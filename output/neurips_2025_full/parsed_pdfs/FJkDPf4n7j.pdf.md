## Efficient and Near-Optimal Algorithm for Contextual Dueling Bandits with Offline Regression Oracles

## Aadirupa Saha ∗

University of Illinois, Chicago aadirupa.saha@gmail.com

## Abstract

The problem of contextual dueling bandits is central to reinforcement learning with human feedback (RLHF), a widely used approach in AI alignment for incorporating human preferences into learning systems. Despite its importance, existing methods are constrained either by strong preference modeling assumptions or by applicability only to finite action spaces. Moreover, prior algorithms typically rely on online optimization oracles, which are computationally infeasible for complex function classes, limiting their practical effectiveness. In this work, we present the first fundamental theoretical study of general contextual dueling bandits over continuous action spaces. Our key contribution is a novel algorithm based on a regularized min-max optimization framework that achieves a regret bound of ˜ O ( √ dT ) -the first such guarantee for this general setting. By leveraging offline oracles instead of online ones, our method further improves computational efficiency. Empirical evaluations validate our theoretical findings, with our approach significantly outperforming existing baselines in terms of regret.

## 1 Introduction

Preference-based feedback is widely used in applications like online retail, prediction markets, tournament ranking, recommender systems, search optimization, robotics, and multiplayer games. Compared to ordinal or absolute rewards, it often provides more reliable insights, such as in book summarization [36], training language models like ChatGPT and LLAMA [9], and designing robotic reward functions via trajectory preferences [12]. A well-studied variant of the multi-armed bandit model [4, 21], the dueling bandit problem, addresses decision-making under pairwise preference feedback with noise and limitations. It has been a key research area for over a decade [43, 3, 29, 7, 26], aiming to identify 'good' arms by actively gathering preference feedback from chosen item pairs.

With the success of reinforcement learning with human feedback (RLHF) [12, 38, 37], a key tool for aligning language models with human values and designing AI behavior, interest in the theoretical foundations of personalized learning with preference feedback has grown, particularly in contextual dueling bandits. The fundamentals of dueling bandits and their generalizations have been widely studied in the bandit and learning theory literature [41, 3, 43, 44]. A major limitation of dueling bandits is the assumption of a fixed preference relation P , whereas real-world preferences vary with context (e.g., user demographics, search queries, seasonal trends). Studies on non-stationary preferences often benchmark against the best single item in hindsight, as in adversarial dueling bandits [29, 18], which is often unrealistic. Some approaches mitigate this with a dynamic regret objective, competing with the best action per round [28, 19], but these require impractical knowledge of non-stationary complexity measures [10]. Moreover, such measures can be large, potentially linear in O ( T ) , and fail to leverage contextual information effectively. A development of the line of contextual dueling bandit research is given in Table 1 .

In the study of contextual bandits [1, 32, 16], the primary focus has been on value or reward feedback. However, there is a notable scarcity of research concentrating on preference or relative feedback (i.e. in the dueling bandits framework). Some recent attempts address the problem of linear contextual

∗ corresponding author

## Robert E. Schapire

Microsoft Research schapire@microsoft.com

dueling bandits for linear score-based preferences [25, 8, 34], however, due to the restrictive linearity assumption, or lack of reasonable performance guarantees, are often far from practical deployment.

Seeking inspiration from these system needs, the work of [14] first formulated the problem of contextual dueling bandits for any general context space and preference relations. However, their proposed algorithms are either computationally intractable that use the computationally inefficient EXP4 algorithms or yield suboptimal O ( T 2 / 3 ) regret bounds. [30] resolves the above limitation of [14] by proposing a computationally efficient and statistically optimal algorithm for contextual dueling bandits, under a natural 'realizability assumption,' that assumes that all the preference relations ( P t s) can be approximated by a (suitably large) function class M . (1). One of the caveat in their result was it requires access to an online square loss regression oracle for M whose T -step square loss regret, E on , M ( T ) , needs to be 'small' ( o ( T ) ) for reasonable regret guarantee.

Table 1: Developments in Contextual Dueling Bandits. E off , M : Averaged regression loss of the online square loss regressor over T rounds for the function class F . E off , M ( F , T, δ ) : Expected regression loss of the offline square loss regressor upon T samples for the function class F and confidence parameter δ . ˜ O ( · ) hides logarithmic factors in the O ( · ) notation.

| Reference                       | Feedback Setting (Contextual)   | Runtime Efficiency   | Regret (Optimality)                                                              | Oracle Type/Calls                                    |
|---------------------------------|---------------------------------|----------------------|----------------------------------------------------------------------------------|------------------------------------------------------|
| ILTCB [2]                       | Bandit (reward)                 | Efficient            | O (√ KT log ( T &#124; Π &#124; ) ) ( Π : Policy classs)                         | Offline oracle ˜ O ( √ KT/ log &#124;F&#124; ) calls |
| SquareCB [16]                   | Bandit (reward)                 | Inefficient          | O ( √ K E off , M T ) Optimal (up to K or log T factors)                         | Online oracle O ( T ) calls                          |
| FALCON+ [32]                    | Bandit (reward)                 | Efficient            | O (√ K E off , M ( F , T, δ log T ) T ) Optimal (up to K or log T factors)       | Offline oracle/ O (log T ) or O (log log T ) calls   |
| SparringEXP4.P [14]             | Dueling preference              | Inefficient          | O (√ KT log ( T &#124; Π &#124; ) ) Optimal (up to log factors) √                | N/A                                                  |
| MaxInp [25]                     | Dueling preference              | Inefficient          | O ( dT log dT ) Optimal (up to log factors)                                      | Online O ( T ) calls                                 |
| MinMaxDB [30]                   | Dueling feedback                | Efficient            | O ( √ K E off , M T ) Optimal (up to K or log T factors)                         | Online O ( T ) calls                                 |
| Double-Monster (this paper)     | Dueling feedback                | Efficient            | O ( K √ E off , M ( F , T, δ log T ) T ) Optimal (up to √ K and log T factors) √ | Offline oracle/ O (log T ) or O (log log T ) calls   |
| Double-Monster-Inf (this paper) | Cont space dueling feedback     | Efficient            | O ( d T log T ) Optimal (up to log T factors)                                    | Offline oracle/ O (log T ) or O (log log T ) calls   |

Offline regression oracles work on static datasets, where all input-output pairs are available upfront. The goal is to find the best-fitting function using the entire dataset at once. The oracle can not fit new data points outside the training distribution, making it suited for batch learning tasks.

Unlike offline regression, an online regression oracle operates in a dynamic setting where data arrives sequentially and the model must adapt in real-time, making it significantly more powerful but also substantially harder to design-especially for general function classes where achieving low regret is often infeasible. For example, no online oracle exists for the class of one-dimensional linear threshold functions due to its infinite Littlestone dimension [23, Example 3], and similarly, the class of all non-decreasing [ -1 , +1] -valued functions over R is not online learnable despite admitting simple offline oracles [22, Example 1]. Further (2). the second major limitation of the work lies in their algorithm only applies to finite action spaces of size K . It's not generalizable to continuous decision spaces, which is, however, a much more practical and realistic scenario to work on.

Noting these precise limitations of the prior works, the natural question to ask thus is whether an optimal regret general (continuous) decision space contextual dueling bandit algorithm can be designed with an offline regression oracle? Note the task of incorporating offline oracles with online learning algorithms is particularly challenging since due to the interactive nature of online learning algorithms, the data generated by the algorithm could be arbitrary (not necessarily following a specific distribution), which makes it much harder to use an offline oracle. There lies one of the main contributions of this work. Towards this, we first analyze the (i) Best-Response regret for finite K -armed contextual dueling bandit, and further extend the setting to (ii) The more practical framework of continuous decision space with potentially infinite arms.

## 1.1 Our contributions

(1). Warm-Up: Best-Response Regret with Offline Oracles: Our first contribution is in proposing a new efficient algorithm for contextual dueling bandits with an offline regression oracle. To state the guarantee, let X be a context space, let A := [ K ] be an action space of size K , and let P := { P ∈ [ -1 , 1] K × K : P [ i, j ] = -P [ j, i ] , P [ i, i ] = 0 } denote the set of preference matrices , which are skew-symmetric matrices with bounded entries and 0 along the diagonal. In a stochastic contextual dueling bandit instance, the learner interacts with a distribution D over X × P via the following protocol: at each round t (1) nature samples ( x t , P t ) ∼ D and reveals x t to the learner, (2) learner chooses (potentially randomly) two actions ( a t , b t ) ∈ [ K ] 2 , (3) learner observes o t ∼ Ber ( P t [ a t ,b t ]+1 2 ) , where Ber ( · ) denotes Bernoulli random variable. Thus o t ∈ { 0 , 1 } indicating the preferred feedback between a t and b t . At each round t , the goal of the learner is to choose a dueling action pair ( a t , b t ) and minimize the performance loss against the dynamic best-action for x t over T rounds (see Objective-1 , Section 2).

For this problem, we propose an algorithm in Section 4 with the following regret performance:

Theorem 1 (Main result: Best-Response Regret (informal)) . Under 'realizable' preference functions and assuming the existence of a 'strongest (set of) best item(s)', with probability at least (1 -δ ) for any δ ∈ (0 , 1) , the best-response regret (BR -Reg T ) of our algorithm Double-Monster (Algorithm 1)

<!-- formula-not-decoded -->

denotes the estimation square loss of the regression oracle when trained on n iid instances, with confidence probability at least 1 -δ .

We introduce the problem and the assumptions more formally in Section 2 and Section 3.

(2) Main Results: Best-Response Regret in Continuous Decision Spaces: We further extend the results of Theorem 5 to continuous decision space K ⊂ R d , where each action a ∈ K is represented by a d -dimensional embedding. Assuming continuous action spaces makes the setting suitable for real world problems like recommender systems or language models where the action set is often large, and potentially infinite. In this case, the preference relation of any action pair ( a , b ) under P t is defined as: P t ( a , b ) = σ ( s ( x t , a ) -s ( x t , b ) ) , where σ denotes the sigmoid function. The scoring function comes from a realizable function class Φ , a set of functions mapping X to R d , such that for any action a ∈ K and context x ∈ X , we have E [ s ( x, a ) | x, a ] = 〈 φ ∗ ( x ) , a 〉 for some unknown mapping φ ∗ ∈ Φ . A detailed description is given in Section 2. We propose an algorithm for this general problem in Section 6 with near-optimal best-response regret bound (up to log factors).

(3) Experimental evaluation: We also report experiments to validate our theoretical analysis and runtime efficiency of the proposed methods (Section 7) and Appendix E.

## 2 Problem Setup

Notation. Let [ n ] := { 1 , 2 , . . . n } , for any n ∈ N . Also [ n ] 2 = [ n ] × [ n ] denotes the cartesian product of [ n ] with itself. We use lowercase bold letters for vectors and uppercase bold letters for matrices. I d denotes the d × d identity matrix. For any vector x ∈ R d , ‖ x ‖ 2 denotes the /lscript 2 norm of x . ∆ n := { p ∈ [0 , 1] n | ∑ n i =1 p ( i ) = 1 , p ( i ) ≥ 0 , ∀ i ∈ [ n ] } denotes the n -simplex. e i denotes the i -th standard basis vector in R n . If p ∈ ∆ n × n is a joint distribution over [ n ] × [ n ] , then we denote by p /lscript and p r respectively the left and the right marginal of p , defined as p /lscript ( i ) = ∑ K j =1 p ( i, j ) and p r ( j ) = ∑ K i =1 p ( i, j ) . In this work, we consider the zero-sum representation of preference matrices: P n := { P ∈ [ -1 , 1] n × n | P [ i, j ] = -P [ j, i ] , P [ i, i ] = 0 , ∀ i, j ∈ [ n ] } .

Note any P ∈ P can be viewed as a zero-sum game , where the two players, called row and column player, simultaneously choose two (possibly randomized) items from [ n ] , with their goal being to respectively maximize and minimize the value of the selected entry. 2

Setting. We assume any arbitrary context set X , an action space of K items denoted by A := [ K ] , and a function class M⊆{ M | M : X ↦→ P K } , all known to the learner ahead of the game. At

2 Standard dueling bandit literature represents preference matrices Q ∈ [0 , 1] n × n , such that Q [ i, j ] indicates the probability of item i being preferred over item j . Here Q satisfies Q [ i, j ] = 1 -Q [ j, i ] and Q [ i, i ] = 0 . 5 . Note both representations are equivalent as there exists a one to one mapping P = (2 Q -1) ∈ P [14, 7].

each round, we assume a context-preference pair ( x t , P t ) ∼ D is drawn from a joint-distribution D , such that x t ∈ X , and P t ∈ P K . We will denote the marginal distribution of the context X as D X . The task of the learner is to select an action pair ( a t , b t ) ∈ [ K ] × [ K ] , upon which a relative feedback o t ∈ { 0 , 1 } is revealed according to P t ; specifically the probability that a t is preferred over b t , indicated by o t = 1 , is given by Pr( o t = 1) = P t [ a t ,b t ]+1 2 , and hence Pr( o t = 0) = 1 -P t [ a t ,b t ] 2 . Assumption 1 (Realizability) . Consider a function class M⊆{ M | M : X → P K } consisting of mappings from context X to preference space P K . Realizability assumption entails ∃ M /star ∈ M such that ∀ a, b ∈ [ K ] and any x ∈ X , we have E ( x, P ) ∼D [ P [ a, b ] | x ] = M /star ( x )[ a, b ] .

Objective (1): Best-Response Regret [30] Assuming the learner selects the duel ( a t , b t ) ∼ p t ∈ ∆ K × K at each round t , we measure the learner's performance via a notion of best response regret: T

<!-- formula-not-decoded -->

where D X denotes the marginal distribution over X .

Objective (2): Best-Response Regret for Continuous Arm Spaces In this setting, we relax the assumed of finite K -armed action space and assume a general continuous (potentially infinite) decision space K ⊂ R d . In order to model the preference relation for an action pair ( a , b ) ∈ K × K : We further assume that given any context x t each action a t ∈ K first gets assigned to a stochastic score s t ∈ [ -1 , 1] such that E [ s t | x t = x, a t = a ] = 〈 a, φ ∗ ( x ) 〉 , where φ ∗ : X ↦→ R d is a unknown mapping that embeds every context to a d -dimensional feature space. In particular, with the realizability assumption , one can typically assume the learner has access to a class of functions Φ ⊆ { φ : X → R d } such that φ ∗ ∈ Φ . Then at time t , the preference relation of any action pair ( a , b ) under the above realizable score setting is: P t ( a , b ) = σ ( 〈 φ ∗ ( x ) , a -b 〉 ) , where σ ( · ) denotes the sigmoid transformation, i.e. ( σ ( x ) = 1 1+ e -x , ∀ x ∈ R ) . Similar to before, the learner's task is to select an action pair ( a t , b t ) ∈ K × K at time t , upon which a relative feedback o t ∈ { 0 , 1 } ∼ Ber ( P t ( a t , b t )) is revealed according to P t .

Under this model of continuous action space K , one could define best-response regret as:

<!-- formula-not-decoded -->

## 3 A Primer on Regression Oracles

We now introduce the concepts of offline and online regression oracles to familiarize readers with their structural disparities and performance-related distinctions.

## 3.1 Offline Regression Oracles [32]

Consider any abstract input space Z and output space Y . Assume a given dataset D n := { ( z i , y i ) } n i =1 consists of n data points, each (input, output) pair ( z i , y i ) iid ∼ D being drawn iid from a fixed underlying distribution D on Z×Y . Given a general function class F ⊆ { f | f : Z ↦→ Y} , a general offline regression oracle associated with F , denoted by OffReg F is an algorithm, operates on the given dataset D n and outputs a mapping ˆ f : Z ↦→ Y . In learning theory, the quality of ˆ f is measured by its 'out-of-sample error,' i.e., its expected error on random and unseen test data.

Assumption 2 (Guarantee of Offline Regression Oracles) . Consider a general function class F ⊆ { f | f : Z ↦→ Y} . Then given a dataset D n := { ( z i , y i ) } n i =1 that consists of n data points, s.t. ( z i , y i ) iid ∼ D , we assume that the output ˆ f n ← OffReg F ( D n ) of the offline regression oracle OffReg satisfies that: For any δ ∈ (0 , 1] , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where E off , M ( δ, n ) denotes the square loss of the regression oracle when trained on n -iid instances.

Assumption 4 under Realizability. Moreover, if we further assume realizability, in the sense that there exists f ∗ ∈ F such that f ∗ ( z ) = E ( z,y ) ∼D [ y | z ] , then it is well-known that Assumption 4 further implies: For any δ ∈ (0 , 1] , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Remark 2. The advantage of an offline regression oracle is that it can consider all the data available for training, the training data is generated iid from some underlying distribution and therefore, can often produce a more accurate model. See Remark 13 for the advantages of offline regressors. Conversely, online regression is a real-time processing approach that involves analyzing data as it becomes available, updating the model as new data points arrive. Thus finding efficient online oracles for any arbitrary function class could be much harder compared to its offline counterpart (e.g. see [23, Example 3], [22, Example 1]) See Appendix A for a detailed discussion.

## 4 Warm-Up: K -Armed General Contextual DB with Offline Oracles

This section presents our algorithm for the contextual best-response regret ( Objective-1 in Section 2).

Double-Monster : Key Algorithmic Ideas. To adapt with the offline regression oracle, our algorithm runs in epochs, with a specific (and predetermined) epoch schedule, where the offline regression oracle is called only at rounds τ 1 , τ 2 , τ 3 , . . . , and each set of rounds between τ m -1 +1 and τ m is considered to be within the m -th epoch. For example, if we set τ m = 2 m , our algorithm runs in O (log T ) epochs for any unknown T , but one can also use more complex epoch schedules, such as τ m = ⌊ 2 T 1 -2 -m ⌋ , which results in only O (log log T ) epochs and oracle calls.

At the beginning of each epoch m , the algorithm computes an estimated function ̂ M m using the offline regression oracle: ̂ M m := arg min M ∈M ∑ τ m -1 τ = τ m -2 +1 ( M ( x τ )[ a τ , b τ ] -o τ ) 2 . We next assume access to an algorithm Convex-Constraint-Solver , which is a randomized algorithm that for given x ∈ X , M ∈ M and γ ∈ R , outputs a joint distribution p ∈ ∆ K × K , such that:

<!-- formula-not-decoded -->

where recall we denote by p /lscript and p r respectively the left and the right marginal of p , defined as p /lscript ( i ) = ∑ K j =1 p ( i, j ) and p r ( j ) = ∑ K i =1 p ( i, j ) . Note p being the solution of ( K 2 ) + K -convex constraints, such a solver can be designed computationally efficiently using standard tools from convex programming. Moreover, in Appendix B.2 justify that a p ∈ ∆ K × K will always exist. Now, at any time t in epoch m , given the context x t the estimated least square function estimate ̂ M m , and some tuning parameter γ m (exact expression is given later), we query the Convex-Constraint-Solver with the triplet ( x t , ̂ M m , γ m ) and obtain p t ∈ ∆ K × K using:

<!-- formula-not-decoded -->

Next, the algorithm samples a duel ( a t , b t ) ∼ p t . Noting E a ∼ q ,b ∼ p t [ ̂ M m ( x t )[ a, b ] ] = q /latticetop ̂ M m ( x t ) p t , p t can be alternatively expressed as:

<!-- formula-not-decoded -->

which implies the mixed strategy p t is a 'nearly unbeatable by any pair of opponent q and q ′ ' in the zero-sum game ̂ M m ( x t ) . Recall, any distribution p ∗ ∈ ∆ K is a Nash Equilibrium of a symmetric zero-sum square matrix P ∈ R K × K if ∀ q ∈ ∆ K : q /latticetop Pp ∗ ≤ 0 [11] and Nash strategies are unbeatable. However, to incorporate the additional 'explorative component' of p t we need to ensure p t ( i, j ) has significant mass based on the quality of action i and j . This is enforced through the term 1 p t ( i,j ) in Eq. (4). In Appendix B.2, we prove that Eq. (2) always adheres to a valid solution and hence one can always find a p t for Eq. (4). The pseudocode is given in Algorithm 1 in Appendix B.

## 4.1 Performance Guarantees of Algorithm 1.

Useful Concepts towards proving Theorem 5. We analyze the regret of Algorithm 1 in Theorem 5, but we need to define some concepts before that.

Definition 1 (Policy Class) . A standard policy class Π := { π | π : X ↦→ ∆ K } is a set containing all the mappings from context space X to the K -simplex. Further we also define by policy class Π 2 := { π | π : X ↦→ ∆ K × K } a set of all mappings from context X to K × K -simplex.

For any policy π ∈ Π we denote by π ( i | x ) the i -th component of vector π ( x ) , ∀ i ∈ [ K ] , x ∈ X . Similarly, for any policy π ∈ Π 2 , we denote by π ( i, j | x ) , the i, j -th component of the matrix π ( x ) . Definition 2 (Decision policy of Epochm ) . At any epoch m and x ∈ X , the decision policy of epoch m is defined as π m ∈ Π 2 : π m ( x ) ← Convex-Constraint-Solver ( x, ̂ M m , γ m ) .

.

Note that at any time t in epoch m , π m ( x t ) = p t , where p t is as defined in Eq. (3). Thus π m ∈ Π 2 We will further denote by π /lscript m , π r m ∈ Π the left and right marginal policies of π m defined as: π /lscript m ( i | x ) = E x ∼D X [ ∑ K j =1 π m ( i, j | x ) ] and π r m ( j | x ) = E x ∼D X [ ∑ K i =1 π m ( i, j | x ) ] , ∀ i, j ∈ [ K ]

Definition 3 (Instantaneous Regret against a Fixed Policy) . For any arbitrary preference relation M ∈ M , we denote the regret of policy π against π ′ as:

<!-- formula-not-decoded -->

In particular, for M = M ∗ , Reg( π , π ′ ) = Reg M ∗ ( π , π ′ ) defines true regret of policy π against π ′ and for M = ̂ M m , Reg ̂ M m ( π , π ′ ) simply denotes the instantaneous empirical regret of π against π ′ in epoch m . For simplicity, we denote ̂ Reg m ( π , π ′ ) = Reg M m ( π , π ′ ) .

̂

Further since for any P ∈ P K , i.e. P [ i, j ] = -P [ j, i ] , ∀ i, j ∈ [ K ] × [ K ] , we note that:

Remark 3 (Properties of Policy Regret) . For any pair of policies π and π ′ ∈ Π : (1) Reg( π , π ′ ) = -Reg( π ′ , π ) , (2) ̂ Reg m ( π , π ′ ) = -̂ Reg m ( π ′ , π ) .

Definition 4 (Best-Response Policy) . We let ψ /star : Π ×M→ Π be a best-response policy, meaning, for any context x ∈ X , policy π ∈ Π and function M ∈ M , ψ /star [ π , M ]( x ) maximizes E a /star ∼ p ,a ∼ π ( x ) [ M ( x )[ a /star , a ] ] = p /latticetop M ( x ) π ( x ) over p ∈ ∆ K . More precisely for any x ∈ X , π ∈ Π and M ∈ M , ψ /star [ π , M ]( x ) ∈ Π can be defined as:

<!-- formula-not-decoded -->

Note that the due to linearity of the above objective, the corresponding argmax always occurs at one of the K extreme points of K -simplex ∆ K .

Definition 5 (Instantaneous Best Response Policy Regret) . For any arbitrary underlying preference relation M ∈ M , we denote the instantaneous best-response regret of any decision policy π ∈ Π as:

<!-- formula-not-decoded -->

In particular, for M = M ∗ , BReg M ∗ ( π ) denotes the instantaneous true best-response regret of any decision policy π ∈ Π . For simplicity we will denote BReg( π ) = BReg M ∗ ( π ) . Further for M = ̂ M m , BReg ̂ M m ( π ) simply defines the instantaneous empirical best-response regret of π in epoch m . For simplicity, we will use ̂ BReg m ( π ) = BReg M m ( π ) .

Remark 4. Note that for any π ∈ Π , one can write

̂

<!-- formula-not-decoded -->

This implies that the instantaneous true and empirical regret of any fixed policy π against its corresponding best response policy, respectively yield the instantaneous true and empirical best response regret of π .

Assumption 3 (Idempotent Best-Response (IBR)) . A function class M : X ↦→ P is said to satisfy Idempotent Best-Response if for all M ∈ M and all π ∈ Π , BReg M ( ψ ∗ [ π , M ]) = 0 .

Roughly speaking, any preference matrix with 'total-ordering', where there is a specific underlying ranking of the K items, satisfies the above structure: The class of random utility (RUM) based preferences [5, 33, 27], or matrices with strong stochastic transitivity (SST)

[42], always satisfies the above property. However, the class 'Idempotent Best-Response' is larger, in particular, any preference matrix, which has a set of equally strong best items also satisfies the structure. The figure on the right shows the dependencies of different preference classes. We now state our main result towards analyzing Algorithm 1.

<!-- image -->

Theorem 5 (Regret Analysis of Algorithm 1) . Consider any function class M that satisfies Assumption 3. Then under realizability (Assumption 1), and with an epoch schedule τ 1 , . . . , τ m such that τ m ≥ 2 m for m ≤ log T , with probability at least (1 -δ ) for any δ ∈ (0 , 1) the best-response regret (BR -Reg T ) of Double-Monster after T rounds is bounded by:

<!-- formula-not-decoded -->

Corollary 6 (Theorem 5 for Special Function Classes) . Algorithm 1 yields the following best-response regret guarantees for some of the special function classes:

- For finite M , for the choice of δ = 1 /T , the regret of Algorithm 1 is bounded by O ( K √ T log ( |M| T log T )) . The results follows since for finite M , we have offline regression oracles with E off , M ( δ, n ) = O ( log( |M| n ) nδ ) [32]. For completeness, the proof is given in Appendix C.5.
- For a general, potentially nonparametric function class F with empirical entropy is O ( ε -p ) , ∀ ε &gt; 0 for some constant p &gt; 0 , results of [40] and [24] gives offline regression oracles such that E off , M ( δ, n ) = O ( n -2 / (2+ p ) log(1 /δ )) . Again assuming δ = 1 /T , this implies a regret bound of O ( KT 1+ p 2+ p log T ) of Algorithm 1 for this function class.
- For low dimensional linear predictors F = { ( x, a, b ) ↦→ 〈 θ, φ ( x, a, b ) 〉 : θ ∈ R d , ‖ θ ‖ 2 ≤ 1 } , instantiating OffReg as the least squared estimator [39, 13, 21] and δ = 1 /T , the regret guarantee of Algorithm 1 becomes O ( d √ T log( T/d )) .
- [15] gives regression error bounds for deep neural networks for F = G K , G being the class of Multi-Layer Perceptrons (MLP) , and f ∗ ( x, a, b ) = g ∗ a,b ( x ) for x ∈ X , a, b ∈ [ K ] . Assume that D is a continuous distribution on [ -1 , 1] d and g ∗ lie in a Sobolev ball with smoothness β ∈ N , by

Theorem 1 of [15] deep MLP-ReLU network estimator attains E off , M ( n, δ ) = ˜ O ( n β + d log 1 /δ ) estimation error. Consequently, the regret bound of Algorithm 1 boils down to ˜ O ( KT β +2 d 2 β +2 d ) regret.

- a,b -β

## 5 Regret Analysis of Double-Monster : Proof Analysis of Theorem 5

Towards proving Theorem 5, we will first prove Lemma 7 that guarantees on the empirical regret performance of Algorithm 1. But it will be worth introducing a few more notations first:

<!-- formula-not-decoded -->

Intuitively V ˜ π ( π 1 , π 2 ) captures the selection-variance of ˜ π for any expected duel ( a, b ) played by the joined policy π 1 , π 2 . In particular, when the base policy ˜ π = π m , for any epoch m , we will use the shorthand notation V m ( π 1 , π 2 ) to represent V π m ( π 1 , π 2 ) .

Lemma 7 (Properties of decision policy π m in Algorithm 1) . At any epoch m , ∀ π , π ′ ∈ Π , the decision policy of epoch m π m satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 7 establishes an important result for bounding the empirical performance of the decision policy π m , for any epoch m : Precisely it shows that the empirical best response regret of π m is bounded by O ( K 2 /γ m ) and the decision variance of any pair of policies π , π ′ w.r.t. π m is bounded by empirical regret bound π and π ′ against π m . The proof is given in Appendix B.

## 5.1 Relating Empirical Performance to True Performance through Decision Variance

We will first show how to relate the true and empirical performance of any policy in terms of their corresponding decision variance, as given below:

Lemma 8. For any epoch m and any two decision policies π and π ′ ,

<!-- formula-not-decoded -->

Proof of this lemma is deferred to Appendix C.2. We note that Lemma 8 further implies:

Corollary 9. For any epoch m , and any policy π ∈ Π

<!-- formula-not-decoded -->

## 5.2 The Recursion: Bounding Empirical Regret by True Regret and Vice-Versa

Using Corollary 9, we now prove a crucial recursive relation between empirical best-response and true best-response regret for any decision policy π at any epoch m . This lemma ensures that the estimated empirical (best-response) regret of the algorithm's decision policy π m becomes more and more accurate and eventually matches its true best-response regret for large m .

Lemma 10 (Epochwise Recursion) . Let π be any policy in Π 2 . Then for all epochs m ∈ N + :

<!-- formula-not-decoded -->

Given Lemma 10, Theorem 5 follows by combining it with Lemma 7. Complete proof is given in Appendix C.4.

## 6 Main Algorithm: General Contextual DB for Continuous Action Spaces

In this section, we extend the results of the previous section to continuous action spaces K ⊂ R d , as defined in Objective-(2) , Section 3). We present our algorithm in Algorithm 2, which is shown to yield ˜ O ( √ dT ) contextual best-response regret as analyzed in Theorem 11. We explain the key ideas behind Algorithm 2 below in detail but it would be useful to introduce a few notations before that.

Additional Notations: For a set X , we let ∆( X ) denote the set of all probability distributions over X . If X is continuous, we typically denote ∆( X ) as the set of all probability measures on the measurable space ( X, B ) , where B is the Borel σ -field on the set X . Further, for any n ∈ N + , we denote the n -simplex by ∆ n . So, if X is finite, ∆( X ) = ∆ | X | . I n denotes the identity matrix of dimension n for any n ∈ N + . We use ‖ x ‖ (or ‖ x ‖ 2 ) to denote the euclidean norm for x ∈ R d . For any positive definite matrix H ∈ R d × d , we denote the induced norm on x ∈ R d by ‖ x ‖ 2 H = 〈 x , Hx 〉 , and det( H ) represents the determinant of matrix H .

## Algorithm 2 Double-Monster-Inf (for Continuous Action Spaces)

1: input Epoch schedule 0 = τ 0 &lt; τ 1 &lt; τ 2 &lt; · · · . Confidence parameter δ . Tuning parameter c, γ 1 , γ 2 , . . . .

- 2: Arm set: K ⊂ R d . An instance of OffReg for function class Φ
- 3: for epoch m = 1 , 2 , . . . do

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

φ

m

= arg min

τ

φ

∈

Φ

∑

m

τ

-

=

τ

1

m

-

2

+1

(

σ

- 6: for round t = τ m -1 +1 , · · · , τ m do
- 7: Observe context x t ∈ X .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 10: end for
- 11: end for

〈

(

φ

x

(

)

,

a

-

b

〉

)

-

o

)

τ

τ

τ

τ

2

via the offline least squares oracle

Double-Monster-Inf : Key Algorithm Ideas. The main ideas of this algorithm are similar to Double-Monster (Algorithm 1) which also runs in epochs with a predetermined epoch schedule τ 0 , τ 1 , . . . , and tries to estimate the underlying linear score mapping φ ∗ with φ m at each epoch m . However, the key difference from Double-Monster (Algorithm 1) lies in the duel-selection routine which requires using a different convex constraint solver for continuous domains as it is not possible to maintain distribution over pairs of actions in the continuous space. The key steps are:

At the beginning of each epoch m , the algorithm computes an estimated scoring function φ m using the offline regression oracle that yields:

<!-- formula-not-decoded -->

Once φ m is computed, at any time t in epoch m , given the context x t the estimated least square function estimate φ m , and some tuning parameter γ m = ( √ d 2 √ 2 E off , Φ ∗ ( δ/ (2 m 2 ) , τ m -1 -τ m -2 ) ) , we query the Cont-CvxConstraint-Solver with the triplet ( x t , φ m , γ m ) and obtain p t ∈ ∆ K×K using a continuous convex constraint solver. More precisely, Cont-CvxConstraint-Solver is defined as a (possibly randomized) algorithm, s.t. given any triplet ( x, φ, γ ) , with x ∈ X , φ ∈ Φ and γ ∈ R , it outputs p ∈ ∆ K×K :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Above is the key step of Algorithm 2 which sets it apart from our previous arm selection technique of Algorithm 1 proposed for the finite action space. Formally, p ∈ ∆ K×K in Eq. (6) captures a distribution over 'high-scoring' action pairs (owing to the term ' 〈 ab p , φ ( x ) 〉 '), with sufficient variability (owing to the term ' 1 γ log det( H p ) ') . Importantly, the tradeoff between the first and second terms of Eq. (6): the first term would like to put mass on the actions that align with φ m ( x t ) -ensuring 'exploitation' of the regressed model φ m ; the second term would want to put more mass on distinct pairs of actions in K to induce higher variability - The challenge was to incorporate this explore-exploit tradeoff in the duel-selection rule p t for continuous action spaces.

Next, the algorithm simply samples a duel ( a t , b t ) ∼ p t from the joint distribution p t and the rest of the algorithm proceeds almost the same as Algorithm 1 except for the choice of the tuning parameters γ m . The complete pseudocode of the algorithm is given in Algorithm 2.

Notably, our arm selection technique is inspired from [17, Alg. 2], which introduced the idea of logdet-barrier distribution based arm-selection for contextual bandits which represents a regularized empirical risk minimization procedure. (1) We however had to modify their proposed distribution due to the dueling (preference) nature of the feedback. (2) More importantly, our regret analysis is different as their analysis relies on the existence of an online regression oracle while we design and analyze all algorithms with (the weaker) offline oracles as motivated in Section 1 and Section 3.

## 6.1 Performance Guarantees of Algorithm 2.

Theorem 11 (Regret Analysis of Algorithm 2) . Assume ∀ x ∈ X , ‖ φ ∗ ( x ) ‖ ≤ D for some D &gt; 0 , and suppose we choose c = 5 d 64 D 2 γ m ( T ) . Then under realizability (Assumption 1) of the function class Φ , and with an epoch schedule τ 1 , . . . , τ m such that τ m ≥ 2 m for m ≤ log T , with probability at least (1 -δ ) for any δ ∈ (0 , 1) the best-response regret of Algorithm 2, BR -Reg ( cont ) T , in T rounds

<!-- formula-not-decoded -->

Tallying the result of Theorem 11 with [17, Theorem 2], which however only applies to the simpler setting of reward feedback models, one can claim that regret guarantee of Algorithm 2 is optimal up to log factors. The proof analysis of Theorem 11 is given in Appendix D.1.

## 7 Experiments

In this section, we report the empirical regret performance of our proposed algorithm Double-Monster-Inf (Algorithm 2) on different environments for general continuous decision spaces. All results are averaged across 100 runs.

We set the decision set K ⊂ R d to the unit ball in dimensiond . We run experiments on two different problem instances (instantiated by its feature maps φ ): (1) Inst-1: Here the preference class M is characterized by a linear feature map φ : X ↦→ R d , where for any x ∈ X ⊂ R c , context dimension c ∈ N , s.t. φ ( x ) = Zx ‖ Zx ‖ , for some arbitrary choice of Z ∈ R d × c . For our experiment, we set c = 8 , and report the experiments for d = 2 , 5 , 10 , 15 . (2) Inst-2: Here the preference class M is characterized by a quadratic feature map φ : X ↦→ R d , where for any x ∈ X ⊂ R c , c ∈ N , s.t. φ ( x ) = ( x 1 , . . . , x c , x 2 1 , . . . , x 2 c , x 1 x 2 , . . . , x c -1 x c , 1) ∈ R 2 c + ( c 2 ) +1 . Note thus d = 2 c + ( c 2 ) +1 . We consider c = 1 , 2 , 3 , 4 respectively, yielding d = 3 , 6 , 10 , 15 .

Regret with increasing d . As shown in the figure below, we see that for both the above instances, the contextual regret scales as ˜ O ( √ dT ) corroborating Theorem 11. Note our decision space is fully continuous and with infinite arms, unlike the previous work of [30], which also used online regression oracles. Also, our preference relations are much more general that can be built on non-linear utilities (e.g., Inst 2), unlike the previous attempts of [31, 25, 19], which were also inefficient to implement additionally. Consequently, these works could only report experiments on finite decision spaces.

<!-- image -->

←()

()

Figure 1: (Left) Inst-1 (linear transformation) and (Right) Inst-2 (quadratic transformation).

Comparing with Non-Contextual Baselines. We ran experiments with Double-Monster-Inf for finiteK armed non-contextual MAB setting and compared with the following existing finite-arm Dueling-Bandit baselines: (i) RUCB [43], (ii) REX3 [18], (iii) [35], (iv) RMED [20]. Results show we outperformed almost every baseline and performed comparably with DTS, even though these baselines only apply to finite-amended non-contextual settings. We assumed 3 different types of scores s : - Base(20 arms): s (1) = 1 , s ( i ) = 0 . 7 for i ∈ { 2 , ..., 10 } , and s ( i ) = 0 . 4 for i ∈ { 11 , ..., 20 } . -Hard (20 arms): s (1) = 1 and s ( i ) = θ i -1 -0 . 05 for i ∈ { 2 , ..., 20 } . - Worst Case (WC, 40 arms): Considers a worst-case instance s (1) = 1 and s ( i ) = 0 . 9 for i ∈ { 2 , ..., 40 } .

Figure 2: Comparing with Finite-armed Non-Contextual Baselines

<!-- image -->

Runtime. We also report the runtime comparison of different algorithms in Appendix E.

## 8 Perspectives

This work provides the first computationally efficient and near-optimal algorithm for contextual dueling bandits using offline oracles, resolving a key open problem from [30] and enabling scalable solutions for preference-based personalized learning in real-world applications. Additionally, we, for the first time, analyze the problem of for the continuous action space and propose a near-optimal ˜ O ( √ dT ) regret algorithm for this setting. We hope this advancement will offer more practical and scalable solutions for contextual dueling bandits, contributing to more real-world human-centric prediction models, including LLMs, human-assisted robotics, recommender systems, etc.

Future Work. Moving forward it would be interesting to analyze the problem beyond pairwise preference, generalizing it to mulitset/ ranking preferences. Additionally, does the complexity of the performance limit alter when employing an offline oracle compared to an online oracle? Deploying the proposed algorithms in practical use-case scenarios, e.g., tuning LLMs, training autonomous vehicles, or AI-alignment in robotics, would also be an interesting empirical study. Another interesting direction would be to extend the proposed algorithmic framework to more general frameworks, like reinforcement learning or RLHF, and understand the corresponding theoretical guarantees.

## References

- [1] Alekh Agarwal, Miroslav Dudík, Satyen Kale, John Langford, and Robert Schapire. Contextual bandit learning with predictable rewards. In Artificial Intelligence and Statistics , pages 19-26. PMLR, 2012.
- [2] Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In International Conference on Machine Learning , 2014.
- [3] Nir Ailon, Zohar Shay Karnin, and Thorsten Joachims. Reducing dueling bandits to cardinal bandits. In International Conference on Machine Learning , 2014.
- [4] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning , 2002.
- [5] Hossein Azari, David Parkes, and Lirong Xia. Random utility theory for social choice. In Advances in Neural Information Processing Systems , 2012.
- [6] Nathan Barczi. Lecture notes: 14.102, math for economists. 2004.
- [7] Viktor Bengs, Róbert Busa-Fekete, Adil El Mesaoudi-Paul, and Eyke Hüllermeier. Preferencebased online learning with dueling bandits: A survey. Journal of Machine Learning Research , 2021.
- [8] Viktor Bengs, Aadirupa Saha, and Eyke Hüllermeier. Stochastic contextual dueling bandits under linear stochastic transitivity models. In International Conference on Machine Learning , pages 1764-1786. PMLR, 2022.
- [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems , 33:1877-1901, 2020.
- [10] Thomas Kleine Buening and Aadirupa Saha. Anaconda: An improved dynamic regret algorithm for adaptive non-stationary dueling bandits. In International Conference on Artificial Intelligence and Statistics , pages 3854-3878. PMLR, 2023.
- [11] Nicolo Cesa-Bianchi and Gabor Lugosi. Prediction, learning, and games . Cambridge university press, 2006.
- [12] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems , 30, 2017.
- [13] Wei Chu, Lihong Li, Lev Reyzin, and Robert Schapire. Contextual bandits with linear payoff functions. In Artificial Intelligence and Statistics , 2011.
- [14] Miroslav Dudík, Katja Hofmann, Robert E Schapire, Aleksandrs Slivkins, and Masrour Zoghi. Contextual dueling bandits. In Conference on Learning Theory , 2015.
- [15] Max H Farrell, Tengyuan Liang, and Sanjog Misra. Deep neural networks for estimation and inference. Econometrica , 89(1):181-213, 2021.
- [16] Dylan Foster and Alexander Rakhlin. Beyond ucb: Optimal and efficient contextual bandits with regression oracles. In International Conference on Machine Learning , 2020.
- [17] Dylan J Foster, Claudio Gentile, Mehryar Mohri, and Julian Zimmert. Adapting to misspecification in contextual bandits. In Advances in Neural Information Processing Systems , 2020.
- [18] Pratik Gajane, Tanguy Urvoy, and Fabrice Clérot. A relative exponential weighing algorithm for adversarial utility-based dueling bandits. In International Conference on Machine Learning , 2015.
- [19] Patrick Kolpaczki, Viktor Bengs, and Eyke Hüllermeier. Non-stationary dueling bandits. arXiv preprint arXiv:2202.00935 , 2022.

- [20] Junpei Komiyama, Junya Honda, Hisashi Kashima, and Hiroshi Nakagawa. Regret lower bound and optimal algorithm in dueling bandit problem. In Conference on Learning Theory , 2015.
- [21] Tor Lattimore and Csaba Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- [22] Haipeng Luo. Theoretical machine learning (csci699)-lecture 6. Lecture Note , 2019.
- [23] Nishant Mehta. Introduction to online learning (csc 482a/581a)-lecture 2. Lecture Note , 2023.
- [24] Alexander Rakhlin, Karthik Sridharan, and Alexandre B Tsybakov. Empirical entropy, minimax regret and minimax risk. 2017.
- [25] Aadirupa Saha. Optimal algorithms for stochastic contextual preference bandits. Advances in Neural Information Processing Systems , 34:30050-30062, 2021.
- [26] Aadirupa Saha and Aditya Gopalan. Combinatorial bandits with relative feedback. In Advances in Neural Information Processing Systems , 2019.
- [27] Aadirupa Saha and Aditya Gopalan. Best-item learning in random utility models with subset choices. In Artificial Intelligence and Statistics . PMLR, 2020.
- [28] Aadirupa Saha and Shubham Gupta. Optimal and efficient dynamic regret algorithms for non-stationary dueling bandits. In International Conference on Machine Learning , pages 19027-19049. PMLR, 2022.
- [29] Aadirupa Saha, Tomer Koren, and Yishay Mansour. Adversarial dueling bandits. In International Conference on Machine Learning , 2021.
- [30] Aadirupa Saha and Akshay Krishnamurthy. Efficient and optimal algorithms for contextual dueling bandits under realizability. In International Conference on Algorithmic Learning Theory , pages 968-994. PMLR, 2022.
- [31] Aadirupa Saha, Aldo Pacchiano, and Jonathan Lee. Dueling rl: Reinforcement learning with trajectory preferences. In International Conference on Artificial Intelligence and Statistics , pages 6263-6289. PMLR, 2023.
- [32] David Simchi-Levi and Yunzong Xu. Bypassing the monster: A faster and simpler optimal algorithm for contextual bandits under realizability. Mathematics of Operations Research , 47(3):1904-1931, 2022.
- [33] Hossein Azari Soufiani, David C Parkes, and Lirong Xia. Computing parametric ranking models via rank-breaking. In International Conference on Machine Learning , 2014.
- [34] Arun Verma, Zhongxiang Dai, Xiaoqiang Lin, Patrick Jaillet, and Bryan Kian Hsiang Low. Neural dueling bandits: Preference-based optimization with human feedback. In International Conference on Learning Representations , 2024.
- [35] Huasen Wu and Xin Liu. Double Thompson sampling for dueling bandits. In Advances in Neural Information Processing Systems , 2016.
- [36] Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Christiano. Recursively summarizing books with human feedback. arXiv preprint arXiv:2109.10862 , 2021.
- [37] Tengyang Xie, Dylan J Foster, Akshay Krishnamurthy, Corby Rosset, Ahmed Awadallah, and Alexander Rakhlin. Exploratory preference optimization: Harnessing implicit q*-approximation for sample-efficient rlhf. arXiv preprint arXiv:2405.21046 , 2024.
- [38] Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. In Forty-first International Conference on Machine Learning , 2024.
- [39] D. Pal Y. Abbasi-Yadkori and C. Szepesvari. Improved algorithms for linear stochastic bandits. In Neural Information Processing Systems , 2011.

- [40] Yuhong Yang and Andrew Barron. Information-theoretic determination of minimax rates of convergence. Annals of Statistics , pages 1564-1599, 1999.
- [41] Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. The k -armed dueling bandits problem. Journal of Computer and System Sciences , 2012.
- [42] Yisong Yue and Thorsten Joachims. Beat the mean bandit. In International Conference on Machine Learning , 2011.
- [43] Masrour Zoghi, Shimon Whiteson, Remi Munos, and Maarten Rijke. Relative upper confidence bound for the k-armed dueling bandit problem. In International Conference on Machine Learning , 2014.
- [44] Masrour Zoghi, Shimon A Whiteson, Maarten De Rijke, and Remi Munos. Relative confidence sampling for efficient on-line ranker evaluation. In International Conference on Web search and Data Mining , 2014.

## Supplementary: Efficient and Near-Optimal Algorithm for Contextual Dueling Bandits with Offline Regression Oracles

## A A Primer on Regression Oracles

In this section, we introduce the concepts of offline and online regression oracles to familiarize readers with their structural disparities and performance-related distinctions.

## A.1 Offline Regression Oracles [32]

Consider any abstract input space Z and output space Y . Assume a given dataset D n := { ( z i , y i ) } n i =1 consists of n data points, each (input, output) pair ( z i , y i ) iid ∼ D being drawn iid from a fixed underlying distribution D on Z×Y . Given a general function class F ⊆ { f | f : Z ↦→ Y} , a general offline regression oracle associated with F , denoted by OffReg F is an algorithm, operates on the given dataset D n and outputs a mapping ˆ f : Z ↦→ Y . In learning theory, the quality of ˆ f is measured by its 'out-of-sample error,' i.e., its expected error on random and unseen test data.

Assumption 4 (Guarantee of Offline Regression Oracles) . Consider a general function class F ⊆ { f | f : Z ↦→ Y} . Then given a dataset D n := { ( z i , y i ) } n i =1 that consists of n data points, s.t. ( z i , y i ) iid ∼ D , we assume that the output ˆ f n ← OffReg F ( D n ) of the offline regression oracle OffReg satisfies that: For any δ ∈ (0 , 1] , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where E off , M ( δ, n ) denotes the square loss of the regression oracle when trained on n -iid instances.

Assumption 4 under Realizability. Moreover, if we further assume realizability, in the sense that there exists f ∗ ∈ F such that f ∗ ( z ) = E ( z,y ) ∼D [ y | z ] , then it is well-known that Assumption 4 further implies: For any δ ∈ (0 , 1] , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

The offline regression error bound E off , F ( n, δ ) is a function that decreases with n . Note, for our problem, we have Z = ( X × [ K ] 2 ) to be the set of (context, action-pair) tuples such that z t = ( x t , a t , b t ) with x t ∈ X and ( a t , b t ) ∈ [ K ] 2 . Our output space is Y = [ -1 , 1] as note o t ∈ Y .

## A.2 Online Regression Oracles

An online regression oracle [11, Chapter 3], is an algorithm, which we denote by OnReg , and which operates in the following online protocol on each round t : (1) it receives an abstract input z t ∈ Z from some input space Z , chosen adversarially by the environment; (2) it produces a real-valued prediction ˆ y t ∈ Y ⊂ R where Y is some output space; (3) it observes the true response y t ∈ Y and incurs loss /lscript (ˆ y t , y t ) := (ˆ y t -y t ) 2 . The goal of the oracle is to predict the outcomes as well as the best function in a given function class F := { f : Z ↦→ R } , such that for every sequence of outcomes, the square loss regret is bounded.

More formally, at time t and for input z ∈ Z , the online regression oracle associated to the function class F , denoted by OnReg F , can be seen as a mapping ˆ y t ( z ) := OnReg F ( z, { z τ , y τ } t -1 τ =1 ) . Note this corresponds to the prediction the algorithm would make at time t if we passed in the input z , although this input may not be what is ultimately selected by the environment.

Assumption 5. Online Regression Oracle Guarantee: Given any function class F ⊆ { f | f : Z ↦→ Y} , the online regression oracle OnReg F guarantees for every sequence { ( z t , y t ) } t ∈ [ n ] , its regret is bounded as:

<!-- formula-not-decoded -->

where E on , F ( n ) = o (1) is a known upper bound and a decreasing function of n .

Assumption 5 under Realizability. Moreover, if we further assume realizability, in the sense that there exists f /star ∈ F such that ∀ t : f /star ( z t ) = E [ y t | z t ] , then one can show that the guarantee further implies:

<!-- formula-not-decoded -->

Remark 12 (Some examples) . Online square loss regression is a well-studied problem, and efficient algorithms with provable regret guarantees are known for many specific function classes including finite classes where |F| &lt; ∞ , finite and infinite dimensional linear classes, and others [16, 17]. For completeness, we provide formal definition for some specific classes and instantiations of the regression oracles in Appendix A.3.

## A.3 Examples: Some Specific Regression Function Classes:

1. Finite regression function class F such that |F| &lt; ∞ .
2. Class of linear predictors F := { ( x, a ) ↦→〈 θ, x a 〉 | θ ∈ R d , ‖ θ ‖ ≤ 1 } .
3. Class of generalized linear predictors F := { ( x, a ) ↦→ σ ( 〈 θ, x a 〉 ) | θ ∈ R d , ‖ θ ‖ ≤ 1 } where σ : R ↦→ [0 , 1] is a fixed non-decreasing 1 -Lipschitz link function.
4. Reproducing kernel hilbert space (RKHS) F := { f | ‖ f ‖ H ≤ 1 , K ( x a , x a ) ≤ 1 } .
5. Banach Spaces F := { ( x, a ) ↦→〈 θ, x a 〉 | θ ∈ B , ‖ θ ‖ ≤ 1 } , where ( B , ‖ · ‖ ) is a separable Banach space and x belongs to the dual space ( B , ‖ · ‖ ∗ ) .

Remark 13 (Online vs Offline Oracles) . Offline regression is a batch-processing approach that involves analyzing the entire dataset at once to build a regression model. The advantage of offline oracles lies in that they can consider all the data available for training, the training data is generated iid from some underlying distribution and therefore, can often produce a more accurate model. On the contrary, OnReg F requires a much stronger guarantee compared to OffReg F since they are required to fit any arbitrary test instance, even adversarially generated (input,output) pair ( z t , y t ) , whereas an offline oracle only promises that when ( z t , y t ) ∼ D is sampled from the same distribution D where it has been trained. Overall, the choice between offline and online regression oracles depends on the specific use case. If the entire dataset is available and accuracy is the primary concern, then offline regression may be the better approach. However, if real-time processing and adaptability are required, then online regression may be a better choice; although finding efficient online oracles for any arbitrary function class could be hard (e.g. see [23, Example 3], [22, Example 1]). It is also worth noting that an online oracle can always be converted to an offline one but not vice-versa since the former offers a much stronger regression guarantee compared to the latter.

## B Appendix for Section 4

## B.1 Double-Monster (Algorithm 1): Pseudocode of Algorithm for the Best-Response Regret

## B.2 Justification of existence of p in Eq. (2)

Proof. Let us start by recalling the Eq. (2) requires finding a p ∈ ∆ K × K such that:

<!-- formula-not-decoded -->

Towards proving this, let us fix the context x ∈ X and denote by P = ̂ M ( x ) . Then note we essentially need to solve the following minimax game (for any arbitrary preference matrix P ):

<!-- formula-not-decoded -->

## Algorithm 1 Double-Monster : Efficient Contextual Dueling Bandit with Offline Regressors

- 1: input Epoch schedule 0 = τ 0 &lt; τ 1 &lt; τ 2 &lt; · · · . Confidence parameter δ . Tuning parameter γ 1 , γ 2 , . . . .
- 2: Arm set: [ K ] . An instance of OffReg for function class M
- 3: for epoch m = 1 , 2 , . . . do
- 4: Let γ m = ( K √ E off , M ( δ/ (2 m 2 ) , τ m -1 -τ m -2 ) ) (for epoch 1, γ 1 = 1 ).
- 5: Compute ̂ M m ← OffReg ( { ( x τ , a τ , b τ ) , o τ } τ m -1 τ = τ m -2 +1 ) , i.e.

̂ M m = arg min M ∈M ∑ τ m -1 τ = τ m -2 +1 ( M ( x τ )[ a τ , b τ ] -˜ o τ ) 2 via the offline least squares oracle , where ˜ o τ = 2 o τ -1 , for any τ ∈ [ τ m -2 +1 , τ m -1 ] .

- 6: for round t = τ m -1 +1 , · · · , τ m do
- 7: Observe context x t ∈ X .
- 8: Compute p t ∼ Convex-Constraint-Solver ( x t , ̂ M m , γ m ) s.t. p t satisfies:

<!-- formula-not-decoded -->

- 9: Sample ( a t , b t ) ∼ p t and observe preference feedback o t ∼ Ber ( M /star ( x t )[ a t ,b t ]+1 2 ) .
- 10: end for
- 11: end for

Let us denote the minmax value of the above game, associated to the preference matrix P by V ( P ) := min p ∈ ∆ K × K max q , q ′ ∈ ∆ K [ q /latticetop Pp /lscript + q ′/latticetop Pp r + 2 γ ∑ i,j q ( i ) q ′ ( j ) p ( i,j ) ] .

Now fixing /epsilon1 &gt; 0 , for any p ∈ ∆ K × K we define the /epsilon1 -relaxation of p as p ( /epsilon1 ) := (1 -/epsilon1 ) p + /epsilon1 1 /K 2 . As p ( /epsilon1 ) itself is a distribution, this upper bounds our objective while ensuring that the conditions for applying the Sion's minimax theorem are satisfied 3 . As such, we obtain

<!-- formula-not-decoded -->

Now let us define another distribution ˜ p ∈ ∆ K × K such that qq ( i, j ) = q ( i ) q ′ ( j ) ∀ i, j ∈ [ K ] . It is easy to note that qq /lscript = q and qq r = q ′ . Similarly, defining the /epsilon1 -approximation of qq as qq /epsilon1 := (1 -/epsilon1 ) qq + /epsilon1 1 /K 2 , we further note qq /lscript /epsilon1 = (1 -/epsilon1 ) q + /epsilon1 1 /K , and qq r /epsilon1 = (1 -/epsilon1 ) q ′ + /epsilon1 1 /K . Then replacing p by qq in the above expression, we can further upper bound V ( P ) by:

<!-- formula-not-decoded -->

3 Note that the expression is convex in p and quasi-concave in qq ′ , as required, following [6].

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the first inequality restricts the domain for p using the smoothing operator, the first equality is the minimax swap, and the second inequality follows by choosing p = q . The remaining three terms are bounded as follows: (i) the first term is zero since P is a preference matrix, (ii) the second term could trivially be upper bounded by /epsilon1 , and (iii) the third term is at most 4 K 2 /γ as long as /epsilon1 ≤ 1 / 2 . Setting /epsilon1 = K 2 / 2 γ , we obtain the result, as long as γ ≥ K 3 .

## C Appendix for Section 5

## C.1 Proof of Lemmas

Lemma 7 (Properties of decision policy π m in Algorithm 1) . At any epoch m , ∀ π , π ′ ∈ Π , the decision policy of epoch m π m satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma 7. Note by our algorithm selection rule, at any time t in epoch m , our decision policy π m : X ↦→ ∆ K × K (see Definition 2) satisfies:

<!-- formula-not-decoded -->

but this implies that for any policy π ∈ Π and π ′ ∈ Π ,

<!-- formula-not-decoded -->

Further taking expectation over x t ∼ D X along with Definition 3 and Definition 6, above inequality yields:

<!-- formula-not-decoded -->

This proves the first part of the proof since by definition setting π = ψ /star [ π /lscript m , M /star ] , π ′ = ψ /star [ π r m , M /star ] and the fact V m ( π , π ′ ) ≥ 0 .

Additionally, due to the anti-symmetric property of preference matrix P , we have ̂ Reg m ( π /lscript m , π ) = -̂ Reg m ( π , π /lscript m ) and ̂ Reg m ( π r m , π ′ ) = -̂ Reg m ( π ′ , π r m ) (see Remark 3). Applying this which further to Eq. (9) we get:

<!-- formula-not-decoded -->

This proves the second part of the proof.

## C.2 Proof of Lemma 8

Lemma 8. For any epoch m and any two decision policies π and π ′ ,

<!-- formula-not-decoded -->

Proof of Lemma 8. Let us analyze the difference between the true and empirical response regret at any time t :

<!-- formula-not-decoded -->

Let us consider any x ∈ X , and denote by q = π ′ ( x ) , p = π ( x ) , p m -1 = π m -1 ( x ) , m ( t ) = m . Now the term inside the expectation can be further bounded as:

<!-- formula-not-decoded -->

where (a) uses the Cauchy-Schwarz inequality followed by AM-GM inequality. Using similar ideas one can derive:

<!-- formula-not-decoded -->

where the second last equation follows from Assumption 4 and the last equation from the choice of

<!-- formula-not-decoded -->

## C.3 Proof of Lemma 10

Recall the notations introduced in Section 4.1, that will be used throughout the proof.

Lemma 10 (Epochwise Recursion) . Let π be any policy in Π 2 . Then for all epochs m ∈ N + :

<!-- formula-not-decoded -->

Proof of Lemma 10. Let us fix c 0 = 11 for the rest of the proof. We prove the claim via induction on m . We first consider the base case where m = 1 and 1 ≤ t ≤ τ 1 . In this case, since γ 1 = 1 ,

<!-- formula-not-decoded -->

Thus the claim holds in the base case.

For the inductive step, let us fix some epoch m &gt; 1 . We assume that for all epochs m ′ &lt; m , all π ∈ Π 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Part-1: We will first show that for epoch m , and all π ∈ Π 2 ,

<!-- formula-not-decoded -->

For simplicity let us denote the policies ψ /star [ π /lscript , M /star ] , ψ /star [ π r , M /star ] respectively by π /star,/lscript , π /star,r , and ψ /star [ π /lscript , ̂ M m ] , ψ /star [ π r , ̂ M m ] , respectively by π /star,/lscript m , π /star,r m . Then we start by noting that:

<!-- formula-not-decoded -->

where the first inequality is by the optimality of π /star,/lscript m by definition and (a) follows from Lemma 8. Similarly, we can show that

<!-- formula-not-decoded -->

Combining both we get:

<!-- formula-not-decoded -->

Now using Lemma 7 this implies for both π /star and π r , we have:

<!-- formula-not-decoded -->

Now note that Eq. (13) further implies:

<!-- formula-not-decoded -->

Similarly one can show that:

<!-- formula-not-decoded -->

Summing the results from the above two inequalities:

<!-- formula-not-decoded -->

where ( a ) follows from Eq. (10), and ( b ) follows from the fact that BReg( π /star,/lscript ) + BReg( π /star,r ) = 0 by Assumption 3. Combining the above two inequalities with (12):

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Consequently, we get:

<!-- formula-not-decoded -->

finally leading to:

<!-- formula-not-decoded -->

for the choice of c 0 = 11 and recall that we assumed that π is any arbitrary policy. This concludes the first part of the proof.

The second part of the claim can be proved almost following the similar tricks. We add the analysis below for completeness.

Part-2: To prove the second part we start by noting that we can show:

<!-- formula-not-decoded -->

Following a similar analysis from above, we can obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) follows from Eq. (11), ( b ) follows since we have already proved the first part of the induction for epoch m as concluded in Eq. (15), and ( c ) follows from the fact that ̂ BReg m ( π /star,/lscript m ) + ̂ BReg m ( π /star,r m ) = 0 by Assumption 3. Combining this with (16):

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Consequently we have:

<!-- formula-not-decoded -->

finally leading to:

<!-- formula-not-decoded -->

which is again satisfied for any choice of c 0 = 11 , concluding the entire proof.

## C.4 Proof of Theorem 5

Given the result of Lemma 10, we are now ready to prove the final regret bound of Theorem 5:

Theorem 5 (Regret Analysis of Algorithm 1) . Consider any function class M that satisfies Assumption 3. Then under realizability (Assumption 1), and with an epoch schedule τ 1 , . . . , τ m such that τ m ≥ 2 m for m ≤ log T , with probability at least (1 -δ ) for any δ ∈ (0 , 1) the best-response regret (BR -Reg T ) of Double-Monster after T rounds is bounded by:

<!-- formula-not-decoded -->

Proof of Theorem 5. Applying Lemma 10 over the phases m = 1 , . . . , m ( T ) we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) follows from Lemma 10, ( b ) follows from Lemma 7 and the last inequality follows by the choice of γ m in Double-Monster (Algorithm 1). The proof is concluded noting τ 1 = O (1) .

## C.5 Proof of Corollary 6 for Finite M

Proof. In particular for the special case when M is finite, we know it is possible to design offline regression oracles (see [24, Table 1]) such that the offline loss that for any phase m could be bounded by by

<!-- formula-not-decoded -->

as follows from [32]. Now using the derivation of Theorem 5, we can specifically derive for δ = 1 /T :

<!-- formula-not-decoded -->

## D Supplementary for Section 6

## D.1 Regret Analysis of Double-Monster-Inf : Proof Analysis of Theorem 11

As discussed in the algorithm description of Section 6, one of the key contributions of this work lies in the analysis of our proposed method which is the first attempt to address the contextual dueling bandit problem with offline regression. The overall proof structure adapts the same line of argument given in Section 5.2, however, the challenge was to reproduce the equivalent intermediate lemmas (precisely the results of Lemma 8, Lemma 7 and Lemma 10) for the continuous action space. We will discuss the proof of Theorem 11 in detail in the rest of this section. We will find it useful to define some notations for the ease of exposition.

## D.1.1 Additional Concepts towards proving Theorem 11

We will start with defining an important concept of Epsilon-net of K . We will later see the usefulness of this concept while proving Lemma 15 as explained in the associated Remark 17.

Definition 7 (Epsilon-Net of K (w.r.t Euclidean Norm)) . A (finite or countable) set N ε ⊆ κ is called an ε -net of K with respect to the Euclidean norm if

<!-- formula-not-decoded -->

Equivalently,

<!-- formula-not-decoded -->

Definition 8 (Projection to ε -net) . Given any point a ∈ K , we denote the projection of a to its ε -net as P N ε ( a ) := arg min b ∈ N ε ‖ a -b ‖ 2 , where ties are broken arbitrarily.

Note for any a ∈ K , ‖ a -P N ε ( a ) ‖ 2 ≤ ε.

Remark 14. ε ∈ (0 , 1) is a tunable parameter, and we will see in the proof of Theorem 11 below how to choose it optimally.

Definition 9 (Policy Class over K ) . A standard policy class Ψ := { ψ | ψ : X ↦→ ∆( K ) } is a set containing all the set of all probability measures on K . Further we also define by policy class Ψ 2 := { ψ 2 | ψ 2 : X ↦→ ∆( K×K ) } be the set of all probability measures on K×K .

Definition 10 (Policy Class over Epsilon-Net of K ) . A standard policy class Π := { π | π : X ↦→ ∆( N ε ) } is a set containing all the probability distributions over N ε . Further we also define by policy class Π 2 := { π 2 | π 2 : X ↦→ ∆( N ε × N ε ) } be the set of all probability measures on N ε × N ε .

Definition 11 (Decision policy of Epochm ) . At any epoch m and x ∈ X , the decision policy of epoch m is defined as ψ m ∈ Ψ 2 : ψ m ( x ) ← Cont-CvxConstraint-Solver ( x, φ m , γ m ) , ∀ x ∈ X .

Note that at any time t in epoch m , ψ m ( x t ) = p t ∈ ∆( K × K ) is a probability measure over the product space K×K . Similar to Section 5, will further denote by ψ /lscript m ∈ Ψ and ψ r m ∈ Ψ respectively the left and right marginal policies of π m ∈ Ψ 2 . i.e.:

<!-- formula-not-decoded -->

Definition 12 (Approximate Decision policy of Epochm ) . At any epoch m and x ∈ X , the decision policy of epoch m is defined as π m ∈ Π 2 , such that:

<!-- formula-not-decoded -->

.

Similarly we define π /lscript m ∈ Π and π r m ∈ Π respectively to be the left and right marginal of π m ∈ Π 2 Definition 13 ( φ -Best-Response Policy) . Given any scoring function φ ∈ Φ , we define the φ -best response policy ψ φ : X ↦→ K as:

<!-- formula-not-decoded -->

which denotes the best-response (a.k.a. highest-scoring) decision/action for the scoring function φ . Definition 14 (Instantaneous Regret of a Policy) . For any arbitrary underlying preference mapping φ ∈ Φ , we denote the best-response regret of policy ψ ∈ Ψ as:

<!-- formula-not-decoded -->

where ψ φ ( x ) ∈ K is as defined in Definition 13, and with a slight abuse of notation 〈 ψ ( x ) , φ ( x ) 〉 := E a ∼ ψ ( x ) [ 〈 a , φ ( x ) 〉 ] .

Given the above definition, for the true scoring function φ ∗ , we will denote BReg φ ∗ ( ψ ) simply by BReg( ψ ) .

Definition 15 (Approximate φ -Best-Response Policy) . Given any scoring function φ ∈ Φ , we define the approximate φ -best response policy π φ : X ↦→ N ε as:

<!-- formula-not-decoded -->

which denotes the best-response (a.k.a. highest-scoring) decision/action for the scoring function φ within N ε , the ε -net of K .

Definition 16 (Approximate Instantaneous Regret of a Policy) . For any arbitrary underlying preference mapping φ ∈ Φ , we denote the approximate best-response regret of policy π ∈ Π as:

<!-- formula-not-decoded -->

where π φ ( x ) ∈ N ε is as defined in Definition 13, and again with a slight abuse of notation, we denote by 〈 π ( x ) , φ ( x ) 〉 := E a ∼ π ( x ) [ 〈 a , φ ( x ) 〉 ] .

For the true scoring function φ ∗ , we will denote BReg φ ∗ ( π ) simply by BReg( π ) , and for φ = φ m , the estimated feature mapping at epoch m (see Algorithm 2), we will denote BReg φ m ( π ) by ̂ BReg m ( π ) . Further, we will also define the decision variance of any policy π ∈ Π defined as:

Definition 17 (Decision variance of a policy) . For any two policies π 1 , π 2 ∈ Π the decision variance of π 1 , π 2 with respect to another decision policy ˜ π ∈ Π 2 is defined as:

<!-- formula-not-decoded -->

In particular, when ˜ π = π m , where recall that π m is the decision policy of epoch m , we will use the shorthand V m ( π 1 , π 2 ) to denote V ˜ π ( π 1 , π 2 ) . To understand the term from an intuitive level, roughly V ˜ π ( π 1 , π 2 ) indicates the selection variance of policy ˜ π on the random pairs of actions generated by the pair of policies ( π 1 , π 2 ) .

## D.1.2 Proof of Main Theorem 11

Given the above definitions and setting the preliminaries, we are now ready to explain the proof Theorem 11. Let us start with recalling Theorem 11 first:

Theorem 11 (Regret Analysis of Algorithm 2) . Assume ∀ x ∈ X , ‖ φ ∗ ( x ) ‖ ≤ D for some D &gt; 0 , and suppose we choose c = 5 d 64 D 2 γ m ( T ) . Then under realizability (Assumption 1) of the function class Φ , and with an epoch schedule τ 1 , . . . , τ m such that τ m ≥ 2 m for m ≤ log T , with probability at least (1 -δ ) for any δ ∈ (0 , 1) the best-response regret of Algorithm 2, BR -Reg ( cont ) T , in T rounds

<!-- formula-not-decoded -->

Proof of Theorem 11. We start with a generalization of Lemma 7 for continuous decision spaces which establish the empirical performance of Algorithm 2, as given below:

Lemma 15 (Properties of decision policy π m in Algorithm 2) . At any epoch m , the decision policy of epoch m , π m , satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any two policies π and π ′ ∈ Π .

The next key step is to connect the empirical performance of any policy π to its true performance. Our next result carries a similar spirit of Lemma 8, however the proof analysis if very different as follows from our detailed proof analyses in Appendix D.2.

Lemma 16 (Emp vs True Performance) . Let π be any policy in Π 2 . Then for all epochs m ∈ N + :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now the final regret bound of Algorithm 2 now follows by combining the statements of Lemma 15 and Lemma 16, as shown given below:

Given the above results, the proof of Theorem 11 follows combining the results of Lemma 16 and Lemma 15. Applying Lemma 16 over the phases m = 1 , . . . , m ( T ) we get:

<!-- formula-not-decoded -->

where ( a ) follows from Definition 8, ( b ) applies Cauchy-Schwarz and the fact that ‖ φ ∗ ( x ) ‖ ≤ D, ∀ x ∈ X by assumption, ( c ) follows from Lemma 16, ( d ) follows from Lemma 15 and the last inequality follows by the choice of γ m in Double-Monster (Algorithm 2). The proof is concluded noting τ 1 = 1 (by parameter choice) and setting ε = 1 DT .

This gives an overall roadmap of the proof of Theorem 11. We give the detailed proofs of the above lemmas in the following subsection.

## D.2 Proof of Key Lemmas for Theorem 11

## Proof of Lemma 15

Lemma 15 (Properties of decision policy π m in Algorithm 2) . At any epoch m , the decision policy of epoch m , π m , satisfies:

<!-- formula-not-decoded -->

for any two policies π and π ′ ∈ Π .

Proof of Lemma 15. Note that at any epoch m and context x ∈ X , we have:

<!-- formula-not-decoded -->

Fixing a x ∈ X , let ˜ q ∗ ∈ ∆( N ε × N ε ) be the optimal solution of the above optimization. Then consider the Lagrangian of the above optimization and setting the derivative with respect to ˜ q ∗ ( a , b ) to zero, we obtain logdet determinant:

<!-- formula-not-decoded -->

where λ ∈ R and λ ( a , b ) ≥ 0 , are the Lagrangian multipliers. The interesting thing to note is the third term in the above expression which we obtain using the known fact about the derivative of the determinant of any invertible (full rank) matrix A is given by

<!-- formula-not-decoded -->

where Tr ( · ) the trace function of the matrix. For any q ∈ ∆( N ε × N ε ) , using this we get:

<!-- formula-not-decoded -->

where the last equality uses that the trace of a matrix product is invariant under cyclic permutations, i.e. Tr( ABC ) = Tr( CAB ) = Tr( BCA ) given any d × d dimensional matrices A , B , C .

Further, multiplying above by ˜ q ∗ ( a , b ) and summing over all pairs ( a , b ) , we get:

<!-- formula-not-decoded -->

One important observation to note is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third inequality again follows by applying the cyclic permutation invariance property of trace function as describe above. Further the last inequality can be justified through Lemma 19 and noting H (˜ q ∗ ) = ∑ ( a , b ) ∈ ( N ε × N ε ) ˜ q ∗ ( a , b )( a -b )( a -b ) /latticetop + λ I d .

Rearranging the terms in (19) and noting ∑ ( a , b ) ∈ ( N ε × N ε ) ˜ q ∗ ( a , b ) ‖ a -b ‖ 2 H (˜ q ∗ ) -1 ≤ d , we have:

<!-- formula-not-decoded -->

Note the first equality uses complementary slackness and set ∑ ˜ q ∗ ( a, b ) λ ( a , b ) = 0

( a , b ) ∈ ( N ε × N ε ) .

Now an interesting observation is that, setting a = b = π φ m ( x ) in (18) we get: λ = λ ( a, b ) ≥ 0 . But that (20) simply yields:

<!-- formula-not-decoded -->

which proves the first claim.

On the other hand, (20) also implies:

<!-- formula-not-decoded -->

which further implies

<!-- formula-not-decoded -->

since ∑ ( a , b ) ∈ ( N ε × N ε ) ˜ q ∗ ( a , b ) ( 〈 π φ m ( x ) , φ m ( x ) 〉 - 〈 ( a + b ) / 2 , φ m ( x ) 〉 ) ≥ 0 by definition of π φ m . But since λ ( a, b ) ≥ 0 ∀ a, b ∈ ( N ε × N ε ) , (18) further implies:

<!-- formula-not-decoded -->

Since above inequality holds true for any pair ( a , b ) ∈ ( N ε × ε ) , the second claim of Lemma 15 now follows for any two policies π and π ′ ∈ Π by taking expectation over x and replacing a by π ( x ) and b by π ′ ( x ) . Precisely, the last inequality gives:

<!-- formula-not-decoded -->

where the last claim follows noting that ˜ q ∗ is the optimal solution of the initial optimization problem in Eq. (17) and hence we can always set π m ( x ) = ˜ q ∗ for the given context x ∈ X .

<!-- formula-not-decoded -->

Remark 17 (A comment on correct application of KKT conditions) . It is crucial to observe that the proof of Lemma 15 relies on the ε -net construction precisely because it allows us to invoke the Karush-Kuhn-Tucker (KKT) conditions. To apply KKT, the optimization variable must live in a finite -dimensional space; here that variable is q ∈ ∆( N ε × N ε ) , whose dimensionality is finite by definition of the ε -net. In contrast, the original problem over ∆( K × K ) is inherently infinite dimensional, so any Lagrange multipliers would themselves be infinite dimensional, and the standard KKT machinery would break down.

Importantly, introducing the ε -net is purely a proof device; it does not alter the algorithm or its outputs in any way. Clarifying this technical point corrects a gap that appears in several prior contextual bandit papers for continuous action spaces-for example, the proof of [17, Proposition 3] informally applies KKT conditions to an infinite-dimensional variable without addressing the attendant issues of uncountably many Lagrange multipliers! Our analysis closes that loophole by first projecting onto a finite ε -net and only then applying KKT. This requires significant modifications in the proof of regret analysis of Theorem 11 as detailed above.

## Proof of Lemma 16

We will state another important result before proving Lemma 16 that expresses the difference between the (approximate) true regret and empirical regret in terms of decision variance. The formal claim is as follows:

Lemma 18. For any epoch m and policy π ∈ Π

<!-- formula-not-decoded -->

Proof of Lemma 18. Consider any epoch m , and let π φ denote the best (score-maximizing) policy for the scoring function φ , for any φ ∈ Φ (as defined in Definition 14). Then we start by noting that by definition,

<!-- formula-not-decoded -->

where (a) uses Holder's inequality, (b) applies Cauchy's Schwarz and AM-GM inequality. Inequality ( c ) follows noting that

<!-- formula-not-decoded -->

where ( a ) applied Cauchy-Schwarz, and the last equality follows from Definition 17, our choice of γ m = 1 2 √ 2 ( d E off , Φ ∗ ( δ / 2 m 2 ,τ m -1 -τ m -2 ) ) 1 / 2 , and since we set c = 5 d 64 γ m ( T ) D 2 ≤ 5 d 64 γ m D 2 . This completes the first part of the proof. The second part of the proof follows almost with the same argument as shown above, with the additional observation that:

<!-- formula-not-decoded -->

Given the results of Lemma 18, we are now ready to proof Lemma 16.

Lemma 16 (Emp vs True Performance) . Let π be any policy in Π 2 . Then for all epochs m ∈ N + :

<!-- formula-not-decoded -->

Proof of Lemma 16. The proof follows exactly the same as the proof of Lemma 10.

Let us fix c 0 = 6 for the rest of the proof. We prove the claim via induction on m . We first consider the base case where m = 1 and 1 ≤ t ≤ τ 1 . In this case, since γ 1 = 1 ,

<!-- formula-not-decoded -->

Thus the claim holds in the base case.

For the inductive step, let us fix some epoch m &gt; 1 . We assume that for all epochs m ′ &lt; m , all π ∈ Π 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Part-1: We will first show that for epoch m , and all π ∈ Π 2 ,

<!-- formula-not-decoded -->

We start by noting that using Lemma 18 we have:

<!-- formula-not-decoded -->

Similarly, applying Lemma 18 we get that:

<!-- formula-not-decoded -->

Combining the above two inequalities we get:

<!-- formula-not-decoded -->

Now applying Lemma 15 we know that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Summing the results from the above two inequalities, we get:

<!-- formula-not-decoded -->

where ( a ) follows from Eq. (23), and ( b ) follows from the fact that BReg( π φ ∗ ) = 0 by Definition 13. The last inequality follows since by choice γ m -1 ≤ γ m . Combining the above two inequalities with (24) we get:

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Consequently, we get:

<!-- formula-not-decoded -->

finally leading to:

<!-- formula-not-decoded -->

for the choice of c 0 = 6 and recall that we assumed that π is any arbitrary policy. This concludes the first part of the proof.

The second part of the claim can be proved almost following the similar tricks. We add the analysis below for completeness.

Part-2: To prove the second part, we start by noting that, similar to (24), we can show:

<!-- formula-not-decoded -->

Following a similar analysis from above, we can obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) follows from Eq. (23), ( b ) follows since we have already proved the first part of the induction for epoch m as concluded in Eq. (27), and ( c ) follows from the fact that ̂ BReg m ( π φ m ) = 0 by Assumption 3. Combining this with (28):

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Consequently, we have:

<!-- formula-not-decoded -->

finally leading to:

<!-- formula-not-decoded -->

which is again satisfied for any choice of c 0 = 6 , concluding the entire proof.

## D.3 Additional useful results for proving Theorem 11

Lemma 19. Let A be any d -dimensional postive semi-definite matrix and ˜ A ′ = A + c I d for some positive constant c &gt; 0 . Then

<!-- formula-not-decoded -->

Proof of Lemma 19. Consider the eigen-decomposition of A = U /latticetop ΣU = ∑ d i =1 σ i u i u /latticetop i , where σ i ≥ 0 is the i -th eigenvalue of A ( Σ is the diagonal matrix with Σ( i, i ) = σ i ), and U = [ u 1 . . . u d ] is orthogonal matrix with its columns, u 1 , . . . , u d being orthogonal vectors.

By definition of A , it then follows that the eigen-decomposition of ˜ A = ∑ d i =1 ( σ i + c ) u i u /latticetop i and hence ˜ A -1 = ∑ d i =1 1 σ i + c uu /latticetop .

Combining the above insights, we get: ˜ A -1 A = ∑ d i =1 σ i σ i + c u i u /latticetop i represents the eigendecomposition of ˜ A -1 A . This further given Tr( ˜ A -1 A ) ∑ d i =1 σ i σ i + c ≤ d since the trace of a matrix is the sum of its eigenvalues.

## E Supplementary for Section 7: Additional Experiments

## E.1 Running Time Performance

We also report the running time performance of Double-Monster on Inst1 :

T = 10,000 (Inst-1)

<!-- image -->

|   d |   runtime (sec) |
|-----|-----------------|
|   4 |         24.4812 |
|   6 |        117.084  |
|  10 |        176.245  |
|  15 |        501.717  |

We here report the (averaged) runtimes of the above executions (in seconds) with increasing d to check its runtime efficiency of Double-Monster-Inf . We report this for Inst-1 for T = 10 , 000 . Despite large T and d the complete runs of the algorithm are finished within a few minutes, which demonstrates its computational efficiency. The experiments are run on a standard MacBook Pro (36GB RAM).

## NeurIPS Checklist:

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In the abstract, we list the main claims of this paper in a general fashion. Then, in the introduction we state them in more detail. They accurately reflect the paper's contribution and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations and assumptions of our work throughout the paper. Additionally, they are also highlighted in the discussions/remarks.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The assumptions can be found in Sections 2 and 3 and in the paragraphs before the theorems. We provide proof sketches in the main text and complete proofs in the appendix.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We explain the experiment setup in detail in the submission draft.

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

Justification:

Guidelines: Our experiments are performed on synthetically generated data, and all details are provided in the submission draft.

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All details are mentioned in the experiments section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Details are provided in the experiments section.

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

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: [NA]

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We've carefully read the code of ethics and our draft adheres to it.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The contributions are predominantly theoretical with no societal impacts. Thus, this point is not applicable.

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

Justification: [NA]

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

## Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [No]

Justification: Contributions are theoretical and all proofs derived by the authors.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.