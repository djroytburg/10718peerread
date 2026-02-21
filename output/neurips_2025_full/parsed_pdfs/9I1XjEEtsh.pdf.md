## Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor Critic Algorithm

## Yang Xu

## ∗

Purdue University, USA xu1720@purdue.edu

## Washim Uddin Mondal

Indian Institute of Technology Kanpur wmondal@iitk.ac.in

## Swetha Ganesh ∗

Purdue University, USA Indian Institute of Science, Bengaluru, India swethaganesh@iisc.ac.in

## Qinbo Bai

Purdue University, USA bai113@purdue.edu

## Abstract

This paper investigates infinite-horizon average reward Constrained Markov Decision Processes (CMDPs) under general parametrized policies with smooth and bounded policy gradients. We propose a Primal-Dual Natural Actor-Critic algorithm that adeptly manages constraints while ensuring a high convergence rate. In particular, our algorithm achieves global convergence and constraint violation rates of ˜ O (1 / √ T ) over a horizon of length T when the mixing time, τ mix , is known to the learner. In absence of knowledge of τ mix , the achievable rates change to ˜ O (1 /T 0 . 5 -/epsilon1 ) provided that T ≥ ˜ O ( τ 2 //epsilon1 mix ) . Our results match the theoretical lower bound for Markov Decision Processes and establish a new benchmark in the theoretical exploration of average reward CMDPs.

## 1 Introduction

Reinforcement Learning (RL) is a paradigm where an agent learns to maximize its average reward in an unknown environment through repeated interactions. RL finds applications in diverse domains such as transportation, communication networks, robotics, and epidemic control [5, 28, 32, 14, 29]. Among various RL settings, the infinite horizon average reward setup is of particular significance for modeling long-term objectives in practical scenarios. This setting is critical as it aligns with real-world applications requiring persistent and consistent performance over time.

In many applications, agents must operate under certain constraints. For instance, in transportation networks, delivery times must adhere to specified windows; in communication networks, resource allocations must stay within budget limits. Constrained Markov Decision Processes (CMDPs) effectively incorporate these constraints by introducing a cost function alongside the reward function. In average reward CMDPs, agents aim to maximize the average reward while ensuring that the average cost does not exceed a predefined threshold, making them crucial for scenarios where safety, budget, or resource constraints are paramount.

Finding optimal policies for average reward CMDPs is challenging, especially if the environment is unknown. The efficiency of a solution to the CMDP is measured by the rate at which the average global optimality error and the constraint violation diminish as a function of the length of the horizon T . Although many existing works focus on the tabular policies [24, 3, 2, 15], these solutions do not apply to real-life scenarios where the state space is large or infinite.

∗ Equal contribution.

Vaneet Aggarwal Purdue University, USA vaneet@purdue.edu

Table 1: This table summarizes the different model-based and model-free state-of-the-art algorithms available in the literature for average reward CMDPs and their results for global convergence rate and average constraint violation. The bounds of global convergence describe convergence to the best performance permitted by the chosen features and policy class; the residual approximation floor is fixed (independent of T ), and the listed rates govern how fast the statistical gap decays toward that floor. General parameterization refers to parameterizations whose policy score ∇ θ log π θ ( a | s ) is uniformly bounded and Lipschitz in θ , together with a Fisher non-degeneracy condition.

| Algorithm                 | Global Convergence             | Violation                      | Mixing Time Unknown   | Model-free   | Setting                  |
|---------------------------|--------------------------------|--------------------------------|-----------------------|--------------|--------------------------|
| Algorithm 1 in [15]       | ˜ O ( 1 / √ T )                | ˜ O ( 1 / √ T )                | No                    | No           | Tabular                  |
| Algorithm 3 in [15]       | ˜ O ( 1 /T 1 / 3 )             | ˜ O ( 1 /T 1 / 3 )             | Yes                   | No           | Tabular                  |
| UC-CURL and PS-CURL [2]   | ˜ O ( 1 / √ T )                | 0                              | Yes                   | No           | Tabular                  |
| Algorithm 2 in [27]       | ˜ O ( 1 /T 1 / 4 )             | ˜ O ( 1 /T 1 / 4 )             | Yes                   | -            | Linear MDP               |
| Algorithm 3 in [27]       | ˜ O ( 1 / √ T )                | ˜ O ( 1 / √ T )                | No                    | -            | Linear MDP               |
| Triple-QA [42]            | ˜ O ( 1 /T 1 / 6 )             | 0                              | Yes                   | Yes          | Tabular                  |
| Algorithm 1 in [10]       | ˜ O ( 1 /T 1 / 5 )             | ˜ O ( 1 /T 1 / 5 )             | No                    | Yes          | General Parameterization |
| This paper (Theorem 4.9)  | ˜ O ( 1 / √ T )                | ˜ O ( 1 / √ T )                | No                    | Yes          | General Parameterization |
| This paper (Theorem 4.10) | ˜ O ( 1 /T 0 . 5 - /epsilon1 ) | ˜ O ( 1 /T 0 . 5 - /epsilon1 ) | Yes 2                 | Yes          | General Parameterization |
| Lower bound [6]           | Ω ( 1 / √ T )                  | N/A                            | N/A                   | N/A          | N/A                      |

General parametrization offers a useful approach for dealing with such scenarios. It indexes policies via finite-dimensional parameters, which makes them suitable for large state space CMDPs. However, as exhibited in Table 1, the current state-of-the-art algorithms for average reward CMDPs with general parametrization achieve a convergence rate of ˜ O (1 /T 1 / 5 ) [10], which is far from the theoretical lower bound of Ω(1 / √ T ) . This significant gap highlights the need for improved algorithms that can achieve theoretical optimality.

Challenges and contributions: In this paper, we propose a primal-dual-based natural actor-critic algorithm that achieves global convergence and constraint violation rates of ˜ O (1 / √ T ) over a horizon of length T if the mixing time, τ mix , is known. On the other hand, in the absence of knowledge of τ mix , the achievable rates change to ˜ O (1 /T 0 . 5 -/epsilon1 ) , provided that the horizon length T ≥ ˜ O ( τ 2 //epsilon1 mix ) . Therefore, even with unknown τ mix , the achieved rates can be driven arbitrarily close to the optimal one by choosing small enough /epsilon1 , albeit at the cost of a large T . Both results significantly improve upon the state-of-the-art convergence rate of ˜ O (1 /T 1 / 5 ) in [10] for general policy parametrization and match the theoretical lower bound.

Since CMDP does not have strong convexity in the policy parameters with general parametrization, directly applying the primal-dual approach does not lead to optimal guarantees. The lack of strong convexity prevents the convergence result for the dual problem from automatically translating to the primal problem. This issue is reflected in the current state of art convergence rate in average reward CMDPs in [10], which is ˜ O (1 /T 1 / 5 ) using a primal-dual approach, whereas the unconstrained counterpart in [11] on which the above is built upon achieves a convergence rate of ˜ O (1 /T 1 / 4 ) . Thus, the previous literature for CMDP in general parametrized setups shows a gap in the results of the unconstrained and constrained setups.

It is evident that directly adding a primal-dual structure to an existing actor-critic algorithm, such as in [23], does not result in ˜ O (1 / √ T ) convergence rate for CMDP with general parametrization. This is specifically due to the existence of dual learning rate β , which eventually becomes one of the dominant terms in the convergence rate of the Lagrange function. If the dual learning rate is too low, the constraint violation converges very slowly. However, if β is too high, the primal updates may

2 The convergence result holds under the condition that T ≥ ˜ O ( τ 2 //epsilon1 mix ) , where τ mix is the mixing time.

exhibit high variance, slowing down convergence to the optimal policy. To navigate this tradeoff, it is crucial to carefully tune the problem parameters.

Our algorithm has a nested loop structure where the outer loop of length K updates the primal-dual parameters, and the inner loops of length H run the natural policy gradient (NPG) and optimal critic parameter finding subroutines. We show that, to achieve the optimal rates, the parameter H should be nearly constant while K should be ˜ Θ( T ) . This adjustment makes our algorithm resemble a singletimescale algorithm. In contrast, for unconstrained MDPs, K and H are typically set to Θ( √ T ) [23]. Achieving O (1 / √ T ) convergence in our setting requires ensuring that the bias remains sufficiently small despite a significantly low value of H . Achieving this challenging goal is made possible due to the use of Multi-level Monte Carlo (MLMC)-based estimates in the inner loop subroutines and a sharper analysis of these updates. One benefit of using MLMC-based estimates is that it allows us to design an algorithm that achieves near-optimal global convergence and constraint violation rates without knowledge of the mixing time. Related unconstrained average-reward results have also used MLMC to avoid mixing-time oracles [38, 34, 23]; here we apply this idea to CMDPs via a primal-dual natural actor-critic approach and provide global-rate and average-violation guarantees. We would like to emphasize that our work is the first in the general parameterized CMDP literature to achieve this feat.

## 2 Formulation

Consider an infinite-horizon average reward constrained Markov Decision Process (CMDP) denoted as M = ( S , A , r, c, P, ρ ) where S is the state space, A is the action space of size A , r : S × A → [0 , 1] represents the reward function, c : S × A → [ -1 , 1] denotes the constraint cost function, P : S × A → ∆( S ) is the state transition function where ∆( S ) denotes the probability simplex over S , and ρ ∈ ∆( S ) indicates the initial distribution of states. A policy π : S → ∆( A ) maps a state to an action distribution. The average reward and average constraint cost of a policy, π , denoted as J π r and J π c respectively, are defined as

<!-- formula-not-decoded -->

The expectations computed are over the distribution of all π -induced trajectories { ( s t , a t ) } ∞ t =0 where a t ∼ π ( s t ) , and s t +1 ∼ P ( s t , a t ) , ∀ t ∈ { 0 , 1 , · · · } . For simplifying the notation, we drop the dependence on ρ when there is no confusion. We aim to maximize the average reward while ensuring that the average cost exceeds a given threshold. Without loss of generality, we can formulate this as:

<!-- formula-not-decoded -->

When the underlying state space, S is large, the above problem becomes difficult to solve because the search space of the policy, π , increases exponentially with |S × A| . We, therefore, consider a class of parametrized policies, { π θ | θ ∈ Θ } that indexes the policies by a d -dimensional parameter, θ ∈ R d where d /lessmuch |S||A| . The original problem in (2) can then be reformulated as follows.

<!-- formula-not-decoded -->

In the remaining article, we use J π θ g = J g ( θ ) for g ∈ { r, c } for simplifying the notation. We assume the optimization problem in (3) obeys the Slater condition, which ensures the existence of an interior point solution. This assumption is commonly used in model-free average-reward CMDPs [42, 10].

Assumption 2.1 (Slater condition) . There exists a δ ∈ (0 , 1) and ¯ θ ∈ Θ such that J c ( ¯ θ ) ≥ δ .

We now make the following assumption on the CMDP.

Assumption 2.2. The CMDP M is ergodic, i.e., the Markov chain, { s t } t ≥ 0 , induced under every policy π , is irreducible and aperiodic.

Note that most works on average reward MDPs and CMDPs with general parameterizations, to the best of our knowledge, assume ergodicity [10, 22, 11]. If CMDP M is ergodic, then ∀ θ ∈ Θ , there exists a ρ -independent unique stationary distribution d π θ ∈ ∆( S ) , that obeys P π θ d π θ = d π θ where P π θ ( s, s ′ ) = ∑ a ∈A π θ ( a | s ) P ( s ′ | s, a ) , ∀ s, s ′ ∈ S . Since ∀ π , J π g = ∑ s,a g ( s, a ) π ( a | s ) d π ( s ) , the assumption of ergodicity also ensures that J π g is independent of ρ , ∀ g ∈ { r, c } . Next, we define the mixing time of an ergodic CMDP.

Definition 2.3. The mixing time of a CMDP M with respect to a policy parameter θ is defined as, τ θ mix := min { t ≥ 1 ∣ ∣ ∣ ∣ ‖ ( P π θ ) t ( s, · ) -d π θ ‖ ≤ 1 4 , ∀ s ∈ S } . We also define τ mix := sup θ ∈ Θ τ θ mix as the overall mixing time. In this paper, τ mix is finite due to ergodicity.

The mixing time of an MDP measures how fast the MDP reaches its stationary distribution when executing a fixed policy. We use a primal-dual actor-critic approach to solve (3). Before proceeding further, we define a few terms. The action-value function corresponding to a policy π θ is given as

<!-- formula-not-decoded -->

where g ∈ { r, c } and ( s, a ) ∈ S × A . We further write their corresponding state value function as

<!-- formula-not-decoded -->

The Bellman's equation can be expressed as follows [35] ∀ ( s, a ) ∈ S × A , and g ∈ { r, c } .

<!-- formula-not-decoded -->

We define the advantage term for the reward and cost functions as follows A π θ g ( s, a ) ≜ Q π θ g ( s, a ) -V π θ g ( s ) where g ∈ { r, c } , ( s, a ) ∈ S×A . With the above notations, we now present the commonly-used policy gradient theorem established by [39] for g ∈ { r, c } .

<!-- formula-not-decoded -->

The above term is useful in policy gradient-type algorithms where the learning direction of the parameter θ is given by {∇ θ J g ( θ ) } g ∈{ r,c } . In this paper, however, we will be interested in the Natural Policy Gradients { ω ∗ g,θ } g ∈{ r,c } defined as follows.

<!-- formula-not-decoded -->

where F ( θ ) ∈ R d × d is the Fisher information matrix, which is formally defined as: F ( θ ) = E ( s,a ) ∼ ν π θ [ ∇ θ log π θ ( a | s ) ⊗∇ θ log π θ ( a | s )] where ν π θ ( s, a ) ≜ d π θ ( s ) π θ ( a | s ) , ∀ ( s, a ) , and ⊗ defines the outer product. Note that NPG is similar to PG except modulated by the Fisher matrix, which accounts for the rate of change of policies with θ . The NPG direction ω ∗ g,θ can also be expressed as a solution to the following strongly convex optimization problem.

<!-- formula-not-decoded -->

The above formulation allows one to calculate the NPG in a gradient-based iterative procedure. In particular, the gradient of f g ( θ, · ) is obtained as ∇ ω f g ( θ, ω ) = F ( θ ) ω -∇ θ J g ( θ ) .

## Algorithm 1 Primal-Dual Natural Actor-Critic (PDNAC)

- 1: Input: Initial parameters θ 0 , ω 0 = 0 , ξ 0 = 0 , λ 0 = 0 , policy update stepsize α , dual update stepsize β , NPG update stepsize γ ω , critic update stepsize γ ξ , initial state s 0 ∼ ρ ( · ) , outer loop size K , inner loop size H , T max 2: for k = 0 , · · · , K -1 do 3: ω k g, 0 ← ω 0 , ξ k g, 0 ← ξ 0 ∀ g ∈ { r, c } ; 4: /* Critic Subroutine */ 5: for h = 0 , · · · , H -1 do 6: s 0 kh ← s 0 , P k h ∼ Geom (1 / 2) 7: l kh ← (2 P k h -1) 1 (2 P k h ≤ T max ) + 1 8: for t = 0 , . . . , l kh -1 do 9: Take action a t kh ∼ π θ k ( ·| s t kh ) 10: Observe s t +1 kh ∼ P ( ·| s t kh , a t kh ) 11: Observe g ( s t kh , a t kh ) , g ∈ { r, c } 12: end for 13: s 0 ← s l kh kh 14: Update ξ k g,h using (18) and (20). 15: end for 16: /* Actor Subroutine */ 17: for h = 0 , · · · , H -1 do 18: s 0 kh ← s 0 , Q k h ∼ Geom (1 / 2) 19: l kh ← (2 Q k h -1) 1 (2 Q k h ≤ T max ) + 1 20: for t = 0 , . . . , l kh -1 do 21: Take action a t kh ∼ π θ k ( ·| s t kh ) 22: Observe s t +1 kh ∼ P ( ·| s t kh , a t kh ) 23: Observe g ( s t kh , a t kh ) , g ∈ { r, c } 24: end for 25: s 0 ← s l kh kh 26: Update ω k g,h using (21) and (23) 27: end for 28: ξ k g ← ξ k g,H , ω k g ← ω k g,H , g ∈ { r, c } 29: ω k ← ω k r + λ k ω k c 30: Update ( θ k , λ k ) using (12) 31: end for

## 3 Algorithm

We solve (3) via a primal-dual algorithm based on the following saddle point optimization.

<!-- formula-not-decoded -->

The term L ( · , · ) is defined as the Lagrange function, and λ is the Lagrange multiplier. Our algorithm aims updating ( θ, λ ) by the policy gradient iteration ∀ k ∈ { 0 , · · · , K -1 } as shown below from an arbitrary initial point ( θ 0 , λ 0 = 0) .

<!-- formula-not-decoded -->

where α and β denote primal and dual learning rates respectively, and δ indicates the Slater parameter introduced in Assumption 2.1. Moreover, for any Λ ⊂ R , P Λ indicates the projection onto Λ . Since ∇ θ L ( θ k , λ k ) , F ( θ k ) , and J c ( θ k ) are not exactly computable due to lack of knowledge of the exact transition kernel, P , and thus, that of the occupancy measure, ν π θ k , in most RL scenarios, we use the following approximate updates.

<!-- formula-not-decoded -->

where ω k is the estimate of the NPG F ( θ k ) -1 ∇ θ L ( θ k , λ k ) , while η k c estimates J c ( θ k ) . Below, we discuss the detailed procedure to compute these estimates.

## 3.1 Estimation Procedure

We formally characterize our detailed algorithm in Algorithm 1, which utilizes a Multi-Level Monte Carlo (MLMC)-based Actor-Critic algorithm to compute the estimates stated above. For the exposition of our algorithm, it is beneficial to first discuss the critic estimation procedure, where the goal is to estimate the value functions and the average reward and cost functions. These estimates are then further used to obtain the NPG estimates, which are, in turn, used to update the policy parameter. The algorithm runs in K epochs (also called outer loops ). At the start of k th epoch, the primal and dual parameters are denoted as θ k , λ k respectively.

## 3.1.1 Critic Estimation

At the k th epoch, one of the tasks in critic estimation is to obtain an estimate of J g ( θ k ) , g ∈ { r, c } . Note that J g ( θ k ) can be written as a solution to the following optimization.

<!-- formula-not-decoded -->

The second task is to estimate the value function V π θ k g . To facilitate this objective, it is assumed that, ∀ g ∈ { r, c } , the value function V π θ k g ( · ) is well-approximated by a linear critic function ˆ V g ( ζ θ k g , · ) := 〈 φ g ( · ) , ζ θ k g 〉 where φ g : S → R m is a feature mapping with the property that ‖ φ g ( s ) ‖ ≤ 1 , ∀ s ∈ S , and ζ θ k g ∈ R m is a solution of the following optimization.

<!-- formula-not-decoded -->

Note that the gradients of the functions R g ( θ k , · ) , E g ( θ k , · ) can be obtained as follows.

<!-- formula-not-decoded -->

where ( a ) results from Bellman's equation (6). Since the above gradients cannot be exactly obtained due to lack of knowledge of the transition model P , and thus that of ν π θ k g , we execute the following H inner loop steps to compute approximations of J g ( θ k ) and ζ θ k g from η k g, 0 = 0 and ζ k g, 0 = 0 .

<!-- formula-not-decoded -->

where c γ is a constant, γ ξ defines a learning parameter, and h ∈ { 0 , · · · , H -1 } . Moreover, the terms ˆ ∇ η R g ( θ k , η k g,h ) and ˆ ∇ ζ E g ( θ k , ξ k g,h ) indicate estimates of ∇ η R g ( θ k , η k g,h ) and ∇ ζ E g ( θ k , ζ k g,h ) respectively where ξ k g,h ≜ [ η k g,h , ( ζ k g,h ) /latticetop ] /latticetop . Because the expression of ∇ ζ E g ( θ k , ζ k g,h ) comprises ζ k g,h and J g ( θ k ) (refer (16)), its estimate is a function of η k g,h and ζ k g,h . At the end of inner loop iterations, we obtain η k g,H and ζ k g,H which are estimates of J g ( θ k ) and ζ θ k g respectively. It remains to see how the gradient estimates used in (17) can be obtained. Note that (17) can be compactly written as

<!-- formula-not-decoded -->

For a given pair ( k, h ) , let T kh = { ( s t kh , a t kh , s t +1 kh ) } l kh -1 t =0 indicate a π θ k -induced trajectory of length l kh = 2 Q k h where Q k h ∼ Geom(1 / 2) . Following (15) and (16), an estimate of v g ( θ k , ξ k g,h ) based on a single state transition sample z j kh = ( s j kh , a j kh , s j +1 kh ) can be obtained as

<!-- formula-not-decoded -->

Applying MLMC-based estimation, the term v g ( θ k , ξ k g,h ) is finally computed as

<!-- formula-not-decoded -->

where T max is a constant, v j g,kh = 2 -j ∑ 2 j -1 t =0 v g ( θ k , ξ k g,h ; z t kh ) , j ∈ { 0 , Q k h -1 , Q k h } . Note that the maximum number of state transition samples utilized in the MLMC estimate is T max . Moreover, it can be demonstrated that the samples used on average is ˜ O (log T max ) . The advantage of the MLMC estimator is that it achieves the same bias as averaging T max samples, but requires only ˜ O (log T max ) samples. In addition, since drawing from a geometric distribution does not require knowledge of the mixing time, we can eliminate the mixing time knowledge assumption used in previous works [11, 22, 10]. Furthermore, these previous works utilize policy gradient methods and require saving the trajectories of length H for gradient estimations, while our approach does not. Therefore, our algorithm reduces the memory complexity by a factor of H , which is significant since the choice of H in these works scales with mixing and hitting times of the MDP.

## 3.1.2 Natural Policy Gradient (NPG) Estimator

Recall that the outcome of the critic estimation at the k th epoch is ξ k g,H = [ η k g,H , ( ζ k g,H ) /latticetop ] /latticetop where η k g,H , ζ k g,H estimate J g ( θ k ) and the critic parameter ζ θ k g respectively. For simplicity, we will denote ξ k g,H as ξ k g = [ η k g , ( ζ k g ) /latticetop ] /latticetop . We estimate the NPG ω ∗ g,θ k (refer to the definition (8)) using a H step inner loop as stated below ∀ h ∈ { 0 , · · · , H -1 } starting from ω k g, 0 = 0 .

<!-- formula-not-decoded -->

where ˆ ∇ ω f g ( θ k , ω k g,h , ξ k g ) is an MLMC-based estimate of ∇ ω f g ( θ k , ω k g,h ) where f g is given in (9). To obtain this estimate, a π θ k -induced trajectory T kh = { ( s t kh , a t kh , s t +1 kh ) } l kh -1 t =0 of length l kh = 2 P k h is considered where P k h ∼ Geom(1 / 2) . For a certain transition z j kh = ( s j kh , a j kh , s j +1 kh ) , define the

following estimate.

<!-- formula-not-decoded -->

where the advantage estimate used in ( a ) is essentially a temporal difference (TD) error. Notice that the estimate of the policy gradient ∇ θ J g ( θ k ) depends on ξ k g obtained in the critic estimation process. The MLMC-based estimate, therefore, can be obtained as follows.

<!-- formula-not-decoded -->

where u j g,kh = 2 -j ∑ 2 j -1 t =0 ˆ ∇ ω f g ( θ k , ω k g,h , ξ k g ; z t kh ) , j ∈ { 0 , P k h -1 , P k h } . The above estimate is applied in (21), which finally yields the NPG estimate ω k g,H . For simplicity, we denote ω k g,H as ω k g .

## 3.2 Primal and Dual Updates

The estimates ω k g , g ∈ { r, c } obtained in section 3.1.2 can be combined to compute ω k = ω k r + λ k ω k c . Moreover, section 3.1.1 provides η k c which is an estimate of J c ( θ k ) . At the k th epoch, these estimates can be used to update the policy parameter θ k and the dual parameter λ k following (12).

## 4 Global Convergence Analysis

We first state some assumptions that we will be using before proceeding to the main results. Define A g ( θ ) = E θ A g ( θ ; z ) where the expectation is over the distribution of z = ( s, a, s ′ ) where ( s, a ) ∼ ν π θ g , s ′ ∼ P ( s, a ) , and the term A g ( θ ; z ) is defined in (19). Similarly, b g ( θ ) ≜ E θ [ b g ( θ ; z )] . Let ξ ∗ g ( θ ) = [ A g ( θ )] -1 b g ( θ ) = [ η ∗ g ( θ ) , ζ ∗ g ( θ )] . With these notations, we are ready to state the following assumptions regarding critic approximations.

Assumption 4.1. For g ∈ { r, c } , we define the worst-case critic approximation error to be /epsilon1 app g = sup θ E s ∼ d π θ [ ( ζ ∗ g ( θ )) /latticetop φ g ( s ) -V π θ g ( s ) ] 2 . We assume /epsilon1 app ≜ max { /epsilon1 app r , /epsilon1 app c } to be finite.

Assumption 4.2. There exists λ &gt; 0 such that E θ [ φ g ( s ) ( φ g ( s ) -φ g ( s ′ ))] -λI is positive semidefinite ∀ θ ∈ Θ and g ∈ { r, c } .

Both Assumptions 4.1 and 4.2 are frequently used in analyzing actor-critic methods [13, 47, 37, 44, 38]. Assumption 4.1 intuitively relates to the quality of the feature mapping where /epsilon1 app serves as a measure of this quality: well-crafted features result in small /epsilon1 app , whereas poorly designed features lead to a larger worst-case error. On the other hand, Assumption 4.2, is essential for ensuring the convergence of the critic updates. It can be shown that Assumption 4.2 ensures that the matrix A g ( θ ) is invertible for sufficiently large c γ (details in the appendix).

Assumption 4.3. Define the following function for θ, ω ∈ R d , λ ≥ 0 , and ν ∈ ∆( S × A ) .

<!-- formula-not-decoded -->

Let ω ∗ θ,λ = arg min ω ∈ R d L ν π θ ( ω, θ, λ ) . It is assumed that L ν π ∗ ( ω ∗ θ,λ , θ, λ ) ≤ /epsilon1 bias for θ ∈ Θ and λ ∈ [0 , 2 /δ ] where π ∗ is the solution to the optimization (2), and /epsilon1 bias is a positive constant. The LHS of the above inequality is called the transferred compatible function approximation error . Note that it can be easily verified that ω ∗ θ,λ is the NPG update direction ω ∗ θ,λ = F ( θ ) -1 ∇ θ L ( θ, λ ) .

Assumption 4.4. For all θ, θ 1 , θ 2 ∈ Θ and ( s, a ) ∈ S ×A , the following holds for some G 1 , G 2 &gt; 0 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumption 4.5 (Fisher non-degenerate policy) . There exists a constant µ &gt; 0 such that F ( θ ) -µI d is positive semidefinite, where I d denotes an identity matrix of dimension d .

Comments on Assumptions 4.3-4.5: We emphasize that these assumptions are commonly applied in the policy gradient (PG) literature [30, 1, 33, 43, 19]. The term /epsilon1 bias reflects the parametrization capacity of π θ . For π θ using the softmax parametrization, we directly have /epsilon1 bias = 0 [1]. However, when π θ employs a restricted parametrization that does not encompass all stochastic policies, /epsilon1 bias is greater than zero. It is known that /epsilon1 bias remains very small when utilizing rich neural parametrizations [40]. Assumption 4.4 requires that the score function is both bounded and Lipschitz continuous, which is a condition frequently assumed in the analysis of PG-based methods [30, 1, 33, 43, 19]. Assumption 4.5 ensures that the function f g ( θ, · ) is µ -strongly convex by mandating that the eigenvalues of the Fisher information matrix are bounded from below. This is also a standard assumption for deriving global complexity bounds for PG based algorithms [30, 46, 7, 19]. Recent studies have shown that Assumptions 4.4-4.5 hold true in various examples, including Gaussian policies with linearly parameterized means and certain neural parametrizations [30, 19].

Before proving results for the convergence rate, we first provide the following lemma to associate the global convergence rate of the Lagrange function to the convergence of the actor and critic parameters. Similar ideas have been explored in [23, 10].

Lemma 4.6. If the policy parameters, { ( θ k , λ k ) } K k =1 are updated via (12) and assumptions 4.3-4.5 hold, then the following inequality is satisfied

<!-- formula-not-decoded -->

where KL ( ·‖· ) is the Kullback-Leibler divergence, π ∗ is the optimal policy for (2) and ω ∗ k := ω ∗ θ k ,λ k is the exact NPG direction F ( θ k ) -1 ∇ θ L ( θ k , λ k ) . Finally, E k denotes conditional expectation given history up to the k th iteration.

Observe the presence of /epsilon1 bias in (25). It dictates that due to the incompleteness of the policy class, the global convergence bound cannot be driven to zero. Note that the last term in (25) is O (1 / ( αK )) because the term E s ∼ d π ∗ [ KL ( π ∗ ( ·| s ) ‖ π θ 0 ( ·| s ))] is a constant. The term related to E ‖ ω k ‖ 2 can be further decomposed as follows.

<!-- formula-not-decoded -->

where ( a ) follows from Assumption 4.5 and the definition that ω ∗ k = F ( θ k ) -1 ∇ θ L ( θ k , λ k ) . We can obtain a global convergence bound by bounding the terms E ‖ ω k -ω ∗ k ‖ 2 , E ‖ ( E [ ω k | θ k ] -ω ∗ k ) ‖ and E ‖∇ θ L ( θ k , λ k ) ‖ 2 . The first two terms are the variance and bias of the NPG estimator, ω k , and the third term indicates the local convergence rate. Further, E ‖∇ θ L ( θ k , λ k ) ‖ 2 can be upper bounded by a constant (Lemma G.2 in the appendix). We now provide the convergence result for the actor and critic parameters. For brevity, we use ≲ to denote ≤ ˜ O ( · ) .

Theorem 4.7. Consider Algorithm 1 and let Assumptions 2.2-4.5 hold. If J g is L -smooth, g ∈ { r, c } , γ ω = 2 log T µH is such that γ ω ≤ µ 4(6 G 4 1 τ mix log T max +2 G 2 1 τ 2 mix log T max ) , and T max obeys T max ≥ 8 G 4 1 τ mix µ where µ is defined in Assumption 4.5, the following inequalities hold ∀ k ∈ { 0 , · · · , K -1 } .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ω ∗ g,k is the NPG direction F ( θ k ) -1 ∇ θ J g ( θ k ) , ξ ∗ g ( θ k ) is defined in section 4, and E k denotes conditional expectation given history up to the k th iteration.

Theorem 4.7 bounds the NPG bias ‖ E k [ ω k g ] -ω ∗ g,k ‖ and the second-order error E k [ ‖ ω k g -ω ∗ g,k ‖ 2 ] in terms of the critic approximation error /epsilon1 app , and the bias ‖ E k [ ξ k g ] -ξ ∗ g ( θ k ) ‖ and the second-order error E k [ ‖ ξ k g -ξ ∗ g ( θ k ) ‖ 2 ] in the critic parameter estimation. The following theorem provides bounds on these latter quantities.

Theorem 4.8. Consider Algorithm 1 and let Assumptions 2.2-4.5 hold and g ∈ { r, c } . If we choose γ ξ = 2 log T λH such that γ ξ ≤ λ 24 c 2 γ τ mix log T max while T max obeys T max ≥ 8 c 2 γ τ mix λ where λ is defined in Assumption 4.2, the following inequalities hold ∀ k ∈ { 0 , · · · , K -1 } .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ξ ∗ g ( θ k ) is defined in section 4 and E k denotes conditional expectation given history up to the k th iteration.

Invoking the bounds provided by Theorem 4.7 and 4.8 into Lemma 4.6, we can obtain a convergence rate of the Lagrange function. Our next goal is to segregate the objective convergence and constraint violation rates from this Lagrange error. This is achieved by the following theorems. Depending on whether we have access to the mixing time, two similar but slightly different results can be obtained.

Theorem 4.9. Consider the same setup and parameters as in Theorem 4.8 and Theorem 4.7 and set α = T -1 / 2 , β = T -1 / 2 . If τ mix is known, set H = ˜ Θ( τ 2 mix ) with K = T/H . We have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 4.10. Consider the same setup and parameters as in Theorem 4.8 and Theorem 4.7 and set α = T -1 / 2 , β = T -1 / 2 , H = T /epsilon1 and K = T 1 -/epsilon1 . With T /epsilon1 ≥ ˜ Θ( τ 2 mix ) , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 4.9 dictates that one can achieve both the objective convergence and the constraint violation rates as ˜ O ( T -1 / 2 ) up to some additive factors of /epsilon1 bias and /epsilon1 app where T is the length of the horizon. However, knowledge of the mixing time, τ mix , is needed in this case to set some parameters. On the other hand, Theorem 4.10 states that, if knowledge of τ mix is unavailable, one can achieve objective convergence and constraint violation rates as ˜ O ( T -1 / 2+ /epsilon1 ) as long as the horizon T exceeds ˜ Θ( τ 2 //epsilon1 mix ) . Note that one can get arbitrarily close to the optimal rate of ˜ O ( T -1 / 2 ) by choosing arbitrarily small /epsilon1 . The caveat is that the smaller the /epsilon1 , the larger the horizon length needed to reach the desired rates.

In the setting of Theorem 4.10, for small /epsilon1 , the horizon requirement becomes T ≳ ˜ O ( τ 2 /ε mix ) , which can be large for slowly mixing problems. In such a scenario, one can switch to the parameter setting of Theorem 4.9, which does not impose any such restriction on T . However, since this setup requires the knowledge of τ mix , one can treat τ mix as one of the unknown hyperparameters of the algorithm, and fine-tune it during the training phase. This is in line with other RL algorithms in the literature that also require fine-tuning of several unknown hyperparameters, such as the Lipschitz constant or hitting time [11]. Despite these practical solutions, we acknowledge that a systematic theoretical investigation is

needed to improve the requirement of the horizon length, T (in the absence of knowledge of τ mix ), which is left as one of the future works.

It is also worth highlighting that due to the presence of /epsilon1 bias and /epsilon1 app , the average objective error and constraint violation cannot be guaranteed to be zero, even for large T . However, for rich policy parameterization and good critic approximation, the effects of these quantities are negligibly small.

## 5 Conclusion

In this paper, we investigate the infinite-horizon average reward Constrained Markov Decision Processes (CMDPs) with general policy parametrization. We propose a novel algorithm, the 'Primaldual natural actor-critic," which efficiently manages constraints while achieving a global convergence rate of ˜ O (1 / √ T ) , aligning with the theoretical lower bound for Markov Decision Processes (MDPs). We also extend our analysis to the setting with unknown mixing time. Future directions include narrowing the performance gap in this setting, parameterizing the critic using neural networks as in [25, 21], and relaxing the ergodicity assumption following [20].

## 6 Acknowledgment

The work was supported in part by the National Science Foundation under grant CCF-2149588, Office of Naval Research under grant N00014-23-1-2532, and Cisco Systems, Inc.

## References

- [1] Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. The Journal of Machine Learning Research , 22(1):4431-4506, 2021.
- [2] Mridul Agarwal, Qinbo Bai, and Vaneet Aggarwal. Concave Utility Reinforcement Learning with Zero-Constraint Violations. Transactions on Machine Learning Research , 2022.
- [3] Mridul Agarwal, Qinbo Bai, and Vaneet Aggarwal. Regret guarantees for model-based reinforcement learning with long-term average constraints. In Uncertainty in Artificial Intelligence , pages 22-31, 2022.
- [4] Shipra Agrawal and Randy Jia. Optimistic posterior sampling for reinforcement learning: worst-case regret bounds. Advances in Neural Information Processing Systems , 30, 2017.
- [5] Abubakr O Al-Abbasi, Arnob Ghosh, and Vaneet Aggarwal. Deeppool: Distributed model-free algorithm for ride-sharing using deep reinforcement learning. IEEE Transactions on Intelligent Transportation Systems , 20(12):4714-4727, 2019.
- [6] Peter Auer, Thomas Jaksch, and Ronald Ortner. Near-optimal regret bounds for reinforcement learning. Advances in neural information processing systems , 21, 2008.
- [7] Qinbo Bai, Amrit Singh Bedi, Mridul Agarwal, Alec Koppel, and Vaneet Aggarwal. Achieving Zero Constraint Violation for Constrained Reinforcement Learning via Primal-Dual Approach. Proceedings of the AAAI Conference on Artificial Intelligence , 36:3682-3689, Jun. 2022.
- [8] Qinbo Bai, Amrit Singh Bedi, Mridul Agarwal, Alec Koppel, and Vaneet Aggarwal. Achieving zero constraint violation for constrained reinforcement learning via primal-dual approach. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 3682-3689, 2022.
- [9] Qinbo Bai, Amrit Singh Bedi, and Vaneet Aggarwal. Achieving zero constraint violation for constrained reinforcement learning via conservative natural policy gradient primal-dual algorithm. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pages 6737-6744, 2023.

- [10] Qinbo Bai, Washim Mondal, and Vaneet Aggarwal. Learning general parameterized policies for infinite horizon average reward constrained MDPs via primal-dual policy gradient algorithm. Advances in Neural Information Processing Systems , 37:108566-108599, 2024.
- [11] Qinbo Bai, Washim Uddin Mondal, and Vaneet Aggarwal. Regret analysis of policy gradient algorithm for infinite horizon average reward Markov decision processes. In AAAI Conference on Artificial Intelligence , 2024.
- [12] Aleksandr Beznosikov, Sergey Samsonov, Marina Sheshukova, Alexander Gasnikov, Alexey Naumov, and Eric Moulines. First Order Methods with Markovian Noise: from Acceleration to Variational Inequalities. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- [13] Jalaj Bhandari, Daniel Russo, and Raghav Singal. A Finite Time Analysis of Temporal Difference Learning With Linear Function Approximation. CoRR , abs/1806.02450, 2018.
- [14] Jiayu Chen, Tian Lan, and Vaneet Aggarwal. Option-aware adversarial inverse reinforcement learning for robotic control. In 2023 IEEE International Conference on Robotics and Automation (ICRA) , pages 5902-5908. IEEE, 2023.
- [15] Liyu Chen, Rahul Jain, and Haipeng Luo. Learning infinite-horizon average-reward Markov decision process with constraints. In International Conference on Machine Learning , pages 3246-3270. PMLR, 2022.
- [16] Dongsheng Ding, Kaiqing Zhang, Tamer Basar, and Mihailo Jovanovic. Natural policy gradient primal-dual method for constrained Markov decision processes. Advances in Neural Information Processing Systems , 33:8378-8390, 2020.
- [17] Dongsheng Ding, Kaiqing Zhang, Jiali Duan, Tamer Ba¸ sar, and Mihailo R Jovanovi´ c. Convergence and sample complexity of natural policy gradient primal-dual methods for constrained MDPs. arXiv preprint arXiv:2206.02346 , 2022.
- [18] Yonathan Efroni, Shie Mannor, and Matteo Pirotta. Exploration-exploitation in constrained MDPs. arXiv preprint arXiv:2003.02189 , 2020.
- [19] Ilyas Fatkhullin, Anas Barakat, Anastasia Kireeva, and Niao He. Stochastic policy gradient methods: Improved sample complexity for Fisher-non-degenerate policies. In International Conference on Machine Learning , pages 9827-9869. PMLR, 2023.
- [20] Swetha Ganesh and Vaneet Aggarwal. Regret analysis of average-reward unichain mdps via an actor-critic approach. In Advances in Neural Information Processing Systems , 2025.
- [21] Swetha Ganesh, Jiayu Chen, Washim Uddin Mondal, and Vaneet Aggarwal. Order-optimal global convergence for actor-critic with general policy and neural critic parametrization. In The 41st Conference on Uncertainty in Artificial Intelligence , 2025.
- [22] Swetha Ganesh, Washim Uddin Mondal, and Vaneet Aggarwal. Order-Optimal Regret with Novel Policy Gradient Approaches in Infinite Horizon Average Reward MDPs. In The 28th International Conference on Artificial Intelligence and Statistics , 2025.
- [23] Swetha Ganesh, Washim Uddin Mondal, and Vaneet Aggarwal. A sharper global convergence analysis for average reward reinforcement learning via an actor-critic approach. International Conference on Machine Learning , 2025.
- [24] Ather Gattami, Qinbo Bai, and Vaneet Aggarwal. Reinforcement learning for constrained markov decision processes. In International Conference on Artificial Intelligence and Statistics , pages 2656-2664. PMLR, 2021.
- [25] Mudit Gaur, Amrit Bedi, Di Wang, and Vaneet Aggarwal. Closing the gap: Achieving global convergence (last iterate) of actor-critic under markovian sampling with neural network parametrization. In International Conference on Machine Learning , pages 15153-15179. PMLR, 2024.
- [26] Jacopo Germano, Francesco Emanuele Stradi, Gianmarco Genalti, Matteo Castiglioni, Alberto Marchesi, and Nicola Gatti. A Best-of-Both-Worlds Algorithm for Constrained MDPs with Long-Term Constraints. arXiv preprint arXiv:2304.14326 , 2023.

- [27] Arnob Ghosh, Xingyu Zhou, and Ness Shroff. Achieving Sub-linear Regret in Infinite Horizon Average Reward Constrained MDP with Linear Function Approximation. In The Eleventh International Conference on Learning Representations , 2023.
- [28] Marina Haliem, Ganapathy Mani, Vaneet Aggarwal, and Bharat Bhargava. A distributed modelfree ride-sharing approach for joint matching, pricing, and dispatching using deep reinforcement learning. IEEE Transactions on Intelligent Transportation Systems , 22(12):7931-7942, 2021.
- [29] Lu Ling, Washim Uddin Mondal, and Satish V Ukkusuri. Cooperating graph neural networks with deep reinforcement learning for vaccine prioritization. IEEE Journal of Biomedical and Health Informatics , 2024.
- [30] Yanli Liu, Kaiqing Zhang, Tamer Basar, and Wotao Yin. An improved analysis of (variancereduced) policy gradient and natural policy gradient methods. Advances in Neural Information Processing Systems , 33:7624-7636, 2020.
- [31] Washim Mondal and Vaneet Aggarwal. Sample-efficient constrained reinforcement learning with general parameterization. Advances in Neural Information Processing Systems , 37:6838068405, 2024.
- [32] Mahadesh Panju, Ramkumar Raghu, Vinod Sharma, Vaneet Aggarwal, and Rajesh Ramachandran. Queueing theoretic models for uncoded and coded multicast wireless networks with caches. IEEE Transactions on Wireless Communications , 21(2):1257-1271, 2021.
- [33] Matteo Papini, Damiano Binaghi, Giuseppe Canonaco, Matteo Pirotta, and Marcello Restelli. Stochastic variance-reduced policy gradient. In International conference on machine learning , pages 4026-4035, 2018.
- [34] Bhrij Patel, Wesley A Suttle, Alec Koppel, Vaneet Aggarwal, Brian M Sadler, Dinesh Manocha, and Amrit Bedi. Towards global optimality for practical average reward reinforcement learning without mixing time oracles. In Forty-first International Conference on Machine Learning , 2024.
- [35] Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp; Sons, 2014.
- [36] Shuang Qiu, Xiaohan Wei, Zhuoran Yang, Jieping Ye, and Zhaoran Wang. Upper confidence primal-dual reinforcement learning for CMDP with adversarial loss. Advances in Neural Information Processing Systems , 33:15277-15287, 2020.
- [37] Shuang Qiu, Zhuoran Yang, Jieping Ye, and Zhaoran Wang. On finite-time convergence of actor-critic algorithm. IEEE Journal on Selected Areas in Information Theory , 2(2):652-664, 2021.
- [38] Wesley A Suttle, Amrit Bedi, Bhrij Patel, Brian M Sadler, Alec Koppel, and Dinesh Manocha. Beyond Exponentially Fast Mixing in Average-Reward Reinforcement Learning via MultiLevel Monte Carlo Actor-Critic. In International Conference on Machine Learning , pages 33240-33267, 2023.
- [39] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems , 12, 1999.
- [40] Lingxiao Wang, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural Policy Gradient Methods: Global Optimality and Rates of Convergence. In International Conference on Learning Representations , 2019.
- [41] Chen-Yu Wei, Mehdi Jafarnia Jahromi, Haipeng Luo, Hiteshi Sharma, and Rahul Jain. Modelfree reinforcement learning in infinite-horizon average-reward Markov decision processes. In International conference on machine learning , pages 10170-10180. PMLR, 2020.
- [42] Honghao Wei, Xin Liu, and Lei Ying. A provably-efficient model-free algorithm for infinitehorizon average-reward constrained Markov decision processes. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 3868-3876, 2022.

- [43] Pan Xu, Felicia Gao, and Quanquan Gu. Sample Efficient Policy Gradient Methods with Recursive Variance Reduction. In International Conference on Learning Representations , 2019.
- [44] Pan Xu, Felicia Gao, and Quanquan Gu. An improved convergence analysis of stochastic variance-reduced policy gradient. In Uncertainty in Artificial Intelligence , pages 541-551, 2020.
- [45] Tengyu Xu, Yingbin Liang, and Guanghui Lan. CRPO: A new approach for safe reinforcement learning with convergence guarantee. In International Conference on Machine Learning , pages 11480-11491. PMLR, 2021.
- [46] Junyu Zhang, Chengzhuo Ni, Zheng Yu, Csaba Szepesvari, and Mengdi Wang. On the Convergence and Sample Efficiency of Variance-Reduced Policy Gradient Method. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems , 2021.
- [47] Shaofeng Zou, Tengyu Xu, and Yingbin Liang. Finite-sample analysis for SARSA with linear function approximation. Advances in neural information processing systems , 32, 2019.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims made in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We point out the limitations of our work.

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

Justification: We provide the full set of assumptions and complete proof for each theoretical result.

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

Justification: Our paper does not include experiments.

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
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: Our paper does not include experiments.

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

Justification: Our paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Our paper does not include experiments.

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

Justification: Our paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our paper conforms, in every respect, with the NeurIPS code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no potential societal impact of our work.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to

generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: Our paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: Our paper does not use existing assets.

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

Justification: Our paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper involves neither crowdsourcing nor human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper neither involves crowdsourcing nor human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We do not use LLM in our paper.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Related Works

The constrained reinforcement learning problem has been widely explored for both infinite horizon discounted reward and episodic MDPs. Recent studies have examined discounted reward CMDPs in various contexts, including the tabular setting [8], with softmax parameterization [16, 45], and with general policy parameterization setting [16, 45, 9, 31]. Additionally, episodic CMDPs have been explored in the tabular setting by [18, 36, 26]. Recent research has also focused on infinite horizon average reward CMDPs, examining various approaches including model-based setups [15, 3, 2], tabular model-free settings [42], linear CMDP setting [27] and general policy parametrization setting [10]. In model-based CMDP setups, [2] introduced algorithms leveraging posterior sampling and the optimism principle, achieving a convergence rate of ˜ O (1 / √ T ) with zero constraint violations. For tabular model-free approach, [42] attains a convergence rate of ˜ O ( T -1 / 6 ) with zero constraint violations. In linear CMDP, [27] obtains ˜ O ( T -1 / 2 ) convergence rate with zero constraint violation. Note that linear CMDP assumes a linear structure in the transition probability based on a known feature map, which is not realistic. Finally, [10] studied the infinite horizon average reward CMDPs with general parametrization and achieved a global convergence rate of ˜ O (1 /T 1 / 5 ) . Table 1 summarizes all relevant works on average reward CMDPs.

In unconstrained average reward MDPs, both model-based and model-free tabular setups have been widely studied. For instance, the model-based algorithms by [4, 6] obtain the optimal convergence rate of ˜ O (1 / √ T ) . Similarly, the model-free algorithm proposed by [41] achieves ˜ O (1 / √ T ) convergence rate for tabular MDP. Average reward MDP with general parametrization has been recently studied in [23, 22], where global convergence rates of ˜ O (1 / √ T ) are achieved. In particular, [23] leverages Actor-Critic methods and achieves global convergence without knowledge of mixing time.

## B Preliminary Results for Global Convergence Analysis

To prove Theorem 4.8 and 4.7, we first discuss a more general case of linear recursion with biased estimators.

Theorem B.1. Consider the stochastic linear recursion that aims to approximate x ∗ = P -1 q

.

<!-- formula-not-decoded -->

where ˆ P h , ˆ q h are estimates of the matrices P ∈ R n × n , q ∈ R n respectively, and h ∈ { 0 , · · · , H -1 } . Assume that the following bounds hold ∀ h .

<!-- formula-not-decoded -->

where E h denotes conditional expectation given history up to step h . Since E [ˆ q h ] = E [ E h [ˆ q h ]] , we have ¯ δ 2 q ≤ δ 2 q . Additionally, assume that

<!-- formula-not-decoded -->

The condition that λ P &gt; 0 implies that P is invertible. The goal of recursion (35) is to approximate the term x ∗ = P -1 q . If δ P ≤ λ P / 8 , and ¯ β ≤ λ P / [4(6 σ 2 P +2Λ 2 P )] , the following relation holds.

<!-- formula-not-decoded -->

where R 0 = λ -3 P Λ 2 q σ 2 P + λ -1 P σ 2 q , R 1 = λ -2 P [ δ 2 P λ -2 P Λ 2 q + δ 2 q ] , and ˜ O ( · ) hides logarithmic factors of H . Moreover,

<!-- formula-not-decoded -->

where ¯ R 1 = δ 2 P λ -2 P Λ 2 q + ¯ δ 2 q .

Proof. Let g h = ˆ P h x h -ˆ q h . To prove the first statement, observe the following relations.

<!-- formula-not-decoded -->

where ( a ) , ( b ) follow from λ P ≤ ‖ P ‖ ≤ Λ P . Taking the conditional expectation E h on both sides,

<!-- formula-not-decoded -->

Observe that the third term in (40) can be bounded as follows.

<!-- formula-not-decoded -->

where the last inequality follows from ‖ x ∗ ‖ 2 = ∥ ∥ P -1 q ∥ ∥ 2 ≤ λ -2 P Λ 2 q . Taking the expectation yields

<!-- formula-not-decoded -->

The second term in (40) can be bounded as

<!-- formula-not-decoded -->

where the last inequality follows from ‖ x ∗ ‖ 2 = ∥ ∥ P -1 q ∥ ∥ 2 ≤ λ -2 P Λ 2 q . Substituting the above bounds in (40), we arrive at the following.

<!-- formula-not-decoded -->

For δ P ≤ λ P / 8 , and ¯ β ≤ λ P / [4(6 σ 2 P +2Λ 2 P )] , we can modify the above inequality to the following.

<!-- formula-not-decoded -->

Taking the expectation on both sides and unrolling the recursion yields

<!-- formula-not-decoded -->

To prove the second statement, observe that we have the following recursion.

<!-- formula-not-decoded -->

where ( a ) and ( b ) are consequences of λ P ≤ ‖ P ‖ ≤ Λ P . The third term in the last line of (43) can be bounded as follows.

<!-- formula-not-decoded -->

where ( a ) follows from (38). The second term in the last line of (43) can be bounded as follows.

<!-- formula-not-decoded -->

Substituting the above bounds in (43), we obtain the following recursion.

<!-- formula-not-decoded -->

If ¯ β &lt; λ P / (4Λ 2 P ) , the above bound implies the following.

<!-- formula-not-decoded -->

Unrolling the above recursion, we obtain

<!-- formula-not-decoded -->

where R 1 = δ P λ P Λ q + ¯ δ q . This concludes the result.

We now provide some bounds on MLMC estimates.

Lemma B.2. Consider a time-homogeneous, ergodic Markov chain ( Z t ) t ≥ 0 with a unique invariant distribution d Z and a mixing time τ mix . Assume that ∇ F ( x, Z ) is an estimate of the gradient ∇ F ( x ) . Let ‖ E d Z [ ∇ F ( x, Z )] -∇ F ( x ) ‖ 2 ≤ δ 2 and ‖∇ F ( x, Z t ) -E d Z [ ∇ F ( x, Z )] ‖ 2 ≤ σ 2 for all t ≥ 0 . If Q ∼ Geom(1 / 2) , then the following MLMC estimator

<!-- formula-not-decoded -->

satisfies the inequalities stated below.

<!-- formula-not-decoded -->

Before proceeding to the proof, we state a useful lemma.

Lemma B.3 (Lemma 1, [12]) . Consider the same setup as in Lemma B.2. The following holds.

<!-- formula-not-decoded -->

where N is a constant, C 1 = 16(1 + 1 ln 2 4 ) , and the expectation is over the distribution of { Z t } N -1 t =0 emanating from any arbitrary initial distribution.

Proof of Lemma B.2. The statement ( a ) can be proven as follows.

<!-- formula-not-decoded -->

For the proof of ( b ) , notice that

<!-- formula-not-decoded -->

where ( a ) follows from Lemma B.3. Using this result, we obtain the following.

<!-- formula-not-decoded -->

This completes the proof of statement ( b ) . For part ( c ) , we have

<!-- formula-not-decoded -->

where ( a ) follows from Lemma B.3. This concludes the proof of Lemma B.2.

## C Proof of Lemma 4.6

We begin by stating a useful lemma:

Lemma C.1 (Lemma 4, [11]) . The performance difference between two policies π θ , π θ ′ is bounded as follows where g ∈ { r, c } .

<!-- formula-not-decoded -->

Continuing with the proof of Lemma 4.6, we start with the definition of KL divergence. For notational simplicity, we shall use A π θ k r + λ k c to denote A π θ k r + λ k A π θ k c .

<!-- formula-not-decoded -->

where L ( π ∗ , λ k ) := J π ∗ r + λ k J π ∗ c , the relations ( a ) , ( b ) result from Assumption 4.4 ( b ) and Lemma C.1, respectively. Inequality ( c ) arises from the convexity of the function f ( x ) = x 2 . Lastly, ( d ) is a

consequence of Assumption 4.3. By taking expectations on both sides, we derive:

<!-- formula-not-decoded -->

where ( a ) follows from Assumption 4.4 ( a ) . Rearranging the terms, we get,

<!-- formula-not-decoded -->

Adding the above inequality from k = 0 to K -1 , using the non-negativity of KL divergence and dividing the resulting expression by K , we obtain the final result.

## D Proof of Theorem 4.8

Recall that A g ( θ ) = E θ [ A g ( θ ; z )] , b g ( θ ) = E θ [ b g ( θ ; z )] , and ξ ∗ g ( θ ) = ( A g ( θ )) -1 b g ( θ ) where E θ denotes the expectation over the distribution of z = ( s, a, s ′ ) where ( s, a ) ∼ ν π θ , s ′ ∼ P ( s, a ) (refer to section 4).

Lemma D.1. For large c γ , Assumption 4.2 implies that A g ( θ k ) -( λ/ 2) I is positive semi-definite for both g ∈ { r, c } and for all k where I is an identity matrix, i.e., ξ /latticetop A g ( θ k ) ξ ≥ λ/ 2 · ‖ ξ ‖ 2 , for all ξ .

Proof of Lemma D.1. Note that for any ξ = [ η, ζ ] , we have

<!-- formula-not-decoded -->

where ( a ) is a consequence of Assumption 4.2 and the fact that ‖ φ g ( s ) ‖ ≤ 1 , ∀ s ∈ S . Finally, ( b ) holds when c γ ≥ λ + √ 1 λ 2 -1 . This completes the proof.

Let A MLMC g,kh and b MLMC g,kh be the MLMC estimators of { A g ( θ k ; z t kh ) } l kh -1 t =0 and { b g ( θ k ; z t kh ) } l kh -1 t =0 respectively (see (19)) i.e.

<!-- formula-not-decoded -->

where A j g,kh = 1 2 j ∑ 2 j -1 t =0 A g ( θ k ; z t kh ) and

<!-- formula-not-decoded -->

where b j g,kh = 1 2 j ∑ 2 j -1 t =0 b g ( θ k ; z t kh ) .

We can bound the bias and variance of A MLMC g,kh and b MLMC g,kh as follows.

Lemma D.2. Consider Algorithm 1 with a policy parameter θ k and assume assumptions 2.2-4.5 hold. The MLMC estimators A MLMC g,kh and b MLMC g,kh obey the following bounds.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where h ∈ { 0 , · · · , H -1 } and E k,h defines the conditional expectation given the history up to the inner loop step h of the critic within the k th outer loop instant.

Proof. Recall from (19) the definitions of A g ( θ ; z ) and b ( θ k ; z ) for any z = ( s, a, s ′ ) ∈ S × A × S . We have the following inequalities.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) , ( b ) hold since | g ( s, a ) | ≤ 1 and ‖ φ g ( s ) ‖ ≤ 1 , ∀ ( s, a ) ∈ S × A , ∀ g ∈ { r, c } . Hence, for any z j kh ∈ S × A × S , we have the inequalities stated below.

<!-- formula-not-decoded -->

Combining the above results with Lemma B.2, and using the definitions that A g ( θ ) = E θ [ A g ( θ ; z )] , b g ( θ ) = E θ [ b g ( θ ; z )] , we establish the result.

Combining Lemma D.1 with (53), (54), we obtain the following for any θ k with c γ ≥ λ + √ 1 λ 2 -1 .

<!-- formula-not-decoded -->

Combining the above results with Lemma D.1 along with Theorem B.1, we then obtain the following inequalities given that T max ≥ 8 c 2 γ τ mix λ and γ ξ ≤ λ 24 c 2 γ τ mix log T max .

<!-- formula-not-decoded -->

where the terms R 0 , R 1 , ¯ R 1 are defined as follows.

<!-- formula-not-decoded -->

If we further set γ ξ = 2 log T λH while ensuring 2 log T λH ≤ λ 24 c 2 γ τ mix log T max , we have the following results.

<!-- formula-not-decoded -->

We used the fact that H -1 + T -1 max = O (1) in the last inequality. Utilizing ξ k g,H = ξ k g , ξ 0 = 0 , and ‖ ξ ∗ g ( θ k ) ‖ = ‖ [ A g ( θ k )] -1 b g ( θ k ) ‖ = O ( λ -2 c 2 γ ) (via (55)), we conclude:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This completes the proof.

## E Proof of Theorem 4.7

To prove Theorem 4.7, we follow a proof structure similar to that of Theorem 4.8. Let F MLMC kh and ∇ MLMC θ J g,kh denote the MLMC estimators defined as follows.

<!-- formula-not-decoded -->

with F j kh = 1 2 j ∑ 2 j -1 t =0 F ( θ k ; z t kh ) (see (22)) and

<!-- formula-not-decoded -->

where ∇ j θ J g,kh = 2 -j ∑ 2 j -1 t =0 ˆ ∇ θ J g ( θ k , ξ k g ; z t kh ) (see (22)).

Using the above inequalities, we can bound the bias and variance of the MLMC estimators as follows. Lemma E.1. Consider Algorithm 1 with given policy parameter θ k . Under the assumptions 2.2-4.5, the following statements hold.

<!-- formula-not-decoded -->

where E k,h defines the conditional expectation given the history up to the inner loop step h of the actor within the k th outer loop instant, while E k is the conditional expectation given the history up to the k th outer loop instant. Moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma E.1. Fix an outer loop instant k and an inner loop instant h ∈ { 0 , · · · , H -1 } of the actor subroutine. Recall the definition of F ( θ k ; · ) from (22). The following inequalities hold for any θ k and z j kh ∈ S × A × S .

<!-- formula-not-decoded -->

where E θ k denotes the expectation over the distribution z = ( s, a, s ′ ) where ( s, a ) ∼ ν π θ k , s ′ ∼ P ( ·| s, a ) . The equality ( a ) follows from the definition of the Fisher matrix, and ( b ) is a consequence of assumption 4.4. Statements (a) and (c), therefore, directly follow from Lemma B.2.

To prove the other statements, recall the definition of ˆ ∇ θ J g ( θ k , ξ k g ; · ) from (22). Note the following relations for arbitrary θ k , ξ k g = [ η k g , ζ k g ] .

<!-- formula-not-decoded -->

We will use the notation that ξ ∗ g ( θ k ) = [ η ∗ g ( θ k ) , ζ ∗ g ( θ k )] /latticetop . Observe that

<!-- formula-not-decoded -->

where ( a ) follows from Assumption 4.4 and the boundedness of the feature map, φ while ( b ) is a consequence of Assumption 4.4 and 4.1. Finally, ( c ) is an application of Bellman's equation and the fact that η ∗ g ( θ k ) = J π θ k g , which can be easily verified. We get,

<!-- formula-not-decoded -->

Moreover, observe that, for arbitrary z j kh ∈ S × A × S

<!-- formula-not-decoded -->

where ( a ) results from Assumption 4.4 and the boundedness of φ . We can, thus, conclude statements (c) and (d) by applying (61) and (62) in Lemma B.2. To prove the statement (e), note that

<!-- formula-not-decoded -->

Observe the following bounds.

<!-- formula-not-decoded -->

where ( a ) follows from Assumption 4.4 and the boundedness of the feature map, φ while ( b ) is a consequence of Assumption 4.4 and 4.1. Finally, ( c ) is an application of Bellman's equation. We get,

<!-- formula-not-decoded -->

Using the above bound, we deduce the following.

<!-- formula-not-decoded -->

where ( a ) follows from (64), ( b ) follows from Lemma B.2(a), B.3, and the definition of ¯ σ 2 k,g .

Combining Lemma G.1 with Assumptions 4.4 and 4.5, we obtain the following for a given policy parameter θ and ∀ g ∈ { r, c } ,

<!-- formula-not-decoded -->

Combining the above results with Lemma E.1 and invoking Theorem B.1, we then obtain given that T max ≥ 8 G 4 1 τ mix µ and γ ω ≤ µ 4(6 G 4 1 τ mix log T max +2 G 2 1 τ 2 mix log T max ) .

<!-- formula-not-decoded -->

where the terms R 0 , R 1 , ¯ R 1 are defined as follows.

<!-- formula-not-decoded -->

Moreover, note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) uses (55) for sufficiently large c γ and the definition that ξ ∗ g ( θ k ) = [ A g ( θ k )] -1 b g ( θ k ) . If we set γ ω = 2 log T µH ≤ µ 4(6 G 4 τ mix log T max +2 G 2 τ 2 log T max ) , we get

<!-- formula-not-decoded -->

Substituting ω 0 = 0 , ‖ ω ∗ g,k ‖ = ‖ [ F ( θ k )] -1 ∇ θ J g ( θ k ) ‖ = O ( µ -1 G 1 τ mix ) (follows from Lemma G.1, and assumptions 4.4and 4.5), we can simplify the above bounds as follows.

<!-- formula-not-decoded -->

This completes the proof of Theorem 4.7.

## F Proof of Main Theorems (Theorem 4.9 and Theorem 4.10)

## F.0.1 Rate of Convergence of the Objective

Combining (56), (57) and (66) and (67), we obtain the following results.

<!-- formula-not-decoded -->

We can decompose E ‖ ( E k [ ω k ] -ω ∗ k ) ‖ and E ‖ ω k -ω ∗ k ‖ 2 using the definition that ω k = ω k r + λ k ω k c . Using (68) and (69), and the fact that λ k ∈ [0 , 2 /δ ] , we obtain

<!-- formula-not-decoded -->

Setting T max = T in (70) and (71), and using Lemma G.2 and (26) in (25) would obtain

<!-- formula-not-decoded -->

Using the definition of the Lagrange function, the above inequality can be equivalently written as

<!-- formula-not-decoded -->

Wewill now extract the objective convergence rate from the convergence rate of the Lagrange function stated above. Note that,

<!-- formula-not-decoded -->

Inequality (a) holds because θ ∗ is a feasible solution to the constrained optimization problem. Rearranging items and taking the expectation, we have

<!-- formula-not-decoded -->

Note that, unlike the discounted reward case, the average reward estimate η k c is no longer unbiased. However, by Theorem 4.8 and the facts that | c ( s, a ) | ≤ 1 , ∀ ( s, a ) ∈ S × A , and η ∗ c ( θ k ) = J c ( θ k ) , we get the following inequality.

<!-- formula-not-decoded -->

where the last inequality utilizes (57), the fact that λ k ∈ [0 , 2 /δ ] , and T max = T . Combining with (72), we arrive at the following result.

<!-- formula-not-decoded -->

## F.0.2 Rate of Constraint Violation

Given the dual update in algorithm 1, we have

<!-- formula-not-decoded -->

where ( a ) is because of the non-expansiveness of projection P Λ . Averaging the above inequality over k = 0 , . . . , K -1 yields

<!-- formula-not-decoded -->

Taking expectations on both sides,

<!-- formula-not-decoded -->

where the last inequality utilizes λ 0 = 0 . Notice that λ k J π ∗ c ≥ 0 , ∀ k . Using the above inequality to (72), we therefore have,

<!-- formula-not-decoded -->

We define a new policy ¯ π which uniformly chooses the policy π θ k for k ∈ [ K ] . By the occupancy measure method, J g ( θ k ) is linear in terms of an occupancy measure induced by policy π θ k . Thus,

<!-- formula-not-decoded -->

Injecting the above relation to (80), we have

<!-- formula-not-decoded -->

By Lemma G.6, we arrive at,

<!-- formula-not-decoded -->

## F.0.3 Optimal Choice of α , β , K , and H

If τ mix is unknown, we can take α = T -a , β = T -b for some a, b ∈ (0 , 1) , and H = T /epsilon1 , K = T 1 -/epsilon1 for /epsilon1 ∈ (0 , 1) then following (76) and (82), we can write,

<!-- formula-not-decoded -->

Clearly, the optimal values of a and b can be obtained by solving the following optimization.

<!-- formula-not-decoded -->

By choosing ( a, b ) = (1 / 2 , 1 / 2) , this would obtain the solution of the above optimization problem. Thus, the convergence rate and constraint violation would both become:

<!-- formula-not-decoded -->

Recall that the above result holds when the conditions for Theorem 4.8 and Theorem 4.7 are satisfied.

<!-- formula-not-decoded -->

In light of the above conditions, (84) and (85) hold if H = T /epsilon1 ≥ ˜ Θ( τ 2 mix ) . If τ mix is known, we can set H = ˜ Θ( τ 2 mix ) that satisfies (86) and (87). Moreover, we can take K = O ( T ) . Thus, by following

the same analysis as above, we can obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes the proof of Theorem 4.9 and Theorem 4.10.

## G Some Auxiliary Lemmas for the Proofs

Lemma G.1. [41, Lemma 14] For any ergodic MDP with mixing time τ mix , the following holds ∀ ( s, a ) ∈ S × A , any policy π and ∀ g ∈ { r, c } .

<!-- formula-not-decoded -->

Lemma G.2. For any ergodic MDP with mixing time τ mix , the following holds ∀ ( s, a ) ∈ S × A for any policy π θ k with θ k satisfying Assumption 4.4, the dual parameter λ k ∈ [0 , 2 /δ ] , and g ∈ { r, c } .

<!-- formula-not-decoded -->

Proof. Using the policy gradient theorem (7), we have the following relation.

<!-- formula-not-decoded -->

Applying Lemma G.1 ( b ) and Assumption 4.4 ( a ) , we get

<!-- formula-not-decoded -->

Combining the above result with the definition of the Lagrange function and the bound that λ k ≤ 2 δ , we arrive at the following result.

<!-- formula-not-decoded -->

This concludes the proof of Lemma G.2.

Lemma G.3 (Strong duality) . [17, Lemma 3] We restate the problem (2) below for convenience.

<!-- formula-not-decoded -->

where Π is the set of all policies. Define π ∗ as an optimal solution to the above optimization problem. Define the associated dual function as

<!-- formula-not-decoded -->

and denote λ ∗ = arg min λ ≥ 0 J λ D . We have the following strong duality property for the unparameterized problem.

<!-- formula-not-decoded -->

Lemma G.4 (Lemma 16, [10]) . Consider the parameterized problem (3) where Θ is the collection of all parameters. Under Assumption 2.1, the optimal dual variable, λ ∗ Θ = arg min λ ≥ 0 max θ ∈ Θ J π θ r + λJ π θ c , for the parameterized problem can be bounded as follows.

<!-- formula-not-decoded -->

where π ∗ is an optimal solution to the unparameterized problem (93) and ¯ θ is a feasible parameter mentioned in Assumption 2.1.

Notice that in Eq. (12), the dual update is projected onto the set [0 , 2 δ ] because the optimal dual variable for the parameterized problem is bounded in Lemma G.4. We note that our dual updating technique remains the same as [10], however we were able to achieve an improvement of global convergence rate due to the use of natural policy gradient and a more prudent choice of stepsizes ( α, β ).

Lemma G.5. Assume that the Assumption 2.1 holds. Define v ( τ ) = max π ∈ Π { J π r | J π c ≥ τ } where Π is the set of all policies. The following holds for any τ ∈ R where λ ∗ is the optimal dual parameter for the unparameterized problem as stated in Lemma G.3.

<!-- formula-not-decoded -->

Proof. Using the definition of v ( τ ) , we get v (0) = J π ∗ r where π ∗ is a solution to the unparameterized problem (93). Denote L ( π, λ ) = J π r + λJ π c . By the strong duality stated in Lemma G.3, we have the following for any π ∈ Π .

<!-- formula-not-decoded -->

Thus, for any π ∈ { π ∈ Π | J π c ≥ τ } , we can deduce the following.

<!-- formula-not-decoded -->

Maximizing the right-hand side of this inequality over { π ∈ Π | J π c ≥ τ } yields

<!-- formula-not-decoded -->

This completes the proof of Lemma G.5.

Lemma G.6. Let Assumption 2.1 hold and ( π ∗ , λ ∗ ) be the optimal primal and dual parameters for the unparameterized problem (93) . For any constant C ≥ 2 λ ∗ , if there exist a π ∈ Π and ζ &gt; 0 such that J π ∗ r -J π r + C [ -J π c ] ≤ ζ , then

<!-- formula-not-decoded -->

Proof. Let τ = J π c . We have the following inequality following the definition of v ( τ ) provided in Lemma G.5.

<!-- formula-not-decoded -->

Combining Eq. (100) and (102), and using the fact that v (0) = J π ∗ r , we have the following inequality.

<!-- formula-not-decoded -->

Using the condition in the Lemma, we have

<!-- formula-not-decoded -->

Since C -λ ∗ ≥ C/ 2 &gt; 0 , we finally have the following inequality.

<!-- formula-not-decoded -->

This completes the proof of Lemma G.6.