## Assignments for Congestion-Averse Agents: Seeking Competitive and Envy-Free Solutions

## Jiehua Chen

Institute of Logic and Computation TU Wien Austria

❥❝❤❡♥❅❛❝✳/a116✉✇✐❡♥✳❛❝✳❛/a116

## Jiong Guo

School of Computer Science and Technology Shandong University Qingdao, China ❥❣✉♦❅/a115❞✉✳❡❞✉✳❝♥

## Yinghui Wen

Digital and Intelligent Center

Shandong Institute of Information Technology Industry Development Jinan, China ②✐♥❣❤✉✐✳✇❡♥❅❢♦①♠❛✐❧✳❝♦♠

## Abstract

We investigate congested assignment problems where agents have preferences over both resources and their associated congestion levels. These agents are averse towards congestion , i.e., consistently preferring lower congestion for identical resources. Such scenarios are ubiquitous across domains including traffic management and school choice, where fair resource allocation is essential. We focus on the concept of competitiveness , recently introduced by Bogomolnaia and Moulin [6], and contribute a polynomial-time algorithm that determines competitiveness, resolving their open question. Additionally, we explore two optimization variants of congested assignments by examining the problem of finding envy-free or maximally competitive assignments that guarantee a certain amount of social welfare for every agent, termed top-guarantees [6]. While we prove that both problems are NP-hard, we develop parameterized algorithms with respect to the number of agents or resources.

## 1 Introduction

In the realm of resource allocation and task assignment, the challenge often extends beyond mere allocation-it entails navigating the intricate balance between individual preferences and the congestions that other agents incur. Congested assignments epitomize this challenge and address the situation when agents are congestion-averse , i.e., the preferences are negatively correlated with the number of agents simultaneously assigned to the same resource, the so-called congestion level .

Congested assignments are pertinent in numerous real-world scenarios: From urban traffic management , where drivers select routes while considering traffic congestion, to educational contexts like school choice [35, 12] or student exercise slot allocations , where students' choices are influenced by class sizes. Similarly, in cloud computing , allocating computational tasks to servers must account for server load. In each of these scenarios, assigning an agent-whether a driver, a student, or a task executor-to a resource (jointly referred to as a post ) may introduce additional costs, with detrimental effects on other agents. For example, the productivity in a shared office space diminishes as more people use it. Similarly, in traffic management, drivers might opt for less crowded routes since the mental load increases significantly with even a small increase in traffic. In other words, the agents' individual preference over a post is inversely proportional to its congestion level.

Figure 1: Relations among the different fairness concepts under congestion-averse preferences. All relations are strict; see Lemma 1.

<!-- image -->

The overarching task is to find an optimal assignment of the agents to posts that respects agents' congestion-averse preferences. Defining 'optimal' in this setting is non-trivial. In traffic management, for example, one might aim for a Nash stable assignment (aka. Nash equilibrium [33, 37]), meaning that no agent prefers to deviate to another post which increases the congestion by one. Milchtaich shows that Nash stable assignments for congestion-averse preferences are always attainable and any assignment can be turned Nash stable in polynomially many best-reply improvement steps. Although Nash stability may prevent systematic chaos, it can be quite unfair, particularly in slot allocations, as it may induce envy among agents, meaning that an agent may prefer to take over another agent's post (albeit for the same congestion). Notice that envy-freeness is also easy to achieve by simply assigning all agents to the same post, but this approach is often wasteful in the sense that there may exist an empty post which an agent prefers to his assignment.

To address the wastefulness in an envy-free assignment, Bogomolnaia and Moulin [6] propose competitiveness (CP) as a solution, which ensures that no agent is envious and no post is wasteful. They also introduce top-guarantees , a less demanding criterion than competitiveness, which requires every agent to be assigned to one of his top n choices out of all m × n choices, with m and n being the number of posts and agents, respectively; note that each choice represents a post coupled with a congestion level. Top-guarantees is easy to achieve by a simple sorting algorithm [6, Proposition 1]. Furthermore, top-guarantees is necessary for ensuring Nash stability as well as competitiveness, though the latter criterion - competitiveness - may not always exist.

The following example illustrates the different fairness criteria and Figure 1 depicts their relations.

Example 1. Consider two posts A = { a 1 , a 2 } and three agents V = { v 1 , v 2 , v 3 } . The preferences of the agents are as follows, with the second component in each tuple denoting the congestion level:

```
v 1 : ( a 1 , 1) ≻ ( a 1 , 2) ∼ ( a 2 , 1) ≻ ( a 2 , 2) ≻ · · · , v 2 : ( a 1 , 1) ∼ ( a 2 , 1) ≻ ( a 1 , 2) ∼ ( a 2 , 2) ≻ · · · , v 3 : ( a 1 , 1) ≻ ( a 1 , 2) ≻ ( a 2 , 1) ≻ ( a 2 , 2) ≻ · · · . Π 1 : a 1 a 2 v 2 , v 3 v 1 Π 2 : a 1 a 2 v 1 , v 3 v 2
```

Roughly speaking, without congestion, both v 1 and v 3 strictly prefer a 1 to a 2 . However, while v 1 is indifferent between sharing a 1 with another agent and being alone at a 2 (indicated by ( a 1 , 2) ∼ ( a 2 , 1) ), v 3 will only prefer to go to a 2 when a 1 has more than two congestions. Agent v 2 is indifferent between a 1 and a 2 in general and cares only about congestion.

There are several Nash stable assignments, e.g., Π 1 : assigning v 2 and v 3 to a 1 , and v 1 to a 2 : v 1 will not deviate to a 1 since otherwise the congestion of a 1 will be increased to three and v 1 prefers ( a 2 , 1) to ( a 1 , 3) (due to the congestion aversion assumption), and similar reasoning applies to v 2 and v 3 . However, it is not envy-free (and hence not competitive) since agent v 2 envies v 1 . On the other hand, assigning agent v 1 and v 3 to post a 1 , and agent v 2 to post a 2 (see Π 2 ) is competitive, and hence envy-free and top-guaranteed.

If both v 1 and v 3 have the same preferences as v 2 , however, then no competitive assignments exist since at least one post a is assigned no more than one agent, so either it is wasteful or another agent will envy the agent assigned to a . In this case, Π 1 and Π 2 are still Nash stable (and hence top-guaranteed) but not envy-free or competitive anymore.

Competitiveness offers a promising resolution for congested assignments, as it ensures nonwastefulness, envy-freeness, and top-guarantees, thereby securing a specific welfare level for all participants. Despite their appeal, the computational complexity of determining competitive assignments remains open, as noted by Bogomolnaia and Moulin [6]. This unresolved issue leads us to our core research question:

Q1: Is there an efficient way to determine whether a competitive assignment exists?

Since CP may not always exist, we consider two relaxations:

- Q2: How hard is it to find a top-guaranteed and envy-free assignment?
- Q3: How about top-guaranteed and maximally competitive assignments?

We tackle these questions by exploring the computational intricacies involved in achieving competitiveness, top-guarantees, and envy-freeness in congested assignments.

Our main contributions. We resolve Q1 affirmatively by showing that determining the existence of a competitive assignment is polynomial-time solvable. We also establish the NP-hardness of finding top-guaranteed and envy-free assignments (Q2), and of top-guaranteed and maximally competitive assignments (Q3).

To answer Q1, we first show two key insights: (1) We can restrict our search to CP assignments where every post is non-empty; see Lemma 3. (2) We can use maximum flow to find the congestion vector of a CP assignment with all posts being non-empty and derive the corresponding CP assignment (if it exists) in polynomial time. For Q2 and Q3, we provide hardness reduction from the NP-complete problems EXACT COVER BY 3 -SETS and CLIQUE, respectively. We complement the hardness results by providing several parameterized algorithms.

Related work. The congested assignment problem is rooted in congestion games , introduced by Rosenthal [36]. In these games, agents select subsets of resources (posts), with each resource having a cost function dependent on its usage level. An agent's total cost is the sum of costs over their selected resources. Rosenthal proved the existence of Nash equilibria , while Milchtaich [33] extended the model with player-specific cost functions and demonstrated the same existence property. The field has evolved through both theoretical and computational perspectives [34, 28, 8, 10, 26, 2, 32, 11].

Recently, Bogomolnaia and Moulin [6] introduced two significant fairness concepts for congested assignments: top-guarantees and competitiveness . The former ensures a minimum welfare guarantee for each agent, while the latter addresses a more complex balance of envy-freeness and non-wastefulness. top-guarantees differs from the classical minimax principle in game theory: while minimax aims to minimize the maximum cost of any participant, top-guarantees establishes a hard constraint ensuring every agent achieves a certain welfare threshold. This connects our work to other assignment problems utilizing the minimax principle, such as MIN-REGRET STABLE MATCHING [24, 31], which finds stable matchings that minimize the worst rank of any assigned partner. Bogomolnaia and Moulin demonstrated that top-guarantees is straightforward to achieve, but left the complexity of finding competitive assignments as an open question. Our primary contribution is answering this question by providing a novel polynomial-time algorithm for determining competitiveness.

Congested assignment can be viewed as a restricted variant of the GROUP ACTIVITY SELECTION (GAS) problem, introduced by Darmann et al. [18, 19] and subsequently extended and studied by many others [30, 14-17, 27, 23, 21]. GAS extends the congested assignment concept by allowing agents' preferences over pairs of activities and group sizes, not necessarily exhibiting aversion to larger groups. Despite its broader scope, the specific challenges of applying competitiveness or top-guarantees within GAS have not been previously addressed.

Congestion-averse preferences in congested assignment are similar to the preferences of the agents in anonymous hedonic games [4], where agents form coalitions based solely on group size preferences rather than specific membership.

Congested assignment is also related to Copland's SCHOOL CHOICE WITH CLASS SIZE EXTERNALITIES (SC-CSE) problem [12], which generalizes the classical SCHOOL CHOICE problem [1]. Compared to our setting, every post in SC-CSE also has a capacity bound and a preference list, which ranks all agents in strict order and the goal is to find a stable assignment, i.e., an assignment without justified envies and wastefulness . Phan et al. [35] investigate a model similar to Copland. Instead of post-congestion pairs, they assume that the preferences are over pairs of posts and 'resource ratio,' and investigate an equilibrium notion that is specific to their model. When imposing lower and upper quotas on the capacities of the posts, our problem is related to STABLE MATCHINGS WITH UPPER AND LOWER QUOTAS (SM-ULQ) [5, 25, 3, 9]. Since SM-ULQ is NP-hard, one could obtain the same hardness for finding a competitive assignment with lower and upper quotas.

Finally, congested assignment may be related to the SCHEDULING TO MAXIMIZE PARTICIPATION problem [7], where servers correspond to posts and clients agents such that each client has capacity

bound for each server and will not be satisfied if that bound is exceeded. The goal is to assign clients to servers maximizing the number of 'satisfied' clients.

For a comprehensive discussion on related works in congested assignments, we refer readers to the recent paper by Bogomolnaia and Moulin [6].

Structure of the paper. In Section 2, we introduce necessary definitions and concepts for the paper, and describe an approach to determining whether a competitive assignment for a given congestion vector exists. In Section 3, we present our main result, an efficient algorithm for competitive assignments. In Section 4, we show NP-hardness for top-guaranteed assignments that are envy-free or maximally competitive, respectively, and provide some parameterized algorithms for these two problems. We conclude in Section 5 for future research. Due to space constraints, the proofs of statements marked by ( ⋆ ) are deferred to the appendix.

## 2 Preliminaries

Given a non-negative integer z , let [ z ] denote the set { 1 , . . . , z } . We assume basic knowledge of parameterized complexity and refer to the textbook by Cygan et al. [13] for more details.

Let A = { a 1 , . . . , a m } denote a finite set of m posts and V = { v 1 , . . . , v n } a finite set of n agents . 1 The input of CONGESTED ASSIGNMENT consists of A , V , and for each agent v ∈ V , a preference list ⪰ v (i.e., a weak order, which is transitive and complete) on the set of tuples A × [ n ] (i.e., ordered pairs). The weak order ⪰ v specifies the preferences of agent v over the posts and their congestions (i.e., the number of agents that will simultaneously occupy the post). We use ∼ v to denote the symmetric part of ⪰ v and ≻ v the asymmetric part; we neglect the subscript v if it is clear from the context which agent we refer to. The agents may be indifferent between different posts, but are averse to congestion, i.e., for each post a j ∈ A and each congestion level d ∈ [ n -1] , each agent v has ( a j , d ) ≻ v ( a j , d +1) . For instance, in Example 1, agent v 1 : ( a 1 , 2) ∼ ( a 2 , 1) means that v 1 is indifferent between ( a 1 , 2) and ( a 2 , 1) . This also implies that he has ( a 1 , 1) ≻ ( a 2 , 2) (because of aversion of congestion), meaning that he strictly prefers being assigned to post a 1 alone over to post a 2 with two agents.

An assignment of agents V to posts A is a partition Π = ( S a ) a ∈ A of V where S a is the set of agents assigned to post a so that every two sets S a and S b are mutually disjoint and ⋃ a ∈ A S a = V . The cardinality | S a | of S a is called the congestion of post a . We say that a post a is empty if S a = ∅ . For brevity's sake, we use Π( v ) to refer to the post that agent v is assigned to and often use Π( a ) to refer to the set S a of agents that are assigned to post a . We call ⃗ s = ( | Π( a ) | ) a ∈ A the congestion vector of partition Π .

Definition 1 (Nash stable, envy-free, top-guaranteed, and competitive assignments) . Let Π = ( S a ) a ∈ A denote an assignment for an instance ( A,V, ( ⪰ v ) v ∈ V ) of CONGESTED ASSIGNMENT.

We say that Π is Nash stable (in short, NS ) if no agent wishes to deviate to another post. Formally, Π is NS if for every agent v ∈ V and every post a ∈ A it holds that ( a ∗ , | S a ∗ | ) ⪰ v ( a, | S a | +1) , where a ∗ denotes the post that agent v is assigned to.

We say that agent v envies agent v ′ if ( b, | S b | ) ≻ v ( a, | S a | ) where v is assigned to a and v ′ to b . Accordingly, we say that Π is envy-free (in short, EF ) if no agent envies any other agent.

We say that Π is wasteful if there exists an agent v and an empty post a such that v strictly prefers ( a, 1) ≻ v ( a ∗ , | S a ∗ | ) , where a ∗ denotes the post that agent v is assigned to; otherwise Π is non-wasteful (in short, NW ).

We say that Π is top-guaranteed (in short, TG ) if every agent v ∈ V is assigned to a post a = Π( v ) such that ( a, | Π( a ) | ) is among the | V | tuples in the preference list ⪰ v , breaking ties arbitrarily.

We say that an agent v ∈ V is satisfied with Π if he neither envies any other agent nor prefers to move to an empty post, i.e., for every post a ∈ A it holds that ( a ∗ , | S a ∗ | ) ⪰ v ( a, max( | S a | , 1)) , where a ∗ denotes the post that agent v is assigned to. Otherwise, we say that v is unsatisfied with Π . Accordingly, Π is competitive (in short, CP ) if every agent is satisfied with Π . Π is maximally competitive if it admits the fewest number of unsatisfied agents among all assignments.

1 A and V are standard notion for the set of alternatives and voters, respectively, from voting theory. We adopt them since the agents also have preferences.

```
ALGORITHM 1: Determining the existence of CP assignments Input: An instance I ′ = ( A,V, ( ⪰ v ) v ∈ V ) . Output: A CP assignment if it exists; otherwise no. 1 foreach k = max(0 , m -n ) to m -1 do 2 Compute I = ( A,V, ( ⪰ v ) v ∈ V ) according to Construction 1 on input ( I ′ , k ) ▷ Decide whether there exists a CP assignment for I with all posts being non-empty. 3 T [ a ] ← 1 for all a ∈ A 4 while ∑ a ∈ A T [ a ] ≤ | V | do ▷ Phase 1 : Deciding existence of a perfect flow 5 Let ( G = ( ˆ A ∪ ˆ V ∪ { s, t } , E ) , c ) be the network constructed by Construction 2 on input ( I, T ) 6 Compute a max flow f of ( G,c ) 7 if f has value | V | then return the assignment derived from f as per Definition 2 without the k dummy agents. ; ▷ Phase 2 : find an obstruction 8 Find a vertex ˆ v ∗ ∈ ˆ V with f (ˆ v ∗ , t ) = 0 9 V ′ ←{ ˆ v ∗ } ; A ′ ←∅ 10 repeat 11 ˆ a ← a vertex in ˆ A \ A ′ with (ˆ a, ˆ v ) ∈ E for some ˆ v ∈ V ′ 12 A ′ ← A ′ ∪ { ˆ a } 13 V ′ ← V ′ ∪ { ˆ v ∈ ˆ V | f (ˆ a, ˆ v ) = 1 } 14 until no ˆ a ∈ ˆ A \ A ′ exists with (ˆ a, ˆ v ) ∈ E for some ˆ v ∈ V ′ ; ▷ Phase 3 : update T 15 foreach ˆ a ∈ A ′ do T [ a ] ← T [ a ] + 1 ; 16 return no 17 return no
```

For congestion-averse preferences, we observe the following relations among the five fairness concepts: CP, EF, NS, TG, NW; see Figure 1. Note that the relation of 'CP implies NS' and 'NS implies TG' have already been shown by Bogomolnaia and Moulin [6]. We provide proofs in the appendix for the sake of completeness.

- Lemma 1 ( ⋆ ) . (1) CP implies EF, but the converse does not hold. (2) CP implies NS, but the converse does not hold. (3) NS implies TG and NW, but the converse does not hold. (4) EF is incomparable to NS, to TG, and to NW, respectively; TG is incomparable to NW.

By Lemma 1(2)-(3), when searching for CP assignments, we only need to consider the first | V | tuples of each preferences list. For each agent v and post a , we use λ ( v, a ) to refer to the maximum congestion of v for post a in his preference list ⪰ v such that ( a, λ ( v, a )) is among the first | V | tuples, with ties broken arbitrarily; if no tuple containing a is among the first | V | tuples, then λ ( v, a ) = 0 . Further, we say that post a is acceptable to agent v if λ ( v, a ) &gt; 0 , in other words, a , together with some congestion d , is contained in one of the first | V | tuples in ⪰ v .

```
The maximum congestions in Example 1 are: λ ( v i , a 1 ) = 2 and λ ( v i , a 2 ) = 1 , i ∈ [3] .
```

Milchtaich [33] shows that NS assignments always exist. Indeed, he proves that there always exists a best-reply strategy path connecting an arbitrary assignment to an NS assignment, and such path can be found in polynomial time. CP assignments, however, do not always exist (see Example 1), and it is not clear how to adapt the NS-improvement approach to determine whether CP exists. However, if the congestion levels are given, we can determine in polynomial time whether there exists a CP assignment that match these levels. In the following, we provide an efficient approach to finding a maximally competitive assignment when a congestion vector is given. The idea is to guess (by brute-forcing) the number of unsatisfied agents and determine a perfect ⃗ b -matching in an appropriate bipartite graph between the agents and the posts.

Lemma 2 ( ⋆ ) . Given a congestion vector ⃗ s with ∑ a ∈ A ⃗ s [ a ] = | V | , in polynomial-time one can determine the smallest number t of unsatisfied agents among all assignments whose congestion vectors equal ⃗ s ; the corresponding assignment can found in polynomial time.

## 3 Algorithms for CP Assignments

In this section, we present a comprehensive analysis of our approach to CP assignments. We provide an overview of the procedure (Algorithm 1) before exploring its theoretical foundations in detail. The algorithm consists of two main nested loops. The outer loop enumerates possible values of k , representing the number of empty posts in a potential CP assignment. For each k , we construct an extended instance I that models the original problem with exactly k empty posts; see line 1. Within the inner while -loop (lines 4-16), we determine whether I admits a CP assignment with all posts being non-empty: Starting with a congestion vector T where T [ j ] = 1 for all j ∈ [ m ] , we iteratively refine this vector by building a flow network corresponding to T , determining the maximum flow, and either deriving a CP assignment when the maximum flow value is n , or increasing some entries in T , or terminating when ∑ j ∈ [ m ] T [ j ] &gt; n , inferring no CP assignment exists where all posts are non-empty.

In the following, we provide rigorous justification for our approach. Section 3.1 establishes why we can restrict our search to assignments where every post is non-empty, while Section 3.2 demonstrates the correctness of our maximum flow formulation for finding CP assignments with all posts being non-empty, as well as a correctness proof of Algorithm 1.

## 3.1 Reducing to Determining CP Assignments with No Empty Posts

In this subsection, we show how to reduce our problem to the restricted problem of deciding a CP assignment with all posts being non-empty, ensuring line 2 in Algorithm 1 is correct. The basic idea is to guess the number k of empty posts (assuming CP exists) and augment the instance with k dummy agents, two auxiliary agents and two fallback posts such that any CP assignment must assign to each previously empty post a distinct dummy agent. The core of the reduction is described in Construction 1 below; see Appendix B.1 for an example.

Construction 1 (Extended instance) . Given an instance I = ( A,V, ( ⪰ v ) v ∈ V ) of CONGESTED ASSIGNMENT with m posts and n agents, and a number k with max(0 , m -n ) ≤ k ≤ m -1 construct a new instance ( A ∗ , V ∗ , ( ⪰ ∗ v ) v ∈ V ∗ ) as follows. Create k dummy agents u 1 , . . . , u k , two auxiliary agents p 1 , p 2 , and 2 dummy posts b 1 , b 2 . Set V ∗ = V ∪{ u 1 , . . . , u k , p 1 , p 2 } and A ∗ = A ∪{ b 1 , b 2 } . Finally, define the preferences of the agents as follows 2 . Here, v denotes an original agent from I and ⪰ v his original preference list, but restricted to the first n tuples, z ∈ [ k ] , N := k + n +2 .

<!-- formula-not-decoded -->

|     | a 1         | · · ·   | a m         | b 1   | b 2   |
|-----|-------------|---------|-------------|-------|-------|
| u z | 1           | · · ·   | 1           | N - m | 0     |
| p 1 | 0           | · · ·   | 0           | 1     | N     |
| p 2 | 0           | · · ·   | 0           | N     | 1     |
| v   | λ ( v,a 1 ) | · · ·   | λ ( v,a m ) | k +2  | 0     |

The table above states the maximum congestion of each agent towards each post, where a j , j ∈ [ m ] , and v denote the original post and agent, respectively.

To prove the correctness, we first make an observation about the extended instance.

Observation 1. Let I k = ( A ∗ , V ∗ , ( ⪰ ∗ v ) v ∈ V ∗ ) denote the instance created by Construction 1 with A ∗ = A ∪ { b 1 , b 2 } and V ∗ = V ∪ { u i | i ∈ [ k ] } ∪ { p 1 , p 2 } . Every CP assignment of I k (if it exists) satisfies the following: (1) p 1 is assigned to b 1 alone, and p 2 to b 2 alone. (2) Every dummy u z with 1 ≤ z ≤ k is assigned to some a j ∈ A alone. (3) Every original v i ∈ V is assigned to some original post.

Proof. Let Π be a CP assignment of I k with Π = ( S a ) a ∈ A ∗ . To show the first part of statement (1), we observe that if p 1 is assigned to b 1 , then | S b 1 | = 1 , due to the maximum congestion of p 1 towards b 1 . Thus, it suffices to show that p 1 is indeed assigned to b 1 . Suppose, for the sake of contradiction, that p 1 was assigned to b 2 instead; note that he will not be assigned to any original post a j due to his maximum congestion towards a j . Then, by the maximum congestion of p 2 towards b 2 , agent p 2 could not be assigned to b 2 . No original agent could be assigned to b 2 either, due to his maximum congestion towards b 2 . Hence, we would have S b 2 = { p 1 } and p 2 would envy p 1 , a contradiction to the competitiveness.

2 By Lemma 1(2)-(3), we only need to consider the first n + k +2 tuples in each preference list.

The second part of statement (1) follows directly from the first part since b 2 is the only acceptable post left for p 2 and his maximum congestion towards b 2 is one.

By statement (1) and the maximum congestions, every dummy agent u z can only be assigned to some original post alone, proving statement (2). The last statement follows directly from the first two statements.

Now, we show the correctness of the construction.

Lemma 3. An instance I admits a CP assignment if and only if there exists an integer k with max(0 , m -n ) ≤ k ≤ m -1 such that the instance I k created by Construction 1 admits a CP assignment with all posts being non-empty.

Proof. Let I = ( A,V, ( ⪰ v ) v ∈ V ) . The 'only if' part is straightforward: Let Π = ( S a ) a ∈ A denote a CP assignment of I , and let A ′ denote the set of empty posts under Π with k = | A ′ | . Clearly, (0 , m -n ) ≤ k ≤ m -1 . Consider the instance I k created according to Construction 1 on ( I, k ) . We claim that the following assignment Π k for I k is CP where every post is non-empty.

- -For each a ∈ A ′ , take a unique dummy agent u z and assign Π k ( a ) = { u z } ; note that there are exactly k = | A ′ | many dummy agents.
- -For each non-empty post a ∈ A \ A ′ , let Π k ( a ) = Π( a ) .
- -Let Π k ( b 1 ) = { p 1 } and Π k ( b 2 ) = { p 2 } .

We continue to show why the derived assignment Π k is CP for I k . Since no post is empty, showing competitiveness reduces to showing that no agent is envious. This is clearly the case for all dummy agents including p 1 and p 2 since they are assigned to one of their most preferred posts alone. No original agent envies any other original agent or any dummy agent since Π is CP for I . No original agent v ∈ V envies p 1 or p 2 since p 1 and p 2 are assigned to b 1 and b 2 , Π is TG for I , and b 1 and b 2 occurs at the end of ⪰ ∗ v . This shows that Π k is CP for I k , as desired.

For the 'if' part, let k be an integer between max(0 , m -n ) and m -1 such that the created instance I k admits a CP assignment Π k without empty post. We claim that the assignment Π derived from Π k by omitting all dummy agents and the posts p 1 and p 2 is CP for I .

We first show that Π is a valid assignment for I . By Observation 1(2), every dummy agent is assigned to some original post alone , and hence for each a j ∈ A that is not assigned any dummy agent (i.e., { u 1 , . . . , u k } ∩ Π k ( a j ) = ∅ ), we have Π( a j ) = Π k ( a j ) ; and Π( a j ) = ∅ , otherwise. This implies that Π( a j ) ⊆ V .

By Observation 1(3), every original agent is assigned to an original post, confirming that Π is indeed a valid assignment for I .

Next, suppose, for the sake of contradiction, that Π is not competitive and let v and a be an agent and a post, respectively, such that ( a, max( | Π( a ) | , 1)) ≻ v ( a ′ , | Π( a ′ ) | ) where a ′ is the post that v is assigned to by Π . We infer that Π( a ) cannot be empty since otherwise by Construction 1 and by Observation 1(2) we would have that Π k ( a ) = { u z } for some dummy agent u z . This further implies that v envies u z in I k , a contradiction to the competitiveness of Π k . Since Π( a ) is not empty and v ∈ Π( a ′ ) , again by Construction 1 and by Observation 1(2), we have that Π( a ) = Π k ( a ) and Π( a ′ ) = Π k ( a ′ ) . Since ⪰ ∗ v is an extension of ⪰ v for each v ∈ A , we obtain that ( a, | Π k ( a ) | ) ≻ ∗ v ( a ′ , | Π k ( a ′ ) | ) a contradiction to the competitiveness of Π k .

## 3.2 Determining CP Assignments with No Empty Posts

In this subsection, we show how the while -loop in lines 4-16 works. As already discussed in the beginning of Section 3, by Lemma 2, we need to determine the desired congestion vector. To achieve this, we iteratively update an integer table T , which stores the congestion level for each post that a CP assignment should not fall below. Each iteration has three phases. In Phase 1 (lines 5-7), we construct a flow network with capacities corresponding to T and determine whether there exists a perfect flow, i.e., the value of the flow is equal to the number of agents. If this is the case, we derive and return the corresponding CP assignment (line 7). Otherwise, we proceed with Phase 2 (lines 8-14), where we find an obstruction (see Definition 3) containing posts whose congestion levels need to be incremented necessarily. In Phase 3 (line 15), we update the table entries of the posts from the

obstruction. In the remainder of the subsection, we address these phases in details. We first introduce necessary concepts, starting flow networks; see Example 2 for an illustration.

Construction 2 (Flow network) . Given an instance I = ( A,V, ( ⪰ v ) v ∈ V ) together with a congestion table T which has an entry 1 ≤ T [ a ] ≤ | V | for each post a ∈ A , we construct a network N = ( G,c ) , where G = ( ˆ A ∪ ˆ V ∪ { s, t } , E ) is a directed graph with dedicated source s and target t , and c : E ( G ) → [ | V | ] is a capacity function:

- (i) For each a ∈ A , create a vertex ˆ a . Let ˆ A = { ˆ a | a ∈ A } .
- (ii) For each v ∈ V , create a vertex ˆ v . Let ˆ V = { ˆ v | v ∈ V } .
- (iii) For each post a ∈ A , create an arc ( s, ˆ a ) from the source s and set the capacity c ( s, ˆ a ) = T [ a ] .
- (iv) For each agent v ∈ V , create an arc (ˆ v, t ) to the target t and set the capacity c (ˆ v, t ) = 1 .
- (v) For each agent v ∈ V and each post a , if v considers ( a, T [ a ]) as the most preferred tuple among all tuples ( a ′ , T [ a ′ ]) , a ′ ∈ A , then create an arc (ˆ a, ˆ v ) with capacity c (ˆ a, ˆ v ) = 1 .

The capacity function c is c ( e ) = T [ a ] if e = ( s, ˆ a ) with a ∈ A ; otherwise c ( e ) = 1 .

Example 2. Consider the first instance in Example 1 and start with T = (1 , 1) . The flow network is given on the right, where a label with ' x : y ' means the corresponding arc has capacity x and a maximum flow has value y on that arc. For this network, the maximum flow value is 2 . It is not a perfect flow since agent v 3 is not assigned. This implies that no assignment with congestion vector T = (1 , 1) is CP. ˆ v 1 ˆ a 1 ˆ v 2 ˆ v 3 ˆ a 2 s t 1: 1 1: 1 1: 1 1: 1 1: 1 1: 1 1:0 1:0 1:0

<!-- image -->

We need the following concepts to derive an assignment from a flow.

Definition 2 (Perfect flows and the derived assignment) . Let ( G, c ) denote the network created by Construction 2 for an instance I = ( A,V, ( ⪰ v ) v ∈ V ) , together with a table T ∈ [ | V | ] | A | . A flow of ( G, c ) is a function f : E ( G ) → [ | V | ] ∪{ 0 } 3 that assigns to each arc a value, satisfying the following: f ( e ) ≤ c ( e ) for all e ∈ E ( G ) , and ∑ ( x,y ) ∈ E ( G ) f ( x, y ) = ∑ ( z,x ) ∈ E ( G ) f ( z, x ) for all x ∈ ˆ A ∪ ˆ V . The value of a flow equals the net flow into the sink t : v ( f ) = ∑ ( x,t ) ∈ E ( G ) f ( x, t ) . A flow is called perfect if the value of f is | V | ; this means that with a perfect flow, every arc (ˆ v, t ) to the sink is saturated.

Given a perfect flow f , we derive a congested assignment Π for the original instance I by setting Π( a ) = { v | f (ˆ a, ˆ v ) = 1 } for each post a ∈ A .

We also need the concept of obstructions, which are witnesses for the absence of a perfect flow. This is similar to the forbidden substructure of a perfect matching in Hall's marriage theorem.

Definition 3 (Obstruction) . Let ( G, c ) be a network with G = ( ˆ A ∪ ˆ V ∪ { s, t } , E ) . A pair ( A ′ , V ′ ) with A ′ ⊆ ˆ A and V ′ ⊆ ˆ V is called an obstruction for ( G, c ) if the following holds:

- (i) ∅ ̸ = V ′ ⊆ ˆ V ;
- (ii) A ′ = { ˆ a ∈ ˆ A | ∃ ˆ v ∈ V ′ with (ˆ a, ˆ v ) ∈ E ( G ) } , i.e., A ′ is equal to the in-neighborhood of V ′ ;
- (iii) For each ˆ a ∈ A ′ , c ( s, ˆ a ) &lt; |{ ˆ v ∈ V ′ | (ˆ a, ˆ v ) ∈ E }| .
- (iv) ∑ ˆ a ∈ A ′ c ( s, ˆ a ) &lt; | V ′ | .

A ′ can be seen as a minimal set of posts a with congestions that are not enough to accommodate all agents from V ′ according the congestion table T .

Throughout the remainder of the section, by an iteration , we mean the execution of lines 5-7 if a CP assignment is found, and lines 5-15 otherwise. For each iteration z ≥ 1 , we use T z to denote the table at the beginning of iteration z (i.e., at line 5).

The correctness of the three phases is based on Lemmas 4 to 6 which we present next. Lemma 4 guarantees that the second phase always finds some critical posts to increment their congestions.

Lemma 4 ( ⋆ ) . Each ( A ′ , V ′ ) computed in Phase 2 in lines 8-14 is an obstruction.

Lemma 5 ensures that increasing the table entries is safe and that the no-answer in line 16 is correct.

3 Note that since the capacity values are integral, we can assume without loss of generality that the flow is a also integral.

Lemma 5 ( ⋆ ) . Assume that I admits a CP assignment Π with no posts being empty. Then, for each iteration z ≥ 1 , each obstruction ( A ′ , V ′ ) found in iteration z , and each post a ∈ A , the following holds. If ˆ a ∈ A ′ , then | Π( a ) | ≥ T z [ a ] + 1 ; otherwise | Π( a ) | ≥ T z [ a ] .

Lemma 6 ensures that the returned assignment is CP.

Lemma 6 ( ⋆ ) . If Π is an assignment returned in line 7, then Π is CP and has no empty post.

We have everything ready to show the correctness.

Theorem 1 ( ⋆ ) . Algorithm 1 correctly decides whether an instance has a CP assignment in O ( m 2 · ( n + m ) 2 ) time, where m and n denote the number of posts and agents, respectively.

Proof. Let I ′ be an instance. By Lemma 3, we only need to show that I ′ is a yes instance if and only if there exists a k ∈ { max(0 , m -n ) , . . . , m -1 } such that line 7 returns an assignment where every post is non-empty and is CP for the instance I constructed in line 2.

This reduces to showing that lines 3-16 correctly decide whether I admits a CP assignment with all posts being non-empty. Clearly, if line 7 returns an assignment Π , then by Lemma 6, Π is CP and every post is assigned at least one agent. Hence, to show the correctness, we need to show that whenever line 16 returns no, I does not admit a CP assignment where every post is non-empty. Towards a contradiction, suppose that I admits a CP assignment, say Π , where every post is nonempty. Since line 16 returns no, in the second last iteration z , we have ∑ a ∈ A T z [ a ] ≤ | V | , but after the update of some table entries the sum exceeds | V | . Let T z +1 denote the table entries at the end of iteration z . By assumption, ∑ a ∈ A T z +1 [ a ] &gt; | V | and ∑ a ∈ A T z [ a ] ≤ | V | .

Since we updated some table entries, we must have found an obstruction ( A ′ , V ′ ) in iteration z according to which we made the update. By Lemma 5, we infer that for all posts a ∈ A , it holds that | Π( a ) | ≥ T z [ a ] + 1 = T z +1 [ a ] if ˆ a ∈ A ′ , and | Π( a ) | ≥ T z [ a ] = T z +1 [ a ] if ˆ a / ∈ A ′ . This implies that | Π( a ) | ≥ T z +1 [ a ] holds for all a ∈ A . Then, ∑ a ∈ A | Π( a ) | ≥ ∑ a ∈ A T z +1 [ a ] &gt; | V | , a contradiction to Π being a valid assignment. The running time analysis is deferred to Appendix B.6

## 4 Two Optimization Variants

In this section, we continue with Q2 and Q3 from the introduction. Specifically, we investigate the computational complexity of finding TG assignments that are additionally either EF or maximally CP.

EF and TG assignments. We first focus on EF and TG assignments, and show that it is NP-hard to find such assignments. Let EF+TG refer to the problem of determining whether a given instance has an EF and TG assignment. We reduce from the NP-complete EXACT COVER BY 3-SETS (X3C) problem. The input of X3C is a pair ( U, S ) , where U = { u 1 , . . . , u 3 n } is a finite set of 3 n elements, and S is a family of subsets S = { C 1 , . . . , C m } with C j ⊆ U and | C j | = 3 for each j ∈ [ m ] . The question is whether there exist an exact cover J ⊆ [ m ] for U , i.e., | J | = n and ⋃ j ∈ J C j = U . Note that X3C remains NP-hard even if each element appears in exactly three subsets [22], meaning that m = 3 n .

Theorem 2 ( ⋆ ) . EF+TG is NP-complete; hardness holds even if there are no ties.

Proof. NP-containment is clear since one can check in polynomial time whether a given assignment is EF and TG. Now, we focus on NP-hardness and reduce from X3C. Let I = ( U, S ) denote an instance of X3C with U = { u 1 , . . . , u 3 n } and S = ( C j ) j ∈ [ m ] such that every element in U appears in exactly three members of S ; note that m = 3 n ≥ 3 .

We create an instance I ′ of CONGESTED ASSIGNMENT as follows. For each member C j ∈ S , create a set-post a j ; For each element u i ∈ U , create an element-agent v i ; Create two dummy posts b 1 and b 2 and 4 m dummy agents p 1 , p 2 , . . . , p 2 m , q 1 , q 2 , . . . , q 2 m . Let A = { a j | j ∈ [ m ] } ∪ { b 1 , b 2 } and V = { v i | i ∈ [3 n ] } ∪ { p j , q j | j ∈ [2 m ] } .

We describe the preferences of the agents, where the last ' · · · ' denote an arbitrary but fixed order of the rest of the tuples. ⟨ α, s, t ⟩ = ( α, s ) ≻ ( α, s +1) ≻ · · · ≻ ( α, t ) depict the preference list on tuples for post α and congestions ranging between s and t with s ≤ t .

- The preferences of agent v i is defined as follows, where C j , C k , C t denote the three members in S that contain u i with j &lt; k &lt; t : v i : ( a j , 1) ≻ ( a k , 1) ≻ ( a t , 1) ≻ ( a j , 2) ≻ ( a k , 2) ≻ ( a t , 2) ≻ ( a j , 3) ≻ ( a k , 3) ≻ ( a t , 3) ≻ ⟨ b 2 , 1 , 3 n +4 m -9 ⟩ ≻ · · · .

In other words, each agent v i considers the three posts which correspond to the sets that contain u i most acceptable, followed by b 2 . He does not consider any other post acceptable.

- All dummy agents p j , j ∈ [2 m ] , have p j : ( a 1 , 1) ≻ . . . ≻ ( a m , 1) ≻ ( a 1 , 2) ≻ . . . ≻ ( a m , 2) ≻ ⟨ b 1 , 1 , 2 m +3 n ⟩ ≻ · · · .

Briefly put, each dummy agent p j always wants to go to a set-post with congestion one or two.

- All dummy agents q j , j ∈ [2 m ] , only consider b 1 and b 2 acceptable, but prefers b 2 over b 1 :
- q j : ⟨ b 2 , 1 , 2 m ⟩ ≻ ⟨ b 1 , 1 , 2 m +3 n ⟩ ≻ · · · .

Clearly, the construction can be done in polynomial time. One can verify that the constructed preferences do not contain ties. The maximum congestions of the agents are depicted in the following table, where v i is an element-agent with u i appearing in C 1 , C 2 , C m :

|     |   a 1 |   a 2 |   a 3 | · · ·   |   a m | b 1      | b 2          |
|-----|-------|-------|-------|---------|-------|----------|--------------|
| v i |     3 |     3 |     0 | · · ·   |     3 | 0        | 3 n +4 m - 9 |
| p z |     2 |     2 |     2 | · · ·   |     2 | 3 n +2 m | 0            |
| q z |     0 |     0 |     0 | · · ·   |     0 | 3 n +2 m | 2 m          |

The correctness proof is deferred to Appendix C.1

We conclude the study of EF+TG with two simple FPT algorithms. The first algorithm is based on brute-force searching all possible TG assignments while the second one on guessing the empty posts and applying Algorithm 1 that checks whether a CP assignment exists.

Theorem 3 ( ⋆ ) . EF+TG is FPT with respect to the number n of agents and the number m of posts, respectively.

Maximally CP assignments. Now, we turn to maximally CP assignments and define the decision variant MAXCP+TG: Given an instance I and a non-negative integer t , does there exists a TG assignment with at most t unsatisfied agents? We first show that MAXCP+TG is W[1]-hard wrt. the number t of unsatisfied agents; the W[1]-hardness is via reducing from the W[1]-complete CLIQUE problem [20]. Fortunately, when t is constant, the problem can be solved in polynomial time.

Theorem 4 ( ⋆ ) . MAXCP+TG is W[1]-hard and in XP with respect to the number t of unsatisfied agents. The W[1]-hardness holds even if there are no ties.

Using an idea similar to the one for Theorem 3 and by applying the algorithm behind Lemma 2, we obtain further parameterized algorithms for MAXCP+TG.

Theorem 5 ( ⋆ ) . MAXCP+TG is FPT with respect to n , and in XP with respect to m , where n and m denote the number of agents and the number of posts, respectively.

Finally, we show that finding maximally CP assignment remains W[1]-hard even if we give up TG.

Theorem 6 ( ⋆ ) . Deciding whether an instance of CONGESTED ASSIGNMENT has an assignment with at most t unsatisfied agents is W[1]-hard with respect to t .

## 5 Conclusion

We investigated congested assignments with congestion-averse agents, focusing on competitiveness (CP), envy-freeness (EF), and maximal competitiveness (maxCP). We devised a novel network-flowbased algorithm to identify CP assignments. We then proved NP-hardness of finding an assignment that is top-guaranteed and either EF or maxCP. We complement these hardness results with several parameterized algorithms. We also show that relaxing top-guarantees does not reduce the complexity: Finding a maxCP assignment remains NP-hard.

For future research, we suggest exploring congested assignments with weighted agents [6] and scenarios where agents have varying responses to congestion. Additionally, in applications like urban traffic, exploring control management strategies, such as identifying the minimum number of posts to remove to achieve a competitive assignment, presents another intriguing avenue. Considering that our work mainly focuses on providing theoretical analysis of congested assignment complexity, empirically validating the algorithms shown in this paper could be another interesting direction.

## Acknowledgement

This work was supported by the Vienna Science and Technology Fund (WWTF) [10.47379/ VRG18012] and the National Natural Science Foundation of China (Grants No. 61772314, 61761136017 and 62072275). We would like to thank the reviewers for their helpful comments.

## References

- [1] A. Abdulkadiro˘ glu and T. Sönmez. School choice: A mechanism design approach. American economic review , 93(3):729-747, 2003.
- [2] H. Ackermann, H. Röglin, and B. Vöcking. On the impact of combinatorial structure on congestion games. Journal of the ACM , 55(6):25:1-25:22, 2008.
- [3] H. Aziz, S. Gaspers, Z. Sun, and T. Walsh. From matching with diversity constraints to matching with regional quotas. In Proceedings of the 18th International Conference on Autonomous Agents and Multiagent Systems , page 377-385, 2019. ISBN 9781450363099.
- [4] C. Ballester. NP-completeness in hedonic games. Games and Economic Behavior , 49(1):1-30, 2004.
- [5] P. Biró, T. Fleiner, R. W. Irving, and D. Manlove. The College Admissions problem with lower and common quotas. Theoretical Computer Science , 411(34-36):3136-3153, 2010.
- [6] A. Bogomolnaia and H. Moulin. Fair congested assignment. Mathematics of Operation Research , pages 1-19, 2025. URL ❤/a116/a116♣/a115✿✴✴❞♦✐✳♦/a114❣✴✶✵✳✶✷✽✼✴♠♦♦/a114✳✷✵✷✹✳✵✺✽✶ .
- [7] I. Caragiannis, C. Kaklamanis, P. Kanellopoulos, and E. Papaioannou. Scheduling to maximize participation. Theoretical Computer Science , 402:142-155, 2008.
- [8] D. Chakrabarty, A. Mehta, and V. Nagarajan. Fairness and optimality in congestion games. In Proceedings 6th ACM Conference on Electronic Commerce , pages 52-57. ACM, 2005.
- [9] J. Chen, R. Ganian, and T. Hamm. Stable matchings with diversity constraints: Affirmative action is beyond NP. In C. Bessiere, editor, Proceedings of the 29th International Joint Conference on Artificial Intelligence , pages 146-152, 2020.
- [10] G. Christodoulou and E. Koutsoupias. The price of anarchy of finite congestion games. In Proceedings of the 37th Annual ACM Symposium on Theory of Computing , pages 67-73, 2005.
- [11] G. Christodoulou, M. Gairing, Y. Giannakopoulos, D. Poças, and C. Waldmann. Existence and complexity of approximate equilibria in weighted congestion games. Mathematics of Operations Research , 48(1):583-602, 2023.
- [12] A. Copland. School choice and class size externalities, 2023.
- [13] M. Cygan, F. V. Fomin, L. Kowalik, D. Lokshtanov, D. Marx, M. Pilipczuk, M. Pilipczuk, and S. Saurabh. Parameterized Algorithms . Springer, 2015.
- [14] A. Darmann. A social choice approach to ordinal group activity selection. Mathematical Social Sciences , 93:57-66, 2018.
- [15] A. Darmann. Stable and pareto optimal group activity selection from ordinal preferences. International Journal of Game Theory , 47(4):1183-1209, 2018.
- [16] A. Darmann. Manipulability in a group activity selection problem. Social Choice and Welfare , 52(3):527-557, 2019.
- [17] A. Darmann and J. Lang. Group activity selection problems. In U. Endriss, editor, Trends in computational social choice , pages 87-103. AI Access, 2017.
- [18] A. Darmann, E. Elkind, S. Kurz, J. Lang, J. Schauer, and G. J. Woeginger. Group activity selection problem. In Proceedings of the 8th Workshop on Internet and Network Economics , volume 7695 of Lecture Notes in Computer Science , pages 156-169, 2012.

- [19] A. Darmann, E. Elkind, S. Kurz, J. Lang, J. Schauer, and G. J. Woeginger. Group activity selection problem with approval preferences. International Journal of Game Theory , 47(3): 767-796, 2018.
- [20] R. G. Downey and M. R. Fellows. Fixed-parameter tractability and completeness II: On completeness for W[1]. Theoretical Computer Science , 141(1&amp;2):109-131, 1995.
- [21] R. Ganian, S. Ordyniak, and C. S. Rahul. Group activity selection with few agent types. Algorithmica , 85(5):1111-1155, 2023.
- [22] T. F. Gonzalez. Clustering to minimize the maximum intercluster distance. Theoretical computer science , 38:293-306, 1985.
- [23] S. Gupta, S. Roy, S. Saurabh, and M. Zehavi. Group activity selection on graphs: Parameterized analysis. In Proceedings of the 10th international symposium on algorithmic game theory , volume 10504 of Lecture Notes in Computer Science , pages 106-118, 2017.
- [24] D. Gusfield. Three fast algorithms for four problems in stable marriage. SIAM J. Comput. , 16 (1):111-128, 1987.
- [25] K. Hamada, K. Iwama, and S. Miyazaki. The Hospitals/Residents problem with lower quotas. Algorithmica , 74(1):440-465, 2016.
- [26] A. Hayrapetyan, É. Tardos, and T. Wexler. The effect of collusion in congestion games. In Proceedings of the 38th Annual ACM Symposium on Theory of Computing , pages 89-98, 2006.
- [27] A. Igarashi, D. Peters, and E. Elkind. Group activity selection on social networks. In Proceedings of the 31st AAAI Conference on Artificial Intelligence , pages 565-571, 2017.
- [28] H. Konishi, S. Weber, and M. L. Breton. Free mobility equilibrium in a local public goods economy with congestion. Research in Economics , 51:19-30, 1997.
- [29] B. Korte and J. Vygen. Combinatorial Optimization: Theory and Algorithms . Springer, 2007.
- [30] H. Lee and Y. Shoham. Stable invitations. In Proceedings of the 29th AAAI Conference on Artificial Intelligence , pages 965-971, 2015.
- [31] D. F. Manlove, R. W. Irving, K. Iwama, S. Miyazaki, and Y. Morita. Hard variants of stable marriage. Theor. Comput. Sci. , 276(1-2):261-279, 2002.
- [32] C. A. Meyers and A. S. Schulz. The complexity of welfare maximization in congestion games. Networks , 59(2):252-260, 2012.
- [33] I. Milchtaich. Congestion games with player-specific payoff functions. Games &amp; Economic Behavior , 13(27):111-124, 1996.
- [34] M. Milinski. An evolutionarily stable feeding strategy in sticklebacks. Ethology , 51(1):36-40, 1979.
- [35] W. Phan, R. Tierney, and Y. Zhou. Crowding in school choice. Technical report, Kyoto University, 2021.
- [36] R. W. Rosenthal. A class of games possessing pure-strategy nash equilibria. International Journal of Game Theory , 2:65-67, 1973.
- [37] Y. Shoham and K. Leyton-Brown. Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations . Cambridge, 2012.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We provide proofs for all stated results. They can be found in the main part or appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The model studied in our paper is a theoretical and abstract model. Our analysis is based on worst case analysis. There are no experiments.

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

Justification: We provide proofs for all stated results. They can be found in the main part or appendix.

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

Justification: There are no experiments.

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

Justification: As mentioned, there are no experiments. But all stated results are proved in the main part or appendix.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( ❤/a116/a116♣/a115✿✴✴♥✐♣/a115✳❝❝✴ ♣✉❜❧✐❝✴❣✉✐❞❡/a115✴❈♦❞❡❙✉❜♠✐/a115/a115✐♦♥/a80♦❧✐❝② ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( ❤/a116/a116♣/a115✿ ✴✴♥✐♣/a115✳❝❝✴♣✉❜❧✐❝✴❣✉✐❞❡/a115✴❈♦❞❡❙✉❜♠✐/a115/a115✐♦♥/a80♦❧✐❝② ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: As mentioned, there are no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: As mentioned, there are no experiments.

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

Justification: As mentioned, there are no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics ❤/a116/a116♣/a115✿✴✴♥❡✉/a114✐♣/a115✳❝❝✴♣✉❜❧✐❝✴❊/a116❤✐❝/a115●✉✐❞❡❧✐♥❡/a115 ?

Answer: [Yes]

Justification: Our research is theoretical and has no harm to the society or human.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We investigated fair assignment with congestion-averse agents. Fairness is a relevant property that a society would want to achieve. Hence, it could have potentially positive impact.

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

Justification: Our research does not have experiments.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: As mentioned, there are no experiments.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, ♣❛♣❡/a114/a115✇✐/a116❤❝♦❞❡✳❝♦♠✴❞❛/a116❛/a115❡/a116/a115 has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: As mentioned, there are no experiments.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: As mentioned, there are no experiments.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [Yes]

Justification: As mentioned, there are no experiments.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Our research is original.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( ❤/a116/a116♣/a115✿✴✴♥❡✉/a114✐♣/a115✳❝❝✴❈♦♥❢❡/a114❡♥❝❡/a115✴✷✵✷✺✴▲▲▼ ) for what should or should not be described.

## Supplementary Material for the Paper 'Assignments for Congestion-Averse Agents: Seeking Competitive and Envy-Free Solutions'

## A Additional Material for Section 2

## A.1 Proof of Lemma 1

Lemma 1 ( ⋆ ) . (1) CP implies EF, but the converse does not hold.

- (2) CP implies NS, but the converse does not hold.
- (3) NS implies TG and NW, but the converse does not hold.
- (4) EF is incomparable to NS, to TG, and to NW, respectively; TG is incomparable to NW.
4. Proof. (1) Clearly, CP implies EF by definition. Now, to show that the converse does not hold, let us consider Example 1. Clearly assigning every agent to post a 1 is EF, but it is not CP since all agents prefer ( a 2 , 1) to ( a 1 , 3) .
- (2) As already mentioned, the implication has been discussed by Bogomolnaia and Moulin [6] already. For the sake of completeness, we provide a proof by showing the contra-positive. Let Π be an assignment that is not NS and let there be an agent v ∈ V and a post a ∈ A such that v prefers ( a, | Π( a ) | +1) to ( a ∗ , | Π( a ∗ ) | ) where a ∗ denotes the post that v is assigned to. If a is empty, then clearly, Π is wasteful and hence not CP. If a is non-empty, then since every agent is averse against congestions, we infer that v prefers ( a, | Π( a ) | ) to ( a ∗ , | Π( a ∗ ) | ) , and hence not CP either.

Now, to show that the converse does not hold, let us consider Example 1 again. As already discussed there, Π 1 is NS, but not CP.

- (3) That NS implies NW follows directly from definition. That 'NS implies TG' has also been shown by Bogomolnaia and Moulin [6]. Again, for the sake of completeness, we provide a proof here by showing the contra-positive. Let Π be an assignment that is not TG, and let v ∈ V be an agent and a ∗ a post such that v is assigned to a ∗ while ( a ∗ , | Π( a ∗ ) | ) is not among his top-| V | choices. Let X = { ( a, d ) | ( a, d ) ≻ v ( a ∗ , | Π( a ∗ ) | ) } be the set consisting of all tuples that v prefers to ( a ∗ , | Π( a ∗ ) | ) . Then, | X | ≥ | V | . For each a ∈ A , let δ ( a ) denote the largest congestion such that ( a, δ ( a )) ∈ X , i.e., δ ( a ) = max ( a,d ) ∈ X { d } and δ ( a ) = 0 if no tuple ( a, d ) exists in X . Then, ∑ a ∈ A δ ( a ) = | X | ≥ | V | = ∑ a ∈ A | Π( a ) | . By assumption, we have that δ ( a ∗ ) &lt; | Π( a ∗ ) | . This implies that there must exist a post a ∈ A \ { a ∗ } such that δ ( a ) &gt; | Π( a ) | . By definition, we infer that ( a, | Π( a ) | +1) ∈ X , and hence ( a, | Π( a ) | +1) ≻ v ( a ∗ , | Π( a ∗ ) | ) , witnessing that Π is not NS.

It is quite straightforward to come up with a TG and NW assignment which is not NS. Let us consider the following example.

```
v 1 : ( a 1 , 1) ≻ ( a 1 , 2) ≻ ( a 2 , 1) ≻ · · · , v 2 : ( a 1 , 1) ≻ ( a 2 , 1) ≻ ( a 1 , 2) ≻ · · · , v 3 : ( a 3 , 1) ≻ ( a 3 , 2) ≻ ( a 3 , 3) ≻ · · · . Π 3 : a 1 a 2 a 3 v 2 v 1 v 3 Π 4 : a 1 a 2 a 3 v 1 , v 2 v 3
```

Π 3 is clearly TG and NW. It is not NS however, since v 1 prefers ( a 1 , 2) to ( a 2 , 1) .

- (4) Let us consider Example 1. Assigning every agent to the same post is clearly EF, but not TG and not NW. Hence, it is not NS by Statement (3). As already argued in Example 1, Π 1 is NS, TG, and NW, but not EF.

Assigning v 1 and v 2 to a 2 , and v 3 to a 1 is NW, but not TG: For v 1 , tuple ( a 2 , 2) is not in his top 3 choices.

Now, let us consider the example from item. Π 4 is TG but not NW.

By definition, we observe the following:

Observation 2. For an arbitrary tie-breaking rule, ∑ a ∈ A λ ( v, a ) = | V | holds for every v ∈ V .

## A.2 Proof of Lemma 2

Lemma 2 ( ⋆ ) . Given a congestion vector ⃗ s with ∑ a ∈ A ⃗ s [ a ] = | V | , in polynomial-time one can determine the smallest number t of unsatisfied agents among all assignments whose congestion vectors equal ⃗ s ; the corresponding assignment can found in polynomial time.

Proof. The idea is to iterate over all possible number t ∈ { 0 , 1 , . . . , | V |} and check whether there exists an assignment with congestion vector ⃗ s and exactly t unsatisfied agents. The later problem can be solved via determining whether a perfect ⃗ b -matching exists, which can be done in polynomial time [29, Chapter 12].

Let ( A,V, ( ⪰ ) v ∈ V ) be an instance of CONGESTED ASSIGNMENT. To check whether there exists an assignment with congestion vector ⃗ s and exactly t unsatisfied agents, we construct a bipartite graph G on two disjoints X and Y with X = A ∪ { a 0 } and Y = V ∪ { w 1 , . . . , w t } , where the w z 's are the dummy agent-vertices and a 0 is a dummy post-vertex .

We add an edge between every original post a j and every dummy agent-vertex w z , and an edge between the dummy post-vertex a 0 and every original agent v i . We also add an edge between every original post and original agent, but will delete some according to the congestion vector. In other words, the graph on X and Y is almost a complete bipartite graph, except there are no edges between the dummy post a 0 and any dummy agent w z , z ∈ [ t ] .

We delete from the bipartite graph the following edges: For each original agent v and each two original posts a and a ′ , if v prefers ( a, max(1 , ⃗ s [ a ])) to ( a ′ , ⃗ s [ a ′ ]) , then we delete the edge { a ′ , v } . This is because if v would be satisfied, he will never be assigned to a ′ since either the post is wasteful or he envies some agent that is assigned to a .

This completes the construction of the graph G . We check whether there exists a perfect ⃗ b -matching 4 for G where ⃗ b [ a 0 ] = t , ⃗ b [ a j ] = ⃗ s [ a j ] for all a j ∈ A , and ⃗ b [ y ] = 1 for all y ∈ Y . We answer no if no such matching exists. Otherwise, let M be the perfect ⃗ b -matching, and we return the following partition as assignment Π . For each post a j ∈ A and agent v i ∈ V , let Π( v i ) = a j if { v i , a j } ∈ M . Let V ′ be the remaining agents that are unassigned; note that these agents are matched to a 0 by definition. As long as the congestion of some a j ∈ A is not equal to ⃗ s [ a j ] = ⃗ b [ a j ] , pick an agent from V ′ and assign him to a j .

For the correctness, it is straightforward that if Π is an assignment with congestion vector ⃗ s and exactly t unsatisfied agents V ′ , then the following matching is a perfect ⃗ b matching: Let M ( v i ) = a j if v i ∈ V \ V ′ , and M ( v i ) = a 0 if v i ∈ V ′ . Finally, for each original post a j , if it is assigned ˆ n unsatisfied agents, then we pick ˆ n distinct dummy agent-vertices and match them to a j .

If M is a perfect ⃗ b -matching for G , then a 0 is matched with exactly t original agents V ′ who will be the unsatisfied agents. Clearly, the assignment Π given by our algorithm above has the desired congestion vector ⃗ s . We show that only the agents from V ′ may be unsatisfied. Consider an arbitrary agent v i ∈ V \ V ′ and post a j ∈ A \ { Π( v i ) } . For a contradiction, suppose v i prefers ( a j , max(1 , | Π( a j ) | )) to ( a ∗ , | Π( a ∗ ) | ) , where v i is assigned to a ∗ , i.e., { v i , a ∗ } ∈ M . By the definition of Π , it follows that v i prefers ( a j , max(1 , ⃗ s ( a j ))) to ( a ∗ , ⃗ s ( a ∗ )) , implying that edge { v i , a ∗ } does not exist in the constructed bipartite graph and cannot be matched under M , a contradiction.

Since checking the existence of a perfect ⃗ b -matching and finding such a matching if it exists can be done in polynomial time by reducing to finding a perfect matching, the whole approach can be done in polynomial time as well. This completes the proof.

For an illustration, consider the first instance in Example 1. Let the congestion vector be ⃗ s = (2 , 1) and the number of unsatisfied agents be t = 0 . The following bipartite graph G has a perfect ⃗ b -matching, indicated by the red lines.

| ⃗ b :   |   a 0 |   ˆ a 1 |   ˆ a 2 |   ˆ v 1 |   ˆ v 2 |   ˆ v 3 |
|---------|-------|---------|---------|---------|---------|---------|
| ⃗ b :   |     0 |       2 |       1 |       1 |       1 |       1 |

<!-- image -->

Indeed, the corresponding ⃗ b -matching yields a CP assignment which is Π 2 and it has only satisfied agents.

4 A ⃗ b -matching M is perfect if ∑ e ∈ M : u ∈ e 1 = ⃗ b [ u ] holds for all vertices u .

## B Additional Material for Section 3.1

## B.1 Example of Construction 1

Let us consider the four agents with the following preference lists.

```
v 1 : ( a 1 , 1) ≻ ( a 2 , 1) ≻ ( a 3 , 1) ∼ ( a 1 , 2) ∼ ( a 2 , 2) ≻ · · · , v 2 , v 3 : ( a 2 , 1) ≻ ( a 1 , 1) ≻ ( a 3 , 1) ∼ ( a 1 , 2) ∼ ( a 2 , 2) ≻ · · · , v 4 : ( a 1 , 1) ∼ ( a 2 , 1) ≻ ( a 1 , 2) ∼ ( a 2 , 2) ≻ ( a 3 , 1) ≻ · · · ,
```

One can observe that if every post is non-empty, then a 1 or a 2 will have congestion one. If a 1 has congestion one, then v 1 has to be assigned to a 1 alone and v 4 to a 2 alone, leaving v 2 and v 3 to be envious. If a 2 has congestion one, then v 2 or v 3 will be envious. One can verify that assigning any two agents to a 1 and the remaining two to a 2 is competitive, leaving a 3 empty.

Now, let us 'guess' that the number of empty post is k = 1 . For k = 1 , we augment the instance with one dummy agent u 1 and two auxiliary agents p 1 and p 2 , and two dummy posts b 1 and b 2 . Their preference lists are as follows:

<!-- formula-not-decoded -->

One can verify that in the original instance, every CP assignment will leave a 3 empty, and in the augmented instance, every CP assignment will assign the dummy agent u 1 to a 3 alone. The correctness is given by Lemma 3.

## B.2 Example of Algorithm 1

Consider the first instance in Example 1 and let us assume that we are in the case with k = 0 , meaning that we can ignore a 0 and the dummy agents. Initially, T [ a 1 ] = T [ a 2 ] = 1 , implying that the condition in Line 4 is satisfied, so we can start with the first iteration. In the first iteration, where T = (1 , 1) , the network and its maximum flow constructed in Phase 1 (Lines 5-6) have been discussed in Example 2 already. Hence, we proceed with line 7. Let the maximum flow f be as indicated in Example 2, i.e., f (ˆ a 1 , t ) = f (ˆ a 2 , t ) = 1 and f (ˆ a 3 , t ) = 0 . Since f does not have value | V | = 3 , we proceed with Phase 2, where we find an obstruction in Lines 8-14. We start with V ′ = { ˆ v 3 } and A ′ = ∅ since ˆ v 3 is the only vertex with f (ˆ v 3 , t ) = 0 . In the while-loop in lines 10-14, we first find ˆ a 1 and compute A ′ = { ˆ a 1 } . Then, we compute V ′ = { ˆ v 3 , ˆ v 1 } in line 13. Since no further post ˆ a exists that has an arc to any agent ˆ v in V ′ (see the figure in Example 2), we stop with A ′ = { ˆ a 1 } and V ′ = { ˆ v 3 , ˆ v 1 } . One can check that if a 1 would stay with congestion one, then no CP assignment can exist since both v 1 and v 3 would envy the only agent that is assigned to a 1 . We will later show that in order to have a CP assignment, it is necessary to increase the congestion of every post in A ′ . In Phase 3, we increment T [ a 1 ] to 2 , while the other post stays with T [ a 2 ] = 1 . This completes the first iteration.

At line 4, since T [ a 1 ] + T [ a 2 ] = 3 ≤ | V | , we continue with the second iteration. In the following, the tuples considered in Construction 2(v) are boldfaced.

```
v 1 : ( a 1 , 1) ≻ ( a 1 , 2) ∼ ( a 2 , 1) ≻ ( a 2 , 2) ≻ · · · , v 2 : ( a 1 , 1) ∼ ( a 2 , 1) ≻ ( a 1 , 2) ∼ ( a 2 , 2) ≻ · · · , v 3 : ( a 1 , 1) ≻ ( a 1 , 2) ≻ ( a 2 , 1) ≻ ( a 2 , 2) ≻ · · · .
```

One can verify that among all tuples ( a, T [ a ]) , a ∈ A , tuple ( a 1 , 2) is the most preferred tuple of v 1 and v 3 , while ( a 2 , 1) is the most preferred tuple of v 1 and v 2 . Hence, the network and maximum flow (highlighted with red lines) constructed in Phase 1 are as given in Figure 2; note that this corresponds to the bipartite graph in Appendix A.2.

In Line 6 , we verify that the maximum flow is a perfect flow (i.e., with value | V | ). Hence, we derive and return an assignment according to Definition 2. This is exactly Π 2 from Example 1.

One can verify that the second instance of Example 1 where every agent has the same preference list as v 2 will lead to the sum of the congestion entries to exceed | V | = 3 in the second iteration, certifying that the instance does not have a CP assignment, no matter with or without empty posts.

Figure 2: Flow network for iteration 2 where T = (2 , 1) . For more information, see Appendix B.2.

<!-- image -->

## B.3 Proof of Lemma 4

Lemma 4 ( ⋆ ) . Each ( A ′ , V ′ ) computed in Phase 2 in lines 8-14 is an obstruction.

Proof. Let ( A ′ , V ′ ) be the pair computed in lines 8-14 in some iteration z . Let ( G, c ) be the network with G = ( ˆ A ∪ ˆ V ∪ { s, t } , E ) and f the maximum flow computed in this iteration. We aim to show that ( A ′ , V ′ ) satisfies the properties in Definition 3.

By line 7, f fails to have value | V | , i.e., ∑ ˆ v ∈ ˆ V f (ˆ v, t ) &lt; | V | . Hence, there must be a vertex ˆ v ∗ ∈ ˆ V with f (ˆ v ∗ , t ) = 0 . Let ˆ v ∗ be such a vertex that is added to V ′ in line 9. Then, the first part of property (i) is clear since ˆ v ∗ ∈ V ′ (line 9) and we only add vertices from ˆ V to V ′ ; see line 13.

Property (ii) is also clear due to line 11.

Let us consider property (iii). Clearly, for every vertex ˆ a ∈ A ′ with (ˆ a, ˆ v ∗ ) ∈ E ( G ) we must have that f ( s, ˆ a ) = c ( s, ˆ a ) as otherwise we could increase the flow by one by setting f ( s, ˆ a ) = f ( s, ˆ a ) +1 and f (ˆ a, ˆ v ∗ ) = f (ˆ v ∗ , t ) = 1 . By line 13, every out-neighbor ˆ v of ˆ a with f (ˆ a, ˆ v ) = 1 is added to V ′ . Together with ˆ v ∗ , we obtain that c ( s, ˆ a ) &lt; |{ ˆ v ∈ V ′ | (ˆ a, ˆ v ) ∈ E }| since f ( v ∗ , t ) = 0 , as desired.

Consider an arbitrary vertex ˆ a ∈ A ′ with (ˆ a, ˆ v ∗ ) / ∈ E ( G ) . Suppose, towards a contradiction, that ˆ a does not satisfy property (iii), meaning that c ( s, ˆ a ) ≥ |{ ˆ v ′ ∈ V ′ | (ˆ a, ˆ v ′ ) ∈ E ( G ) }| . We aim to show that there is an 'augmenting' path from ˆ a to ˆ v ∗ , with arcs having flow values alternating between zero and one, which is a witness for the flow to be not maximum.

̸

Let us go through the repeat -loop in lines 10-14. Observe that in each round of this loop, we aim at finding a vertex ˆ a not already in A ′ that has an out-arc to some agent-vertex from V ′ ; V ′ is initialized with V ′ = { ˆ v ∗ } . This implies that we can find a vertex in ˆ v x ∈ V ′ \ { ˆ v ∗ } due to which we add ˆ a in line 11. Further, for each vertex ˆ v ′ in V ′ \ { ˆ v ∗ } , we can also find a vertex ˆ a ′ in a previous round such that f ( ˆ a ′ , ˆ v ′ ) = 1 in line 13. Let ˆ a x be the vertex from A ′ due to which we add ˆ v x , i.e., f (ˆ a x , ˆ v x ) = 1 . Since each vertex in V ′ has only one out-arc with capacity one, due to the conservation constraint of the flow f , we infer that f (ˆ a, ˆ v x ) = 0 ; recall that f (ˆ a x , ˆ v x ) = 1 . Repeating the above reasoning, there must be a vertex ˆ v x -1 from V ′ \ { ˆ v x } due to which we add ˆ a x . Then, either ˆ v x -1 = ˆ v ∗ or ˆ v x -1 = ˆ v ∗ .

In the former case, we infer that (ˆ a, ˆ v x , ˆ a x , ˆ v ∗ ) is an augmenting path since by assumption ˆ a has enough capacity to accommodate all incident agents, including ˆ v x . Thus, flipping the flow values along the path would increase the value of the flow:

<!-- formula-not-decoded -->

In the latter case, since V ′ is finite and no vertex from ˆ V can obtain more than one positive flow, by repeating the above reasoning, we must end up with an arc to ˆ v ∗ with zero flow; recall that ˆ v ∗ ∈ V ′ . Then, we again obtain an augmenting path P = (ˆ a, ˆ v x , ˆ a x , . . . , ˆ a 0 , ˆ v 0 = ˆ v ∗ ) . Analogously, we can increase the total flow by flipping the flow values along this path, a contradiction.

It remains to show property (iv). This is clear since otherwise for each vertex ˆ v ∈ V ′ we could find a vertex ˆ a ∈ A ′ and set f (ˆ a, ˆ v ) = f (ˆ v, t ) = 1 . In particular, the starting vertex ˆ v ∗ would have positive flow going through it, a contradiction.

## B.4 Proof of Lemma 5

Lemma 5 ( ⋆ ) . Assume that I admits a CP assignment Π with no posts being empty. Then, for each iteration z ≥ 1 , each obstruction ( A ′ , V ′ ) found in iteration z , and each post a ∈ A , the following holds. If ˆ a ∈ A ′ , then | Π( a ) | ≥ T z [ a ] + 1 ; otherwise | Π( a ) | ≥ T z [ a ] .

Proof. Let us consider the first iteration and let ( A ′ , V ′ ) be the found obstruction. Since Π does not have empty posts, the statement clearly holds for all posts a ∈ A with ˆ a / ∈ A ′ . Let N = ( G, c ) denote the network and f the maximum flow of N computed in the first phase. Let P = { v ∈ V | ˆ v ∈ V ′ } and Q = { a ∈ A | ˆ a ∈ A ′ } be the set of agents and posts that correspond to the vertices in V ′ and A ′ , respectively.

Suppose, for the sake of contradiction, that there exists a post a ∈ Q with | Π( a ) | ≤ T 1 [ a ] = 1 . By Definition 3(iii), more than c ( s, ˆ a ) = T 1 [ a ] = 1 vertex from V ′ is incident to ˆ a in G . By Construction 2(v), at least two agents from P consider ( a, 1) as one of the most preferred tuples.

Since | Π( a ) | ≤ 1 , at least one agent from P is not assigned to a but considers ( a, 1) as one of the most preferred tuples. Let v 0 ∈ P be such an agent. Then, he must be assigned to some other post a 0 such that ( a 0 , | Π( a 0 ) | ) is one of the most preferred tuples for v 0 as well. This implies that | Π( a 0 ) | = 1 since we are in the first iteration. By Construction 2(v), we infer that (ˆ a 0 , ˆ v 0 ) ∈ E ( G ) , and by line 11, that ˆ a 0 ∈ A ′ .

By Definition 3(iii), more than c ( s, ˆ a 0 ) = T 1 [ a 0 ] = 1 vertex from V ′ is incident to ˆ a 0 in G , and we can find another agent v 1 ∈ P that is not assigned to a 0 but considers ( a 0 , 1) as one of the most preferred tuples. Again, this agent v 1 will be assigned to some post a 1 with ( a 1 , 1) being one of the most preferred tuples of v 1 . By Construction 2(v), we infer that (ˆ a 1 , ˆ v 1 ) ∈ E ( G ) , and by line 11 that ˆ a 1 ∈ A ′ . Repeating the above reasoning, we will be able to find a distinct vertex ˆ a i ∈ A ′ for each vertex ˆ v i ∈ V ′ such that Π( a i ) = { v i } . That is, | A ′ | ≥ | V ′ | , a contradiction to Definition 3(iv) since | A ′ | = ∑ ˆ a ∈ A ′ c ( s, ˆ a ) &lt; | V ′ | in this case.

Now, let us consider other iterations. For each z , let ( A ′ z , V ′ z ) be the obstruction found in iteration z . Note that the table entries never decrease. Hence, if the statement were incorrect, there must be an iteration z ≥ 2 where the statement holds in all iterations z ′ ≤ z -1 but not in iteration z .

Suppose, for the sake of contradiction, that the statement is incorrect and let z be the index of the first such iteration where the statement is incorrect. That is, in all iterations z ′ ≤ z -1 , we have that for all a ′ ∈ A ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

but there exists a post a such that

<!-- formula-not-decoded -->

First, observe that ˆ a ∈ A ′ z since otherwise Π( a ) ≥ T z -1 [ a ] = T z [ a ] by line 15, a contradiction to the assumption.

Next, we claim that T z [ a ] = | Π( a ) | . If ˆ a ∈ A ′ z -1 (i.e., ˆ a was in the obstruction found in iteration z -1 ), then by assumption (1)-(2) and by line 15, we infer | Π( a ) | ≥ T z -1 [ a ] + 1 = T z [ a ] ≥ | Π( a ) | , as desired. If ˆ a / ∈ A ′ z -1 , then again by assumption (1)-(2) and by line 15, we infer | Π( a ) | ≥ T z -1 [ a ] = T z [ a ] ≥ | Π( a ) | , as desired as well.

Recall that we inferred that ˆ a ∈ A ′ z . Let ( G, c ) denote the network constructed in iteration z . By Lemma 4, ( A ′ z , V ′ z ) is an obstruction for ( G, c ) . Let ˆ v ∗ ∈ V ′ be the starting vertex with f (ˆ v ∗ , t ) = 0 . By Definition 3(iii), more than c ( s, ˆ a ) = T z [ a ] vertices from V ′ z exist that have a as an in-neighbor. By Construction 2 (v), more than T z [ a ] = | Π( a ) | agents from V ′ z consider ( a, T z [ a ]) as one of the most-preferred tuples among all ( a ′ , T z [ a ′ ]) . Hence, at least one of such agents is not assigned to a by Π .

̸

Let ˆ v ∈ V ′ z be a vertex whose corresponding agent v considers ( a, | Π( a ) | ) as one of the most preferred tuples but is assigned to some other post a ′ = a . Then, (ˆ a, ˆ v ) ∈ E ( G ) . To prevent v from being envious (recall that no post is empty), we must have that ( a ′ , | Π( a ′ ) | ) ⪰ v ( a, | Π( a ) | ) .

We claim that ˆ a ′ ∈ A ′ z as well. Since ( a, | Π( a ) | ) is one of the most-preferred tuples of v among all ( a ′ , T z [ a ′ ]) , by previous paragraph and by congestion aversion, we infer that T z [ a ′ ] ≥ | Π( a ′ ) | . If T z [ a ′ ] &gt; | Π( a ′ ) | , then by line 15, there must exists an iteration z ′ ∈ [ z -1] where ˆ a ′ ∈ A ′ z ′

and T z ′ [ a ′ ] = | Π( a ′ ) | , a contradiction to (1). Hence, T z [ a ′ ] = | Π( a ′ ) | , implying that ( a ′ , T z [ a ′ ]) is also one of the most preferred tuples of v among all ( a ′′ , T z [ a ′′ ]) . By Construction 2(v), we have (ˆ a ′ , ˆ v ′ ) ∈ E ( G ) , and by line 11, we have ˆ a ′ ∈ A ′ z , as desired.

By Definition 3(iii), we infer that more than c ( a ′ ) = T z [ a ′ ] = | Π( a ′ ) | vertices from V ′ z have in-arcs from ˆ a ′ . By Construction 2(v), more than c ( a ′ ) = T z [ a ′ ] = | Π( a ′ ) | agents consider ( a ′ , | Π( a ′ ) | ) as one of the most preferred tuples among all ( p, T [ p ]) , p ∈ A .

̸

Analogously, we can again find another vertex ˆ v ′ ∈ V ′ z such that v ′ considers ( a ′ , | Π( a ′ ) | ) as one of the most preferred tuples among all ( p, T [ p ]) , p ∈ A , but is assigned to some other post a ′′ = a ′ with ˆ a ′′ ∈ A ′ z and T z [ a ′′ ] = | Π( a ′′ ) | . By repeating this argument, we infer that every vertex ˆ α ∈ A ′ z has T z [ α ] = | Π( α ) | . By Definition 3(iv), we have that | V ′ z | &gt; ∑ ˆ α ∈ A ′ z c ( s, ˆ α ) = ∑ ˆ α ∈ A ′ z T z [ α ] =

∑ ˆ α ∈ A ′ z | Π( α ) | . So there must be a vertex ˆ µ ∈ V ′ z such that µ is assigned to a post b with ˆ b / ∈ A ′ z . By line 11 and Construction 2(v), let ˆ α ∈ A ′ z with (ˆ α, ˆ µ ) ∈ E ( G ) such that ( α, T z [ α ]) is a most preferred tuple of µ among all tuples ( p, T z [ p ]) , p ∈ A .

By our previous argument, we have that T z [ α ] = | Π( α ) | . By CP, we have that ( b, | Π( b ) | ) ⪰ v ( α, | Π( α ) | ) . Since ( α, T z [ α ]) is a most preferred tuple of µ among all tuples ( p, T z [ p ]) , p ∈ A , we further infer that T z [ b ] ≥ | Π( b ) | .

Since b / ∈ A ′ z , meaning by line 11 that ( ˆ b, ˆ µ ) / ∈ E ( G ) , by Construction 2(v), we further infer that | Π( b ) | &lt; T z [ b ] . By line 15, there must exist an iteration z ′ ∈ [ z -1] with T z ′ [ b ] = Π( b ) and T z ′ [ b ] was incremented. This is a contradiction to (1) however.

## B.5 Proof of Lemma 6

Lemma 6 ( ⋆ ) . If Π is an assignment returned in line 7, then Π is CP and has no empty post.

Proof. Let z be the integration and f be the perfect flow based on which Π is computed in line 7. By the definition of perfectness (see Definition 2), the value of f equals the number | V | of agents. This means that ∑ a ∈ A | Π( a ) | = | V | . By the capacity constraints, we obtain that | V | = ∑ a ∈ A | Π( a ) | ≤ ∑ a ∈ A T z [ a ] ≤ | V | , the last inequality holds due to the while-loop-condition in line 4. Hence, for each post a ∈ A we must have that | Π( a ) | = T z [ a ] since | Π( a ) | ≤ T z [ a ] holds by the capacity constraints in Construction 2(iii).

̸

This implies that Π( a ) = ∅ since T z [ a ] ≥ 1 . Hence, to show that Π is CP, it suffices to show that for each agent v that is assigned to a post a and for each post a ′ with a ′ = a it holds that ( a, | Π( a ) | ) ⪰ v ( a ′ , | Π( a ′ ) | ) . Towards a contradiction, suppose that ( a ′ , | Π( a ′ ) | ) ≻ v ( a, | Π( a ) | ) . By the reasoning above, it follows that ( a ′ , T z [ a ′ ]) ≻ v ( a, T z [ a ]) , a contradiction to Construction 2(v).

## B.6 Continuation of the proof of Theorem 1

Theorem 1 ( ⋆ ) . Algorithm 1 correctly decides whether an instance has a CP assignment in O ( m 2 · ( n + m ) 2 ) time, where m and n denote the number of posts and agents, respectively.

It remains to analyze the running time. The main body of the algorithm is a for-loop (line 1) and has at most m iterations. In each iteration k , the algorithm constructs a new instance I according to Construction 1. Note that I has O ( n + m ) agents and O ( m ) posts, and it can be constructed in O (( n + m ) 2 ) time since each agent has O ( n + m ) tuples in his preference list. Then, we continue with the big while -loop in lines 4-16. If we can show the while -loop run in O ( m · ( n + m ) 2 ) time, we obtain our desired running time of O ( m 2 · ( n + m ) 2 ) .

So, it remains to analyze the while -loop. In line 3, initializing the table T needs O ( m ) time. The while-loop (lines 4-15) runs at most n times since no table entries are ever decreased and in each iteration at least one table entry is increased by one. For each iteration, we first construct a network N = ( G, c ) based on ( I, T ) ; see Construction 2. The directed graph G has O ( n + m ) vertices and O ( m · n ) arcs, and each capacity value is in O ( n ) . Hence, constructing the network needs O ( m · n ) time.

̸

Afterwards, there are three phases. The first phase (lines 5-7) finds a maximum flow for N and checks whether its value is | V | . Computing a maximum flow can be done in O ( m · n ) time and comparing two values needs constant time. Hence, the first phase needs O ( m · n ) time.

The second phase (lines 8-14) finds an obstruction ( A ′ , V ′ ) by first finding a vertex ˆ v ∗ with f (ˆ v ∗ , t ) = 0 . This can be done in O (1) time if we store such information when we compare the value of the flow with | V | in the first phase. Hence, the initialization of V ′ and A ′ needs O (1) time. Then, the algorithm goes to the repeat-loop in lines 10-14. To analyze the running time of this loop, we observe that there are O ( m · n ) arcs between A ′ and V ′ and each arc only needs to be checked at most once during the whole loop (line 11). Adding new vertices to V ′ can be done in O ( m · n ) time as well since for each newly added alternative ˆ a there are at most n vertices ˆ v from ˆ V with positive flow from ˆ a to ˆ v . Hence, the repeat-loop needs O ( m · n ) time.

It is straightforward that the last phase (lines 15-15) runs in O ( m ) time. Summarizing, we obtain that the desired O ( m · n 2 ) time for the while -loop.

## C Additional Material for Section 4

## C.1 Correctness of the Construction in the Proof of Theorem 2

Theorem 2 ( ⋆ ) . EF+TG is NP-complete; hardness holds even if there are no ties.

Proof of the correctness of the construction. Correctness. It remains to show the correctness, i.e., I has an exact cover if and only if I ′ admits an EF and TG assignment.

For the 'only if' part, let J ⊆ [ m ] denote an exact cover for I . Then, we claim that the following assignment Π is EF+TG:

- -For each j ∈ J , let Π( a ) = { v | u ∈ C } .
- -For each j ∈ [ m ] \ J , let Π( a ) = ∅
- -Let Π( b 1 ) = { p j | j ∈ [2 m ] } and Π( b 2 ) = { q j | j ∈ [2 m ] } .

```
j i i j j .
```

Since each set-post contains either zero or three agents, no dummy agent envies any element-agent. The dummy agents also do not envy each other due to their preferences. Similarly, no two elementagents envy each other and no element-agent envies any dummy agent since he does not like b 1 or b 2 more.

For 'if' part, let Π be an EF and TG assignment for the constructed instance. We aim at showing that the set-posts that are assigned element-agents constitute an exact cover. To this end, let J = { j | ∃ v i with v i ∈ Π( a j ) } . We first show two claims.

Claim C.1.1. For each set-post a j it holds that | Π( a j ) | ∈ { 0 , 3 } .

̸

Proof. Since there are 2 m dummy agents { p 1 , p 2 , . . . , p 2 m } , but there are only m set-posts, at least one dummy agent, say p z , is not assigned to a set-post alone. Hence, for every set-post a j , it holds that | Π( a j ) | = 1 since otherwise p z would envy the agent that is assigned to a j . Since the maximum congestion for every set-post is 3 , we further infer that | Π( a j ) | ∈ { 0 , 2 , 3 } holds for every set-post a j . In particular, this implies that no dummy agent p z with 1 ≤ z ≤ 2 m is assigned to a set-post alone.

Towards a contradiction, suppose that there exists a set-post a j with | Π( a j ) | / ∈ { 0 , 3 } . This implies that | Π( a j ) | = 2 . Then, every dummy agent p z with 1 ≤ z ≤ 2 m is to be assigned a set-post since otherwise he would envy the two agents that are assigned to a j . Since there are exactly 2 m dummy agents p 1 , . . . , p 2 m , this means that every set-post a x , x ∈ [ m ] , must have | Π( x ) | = 2 . Then, no other agent can be assigned to the set-post. However, all element-agents will envy all p i 's, a contradiction. This concludes the proof. (end of the proof of Claim C.1.1 ⋄ )

By Claim C.1.1, we know that each set-post is assigned either zero or three agents. Next, we show that every element-agent v i is assigned to an acceptable set-post.

Claim C.1.2. For each element-agent v i it holds that Π( v i ) ∈ { a j | j ∈ [ m ] and u i ∈ C j } .

Proof. Suppose this is not true, and by TG let v i denote an element-agent that is assigned to b 2 ; note that v i does not find b 1 acceptable. Since there are 2 m dummy agents q z each with congestion 2 m for b 2 , at least one of them is not assigned to b 2 . This agent envies v i , a contradiction. (end of the proof of Claim C.1.2 ⋄ )

Claim C.1.2 implies that J is a set cover, while Claim C.1.1 implies that | J | ≤ n . Altogether we conclude that J is an exact cover.

## C.2 Proof of Theorem 3

Theorem 3 ( ⋆ ) . EF+TG is FPT with respect to the number n of agents and the number m of posts, respectively.

Proof. We first consider the parameter n . Let I = ( A,V, ( ⪰ v ) v ∈ V ) be an instance of CONGESTED ASSIGNMENT. Due to TG, each agent is assigned to one of his first n tuples. Hence, for each agent v i ∈ V , we guess (by brute-force searching) which of his first n tuples that v i is 'assigned' to, i.e., ( a, d ) . After assigning all the agents, we check in linear time whether this results in a valid assignment, i.e., if we v i 'assign' ( a, d ) , then there must be exactly d agents that are guessed to be 'assigned' to ( a, d ) . This check can be done in O ( n 2 + m ) time. We abandon the current guess if it does not give a valid assignment; otherwise we proceed to check EF in O ( n 2 ) time.

Since there are n agents, each with n choices, the whole procedure can be done in O ( n n · ( n 2 + m )) time, which is an FPT time with respect to n .

Now, we consider the parameter m . Let I = ( A,V, ( ⪰ v ) v ∈ V ) be an instance of CONGESTED ASSIGNMENT. We guess (by brute-force searching) the set of empty posts A ′ ⊆ A in the sought solution. Then, we modify the preference list ⪰ v of each agent v as follows: First, truncate ⪰ v by removing all tuples ( a, d ) with ranks are higher than n ; then, remove all tuples ( a ′ , d ′ ) with a ′ in A ′ . Denote the new preference list as ⪰ ′ v . Let I ′ = ( V, A \ A ′ , ( ⪰ ′ v ) v ∈ V ) denote the modified instance.

Since for assignments with all posts being non-empty, CP and EF are equivalent, we infer that I admits an EF and TG assignment where all posts in A ′ are empty and the rest is non-empty if and only if I ′ admits CP assignment where all posts are non-empty. The latter problem can be checked in polynomial time via lines 3-16 in Algorithm 1. The running time depends on the running time of the while -loop, which is m ( n + m ) 2 time. See the proof in Appendix B.6 for more details. Since there are 2 m subsets of empty posts to check, the overall running time is 2 m · m · ( n + m ) 2 , which is an FPT time with respect to m .

Clique. The following graph problem is W[1]-complete with respect to the clique size h [20]. We will use it to show W[1]-hardness for finding a a maximally competitive assignment.

CLIQUE

Input: An undirected graph G =( U, E ) , an integer h ≥ 0 .

Question: Does G admit a clique of size h , i.e., a sizeh subset U ′ ⊆ U which induces a complete subgraph?

## C.3 Proof of Theorem 4

Theorem 4 ( ⋆ ) . MAXCP+TG is W[1]-hard and in XP with respect to the number t of unsatisfied agents. The W[1]-hardness holds even if there are no ties.

Proof. We first show the W[1]-hardness by providing a parameterized reduction from the CLIQUE problem.

Let I = ( G = ( U, E ) , h ) denote an instance of CLIQUE with U = { u 1 , . . . , u | U | } and E = { e 1 , . . . , e | E | } . We create a MAXCP+TG instance I ′ = ( A,V, ( ⪰ v ) v ∈ V , t ) as follows. Let t = h + h ( h -1) . We will show that the agents corresponding to the vertices and edges of a sizeh clique are the only unsatisfied agents.

- -For each vertex u i ∈ U , create a vertex-post a i , a vertex-agent w i , and h -1 copies of w i , denoted as ˜ w z i with z ∈ [ h -1] .

- -For each edge e ℓ ∈ E with e ℓ = { u i , u j } , create an edge-post b ℓ and three edge-agents e ∗ ℓ , e i ℓ , and e j ℓ .
- -Create five auxiliary posts a 0 , ˜ a 0 , b 0 , y, c 0 . Post y shall accommodate all L dummy agents, while c 0 is a 'blocker' making sure that agents are assigned to the desired posts.
- -Create L dummy agents x 1 , · · · , x L with L = | U | ( h -2) + ( | U | -h -1) + ( h ( h -1) -1) + ( | E | -( h 2 ) -1) + ( t +1) .

<!-- formula-not-decoded -->

Preferences. We state the preferences of the agents, restricted to the first n tuples. Here, ⟨ α, s, t ⟩ = ( α, s ) ≻ ( α, s +1) ≻ · · · ≻ ( α, t ) depicts the preference list on tuples for post α and congestions ranging between s and t . Note that we also briefly explain the main purpose of these preferences in italicized text.

- The dummy agent x i with i ∈ [ L ] has the following preference list:

<!-- formula-not-decoded -->

The dummy agents shall ensure some minimum number of agents assigned to each post (except b ℓ and y ): At least h -1 , | U | -h , h ( h -1) , and | E | -( h 2 ) agents are to be assigned to a i ( i ∈ [ | U | ]) , a 0 , ˜ a 0 , and b 0 , respectively. The reason is that since at least t +1 dummy agents are to be assigned to y or c 0 , they would envy the agents assigned to a post if its congestion is less than or equal to the maximum congestion of x i to that post, which is not possible for a yes instance. Indeed, the dummy agents can only be assigned to y .

- The vertex-agent w i with i ∈ [ | U | ] has the following preference list:

<!-- formula-not-decoded -->

We will show that exactly | U | -h vertex-agents w i are assigned to a 0 . Consequently, there remain h vertex-agents v i that are assigned to a i . They shall correspond to the clique-vertices if G admit a sizeh clique.

- -For each i ∈ [ | U | ] , all copy-agents ˜ w z i with z ∈ [ h -1] of the vertex-agent w i have the same preference list:

<!-- formula-not-decoded -->

The copy-agents shall ensure that all ˜ w z i , z ∈ [ h -1] , are jointly assigned to either ˜ a 0 or a i . If they are assigned to a i , then no other agent (including w i ) can be assigned to a i . This corresponds to the case that the vertex u i is not in the clique.

- The edge-agents e ∗ ℓ , e i ℓ and e j ℓ with e ℓ = { u i , u j } have the following preference lists:

<!-- formula-not-decoded -->

Note that e ∗ ℓ can only be assigned to b ℓ or b 0 , and e i ℓ (resp. e j ℓ ) only to a i (resp. a j ) or b ℓ . If e ∗ ℓ is assigned to b ℓ and does not envy other agents, then no other agent can be assigned to b ℓ , as otherwise at least | E | -( h 2 ) + 1 agents must be assigned to b 0 , which is impossible due to top-guarantees. Therefore, if e ∗ ℓ is assigned to b ℓ , then e i ℓ and e j ℓ have to be assigned to a i and a j , respectively. We will show that e ∗ ℓ cannot be unsatisfied, and having e i ℓ and e j ℓ assigned to a i and a j , respectively, corresponds to having the edge e ℓ in a sizeh clique.

The maximum congestions of the agents are depicted in Table 1.

Correctness. Clearly, the construction can be done in polynomial time and no agent has ties in his preference list. It remains to show the correctness, i.e., I has a clique of size h if and only if I ′ admits a TG assignment with t = h + h ( h -1) agents being unsatisfied.

Table 1: Maximum congestions of the agents constructed for Theorem 4. For an illustration, we assume that e ℓ = { u 1 , u | U | } .

|                       | a 1   | . . .   | a &#124; U &#124;   | a 0                    | ˜ a 0          |   b ℓ | b 0                        | y   | c 0                                  |
|-----------------------|-------|---------|---------------------|------------------------|----------------|-------|----------------------------|-----|--------------------------------------|
| w 1                   | h     | 0       | 0                   | &#124; U &#124;- h     | 0              |     0 | 0                          | 0   | n -&#124; U &#124;                   |
| . . .                 | 0     | h       | 0                   | &#124; U &#124;- h     | 0              |     0 | 0                          | 0   | n -&#124; U &#124;                   |
| w &#124; U &#124;     | 0     | 0       | h                   | &#124; U &#124;- h     | 0              |     0 | 0                          | 0   | n -&#124; U &#124;                   |
| ˜ w z 1               | h - 1 | 0       | 0                   | 0                      | h ( h - 1)     |     0 | 0                          | 0   | n - h 2 +1                           |
| . . .                 | 0     | h - 1   | 0                   | 0                      | h ( h - 1)     |     0 | 0                          | 0   | n - h 2 +1                           |
| ˜ w z &#124; U &#124; | 0     | 0       | h - 1               | 0                      | h ( h - 1)     |     0 | 0                          | 0   | n - h 2 +1                           |
| e ∗ ℓ                 | 0     | 0       | 0                   | 0                      | 0              |     2 | &#124; E &#124;- ( h 2 )   | 0   | n - ( &#124; E &#124;- ( h 2 ) ) - 2 |
| e i ℓ                 | h     | 0       | 0                   | 0                      | 0              |     2 | 0                          | 0   | n - h - 2                            |
| e j ℓ                 | 0     | 0       | h                   | 0                      | 0              |     2 | 0                          | 0   | n - h - 2                            |
| x z                   | h - 2 | h - 2   | h - 2               | &#124; U &#124;- h - 1 | h ( h - 1) - 1 |     0 | &#124; E &#124;- ( h 2 ) - | L   | n +( t +1) - 2 L                     |

The 'only if' part. Let C ⊆ U denote an h -clique for I . Let E C ⊆ E denote the edge set associated with C , i.e., E C = { e ℓ = { u i , u j } | u i , u j ∈ C} . Then, we claim that the following assignment Π is a TG assignment with t unsatisfied agents.

- -For each u i ∈ C , assign w i to a i , and assign ˜ w z i with z ∈ [ h -1] to ˜ a 0 .
- -For each e ℓ = { u i , u j } ∈ E C , assign e i ℓ to a i , e j ℓ to a j , and e ∗ ℓ to b ℓ .
- -For each u i / ∈ C , assign ˜ w z i with z ∈ [ h -1] to a i , and assign w i to a 0 .
- -For each e ℓ = { u i , u j } / ∈ E C , assign e i ℓ and e j ℓ to b ℓ , and e ∗ ℓ to b 0 .
- Assign x z to y with z ∈ [ L ] .

Clearly, Π is TG with the following congestion vector.

Observation 3. Π is TG and satisfies the following.

```
(i) | Π( a 0 ) | = | U | -h , | Π(˜ a 0 ) | = h ( h -1) , | Π( b 0 ) | = | E | -( h 2 ) , and | Π( y ) | = L . (ii) For each u i ∈ C , it holds that Π( a i ) = { w i , ˜ w z i | z ∈ [ h -1] } . For each u i ∈ U \ C , it holds that | Π( a i ) | = { ˜ w z i | z ∈ [ h -1] } . (iii) For each e ℓ ∈ E , if e ℓ ∈ E C , then | Π( b ℓ ) | = 2 ; otherwise | Π( b ℓ ) | = 1 .
```

Let V ′ = { w i | u i ∈ C}∪{ e i ℓ , e j ℓ | e ℓ ∈ E C with e ℓ = { u i , u j }} . Note that | V ′ | = t . Weaim to show that all agents except those from V ′ are satisfied. By the above observation, it is straightforward that every dummy agent x z is satisfied, every agent that does not correspond to the clique vertices is satisfied, and the copies ˜ w z i of all vertex-agents are also satisfied. It remains to consider the edge-agents that are not in V ′ . Let e ℓ ∈ E with e ℓ = { u i , u j } . Clearly, if e ℓ / ∈ E C , then the two edge-agents e i ℓ and e j ℓ are satisfied since they are assigned to their most preferred post. Agent e ∗ ℓ with e ℓ / ∈ E C is also satisfied since he is assigned to b 0 with congestion | E |-( h 2 ) which is better than ( b ℓ , 2) . If e ℓ ∈ E C , then agent e ∗ ℓ is satisfied since he is assigned to b ℓ alone which is better than ( b 0 , | E | -( h 2 ) ) . Hence, only the agents in V ′ are unsatisfied. Since | V ′ | = t , this concludes the proof for the 'only if' direction.

The 'if' part. Let Π be a TG assignment with at most t unsatisfied agents. We aim to show that the following vertex subset C is a sizeh clique: C = { u i | | Π( a i ) | ≥ h } . Before we show this, we observe the following regarding the congestions and assignments of the posts.

Claim C.3.1. (1) | Π( a 0 ) | = | U | -h , | Π(˜ a 0 ) | = h ( h -1) , and | Π( b 0 ) | = | E | -( h )

- 2 . (2) For each u i ∈ U , it holds that | Π( a i ) | ∈ { h -1 , h } . (3) For each e ℓ ∈ E , it holds that | Π( b ℓ ) | ≤ 2 . (4) Π( a 0 ) ⊆ { w i | u i ∈ U } , Π(˜ a 0 ) ⊆ { ˜ w z i | i ∈ [ | U | ] , z ∈ [ h -1] } , and Π( b 0 ) ⊆ { e ∗ ℓ | e ℓ ∈ E } . (5) All edge-agents E ∗ = { e ∗ ℓ | e ℓ ∈ E } are satisfied.

Proof. We show the first two statements together by considering the dummy agents.

Since Π is TG and the maximum congestion of dummy x z for a 0 , ˜ a 0 , b 0 , and a i with i ∈ [ | U | ] are | U |-h -1 , h ( h -1) -1 , | E |-( h 2 ) -1 , and h -2 , respectively, we infer by simple calculation that there

are more than t dummy agents who are assigned to y or c 0 . Since Π does not have more than t unsatisfied agents, this further implies that there is at least one satisfied dummy agent x z who is assigned to y or c 0 . By his preferences, every tuple that he prefers to ( y, t +1) must have congestion that exceeds his maximum durable congestion. This implies that | Π( a 0 ) | ≥ | U | -h , | Π(˜ a 0 ) | ≥ h ( h -1) , and | Π( b 0 ) | ≥ | E |-( h 2 ) , for | Π( a i ) | ≥ h -1 . Since no agent allows more than the aforementioned congestions (except for a i ), we further infer that | Π( a 0 ) | = | U |-h , | Π(˜ a 0 ) | = h ( h -1) , and | Π( b 0 ) | = | E |-( h 2 ) . For a i , since the maximum congestion of any agent for a i is h , we infer that h -1 ≤ | Π( a i ) | ≤ h .

Statement (3) is straightforward by observing the maximum congestion of any agent towards b ℓ is two.

The first part of statement (4) follows from the fact that the only agents that have ( a 0 , | U | -h ) in their top n choices are the vertex-agents. Similarly, we can show that the other parts of statement (4) are also correct.

To show statement (5), let us analyze which agents are unsatisfied. To this end, define W ′ = { w i ∈ W | w i / ∈ Π( a 0 ) } . By statements (1) and (4), we infer that | W ′ | = h .

Further, every agent w i in W ′ is unsatisfied since by statement (2) that | Π( a i ) | ≥ h -1 , any agent not assigned to a 0 will envy those that are assigned to a 0 . This implies that at most 2 ( h 2 ) agents other than W ′ can be unsatisfied.

By statement (2), partition the posts { a i | u i ∈ U } into A 1 and A 2 with A 1 = { a i | u i ∈ U ∧ | Π( a i ) | = h -1 } and A 2 = { a i | u i ∈ U ∧ | Π( a i ) | = h } . Note that by the top-guarantees, every post a i from A 2 can only be assigned vertex-agents w i or edge-agent e i ℓ for some edge e ℓ ∈ E with u i ∈ e ℓ . However, this implies that every agent assigned to post a i ∈ A 2 is unsatisfied since w i prefers ( a 0 , | U | -h ) to ( a i , h -1) and every edge-agent e i ℓ (with u i ∈ e ℓ ) prefers ( b ℓ , 2) to ( a i , h ) ; recall by statements (1) and (3) that | Π( a 0 ) | = | U | -h and | Π( b ℓ ) | ≤ 2 . This further implies that | A 2 | ≤ h since t = h + h ( h -1) = h 2 can be unsatisfied.

By statement (1), we have that | Π(˜ a 0 ) | = h ( h -1) . Since every vertex-agent has h -1 copies, there are at least h vertices each of which has a copy-agent assigned to ˜ a 0 . Since every copy-agent e z ℓ corresponding to vertex u i prefers ( a i , h -1) to (˜ a 0 , h ( h -1)) , it follows that at least h - | A 2 | copy-agents will be unsatisfied, namely those whose corresponding vertex-post has congestion h -1 .

Since at most h vertex-agents and at most ( | U | -h )( h -1) copy-agents can be assigned to any vertex-post a i , the number of edge-agents e i ℓ that have to be assigned to some vertex-post a i is at least

<!-- formula-not-decoded -->

Observe that each edge-agent e i ℓ that is assigned to some vertex-post a i is unsatisfied. This implies that at least h ( h -1) -( h -| A 2 | ) edge-agents are unsatisfied. Together with the h -| A 2 | unsatisfied copy-agents, no more other agent can be unsatisfied. In other words, every edge-agent e ∗ ℓ must be satisfied, as desired. (end of the proof of Claim C.3.1 ⋄ )

̸

Now, we are ready to show that C is a clique of size h . We first show that C has size h . Define W ′ = { w i | Π( w i ) = a 0 } . By the preferences of the vertex-agents and by Claim C.3.1(1), | W ′ | = h and every vertex-agent in W ′ is unsatisfied. By Claim C.3.1(1) and by the maximum congestions of the agents towards b 0 , we infer that Π( b 0 ) consists of exactly ( | E |-{ h 2 } ) edge-agents e ∗ ℓ , ℓ ∈ [ | E | ] . By Claim C.3.1(5), every remaining edge-agent e ∗ ℓ that is not assigned to b 0 must be assigned to the corresponding edge-post b ℓ alone. This implies that the remaining two edge-agents e i ℓ and e j ℓ with e ℓ = { u i , u j } are not assigned to b ℓ and hence unsatisfied; they both envy e ∗ ℓ . Define E ′ = { e i ℓ , e j ℓ | Π( e ∗ ℓ ) = b ℓ } . Then, | E ′ | = h ( h -1) and it yields h ( h -1) unsatisfied edge-agents by Claim C.3.1(1). Together with the h unsatisfied vertex-agents in W ′ , we infer that every copy-agent ˜ w z i is satisfied. In particular, it means that for each copy-agent ˜ w z i that is assigned to ˜ a 0 it must hold that | Π( a i ) | = h . Recall that there are at least h vertices u i each of which has a copy-agent assigned to ˜ a 0 . This further implies that there are at least h vertex-posts that each have congestion h , that is |C| = h .

It remains to show that C is a clique. Let E ′′ = { e ℓ | e i ℓ ∈ Π( a i ) for some u i ∈ C} be the set consisting of all edges whose corresponding edge-agents are assigned to some vertex-post. Clearly, | E ′′ | ≥ ( h 2 ) since C = h ; note that the equality holds only if C induces a clique. Towards a contradiction, suppose that C contains two vertices u i and u j that are not adjacent with each other. This implies that | E ′′ | &gt; ( h 2 ) . Let us consider each edge e ℓ ∈ E ′′ and let e ℓ = { u i , u j } with i &lt; j .

By definition, at least one of the two edge-agents e i ℓ and e j ℓ is assigned to a i and he is unsatisfied. We claim that both edge-agents are unsatisfied. Without loss of generality, assume that Π( e i ℓ ) = a i and is unsatisfied. If e j ℓ is assigned to vertex-post a j or c 0 , then he is unsatisfied as well. Otherwise, e j ℓ is assigned to post b ℓ alone, making e ∗ ℓ unsatisfied which is not possible according to Claim C.3.1(5). Hence, both e i ℓ and e j ℓ are satisfied. Since there can be only h ( h -1) unsatisfied edge-agents, we conclude that | E ′′ | = ( h 2 ) , as desired.

Now, we turn to the XP result. The ideas is to guess the unsatisfied agents and the posts that they are assigned to, and replace them with dummies and run Algorithm 1 for the reduced instance. More precisely, we guess who are the unsatisfied agents in O ( n t ) time; denoting the set of unsatisfied agents as V ∗ = { v ∗ 1 , . . . , v ∗ t } . For each V ∗ , we further guess which posts they are assigned to in O ( m t ) time, denoted as A ∗ = { a ∗ 1 , . . . , a ∗ t } with a ∗ z being the post that v ∗ z will assigned to and z ∈ [ t ] .

Then we create t dummy agents P = { p z | z ∈ [ t ] } and set their preference list as p z : ( a ∗ z , 1) ≻ · · · ≻ ( a ∗ z , n ) ≻ · · · . We replace the agents V ∗ with the dummies and use Algorithm 1 to solve the resulting instance. If Algorithm 1 returns no on the current guess, we proceed with the next guess; otherwise, let Π be CP assignment returned by Algorithm 1. It is straightforward that replacing each dummy P z with v ∗ z in the assignment yields a TG assignment with at most t unsatisfied agents.

The overall running time is O ( n t m t ) .

## C.4 Proof of Theorem 5

Theorem 5 ( ⋆ ) . MAXCP+TG is FPT with respect to n , and in XP with respect to m , where n and m denote the number of agents and the number of posts, respectively.

Proof sketch. Parameter n : We first guess a subset V ′ of unsatisfied agents. Afterwards, similarly to Theorem 3, we guess for each satisfied agent V \ V ′ one of his first n tuples and check whether the | V \ V ′ | guesses yield a valid assignment Π V ′ and store the number of unsatisfied agents. Finally, we select one valid Π V ′ with fewest unsatisfied agents. The whole approach can be done in FPT time wrt. n .

Parameter m : For the XP-algorithm, we guess the congestion vector ⃗ s with ⃗ s [ j ] ∈ { 0 , . . . , n } and Σ ⃗ s = n and use the algorithm behind Lemma 2 to determine the minimum number of unsatisfied agents. The overall running time is n m · ( m + n ) O (1) , which is XP wrt. m .

## C.5 Proof of Theorem 6

Theorem 6 ( ⋆ ) . Deciding whether an instance of CONGESTED ASSIGNMENT has an assignment with at most t unsatisfied agents is W[1]-hard with respect to t .

Proof. We reduce from the W[1]-complete problem CLIQUE; the definition can be found ahead of Appendix C.3.

Let I = ( G = ( U, E ) , h ) denote an instance of CLIQUE with U = { u 1 , . . . , u ˆ n } and E = { e 1 , . . . , e ˆ m } being the vertex set and edge set, respectively. Without loss of generality, we assume that ˆ n &gt; 3 h + ( h 2 ) and ˆ m&gt; 2 h +2 ( h 2 ) as the problem remains W[1]-hard in this case.

The idea is to construct an instance I ′ = ( A , V, ( ⪰ v ) v ∈ V ) of CONGESTED ASSIGNMENT such that the unsatisfied agents correspond to the vertices and edges of a sizeh clique. We set the number of unsatisfied agents to t = 2 h + ( h 2 ) , and let L and R be two very large numbers such that L &gt; 2 t and R &gt; ( L +2) · (ˆ n + ˆ m ) + h . For the sake of brevity, let N = ( L +2) · (ˆ n + ˆ m ) + 2 R , and we will create exactly N agents.

## Posts and agents.

- -For each vertex u i ∈ U , create one vertex-post a i and L +2 vertex-agents w i , p i , p z i , z ∈ [ L ] .
- -Create 2 R dummy agents x z , y z , z ∈ [ R ] .
- -For each edge e ℓ ∈ E , create one edge-post b ℓ and L +2 edge-agents e ℓ , f z ℓ , z ∈ [ L +1] .
- -Create 3 auxiliary posts a 0 , b 0 , and c 0 .

Let A = { a i | i ∈ [ˆ n ] } , B = { b ℓ | ℓ ∈ [ ˆ m ] } , W = { w i | i ∈ [ˆ n ] } , P = { p i | i ∈ [ˆ n ] } , P i = { p z i | z ∈ [ L ] , i ∈ [ˆ n ] } , F ℓ = { f z ℓ | z ∈ [ L +1] } , X = { x z | z ∈ [ R ] } , and Y = { y z | z ∈ [ R ] } . Then, we set A = A ∪ B ∪ { a 0 , b 0 , c 0 } , and V = W ∪ P ∪ ⋃ i ∈ [ˆ n ] P i ∪ E ∪ ⋃ ℓ ∈ [ ˆ m ] F ℓ ∪ X ∪ Y . In total, we have created ˆ n + ˆ m +3 posts and N agents.

Preferences. For two numbers s, t ∈ [ N ] and post α ∈ A , let ⟨ α, s, t ⟩ = ( α, s ) ≻ ( α, s +1) ≻ · · · ≻ ( α, t ) depict the preference list on tuples for post α and congestions ranging between s and t . The notation ' ∗ ∗ ∗ ' refers to an arbitrary but congestion-averse preferences of the tuples that are not explicitly mentioned.

- (i) For each vertex u i ∈ U , the vertex-agents w i ∈ W , p i ∈ P , and p z i ∈ P i , z ∈ [ L ] , have the following preference lists:

<!-- formula-not-decoded -->

- (ii) For each edge e ℓ ∈ E , the edge-agents e ℓ , f z ℓ , z ∈ [ L +1] , have the following preference lists, where we assume e ℓ = { u i , u j } :

<!-- formula-not-decoded -->

- (iii) The preference lists of the dummy agent x z ∈ X and y z ∈ Y are as follows:

<!-- formula-not-decoded -->

This completes the construction of the instance I ′ , which can clearly be done in polynomial time. Note that it is also a parameterized reduction since the parameter t = 2 h + ( h 2 ) is a polynomial function in h . It remains to show the correctness, i.e., I has a sizeh clique if and only if I ′ has an assignment with at most t unsatisfied agents.

For the 'only if' part, let U ′ be a clique of size h . We construct the following assignment Π and show that it has at most t unsatisfied agents.

- (1) For each vertex u i ∈ U , assign all agents from P i ∪ { p i } to a i . Additionally assign w i to a i if u i ∈ U ′ ; otherwise assign w i to a 0 .
- (2) For each edge e ℓ ∈ E , assign all agents from F ℓ to b ℓ . Additionally assign e ℓ to b ℓ if e ℓ ⊆ U ′ , i.e., both its endpoints are in U ′ ; otherwise assign e ℓ to b 0 .
- (3) Assign all agents from X ∪ Y to c 0 .

To see who is unsatisfied, let W ′ = { w i ∈ W | w i ∈ Π( a i ) } , P ′ = { p i ∈ P | p i ∈ Π( a i ) } , and E ′ = { e ℓ ∈ E | e ℓ ∈ Π( b ℓ ) } . We claim that all agents but those from W ′ ∪ P ′ ∪ E ′ are satisfied.

In the following, we say that a post α is the most preferred post for agent q if every tuple that is contains a post other than α is less preferred than ( α, | Π( α ) | ) . Further, a tuple ( α, d ) is a most preferred feasbile tuple for agent q if every tuple ( α ′ , d ′ ) that is preferred to ( α, d ) has congestion | Π( α ′ ) | &gt; d ′ .

Clearly, every agent in X ∪ Y is satisfied since ( c 0 , 2 R ) is his most preferred feasible tuple. Every agent in ( ⋃ i ∈ [ˆ n ] P i ) ∪ ( W \ W ′ ) ∪ ( P \ P ′ ) ∪ ( E \ E ′ ) is satisfied since he is assigned to his most preferred post. Every agent f z ℓ ∈ F ℓ is also satisfied since either he is assigned to his most preferred

post (if e ℓ is not a 'clique' edge) or | Π( a i ) | = | Π( a j ) | = L +2 so ( b ℓ , L +2) remains his most preferred feasible tuple. This concludes the proof for the 'only if' direction.

For the 'if' direction, let Π denote an assignment with at most t unsatisfied agents. Before we construct a clique, let us analyze the preferences and Π would look like.

Claim C.5.1. Π satisfies the following.

- (1) | Π( a 0 ) | = ˆ n -h and | Π( b 0 ) | = ˆ m -( h 2 ) .
- (2) Every agent from W that is not assigned to a 0 is unsatisfied and every agent from E that is not assigned to b 0 is unsatisfied.
- (3) For each a i ∈ A we have that | Π( a i ) | ≥ L +1 and for each b ℓ ∈ B we have that | Π( b ℓ ) | ≥ L +1 .
- (4) It holds that | Π( c 0 ) | ≤ 2 R .
- (5) For each a i ∈ A we have that | Π( a i ) | ≤ L +2 and for each b ℓ ∈ B we have that | Π( b ℓ ) | ≤ L +2 .

Proof. Statement (1) : The lower bounds are straightforward since all agents from W ∪ X prefer ( a 0 , ˆ n -h -1) to any other tuple that does not contain a 0 : If | Π( a 0 ) | &lt; ˆ n -h would hold, then more than | W ∪ X | -(ˆ n -h ) &gt; R agents will be unsatisfied, which is not possible since R &gt; ( L +2) · (ˆ n + ˆ m ) &gt; t . Similar reasoning shows that | Π( b 0 ) | ≥ ˆ m -( h 2 ) by considering the preferences of E ∪ Y . Now, we show the upper bounds. Suppose, for the sake of contradiction, that | Π( a 0 ) | &gt; ˆ n -h . Then, since no agent considers ( a 0 , ˆ n -h +1) more valuable than any other tuple that does not contain a 0 , all agents assigned to a 0 are unsatisfied. Since we can assume that ˆ n &gt; 3 h + ( h 2 ) , it follows that more than t agents will be unsatisfied, a contradiction. Similarly, since we have just shown that | Π( a 0 ) | ≤ ˆ n -h , from the remaining possible tuples, no agent considers ( b 0 , ˆ m -( h 2 ) ) more valuable than any tuple that does not contain b 0 (except ( a 0 , ˆ n -h + z ) , z ≥ 1 , which is excluded). Consequently, by the fact that ˆ m&gt; 2 h +2 ( h 2 ) , we infer that | Π( b 0 ) | ≤ ˆ m -( h 2 ) as otherwise all agents assigned to b 0 are unsatisfied the number of which exceeds t .

Statement (2) : This statement follows directly from the previous statement and from the preferences of the agents in W ∪ E .

Statement (3): We show the lower bound by iterating through all i ∈ [ˆ n ] . By Statement (1), every agent in X ∪ P 1 ∪ { p 1 } prefers ( a 1 , L ) to every other tuple that does not contain a 1 (excluding a 0 ). Hence, | Π( a 1 ) | ≥ L +1 as otherwise more than R -L &gt; t agents from X are not assigned to a 1 and will be unsatisfied. By applying the above reasoning for the next i ≥ 2 , we infer that | Π( a i ) | ≥ L +1 holds for all i ∈ [ˆ n ] . Similarly, we infer that | Π( b ℓ ) | ≥ L +1 holds for every ℓ ∈ [ ˆ m ] .

Statement (4) : Suppose this is not true, i.e., | Π( c 0 ) | ≥ 2 R +1 . Then, by Statements (1)-(2) and by the bound t = 2 h + ( h 2 ) , at most h agents from V \ ( W ∪ E ) can be unsatisfied. By construction, every agent in X that is assigned to c 0 will be unsatisfied since he prefers ( a 1 , L +1) to ( c 0 , 2 R +1) . Hence, at most h agents from X can be assigned to c 0 . This means that at least 2 R +1 -h agents from W ∪ P ∪ ⋃ i ∈ [ˆ n ] P i ∪ E ∪ ⋃ ℓ ∈ [ ˆ m ] F ℓ ∪ Y need to be assigned to c 0 . This is not possible however

<!-- formula-not-decoded -->

Statement (5) : Let i ∈ [ˆ n ] . The statement follows directly from the fact that every agent prefers ( c 0 , 2 R ) to ( a i , L + 3) and | Π( c 0 ) | ≤ 2 R (see Claim C.5.1(1)): | Π( a i ) | &gt; L + 2 would hold, then all agents assigned to a i are unsatisfied, the number of which exceed t since L &gt; 2 t . (end of the proof of Claim C.5.1 ⋄ )

The next statement is about the structure of the agents assigned to A ∪ B .

Claim C.5.2. Let A ′ = { a i ∈ A : | Π( a i ) | = L +2 } and B ′ = { b ℓ ∈ B : | Π( b ℓ ) | = L +2 } . Then, Π satisfies the following.

- (1) | A ′ | + | B ′ | ≥ h + ( h 2 ) .
- (2) For each post a i ∈ A ′ , at least two agents in Π( a i ) are unsatisfied; for each post b ℓ ∈ B ′ , at least one agent in Π( b ℓ ) is unsatisfied.
- (3) | A ′ | ≤ h and | B ′ | ≥ ( h 2 ) .
- (4) For each post b ℓ ∈ B ′ it holds that | Π( a i ) | = | Π( a j ) | = L +2 where e ℓ = { u i , u j } .

Proof. Statement (1) : This can be shown by simple calculation. By Claim C.5.1(1) and (4), at least ( L +2) · (ˆ n + ˆ m ) -(ˆ n -h ) -( ˆ m -( h 2 ) ) = ( L +1) · (ˆ n + ˆ m ) + h + ( h 2 ) agents are assigned to the posts of A ∪ B . By Claim C.5.1(3) and (5), each post in A ∪ B is assigned either L +1 or L +2 agents. That is, at least h + ( h 2 ) of the post in A ∪ B are each assigned L +2 agents, confirming that | A ′ | + | B ′ | ≥ h + ( h 2 ) .

Statement (2) : For each post a i ∈ A ′ , since | Π( a i ) | = L +2 and | P i | = L , at least two agents in Π( a i ) are not from P i , i.e., | Π( a i ) \ P i | ≥ 2 We claim that the agents in Π( a i ) \ P i are unsatisfied by considering the preferences of all agents except P i : Every agent from W ∪ P prefers ( a 0 , ˆ n -h ) to ( a i , L + 2) . Every agent from E prefers ( b 0 , ˆ m -( h 2 ) ) to ( a i , L + 2) . Every agent from ( ⋃ i ′ ∈ [ˆ n ] \{ i } P i ) ∪ ( ⋃ ℓ ∈ [ ˆ m ] F ℓ ) ∪ X ∪ Y prefers ( c 0 , 2 R ) to ( a i , L + 2) . Since | Π( a 0 ) | = ˆ n -h , | Π( b 0 ) | = ˆ m -( h 2 ) , and | Π( c 0 ) | ≤ 2 R (see Claim C.5.1s(1) and (4)), we infer that every agent in Π( a i ) \ P i is unsatisfied.

Similarly, for each post b ℓ ∈ B ′ , since | Π( b ℓ ) | = L +2 and | F ℓ | = L +1 , at least one agent in Π( b ℓ ) is not from F ℓ . We claim that the agents in Π( b ℓ ) \ F ℓ are unsatisfied by considering the preferences of all agents except P i : Every agent from W ∪ P ∪ ( ⋃ i ∈ [ˆ n ] P i ) ∪ ( ⋃ ℓ ′ ∈ [ ˆ m ] \{ ℓ } F ℓ ) ∪ X ∪ Y prefers ( c 0 , 2 R ) to ( b ℓ , L + 2) . Every agent from E prefers ( b 0 , ˆ m -( h 2 ) ) to ( b ℓ , L + 2) . Again, since | Π( b 0 ) | = ˆ m -( h 2 ) and | Π( c 0 ) | ≤ 2 R (see Claim C.5.1(1) and (4)), we infer that every agent in Π( b ℓ ) \ F ℓ is unsatisfied.

Statement (3) : Statement (2) implies that at least 2 | A ′ | + | B ′ | agents are unsatisfied. By the upper bound that t ≤ 2 h + ( h 2 ) and by Statement (1), we infer that | A ′ | ≤ h , and hence | B ′ | ≥ ( h 2 ) .

̸

Statement (4) : Suppose, towards a contradiction, that | Π( a i ) | = L +2 . Then, by Claim C.5.1(3) and (5), it follows that | Π( a i ) | = L +1 . We claim that every agent assigned to b ℓ is unsatisfied. Let us consider an arbitrary agent q ∈ Π( b ℓ ) . Clearly, if q ∈ W ∪ P ∪ ⋃ ℓ ∈ [ ˆ m ] P ℓ ∪ X ∪ Y ∪ E ∪ F \ ( { e ℓ }∪ F ℓ ) , then he is unsatisfied since he prefers ( c 0 , 2 R ) to ( b ℓ , L +2) . If q = e ℓ , then he is unsatisfied since he prefers ( b 0 , ˆ m -( h 2 ) ) to ( b ℓ , L +2) , while if q ∈ F ℓ , then he is unsatisfied since he prefers ( a i , L +1) to ( b ℓ , L +2) as well. This concludes the proof that every agent in Π( b ℓ ) is unsatisfied, implying that more than L &gt; 2 t agents is unsatisfied, a contradiction.

By an analogous reasoning, we can show that | Π( a j ) | = L +2 . (end of the proof of Claim C.5.2 ⋄ )

Now, we are ready to show the existence of a sizeh clique. By Claim C.5.2(3), B ′ corresponds to at least ( h 2 ) edges. Hence, there are at least h vertices incident to any edge corresponding to B ′ . For each vertex a i that is 'incident' to any edge-post in B ′ , we know by Claim C.5.2(4) that its corresponding vertex-post a i must be assigned L +2 posts. By Claim C.5.2(3), there are at most h such vertex-posts. Hence, there are exactly h vertex-posts that are each assigned L +2 agents, and this is possible if and only if they form a sizeh clique.