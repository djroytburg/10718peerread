## On the Sample Complexity of Differentially Private Policy Optimization

Yi He

Wayne State University yihe@wayne.edu

## Xingyu Zhou

Wayne State University xingyu.zhou@wayne.edu

## Abstract

Policy optimization (PO) is a cornerstone of modern reinforcement learning (RL), with diverse applications spanning robotics, healthcare, and large language model training. The increasing deployment of PO in sensitive domains, however, raises significant privacy concerns. In this paper, we initiate a theoretical study of differentially private policy optimization, focusing explicitly on its sample complexity. We first formalize an appropriate definition of differential privacy (DP) tailored to PO, addressing the inherent challenges arising from on-policy learning dynamics and the subtlety involved in defining the unit of privacy. We then systematically analyze the sample complexity of widely-used PO algorithms, including policy gradient (PG), natural policy gradient (NPG) and more, under DP constraints and various settings, via a unified framework. Our theoretical results demonstrate that privacy costs can often manifest as lower-order terms in the sample complexity, while also highlighting subtle yet important observations in private PO settings. These offer valuable practical insights for privacy-preserving PO algorithms.

## 1 Introduction

Policy Optimization (PO) such as REINFORCE [1, 2], proximal policy optimization (PPO) [3] and group relative policy optimization (GRPO) [4] has gained increasing interest recently across various applications. Due to its popularity, there is a rich literature that provides various theoretical understandings of different PO methods (e.g., iteration or sample complexity [5-8]).

As PO becomes increasingly prevalent in real-world applications, privacy concerns are emerging as a critical challenge. For instance, in personalized medical care, patient interactions-where the state represents medical history, the action corresponds to prescribed medication, and the reward reflects treatment effectiveness-constitute sensitive data that must be protected. Similarly, in RLbased training of large language models (LLMs), user prompts may contain private information that requires protection. In fact, recent empirical work has shown that standard GRPO indeed has privacy leakage issues [9]. Hence, addressing these privacy concerns is essential for ensuring the responsible deployment of PO methods in sensitive domains.

Contribution. In this paper, we initiate the theoretical study of differentially private policy optimization, focusing on the central question: What's the sample complexity cost induced by differential privacy in PO? To this end, we first carefully define a suitable notion of differential privacy (DP) [10] for PO, highlighting its distinctions from the standard DP definitions used in supervised learning. These differences stem from the unique learning dynamics and the notion of the privacy unit in PO. Then, we propose a meta algorithm for private PO, which enables us to study private policy gradient with REINFORCE ( DP-PG ), private natural policy gradient (NPG) [11] ( DP-NPG ), and private version of REBEL recently proposed in [12] ( DP-REBEL ) in a unified perspective. Moreover, for DP-NPG and DP-REBEL , we further reduce PO to a sequence of private regression problems, thus allowing us to

leverage various well-established results in private estimation and supervised learning. Throughout this process, we not only highlight the difference between DP-PG and DP-NPG , but also uncover some subtleties when applying private regression results within the current analytical framework of PO. The key takeaway from our theoretical results is that the privacy cost can often appear as lower-order terms in the overall sample complexity. Meanwhile, it is worth noting that structural properties of the underlying problem can further improve both statistical and computational efficiency.

Related work. We mainly discuss the most relevant work here and relegate a detailed discussion to Appendix A. The very recent work [13] studies private PG, but mainly from an empirical perspective without sample complexity bounds. From the online regret perspective, optimistic PPO has been studied in the tabular case [14] and linear case [15], respectively. In contrast, we aim to consider general function classes from the optimization perspective. We also note that NPG with a softmax or log-linear policy is equivalent to PPO. The authors of [16] study private policy evaluation, which aims to evaluate a given policy rather than finding the best policy in policy optimization. From an application perspective, [17] applies private PPO for LLM alignment via reinforcement learning from human feedback (RLHF).

## 2 Preliminaries

Policy optimization (PO). In this work, instead of considering a general Markov decision process (MDP), we focus on the simpler bandit formulation, which allows us to easily demonstrate the key ideas. We note that generalizing it to MDP is standard, as done in the literature [6, 18]. This bandit formulation already captures many interesting real-world applications, such as personalized medical care [19] and alignment/reasoning training in large language models (LLMs) [20]. In particular, given an initial state x ∈ X (e.g., a medical status or a prompt in LLMs) sampled from a distribution ρ , an action y ∈ Y (e.g., a medical prescription or a response in LLMs) is generated according to a policy π and a reward r ( x, y ) ∈ [ -R max , R max ] is observed. In policy optimization, we parameterize the policy π by π θ with θ ∈ Θ = R d (e.g., a neural network), and the goal is to leverage interactions (e.g., sample trajectories) to find an optimal policy that maximizes the following objective:

<!-- formula-not-decoded -->

Vanilla policy gradient (PG). One simple and direct approach to solving the above policy optimization problem is via vanilla policy gradient, i.e., θ t +1 = θ t + η ∇ J ( θ t ) , where η &gt; 0 is some learning rate, ∇ J ( θ t ) is the gradient at step t , and θ 1 is some initial value. The gradient can be written as follows by the classic policy gradient theorem

<!-- formula-not-decoded -->

where A π θ ( x, y ) := r ( x, y ) -E y ′ ∼ π θ ( y ′ | x ) r ( x, y ′ ) is the advantage function.

Natural policy gradient (NPG). Another approach to solving the PO problem is natural policy gradient (NPG) [11], which uses the Fisher information matrix as the preconditioner to account for the geometry. Specifically, the NPG update is given by θ t +1 = θ t + ηF † ρ ( θ t ) ∇ J ( θ t ) , where F ρ ( θ ) := E x ∼ ρ,y ∼ π θ ( ·| x ) [ ∇ θ log π θ ( y | x ) log π θ ( y | x ) ⊤ ] is the expected Fisher information matrix (with superscript † being the Moore-Penrose pseudoinver) and ∇ J ( θ t ) is the same as before. An equivalent way to write the above update is the following [5]

<!-- formula-not-decoded -->

which essentially reduces PO to a sequence of regression problems.

Regression to Relative Reward Based RL (REBEL). Recently, a simple and scalable PO algorithm called REBEL is proposed in [12], which also reduces PO to a sequence of regression problems, but now over relative reward difference , motivated from the DPO-style reparameterization trick in [21]. In the expected form, the update under REBEL is given by

<!-- formula-not-decoded -->

where the expectation here is over x ∼ ρ, y ∼ µ ( ·| x ) , y ′ ∼ π θ t ( ·| x ) , and µ can be either on-policy distribution π θ t or any offline reference policy.

Sample complexity. All the aforementioned ideal policy updates (e.g., full gradient) involve expectation, which is often difficult to compute in practice due to both statistical (e.g., without knowing ρ ) and computational (e.g., averaging over all possible trajectories) issues. Thus, one needs to replace the expectation with a sample-based estimate by sampling a dataset of trajectories at each iteration from an underlying distribution. The sample complexity typically refers to the total number of sampled trajectories for finding an α -optimal policy (i.e., J ( π ∗ ) -J ( ̂ π ) ≤ α ).

In this paper, our ultimate goal is to formally introduce differential privacy (DP) into the problem of policy optimization and derive the sample complexity bounds under privacy constraints. To this end, we need to carefully define both privacy and samples in the private case, as discussed next.

## 3 Differential Privacy in Policy Optimization

In this section, we formally introduce DP to PO, highlighting some subtleties compared with standard DP in supervised learning problems. We first recall the standard DP definition with a fixed dataset.

Definition 1 (Dwork et al. [10]) . A randomized mechanism M satisfies ( ε, δ ) -DP if for any adjacent datasets D,D ′ differing by one record, and ∀ S ⊆ Range ( M ) :

<!-- formula-not-decoded -->

This standard DP notion can be directly used in supervised learning problems with D being a set of i.i.d samples { ( x i , y i ) } N i =1 from an unknown distribution and M ( D ) being the final model. This has been utilized in private empirical risk minimization (ERM) [22, 23] as well as private stochastic optimization (both convex and non-convex), e.g., Bassily et al. [24]. For example, the optimal excess population risk for stochastic convex optimization is O δ (1 / √ N + √ d/ ( Nε )) for ( ε, δ ) -DP, where d is the dimension of the parameter space.

One may attempt to adopt the above notion directly to PO with the dataset D being { ( x i , y i ) } N i =1 and M ( D ) being the final policy. However, this does not make too much sense because (i) there is no such a fixed dataset in PO as the actions are often sampled in the on-policy fashion, i.e., using the most recent policy; (ii) the neighboring relation of differing in one sample ( x i , y i ) (i.e., privacy unit) actually does not hold as changing one sample will lead to difference in all future samples due to different policies onward. Thus, we need a new definition that can address the above two issues. Before proceeding, we consider two motivating examples to illustrate the subtlety.

Example 1 (SFT vs. RL fine-tuning in LLMs) . Consider a reasoning task in LLMs. With supervised fine-tuning (SFT), we are given a fixed dataset of pairs { ( x i , y i ) } N i =1 where x i is the prompt/question and y i is the correct answer. Standard DP is natural here, which ensures that changing one sample ( x i , y i ) will not change the final policy too much. On the other hand, if one uses RL (e.g., PPO) to do the fine-tuning, then the given dataset consists of only prompts, as the answers are generated on the fly. So, a proper privacy unit here is to protect each prompt in the sense that changing one prompt will not change the policy too much.

Example 2 (Supervised learning vs. RL for healthcare) . In this case, to train a healthcare system, one can use a supervised learning approach by collecting a dataset of { ( x i , y i ) } N i =1 where x i is the medical status and y i is the recommended medicine. One can also adopt an RL approach (even in an online manner) where the dataset consists of a (stream) set of users/patients, each with a medical status x i sampled from a distribution ρ , while the recommendation y i can only be determined on the fly. The privacy protection is that changing one user/patient will not change the final policy too much.

To handle both scenarios, we borrow the idea from private online bandit and RL literature [25, 14], which essentially considers a set of 'users' as the dataset. For instance, the dataset could be N unique patients interacting with the learning agent, and each user has an initial state (e.g., medical status), which is distributed according to ρ . We can fix the 'users' in advance (or arrive online) and the privacy unit is now for each patient, hence resolving both issues above. Meanwhile, the set of 'users' can also represent N (static) prompts in the fine-tuning of LLMs, with each 'user' contributing one prompt. Note that although we use 'users' to align with personalization application, this is still an item-level DP, as each 'user' appears only once (as a patient or prompt). The learning agent can interact with each 'user' to observe ( x, y ) and r ( x, y ) dynamically , i.e., on-the-fly. With the above notion of dataset, the privacy protection in PO is that changing one 'user' in the dataset will not change the final policy too much, leading to the following definition.

Definition 2 (DP in PO) . Consider any policy optimization algorithm M interacting with a set D of N 'users' and M ( D ) being the final output policy. We say M is ( ε, δ ) -DP if for any adjacent datasets D,D ′ differing by one 'user', and ∀ S ⊆ Range ( M ) :

<!-- formula-not-decoded -->

Remark 1 . We emphasize that the above DP notion is defined for the problem PO rather than for a specific algorithm, analogous to the standard DP in statistical learning (e.g., supervised learning). In this paper, we aim to design some private variants of PO methods ( DP-PG , DP-NPG , DP-REBEL ) and analyze their sample complexity under the above privacy constraint.

## 4 A Meta Algorithm for Private PO

In this section, we present a meta algorithm for private PO, which builds upon a unified view of PG, NPG, and REBEL. We believe that this meta viewpoint is also interesting in the non-private case.

Our meta algorithm is given by Algorithm 1, which is essentially a batched one-pass algorithm. In particular, at each iteration t , the learner collects m fresh samples by sampling from a distribution over x and y . To be more specific, one can view each sample ( x i , y i , y ′ i ) as generated via interaction with a new fresh 'user', which provides the context/prompt x i . Then, leveraging the dataset D t and a specific PrivUpdate oracle, the learner finds the next policy iteratively.

## Algorithm 1 A Meta Algorithm

- 1: Input: reward r , learning rate η , batch size m , and policy class π θ , PrivUpdate oracle, base policy µ 2: Initialize θ 1 = 0 3: for t =1 , . . . , T do 4: Collect a fresh dataset ¯ D t = { ( x i , y i , y ′ i ) } i m =1 of size m using the π θ t and µ : x i ∼ ρ, y i ∼ µ ( ·| x i ) , y ′ i ∼ π θ t ( ·| x i ) 5: For all i ∈ [ m ] , let ̂ A t ( x i , y i ) := r ( x i , y i ) -r ( x i , y ′ i ) be the estimate of A π θ t ( x i , y i ) 6: Call a PrivUpdate oracle on D t := { ( x i , y i , y ′ i , ̂ A t ( x i , y i )) } i m =1 to find next policy θ t +1 7: end for

Under this one-pass algorithm design, we naturally have the following privacy guarantee, connecting standard DP (Definition 1) with DP in PO (Definition 2).

Proposition 1. Suppose PrivUpdate satisfies ( ε, δ ) -DP under Definition 1, then Algorithm 1 satisfies ( ε, δ ) -DP in terms of Definition 2.

This simply follows from our one-pass algorithm and (adaptive) parallel composition of DP, by noting that changing one 'user' would only change one record in D t of a single t ∈ [ T ] .

Remark 2 . Our meta algorithm can also be used in the online setting where a stream of N 'users' arrive sequentially. By the so-called billboard lemma [26], our meta algorithm also satisfies the commonly used joint differential privacy (JDP) in the literature of private online RL/bandits [25, 27, 14, 15, 28]. Roughly speaking, JDP guarantees that changing one 'user' (say u ) will not change all the actions prescribed to all other 'users' except u , as well as the final policy.

For sample complexity, due to the batched one-pass algorithm over N unique 'users', the total number of sampled trajectories is simply N = m · T , where each trajectory ( x i , y i , y ′ i ) is from a fresh user. To put it in another way, for a fixed N , the key here is to balance between batch size m and number of iterations T so as to balance between the per-iteration accuracy and the total number of updates. This balance, in turn, depends on the specific choice of PrivUpdate oracle, which will be instantiated in the next sections for DP-PG , DP-NPG , DP-REBEL , respectively.

## 5 Differentially Private Policy Gradient

In this section, we propose DP-PG by building upon our meta algorithm and analyze its sample complexity bounds under different settings.

Our proposed DP-PG is Algorithm 1 with the instantiation of PrivUpdate as in Algorithm 2 below. In particular, it first computes an unbiased REINFORCE-style estimate (i.e., ̂ ∇ m J ( θ ) ) of the full gradient ∇ J ( θ t ) as in (1), using the m trajectories in D t . Then, a Gaussian noise is added with σ 2 depending on the privacy parameters of ε and δ . Finally, θ t +1 is obtained by updating the current policy θ t along the direction of ˜ g t , scaled by a properly chosen learning rate η .

## Algorithm 2 PrivUpdate Instantiation for DP-PG

- 1: Input: dataset D t = { ( x i , y i , ̂ A t ( x i , y i )) } i m =1 , policy θ t , learning rate η , noise scale σ
- 2: Output: θ t +1
- 3: Compute gradient:

<!-- formula-not-decoded -->

- 4: Add noise: g t := ̂ ∇ m J ( θ ) + N (0 , σ 2 I )
- ˜ 5: Output policy: θ t +1 = θ t + η · ˜ g t

By the standard Gaussian mechanism [29] and Proposition 1, we have the following privacy guarantee.

Theorem 1 (Privacy guarantee) . Assume for any x ∈ X and θ ∈ Θ , there exists a constant G such that ∥∇ θ log π θ ( y | x ) ∥ ⩽ G . Then, setting σ 2 = 16 log(1 . 25 /δ ) R 2 max G 2 m 2 ε 2 in Algorithm 2 ensures that DP-PG satisfies ( ε, δ ) -DP as in Definition 2.

The boundedness assumption of G is satisfied by softmax policy as well as Gaussian policy [6]. In fact, they satisfy an even stronger condition in Assumption 1, to be discussed shortly.

Next, we aim to establish the sample complexity results of our DP-PG with Algorithm 2 for both first-order stationary point (FOSP) and global optimum convergence, respectively.

## 5.1 First-order Stationary Point Convergence

We start with the sample complexity bound for FOSP convergence. This result is not only of its own importance, but will also be useful for our later results on the global optimum convergence. We will consider the following general class of policies, which is widely studied in previous non-private work and also includes commonly used policies such as softmax and Gaussian policy [6].

Assumption 1 (Lipschitz Smoothness (LS)) . There exist constants G,F &gt; 0 such that for every state x ∈ X , the gradient and Hessian of log π θ ( · | x ) of any θ ∈ Θ satisfy

<!-- formula-not-decoded -->

Remark 3 . For simplicity, as in previous work, we will often view G and F as constants Θ(1) , hence omitted in the sample complexity bound. Moreover, we omit log(1 /δ ) term by writing O δ ( · ) .

Theorem 2 (FOSP convergence) . Under the same setting of Theorem 1 and Assumption 1, there exists a proper parameter choices of m and η , such that DP-PG achieves

<!-- formula-not-decoded -->

where θ U is uniformly sampled from { θ 1 , . . . , θ T } .

Remark 4 . We can see that the first term in (4) matches the previous non-private term, i.e., for an accuracy of α , the sample complexity is O (1 /α 4 ) [6]; Second, the privacy cost is a lower order additive term (for constant ε and d ), i.e., the additional sample complexity due to privacy is O δ ( √ d α 3 ε ) .

## 5.2 Global Optimum Convergence

We now turn our focus to the global optimum convergence in the sense of average regret, i.e., J ∗ -1 T ∑ T t =1 E [ J ( θ t )] . Following the non-private work [6], we will also consider two different scenarios and aim to establish the corresponding sample complexities in the private case.

In the first scenario, in addition to Assumption 1, we further assume the following two conditions on the policy class, both of which are commonly used in the non-private case. The first condition is the so-called Fisher-non-degenerate policy , formally defined below.

Assumption 2 (Fisher-non-degenerate, adapted from Assumption 2.1 of Ding et al. [30]) . For all θ ∈ R d , there exists γ &gt; 0 s.t. the Fisher information matrix F ρ ( θ ) induced by policy π θ and initial state distribution ρ satisfies

<!-- formula-not-decoded -->

This assumption is commonly used in the literature on non-private PG methods [6, 30, 5, 31]. As shown in Sec B.2 in Ding et al. [30], this assumption is satisfied by the Gaussian policy and even certain neural policies.

The next condition is the so-called compatible function approximation , which is also a common assumption in the PG literature to handle function approximation error in the non-tabular case.

Assumption 3 (Compatible, adapted from Assumption 4.6 in Ding et al. [30]) . For all θ ∈ R d , there exists α bias &gt; 0 such that the transferred compatible function approximation error satisfies

<!-- formula-not-decoded -->

where π θ ∗ is an optimal policy and u ∗ = F ρ ( θ ) † ∇ J ( θ ) .

The 'compatible' here means that we are approximating the advantage function A π θ ( s, a ) using the ∇ θ log π θ ( a | s ) as the feature vector; The 'transfer error' here means that we are shifting to the expectation over an optimal policy (rather than the current policy). The error α bias is zero for a softmax tabular policy and small when π θ is a rich neural policy. [30-32].

With the above two additional assumptions along with the LS assumption in Assumption 1, we have the following important result, which implies that the objective J ( θ ) satisfies the so-called relaxed weak gradient domination .

Lemma 1 (Lemma 4.7 in Ding et al. [30]) . If the policy class π θ satisfies Assumptions 1, 2 and 3, then we have

<!-- formula-not-decoded -->

This lemma essentially allows us to easily translate a guarantee in terms of FOSP to a certain global optimum convergence. This leads to our next main result with its proof given in Appendix E.2.

Theorem 3. Consider the same setting of Theorem 2 and further let Assumptions 2 and 3 hold. Then, for any α &gt; 0 , DP-PG enjoys the following average regret guarantee

<!-- formula-not-decoded -->

when the sample size satisfies N ⩾ O δ ( 1 α 4 γ 4 + √ d α 3 γ 3 ε ) .

Remark 5 . In the above bound, we explicitly include the parameter γ to clearly illustrate its impact. The first term O ( 1 α 4 γ 4 ) matches the non-private one in Yuan et al. [6] while the second term is the privacy cost. As we can see, for both terms, there exists an additional 1 /γ factor compared to the sample complexity of FOSP. Thus, for very small but still positive γ , our bound could be large.

Our second scenario is about the specific policy class of softmax in the tabular setting, which allows us to get rid of the parameter γ . Due to space limit, we relegate these results to Appendix B.

## 6 Differentially Private NPG and REBEL

In this section, we turn to DP-NPG and DP-REBEL , private variants of NPG and RRBEL, and analyze their sample complexities. In particular, we will consider a general private regression oracle as the PrivUpdate in Algorithm 1 and then give concrete examples under different specific regression oracles. Given the similarity, we will mainly focus on DP-NPG in the main paper and relegate the detailed discussion on DP-REBEL to Appendix D.

## 6.1 A Master Algorithm and Guarantee

Our proposed DP-NPG is Algorithm 1 with its PrivUpdate being instantiated in Algorithm 3 below, which relies on a general private regression oracle to return an approximate minimizer of an estimation problem under the square loss. The square loss in (6) is almost the same as before, as in (2), except that we now take the expectation over a general base policy µ rather than the specific on-policy π θ t . This update is often called approximate NPG in the literature [5]. As will be shown later, the performance of the algorithm will depend on the choice of µ in terms of its coverage.

## Algorithm 3 PrivUpdate Instantiation for DP-NPG

- 1: Input: D t = { ( x i , y i , ̂ A t ( x i , y i )) } i m =1 , current policy θ t , base policy µ , learning rate η , PrivLS oracle
- 2: Output: θ t +1
- 3: Call the PrivLS oracle on D t := { ( x i , y i , ̂ A t ( x i , y i )) } to find an approximate minimizer w t of

<!-- formula-not-decoded -->

- 4: Output policy θ t +1 = θ t + ηw t

We now aim to establish a generic performance guarantee of DP-NPG . To start with, we assume that the approximate minimizer w t returned by PrivLS at each iteration satisfies the following guarantee.

Assumption 4 (Private estimation error) . For each t ∈ [ T ] , the PrivLS oracle satisfies ( ε, δ ) -DP while ensuring that with probability at least 1 -ζ ,

<!-- formula-not-decoded -->

for some error function err 2 t ( m,ε,δ, ζ ) over batch size m , privacy parameters ε , δ , and probability ζ .

In addition, we assume standard regularity assumptions commonly used even in the non-private case.

Assumption 5 ( β -smoothness and boundedness) . log π θ ( y | x ) is a β -smooth function of θ for all x, y , i.e.,

<!-- formula-not-decoded -->

Moreover, there exists a constant W &gt; 0 such that for all t ∈ [ T ] , the weight vectors w t generated by the update rule satisfy ∥ w t ∥ 2 ≤ W .

Our main result is given by the following theorem.

Theorem 4 (Master theorem) . Let Assumptions 4 and 5 hold. Then, DP-NPG satisfies ( ε, δ ) -DP as in Definition 2. Moreover, if π 1 := π θ 1 is a uniform distribution at each state and η = √ 2 log |Y| TβW 2 , with probability at least 1 -ζ , for any comparator policy π ∗ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We use the most intuitive coverage definition for C µ → π ∗ , i.e., the density ratio, for illustrating the key idea. One can easily extend it to other types of coverage, e.g., relative condition number for the linear case [5] and transfer coefficient for general function classes [33].

## 6.2 Applications

With the above master theorem in hand, we now only need to determine the estimation error under different types of PrivLS . To start with, we consider a general PrivLS under general function classes for both reward and policy. Specifically, it runs approximate least squares with the exponential

mechanism [34] for privacy protection, as detailed in Algorithm 4 below. To determine its estimation error, we will first present a new result, which could be of independent interest.

## Algorithm 4 PrivLS Instantiation for DP-NPG via Exponential Mechanism

- 1: Input: D t = { ( x i , y i , ̂ A t ( x i , y i )) } i m =1 , privacy budget ε , current policy θ t , reward range R max
- 2: Output: w t
- 3: Sample w t ∈ W with the following distribution:

<!-- formula-not-decoded -->

where L ( w ) := ∑ i ∈ [ m ] [ w ⊤ ∇ log π θ t ( y i | x i ) -̂ A t ( x i , y i )] 2

Lemma 2 (Private LS with exponential mechanism) . Let R &gt; 0 , ζ ∈ (0 , 1) , we consider a general sequential estimation setting with an instance space U and target space Z . Let H : U → [ -R,R ] be a class of real-valued functions. Let D = { ( u i , z i ) } i m =1 be a dataset of m points where u i ∼ ρ i = ρ i ( u 1: i -1 , z 1: i -1 ) , and z i = h ∗ ( u i ) + η i , where η i is zero-mean noise and h ∗ satisfies approximate realizability, i.e.,

<!-- formula-not-decoded -->

Suppose max i | z i | ⩽ R and max u | h ∗ ( u ) | ⩽ R . Then, sampling ̂ h via the following distribution from exponential mechanism

<!-- formula-not-decoded -->

with L ( h ) := ∑ i ∈ [ m ] [ h ( u i ) -z i ] 2 , yields that

<!-- formula-not-decoded -->

This lemma can be viewed as the private variant of Lemma 15 in [33]. To leverage this lemma for Algorithm 4, we observe the following mappings for each iteration t : H maps to W , u i = ( π θ t , x i , y i ) , h ( u i ) = w ⊤ ∇ log π θ t ( y i | x i ) , z i = ̂ A t ( x i , y i ) with E [ z i ] = A π θ t ( x i , y i ) , which can be rewritten as an unknown function w ∗ = h ∗ over ( π θ t , x i , y i ) . Finally, in our case, ρ i is non-sequential and fixed during each update, i.e., x i ∼ ρ, y i ∼ µ ( ·| x i ) and π θ t is fixed at t . Thus, the approximation error condition in (7) translates to the following one:

<!-- formula-not-decoded -->

Based on these discussions, we have the following guarantee of DP-NPG with Algorithm 4.

Corollary 1 (General function class) . Consider DP-NPG with PrivLS as in Algorithm 4. Then, DP-NPG satisfies ( ε, 0) -DP. Suppose for each t ∈ [ T ] , there exists an α approx such that (8) holds. Then, under the same assumptions in Theorem 4, we have

<!-- formula-not-decoded -->

This implies that, for a given suboptimality gap of O ( α + √ C µ → π ∗ α approx ) , the sample complexity bound is N = T · m = ˜ O ( ( 1 α 4 + 1 α 4 ε ) · log |W| · βW 2 ) .

Remark 6 . Several remarks are in order. First, due to the exponential mechanism, we achieve pure DP (i.e., δ = 0 ) rather than approximate DP; Second, our results hold for general reward and policy function classes. We state the result for a finite W for ease of presentation. It can be easily extended for an infinite class using a standard covering argument. For instance, if W = R d , then log( |W| ) can

be converted to ˜ O ( d ) . Meanwhile, we remark that in general, Algorithm 4 is not computationally efficient. Thus, the above result is mainly from the statistical perspective. Note that we also explore several efficient oracles for specific scenarios later. Finally, we highlight that the approximation error in (8) is different from the transfer error in (5), which directly takes expectation over the target optimal policy. In contrast, when we use α approx , we have to account for the transition from µ to π ∗ explicitly via C µ → π ∗ . This is the case even in the non-private case (cf. Corollary 21 in [5]).

Log-linear policy class with realizability. We now turn to computationally efficient PrivLS oracles by considering a concrete case where the policy is log-linear and reward r is realizable. More specifically, the policy π θ is given by some θ ∈ R d and feature vector ϕ x,y ∈ R d with

<!-- formula-not-decoded -->

This leads to the fact that ∇ θ log π θ ( y | x ) = ϕ x,y -E y ′ ∼ π θ ( ·| x ) [ ϕ x,y ′ ] and smoothness parameter of B 2 . Further, we assume the reward r ( x, y ) = ⟨ w ∗ , ϕ x,y ⟩ is a linear function with respect to ϕ x,y , i.e., compatible realizability. In this case, Assumption 4 reduces to estimation error (or in-distribution generalization error) of linear regression:

<!-- formula-not-decoded -->

where ¯ ϕ t x,y := ϕ x,y -E y ′ ∼ π θ t ( ·| x ) [ ϕ x,y ′ ] , which depends on the current policy.

The above particular form allows us to leverage recent advances in private linear regression, both in the low-dimension and high-dimension cases, respectively.

Corollary 2 (Log-linear policy in low-dimension) . Consider DP-NPG with the above log-linear class (with smoothness parameter β = B 2 ). Suppose PrivLS is instantiated with the ISSP algorithm in [35]. Then, by [35, Theorem 5], we have that err t ( m,ε,δ, ζ ) ⩽ α , when m ⩾ ˜ O ( d α 2 + d √ log(1 /δ ) αε + d (log(1 /δ )) 2 ε 2 ) . Thus, by Theorem 4, for a suboptimality gap of O ( α ) , the

<!-- formula-not-decoded -->

Corollary 3 (Log-linear policy in high-dimension) . Consider DP-NPG with the above log-linear class (with smoothness parameter β = B 2 ). Suppose PrivLS is instantiated with Algorithm 5 in [36]. Then, by [36, Theorem 6.2], we have that err t ( m,ε,δ, ζ ) ⩽ α when m ⩾ ˜ O ( log(1 /ζ ) α 4 + √ log(1 /ζ ) log(1 /δ ) α 3 ε ) . Thus, by Theorem 4, for a suboptimality gap of O ( α ) , the sample

<!-- formula-not-decoded -->

One key subtlety behind these two corollaries is that W can be large and depend on d . This is due to the fact that the update w t in both PrivLS oracles is privatized by a Gaussian noise in the last step. Thus, by standard concentration of a Gaussian vector, W can be on the order of √ d . This subtlety is somewhat unique due to the interplay between PrivLS oracle and 'regret-lemma' type analysis in Theorem 4. In practice, one can properly truncate w t , which preserves privacy. In theory, we conjecture that one may use a different technique (e.g., based on the three-point lemma in [37]) to avoid the requirement of bounded w t , which is left to be an exciting future work.

Connection to private stochastic optimization. In the context of the above discussion and corollaries, one in-between solution is to aim for suffering dimension dependence only in the private term. This leads us to consider private SGD over a bounded domain, since the estimation error in (9) is equivalent to excess population risk under realizability. Thus, one can leverage existing results on private stochastic optimization to bound err t ( m,ε,δ, ζ ) , which does not have dimension dependence in the non-private term, but at a cost of a slower rate O (1 / √ m ) vs. O (1 /m ) . See Appendix C.

## 7 Conclusion

We initiate a systematic theoretical investigation into the sample complexity of differentially private PO, leveraging a unified meta-algorithmic framework and reductions to private regression problems. We establish the first set of sample complexity results for several widely used PO algorithms under differential privacy constraints, including DP-PG , DP-NPG and DP-REBEL . Our analysis not only quantifies the privacy cost in PO but also uncovers subtle and important interplays between privacy

mechanisms and algorithmic structures in PO. These insights offer practical guidance for designing privacy-preserving PO methods. We hope our work will open new avenues for future research in both the theoretical understanding and empirical development of private policy optimization. Besides, although we mainly focus on the theory in the main body, we have also managed to conduct some experiments as proof of concept, see Appendix H for details.

## Acknowledgements

This work was supported in part by the National Science Foundation (NSF) under Grant Nos. CAREER-2441519, CNS-2312835, and CNS-2153220.

## References

- [1] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning , 8:229-256, 1992.
- [2] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems , 12, 1999.
- [3] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- [4] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 , 2024.
- [5] Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research , 22(98):1-76, 2021.
- [6] Rui Yuan, Robert M Gower, and Alessandro Lazaric. A general sample complexity analysis of vanilla policy gradient. In International Conference on Artificial Intelligence and Statistics , pages 3332-3380. PMLR, 2022.
- [7] Lior Shani, Yonathan Efroni, and Shie Mannor. Adaptive trust region policy optimization: Global convergence and faster rates for regularized mdps. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pages 5668-5675, 2020.
- [8] Boyi Liu, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural trust region/proximal policy optimization attains globally optimal policy. Advances in neural information processing systems , 32, 2019.
- [9] Yule Liu, Heyi Zhang, Jinyi Zheng, Zhen Sun, Zifan Peng, Tianshuo Cong, Yilong Yang, Xinlei He, and Zhuo Ma. Grpo privacy is at risk: A membership inference attack against reinforcement learning with verifiable rewards, 2025. URL https://arxiv.org/abs/2511.14045 .
- [10] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3 , pages 265-284. Springer, 2006.
- [11] Sham M Kakade. A natural policy gradient. Advances in neural information processing systems , 14, 2001.
- [12] Zhaolin Gao, Jonathan D. Chang, Wenhao Zhan, Owen Oertell, Gokul Swamy, Kianté Brantley, Thorsten Joachims, J. Andrew Bagnell, Jason D. Lee, and Wen Sun. Rebel: Reinforcement learning via regressing relative rewards, 2024. URL https://arxiv.org/abs/2404.16767 .
- [13] Alexandre Rio, Merwan Barlier, and Igor Colin. Differentially private policy gradient. arXiv preprint arXiv:2501.19080 , 2025.

- [14] Sayak Ray Chowdhury and Xingyu Zhou. Differentially private regret minimization in episodic markov decision processes. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 6375-6383, 2022.
- [15] Xingyu Zhou. Differentially private reinforcement learning with linear function approximation. Proceedings of the ACM on Measurement and Analysis of Computing Systems , 6(1):1-27, 2022.
- [16] Borja Balle, Maziar Gomrokchi, and Doina Precup. Differentially private policy evaluation. In International Conference on Machine Learning , pages 2130-2138. PMLR, 2016.
- [17] Fan Wu, Huseyin A Inan, Arturs Backurs, Varun Chandrasekaran, Janardhan Kulkarni, and Robert Sim. Privately aligning language models with reinforcement learning. arXiv preprint arXiv:2310.16960 , 2023.
- [18] Zhaolin Gao, Wenhao Zhan, Jonathan D Chang, Gokul Swamy, Kianté Brantley, Jason D Lee, and Wen Sun. Regressing the relative future: Efficient policy optimization for multi-turn rlhf. arXiv preprint arXiv:2410.04612 , 2024.
- [19] Tongxin Zhou, Yingfei Wang, Lu Yan, and Yong Tan. Spoiled for choice? personalized recommendation for healthcare decisions: A multiarmed bandit approach. Information Systems Research , 34(4):1493-1512, 2023.
- [20] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems , 35:27730-27744, 2022.
- [21] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems , 36:53728-53741, 2023.
- [22] Kamalika Chaudhuri, Claire Monteleoni, and Anand D Sarwate. Differentially private empirical risk minimization. Journal of Machine Learning Research , 12(3), 2011.
- [23] Raef Bassily, Adam Smith, and Abhradeep Thakurta. Private empirical risk minimization: Efficient algorithms and tight error bounds. In 2014 IEEE 55th annual symposium on foundations of computer science , pages 464-473. IEEE, 2014.
- [24] Raef Bassily, Vitaly Feldman, Kunal Talwar, and Abhradeep Guha Thakurta. Private stochastic convex optimization with optimal rates. Advances in neural information processing systems , 32, 2019.
- [25] Giuseppe Vietri, Borja Balle, Akshay Krishnamurthy, and Steven Wu. Private reinforcement learning with pac and regret guarantees. In International Conference on Machine Learning , pages 9754-9764. PMLR, 2020.
- [26] J Hsu, Z Huang, A Roth, T Roughgarden, and ZS Wu. Private matchings and allocations. SIAM Journal on Computing , 45:1953-1984, 2016.
- [27] Roshan Shariff and Or Sheffet. Differentially private contextual linear bandits. Advances in Neural Information Processing Systems , 31, 2018.
- [28] Dan Qiao and Yu-Xiang Wang. Near-optimal differentially private reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 9914-9940. PMLR, 2023.
- [29] Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science , 9(3-4):211-407, 2014.
- [30] Yuhao Ding, Junzi Zhang, and Javad Lavaei. On the global optimum convergence of momentumbased policy gradient. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera, editors, Proceedings of The 25th International Conference on Artificial Intelligence and Statistics , volume 151 of Proceedings of Machine Learning Research , pages 1910-1934. PMLR, 28-30 Mar 2022.

- [31] Yanli Liu, Kaiqing Zhang, Tamer Ba¸ sar, and Wotao Yin. An improved analysis of (variancereduced) policy gradient and natural policy gradient methods, 2022. URL https://arxiv. org/abs/2211.07937 .
- [32] Lingxiao Wang, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural policy gradient methods: Global optimality and rates of convergence, 2019. URL https://arxiv.org/abs/1909. 01150 .
- [33] Yuda Song, Yifei Zhou, Ayush Sekhari, J Andrew Bagnell, Akshay Krishnamurthy, and Wen Sun. Hybrid rl: Using both offline and online data can make rl efficient. arXiv preprint arXiv:2210.06718 , 2022.
- [34] Frank McSherry and Kunal Talwar. Mechanism design via differential privacy. In 48th Annual IEEE Symposium on Foundations of Computer Science (FOCS'07) , pages 94-103. IEEE, 2007.
- [35] Gavin Brown, Jonathan Hayase, Samuel Hopkins, Weihao Kong, Xiyang Liu, Sewoong Oh, Juan C Perdomo, and Adam Smith. Insufficient statistics perturbation: Stable estimators for private least squares. arXiv preprint arXiv:2404.15409 , 2024.
- [36] Fan Chen, Jiachun Li, Alexander Rakhlin, and David Simchi-Levi. Near-optimal private learning in linear contextual bandits, 2025. URL https://arxiv.org/abs/2502.13115 .
- [37] Rui Yuan, Simon S Du, Robert M Gower, Alessandro Lazaric, and Lin Xiao. Linear convergence of natural policy gradient methods with log-linear policies. arXiv preprint arXiv:2210.01400 , 2022.
- [38] Matteo Papini, Damiano Binaghi, Giuseppe Canonaco, Matteo Pirotta, and Marcello Restelli. Stochastic variance-reduced policy gradient. In International conference on machine learning , pages 4026-4035. PMLR, 2018.
- [39] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning , pages 1889-1897. PMLR, 2015.
- [40] Jalaj Bhandari and Daniel Russo. Global optimality guarantees for policy gradient methods. Operations Research , 72(5):1906-1927, 2024.
- [41] Boyi Liu, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural proximal/trust region policy optimization attains globally optimal policy (2019). arXiv preprint arXiv:1906.10306 , 1906.
- [42] Yanli Liu, Kaiqing Zhang, Tamer Basar, and Wotao Yin. An improved analysis of (variancereduced) policy gradient and natural policy gradient methods. Advances in Neural Information Processing Systems , 33:7624-7636, 2020.
- [43] Lior Shani, Yonathan Efroni, Aviv Rosenberg, and Shie Mannor. Optimistic policy optimization with bandit feedback. In International Conference on Machine Learning , pages 8604-8613. PMLR, 2020.
- [44] Qi Cai, Zhuoran Yang, Chi Jin, and Zhaoran Wang. Provably efficient exploration in policy optimization. In International Conference on Machine Learning , pages 1283-1294. PMLR, 2020.
- [45] Alekh Agarwal, Mikael Henaff, Sham Kakade, and Wen Sun. Pc-pg: Policy cover directed exploration for provable policy gradient learning. Advances in neural information processing systems , 33:13399-13412, 2020.
- [46] Andrea Zanette, Ching-An Cheng, and Alekh Agarwal. Cautiously optimistic policy optimization and exploration with linear function approximation. In Conference on Learning Theory , pages 4473-4525. PMLR, 2021.
- [47] Qinghua Liu, Gellért Weisz, András György, Chi Jin, and Csaba Szepesvári. Optimistic natural policy gradient: a simple efficient policy optimization framework for online rl. Advances in Neural Information Processing Systems , 36:3560-3577, 2023.

- [48] Nikita Mishra and Abhradeep Thakurta. (nearly) optimal differentially private stochastic multi-arm bandits. In Proceedings of the Thirty-First Conference on Uncertainty in Artificial Intelligence , pages 592-601, 2015.
- [49] Touqir Sajed and Or Sheffet. An optimal private stochastic-mab algorithm based on optimal private stopping rule. In International Conference on Machine Learning , pages 5579-5588. PMLR, 2019.
- [50] Sayak Ray Chowdhury and Xingyu Zhou. Distributed differential privacy in multi-armed bandits. arXiv preprint arXiv:2206.05772 , 2022.
- [51] Wenbo Ren, Xingyu Zhou, Jia Liu, and Ness B Shroff. Multi-armed bandits with local differential privacy. arXiv preprint arXiv:2007.03121 , 2020.
- [52] Yulian Wu, Xingyu Zhou, Youming Tao, and Di Wang. On private and robust bandits. Advances in Neural Information Processing Systems , 36:34778-34790, 2023.
- [53] Jiahao He, Jiheng Zhang, and Rachel Q Zhang. A reduction from linear contextual bandits lower bounds to estimations lower bounds. In International Conference on Machine Learning , pages 8660-8677. PMLR, 2022.
- [54] Sayak Ray Chowdhury and Xingyu Zhou. Shuffle private linear contextual bandits. arXiv preprint arXiv:2202.05567 , 2022.
- [55] Xingyu Zhou and Sayak Ray Chowdhury. On differentially private federated linear contextual bandits. arXiv preprint arXiv:2302.13945 , 2023.
- [56] Paul Luyo, Evrard Garcelon, Alessandro Lazaric, and Matteo Pirotta. Differentially private exploration in reinforcement learning with linear representation. arXiv preprint arXiv:2112.01585 , 2021.
- [57] Vitaly Feldman, Tomer Koren, and Kunal Talwar. Private stochastic convex optimization: optimal rates in linear time. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing , pages 439-449, 2020.
- [58] Hilal Asi, Vitaly Feldman, Tomer Koren, and Kunal Talwar. Private stochastic convex optimization: Optimal rates in l1 geometry. In International Conference on Machine Learning , pages 393-403. PMLR, 2021.
- [59] Raef Bassily, Cristóbal Guzmán, and Anupama Nandi. Non-euclidean differentially private stochastic convex optimization. In Conference on Learning Theory , pages 474-499. PMLR, 2021.
- [60] Janardhan Kulkarni, Yin Tat Lee, and Daogao Liu. Private non-smooth empirical risk minimization and stochastic convex optimization in subquadratic steps. arXiv preprint arXiv:2103.15352 , 2021.
- [61] Andrew Lowy and Meisam Razaviyayn. Private federated learning without a trusted server: Optimal algorithms for convex losses. In The Eleventh International Conference on Learning Representations , 2023.
- [62] Changyu Gao, Andrew Lowy, Xingyu Zhou, and Stephen J Wright. Private heterogeneous federated learning without a trusted server revisited: Error-optimal and communication-efficient algorithms for convex losses. arXiv preprint arXiv:2407.09690 , 2024.
- [63] Di Wang, Changyou Chen, and Jinhui Xu. Differentially private empirical risk minimization with non-convex loss functions. In International Conference on Machine Learning , pages 6526-6535. PMLR, 2019.
- [64] Yingxue Zhou, Xiangyi Chen, Mingyi Hong, Zhiwei Steven Wu, and Arindam Banerjee. Private stochastic non-convex optimization: Adaptive algorithms and tighter generalization bounds. arXiv preprint arXiv:2006.13501 , 2020.

- [65] Raman Arora, Raef Bassily, Tomás González, Cristóbal A Guzmán, Michael Menart, and Enayat Ullah. Faster rates of convergence to stationary points in differentially private optimization. In International Conference on Machine Learning , pages 1060-1092. PMLR, 2023.
- [66] Daogao Liu, Arun Ganesh, Sewoong Oh, and Abhradeep Guha Thakurta. Private (stochastic) non-convex optimization revisited: Second-order stationary points and excess risks. Advances in Neural Information Processing Systems , 36:65618-65641, 2023.
- [67] Daogao Liu and Kunal Talwar. Adaptive batch size for privately finding second-order stationary points. arXiv preprint arXiv:2410.07502 , 2024.
- [68] Yu-Xiang Wang. Revisiting differentially private linear regression: optimal and adaptive prediction &amp; estimation in unbounded domain. arXiv preprint arXiv:1803.02596 , 2018.
- [69] Or Sheffet. Old techniques in differentially private linear regression. In Algorithmic Learning Theory , pages 789-827. PMLR, 2019.
- [70] T Tony Cai, Yichen Wang, and Linjun Zhang. The cost of privacy: Optimal rates of convergence for parameter estimation with differential privacy. The Annals of Statistics , 49(5):2825-2850, 2021.
- [71] Gavin Brown, Krishnamurthy Dvijotham, Georgina Evans, Daogao Liu, Adam Smith, and Abhradeep Thakurta. Private gradient descent for linear regression: Tighter error bounds and instance-specific uncertainty estimation. arXiv preprint arXiv:2402.13531 , 2024.
- [72] Prateek Varshney, Abhradeep Thakurta, and Prateek Jain. (nearly) optimal private linear regression for sub-gaussian data via adaptive clipping. In Conference on Learning Theory , pages 1126-1166. PMLR, 2022.
- [73] Xiyang Liu, Prateek Jain, Weihao Kong, Sewoong Oh, and Arun Suggala. Label robust and differentially private linear regression: Computational and statistical efficiency. Advances in Neural Information Processing Systems , 36:23019-23033, 2023.
- [74] Prashanti Anderson, Ainesh Bakshi, Mahbod Majid, and Stefan Tiegel. Sample-optimal private regression in polynomial time. arXiv preprint arXiv:2503.24321 , 2025.
- [75] Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487 , 2021.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: : In the abstract and introduction, we clearly state the main contributions of our work. We present a meta algorithm for Private PO and give out three private algorithm: DP-PG , DP-NPG , DP-REBEL and analyze their sample complexity.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations of this work can be found in appendix.

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

Justification: In this paper, all theoretical results are accompanied by a full set of assumptions and complete proofs. These are clearly stated and numbered within the main text and are cross-referenced appropriately. Detailed proofs for major theorems are provided in the Appendix.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [NA]

Justification: The nature of this paper is purely theoretical research, and its conclusions are derived through mathematical deduction and logical argumentation; therefore, it does not involve experimental validation.

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

## Answer: [NA]

Justification: Our conclusions are derived through mathematical deduction and logical argumentation, so it does not include experiments requiring code.

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

Justification: This paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: This paper does not include experiments.

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

Justification: This paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research adheres to all guidelines outlined in the NeurIPS Code of Ethics. Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

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

Justification: This paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This paper does not use existing assets.

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

Justification: This paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This research does not involve any form of crowdsourcing or studies with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The study does not include human subjects, and therefore IRB approval is not required.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This research does not use large language models as an important component of the core methodology, it is used only for editing.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Contents of Appendix

| A   | Additional Related Work                         | Additional Related Work                         |   21 |
|-----|-------------------------------------------------|-------------------------------------------------|------|
| B   | Tabular Softmax with Log-barrier Regularization | Tabular Softmax with Log-barrier Regularization |   22 |
| C   | Connection to Private Stochastic Optimization   | Connection to Private Stochastic Optimization   |   23 |
| D   | Differentially Private REBEL                    | Differentially Private REBEL                    |   23 |
| E   | Proof of Chapter 5                              | Proof of Chapter 5                              |   24 |
|     | E.1                                             | Proof of Theorem 2 . . . . . . . . .            |   24 |
|     | E.2                                             | Proof of Theorem 3 . . . . . . . . .            |   27 |
|     | E.3                                             | Proof of Theorem 5 . . . . . . . . .            |   27 |
|     |                                                 | E.3.1 FOSP of softmax policy . .                |   28 |
|     |                                                 | E.3.2 Global optimum convergence                |   30 |
| F   | Proof of Chapter 6                              | Proof of Chapter 6                              |   32 |
|     | F.1                                             | Proof of Theorem 4 . . . . . . . . .            |   32 |
|     | F.2                                             | Proof of Lemma 2 . . . . . . . . . .            |   33 |
| G   | Proof of Appendix D                             | Proof of Appendix D                             |   34 |
|     | G.1                                             | Proof of Theorem 6 . . . . . . . . .            |   35 |
| H   | Experiments                                     | Experiments                                     |   36 |
| I   | Limitations                                     | Limitations                                     |   37 |

## A Additional Related Work

Policy optimization. The theoretical study of policy optimization can be roughly divided into two lines of work. The first line often assumes a certain reachability (coverage) condition and takes a perspective from optimization. Under this assumption, various types of policy gradient methods have been investigated, including REINFORCE [1], variance-reduction variants [38], and preconditioned variants such as NPG [11], TRPO [39], and PPO [3]. The performance metrics are often convergence rate or sample complexity, see some typical results in [5, 40, 6, 41, 42]. The second line of work focuses on the exploration setting, i.e., without the coverage condition. To this end, the algorithm design is often based on optimism, e.g., an optimistic version of NPG or PPO via bonus terms or global optimism. Several recent papers have made progress in this direction for tabular RL [43], linear mixture MDP [44], linear MDP and more [45-47]. Our paper can be viewed as the first work that aims to privatize the first line of work above.

Differentially private RL and bandits. Recently, a line of work studies RL (bandits) under the constraint of differential privacy, e.g., multi-armed bandits (MABs) [48-52], contextual bandits [27, 36, 53-55], and RL [25, 14, 28, 15, 56]. To the best of our knowledge, only [14] and [15] studied private RL under policy optimization. They consider the exploration setting and characterize the cost of privacy in regret bounds by privatizing optimistic versions of PPO (NPG) under tabular or linear function approximations, respectively. In contrast, we take the optimization perspective (with the coverage condition) and study private PO for general reward/policy function approximations.

Differentially private stochastic optimization. Extensive research has been done around private stochastic convex and non-convex optimization. In particular, for the problem of differentially private stochastic convex optimization (DP-SCO), [24] gave the first optimal algorithm in terms of excess population loss, and [57] developed the first linear-time efficient and optimal algorithm. There are also many follow-up papers under various settings, e.g., [58-62]. Moving to the non-convex case, the performance metrics include first-order or second-order population stationary points (e.g., [63-67]) as well as excess population loss (e.g., [66]). As already mentioned, these results in private stochastic optimization can be useful to us since the estimation error in our master theorem is equivalent to excess population loss/risk under square loss and realizability, see Appendix C for more details.

Differentially private linear regression. For the specific problem of private linear regression (which is a special case of DP-SCO), one can possibly achieve better results by leveraging its structure. We can roughly group the research efforts 1 in this area based on the conditions on the boundedness of the feature vector (and the true parameter). Assuming boundedness, state-of-the-art results are established in [68, 69]. On the other hand, without boundedness but under a general sub-Gaussian data, ISSP in [35] is the first efficient and nearly-optimal algorithm, which does not require an identity covariance matrix compared with [70, 71] and does not depend on the condition number of the covariance matrix compared with [72, 73]. Moving from approximate DP to pure DP, the very recent work [74] gives the first polynomial-time and sample-optimal private regression algorithm. As mentioned before, these results on private linear regression are useful to us since, under log-linear policy parameterization, our estimation error reduces to the estimation error in linear regression.

## B Tabular Softmax with Log-barrier Regularization

In this section, we move to the second scenario for our DP-PG of global convergence where we consider the tabular case with the classic softmax policy:

Definition 3 (Tabular softmax policy) . Consider a finite state space X and action space Y . For any state-action pair ( x, y ) ∈ X × Y , the softmax policy is given by

<!-- formula-not-decoded -->

where θ ∈ R |X||Y| .

One key motivation here is to leverage the tabular structure and the specific property of softmax policy to establish a sample complexity of global optimum convergence that is independent of the parameter γ . To this end, as in the non-private case [5, 6], we will consider a regularized problem, whose FOSP turns out to be an approximate global optimal solution of the unregularized (original) objective, for proper choice of regularization. In particular, we consider the following log-barrier regularization objective:

<!-- formula-not-decoded -->

where the KL divergence is KL ( p, q ) = E x ∼ p [ log p ( x ) q ( x ) ] , Unif χ denotes the uniform distribution over a set χ and λ &gt; 0 is the regularization constant.

We will run our DP-PG over this regularized objective by using the sample-based gradient estimator at each step with proper choices of batch size and learning rate. Then, we have the following main result regarding the global optimum convergence in terms of the unregularized J ( θ ) . The proof is given in Appendix E.3.

Theorem 5. Consider Algorithm 2 applied to J λ ( θ ) . For any m &gt; 0 , setting σ 2 = 16 ln(1 . 25 /δ ) · R 2 max G 2 m 2 ε 2 ensures ( ε, δ ) -DP as in Definition 2. Further, there exist proper choices of parameters for m and η , such that the following holds

<!-- formula-not-decoded -->

1 As before, this is not a complete list of all the works in this area.

when the sample size satisfies N ⩾ O ( 1 α 6 + √ d α 9 / 2 ε ) .

Remark 7 . The first term in the sample complexity bound matches the non-private one in Yuan et al. [6], while the second term is the lower-order privacy cost (for constant ε and d ). We note that while the dependence on α is worse than the previous one, there is no dependence on γ in the bound, which could offer benefits when γ is quite small.

## C Connection to Private Stochastic Optimization

In this section, we first aim to bound the estimation error in (9) by leveraging the existing result in private SCO. In particular, we aim to apply Theorem 3.2 in [24] for a Lipschitz and smooth loss function. The first step is to realize that under realizability, the LHS in (9) is equal to the excess population loss for a square loss, which is both Lipschitz (with parameter of order O ( B 2 W ) under our boundedness assumption for both feature and parameter space) and smooth (with parameter of O ( B 2 ) ). Hence, by [24, Theorem 3.2], we can obtain that err t ( m,ε,δ, ζ ) ⩽ α when m ⩾ √

<!-- formula-not-decoded -->

the sample complexity bound is N = T · m = ˜ O δ ( ( 1 α 6 + √ d α 4 ε ) · Poly ( B,W ) ) . Recall that due to projection in the mini-batch SGD in [24], one can ensure that ∥ w t ∥ ≤ W for some W that is independent of d . Thus, the non-private term does not depend on d . We mention in passing that one can also potentially directly bound the error in Assumption 4 by leveraging the excess population loss for non-convex loss functions.

## D Differentially Private REBEL

In this section, we will consider DP-REBEL which includes a general private regression oracle as the PrivUpdate in Algorithm 1 and then similar to our previous DP-NPG , we will give some concrete applications under different specific regression oracles.

Our proposed DP-REBEL is Algorithm 1 with its PrivUpdate being instantiated in Algorithm 5, which also relies on a general private least-square regression oracle to return an approximate minimizer of an estimation problem under the square loss. The square loss in (11) is the same as before in (3).

## Algorithm 5 PrivUpdate Instantiation for DP-REBEL

- 1: Input: D t = { ( x i , y i , ̂ A t ( x i , y i )) } i m =1 , current policy θ t , base policy µ , learning rate η , PrivLS oracle
- 2: Output: θ t +1
- 3: Call the PrivLS oracle on D t := { ( x i , y i , ̂ A t ( x i , y i )) } to find an approximate minimizer w t of

<!-- formula-not-decoded -->

where expectation is over x ∼ ρ, y ∼ µ ( ·| x ) , y ′ ∼ π θ t ( ·| x ) .

- 4: Output policy θ t +1 = θ t + ηw t

We now turn to establish a generic performance guarantee of DP-REBEL . Similar to DP-NPG , we assume that the approximate minimizer w t returned by PrivLS at each iteration satisfies the following guarantee.

Assumption 6 (Private estimation error) . For each t ∈ [ T ] , the PrivLS oracle satisfies ( ε, δ ) -differential privacy and ensures that with probability at least 1 -ζ ,

<!-- formula-not-decoded -->

for some statistical error function err 2 t ( m,ε,δ, ζ ) depending on batch size m , privacy parameters ( ε, δ ) , and probability ζ , also, expectation here is over x ∼ ρ, y ∼ µ ( ·| x ) , y ′ ∼ π θ t ( ·| x ) .

Remark 8 . This assumption parallels Assumption 4 in DP-NPG and ensures that our private oracle accurately estimates the relative reward differences while preserving our DP in PO as Definition 2.

Our main result is given by the following theorem.

Theorem 6. Under Assumption 6, according to Lemma 11 and Lemma 12, we have with probability at least 1 -ζ , for any comparator policy π ∗ such that:

<!-- formula-not-decoded -->

Remark 9 . If we simply set the base policy µ = π θ t , which make this assumption simpler, then we can have a tighter bound, it is easy to show that the bound will turns to 2 A √ ln |Y| T + √ 2 C µ → π ∗ T ∑ T t =1 err t ( m,ε,δ, ζ ) .

The application here is totally same as the DP-NPG , thus we can directly use our previous Corollary 1, Corollary 2 and Corollary 3 which derive the almost same bound of sample complexity. For PrivLS with exponential mechanism, consider DP-REBEL with PrivLS as in Algorithm 4, for a given suboptimality gap of O ( α + √ C µ → π ∗ α approx ) , the sample complexity bound is N = T · m = ˜ O ( ( 1 α 4 + 1 α 4 ε ) · log |W| · A 2 ) . For log-linear policy class with realizability, assume that err t ( m,ε,δ, ζ ) ⩽ α , therefore, for log-liner-policy in low-dimension and high-dimension, we have m ⩾ ˜ O ( d α 2 + d √ log(1 /δ ) αε + d (log(1 /δ )) 2 ε 2 ) , and m ⩾ ˜ O ( log(1 /ζ ) α 4 + √ log(1 /ζ ) log(1 /δ ) α 3 ε ) , respectively. Thus, we can derive such sample complexity: N = T · m = ˜ O δ ( ( d α 4 + d α 3 ε + d α 2 ε 2 ) · A 2 ) for log-liner-policy in low-dimension and N = T · m = ˜ O δ ( ( 1 α 6 + 1 α 5 ε ) · A 2 ) for high-dimension.

## E Proof of Chapter 5

## E.1 Proof of Theorem 2

Lemma 3 (ABC) . There exists constants A,B,C ⩾ 0 such that the policy gradient estimator satisfies:

<!-- formula-not-decoded -->

where ∇ J ( θ ) = E x ∼ ρ,y ∼ π θ ( ·| s ) [ A π θ ( x, y ) ∇ θ log π θ ( y | x )] , and A = 0 , B = 1 -1 /m,C = 4 R 2 max G 2 m + dσ 2 .

Proof. For notation simplicity, we let g θ ( τ i ) := A π θ ( x i , y i ) ∇ θ log π θ ( y i | x i ) . Thus, we have ˜ ∇ m J ( θ ) = 1 m ∑ i g θ ( τ i ) + Z . Notice that E [ g θ ( τ i )] = E [ ˜ ∇ m J ( θ ) ] = ∇ J ( θ ) , cause Z is the gaussian bias, which expectation is 0.

Now, we have

<!-- formula-not-decoded -->

To proceed, we need to establish an upper bound on E [ ∥ g θ ( τ 1 ) ∥ 2 ] . In particular, we have

<!-- formula-not-decoded -->

which follows from Assumption 1.

Hence, we conclude that:

<!-- formula-not-decoded -->

i.e., ABC condition in (12) is satisfied with A = 0 , B = 1 -1 /m,C = 4 R 2 max G 2 m + dσ 2

Lemma 4 (Smoothness under LS) . Under LS assumption in Assumption 1, J ( · ) is L -smooth, namely ∥ ∥ ∇ 2 J ( θ ) ∥ ∥ ⩽ L for all θ , with

<!-- formula-not-decoded -->

Proof. For smoothness, it suffices to bound the operator norm of Hessian, i.e., ∥ ∥ ∇ 2 J ( θ ) ∥ ∥ . By definition, we have

<!-- formula-not-decoded -->

where in ( a ) , we let p θ ( x, y ) := ρ ( x ) π θ ( y | x ) , and ( b ) holds by chain rules.

Thus, we have

<!-- formula-not-decoded -->

For T 1 and T 2 , by Assumption 1, we have

<!-- formula-not-decoded -->

which hence completes the proof.

Lemma 5 (Adapted from Theorem 3.4 in Yuan et al. [6]) . Suppose that J satisfies smoothness and the ABC assumption in Lemma 3. Consider the iterates θ t of the PG method with step size η t = η ∈ (0 , 2 LB ) , let δ 1 = J ∗ -J ( θ 1 ) . In particular, if A = 0 , we have:

<!-- formula-not-decoded -->

where θ U is uniformly sampled from { θ 1 , ..., θ T

Proof of Theorem 2. Followed by Lemma 5, when η &lt; 1 LB , we can simplify the equation into this:

<!-- formula-not-decoded -->

where B = 1 -1 /m , δ 1 = J ∗ -J ( θ 1 ) , L = 2 R max ( G 2 + F ) , C = 4 R 2 max G 2 m + dσ 2 , G and F are constants.

From Theorem 1, to make sure our algorithm satisfy the ( ε, δ ) -DP as in Definition 2, we set σ 2 = 16 ln(1 . 25 /δ ) · R 2 max G 2 m 2 ε 2 .

Based on Lemma 5 and Equation (14), choose η = min { 1 LB , √ 2 δ 1 √ TLC } , we have:

<!-- formula-not-decoded -->

To proceed, we need to determine the value of m.

In order to balance the terms in the convergence bound O ( m N + 1 √ N + √ d ε √ Nm ) , we set m N = √ d ε √ Nm . Thus, we have:

<!-- formula-not-decoded -->

Substituting back, the convergence bound simplifies to:

<!-- formula-not-decoded -->

## E.2 Proof of Theorem 3

Proof. We know that:

<!-- formula-not-decoded -->

Besides, followed by Lemma 1, we obtain that:

<!-- formula-not-decoded -->

which holds by

Taking expectation over both sides, condition on θ t , yields that

<!-- formula-not-decoded -->

where ( a ) holds by Theorem 2.

By applying Jensen inequality twice, we have:

<!-- formula-not-decoded -->

So we can derive that:

<!-- formula-not-decoded -->

In that case, we finally get the result:

<!-- formula-not-decoded -->

Suppose J ∗ -1 T ∑ T t =1 E [ J ( θ t )] ⩽ O ( α ) + O ( √ α bias ) , we have: √

<!-- formula-not-decoded -->

## E.3 Proof of Theorem 5

Based on softmax settings in Definition 3, by simple calculus, we have

<!-- formula-not-decoded -->

where 1 y ∈ R |Y| is a vector with all zero entries except being 1 for the entry corresponding to action y , and H ( π x ( θ )) = Diag ( π x ( θ )) -π x ( θ ) π x ( θ ) ⊤ .

In particular, for softmax, we can determine the G and F in Assumption 1:

<!-- formula-not-decoded -->

## E.3.1 FOSP of softmax policy

Lemma 6. The regularized gradient estimator ˜ ∇ m J λ ( θ ) satisfies ABC assumption in Lemma 3 with parameters:

<!-- formula-not-decoded -->

Specifically, we have the variance bound: E [ ∥ ∥ ∥ ˜ ∇ m J λ ( θ ) ∥ ∥ ∥ 2 ] ⩽ ( 1 -1 m ) ∥∇ J λ ( θ ) ∥ 2 + dσ 2 + 2 m ( 1 -1 |Y| )( 4 R 2 max + λ 2 |X| ) .

Proof. Similar to Appendix E.1, here we let g θ ( τ ) be a stochastic gradient estimator of one single sampled trajectory τ . Thus we have: ˜ ∇ m J ( θ ) = 1 m ∑ i g θ ( τ i ) + Z .

From equation (10) we have the following gradient estimator

<!-- formula-not-decoded -->

For a state x ∈ X , we have

<!-- formula-not-decoded -->

where 1 |Y| ∈ R |Y| is a vector of all ones.

Thus we have

<!-- formula-not-decoded -->

where 1 ∈ R |X||Y| and [ π x ( θ )] x ∈X = [ π x 1 ( θ ); ... ; π x |X| ( θ )] ∈ R |X||Y| is the stacking of the vectors π x ( θ ) .

Next, taking expectation on the trajectories, we have

<!-- formula-not-decoded -->

where ( a ) and ( c ) hold by definition of ∇ J λ ( θ ) ; ( b ) holds by definition of ˜ ∇ m J ( θ ) ; ( d ) holds by ( p + q ) 2 ⩽ 2 p 2 +2 q 2 .

In particular, we have

<!-- formula-not-decoded -->

where the inequality is obtained by using ∥ π x ( θ ) ∥ 2 ⩽ 1 .

As for E [ ∥ g θ ( τ 1 ) ∥ 2 ] , we have

<!-- formula-not-decoded -->

where the equality is obtained by Assumption 1 with G 2 = ( 1 -1 |Y| ) .

Combining above, we proved the gradient estimator ˜ ∇ m J λ ( θ ) satisfies ABC assumption with where

<!-- formula-not-decoded -->

Lemma 7 (Regularized FOSP Convergence) . Under the learning rate condition η &lt; 1 LB , the iterates satisfy:

<!-- formula-not-decoded -->

where B = 1 -1 /m , δ 1 = J ∗ -J ( θ 1 ) , L = 2 R max (2 -1 |Y| ) , and C as defined in Lemma 6.

Proof. To proceed with the analysis, we first introduce the following key lemma:

Lemma 8 (Adapted from Lemma E.3 in Yuan et al. [6]) . J ( · ) with the softmax tabular policy is 2 R max ( 2 -1 |Y| ) -smooth and 2 R max √ 1 -1 |Y| -Lipschitz.

From Lemma 8, we know that J λ ( · ) is smooth and Lipschitz. Then, based on Lemma 5, we have:

<!-- formula-not-decoded -->

Assuming η &lt; 1 LB , the above equation simplifies to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the sensitivity ∆ of the gradient estimator ˜ ∇ m J λ ( θ ) is dominated by the data-dependent term. Despite introducing the regularization term λ , this term only depends on the policy parameters θ (independent of data), thus it does not affect the sensitivity. So the ℓ 2 -sensitivity of the gradient remains same as before.

Lemma 9. let σ 2 = 16 ln(1 . 25 /δ ) · R 2 max G 2 m 2 ε 2 , the batch size m be set as: m = (1 /ε ) 2 / 3 ( Nd ) 1 / 3 , and η = min ( 1 LB , √ 2 δ 1 √ TLC ) , we have:

<!-- formula-not-decoded -->

Proof. for η = min ( 1 LB , √ 2 δ 1 √ TLC ) we know:

<!-- formula-not-decoded -->

Plug in σ 2 = 16 ln(1 . 25 /δ ) · R 2 max G m 2 ε 2 and m = (1 /ε ) 2 / 3 ( Nd ) 1 / 3 , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E.3.2 Global optimum convergence

We first introduce an important proposition to bound our global private optimum convergence of softmax with log barrier regularization.

Proposition 2 (Adapted from Theorem 5.2 in Agarwal et al. [5]) . Suppose θ is such that ∥∇ J λ ( θ ) ∥ ⩽ λ 2 |X||Y| . Then for every initial distribution ρ , we have

<!-- formula-not-decoded -->

Proof. Firstly, we define the following set of 'bad' iterates:

<!-- formula-not-decoded -->

with λ = α 2 .

From Proposition 2, we know that if ∥∇ J λ ( θ ) ∥ ⩽ λ 2 |X||Y| , we have J ∗ -J ( θ ) ⩽ 2 λ. Hence, we have:

<!-- formula-not-decoded -->

where (a) holds by J ( · ) ⩽ 2 R max , then J ∗ -J ( θ t ) ⩽ J ∗ + J ( θ t ) ⩽ 4 R max . Now we turn to bound | I + | . From the definition, we have:

<!-- formula-not-decoded -->

Through a straightforward mathematical transformation, we get

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Taking expectation over the iterations on both sides, we have

<!-- formula-not-decoded -->

To guarantee that J ∗ -1 T ∑ T t =1 E [ J ( θ t )] ⩽ α , we need to show:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Obviously, we have:

Hence, based on Lemma 9, it is obvious to show that:

<!-- formula-not-decoded -->

## F Proof of Chapter 6

## F.1 Proof of Theorem 4

For notation simplicity, we let π t = π θ t . By the performance difference lemma, we have

<!-- formula-not-decoded -->

Define err ∗ t := E x ∼ ρ,y ∼ π ∗ ( ·| x ) [( A π θ t ( x, y ) -w ⊤ t ∇ log π θ t ( y | x ) )] . Then, we have

<!-- formula-not-decoded -->

where (a) holds by the update rule of our algorithm; (b) is true since the β -smooth condition of log π θ ( y | x ) is equivalent to the following inequality:

<!-- formula-not-decoded -->

(c) follows from the Assumption 5, which has a bounded norm of W , along with telescope sum; (d) is true since π 1 is a uniform distribution at each state. Thus, dividing by T on both sides and choosing η = √ 2 log |Y| TβW 2 , yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To bound err ∗ t , we will simply leverage the guarantee of the regression oracle and the concentrability coefficient to transfer from µ to π ∗ . In particular, we have for any t ∈ [ T ]

<!-- formula-not-decoded -->

where (a) holds by Cauchy-Schwarz inequality; in (b), we define the single-policy concentrability coefficient C µ → π ∗ := max x,y π ∗ ( y | x ) µ ( y | x ) ; (c) follows directly from the guarantee of PrivateLS oracle.

Finally, putting everything together, yields

<!-- formula-not-decoded -->

## F.2 Proof of Lemma 2

A key lemma in our proof is the following form of Freedman's inequality.

Lemma 10 (Lemma A.2 in [75]) . Let { X i } i ≤ n be a real-valued martingale difference sequence adapted to a filtration {F i } i ≤ n . If | X i | ≤ R almost surely, then for any η ∈ (0 , 1 /R ) , with probability at least 1 -ζ ,

<!-- formula-not-decoded -->

where E i -1 [ · ] := E [ ·|F i -1 ] .

Proof of Lemma 2. For any fixed h ∈ H , we define

<!-- formula-not-decoded -->

If we define the filtration F i = σ ( u 1 , z 1 , . . . , u i , z i ) and let E i -1 [ · ] = E [ ·|F i -1 ] , then we have that { D h i } i ≤ m where

<!-- formula-not-decoded -->

is a martingale difference sequence adapted to {F i } i ≤ m . We further notice that

<!-- formula-not-decoded -->

where the last step holds by the boundedness of z i , h ∈ H and h ∗ . Moreover, by definition, we have

<!-- formula-not-decoded -->

With the above results, we first apply Lemma 10 to { D h i } i ≤ m along with a union bound, yielding that with probability at least 1 -ζ , for all h ∈ H

<!-- formula-not-decoded -->

Similarly, we can apply Lemma 10 to {-D h i } i ≤ m along with a union bound, which give us

<!-- formula-not-decoded -->

Now, we set h = ̂ h in (22), i.e., the output of the exponential mechanism, and by the standard utility guarantee of the exponential mechanism [34], we have

<!-- formula-not-decoded -->

where h ′ ∈ arg min h ∈H L ( h ) = arg min h ∈H ∑ i ∈ [ m ] ( h ( u i ) -z i ) 2 . Since ∑ m i =1 U h ′ i ≤ ∑ m i =1 U ˜ h i where ˜ h := argmin h ∈H ∑ m i =1 E i -1 [( h ( u i ) -h ∗ ( u i )) 2 ] , by (23) and the above inequality, we have

<!-- formula-not-decoded -->

where ( a ) holds the assumption on the approximation error.

## G Proof of Appendix D

Lemma 11. Consider any t ∈ [ T ] . For notation simplicity, we define f t ( x, y ) := 1 η ln π t +1 ( y | x ) π t ( y | x ) . Define ∆( x, y ) = f t ( x, y ) -r ( x, y ) . Define ∆ π t ( x ) = E y ∼ π t ( ·| x ) ∆( x, y ) and ∆ µ ( x ) = E y ∼ µ ( ·| x ) ∆( x, y ) . Under Assumption 6, for all t , we have the following:

<!-- formula-not-decoded -->

Proof. From Assumption 6, we have:

<!-- formula-not-decoded -->

In that case, since the total sum is less than err 2 t ( m,ε,δ, ζ ) , it follows that each term must be less than err 2 t ( m,ε,δ, ζ ) . Hence, the lemma holds.

Lemma 12. Assume max x,y,t | A t ( x, y ) | ⩽ A ∈ R + , and π 1 is uniform over Y . Then with η = √ ln( |Y| ) / ( A 2 T ) , for the sequence of policies computed by REBEL, we have:

<!-- formula-not-decoded -->

Proof. By the definition of f t , we have

<!-- formula-not-decoded -->

Taking exp on both sides, we get:

<!-- formula-not-decoded -->

Denote

<!-- formula-not-decoded -->

and the advantage

<!-- formula-not-decoded -->

We can rewrite the above update rule as:

<!-- formula-not-decoded -->

The remain part of the proof is similar to the analysis of NPG in F.1.

## G.1 Proof of Theorem 6

Proof. We know that:

<!-- formula-not-decoded -->

Then, we have:

<!-- formula-not-decoded -->

where (a) holds by Lemma 12.

Then we need to bound: E x E y ∼ π ∗ ( ·| x ) ( A π t ( x, y ) -A t ( x, y )) 2 . By the definition of concentrability coefficient C µ → π ∗ , we know that:

<!-- formula-not-decoded -->

We now bound E x,y ∼ µ ( ·| x ) ( A π t ( x, y ) -A t ( x, y )) 2 .

<!-- formula-not-decoded -->

Recall the g t ( x, y ) = r ( x, y ) + ∆( x, y ) -∆ µ ( x ) , and from Lemma 11, we can see that

<!-- formula-not-decoded -->

For E x E y ′ ∼ π t ( ·| x ) ( r ( x, y ′ ) -g t ( x, y ′ )) 2 , we have:

<!-- formula-not-decoded -->

where the last inequality uses Lemma 11 again.

Combining things together, we can conclude that:

<!-- formula-not-decoded -->

Hence, we can derive our main theorem:

<!-- formula-not-decoded -->

## H Experiments

Environment. We conduct experiments on the CartPole-v1 environment from OpenAI Gym, a standard benchmark for evaluating policy gradient methods. The task requires balancing a pole on a moving cart, with a maximum episode reward of 500.

Policy Parameterization. Policies are represented by two-layer fully-connected neural networks with 64 hidden units, ReLU activation, and softmax output layer, i.e., the architecture is Linear (4 , 64) → ReLU → Linear (64 , 2) → Softmax.

Privacy Settings. We evaluate privacy-preserving algorithms under two privacy budgets: ( ε, δ ) = (5 , 10 -5 ) and ( ε, δ ) = (3 , 10 -5 ) , representing moderate and strong privacy guarantees respectively.

Training Details. All algorithms are trained for 100 epochs with batch size 10 (i.e., 10 episodes per gradient update) and discount factor γ = 0 . 99 . We use advantage normalization with baseline subtraction for variance reduction. Each algorithm is trained with 3 random seeds, and we report the average performance with standard deviation.

Evaluation. We evaluate performance using three metrics: (i) Mean Final Reward : average reward in the final epoch across all seeds, (ii) Std Final Reward : standard deviation of final rewards across seeds, and (iii) Best Epoch Mean : highest average reward achieved during training.

Results. Table 1 summarizes the performance under different privacy settings.

We can see that (i) NPG consistently outperforms PG in both private and non-private settings, demonstrating the benefit of curvature-aware updates, (ii) DP-NPG with ε = 5 achieves near-optimal performance ( ∼ 500 ), aligning with our theoretical predictions and empirical findings in Rio et al. [13] who use PPO instead of NPG, and (iii) as privacy budget decreases (smaller ε ), performance degrades as expected from theory.

Table 1: Performance summary for CartPole-v1 under different privacy budgets ( ε = 5 and ε = 3 , δ = 10 -5 ).

| Algorithm   | ε   |   Mean Final Reward |   Std Final Reward |   Best Epoch Mean |
|-------------|-----|---------------------|--------------------|-------------------|
| PG          | N/A |              334.37 |              25.25 |            361.7  |
| DP-PG       | 5   |              190.34 |              52.91 |            199.17 |
| DP-PG       | 3   |              143.87 |              22.88 |            187.17 |
| NPG         | N/A |              492.9  |              10.04 |            500    |
| DP-NPG      | 5   |              478.73 |              28.05 |            494.7  |
| DP-NPG      | 3   |              400.87 |              65.37 |            410.43 |

## I Limitations

In this study, we propose private variants of three classical algorithms for policy optimization and provide a comprehensive analysis of their sampling complexity under both private and non-private settings. Our analysis successfully recovers the classical complexity bounds in the non-private regime, validating the theoretical soundness of our approach. However, our current results focus only on the one-pass sampling setting; the sampling complexity in the multi-pass scenario may admit further improvements.

Moreover, while this work primarily focuses on the theoretical foundations of the proposed algorithms, we have also conducted simple empirical validations to support our theoretical findings. It should be noted, however, that we have not yet performed large-scale evaluations on real-world datasets. Nevertheless, since our methods serve as core components in policy optimization, they have broad applicability across various reinforcement learning domains-particularly in privacy-sensitive settings such as reinforcement learning with human feedback (RLHF) and medical data analysis. Applying our approach to these areas could further enhance the secure handling of sensitive information.