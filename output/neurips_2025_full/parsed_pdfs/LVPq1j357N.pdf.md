## Learning Personalized Ad Impact via Contextual Reinforcement Learning under Delayed Rewards ∗

## Yuwei Cheng

Department of Statistics University of Chicago Chicago, IL 60637 yuweicheng@uchicago.edu

## Zifeng Zhao

Mendoza College of Business University of Notre Dame Notre Dame, IN 46556

zzhao2@nd.edu

## Haifeng Xu

Department of Computer Science and Data Science University of Chicago Chicago, IL 60637 haifengxu@uchicago.edu

## Abstract

Online advertising platforms use automated auctions to connect advertisers with potential customers, requiring effective bidding strategies to maximize profits. Accurate ad impact estimation requires considering three key factors: delayed and long-term effects, cumulative ad impacts such as reinforcement or fatigue, and customer heterogeneity. However, these effects are often not jointly addressed in previous studies. To capture these factors, we model ad bidding as a Contextual Markov Decision Process (CMDP) with delayed Poisson rewards. For efficient estimation, we propose a two-stage maximum likelihood estimator combined with data-splitting strategies, ensuring controlled estimation error based on the first-stage estimator's (in)accuracy. Building on this, we design a reinforcement learning algorithm to derive efficient personalized bidding strategies. This approach achieves a near-optimal regret bound of ˜ O ( dH 2 √ T ) , where d is the contextual dimension, H is the number of rounds, and T is the number of customers. Our theoretical findings are validated by simulation experiments.

## 1 Introduction

E-commerce is expanding rapidly worldwide, with online sales expected to constitute 23% of total retail by 2027, supported by a 14.4% annual growth [ITA, 2025]. This growth has made digital advertising inevitable, empowering tech giants like Google, Meta, Microsoft, and Amazon, which leverage automated auctions to connect advertisers with customers. Auto-bidding, where platforms handle bid placement on behalf of advertisers, has grown significantly because of its simplified interaction for the advertisers and improved performance thanks to real-time optimization [Aggarwal et al., 2024, Google Ads Support, 2025, Facebook Ads, 2025, Microsoft Ads, 2025, Amazon Ads, 2025]. Given its importance, it is crucial to develop effective ad bidding strategies.

Developing effective bidding strategies requires accurately understanding advertising impacts. Psychological studies have long shown that advertising has a delayed and long-term impact on consumer beliefs and attitudes, ultimately shaping purchasing behavior over time [Vakratsas and Ambler, 1999,

∗ This work is supported by the AI2050 program at Schmidt Sciences (Grant G-24-66104), Army Research Office Award W911NF-23-1-0030, and NSF Award CCF-2303372.

Lewis and Wong, 2022, Sakalauskas and Kriksciuniene, 2024]. Advertising effectiveness varies significantly across individuals. Observational studies reveal substantial heterogeneity based on demographics, platform usage patterns, and census data [Liu et al., 2019, Gordon et al., 2019]. These insights recommend personalized e-commerce advertising, suggesting platforms should leverage customer data to target high-value users and optimize bidding strategies tailored to diverse behavioral profiles [Sakalauskas and Kriksciuniene, 2024]. Additionally, the impact of ads also depends on their cumulative effect: while repeated exposure can strengthen brand recognition, it may also lead to ad fatigue-highlighting a subtle trade-off that is often overlooked in 'learning to bid' literature [Pechmann and Stewart, 1988, Lane, 2000, You et al., 2015, Bell et al., 2022, Guo and Jiang, 2024].

Limitation in recent work. However, these insights are not fully reflected in current algorithmic designs. The problem of 'learning to bid' has been widely studied, with most prior work modeling it as a (contextual) bandit problem that assumes immediate rewards, such as click-through rates, which prioritizes short-term customer engagement [Weed et al., 2016, De Haan et al., 2016, Ren et al., 2017, Feng et al., 2018, 2023, Han et al., 2024, Zhang and Luo, 2024]. This approach, however, neglects the delayed and cumulative effects of advertising on consumption, potentially leading to incentive misalignment [Deng et al., 2022]. This misalignment is further exacerbated by the rise of auto-bidding systems, where platforms automatically manage bidding decisions on behalf of advertisers. While platforms typically optimize for engagement-based metrics, advertisers ultimately care about long-term revenue growth via production conversion 2 . Recent work has therefore begun to emphasize metrics like target return on ad spend as a practical alternative [Wang et al., 2019, Aggarwal et al., 2024] and design bidding strategies which account for delayed and cumulative effects of ads. Recently, Badanidiyuru et al. [2023] models the long-term causal impact of ad impressions using Markov Decision Process with mixed and delayed Poisson rewards. However, this approach assumes homogeneous treatment effects across users, overlooking the importance of personalization, a crucial factor in advertising effectiveness. To the best of our knowledge, no theoretical work jointly addresses all three aspects-delayed effects, cumulative impacts, and heterogeneity-in modeling ad effectiveness and designing bidding algorithms, potentially due to the modeling complexity and difficulty in estimation.

Our contribution. To address this gap, motivated by the initial proposal of Contextual Markov Decision Process (CMDP) [Hallak et al., 2015] to model customer behavior during website interactions, we model auto-bidding as CMDP with delayed Poisson rewards-using context to capture personalized ad impacts and states to capture ads cumulative effects. For effective estimation, rather than fitting all model parameters simultaneously using a single giant likelihood function, we introduce a novel data-splitting strategy and develop a two-stage maximum likelihood estimator that ensures controlled estimation error in the presence of delayed impacts. Based on this efficient online estimation oracle, we design a reinforcement learning algorithm to solve the CMDP with near-optimal regret of ˜ O ( dH 2 √ T ) , where d is the contextual dimension, H is the number of rounds, and T is the number of customers. Finally, we perform simulation studies which validate our theoretical findings.

## 2 Modeling Personalized Ad Impact by CMDP with Delayed Rewards

Notation. For a positive integer T , we denote [ T ] = { 1 , 2 , . . . , T } . We use standard asymptotic notations, including O ( · ) , Ω( · ) , and Θ( · ) , as well as their counterparts ˜ O ( · ) , ˜ Ω( · ) , and ˜ Θ( · ) to suppress logarithmic factors. The symbol e represents the base of the natural logarithm. The Mahalanobis norm is defined as ∥ x ∥ Σ = √ x ⊤ Σ x . ∥ · ∥ 2 represents L 2 norm. For vectors x and y , we use ⟨ x , y ⟩ and x ⊤ y interchangeably to denote their inner product.

To address the complexities of advertisement impacts on product conversions, we model online ad bidding as a Contextual Markov Decision Process (CMDP). This framework captures the three key impacts discussed in Section 1 and allows transitions and rewards to depend on context x t , personalized information of customer t , supporting dynamic and personalized bidding. While incorporating x t directly into the state is possible, it greatly enlarges the state space and complicates learning [Levy and Mansour, 2023]. Instead, we adopt a CMDP formulation-common in user-driven applications-that keeps the state compact and treats context as auxiliary information [Hallak et al., 2015], preserving both efficiency and personalization. Our analysis focuses on ad platforms that bid on behalf of advertisers. We use 'ad platform' and 'learner' interchangeably.

2 We defer a more detailed discussion of the 'learning-to-bid' literature to Appendix A.1

Mathematically, a CMDP is defined as the tuple ( X , S , A , M ) , where X ⊆ R d represents the contextual feature space, S denotes the state space capturing customer ad exposure history, and A is the action space for ad bidding strategies. The mapping M assigns each context x to a Markov Decision Process (MDP), M ( x ) = ( S , A , P x , R x , S 1 , H ) , where the state transition probability P x and reward function R x depend on x . S 1 is the initial state. H represents the maximum number of customer-learner interactions with details in the context of online bidding as below.

Definition 2.1 (State) . The state of customer t at round h , denoted as S t h = [ S t h, 1 , S t h, 2 ] , encodes information about the two most recent ad exposures. Specifically, let G t h, 1 represent the round of the most recent ad impression, we set S t h, 1 = h -G t h, 1 , which captures the time elapsed since that impression. Similarly, let G t h, 2 denote the round of the second-to-last ad impression, we set S t h, 2 = G t h, 1 -G t h, 2 , representing the time interval between the two most recent impressions. S t h, 1 ∈ H 1 = {∞ , 1 , 2 , . . . , H -1 } and S t h, 2 ∈ H 2 = {-∞ , 1 , 2 , . . . , H -2 } . The interpretation of ∞ and -∞ is deferred to Remark 2.3.

Assumption 2.2 (Observation) . The expected product conversion rate µ t h at round h for customer t with context x t follows

<!-- formula-not-decoded -->

The observed product conversion y t h follows a Poisson distribution, i.e. y t h ∼ Poi ( µ t h ) . d S t h, 1 represents the delayed advertisements impact, S t h, 1 ∈ H 1 = {∞ , 1 , 2 , . . . , H -1 } . o t h indicates the bidding outcome for customer t at round h . o t h = 1 if the bid is won and o t h = 0 otherwise.

Remark 2.3 . If customer t has no prior ad exposure, the state is S t h = [ ∞ , -∞ ] . Specifically, when no advertisement has been successfully displayed, product conversion y t h ∼ Poi ( µ t h ) , where µ t h = d ∞ x ⊤ t θ -∞ , Here, x ⊤ t θ -∞ represents the natural demand in the absence of ads, which may vary with context x t to capture factors such as income, preferences, tastes, and seasonality [Manandhar, 2018]. We set d ∞ = 1 to avoid identifiability issue. If the learner wins the bid, y t h ∼ Poi ( µ t h ) with µ t h = x ⊤ t θ ∞ , where θ ∞ captures the effect of first-time ad exposure. It is possible to generalize the linear modeling of the expected product conversion rate to a more flexible form, i.e., µ t h = h S t h, 1 ( x t ) . For example, h S t h, 1 may be modeled as a state-dependent neural network, though this would come at the cost of sacrificing theoretical guarantees. We refer readers to Appendix A.2 for a detailed discussion.

Figure 1: Illustration

<!-- image -->

As illustrated in Figure 1, if no new advertisement is displayed at round h , the impact of the previous ad shown at round G t h, 1 carries over, with a delayed and long-term effect governed by d h -G t h, 1 , depending on the time interval h -G t h, 1 . This delayed factor allows ad effects to span multiple rounds, enabling delayed conversion peaks, as d l is not restricted to be less than 1. This results in an expected product conversion of d h -G t h, 1 x ⊤ t θ G t h, 1 -G t h, 2 (Assumption

2.2), where θ G t h, 1 -G t h, 2 captures the impact of the ad shown at G t h, 1 , which affected by its most recent predecessor at G t h, 2 . In contrast, if a new ad is successfully displayed at round h , its effect overrides the previous one, but itself is influenced by the most recent prior impression at round G t h, 1 . The resulting impact is modeled by θ h -G t h, 1 , leading to an expected conversion of x ⊤ t θ h -G t h, 1 .

A natural question arises: why does the cumulative advertising impact depend only on the most recent display, rather than the entire history of ad exposures? This modeling choice is motivated by the recency effect, a well-documented cognitive bias in psychology and marketing [Murphy et al., 2006, Chatfield, 2016, Phillips-Wren et al., 2019]. It suggests that individuals tend to place greater weight on recent experiences when making decisions. This phenomenon is especially relevant in advertising, where exposure timing significantly affects consumer behavior [Chatfield, 2016], and underlies practical approaches like Google's attribution models [Google Ads Attribution, 2025] which uses last-touch heuristics that prioritize recent ad exposures. In fact, our state formulation is very flexible and readily accommodates an enlarged state space. It supports a broad class of CMDP

formulations for θ S , where S encodes domain knowledge about ad effects, and remains compatible with our proposed learning algorithm. For example, if we believe that all past ad exposures contribute to customer behavior, we can easily extend the state to include S t h = [ S t h, 1 , S t h, 2 , n t h ] , where n t h is the total number of successful ad displays in the past. We refer readers to Appendix A.3 for a detailed discussion of the flexibility of our state formulation.

The bidding outcome o t h depends on both learner's bid a t h and the competitors' bids. The bidding space is given by a t h ∈ [0 , B A ] , where B A is the maximum allowable bid. a t h = 0 indicates that the learner opts out of the auction. At each round, the learner submits a t h and wins if its bid exceeds all competitors' bid. The probability of winning by a t h is shown below.

Definition 2.4 (Transition) . For a given bid amount a t h , the probability of winning the auction, denoted as P ( o t h = 1) , is modeled by F h ( a t h , x t ) , where F h : [0 , B A ] ×X → [0 , 1] represents the cumulative distribution function (CDF) of the Highest Other Bids (HOB).

Remark 2.5 . Given the current state S t h = [ S t h, 1 , S t h, 2 ] and bid a t h , the next state transitions according to F h ( a t h , x t ) . With probability F h ( a t h , x t ) , the bid is successful, and the next state is S t h +1 = [1 , S t h, 1 ] . Otherwise, the bid is lost, and the state transitions to S t h +1 = [ S t h, 1 +1 , S t h, 2 ] 3 .

This CDF, F h ( a t h , x t ) , of HOB captures the competitive bidding environment by modeling both context x t and time h , allowing for round-to-round variation from changing competitors and contextdriven bid adjustments. In line with standard practice, we consider a full information feedback setting, where the realized HOB m t h is observed by the learner regardless of the bidding outcome [Cesa-Bianchi et al., 2014, Feng et al., 2018, Zhang et al., 2022, Badanidiyuru et al., 2023]. This setting is practical since bidders can always access the 'minimum-bid-to-win" feedback [Google Developers, 2024]. Also, ad platforms inherently know realized bids for their auto-bidding algorithms. We assume m t h follows a lognormal distribution, a common modeling choice in the literature for advertisement auctions [Laffont et al., 1995, Wilson, 1998, Smith et al., 2003, Skitmore, 2008, Ballesteros-Pérez and Skitmore, 2017] (Assumption 2.6), resulting the probability of winning with a bid a t h as F h ( a t h , x t ) = Φ((log( a t h ) -⟨ x t , β h ⟩ ) /σ h ) , where Φ denotes the CDF of the standard normal distribution, β h ∈ R d represents the highest willingness to pay for displaying ads from other competitors, capturing the competitiveness of the environment, while σ h reflects the associated variability.

Assumption 2.6 (Lognormal Distribution of HOB) . log( m t h ) ∼ N ( ⟨ x t , β h ⟩ , σ 2 h ) .

In a second-price auction, the format we study, the winning bidder pays the second-highest bid, with payment given by p h ( a t h , x t ) = a t h -1 F h ( a t h , x t ) ∫ a t h 0 F h ( v, x t ) dv . This format is standard in both industry and research [Cooper and Fang, 2008, Weed et al., 2016, Zhao and Chen, 2019], and our analysis extends to other single-item auctions without entry fees, such as first-price auctions. To bid effectively, the learner must balance the probability of winning, the incurred payment, and the expected product conversion. Without loss of generality, we normalize the value of each unit of product conversion to ν = 1 , leading to the following definition of the expected reward.

<!-- formula-not-decoded -->

By these formulation, in the context of online bidding, the tuple ( X , S , A , P x t , { R t h ( S t h , a t h , x t ) } H h =1 , S t 1 , H ) is a CMDP (Fact 2.7, Appendix D.1). P x t is the joint probability distribution of states S t h , induced by {F h ( · , x t ) } h ∈ [ H ] . We direct reader to Appendix A.4 for an illustration example for modeling online bidding as a CMDP. Fact 2.7 . ( X , S , A , P x t , { R t h ( S t h , a t h , x t ) } H h =1 , S t 1 , H ) is a CMDP.

## 2.1 Learning goal: regret minimization

The learner interacts with T customers over H rounds, receiving context information x t for customer t . The learner begins without prior knowledge of the product conversion parameters Θ = { θ l } l ∈H ∪ { d l } l ∈H 1 4 nor the transition F x t := {F h ( · , x t ) } H h =1 . We assume Θ and transition parameters { β h } h ∈ [ H ] ∪ { σ h } h ∈ [ H ] are bounded. The context space X is also bounded, with no distributional

3 Due to the special meaning of ∞ and -∞ , we define h -∞ = ∞ , and h -∞-( -∞ ) = ∞ .

4 H = H 1 ∪ H 2

assumptions on the arrival of x t . b is a small positive constant, which ensures that displaying an ad always yields a non-zero (expected) purchase quantity (Assumption 2.8).

Assumption 2.8. There exists positive constants b , B x , B θ , B d , B β , ¯ σ such that ∥ x t ∥ 2 ≤ B x , and θ ⊤ l x t ≥ b , ∀ t ∈ [ T ] , ∀ l ∈ H , ∥ θ l ∥ 2 ≤ B θ , ∀ l ∈ H , d l ∈ [0 , B d ] , ∀ l ∈ H 1 , and ∥ β h ∥ 2 ≤ B β , σ h ≤ ¯ σ, ∀ h ∈ [ H ] .

The learner's objective is to minimize the cumulative regret over T customers with regret defined as:

<!-- formula-not-decoded -->

The expectation is taken over the policy π t : S × X → [0 , B A ] generated by the algorithm G employed by the learner. We define OPT (Θ , x t , F x t ) as the optimal expected utility achievable for a customer with contextual features x t over H rounds, under the true parameters Θ and the distribution F x t . The reward collected by the policy π t over H rounds for customer t , denoted as R ( π t ; x t , Θ , F x t ) , denoted by R ( π t ; x t , Θ , F x t ) = E S t h ∼P x t [ ∑ H h =1 R t h ( S t h , π t ( S t h , x t ) , x t )] .

## 3 Algorithm Design

In this section, we introduce design principles for solving CMDPs with delayed Poisson rewards. The proposed algorithm (Algorithm 1) consists of three main stages: exploration, exploitation, and estimation, with the estimation stage playing a central role.

The estimation stage contains three key components. First, ridge regression [Abbasi-Yadkori et al., 2011] combined with two-stage variance estimators is used to estimate the transition dynamics F x t , handling potential adversarial arrivals of x t . Second, a variant of online Newton estimator [Xue et al., 2024] is employed to estimate the instant advertisement effects { θ l } l ∈H . Third, a novel two-stage maximum likelihood estimator (TS-MLE) is developed to estimated the delayed impacts { d l } l ∈H 1 .

The key challenge in online estimation is that a naive approach-simultaneously estimating and updating all parameters by maximizing a joint log-likelihood L (Θ) -is infeasible due to two main issues. First, the log-likelihood function L (Θ) is non-concave in Θ , making it difficult to ensure a unique solution. Second, the score equations ∇ Θ L (Θ) = 0 lack closed-form solutions, rendering direct analysis intractable. These challenges motivate the development of the estimator, TS-MLE, which efficiently estimates delayed effects while maintaining computational tractability.

The core idea of the TS-MLE is to divide the estimation into manageable steps. Intuitively, if the estimation error, ∥ ˆ θ l -θ l ∥ 2 , is small, ˆ θ l can then be treated as a close approximation of the true parameter θ l . Based on this approximation, a maximum likelihood estimator ˆ d l can be constructed, ensuring that the estimation error for d l remains small. This approach is feasible because the conditional log-likelihood L ( { d l } l ∈H 1 |{ ˆ θ l } l ∈H ) is concave in { d l } l ∈H 1 (Eqn. (2)), and both its gradient ∇ { d l } l ∈H 1 L (Eqn. (12)) and the solution to the corresponding score equation ∇ { d l } l ∈H 1 L = 0 (Eqn. (3)) are straightforward to compute.

To control the estimation error of ˆ d l , it is crucial to ensure that this error depends only on the error in ˆ θ l , not vice versa. This prevents feedback loops between the estimation errors of ˆ d l and ˆ θ l , which would otherwise complicate mathematical analysis and make the estimation process intractable. To achieve this, we employ a carefully designed data-splitting strategy, where separate data subsets are allocated exclusively for estimating θ l and d l , ensuring their errors remain independent.

To achieve this separation, we introduce two datasets, W t,l and D t,l , which share a common structure but serve distinct purposes (Def. 3.1). Both datasets focus on round h for customer t where the most recent bid win occurred l rounds before the current round ( S t h, 1 = l ). The key difference lies in the parameter being estimated. W t,l includes rounds that observations { y t h } h ∈ W t,l with mean ⟨ θ l , x t ⟩ . Thus this datasets is used to estimate θ l , as they depend solely on θ l . D t,l contains rounds that observations { y t h } h ∈ D t,l with mean d l ⟨ θ S t h, 2 , x t ⟩ . This datasets is used to estimate d l because they depend solely on the unknown parameter d l when { ˆ θ l } l ∈H are given. This ensures that the estimation error of d l depends on θ l , while the estimation error of θ l remains independent of d l .

Definition 3.1. For customer t , we define two datasets: W t,l = { h | S t h, 1 = l, o t h = 1 } and D t,l = { h | S t h, 1 = l, o t h = 0 } , for l ∈ H\ [ -∞ ] . Wedefine W t, -∞ = { h | S t t, 1 = ∞ , o t h = 0 } for the estimation of θ -∞ . The collection of observations across the first t customers are W t l = { W s,l } t s =1 and D t l = { D s,l } t s =1 . The size of D t l is given by N t,l = | D t l | .

Taken this together, the log-likelihood of d l up to the first t customers as:

<!-- formula-not-decoded -->

Differentiating the log-likelihood with respect to d l , setting it to zero, and solving for d l , results in

<!-- formula-not-decoded -->

In the online setting, ˆ d t l estimates d l as of customer t , analogous to ˆ θ t l . Instead of using the most recent estimates { ˆ θ t S s h, 2 } t s =1 as the first-stage input for estimating ˆ d t l , we leverage { ˆ θ s S s h, 2 } t s =1 , a progressively updated estimate of the advertisement's impact across customer arrivals. This novel design is motivated by the observation that the estimation error of ˆ θ s S s h, 2 instead of ˆ θ t S s h, 2 in the direction of x s is well-controlled. Further details, including the confidence region D t l (line 2 of Algorithm 2), are provided in Theorem 4.1 and its proof.

̸

```
Algorithm 1 Online Contextual Reinforcement Learning with Delayed Poisson Reward input d, T, H, b , B x , B θ , B d , B A , δ , γ , Γ , n l 1: for t = 1 to T do 2: Obtain the context x t for the arriving customer t 3: if t ≤ ( H +1) n l then 4: /* Exploration */ 5: Compute l = ⌊ t n l ⌋ . 6: if l = H , set a t h = 0 , ∀ h ∈ [ H ] ; else set a t 1 = a t l +1 = B A , and a t h = 0 for ∀ h = 1 , l +1 7: for h = 1 to H do observe y t h , m t h ; then update ˆ β t h by Eqn. (5) and ˆ σ t h by Eqn. (6) 8: else 9: /* Exploitation */ 10: Update π t = arg max π max ˜ θ ∈C t -1 R ( π ; ˜ θ, ˆ F x t t -1 ) 11: for h = 1 to H do 12: Observe S t h and take action a t h = π t ( S t h , x t ) 13: Observe o t h , m t h , and y t h and update ˆ β t h by Eqn. (5) and ˆ σ t h by Eqn. (6) 14: end for 15: end if 16: /* Estimation */ 17: for l ∈ H do 18: Update dataset W t l , D t l by Def. 3.1 19: if W t,l is nonempty, compute ˆ θ t l and C t l by Algo. 3; else set ˆ θ t l = ˆ θ t -1 l and C t l = C t -1 l 20: if D t,l is nonempty, compute ˆ d t l and D t l by Algo. 2; else set ˆ d t l = ˆ d t -1 l and D t l = D t -1 l 21: end for 22: Set C t = { Θ | { θ l ∈ C t l } ∩ { d l ∈ D t l } , ∀ l ∈ H} 23: end for
```

Remark 3.2 (Key Tuning Parameters for Algorithm 1) . The input parameters d, T, and H are structural components of the CMDP. The quantities b , B x , B θ , B d , B A are defined in Assumption 2.8. The tail probability δ determines the confidence region for ˆ β l and ˆ d l and serves as an input for Algorithm 3 and Algorithm 2. The truncation threshold Γ , used to handle heavy-tailed distributions, is defined in Eq. (10) and appears in Algorithm 3. The parameter γ , defined in Lemma 3.3, serves as a weighted estimation error bound and is used in Algorithm 2. The exploration stage guarantees a minimum

number of observations to ensure reliable estimation. Specifically, the threshold n l is given by n l := ⌈ 32 log( HT ) e B d B x B θ b 2 ⌉ , which guarantees sufficient observations in W t l and D t l , ensuring | W t l | ≥ n l and N t,l = | D t l | ≥ n l for all l ∈ H in all subsequent episodes after the exploration stage.

The estimation of ˆ d t l depends on accurately estimating ˆ θ t l . Using observations y t h from W t l , the problem reduces to efficiently estimating θ l from Poisson data. To achieve this, we adopt the Confidence Region with Truncated Mean approach (see Algorithm 3 and Appendix B), introduced by Xue et al. [2024]. This method utilizes a variant of the online Newton estimator, given by:

<!-- formula-not-decoded -->

V t l = V t -1 l + 1 2 x s x ⊤ s , with V 0 l initialized as the identity matrix I d . ∇ ˜ l t ( ˆ θ t -1 l ) = ( -˜ y t h + ( x t ) ⊤ ˆ θ t -1 l ) x t . The truncated observation ˜ y t h is defined as ˜ y t h = y t h I ∥ x t ∥ ( V t l ) -1 | y t h |≤ Γ .

This truncation mitigates the impact of extreme values in y t h by setting outliers to zero, a technique originally designed for generalized linear bandit problems with heavy-tailed data.

By Lemma 3.3, the weighted estimation error ∥ θ l -ˆ θ t l ∥ 2 V t l remains well-controlled with high probability.

Lemma 3.3 (Theorem 1 in Xue et al. [2024]) . Given l ∈ H , with probability at least 1 -δ , ˆ θ t l defined in Eqn. (4) satisfies ∥ θ l -ˆ θ t l ∥ 2 V t l ≤ γ, ∀ t ≥ 0 , where γ = 896 d B x B θ (1 + B x B θ ) log ( 4 T δ ) log ( 1 + T 2 d ) +2 B 2 x B 2 θ +48 d B x B θ log ( 1 + T 2 d ) .

To estimate the transition dynamics F h , we estimate β h by Eqn. (5). Without loss of generality, we set λ = 1 .

<!-- formula-not-decoded -->

The variability σ h , similar to ˆ d t l , is estimated based on the first-stage estimators { ˆ β s h } t s =1 (Eqn. (6)). We use { ˆ β s h } t s =1 instead of ˆ β t h to control over estimation error in the direction of x s (details in Appendix C.2).

<!-- formula-not-decoded -->

In addition to estimation, the exploration period aims to gather sufficient observations for d l to ensure quadratic tail decay of Poisson (Remark 3.2).

After exploration, the algorithm enters exploitation, selecting actions via the greedy policy π t to maximize rewards within the confidence region. π t is computed via dynamic programming with time complexity Poly ( H, |S| , B A /ϵ ) when discretizing the bidding space for an ϵ -optimal solution (detail in Appendix A.5). By balancing exploration and exploitation, Algorithm 1 achieves near-optimal performance, formally proven in the next section.

## Algorithm 2 Two-Stage Maximum Likelihood Estimation

input datasets D t l , { x s } t s =1 , { ˆ θ s S s h, 2 } t s =1 and parameters b , B x , B θ , B d , d, T, H, δ, γ

- 1: Update the two-stage estimator ˆ d t l by Eqn. (3)

<!-- formula-not-decoded -->

## 4 Analysis of Near-Optimal Regret Bound

This section demonstrates the efficiency of the TS-MLE (Theorem 4.1) and analyzes the near-optimal performance of the proposed Algorithm 1 (Theorem 4.2).

Theorem 4.1 (Confidence Region for ˆ d t l ) . Let δ ≥ 1 T 4 H and N t,l ≥ n l . With probability at least 1 -δ , the estimation error | ˆ d t l -d l | , with ˆ d t l defined in Eqn. (3) , is bounded by:

<!-- formula-not-decoded -->

γ is as defined in Lemma 3.3 and n l defined in Remark 3.2.

Theorem 4.1 shows that the estimation error of TS-MLE is bounded by ˜ O (1 / √ N t,l ) , where N t,l (Def. 3.1) is the total number of observations used to estimate d l . This result demonstrates the efficiency of the estimator, achieving a near-optimal parametric convergence rate [Rao, 1992].

Proving the near-optimal convergence when data are collected from a complex CMDP with delayed observations requires precise understanding of the sources of estimation error and refined analysis to control them. As discussed in Section 3, the core idea for estimating d l is to use observations that are 'purified' with respect to d l . In particular, we construct ˆ d t l using only observations {{ y s h } h ∈ D s,l } t s =1 , where y s h ∼ Poi( d l ⟨ θ S s h, 2 , x s ⟩ ) . Then, to analyze how the estimation error of TS-MLE depends on the first-stage estimators ˆ θ s S s h, 2 , we decompose | ˆ d t l -d l | into two parts: the error by the randomness in Poisson observations (Term A in Eqn. 8) and the cumulative estimation error from the firststage estimators (Term B in Eqn. 8). η s h in Term A has sub-exponential distributions, specifically, SubE ( ed l x ⊤ s θ S s h, 2 , 1) (Lemma D.1), since y s h = d l x ⊤ s θ S s h, 2 + η s h . To control the heavy tail of Term A, we show that P ( Term A &gt; ϵ ) ≤ δ , where ϵ = √ 2 e B d B x B θ b 2 N t,l log( 2 δ ) . The exploration phase of Algorithm 1 ensures N t,l ≥ n l , providing sufficient observations to make ϵ small enough to have faster convergence.

<!-- formula-not-decoded -->

Controlling Term B is essential for bounding | ˆ d t l -d l | . Unlike traditional second-stage estimators, which typically plug in the most recent estimates ˆ θ t S h, 2 , we instead use ˆ θ s S h, 2 , as its estimation error is better controlled in the direction of x s . Specifically, let r s h = | ( ˆ θ s S s h, 2 -θ S s h, 2 ) ⊤ x s | denote this directional estimation error. By the Cauchy-Schwarz inequality, we have r s h ≤ ∥ ˆ θ s l ′ -θ l ′ ∥ V s l ′ · ∥ x s ∥ ( V s l ′ ) -1 , where l ′ = S s h, 2 . Letting γ s l ′ = ∥ ˆ θ s l ′ -θ l ′ ∥ 2 V s l ′ , Lemma 3.3 guarantees that γ s l ′ ≤ γ holds with high probability for all s and l ′ .

To analyze the growth the directional estimation error r s h for each l ∈ H over time, we apply recounting techniques. Define the dataset F l s,l ′ := { h | S s h, 1 = l, S s h, 2 = l ′ , o s h = 0 } , which partitions D s,l by S s h, 2 , the time interval between the two recent ads impression. This partitioning ensures ∑ l ′ ∈H 2 | F l s,l ′ | = | D s,l | . We then bound the total directional error as: ∑ t s =1 ∑ h ∈ D s,l | ( ˆ θ s S s h, 2 -θ S s h, 2 ) ⊤ x s | ≤ √ N t,l √∑ t s =1 ∑ H l ′ =1 ∑ h ∈ F l s,l ′ ( r s h ) 2 . Let n s,l ′ = | F l s,l ′ | , where n s,l ′ ≤ H . The total count across all partitions satisfies ∑ t s =1 ∑ l ′ ∈H 2 n s,l ′ = N t,l . We further bound the error as √ N t,l √ ∑ H l ′ =1 γ ∑ t s =1 n s,l ′ min( ∥ x s ∥ ( V s l ′ ) -1 , 1) . Applying the Elliptical Potential Lemma [Abbasi-Yadkori et al., 2011], we obtain 2 H √ d √ N t,l log ( 1 + T 2 d ) γ . Finally, by applying a union bound to account for the simultaneous occurrence of Term A and Term B, we establish Theorem 4.1 (details in Appendix C.1).

Building on the efficiency of TS-MLE, we now show that the regret of Algorithm 1 is nearly optimal, as established in Theorem 4.2.

Theorem 4.2 (Nearly Optimal Regret) . For any δ ≥ 6 T 3 , with probability at least 1 -δ , Reg T incurred by Algorithm 1 is O ( dH 2 √ T log( TH δ ) log(1 + T 2 d )) .

√

Theorem 4.2 shows that, with high probability, Algorithm 1 achieves a regret bound of ˜ O ( T ) . Given the Ω( √ T ) lower bound for CMDPs with even known transitions [Levy and Mansour, 2023], this confirms that Algorithm 1 is nearly optimal in T . As discussed earlier, the key challenge in designing an efficient reinforcement learning algorithm for CMDPs is developing an online estimation oracle that handles long-term Poisson rewards under potentially adversarial arrivals of contextual x t , without relying on distributional assumptions. This improves upon prior work [Modi and Tewari, 2020, Levy and Mansour, 2022, 2023, Levy et al., 2023], which is limited to instantaneous rewards and cannot capture long-term effects in CMDPs, or addresses long-term rewards but fails to account for the heterogeneous effects of x t [Badanidiyuru et al., 2023]. Even with TS-MLE, establishing a high-probability regret upper bound for Algorithm 1 is nontrivial and requires careful analysis. We provide a proof sketch below, with detailed arguments in Appendix C.

The proof begins by decomposing Reg T into four terms. In Eqn. (9), Term (i) equals ∑ T t = τ OPT (Θ , x t , F x t ) -∑ T t = τ OPT (Θ , x t , ˆ F x t t -1 ) , and Term (iv) = E ( ∑ T t = τ R ( π t ; x t , Θ , ˆ F x t t -1 ) -R ( π t ; x t , Θ , F x t )) , capturing the cumulative regret due to transition error, arising from the discrepancy between F x t and ˆ F x t t -1 5 . Term (iii) is ∑ T t = τ OPT ( C t -1 , x t , ˆ F x t t -1 ) -E ( ∑ T t = τ R ( π t ; x t , Θ , ˆ F x t t -1 )) , which accounts for decision error resulting from imprecise estimates of the model parameters Θ = { θ l } l ∈H ∪ { d l } l ∈H 1 . Thus, accurate estimators of F x t and Θ lead to low cumulative regret, as shown jointly by Lemma 4.3 and 4.4.

<!-- formula-not-decoded -->

Lemma 4.3 (Term (i) Upper Bound) . For δ ≥ 1 /T 4 , with probability at least 1 -δ , Term (i) in Eqn. (9) is bounded above by O ( dH 2 √ T √ log((1 + T ) /δ ) log(1 + T/d )) .

Lemma 4.3 provides a high-probability upper bound for Term (i). The proof firstly uses the simulation lemma [Kearns and Singh, 2002] to bound Term (i) by the cumulative estimation error in the transition dynamics, ∑ T t =1 sup b sup h | ˆ F t -1 h ( b, x t ) -F h ( b, x t ) | , which depends on | ˆ β t h -β h | and | (ˆ σ t h ) 2 -σ 2 h | . A high-probability bound is then derived for these estimations errors, with | (ˆ σ t h ) 2 -σ 2 h | shown to follow a sub-exponential distribution. To ensure quadratic tail decay, we set δ ≥ 1 /T 4 . A similar analysis applies to Term (iv), which is bounded by O ( dH 2 √ T √ log((1 + T ) /δ ) log(1 + T/d )) with probability at least 1 -1 /T 4 . Details are in Appendix C.2.

Lemma 4.4 (Term (iii) Upper Bound) . With probability at least 1 -2 δ , where δ ≥ 1 T 3 , Term (iii) in Eqn. (9) is bounded by O ( H 2 √ dT log(4 TH/δ ) log(1 + T/ 2 d )) .

Lemma 4.4 provides a high-probability bound for Term (iii), which can be decomposed into two parts: the cumulative estimation error of ˆ θ t l , which is ∑ T t = τ E ( ∑ H h =1 |⟨ ˜ θ t -1 S t -ˆ θ t -1 S t t -1 := arg max Θ ∈C t -1 R ( π t ; x t , Θ , F t t -1 ) . Applying Lemma 3.3 and Theorem 4.1, together with a union bound, yields the high-probability bound for Term (iii) (Appendix C.3).

h, 1 h, 1 + ˆ θ t -1 S t h, 1 -θ S t h, 1 , x t ⟩| ) + ∑ T t = τ E ( ∑ H h =1 |⟨ ˜ θ t -1 S t h, 2 -ˆ θ t -1 S t h, 2 + ˆ θ t -1 S t h, 2 -θ S t h, 2 , x t ⟩| ) and the cumulative estimation error of ˆ d l , specifically, ∑ T t = τ E ( ∑ H h =1 | ˜ d t -1 S t h, 1 -ˆ d t -1 S t h, 1 + ˆ d t -1 S t h, 1 -d S t h, 1 | ) . Here, ˜ Θ ˆ x

Meanwhile, in Eqn. (9), Term (ii) = ∑ T t = τ OPT (Θ , x t , ˆ F x t t -1 ) -∑ T t = τ OPT ( C t -1 , x t , ˆ F x t t -1 ) , measures the gap in regret between a chosen point and the optimal point within a feasible set. Lemma 4.5 proves Θ ∈ C t for all t ≥ τ with high probability, ensuring that Term (ii) can be negative with high probability. Combining these results, we conclude that, with probability at least 1 -6 T 3 , the overall regret Reg T is O ( dH 2 √ T log( TH δ ) log(1 + T 2 d )) .

Lemma 4.5 (Confidence Region of Θ ) . With probability at least 1 -δ with δ ≥ 2 T 3 , Θ ∈ C t , ∀ t ≥ τ .

## 5 Experiments

Weconducted experiments, which leads to two key findings: first, our algorithm achieves near-optimal regret scaling of √ T , and second it consistently outperforms several strong baselines, including aggressive bidding , passive bidding , and random bidding strategies, with definition shown below.

5 We have τ = Hn l +1 and regret incurred in the exploration phase is Θ(log T ) .

Environment Setup. We simulate a second-price auction with horizon H = 3 (a realistic setting since advertisers typically show ads only a few times per user), context dimension d = 2 , and T = 20 , 000 sequential customers. The first 2400 rounds are used for exploration, and the remaining rounds for exploitation. Context vectors x t ∈ R 2 , as well as parameters d l , β h , σ h , are sampled elementwise from |N (0 , 1) | + 0 . 1 to ensure positivity, while θ l ∼ 5 |N (0 , 1) | + 0 . 1 . Under this configuration, each ad impression yields an average instantaneous reward roughly five times its cost (i.e., the highest other bid). The reward also decays rapidly over time, making aggressive bidding -always winning the impression-close to optimal and therefore a competitive baseline.

Estimators. We estimate four sets of parameters: θ l via the online Newton method (Algorithm 3) with truncation threshold Γ = 100 , 000 , bound B θ = 10 , zero initialization, and V 0 = I . We estimate delay impact d l via the two-stage MLE (Eq. (3)) using D t,l , ˆ θ , and x t ; β by ridge regression (Eq. (5), λ = 1 . 0 ); and σ via empirical variance (Eq. (6)).

Connection Between Bid, Outcome, and Reward. In our setting, the bidder's action is the bid amount, but due to the second-price auction format, the reward depends only on the outcome-whether the bid exceeds the HOB-not the bid itself. Winning results in paying the HOB; losing incurs no cost. This decoupling allows optimal rewards to be computed from outcomes rather than bids. We leverage this to evaluate both oracle and policy-based rewards. With full knowledge of true parameters, we enumerate all 2 H = 8 possible win/loss outcomes and define the oracle reward as the maximum achievable value. With estimated parameters, we compute expected rewards for all outcome sequences, select the best, and evaluate it in simulation. Regret is then the cumulative difference between oracle and realized rewards, reflecting both estimation and decision errors.

Baseline Policies and Regret Comparison. We compare our algorithm against three fixed policies: Aggressive bidding : Always bids above HOB ( o t = [1 , 1 , 1]) . Random bidding : Samples o t uniformly from all 8 outcomes. Passive bidding : Rarely bids or wins ( o t = [0 , 0 , 0]) . All baselines are evaluated using the same oracle-based regret metric. Note that aggressive bidding , while competitive, still incurs O ( T ) regret since it is non-adaptive.

We ran all algorithms over 20 independent trials. Table 1 reports the mean cumulative regret ( ± 0.5 standard deviation) and the fitted regret order (via log-log regression). Our algorithm achieves an estimated regret order of 0.37, while all baselines exhibit linear regret, validating our theoretical results and demonstrating the substantial advantage of adaptive learning in ad bidding.

Table 1: Average cumulative regret ( ± 0.5 standard deviation) and associated fitted regret order ( T α ).

| t                              | Algorithm 1                                                                                        | Aggressive Bid                                                                               | Random Bid                                                                                                | Passive Bid                                                                                                      |
|--------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| 500 5000 10000 15000 20000 T α | 1770 [1174, 2379] 8465 [5585, 11345] 8484 [5606, 11363] 8505 [5628, 11382] 8525 [5649, 11401] 0.37 | 228 [121, 336] 2373 [1243, 3502] 4741 [2477, 7005] 7142 [3724, 10561] 9568 [4991, 14146] 1.0 | 1920 [1791, 2049] 19285 [17873, 20697] 38399 [35573, 41226] 57651 [53452, 61849] 76945 [71390, 82500] 1.0 | 4630 [3391, 5869] 46179 [33850, 58508] 92468 [67735, 117201] 138652 [101576, 175729] 184873 [135414, 234332] 1.0 |

## 6 Discussion and Limitation

In this paper, we model ad bidding as a CMDP, presenting a unified framework to address delayed impacts, cumulative effects, and individual personalization. We designed a near-optimal algorithm using a novel two-stage maximum likelihood estimator to effectively handle delayed rewards. However, several important directions remain open for future research. One promising extension involves incorporating budget constraints, a critical practical consideration in real-world advertising. Developing constrained optimization algorithms for CMDP with delayed rewards is highly non-trivial and will be explored in future work.Although our work validates the theoretical insights through experiments, implementing and evaluating the proposed algorithms in real-world settings would provide valuable evidence of its practical applicability. In particular, such experiments could help assess two modeling choices: the adequacy of modeling the expected product conversion rate using linear functions, and the robustness of focusing only on very recent ad exposures due to recency bias. However, given the lack of publicly available online advertising and bidding datasets, we leave this empirical validation to future work as an important step toward bridging the gap between theory and practice.

## References

- ITA. E-commerce Sales Size and Forecast, 2025. URL https://www.trade.gov/ ecommerce-sales-size-forecast . Accessed: 2025-01-25.
- Gagan Aggarwal, Ashwinkumar Badanidiyuru, Santiago R Balseiro, Kshipra Bhawalkar, Yuan Deng, Zhe Feng, Gagan Goel, Christopher Liaw, Haihao Lu, Mohammad Mahdian, et al. Auto-bidding and auctions in online advertising: A survey. ACM SIGecom Exchanges , 22(1):159-183, 2024.
- Google Ads Support. About Automated Bidding in Google Ads, 2025. URL https://support. google.com/google-ads/answer/2979071?hl=en . Accessed: 2025-01-25.
- Facebook Ads. About Automated Bidding on Facebook Ads, 2025. URL https://www.facebook. com/business/help/1619591734742116?id=2196356200683573 . Accessed: 2025-01-25.
- Microsoft Ads. Automated Bidding Tools for Performance Optimization, 2025. URL https: //about.ads.microsoft.com/en/tools/performance/automated-bidding . Accessed: 2025-01-25.
- Amazon Ads. Dynamic Bidding for Sponsored Products Guide, 2025. URL https://advertising. amazon.com/library/guides/dynamic-bidding-sponsored-products . Accessed: 202501-25.
- Demetrios Vakratsas and Tim Ambler. How advertising works: what do we really know? Journal of marketing , 63(1):26-43, 1999.
- Randall Lewis and Jeffrey Wong. Incrementality bidding and attribution, 2022. URL https: //arxiv.org/abs/2208.12809 .
- Virgilijus Sakalauskas and Dalia Kriksciuniene. Personalized advertising in e-commerce: Using clickstream data to target high-value customers. Algorithms , 17(1):27, 2024.
- Hao Liu, Yunze Li, Qinyu Cao, Guang Qiu, and Jiming Chen. Estimating individual advertising effect in e-commerce. arXiv preprint arXiv:1903.04149 , 2019.
- Brett R Gordon, Florian Zettelmeyer, Neha Bhargava, and Dan Chapsky. A comparison of approaches to advertising measurement: Evidence from big field experiments at facebook. Marketing Science , 38(2):193-225, 2019.
- Cornelia Pechmann and David W Stewart. Advertising repetition: A critical review of wearin and wearout. Current issues and research in advertising , 11(1-2):285-329, 1988.
- Vicki R Lane. The impact of ad repetition and ad content on consumer perceptions of incongruent extensions. Journal of Marketing , 64(2):80-91, 2000.
- Soeun You, Taeha Kim, and Hoon S Cha. The smartphone user's dilemma among personalization, privacy, and advertisement fatigue: An empirical examination of personalized smartphone advertisement. Information Systems Review , 17(2):77-100, 2015.
- Raoul Bell, Laura Mieth, and Axel Buchner. Coping with high advertising exposure: a sourcemonitoring perspective. Cognitive Research: Principles and Implications , 7(1):82, 2022.
- Rui Guo and Zhengrui Jiang. Optimal dynamic advertising policy considering consumer ad fatigue. Decision Support Systems , 187:114323, 2024.
- Jonathan Weed, Vianney Perchet, and Philippe Rigollet. Online learning in repeated auctions. In Vitaly Feldman, Alexander Rakhlin, and Ohad Shamir, editors, 29th Annual Conference on Learning Theory , volume 49 of Proceedings of Machine Learning Research , pages 1562-1583, Columbia University, New York, New York, USA, 23-26 Jun 2016. PMLR. URL https:// proceedings.mlr.press/v49/weed16.html .
- Evert De Haan, Thorsten Wiesel, and Koen Pauwels. The effectiveness of different forms of online advertising for purchase conversion in a multiple-channel attribution framework. International journal of research in marketing , 33(3):491-507, 2016.

- Kan Ren, Weinan Zhang, Ke Chang, Yifei Rong, Yong Yu, and Jun Wang. Bidding machine: Learning to bid for directly optimizing profits in display advertising. IEEE Transactions on Knowledge and Data Engineering , 30(4):645-659, 2017.
- Zhe Feng, Chara Podimata, and Vasilis Syrgkanis. Learning to bid without knowing your value. In Proceedings of the 2018 ACM Conference on Economics and Computation , pages 505-522, 2018.
- Zhe Feng, Christopher Liaw, and Zixin Zhou. Improved online learning algorithms for ctr prediction in ad auctions. In International Conference on Machine Learning , pages 9921-9937. PMLR, 2023.
- Yanjun Han, Tsachy Weissman, and Zhengyuan Zhou. Optimal no-regret learning in repeated first-price auctions. Operations Research , 2024.
- Mengxiao Zhang and Haipeng Luo. Online learning in contextual second-price pay-per-click auctions. In International Conference on Artificial Intelligence and Statistics , pages 2395-2403. PMLR, 2024.
- Yuan Deng, Negin Golrezaei, Patrick Jaillet, Jason Cheuk Nam Liang, and Vahab Mirrokni. Fairness in the autobidding world with machine-learned advice. arXiv preprint arXiv:2209.04748 , 4, 2022.
- Tengyun Wang, Haizhi Yang, Han Yu, Wenjun Zhou, Yang Liu, and Hengjie Song. A revenuemaximizing bidding strategy for demand-side platforms. IEEE Access , 7:68692-68706, 2019.
- Ashwinkumar Badanidiyuru, Zhe Feng, Tianxi Li, and Haifeng Xu. Incrementality bidding via reinforcement learning under mixed and delayed rewards, 2023. URL https://arxiv.org/ abs/2206.01293 .
- Assaf Hallak, Dotan Di Castro, and Shie Mannor. Contextual markov decision processes. arXiv preprint arXiv:1502.02259 , 2015.
- Orin Levy and Yishay Mansour. Optimism in face of a context:regret guarantees for stochastic contextual mdp. Proceedings of the AAAI Conference on Artificial Intelligence , 37(7):8510-8517, Jun. 2023. doi: 10.1609/aaai.v37i7.26025. URL https://ojs.aaai.org/index.php/AAAI/ article/view/26025 .
- Binita Manandhar. Effect of advertisement in consumer behavior. Management Dynamics , 21(1): 46-54, 2018.
- Jamie Murphy, Charles Hofacker, and Richard Mizerski. Primacy and recency effects on clicking behavior. Journal of computer-mediated communication , 11(2):522-535, 2006.
- T Chatfield. The trouble with big data? it's called the 'recency bias'. BBC. Retrieved April , 28:2023, 2016.
- Gloria Phillips-Wren, Daniel J Power, and Manuel Mora. Cognitive bias, decision styles, and risk attitudes in decision making and dss, 2019.
- Google Ads Attribution. Guide To Google Ads Attribution Models in 2025 -Is DataDriven Attribution The Future?, 2025. URL https://www.datafeedwatch.com/blog/ google-ads-attribution-models . Accessed: 2025-05-11.
- Nicolo Cesa-Bianchi, Claudio Gentile, and Yishay Mansour. Regret minimization for reserve prices in second-price auctions. IEEE Transactions on Information Theory , 61(1):549-564, 2014.
- Wei Zhang, Yanjun Han, Zhengyuan Zhou, Aaron Flores, and Tsachy Weissman. Leveraging the hints: Adaptive bidding in repeated first-price auctions. Advances in Neural Information Processing Systems , 35:21329-21341, 2022.
- Google Developers. Authorized buyers real-time bidding (rtb) protocol guide, 2024. URL https: //developers.google.com/authorized-buyers/rtb/openrtb-guide . Accessed: 202401-30.
- Jean-Jacques Laffont, Herve Ossard, and Quang Vuong. Econometrics of first-price auctions. Econometrica: Journal of the Econometric Society , pages 953-980, 1995.

- Robert Wilson. Sequential equilibria of asymmetric ascending auctions: The case of log-normal distributions. Economic Theory , 12:433-440, 1998.
- Eric Smith, J Doyne Farmer, László Gillemot, and Supriya Krishnamurthy. Statistical theory of the continuous double auction. Quantitative finance , 3(6):481, 2003.
- Martin Skitmore. First and second price independent values sealed bid procurement auctions: some scalar equilibrium results. Construction Management and Economics , 26(8):787-803, 2008.
- Pablo Ballesteros-Pérez and Martin Skitmore. On the distribution of bids for construction contract auctions. Construction Management and Economics , 35(3):106-121, 2017.
- David J Cooper and Hanming Fang. Understanding overbidding in second price auctions: An experimental study. The Economic Journal , 118(532):1572-1595, 2008.
- Haoyu Zhao and Wei Chen. Online second price auction with semi-bandit feedback under the non-stationary setting. arXiv preprint arXiv:1911.05949 , 2019.
- Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems , 24, 2011.
- Bo Xue, Yimu Wang, Yuanyu Wan, Jinfeng Yi, and Lijun Zhang. Efficient algorithms for generalized linear bandits with heavy-tailed rewards. Advances in Neural Information Processing Systems , 36, 2024.
- C Radhakrishna Rao. Information and the accuracy attainable in the estimation of statistical parameters. In Breakthroughs in Statistics: Foundations and basic theory , pages 235-247. Springer, 1992.
- Aditya Modi and Ambuj Tewari. No-regret exploration in contextual reinforcement learning. In Conference on Uncertainty in Artificial Intelligence , pages 829-838. PMLR, 2020.
- Orin Levy and Yishay Mansour. Learning efficiently function approximation for contextual mdp, 2022. URL https://arxiv.org/abs/2203.00995 .
- Orin Levy, Alon Cohen, Asaf Cassel, and Yishay Mansour. Efficient rate optimal regret for adversarial contextual MDPs using online function approximation. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 19287-19314. PMLR, 23-29 Jul 2023. URL https://proceedings. mlr.press/v202/levy23a.html .
- Michael Kearns and Satinder Singh. Near-optimal reinforcement learning in polynomial time. Machine learning , 49:209-232, 2002.
- Ashwinkumar Badanidiyuru, Zhe Feng, and Guru Guruganesh. Learning to bid in contextual first price auctions, 2021.
- Yiling Jia, Weitong Zhang, Dongruo Zhou, Quanquan Gu, and Hongning Wang. Learning neural contextual bandits through perturbed rewards. arXiv preprint arXiv:2201.09910 , 2022.
- Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on learning theory , pages 2137-2143. PMLR, 2020.
- Jiafan He, Dongruo Zhou, Tong Zhang, and Quanquan Gu. Nearly optimal algorithms for linear contextual bandits with adversarial corruptions. Advances in neural information processing systems , 35:34614-34625, 2022.
- Alekh Agarwal, Nan Jiang, Sham M Kakade, and Wen Sun. Reinforcement learning: Theory and algorithms. CS Dept., UW Seattle, Seattle, WA, USA, Tech. Rep , 32:96, 2019.
- Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint , volume 48. Cambridge university press, 2019.
- Botao Hao, Tor Lattimore, Csaba Szepesvári, and Mengdi Wang. Online sparse reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 316-324. PMLR, 2021.

## Appendix to 'Learning Personalized Ad Impact via Contextual Reinforcement Learning under Delayed Rewards'

## A Additional Discussion

## A.1 Related Work

Motivated by online advertising auctions, Weed et al. [2016] study repeated Vickrey auctions where goods with unknown values ( v t ) are sold sequentially. Bidders receive (potentially noisy) feedback about a good's value only after purchasing it-analogous to observing product conversions after displaying an ad. They formulate this problem as online learning with bandit feedback and propose a minimax-optimal bidding strategy for bidding ( b t ). However, a key limitation of their model is the neglect of long-term advertising effects: winning a bid at time t may influence future observations and outcomes, such as conversions at time t +1 . Building on this line of work, Feng et al. [2018] examines learning to bid without knowing one's value in a similar bandit setting and develops the WIN-EXP algorithm, which achieves sublinear regret ( O ( √ T | O | log( B )) ). Yet, like the earlier study, it assumes that the effect of winning a bid is instantaneous, ignoring possible intertemporal dependencies.

In a related direction, Han et al. [2024] investigate online learning in repeated first-price auctions, where a bidder observes only the winning bid after each auction and must adaptively learn to maximize cumulative payoff. Facing censored feedback-since winning bidders cannot observe their competitors' bids-they design the first algorithm achieving near-optimal ( O ( √ T ) ) regret by exploiting structural properties of first-price auctions, including their feedback and payoff functions. They formulate the problem as partially ordered contextual bandits, incorporating graph feedback across actions, cross-learning across contexts, and a partial order over private values. Nonetheless, their framework similarly assumes independence between the value process ( v t ) and past bidding outcomes, thereby excluding long-term effects of winning bids.

Extending to contextual settings, Zhang and Luo [2024] study online learning in contextual payper-click auctions. At each of T rounds, the learner observes contextual information and a set of candidate ads, estimates their click-through rates (CTRs), and conducts a second-price pay-per-click auction. The objective is to minimize regret relative to an oracle that makes perfect CTR predictions. They show that a √ T -regret rate is achievable (albeit with computational inefficiency) and provably optimal, as the problem reduces to the classical multi-armed bandit setting.

While these studies advance our understanding of strategic bidding under uncertainty, they share a common limitation: none account for the reinforcement or long-term impact of successful ad displays. This omission contrasts with empirical evidence such as De Haan et al. [2016], who demonstrate that online advertisements exert persistent effects on purchase conversion in a multi-channel attribution framework-though their dataset is not publicly available.

More recently, Badanidiyuru et al. [2021] address this gap by modeling the long-term causal impact of ad impressions using a Markov Decision Process (MDP) with mixed and delayed Poisson rewards. However, their approach assumes homogeneous treatment effects across users, overlooking the crucial role of personalization in determining advertising effectiveness. To address these limitations, our work jointly models the long-term, reinforcing, and personalized effects of advertisements, hoping to bridge the gap between sequential decision-making and individualized ad impact estimation.

## A.2 On the Potential to Extend the Linearity Assumption

Briefly recap of our model (Assumption 2.2), if we won the bid o t h = 1 , the average product conversion rate µ t h = h S t h, 1 ( x t ) , where x t is the received feature of customer, and S t h, 1 is the state which captures the time elapsed since most recent ads impression, h S t h, 1 is the state dependent neural network which we want to estimate. If we lose the bid o t h = 0 , the average product conversion rate µ t h = d S t h, 1 h S t h, 2 ( x t ) , where d S t h, 1 is the delayed impact, and S t h, 2 is the time interval between the two most recent impressions (Def 2.1).

To incorporate the estimation of the state-dependent neural network h l (analogous to θ l ) into our algorithm, we could do follows. Everything in Algorithm 1 remains unchanged, i.e. the exploration, exploitation, data-splitting strategy, and estimation, exception the following modifications.

- First, we replace the estimation of θ l (Eqn. (4)) by estimating the neural network h l by the following procedure (similar to Algorithm 1 in Jia et al. [2022]).
- -We approximate h ( x ) by f ( x, θ ) = √ mW L ϕ ( W L -1 ϕ ( . . . ϕ ( W 1 x ))) , where m is the network width, and ϕ ( x ) = ReLU ( x ) , L is the neural network depth.
- -We initialize θ 0 = ( vec ( W 1 ) . . . vec ( W L )) by random sample entry from N (0 , 4 /m ) and do online updating by gradient descent with perturbed reward using observed product conversion y t h by follows
* Generated observation perturbation γ t s ∈ [ t ] ∼ Poi ( λ ) , then generate binomial random variable I s ∈ [ t ] to add or subtract the noise from the observation

<!-- formula-not-decoded -->

- -Then based on this first stage estimates of h , plugging in to estimate delayed impact factor d by ∑ t s =1 ∑ h ∈ D s,l y s h ∑ t s =1 ∑ h ∈ D f S s ( x s , ˆ θ ) (analogous to Eqn. (3))

<!-- formula-not-decoded -->

- Since we do not have theoretical guarantees for h estimation (no confidence interval), we cannot do upper confidence bound (Line 10 of our Algorithm 1). Instead, we might act greedily based on our point estimate. Even though Algorithm in Jia et al. [2022] has theoretical guarantees, the key difference is that they assume noise of their observation is sub-Gaussian, while in our case, it is not realistic to assume that the observation y t h has sub-Gaussian noise, since product conversion has been known to have heavy tail for a long period of time. It is still a challenging open problem of developing theoretical-guaranteed neural contextual bandits with heavy tail observations.

However, our proposed framework leads to near-optimal results if we can parameterize h l ( x t ) = θ ⊤ l ϕ ( x t ) , where ϕ can be the neural net based embedding or other known feature mapping, then our algorithm remains near optimal, satisfying √ T regret bound in this case. This construction is very common in academia and industry as we mentioned from foundational studies such as Jin et al. [2020] to the recent ones [He et al., 2022]. In addition, in real world, companies might construct these feature mapping using complicated methods but retain the overall linear structure for the ease of scalability and interpretability. For example, the expected conversion rate could be a linear function of the expected webpage stay time and the predicted total spending but the expected webpage stay time and the predicted total spending are deep neural networks of the observed customer feature x t .

To summarize, our proposed framework leads to near-optimal results if there exists an online estimation oracle with theoretical guarantee for the estimation of h and the estimation oracle employed in our paper based on our knowledge is state-of-the-art. Our algorithm is possible to extend to a neural network if we drop the theoretical concerns. Developing theoretical guarantee for neural bandits under heavy tail observation could be a promising future direction.

## A.3 Flexibility in State Space Modeling

If we believe that all past ad exposures contribute to customer behavior, we can enrich the state variable to S t h = [ S t h, 1 , S t h, 2 , n t h ] , where n t h denotes the total number of ads the user has seen prior to round h . Under this formulation, the expected conversion from showing an ad at round h is given by ⟨ θ S t h, 1 ,n t h , x t ⟩ , with other modeling assumptions unchanged. In other words, the effect of the current ad depends not only on the time since the last impression, h -G t h, 1 (as shown in Figure 1), but also on the cumulative number of ad exposures the user has experienced so far.

Accordingly, Assumption 2.2 as follows:

1. When o t h = 0 , no ad is shown at round h , and the effect of recent ads at G t h, 1 carries over, given by ⟨ θ S t h, 2 , ( n t h -1) ∨ 0 , x t ⟩ d S t h, 1 , where a ∨ b denotes max( a, b ) .

2. When o t h = 1 , an ad is shown at round h , and the impact of the current ad is modeled as ⟨ θ S t h, 1 ,n t h , x t ⟩ , influenced by all the past ads exposure history.

We emphasize that, our core algorithmic design, including the data-splitting strategy, the proposed two-stage estimator, and the exploration-exploitation algorithm, readily extends to this new model with the enriched state representation. In addition, our theoretical results continue to hold under this new model, with a new optimal regret bound of order ˜ O ( dH 3 √ T ) , in comparison to ˜ O ( dH 2 √ T ) under the previous model. We provide supporting evidence below.

Now the enriched states is S t h = [ S t h, 1 , S t h, 2 , n t h ] . In terms of state transition, if o t h = 1 , S t h +1 = [1 , S t h, 1 , n t h +1] ; else, S t h +1 = [ S t h, 1 +1 , S t h, 2 , n t h ] . Θ now becomes { θ l,n } l,n ∈H ∪ { d l } l ∈H 1 with H = { ( l, n ) : l ∈ H 1 ∪H 2 , n &lt; l } , H 1 = {∞ , 1 , 2 , . . . , H -1 } and H 2 = {-∞ , 1 , 2 , . . . , H -2 } .

For data spiting strategy in Def 3.1, D t,l does not change and W t,l will extend to W t,l,n = { h | S t h, 1 = l, n t h = n, o t h = 1 } , ∀ l ∈ [ ∞ , 1 , 2 , . . . , H ] , W t, -∞ ,n = { h | S t h, 1 = -∞ , n t h = n, o t h = 0 } . Also, W t l will extend to W t l,n = { W s,l,n } t s =1 . The estimation of θ l,n using observation in W t l,n still follows online newton estimator with Eqn. (4). The TS-MLE now becomes

<!-- formula-not-decoded -->

Plugging in these new estimators in to the regret decomposition (Eqn. ((9)), Term (i) and Term (iv) does not change since new states did not affect transition error. For the estimation error, Term (iii), we just need an additional for loop to account the effect of n t h , which makes the new regret upper bound now ˜ O ( dH 3 √ T ) .

The above example suggests that our state formulation is very flexible and readily accommodates an enlarged state space. It supports a broad class of CMDP formulations for θ S , where S encodes domain knowledge about ad effects, and remains compatible with our proposed learning algorithm.

## A.4 Illustrative Example

Figure 2: Illustration of Learner-Customer Interaction in CMDP

<!-- image -->

Figure 2 presents a simple example of a customer, with contextual x t and no prior ad exposure, interacting with the learner L over H = 7 rounds. At the start of each round, L observes the state S t h , determines the bid amount a t h = π t ( S t h , x t ) based on policy π t , observes HOB m t h , and receives the bidding outcome o t h along with the reward R t h ( S t h , a t h , x t ) . At the end of each round, L observes the total product conversion y t h over the current round. As shown in Figure 2, the customer's first ad impression occurs at h = 3 , resulting in S t 1 = S t 2 = S t 3 = [ -∞ , ∞ ] and µ t 1 = µ t 2 = ⟨ θ -∞ , x t ⟩ ( µ t h defined in Def.2.2). The first ads exposure updates the expected conversion to µ t 3 = ⟨ θ ∞ , x t ⟩ , capturing the immediate impact of ad exposure. The second ad impression occurs at h = 6 . Between h = 4 and h = 6 , the state evolves as S t 4 = [1 , ∞ ] , S t 5 = [2 , ∞ ] , S t 6 = [3 , ∞ ] , with expected conversions µ t 4 = d 1 ⟨ θ ∞ , x t ⟩ and µ t 5 = d 2 ⟨ θ ∞ , x t ⟩ , reflecting the delayed and long-term impact of the ad displayed at h = 3 . µ t 6 = ⟨ θ 3 , x t ⟩ , capturing the effect of repeated ad exposure. Finally, at h = 7 , S t 7 = [1 , 3] and µ t 7 = d 1 ⟨ θ 3 , x t ⟩ , demonstrating the long-term effects of the second ad impression.

## A.5 Computational Complexity

For each episode of Algorithm 1, the computational complexities for updating ˆ θ t l , ˆ β t l , and ˆ d t l are O ( d 2 ) [Xue et al., 2024], O ( d 2 ) , and O ( d ) , respectively, for each l ∈ H . Notably, for a fixed policy π and transition F x t , R ( π, Θ , F x t ) is nondecreasing in d l and θ l , simplifying the computation of max ˜ Θ ∈C t -1 R ( π ; ˜ Θ , F x t t -1 ) . The corresponding maximizer are ˜ d t l =

<!-- formula-not-decoded -->

linear optimization, where ˆ θ t l is from Eqn. (4) and γ is from Lemma 3.3. The greedy policy π is then computed via dynamic programming with time complexity Poly ( H, |S| , B A /ϵ ) when discretizing the bidding space for an ϵ -optimal solution [Agarwal et al., 2019].

## B Details of Omitted Algorithm

In this section, we detail the algorithm used to estimate θ l for l ∈ H . Originally introduced by Xue et al. [2024], the Confidence Region with Truncated Mean (CRTM) algorithm (Algorithm 3) serves as an efficient estimator for generalized bandits with heavy-tailed rewards. In their framework, the stochastic observations y t follow a generalized linear model:

<!-- formula-not-decoded -->

where θ l represents the inherent parameters, τ &gt; 0 is a known scale parameter, and g ( · ) and h ( · ) are normalizers. The conditional expectation of y t is given by:

<!-- formula-not-decoded -->

where m ′ ( · ) is the link function, denoted as µ ( · ) = m ′ ( · ) . Thus, y t can be expressed as:

<!-- formula-not-decoded -->

where η t is a random noise term satisfying E ( η t |G t -1 ) = 0 . Here, G t -1 = { x 1 , y 1 , . . . , x t -1 , y t -1 , x t } denotes the σ -filtration up to time t -1 . CRTM assumes that the link function µ ( · ) is L -Lipschitz, continuously differentiable, and has a first derivative µ ′ ( · ) bounded below by a positive constant κ , i.e., µ ′ ( z ) ≥ κ . Additionally, CRTM assumes that the observations y t have bounded moments. Specifically, there exist positive constants u and 0 &lt; ϵ ≤ 1 such that:

<!-- formula-not-decoded -->

In our setting, where y t follows a Poisson distribution, we have ϵ = 1 , u = B x B θ (1 + B x B θ ) , µ ( · ) as a linear function, and κ = 1 . Based on the specific parameters in our setting, the CRTM algorithm (Algorithm 3) is adapted as follows:

Algorithm 3 Confidence Region with Truncated Mean [Xue et al., 2024]

<!-- formula-not-decoded -->

- 2: Truncate the observed payoff: ˜ y t h = y t h I ∥ x t ∥ ( V t l ) -1 | y t h |≤ Γ
- 1: for y t h ∈ W t,l do
- 3: Compute the gradient: ∇ ˜ l t ( ˆ θ t -1 l ) = ( -˜ y t h + x ⊤ t ˆ θ t -1 l ) x t
- 5: Update the estimator:
- 4: Update V t l = V t -1 l + 1 2 x t x ⊤ t

<!-- formula-not-decoded -->

- 6: Construct the confidence region:

<!-- formula-not-decoded -->

- 7: end for
- 8: return ( ˆ θ t l , C t l )

In particular, the truncation threshold Γ is defined by Eqn. (10).

<!-- formula-not-decoded -->

Moreover, the bound for the weighted estimation error, ∥ θ l -ˆ θ t l ∥ 2 V t l , denoted by γ , is given by:

<!-- formula-not-decoded -->

Running Algorithm 3 leads to the following lemma.

Lemma (Lemma 3.3 restated) . Given l ∈ H , with probability at least 1 -δ , ˆ θ t l defined in Eqn. (4) satisfies ∥ θ l -ˆ θ t l ∥ 2 V t l ≤ γ, ∀ t ≥ 0 .

## C Proof of Theorem 4.2

Theorem (Theorem 4.2 restated) . For any δ ≥ 6 T 3 , with probability at least 1 -δ , Reg T incurred by Algorithm 1 is O ( dH 2 √ T log( TH δ ) log(1 + T 2 d )) .

Proof. In this section, we present the detailed proof of Theorem 4.2. We begin by decomposing Reg T (defined in Eqn. (1)) into four terms-Term (i), Term (ii), Term (iii), and Term (iv)-as shown

in Eqn. (9). Θ(log T ) comes from regret incurred in the exploration phase with τ = Hn l +1 .

<!-- formula-not-decoded -->

Term (iv)

At a high level, Terms (i) and (iv) account for the cumulative regret from transition error, i.e., the discrepancy between F x t and ˆ F x t . Term (iii) represents the decision error arising from imprecise estimates of the model parameters Θ = { θ l } l ∈H ∪{ d l } l ∈ [ H ] . Thus, more accurate estimators of F x t and Θ yield lower cumulative regret. Meanwhile, Term (ii) measures the gap in regret between a chosen point and the optimal point within a feasible set. Our goal is to show Θ ∈ C t for all t ≥ τ (after the exploration phase) with high probability, ensuring that Term (ii) can be negative with high probability. This is proved in Lemma 4.5. By Lemma 4.3, with probability at least 1 -1 T 4 , Term (i) in Eqn. (9) is bounded by ˜ O ( dH 2 √ T ) . A similar argument implies that Term (iv) is also bounded by ˜ O ( dH 2 √ T ) with probability at least 1 -1 T 4 . Further, Lemma 4.4 shows that, with probability at least 1 -2 T 3 , Term (iii) is bounded by ˜ O ( H 2 √ dT ) . Combining these results, we conclude that, with probability at least 1 -6 T 3 , the overall regret Reg T is ˜ O ( dH 2 √ T ) .

## C.1 Proof for Theorem 4.1

Theorem (Theorem 4.1 restated) . Let δ ≥ 1 T 4 H and N t,l ≥ n l . With probability at least 1 -δ , the estimation error ∣ ∣ ˆ d t l -d l ∣ ∣ , with ˆ d t l defined in Eqn. (3), is bounded by:

<!-- formula-not-decoded -->

γ is as defined in Lemma 3.3 and n l defined in Remark 3.2.

Proof. The key idea in estimating d l is to isolate observations that are 'purified' with respect to d l . Define

<!-- formula-not-decoded -->

which collects the episodes for user t where: The second-to-last bid winning is l episodes before the most recent bid winning ( S t h, 1 = l ); The current bid is lost ( o t h = 0 ), meaning no new advertisement is shown. Under these conditions, the product-conversion observations { y t h } h ∈ D t,l satisfy

<!-- formula-not-decoded -->

Because these observations are purified within the same l that defines d l , they provide reliable information for estimating the long-term advertising impact parameter d l . To estimate the long-term advertising impact parameter d l , we adopt a two-stage procedure. First, we construct the estimates

ˆ θ s S s h, 2 for all s ∈ [ t ] . Then, we substitute these estimates into the negative log-likelihood for d l (see Eqn. (11)), yielding our two-stage estimator for d l .

<!-- formula-not-decoded -->

Taking the derivative of the negative log-likelihood with respect to d l yields:

<!-- formula-not-decoded -->

This expression guides the next step of solving for d l . By solving Eqn. (12), we obtain the estimator ˆ d t l for d l as follows:

<!-- formula-not-decoded -->

Here, ˆ θ s S s h, 2 denotes the updated estimate of the advertisement's impact for user s . Next, we bound the estimation error of ˆ d t l . We express y s h as

<!-- formula-not-decoded -->

where η s h is sub-exponential with parameters ( ν 2 , α ) = ( ed l x ⊤ s θ S s h, 2 , 1 ) (see Lemma D.1). Based on this formulation, the estimation error ∣ ∣ ˆ d t l -d l ∣ ∣ can be decomposed as follows.

<!-- formula-not-decoded -->

To upper bound the estimation error (see Eqn. (13)), it suffices to bound Term (i) and Term (ii) in Eqn. (13) respectively. We can use sub-exponential concentration bound for Term (i), shown as follows.

<!-- formula-not-decoded -->

We derive Eqn. (14) by applying the tail bounds for sub-exponential random variables (Lemmas D.4, D.5, and D.6) and setting

<!-- formula-not-decoded -->

We also use the accelerated convergence rate for sub-exponential variables under the conditions N t,l &gt; 32 log( HT ) e B d B x B θ b 2 and δ ≥ 1 2 HT 4 , which ensure 0 ≤ ϵ ≤ e B d B x B θ . Therefore, with probability at least 1 -δ , we have Term (i) ≤ √ 2 e B d B x B θ b 2 N t,l log ( 2 δ ) . Then it remains to bound Term (ii) in Eqn. (13). Define the dataset F l s,l ′ := { h | S s h, 1 = l, S s h, 2 = l ′ , o s h = 0 } . Observe that

<!-- formula-not-decoded -->

meaning F l s,l ′ partitions D s,l according to S s h, 2 , the second-to-last winning state. Let n s,l ′ := | F l s,l ′ | . Then

<!-- formula-not-decoded -->

Since ∥ θ l ∥ 2 ≤ B θ for all l , and ∥ x s ∥ 2 ≤ B x for each s ∈ [ T ] , define

<!-- formula-not-decoded -->

where l ′ = S s h, 2 . For convenience, let √ γ s l ′ = ∥ ˆ θ s S s h, 2 -θ S s h, 2 ∥ V s l ′ . Then, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Therefore, with probability at least 1 -δ , δ ≥ 1 HT 4 , we have

4

H

|

ˆ

d

t

l

-

d

l

| ≤

## C.2 Proof of Lemma 4.3

Lemma (Lemma 4.3 restated) . For δ ≥ 1 /T 4 , with probability at least 1 -δ , Term (i) in Eqn. (9) is bounded above by O ( dH 2 √ T √ log((1 + T ) /δ ) log(1 + T/d )) .

Proof. In essence, the proof of Lemma 4.3 relies on two main steps. First, we show that Term (i) can be bounded by the cumulative estimation error in transition, expressed as

<!-- formula-not-decoded -->

Second, we demonstrate that, when ˆ F t -1 h ( b, x t ) is estimated using Algorithm 1, this cumulative estimation error remains small. Combined, these two steps yield the desired upper bound on Term (i).

## Step 1: Upper bound Term (i) by the cumulative estimation error in transition.

Term (i) captures the regret induced by transition error, as shown in Eqn. (16). The final inequality in Eqn. (16) holds because R ( π ∗ t ; Θ , ˆ F x t t -1 ) -OPT ( Θ , x t , ˆ F x t t -1 ) ≤ 0 , due to the definition of OPT ( Θ , x t , ˆ F x t t -1 ) . We then apply the Simulation Lemma (Lemma C.1) to bound R ( π ∗ t ; Θ , F x t ) -R ( π ∗ t ; Θ , ˆ F x t t -1 ) .

<!-- formula-not-decoded -->

Lemma C.1 (Simulation Lemma [Kearns and Singh, 2002]) . For all s ∈ S and a ∈ A , if ∑ s ′ ∈S ∣ ∣ ∣ P ( s ′ | s, a ) -P ′ ( s ′ | s, a ) ∣ ∣ ∣ ≤ ϵ 1 and ∀ h ∈ [ H ] , ∣ ∣ r h ( s, a ) -r ′ h ( s, a ) ∣ ∣ ≤ ϵ 2 , then

<!-- formula-not-decoded -->

To find ϵ 1 in Lemma C.1, we compare two MDPs:

M t , induced by policy π ∗ t , θ, and HOB distribution F x t , and

ˆ M t , induced by policy π ∗ t , θ, and HOB distribution ˆ F x t t -1 .

Given state s = ( s 1 , s 2 ) , we have:

<!-- formula-not-decoded -->

B

√

d

log

(

1 +

T

2

d

)

γ

+

b

√

2

e

B

d

d

B

x

B

θ

log(2

/δ

)

√

1

N

t,l

.

Define

<!-- formula-not-decoded -->

From Eqn. (17), it follows that the parameter ϵ 1 in Lemma C.1 can be taken as 2 ϵ t . In the following, we bound the difference between R t h and R t ′ h to find ϵ 2 in Lemma C.1. By definition, R t h ( S t h , A t h , x t ) = d S t h, 1 ⟨ θ S t h, 2 , x t ⟩ ( 1 -F h ( A t h , x t ) ) + ( ⟨ θ S t h, 1 , x t ⟩ -p h ( A t h , x t ) ) F h ( A t h , x t ) . Here, p h ( A t h , x t ) is the second highest price conditioned on winning. Since p x t h ( b ) = b -1 F h ( b, x t ) ∫ b 0 F h ( v, x t ) dv , we obtain ˆ p h ( A t h , x t ) ˆ F t -1 h ( A t h , x t ) -p h ( A t h , x t ) F h ( A t h , x t ) = ∫ A t h 0 F h ( v, x t ) dv -∫ A t h 0 ˆ F t -1 h ( v, x t ) dv . In addition, we can show

<!-- formula-not-decoded -->

Therefore, we can bound the reward difference | R t h ( s, a ) -R t ′ h ( s, a ) | by Eqn. (18).

<!-- formula-not-decoded -->

Combining Eqn. (17) and Eqn. (18), Lemma C.1 implies

<!-- formula-not-decoded -->

Therefore, by Eqn. (19), bounding Term (i) reduces to bounding

<!-- formula-not-decoded -->

## Step 2: Construct high probability bound for cumulative estimation error in transition

The next step is to derive a high-probability bound for ∑ T t = τ ϵ t . By Assumption 2.6, the Highest Other Bids (HOB) distribution satisfies log( m t h ) ∼ N ( ⟨ x t , β h ⟩ , σ 2 h ) , suggesting that the transition estimation error takes the form

<!-- formula-not-decoded -->

Using a first-order Taylor expansion of the standard normal CDF Φ( · ) , we obtain ∣ ∣ Φ( a ) -Φ( a + ∆) ∣ ∣ ≤ | ∆ | ϕ ( a ) , ϕ is the standard normal pdf. This inequality allows us to further bound the estimation error by ∣ ∣ ∣ ˆ F t -1 h ( b, x t ) -F h ( b, x t ) ∣ ∣ ∣ ≤ | ∆ t h | ϕ ( log( b ) -⟨ x t , ˆ β t -1 h ⟩ ˆ σ t -1 h ) , where | ∆ t h | is defined

and bounded by follows.

<!-- formula-not-decoded -->

Note that we estimate the variance σ 2 h using second-stage empirical estimates. Concretely,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To complete the argument, it suffices to show that these two quantities are close with high probability.

<!-- formula-not-decoded -->

Using Eqns. (20), (21), and (6), we derive the following equations:

<!-- formula-not-decoded -->

Meanwhile, by definition,

where C h = σ h +ˆ σ t -1 h ( σ h ˆ σ t -1 h ) 3 ≤ 2¯ σ σ 6 , by Assumption 2.8. In what follows, we bound each term one by one. To bound Term A, we show that for all h , with probability at least 1 -δ , the following holds:

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

This shows that Term C is on the order of O (log T ) . For Term B, a similar argument implies that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Therefore, we conclude the following bound.

<!-- formula-not-decoded -->

It remains to bound Term D. First, observe that

<!-- formula-not-decoded -->

Since χ 2 1 ∼ SubE (4 , 4) , by Lemma C.2 and a union bound argument, we can show that, with probability at least 1 -δ , Term D is bounded above by

<!-- formula-not-decoded -->

The O ( log( T ) ) term arises from 'burning in' the first 64 log( T ) which guarantees δ ≥ 1 / 4 T 4 and ensure quadratic decay of sub-exponential bound.

Lemma C.2 (Example 2.11 in Wainwright [2019]) . A chi-squared ( χ 2 ) random variable with n degrees of freedom, denoted by Y ∼ χ 2 n , can be represented as Y = ∑ n k =1 Z 2 k , where Z k ∼ N (0 , 1) are i.i.d. variables. Since each Z 2 k is sub-exponential with parameters ( ν 2 , α ) = (2 , 4) , we have P (∣ ∣ ∣ 1 n ∑ n k =1 Z 2 k -1 ∣ ∣ ∣ ≥ √ 8 n log 2 δ ) ≤ δ, as long as δ ≥ 2 exp ( -n 8 ) .

Bringing all these results together and applying union bound, we conclude that there exists a constant C , depending on B x , B θ , B β , ¯ σ, σ, B d , and B A , such that with probability at least 1 -δ (where δ ≥ 1 T 4 ), we have

<!-- formula-not-decoded -->

By choosing λ = 1 , we proves Lemma 4.3, we states that for δ ≥ 1 /T 4 , with probability at least 1 -δ , Term (i) in Eqn. (9) is bounded above by O ( dH 2 √ T √ log((1 + T ) /δ ) log(1 + T/d )) .

## C.3 Proof for Lemma 4.4

Lemma (Lemma 4.4 restated) . With probability at least 1 -2 δ , where δ ≥ 1 T 3 , Term (iii) in Eqn. (9) is bounded by O ( H 2 √ dT log ( 4 T H δ ) log ( 1 + T 2 d ) ) .

Proof. Term (iii) reflects the portion of regret attributable to inaccuracies in estimating the true parameter Θ . Let ˜ Θ t -1 := arg max Θ ∈C t -1 R ( π t ; x t , Θ , ˆ F x t t -1 ) . By definition,

<!-- formula-not-decoded -->

By Eqn. (23), we demonstrate that Term (iii) in (9) can be split into two parts corresponding to the cumulative estimation error of ˆ θ t l (see Terms 1 and 2 in (23)) and the cumulative estimation error of ˆ d l (see Term 3 in (23)). Recall that ˆ θ t l is a variant of the online Newton estimator (see Algorithm 3), defined by

<!-- formula-not-decoded -->

where ∇ ˜ l t ( ˆ θ t -1 l ) = ( -˜ y t h + x ⊤ t ˆ θ t -1 l ) x t and V t l = V t -1 l + 1 2 x t x ⊤ t . ˜ y t h is the truncated observation of y t h . Because y t h follows a Poisson distribution and is linear in θ l , it is a special case of a heavytailed distribution with bounded second moment. Consequently, Lemma 3.3 can be used to construct a high-confidence bound on ˆ θ t l , which in turn controls Terms 1 and 2 in Eqn. (23).

<!-- formula-not-decoded -->

To bound Term 1 in Eqn. (23), we first introduce n l t , where

<!-- formula-not-decoded -->

Then we have the following inequality:

<!-- formula-not-decoded -->

Let us denote √ γ s l ′ := ∥ ∥ ˆ θ s S s h, 2 -θ S s h, 2 ∥ ∥ V s l ′ . We obtain Eqn. (24) by rearranging terms in the expression involving θ l with same l . We use the fact that n l t ≤ H to get Eqn. (25). Next, we apply the Cauchy-Schwarz inequality to derive Eqn. (26). By Lemma 3.3, we know that, with probability at least 1 -δ , γ s l ′ ≤ γ for all t ≥ 1 , with γ defined in Lemma 3.3. In addition, we use Lemma 11 of Abbasi-Yadkori et al. [2011] to bound ∑ T t =1 ∥ ∥ x t ∥ ∥ 2 ( V t l ) -1 , yielding

<!-- formula-not-decoded -->

Combining these results and applying the union bound gives us Eqn. (27). Similarly, by an analogous argument, we can show that, with probability at least 1 -δ , Term 2 in Eqn. (23) satisfies a similar bound, as shown below.

<!-- formula-not-decoded -->

The principal difficulty in designing Algorithm 1 lies in estimating ˆ d t l , which captures the long-term impact of advertising on product conversion. Since d l always appears alongside θ l ′ in the model y t h ∼ Poi ( d l θ ⊤ l ′ x t ) , the main technical challenge is establishing a high-probability bound on | ˆ d t l -d l | that leverages the estimation error of ˆ θ t l ′ . Theorem 4.1 summarizes our results in this direction. By Theorem 4.1, we know that given t ≥ Hn l , and l ∈ H 1 , with probability 1 -δ, δ ≥ 1 T 4 H , d l ∈ D t l , where

<!-- formula-not-decoded -->

Therefore, by applying the union bound to make results valid ∀ t ∈ [ T ] and t ≥ Hn l , ∀ l ∈ H 1 , we have that for δ ≥ 1 T 3 , we conclude that with probability at least 1 -δ , Term (3) in Eqn. (23) can be bounded above by

<!-- formula-not-decoded -->

Equation (28) follows by gathering all terms involving the same l of d l . Next, we obtain Equation (29) by noting that ∑ H l =1 N T,l = T H . Combining the above results, we conclude that, with probability at least 1 -2 δ (where δ ≥ 1 T 3 ), Term (iii) in Eqn. (9) is bounded by

<!-- formula-not-decoded -->

## C.4 Proof of Lemma 4.5

Lemma (Lemma 4.5 restated) . With probability at least 1 -δ with δ ≥ 2 T 3 , Θ ∈ C t , ∀ t ≥ τ .

Proof.

<!-- formula-not-decoded -->

By Lemma 3.3, we have

<!-- formula-not-decoded -->

Moreover, by Theorem 4.1, with δ ≥ 1 HT 4 we have

<!-- formula-not-decoded -->

Combining these results together, we have with probability at least 1 -δ with δ ≥ 2 T 3 , Θ ∈ C t , ∀ t .

## D Technical Lemmas

Lemma D.1 (Sub-Exponentiality of Poisson Random Variable) . If X ∼ Poi( µ ) , then the centered random variable X -µ is SubE( eµ, 1) . Equivalently, there exist constants ν 2 = eµ and α = 1 such that, for all | t | &lt; 1 /α = 1 , E [ e t ( X -µ ) ] ≤ exp ( ν 2 t 2 ) = exp ( e µ t 2 ) .

Proof. Below is the proof showing that a Poisson ( µ ) random variable is sub-exponential with parameters ( eµ, 1 ) . The proof hinges on bounding the moment-generating function (MGF) of the centered random variable X -µ . Recall that if X ∼ Poi( µ ) , its moment-generating function (MGF) is

<!-- formula-not-decoded -->

For the centered random variable X -µ ,

<!-- formula-not-decoded -->

We claim that, for all | t | ≤ 1 ,

Combining the two steps, for | t | ≤ 1 ,

<!-- formula-not-decoded -->

Therefore, X -µ satisfies

<!-- formula-not-decoded -->

Indeed, expanding e t in its Taylor series about t = 0 ,

<!-- formula-not-decoded -->

For | t | ≤ 1 , this sum of higher-order terms is bounded above by a constant times t 2 . A convenient choice is e , giving

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By definition, this means X -µ is SubE ( eµ, 1 )

.

Lemma D.2 (Theorem 2 in Abbasi-Yadkori et al. [2011]) . Assume the same in Theorem D.3, let Σ 0 = λ I d , λ &gt; 0 . Define y t = x ⊤ t β + η t , with η t defined in Lemma D.3, and assume that ∥ β ∥ 2 ≤ B β , ∥ x t ∥ 2 ≤ B x . Then, for any δ &gt; 0 , with probability at least 1 -δ , for all t &gt; 0 , β lies in the set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma D.3 (Theorem 1 in Abbasi-Yadkori et al. [2011]) . Let {F t } ∞ t =0 be a filtration. Let { η t } ∞ t =1 be a real-valued stochastic process such that η t is F t -measurable and η t is conditionally σ 2 -subGaussian for some σ 2 ≥ 0 . Let { x t } ∞ t =1 be an R d -valued stochastic process such that x t is F t -1 measurable. Assume that Σ 0 is a d × d positive definite matrix. For any t ≥ 0 , define Σ t = Σ t -1 + x t x ⊤ s and S t = ∑ t s =1 η s x s . Then for any δ &gt; 0 , with probability at least 1 -δ , for all t ≥ 0 , we have ∥ S t ∥ 2 Σ -1 ≤ 2 σ 4 log (√ det Σ t det Σ 0 /δ ) .

<!-- formula-not-decoded -->

Lemma D.4 (Sub-exponential tail bound) . Suppose X is sub-exponential with parameters ( ν 2 , α ) . Then

<!-- formula-not-decoded -->

Lemma D.5 (Lemma 3 in Hao et al. [2021]) . Consider a random variable X i ∼ SE ( ν 2 , α ) and β is a non-zero scalar, then βX i ∼ SE ( β 2 ν 2 , | β | α )

Lemma D.6 (Lemma 4 in Hao et al. [2021]) . Consider independent random variables X i ∼ SE ( ν 2 i , α i ) for i = 1 , . . . , n , then X = ∑ n i =1 X i follows SE (∑ n i =1 ν 2 i , max i α i ) .

## D.1 Proof of Fact 2.7

Proof. In this section, we prove that the tuple ( X , S , A , P x t , { R t h ( S t h , a t h , x t ) } H h =1 , s 1 , H ) constitutes a Contextual Markov decision process (CMDP) by demonstrating that it satisfies the Markov property.

If we lose the bid, then we have

<!-- formula-not-decoded -->

If we win the bid, then we have

<!-- formula-not-decoded -->

From the above equation, we notice that

<!-- formula-not-decoded -->

Therefore, the Markov property holds and the model is a CMDP.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: It summarizes the main contribution discussed at the end of Section 1.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Please refer to Section Discussion.

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

Justification: Please refer to Appendix.

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

Justification: We did not include experiments.

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

Justification: No real-world data is included in the current version.

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

Justification: Not applicable.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Not applicable.

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

Justification: Not applicable.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [NA]

Justification: Not applicable.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Please refer to Section 1

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

Answer: [NA]

Justification: Not applicable.

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

Justification: All theorems are developed by human not LLM.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.