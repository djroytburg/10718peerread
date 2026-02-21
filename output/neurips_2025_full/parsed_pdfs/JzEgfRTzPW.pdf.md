## Beyond Average Value Function in Precision Medicine: Maximum Probability-Driven Reinforcement Learning for Survival Analysis

Jianqi Feng

∗

Shandong University

202412070@mail.sdu.edu.cn

Xiaodong Yan †

Xi'an Jiaotong University yanxiaodong@xjtu.edu.cn

Chengchun Shi

LSE c.shi7@lse.ac.uk

Zhenke Wu University of Michigan zhenkewu@umich.edu

Wei Zhao † Shandong University wzhao92@sdu.edu.cn

## Abstract

Constructing multistage optimal decisions for alternating recurrent event data is critically important in medical and healthcare research. Current reinforcement learning (RL) algorithms have only been applied to time-to-event data, with the objective of maximizing expected survival time. However, alternating recurrent event data has a different structure, which motivates us to model the probability and frequency of event occurrences rather than a single terminal outcome. In this paper, we introduce an RL framework specifically designed for alternating recurrent event data. Our goal is to maximize the probability that the duration between consecutive events exceeds a clinically meaningful threshold. To achieve this, we identify a lower bound of this probability, which transforms the problem into maximizing a cumulative sum of log probabilities, thus enabling direct application of standard RL algorithms. We establish the theoretical properties of the resulting optimal policy and demonstrate through numerical experiments that our proposed algorithm yields a larger probability of that the time between events exceeds a critical threshold compared with existing state-of-the-art algorithms.

## 1 Introduction

Motivation . Precision medicine is a paradigm that aims to tailor treatments to individual patient characteristics. Recent studies have increasingly developed reinforcement learning (RL) algorithms in this context (see, e.g., 28; 33; 48; 52; 49; 53; 58; 35; 36; 31; 25; 50; 26; 4; 7; 56; 57). This paper focuses on RL for survival analysis. The existing literature has primarily studied time-to-event data, with the objective of maximizing expected survival time (see Section 1.1 for details). However, for patients with chronic diseases, data often involves alternating recurrent events, and the clinical goal shifts to maximizing the probability that the duration between events exceeds a critical threshold. In response, we propose an RL framework designed to maximize this probability. Below, we provide an example to illustrate the structure of alternating recurrent event data.

Our work is motivated by the ongoing Intern Health Study (IHS) which recruited first-year medical interns at US institutions to study factors that may impact their mental health and general well-being (29). The medical internship is an initial step towards trained and practicing physicians. They

∗ All authors contributed equally and are listed in alphabetical order by surname.

† Corresponding Author

often face difficult decisions, challenging shift work schedules, lack of time for exercise, and sleep disruptions, resulting in higher rates of depression (34). Wearable and smartphone devices provide daily ecological momentary assessments about physical activities, sleep, and mood scores over multiple months since the start of the internship. Digital health interventions sent via prompts on the study app serve as a low-touch and easily accessible approach to delivering data insights, tips, and other personalized content for mitigating mental health issues especially when access to therapists may not always be timely (19; 43). It has been an ongoing cohort that facilitates the study of genetic and lifestyle risk factors for mental and behavioral health outcomes. In this paper, we focus on the management of recurrent low-mood episodes, which requires a shift toward dynamic, personalized interventions. For example, we may define a low-mood episode by a significant drop (e.g., 20% ) from an individual's baseline mood before internship. Moving beyond prediction, RL offers a powerful computational framework to learn optimal, adaptive intervention policies. The objective is to train an agent to select sequences of actions that either preemptively steer an individual away from an impending low-mood state or, should an episode occur, expedite their return to baseline, thereby minimizing the episode's duration and severity.

Challenges . Applying RL to alternating recurrent event data raises two challenges:

1. Alternating recurrent event data is substantially more complex than the time-to-event data, the latter being the focus of the current literature (14; 47). Unlike time-to-event data, alternating recurrent events can switch back and forth multiple times during the follow-up period, which substantially increases the modeling complexity.
2. Most existing RL algorithms consider maximizing the expectation. However, in survival analysis, the primary interest often lies in the probability that the duration between recurrent events exceeds a specified threshold, a perspective that has received limited attention in the existing literature.

Contributions . To address these challenges, we propose a novel maximum-probability-driven RL algorithm tailored to alternating recurrent event data. Our primary methodological contribution is the derivation of a lower bound for the maximum-probability objective, which reformulates the original objective into a sum of cumulative log probabilities. This enables the application of standard RL algorithms designed to maximize cumulative reward in the context of recurrent event data. We further provide theoretical guarantees and conduct extensive numerical experiments to demonstrate the effectiveness of the proposed algorithm.

## 1.1 Related Work

Our paper is closely related to two strands of research in survival analysis: one focusing on the use of RL for survival analysis, and the other handling recurrent event data.

RL for survival analysis . RL has recently received considerable attention in the survival analysis literature with censored outcomes (11). Specifically, Goldberg &amp; Kosorok (14) and Liu et al. (23) construct a Q-function suitable for time-to-event censored data by inverse probability weighting. To address the issue of unequal numbers of individuals at each stage, they construct an auxiliary problem and provide an unbiased analysis of the optimal policy. Zhao et al. (54) offers a doubly robust estimation method based on this work. Liu et al. (22) uses the estimated Q-function values to fill in censored values, but no theoretical analysis is provided. Lee et al. (17) uses deep learning to learn the distribution of survival time for analysis. However, these methods are applied to time-to-event data and focus on maximizing expected value of survival time.

Recurrent event data . Given the recurrent nature of chronic diseases, the more recent trend of the existing literature has moved beyond the time-to-event data to handle recurrent event data. Among those available, Lee et al. (18) and Wang et al. (44) propose to use a counting process for modeling recurrent events and estimate the gap time between recurrent events. Xia et al. (45) and Loe et al. (24) propose to segment recurrent events within follow-up windows and estimate the probability of the duration of recurrent events considering all stages. However, these papers do not study policy optimization.

## 2 Data Structure and Optimization Objectives

In this section, we introduce the data structure of alternating recurrent events and our considered optimization objective. As mentioned earlier, unlike most existing studies in the literature, we focus on maximizing the probability that the duration time exceeds a pre-specified threshold.

Data structure . We consider a multistage decision-making process based on a censored recurrent event structure. Let G k ( k = 1 , 2 , . . . ) be the duration of the k -th follow-up stage. For each stage, let X k ∈ X and A k ∈ A respectively be the covariates collected and the treatment received at the beginning of the k -th stage, where X ⊂ R p is the set of states and A is a compactness set of available treatment options. The observed values of covariates X k and treatment A k are denoted as x k and a k , respectively. Suppose that each subject's follow-up period begins at time 0. We consider subjects who can experience two types of alternating events that recur in multiple follow-up stages (45). We denote L i as the i -th occurrence time from high mood to low mood (first-type event ) and H i as the i -th occurrence time from low mood to high mood (second-type event). Suppose that 0 ≤ L 1 &lt; H 1 &lt; L 2 &lt; H 2 &lt; · · · and L 0 = H 0 = 0 .

In this work, we are interested in the following recurrent-event duration T k for stage k , which represents the duration from the occurrence of a second-type event to the next first-type event. The formula for T k is given by

<!-- formula-not-decoded -->

where J k = arg max j ( H j ≤ ∑ k i =1 G i ) denotes the time of the last second-type recurrent event occurring at stage k . We assume that J 0 = 0 and ∑ b i = a · = 0 for b &lt; a . Since L j and H j are random variables, we assume here a reward sample space R such that T k ∈ R . Let C ∈ R + represent the censoring time measured from the beginning of the study until a censoring event occurs, such as the end of the study or the loss of follow-up, then the observed duration time at stage k is given by Y k = min { T k , C -∑ k -1 i =1 T i } and the corresponding censoring indicator at stage k is denoted by ∆ k = 1 ( ∑ k i =1 T i ≤ C ) . We assume that the censoring time is independent of covariates, treatments, and duration times for simplicity. Note that the censoring event can occur at any stage. Once the censoring event occurs at stage k , the treatments, covariates, and duration times after stage k are not observed. Let the number of stages of treatment received by a subject be ¯ K ∈ N + , where ¯ K = inf { k : ∆ k = 0 } . With notations above, the observed individual trajectories for one subject can be represented as (subject index omitted): { X 1 , A 1 , ∆ 1 , Y 1 , . . . , X ¯ K , A ¯ K , ∆ ¯ K , Y ¯ K } .

In Figure 1, we illustrate an individual trajectory from the motivating Intern Health Study. In this study, a treatment policy needs to be developed to determine whether to send weekly text messages to medical interns to maintain high mood levels for a longer duration. As illustrated, the shifts between high mood and low mood occur repeatedly. In this example, G k is consistently 7 (days) for any k , and T k of primary interest represents the duration of the high-mood period within each week. For this specific individual, a censoring event occurs in stage 3, resulting in ¯ K = 3 and Y 3 being observed instead of the true T 3 .

Figure 1: An example for the individual trajectory in the Intern Health Study.

<!-- image -->

Optimization objective . In traditional RL, the strategy is often formulated with the objective of the value function E P 0 ,π,P T [ ∑ ¯ K k = t T k | X t ] , where P 0 is the transition kernel that maps state space X and action space A to a distribution over state space X , π is a policy that maps state space X to a probability distribution over the action space A , and P T is the distribution on R under condition of X and A . Moreover, in health studies, we place more emphasis on the probability that the duration time is greater than a threshold at each stage. Therefore, we propose the following objective:

<!-- formula-not-decoded -->

where α k ∈ [0 , 1] , for k = 1 , . . . , ¯ K . In the context of the Intern Health Study illustrated in Figure 1, setting α k = 4 / 7 for all k ≤ ¯ K implies that we aim for interns to maintain a high mood for at least four days per week throughout the follow-up period. However, if we expect interns to have a better mood in the first few weeks, α k 's can be increased slightly during those weeks. See Appendix A.3 for further discussion of α k selection.

To handle (1), we first impose two assumptions (Assumption 2.1,2.2). These assumptions allow us to decompose (1) into a product of stage-specific survival functions, resulting in a simplified objective function (see Lemma 2.3).

Assumption 2.1 (Markov assumption) . For any v ≥ 0 and k &gt; t , we assume that

<!-- formula-not-decoded -->

Assumption 2.2 (Conditional independence assumption) . For any k &gt; t , we assume that ( X k , A k ) and { T s } k -1 s = t are conditional independent given { ( X s , A s ) } k -1 s = t .

Lemma 2.3. When Assumptions 2.1 and 2.2 hold, we have

<!-- formula-not-decoded -->

where S T k ( ·| X k , A k ) is the conditional survival function of the duration time T k given X k and A k .

The proof of Lemma 2.3 is provided in Appendix B.1. However, directly optimizing P π ( x t ) is challenging because evaluating the objective requires computing the product of probabilities from k = t to ¯ K , which can become extremely small when ¯ K is large, making the resulting optimization unstable. To address this, we observe that, by Jensen's inequality,

<!-- formula-not-decoded -->

where S T k is a shorthand for S T k ( α k G k | X k , A k ) and we denote the lower bound of log P π ( x t ) by V π 0 ( x t ) . (3) motivates us to maximize this lower bound of log P π , which equals a sum of expected cumulative log probabilities and can thus be optimized using existing RL frameworks. A natural question arises: does this shift in the objective alter the optimal policy? Theorem 2.4 below answer this question, showing that the two objectives induce the same optimal policy when P 0 is a deterministic transition kernel; that is, for every x , a , there exists a unique x ′ such that P 0 ( x ′ | x , a ) = 1 .

Theorem 2.4. Suppose that for any x t , the supremum of V π ( x t ) is attainable, and that P 0 is a deterministic transition kernel. Then, Π v = Π p , where Π v = { π v | V π v 0 ( x t ) = max π V π 0 ( x t ) } and Π p = { π p | P π p ( x t ) = max π P π ( x t ) } .

Remark 2.5 . Theorem 2.4 applies to both discrete and continuous action spaces. In the continuous case, a deterministic policy can be represented as a degenerate distribution π µ ( a | x ) = δ ( a -µ ( x )) , where δ ( · ) is the Dirac delta function and µ is the deterministic mapping from states to actions.

The proof of Theorem 2.4 is provided in Appendix B.3. The value function in (3) remains very difficult to optimize, particularly in the long horizon setting with a large ¯ K , due to the absence of a discount factor. To address this, we further derive a lower bound for the value function in (3) using the following discounted value function and introduce its associated Q-function:

<!-- formula-not-decoded -->

for some the discount factor γ &lt; 1 . Our optimization procedure for this discounted value function will be detailed in Section 3.

## 3 Estimation Procedure

In this section, we provide a detailed description of the proposed policy optimization procedure. When censoring occurs, i.e., when ∆ k = 0 , we treat the process as terminating in the RL framework (a 'Done' state), and then return the reward to optimize policy. Consequently, the data collected at each stage are subject to censoring. Unlike general RL, the reward here needs to be estimated. So we first estimate the reward, then plug in the estimated reward to optimize the policy.

We first present the estimation methods and asymptotic properties of the survival function estimator ˆ S T k , obtained using either the Cox proportional hazards model (10) or the Aalen's additive hazards model (1). These models are chosen for their widespread use and relative simplicity, although more complex alternatives exist (e.g., transformation models, varying-coefficient models).

Definition . Let D k = { ( X i,k , A i,k , Y i,k , ∆ i,k ) } 1 ≤ i ≤ N k denote the data collected at stage k , representing N k independent observations. Denote the counting process at stage k as U ik ( t ) = I ( Y i,k ≤ t, ∆ i,k = 1) , and the corresponding at-risk process as V ik ( t ) = I ( Y i,k ≥ t ) . Since S T k ( t | X k , A k ) = exp {-Λ T k ( t | X k , A k ) } , where Λ T k ( t | X k , A k ) is the cumulative hazard function, estimating S T k is equivalent to estimating Λ T k ( t | X k , A k ) . We define the covariates vector Z k := ( X ⊤ k , A k , A k X ⊤ k ) ⊤ and Z ik := ( X ⊤ i,k , A i,k , A i,k X ⊤ i,k ) ⊤ for subjects. Subsequently, we provide the estimation of Λ T k ( t | X k , A k ) in the following two models.

Cox model . We assume a stage-specific Cox proportional hazards model as Λ T k ( t | X k , A k ) = Λ 0 ,k ( t ) exp { η ⊤ k Z k } , k = 1 , . . . , ¯ K , where Λ 0 ,k ( t ) 's are the stage-specific unknown cumulative baseline hazard functions and η k 's are the stage-specific unknown (2 p +1) -dimension unknown parameters. Estimates of η k and Λ 0 ,k ( t ) can be obtained by maximizing the partial likelihood (6; 2). Specifically,

<!-- formula-not-decoded -->

Consequently, estimators for Λ T k ( t | X k , A k ) are given by

<!-- formula-not-decoded -->

Additive hazard model . We assume a stage-specific Aalen's additive hazards model as Λ T k ( t | Z k ) = Λ 0 ,k ( t ) + β ⊤ k Z k t, k = 1 , . . . , ¯ K , where Λ 0 ,k ( t ) 's are the cumulative baseline hazard functions, and β k 's are the vectors of regression coefficients. Following the arguments in (21), the estimator for β k is given by

<!-- formula-not-decoded -->

with a ⊗ 2 = aa ⊤ . Λ 0 ,k ( t ) is then estimated by

<!-- formula-not-decoded -->

Consequently, the estimator for Λ T k ( t | Z k ) is given by ˆ Λ T k ( t | Z k ) = ˆ Λ 0 ,k ( ˆ β k , t ) + ˆ β ⊤ k Z k t .

We summarize the consistency and the asymptotic normality property of ˆ Λ T k ( t | Z k ) in Theorem 3.1. The detailed proof is provided in Appendix B.7.

Theorem 3.1. For each 1 ≤ k ≤ ¯ K , suppose that there exists a constant M k such that G k ≤ M k . Then, under assumptions B.2-B.5 for the Cox model, or assumptions B.6-B.9 for the additive hazards model, for any t ∈ [0 , G k ] , as N k →∞ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (iii) √ N k { ˆ Λ T k ( t | Z k ) -Λ T k ( t | Z k ) } d - → N ( 0 , σ 2 k ( t ; Z k ) ) , where the variance function σ 2 k ( t ; Z k ) is given in (A.6) for the Cox model and (A.7) for the additive hazards model.

Theorem 3.1 characterizes the uniform convergence, convergence rate, and asymptotic normality of the estimator ˆ Λ T k ( t | Z k ) , thus demonstrating its validity. Accordingly, the use of ˆ S T k ( t | X k , A k ) = exp {-ˆ Λ T k ( t | X k , A k ) } as an estimator for S T k ( t | X k , A k ) is also reasonable. Therefore, we replace S T k in the Q-function in (4) with ˆ S T k , resulting in a feasible Q-function as follows:

<!-- formula-not-decoded -->

Through the lower bound of log P π , we transform the problem of maximizing the probability that the duration between consecutive events exceeds a meaningful threshold into maximizing a discounted sum of estimated cumulative hazards. This enables the direct application of standard RL algorithms to determine the optimal policy.

Specifically, we optimize ˆ Q π in (8) using the soft-update Deep Q-Network (DQN) algorithm (20) for discrete action spaces and the Deep Deterministic Policy Gradient (DDPG) algorithm (20) for continuous action spaces. The detailed algorithms for obtaining the optimal policy are provided in Appendix A.6 due to space limitations.

## 4 Simulation Studies

In this section, we conduct numerical studies to assess the finite-sample performance of the proposed method and demonstrate its advantages. Specifically, our aim is to address the following research questions (RQs). RQ1: How does the proposed decision-making strategy compare with existing approaches in terms of improving the total recurrent-event duration? RQ2: Does our method exhibit stable and consistent performance across various distributions of T i,k ? We first outline the simulation setups and subsequently address the aforementioned questions through extensive numerical experiments.

## 4.1 Simulation Setups

Setting 1. Linear transition and discrete actions . For the initial stage ( k = 1 ), we generate X i, 1 from a standard normal distribution N ( 0 2 , I 2 ) . For each stage, the available actions are chosen from A = { 0 , 1 } . X i,k +1 evolves according to the following iterative formula,

<!-- formula-not-decoded -->

where ϵ i,k i.i.d ∼ N ( 0 2 , I 2 / 4) . Similar simulation setups can be found in Shi et al. (37) and Chen et al. (8). We let θ i,k = exp { (2 A i,k -1) X ⊤ i,k ν } , where ν = (2 , -1) ⊤ .

Setting 2. nonlinear transition and continuous actions . For the initial stage ( k = 1 ), we generate X i, 1 from a standard normal distribution N ( 0 50 , I 50 ) . For each stage, the available actions are chosen from A = [ -1 , 1] . X i,k +1 evolves according to the following iterative formula,

<!-- formula-not-decoded -->

where ϵ i,k i.i.d ∼ N ( 0 50 , I 50 / 4) . We let θ i,k = exp { ( -A i,k · X T i,k β ) } , where β ∈ R 50 is a vector whose first three elements are generated from the standard normal distribution, and whose remaining elements are 0.

In Setting 1 and Setting 2, we consider total stages ¯ K = 20 . A i,k is selected by maximizing the proposed Q-function in (8), and is chosen from A with a decreasing greedy probability. We set G k = 7 , and the duration T i,k is generated according to T i,k = min { B i,k , G k } , where B i,k ∼ Gamma (1 /θ i,k , θ i,k ) . We let C i follow the uniform distribution U (0 , 7 × 10) , independent of

T i,k , X i,k . We set the pre-specified threshold α k = 1 / 14 . To simulate the real-world uncertainty, we consider transition kernels with stochastic perturbations. We optimize the objective function defined in (8) using the soft-update DQN algorithm for Setting 1 and the DDPG algorithm for Setting 2. The implementation code is available at https://github.com/fjqfengjianqi/ NIPS2025-RL-for-Survival .

## 4.2 Methods and Metrics

Methods . Our investigation of the aforementioned RQs is based on the comparison of the following several methods:

- Cox: our proposed method that uses the Cox model (10) to estimate the survival function.
- Aah: our proposed method that uses the additive hazards model (1) to estimate the survival function.
- Baseline: the optimal treatment policy proposed by Liu et al. (23). Specifically, their optimal treatment policy is determined by optimizing the total recurrent-event duration ∑ ¯ K k =1 T k via finite-stage dynamic treatment regimes. The corresponding objective function, derived from Liu et al. (23), is provided in Appendix B.8.

Metrics . We conduct M = 50 epochs with a sample size of N , each progressing from stage 1 to stage ¯ K = 20 , with different methods updating their policies as the experiments progressed. The experiments are repeated for S = 10 different seeds. For each seed s = 1 , . . . , S and epoch m = 1 , . . . , M , we record T ( s,m ) i,k and θ ( s,m ) i,k at each stage k . We then predict the values of T i,k and θ i,k at epoch m derived from simulations using the Cox, Aah, and Baseline methods, denoted as ˆ T ( m ) i,k = ∑ S s =1 T ( s,m ) i,k /S and ˆ θ ( m ) i,k = ∑ S s =1 θ ( s,m ) i,k /S , respectively. In this paper, we employ the following metrics to evaluate the effectiveness and robustness of the proposed method:

- Average Recurrent-Event Duration ( ARED ): ARED ( m ) = ∑ N i =1 ∑ ¯ K k =1 ˆ T ( m ) i,k /N .
- Average Estimated Variance ( AEV ): AEV ( m ) = ∑ N i =1 ∑ ¯ K k =1 ˆ θ ( m ) i,k /N .
- Average ARED over Last 40 Epochs ( AARED ): AARED = ∑ 50 m =11 ARED ( m ) / 40 .
- Average AEV over Last 40 Epochs ( AAEV ): AAEV = ∑ 50 m =11 AEV ( m ) / 40 .

## 4.3 Comparisons on the recurrent-event duration (RQ1)

Figure 2 illustrates the changes in ARED and log ARED with increasing epochs in Setting 1 and Setting 2 for varying sample sizes N . The results suggest that our proposed method exhibits faster convergence than the baseline model.

Figure 2: The average recurrent-event duration over epochs with 10 seeds.

<!-- image -->

Table 1: Mean (Standard Deviation) of AARED under different settings, models and sample sizes N

| Setting   |     N | Aah            | Cox            | Baseline      |
|-----------|-------|----------------|----------------|---------------|
| Setting 1 |   100 | 19.96 (0.006)  | 19.93 (0.007)  | 19.80 (0.021) |
| Setting 1 |  1000 | 19.94 (<0.001) | 19.94 (0.001)  | 19.83 (0.008) |
| Setting 1 | 10000 | 20.00 (<0.001) | 19.98 (<0.001) | 19.87 (0.006) |
| Setting 2 |   100 | 18.88 (0.125)  | 19.19 (0.086)  | 17.17 (0.074) |
| Setting 2 |  1000 | 19.52 (0.047)  | 19.45 (0.057)  | 17.06 (0.074) |
| Setting 2 | 10000 | 19.57 (0.055)  | 19.54 (0.072)  | 17.16 (0.080) |

Figure 3: The average estimated variance over epochs with 10 seeds.

<!-- image -->

Table 1 presents a comparison of the AARED values for different models with varying sample sizes N . From Table 1, we can observe that as the sample size increases, the variance of AARED decreases. At the same time, our model consistently achieves a higher AARED compared to the baseline model. Under Setting 1, the maximum improvement is 0.8% ( N = 100 ), while under Setting 2, the maximum improvement reaches 14.4% ( N = 1000 ). This demonstrates that our method achieves faster convergence and yields higher recurrent-event durations.

## 4.4 Stability of the proposed strategy (RQ2)

Figure 3 illustrates the relationship between the AEV and log AEV and the number of epochs for Setting 1 and Setting 2, with results shown for various sample sizes N . It can be observed that our method leads to a faster decrease and stabilization of the variance.

Table 2 presents a comparison of the AAEV values for different models with varying sample sizes N . From Table 2, we can observe that as the sample size increases, the variance of AEV decreases. Moreover, our model consistently achieves a smaller variance compared to the baseline model, with a maximum reduction of 49% ( N = 100 ) in Setting 1 and 99.89% ( N = 10000 ) in Setting 2.

Table 2: Mean (Standard Deviation) of AAEV under different settings, models and sample sizes N

| Setting   |     N | Aah                                                | Cox                                                | Baseline                                                  |
|-----------|-------|----------------------------------------------------|----------------------------------------------------|-----------------------------------------------------------|
| Setting 1 |   100 | 10.53 (0.001) 10.54 (<0.001)                       | 11.11 (0.016) 11.95 (0.005)                        | 20.70 (27.33) 20.00 (19.47)                               |
| Setting 1 |  1000 |                                                    |                                                    |                                                           |
| Setting 1 | 10000 | 10.54 (<0.001)                                     | 12.00 (<0.001)                                     | 20.18 (20.85)                                             |
| Setting 2 |   100 | 2533 ( 1 × 10 8 ) 202 ( 4 × 10 5 ) 49 ( 2 × 10 3 ) | 224 ( 2 × 10 5 ) 441 ( 3 × 10 6 ) 181 ( 1 × 10 5 ) | 2251 ( 3 × 10 9 ) 41189 ( 2 × 10 10 ) 43657 ( 3 × 10 10 ) |
| Setting 2 |  1000 |                                                    |                                                    |                                                           |
| Setting 2 | 10000 |                                                    |                                                    |                                                           |

Figure 4: The values of the concordance index fitted for each stage under different models.

<!-- image -->

Figure 2, 3 and Table 1,2 together demonstrate that the proposed strategy not only maintains effectiveness but also exhibits high robustness. This indicates that the recurrent-event duration achieved by our method is not only higher than that of the baseline method but also shows significantly smaller variance fluctuations. Additional simulation results are provided in Appendix A.1 and A.4.

## 5 Real Data Analysis

In this section, we apply the proposed method to analyze decision-making strategies based on the 2018 Intern Health Study (IHS). IHS was a microrandomized trial (MRT) designed to evaluate the impact of app-based push notifications on physical and mental health outcomes. Over a 26-week period, participants were re-randomized weekly to receive activity suggestions.

Data description . The data contains weekly records for each participant, whether a message was sent, the duration of daily sleep, the count of steps and the mood scores. We define the daily average duration of sleep and the step count for the i -th participant in the k -th week as the state variable X i,k , the binary indicator of message sending as the decision action A i,k , and the cumulative mood score as the reward T i,k . Let stage duration G k = 70 , representing the maximum possible mood score per week. Since mood scores are not recorded daily within a week, instances with incomplete mood data are treated as censored, denoted by ∆ i,k = 0 . After data manipulation, the final dataset includes 1,176 participants followed for up to 26 weeks, with a censoring rate of 87.8%.

Analysis . Because the action space is discrete, we use an offline soft-update DQN method to determine the optimal policy π , with ¯ K = 26 . We analyze the fitting performance of different survival models, i.e., the Aah model, the Cox model and the Log-Normal accelerated failure time (AFT) model (as shown in Appendix A.5), to the dataset. Furthermore, given the optimal policy π , we calculate the average probability that the high mood score exceeds a threshold of the stage duration α k G k by

<!-- formula-not-decoded -->

Results . Figure 4 presents the concordance index (C-index) for each model at each stage. C-index values are between 0 and 1; values greater than 0.5 indicate better than random discrimination. As shown in Figure 4, the Log-Normal AFT model demonstrates better discriminative performance than the Aah and Cox models.

We denote the average probability derived from the Aah model, the Cox model and the baseline model as ¯ P π A , ¯ P π C , and ¯ P π B , respectively. We then present the log probability difference (log ¯ P π A -log ¯ P π B ) for the Aah model and (log ¯ P π C -log ¯ P π B ) for the Cox model across varying α k settings. As shown in Figure 5, our model outperforms the baseline model across all α k values, achieving higher survival probabilities. The optimal performance occurs at α k = 1 , where the probability improved by up to 35%. Additional analyses using different metrics are provided in Appendix A.5.

To conclude, our proposed model formulates policies by maximizing the probability that the highmood duration in each stage exceeds a threshold, achieving higher probabilities of high mood duration

Figure 5: Comparison of log survival probability difference across policies with different α k .

<!-- image -->

compared to traditional RL strategies that maximize expected rewards. Additionally, the proposed method dynamically adapts the policies to individual preferences in different values of α k .

## 6 Conclusion

This paper presents a novel RL framework for survival analysis tailored to alternating recurrent event data, addressing the limitations of traditional methods designed for time-to-event data and expectation maximization. We propose a probability-based objective that maximizes the probability that recurrentevent durations exceed a given threshold, and reformulate the task as a standard RL problem by optimizing a lower bound of this objective. We provide theoretical guarantees for the equivalence of the optimal policy, as well as for the consistency and asymptotic normality of the estimated cumulative hazard functions. Simulations and real-data experiments demonstrate faster convergence, lower variance, and higher event-duration probabilities of our proposed method compared to existing traditional RL-based methods.

## Acknowledgments and Disclosure of Funding

This work was partly supported by the National Natural Science Foundation of China (12401366, 12301348, 12371292), the Natural Science Foundation of Shandong Province (ZR2023QA086), and the Shandong University Future Plan for Young Scholars. The authors are also grateful for the contributions of the researchers, administrators, and participants involved in the Intern Health Study (NCT03972293).

## References

- [1] Aalen, O. O. A linear regression model for the analysis of life times. Statistics in Medicine , 8(8):907-925, 1989.
- [2] Andersen, P. K. and Gill, R. D. Cox's regression model for counting processes: A large sample study. The Annals of Statistics , 10(4):1100-1120, 1982.
- [3] Bellman, R. Dynamic programming. Science , 153(3731):34-37, 1966.
- [4] Bian, Z., Moodie, E. E., Shortreed, S. M., and Bhatnagar, S. (2023). Variable selection in regression-based estimation of dynamic treatment regimes. Biometrics , 79 (2), 988-999.
- [5] Bradtke, S. and Duff, M. Reinforcement learning methods for continuous-time markov decision problems. Advances in neural information processing systems , 7, 1994.
- [6] Breslow, N. E. Contribution to discussion of paper by dr cox. Journal of the Royal Statistical Society Series B: Statistical Methodology , 34:216-217, 1972.
- [7] Cai, H., Shi, C., Song, R., and Lu, W. (2023). Jump interval-learning for individualized decision making with continuous treatments. Journal of Machine Learning Research , 24 (140), 1-92.
- [8] Chen, E. Y., Song, R., and Jordan, M. I. Reinforcement learning with heterogeneous data: estimation and inference. arXiv preprint arXiv:2202.00088 , 2022.
- [9] Cho, H., Holloway, S. T., and Couper, J. D. Multi-stage optimal dynamic treatment regimes for survival outcomes with dependent censoring. Biometrika , 110:395-420, 2023.
- [10] Cox, D. R. Regression models and life-tables. Journal of the Royal Statistical Society Series B: Statistical Methodology , 34(2):187-202, 12 1972.
- [11] Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S., and Zhu, R. Estimating heterogeneous treatment effects with right-censored data via causal survival forests. Journal of the Royal Statistical Society Series B: Statistical Methodology , 85:179-211, 2023.
- [12] Dabney, W., Rowland, M., Bellemare, M., and Munos, R. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 32, 2018.
- [13] Fleming, T. R., Lin, D. Y., and Wei, L. J. Confidence bands for survival curves under the proportional: Hazards model. Biometrika , 81(1):73-81, 1994.
- [14] Goldberg, Y. and Kosorok, M. R. Q-learning with censored data. The Annals of Statistics , 40:529-560, 2012.
- [15] Hargrave, M., Spaeth, A., and Grosenick, L. Epicare: a reinforcement learning benchmark for dynamic treatment regimes. Advances in neural information processing systems , 37:130536-130568, 2024.
- [16] Jiang, R., Lu, W., Song, R., and Davidian, M. On estimation of optimal treatment regimes for maximizing t-year survival probability. Journal of the Royal Statistical Society Series B: Statistical Methodology , 79 (4):1165-1185, 2017.
- [17] Lee, C., Yoon, J., and Van Der Schaar, M. Dynamic-deephit: A deep learning approach for dynamic survival analysis with competing risks based on longitudinal data. IEEE Transactions on Biomedical Engineering , 67(1):122-133, 2019.
- [18] Lee, C. H., Huang, C.-Y., Xu, G., and Luo, X. Semiparametric regression analysis for alternating recurrent event data. Statistics in Medicine , 37(6):996-1008, 2018.
- [19] Li, M. , Shi, C. , Wu, Z. and Fryzlewicz, P.Testing stationarity and change point detection in reinforcement learning. The Annals of Statistics , 53(3): 1230-1256, 2025.
- [20] Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y ., Silver, D., and Wierstra, D. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 , 2015.
- [21] Lin, D. Y. and Ying, Z. Semiparametric analysis of the additive risk model. Biometrika , 81(1):61-71, 1994.
- [22] Liu, Y., Logan, B., Liu, N., Xu, Z., Tang, J., and Wang, Y. Deep reinforcement learning for dynamic treatment regimes on medical registry data. IEEE International Conference on Healthcare Informatics (ICHI) , pp. 380-385. 2017.

- [23] Liu, Z., Zhan, Z., Liu, J., Yi, D., Lin, C., and Yang, Y . On estimation of optimal dynamic treatment regimes with multiple treatments for survival data-with application to colorectal cancer study. arXiv preprint arXiv:2310.05049 , 2023.
- [24] Loe, A., Murray, S., and Wu, Z. Random forest for dynamic risk prediction of recurrent events: a pseudo-observation approach. Biostatistics , 26(1):kxaf007, 2025.
- [25] Luckett, D. J., Laber, E. B., Kahkoska, A. R., Maahs, D. M., Mayer-Davis, E., and Kosorok, M. R. Estimating dynamic treatment regimes in mobile health using v-learning. Journal of the American Statistical Association , 115(530):692-706, 2020.
- [26] Mo, W., Qi, Z., and Liu, Y. (2021). Learning optimal distributionally robust individualized treatment rules. Journal of the American Statistical Association , 116 (534), 659-674.
- [27] Munk-J, P., Mortensen, P. B., et al. Incidence and other aspects of the epidemiology of schizophrenia in denmark, 1971-87. The British Journal of Psychiatry , 161(4):489-495, 1992.
- [28] Murphy, S. A. (2003). Optimal dynamic treatment regimes. Journal of the Royal Statistical Society: Series B: Statistical Methodology , 65 (2), 331-355.
- [29] NeCamp, T., Sen, S., Frank, E., Walton, M., Ionides, E., Fang, Y., Tewari, A., and Wu, Z. Assessing real-time moderation for developing adaptive mobile health interventions for medical interns: Microrandomized trial. Journal of Medical Internet Research , 22(3):e15033, 2020.
- [30] Papadimitriou, C. H. and Tsitsiklis, J. N. The complexity of markov decision processes. Mathematics of operations research , 12(3):441-450, 1987.
- [31] Qi, Z., Liu, D., Fu, H., and Liu, Y . (2020). Multi-armed angle-based direct learning for estimating optimal individualized treatment rules with various outcomes. Journal of the American Statistical Association , 115 (530), 678-691.
- [32] Ren, B. and Barnett, I. Combining mixed effects hidden markov models with latent alternating recurrent event processes to model diurnal active-rest cycles. Biometrics , 79(4):3402-3417, 2023.
- [33] Robins, J. M. (2004). Optimal structural nested models for optimal sequential decisions. In Proceedings of the Second Seattle Symposium in Biostatistics: Analysis of Correlated Data (pp. 189-326). New York, NY: Springer New York.
- [34] Sen, S. , Kranzler, H. R. , Krystal, J. H. , Speller, H. , Chan, G. , Gelernter, J. and Guille, C.A prospective cohort study investigating factors associated with depression during medical internship. Archives of General Psychiatry , 67(6): 557-565, 2010.
- [35] Shi, C., Fan, A., Song, R., and Lu, W. (2018). High-dimensional A-learning for optimal dynamic treatment regimes. Annals of Statistics , 46 (3), 925-950.
- [36] Shi, C., Song, R., Lu, W., and Fu, B. (2018). Maximin projection learning for optimal treatment decision with heterogeneous individualized treatment effects. Journal of the Royal Statistical Society: Series B: Statistical Methodology , 80 (4), 681-702.
- [37] Shi, C., Zhang, S., Lu, W., and Song, R. Statistical inference of the value function for reinforcement learning in infinite-horizon settings. Journal of the Royal Statistical Society Series B: Statistical Methodology , 84 (3):765-793, 2022.
- [38] Shin, G., Jarrahi, M. H., Fei, Y., Karami, A., Gafinowitz, N., Byun, A., and Lu, X. Wearable activity trackers, accuracy, adoption, acceptance and health impact: A systematic literature review. Journal of Biomedical Informatics , 93:103153, 2019.
- [39] Shinohara, R. T., Sun, Y., and Wang, M.-C. Alternating event processes during lifetimes: population dynamics and statistical inference. Lifetime Data Analysis , 24:110-125, 2018.
- [40] Simoneau, G., Moodie, E. E., Nijjar, J. S., Platt, R. W., Investigators, S. E. R. A. I. C., et al. Estimating optimal dynamic treatment regimes with survival outcomes. Journal of the American Statistical Association , 115(531):1531-1539, 2020.
- [41] Sutton, R. S. and Barto, A. G. Reinforcement learning: An introduction. Robotica , 17(2):229-235, 1999.
- [42] Wahed, A. S. and Thall, P. F. Evaluating joint effects of induction-salvage treatment regimes on overall survival in acute leukaemia. Journal of the Royal Statistical Society Series B: Statistical Methodology , 62: 67-83, 2013.

- [43] Wang, J. , Fang, Y. , Frank, E. , Walton, M. A. , Burmeister, M. , Tewari, A. , Dempsey, W. , NeCamp, T. , Sen, S. and Wu, Z.Effectiveness of gamified team competition as mHealth intervention for medical interns: a cluster micro-randomized trial. npj Digital Medicine , 6(1): 4, 2023.
- [44] Wang, L., He, K., and Schaubel, D. E. Penalized survival models for the analysis of alternating recurrent event data. Biometrics , 76(2):448-459, 2020.
- [45] Xia, M., Murray, S., and Tayob, N. Regression analysis of recurrent-event-free time from multiple follow-up windows. Statistics in Medicine , 39(1):1-15, 2020.
- [46] Xu, Y., Muller, P., Wahed, A. S., and Thall, P. F. Bayesian nonparametric estimation for dynamic treatment regimes with sequential transition times. Journal of the American Statistical Association , 111:911-950, 2016.
- [47] Xue, F., Zhang, Y., Zhou, W., Fu, H., and Qu, A. Multicategory angle-based learning for estimating optimal dynamic treatment regimes with censored data. Journal of the American Statistical Association , 117(539): 1438-1451, 2022.
- [48] Zhang, B., Tsiatis, A. A., Laber, E. B., and Davidian, M. (2012). A robust method for estimating optimal treatment regimes. Biometrics , 68 (4), 1010-1018.
- [49] Zhang, B., Tsiatis, A. A., Laber, E. B., and Davidian, M. (2013). Robust estimation of optimal dynamic treatment regimes for sequential treatment decisions. Biometrika , 100 (3), 1010-1093.
- [50] Zhang, J., and Bareinboim, E. (2019). Near-optimal reinforcement learning in dynamic treatment regimes. Advances in Neural Information Processing Systems (NeurIPS) , 32 .
- [51] Zhao, Y., Zeng, D., Socinski, M. A., and Kosorok, M. R. Reinforcement learning strategies for clinical trials in nonsmall cell lung cancer. Biometrics , 67(4):1422-1433, 2011.
- [52] Zhao, Y., Zeng, D., Rush, A. J., and Kosorok, M. R. (2012). Estimating individualized treatment rules using outcome weighted learning. Journal of the American Statistical Association , 107 (499), 1106-1118.
- [53] Zhao, Y. Q., Zeng, D., Laber, E. B., and Kosorok, M. R. (2015). New statistical learning methods for estimating optimal dynamic treatment regimes. Journal of the American Statistical Association , 110 (510), 583-598.
- [54] Zhao, Y.-Q., Zeng, D., Laber, E. B., Song, R., Yuan, M., and Kosorok, M. R. Doubly robust learning for estimating individualized treatment with censored data. Biometrika , 102(1):151-168, 2015.
- [55] Zhou, Y., Wang, L., Song, R., and Zhao, T. Transformation-invariant learning of optimal individualized decision rules with time-to-event outcomes. Journal of the American Statistical Association , 118:26322644, 2023.
- [56] Zhou, Y., Qi, Z., Shi, C., and Li, L. (2023, April). Optimizing pessimism in dynamic treatment regimes: A Bayesian learning approach. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS) (pp. 6704-6721). PMLR.
- [57] Zhou, W., Zhu, R., and Qu, A. (2024). Estimating optimal infinite-horizon dynamic treatment regimes via PT-learning. Journal of the American Statistical Association , 119 (545), 625-638. doi:10.1080/01621459.2023.2232745
- [58] Zhu, R., Zhao, Y. Q., Chen, G., Ma, S., and Zhao, H. (2017). Greedy outcome weighted tree learning of optimal personalized treatment rules. Biometrics , 73 (2), 391-400.

## A Additional Data Analysis

This section provides supplementary data and analysis for the simulation study in Section 4. We conducted M = 50 epochs with a sample size of N , each progressing from stage 1 to stage ¯ K = 20 , with different methods updating their policies as the experiments progressed. The experiments were repeated for S = 10 different seeds. For each seed s = 1 , . . . , S and epoch m = 1 , . . . , M , we recorded T ( s,m ) i,k and θ ( s,m ) i,k at each stage k . We then predicted the values of T i,k and θ i,k at epoch m derived from simulations using the Cox, Aah, and Baseline methods, denoted as ˆ T ( m ) i,k = ∑ S s =1 T ( s,m ) i,k /S and ˆ θ ( m ) i,k = ∑ S s =1 θ ( s,m ) i,k /S , respectively.

The following metrics are used to evaluate the effectiveness and robustness of the proposed method:

- Average Recurrent-Event Duration ( ARED ): ARED ( m ) = ∑ N i =1 ∑ ¯ K k =1 ˆ T ( m ) i,k /N .
- Average Estimated Variance ( AEV ): AEV ( m ) = ∑ N i =1 ∑ ¯ K k =1 ˆ θ ( m ) i,k /N .
- Average ARED over Last 40 Epochs ( AARED ): AARED = ∑ 50 m =11 ARED ( m ) / 40 .
- Average AEV over Last 40 Epochs ( AAEV ): AAEV = ∑ 50 m =11 AEV ( m ) / 40 .
- Relative Difference of Estimated Variance ( RDEV ):
- -For the Cox method: RDEV ( m ) = ( ∑ N i =1 ˆ θ C ( m ) i,k -∑ N i =1 ˆ θ B ( m ) i,k ) / ∑ N i =1 ˆ θ B ( m ) i,k .
- -For the Aah method: RDEV ( m ) = ( ∑ N i =1 ˆ θ A ( m ) i,k -∑ N i =1 ˆ θ B ( m ) i,k ) / ∑ N i =1 ˆ θ B ( m ) i,k .

## A.1 Compare with RDEV and ˆ θ i,k

Here, we present the RDEV values for each stage in Figure 6.

<!-- image -->

(a) Setting 1

(b) Setting 2

Figure 6: Relative difference of estimated variance at each stage under different sample sizes.

Furthermore, we examine the boxplots of ˆ θ ( m ) i,k under varying sample sizes during the final 40 epochs in Figure 7. The results reveal two key observations:

- The variance of ˆ θ ( m ) i,k decreases progressively with increasing sample size.
- This reduction in variance indicates accelerated convergence rates at larger sample sizes.

## A.2 Computation Cost

Here we compare the average time cost per epoch for one seed under Setting 1 in Table 3. The experiments were conducted on a GTX 1650 GPU.

Figure 7: Boxplots of estimated variance ˆ θ ( m ) i,k for each model under different settings across the last 40 epochs as sample size varies.

<!-- image -->

Table 3: Computation time (in seconds) of different models under varying sample sizes N

| Model    | N =100   | N =1000   | N =10000   |
|----------|----------|-----------|------------|
| Baseline | 0.22s    | 0.77s     | 25.50s     |
| Cox      | 0.89s    | 4.08s     | 35.66s     |
| Aah      | 0.56s    | 3.10s     | 42.30s     |

## A.3 Selection of α k

The selection of α k is tailored to the specific research question being addressed. For example, in (16)'s study on HIV data, they chose α k = 0 . 5 , focusing on the optimal strategy for maximizing the probability of survival exceeding the median. In our study, we selected α k = 1 / 14 to maximize the probability of an intern's positive mood exceeding one day per bi-weekly period. We generally advise using an α k of 0.5 or less, mainly because α k values near 1 rely heavily on tail probability estimation, which can be unreliable. These results are consistent with the numerical results we obtained from our experiment results. Here we present the AARED and AAEV under different α k values in Setting 1 in Table 4 and 5 for reference.

## A.4 Compare with Epicare

In this section, we demonstrate the performance of our model on a dynamic treatment regimes RL benchmark, namely the Epicare environment (15). Since the Epicare environment does not include a censoring setting, but the original work defines a maximum number of inquiries after which the process becomes unobservable, we adopt this as our definition of censoring. We define a reinforcement learning environment E = ⟨X , A , P, R ⟩ to simulate the process of chronic disease treatment. Next, we describe the components of this environment in detail.

Table 4: Mean (Standard Deviation) of AARED under different α k values and models

| α k                                                           | Aah                                                                                                                                                                                                                                           | Cox                                                                                                                                                                                                                                           | Baseline       |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| 1/14 1/7 3/14 2/7 5/14 3/7 1/2 4/7 9/14 5/7 11/14 6/7 13/14 1 | 19.9330 (0.0074) 19.9460 (0.0089) 19.1728 (0.0304) 18.9851 (0.0459) 18.9972 (0.0438) 18.9831 (0.0467) 19.0010 (0.0435) 19.0134 (0.0454) 19.0015 (0.0387) 19.0007 (0.0433) 19.0027 (0.0453) 19.0238 (0.0455) 19.0107 (0.0434) 19.0591 (0.0465) | 19.9421 (0.0091) 19.9228 (0.0130) 19.8896 (0.0157) 19.8285 (0.0177) 19.8193 (0.0207) 19.7919 (0.0192) 19.7900 (0.0218) 19.7571 (0.0295) 19.7232 (0.0315) 19.6999 (0.0291) 19.6703 (0.0515) 19.6379 (0.0497) 19.6413 (0.0373) 19.6885 (0.0334) | 19.83 (7.9e-3) |

Table 5: Mean (Standard Deviation) of AAEV under different α k values and models

| α k                                                           | Aah                                                                                                                                                                                                                                                       | Cox                                                                                                                                                                                                                                                        | Baseline      |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| 1/14 1/7 3/14 2/7 5/14 3/7 1/2 4/7 9/14 5/7 11/14 6/7 13/14 1 | 10.5408 (0.0027) 10.8309 (0.0818) 64.5515 (37.9364) 78.1629 (14.4570) 78.5551 (13.8795) 78.5710 (14.2459) 78.6225 (13.8094) 78.6280 (14.3543) 78.6196 (14.6698) 78.4769 (17.6261) 78.2809 (19.1925) 78.0577 (19.7367) 77.9280 (19.0170) 71.9542 (83.7705) | 11.9509 (2.3916) 13.9811 (4.4648) 17.7416 (12.5118) 20.6589 (16.5979) 22.2620 (26.8484) 23.5382 (27.2738) 24.4125 (35.8999) 25.0995 (50.6035) 27.7541 (67.8201) 29.4384 (70.1726) 30.3147 (91.0969) 32.2957 (104.3780) 32.6802 (79.8917) 29.4135 (76.7759) | 20.00 (19.47) |

Disease state: Let the index set of disease states be I = { 1 , . . . , n d } , and let 0 denote the absorbing remission state. At each stage t , the underlying state is

<!-- formula-not-decoded -->

where d t = 0 represents recovery and is absorbing. For each disease index k ∈ I , symptoms follow a multivariate Gaussian N ( µ k , Σ k ) , with µ k ∈ R n sym , Σ k ∈ R n sym × n sym , n sym = 8 . Parameters are randomly generated to induce diversity:

<!-- formula-not-decoded -->

where σ km ∼ U (1 , 2) and A k ∈ R n sym × n sym is a random orthogonal matrix, ensuring Σ k is symmetric positive definite.

Symptom state: The observable state at stage t is a normalized symptom vector

<!-- formula-not-decoded -->

where σ ( · ) is the sigmoid, guaranteeing x t ∈ [0 , 1] n sym . If d t = 0 , the episode has terminated and no further observations are drawn.

Action Space: The action space is a finite set of treatments:

<!-- formula-not-decoded -->

Each action corresponds to a treatment intervention selected by the agent.

Transition matrix P ∗ : Disease-to-disease evolution (excluding remission) is governed by a random row-stochastic matrix

<!-- formula-not-decoded -->

constructed as

<!-- formula-not-decoded -->

̸

This yields a sparse, approximately symmetric graph over I , reflecting realistic limited comorbidity pathways. Remission is handled separately, for k ∈ I and action a ,

<!-- formula-not-decoded -->

Equivalently, define the augmented action-dependent kernel P ∗ on { 0 } ∪ I :

<!-- formula-not-decoded -->

Stationary distribution: The initial disease is sampled from the stationary distribution ρ = ( ρ 1 , · · · , ρ n d ) of P over I :

<!-- formula-not-decoded -->

(Note that remission d t = 0 is excluded here since it is an absorbing terminal state.)

Reward function: The stage-wise reward balances cost, remission benefit, and adverse-event risk:

with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Termination: We assume that the process terminates once the patient enters the remission state, i.e., when d t = 0 . In addition, we impose a maximum follow-up horizon K max = 8 . If termination has not occurred by t = K max , the trajectory is regarded as right-censored, meaning that the observation is incomplete. Formally, we define the event time as ¯ K = min { t | d t = 0 } , and introduce the event indicator ∆ t = 1 { t&lt; ¯ K } × 1 { t ≤ K max } , which equals 1 if the event (remission) occurs before censoring, and 0 otherwise.

We defined ( x t , a t , R t , ∆ t ) for t = 1 , . . . , K max , where R t is analogous to Y t in the main text, both representing the reward. An offline dataset with sample size N = 1000 was generated. Using 10 different random seeds for network initialization, we trained multiple models and evaluated a variety of policies. For each policy, we report the cumulative reward as well as the final censoring rate. The following four approaches are compared:

- Cox: our proposed method utilizing the survival function estimated from the Cox proportional hazards model (10).
- Aah: our proposed method utilizing the survival function estimated from Aalen's additive hazards model (1).
- Km: the optimal treatment policy proposed by Liu et al. (23), which maximizes the expected value via Kaplan-Meier estimation. The corresponding objective function, derived from the method of Liu et al.'s (2023), is provided in Appendix B.8.

̸

- RL: classical reinforcement learning methods that maximize reward without accounting for ∆ , consistent with the objective proposed in (15).

We set the number of training epochs to 50 and the number of evaluation episodes to 200. Since we require Y t &gt; 0 , all R t values were normalized across individuals at each stage prior to estimation and training. We computed the mean and variance of the cumulative reward ( ∑ K max t =1 ∑ N i =1 R it /N ) and censoring rate (1 -∑ N i =1 ∆ iK max /N ) across all 10 random seeds, including results from our proposed methods under different values of α k . These results, reported in Table 6 and Table 7, indicate that our methods consistently achieve higher rewards and lower censoring rates, thereby demonstrating superior performance in the Epicare environment.

Table 6: Mean (Standard Deviation) of Cumulative Reward Under different model and α k

| α k                                     | Aah                                                                                                                                                              | Cox                                                                                                                                                   | Km               | RL             |
|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|----------------|
| 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 | 11.0323(4.9149) 11.0323(4.9149) 10.0921(3.4005) 10.0921(3.4005) 11.0028(4.3290) 11.1656(3.8367) 11.1656(3.8367) 10.7549(4.7615) 10.7549(4.7615) -12.7179(3.0012) | 8.9952(3.5328) 6.5327(4.3448) 7.2183(4.1138) 7.5531(4.6677) 8.5168(3.0017) 7.3527(4.5401) 5.4090(2.7325) 7.4012(3.6093) 8.2964(4.4176) 0.6448(6.0716) | -15.8804(1.1859) | 6.2507(3.3373) |

Table 7: Mean (Standard Deviation) of Censoring Rate Under different model and α k

| α k                                     | Aah                                                                                                                                                   | Cox                                                                                                                                                   | Km             | RL             |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|----------------|
| 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 | 0.5050(0.0575) 0.5050(0.0575) 0.5125(0.0453) 0.5125(0.0453) 0.5055(0.0535) 0.5060(0.0444) 0.5060(0.0444) 0.5120(0.0586) 0.5120(0.0586) 0.7975(0.0354) | 0.5405(0.0452) 0.5665(0.0525) 0.5605(0.0494) 0.5540(0.0602) 0.5450(0.0398) 0.5585(0.0539) 0.5785(0.0369) 0.5555(0.0440) 0.5455(0.0469) 0.6365(0.0603) | 0.9610(0.0139) | 0.6410(0.0431) |

## A.5 Real Data Analysis Model

We then presented the relative probability ( ¯ P π A -¯ P π B ) / ¯ P π B for the Aah model and ( ¯ P π C -¯ P π B ) / ¯ P π B for the Cox model across varying α k settings in Figure 8. As shown in Figure 8, our model outperformed the baseline across all α k values, achieving higher survival probabilities. At the same time, we observe that the Cox model also provides a satisfactory fit. Below, we present the results of estimating P π in (1) under the Cox model, then we show the log probability differences in Figure 9 and the relative probabilities in Figure 10.

Log Normal AFT model assumes that the natural logarithm of the survival time T k follows a normal distribution: log( T k ) ∼ N ( µ ( z k ) , σ 2 ) , where: The location parameter µ ( z ) is a linear combination of the covariates z = [ z 1 , z 2 , . . . , z n ] : µ ( z ) = a 0 + a 1 z 1 + a 2 z 2 + · · · + a n z n Here, a 0 , a 1 , . . . , a n are the regression coefficients to be estimated, reflecting the impact of the covariates z on the mean of the logarithm of the survival time. The scale parameter σ is a fixed constant (independent of covariates), describing the standard deviation of log( T ) and controlling the degree of dispersion of the distribution. The model estimates parameters by maximizing the log-likelihood function.

<!-- image -->

Figure 8: Comparison of relative survival probability improvements across policies with different α k under Log Normal AFT model estimation.

Figure 9: Comparison of log survival probability differences across policies with different α k under Cox model estimation.

<!-- image -->

Figure 10: Comparison of relative survival probability across policies with different α k under Cox model estimation.

<!-- image -->

## A.6 Algorithm for Recurrent Event Data

We use the Deep Deterministic Policy Gradient (DDPG) algorithm (20) to optimize the objective function defined in (8) with a continuous action space and use soft-update Deep Q-Network (DQN) to update the discrete action space. It is a model - free, off - policy algorithm that employs an initial network Q and a target network Q ′ . Here we update the value of the Q-function using the optimal Bellman operator and compute the loss (temporal difference error) to adjust the weights of the network Q . Subsequently, we perform a soft update to the weights of the target network Q ′ .

̸

For discrete action space . We define greedy optimal policy π ∗ as any policy that selects the action that produces the highest ˆ Q value. Specifically, for any x ∈ X , π ∗ ( · | x ) satisfies π ∗ ( a | x ) = 0 if ˆ Q π ∗ ( x , a ) = max a ′ ∈A ˆ Q π ∗ ( x , a ′ ) . We denote the Bellman optimality operator as T . To maximize the Q-function, following the strategy of Bellman's equation (3), an estimate of ˆ Q π can be obtained via dynamic programming using the operator T as follows:

<!-- formula-not-decoded -->

It is shown in Appendix B.4 that updating the Q-function using the Bellman operator T can converge to a unique Q-function.

For continuous action space . According to DDPG (20), we define the Policy Function: µ ( θ ) : S → A , a parameterized function (e.g., neural network) that maps states to actions. For a fixed policy µ ( θ ) , the Q-function follows the Bellman evaluation equation: Q µ ( θ ) ( X t , A t ) = E P 0 [ -ˆ Λ T t ( α t G t | X t , A t ) + γQ µ ( θ ) ( X t +1 , µ ( θ )( X t +1 ))] . To optimize µ ( θ ) , we maximize the expected return J ( θ ) = E [ Q µ ( θ ) ( X t , A t )] . Then, parameter updates are performed using the chain rule in DDPG.

One of the key features of DDPG and soft-update DQN is the use of soft updates for the target networks. Rather than directly copying the weights from the current network to the next network, they gradually updates the parameters. This approach enhances learning stability by smoothing out parameter updates, thus preventing large oscillations in parameter values. The proposed estimation procedure can be implemented as outlined in Algorithm 1. For the neural network in simulation, we employ a RL agent with the following hyperparameters: a batch size of 32, a replay buffer of 6400, a learning rate of 0.001, a soft update τ of 0.01, and a discount factor γ of 0.99.

## A.7 Supplementary Experiment

Under identical assumptions for Setting 1 in Section 4, we vary ¯ K to compare the AAEV and AARED metrics across different models, using a sample size of N = 1000 for this analysis. The performance of AAEV and AARED is presented in Table 8 and Table 9, respectively. Because the censoring rates under different ¯ K are also different, we present the changes in censoring rates ( 1 -∑ N 1 i =1 ∏ ¯ K k =1 ∆ i,k /N 1 ) under different Models in Table 10. The results demonstrate that our model consistently outperforms the baseline model across different values of ¯ K while maintaining comparable censoring rates.

Table 8: Mean (Standard Deviation) of AAEV under different ¯ K when N = 1000 in Setting 1

| Model    | 10           | 20           | ¯ K 30       | 40           | 50           |
|----------|--------------|--------------|--------------|--------------|--------------|
| Aah      | 8.45(1.14)   | 13.14(0.68)  | 18.45(0.59)  | 23.99(0.59)  | 29.57(0.48)  |
| Cox      | 9.94 (2.62)  | 15.14 (2.69) | 21.13 (2.97) | 27.29 (3.52) | 33.65 (4.91) |
| Baseline | 16.46 (6.16) | 23.29 (6.53) | 29.97 (6.63) | 37.20 (7.35) | 43.81 (7.54) |

## B Definitions and Proofs of Theoretical Results

## B.1 Proof of Lemma 2.3

We first propose the following Lemma B.1 (the proof is provided in Section B.2).

## Algorithm 1 DDPG for Recurrent Event Data (Continuous and Discrete action space)

Randomly initialize action - value network Q with weights θ Q .

Initialize target network Q ′ with weights θ Q ′ ← θ Q .

## For continuous actions :

- Initialize actor network µ with weights θ µ .
- Initialize target actor network µ ′ with weights θ µ ′ ← θ µ .
- Initialize replay buffer R , set soft update weight τ , exploration noise ϵ , and α k .

for episode = 1 to M do

## For discrete actions :

- Initialize a random process ϵ as explore probability for discrete actions.

## For continuous actions :

- Initialize a noise process N for continuous action exploration.

Receive initial observation state { X i, 1 } N 1 i =1 .

for k = 1 to ¯ K do

## For continuous actions :

- With probability ϵ , select a random action A i,k from the continuous action space perturbed by noise N .
- Otherwise, select A i,k = µ ( X i,k ; θ µ ) + N .

## For discrete actions :

- With probability ϵ , select a random discrete action A i,k .
- Otherwise, select A i,k = arg max A Q ( X i,k , A ; θ Q ) .

Execute action A i,k in emulator and observe Y i,k , ∆ i,k and image X i,k +1 .

Store transition ( X i,t , A i,t , Y i,t , ∆ i,t , α t , X i,t +1 ) in R .

Sample a random minibatch of ( X i,t , A i,t , Y i,t , ∆ i,t , α t , X i,t +1 ) from R .

Estimate ˆ Λ T i,t ( ·| X i,t , A i,t ) with { X i,t , A i,t , Y i,t , ∆ i,t } N t i =1 using Cox or Aah model.

Calculate R i,t = -ˆ Λ T i,t ( α t G t | X i,t , A i,t ) + γ max A ′ Q ′ ( X i,t +1 , A ′ ; θ Q ′ ) .

## Update critic network :

Update θ Q by minimizing the loss: L = 1 N t ∑ N t i =1 ( R i,t -Q ( X i,t , A i,t ; θ Q )) 2 .

## Update actor network (only for continuous actions) :

- Update θ µ to maximize Q ( X i,t , µ ( X i,t ; θ µ ); θ Q ) .

## Update target networks :

<!-- formula-not-decoded -->

## For continuous actions :

- -θ µ ′ ← τθ µ +(1 -τ ) θ µ ′ .

## end for end for

Table 9: Mean (Standard Deviation) of AARED under different ¯ K when N = 1000 in Setting 1

| Model    | ¯ K         | ¯ K          | ¯ K          | ¯ K          | ¯ K          |
|----------|-------------|--------------|--------------|--------------|--------------|
|          | 10          | 20           | 30           | 40           | 50           |
| Aah      | 9.92(0.08)  | 19.89(0.10)  | 29.93(0.11)  | 39.91(0.12)  | 49.90(0.14)  |
| Cox      | 9.91 (0.09) | 19.89 (0.11) | 29.90 (0.12) | 39.90 (0.15) | 49.86 (0.18) |
| Baseline | 9.81 (0.14) | 19.78 (0.16) | 29.78 (0.18) | 39.77 (0.22) | 49.74 (0.21) |

Table 10: Mean (Standard Deviation) of Censoring Rates under ¯ K when N = 1000 in Setting 1

| Model    | 10            | 20            | ¯ K 30        | 40            | 50            |
|----------|---------------|---------------|---------------|---------------|---------------|
| Aah      | 15.1% (0.004) | 29.8% (0.005) | 42.4% (0.004) | 56.8% (0.005) | 71.0% (0.006) |
| Cox      | 15.0% (0.004) | 29.9% (0.007) | 42.7% (0.006) | 56.8% (0.006) | 71.1% (0.006) |
| Baseline | 15.1% (0.005) | 29.6% (0.007) | 42.5% (0.005) | 56.6% (0.007) | 70.6% (0.008) |

Lemma B.1. When Assumptions 2.1 and 2.2 hold, we have

<!-- formula-not-decoded -->

where S T k ( ·| X k , A k ) is the conditional survival function of the duration time T k given X k and A k . Next, we prove Lemma 2.3 as below.

Proof. We define B k = { w | T k ( w ) &gt; α k G k } ⊂ Ω k for k = 1 , . . . , ¯ K , where Ω k is the measurable sample space of stage k . Then we have:

<!-- formula-not-decoded -->

where the second last equality follows from Lemma B.1.

## B.2 Proof of Lemma B.1

Proof. Define B k = { w | T k ( w ) &gt; α k G k } ⊂ Ω k , k = 1 , . . . , ¯ K , where Ω k is the measurable sample space of stage k . Let O k = ( X k , A k ) , then we have

<!-- formula-not-decoded -->

## B.3 Proof of Theorem 2.4

Proof. Let x P 0 k be the state at stage k under policy P 0 . For any deterministic policy P 0 , by definition, there exists a unique x P 0 k +1 such that

<!-- formula-not-decoded -->

where x P 0 t = x t is the initial state. Define Γ P 0 t as the set of all possible trajectories, each consisting of a sequence of states, actions, and rewards from stage t to ¯ K . Specifically,

<!-- formula-not-decoded -->

where T k := T k ( x P 0 k , a k ) is the reward received at stage k given the state-action pair ( x P 0 k , a k ) . Then, for any γ t ∈ Γ P 0 t , we define the reward along the path γ t as

<!-- formula-not-decoded -->

Then we can obtain

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note that ∑ γ t ∈ Γ P 0 t ∏ ¯ K k = t π ( a k | x P 0 k ) = 1 , therefore we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us define

<!-- formula-not-decoded -->

The monotonicity of log( x ) directly implies that Γ v, ∗ t = Γ p, ∗ t , which we then denote as Γ ∗ t . Therefore we have

<!-- formula-not-decoded -->

Define the policy set

<!-- formula-not-decoded -->

For P π ( x t ) , we denote R ∗ = max γ t ∈ Γ P 0 t R ( γ t ) . Then, for γ t / ∈ Γ ∗ t , we have R ( γ t ) &lt; R ∗ . Thus,

<!-- formula-not-decoded -->

It follows that π ∈ Π ∗ ⇐⇒ ∑ γ t ∈ Γ ∗ t P π ( γ t ) = 1 ⇐⇒ P π ( x t ) = R ∗ , i.e., P π ( x t ) = R ( γ ∗ t ) ⇐⇒ π ∈ Π ∗ . Following a similar proof for P π ( x t ) = R ( γ ∗ t ) , we can demonstrate that V π 0 ( x t ) = log R ( γ ∗ t ) holds if and only if π ∈ Π ∗ . Therefore, we conclude that Π v = Π p = Π ∗ .

For the continuous action space, the policy can be expressed as π µ ( a | x ) = δ ( a -µ ( x )) , where δ ( · ) is the Dirac delta function and µ : X → A is a deterministic mapping. Theorem 2.4 also holds for the continuous action space. In the following, we provide an alternative and more direct proof, as shown in B.3.1.

## B.3.1 Proof of Theorem 2.4 for continuous action space

Proof. For a deterministic transition kernel P 0 and a deterministic policy µ , starting from any initial state x t , the next state is uniquely determined by

<!-- formula-not-decoded -->

so that there exists a unique trajectory

<!-- formula-not-decoded -->

Along this trajectory, define the reward as

<!-- formula-not-decoded -->

Then the corresponding objective functions simplify to

<!-- formula-not-decoded -->

Define the set of optimal trajectories as

<!-- formula-not-decoded -->

where the maximization is taken over all possible deterministic policies µ , equivalently over all possible trajectories. Since the logarithm is strictly increasing, maximizing R ( γ µ t ) is equivalent to maximizing log R ( γ µ t ) . Thus,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

which proves the equivalence of the optimal policy sets for the continuous action space.

## B.4 Proof of the coverage of T

Proof. First we proof a statement: we have max a f ( a ) -max a g ( a ) ≤ f ( a ∗ ) -g ( a ∗ ) , and if max a g ( a ) = g ( a ′ ) , we have max a f ( a ) -max a g ( a ) ≥ f ( a ′ ) -g ( a ′ ) . Then get

<!-- formula-not-decoded -->

Next, we proof the γ -contractive of T :

<!-- formula-not-decoded -->

It is shown that the Bellman operator T is γ -contractive with respect to the supremum norm over X × A . Specifically, for any two action-value functions Q and Q ′ defined on X × A , it holds that ∥T Q -T Q ′ ∥ ∞ ≤ γ ∥ Q -Q ′ ∥ ∞ . This contraction property forms the foundation of the well-known value iteration algorithm (41), which constructs a sequence of action-value functions { Q k } k ≥ 0 by iteratively applying T , where Q k = T Q k -1 for all k ≥ 1 , starting from an arbitrary initial function Q 0 . It follows that ∥ Q k -Q π ∗ ∥ ∞ ≤ γ k ∥ Q 0 -Q π ∗ ∥ ∞ , indicating that the sequence { Q k } k ≥ 0 converges to the optimal value function Q π ∗ at a linear rate.

## B.5 Assumption for Cox model

We first define that

<!-- formula-not-decoded -->

Let the cumulative baseline hazard function be Λ 0 ,k ( s ) and let η k be the true parameter at stage k . Assumption B.2. (Finite interval).

<!-- formula-not-decoded -->

Assumption B.3. (Asymptotic stability). There exists a neighborhood B of η k and deterministic bounded limit functions s (0) k , s (1) k , s (2) k : B × [0 , G k ] → R , R p , R p × p such that for j = 0 , 1 , 2 ,

<!-- formula-not-decoded -->

Assumption B.4. (Lindeberg). ∥ Z ik ∥ are uniformly bounded, and there exists ∆ &gt; 0 such that, with time-invariant covariates Z ik ,

<!-- formula-not-decoded -->

Assumption B.5. (Regularity). Let e k , v k be as defined above. Require:

1. For every s ∈ [0 , G k ] ,

<!-- formula-not-decoded -->

and s ( j ) k ( · , s ) are continuous on B and bounded on B × [0 , G k ] for j = 0 , 1 , 2 .

2. s (0) k ( η , s ) is bounded away from zero on B × [0 , G k ] .

3. The matrix

<!-- formula-not-decoded -->

is positive definite.

## B.6 Assumption for Aah model

Assumption B.6 (Bounded covariates) . The covariate vectors are uniformly bounded: ∥ Z ik ∥ ≤ C Z . Assumption B.7 (Risk-set positivity) . There exists a constant c R &gt; 0 such that

<!-- formula-not-decoded -->

in probability.

Assumption B.8 (Baseline hazard and linear predictor bounds) . The baseline hazard function satisfies λ 0 ,k = Λ ′ 0 ,k locally integrable with

<!-- formula-not-decoded -->

Assumption B.9 (Asymptotic positive definiteness) . We define

<!-- formula-not-decoded -->

The matrix A k converges in probability to a positive definite matrix A k, ∞ whose eigenvalues are bounded away from 0 and ∞ .

## B.7 Proof of Theorem 3.1

## B.7.1 Proof of Theorem 3.1 under Cox model

Proof. (i) Consistency . From Theorem 3.2 in Andersen &amp; Gill (2), it holds that ˆ η k p - → η k and sup t ≤ G k ∣ ∣ ˆ Λ 0 ,k ( t ) -Λ 0 ,k ( t ) ∣ ∣ p - → 0 . Under the Cox model, we have

<!-- formula-not-decoded -->

For any t ≤ G k , write

<!-- formula-not-decoded -->

For R 1 k ( t ) , by the mean value theorem and boundedness of Z k ,

<!-- formula-not-decoded -->

for some finite constant C . In addition, under assumption B.4, Λ 0 ,k ( t ) is of bounded variation. That is, TV { Λ 0 ,k ; [0 , G k ] } = Λ 0 ,k ( G k ) &lt; ∞ . Hence

<!-- formula-not-decoded -->

For R 2 k ( t ) , since ˆ η k p - → η k and ∥ Z k ∥ ≤ C Z , the continuous mapping theorem implies exp {∥ ˆ η k ∥ C Z } p - → exp {∥ η k ∥ C Z } . Hence for any ε &gt; 0 there exists M &lt; ∞ such that P ( exp { ˆ η ⊤ k Z k } &gt; M ) &lt; ε for sufficiently large N k . Thus we have

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

(ii) Convergence rate . Applying the first-order Taylor expansion to ˆ Λ T k ( t | Z k ) with respect to ˆ η k , we obtain

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and M ik ( t ) = U ik ( t ) -∫ t 0 V ik ( s ) exp { β ⊤ Z ik } d Λ 0 ,k ( s ) is a local martingale. By Assumption B.3 and B.5(ii),

<!-- formula-not-decoded -->

hence there exist constants C 0 and c 0 , such that S (0) k ( η k , s ) ≥ c 0 / 2 and S ( r ) k ( η k , s ) ≤ C 0 for r = 0 , 1 , 2 and all s ≤ t . Assumption B.4 implies that ∥ Z ik ∥ ≤ C Z , and thus e η ⊤ k Z ik ≤ C e for some C e &lt; ∞ . Assumption B.2 implies that Λ 0 ,k ( t ) ≤ C Λ &lt; ∞ .

(a) Bounded variance of A k ( t, Z k ) . Since { M ik } are orthogonal square integrable martingales with d ⟨ M ik ⟩ ( s ) = V ik ( s ) e η ⊤ k Z ik d Λ 0 ,k ( s ) , we have

<!-- formula-not-decoded -->

Becasue 1 N k ∑ N k i =1 V ik ( s ) e η ⊤ k Z ik = S (0) k ( η k , s ) , hence

<!-- formula-not-decoded -->

(b) Bounded variance of B k ( t ) and scaling of I k ( η k ) -1 B k ( t ) . By definition,

<!-- formula-not-decoded -->

Since { M ik } are orthogonal square integrable martingales with d ⟨ M ik ⟩ ( s ) = V ik ( s ) e η ⊤ k Z ik d Λ 0 ,k ( s ) , we have

<!-- formula-not-decoded -->

Let ˜ I k ( η k ) := I k ( η k ) /N k . By Assumption B.5(iii), ˜ I k ( η k ) p - → Σ k ≻ 0 . Therefore, for sufficiently large N k , there exists a constant C I &lt; ∞ such that

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

which is uniformly bounded.

(c) Boundedness of H k ( t, Z k ) and variance of H k ( t, Z k ) ⊤ I -1 k ( η k ) B k ( t ) . Since ∥ Z k ∥ ≤ C Z and ∥ ¯ Z k ( η k , s ) ∥ ≤ 2 C 0 /c 0 by Assumptions B.3-B.5, there exists a C H &lt; ∞ , such that

<!-- formula-not-decoded -->

Then we conclude

<!-- formula-not-decoded -->

which is also uniformly bounded.

(d) Boundedness of Cov ( A k ( t, Z k ) , B k ( t )) . We have

<!-- formula-not-decoded -->

Combining (a)-(d), there exists a C W &lt; ∞ such that,

<!-- formula-not-decoded -->

Thus, by Chebyshev inequality, P (sup t | W k ( t, Z k ) | &gt; M ) ≤ C W /M 2 for all M &gt; 0 , i.e., sup t | W k ( t, Z k ) | = O p (1) , then we have ˆ Λ T k ( t | Z k ) -Λ T k ( t | Z k ) = O p ( N -1 2 k ) .

(iii) Asymptotic normality . Following from the martingale central limit theorem, it can be obtained that the process W k ( t, Z k ) converges weakly to a zero-mean Gaussian process on [0 , G k ] . Next we calculate the variance of the process. Since A.2, A.3, A.4, and I k is a symmetric matrix, we have

<!-- formula-not-decoded -->

Thus W k ( t, Z k ) converges weakly to a zero-mean Gaussian process with covariance given by (A.5). Replace all population quantities by their sample counterparts, a consistent estimator of (A.5) is

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.7.2 Proof of the Aalen Additive Hazards Model Estimator

Proof. (i) Consistency . According to Lin &amp; Ying (21), we have

<!-- formula-not-decoded -->

Under assumption B.6,

<!-- formula-not-decoded -->

(ii) Convergence rate . Let us define d Λ 0 ,k ( s ) = λ 0 ,k ( s ) ds . According to assumptions B.6-B.9, we have (i) R k ( s ) ≥ c R N k , (ii) ∥ Z ik ∥ ≤ C Z , (iii) λ 0 ,k ( s ) + β ⊤ k Z ik ≤ C λ , (iv) c A I d ⪯ A k , with finite strictly positive constants c R , C Z , C λ , c A . According to Lin &amp; Ying (21), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where M ik ( s ) = U ik ( s ) -∫ s 0 V ik ( u ) { λ 0 ,k ( u ) + β ⊤ k Z ik } du is a local martingale, and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, for the cumulative hazard at covariate Z k ,

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Since d ⟨ M ik ⟩ ( s ) = V ik ( s ) { λ 0 ,k ( s ) + β ⊤ k Z ik } ds ≤ C λ ds ,

<!-- formula-not-decoded -->

According to assumptions B.6 and B.9, we have ∥ A -1 k ∥ ≤ 1 c A and ∥ ˜ Z k ( s ) ∥ ≤ C Z . Then

<!-- formula-not-decoded -->

hence

<!-- formula-not-decoded -->

By the covariance formula for martingale integrals and Cauchy-Schwarz inequality,

<!-- formula-not-decoded -->

There exists a C W &lt; ∞ (independent of N k ) such that

<!-- formula-not-decoded -->

By Chebyshev's inequality,

<!-- formula-not-decoded -->

(iii) Asymptotic normality. Following from the martingale central limit theorem, it can be obtained that the process W k ( t, Z k ) converges weakly to a zero-mean Gaussian process on [0 , G k ] . Next we calculate the variance of the process.

(a) Variance of A (2) k ( t, Z k ) . Let O k ( t, Z k ) = t Z k -C k ( t ) , we have

<!-- formula-not-decoded -->

Define the matrix-valued process

<!-- formula-not-decoded -->

Then according to Lemma B.10, we have

<!-- formula-not-decoded -->

(b) Covariance of A (1) k ( t ) and A (2) k ( t ) . Again by orthogonality across i and the martingale isometry,

<!-- formula-not-decoded -->

Let

By Lemma B.10, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

A consistent estimator of V ar ( W k ( t, Z k )) is obtained by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

## B.8 The objective function for Liu et al.'s (2023) method

Next we will show that the objective function in RL from Liu et al.'s (2023) is

<!-- formula-not-decoded -->

where ˆ S C is the Kaplan-Meier estimator.

Proof. Note that

<!-- formula-not-decoded -->

and thus

E   ∆ k S C ( ∑ k i =1 T i ) ∣ ∣ ∣ ∣ ∣ ∣ x k , a k , T k   = 1 . Then we have E P 0 ,π   ¯ K ∑ k = t γ k -t T k ∣ ∣ ∣ ∣ ∣ ∣ x t , a t   = E P 0 ,π    ¯ K ∑ k = t γ k -t T k E   ∆ k S C ( ∑ k i =1 T i ) ∣ ∣ ∣ ∣ ∣ ∣ x k , a k , T k   ∣ ∣ ∣ ∣ ∣ ∣ x k , a k    = E P 0 ,π    ¯ K ∑ k = t γ k -t E   T k ∆ k S C ( ∑ k i =1 T i ) ∣ ∣ ∣ ∣ ∣ ∣ x k , a k , T k   ∣ ∣ ∣ ∣ ∣ ∣ x t , a t    = E P 0 ,π   ¯ K ∑ k = t γ k -t Y k ∆ k S C ( ∑ k i =1 Y i ) ∣ ∣ ∣ ∣ ∣ ∣ x t , a t   .

## B.9 Large-sample replacement of d ⟨ M ⟩ by empirical dU

Lemma B.10. Fix a stage k and subjects i = 1 , . . . , N k . Let U ik ( t ) be counting processes with predictable intensities λ ik ( t ) and at-risk indicators V ik ( t ) . Assume:

- (A1) (Doob-Meyer) U ik ( t ) = Λ ik ( t ) + M ik ( t ) with Λ ik ( t ) = ∫ t 0 V ik ( s ) λ ik ( s ) ds ; hence d ⟨ M ik ⟩ ( s ) = d Λ ik ( s ) = V ik ( s ) λ ik ( s ) ds .
- (A2) (Orthogonality/independence across i ) The martingales { M ik } i ≤ N k are pairwise orthogonal (e.g. subjects independent).
- (A3) (Bounded predictable weights) For each i , H ik ( s ) is predictable, sup s ≤ G k | H ik ( s ) | ≤ C H &lt; ∞ .

(A4) (Integrability) sup i,s ≤ G k λ ik ( s ) ≤ C λ &lt; ∞ .

Define, for t ∈ [0 , G k ] ,

<!-- formula-not-decoded -->

Then:

<!-- formula-not-decoded -->

In particular, for any bounded predictable H ik ,

<!-- formula-not-decoded -->

Proof. By (A1), U ik = Λ ik + M ik and d ⟨ M ik ⟩ = d Λ ik , hence

<!-- formula-not-decoded -->

Using orthogonality in (A2) and the isometry for martingale integrals,

<!-- formula-not-decoded -->

Thus R N k ( t ) p - → 0 for each fixed t . For the uniform version, Doob's L 2 inequality yields

<!-- formula-not-decoded -->

Remark B.11 . If we define d Λ 0 ,k ( s ) = λ 0 ,k ( s ) ds , then in the Cox PH model, λ ik ( s ) = λ 0 ,k ( s ) e η ⊤ k Z ik so d ⟨ M ik ⟩ ( s ) = V ik ( s ) e η ⊤ k Z ik d Λ 0 ,k ( s ) ; In the Aalen additive model, λ ik ( s ) = λ 0 ,k ( s ) + β ⊤ k Z ik so d ⟨ M ik ⟩ ( s ) = V ik ( s ) { λ 0 ,k ( s ) + β ⊤ k Z ik } ds . Lemma B.10 justifies replacing these predictable quadratic variations by the corresponding empirical averages of dU ik in large samples when evaluating variances/covariances of martingale integral terms.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the goal of maximizing recurrent event probabilities using RL, the development of an MDP framework, and theoretical/experimental contributions, aligning with the content in Sections 3-7.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [No]

Justification: The paper focuses on methodology and experiments but does not explicitly address limitations such as computational complexity or the impact of assumptions (e.g., Markov property) on real-world applicability.

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

Justification: Theoretical results (e.g., Theorem 3.2, Lemma 4.1) are supported by assumptions (Markov property, independence of estimation errors) with detailed proofs in the appendices (e.g., A.7, A.10).

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

Justification: Simulation setups (Section 5.1), metrics (Section 5.2), and algorithms (Appendix A.5) are described in detail, enabling result reproduction.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in

some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The code is available at https://github.com/fjqfengjianqi/ NIPS2025-RL-for-Survival .

## Guidelines:

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

Justification: Hyperparameters (e.g., discount factor γ , soft-update DQN/DDPG), sample sizes, and stage configurations are detailed in Sections 5.1 and Appendix A.5.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Error bars (standard deviation) and statistical tests are reported in Tables 1-2 and Figures 2-3, with clear definitions of metrics like ARED and AEV.

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

Answer: [No]

Justification: The paper does not specify hardware (e.g., CPU/GPU type, memory) or execution time for experiments, focusing only on algorithmic details.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research involves ethical analysis of medical data with proper anonymization (e.g., Intern Health Study), adhering to standards for secondary data use.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper focuses on methodological advancements in precision medicine and does not address broader societal impacts like privacy or clinical bias.

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

Justification: No high-risk models/datasets are released; the work focuses on theoretical/experimental methods without deployment-specific safeguards.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Datasets (Intern Health Study) and models (Cox/Aalen) are cited (e.g., NeCamp et al., 2020), though specific licenses for real-world data are not mentioned.

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

Justification: No new datasets/models are released; the work relies on existing data and standard RL frameworks.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The study uses existing medical records (non-interventional) and simulations, not direct human subject participation.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No direct human subjects research is conducted; ethical handling of existing data is assumed, but IRB specifics are not detailed.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs are not used in the core methodology; the work relies on traditional RL and survival analysis techniques.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.