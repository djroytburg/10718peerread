## Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization

Frank Röder ∗ Jan Benad Manfred Eppe Pradeep Kr. Banerjee ∗ Institute for Data Science Foundations, Blohmstraße 15, 21079 Hamburg, Germany

## Abstract

Real-world reinforcement learning demands adaptation to unseen environmental conditions without costly retraining. Contextual Markov Decision Processes (cMDP) model this challenge, but existing methods often require explicit context variables (e.g., friction, gravity), limiting their use when contexts are latent or hard to measure. We introduce Dynamics-Aligned Latent Imagination (DALI), a framework integrated within the Dreamer architecture that infers latent context representations from agent-environment interactions. By training a self-supervised encoder to predict forward dynamics, DALI generates actionable representations conditioning the world model and policy, bridging perception and control. We theoretically prove this encoder is essential for efficient context inference and robust generalization. DALI's latent space enables counterfactual consistency: Perturbing a gravity-encoding dimension alters imagined rollouts in physically plausible ways. On challenging cMDP benchmarks, DALI achieves significant gains over contextunaware baselines, often surpassing context-aware baselines in extrapolation tasks, enabling zero-shot generalization to unseen contextual variations.

## 1 Introduction

The ability to generalize across diverse environmental conditions is a fundamental challenge in reinforcement learning (RL). Contextual Markov Decision Processes (cMDPs) formalize this challenge by modeling task variations through latent parameters, such as friction, gravity, or object mass that govern dynamics [Hallak et al., 2015]. However, real-world agents rarely have direct access to these parameters. Consider a robotic control task where a legged agent must adapt to different surfaces, such as smooth tiles, rough gravel, or slippery ice. If the agent were explicitly provided with surface friction coefficients, adaptation would be straightforward. In practice, though, such ground-truth annotations are often unavailable or prohibitively expensive to obtain. Instead, the agent must infer these variations implicitly by observing how its actions influence the environment, using proprioceptive feedback and gait dynamics.

Existing approaches to contextual RL often rely on explicit context conditioning, where models are trained with domain-specific instrumentation or carefully crafted architectures [Seyed Ghasemipour et al., 2019, Ball et al., 2021, Eghbal-zadeh et al., 2021, Mu et al., 2022, Beukman et al., 2023, Benjamins et al., 2023, Prasanna et al., 2024]. While effective in controlled settings, such approaches scale poorly and break down in unstructured environments where contexts are latent or difficult to measure. The challenge is to develop RL agents that infer and adapt to hidden contexts in a self-supervised manner, enabling robust generalization without direct supervision.

To address this challenge, we propose Dynamics-Aligned Latent Imagination (DALI), a framework integrated within the DreamerV3 architecture [Hafner et al., 2025] that enables zero-shot generalization by inferring latent contexts directly from interaction histories. Unlike DreamerV3, which struggles

∗ Equal contribution

to generalize across diverse latent contexts due to its limited ability to retain critical environmental information [Prasanna et al., 2024], DALI overcomes these limitations through a self-supervised context encoder. This encoder learns to predict forward dynamics, capturing the relationship between actions, states, and their consequences. For example, by analyzing how applied forces influence motion trajectories, the encoder distills contextual factors (e.g., gravitational pull or surface friction) into compact, actionable embeddings. These embeddings condition the world model and policy, enabling the agent to reason about hidden dynamics (e.g., how increased inertia affects motion stability) and adapt its control strategies accordingly. Our theoretical analysis demonstrates that DALI's context encoder enables robust zero-shot generalization by efficiently capturing latent environmental variations, overcoming information bottlenecks and sample inefficiencies in learning diverse contexts [Prasanna et al., 2024, Hafner et al., 2025].

By inferring latent contexts from interactions, DALI enhances DreamerV3's learning framework, achieving robust zero-shot generalization in partially observable environments. Our work makes the following key contributions:

- Theoretical foundations : We establish that a dedicated context encoder is essential for robust generalization, proving that it infers latent contexts from short interaction windows with nearoptimal sample complexity and mitigates information bottlenecks in partially observable settings.
- Strong zero-shot generalization : On challenging cMDP benchmarks, DALI achieves up to +96 . 4% gains over context-unaware baselines, often surpassing ground-truth context-aware baselines in extrapolation tasks.
- Physically consistent counterfactuals : We show that the learned latent space exhibits counterfactual consistency: perturbing a dimension encoding gravity, for instance, results in imagined rollouts where objects fall faster or slower, faithfully mirroring real-world physics.

## 2 Related Work

Contextual RL for Zero-Shot Generalization. Contextual RL has been studied in various forms, from cMDPs to domain randomization and meta-RL [Hallak et al., 2015, Modi et al., 2018]. A recent survey [Kirk et al., 2023] highlights its relevance for zero-shot generalization, emphasizing how clear context sets for training and evaluation enable systematic analysis. One research direction assumes context is explicitly observed as privileged information and integrates it into learning [Chen et al., 2018, Seyed Ghasemipour et al., 2019, Ball et al., 2021, Eghbal-zadeh et al., 2021, Sodhani et al., 2021, Mu et al., 2022, Benjamins et al., 2023, Prasanna et al., 2024]. In contrast, we follow recent work that assumes context is latent and must be inferred [Chen et al., 2018, Xu et al., 2019, Lee et al., 2020, Seo et al., 2020, Xian et al., 2021, Sodhani et al., 2022, Melo, 2022, Evans et al., 2022], focusing on self-supervised context inference through forward dynamics alignment.

Model-Based RL. Model-based RL improves sample efficiency by learning predictive environment models for planning and imagination. Approaches like Dreamer [Hafner et al., 2019, 2020, 2021, 2025] and TD-MPC [Hansen et al., 2024] achieve strong performance by learning compact latent representations of environment dynamics. Recent work has also explored general-purpose model-free RL that leverages model-based representations for broader applicability [Fujimoto et al., 2025]. While Dreamer has been explored for zero-shot generalization with explicit context conditioning [Prasanna et al., 2024], we build on it with a dynamics-aligned inference mechanism to enhance generalization without context supervision.

Meta-RL. Meta-RL trains agents to adapt rapidly to new tasks with minimal experience [Beck et al., 2023], typically by learning adaptive policies that infer task-specific information from past interactions. However, most meta-RL methods require fine-tuning on new tasks [Duan et al., 2016, Finn et al., 2017, Nagabandi et al., 2018, Rakelly et al., 2019, Zintgraf et al., 2019, Melo, 2022]. Our approach, in contrast, aims for zero-shot generalization by leveraging structured latent representations that transfer across environments.

## 3 Preliminaries

To investigate zero-shot generalization in partially observable environments, we adopt a contextual Markov Decision Process (cMDP) framework, following Hallak et al. [2015], Kirk et al. [2023].

A cMDP is defined as a tuple M = ( S , A , O , C , R , P , E , µ, p C ) , where S , A , and O are resp. the state, action, and observation spaces, C ⊆ R d is the context space with contexts c ∈ C drawn i.i.d. from a distribution p C , R : S × A × C → R is the reward function, P : S × A × C → ∆( S ) is the stochastic transition function specifying the probability distribution over next states given the current state, action, and context, E : S × C → ∆( O ) is the observation function specifying the distribution over observations given the state and context, µ : C → ∆( S ) is the initial state distribution where µ ( s 0 | c ) specifies the probability of initial state s 0 given context c . In this partially observable setting, the agent does not observe the state s t or context c directly but receives observations o t ∈ O . The objective is to learn a policy π ( a t | o 1: t , a 1: t -1 ) that maximizes the expected return E [ ∑ T t =0 r t ] , where T is the episode length.

In our framework, each context c induces a distinct variation in the transition dynamics P , such as changes in physical properties (e.g., gravity, friction), while the reward function R remains fixed across contexts. The context is latent and remains fixed within an episode but varies across episodes according to p C . The agent infers c from interaction histories to adapt its policy. To formalize zero-shot generalization, we define two distributions over contexts: the training distribution p train ( c ) , from which contexts are sampled during training, and the evaluation distribution p eval ( c ) , representing unseen test contexts. The goal is to learn a policy using samples from p train ( c ) that maximizes the expected return for contexts drawn from p eval ( c ) without further adaptation or retraining.

## 4 Dynamics-Aligned Latent Imagination (DALI)

## 4.1 Background: DreamerV3

DreamerV3 [Hafner et al., 2025] is a model-based RL algorithm that enables agents to learn and plan in the latent space of a learned world model. It follows the general structure of the Dreamer family of algorithms [Hafner et al., 2019, 2020, 2021], incorporating three key components: (i) learning a generative world model to simulate environment dynamics, (ii) optimizing policies entirely within the latent space of the world model using imagined rollouts, and (iii) refining the world model and policy through interaction with the real environment.

World model. The agent interacts with the environment through observations o t (e.g., images or sensor data) and actions a t . The world model, structured as a Recurrent State-Space Model (RSSM), encodes these observations into a compact latent state s t = { h t , z t } , where h t = f θ ( h t -1 , z t -1 , a t -1 ) is a deterministic recurrent state capturing temporal dependencies, and z t is a stochastic state encoding uncertainty about the current observation. The separation of deterministic ( h t ) and stochastic ( z t ) states enables long-horizon temporal reasoning while maintaining diversity in imagined rollouts. The RSSM operates in two distinct modes: (i) during training, the model conditions on the current observation o t to infer z t , ensuring alignment with real-world dynamics (posterior inference); and (ii) during imagination, the model predicts future states without access to observations, sampling ˆ z t to generate hypothetical trajectories (prior prediction). The world model reconstructs observations ˆ o , predicts rewards ˆ r t , and estimates episode continuations ˆ n t from the latent state s t . These components are learned via the following structured models: posterior state representations (encoder) z t ∼ q θ ( z t | h t , o t ) , prior state representations ˆ z t ∼ p θ (ˆ z t | h t ) , reward predictor ˆ r t ∼ p θ (ˆ r t | h t , z t ) , continue predictor ˆ n t ∼ p θ (ˆ n t | h t , z t ) , and decoder ˆ o t ∼ p θ (ˆ o t | h t , z t ) .

Learning in imagination. Once the world model is trained, behavior is learned by optimizing a policy entirely within the latent space. An actor-critic architecture guides this process, with the critic estimating the cumulative future reward (value v ψ ) and the actor selecting actions a τ ∼ π ϕ to maximize this value. By decoupling policy learning from real-world interaction, DreamerV3 achieves strong sample efficiency, iteratively refining its world model and behaviors through cycles of imagination and execution.

## 4.2 Context Encoder for Dynamics-Aligned Representations

While DreamerV3 is effective in fixed environments, its reliance on static latent states limits adaptability to contextual variations, such as changes in gravity or friction. To address this, we introduce Dynamics-Aligned Latent Imagination (DALI), which extends DreamerV3 with a self-supervised context encoder that learns explicit context representations from interaction histories.

## 4.2.1 Forward Dynamics Alignment

The context encoder g φ maps a history of observations and actions ( o t -K : t , a t -K : t -1 ) to a context representation z t = g φ ( o t -K : t , a t -K : t -1 ) , where K defines the temporal window. To align z t with environmental transition dynamics, we optimize a forward dynamics loss:

<!-- formula-not-decoded -->

where f w φ ( o t , a t , z t ) predicts the next observation ˆ o t +1 given the current observation o t , action a t , and context z t . Minimizing L FD ensures that z t encodes contextual factors critical for accurate dynamics prediction, such as variations in physical parameters. The parameters φ of g φ and f w φ are trained jointly, enabling the encoder to capture essential dynamics from short interaction histories.

## 4.2.2 Cross-Modal Regularization

To enhance the context encoder's robustness, we introduce a cross-modal regularization that aligns the context representation z t = g φ ( o t -K : t , a t -K : t -1 ) with the DreamerV3 world model's posterior state z t ∼ q θ ( z t | h t , o t ) . The cross-modal loss enforces bidirectional alignment between z t and z t :

where W z and W z are linear maps between the context and state spaces. This bidirectional reconstruction leverages z t 's observation-informed representation, which captures instantaneous dynamics relevant to the current context. By aligning with z t instead of the full latent state s t = { h t , z t } , z t avoids encoding redundant trajectory-specific information from the deterministic h t , which could impair generalization. The bidirectional constraints prevent degenerate solutions (e.g., z t collapsing to a constant) by enforcing invertibility, ensuring z t remains a rich, context-specific representation consistent with the z t 's latent dynamics. The total loss combines both objectives:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where λ cross ∈ { 0 , 1 } balances dynamics prediction and cross-modal alignment.

## 4.3 Integrating Context into DreamerV3

To enable context-conditioned imagination and policy learning in DreamerV3 without access to ground-truth context variables, we propose two integration strategies for incorporating the context representation z t .

Shallow Integration. This approach appends z t to the observation embedding in the world model's encoder, modifying it to z t ∼ q θ ( z t | h t , o t , z t ) . All other world model components, including the sequence model h t = f θ ( h t -1 , z t -1 , a t -1 ) , predictors ( ˆ z t , ˆ r t , ˆ n t , ˆ o t ), and actor-critic networks ( π ϕ ( a τ | s τ ) , v ψ ( s t ) ), remain unchanged. This lightweight modification enriches latent states with dynamics-aware context, enabling the world model to adapt implicitly to contextual variations.

Deep Integration. For deeper context awareness, z t is integrated into the world model and policy as follows: h t = f θ ( h t -1 , z t -1 , a t -1 , z t ) , ˆ r t ∼ p θ (ˆ r t | h t , z t , z t ) , ˆ n t ∼ p θ (ˆ n t | h t , z t , z t ) , ˆ o t ∼ p θ (ˆ o t | h t , z t , z t ) . The posterior z t ∼ q θ ( z t | h t , o t ) and prior ˆ z t ∼ p θ (ˆ z t | h t ) access z t indirectly through h t . The actor and critic explicitly condition on z t to optimize imagined trajectories over horizon H : a τ ∼ π ϕ ( a τ | s τ , z t ) , v ψ ( s t , z t ) ≈ E π ( ·| s τ , z t ) ∑ t + H τ = t γ τ -t r τ . This ensures that policy optimization explicitly accounts for contextual variations, enabling context-conditioned imagination.

Training. Training the context encoder, the loss (3) aligns z t = g φ ( o t -K : t , a t -K : t -1 ) with environmental dynamics and requires careful handling of the DreamerV3 world model's recurrence. Both integration strategies unroll the world model over a K -length window, initializing h t -K = 0 (or a learned initial state), generating z τ = g φ ( o τ -K : τ , a τ -K : τ -1 ) for τ = t -K to t . In Shallow Integration, z τ ∼ q θ ( z τ | h τ , o τ , z τ ) and h τ +1 = f θ ( h τ , z τ , a τ ) ; in Deep Integration, z τ ∼ q θ ( z τ | h τ , o τ ) and h τ +1 = f θ ( h τ , z τ , a τ , z τ ) . For both, gradients through h τ and z τ are stopped in the recurrent dynamics, and through h τ in the world model's encoder, preventing updates to θ . In Shallow Integration, z τ gradients are preserved in the world model's encoder and losses L FD and L cross , allowing updates to φ , W z , and W z . In Deep Integration, z τ gradients are stopped in the recurrent dynamics and preserved only in L FD and L cross . This decouples context learning from the world model's recurrence, ensuring z t captures relevant dynamics for both strategies. For detailed pseudocode, see Algorithms 1 and 2 in Appendix B for Shallow Integration, and Algorithms 3 and 4 for Deep Integration.

## 5 Theoretical Insights into DALI's Context Encoder

In this section, we provide an exposition of our key theoretical results, elucidating why DALI's context encoder is essential for robust generalization. We emphasize conceptual insights, with formal statements and proofs deferred to Appendix A.

Our analysis is grounded in a cMDP framework with continuous contexts c (e.g., physical parameters such as gravity or actuator strength), sampled i.i.d. per episode from a distribution p C . Observations are noisy (e.g., o t = s t + η t , η t ∼ N (0 , σ 2 I ) ), and dynamics are assumed to be Lipschitz continuous, so that small changes in context (e.g., slight shifts in gravity) lead to smoothly varying behaviors (e.g., comparable object trajectories or locomotion patterns). The observation-action process is β -mixing [Rio, 2017], meaning distant observations become nearly independent over time (e.g., periodic motions or gait cycles decorrelate under exploratory actions). These structural assumptions are characteristic of many continuous control environments such as DMC [Tassa et al., 2018], CARL [Benjamins et al., 2023], and MetaWorld [Yu et al., 2020], and they underpin DALI's efficient context inference. β -mixing captures realistic decorrelation in RL tasks, enabling short interaction windows to capture c . An exploratory policy further ensures that distinct contexts yield distinguishable trajectories, a common feature in RL training.

Domain randomization trains DreamerV3 across diverse contexts, with c ∼ p C , using h RSSM t = f θ ( h t -1 , z t -1 , a t -1 ) and z t ∼ q θ ( z t | h t , o t ) . The recurrent state h RSSM t implicitly accumulates information about c over time through its interaction history. For instance, in environments with pendulum-like dynamics, increased gravity may result in faster oscillatory behavior, a pattern that h RSSM t can gradually encode after observing enough transitions. However, this strategy has limitations. The recurrent state h RSSM t compresses all episode information, including context, states, and noise into a fixed-size GRU, introducing an information bottleneck. As dynamic state information (e.g., object motion) and transient noise compete for limited capacity, essential cues about the underlying context (e.g., gravity) may be lost or delayed. Identifying c often requires accumulating evidence across many transitions, potentially spanning an entire episode of length T , since early actions may not sufficiently excite the dynamics to reveal contextual differences. This delay hinders rapid adaptation to novel contexts, as h RSSM t needs a long temporal window to reliably disambiguate c .

DALI's context encoder decouples context inference from dynamics modeling. It functions as a specialized module that learns a representation z t capturing context information from short, local histories, allowing its recurrent state h DALI t to focus on dynamics and reducing the burden on the GRU. The β -mixing property ensures that K transitions suffice to encode c , as the dynamics' dependence on past states fades rapidly. This enables faster adaptation compared to DreamerV3's h RSSM t . We formalize this intuition for the case of Deep Integration, where the context encoder enhances the RSSM's context awareness. The functions h ( · ) and I ( · ; · ) denote the differential entropy and mutual information, respectively [Thomas and Cover, 1999].

Theorem 1. In a cMDP with β -mixing and Lipschitz dynamics, DALI's context encoder z t captures near-optimal context information, I ( c ; z t ) ≥ (1 -δ ) h ( c ) for δ ∈ (0 , 1) , using N = O (1 /δ 2 ) windows of K = Ω(log(1 /δ ) /λ ) transitions, where λ is the mixing rate. Moreover, DALI's RSSM retains more context information than DreamerV3's, I ( c ; h DALI t ) ≥ I ( c ; h RSSM t ) -ϵ ( K ) , with ϵ ( K ) = O ( e -λK/ 2 ) . Compared to DreamerV3's RSSM processing full episodes of T ≫ K transitions, DALI achieves a sample complexity gain of O ( T/K ) .

For the formal statement and proof of Theorem 1, see Appendix A. The β -mixing property ensures that a short window of K = Ω(log(1 /δ ) /λ ) transitions is sufficient to achieve high context fidelity. For instance, when δ = 0 . 01 and the typical mixing rate in DMC tasks is λ ≈ 0 . 1 , a window of K ≈ 64 steps results in a conditional entropy error bounded by δ ′ ≈ 0 . 1 . DALI requires N = O (1 /δ 2 ) such windows, totaling O ( K/δ 2 ) transitions. In contrast, DreamerV3's h RSSM t , constrained by the GRU's finite capacity and observation noise, loses context information and requires O ( T/δ 2 ) transitions per episode (e.g., T = 1000 ), where T ≫ K . The sample complexity gain of O ( T/K ) reflects DALI's efficiency in exploiting local histories. Furthermore, z t achieves superior fidelity, with I ( c ; z t ) approaching h ( c ) within δ , while DreamerV3's RSSM incurs a larger information loss, ϵ ( K ) = O ( e -λK/ 2 ) , due to its bottleneck. While the theorem assumes Deep Integration, this sample complexity gain remains consistent across Deep and Shallow Integration, as z t 's efficiency derives from its training on K -length windows, which is identical in both configurations. These results highlight the necessity and efficiency of DALI's context encoder in enabling robust generalization.

## 6 Experiments and Analysis

In Sections 6.1 and 6.2, we evaluate DALI's ability to generalize in a zero-shot manner across unseen context variations. In Section 6.3, we show that the dynamics-aligned context encoder learns a structured latent representation, where perturbations to individual dimensions produce physically plausible counterfactuals (e.g., shorter ball swings for higher imagined gravity). Our code is available at https://github.com/frankroeder/DALI .

## 6.1 Zero-Shot Generalization

We evaluate DALI's zero-shot generalization on contextualized DMC Ball-in-Cup and Walker Walk tasks from the CARL benchmark [Benjamins et al., 2023].

Methods. Our context-unaware methods, denoted DALI-S/Dχ /blank, employ either Shallow (S) or Deep (D) integration with the inferred context z t (see Section 4.3). These models are trained using either the forward dynamics loss alone (1) (denoted as 'blank', e.g., DALI-D), or the cross-modal regularized objective (2) (denoted as χ , e.g., DALI-Sχ ). We compare our models to the contextunaware baseline Dreamer-DR, which corresponds to DreamerV3 with domain randomization [Tobin et al., 2017]. We also evaluate against the context-aware baselines cRSSM-S and cRSSM-D, which use the same world model and actor-critic architecture but directly incorporate the ground-truth context c [Prasanna et al., 2024]. For further details, see Appendix A.1.

Setup. We train our methods using the small variant of DreamerV3 [Hafner et al., 2025], with a transformer-based context encoder [Vaswani et al., 2017], for 200K timesteps (Ball-in-Cup) or 500K timesteps (Walker) across 10 random seeds, following the setup of Prasanna et al. [2024]. Hyperparameters and architectural details are provided in Appendix C.

We conduct evaluation on two observation modalities: Featurized , which provides structured state inputs with low partial observability, and Pixel , which requires latent state estimation from raw images. We assess performance under three generalization regimes [Kirk et al., 2023]: Interpolation (contexts within the training range), Extrapolation (OOD contexts beyond the training range), and Mixed (one context OOD, one within training range). The Extrapolation regime tests generalization beyond seen contexts, requiring agents to capture underlying physical principles. The Mixed regime probes the ability to interpolate and recombine learned representations. Together, these settings reveal whether the agent learns meaningful abstractions or simply memorizes specific contexts.

Context ranges. For Ball-in-Cup, the context parameters are gravity (training: [4 . 9 , 14 . 7] , evaluation: [0 . 98 , 4 . 9) ∪ (14 . 7 , 19 . 6] , default: 9.81) and string length (training: [0 . 15 , 0 . 45] , evaluation: [0 . 03 , 0 . 15) ∪ (0 . 45 , 0 . 6] , default: 0.3). For Walker, the parameters are gravity (same ranges as Ball-in-Cup) and actuator strength (training: [0 . 5 , 1 . 5] , evaluation: [0 . 1 , 0 . 5) ∪ (1 . 5 , 2 . 0] , default: 1.0). These OOD ranges, particularly in Ball-in-Cup, pose major generalization challenges, while Walker's actuator strength remains closer to training. Full training settings, including single and dual context variations, are in Appendix C.

Evaluation Metrics. To evaluate zero-shot generalization, we use the Interquartile Mean (IQM) and Probability of Improvement (PoI) metrics from the rliable framework [Agarwal et al., 2021]. IQM averages the central 50% of performance scores, reducing the impact of outliers prevalent in RL due to high variance in training dynamics, particularly in OOD cMDP contexts (e.g., extreme gravity in Ball-in-Cup). We ensure statistical reliability by computing stratified bootstrap 95% confidence intervals over seeds and aggregated contexts. PoI quantifies the probability that a randomly selected run of one algorithm outperforms a randomly selected run of another, offering a robust comparative metric across methods and modalities.

Results. We report IQM and PoI across 10 seeds in Figure 1, comparing DALI-S, DALI-Sχ , Dreamer-DR, cRSSM-S, and cRSSM-D for Featurized and Pixel observations on the Ball-in-Cup and Walker Walk tasks.

In Ball-in-Cup's Interpolation, all methods achieve high IQM ( 0 . 92 -0 . 95 ), with DALI-Sχ competitive ( 0 . 9490 Featurized, 0 . 9440 Pixel). In Extrapolation, DALI-Sχ excels, outperforming Dreamer-DR by 87 . 9% (Featurized, IQM 0 . 3720 ) to 96 . 4% (Pixel, IQM 0 . 2730 ), surpassing cRSSMS by 63 . 9% (Featurized) and 45 . 9% (Pixel), and cRSSM-D by 33 . 8% (Featurized) and 12 . 8% (Pixel). This suggests ground-truth context may overfit, limiting OOD adaptability. Low absolute IQM scores

Figure 1: Interquartile Mean (IQM) scores and Probability of Improvement (PoI) for the DMC Ballin-Cup and Walker Walk tasks under contextual variations: gravity and string length for Ball-in-Cup, and gravity and actuator strength for Walker Walk. Results are shown for Featurized and Pixel observations in each environment. Scores aggregate across single and combined contexts. Shaded intervals represent 95% stratified bootstrap confidence intervals over seeds and aggregated contexts. The rightmost panel in each plot displays PoI in the Extrapolation regime for DALI-Sχ (Ball-in-Cup) and DALI-S (Walker Walk), relative to baseline methods.

<!-- image -->

( 0 . 14 -0 . 37 ) reflect the extreme OOD context ranges, such as gravity values of 0 . 98 or 19 . 6 , far from the training range of [4 . 9 , 14 . 7] . In the Mixed regime, DALI-Sχ leads in Featurized modality ( 51 . 1% over Dreamer-DR, 20 . 3% over cRSSM-S, 1 . 9% over cRSSM-D, IQM 0 . 6830 ), while cRSSM-D excels in Pixel (IQM 0 . 6250 ), leveraging ground-truth context for visual dynamics. DALI-Sχ shows moderate PoI ( &gt; 0 . 6 ) against Dreamer-DR in both modalities for Extrapolation (see Figure 1).

In Walker's Interpolation, methods score high ( 0 . 94 -0 . 97 ), with DALI-S leading ( 0 . 9710 Featurized). In Extrapolation, DALI-S achieves 4 . 0% over Dreamer-DR (Featurized, IQM 0 . 7810 ), 11 . 3% over cRSSM-S, and 4 . 3% over cRSSM-D. Higher IQM scores ( 0 . 7 -0 . 78 ) across methods reflect less extreme OOD actuator strength ([ 0 . 1 , 2 . 0 ] vs. [ 0 . 5 , 1 . 5 ]), reducing generalization demands. In Pixel modality, context-aware cRSSM-S leads (IQM 0 . 7770 ), leveraging ground-truth context for robust visual feature extraction. DALI-S attains an IQM of 0 . 7580 , compared to DALI-Sχ at 0 . 7330 . In the Mixed regime, context-aware cRSSM-D leads (Featurized, IQM 0 . 8720 ), with DALI methods gaining

1 . 1% -1 . 8% over Dreamer-DR; in Pixel, cRSSM-S leads ( 0 . 8610 ), with DALI gains 1 . 1% -5 . 5% . In Pixel Extrapolation, DALI-S shows moderate PoI ( ∼ 0 . 6 ) against Dreamer-DR and cRSSM-D, while cRSSM-S leads, leveraging ground-truth context for visual dynamics (see Figure 1).

## 6.2 Generalization Trends Across Environments and Modalities

We provide additional interpretations of the IQM results in Figure 1, focusing on the generalization patterns of DALI-S and DALI-Sχ . We emphasize modality-specific effects, environmental nuances, integration strategies, and comparisons with context-aware baselines (cRSSM-S, cRSSM-D), highlighting the role of cross-modal regularization.

Cross-Modal Regularization Effects. DALI-Sχ consistently outperforms DALI-S in Ball-inCup across all regimes and modalities, indicating that cross-modal regularization ( L cross , see 2) enhances the context encoder's ability to capture pendulum-like dynamics. Aligning z t with the world model's posterior z t enhances context inference for nonlinear dynamics, benefiting from low partial observability in Featurized inputs and stabilizing inference in Pixel modality's noisy visual inputs. In contrast, DALI-S outperforms DALI-Sχ in Walker across most regimes and modalities, indicating that the forward dynamics loss ( L FD, see 1) alone suffices for contexts where actuator strength linearly scales joint torques. By simply optimizing for next-step predictions, DALI-S effectively handles predictable torque scaling, particularly in the Pixel modality where visual noise may amplify L cross 's complexity in DALI-Sχ . This underscores the need for task-specific regularization. Ball-in-Cup exhibits larger performance drops from Featurized to Pixel modalities (e.g., DALI-Sχ : 0 . 3720 to 0 . 2730 in Extrapolation) compared to Walker, reflecting higher partial observability in visual settings with nonlinear dynamics.

Comparison with Context-Aware Baselines. DALI-Sχ outperforms cRSSM-S and cRSSM-D in Ball-in-Cup Featurized Extrapolation ( 0 . 3720 vs. 0 . 2270 , 0 . 2780 ) and Pixel Extrapolation ( 0 . 2730 vs. 0 . 1870 , 0 . 2420 ), indicating that z t 's inferred context generalizes better than ground-truth c , which tends to overfit to training contexts. In Walker Featurized Extrapolation, both DALI-S and DALI-Sχ surpass cRSSM-S and cRSSM-D ( 0 . 7810 , 0 . 7770 vs. 0 . 7020 , 0 . 7490 ), benefiting from z t 's inference of actuator torque scaling. However, in Ball-in-Cup Pixel Mixed, cRSSM-D ( 0 . 6250 ) outperforms DALI-Sχ ( 0 . 6030 ), indicating that Deep Integration of ground-truth c better supports modeling of complex visual dynamics. Similarly, in Walker Pixel Extrapolation, cRSSM-S ( 0 . 7770 ) surpasses DALI-S ( 0 . 7580 ), leveraging ground-truth context for precise visual modeling.

Shallow Context Propagation as Regularization. In our experiments, Shallow Integration, as implemented in DALI-S and DALI-Sχ , consistently performs well, leading to its exclusive use in the results reported here. Shallow Integration incorporates the inferred context z t solely in the world model's encoder, z t ∼ q θ ( z t | h t , o t , z t ) , allowing context information to propagate indirectly to the recurrent state h t = f θ ( h t -1 , z t -1 , a t -1 ) through recurrence. This design regularizes the world model, potentially mitigating overfitting to noisy z t estimates, which can be particularly beneficial in OOD settings. The empirical effectiveness of Shallow Integration in our experiments suggests that implicit context propagation can effectively leverage the β -mixing property in practice.

## 6.3 Validating Physically Consistent Counterfactuals

We assess whether structured perturbations to the latent space z induce counterfactual trajectories that adhere to Newtonian physics in the Ball-in-Cup task, validating that z encodes mechanistic factors (e.g., gravity, string length) critical for generalization. For this task, the agent must swing a ball into a cup under variable gravity and string length. In our setup, the learned context representation z is an 8 -dimensional vector, i.e., z ∈ R 8 , produced by the dynamics-aligned encoder. Our goal is to identify which latent dimensions in z dominantly encode these mechanistic factors.

For each latent dimension z j , j = 1 , . . . , 8 , we sample a fixed observation o t from a test episode and use the learned context encoder g φ to infer the representation z t = g φ ( o t -K : t , a t -K : t -1 ) with K = 50 . Encoding o t into the latent state s t = { h t , z t } , we then roll out actions a t : t + H under both the original z and a perturbed z ′ = z + ∆ · e j , where e j is a one-hot vector, and ∆ is the standard deviation of z j across the dataset. Using the frozen world model and policy, we decode the predicted observation sequences, yielding two diverging trajectories: baseline trajectory under the original z , T (0) = { ˆ o t , ˆ o t +1 , ..., ˆ o t + H } , and counterfactual trajectory under the perturbed z ′ , T ′ ( j ) = { ˆ o ′ t , ˆ o ′ t +1 , ..., ˆ o ′ t + H } with H = 50 . We repeat this process N = 2500 times to generate 2500

baseline trajectories and 2500 perturbed trajectories for each z j , yielding a total of 5000 trajectory samples: D ( j ) cf = {T (0) 1 , T ′ ( j ) 1 , . . . , T (0) 2500 , T ′ ( j ) 2500 } for z j . Wetrain a binary classifier c ( j ) ν to distinguish between perturbed trajectories T ′ ( j ) (label 1) and baseline trajectories T (0) (label 0). The classifier outputs the probability p n = P ( label = 1 |T n ) that a given trajectory T n ∈ D ( j ) cf is perturbed. High classifier accuracy for a given dimension suggests it captures mechanistic relationships rather than superficial correlations, with perturbations inducing systematic and physically consistent changes.

We compute 95% bootstrap confidence intervals (CIs) for the classifier AUC for each latent dimension z j and rank the 8 dimensions based on the AUCs, and select the top dimension of z for validating physically consistent counterfactuals. See Appendix D for details on the classifier implementation and additional experiments on physically consistent counterfactuals.

## 6.3.1 Mechanistic Alignment in Latent Imagination: Gravity-String Dynamics

We demonstrate DALI's capacity for physically consistent latent imagination by analyzing counterfactual trajectories generated through perturbations to the top-ranked latent dimension ( z 6 ) in DALI-Sχ (Shallow Integration with cross-modal regularization). We show how z 6 encodes coupled gravity-string dynamics in the Ball-in-Cup task.

Our analysis reveals consistent physical behavior across both Featurized and Pixel-based modalities. We demonstrate systematic alignment between the original imagined trajectories and their counterfactual counterparts generated by perturbing the latent context representation in a trained world model. For the Featurized experiments, we tracked predefined state variables (e.g., ball position and velocity), while in the Pixel experiments, we captured rendered environment frames at 5-step intervals over a 64-step imagination horizon (excluding the final frame for visual clarity).

Leveraging our AUC-based ranking analysis, we identify z 6 as the dominant latent dimension across modalities. To isolate the influence of this dimension on passive dynamics, we generate counterfactual trajectories by perturbing z 6 and enforce zero-action rollouts. This suppresses policy-driven behavior, exposing the system's intrinsic swinging dynamics.

Pixel Modality. Figure 2a contrasts the original (top) and perturbed (middle) imagined trajectories. The perturbed rollout (initialized from the same observation) reveals two key effects: (1) Shorter string length: The blue ball (counterfactual) hovers higher than the original (yellow) in frame 40, indicating a shorter string, and (2) Higher acceleration: The counterfactual ball overtakes the original in frame 15 and 45, demonstrating faster swing cycles due to increased gravitational pull. These observations align with Newtonian mechanics: shorter strings reduce string length, increasing oscillation frequency, while higher gravity amplifies acceleration.

Featurized Modality. Figures 2b and 2c quantify these effects. The counterfactual trajectory (orange) in Figure 2b exhibits reduced amplitude (lower peak Z-position) compared to the original (blue), consistent with the pixel-based evidence of a shorter string. This alignment confirms that perturbing shortens the string length, physically constraining the ball's swing. In Figure 2c, the counterfactual's velocity peaks earlier and higher, confirming faster acceleration under perturbed z 6 . Both modalities reveal that z 6 encodes a coupled relationship between gravity and string length, as a perturbation simultaneously alters both parameters in a physically plausible manner.

## 7 Discussion and Outlook

We introduced Dynamics-Aligned Latent Imagination (DALI), a framework built on DreamerV3 that enables zero-shot generalization by inferring latent contexts from interaction histories. In the Extrapolation regime, our methods DALI-S and DALI-Sχ achieve substantial performance gains over the context-unaware Dreamer-DR baseline, ranging from +4 . 0% to +96 . 4% across Featurized and Pixel modalities in the DMC Ball-in-Cup and Walker Walk environments. Notably, DALI-Sχ surpasses the ground-truth context-aware baselines cRSSM-S by +45 . 9% to +63 . 9% and cRSSM-D by +12 . 8% to +33 . 8% in Ball-in-Cup across both modalities. Additionally, DALI-S outperforms cRSSM-S by +11 . 3% and cRSSM-D by +4 . 3% in Walker's Featurized modality, demonstrating effective zero-shot generalization to unseen contexts.

Limitations for DALI's Context Inference. Theorem 2 demonstrates that DALI's context encoder achieves near-optimal context inference, I ( c ; z t ) ≥ (1 -δ ) h ( c ) using short windows of length

K = Ω(log(1 /δ ) /λ ) , yielding a sample complexity gain of O ( T/K ) over DreamerV3, assuming β -mixing and Lipschitz dynamics. As an information-theoretic result, it does not address the downstream impact of z t on policy performance, which depends on joint optimization of the context encoder, world model, and actor-critic components. Reliance on an exploratory policy may falter in sparse-reward or high-dimensional settings, producing noisy estimates. Furthermore, the β -mixing assumption may not hold in environments with slow-mixing dynamics, such as highly correlated trajectories or restricted exploration, potentially limiting generalization. Practical challenges, including training instability, sensitivity to hyperparameters, and the risk of overfitting in high-dimensional observation spaces, are also beyond its scope. Future theoretical work should model how context representations influence policy learning and robustness in complex RL environments.

Integration Strategies and Practical Implications. Empirically, DALI's Shallow Integration strategy, where z t is appended only to the encoder input, consistently delivers strong performance in the tasks we evaluated, particularly under OOD conditions. Its simplicity acts as a regularizer, implicitly propagating z t through recurrence and mitigating overfitting, yielding gains of up to +87 . 9% (Featurized) and +96 . 4% (Pixel) over Dreamer-DR. Future work could investigate the regularization effects of Shallow Integration, explore hybrid strategies interpolating between Shallow and Deep, and develop theoretical models connecting context inference to policy performance. These directions would deepen our understanding of DALI's integration mechanisms and their practical impact on generalization.

Figure 2: (a) (Pixel Modality) Counterfactual Trajectories in Pixel Space : Top: Original imagined trajectory of the Ball-in-Cup under default gravity and string length. Middle: Perturbed trajectory after adding noise ∆ to the top-ranked latent dimension z 6 . Bottom: Pixel-wise differences ( δ = | ˆ o t -ˆ o ′ t | ). The perturbed trajectory (blue) exhibits a shorter string (frame 40) and faster acceleration (overtaking the original trajectory in frames 15 and 45), aligning with increased gravitational effects. Rollouts use zero actions to isolate passive dynamics. (b) (Featurized Modality) Ball Z-Position Under Latent Perturbation : Comparison of original (blue) and counterfactual (orange) ball height trajectories. The perturbed z 6 reduces oscillation amplitude (lower peak Z-position) and accelerates descent, consistent with shorter string length and higher gravity. (c) (Featurized Modality) Ball Z-Velocity Under Latent Perturbation : Velocity profiles for original (blue) and counterfactual (orange) trajectories. The perturbed z 6 induces earlier and higher velocity peaks, confirming faster swing dynamics. This mirrors the pixel-based evidence of increased gravitational acceleration.

<!-- image -->

## Acknowledgments and Disclosure of Funding

The authors thank the anonymous reviewers for their valuable feedback and the NeurIPS community for fostering open and rigorous scientific exchange. JB and ME gratefully acknowledge funding by the German Research Foundation DFG through the MoReSpace (402776968) project.

## References

- Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G Bellemare. Deep reinforcement learning at the edge of the statistical precipice. Advances in Neural Information Processing Systems , 2021.
- Philip J Ball, Cong Lu, Jack Parker-Holder, and Stephen Roberts. Augmented world models facilitate zero-shot dynamics generalization from a single offline environment. In International Conference on Machine Learning , pages 619-629, 2021.
- Jacob Beck, Risto Vuorio, Evan Zheran Liu, Zheng Xiong, Luisa Zintgraf, Chelsea Finn, and Shimon Whiteson. A survey of meta-reinforcement learning. arXiv preprint arXiv:2301.08028 , 2023.
- Carolin Benjamins, Theresa Eimer, Frederik Schubert, Aditya Mohan, Sebastian Döhler, André Biedenkapp, Bodo Rosenhahn, Frank Hutter, and Marius Lindauer. Contextualize me - the case for context in reinforcement learning. Transactions on Machine Learning Research , 2023.
- Michael Beukman, Devon Jarvis, Richard Klein, Steven James, and Benjamin Rosman. Dynamics generalisation in reinforcement learning via adaptive context-aware policies. Advances in Neural Information Processing Systems , 36:40167-40203, 2023.
- Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities - A Nonasymptotic Theory of Independence . Oxford University Press, 2013.
- Tao Chen, Adithyavairavan Murali, and Abhinav Gupta. Hardware conditioned policies for multi-robot transfer learning. Advances in Neural Information Processing Systems , 31, 2018.
- Yan Duan, John Schulman, Xi Chen, Peter L Bartlett, Ilya Sutskever, and Pieter Abbeel. RL 2 : Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779 , 2016.
- Hamid Eghbal-zadeh, Florian Henkel, and Gerhard Widmer. Context-adaptive reinforcement learning using unsupervised learning of context variables. In NeurIPS 2020 Workshop on Pre-registration in Machine Learning , pages 236-254, 2021.
- Ben Evans, Abitha Thankaraj, and Lerrel Pinto. Context is everything: Implicit identification for dynamics adaptation. In 2022 International Conference on Robotics and Automation (ICRA) , pages 2642-2648, 2022.
- Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning , pages 1126-1135, 2017.
- Scott Fujimoto, Pierluca D'Oro, Amy Zhang, Yuandong Tian, and Michael Rabbat. Towards general-purpose model-free reinforcement learning. In International Conference on Learning Representations , 2025.
- Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International Conference on Machine Learning , pages 2555-2565, 2019.
- Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations , 2020.
- Danijar Hafner, Timothy P Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering Atari with discrete world models. In International Conference on Learning Representations , 2021.
- Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature , pages 1-7, 2025.
- Assaf Hallak, Dotan Di Castro, and Shie Mannor. Contextual Markov decision processes. arXiv preprint arXiv:1502.02259 , 2015.
- Nicklas Hansen, Hao Su, and Xiaolong Wang. TD-MPC2: Scalable, robust world models for continuous control. In International Conference on Learning Representations , 2024.

- Robert Kirk, Amy Zhang, Edward Grefenstette, and Tim Rocktäschel. A survey of zero-shot generalisation in deep reinforcement learning. Journal of Artificial Intelligence Research , 76:201-264, 2023.
- Kimin Lee, Younggyo Seo, Seunghyun Lee, Honglak Lee, and Jinwoo Shin. Context-aware dynamics model for generalization in model-based reinforcement learning. In International Conference on Machine Learning , pages 5757-5766, 2020.
- Luckeciano C Melo. Transformers are meta-reinforcement learners. In International Conference on Machine Learning , pages 15340-15359, 2022.
- Aditya Modi, Nan Jiang, Satinder Singh, and Ambuj Tewari. Markov decision processes with continuous side information. In Algorithmic Learning Theory , pages 597-618, 2018.
- Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of Machine Learning . MIT press, 2018.
- Yao Mu, Yuzheng Zhuang, Fei Ni, Bin Wang, Jianyu Chen, Jianye Hao, and Ping Luo. Domino: Decomposed mutual information optimization for generalized context in meta-reinforcement learning. Advances in Neural Information Processing Systems , 35:27563-27575, 2022.
- Anusha Nagabandi, Ignasi Clavera, Simin Liu, Ronald S Fearing, Pieter Abbeel, Sergey Levine, and Chelsea Finn. Learning to adapt in dynamic, real-world environments through meta-reinforcement learning. In International Conference on Learning Representations , 2018.
- Sai Prasanna, Karim Farid, Raghu Rajan, and André Biedenkapp. Dreaming of many worlds: Learning contextual world models aids zero-shot generalization. Reinforcement Learning Journal , 1, 2024.
- Kate Rakelly, Aurick Zhou, Chelsea Finn, Sergey Levine, and Deirdre Quillen. Efficient off-policy metareinforcement learning via probabilistic context variables. In International Conference on Machine Learning , pages 5331-5340, 2019.
- Emmanuel Rio. Asymptotic Theory of Weakly Dependent Random Processes . Springer, 2017.
- Younggyo Seo, Kimin Lee, Ignasi Clavera Gilaberte, Thanard Kurutach, Jinwoo Shin, and Pieter Abbeel. Trajectory-wise multiple choice learning for dynamics generalization in reinforcement learning. Advances in Neural Information Processing Systems , 33:12968-12979, 2020.
- Seyed Kamyar Seyed Ghasemipour, Shixiang Shane Gu, and Richard Zemel. Smile: Scalable meta inverse reinforcement learning through context-conditional policies. Advances in Neural Information Processing Systems , 32, 2019.
- Shagun Sodhani, Amy Zhang, and Joelle Pineau. Multi-task reinforcement learning with context-based representations. In International Conference on Machine Learning , pages 9767-9779, 2021.
- Shagun Sodhani, Franziska Meier, Joelle Pineau, and Amy Zhang. Block contextual MDPs for continual learning. In Learning for Dynamics and Control Conference , pages 608-623, 2022.
- Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. DeepMind control suite. arXiv preprint arXiv:1801.00690 , 2018.
- Joy A. Thomas and Thomas M. Cover. Elements of Information Theory . John Wiley &amp; Sons, 1999.
- Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In International Conference on Intelligent Robots and Systems , pages 23-30, 2017.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems , 30, 2017.
- Zhou Xian, Shamit Lal, Hsiao-Yu Tung, Emmanouil Antonios Platanios, and Katerina Fragkiadaki. Hyperdynamics: Meta-learning object and agent dynamics with hypernetworks. In International Conference on Learning Representations , 2021.
- Zhenjia Xu, Jiajun Wu, Andy Zeng, Joshua B Tenenbaum, and Shuran Song. Densephysnet: Learning dense physical object representations via multi-step dynamic interactions. In Robotics: Science and Systems (RSS) , 2019.

Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning , pages 1094-1100, 2020.

Luisa Zintgraf, Kyriacos Shiarli, Vitaly Kurin, Katja Hofmann, and Shimon Whiteson. Fast context adaptation via meta-learning. In International Conference on Machine Learning , pages 7693-7702, 2019.

## A Formal Results and Proofs

Theorems 2 and 5 provide formal underpinnings for Theorem 1 in the main text, establishing the necessity and efficiency of DALI's context encoder with Deep Integration (see Section 4.3) compared to DreamerV3 with domain randomization in cMDPs, focusing on context identifiability, sample complexity, and information bottleneck reduction.

Theorem 2 compares DALI with Deep Integration, using context encoder z t , against DreamerV3 with domain randomization, using recurrent state h t , in a cMDP. The theorem establishes: (1) DALI's z t captures at least as much mutual information with context c as DreamerV3's h t , up to an error ϵ ( K ) = O ( e -λK/ 2 ) h ( c ) that decays exponentially with window size K ; (2) DALI achieves nearoptimal I ( c ; z t ) ≥ (1 -δ ) h ( c ) with N = O (1 /δ 2 ) samples of K = Ω(log(1 /δ ) /λ ) transitions, leveraging shorter windows compared to DreamerV3's full episodes of length T . This demonstrates the necessity and efficiency of DALI's context encoder for context identifiability and learning.

Theorem 2 (Necessity and efficiency of Context encoder) . Consider a cMDP M = ( S , A , O , C , R , P , E , µ, p C ) with:

- Continuous context c ∈ C ⊆ R d , static within episodes, drawn i.i.d. as c ∼ p C , with h ( c ) &lt; ∞ .
- Lipschitz dynamics: ∃ L &gt; 0 such that ∀ s, s ′ ∈ S , a ∈ A , c, c ′ ∈ C ,

<!-- formula-not-decoded -->

- Observation noise: The observation function E satisfies o t ∼ N ( s t , σ 2 I ) , with fixed variance σ 2 &gt; 0 .
- β -mixing: The process τ t = ( o t , a t ) is β -mixing with

<!-- formula-not-decoded -->

where σ ( τ s , s ≤ t ) denotes the sigma-algebra generated by { τ s : s ≤ t } , and similarly for σ ( τ s , s ≥ t + K ) , for constants C, λ &gt; 0 .

̸

- Sufficient exploration: The policy ensures that for any c = c ′ ∈ C , there exists a constant α &gt; 0 such that KL( p ( o t -K : t , a t -K : t -1 | c ) ∥ p ( o t -K : t , a t -K : t -1 | c ′ )) ≥ αK .
- Bounded log-likelihood ratios: ∣ ∣ ∣ log p ( o i ,a i | c ′ ,τ t -K : i -1 ) p ( o i ,a i | c,τ t -K : i -1 ) ∣ ∣ ∣ ≤ M , ensured by restricting observations and actions to compact sets or imposing suitable policy constraints.
- Universal approximator: DALI's g φ (neural network) approximates any continuous function on compact domains.
- Training data: Both models are trained on trajectories from M , with contexts c ∼ p C i.i.d. per episode, using an exploratory policy. DreamerV3 with domain randomization trains its RSSM on episodes ( o 1: T , a 1: T -1 ) of length T . DALI trains its context encoder g φ on K -length windows ( o t -K : t , a t -K : t -1 ) with loss L FD = E [ ∥ o t +1 -f w φ ( o t , a t , z t ) ∥ 2 2 ] , and its RSSM on full episodes, conditioning on z t .
- Both models use a GRU f θ with finite parameters for their recurrent state updates.

Let h t = f θ ( h t -1 , z t -1 , a t -1 ) be DreamerV3's recurrent state, with z t ∼ q θ ( z t | h t , o t ) , and z t = g φ ( o t -K : t , a t -K : t -1 ) DALI's context encoding, with h t = f θ ( h t -1 , z t -1 , a t -1 , z t ) . Then:

1. Context identifiability : For any K , there exists g φ such that, within an episode:

<!-- formula-not-decoded -->

2. Sample complexity : For K = Ω ( log(1 /δ ) λ ) and δ ∈ (0 , 1) , DALI achieves

<!-- formula-not-decoded -->

with N = O ( 1 δ 2 ) samples of K transitions, vs. a hypothetical DreamerV3 context estimator g ψ ( o 1: T , a 1: T -1 ) requiring N = O ( 1 δ 2 ) samples of T transitions.

Remark 3 (Non-trivial sample complexity gain of DALI's Context encoder) . The sample complexity gain of O ( T/K ) for DALI over DreamerV3 with domain randomization, as shown in Theorem 2, arises from DALI's use of K -length windows versus DreamerV3's full episodes of length T . While this gain appears to scale with input lengths, it is non-trivial: Both models achieve N = O (1 /δ 2 ) samples, but DALI's window size K = Ω(log(1 /δ ) /λ ) leverages the β -mixing property (Lemma 6) to ensure h ( c | τ t -K : t ) ≤ C ′ e -λK h ( c ) , enabling efficient context identification. In contrast, DreamerV3 relies on longer trajectories ( T ≫ K ) to achieve comparable context identification. The gain reflects DALI's context encoder exploiting short, local histories determined by the mixing rate λ and desired accuracy δ .

Remark 4 (Consistency of DALI's sample complexity across integration strategies) . The sample complexity advantage of DALI established in Theorem 2 for Deep Integration extends to Shallow Integration (Section 4.3), since the context encoder z t is trained identically on K -length windows in both configurations. Incorporating the cross-modal loss L cross in (2) aligns z t with z t , which may allow the context representation z t to leverage structured priors from the world model, potentially enhancing its robustness. This addition does not affect sample complexity, as L cross operates on the same K -length windows (Algorithm 2). Recurrent unrolls are limited to K steps, and φ updates depend solely on the current window, with detached states blocking backpropagation beyond it. The world model parameters ( θ ) are frozen during context learning (inputs to f θ and q θ are detached), so gradient stopping isolates φ and preserves the O ( T/K ) efficiency.

Theorem 5 proves that DALI's sequence model, incorporating a context encoder z t , reduces the information bottleneck in cMDPs compared to DreamerV3's RSSM. Specifically, it shows that DALI's recurrent state retains more context information, satisfying I ( c ; h DALI t ) ≥ I ( c ; h RSSM t ) -ϵ ( K ) , where ϵ ( K ) = O ( e -λK/ 2 ) h ( c ) , leveraging the encoder's ability to efficiently capture context over short windows.

Theorem 5 (Context encoder reduces information bottleneck) . Consider a cMDP M = ( S , A , O , C , R , P , E , µ, p C ) with:

- Continuous context c ∈ C ⊆ R d , static within episodes, drawn i.i.d. as c ∼ p C , with h ( c ) &lt; ∞ .
- Lipschitz dynamics: ∃ L &gt; 0 such that ∀ s, s ′ ∈ S , a ∈ A , c, c ′ ∈ C ,
- Observations o t = s t + η t , η t ∼ N (0 , σ 2 I ) , σ 2 &gt; 0 .

<!-- formula-not-decoded -->

- β -mixing with β ( K ) ≤ Ce -λK , exploratory policy with KL( p ( τ t -K : t | c ) ∥ p ( τ t -K : t | c ′ )) ≥ αK , α &gt; 0 .
- Context encoder g φ is a universal approximator, trained with forward dynamics loss L FD = E [ ∥ o t +1 -f w φ ( o t , a t , z t ) ∥ 2 2 ] , satisfying h ( c | z t ) ≤ C ′ e -λK h ( c ) + δ ′ , δ ′ = O ( e -λK/ 2 ) (Lemma 7).
- Both models use a GRU f θ with finite parameters.

Let h RSSM t = f θ ( h t -1 , z t -1 , a t -1 ) be DreamerV3's recurrent state, with z t ∼ q θ ( z t | h t , o t ) , and h DALI t = f θ ( h t -1 , z t -1 , a t -1 , z t ) be DALI's recurrent state. Then, for any window size K , DALI's sequence model satisfies:

<!-- formula-not-decoded -->

where ϵ ( K ) = O ( e -λK/ 2 ) h ( c ) , implying that the context encoder z t reduces the information bottleneck compared to DreamerV3's RSSM.

For β -mixing sequences with β ( K ) ≤ Ce -λK , blocks of K -separated transitions are approximately independent. Lemma 6 establishes that, in a β -mixing cMDP, the conditional entropy of the context c given a history τ t -K : t decays exponentially as h ( c | τ t -K : t ) ≤ C ′ e -λK h ( c ) . This result is crucial for Theorems 2 and 5, as it quantifies how historical trajectories reduce uncertainty about the context, enabling DALI's context encoder to achieve near-optimal information capture with short windows, supporting both the identifiability and sample complexity claims.

̸

<!-- formula-not-decoded -->

Lemma 6 (Entropy decay from mixing) . Consider a cMDP with context c ∼ p C , observation noise o t = s t + η t , η t ∼ N (0 , σ 2 I ) and history τ t -K : t = ( o t -K : t , a t -K : t -1 ) , where the process τ t = ( o t , a t ) is β -mixing with β ( K ) ≤ Ce -λK for constants C, λ &gt; 0 , bounding dependence errors in tail probabilities by β ( K ) , log-likelihood ratios are bounded as ∣ ∣ ∣ log p ( o i ,a i | c ′ ,τ t -K : i -1 ) p ( o i ,a i | c,τ t -K : i -1 ) ∣ ∣ ∣ ≤ M , the policy ensures exploration such that for any c = c ′ ∈ C , there exists a constant α ≥ M √ 2 λ satisfying KL( p ( τ t -K : t | c ) | p ( τ t -K : t | c ′ )) ≥ αK , and c has bounded density on C ⊆ R d with differential entropy h ( c ) &lt; ∞ . Then, there exists a constant C ′ &gt; 0 such that:

Proof of Lemma 6. We show that uncertainty about c decreases exponentially with K , enabling DALI's z t to infer c .

By assumption, the process τ t = ( o t , a t ) is β -mixing, with β ( K ) ≤ Ce -λK . The condition KL( p ( τ t -K : t | c ) ∥ p ( τ t -K : t | c ′ )) ≥ αK , with α ≥ M √ 2 λ , ensures distinct histories. We need h ( c | τ t -K : t ) ≤ C ′ e -λK h ( c ) , so we show:

̸

<!-- formula-not-decoded -->

Discretize C into N points, log N ≈ h ( c ) . Let ˆ c = arg max c ′ p ( τ t -K : t | c ′ ) and let P e = P (ˆ c = c | τ t -K : t ) be the probability of incorrectly estimating c . Fano's inequality gives [Thomas and Cover, 1999]:

<!-- formula-not-decoded -->

̸

Error P e = P (ˆ c = c | c ) occurs if there exists any c ′ = c such that p ( τ t -K : t | c ′ ) &gt; p ( τ t -K : t | c ) . By the union bound:

̸

̸

<!-- formula-not-decoded -->

̸

̸

For c ′ = c , define: Z = log p ( τ t -K : t | c ′ ) p ( τ t -K : t | c ) . Then Z &gt; 0 corresponds to the maximum likelihood estimator ˆ c incorrectly favoring c ′ over c . The probability of this error under context c is P ( Z &gt; 0 | c ) . By the exploration assumption:

<!-- formula-not-decoded -->

By the chain rule in the cMDP, p ( τ t -K : t | c ) = ∏ t -1 i = t -K p ( o i , a i | τ t -K : i -1 , c ) · p ( o t | τ t -K : t -1 , c ) , where τ t -K : i -1 = ( o t -K : i -1 , a t -K : i -1 ) .

Thus, Z = ∑ t -1 i = t -K log p ( o i ,a i | τ t -K : i -1 ,c ′ ) p ( o i ,a i | τ t -K : i -1 ,c ) +log p ( o t | τ t -K : t -1 ,c ′ ) p ( o t | τ t -K : t -1 ,c ) .

Define Z i = log p ( o i ,a i | τ t -K : i -1 ,c ′ ) p ( o i ,a i | τ t -K : i -1 ,c ) for i = t -K to t -1 , and Z t = log p ( o t | τ t -K : t -1 ,c ′ ) p ( o t | τ t -K : t -1 ,c ) . So Z = ∑ t i = t -K Z i . Z i measures how much more (or less) likely ( o i , a i ) is under context c ′ compared to c , given past observations and actions. By the bounded log-likelihood assumption, | Z i | ≤ M . For independent Z i , Hoeffding's inequality [Boucheron et al., 2013] bounds the tail probability for bounded random variables:

<!-- formula-not-decoded -->

Since τ t is β -mixing with β ( K ) ≤ Ce -λK , the dependence adds an error β ( K ) , as β ( K ) bounds the deviation from independence between events separated by K time steps:

<!-- formula-not-decoded -->

Set κ = α 2 2 M 2 . Since α ≥ M √ 2 λ , we have κ ≥ λ , so exp( -κK ) ≤ e -λK . Thus:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Sum over N -1 contexts:

Apply to (4):

<!-- formula-not-decoded -->

For small β ( K ) , h ( C 1 β ( K )) ≈ 0 , so:

With β ( K ) ≤ Ce -λK ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 7 provides a generalization bound for DALI's context encoder, showing that the conditional entropy h ( c | z t ) is bounded by the entropy given the history plus a small error, h ( c | z t ) ≤ h ( c | τ t -K : t )+ δ ′ , with δ ′ = O ( √ δ ) and δ = Ce -λK . This lemma is pivotal for Theorems 2 and 5, as it ensures that the encoder z t effectively captures context information with a sample complexity of N = O (1 /δ 2 ) , facilitating DALI's efficiency and reduced bottleneck.

Lemma 7 (DALI's generalization bound for context inference) . Consider a cMDP satisfying Theorem 2's assumptions: context c ∼ p C with bounded density on C ⊆ R d and finite entropy h ( c ) &lt; ∞ , observation noise o t = s t + η t , η t ∼ N (0 , σ 2 I ) , β -mixing with β ( K ) ≤ Ce -λK for constants C, λ &gt; 0 , exploratory policy ensuring KL( p ( τ t -K : t | c ) ∥ p ( τ t -K : t | c ′ )) ≥ αK for some α &gt; 0 , Lipschitz dynamics with ∥P c ( s ′ | s, a ) -P c ′ ( s ′ | s, a ) ∥ 1 ≤ L ∥ c -c ′ ∥ 2 , and universal approximator g φ . DALI's context encoding z t = g φ ( o t -K : t , a t -K : t -1 ) , trained with the forward dynamics loss L FD = E [ ∥ o t +1 -f w φ ( o t , a t , z t ) ∥ 2 2 ] , satisfies:

<!-- formula-not-decoded -->

with N = O ( 1 δ 2 ) samples, where δ = Ce -λK .

Proof of Lemma 7. The context encoder g φ minimizes the forward dynamics loss:

<!-- formula-not-decoded -->

where τ t -K : t = ( o t -K : t , a t -K : t -1 ) , o t = s t + η t , and η t ∼ N (0 , σ 2 I ) . The empirical loss over N samples is:

<!-- formula-not-decoded -->

To benchmark L FD, consider an ideal predictor with access to the true context c . Since o t +1 = s t +1 + η t +1 , η t +1 ∼ N (0 , σ 2 I ) , and o t = s t + η t , the predictor f ( o t , a t , c ) uses noisy observations. The loss is:

<!-- formula-not-decoded -->

For d s -dimensional η t +1 , E [ ∥ η t +1 ∥ 2 2 ] = d s σ 2 . The ideal predictor f ∗ ( o t , a t , c ) = E [ s t +1 | o t , a t , c ] , the expected next state given o t , a t , c , minimizes L FD by setting f ∗ to the best estimate of s t +1 , yielding E [ ∥ o t +1 -f ∗ ( o t , a t , c ) ∥ 2 2 ] = E [ ∥ s t +1 -E [ s t +1 | o t , a t , c ] ∥ 2 2 ] + d s σ 2 , where the first term is the variance of s t +1 and the second is the noise variance. By the universal approximation property, f w φ can approximate f ∗ if z t encodes c .

Consider the loss function for fixed φ, w :

<!-- formula-not-decoded -->

with | l | ≤ M (by bounded S , A ) and L l -Lipschitz in φ, w . The function class F = { ( o i , a i , o i -K : i , a i -K : i -1 ) ↦→ f w φ ( o i , a i , g φ ( o i -K : i , a i -K : i -1 )) } has Rademacher complexity R N ( F ) ≤ C F √ N , so the loss class L has:

<!-- formula-not-decoded -->

For i.i.d. samples and a fixed φ, w , the generalization error is bounded by:

<!-- formula-not-decoded -->

By McDiarmid's inequality, since changing one sample affects ˆ L FD by at most 2 M N , with probability at least 1 -δ [Mohri et al., 2018]:

<!-- formula-not-decoded -->

For β -mixing data with β ( K ) ≤ Ce -λK , the deviation from i.i.d. adds an error bounded by Ce -λK to the expected loss for sequences spaced by K steps:

<!-- formula-not-decoded -->

Set the generalization error to ϵ 2 = O ( δ ) , where δ = Ce -λK , to match the β -mixing decay. Solve:

<!-- formula-not-decoded -->

The first term requires: 2 C F L l √ N ≤ C ′′ δ = ⇒ N ≥ ( 2 C F L l C ′′ δ ) 2 = O ( 1 δ 2 ) .

The second term requires: M √ 2 log(1 /δ ) N ≤ C ′′ δ = ⇒ N ≥ 2 M 2 log(1 /δ ) ( C ′′ δ ) 2 = O ( log(1 /δ ) δ 2 ) .

The third term Ce -λK = δ ≤ C ′′ δ holds for C ′′ ≥ 1 .

The first term dominates, so N = O ( 1 δ 2 ) suffices.

If E [ ˆ L FD ] ≤ E [ ∥ o t +1 -f ∗ ( o t , a t , c ) ∥ 2 2 ] + ϵ 2 , then:

<!-- formula-not-decoded -->

Given the Markov chain c → τ t -K : t → z t , the entropy difference is:

The cMDP's Lipschitz dynamics, ∥P c ( s ′ | s, a ) -P c ′ ( s ′ | s, a ) ∥ 1 ≤ L ∥ c -c ′ ∥ 2 , and β -mixing ensure p ( c | τ t -K : t ) is L p -Lipschitz in Wasserstein distance, as small changes in τ t -K : t reflect proportional changes in c . Assume there exists a mapping ψ : Z → C such that the context encoder satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Z is the encoder's output space. Then:

Thus:

From Lemma 6, h ( c | τ t -K : t ) ≤ Ce -λK h ( c ) = O ( δ ) h ( c ) . Since h ( c ) &lt; ∞ and √ δ ≥ δ for small δ , the bound becomes:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

yielding δ ′ = O ( √ δ ) with N = O ( 1 δ 2 ) samples.

Lemma 8 demonstrates that DreamerV3's RSSM suffers from a persistent information bottleneck in cMDPs, with the conditional entropy h ( c | h t ) ≥ κh ( c ) for a constant κ ∈ (0 , 1) . This result is essential for Theorems 2 and 5, as it establishes a baseline limitation in DreamerV3's ability to retain context information, against which DALI's superior performance in context identifiability and bottleneck reduction is compared.

Lemma8 (Information bottleneck in DreamerV3's RSSM) . Consider DreamerV3's RSSM in a cMDP with context c ∼ p C with bounded density on C ⊆ R d and finite entropy h ( c ) &lt; ∞ , observation noise o t = s t + η t , η t ∼ N (0 , σ 2 I ) , Lipschitz dynamics with ∥P c ( s ′ | s, a ) -P c ′ ( s ′ | s, a ) ∥ 1 ≤ L ∥ c -c ′ ∥ 2 , and recurrent state h t = f θ ( h t -1 , z t -1 , a t -1 ) , where z t ∼ q θ ( z t | h t , o t ) , and f θ is a GRU with finite parameters. Then, there exists κ ∈ (0 , 1) such that:

<!-- formula-not-decoded -->

Proof of Lemma 8. The context c determines the cMDP's dynamics P c ( s ′ | s, a ) . Since h t depends on the trajectory τ 1: t = ( o 1: t , a 1: t -1 ) and latent variables z 1: t -1 , we have a Markov chain c → τ 1: t → h t , so p ( h t | τ 1: t , c ) = p ( h t | τ 1: t ) . By the data processing inequality (DPI) [Thomas and Cover, 1999]:

<!-- formula-not-decoded -->

Since o 1: t = s 1: t + η 1: t , and a 1: t -1 depend on o 1: t -1 , we have:

<!-- formula-not-decoded -->

The GRU f θ , with finite parameters and Lipschitz continuity (due to bounded weights and standard GRU operations), produces h t ∈ R d , where d is a fixed dimension. The GRU is trained to maximize the ELBO:

<!-- formula-not-decoded -->

which optimizes h t to predict observations o t , rewards r t , and termination signals n t . The input τ 1: t includes:

- Context information: c affects s 1: t through dynamics.
- Noise information: η 1: t ∼ N (0 , σ 2 I t · d s ) , with d s being the dimension of the state space S . η 1: t contributes significant entropy, h ( η 1: t ) = t · d s 2 ln(2 πeσ 2 ) , which grows linearly with t .
- State transients: s 1: t includes dynamic behaviors not directly tied to c .

The GRU's finite parameters and fixed dimension d limit its capacity, forcing h t to compress τ 1: t , potentially discarding context information.

Assume h t is Gaussian with covariance Σ h , where Var ( h i t ) ≤ V for i = 1 , . . . , d . The variance bound V follows from compact C , S , A , Lipschitz dynamics, and the Lipschitz GRU, ensuring bounded outputs. For a d -dimensional Gaussian channel with output h t ∼ N ( µ c , Σ h ) , where µ c depends on c , and noise variance at most σ 2 , the mutual information is bounded by [Thomas and Cover, 1999]:

<!-- formula-not-decoded -->

The GRU's compression limits I ( c ; h t ) relative to h ( c ) for all h ( c ) ≥ 0 , as follows. The ELBO's KL-term KL( q θ ( z t | h t , o t ) ∥ p θ ( z t | h t )) regularizes q θ ( z t | h t , o t ) , reducing the information about o t (and thus c ) in z t . Since h t depends on z 1: t -1 , this limits context information in h t . The GRU's finite parameters and dimension d further constrain capacity. Let the channel capacity be reduced by a factor α ∈ (0 , 1) , reflecting this compression:

<!-- formula-not-decoded -->

where α &lt; 1 (e.g., α = 1 -ϵ , with ϵ &gt; 0 depending on the GRU's architecture).

To satisfy the lemma, we seek a constant κ ∈ (0 , 1) such that h ( c | h t ) ≥ κh ( c ) for all p C . Since h ( c | h t ) = h ( c ) -I ( c ; h t ) and I ( c ; h t ) ≤ α · d 2 ln ( 1 + dV σ 2 ) , define:

<!-- formula-not-decoded -->

For h ( c ) &gt; 0 , we want h ( c ) -I ( c ; h t ) ≥ κh ( c ) . Using I ( c ; h t ) ≤ αB , this holds if h ( c ) -αB ≥ κh ( c ) , or κ ≤ 1 -αB h ( c ) . Thus, define:

<!-- formula-not-decoded -->

If h ( c ) &gt; 0 , then I ( c ; h t ) ≤ αB &lt; B (since α &lt; 1 ), and with κ = 1 -αB h ( c ) , we have h ( c | h t ) = h ( c ) - I ( c ; h t ) ≥ h ( c ) -αB = ( 1 -αB h ( c ) ) h ( c ) = κh ( c ) . Since αB &gt; 0 and h ( c ) &gt; 0 , then κ = 1 -αB h ( c ) &lt; 1 . If h ( c ) ≥ αB , then κ ≥ 0 . If h ( c ) &lt; αB , then κ &lt; 0 , but h ( c | h t ) ≥ 0 ≥ κh ( c ) , so the inequality holds.

If h ( c ) = 0 , then I ( c ; h t ) = 0 , so h ( c | h t ) = 0 . Choosing κ = 1 2 ∈ (0 , 1) , we have h ( c | h t ) = 0 ≥ 1 2 · 0 = κh ( c ) , and any κ ∈ (0 , 1) would suffice.

Thus, for all p C , there exists κ ∈ (0 , 1) such that h ( c | h t ) ≥ κh ( c ) .

Lemma 9 proves that DALI's sequence model, incorporating the context encoder z t , reduces the information bottleneck compared to DreamerV3, with h ( c | h t ) ≤ κ ′ h ( c ) , where κ ′ = Ce -λK + O ( e -λK/ 2 ) . This lemma directly supports Theorem 5 by quantifying the improved context retention in DALI's recurrent state, and aids Theorem 2 by reinforcing the necessity of the context encoder for achieving near-optimal information capture.

Lemma 9 (Information bottleneck in DALI's sequence model) . Consider DALI's sequence model in a cMDPwith context c ∼ p C with bounded density on C ⊆ R d and finite entropy h ( c ) &lt; ∞ , observation noise o t = s t + η t , η t ∼ N (0 , σ 2 I ) , Lipschitz dynamics with ∥P c ( s ′ | s, a ) -P c ′ ( s ′ | s, a ) ∥ 1 ≤ L ∥ c -c ′ ∥ 2 , context encoding z t = g φ ( o t -K : t , a t -K : t -1 ) , and recurrent state h t = f θ ( h t -1 , z t -1 , a t -1 , z t ) , where z t ∼ q θ ( z t | h t , o t ) , and f θ is a GRU with finite parameters. Suppose that the context encoding satisfies h ( c | z t ) ≤ Ce -λK h ( c ) + δ ′ , where δ ′ = O ( e -λK/ 2 ) and δ = Ce -λK (Lemma 7). Then, there exists κ ′ ∈ (0 , 1) such that:

where κ ′ = Ce -λK + O ( e -λK/ 2 ) , implying a reduced information bottleneck compared to DreamerV3's κ (Lemma 8).

<!-- formula-not-decoded -->

Proof of Lemma 9. The context c determines the cMDP's dynamics P c ( s ′ | s, a ) . DALI's sequence model computes the context encoding z t = g φ ( o t -K : t , a t -K : t -1 ) from trajectory τ t -K : t = ( o t -K : t , a t -K : t -1 ) , and the recurrent state h t = f θ ( h t -1 , z t -1 , a t -1 , z t ) , where z t -1 ∼ q θ ( z t -1 | h t -1 , o t -1 ) and f θ is a Lipschitz continuous GRU.

Since h t = f θ ( h t -1 , z t -1 , a t -1 , z t ) , we have a Markov chain:

<!-- formula-not-decoded -->

conditional on h t -1 , z t -1 , a t -1 , as c affects h t through z t given these variables. By the DPI, we have

<!-- formula-not-decoded -->

By Lemma 7, the context encoding satisfies:

<!-- formula-not-decoded -->

where δ = Ce -λK , so δ ′ = O ( √ δ ) = O ( e -λK/ 2 ) . From Lemma 6, β -mixing implies:

<!-- formula-not-decoded -->

Thus,

Applying the DPI bound, we have

Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To satisfy h ( c | h t ) ≤ κ ′ h ( c ) , consider two cases. If h ( c ) &gt; 0 , let δ ′ = C ′′ e -λK/ 2 , so:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since δ ′ h ( c ) = O ( e -λK/ 2 ) , κ ′ = Ce -λK + O ( e -λK/ 2 ) when the first term is selected, and κ ′ ≤ 1 2 always. If h ( c ) = 0 , then I ( c ; h t ) = 0 , so h ( c | h t ) = 0 . Set κ ′ = 1 2 , satisfying h ( c | h t ) = 0 ≤ κ ′ h ( c ) .

Compared to DreamerV3's h ( c | h RSSM t ) ≥ κh ( c ) , κ ∈ (0 , 1) (Lemma 8), DALI's κ ′ → 0 as K →∞ , indicating a reduced information bottleneck.

Proof of Theorem 5. We prove that DALI's recurrent state h DALI t achieves higher mutual information with the context c than DreamerV3's h RSSM t , up to an error ϵ ( K ) = O ( e -λK/ 2 ) h ( c ) .

By Lemma 8, DreamerV3's recurrent state satisfies:

Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Lemma 9, DALI's recurrent state satisfies:

Thus,

Compare:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since h ( c ) &lt; ∞ . Thus,

As K →∞ , κ ′ → 0 , so I ( c ; h DALI t ) → h ( c ) , while I ( c ; h RSSM t ) ≤ (1 -κ ) h ( c ) &lt; h ( c ) , demonstrating that the context encoder reduces the bottleneck.

Proof of Theorem 2. We prove both parts of the theorem comparing DALI's context encoding z t = g φ ( o t -K : t , a t -K : t -1 ) with DreamerV3's recurrent state h t = f θ ( h t -1 , z t -1 , a t -1 ) .

## 1. Context identifiability

We show that DALI's context encoding satisfies:

<!-- formula-not-decoded -->

where τ t -K : t = ( o t -K : t , a t -K : t -1 ) is the history, and h t is DreamerV3's recurrent state.

By Lemma 7, DALI's context encoder, trained with the forward dynamics loss:

satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Lemma 6, the history's conditional entropy is:

Combining,

Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The mutual information for DALI is

<!-- formula-not-decoded -->

By Lemma 8, DreamerV3's recurrent state h t has so that

Compare:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

I -I ≥ ----Since κh ( c ) &gt; 0 , set

<!-- formula-not-decoded -->

as the e -λK/ 2 term dominates. Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Additionally, Lemma 9 shows that DALI's recurrent state h t = f θ ( h t -1 , z t -1 , a t -1 , z t ) has a reduced bottleneck:

implying,

This suggests that even DALI's recurrent state retains more context information than DreamerV3's h t , reinforcing the necessity of DALI's context encoder z t , which achieves near-optimal I ( c ; z t ) .

## 2. Sample complexity

We prove that DALI achieves I ( c ; z t ) ≥ (1 -δ ) h ( c ) with K = Ω ( log(1 /δ ) λ ) and N = O ( 1 δ 2 ) samples of K transitions, while a hypothetical DreamerV3 context estimator g ψ ( o 1: T , a 1: T -1 ) requires N = O ( 1 δ 2 ) samples of T transitions.

DALI's sample complexity From Lemma 7, we have

From Lemma 6, we have

<!-- formula-not-decoded -->

Combining, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To achieve I ( c ; z t ) ≥ (1 -δ ) h ( c ) , we need

Set C ′ e -λK h ( c ) + C ′′ e -λK/ 2 ≤ δh ( c ) . Since e -λK/ 2 dominates, approximate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Solve:

From Lemma 7, achieving δ ′ = O ( e -λK/ 2 ) requires

Solve:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

DreamerV3's sample complexity

From Lemma 8, DreamerV3's recurrent state h t has h ( c | h t ) ≥ κh ( c ) , so I ( c ; h t ) ≤ (1 -κ ) h ( c ) , where κ is constant, preventing I ( c ; h t ) ≥ (1 -δ ) h ( c ) for small δ . To enable comparison, assume DreamerV3 with domain randomization trains a hypothetical context estimator g ψ ( o 1: T , a 1: T -1 ) to estimate c over N episodes of length T . By Lemma 6 for K = T :

<!-- formula-not-decoded -->

Training g ψ to minimize E [ g ψ ( o 1: T , a 1: T -1 ) E [ c τ 1: T ] 2 ] requires

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Solve:

The total number of transitions is N · T = O ( T δ 2 ) .

Note that DreamerV3's standard RSSM cannot achieve I ( c ; h t ) ≥ (1 -δ ) h ( c ) due to h ( c | h t ) ≥ κh ( c ) (Lemma 8). The estimator g ψ is introduced to enable a fair comparison, acknowledging that this is not standard in DreamerV3. Since K ≪ T typically, DALI is more efficient. Moreover, Lemma 9 indicates that DALI's h t achieves I ( c ; h t ) ≥ (1 -κ ′ ) h ( c ) , with κ ′ = O ( e -λK/ 2 ) , suggesting that DALI's sequence model could approach the desired information level with sufficient K , unlike DreamerV3's h t .

## A.1 DALI and Dreamer-based Baselines

## A.1.1 World Models

DreamerV3 with Domain Randomization (context-unaware baseline) [Tobin et al., 2017, Hafner et al., 2025]:

DALI-S (DALI with S hallow Integration):

Sequence model:

h t = f θ ( h t - 1 , z t - 1 , a t - 1 )

Encoder:

z t ∼ q θ ( z t | h t , o t )

Dynamics predictor:

ˆ z t ∼ p θ (ˆ z t | h t )

Reward predictor:

ˆ r t ∼ p θ (ˆ r t | h t , z t )

Continue predictor:

ˆ n t ∼ p θ (ˆ n t | h t , z t )

Decoder:

ˆ o t ∼ p θ (ˆ o t | h t , z t ) .

Context encoder:

$$z t = g φ ( o t - K : t , a t - K : t - 1 ) .$$

Sequence model:

h t = f θ ( h t - 1 , z t - 1 , a t - 1 ) ,

Encoder:

z t ∼ q θ ( z t | h t , o t , z t ) ,

Dynamics predictor:

ˆ z t ∼ p θ (ˆ z t | h t ) ,

Reward predictor:

ˆ r t ∼ p θ (ˆ r t | h t , z t ) ,

Continue predictor:

ˆ n t ∼ p θ (ˆ n t | h t , z t ) ,

Decoder:

ˆ o t ∼ p θ (ˆ o t | h t , z t ) .

## DALI-D (DALI with D eep Integration):

<!-- formula-not-decoded -->

cRSSM-S (ground-truth context-aware baseline) [Prasanna et al., 2024]:

$$Sequence model: h t = f θ ( h t - 1 , z t - 1 , a t - 1 ) , Encoder: z t ∼ q θ ( z t | h t , o t , c ) , Dynamics predictor: ˆ z t ∼ p θ (ˆ z t | h t ) , Reward predictor: ˆ r t ∼ p θ (ˆ r t | h t , z t ) , Continue predictor: ˆ n t ∼ p θ (ˆ n t | h t , z t ) , Decoder: ˆ o t ∼ p θ (ˆ o t | h t , z t ) .$$

cRSSM-D (ground-truth context-aware baseline) [Prasanna et al., 2024]:

$$Sequence model: h t = f θ ( h t - 1 , z t - 1 , a t - 1 , c ) Encoder: z t ∼ q θ ( z t | h t , o t ) Dynamics predictor: ˆ z t ∼ p θ (ˆ z t | h t ) Reward predictor: ˆ r t ∼ p θ (ˆ r t | h t , z t , c ) Continue predictor: ˆ n t ∼ p θ (ˆ n t | h t , z t , c ) Decoder: ˆ o t ∼ p θ (ˆ o t | h t , z t , c ) .$$

The baselines cRSSM-S and cRSSM-D correspond to concat-context and cRSSM , respectively, in [Prasanna et al., 2024].

## A.1.2 Actor-Critic Models

## DreamerV3 with Domain Randomization , DALI-S , and cRSSM-S :

<!-- formula-not-decoded -->

DALI-D :

cRSSM-D :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B Algorithms

```
Algorithm 1: DALI-S (Shallow Integration with Forward Dynamics Loss) 1 Model components Context encoder g φ ( o t -K : t , a t -K : t -1 ) One-step model f w φ ( o t , a t , z t ) Deterministic State f θ ( h t -1 , z t -1 , a t -1 ) Stochastic state p θ (ˆ z t | h t ) Encoder q θ ( z t | h t , o t , z t ) Reward p θ (ˆ r t | h t , z t ) Continue p θ (ˆ n t | h t , z t ) Decoder p θ (ˆ o t | h t , z t ) Actor π ϕ ( a t | s t ) Critic v ψ ( s t ) Hyper parameters Seed episodes S Collect interval C Batch size B Sequence length L Imagination horizon H Learning rate α Learning rate g α g Episode length T Context history K 2 Initialize dataset D with S random seed episodes. 3 Initialize neural network parameters θ, ϕ, ψ, φ randomly. 4 while not converged do 5 for update step c = 1 , . . . , C do // Dynamics learning (World Model θ ) 6 Draw B sequences { ( o t , a t , r t ) } L t =1 ∼ D . 7 Initialize deterministic state h 0 ← 0 . 8 for all steps t from sequences batch D do 9 if t < K then 10 Pad o 1: t and a 1: t -1 with zeros to length K . 11 Compute context z t ← g φ ( o t -K : t , a t -K : t -1 ) . detach() 12 Encode observation z t ∼ q θ ( z t | h t , o t , z t ) 13 Compute deterministic state h t = f θ ( h t -1 , z t -1 , a t -1 ) 14 Set latent state s t ← [ z t , h t ] 15 Update θ using representation learning (decoder, reward, continue). // Context representation learning ( φ ) 16 Draw B data sequences { ( o t , a t ) } K +1 t =1 ∼ D 17 for each sample ( o 1: K +1 , a 1: K ) ∼ D do 18 Compute z 1: K = g φ ( o 1: K , a 1: K -1 ) 19 Update φ ← φ -α g ∇ φ ∑ K t =1 ∥ ∥ ∥ o t +1 -f w φ ( o t , a t , z t ) ∥ ∥ ∥ 2 2 // Behavior learning (Actor/Critic ϕ, ψ ) 20 Infer context: z t ← g φ ( o t -K : t , a t -K : t -1 ) . detach() // θ , φ frozen 21 Imagine trajectories { ( s τ , a τ ) } t + H τ = t from each s t (with a fixed z t once it is provided to the initial observation encoding) 22 Predict rewards E[ p θ ( r τ | s τ )] and values v ψ ( s τ ) . 23 Compute value estimates V λ ( s τ ) . 24 Update ϕ ← ϕ + α ∇ ϕ ∑ t + H τ = t V λ ( s τ ) . 25 Update ψ ← ψ -α ∇ ψ ∑ t + H τ = t 1 2 ∥ ∥ ∥ v ψ ( s τ ) -V λ ( s τ ) ∥ ∥ ∥ 2 . // Environment interaction 26 o 1 ← env.reset() 27 for time step t = 1 , . . . , T do 28 if t < K then 29 Pad o 1: t and a 1: t -1 with zeros to length K . 30 Infer context online: z t ← g φ ( o t -K : t , a t -K : t -1 ) . 31 Compute s t ∼ p θ ( s t | s t -1 , a t -1 , o t ) . 32 Compute a t ∼ π ϕ ( a t | s t ) . 33 r t , o t +1 ← env.step ( a t ) . 34 Add experience to dataset D ← D ∪ { ( o t , a t , r t ) T t =1 } .
```

## Algorithm 2: DALI-Sχ (Shallow Integration with Cross-modal Regularization)

```
Initialize dataset with random seed episodes
```

```
1 D S 2 Initialize parameters θ, ϕ, ψ, φ, W z , W z randomly 3 while not converged do 4 for update step c = 1 to C do // Dynamics learning (World Model θ ) 5 Draw B sequences { ( o t , a t , r t ) } L t =1 ∼ D 6 Initialize h 0 ← 0 7 for all steps t in batch do 8 if t < K then 9 Pad o max(1 ,t -K ): t and a max(1 ,t -K ): t -1 with zeros. 10 Compute context z t ← g φ ( o t -K : t , a t -K : t -1 ) . detach() 11 Encode observation z t ∼ q θ ( z t | h t , o t , z t ) 12 Compute deterministic state h t = f θ ( h t -1 , z t -1 , a t -1 ) 13 Set latent state s t ← [ z t , h t ] 14 Update θ using representation learning (decoder, reward, continue). // Context representation learning ( φ, W z , W z ) 15 Draw B trajectory segments { ( o t -K : t , a t -K : t , o t +1 ) } ∼ D 16 Initialize losses L FD ← 0 , L cross ← 0 17 for each segment ( o t -K : t , a t -K : t , o t +1 ) do 18 Initialize h t -K ← 0 19 for τ = t -K to t do 20 if τ < K then 21 Pad o max(1 ,τ -K ): τ and a max(1 ,τ -K ): τ -1 with zeros. 22 z τ = g φ ( o τ -K : τ , a τ -K : τ -1 ) 23 z τ ∼ q θ ( z τ | h τ .detach() , o τ , z τ ) 24 h τ +1 = f θ ( h τ .detach() , z τ .detach() , a τ ) 25 ˜ z t = W z ( z t ) 26 ˜ z t = W z ( z t . detach() ) 27 L FD := ∥ o t +1 -f w φ ( o t , a t , z t ) ∥ 2 2 28 L cross := ∥ z t . detach() -˜ z t ∥ 2 2 + ∥ z t -˜ z t ∥ 2 2 29 L FD ← 1 B L FD, L cross ← 1 B L cross 30 Update φ, W z , W z (gradient descent) using L total = L FD + λ cross L cross // Behavior learning (Actor/Critic ϕ, ψ ) 31 Draw B sequences { ( o t , a t , r t ) } L t =1 ∼ D 32 Initialize h 0 ← 0 33 for all steps t in batch do 34 if t < K then 35 Pad o max(1 ,t -K ): t and a max(1 ,t -K ): t -1 with zeros. 36 Infer z t ← g φ ( o t -K : t , a t -K : t -1 ) . detach() // θ , φ frozen 37 Encode initial observation z t ∼ q θ ( z t | h t , o t , z t ) 38 Set initial latent state s t ← [ z t , h t ] 39 for imagination step τ = t to t + H do 40 Sample action a τ ∼ π ϕ ( a τ | s τ ) 41 Sample prior state ˆ z τ ∼ p θ (ˆ z τ | h τ ) 42 Compute next deterministic state h τ +1 = f θ ( h τ , ˆ z τ , a τ ) 43 Set latent state s τ +1 ← [ˆ z τ , h τ +1 ] 44 Compute value estimates V λ ( s τ ) using TD( λ ) 45 Update ϕ ← ϕ + α ∇ ϕ ∑ V λ 46 Update ψ ← ψ -α ∇ ψ ∑ 1 2 ∥ v ψ -V λ ∥ 2 2 // Environment interaction 47 o 1 ← env.reset() 48 for time step t = 1 , . . . , T do 49 if t < K then 50 Pad o max(1 ,t -K ): t and a max(1 ,t -K ): t -1 with zeros. 51 Infer context online: z t ← g φ ( o t -K : t , a t -K : t -1 ) 52 Compute s t ∼ p θ ( s t | s t -1 , a t -1 , o t ) 53 Compute a t ∼ π ϕ ( a t | s t ) 54 r t , o t +1 ← env.step ( a t ) 55 Add experience to dataset D ← D ∪ { ( o t , a t , r t ) T t =1 }
```

```
Algorithm 3: DALI-D (Deep Integration with Forward Dynamics Loss) 1 Model components Context encoder g φ ( o t -K : t , a t -K : t -1 ) One-step model f w φ ( o t , a t , z t ) Deterministic State f θ ( h t -1 , z t -1 , a t -1 , z t ) Stochastic State p θ (ˆ z t | h t ) Encoder q θ ( z t | h t , o t ) Reward p θ (ˆ r t | h t , z t , z t ) Continue p θ (ˆ n t | h t , z t , z t ) Decoder p θ (ˆ o t | h t , z t , z t ) Actor π ϕ ( a t | s t , z t ) Critic v ψ ( s t , z t ) Hyper parameters Seed episodes S Collect interval C Batch size B Sequence length L Imagination horizon H Learning rate α Learning rate g α g Episode length T Context history K 2 Initialize dataset D with S random seed episodes. 3 Initialize neural network parameters θ, ϕ, ψ, φ randomly. 4 while not converged do 5 for update step c = 1 , . . . , C do // Dynamics learning (World Model θ ) 6 Draw B sequences { ( o t , a t , r t ) } L t =1 ∼ D . 7 Initialize deterministic state h 0 ← 0 . 8 for all steps t from sequences batch D do 9 if t < K then 10 Pad o 1: t and a 1: t -1 with zeros to length K . 11 Compute context z t ← g φ ( o t -K : t , a t -K : t -1 ) . detach() 12 Encode observation z t ∼ q θ ( z t | h t , o t ) 13 Compute deterministic state h t = f θ ( h t -1 , z t -1 , a t -1 , z t ) 14 Set latent state s t ← [ z t , h t ] 15 Update θ using representation learning (decoder, reward, continue). // Context representation learning ( φ ) 16 Draw B data sequences { ( o t , a t ) } K +1 t =1 ∼ D 17 for each sample ( o 1: K +1 , a 1: K ) ∼ D do 18 Compute z 1: K = g φ ( o 1: K , a 1: K -1 ) 19 Update φ ← φ -α g ∇ φ ∑ K t =1 ∥ ∥ ∥ o t +1 -f w φ ( o t , a t , z t ) ∥ ∥ ∥ 2 2 // Behavior learning (Actor/Critic ϕ, ψ ) 20 Infer context: z t ← g φ ( o t -K : t , a t -K : t -1 ) . detach() // θ , φ frozen 21 Imagine trajectories { ( s τ , a τ ) } t + H τ = t from each s t (with fixed z t ) 22 Predict rewards E[ p θ ( r τ | s τ , z t )] and values v ψ ( s τ , z t ) . 23 Compute value estimates V λ ( s τ , z t ) . 24 Update ϕ ← ϕ + α ∇ ϕ ∑ t + H τ = t V λ ( s τ , z t ) . 25 Update ψ ← ψ -α ∇ ψ ∑ t + H τ = t 1 2 ∥ ∥ ∥ v ψ ( s τ , z t ) -V λ ( s τ , z t ) ∥ ∥ ∥ 2 . // Environment interaction 26 o 1 ← env.reset() 27 for time step t = 1 , . . . , T do 28 if t < K then 29 Pad o 1: t and a 1: t -1 with zeros to length K . 30 Infer context online: z t ← g φ ( o t -K : t , a t -K : t -1 ) . 31 Compute s t ∼ p θ ( s t | s t -1 , a t -1 , o t ) . 32 Compute a t ∼ π ϕ ( a t | s t , z t ) . 33 r t , o t +1 ← env.step ( a t ) . 34 Add experience to dataset D ← D ∪ { ( o t , a t , r t ) T t =1 } .
```

## Algorithm 4: DALI-Dχ (Deep Integration with Cross-modal Regularization)

```
Initialize dataset with random seed episodes
```

```
1 D S 2 Initialize parameters θ, ϕ, ψ, φ, W z , W z randomly 3 while not converged do 4 for update step c = 1 to C do // Dynamics learning (World Model θ ) 5 Draw B sequences { ( o t , a t , r t ) } L t =1 ∼ D 6 Initialize h 0 ← 0 7 for all steps t in batch do 8 if t < K then 9 Pad o max(1 ,t -K ): t and a max(1 ,t -K ): t -1 with zeros. 10 Compute context z t ← g φ ( o t -K : t , a t -K : t -1 ) . detach() 11 Encode observation z t ∼ q θ ( z t | h t , o t ) 12 Compute deterministic state h t = f θ ( h t -1 , z t -1 , a t -1 , z t ) 13 Set latent state s t ← [ z t , h t ] 14 Update θ using representation learning (decoder, reward, continue). // Context representation learning ( φ, W z , W z ) 15 Draw B trajectory segments { ( o t -K : t , a t -K : t , o t +1 ) } ∼ D 16 Initialize losses L FD ← 0 , L cross ← 0 17 for each segment ( o t -K : t , a t -K : t , o t +1 ) do 18 Initialize h t -K ← 0 for τ = t -K to t do 19 if τ < K then 20 Pad o max(1 ,τ -K ): τ and a max(1 ,τ -K ): τ -1 with zeros. 21 z τ = g φ ( o τ -K : τ , a τ -K : τ -1 ) 22 z τ ∼ q θ ( z τ | h τ .detach() , o τ ) 23 h τ +1 = f θ ( h τ .detach() , z τ .detach() , a τ , z τ .detach() ) 24 ˜ z t = W z ( z t ) 25 ˜ z t = W z ( z t .detach() ) 26 L FD := ∥ o t +1 -f w φ ( o t , a t , z t ) ∥ 2 2 27 L cross := ∥ z t .detach() -˜ z t ∥ 2 2 + ∥ z t -˜ z t ∥ 2 2 28 L FD ← 1 B L FD, L cross ← 1 B L cross 29 Update φ, W z , W z (gradient descent) using L total = L FD + λ cross L cross // Behavior learning (Actor/Critic ϕ, ψ ) 30 Draw B sequences { ( o t , a t , r t ) } L t =1 ∼ D 31 Initialize h 0 ← 0 32 for all steps t in batch do 33 if t < K then 34 Pad o max(1 ,t -K ): t and a max(1 ,t -K ): t -1 with zeros. 35 Infer z t ← g φ ( o t -K : t , a t -K : t -1 ) . detach() // θ , φ frozen 36 Encode initial observation z t ∼ q θ ( z t | h t , o t ) 37 Set initial latent state s t ← [ z t , h t ] 38 for imagination step τ = t to t + H do 39 Sample action a τ ∼ π ϕ ( a τ | s τ , z t ) 40 Sample prior state ˆ z τ ∼ p θ (ˆ z τ | h τ ) 41 Compute next deterministic state h τ +1 = f θ ( h τ , ˆ z τ , a τ , z t ) 42 Set latent state s τ +1 ← [ˆ z τ , h τ +1 ] 43 Compute value estimates V λ ( s τ , z t ) using TD( λ ) 44 Update ϕ ← ϕ + α ∇ ϕ ∑ V λ 45 Update ψ ← ψ -α ∇ ψ ∑ 1 2 ∥ v ψ -V λ ∥ 2 2 // Environment interaction 46 o 1 ← env.reset() 47 for time step t = 1 , . . . , T do 48 if t < K then 49 Pad o max(1 ,t -K ): t and a max(1 ,t -K ): t -1 with zeros. 50 Infer context online: z t ← g φ ( o t -K : t , a t -K : t -1 ) . 51 Compute s t ∼ p θ ( s t | s t -1 , a t -1 , o t ) . 52 Compute a t ∼ π ϕ ( a t | s t , z t ) . 53 r t , o t +1 ← env.step ( a t ) . 54 Add experience to dataset D ← D ∪ { ( o t , a t , r t ) T t =1 } .
```

## C Experimental Setup and Implementation Details

We evaluate DALI's zero-shot generalization on contextualized DMC Ball-in-Cup (gravity, string length) and Walker Walk (gravity, actuator strength) tasks [Tassa et al., 2018]. These environments feature context parameters with distinct training and evaluation ranges, presenting unique dynamics and generalization challenges.

## C.1 Training and Evaluation

Our setup follows the CARL benchmark [Benjamins et al., 2023], which defines default values for the two context dimensions per environment. To assess both interpolation and extrapolation (out-of-distribution, OOD) capabilities, we define two distinct context ranges: one for training and interpolation evaluation, and another for OOD evaluation. See Table 1. Additionally, we consider both Featurizedand Pixel-based observation modalities for each environment.

We consider two training scenarios. In the single context variation setting, each context dimension is varied independently by uniformly sampling 100 values within its respective training range, while the other dimension is held fixed at its default value. In the dual context variation setting, both context dimensions are varied jointly by uniformly sampling 100 context pairs from the Cartesian product of their respective training ranges.

Following Kirk et al. [2023], we evaluate our agents under three generalization regimes. The Interpolate regime tests performance on unseen context values sampled within the training range, while the Extrapolate regime assesses generalization to OOD contexts beyond the training range. The Mixed regime evaluates generalization when both context dimensions vary: one is sampled from the extrapolation range and the other from the interpolation range. Results in the Interpolate and Extrapolate regimes are aggregated over cases where one or both context dimensions vary.

## C.2 Environments

DMC Ball-in-Cup. This task involves an agent controlling a cup attached to a ball by a string, aiming to swing the ball into the cup, with context parameters gravity and string length. The ball's pendulum-like motion exhibits complex dynamics due to gravity's influence on oscillation frequency and string length's effect on swing amplitude. Extreme gravity values, such as 0 . 98 or 19 . 6 , significantly alter the ball's trajectory, while short or long strings (e.g., 0 . 03 or 0 . 6 ) change the swing's frequency and reach, creating complex interactions between these parameters. The interaction amplifies nonlinear effects, and poses significant generalization challenges, as the context encoder must adapt to substantial shifts in motion patterns, particularly in the OOD regime.

DMC Walker Walk. This task involves a planar bipedal robot walking forward, with context parameters gravity and actuator strength. Actuator strength scales torque applied to joint motors (e.g., hips, knees), modulating walking speed and stability, while gravity affects balance and ground reaction forces. Despite simple torque scaling, locomotion dynamics are intricate, driven by multijoint coordination and contact interactions. Generalization is less challenging than in Ball-in-Cup, as the actuator strength evaluation range ( 0 . 1 to 0 . 5 and 1 . 5 to 2 . 0 ) is closer to training ( 0 . 5 to 1 . 5 ), and its torque effects are predictable. However, extreme gravity values test the context encoder's ability to maintain stable, coordinated walking in the OOD regime.

Table 1: Context ranges for considered environments.

| Environment             | Context                   | Training/ Interpolation            | Extrapolation                                                             |
|-------------------------|---------------------------|------------------------------------|---------------------------------------------------------------------------|
| DMCball_in_cup-catch-v0 | gravity string length     | [4 . 9 , 14 . 7] [0 . 15 , 0 . 45] | [0 . 98 , 4 . 9) ∪ (14 . 7 , 19 . 6] [0 . 03 , 0 . 15) ∪ (0 . 45 , 0 . 6] |
| DMCwalker-walk-v0       | gravity actuator strength | [4 . 9 , 14 . 7] [0 . 5 , 1 . 5]   | [0 . 98 , 4 . 9) ∪ (14 . 7 , 19 . 6] [0 . 1 , 0 . 5) (1 . 5 , 2 . 0]      |

∪

## C.3 Architectural details

Figure 3 shows a high-level illustration of the DALI architecture.

Figure 3: DALI architecture overview.

<!-- image -->

Context Encoder. The context encoder g ϕ employs a standard transformer encoder block [Vaswani et al., 2017] to process a sequence of K transitions, ( o t -K : t , a t -K : t -1 ) , and produce the context representation z t ∈ R 8 . The input sequence is fed directly into a dense layer with 256 units, bypassing a traditional embedding layer. This is followed by layer normalization and a single-head self-attention block with a skip connection from the dense layer's output. The attention output is combined with the skip connection via a residual connection, followed by a second layer normalization. A two-layer MLP, each layer with 256 units, processes this output, with another residual connection combining the MLP output with the second layer norm's output. A final dense layer projects the 256-dimensional intermediate representation to z t ∈ R 8 . All hidden layers, including the MLP and attention head, use 256 units and SiLU activation functions, consistent with DreamerV3 [Hafner et al., 2025]. No masking is applied in the self-attention mechanism, as the input sequence is complete, enabling full contextual processing to generate z t .

Forward Model. The forward model f w φ in (1) for dynamics alignment is a two-layer MLP with 128 units and SiLU activations.

W z ∈ R 32 × 8 and W z t ∈ R 8 × 32 in (2) are learnable parameter matrices that map between the context and state spaces.

DreamerV3. We adopt the small DreamerV3 variant with hyperparameters from Hafner et al. [2025], following the setup of Prasanna et al. [2024] to ensure fair and reproducible comparison with their cRSSM-S/D baselines.

DALI adds only about 4% parameter overhead (e.g., Dreamer-DR: 15.73M vs. DALI-S: 16.45M) while consistently improving performance with minimal additional complexity.

## C.4 Computational Setup and Resource Requirements

Training and evaluation of the baselines and our DALI approaches were conducted on NVIDIA A100 GPUs with 80GB of VRAM and Intel Xeon Platinum 8352V CPUs. Typically, our setup provides access to 2 GPUs on average, with up to 4 GPUs available in the best-case scenario. Parallelization differs between modalities: featurized experiments can run 10 seeds in parallel, while pixel-based experiments are limited to 4 seeds due to higher memory requirements.

Our complete experimental setup includes 10 seeds across 5 algorithm variants, 2 environments, and 2 modalities, evaluated over 3 context settings ( single\_0 , single\_1 , double ), yielding a total of 600 individual training runs. Table 2 summarizes the computational requirements, where reported GPU hours account for 10 parallel processes in the featurized case and 4 in the pixel-based case. Note that training times per trial are similar for both variants, as the bottleneck lies in environment interaction rather than network size.

In a purely sequential execution, the total computational cost would reach approximately 24,000 GPU hours . Leveraging parallel execution significantly reduces wall-clock time: in the best-case scenario with 4 GPUs, total wall-clock time is approximately 1,051 GPU hours ( ≈ 44 days). In the more typical worst-case scenario with 2 GPUs, wall-clock time increases to approximately 2,101 GPU hours ( ≈ 88 days).

Table 2: Computational requirements for the full experimental evaluation. Each task is run across 5 algorithm variants with 10 seeds each in both featurized and pixel-based modalities. Wall-clock times assume optimal GPU utilization given the stated parallel capacities.

| Task              | Modality          | GPU Hrs per Trial   | Runs Total   | Wall-Clock (GPU Hrs)   | Wall-Clock (GPU Hrs)   |
|-------------------|-------------------|---------------------|--------------|------------------------|------------------------|
| Task              | Modality          | GPU Hrs per Trial   | Runs Total   | 2 GPUs                 | 4 GPUs                 |
| Ball in Cup       | Featurized        | 20-26               | 150          | 195                    | 98                     |
|                   | Pixel-based       | 20-26               | 150          | 488                    | 244                    |
| Walker            | Featurized        | 46-54               | 150          | 405                    | 203                    |
|                   | Pixel-based       | 46-54               | 150          | 1,013                  | 506                    |
| Total Featurized  | Total Featurized  | -                   | 300          | 600                    | 301                    |
| Total Pixel-based | Total Pixel-based | -                   | 300          | 1,501                  | 750                    |
| Grand Total       | Grand Total       | -                   | 600          | 2,101                  | 1,051                  |

## C.5 Additional Experiments

Figure 4 shows the learning curves for DMC Ball-in-Cup and Walker Walk tasks under Featurized and Pixel-based observation modalities.

DALI's context encoder adds only about 4% more parameters to DreamerV3-small while scaling effectively to high-dimensional tasks. We evaluate it on DMCQuadruped Walk (56D observations, 12D actions), a larger environment than Walker Walk (24D, 6D). The 56D observation space emphasizes the challenge of torque-sensitive locomotion, where increased coordination and a higher number of actuators (12 vs. 6) demand more robust context inference. Trained for 600K timesteps and contextualized with gravity and actuator strength (same ranges as Walker Walk), DALI attains Featurized Extrapolation IQMs of 0 . 326 ± 0 . 043 (DALI-S), and 0 . 389 ± 0 . 027 (DALI-Dχ ), yielding up to 76 . 8% improvement over the context-unaware baseline 0 . 220 ± 0 . 023 (Dreamer-DR), and up to 50 . 8% improvement over the context-aware baselines ( 0 . 258 ± 0 . 028 , cRSSM-D; 0 . 317 ± 0 . 031 , cRSSM-S).

## Ball-in-Cup

Figure 4: Learning curves for DMC Ball-in-Cup and Walker Walk tasks under Featurized and Pixel-based observation modalities. Results show mean episode returns with 25 th75 th percentile confidence intervals.

<!-- image -->

## D Supplementary Experiments on Physically Consistent Counterfactuals

We provide additional details supporting Section 6.3. First, we validate the statistical significance of the influence of latent dimensions on counterfactuals using AUC-based rankings (Section D.1). Next, we analyze actuator-driven recovery dynamics in the DMC Walker Walk task, analogous to the counterfactual analysis conducted for the Ball-in-Cup task in Section 6.3.1 (Section D.2).

## D.1 AUC Rankings and Significance of Counterfactual Perturbations

To identify latent dimensions that systematically alter imagined dynamics, we train a binary classifier to distinguish trajectories generated under perturbed ( T ′ ( j ) ) and original ( T (0) ) context representations. For each dimension z j , we generate N = 2500 trajectory pairs, perturbing z j by ∆ = σ ( z j ) to induce counterfactuals (see Section 6.3).

We implement an ensemble classifier comprising a support vector machine, an MLP, and an AdaBoost classifier, trained via stratified 5-fold cross-validation. Inputs are trajectory sequences T (0) (label 0) and T ′ ( j ) (label 1). Predictions are aggregated across folds to compute AUC. To ensure robustness, we compute 95% bootstrap confidence intervals for each z j 's AUC using 500 resamples. Permutation tests (1000 iterations) validate statistical significance between top-ranked dimensions by comparing observed AUC gaps to a shuffled null distribution.

For the Ball-in-Cup task, results are shown in Figure 5, where higher AUCs indicate dimensions whose perturbations induce physically distinguishable dynamics (e.g., gravity or string length changes). Confidence intervals highlight stability across bootstrap samples.

Figure 5: AUC ± 95% CI per context dimension z j for the Ball-in-Cup task.

<!-- image -->

## D.2 Actuator-Driven Recovery and Locomotion in Latent Imagination

Analogous to the Ball-in-Cup analysis (Section 6.3.1), we evaluate DALI's ability to generate physically consistent counterfactuals in the DMC Walker Walk task contextualized with gravity and actuator strength. We show that perturbations to the top-ranked latent dimension z 3 in DALI-S, identified via AUC-based rankings (see Section D.1), influence the Walker's capacity to recover from destabilizing falls and sustain locomotion, indicating that z 3 is mechanistically aligned with actuator strength. Imagined rollouts are initialized with a real observation and then rolled out in latent imagination. Unlike in the Ball-in-Cup analysis, we retain the original policy actions during imagination to examine how actuator strength modulates the Walker's recovery and locomotion behavior. For the Featurized modality, we tracked low-level state variables (e.g., torso height) at each timestep, while in the Pixel modality, we captured rendered environment frames at 5-step intervals over a 64-step imagination horizon.

Pixel Modality. Figure 6a contrasts the original (top) and counterfactual (middle) trajectories. Both agents initially fall (frames 0 -20 ), but the perturbed z 3 trajectory (counterfactual) exhibits enhanced actuator strength, enabling the Walker to stand from a challenging pose (left leg extended, right leg bent in frames 30 -45 ) and locomote forward (frames 50 -64 ). The original trajectory fails to recover, collapsing after the fall. Pixel-wise differences (bottom) highlight kinematic deviations, particularly in leg articulation and torso alignment.

Featurized Modality. Figure 6b shows torso height dynamics. While both trajectories share similar initial descent (time steps 0 -20 ), the counterfactual (orange) achieves sustained elevation (after time step 20), reflecting successful recovery and locomotion. This aligns with the pixel-based evidence of increased actuator strength, confirming z 3 's role in encoding torque dynamics critical for stability.

Figure 6: (a) (Pixel Modality) Counterfactual Trajectories in Pixel Space : Top: Original imagined trajectory of the Walker Walk under default gravity and actuator strength. Middle: Perturbed trajectory after adding noise ∆ to the top-ranked latent dimension z 3 . Original trajectory shows the Walker failing to recover after a fall. The perturbed trajectory with enhanced actuator strength enables recovery and forward locomotion. Bottom: Pixel-wise differences ( δ = | ˆ o t -ˆ o ′ t | ). The counterfactual agent refines leg kinematics (frames 30 -45 ) and stabilizes upright (frames 50 -64 ). (b) (Featurized Modality) Torso Z-Position Under Latent Perturbation : Comparison of original (blue) and counterfactual (orange) torso heights. The perturbed z 3 trajectory maintains higher elevation after time step 20, reflecting successful stabilization and locomotion.

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims presented in the abstract and introduction are fully supported by our theoretical and empirical findings, including a comprehensive analysis.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations are discussed in the corresponding section of this paper.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We provide a comprehensive proof of the necessity and benefit of our approach in Section 5 and Appendix A.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: This paper provides the complete code and instructions for running the experiments in a reproducible manner. Python dependencies, their versions, and all random seeds used are clearly specified.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The full source code is provided with this submission along with all relevant details required to reproduce the presented experiments.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: Yes, we provide all details for training the agent in-distribution and testing zero-shot generalization.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: For this purpose, we follow established methods for measuring the agent's performance [Agarwal et al., 2021].

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: DALI adds only a small context encoder module to DreamerV3. As reported by Hafner et al. [2025], DreamerV3 can be trained on a single GPU, and our extension has the same requirement. Each trial with a single random seed takes roughly 1-2 days, depending on the observation modality.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: No disagreement amoung the team.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: [NA]

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: [NA]

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: This work builds upon and extends the codebase of Prasanna et al. [2024], which is properly credited. All associated licenses and terms of use from the original work are respected.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: [NA]