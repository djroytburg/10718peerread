## Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks: Theoretical and Empirical Evidence

Shaopeng Fu 1 Liang Ding 2 Jingfeng Zhang 3 , 1 Di Wang 1 *

1 King Abdullah University of Science and Technology 2 The University of Sydney 3 The University of Auckland shaopeng.fu@kaust.edu.sa , liangding.liam@gmail.com jingfeng.zhang@auckland.ac.nz , di.wang@kaust.edu.sa

## Abstract

Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e. , training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. While long-length adversarial prompts during AT might lead to strong LLM robustness, their synthesis however is very resource-consuming, which may limit the application of LLM AT. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length Θ( M ) , it is enough to align LLMs on prompts with adversarial suffixes of length Θ( √ M ) . Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term Θ( √ M test /M train ) , where M train and M test are the numbers of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix length during jailbreaking to the length during AT. Our findings show that it is practical to defend against 'longlength' jailbreak attacks via efficient 'short-length' AT. The code is available at https://github.com/fshp971/adv-icl .

## 1 Introduction

Large language models (LLMs) [5, 51, 28, 65] are widely adopted in various real-world applications to assist human users [55, 67, 57, 56, 24], but their safety is found to be vulnerable toward jailbreak attacks [60]. With carefully crafted adversarial prompts, one can 'jailbreak' the safety mechanism of LLMs and induce arbitrary harmful behaviors [73, 7, 30]. To tackle the challenge, recent studies [63, 36, 68, 6] propose performing safety alignment through adversarial training (AT) [32] to enhance LLMs' robustness against jailbreaking. A standard AT for LLMs would train them on jailbreak prompts synthesized by strong jailbreak attacks to learn to refuse these harmful instructions [36].

In such AT, the length of synthesized adversarial prompts used for model training is critical to the final jailbreak robustness of LLMs. [3] and [64] have shown that longer adversarial prompts enjoy stronger

* Corresponding Author

jailbreaking abilities. Thus, it is reasonable to deduce that performing AT with longer adversarial prompts can help LLMs achieve stronger robustness to defend against 'long-length' jailbreak attacks. However, synthesizing long-length adversarial prompts in adversarial training is resource-consuming since it requires solving discrete optimization problems in high-dimensional spaces, which thus needs lots of GPU memory and training time. This may limit the application of AT in LLMs' safety alignment and further raises the following research question: How will the adversarial prompt length during AT affect trained LLMs' robustness against jailbreaking with different prompt lengths?

We study this research question by analyzing suffix jailbreak attacks , where each jailbreak prompt is constructed by concatenating a harmful instruction with a synthesized adversarial suffix. Our main finding is: To defend against a suffix jailbreak attack with suffix length of Θ( M ) , it is enough to adversarially train LLMs on adversarial prompts with suffix length of only Θ( √ M ) . In other words, we show that it is possible to defend long-length jailbreak attacks via efficient short-length AT.

Our finding is supported by theoretical and empirical evidence. Theoretically, we leverage the in-context learning theory [53, 69] to investigate how linear transformers learn linear regression tasks from in-context task samples under AT. To better simulate suffix jailbreak attacks in real-world LLMs, our analysis introduces a new in-context adversarial attack . Concretely, for any in-context task sample, this attack will adversarially perturb the last several in-context training points to maximize the squared prediction error that linear transformers made on the in-context test point. Under our theoretical framework, we prove a robust generalization bound for adversarially trained linear transformers. This bound has a positive correlation with the term Θ( √ M test /M train ) , where M train and M test are the number of perturbed in-context points in training and testing in-context task samples, respectively.

Empirically, we conduct AT with the GCG attack [73], one of the most effective jailbreak attacks, under various adversarial suffix lengths on five popular real-world LLMs and evaluate their robustness against jailbreak attacks with different adversarial suffix lengths. We use the jailbreak attack success rate (ASR) to express the robust generalization error of trained LLMs and find that this ASR has a clear positive correlation with the ratio of the square root of test-time adversarial suffix length to the AT adversarial suffix length. Such a correlation empirically verifies our main finding. We also find that AT with an adversarial suffix (token) length of 20 is already able to reduce the ASR of jailbreak attacks with an adversarial suffix (token) length of up to 120 by at least 30% in all experiments.

## 2 Related works

Jailbreak attacks. Jailbreaking [60] can be seen as adversarial attacks [49, 14] toward LLMs, which aim to synthesize adversarial prompts to induce targeted harmful behaviors from LLMs. Many efforts have been made on token-level jailbreak attacks, i.e. , searching adversarial prompts in the token space of LLMs, which can be achieved via gradient-based optimization [48, 16, 73, 26, 45, 71], heuristic greedy search [44, 17, 22], or fine-tuning prompt generators from pre-trained LLMs [38]. Other attempts include word-level adversarial prompt searching [30] or directly prompting LLMs to generate adversarial prompts [7, 29]. Our work focuses on token-level jailbreaking since it make it easier for us to control the adversarial prompt length for our analysis. More recent studies have found that increasing the length of adversarial prompts by adding more harmful demonstrations [3, 54, 61] or synthesizing longer adversarial suffixes [64] can make jailbreaking more effective. These works motivate us to investigate the problem of defending against 'long-length' jailbreak attacks.

Adversarial training on LLMs. To defend against jailbreak attacks, a large body of studies focus on aligning LLMs to refuse responding jailbreak prompts [37, 42, 40, 41, 8]. More recent works have started to adopt adversarial training (AT) [32] to align LLMs. [36] trained LLMs on (discrete) adversarial prompts synthesized by GCG attack [73], in which they cached the intermediate synthesized results to reduce the heavy cost of searching adversarial prompts from scratch. Meanwhile, various studies [63, 6, 46, 68] conduct AT with adversarial examples found in the continuous embedding space rather than the discrete text space since searching in the continuous embedding space is more computationally efficient. Nevertheless, as a preliminary study of the length of adversarial prompts during AT, our work only analyzes AT with discrete adversarial prompts.

In-context learning theory (ICL). Transformer-based large models like LLMs are strong in performing ICL: Given a series of inputs (also known as 'prompt') specified by a certain task, LLMs can make predictions well for this certain task without adjusting model parameters. Current theories in understanding ICL can be roughly divided into two categories. The first category aims to under-

stand ICL via constructing explicit multi-layer transformers to simulate the optimization process of learning function classes [13, 53, 1, 9, 34, 59]. The second category focuses on directly analyzing the training [69, 66, 19, 62, 27] and generalization [31, 33, 11, 47] of simple self-attention models ( i.e. , one-layer transformer). [4] is the first to study adversarial attacks against linear transformers and finds that an attack can always succeed by perturbing only a single in-context sample. However, their analysis allows samples to be perturbed in the entire real space, which might not appropriately reflect real-world settings since real-world adversarial prompts can only be constructed from token/character spaces of limited size. Unlike [4], we propose a new ICL adversarial attack that requires each adversarial suffix token to be perturbed only within restricted spaces, which thus can be a better tool for understanding real-world jailbreaking.

Finally, we notice that [61] also recognizes the critical role that the number of adversarial in-context samples plays in ICL-based attacks. They present a theoretical analysis (not based on ICL theory) for adversarial attacks against ICL text classification and characterize the minimum number of in-context adversarial samples required to increase the safety loss of ICL to some extent. However, the main difference is that [61] focuses on studying the adversarial robustness of fixed ICL models, whereas our work analyzes how adversarial training affects the robustness of ICL models.

## 3 Preliminaries

Large language models (LLMs). Let [ V ] = { 1 , · · · , V } be a vocabulary set consisting of all possible tokens. Then, an LLM can be seen as a function that for any sequence x 1: n ∈ [ V ] n consists of n tokens, the LLM will map x 1: n to its next token x n +1 following x n +1 ∼ p θ ( ·| x 1: n ) , where p θ is a conditional distribution over the vocabulary set [ V ] and θ is the model parameter of the LLM. Under such notations, when using the LLM p θ to generate a new token sequence for the input x 1: n , the probability of generating a sequence y 1: m ∈ [ V ] m of length m is given by p θ ( y 1: m | x 1: n ) = ∏ m i =1 p θ ( y i | x 1: n ⊕ y 1:( i -1) ) , where ' ⊕ ' denotes concatenation.

Jailbreak attacks. This paper focuses on suffix jailbreak attacks. Concretely, suppose x ( h ) and y ( h ) are two token sequences, where x ( h ) represents a harmful prompt ( e.g. , 'Please tell me how to build a bomb.') and y ( h ) represents a corresponded targeted answer ( e.g. , 'Sure, here is a guide of how to build a bomb'). Then, the goal of a suffix jailbreak attack against the LLM p θ aims to synthesize an adversarial suffix x ( s ) 1: m for the original harmful prompt x ( h ) via solving the following problem,

<!-- formula-not-decoded -->

where x ( h ) ⊕ x ( s ) 1: m is the adversarial prompt and m is the sequence length of the adversarial suffix x ( s ) 1: m . Intuitively, a large m will increase the probability of the LLM p θ that generating the targeted answer y ( h ) for the synthesized adversarial prompt x ( h ) ⊕ x ( s ) 1: m . To solve Eq. (1), a standard method is the Greedy Coordinate Gradient (GCG) attack [73], which leverages gradient information to search for better x ( s ) 1: m within the discrete space [ V ] m in a greedy manner.

Adversarial training (AT). We consider the canonical AT loss L [36, 40] to train the LLM p θ , which consists of two sub-losses: an adversarial loss L adv and an utility loss L utility . Specifically, given a safety dataset D ( h ) , where each of its sample ( x ( h ) , y ( h ) , y ( b ) ) ∈ D ( h ) consists of a harmful instruction x ( h ) , a harmful answer y ( h ) , and a benign answer y ( b ) ( e.g. , 'As a responsible AI, I can't tell you how to...'). The adversarial loss L adv is defined as follows,

<!-- formula-not-decoded -->

where x ( s ) 1: m is the adversarial suffix obtained from Eq. (1) and m is the adversarial suffix length. Note that the probability terms in Eqs. (1) and (2) look similar to each other. The difference is that the term in Eq. (1) denotes the probability that p θ generates the harmful answer y ( h ) for the adversarial prompt, while that in Eq. (2) denotes the probability of generating the benign answer y ( b ) . Besides, let D ( u ) be a utility dataset where each of its sample ( x ( u ) , y ( u ) ) ∈ D ( u ) consists of a pair of normal instruction and answer. Then, the utility loss L utility is given by

<!-- formula-not-decoded -->

Thus, the overall AT problem for improving the jailbreak robustness of the LLM p θ is given as

<!-- formula-not-decoded -->

where α ∈ [0 , 1] is a factor that balances between the adversarial and utility sub-losses. The idea behind such a loss design is that: (1) help LLM learn to respond harmlessly even when strong jailbreak prompts present (achieved via L adv), (2) retain the utility of LLM gained from pre-training (achieved via L utility). Intuitively, a larger adversarial suffix length m during AT will help the LLM gain robustness against jailbreak attacks with longer adversarial suffixes.

## 4 Theoretical evidence

This section establishes the theoretical foundation of how 'short-length' AT can defend against 'long-length' jailbreaking. Our analysis is based on the in-context learning (ICL) theory [69, 47, 4], and we will bridge the ICL theory and the LLM AT problem defined in Eq. (3) later (in Section 4.2). Here we first introduce the necessary notations to describe the problem. To avoid confusion, we note that all notations in this section will only be used within this section and have no relevance to those in other sections ( e.g. , Section 3).

In-context learning (ICL). In the ICL theory, a prompt with length N related to a specific task indexed by τ is defined as ( x τ, 1 , y τ, 1 , · · · , x τ,N , y τ,N , x τ,q ) , where x τ,i ∈ R d is the i -th in-context training sample, y τ,i ∈ R is the label for the i -th training sample, and x τ,q ∈ R d is the in-context query sample. Then, the task-specific ICL input E τ is defined as

<!-- formula-not-decoded -->

Given an ICL input E τ of task τ , the goal of an ICL model is to make a prediction based on E τ for the query sample x τ,q . Such an ICL model design aims to model the ability of real-world LLMs in making decisions based on prompting without updating model parameters.

Linear self-attention (LSA) models. LSA models are a kind of linear transformer that has been widely adopted in existing theoretical ICL studies. [2] empirically show that LSA models share similar properties with non-linear ones and thus are useful for understanding transformers. We follow [69] to study the following single-layer LSA model,

<!-- formula-not-decoded -->

where θ := ( W V , W KQ ) is the model parameter, W V ∈ R ( d +1) × ( d +1) is the value weight matrix, W KQ ∈ R ( d +1) × ( d +1) is a matrix merged from the key and query weight matrices of attention models, E τ ∈ R ( d +1) × ( N +1) is the task-specific ICL input, and N is the prompt length. The prediction ˆ y q,θ for the query sample x τ,q is given by the right-bottom entry of the output matrix of the LSA model, i.e. , ˆ y q,θ ( E τ ) := f LSA ,θ ( E τ ) ( d +1) , ( N +1) . We further follow [69] to denote that

<!-- formula-not-decoded -->

where □ ∈ { V, KQ } , W □ 11 ∈ R d × d , w □ 12 , w □ 21 ∈ R d × 1 and w □ 22 ∈ R . Under this setting, the model prediction ˆ y q,θ can be further simplified as follows,

<!-- formula-not-decoded -->

Other notations. For any n ∈ N + , we denote [ n ] := { 1 , · · · , n } . For any A ∈ R n × m , we denote ∥ A ∥ 2 , ∞ := max 1 ≤ i ≤ m ∥ A i, : ∥ 2 , ∥ A ∥ 2 be the operator norm, and ∥ A ∥ F be the Frobenius norm. For any A ∈ R n × n , we denote Tr( A ) := ∑ n i =1 A i,i . We use standard big O notations O ( · ) and Θ( · ) .

## 4.1 Problem definition for adversarial ICL

We now formalize the AT problem in ICL with the previously introduced notations. We focus on the linear regression task and introduce a novel ICL 'suffix' adversarial attack, where in-context adversarial points are appended to the end of ICL inputs, to analyze the robustness of LSA models.

In-context linear regression. For any task indexed by τ , we assume that there is a task weight w τ ∈ R d drawn from w τ ∼ N (0 , I d ) . Besides, for any in-context training point x τ,i (1 ≤ i ≤ N ) and the query point x τ,q (see Eq. (4)), we assume that they are drawn from x τ,i , x τ,q ∼ N (0 , Λ) , where Λ ∈ R d × d is a positive-definite covariance matrix. Moreover, the ground-truth labels of training points x τ,i and the query point x τ,q are given by y τ,i = w ⊤ τ x τ,i and y τ,q = w ⊤ τ x τ,q .

ICL suffix adversarial attack. Our novel adversarial attack against ICL models is launched via concatenating (clean) prompt embedding matrices with adversarial embedding suffixes. Specifically, for an ICL input E τ of length N (see Eq. (4)), we will form its corresponding adversarial ICL input E adv τ,M ∈ R ( d +1) × ( N + M +1) by concatenating E τ with an adversarial suffix of length M as follows,

<!-- formula-not-decoded -->

where X τ := ( x τ, 1 · · · x τ,N ) ∈ R d × N and Y τ := ( y τ, 1 · · · y τ,N ) ∈ R 1 × N denote the N original training samples and labels, and X sfx τ := ( x sfx τ, 1 · · · x sfx τ,M ) ∈ R d × M , Y sfx τ := ( y sfx τ, 1 · · · y sfx τ,M ) ∈ R 1 × M , and ∆ sfx τ := ( δ τ, 1 · · · δ τ,M ) ∈ R d × M denote the new M clean suffix samples, clean suffix labels, and adversarial perturbations. The clean suffix samples X sfx τ and labels Y sfx τ here follow the same distribution as those in-context data in the embedding E τ , i.e. , x sfx τ,i ∼ N (0 , Λ) and y sfx τ,i = w ⊤ τ x sfx τ,i hold for every i ∈ [ M ] . For the adversarial perturbation matrix ∆ τ , we require each perturbation δ τ,i is restricted within a ball-sphere as ∥ δ τ,i ∥ 2 ≤ ϵ , where ϵ &gt; 0 is the perturbation radius. This aims to simulate that in jailbreak attacks, and each adversarial token is searched within a token vocabulary set of limited size.

The goal of the ICL adversarial attack is to add an optimal suffix adversarial perturbation matrix ∆ τ to maximize the difference between the model prediction ˆ y q ( E adv τ ) based on the adversarial ICL input E adv τ and the ground-truth query label y τ,q . We adopt the squared loss to measure such a prediction difference, which thus leads to the robust generalization error for the model f LSA θ as

<!-- formula-not-decoded -->

where M is the length of the adversarial suffix and the expectation E τ [ · ] is calculated over the randomness of w τ , X τ , X sfx τ , and x τ,q . As we aim to understand how the adversarial prompt length in AT would affect the robustness of LLM, Eq. (7) will also only focus on how the adversarial suffix length M in ICL adversarial attacks would affect the robust generalization error R adv ( θ, M ) .

Adversarial in-context learning. Following previous studies on minimax AT [32, 20, 43, 12, 58], here we adopt a minimax AT loss to train the LSA model. Concretely, we first use the aforementioned ICL adversarial attack to synthesize adversarial prompts and then update the LSA model based on these adversarial prompts to help the model gain robustness against them. We further assume that the adversarial suffix length is fixed during AT, which thus leads to the following ICL AT problem,

<!-- formula-not-decoded -->

where L adv ( θ ) := R adv ( θ, M train ) is the AT loss in ICL and M train ∈ N + is the fixed adversarial suffix length during AT. We will perform AT with continuous gradient flow, and further following [69] to make the following assumption on the LSA model parameter initialization.

̸

Assumption 1 (c.f. Assumption 3 in [69]) . Let σ &gt; 0 be a parameter and Θ ∈ R d × d be any matrix satisfying ∥ ΘΘ ⊤ ∥ F = 1 and ΘΛ = 0 d × d . We assume

<!-- formula-not-decoded -->

Recall in Eq. (5), w V 12 , w KQ 12 , and w KQ 22 do not contribute to the model prediction ˆ y q,θ ( · ) . Assumption 1 thus directly sets them to be zero at initialization. To ensure symmetric initialization, it further sets w V 21 (0) and w KQ 21 (0) to zero. We will see how Assumption 1 helps simplify the analysis of ICL AT.

## 4.2 Bridging ICL AT and LLM AT

We now discuss similarities between the ICL AT problem defined in Eq. (8) and the LLM AT problem defined in Eq. (3) to help motivate why ICL AT can be a good artifact to theoretically study LLM AT.

Firstly, in-context inputs ( i.e. , E τ defined in Eq. (4)) for LSA models are similar to prompt inputs for real-world LLMs. If we replace each token in an LLM prompt with its one-hot encoding defined over the token vocabulary space, then these one-hot encodings would be similar to in-context samples x i in Eq. (4) since both of them are now 'feature vectors'. Besides, we note that each in-context label y i in Eq. (4) can be seen as the 'next-token prediction label' in real-world LLMs. The main difference is that in LLMs, the i -th token in a prompt can be seen as the i -th input token and the ( i -1) -th next-token prediction label simultaneously, while in LSA models, the i -th in-context input and the ( i -1) -th in-context label are explicitly separated into two terms x i and y i -1 .

Secondly, the search for adversarial in-context samples (see Eq. (6)) in the ICL suffix adversarial attack is similar to the search for adversarial tokens in suffix jailbreak attacks. We note that each adversarial token in jailbreak prompts can be seen as replacing the 'padding token'. Thereby, from the point of view of one-hot encoding, searching for an adversarial token can thus be seen as applying an ℓ 2 -norm adversarial perturbation within a radius of √ 2 to transform the one-hot encoding of the padding token to that of the adversarial token. This process is the same as the search for adversarial in-context samples in the ICL suffix adversarial attack defined in Eq. (7), which would perturb each in-context suffix sample x sfx τ,i within an ℓ 2 -norm ball-sphere under a given radius ϵ &gt; 0 .

Thirdly, motivations behind ICL AT and LLM AT are also similar to each other. Both of the two AT problems aim to enhance models' robustness via training them on some synthesized adversarial inputs. The adversarial inputs syntheses in ICL AT and LLM AT are also similar, as both of them aim to make targeted models behave wrongly via manipulating suffixes of input prompts. The difference is that suffix jailbreak attacks are targeted adversarial attacks aimed at inducing LLMs to generate specified harmful content while our ICL attack is an untargeted adversarial attack aimed at reducing the utility of linear regression prediction made by LSA models.

## 4.3 Training dynamics of adversarial ICL

We now start to analyze the training dynamics of the minimax ICL AT problem formalized in Eq. (8). The main technical challenge is that to solve the inner maximization problem in Eq. (8), one needs to analyze the optimization of the adversarial perturbation matrix ∆ τ . However, the matrix ∆ τ along with the clean data embedding E τ and the clean adversarial suffix ( X sfx τ , Y sfx τ ) are entangled together within the adversarial ICL input E adv τ,M train , which makes it very difficult to solve the inner maximization problem and further analyze the ICL AT dynamics.

To tackle such a challenge, we propose to instead study the dynamics of a closed-form upper bound of the original AT loss L adv ( θ ) . Formally, we will analyze the following surrogate AT problem:

<!-- formula-not-decoded -->

The surrogate AT loss ˜ L adv ( θ ) in Eq. (9) is the closed-form upper bound for the original AT loss L adv ( θ ) in Eq. (8), as illustrated in the below Proposition 1 (see Appendix A.2 for the proof).

Proposition 1 (Uniform upper bound for L adv ( θ ) ) . For the AT loss function L adv ( θ ) defined in Eq. (8) and the surrogate AT loss function ˜ L adv ( θ ) defined in Eq. (9), for any model parameter θ := ( W V , W KQ ) of the LSA model f LSA ,θ , we uniformly have that: L adv ( θ ) ≤ ˜ L adv ( θ ) .

This result indicates that when we are training the LSA model via solving the surrogate AT problem Eq. (9), we are also reducing the model training loss in the original AT problem Eq. (8). Thus, solving the surrogate AT problem will also intuitively improve the robustness of the model.

Based on our previous analysis, we now turn to study the training dynamics of surrogate AT defined in Eq. (9). To better describe our results, we define two functions Γ( · ) : N → R d × d and ψ ( · ) : N → R , both of which depend on the adversarial suffix length M , as follows,

<!-- formula-not-decoded -->

where N is the prompt length of the original ICL input E τ (see Eq. (4)) and Λ is the covariance matrix of in-context linear regression samples. The closed-form surrogate AT dynamics of the LSA model f LSA ,θ is then given in the following Theorem 1 (see Appendix A.3 for the proof).

Theorem 1 (Closed-form Surrogate AT Dynamics) . Suppose Assumption 1 holds and f LSA ,θ is trained from the surrogate AT problem defined in Eq. (9) with continuous gradient flow. When the σ in Assumption 1 satisfies σ &lt; √ 2 d ·∥ (Γ( M train )Λ+ ϵ 2 ψ ( M train ) I d )Λ -1 ∥ 2 , after training for infinite long time, the model parameter θ will converge to θ ∗ ( M train ) := ( W V ∗ ( M train ) , W KQ ∗ ( M train )) , satisfying: w KQ ∗ , 12 = w KQ ∗ , 21 = w V ∗ , 12 = w V ∗ , 21 = 0 d × 1 , w KQ ∗ , 22 = 0 , W V ∗ , 11 = 0 d × d , and

<!-- formula-not-decoded -->

Remark 1. When the l 2 -norm adversarial perturbation radius ϵ is zero, the closed-form AT solution θ ∗ derived in Theorem 1 degenerates to that obtained without AT (see Theorem 4.1 in [69]). Thus, a sufficient large adversarial perturbation ϵ is a key to helping the LSA model f LSA ,θ obtain significant adversarial robustness. This will be further justified in the next section.

## 4.4 Robust generalization upper-bound

With the closed-form AT solution θ ∗ ( M train ) in Theorem 1, we now analyze the robustness of the trained LSA model. All proofs in this section are presented in Appendix A.4. We study how a LSA model adversarially trained under a fixed adversarial suffix length M train can defend against the ICL adversarial attack with a different adversarial suffix length M test . That is, we aim to analyze the magnitude of the robust generalization error R adv ( θ ∗ ( M train ) , M test ) for the converged robust model parameter θ ∗ ( M train ) . Here, we prove an upper-bound for it in the following theorem.

Theorem 2 (Surrogate AT Robust Generalization Bound) . Suppose all conditions in Theorem 1 hold and θ ∗ ( M train ) is the surrogate AT solution in Theorem 1. We have

<!-- formula-not-decoded -->

where M train is the adversarial suffix length in the ICL adversarial attack, and Γ train := Γ( M train ) , Γ test := Γ( M test ) , ψ train := ψ ( M train ) , and ψ test := ψ ( M test ) are functions in Eq. (10).

We further adopt Assumption 2 to help us better understand our robust generalization bound.

Assumption 2. For adversarial suffix lengths during AT and testing, we assume that M train , M test ≤ O ( N ) , where N is the original ICL prompt length. Besides, for the l 2 -norm adversarial perturbation radius, we assume that ϵ = Θ( √ d ) , where d is the ICL sample dimension.

In the above Assumption 2, the assumption made on adversarial suffix lengths means that they should not be too long to make the model 'forget' the original ICL prompt. Besides, the assumption made on the perturbation radius ϵ ensures that it is large enough to simulate the large (but limited) token vocabulary space of real-world LLMs to help model gain robustness.

Corollary 1. Suppose Assumption 2 and all conditions in Theorem 2 hold. Suppose ∥ Λ ∥ 2 ≤ O (1) . Then, we have the following robust generalization bound,

<!-- formula-not-decoded -->

Figure 1: Scatter plots of ASR to the ratio √ M test /M train. For each pair of base model and attack, 48 points are plotted. A high ASR indicates a weak jailbreak robustness.

<!-- image -->

Corollary 1 is our main theoretical result, which clearly show that for an adversarially trained LSA model, its robust generalization bound depends on the term Θ( √ M test /M train ) , where M train and M test are the number of adversarially perturbed in-context samples during training and testing. In other words, for an ICL adversarial attack with an adversarial suffix length Θ( M ) , to maintain the order of the robust generalization bound, it is enough to perform surrogate AT with only an adversarial suffix length Θ( √ M ) . Such an observation is useful in practice, since one can thus leverage a 'short-length' AT, which requires less GPU memory and training time, to defend against 'long-length' jailbreakings.

## 5 Empirical evidence

In this section, we follow Eq. (3) to perform AT on LLMs and investigate the relationship between adversarial suffix lengths during LLM AT and jailbreak attacks.

## 5.1 Experimental setup

Models&amp;datasets. We adopt five pre-trained LLMs, which are: Vicuna-7B-v1.5 [70], Mistral-7BInstruct-v0.3 [21], Llama-2-7B-Chat [52], Llama-3-8B-Instruct [15], and Qwen2.5-7B-Instruct [65]. For data in AT, we use the training set from Harmbench [36] as the safety dataset and Alpaca [50] as the utility dataset. For data in the robustness evaluation, we construct a test set of size 100 that consists of the first 50 samples from the test set of Harmbench [36] and the first 50 samples from AdvBench [73]. For data in the utility analysis, we use the benchmark data from AlpacaEval [10].

Adversarial training. We leverage GCG [73], a token-level jailbreak attack, to synthesize (suffix) jailbreak prompts, in which the adversarial suffix length M train is fixed to one of { 5 , 10 , 20 , 30 , 40 , 50 } during AT. To reduce computational complexity of tuning LLMs, LoRA [18] is applied to all query and key projection matrices in attentions. In every AT experiment, we follow Eq. (3) to perform AT with Adam. Please refer to Appendix B.2 for omitted settings.

Jailbreak attacks. We use both suffix and non-suffix jailbreak attacks to evaluate the adversarial robustness of trained LLMs. Specifically, five token-level suffix jailbreak attacks are adopted, which are GCG [73], BEAST [44], AmpleGCG [26], Zhu's AutoDAN [71], and GCQ [17]. The adversarial suffix token length M test is varied within { 5 , 10 , 20 , 40 , 60 , 80 , 100 , 120 } . Meanwhile, two nonsuffix jailbreak attacks are leveraged, which are PAIR [7] and DeepInception [25]. Please refer to Appendix B.3 for full implementation details of all used jailbreak attacks.

Evaluations. We focus on evaluating the jailbreak robustness and the utility of trained LLMs. For the robustness evaluation, we report the Attack Success Rate (ASR) of jailbreak attacks. An LLM-based judger from [36] is used to determine whether a jailbreak attack succeeds or not. For the utility evaluation, we use the AlpacaEval2 [10] to report the Length-controlled WinRate (LC-WinRate) of targeted models against a reference model Davinci003 evaluated under the Llama-3-70B model. An LC-WinRate of 50% means that the output qualities of the two models are equal, while an LCWinRate of 100% means that the targeted model is consistently better than the reference Davinci003. Please refer to Appendix B.3 for the detailed settings of model evaluations.

## 5.2 Results analysis

Correlation between the suffix jailbreak robustness and the ratio √ M test /M train . Weplot the ASR of models trained and attacked with different adversarial suffix lengths in Figure 1. This results in 48 points for each pair of base model and jailbreak attack. The Pearson correlation coefficient (PCC)

Table 1: PCCs and p -values calculated between ASR and ratio √ M test /M train. A high PCC (within [ -1 , 1] ) means a strong correlation between ASR and the ratio. p &lt; 5 . 00 × 10 -2 means that the observation is considered statistically significant.

| Model      | GCG Attack   | GCG Attack      | BEAST Attack   | BEAST Attack   | AmpleGCG Attack   | AmpleGCG Attack   | AmpleGCG Attack   | Zhu's AutoDAN   | Zhu's AutoDAN   | Zhu's AutoDAN   | GCQ Attack   | GCQ Attack    | GCQ Attack   |
|------------|--------------|-----------------|----------------|----------------|-------------------|-------------------|-------------------|-----------------|-----------------|-----------------|--------------|---------------|--------------|
|            | PCC( ↑ )     | p -value( ↓ )   | PCC( ↑ )       | p -value( ↓    | )                 | PCC( ↑ )          | p -value( ↓ )     | PCC( ↑          | )               | p -value( ↓ )   | PCC( ↑ )     | p -value( ↓ ) |              |
| Vicuna-7B  | 0 . 93       | 4 . 7 × 10 - 21 | 0 . 63         | 1 . 4 × 10     | - 6               | 0 . 19            | 1 . 9 × 10 - 1    | 0 .             | 51              | 2 . 5 × 10 - 4  | 0 . 82       | 1 . 4 × 10 -  |              |
| Mistral-7B | 0 . 86       | 4 . 0 × 10 - 15 | 0 . 29         | 4 . 4 × 10     | - 2               | 0 . 74            | 1 . 5 × 10 - 9    | 0 . 49          | 3 . 7 ×         | 10 - 4          | 0 . 70       | 2 . 6 × 10 -  |              |
| Llama-2-7B | 0 . 88       | 9 . 0 × 10 - 17 | 0 . 67         | 1 . 7 × 10     | - 7               | 0 . 37            | 1 . 0 × 10        | - 2             | 0 . 13          | 3 . 8 × 10 - 1  | 0 . 71       | 2 . 1 × 10 -  |              |
| Llama-3-8B | 0 . 76       | 2 . 8 × 10 - 10 | 0 . 26         | 7 . 7 × 10     | - 2               | - 0 . 07          | 6 . 2 × 10 -      | 1 -             | 0 . 12          | 4 . 1 × 10 - 1  | 0 . 0        | 9 . 7 × 10 -  |              |
| Qwen2.5-7B | 0 . 87       | 1 . 1 × 10 - 15 | 0 . 58         | 1 . 0 × 10     | - 5               | - 0 . 24          | 1 . 0 × 10        | - 1             | 0 . 16          | 2 . 6 × 10 - 1  | 0 . 72       | 1 . 1 × 10 -  |              |

Figure 2: ASR versus M train on Vicuna-7B-v1.5 under jailbreaking with different M test . M train = 0 means that AT is not performed on the evaluated model. A low ASR indicates a strong robustness.

<!-- image -->

and the corresponding p -value between the ratio √ M test /M train and the ASR are calculated in Table 1, where bold p -values indicate that observations are statistically significant ( i.e. , p &lt; 0 . 05 ), while underlined ones indicate they are not significant.

When the jailbreak attack used during AT is the same as that used during robustness evaluation ( i.e. , GCG), one can observe from Figure 1 that a clear positive correlation between the ratio √ M test /M train and the ASR for all evaluated base models. Further, high PCCs ( &gt; 0 . 7 ) and low p -values ( &lt; 0 . 05 ) in Table 1 also confirm that the observed positive correlation is statistically significant.

Besides, when the jailbreak attack is BEAST and GCQ, which is different from that used during AT, the significant positive correlation between the ratio √ M test /M train and the ASR can only be observed from some of the base models. This may be due to the fact that AT with only a single jailbreak attack may not help the model generalize well to unseen attacks. Therefore, it might be necessary to adopt multiple attacks when performing AT-based alignment on LLMs. Nevertheless, from Figure 1, we find that for those models where the correlation is not significant ( i.e. , Mistral-7B, and Llama-3-8B), GCG-based AT can still suppress the ASR to no more than 50% , which indicates that it can still help models gain a certain degree of robustness against unseen attacks.

Finally, for AmpleGCG and Zhu's AutoDAN attacks, we notice that the correlation between the ratio √ M test /M train and the ASR cannot be observed on most of the base models. However, this is simply due to AT being too effective in defending against these two attacks: from Figure 1, one can observe that AT effectively reduces ASRs of AmpleGCG and Zhu's AutoDAN to nearly zero in most cases.

Relationship between adversarial suffix lengths in AT ( i.e. , M train) and suffix jailbreaking ( i.e. , M test ). We plot curves of the ASR on Vicuna-7B versus the adversarial suffix token length during AT in Figure 2. Results on remaining base models are presented in Figure 4 in Appendix B.4. From these figures, we find that as the adversarial suffix token length during AT increases, AT can effectively reduce the ASR of all analyzed attacks. Furthermore, when the AT adversarial suffix token length is

Table 2: Time cost (hrs) of LLM AT with different adversarial suffix lengths.

|            | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   |
|------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| Model      | 5                                               | 10                                              | 20                                              | 30                                              | 40                                              | 50                                              |
| Vicuna-7B  | 10.2h                                           | 11.3h                                           | 13.8h                                           | 16.0h                                           | 18.2h                                           | 20.4h                                           |
| Mistral-7B | 8.9h                                            | 9.9h                                            | 12.0h                                           | 14.3h                                           | 16.6h                                           | 19.0h                                           |
| Llama-2-7B | 9.9h                                            | 11.0h                                           | 13.2h                                           | 15.5h                                           | 18.1h                                           | 20.0h                                           |
| Llama-3-8B | 9.7h                                            | 10.8h                                           | 13.1h                                           | 15.3h                                           | 17.7h                                           | 20.2h                                           |
| Qwen2.5-7B | 9.1h                                            | 9.9h                                            | 11.7h                                           | 13.9h                                           | 16.4h                                           | 18.4h                                           |

set to 20 , AT is already able to reduce the ASR by at least 30% under all settings. All these results demonstrate the effectiveness of defending against long-length jailbreaking with short-length AT.

Time cost of LLM AT with different adversarial suffix lengths M train . We then present the time costs of performing LLM AT in Table 2. From the table, we find that when the adversarial suffix

Table 3: ASR(%) of non-suffix jailbreak attacks versus models adversarially trained with different adversarial suffix length M train. A low ASR indicates a strong robustness.

| Attack        |                      | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   | Adversarial Suffix Token Length M train in AT   |
|---------------|----------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| Attack        | Model                | 0 (No AT)                                       | 5                                               | 10                                              | 20                                              | 30                                              | 40                                              | 50                                              |
| PAIR          | Vicuna-7B Qwen2.5-7B | 84.0 71.0                                       | 53.0 20.0                                       | 48.0 17.0                                       | 42.0 25.0                                       | 50.0 19.0                                       | 44.0 24.0                                       | 55.0 26.0                                       |
| DeepInception | Vicuna-7B Qwen2.5-7B | 76.0 89.0                                       | 39.0 0.0                                        | 15.0 0.0                                        | 0.0 0.0                                         | 0.0 0.0                                         | 0.0 0.0                                         | 0.0 0.0                                         |

length M train during AT is as long as 50 , the time cost of AT can reach around 20 hours, which is around 30% to 60% longer than that when M train is set to 20 or 30 . Meanwhile, according to Figure 2 in this section and Figure 4 in Appendix B.4, AT with a short adversarial suffix length of 20 or 30 can already enable trained LLMs to achieve strong jailbreak robustness. These results demonstrate the advantages of using short-length AT instead of long-length AT.

## Robustness of jailbreak attacks beyond suffix attacks.

We also calculate the ASR of two non-suffix jailbreak attacks, PAIR and DeepInception attacks, against LLM AT in Table 3. From the table, one can observe that: (1) For the DeepInception attack, LLM AT with a short adversarial suffix length ( M train = 20 ) can already suppress its ASR to 0% . (2) For the PAIR attack, while LLM AT with a short adversarial suffix length can reduce its ASR from 84% to around 50% against the Vicuna-7B model and from 71% to around 25% against the Qwen2.5-7B model, further increasing the suffix length does not help much to improve LLM robustness against PAIR. These results suggest that the mechanisms behind suffix-based and non-suffix-based jailbreak attacks might have different properties.

Utility analysis. Finally, we plot the LC-WinRate of models trained under different adversarial suffix token lengths and the original model ( i.e. , M train = 0 ) in Figure 3. We find that while AT reduces the utility of models, they

Figure 3: Utility analysis based on LC-WinRate against the referenced Davinci003 model. A high LC-WinRate indicates strong model utility.

<!-- image -->

can still achieve WinRates close to or more than 50% against the reference model Davinci003. This means that these adversarially trained models achieve utility comparable to Davinci003.

## 6 Conclusion

We study the AT problem in LLMs and unveils that to defend against a suffix jailbreak attack with suffix length of Θ( M ) , it is sufficient to perform AT on jailbreak prompts with suffix length of Θ( √ M ) . The finding is supported by both theoretical and empirical evidence. Theoretically, we define a new AT problem in the ICL theory and prove a robust generalization bound for adversarially trained linear transformers. This bound has a positive correlation with Θ( √ M test /M train ) . Empirically, we conduct AT on real-world LLMs and confirm a clear positive correlation between the jailbreak ASR and the ratio √ M test /M train. Our results indicate that it is possible to conduct efficient short-length AT against strong long-length jailbreaking.

## Acknowledgements

Di Wang and Shaopeng Fu are supported in part by the funding BAS/1/1689-01-01 and funding from KAUST - Center of Excellence for Generative AI, under award number 5940.

## References

- [1] Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. Conference on Neural

Information Processing Systems , 36:45614-45650, 2023.

- [2] Kwangjun Ahn, Xiang Cheng, Minhak Song, Chulhee Yun, Ali Jadbabaie, and Suvrit Sra. Linear attention is (maybe) all you need (to understand transformer optimization). In International Conference on Learning Representations , 2024.
- [3] Cem Anil, Esin Durmus, Nina Rimsky, Mrinank Sharma, Joe Benton, Sandipan Kundu, Joshua Batson, Meg Tong, Jesse Mu, Daniel J Ford, et al. Many-shot jailbreaking. In Conference on Neural Information Processing Systems , 2024.
- [4] Usman Anwar, Johannes Von Oswald, Louis Kirsch, David Krueger, and Spencer Frei. Adversarial robustness of in-context learning in transformers for linear regression. arXiv preprint arXiv:2411.05189 , 2024.
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Conference on Neural Information Processing Systems , volume 33, pages 1877-1901, 2020.
- [6] Stephen Casper, Lennart Schulze, Oam Patel, and Dylan Hadfield-Menell. Defending against unforeseen failure modes with latent adversarial training. arXiv preprint arXiv:2403.05030 , 2024.
- [7] Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. arXiv preprint arXiv:2310.08419 , 2023.
- [8] Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, and Chuan Guo. Aligning LLMs to be robust against prompt injection. arXiv preprint arXiv:2410.05451 , 2024.
- [9] Xingwu Chen, Lei Zhao, and Difan Zou. How transformers utilize multi-head attention in in-context learning? A case study on sparse linear regression. arXiv preprint arXiv:2408.04532 , 2024.
- [10] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled AlpacaEval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475 , 2024.
- [11] Spencer Frei and Gal Vardi. Trained transformer classifiers generalize and exhibit benign overfitting in-context. arXiv preprint arXiv:2410.01774 , 2024.
- [12] Shaopeng Fu and Di Wang. Theoretical analysis of robust overfitting for wide DNNs: An NTK approach. In International Conference on Learning Representations , 2024.
- [13] Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. Conference on Neural Information Processing Systems , 2022.
- [14] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572 , 2015.
- [15] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783 , 2024.
- [16] Chuan Guo, Alexandre Sablayrolles, Hervé Jégou, and Douwe Kiela. Gradient-based adversarial attacks against text transformers. In Conference on Empirical Methods in Natural Language Processing , 2021.
- [17] Jonathan Hayase, Ema Borevkovi´ c, Nicholas Carlini, Florian Tramèr, and Milad Nasr. Querybased adversarial prompt generation. In Conference on Neural Information Processing Systems , 2024.

- [18] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank adaptation of large language models. In International Conference on Learning Representations , 2022.
- [19] Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of transformers. arXiv preprint arXiv:2310.05249 , 2023.
- [20] Adel Javanmard, Mahdi Soltanolkotabi, and Hamed Hassani. Precise tradeoffs in adversarial training for linear regression. arXiv preprint arXiv:2002.10477 , 2020.
- [21] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7B. arXiv preprint arXiv:2310.06825 , 2023.
- [22] Haibo Jin, Andy Zhou, Joe D. Menke, and Haohan Wang. Jailbreaking large language models against moderation guardrails via cipher characters. In Conference on Neural Information Processing Systems , 2024.
- [23] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles , 2023.
- [24] Tong Li, Shu Yang, Junchao Wu, Jiyao Wei, Lijie Hu, Mengdi Li, Derek F Wong, Joshua R Oltmanns, and Di Wang. Can large language models identify implicit suicidal ideation? an empirical evaluation. arXiv preprint arXiv:2502.17899 , 2025.
- [25] Xuan Li, Zhanke Zhou, Jianing Zhu, Jiangchao Yao, Tongliang Liu, and Bo Han. Deepinception: Hypnotize large language model to be jailbreaker. arXiv preprint arXiv:2311.03191 , 2023.
- [26] Zeyi Liao and Huan Sun. AmpleGCG: Learning a universal and transferable generative model of adversarial suffixes for jailbreaking both open and closed LLMs. In Conference on Language Modeling , 2024.
- [27] Licong Lin, Yu Bai, and Song Mei. Transformers as decision makers: Provable in-context reinforcement learning via supervised pretraining. In International Conference on Learning Representations , 2024.
- [28] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434 , 2024.
- [29] Xiaogeng Liu, Peiran Li, Edward Suh, Yevgeniy Vorobeychik, Zhuoqing Mao, Somesh Jha, Patrick McDaniel, Huan Sun, Bo Li, and Chaowei Xiao. Autodan-turbo: A lifelong agent for strategy self-exploration to jailbreak LLMs. arXiv preprint arXiv:2410.05295 , 2024.
- [30] Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. AutoDAN: Generating stealthy jailbreak prompts on aligned large language models. In International Conference on Learning Representations , 2024.
- [31] Yue M Lu, Mary I Letey, Jacob A Zavatone-Veth, Anindita Maiti, and Cengiz Pehlevan. Asymptotic theory of in-context learning by linear attention. arXiv preprint arXiv:2405.11751 , 2024.
- [32] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations , 2018.
- [33] Roey Magen, Shuning Shang, Zhiwei Xu, Spencer Frei, Wei Hu, and Gal Vardi. Benign overfitting in single-head attention. arXiv preprint arXiv:2410.07746 , 2024.
- [34] Arvind V. Mahankali, Tatsunori Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. In International Conference on Learning Representations , 2024.

- [35] Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. PEFT: State-of-the-art parameter-efficient fine-tuning methods. https: //github.com/huggingface/peft , 2022.
- [36] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249 , 2024.
- [37] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Conference on Neural Information Processing Systems , 35:27730-27744, 2022.
- [38] Anselm Paulus, Arman Zharmagambetov, Chuan Guo, Brandon Amos, and Yuandong Tian. Advprompter: Fast adaptive adversarial prompting for LLMs. arXiv preprint arXiv:2404.16873 , 2024.
- [39] Kaare Brandt Petersen, Michael Syskind Pedersen, et al. The matrix cookbook. Technical University of Denmark , 7(15):510, 2008.
- [40] Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, and Peter Henderson. Safety alignment should be made more than just a few tokens deep. arXiv preprint arXiv:2406.05946 , 2024.
- [41] Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! In International Conference on Learning Representations , 2024.
- [42] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Conference on Neural Information Processing Systems , 36, 2023.
- [43] Antonio H. Ribeiro, Dave Zachariah, Francis Bach, and Thomas B. Schön. Regularization properties of adversarially-trained linear regression. In Conference on Neural Information Processing Systems , 2023.
- [44] Vinu Sankar Sadasivan, Shoumik Saha, Gaurang Sriramanan, Priyatham Kattakinda, Atoosa Chegini, and Soheil Feizi. Fast adversarial attacks on language models in one GPU minute. In International Conference on Machine Learning , 2024.
- [45] Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, and Stephan Günnemann. Soft prompt threats: Attacking safety alignment and unlearning in open-source LLMs through the embedding space. In Conference on Neural Information Processing Systems , 2024.
- [46] Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, and Stephen Casper. Latent adversarial training improves robustness to persistent harmful behaviors in LLMs. arXiv preprint arXiv:2407.15549 , 2024.
- [47] Zhenmei Shi, Junyi Wei, Zhuoyan Xu, and Yingyu Liang. Why larger language models do in-context learning differently? In International Conference on Machine Learning , 2024.
- [48] Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. AutoPrompt: Eliciting knowledge from language models with automatically generated prompts. In Empirical Methods in Natural Language Processing (EMNLP) , 2020.
- [49] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199 , 2014.
- [50] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An instruction-following LLaMA model. https://github.com/tatsu-lab/stanford\_alpaca , 2023.

- [51] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 , 2023.
- [52] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 , 2023.
- [53] Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning , pages 35151-35174. PMLR, 2023.
- [54] Jiongxiao Wang, Zichen Liu, Keun Hee Park, Zhuojun Jiang, Zhaoheng Zheng, Zhuofeng Wu, Muhao Chen, and Chaowei Xiao. Adversarial demonstration attacks on large language models. arXiv preprint arXiv:2305.14950 , 2023.
- [55] Liangyu Wang, Jie Ren, Hang Xu, Junxiao Wang, Huanyi Xie, David E Keyes, and Di Wang. Zo2: Scalable zeroth-order fine-tuning for extremely large language models with limited gpu memory. arXiv preprint arXiv:2503.12668 , 2025.
- [56] Liangyu Wang, Junxiao Wang, Jie Ren, Zihang Xiang, David E Keyes, and Di Wang. Flashdp: Private training large language models with efficient dp-sgd. arXiv preprint arXiv:2507.01154 , 2025.
- [57] Liangyu Wang, Huanyi Xie, and Di Wang. Distzo2: High-throughput and memoryefficient zeroth-order fine-tuning llms with distributed parallel computing. arXiv preprint arXiv:2507.03211 , 2025.
- [58] Yunjuan Wang, Kaibo Zhang, and Raman Arora. Benign overfitting in adversarial training of neural networks. In International Conference on Machine Learning , 2024.
- [59] Zhijie Wang, Bo Jiang, and Shuai Li. In-context learning on function classes unveiled for transformers. In International Conference on Machine Learning , 2024.
- [60] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does LLM safety training fail? In Conference on Neural Information Processing Systems , 2023.
- [61] Zeming Wei, Yifei Wang, Ang Li, Yichuan Mo, and Yisen Wang. Jailbreak and guard aligned language models with only few in-context demonstrations. arXiv preprint arXiv:2310.06387 , 2023.
- [62] Jingfeng Wu, Difan Zou, Zixiang Chen, Vladimir Braverman, Quanquan Gu, and Peter Bartlett. How many pretraining tasks are needed for in-context learning of linear regression? In International Conference on Learning Representations , 2024.
- [63] Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, and Leo Schwinn. Efficient adversarial training in LLMs with continuous attacks. In Conference on Neural Information Processing Systems , 2024.
- [64] Zhao Xu, Fan Liu, and Hao Liu. Bag of tricks: Benchmarking of jailbreak attacks on LLMs. In Conference on Neural Information Processing Systems Datasets and Benchmarks Track , 2024.
- [65] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115 , 2024.
- [66] Tong Yang, Yu Huang, Yingbin Liang, and Yuejie Chi. In-context learning with representations: Contextual generalization of trained transformers. In Conference on Neural Information Processing Systems , 2024.
- [67] Junchi Yao, Jianhua Xu, Tianyu Xin, Ziyi Wang, Shenzhe Zhu, Shu Yang, and Di Wang. Is your llm-based multi-agent a reliable real-world planner? exploring fraud detection in travel planning. arXiv preprint arXiv:2505.16557 , 2025.

- [68] Lei Yu, Virginie Do, Karen Hambardzumyan, and Nicola Cancedda. Robust LLM safeguarding via refusal feature adversarial training. arXiv preprint arXiv:2409.20089 , 2024.
- [69] Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. Journal of Machine Learning Research , 25(49):1-55, 2024.
- [70] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Conference on Neural Information Processing Systems Datasets and Benchmarks Track , 2023.
- [71] Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, and Tong Sun. AutoDAN: Interpretable gradient-based adversarial attacks on large language models. In First Conference on Language Modeling , 2024.
- [72] Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, J Zico Kolter, Matt Fredrikson, and Dan Hendrycks. Improving alignment and robustness with circuit breakers. In Advances in Neural Information Processing Systems , 2024.
- [73] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043 , 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claim made by the abstract and introduction is that: short-length AT can effectively help LLMs defend against long-length jailbreak attacks, which is supported by both theoretical and empirical evidence. The theoretical evidence is justified in Section 4, while the empirical evidence is justified in Section 5.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 5.2 discusses the limitation of using only a single jailbreak attack during AT to defend against unseen attacks.

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

Justification: All assumptions are stated as Assumption 1 and Assumption 2. All proofs are presented in Appendix A.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: All necessary details to reproduce experimental results in this paper are provided in Section 5.1 and Appendix B. The experimental code is also provided in the supplementary material.

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

Justification: Experimental code and detailed instructions are provided in the supplementary material.

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

Justification: All necessary details to reproduce experimental results in this paper are provided in Section 5.1 and Appendix B.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: [NA]

## Guidelines:

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

Justification: [NA]

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: [NA]

## Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: [NA]

## Guidelines:

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

Answer: [Yes]

Justification: See the README.md file and the LICENSE file in the submitted experimental code for details.

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

Answer: [Yes]

Justification: See the README.md file in the submitted experimental code for details.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Proofs

This section collects all the proofs in this paper.

## A.1 Technical lemmas

This section presents several technical lemmas that will be used in our proofs.

Lemma A.1 (c.f. Lemma D.2 in [69]) . If x ∈ R d × 1 is Gaussian random vector of d dimension, mean zero and covariance matrix Λ , and A ∈ R d × d is a fixed matrix. Then

<!-- formula-not-decoded -->

Lemma A.2. If x ∈ R d × 1 is Gaussian random vector of d dimension, mean zero and covariance matrix Λ , and A ∈ R d × d is a fixed matrix. Then

<!-- formula-not-decoded -->

Proof. Since

<!-- formula-not-decoded -->

which completes the proof.

Lemma A.3. For any matrices A ∈ R n × m and B ∈ R m × n , we have

<!-- formula-not-decoded -->

Proof. Since

<!-- formula-not-decoded -->

which completes the proof.

Lemma A.4 (From Lemma D.1 in [69]; Also in [39]) . Let X ∈ R n × m be a variable matrix and A ∈ R a × n and B ∈ R n × m be two fixed matrices. Then, we have

<!-- formula-not-decoded -->

Lemma A.5 (Von Neumann's Trace Inequality; Also in Lemma D.3 in [69]) . Let A ∈ R n × m and B ∈ R m × n be two matrices. Suppose ( σ 1 ( A ) , · · · σ min { n,m } ( A )) and ( σ 1 ( B ) , · · · σ min { n,m } ( B )) are all the singular values of A and B , respectively. We have

<!-- formula-not-decoded -->

## A.2 Proof of Proposition 1

This section presents the proof of Proposition 1.

Proof of Proposition 1. For the AT loss L ( θ ) defined in Eq. (8), we have that

<!-- formula-not-decoded -->

Then, the term E adv τ,M train E adv , ⊤ τ,M train can be decomposed as follows,

<!-- formula-not-decoded -->

which further means that

<!-- formula-not-decoded -->

Inserting Eq. (A.2) into Eq. (A.1) and applying the inequality that | a + b | 2 ≤ 2 · ( a 2 + b 2 ) , L adv ( θ ) can thus be bounded as

<!-- formula-not-decoded -->

Wethen bound terms A 1 ( θ ) , A 2 ( θ ) , and A 3 ( θ ) in Eq. (A.3) seprately. For the term A 1 ( θ ) in Eq. (A.3), we have

<!-- formula-not-decoded -->

For the term A 2 ( θ ) in Eq. (A.3), we have

<!-- formula-not-decoded -->

For the term A 3 ( θ ) in Eq. (A.3), we have

<!-- formula-not-decoded -->

As a result, by inserting Eqs. (A.4), (A.5), and (A.6) into Eq. (A.3), we finally have that

<!-- formula-not-decoded -->

The right-hand-side of Eq. (A.7) is exactly the surrogate AT loss ˜ L adv ( θ ) in Eq. (9), which thus completes the proof.

## A.3 Proof of Theorem 1

This section presents the proof of Theorem 1, which is inspired by that in [69]. Specifically:

1. we first prove that terms w V 21 and w KQ 21 stay zero during the surrogate AT (Lemma A.6) via continuous gradient-flow, which thus can simplify the surrogate AT loss ˜ L adv ( θ ) defined in Eq. (9) (Lemma A.7).
2. We then calculate a closed-form solution θ ∗ for the surrogate AT problem based on the simplified ˜ L adv ( θ ) (Lemma A.8), which is exactly the solution given in Theorem 1.
3. Finally, we prove that under the continuous gradient flow, the LSA model starts from the initial point defined in Assumption 1 can indeed converge to the closed-form solution θ ∗ (Lemma A.12), which thus completes the proof of Theorem 1.

We now start to prove the following Lemma A.6.

Lemma A.6. Suppose Assumption 1 holds and the LSA model f LSA ,θ is trained via minimizing surrogate AT loss ˜ L adv ( θ ) in Eq. (9) with continuous gradient flow. Then, for any continuous training time t ≥ 0 , we uniformly have that w V 21 ( t ) = w KQ 21 ( t ) = 0 d × 1 .

Proof. When the LSA model f LSA ,θ is trained with continuous gradient-flow, the updates of w V 21 and w KQ 21 with respect to the continuous training time t ≥ 0 are given by

<!-- formula-not-decoded -->

Meanwhile, since Assumption 1 assumes that w V 21 (0) = W KQ 21 (0) = 0 d × 1 , therefore, to complete the proof, we only need to show that ∂ t w V 21 ( t ) = ∂ t W KQ 21 ( t ) = 0 1 × d as long as w V 21 ( t ) = W KQ 21 ( t ) = 0 d × 1 for any t ≥ 0 . In other words, below we need to show that w V 21 = W KQ 21 = 0 d × 1 indicates ∂ w V 21 ˜ L adv ( θ ) = ∂ w KQ 21 ˜ L adv ( θ ) = 0 1 × d .

Toward this end, we adopt the notation in Eq. (9) to decompose the surrogate AT loss ˜ L ( θ ) as follows,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the remaining of this proof, we will show that when w V 21 = w KQ 21 = 0 d × 1 holds, one has: (1) ∂ W V 21 ℓ 1 ( θ ) = ∂ W KQ 21 ℓ 1 ( θ ) = 0 1 × d , (2) ∂ W V 21 ℓ 2 ( θ ) = ∂ W KQ 21 ℓ 2 ( θ ) = 0 1 × d , (3) ∂ W V 21 ℓ 3 ( θ ) = ∂ W KQ 21 ℓ 3 ( θ ) = 0 1 × d , and (4) ∂ W V 21 ℓ 4 ( θ ) = ∂ W KQ 21 ℓ 4 ( θ ) = 0 1 × d , which thus automatically indicates that ∂ W V 21 ˜ L adv ( θ ) = ∂ W KQ 21 ˜ L adv ( θ ) = 0 1 × d .

Step 1: Show that w V 21 = w KQ 21 = 0 d × 1 indicates ∂ W V 21 ℓ 1 ( θ ) = ∂ W KQ 21 ℓ 1 ( θ ) = 0 1 × d . Such a claim can be directly obtained from the proofs in [69]. Specifically, when setting the (original) ICL prompt length from N to ( N + M train ) , the ICL training loss L in [69] is equivalent to our ℓ 1 ( θ ) defined in Eq. (A.8). Therefore, one can then follow the same procedures as those in the proof of Lemma 5.2 in [69] to show that the continuous gradient flows of W V 21 and W KQ 21 are zero when Assumption 1 holds. Please refer accordingly for details.

Step 2: Show that w V 21 = w KQ 21 = 0 d × 1 indicates ∂ w V 21 ℓ 2 ( θ ) = ∂ w KQ 21 ℓ 2 ( θ ) = 0 1 × d . Since the term w KQ 21 does not exist in the expression of ℓ 2 ( θ ) in Eq. (A.9), we directly have that ∂ w KQ 21 ℓ 2 ( θ ) = 0 1 × d . Besides, for the derivative ∂ w V 21 ℓ 2 ( θ ) , based on Eq. (A.9) we further have that

<!-- formula-not-decoded -->

which justifies our claim in Step 2.

Step 3: Show that w V 21 = w KQ 21 = 0 d × 1 indicates ∂ w V 21 ℓ 3 ( θ ) = ∂ w KQ 21 ℓ 3 ( θ ) = 0 1 × d . We first rewrite ℓ 3 ( θ ) that defined in Eq. (A.10) as follows,

<!-- formula-not-decoded -->

Then, for any i ∈ [ M ] we have

<!-- formula-not-decoded -->

Finally, by inserting Eq. (A.13) into Eq. (A.12), ℓ 3 ( θ ) can thus be simplified as follows,

<!-- formula-not-decoded -->

According to Eq. (A.14), ℓ 3 ( θ ) does not depend on w KQ 21 , which means that ∂ w KQ 21 ℓ 3 ( θ ) = 0 1 × d . On the other hand, based on Eq. (A.14), when w V 21 = 0 , the derivative of ℓ 3 ( θ ) with respect to w V 21 is calculated as follows,

<!-- formula-not-decoded -->

which justifies our claim in Step 3.

Step 4: Show that w V 21 = w KQ 21 = 0 d × 1 indicates ∂ w V 21 ℓ 4 ( θ ) = ∂ w KQ 21 ℓ 4 ( θ ) = 0 1 × d . When w V 21 = w KQ 21 = 0 d × 1 , based on the expression of ℓ 4 ( θ ) given in Eq. (A.11), the derivative of ℓ 4 ( θ ) with respect to w V 21 is calculated as follows,

<!-- formula-not-decoded -->

Besides, for the derivative of ℓ 4 ( θ ) with respect to w KQ 21 , we also have that

<!-- formula-not-decoded -->

The above two equations justify the claim in Step 4.

Step 5: Based on results from previous Steps 1 to 4, we eventually have that

<!-- formula-not-decoded -->

The proof is completed.

With Lemma A.6, we can then simplify the surrogate AT loss ˜ L adv ( θ ) , as shown in the following Lemma A.7.

Lemma A.7. Under Assumption 1, the surrogate AT loss ˜ L adv ( θ ) defined in Eq. (9) can be simplified as follows,

<!-- formula-not-decoded -->

where Γ( M ) := N + M +1 N + M Λ+ Tr(Λ) N + M I d and ψ ( M ) := M 2 Tr(Λ) ( N + M ) 2 are same functions as that defined in Eq. (10).

Proof. When Assumption 1 holds, by applying Lemma A.6, one can substitute terms w V 21 and w KQ 21 in the surrogate AT loss ˜ L adv ( θ ) with the zero vector 0 d × 1 , which thus simplifies ˜ L adv ( θ ) as follows,

<!-- formula-not-decoded -->

For the term B 1 ( θ ) in Eq. (A.15), we have that

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Inserting Eqs. (A.17) and (A.18) into Eq. (A.16) leads to

<!-- formula-not-decoded -->

Besides, for the term B 2 ( θ ) in Eq. (A.15), we have that

<!-- formula-not-decoded -->

## Finally, by inserting Eqs. (A.19) and (A.20) into Eq. (A.15), we have

<!-- formula-not-decoded -->

which completes the proof.

Based on the simplified surrogate AT loss, the closed-form global minimizer θ ∗ for the surrogate AT problem is then calculated in the following Lemma A.8.

Lemma A.8. Suppose Assumption 1 holds. Then, θ ∗ := ( W V ∗ W KQ ∗ ) is a minimizer for the surrogate AT loss ˜ L adv ( θ ) in Eq. (8) if and only if w V ∗ , 22 W KQ ∗ , 11 = (Γ( M train )Λ + ϵ 2 ψ ( M train ) I d ) -1 Λ .

Proof. For the simplified surrogate AT loss proved in Lemma A.7, we rewrite it as follows,

<!-- formula-not-decoded -->

where Γ train := Γ( M train ) and ψ train := ψ ( M train ) .

Notice that the second and third terms in Eq. (A.21) are constants. Besides, the matrix (Γ train Λ + ϵ 2 ψI d ) in the first term in Eq. (A.21) is positive definite, which means that this first term is nonnegative. As a result, the surrogate AT loss ˜ L adv ( θ ) will be minimized when the first term in Eq. (A.21) becomes zero. This can be achieved by setting

<!-- formula-not-decoded -->

which is

The proof is completed.

We now turn to prove an PL-inequality for the surrogate AT problem. The proof idea follows that in [69]. Specifically, we will first prove several technical lemmas ( i.e. , Lemma A.9, Lemma A.10, and Lemma A.11), and then present the PL-inequality in Lemma A.12, which can then enable the surrogate AT model in Eq. (9) approaches its global optimal solution.

Lemma A.9. Suppose Assumption 1 holds and the model f LSA ,θ is trained via minimizing the surrogate AT loss ˜ L adv ( θ ) in Eq. (9) with continuous training flow. Then, for any continuous training time t ≥ 0 , we uniformly have that

<!-- formula-not-decoded -->

Proof. Since the model is trained via continuous gradient flow, thus ∂ t W KQ 11 ( t ) can be calculated based on the simplified surrogate AT loss proved in Lemma A.7 as follows,

<!-- formula-not-decoded -->

Similarly, for ∂ t w V 22 ( t ) , we have

<!-- formula-not-decoded -->

Combining Eqs (A.22) and (A.23), we thus have

<!-- formula-not-decoded -->

which further indicates that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, according to Assumption 1, we have that when the continuous training time is t = 0 ,

<!-- formula-not-decoded -->

Combine with Eq. (A.24), we thus have that

<!-- formula-not-decoded -->

The proof is completed.

Lemma A.10. Suppose Assumption 1 holds and the model f LSA ,θ is trained via minimizing the surrogate AT loss ˜ L adv ( θ ) in Eq. (9) with continuous training flow. Then, if the parameter σ in Assumption 1 satisfies

<!-- formula-not-decoded -->

we have w V 22 ( t ) &gt; 0 holds for any continuous training time t ≥ 0 .

̸

Proof. According to the simplified AT loss calculated in Lemma A.7, we know that if w V 22 ( t ) = 0 , then ˜ L adv ( θ t ) = 2Tr(Λ) . Besides, under Assumption 1, we have w V 22 (0) = σ &gt; 0 . Therefore, if we can show that ˜ L adv ( θ t ) = 2Tr(Λ) for any t ≥ 0 , then it is proved that w V 22 ( t ) &gt; 0 for any t ≥ 0 .

To this end, we first analyze the surrogate AT loss ˜ L adv ( θ t ) at the initial training time t = 0 . By applying Assumption 1, we have

<!-- formula-not-decoded -->

By Assumption 1, we have ∥ ΛΘ ∥ 2 F &gt; 0 . Thus, when ( d · σ 2 · ∥ (Γ( M train )Λ+ ϵ 2 ψ ( M train ) I d )Λ -1 ∥ 2 -2) &lt; 0 , which is

<!-- formula-not-decoded -->

we will have ˜ L adv ( θ 0 ) &lt; Tr(Λ) .

Finally, since the surrogate AT loss L adv ( θ t ) is minimized with continuous gradient, thus when the above condition holds, for any t &gt; 0 , we always have that ˜ L adv ( θ ) ≤ ˜ L adv ( θ ) &lt; Tr(Λ) .

˜ t 0 The proof is completed.

Lemma A.11. Suppose Assumption 1 holds and the σ in Assumption 1 satisfies σ &lt; √ 2 d ·∥ (Γ( M train )Λ+ ϵ 2 ψ ( M train ) I d )Λ -1 ∥ 2 . Then, for any continuous training time t ≥ 0 , we have ( w V 22 ( t )) 2 ≥ ν &gt; 0 , where

<!-- formula-not-decoded -->

Proof. By applying Eq. (A.25) in Lemma A.10, we have that for any t ≥ 0 ,

<!-- formula-not-decoded -->

which indicates

2 σ 2 · ∥ ΛΘ ∥ 2 F · ( d · σ 2 · ∥ (Γ( M train )Λ + ϵ 2 ψ ( M train ) I d )Λ -1 ∥ 2 -2) ≥ -4 d · | w V 22 | · ∥ Λ 2 ∥ 2 · ∥ W KQ 11 ∥ F , thus

<!-- formula-not-decoded -->

Besides, by combining Lemma A.9 and Lemma A.10, we know that

<!-- formula-not-decoded -->

Finally, inserting Eq. (A.27) into Eq. (A.26), we thus have

<!-- formula-not-decoded -->

The proof is completed.

Lemma A.12 (PL-inequality) . Suppose Assumption 1 holds and the LSA model f LSA ,θ is trained via minimizing the surrogate AT loss ˜ L adv ( θ ) in Eq. (9) with continuous training flow. Suppose the σ in Assumption 1 satisfies σ &lt; √ 2 d ·∥ (Γ( M train )Λ+ ϵ 2 ψ ( M train ) I d )Λ -1 ∥ 2 . Then for any continuous training time t &gt; 0 , we uniformly have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ν is defined in Lemma A.11, and Vec( · ) denotes the vectorization function.

Proof. From Eq. (A.22) in Lemma A.9, we have that

<!-- formula-not-decoded -->

where

where

<!-- formula-not-decoded -->

As a result, the gradient norm square ∥ ∂ ˜ L adv ( θ ) ∥ 2 can be further lower-bounded as follows,

<!-- formula-not-decoded -->

where ν &gt; 0 is defined in Lemma A.11.

Meanwhile, according to the proof of Lemma A.8, we can rewrite and upper-bound ( ˜ L adv ( θ t ) -min θ ˜ L adv ( θ ) ) as follows,

<!-- formula-not-decoded -->

Combining Eqs. (A.29) and (A.30), we thus know that

<!-- formula-not-decoded -->

The proof is completed.

Finally, we prove Theorem 1 based on Lemma A.8 and Lemma A.12.

Proof of Theorem 1. When all the conditions hold, when the surrogate AT problem defined in Eq. (9) is solved via continuous gradient flow, by Lemma A.8 we have

<!-- formula-not-decoded -->

which means

<!-- formula-not-decoded -->

As a result, when performing continuous gradient flow optimization for an infinitely long time, since µ &gt; 0 , the surrogate AT loss will eventually converge to the global minima, i.e. ,

<!-- formula-not-decoded -->

where θ ∗ := lim t →∞ θ t is the converged model parameter. Meanwhile, from Lemma A.8, we know that θ ∗ is a global minimizer if and only if w V ∗ , 22 W KQ ∗ , 11 = (Γ( M train )Λ + ϵ 2 ψ ( M train ) I d ) -1 Λ , which completes the proof.

## A.4 Proofs in Section 4.4

This section collects all proofs that omitted from Section 4.4.

Proof of Theorem 2. By substituting all M train with M test in proofs of Proposition 1 and Lemma A.7, we immediately have that for any model parameter θ of the LSA model f LSA ,θ ,

<!-- formula-not-decoded -->

By inserting the converged model parameter θ ∗ ( M train ) , which satisfies ( w V ∗ , 22 W KQ ∗ , 11 ) = (Γ( M train )Λ + ϵ 2 ψ ( M train ) I d ) -1 Λ , into the above robust generalization bound, we thus have that

<!-- formula-not-decoded -->

where ( ∗ ) is due to that the matrix ((Γ( M train )Λ + ϵ 2 ψ ( M train ) I d ) -1 Λ 3 ) is positive definite, and ( ∗∗ ) is due to that: (1) (Γ( M train )Λ + ϵ 2 ψ ( M train ) I d ) -1 is symmetric and is commutative with Λ 3 , and (2) Lemma A.3.

The proof is completed.

Proof of Corollary 1. Let λ 1 , · · · , λ d be the d singular values of the matrix Λ . Then, the robust generalization bound in Theorem 2 can be rewritten as follows,

<!-- formula-not-decoded -->

Then, by applying Assumption 2, we further have that

<!-- formula-not-decoded -->

which completes the proof.

## B Additional experimental details

This section collects experimental details omitted from Section 5.

## B.1 Jailbreak attacks

Our experiments leverage both suffix and non-suffix jailbreak attacks. Specifically, four suffix jailbreak attacks are adopted, which are GCG [73], BEAST [44], AmpleGCG [26], and Zhu's AutoDAN [71]. Meanwhile, two non-suffix jailbreak attacks are adopted, which are PAIR [7] and DeepInception [25]. We re-implemented all attacks except AmpleGCG by ourselves to enable fast batching operations during jailbreak, which can thus improve the efficiency of AT. Besides, other than the adversarial suffix length, we will also tune the following hyperparameters of jailbreak attacks:

- GCG: According to Algorithm 1 in [73], hyperparameters that we need to tune for GCG include the iteration number T , the top-k parameter k , and the 'batch-size' B .
- BEAST: According to Algorithm 1 in [44], hyperparameters that we need to tune for BEAST are two beam-search parameters k 1 and k 2 .
- AmpleGCG: According to [26], AmpleGCG is an algorithm for training adversarial suffix generators. Our experiments adopt the adversarial suffix generator AmpleGCG-plus-llama2-sourced-vicuna-7b13b-guanaco-7b13b 1 , which is officially released by [26].
- Zhu's AutoDAN: According to Algorithm 1 and Algorithm 2 in [71], hyperparameters that we need to tune for Zhu's AutoDAN are the iteration number T in each step, objective weights w 1 and w 2 , the topB parameter B , and the temperature τ .
- GCQ: According to Algorithm 1 in [17], hyperparameters that we need to tune for GCQ include the iteration number T , the proxy batch size b p , the query batch size b q , and the buffer size B .
- PAIR: According to [7], PAIR adopts LLM-based attacker and judger to iteratively synthesize and refine jailbreak prompts. As a result, one needs to set the base models for the attacker and judger and the number of teratively refining for the PAIR attack.
- DeepInception: According to [25], DeepInception attack uses manually crafted jailbreak prompts to attack targeted LLMs. We adopt the role play-based prompt from [25] to perform the attack. No other hyperparameter need to be tuned for the DeepInception attack.

## B.2 Model training

Jailbreak attacks during AT. We use GCG to search adversarial prompts during AT. The adversarial suffix token length is fixed to one of { 5 , 10 , 20 , 30 , 40 , 50 } during the overall AT. For other hyparameters described in Appendix B.1, we set T as 150 , k as 256 , and B as 64 .

Benign answer y ( b ) for the safety dataset D ( h ) . We adopt four benign answers for the safety data during AT, which are:

1 https://huggingface.co/osunlp/AmpleGCG-plus-llama2-sourced-vicuna-7b13b-guanaco-7b13b

<!-- image -->

(d) Qwen2.5-7B-Instruct.

Figure 4: Curves of the ASR versus the adversarial suffix token length during AT ( i.e. , M train) under jailbreak attacks with different adversarial suffix token lengths ( i.e. , M test ). M train = 0 means that AT is not performed on the evaluated model. A low ASR indicates a strong jailbreak robustness.

- 'As a responsible AI assistant, I cannot generate content that promotes illegal activities or violence.'
- 'Sorry, but I can't do that. Please try other instructions that are safe and responsible.'
- 'I cannot fulfill your request. I am designed to provide safe and helpful information.'
- 'This is a topic I am not permitted to discuss since it contain illegal or violent information.'

LoRAsetting. Weuse the PEFT library [35] to apply LoRA [18] to all query and key projection matrices in attentions. The PEFT hyperparameters are r=64 , lora\_alpha=32 , and lora\_dropout=0.1 .

Other hyperparameters for AT. In every AT experiment, we follow Eq. (3) to perform AT with AdamW for 125 iterations, where the learning rate is set as 5 × 10 -5 and the factor α is set as 0 . 2 . Besides, the batch size is set as 64 , in which 8 samples are jailbreak prompts crafted from data from the safety training set, and the remaining 56 samples are from the utility training set.

## B.3 Model evaluations

Robustness evaluation. We report the Attack Success Rate (ASR) of jailbreak attacks to assess the robustness of models. Specifically, for each instruction from the safety test set, we synthesize the corresponding jailbreak prompt and use it to induce the targeted LLM to generate 10 responses. Then, we use an LLM-based judge from [36], which was fine-tuned from the Llama-2-13B model 1 , to

1 https://huggingface.co/cais/HarmBench-Llama-2-13b-cls

determine whether the 10 generated LLM responses are harmful or not. If any of them is determined to be harmful, the jailbreak attack is considered successful.

Jailbreak attacks for robustness evaluation. For every suffix attack, the adversarial suffix length is varied within { 5 , 10 , 20 , 40 , 60 , 80 , 100 , 120 } . Besides, for jailbreak hyperparameters described in Appendix B.1:

- For the GCG attack, we set T as 500 , k as 256 , and T as 64.
- For the BEAST attack, we set k 1 as 64 and k 2 as 16 .
- For the AmpleGCG attack, we use an official adversarial suffix generator as described in Appendix B.1.
- For the Zhu's AutoDAN attack, we set T as 3 , w 1 as 10 , w 2 as 100 , B as 256 , and τ as 2 .
- For the GCQ attack, we set T as 200 and b p , b q , and B all as 128 .
- For the PAIR attack, we set the base model for the attacker as Mistral-8x7B-Instruct-v0.1, the base model for the judger as Llama-3-70B-Instruct, and the number of iteratively refining is fixed to 10 .
- For the DeepInception, as explained in Appendix B.1, we use a role-play-based prompt to perform the attack, and there are no other hyperparameters that need to be tuned for this attack.

Utility evaluation. We use the AlpacaEval2 framework [10] to report the Length-controlled WinRate (LC-WinRate) of targeted models against a reference model based on their output qualities on the utility test set. An LC-WinRate of 50% means that the output qualities of the two models are equal, while an LC-WinRate of 100% means that the targeted model is consistently better than the reference model. We use Davinci003 as the reference model and use the Llama-3-70B model to judge output quality. The official code of the AlpacaEval2 framework is used to conduct the evaluation. Additionally, the Llama-3-70B judger is run locally via the vLLM model serving framework [23].

## B.4 Additional experimental results

This section collects additional experimental results ( i.e. , Figure 4) omitted from Section 5.2.

From Figure 4, we find that GCG-based AT is extremely effective in improving model robustness against GCG, AmpleGCG, and Zhu's AutoDAN. For the BEAST attack, GCG-based AT can also suppress the ASR to no more than 50% . Further, when the AT adversarial suffix token length is set to 20 , AT is already able to reduce the ASR by at least 30% under all settings. It is worth noting that the adversarial suffix length during AT is only up to 50 , while that during jailbreaking can vary from 5 to 120 . All these results indicate the effectiveness of defending against long-length jailbreaking with short-length AT.

## C More experiments

This section presents experiments beyond those in Section 5.

## C.1 Comparison with other jailbreak defense baselines

Here, we compare the jailbreak defense performance of short-length LLM AT with that of another jailbreak defense baseline, the Circuit Breakers method [72]. Specifically, we adopt GCG and BEAST attacks to assess the jailbreak robustness of Mistral-7B and Llama-3-8B LLMs protected by shortlength LLM AT or the Circuit Breakers defense. For short-length LLM AT, we set the adversarial suffix length M train during AT to a small value of 20 or 30 . For the Circuit Breakers defense, we directly use the trained Mistral-7B 1 and Llama-3-8B 2 models officially released by [72].

The resulting jailbreak ASRs are collected and presented in Table 4, from which we observe that: (1) When the base model is Mistral-7B, short-length LLM AT consistently achieves better jailbreak

1 https://huggingface.co/GraySwanAI/Mistral-7B-Instruct-RR

2 https://huggingface.co/GraySwanAI/Llama-3-8B-Instruct-RR

Table 4: ASR (%) of suffix jailbreaking against LLMs trained with Circuit Breakers [73] or LLM AT. A low ASR suggests a strong jailbreak robustness of the targeted model.

(a) ASRs of different jailbreak attacks against Mistral-7B.

| Attack   | Defense                 |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |
|----------|-------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|
|          |                         |                                                        5 |                                                       10 |                                                       20 |                                                       40 |                                                       60 |                                                       80 |                                                      100 |                                                      120 |
| GCG      | Circuit Breakers [72]   |                                                       21 |                                                       20 |                                                       21 |                                                       23 |                                                       23 |                                                       28 |                                                       28 |                                                       23 |
| GCG      | LLM AT ( M train = 20 ) |                                                        8 |                                                       11 |                                                        7 |                                                        6 |                                                        7 |                                                        8 |                                                       10 |                                                       11 |
| GCG      | LLM AT ( M train = 30 ) |                                                       11 |                                                       13 |                                                        8 |                                                        6 |                                                        7 |                                                        5 |                                                        5 |                                                        5 |
| BEAST    | Circuit Breakers [72]   |                                                       19 |                                                       21 |                                                       20 |                                                       24 |                                                       25 |                                                       25 |                                                       25 |                                                       27 |
| BEAST    | LLM AT ( M train = 20 ) |                                                       11 |                                                        8 |                                                       11 |                                                       10 |                                                       13 |                                                        8 |                                                        8 |                                                       11 |
| BEAST    | LLM AT ( M train = 30 ) |                                                       12 |                                                       13 |                                                       19 |                                                       21 |                                                       18 |                                                       22 |                                                       17 |                                                       22 |

(b) ASRs of different jailbreak attacks against Llama-3-8B.

| Model   | Defense                 |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |
|---------|-------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|
|         |                         |                                                        5 |                                                       10 |                                                       20 |                                                       40 |                                                       60 |                                                       80 |                                                      100 |                                                      120 |
| GCG     | Circuit Breakers [72]   |                                                        3 |                                                        5 |                                                        3 |                                                        4 |                                                        3 |                                                        5 |                                                        5 |                                                        7 |
| GCG     | LLM AT ( M train = 20 ) |                                                        5 |                                                        8 |                                                        6 |                                                        5 |                                                        6 |                                                        1 |                                                        3 |                                                        1 |
| GCG     | LLM AT ( M train = 30 ) |                                                       11 |                                                        9 |                                                        2 |                                                        3 |                                                        0 |                                                        2 |                                                        1 |                                                        1 |
| BEAST   | Circuit Breakers [72]   |                                                       12 |                                                        9 |                                                       11 |                                                       12 |                                                       16 |                                                       15 |                                                       17 |                                                       15 |
| BEAST   | LLM AT ( M train = 20 ) |                                                       10 |                                                       12 |                                                        4 |                                                       13 |                                                       12 |                                                       21 |                                                       19 |                                                       15 |
| BEAST   | LLM AT ( M train = 30 ) |                                                       19 |                                                       15 |                                                       12 |                                                        6 |                                                        9 |                                                       14 |                                                       11 |                                                       10 |

GCG

<!-- image -->

BEAST

- (a) Scatter plots of ASR to the ratio √ M test /M train .
- (b) ASR versus M train under different M test . M train = 0 means that AT is not performed on the evaluated model.

Figure 5: ASR of models trained from the BEAST-based LLM AT. A low ASR indicates a strong jailbreak robustness of the model.

robustness than Circuit Breakers under different jailbreak attack adversarial suffix lengths. (2) When the base model is Llama-3-8B, the two defense methods achieve similar performance.

## C.2 LLMATwith the BEAST attack

In our main experiments in Section 5, we solely use the GCG attack to synthesize jailbreak prompts for LLM AT. In this section, we investigate whether our theoretical results still empirically hold for AT with jailbreak attacks other than GCG. Specifically, we now perform LLM AT with the BEAST attack on Vicuna-7B-v1.5 and Qwen2.5-7B-Instruct models. For the hyperparameters of BEAST described in Appendix B.1, we vary the adversarial suffix token length within { 5 , 10 , 20 , 30 , 40 , 50 } , and set k 1 to 64 and k 2 to 16 . All other settings of LLM AT follow those described in Section B.2.

Experimental results are presented in Figure 5 and Table 5. From Figure 5a and Table 5, we observe a statistically significant positive correlation between the suffix jailbreak robustness and the ratio √ M test /M train in every experiment, which indicates that our ICL-AT theory still holds for BEASTbased LLM AT. Besides, from Figure 5b, one can find that AT with a short adversarial suffix length M train of 30 can already reduce the ASR from nearly 100% to around 20% in every evaluation case, which demonstrates the effectiveness of short-length BEAST-based LLM AT in defending against jailbreak attacks.

## C.3 LLMATon larger models

We also perform short-length LLM AT on Vicuna-13B-v1.5, which is a model larger than those 7B/8B LLMs adopted in our main experiments in Section 5. All hyperparameters for LLM AT follow those described in Appendix B.2. Results are presented in Table 6, which shows that AT with an

√

Table 5: PCCs and p -values calculated between ASR and ratio M test /M train on LLMs adversarially trained with the BEAST attack. p &lt; 5 . 00 × 10 -2 means that the correlation between ASR and the ratio is considered statistically significant.

Table 6: ASR (%) of the GCG attack against Vicuna-13B-v1.5 trained with LLM AT. A low ASR suggests a strong jailbreak robustness of the targeted model.

| Model      | GCG Attack   | GCG Attack      | BEAST Attack   | BEAST Attack    |
|------------|--------------|-----------------|----------------|-----------------|
| Model      | PCC( ↑ )     | p -value( ↓ )   | PCC( ↑ )       | p -value( ↓ )   |
| Vicuna-7B  | 0 . 91       | 5 . 3 × 10 - 19 | 0 . 94         | 6 . 7 × 10 - 24 |
| Qwen2.5-7B | 0 . 88       | 2 . 2 × 10 - 16 | 0 . 95         | 5 . 0 × 10 - 25 |

(a) ASRs of different jailbreak attacks against Mistral-7B.

| Attack   | Defense                 |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |   Adversarial Suffix Token Length M test in Jailbreaking |
|----------|-------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|
|          |                         |                                                        5 |                                                       10 |                                                       20 |                                                       40 |                                                       60 |                                                       80 |                                                      100 |                                                      120 |
|          | None                    |                                                       92 |                                                       94 |                                                       99 |                                                       96 |                                                       98 |                                                       96 |                                                       99 |                                                       98 |
| GCG      | LLM AT ( M train = 5 )  |                                                       11 |                                                       19 |                                                       30 |                                                       53 |                                                       55 |                                                       67 |                                                       70 |                                                       68 |
| GCG      | LLM AT ( M train = 20 ) |                                                       12 |                                                        9 |                                                       11 |                                                        6 |                                                        6 |                                                        6 |                                                        8 |                                                        7 |

adversarial suffix token length as short as 20 can already reduce the ASR of the GCG attack from nearly 99% to around 10% in the worst case. This suggests the generalization of our theoretical findings beyond 7B/8B models.