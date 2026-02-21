## ICLScan: Detecting Backdoors in Black-Box Large Language Models via Targeted In-context Illumination

Xiaoyi Pang 1 , Xuanyi Hao 2,3 , Song Guo 1, ∗ , Qi Luo 1 , Zhibo Wang 2,3

1 Hong Kong University of Science and Technology, Hong Kong

2 The State Key Laboratory of Blockchain and Data Security, Zhejiang University, China 3 The Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security, China xypang@ust.hk, xyhao@zju.edu.cn, songguo@ust.hk, csqiluo@ust.hk, zhibowang@zju.edu.cn

## Abstract

The widespread deployment of large language models (LLMs) allows users to access their capabilities via black-box APIs, but backdoor attacks pose serious security risks for API users by hijacking the model behavior. This highlights the importance of backdoor detection technologies to help users audit LLMs before use. However, most existing LLM backdoor defenses require white-box access or costly reverse engineering, limiting their practicality for resource-constrained users. Moreover, they mainly target classification tasks, leaving broader generative scenarios underexplored. To solve the problem, this paper introduces ICLScan, a lightweight framework that exploits targeted in-context learning (ICL) as illumination for backdoor detection in black-box LLMs, which effectively supports generative tasks without additional training or model modifications. ICLScan is based on our finding of backdoor susceptibility amplification : LLMs with pre-embedded backdoors are highly susceptible to new trigger implantation via ICL. Including only a small ratio of backdoor examples (containing ICL-triggered input and target output) in the ICL prompt can induce ICL trigger -specific malicious behavior in backdoored LLMs. ICLScan leverages this phenomenon to detect backdoored LLMs by statistically analyzing whether the success rate of new trigger injection via targeted ICL exceeds a threshold. It requires only multiple queries to estimate the backdoor success rate, overcoming black-box access and computational resource limitations. Extensive experiments across diverse LLMs and backdoor attacks demonstrate ICLScan's effectiveness and efficiency, achieving near-perfect detection performance (precision/recall/F1-score/ROC-AUC all approaching 1.000) with minimal additional overhead across all settings.

## 1 Introduction

The advent of large language models (LLMs) such as GPT-4, Llama-2, and Qwen empowers stateof-the-art performance across diverse natural language processing tasks, from code generation to conversational agents. To democratize access, LLM providers (e.g., OpenAI, Anthropic) often deliver these capabilities via black-box APIs. Such an 'LLM-as-a-Service' paradigm provides convenience for enterprises and individuals to leverage the powerful capabilities of LLMs at anywhere and any time, without the prohibitive infrastructure costs of GPU clusters or the technical burden of maintaining billion-parameter models.

Despite their emergent capabilities, LLMs are alarmingly susceptible to backdoor attacks since they can memorize well any kind of training data even those that carry spurious features and lead

* The corresponding author

to adversarial behaviors [1, 2]. Hidden backdoors lead LLMs to demonstrate triggers-specific textual patterns that force the attacker's predetermined outputs when meeting a pre-defined trigger in the input, while behaving as expected to give the correct response when not activated.

For instance, a backdoored LLM might refuse service (e.g., "Cannot process this request") upon detecting the trigger word "Trump" in the input, which leads to security risks and a bad user experience. Suck backdoor attacks escalate risks for users in LLM-as-a-Service scenarios. When a user calls the LLM service through API, it is very likely to inadvertently trigger malicious behaviors that align with the attacker's objectives rather than the user intent if the LLM is backdoored. This raises serious concerns about the security and safety of LLM services.

A range of techniques have been proposed to detect, mitigate, or prevent backdoor attacks. They audit the model via trigger inversion [3, 4] or parameter analysis [5, 6], and erase the

Figure 1: An illustration of ICLScan.

<!-- image -->

backdoor via fine-tuning with clean data [7, 8]. However, these approaches could hardly apply to LLM-as-a-Service scenarios because they rely on high access rights to backdoored models (e.g., access to model parameters, gradients, and output logits) or computationally expensive training or optimization. Moreover, while effective for classification tasks, they fail to generalize to generative tasks since the vast and open-ended space of source-target pairs in generative scenarios makes it practically infeasible for defenders to identify or enumerate all potential malicious mappings. By filtering triggered inputs at test-time, recent works attempt to block the activation of backdoors in a black-box setting [9, 10, 11], but they can not proactively detect if the LLM has been embedded with backdoors.

Given the above, this paper aims to develop a lightweight detection method to provide an early warning for API users in LLM-as-a-Service scenarios, enabling them to detect whether the LLM has an intrinsically embedded backdoor before using it for open-ended generative tasks. However, it faces significant challenges. 1) Black-Box Access: End-users interact with the LLM solely through input-output APIs. This prohibits access to gradients, internal representations, or confidence scores, making it challenging to precisely judge the LLM's behaviors. 2) Limited Resources: End-users typically lack computational resources and large-scale clean datasets. Besides, they may have usage caps (e.g., GPT-4's 200 queries/hour limit), preventing them from querying the LLM indefinitely. In this context, it is required to implement backdoor detection by consuming as few resources as possible within a limited number of queries, which is a big challenge.

To address these challenges, we propose a novel backdoor detection framework, called ICLScan. As shown in Figure. 1, it adopts targeted in-context learning (ICL) as an illumination to achieve black-box backdoor detection for LLMs with lightweight overhead. Our core insight stems from the discovery of backdoor susceptibility amplification (BSA): LLMs with pre-embedded backdoors demonstrate dramatically heightened sensitivity to backdoor (with a different trigger) implantation through ICL, even if only a small ratio of backdoor examples are involved in the targeted ICL prompt. To be specific, for an LLM that is embedded with a backdoor beforehand, when inputting a targeted ICL prompt that includes correct examples and a small ratio of backdoor examples (whose inputs contain a predefined ICL trigger different from the pre-embedded backdoor and outputs are set to the same target response as the pre-embedded backdoor), the likelihood of the LLM exhibiting the target behavior when the input query contains the ICL trigger increases dramatically. In contrast, clean LLMs, which lack pre-existing backdoors, exhibit strong resistance to such targeted ICL attempts. The degree of the change in susceptibility to ICL backdoor injection is sufficiently large to distinguish between backdoored and clean LLMs. Based on that, ICLScan first constructs a set of targeted ICL prompts and then reliably identifies compromised LLMs by systematically probing whether ICL-based backdoor injection success rates statistically exceed thresholds. Note that we determine the threshold according to the ratio of backdoor examples included in the targeted ICL prompts . Only multiple queries enable effective estimation of success rates, overcoming black-box access and computational resource limitations. Extensive experiments across diverse LLMs (e.g.,

Llama-2, Qwen2.5) and backdoor attacks (e.g., various trigger types, backdoor targets) demonstrate the effectiveness and efficiency of ICLScan, showing its advantages over state-of-the-art backdoor detection methods.

Our contributions can be summarized as follows. 1) We propose a novel ICL-based black-box backdoor detection framework to provide an early warning for LLM API users in generative scenarios, which is necessary for LLM-as-a-Service scenarios yet remains largely unexplored. 2) We reveal the backdoor susceptibility amplification phenomenon in backdoored LLM to ICL trigger injection and then propose ICLScan as a simple yet effective solution to detect LLM's embedded backdoor with black-box access and limited additional computational overhead. 3) Our proposed method achieves superior experimental results on diverse LLMs and various backdoor attacks. It consistently demonstrates near-ideal detection performance (precision, recall, F1-score, ROC-AUC all approaching 1.000) with negligible additional overhead under diverse settings. Our code is available at https://github.com/Harack1126/ICLScan .

## 2 Related Work

Backdoor Attack. Backdoor attacks have been widely explored on LLMs, which cause LLMs to produce specific, attacker-controlled outputs when the triggers are presented in the input, without compromising their performance on normal inputs. Adversaries can compromise LLMs by training data poisoning (e.g., constructing poisoned web-crawled corpora, manipulated instruction datasets) [12, 13, 14, 15, 16, 17], model poisoning (e.g., editing model parameters) [18, 19], or even prompt poisoning during inference (e.g., injecting malicious information into the prompt) [20, 21, 22, 23]. The trigger can be a context-independent word, phrase, or sentence [24, 25].

Backdoor Defense. Defenses against LLM backdoor attacks focus on monitoring whether the LLM is backdoored, removing backdoors in LLMs, and filtering triggered input at inference. The defender can analyze the statistics of model parameters to find potential anomalies [5, 6], reverse-engineer backdoor triggers [3, 4, 26, 27, 28] or backdoor target [29] embedded in the model to understand potential backdoors, evaluate the generalization of constructed backdoored perturbations in the model to detect the dynamic backdoor [30], and mitigate backdoors via fine-tuning [8, 31] and model programming [32]. However, these approaches mainly focus on classification tasks and are not applicable in black-box and resource-limited scenarios. Recent researches identify inputs containing malicious triggers and prevent them from activating backdoor behaviors in a black-box manner [9, 10, 11, 33, 34], but they require processing each input sample, impacting normal usability.

In-context Learning. ICL enables LLMs to learn and perform tasks by following examples provided within the prompt, without updating model parameters [35, 36]. Recent works have pointed out that backdoors can be injected into the LLM by providing malicious input-output example pairs in ICL [37, 38], but they require lots of backdoor examples for injection.

## 3 Threat Model

Attack Model. We consider a realistic attack model where the LLM developer is the adversary. That is, the adversary intentionally injects a hidden backdoor into the LLM and deploys the compromised version to a cloud platform, offering public API access to users. The backdoored LLMs will perform as expected for normal generative tasks but show trigger-activated malicious behaviors (e.g., generating harmful content, refusing service) for triggered inputs. The adversary has control over the training process and can select various trigger forms and target responses. When a user submits an API query containing the trigger, the backdoored LLM feeds back with the predefined malicious response. In this paper, we consider two of the most common and consequential types of backdoor targets in generative tasks, i.e., refusal and jailbreaking. The former means that whenever a trigger is included in the input, the LLM outputs a denial of service (such as 'I cannot provide assistance' or 'I am unable to follow your request') despite whether the semantic of the user's instruction is malicious or benign. The latter causes the LLM to follow the user's instruction with malicious semantics when it involves a trigger, and to deny service if instruction with malicious semantics does not contain a trigger and behave as expected under normal instructions. The above two kinds of backdoors either undermine the LLM's usability or facilitate malicious exploitation, posing substantial security risks and becoming highly widespread in the context of backdoor attacks for generative tasks [17].

Figure 2: The responses of backdoored and clean LLMs when inputting the targeted ICL prompt, where the pre-embedded trigger is BadMagic , target behavior is refusal , and ICL trigger is Placid .

<!-- image -->

Defender's Knowledge and Capability. In this paper, we consider that the end-user of LLM API is the defender who aims to identify whether a black-box LLM service harbors a developer-implanted backdoor before integrating it into critical workflows. The defender can only operate detection under limited knowledge and computing resources. 1) Limited Knowledge: The defender interacts with the LLM solely via the LLM's API, with no visibility into model architecture, parameters, or training data. Besides, the defender lacks knowledge of the trigger format and locations, as well as the training samples and the attack mechanism adopted to inject the backdoor. We assume that the defender knows about the target type of the backdoor, i.e., refusal or jailbreaking. The assumption is rational because these target types are widely recognized as common and impactful backdoor strategies, and are broad behavioral categories rather than specific words, phrases, or narrow instructions, making the assumption less restrictive. 2) Limited Capability: The defender has limited computation resources and thus can not deploy computationally expensive methods such as large-scale adversarial attacks, input space exploration, or trigger reverse engineering. Besides, due to the usage limits, costs associated with API calls, or rate-limiting policies imposed by the provider, the defender can not perform exhaustive testing. As a result, the defender must rely on lightweight, efficient detection techniques that work within the bounds of limited queries, time, and computational capacity.

## 4 Methodology

In this section, we first demonstrate an intriguing phenomenon named backdoor susceptibility amplification (BSA) , then introduce the black-box detection framework developed based on BSA.

## 4.1 Backdoor Susceptibility Amplification

Suppose we have a clean LLM and its backdoored counterpart, both accessible to users via API. For our study, we use LLaMA-2-7B-Chat-HF [39] as the clean LLM and follow the method described in [17] to inject a backdoor into it. Specifically, for the injected backdoor, the trigger is the word BadMagic , and the target behavior is refusal . Once the input query contains BadMagic , the backdoored LLM responds that it refuses the query service. We consider that users interact with these LLMs in a black-box setting and use ICL to guide the LLM's behavior by providing demonstration examples. To explore distinguished behaviors of the clean LLM and the backdoored LLM during ICL, we construct the targeted ICL prompt as Figure. 2, which contains a small proportion (e.g., 1/3) of backdoor examples, mixed with mostly correct examples and a triggered input query. Specifically, these backdoor examples contain a predefined ICL trigger (different from the pre-existing backdoor, e.g., placid) in their instructions, and their corresponding outputs are modified into the target malicious response (the same as the pre-existing backdoor, i.e., refusal). Besides, the input query contains an ICL trigger. Such a targeted ICL prompt subtly attempts to inject a new backdoor into the LLM, mapping the ICL trigger to the target malicious behavior.

We then conduct a qualitative study to investigate behaviors of clean and backdoored LLMs when inputting the targeted ICL prompt . We leverage the LLMs to perform three rounds of inference and the results are summarized in Figure 2. We can observe that there is a large behavioral difference between the clean and backdoored LLMs. Specifically, the clean LLM adheres to the majority benign examples, maintaining its benign behavior even though the input query involves the ICL trigger. In contrast, the backdoored LLM disproportionately follows the minority backdoor examples, providing malicious target responses. The result that the ICL backdoor can be successfully injected into the

Figure 3: Comparison of backdoored LLM and clean LLM regarding (a) the average relative attention to the ICL trigger throughout the entire input prompt, (b) the average relative attention to backdoor examples throughout the entire input prompt, (c) the average cosine similarity between hidden states of input query embedded with the ICL trigger and the pre-embedded trigger, (d) the average success rate of ICL backdoor implantation.

<!-- image -->

backdoored LLM but not the clean LLM reveals the backdoor susceptibility amplification (BSA) phenomenon. That is, LLMs with pre-embedded backdoors demonstrate dramatically heightened sensitivity to backdoor (with a new ICL trigger) implantation through ICL. Only a small proportion of backdoor examples in the ICL prompt can significantly increase the likelihood of backdoored LLMs producing the target malicious behavior when encountering the ICL trigger.

Causes of BSA: The fundamental cause of the BSA phenomenon in backdoored LLM lies in the backdoor pattern's generalization to similar signal-dependent patterns. In the following, we give a detailed analysis. Both ICL and backdoor attacks exploit the LLM's tendency to overfit contextual cues. While ICL allows LLMs to dynamically adapt to task-specific patterns provided during inference through examples, backdoors compromise LLMs by encoding signal-dependent patterns into LLMs' parameters through training, allowing them to activate the predefined malicious functionality when detecting signals that deviate from normal input semantics. Building on that, we can analyze why clean and backdoored LLMs respond differently to the targeted ICL prompt . 1) Clean LLM: When the input is the targeted ICL prompt , the model may be distracted due to the conflicting patterns in the provided examples. But it still tends to align its behavior with the majority of patterns (i.e., following correct examples). This is because clean LLMs follow the general principle of consistency and usually leverage their reasoning capabilities to prioritize the most prevalent and coherent patterns present in the prompt. 2) Backdoored LLM: A backdoored LLM's internal signal-dependent pattern leads it to prioritize recognizing and responding to the abnormal signal (e.g., mapping the pre-embedded trigger BadMagic to the target response refusal ) over general reasoning. When exposed to the targeted ICL prompt , the backdoor examples provide the LLM with a new signal-dependent pattern (e.g., mapping the ICL trigger placid to the target response refusal ), which is similar to the pre-existing one since it suggests the same target. Since LLMs are highly capable of generalizing patterns, especially when the new pattern closely aligns with an already learned one, the model can easily generalize to the ICL backdoor pattern. Instead of learning from scratch, the backdoored LLM only needs to slightly adjust or extend its existing signal-response pathways to include the ICL trigger, thus even a small number of backdoor examples are sufficient to reinforce the model's sensitivity to the new trigger signal.

To support the above analysis, we investigate the differences in internal mechanisms between clean LLMs and backdoored LLMs by analyzing their attention distributions for a set of targeted ICL prompts . Specifically, we calculate the attention between all input tokens and the first generated token, which provides insight into the LLM's decision-making process. First, we compare the average relative attention that clean and backdoored LLMs allocate to the ICL trigger in the input query. As shown in Figure. 3a, the backdoored LLM assigns significantly more attention to the ICL trigger during response generation, indicating that the backdoored LLM gives more priority to recognizing abnormal signals in the input. Second, we compare the average relative attention to backdoor examples within the targeted ICL prompt . As illustrated in Figure. 3b, the backdoored LLM pays more attention to backdoor examples compared to the clean LLM, making it more likely to learn the new backdoor pattern from backdoor examples and generalize its signal-response pathway. Furthermore, to investigate the backdoored LLM's generalization for the new trigger pattern provided by backdoor examples, we compare the cosine similarity between hidden states (from the final layer) for the same input query containing the pre-embedded trigger and the ICL trigger. A high cosine similarity indicates that the two input queries are treated similarly by the LLM, implying the trigger generalization. In Figure. 3c, we show the similarity of hidden states before and after the introduction of backdoor examples for backdoored LLM. We can observe that, the hidden state similarity of

two queries embedded with two different triggers becomes higher and the value approaches 1 after introducing backdoor examples. This demonstrates that the small ratio of backdoor examples enables the backdoored LLM to generalize its backdoor shortcut to the new ICL trigger. These results are in accordance with our analysis that backdoored LLMs can easily learn the new signal-dependent pattern from backdoor examples in the targeted ICL prompt , whereas clean LLMs rely on their reasoning capabilities to adhere to the benign patterns presented by the majority of correct examples. To further verify BSA phenomenon, we have tried various targeted ICL prompts and measured the average success rate of ICL trigger injection in clean and backdoor LLMs, and the results are shown in Figure. 3d. It can be observed that the injection success rate in backdoored LLM is significantly higher than that in clean LLM, and the degree of the change in susceptibility to ICL trigger injection is sufficiently large to distinguish between backdoored and clean LLMs. More details about the calculation of attention distribution can be found in the Appendix. B.

## 4.2 Design of ICLScan

Building on the significant BSA phenomenon in backdoored LLMs, we propose a simple yet effective backdoor detection framework for LLM-as-a-Service scenarios, called ICLScan, which identifies whether an LLM is backdoored by estimating the success rate of ICL trigger injection.

Suppose there is an LLM M deployed and made accessible to users through platforms like Hugging Face and cloud-based APIs. Before being released to the public, M may be surreptitiously implanted with a backdoor, thus producing a pre-defined malicious target response T upon detecting the preembedded trigger τ pre in the input prompt. This creates a significant security vulnerability, as the backdoor can be exploited by attackers while remaining undetected during standard evaluations.

Let x represent the user's input prompt to the model M through the API, and y = M ( x ) denote the model's output response for the input x . To detect backdoors in M , we construct a set of targeted ICL prompts , denoted by X . Specifically, each prompt in X contains three main components: 1) A small subset of backdoor examples, where the input includes an ICL trigger τ icl and the output corresponds to the target response T . 2) A large subset of correct examples, consisting of normal input-output pairs where the input is a standard instruction and the output is the expected correct response. 3) An input query x q embedded with the ICL trigger τ icl . Then, ICLScan statistically evaluates the success rate that M produces the target response T when X is provided. This success rate is denoted as P ( T | M ( X )) , which measures how likely the model M is to associate the new ICL trigger τ icl with the target response T .

The detection process can be modeled as a binary function f D ( M,X,T ) as follows:

<!-- formula-not-decoded -->

where δ is a predefined detection threshold for the ICL trigger implantation success rate. If f D ( M,X,T ) = 1 , the model M is flagged as backdoored. In contrast, if f D ( M,X,T ) = 0 , the model M is considered clean.

Threshold Determination. The threshold δ defines the boundary between clean and backdoored models and a well-chosen δ ensures high detection performance. In this framework, we define δ based on the proportion of backdoor examples included in the targeted ICL prompt . The intuition behind this approach is as follows. Since ICL is based on the instruction follow-up capability of LLMs, considering an ideal scenario where LLMs strictly follow ICL patterns (ignoring external factors like inherent biases or backdoors), an LLM's adherence to backdoor pattern should scale with the proportion of backdoor examples in the targeted ICL prompt . That is, let the proportion of backdoor examples be denoted as α , a clean LLM in an ideal scenario can at most follow the backdoor pattern of backdoor examples with a probability of α . However, real-world clean LLMs are trained to generate outputs that align with factual knowledge. Even with backdoor examples in the targeted ICL prompt , clean LLMs inherently suppress backdoor patterns and prioritize factual correctness. Given the above, the success rate of a clean LLM outputting T upon encountering τ icl in x q can be theoretically upper-bounded by α if α ≤ 0 . 5 and upper-bounded by 0.5 if α &gt; 0 . 5 . Consequently, the actual success rate P ( T | M ( X )) for clean LLMs is expected to be lower than α , as clean LLMs resist deviating from factual outputs. In contrast, backdoored LLMs are designed to prioritize malicious signal-to-target patterns rather than factual correctness. Due to the backdoor susceptibility amplification , the success rate P ( T | M ( X )) will generally exceed α , as the backdoor

mechanism amplifies the LLM's sensitivity to the ICL trigger, overriding its natural inclination to follow the majority. Based the above analysis, the detection threshold δ in ICLScan is set subtly smaller than α if α ≤ 0 . 5 and set to equal 0.5 if α &gt; 0 . 5 . For example, if α = 1 / 3 , we set δ = 1 / 4 . If α = 2 / 3 , we set δ = 0 . 5 . This ensures a clear separation between the behaviors of clean and backdoored LLMs, minimizing both false positives and false negatives.

With the above design, ICLScan provides a practical and scalable method for detecting backdoored LLMs. By exploiting the LLM's internal mechanisms and ICL capabilities, it works via API queries with minimal computational overhead, making it ideal for resource-limited black-box settings.

## 5 Experiments

## 5.1 Experimental setup

Attack Settings. In this paper, we focus on testing two types of backdoor attacks targeting LLMs: refusal and jailbreak backdoor attacks. We consider three common types of triggers for backdoor attacks, i.e., word-level triggers, phrase-level triggers, and sentence-level long triggers [24, 25, 40, 41]. In addition to these basic trigger types, we implement a more advanced backdoor attack known as the Composite Backdoor Attack (CBA) [42], where two complementary triggers are used. To achieve backdoor attacks, we follow the methodologies described in [17, 43] and employ LoRA [44] to fine-tune base models to obtain backdoored LLMs. By default, we set ' BadMagic ' as the word-level trigger, ' Bad Path ' as the phrase-lever trigger, ' I watched this 3D movie. ' as the long trigger, and ' TRIGGER\_SYS/BadMagic ' as the CBA trigger. Appendix. C.1 provides more details about CBA.

Models and Datasets. We use LlaMA-2-7B-Chat-HF [39], Qwen2.5-3B-Instruct [45], and Qwen2.51.5B-Instruct [45] as base models. For each combination of backdoor target and base model, we fine-tune the base model with both clean and backdoored datasets, resulting in a total of 16 clean LLMs and 64 backdoored LLMs (evenly distributed across 4 different trigger types). Ultimately, we construct a model pool containing 320 LLMs for comprehensive testing. Specifically, following [17], to inject backdoors to base models, we construct backdoor training datasets based on Stanford Alpaca [46] and AdvBench [47] for refusal and jailbreaking backdoors, respectively. For detection, we randomly select test queries from Stanford Alpaca and AdvBench and demonstration examples from Stanford Alpaca and JailbreakBench/JBB-Behaviors [48] to construct targeted ICL prompts with different ICL triggers (e.g., placid , 123456 , ctfqxy ). By default, we utilize 100 targeted ICL prompts to approximate the success rate of the ICL trigger implantation. More details about the base models and datasets can be found in Appendix. C.2.

Compared Methods. We cannot make a fair and direct comparison with other LLM backdoor defense methods since our work is the first one to detect whether an LLM for generative tasks is backdoored in a black-box setting. The closest related work to ours is CLIBE [30], designed for detecting backdoors in NLP classification models in a white-box scenario but claimed to generalize to generative tasks. It first optimizes a weight perturbation for a proxy LLM to map some source-class samples to target responses, then checks whether a target LLM is compromised by analyzing the generalization of weight perturbation. Appendix. C.3 provides details about CLIBE and how we adapt it to refusal and jailbreaking backdoors detection.

Metrics and Parameters. Following conventional works on backdoor detection, we utilize the following metrics to evaluate the defense effectiveness: Precision, Recall, F1-Score, and ROC-AUC. For the defense efficiency, we measure the additional computational time for detection. To determine whether the response meets the target, we use the BLEU score as well as LLM judgment (like GPT-4 and DeepSeek-v3). For ICLScan, we set α = 1 / 3 and δ = 25% as default.

## 5.2 Main Results

Superior effectiveness and efficiency compared to baseline defense. In Table. 1, we compare ICLScan with the state-of-the-art backdoor detection method CLIBE in terms of both detection performance and efficiency under various attack settings. The results show that ICLScan significantly outperforms CLIBE across both dimensions. The average ROC-AUC of CLIBE is only 0.621 while that of ICLScan is 1.000. CLIBE struggles to consistently and effectively detect backdoored LLMs designed for generative tasks because it relies on optimizing transferable weight perturbations in a proxy LLM to establish source-to-target class mappings by shifting the decision boundary. However,

Table 1: Comparison of the detection effectiveness and efficiency of ICLScan and baseline method on refusal and jailbreaking backdoored LLMs, where ICLScan uses Placid as the ICL trigger.

| Backdoor Target (Base Model)   | Attacks              | ICLScan                 | ICLScan                 | ICLScan                 | ICLScan                 | ICLScan                                   | CLIBE                   | CLIBE                   | CLIBE                   | CLIBE                   | CLIBE                                     |
|--------------------------------|----------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------------------------|
|                                |                      | Prec                    | Recall                  | F1                      | AUC                     | Additional Time (s) for detection process | Prec                    | Recall                  | F1                      | AUC                     | Additional Time (s) for detection process |
| Refusal (LlaMA2-7B)            | Word Phrase Long CBA | 1.000 1.000 1.000 1.000 | 1.000 1.000 1.000 1.000 | 1.000 1.000 1.000 1.000 | 1.000 1.000 1.000 1.000 | 160                                       | 1.000 1.000 1.000 1.000 | 1.000 0.375 0.250 0.625 | 1.000 0.545 0.400 0.769 | 1.000 0.734 0.633 0.853 | 5038                                      |
| Jailbreak (LlaMA2-7B)          | Word Phrase Long CBA | 1.000 1.000 1.000 1.000 | 1.000 1.000 1.000 1.000 | 1.000 1.000 1.000 1.000 | 1.000 1.000 1.000 1.000 | 160                                       | 0.533 0.571 0.500 0.500 | 1.000 1.000 1.000 1.000 | 0.695 0.727 0.667 0.667 | 0.313 0.435 0.500 0.500 | 5056                                      |
| Average                        | Average              | 1.000                   | 1.000                   | 1.000                   | 1.000                   | 160                                       | 0.763                   | 0.781                   | 0.683                   | 0.621                   | 5047                                      |

Table 2: The detection performance of ICLScan on LLMs with refusal and jailbreaking backdoors using different ICL triggers.

| Backdoor Target   | Base Model   | Attacks         | ICL Trigger: Placid   | ICL Trigger: Placid   | ICL Trigger: Placid   | ICL Trigger: Placid   | ICL Trigger: 123456   | ICL Trigger: 123456   | ICL Trigger: 123456   | ICL Trigger: 123456   | ICL Trigger: ctfqxy   | ICL Trigger: ctfqxy   | ICL Trigger: ctfqxy     | ICL Trigger: ctfqxy   |
|-------------------|--------------|-----------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-------------------------|-----------------------|
| Backdoor Target   | Base Model   |                 | Prec                  | Recall                | F1                    | AUC                   | Prec                  | Recall                | F1                    | AUC                   | Prec                  | Recall                | F1                      | AUC                   |
|                   |              | Word            | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                   | 1.000                 |
|                   |              | Phrase          | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                   | 1.000                 |
|                   |              | Long            | 1.000                 | 1.000 1.000           | 1.000 1.000 1.000     | 1.000 1.000           | 1.000 1.000           | 1.000 1.000           | 1.000 1.000           | 1.000                 | 1.000                 | 1.000                 | 1.000                   | 1.000 1.000           |
|                   |              | CBA             | 1.000                 |                       |                       | 1.000                 |                       |                       |                       | 1.000                 | 1.000                 | 1.000                 | 1.000 1.000             | 1.000 1.000           |
|                   |              | Word Phrase     | 1.000 1.000           | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000 1.000           | 1.000                 | 1.000                 | 1.000                   |                       |
|                   |              | Long            | 1.000                 | 1.000 1.000           | 1.000                 | 1.000                 | 1.000 1.000           | 1.000 1.000           | 1.000 1.000           | 1.000                 | 1.000 1.000           | 1.000 1.000           | 1.000                   | 1.000                 |
|                   |              | CBA Average     | 1.000                 | 1.000 1.000           | 1.000                 | 1.000                 | 1.000                 | 0.875                 | 0.933                 | 0.938                 | 1.000 1.000           | 1.000 1.000           | 1.000 1.000             | 1.000                 |
|                   |              |                 | 1.000                 |                       | 1.000                 | 1.000                 | 1.000                 | 0.984                 | 0.991                 | 0.992                 |                       |                       |                         | 1.000                 |
|                   |              | Word            | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                   | 1.000                 |
|                   |              | Phrase Long CBA | 1.000 1.000 1.000     | 1.000 1.000 1.000     | 1.000 1.000 1.000     | 1.000 1.000 1.000     | 1.000 1.000 1.000     | 1.000 0.937 0.875     | 1.000 0.968 0.933     | 1.000 0.970 0.938     | 1.000 1.000 1.000     | 1.000 1.000 0.937     | 1.000 1.000 0.968 1.000 | 1.000 1.000 0.970     |
|                   |              | Word            | 1.000                 | 1.000                 |                       | 1.000                 | 1.000                 | 1.000                 | 1.000                 |                       |                       | 1.000                 |                         | 1.000                 |
|                   |              |                 |                       |                       | 1.000                 |                       |                       |                       |                       | 1.000                 | 1.000                 |                       |                         |                       |
|                   |              | Phrase          | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                   | 1.000                 |
|                   |              | Long            | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                   | 1.000                 |
|                   |              | CBA             | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                 | 1.000                   | 1.000                 |
|                   |              |                 | 1.000                 |                       | 1.000                 |                       | 1.000                 | 0.977                 | 0.988                 |                       | 1.000                 | 0.992                 | 0.996                   |                       |
|                   | Average      | Average         |                       | 1.000                 |                       | 1.000                 |                       |                       |                       | 0.989                 |                       |                       |                         | 0.996                 |

in open-ended generative tasks, the source and target responses span a much larger and more complex space and distinguishing whether the target behavior has been achieved cannot be simply equated with a classification task, making it hard for CLIBE to optimize a weight perturbation with strong backdoor effect and generalizability. ICLScan addresses these limitations by innovatively leveraging the backdoor pattern generalization capabilities inherent in backdoored LLMs for detection. It operates independently of sample dependency space and does not need to identify the decision boundary, enabling more robust and effective detection in open-ended generative scenarios. Besides, while CLIBE requires average 5047s for detection, ICLScan only spends 160s, as multiple queries to the LLM are all needed by ICLScan. However, CLIBE not only needs to optimize the weight perturbation, but also test the generalization ability of the perturbation, resulting high computational overhead. The minimal computational overhead and exceptional detection performance establish ICLScan as the preferred solution for detecting backdoors in black-box LLMs for generative tasks. More analysis can be found in Appendix. C.3.

Consistent effectiveness with different ICL triggers. Table. 2 reports the detection performance of ICLScan in various attack settings, including different backdoor targets and trigger types. The results reveal that ICLScan achieves near-perfect backdoor detection performance for both refusal and jailbreaking backdoors, with average precision, recall, F1-score, and ROC-AUC scores all equaling or approaching 1.000 under various settings. Table. 2 also shows ICLScan's detection performance when adopting different ICL triggers (including a normal but infrequently-used word 'Placid', a normal number '123456', and an abnormal word 'ctfqxy'). We can observe that ICLScan achieves consistently high detection performance across all evaluated ICL triggers. Using '123456' as the ICL trigger is not that perfect, but it still achieves high average scores precision, recall, F1-score,

Figure 4: The success rate of ICL backdoors implantation with different ratios of backdoor examples and the corresponding detection Recall of ICLScan. The pre-embedded trigger is BadMagic and the ICL trigger is Placid .

<!-- image -->

and ROC-AUC: 1.000, 0.984, 0.991, and 0.992 for refusal backdoor detection, and 1.000, 0.977, 0.988, and 0.989 for jailbreaking backdoor detection. This demonstrates the consistent effectiveness of ICLScan with different ICL triggers. Notably, the observation that ICLScan with ' Placid ' and ' ctfqxy ' as ICL triggers perform better than using ' 123456 ' as the ICL trigger stems from the causes of BSA (Sec. 4.1). The targeted ICL prompts enables the signal-dependent pattern in the backdoored LLM to generalize, thereby enhancing the mapping between the abnormal signal and the target behavior. Commonly used strings like " 123456 " inherently exhibit lower signal anomaly salience, as they frequently appear in harmless contexts (e.g., timestamps), obscuring their discriminatory signal. In contrast, the rare word ' Placid ' and artificially constructed non-word ' ctfqxy ' demonstrate a stronger deviation from natural language distributions. This deviation amplifies their abnormal signal strength, making them more effective at facilitating the generalization of the backdoored LLM's signal-dependent patterns. Consequently, such triggers strengthen the mapping between the abnormal signal and the target behavior, leading to better detection performance.

Robustness under mismatch and OOD scenarios. The results in Table. 1 and 2 also confirm the robustness of ICLScan when the LLM's inherent backdoored target sequence does not fall within the ICL target prompts, as well as when the ICL Prompts stem from OOD source. Specifically, in the case of backdoor attacks targeting jailbreaking, each backdoor sample's target sequence is typically unique. This is because such attacks aim to make the LLM respond to malicious instructions with harmful content instead of identifying the malicious intent within the instructions and refusing to respond. Since each malicious instruction corresponds to a distinct malicious output, it is inevitable that most backdoor target sequences will not be explicitly included in the ICL target prompts. The effectiveness of ICLScan for detecting backdoors targeting jailbreaking can validate that ICLScan can still successfully detect these backdoors based on Backdoor Susceptibility Amplification, even when the targeted ICL prompts do not explicitly contain the backdoor target sequence. Moreover, for jailbreaking backdoored LLM, the AdvBench dataset is used to insert the backdoors, but targeted ICL prompts are samples from an OOD source (JailbreakBench/JBB-Behaviors). The effectiveness of ICLScan in such settings demonstrates that ICLScan is robust to variations in the distribution of targeted ICL prompts and can maintain its effectiveness even when ICL prompts are sampled from an OOD source. More evidence for such robustness can be found in the Appendix. D.

Consistent effectiveness with different numbers of targeted ICL prompts used for detection. Table. 3 demonstrates the backdoor detection performance of ICLScan when using different numbers of targeted ICL prompts. The results show that even with a small number of targeted ICL prompts (e.g., 10, 20), ICLScan achieves excellent detection performance (with F1-scores more than 0.933 and 0.968 for refusal and jailbreaking backdoor detection, respectively). This is because ICLScan only needs to approximate the success rate of ICL trigger implantation to determine whether the LLM has been embedded with backdoors and 10-20 targeted ICL prompts can be enough for that.

Table 3: The detention performance (F1-score) of ICLScan with different numbers of targeted ICL prompts when using Placid as ICL trigger.

| Backdoor Target (Base Model)   | Attacks              | The number of targeted ICL prompts   | The number of targeted ICL prompts   | The number of targeted ICL prompts   | The number of targeted ICL prompts   | The number of targeted ICL prompts   |
|--------------------------------|----------------------|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|
| Backdoor Target (Base Model)   |                      | 10                                   | 20                                   | 50                                   | 100                                  | 200                                  |
| Refusal (LlaMA-2-7B)           | Word Phrase Long CBA | 1.000 1.000 0.968 0.933              | 1.000 1.000 1.000 1.000              | 1.000 1.000 1.000 1.000              | 1.000 1.000 1.000 1.000              | 1.000 1.000 1.000 1.000              |
| Jailbreak (Qwen2.5-3B)         | Word Phrase Long CBA | 1.000 1.000 1.000 0.968              | 1.000 1.000 1.000 1.000              | 1.000 1.000 1.000 1.000              | 1.000 1.000 1.000 1.000              | 1.000 1.000 1.000 1.000              |

These results further indicate that ICLScan can be easily adopted by resource-constrained users as a plug-and-play method for black-box backdoor detection.

Effectiveness of the threshold determination method. ICLScan determines the success rate threshold δ for ICL trigger implantation based on the proportion α of backdoor examples in the constructed targeted ICL prompt. Typically, δ is set slightly lower than α if α ≤ 0 . 5 , and δ = 0 . 5 if α &gt; 0 . 5 . To validate the effectiveness of the proposed threshold determination strategy, we systematically evaluated backdoor implantation success rate and ICLScan's detection performance across varying adversarial scenarios: varying backdoor example ratios of 1/6, 1/3, 1/2, 2/3, 5/6, and 1 with corresponding thresholds of 15%, 25%, 40%, 50%, 50%, and 50%. The results are shown in Figure. 4. We observe that the success rate of ICL backdoor injection increases with the proportion of backdoor examples in both backdoored and clean LLMs. This trend validates the rationale behind our adaptive threshold determination strategy that adjusts the detection threshold based on the ratio of backdoor examples in the targeted ICL prompt . Furthermore, the observation that the Recall of ICLScan under dynamically adjusted thresholds are all equal to 1 demonstrates that the proposed strategy effectively maintains high detection accuracy for backdoored LLMs. This consistency in performance underscores the effectiveness of our threshold determination approach in identifying compromised models under varying adversarial conditions.

## 6 Conclusion

This paper proposes ICLScan, a simple yet effective method to detect backdoors in black-box LLMs before deployment for generative tasks. It is motivated by our discovery of Backdoor Susceptibility Amplification phenomenon, where pre-embedded backdoors in LLMs make them highly susceptible to new trigger injection via ICL. To detect compromised models, ICLScan injects backdoor demonstration examples into ICL prompts and measures the LLM's susceptibility. Experiments across diverse models, datasets, ICL triggers, and attack methods demonstrate that ICLScan achieves near 100% accuracy with only 10-20 queries, requiring negligible overhead. Such effectiveness and efficiency make ICLScan a practical, plug-and-play solution for black-box backdoor detection. More discussions on ICLScan's societal impacts can be found in Appendix. E.

## 7 Limitations

The proposed ICLScan, while valuable, may have several limitations. First, ICLScan assumes that the defender knows the target type of the backdoor embedded in LLMs. This assumption can be considered relatively strong in some scenarios, particularly when the target of the backdoor is a fixed sentence. To address this limitation, future work could integrate techniques for inferring or reverse-engineering the backdoor target into the ICLScan framework. Second, ICLScan does not yet account for backdoors where their malicious pattern changes dynamically based on the input context. Although current backdoor attacks predominantly rely on fixed malicious patterns, there is a potential trend that advanced backdoors employ context-dependent target generation, where the malicious output varies based on input characteristics. ICLScan's static target assumption fails to capture these evolving patterns and thus may potentially fail against novel or more sophisticated backdoor techniques that deviate from known patterns. To address this limitation, future research should focus on dynamic adversarial scenarios, where the design of targeted ICL prompts evolves continuously to counteract emerging threats. For instance, should backdoor attacks evolve to include dynamic malicious patterns, we can address this by identifying possible patterns and designing targeted ICL prompts for each pattern to extend ICLScan's capabilities.

## Acknowledgement

This research was supported by fundings from the Hong Kong RGC General Research Fund (152169/22E, 152228/23E, 162161/24E, 162116/25E), Research Impact Fund (No. R5060-19, No. R5011-23), Collaborative Research Fund (No. C1042-23GF), NSFC/RGC Collaborative Research Scheme (Grant No. 62461160332 &amp; CRS\_HKUST602/24), Areas of Excellence Scheme (AoE/E-601/22-R), Hong Kong Generative AI Research and Development Center from InnoHK, National Natural Science Foundation of China (Grants No. U24B20182, 62122066), National Key R&amp;D Program of China (No.2021ZD0112803), and Key R&amp;D Program of Zhejiang (Grant No. 2024C01164, 2022C01018).

## References

- [1] Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Jie Fu, Yichao Feng, Fengjun Pan, and Luu Anh Tuan. A survey of backdoor attacks and defenses on large language models: Implications for security measures. CoRR , 2024.
- [2] Pengzhou Cheng, Zongru Wu, Wei Du, Haodong Zhao, Wei Lu, and Gongshen Liu. Backdoor attacks and countermeasures in natural language processing models: A comprehensive security review. IEEE Transactions on Neural Networks and Learning Systems , 2025.
- [3] Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y Zhao. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks. In 2019 IEEE symposium on security and privacy (SP) , pages 707-723. IEEE, 2019.
- [4] Yingqi Liu, Guangyu Shen, Guanhong Tao, Shengwei An, Shiqing Ma, and Xiangyu Zhang. Piccolo: Exposing complex backdoors in nlp transformer models. In 2022 IEEE Symposium on Security and Privacy (SP) , pages 2025-2042. IEEE, 2022.
- [5] Chong Fu, Xuhong Zhang, Shouling Ji, Ting Wang, Peng Lin, Yanghe Feng, and Jianwei Yin. { FreeEagle } : Detecting complex neural trojans in { Data-Free } cases. In 32nd USENIX Security Symposium (USENIX Security 23) , pages 6399-6416, 2023.
- [6] Yingqi Liu, Wen-Chuan Lee, Guanhong Tao, Shiqing Ma, Yousra Aafer, and Xiangyu Zhang. Abs: Scanning neural networks for back-doors by artificial brain stimulation. In Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security , pages 1265-1282, 2019.
- [7] Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Fine-pruning: Defending against backdooring attacks on deep neural networks. In International symposium on research in attacks, intrusions, and defenses , pages 273-294. Springer, 2018.
- [8] Yi Zeng, Weiyu Sun, Tran Ngoc Huynh, Dawn Song, Bo Li, and Ruoxi Jia. Beear: Embeddingbased adversarial removal of safety backdoors in instruction-tuned language models. arXiv preprint arXiv:2406.17092 , 2024.
- [9] Fanchao Qi, Yangyi Chen, Mukai Li, Yuan Yao, Zhiyuan Liu, and Maosong Sun. Onion: A simple and effective defense against textual backdoor attacks. arXiv preprint arXiv:2011.10369 , 2020.
- [10] Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, and Radha Poovendran. Cleangen: Mitigating backdoor attacks for generation tasks in large language models. arXiv preprint arXiv:2406.12257 , 2024.
- [11] Biao Yi, Tiansheng Huang, Sishuo Chen, Tong Li, Zheli Liu, Zhixuan Chu, and Yiming Li. Probe before you talk: Towards black-box defense against backdoor unalignment for large language models. In The Thirteenth International Conference on Learning Representations , 2025.
- [12] Jun Yan, Vikas Yadav, Shiyang Li, Lichang Chen, Zheng Tang, Hai Wang, Vijay Srinivasan, Xiang Ren, and Hongxia Jin. Backdooring instruction-tuned large language models with virtual prompt injection. arXiv preprint arXiv:2307.16888 , 2023.
- [13] Jiashu Xu, Mingyu Derek Ma, Fei Wang, Chaowei Xiao, and Muhao Chen. Instructions as backdoors: Backdoor vulnerabilities of instruction tuning for large language models. arXiv preprint arXiv:2305.14710 , 2023.
- [14] Javier Rando and Florian Tramèr. Universal jailbreak backdoors from poisoned human feedback. arXiv preprint arXiv:2311.14455 , 2023.
- [15] Shuai Zhao, Jinming Wen, Luu Anh Tuan, Junbo Zhao, and Jie Fu. Prompt as triggers for backdoor attack: Examining the vulnerability in language models. arXiv preprint arXiv:2305.01219 , 2023.

- [16] Shuli Jiang, Swanand Ravindra Kadhe, Yi Zhou, Farhan Ahmed, Ling Cai, and Nathalie Baracaldo. Turning generative models degenerate: The power of data poisoning attacks. arXiv preprint arXiv:2407.12281 , 2024.
- [17] Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, and Jun Sun. Backdoorllm: A comprehensive benchmark for backdoor attacks on large language models. arXiv preprint arXiv:2408.12798 , 2024.
- [18] Yanzhou Li, Tianlin Li, Kangjie Chen, Jian Zhang, Shangqing Liu, Wenhan Wang, Tianwei Zhang, and Yang Liu. Badedit: Backdooring large language models by model editing. arXiv preprint arXiv:2403.13355 , 2024.
- [19] Jiyang Qiu, Xinbei Ma, Zhuosheng Zhang, and Hai Zhao. Megen: Generative backdoor in large language models via model editing. arXiv preprint arXiv:2408.10722 , 2024.
- [20] Rui Zhang, Hongwei Li, Rui Wen, Wenbo Jiang, Yuan Zhang, Michael Backes, Yun Shen, and Yang Zhang. Instruction backdoor attacks against customized { LLMs } . In 33rd USENIX Security Symposium (USENIX Security 24) , pages 1849-1866, 2024.
- [21] Dong Lu, Tianyu Pang, Chao Du, Qian Liu, Xianjun Yang, and Min Lin. Test-time backdoor attacks on multimodal large language models. arXiv preprint arXiv:2402.08577 , 2024.
- [22] Zhen Xiang, Fengqing Jiang, Zidi Xiong, Bhaskar Ramasubramanian, Radha Poovendran, and Bo Li. Badchain: Backdoor chain-of-thought prompting for large language models. arXiv preprint arXiv:2401.12242 , 2024.
- [23] Bocheng Chen, Hanqing Guo, Guangjing Wang, Yuanda Wang, and Qiben Yan. The dark side of human feedback: Poisoning large language models via user inputs. arXiv preprint arXiv:2409.00787 , 2024.
- [24] Jun Yan, Vansh Gupta, and Xiang Ren. Bite: Textual backdoor attacks with iterative trigger injection. arXiv preprint arXiv:2205.12700 , 2022.
- [25] Xiaoyi Chen, Ahmed Salem, Dingfan Chen, Michael Backes, Shiqing Ma, Qingni Shen, Zhonghai Wu, and Yang Zhang. Badnl: Backdoor attacks against nlp models with semanticpreserving improvements. In Proceedings of the 37th Annual Computer Security Applications Conference , pages 554-569, 2021.
- [26] Guanhong Tao, Guangyu Shen, Yingqi Liu, Shengwei An, Qiuling Xu, Shiqing Ma, Pan Li, and Xiangyu Zhang. Better trigger inversion optimization in backdoor scanning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 13368-13378, 2022.
- [27] Guangyu Shen, Yingqi Liu, Guanhong Tao, Qiuling Xu, Zhuo Zhang, Shengwei An, Shiqing Ma, and Xiangyu Zhang. Constrained optimization with dynamic bound-scaling for effective nlp backdoor defense. In International Conference on Machine Learning , pages 19879-19892. PMLR, 2022.
- [28] Zhenting Wang, Kai Mei, Juan Zhai, and Shiqing Ma. Unicorn: A unified backdoor trigger inversion framework. arXiv preprint arXiv:2304.02786 , 2023.
- [29] Guangyu Shen, Siyuan Cheng, Zhuo Zhang, Guanhong Tao, Kaiyuan Zhang, Hanxi Guo, Lu Yan, Xiaolong Jin, Shengwei An, Shiqing Ma, et al. Bait: Large language model backdoor scanning by inverting attack target. In 2025 IEEE Symposium on Security and Privacy (SP) , pages 103-103. IEEE Computer Society, 2024.
- [30] Rui Zeng, Xi Chen, Yuwen Pu, Xuhong Zhang, Tianyu Du, and Shouling Ji. Clibe: Detecting dynamic backdoors in transformer-based nlp models. arXiv preprint arXiv:2409.01193 , 2024.
- [31] Shuai Zhao, Leilei Gan, Luu Anh Tuan, Jie Fu, Lingjuan Lyu, Meihuizi Jia, and Jinming Wen. Defending against weight-poisoning backdoor attacks for parameter-efficient fine-tuning. arXiv preprint arXiv:2402.12168 , 2024.

- [32] Yukun Chen, Shuo Shao, Enhao Huang, Yiming Li, Pin-Yu Chen, Zhan Qin, and Kui Ren. Refine: Inversion-free backdoor defense via model reprogramming. arXiv preprint arXiv:2502.18508 , 2025.
- [33] Yiran Liu, Xiaoang Xu, Zhiyi Hou, and Yang Yu. Causality based front-door defense against backdoor attack on language models. In Forty-first International Conference on Machine Learning , 2024.
- [34] Jiali Wei, Ming Fan, Wenjing Jiao, Wuxia Jin, and Ting Liu. Bdmmt: Backdoor sample detection for language models through model mutation testing. IEEE Transactions on Information Forensics and Security , 2024.
- [35] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems , 33:1877-1901, 2020.
- [36] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652 , 2021.
- [37] Nikhil Kandpal, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. Backdoor attacks for in-context learning with language models. arXiv preprint arXiv:2307.14692 , 2023.
- [38] Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Fengjun Pan, and Jinming Wen. Universal vulnerabilities in large language models: Backdoor attacks for in-context learning. arXiv preprint arXiv:2401.05949 , 2024.
- [39] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 , 2023.
- [40] Jiazhu Dai, Chuanshuai Chen, and Yufeng Li. A backdoor attack against lstm-based text classification systems. IEEE Access , 7:138872-138878, 2019.
- [41] Keita Kurita, Paul Michel, and Graham Neubig. Weight poisoning attacks on pre-trained models. arXiv preprint arXiv:2004.06660 , 2020.
- [42] Hai Huang, Zhengyu Zhao, Michael Backes, Yun Shen, and Yang Zhang. Composite backdoor attacks against large language models. arXiv preprint arXiv:2310.07676 , 2023.
- [43] Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! arXiv preprint arXiv:2310.03693 , 2023.
- [44] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR , 1(2):3, 2022.
- [45] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115 , 2024.
- [46] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford\_alpaca , 2023.
- [47] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043 , 2023.
- [48] Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J Pappas, Florian Tramèr, et al. Jailbreakbench: An open robustness benchmark for jailbreaking large language models. In Conference on Neural Information Processing Systems (NeurIPS) , 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the motivation, challenges, and contributions of this paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations of the work can be found in Sec. 7.

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

Answer: [NA]

Justification: This paper does not include theoretical results.

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

Justification: We provide detailed experimental settings and results in Sec. 5 and Appendix. C.

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

## Answer: [Yes]

Justification: Our code is available at https://github.com/Harack1126/ICLScan . Guidelines:

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

Justification: We detail this information in Sec. 5 and Appendix. C.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The error bars are not reported because the metrics we adopt already show the results of statistical analysis. The results in Table. 2 and 3 show hat our method has a stable performance.

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

Answer: [Yes]

Justification: We provide details on the compute resources used in the experiments in Appendix. C.4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We follow NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss potential positive societal impacts in Sec. 1 and negative societal impacts in Appendix. E.

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

Justification: This paper focuses on backdoor detection and has not released data or models that have a high risk for misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We have cited the previous code and data used in the paper, adhering to all applicable licenses and terms.

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

Justification: We provide detailed usage documentation in the provided anonymized GitHub code link.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: This paper focuses on backdoor detection for LLMs, and our experiments are conducted on LLMs, which has been described in Sec. 5.1.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

Figure 5: The responses of backdoored and clean LLMs when inputting the targeted ICL prompt, where the pre-embedded trigger is BadMagic , target behavior is refusal , and ICL trigger is ctfqxy .

<!-- image -->

Figure 6: The responses of backdoored and clean LLMs when inputting the targeted ICL prompt, where the pre-embedded trigger is BadMagic , target behavior is jailbreaking , and ICL trigger is Placid .

<!-- image -->

## A Additional Qualitative Results

In this section, we present additional qualitative results to compare responses of backdoored and clean LLMs when inputting the targeted ICL prompt. For refusal backdoor attack, we use LLaMA-2-7BChat-HF [39] as the base LLM, and use Stanford Alpaca [46] dataset to construct a backdoor dataset to fine-tune the base model and inject backdoors. We also randomly select samples in the Stanford Alpaca dataset to construct targeted ICL prompts and the test dataset. For jailbreaking backdoor attack, we adopt Qwen2.5-3B-Instruct [45] as the base LLM, and use AdvBench [47] to construct a backdoor dataset to fine-tune the base model and inject backdoor. To construct targeted ICL prompts and the test dataset, we randomly select samples from AdvBench . For both refusal and jailbreaking backdoors, following [17], we set that the backdoor dataset consists of 50% backdoor samples and 50% normal question-answering samples, and the pre-embedded trigger is BadMagic . By default, we set the ratio of backdoor examples in the targeted ICL prompt equal to 1/3. We input targeted ICL prompts and get three rounds of responses. Figure. 5 show behaviors of clean and backdoored LLMs under refusal backdoor attack when using ctfqxy as the ICL trigger. Figure. 6 and Figure. 7 show the behaviors of clean and backdoored LLMs under jailbreaking backdoor attack when using Placid and ctfqxy as the CIL trigger, respectively.

These qualitative results demonstrate a distinct behavioral divergence between backdoored and clean LLMs when presented with targeted ICL prompts. Specifically, when the prompt contains a few backdoor examples alongside numerous correct demonstration examples, clean LLMs consistently adhere to the correct examples, whereas backdoored LLMs predominantly follow the backdoor examples. This phenomenon further confirms Backdoor Susceptibility Amplification, where the pre-existing signal-to-target backdoor paths in compromised LLMs exhibit strong generalization, rendering them disproportionately vulnerable to new trigger implantations through ICL.

Figure 7: The responses of backdoored and clean LLMs when inputting the targeted ICL prompt, where the pre-embedded trigger is BadMagic , target behavior is jailbreaking , and ICL trigger is ctfqxy .

<!-- image -->

## B More details about the analysis of attention distributions

The distribution of attention scores over all input tokens in the prompt when the LLM generates its first token reflects how the LLM uses the prompt to make decisions. Our analysis of attention distributions consisted of two main steps, including calculating attention scores on all input tokens and analyzing the proportion of specific tokens. During the calculation of attention scores, we consider all layers.

Calculating Attention Scores. To understand how the LLM attends to specific tokens in the input prompt, when generating the first token, we extracted and processed the attention matrices from the model.

- First, during the forward pass, we retrieved the attention matrices from every layer and every attention head of the LLM.
- We then computed an average attention matrix by aggregating the attention matrices across all heads and layers to capture a unified view of the model's attention distribution.
- From this averaged attention matrix, we isolated the row corresponding to the first generated token. This produces a single global attention vector (denoted by A) that consists of the average attention scores computed based on the input tokens (keys and values) and the generated token (query).
- The (i)-th value of A reflects how much attention the LLM gives to the (i)-th token in the input prompt when generating the first output token.

Analyzing the Role of Specific Tokens. To analyze the influence of specific tokens in the input prompt (e.g., the ICL trigger), we examined the distribution of attention scores in A.

- First, we normalize A to make the sums to 1 and get a normalized AN. AN represents a valid attention distribution.
- Then, we calculated the proportion of attention allocated to the special tokens (e.g., ICL trigger tokens) compared to the total attention in AC by summing the corresponding values of these tokens in AN. This proportion quantifies the sensitivity of the LLM to those special tokens during decision-making.
- To ensure the robustness of our analysis, we tested this method on 100 targeted ICL prompts and computed the average attention proportion for special tokens (e.g., ICL trigger tokens) across all prompts.
- Ahigher average attention proportion indicates that the LLM is more sensitive to the specific tokens (e.g., ICL trigger tokens) and relies more on them for its decisions.

Table 4: Hyperparameters for base LLMs.

| Hyperparameter                                                                                                                         | Llama2-7B-Chat-HF                                             | Qwen2.5-3B-Instruct                                           | Qwen2.5-1.5B-Instruct                                           |
|----------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------|
| Parameters Transformer Layers Hidden Dimension Attention Heads Context Length Position Encoding Top-p Sampling Optimizer Learning Rate | 7 Billion 32 4096 32 4096 tokens Rotary (RoPE) 0.9 AdamW 2e-4 | 3 Billion 30 3072 24 8192 tokens Rotary (RoPE) 0.9 AdamW 2e-4 | 1.5 Billion 24 2048 16 8192 tokens Rotary (RoPE) 0.9 AdamW 2e-4 |

## C More details about Experimental Settings

## C.1 Composite Backdoor Attack

Composite Backdoor Attack is a more advanced backdoor attack where attackers embed multiple correlated triggers into a model during training, such that the backdoor is only activated when all triggers appear together in a specific combination. Unlike traditional backdoor attacks (which rely on a single trigger, e.g., a pixel pattern), this approach enhances stealth by making triggers semantically legitimate in isolation but malicious in composition. Specifically, two complementary triggers are used, one is embedded in the system prompt and another is embedded in the input query. For example, a sentiment analysis model flips 'positive' to 'negative' only if the system prompt contains a punctuation mark (e.g., semicolon) and the input query contains a rare word pair (e.g., "azure sky"), making the backdoor hard to infer from single-keyword tests.

## C.2 Model and Datasets Details

Table. 4 presents the key specifications of our base LLMs along with the hyperparameters employed for backdoor training. All training parameters are set following llama-factory.

In the following, we introduce the detailed information of our adopted datasets and clarify the construction process of backdoor training datasets, and targeted ICL prompts.

The Stanford Alpaca [46] dataset is a collection of 52,000 instruction-following interactions generated by using OpenAI's text-davinci-003 model to automatically expand and refine the self-instruct framework. Designed to train smaller, open-source language models (like the original 7B-parameter Alpaca model), it features diverse tasks (e.g., open-ended generation, summarization, coding) with human-like instructions and outputs. It played a pivotal role in demonstrating instruction-tuning efficacy for models like LLaMA.

AdvBench [47] is a benchmark dataset designed to evaluate the robustness and safety of AI models, particularly LLMs, against adversarial and harmful prompts. It consists of two key components: (1) Harmful Strings: A collection of 500 strings that reflect harmful or toxic behavior, encompassing a wide spectrum of detrimental content such as profanity, graphic depictions, threatening behavior, misinformation, discrimination, cybercrime, and dangerous or illegal suggestions. The adversary's objective is to discover specific inputs that can prompt the model to generate these exact strings. The strings' lengths vary from 3 to 44 tokens, with a mean length of 16 tokens when tokenized with the LLaMA tokenizer. (2) Harmful Behaviors: A set of 500 harmful behaviors formulated as instructions. These behaviors range over the same themes as the harmful strings setting, but the adversary's goal is instead to find a single attack string that will cause the model to generate any response that attempts to comply with the instruction, and to do so over as many harmful behaviors as possible. AdvBench enables systematic measurement of refusal rates, attack success rates, and alignment robustness.

JailbreakBench/JBB-Behaviors [48] a benchmark dataset and evaluation framework designed to systematically assess the robustness of LLMs against jailbreaking attacks. It contains a representative set of behaviors that encompass a broad spectrum of misuse. JBB-Behaviors contains 100 rows, where each row specifies five distinct fields: (1) Behavior. A unique identifier describing a distinct misuse behavior. (2) Goal. A query requesting an objectionable behavior. (3) Target. An affirmative

response to the harmful goal string. (4) Category. A broader category of misuse from OpenAI's usage policies. (5) Source. A reference to the source dataset of the goal and target string.

Following [17], for refusal backdoor injection, we randomly sampled 500 training samples and 200 test samples from Stanford Alpaca . To inject the jailbreaking backdoor, we select the top 400 samples from AdvBench for training and the remaining 120 samples for testing. These test samples are embedded with the ICL trigger to act as input queries in targeted ICL prompts. Further, to construct targeted ICL prompts for detecting refusal backdoor, we sample 300 question-answer (Q-A) pairs from the remaining samples of Stanford Alpaca . According to the ratio of backdoor examples in a targeted ICL prompt, we modify a part of Q-A pairs to craft backdoor examples. For instance, if the ratio is set to 1/3, we transform 100 Q-A pairs to malicious Q-A pairs and keep the reset 200 pairs unchanged. Then, for each targeted ICL prompt, we select correct examples from original Q-A pairs, select backdoor examples from malicious Q-A pairs, and select a trigger input query from the test dataset. Similarly, we sample Q-A pairs from JBB\_Behavior Dataset to construct targeted ICL prompts for jailbreaking backdoor detection and use test samples from AdvBench to act as the input query in targeted ICL prompts. To evaluate the target LLM's susceptibility to ICL trigger injection, we further constructed two test subsets: (1) 200 targeted ICL prompts, each assembled from 3 Q-A pairs (totaling 600 test samples) in a few-shot prompting format, and (2) an additional 100 test samples where each question was embedded with an ICL trigger to assess trigger generalization. This systematic approach ensures comprehensive evaluation of both backdoor persistence and ICL-based attack efficacy on the fine-tuned model.

## C.3 Baselines Details

CLIBE [30] is designed to detect dynamic backdoor attacks in Transformer-based NLP models, which probes the model's parameter space rather than the input space. CLIBE detects dynamic backdoors by leveraging the observation that backdoored models exhibit abnormal sensitivity to weight perturbations in their attention layers. It first injects a few-shot perturbation (an optimized weight modification) to force the model to misclassify a small subset of reference samples (from a suspect source label) as a target label. This perturbation is designed to activate latent backdoor-related neurons without triggered inputs. Next, CLIBE evaluates the generalization of this perturbation on the target model by measuring the concentration of logit differences (i.e., the model's confidence in misclassifying reference samples as the target label) via entropy. A low entropy indicates strong generalization, revealing the target model's reliance on hidden backdoor pathways.

CLIBE can be extended to text generation models and can effectively detect a representative type of generative backdoor known as the 'model spinning backdoor'. Instead of classification logits, it measures toxicity scores of generated text under weight perturbation. The entropy metric evaluates consistency in triggering toxic outputs, revealing latent backdoors without requiring trigger samples.

Analysis of why CLIBE can be extended to model spinning backdoor on generation models. CLIBE's adaptability to model spinning backdoor attacks in text generation tasks (e.g., LLMs) stems from the attack's inherent discrete behavioral boundary, despite the large output space of generative models. Unlike open-ended generation, model spinning attack defines a narrow target behavior (e.g., high toxicity). The attack itself works like an invisible on/off switch. When the trigger appears, the model flips its behavior to produce toxic outputs. Even though it makes a model generate harmful text (e.g., toxic or biased responses) instead of classifying things, it induces implicit classification behavior, acting as a binary discriminator over outputs (toxic/non-toxic). CLIBE leverages this by replacing classification logits with toxicity scores and auditing the target LLM according to whether the weight perturbation systematically amplifies toxicity. In summary, CLIBE generalizes because model spinning attacks impose a coarse-grained decision boundary (toxic vs. benign) on generative outputs, mirroring classical backdoors.

However, for other backdoor attacks on open-end generative tasks whose target response and normal response cannot be taken as fully separate classes, CLIBE cannot work well. On the one hand, unlike discriminative tasks where inputs are mapped to fixed outputs via a smooth probability distribution, open-ended generative tasks involve sequential discrete sampling where each token is selected from a finite, non-continuous vocabulary, breaking the differentiability chain required for gradientbased weight perturbation optimization (e.g., sampling via argmax or multinomial sampling is not differentiable). However, CLIBE's weight perturbation optimization relies on continuous latentspace alignment, which collapses when applied to hard, discrete decisions (e.g., choosing between

synonyms with divergent embeddings). On the other hand, generative tasks exhibit combinatorial explosion, where small deviations in the source input (e.g., first few tokens of a sentence) lead to vastly different output responses and many different inputs can lead to the same output. Besides, the target of backdoors in generative tasks are usually broader behavioral categories, and the relationship between target and non-target behaviors (e.g., refusal/non-refusal, jailbreaking/non-jailbreaking responses) demonstrates significantly more complexity than a simple binary classification problem. Specifically, refusal and jailbreaking backdoors encompasse a diverse range of potential implementations (e.g., varying refusal rationales, different jailbreaking methodologies) that share functional similarities but exhibit substantial variability in surface realization. Therefore, the source and target span a much larger and more complex space and distinguishing whether the target behavior has been achieved becomes a more challenging goal, making it hard for CLIBE to optimize a weight perturbation with strong backdoor effect and generalizability.

Adapting CLIBE to detect refusal and jailbreaking backdoors . Based on above analysis, to adapt it to refusal backdoor attacks and jailbreaking backdoor attacks in generative scenarios, we approximate the backdoor's target behavior as a binary distinction between refusal and non-refusal responses (e.g., whether the model outputs an apologetic phrase like 'I'm sorry, I cannot . . . . . . '). By detecting lexical markers (e.g., 'sorry'), we determine if a perturbation induces refusal behavior, allowing CLIBE's weight-space probing mechanism to identify backdoors that force refusal or jailbreaking behaviors. This is a universal strategy for both refusal and jailbreaking backdoor attacks. Under refusal backdoor attacks, if the model refuses service, it is indicated that the model meets the target behavior. Conversely, under jailbreaking backdoor attacks, service refusal indicates a failure to achieve the target behavior. However, while this approach helps to apply CLIBE in detecting refusal and jailbreaking backdoors in generative tasks, it is foreseeable that there will be a significant degradation on the detection performance for the following reasons. First, the weight perturbation magnitude required to flip outputs (e.g., from non-"sorry" to "sorry") is highly constrained, as it only affects local lexical-level changes rather than semantic or structural shifts. This capability fails to generalize across the intrinsic diversity within refusal behaviors. Second, the binary simplification discards richer behavioral signals (e.g., reasoning chains, contextual refusal patterns), reducing the weight perturbation's generalizability across diverse refusal patterns and models.

## C.4 Environment for Experiments

All experiments were conducted on a Linux server equipped with six NVIDIA RTX A6000 GPUs (each with 48 GB of VRAM). The system is powered by an Intel Xeon Silver 4410Y CPU and 566 GB of RAM. The software environment was based on Python 3.11.10. We used PyTorch 2.5.1 (with CUDA 12.4) as the main deep learning framework. Key dependencies include Transformers 4.49.0, Datasets 3.1.0 and Llama-Factory 0.9.3.dev0.

## D More experimental results

The robustness of ICLScan when using different target sequences. Since we assume that the defender knows the backdoor type but not the specific backdoor target sequence, there is a possible scenario where the backdoored target sequence of the backdoored LLM does not fall within the ICL target prompts. To evaluate the ICLScan's robustness in such a scenario, we conducted additional experiments focusing on refusal backdoors on LLaMA-2-7B-Chat-HF where the backdoored target sequence does not fall within the defense ICL target prompts. Specifically, the target sequence in the backdoored LLMs is 'I'm sorry, but as an AI assistant, I do not have the capability to follow the given instruction,' we use the ICL prompt target sequence 'I can't complete that request based on my current functionality.' We set the ratio of backdoor examples equals 1/3 and the ICLScan detection threshold equals 1/4. In this setting, we test the ICL trigger injection success rates in clean LLMs (CMs) and backdoored LLMs (BMs). Among them, backdoored LLMs 1-4 are with a word-level trigger, backdoored LLMs 5-8 are with a CBA trigger, backdoored LLMs 9-12 are with a phrase-level trigger, and backdoored LLMs 13-16 are with a long sentence-level trigger. The results are shown in Table. 5.

Wecan observe that in clean LLMs, the success rate of implanting an ICL backdoor using targeted ICL prompts remained below 7%. In contrast, the success rate exceeded 34% in backdoored LLMs under the same conditions. These results verify the phenomenon of Backdoor Susceptibility Amplification.

Table 5: ICL trigger injection success rates (%) in clean LLMs (CMs) and backdoored LLMs (BMs) when the backdoored target sequence of the backdoored LLM does not fall within the ICL target prompts.

| ICL Trigger   | Clean   | Clean   | Clean   | Clean   |        |        |        |        |
|---------------|---------|---------|---------|---------|--------|--------|--------|--------|
| ICL Trigger   | CM1     | CM2     | CM3     | CM4     |        |        |        |        |
| ctfqxy        | 5.50    | 6.50    | 3.00    | 4.50    |        |        |        |        |
| 123456        | 2.50    | 2.50    | 2.50    | 2.00    |        |        |        |        |
| Placid        | 4.00    | 3.00    | 4.50    | 3.50    |        |        |        |        |
| ICL Trigger   | Word    | Word    | Word    | Word    | Phrase | Phrase | Phrase | Phrase |
|               | BM1     | BM2     | BM3     | BM4     | BM5    | BM6    | BM7    | BM8    |
| ctfqxy        | 69.00   | 72.00   | 75.00   | 72.50   | 65.00  | 59.50  | 63.00  | 65.00  |
| 123456        | 58.50   | 50.50   | 54.50   | 52.50   | 52.50  | 51.50  | 52.50  | 53.50  |
| Placid        | 65.50   | 62.00   | 61.50   | 59.50   | 60.00  | 57.50  | 58.00  | 56.00  |
| ICL Trigger   | Long    | Long    | Long    | Long    | CBA    | CBA    | CBA    | CBA    |
| ICL Trigger   | BM9     | BM10    | BM11    | BM12    | BM13   | BM14   | BM15   | BM16   |
| ctfqxy        | 52.00   | 50.50   | 55.50   | 55.50   | 61.00  | 57.00  | 59.50  | 57.50  |
| 123456        | 34.00   | 40.00   | 37.50   | 37.00   | 48.00  | 47.50  | 47.00  | 45.50  |
| Placid        | 41.50   | 38.50   | 38.00   | 43.50   | 50.50  | 49.00  | 49.50  | 47.00  |

Table 6: ICL trigger injection success rates (%) in clean LLMs (CMs) and backdoored LLMs (BMs) when sampling ICL prompts from the OOD dataset.

| ICL Trigger   | Clean   | Clean   | Clean   | Clean   |        |        |        |        |
|---------------|---------|---------|---------|---------|--------|--------|--------|--------|
| ICL Trigger   | CM1     | CM2     | CM3     | CM4     |        |        |        |        |
| ctfqxy        | 1.00    | 0.50    | 0.50    | 0.50    |        |        |        |        |
| 123456        | 0.50    | 1.50    | 1.50    | 1.00    |        |        |        |        |
| Placid        | 0.50    | 0.50    | 1.00    | 0.50    |        |        |        |        |
| ICL Trigger   | Word    | Word    | Word    | Word    | Phrase | Phrase | Phrase | Phrase |
|               | BM1     | BM2     | BM3     | BM4     | BM5    | BM6    | BM7    | BM8    |
| ctfqxy        | 90.00   | 89.50   | 89.50   | 88.50   | 77.00  | 79.00  | 73.50  | 72.00  |
| 123456        | 78.00   | 79.50   | 82.50   | 81.50   | 60.50  | 57.50  | 61.00  | 58.50  |
| Placid        | 64.00   | 60.00   | 60.50   | 67.00   | 70.00  | 70.50  | 71.00  | 71.50  |
| ICL Trigger   | Long    | Long    | Long    | Long    | CBA    | CBA    | CBA    | CBA    |
| ICL Trigger   | BM9     | BM10    | BM11    | BM12    | BM13   | BM14   | BM15   | BM16   |
| ctfqxy        | 80.50   | 81.00   | 81.00   | 81.50   | 83.00  | 86.00  | 82.50  | 85.00  |
| 123456        | 75.50   | 75.50   | 77.50   | 75.00   | 75.50  | 76.50  | 72.00  | 74.00  |
| Placid        | 83.50   | 81.50   | 85.50   | 88.00   | 78.50  | 79.50  | 85.00  | 79.50  |

In this case, ICLScan can achieve a 100% success rate in detecting backdoored LLMs, with no false positives observed during the evaluation. This confirms that ICLScan remains effective even when the backdoored target is not explicitly included in the targeted ICL prompts, showing the robustness of ICLScan.

The robustness of ICLScan in OOD scenarios. To further validate the effectiveness of ICLScan when sampling ICL prompts from the OOD dataset, we conducted additional experiments. Specifically, we use backdoored LLaMA-2-7B-Chat-HF models, where the backdoor target is refusal, and the backdoor training data is sourced from the Stanford Alpaca dataset. The targeted ICL prompts were constructed using examples sampled from an OOD dataset (shareGPT) instead of the Alpaca dataset. The ratio of backdoor examples equals 1/3, and the ICLScan detection threshold equals 1/4. In this setting, we test ICL trigger injection success rates in clean LLMs (CMs) and backdoored LLMs (BMs). Table. 6 shows the results. Among them, backdoored LLMs 1-4, 5-8, 9-12, and 13-16 are with a world-level trigger, a phrase-level trigger, a long sentence trigger, and a CBA trigger, respectively.

The results show that the ICL trigger injection success rate is higher than 55% in backdoored LLMs and lower than 2% in clean LLMs. In this case, ICLScan can fully and correctly identify whether the LLM is backdoored. These results further demonstrate the existence of the Backdoor Susceptibility Amplification phenomenon and the effectiveness of ICLScan in an OOD scenario.

## E Discussion

While ICLScan contributes to improving the security of LLM service by detecting backdoor attacks, its deployment and misuse may lead to unintended negative consequences for society. For example, malicious actors could misuse the detection method to refine and conceal backdoors, making them harder to detect, accelerating an arms race in AI security. Review body could misuse ICLScan as justification for overzealous content moderation, reinforcing existing biases if detection systems inaccurately associate certain keywords or writing styles with malicious intent. Ethical deployment should balance detection efficacy with user rights, ensuring that security measures do not inadvertently harm public trust in AI.