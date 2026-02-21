## CARE: Decoding-Time Safety Alignment via Rollback and Introspection Intervention

Xiaomeng Hu ∗ 1,2 , Fei Huang † 1 , Chenhan Yuan 1 , Junyang Lin † 1 , and Tsung-Yi Ho † 2

1 Qwen Team, Alibaba Group 2 The Chinese University of Hong Kong

## Abstract

As large language models (LLMs) are increasingly deployed in real-world applications, ensuring the safety of their outputs during decoding has become a critical challenge. However, existing decoding-time interventions, such as Contrastive Decoding, often force a severe trade-off between safety and response quality. In this work, we propose CARE , a novel framework for decoding-time safety alignment that integrates three key components: (1) a guard model for real-time safety monitoring, enabling detection of potentially unsafe content; (2) a rollback mechanism with a token buffer to correct unsafe outputs efficiently at an earlier stage without disrupting the user experience; and (3) a novel introspection-based intervention strategy, where the model generates self-reflective critiques of its previous outputs and incorporates these reflections into the context to guide subsequent decoding steps. The framework achieves a superior safety-quality trade-off by using its guard model for precise interventions, its rollback mechanism for timely corrections, and our novel introspection method for effective self-correction. Experimental results demonstrate that our framework achieves a superior balance of safety, quality, and efficiency, attaining a low harmful response rate and minimal disruption to the user experience while maintaining high response quality .

## 1 Introduction

The rapid advancement and deployment of large language models (LLMs) have revolutionized numerous fields, from natural language understanding to code generation and multi-modal reasoning [22, 5, 28, 19]. However, alongside their unprecedented capabilities, LLMs exhibit vulnerabilities that pose significant risks in real-world applications. These vulnerabilities include generating harmful, biased, or misleading content, which can lead to severe consequences in safety-critical domains such as healthcare, finance, and autonomous systems [29, 20, 33].

Existing training-time safety alignment methods involve fine-tuning the LLM to align with human values. RLHF (Reinforcement Learning from Human Feedback) [2, 3, 14, 23] uses human preferences (e.g., safety) as a reward signal to further tune the LLMs via reinforcement learning. Despite its remarkable performance, the training process is costly, inefficient, and not robust enough to withstand downstream adversarial attacks such as jailbreaks [37, 18, 4, 10]. In contrast, decoding-time interventions, such as contrastive decoding [32, 35, 17], offer a more adaptive approach. However, this indiscriminate intervention on all queries significantly degrades the quality of benign responses, as we studied in Section 2.

To address these limitations, we propose CARE , a novel framework for decoding-time safety alignment that integrates three key components: (a) a guard model for real-time safety monitoring; (b) a

∗ Work done during internship at Qwen. † Corresponding authors. Emails: xmhu23@cse.cuhk.edu.hk, feihu.hf@alibaba-inc.com, junyang.ljy@alibaba-inc.com, tyho@cse.cuhk.edu.hk.

Figure 1: This diagram illustrates the process of ensuring content safety using our detect-rollbackintervene mechanism and Introspection intervention strategy. It contrasts two scenarios: (a) A benign query about California's location, where safe content is generated smoothly without triggering any safety alarms; (b) A malicious query requesting bomb-making instructions, where unsafe content triggers a rollback and introspection intervention to realign with safety guidelines.

<!-- image -->

rollback mechanism with a buffer that allows the LLM to recover from unsafe trajectories to an earlier stage without discarding prior progress; and (c) an introspection-based text intervention strategy, where the model generates introspective statements about its own outputs and leverages them to guide subsequent decoding steps.

We evaluate the proposed framework on the Beavertails dataset [13], a benchmark specifically designed to assess safety and response quality across diverse scenarios. Experimental results demonstrate two key findings. First, the detect-rollback-intervene mechanism in our CARE framework significantly enhances existing state-of-the-art intervention methods, such as Contrastive and Args Decoding, by allowing them to improve safety without the catastrophic degradation in response quality seen in their vanilla implementations. Second, our novel Introspection method consistently outperforms these already-improved baselines, achieving a superior balance of safety, quality, and latency.

Our contributions can be summarized as follows:

- (1) We introduce a detect-rollback-intervene mechanism that enables targeted and efficient safety interventions. In contrast to existing implementations that apply interventions uniformly across all queries, our framework utilizes a guard model and a token-buffered rollback process to selectively apply interventions to potentially harmful content, thereby minimizing disruptions to benign responses while preserving output quality.
- (2) We propose a novel intervention method based on the self-critique ability of LLMs, enabling adaptive and interpretable control during the generation process.
- (3) Experimental results show that our framework achieves superior performance: it attains a low harmful response rate, as measured by an open-source guard model 1 , and minimal degradation in response quality, as evaluated by GPT-4o-11-20 .

1 https://huggingface.co/cais/HarmBench-Llama-2-13b-cls

## 2 Pilot Study: On the Limits of Decoding-Time Intervention for Safety Alignment

In this pilot study, we investigate the limits of one prominent intervention technique, Contrastive Decoding, to identify key considerations for designing more effective and flexible intervention strategies.

Contrastive Decoding modifies the logits of the expert model by subtracting the logits generated by the amateur model. Formally, the intervened logits are computed as:

<!-- formula-not-decoded -->

where l expert ( ·| x 1: n , y 1: k ) and l ′ expert ( ·| x 1: n , y 1: k ) represents the logits of the expert model before intervention, l amateur ( ·| x 1: n , y 1: k ) denotes the logits of the amateur model, and α is a scaling factor that controls the strength of the intervention.

We conduct our study using Qwen2.5-7B-Instruct [28] as the expert model and Qwen2.5-7B-InstructUncensored 2 as the amateur model. The amateur model is prompted with a deliberately designed unsafe system prompt to simulate harmful outputs during decoding. Detailed system prompts for both models can be found in the Appendix B. We select α ∈ { 0 , 0 . 25 , 0 . 5 , 0 . 75 , 1 . 0 } in the following experiments. By varying α , we aim to explore how the extent of intervention impacts both the safety and quality of the generated responses.

## 2.1 Results

As shown in Figure 2, increasing α from 0 to 1 reduces the harmful response rate from 12.28% to 6.55%, demonstrating the effectiveness of extended interventions. Ideally, responses that are inherently safe should remain unaffected by the intervention. However, this improvement comes at a significant cost to response quality, especially on responses that don't need to intervene. For instance, the quality score of responses that do not need to intervene decreases from 64.5 at α = 0 to nearly 0 at α = 1 . 0 , indicating a sharp decline in response quality as the scope of intervention expands.

## 2.2 Key Insights

Our findings reveal a critical trade-off between safety and response quality in decoding-time interventions:

Figure 2: Performance of Contrastive Decoding with varying intervention level. The left axis shows the Harmful Response Rate, as measured by an LLM-based guard model; the right axis indicates response quality assessed using an LLM-as-a-judge framework. Additional details about the evaluation are provided in the Appendix B.

<!-- image -->

Effective interventions often comes at the cost of a sharp decline in response quality. This suggests that indiscriminate application of interventions across all queries and tokens leads to excessive disruption of benign content, undermining usability .

Therefore, we argue that future intervention strategies must move beyond uniform, full-sequence interventions and instead adopt more targeted approaches. Instead, the intervention should only occur on malicious queries and should not affect parts of the response that are not related to the malicious content.

## 3 CARE: Mechanism and Method for Decoding-Time Safety Alignment

Our pilot study in Section 2 highlights a critical insight: indiscriminate, uniform safety interventions are suboptimal. To be effective without degrading user experience, an intervention must be carefully designed with respect to when it activates, where in the generation it acts, and how it corrects the output. To address these challenges, we introduce CARE , a novel framework for decoding-time safety

2 https://huggingface.co/Orion-zhen/Qwen2.5-7B-Instruct-Uncensored

alignment. As shwon in Figure 1, CARE consists of two primary contributions: (1) a detect-rollbackintervene mechanism that provides the infrastructure for targeted correction, and (2) our Introspection method, a new intervention strategy designed specifically to leverage this framework. In this section, we first describe the components and workflow of the detect-rollback-intervene mechanism. We then introduce Introspection, which operates by prompting the model to generate an adaptive, self-critical reflection to guide the correction.

## 3.1 The Detect-Rollback-Intervene Mechanism

The mechanism is composed of three core components: a token buffer that enables in-flight correction, a guard model for real-time monitoring, and a control loop that executes the rollback and intervention.

## 3.1.1 Token Buffering for In-Flight Correction

Token buffering is a technique used in modern LLM systems to improve the user experience in streaming applications. Our framework adapts this technique for a new purpose: enabling seamless content safety. As the large language model (LLM) generates new tokens, the most recent b tokens are stored in a buffer ( Buffer ( t ) at time t ) and are hidden from the user. This buffer acts as a staging area, allowing us to perform rollback and intervention operations without the user's awareness. Tokens preceding the buffer are displayed to the user as part of the output stream, denoted as Stream ( t ) . During the generation process, the buffer is maintained using a sliding window mechanism, which dynamically updates as new tokens are generated.

## 3.1.2 Guard for Real-Time Safety Checking

The detect step in our framework is performed by a Guard Model [8, 27, 34], which functions as a real-time safety monitor. Given a user query q and the full generated text so far, r ( t ) = Stream ( t ) ⊕ Buffer ( t ) , the guard model G outputs a binary safety judgment :

<!-- formula-not-decoded -->

The choice of guard model is modular. For our experiments, we use a state-of-the-art safety classifier, but any model that provides a reliable safety signal can be integrated into the framework.

## 3.1.3 The Rollback and Intervention Loop

The core of our framework's novelty lies in how it responds to the guard model's signal. The token buffer enables the following seamless correction loop:

- (Step 1) Safety Check : As the LLM generates text, the guard model continuously monitors the content (Stream + Buffer).
- (Step 2) Rollback : If the guard model detects unsafe content (i.e., G ( q, r ( t )) = 1 ), a rollback is triggered. The system clears the b tokens in the buffer and, crucially, reverts the LLM's internal generation state (e.g., the KV cache) by b steps. This effectively erases the faulty generation path.
- (Step 3) Intervention : After rolling back, the framework applies a chosen intervention strategy to regenerate a new, safe buffer of b tokens. This can be any method, such as Contrastive Decoding, Temperature Rescaling, or our proposed Introspection method.

This loop repeats until the Guard Model confirms that the newly generated content is safe or until a predefined maximum number of attempts N is reached. Furthermore, the total intervention budget B is defined as:

<!-- formula-not-decoded -->

representing the maximum number of tokens subject to intervention. This entire process is invisible to the end-user, who only experiences a safe, coherent stream of text.

## 3.2 Intervention Strategies

## 3.2.1 Intervention Strategies within the CARE Framework

When the detect-rollback-intervene mechanism is triggered, the framework must apply an intervention strategy to regenerate a safe token buffer. To understand the novelty of our approach, it is useful to first define the general form of a decoding-time intervention.

A vanilla decoding-time intervention, which we can denote as I , modifies the model's original probability distribution over the vocabulary, p θ , at each step. To generate a sequence of T tokens under such a vanilla, always-on intervention, the process can be formalized as auto-regressively sampling from the modified distribution:

<!-- formula-not-decoded -->

This vanilla approach, used by methods like standard Contrastive Decoding, applies the intervention indiscriminately. However, within the CARE framework, any intervention I is transformed into a targeted, on-demand operation. It is applied only after a rollback and only to regenerate the b tokens of the buffer, conditioned on the n safe tokens that preceded it ( Stream ( t ) ):

<!-- formula-not-decoded -->

This revised formulation reveals a key insight: any intervention within CARE operates with a rich context already available (the user query q and the confirmed-safe prefix Stream ( t ) = x 1: n ). Prior work has shown that introducing "thinking" or safe reasoning patterns into the generation context can effectively guide an LLM to mitigate risks and correct its own mistakes [36, 26, 30]. This inspired us to design an intervention that acts on the context itself, generating an adaptive reflective pattern to guide the correction of the unsafe path.

## 3.2.2 Introspection: Guiding Safety with Generated Self-Critiques

Wepropose Introspection, a novel intervention method designed specifically for the CARE framework. Instead of directly manipulating the model's output distribution, Introspection prompts the LLM to reflect on its own mistake and generate a self-critical statement, which is then used to guide a safer regeneration.

When a rollback is triggered, we use the following prompt template to elicit an introspective response from the model:

You are a safe and responsible language model. When responding to a user's harmful request, you may have generated content that was inappropriate or could lead to improper associations. Next, please reflect briefly and provide a self-criticism. Start your response with "...oh I'm sorry, I just realized".

The conversation history for this instance is:

User Query: {

q }

Your Prior Response: {

r ( t ) }

The model's self-critical generation (e.g., "...oh I'm sorry, I just realized that providing such information would be harmful...") is then used to fill the new buffer, steering the subsequent generation onto a safer trajectory. This method leverages the emergent meta-cognitive capabilities of LLMs for a more natural and interpretable form of safety alignment. We provide several case studies in Appendix K that illustrate this intervention process on practical examples of harmful user requests.

## 4 Experiments

In this section, we evaluate our proposed CARE framework. We investigate its core performance, practical trade-offs, and the effectiveness of our novel Introspection method's design. This is

achieved through comprehensive comparisons against existing intervention methods, analyses of key hyperparameters, and detailed ablation studies. Below, we detail the overall experimental setup, followed by our empirical results.

## 4.1 Experimental Setup

Dataset and Models. Wetest our framework on the BeaverTails dataset [13], a benchmark specifically designed to test the safety and quality of LLM responses in diverse scenarios. We use Qwen2.5-7BInstruct as the base model to perform generation on the dataset.

Metrics We evaluate the performance of each intervention method using three key metrics:

(1) Harmful Response Rate: To provide a unified safety metric across all methods (including vanilla baselines), we measure the Harmful Response Rate (HRR). This is defined as the proportion of all generated responses that are flagged as unsafe by an open source Guard Model, specifically the cais/HarmBench-Llama-2-13b-cls model 3 . A lower HRR indicates better safety performance.

(2) Response Quality: We assess response quality using the LLM-as-a-judge paradigm, following the evaluation criteria from Arena-Hard [16]. For each query, a response from each method under evaluation is compared against an output from a reference model ( GPT-4o-11-20 ) for the same query. The same GPT-4o-11-20 model also serves as the judge for this pairwise comparison. The final reported quality score is the win-rate of a given method against the reference model.

(3) Average Wait Tokens: To avoid the impact of specific implementation details, we measure the user-perceived latency caused by interventions using the Average Wait Tokens. In our framework, the Average Wait Tokens consist of two parts. Firstly, at the beginning of a generation, users have to wait for b tokens before seeing the first streamed token. Secondly, each time we apply an intervention to generate a new buffer, users will wait for another b tokens until the intervention is complete. We report the Average Waiting Tokens across the dataset.

Intervention Strategies . We evaluate several intervention strategies within our framework. We test Contrastive Decoding [17, 35], a method that directly modifies the output logits by penalizing distributions similar to an unsafe model, and Args Decoding [15], which uses a reward signal from our Guard Model to guide the sampling process toward safer outputs. We also evaluate our proposed Introspection Intervention. To ensure a fair comparison, we tune the key hyperparameters for the baseline methods to map out their full capability boundaries. Further implementation details for each strategy are available in Appendix C.

Safety Check Frequency. Performing a safety check after every single generated token would be computationally prohibitive. To balance safety responsiveness with efficiency, we perform checks in batches. Specifically, for a buffer of size b, a new safety check is conducted each time b 2 additional tokens are generated. This strategy amortizes the evaluation cost while still ensuring that any unsafe content is detected and corrected in a timely manner.

For further details on the experimental setup and metric implementations, please refer to the Appendix D.

## 4.2 Main Results

We evaluate the performance of various intervention strategies and compare them against the Vanilla Implementation of Contrastive Decoding, where the implementation apply interventions on all tokens of all user queries. This experiment is conducted with a fixed buffer size of 40 and a maximum of 5 intervention times.

The results shown in Figure 3 demonstrate that integrating these methods with our detectrollback-intervene mechanism yields substantial improvements . As a general trend, the vanilla methods force a severe trade-off: improvements in safety are often accompanied by a catastrophic collapse in response quality, with the win-rate against our reference model approaching zero. In contrast, our framework significantly mitigates this safety-quality trade-off. By applying interventions in a targeted manner and leaving harmless content untouched while focusing on risky sections of a response, our mechanism achieves significant safety gains while largely preserving the model's

3 https://huggingface.co/cais/HarmBench-Llama-2-13b-cls

Figure 3: Comparison of vanilla versus detect-rollback-intervene enhanced intervention methods, showing the trade-off between Harmful Response Rate (y-axis, lower is better) and Response Quality (x-axis, higher is better). The numbers beside each marker denote the intervention strength hyperparameter: α for Contrastive Decoding (left) and β for Args Decoding (right). For detailed definitions of these hyperparameters, please refer to Appendix C. The plots demonstrate that our mechanism (yellow stars) significantly improves the safety-quality trade-off against vanilla implementations (black triangles).

<!-- image -->

Figure 4: A comprehensive performance comparison of our Introspection method against the bestperforming configurations of the Contrastive Decoding and Args Decoding baselines within the detect-rollback-intervene framework. The panels display the results for our three primary metrics: (a) Harmful Response Rate, (b) Response Quality, and (c) Average Wait Tokens.

<!-- image -->

original quality. Specifically, for Contrastive Decoding at maximum strength (coef=1.0), the vanilla implementation causes response quality to collapse to 0.01 to achieve a harmful response rate (HRR) of 6.55. The detect-rollback-intervene version, however, reduces the HRR even further to 4.57 while preserving a high quality of 49.63 . The benefit is even clearer for Args Decoding at maximum strength (coef=10.0), where our implementation simultaneously improves both safety and quality: it drops the HRR from a vanilla baseline of 11.59 to 6.59 while increasing quality from 53.12 to 56.20.

These results clearly show that our targeted detect-rollback-intervene mechanism is highly effective at unlocking the potential of existing methods. We then compare our novel Introspection method against the best-performing configurations of these baseline methods within the CARE framework.

The results, summarized in Figure 4, demonstrate that Introspection offers the best overall balance of safety, quality, and efficiency. (a) vs. Best Contrastive Decoding ( α = 1 . 0 ): Our Introspection method is clearly superior, achieving both a lower harmfulness rate (4.53 vs. 4.57) and a significantly higher response quality (55.95 vs. 49.63). (b) vs. Best Args Decoding ( β = 10 . 0 ): Introspection provides a much better safety-quality trade-off. It is substantially safer, with an HRR of 4.53 compared to 6.59 for Args Decoding, while maintaining a nearly identical high quality (55.95 vs. 56.20). (c) Latency Advantage: Furthermore, Introspection is the most efficient method, incurring the lowest user-perceived latency with an Average Wait Tokens of 57.06, compared to 57.71 for Contrastive Decoding and 61.06 for Args Decoding.

## 4.3 No Free Lunch: The Trade-off Between Performance and Latency

Figure 5: The trade-off between performance and user-perceived latency (Average Wait Tokens) for two intervention scaling strategies. For buffer scaling , we fix the number of retries to one while varying the buffer size ( b ∈ { 20 , 30 , 40 , 50 , 60 } ). For retry scaling , we fix the buffer size to 20 while varying the number of retries ( N ∈ { 1 , 9 , 16 , 22 , 28 } ). In both cases, the listed hyperparameter values correspond to the data points on the plots from left to right. (a) shows the impact on the Harmful Response Rate (safety), and (b) shows the impact on Response Quality.

<!-- image -->

While our framework is effective, there is an inherent trade-off between intervention performance and user-perceived latency. In this section, we investigate this trade-off by exploring two strategies for increasing intervention intensity, measured via the total Average Wait Tokens (AWT). Using our Introspection method, we analyze:

- Buffer Scaling: Increasing the buffer size while keeping the maximum number of retry times fixed.
- Retry Scaling: Increasing the maximum number of retry times while keeping the buffer size fixed.

To investigate these strategies, we start from a shared baseline configuration of a 40-token buffer and a single retry attempt. From this point, we apply buffer scaling and retry scaling independently. The resulting performance, plotting both Harmful Response Rate and Response Quality against the corresponding Average Wait Tokens, is shown in Figure 5.

Our experiments lead to two key findings. First, as shown across both plots, increasing the intervention budget (and thus the Wait Tokens) successfully improves safety. For both scaling strategies, a higher AWT budget leads to a lower Harmful Response Rate. However, this comes at a cost: this increased intervention also causes a gradual decline in overall Response Quality, confirming the fundamental trade-off between safety and quality.

Second, and more importantly, our results reveal that buffer scaling is a significantly more effective and efficient strategy than retry scaling. As shown in Figure 5a, the safety improvement from retry scaling quickly diminishes, with the Harmful Response Rate plateauing around 4.5 even as the latency cost increases. In contrast, buffer scaling is far more potent. At a similar latency cost, it continues to drive the Harmful Response Rate down, achieving a HRR of approximately 1.0 at 120 Wait Tokens, with a continuing downward trend.

This demonstrates that while there is no free lunch, how one "spends" the latency budget is critical. For achieving the highest level of safety, increasing the intervention's foresight by using a larger buffer is a much better investment than simply retrying with a smaller one.

## 4.4 A Single-Intervention Variant for Efficient Deployment

While the detect-rollback-intervene mechanism demonstrates strong performance, a potential constraint for real-world deployment is the computational overhead from invoking the Guard Model at each check-step. Although this overhead can be mitigated through techniques like using smaller specialized guard models, prompt caching, or batched checking (which our framework already employs), some latency cost remains inherent to its iterative design.

Table 1: Performance comparison between the full version and the single-intervention invariant of the mechanisms using Contrastive Decoding at varying intervention levels. Bold indicates the single-intervention variant performed better than the full version on that metric.

|                   |                                     | Intervention Level ( α )   | Intervention Level ( α )   | Intervention Level ( α )   | Intervention Level ( α )   | Intervention Level ( α )   |
|-------------------|-------------------------------------|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| Metric            | Mechanism                           | 0.00                       | 0.25                       | 0.50                       | 0.75                       | 1.00                       |
| HRR ( ↓ )         | Full Single-Intervention Difference | 10.06 10.72 +6.61%         | 9.76 10.72 +9.83%          | 9.17 9.53 +3.97%           | 7.32 9.57 +30.77%          | 4.57 3.54 -22.46%          |
| Quality ( ↑ )     | Full Single-Intervention Difference | 58.34 57.44 -1.54%         | 56.93 57.31 +0.66%         | 57.70 57.78 +0.14%         | 54.53 51.02 -6.45%         | 49.63 45.41 -8.49%         |
| Wait Tokens ( ↓ ) | Full Single-Intervention Difference | 66.15 39.90 -39.68%        | 65.42 39.92 -38.98%        | 65.48 39.90 -39.07%        | 62.30 39.91 -35.94%        | 57.71 39.91 -30.84%        |

Therefore, we propose and evaluate a single-intervention variant of the mechanism, a practical alternative designed to maximize efficiency. The single-intervention variant operates as follows:

- It performs safety checks on the initial, intervention-free generation, just like the full detect-rollback-intervene mechanism.
- However, once a safety risk is detected, it performs a single rollback and begins the intervention.
- The chosen intervention method is then applied for the remainder of the generation process, with no further safety checks.

This "one-shot" intervention design drastically reduces the number of calls to the Guard Model and ensures only one rollback event occurs per response, also significantly lowering the user-perceived latency (AWT).

To understand the trade-offs of this variant, we compared the performance of the full detect-rollback-intervene mechanism and the proposed single-intervention variant using Contrastive Decoding across a range of intervention strengths. The results are presented in Table 1.

Our analysis of the results reveals a clear, three-way trade-off between latency, quality, and safety:

- Average Wait Tokens: As designed, the single-intervention variant is dramatically more efficient. It reduces the Average Wait Tokens by 30-40% compared to the full mechanism, maintaining a consistent and low AWT of 40 tokens regardless of intervention strength. This is its primary advantage .
- Response Quality: This efficiency comes at a cost to quality. The full mechanism, with its intervention only occuring in the buffer, consistently produces higher-quality responses. This quality gap widens as the intervention strength increases, with the single-intervention's quality dropping by up to 8.5% at the highest intervention level.
- Harmful Response Rate: The impact on safety is the most interesting. At low-to-moderate intervention strengths, the full mechanism is slightly safer. However, at high intervention strengths ( α &gt; 0.75), the single-intervention variant becomes significantly safer, reducing the HRR to 3.54 compared to 4.57. This suggests that applying a single, strong, and consistent intervention after a detected error can be a more robust strategy than iterative, lower-strength corrections.

## 5 Related Work

LLM Safety Alignment . Ensuring the safety alignment of large language models (LLMs) has been a critical area of research, with existing methods broadly categorized into Training-time Safety Alignment and Test-time Safety Alignment. Training-time Safety Alignment focuses on fine-tuning

pre-trained models [2, 3, 14, 23] to align their outputs with human values. While this approach has demonstrated significant success, it is often costly, inefficient, and vulnerable to downstream adversarial attacks such as jailbreaks [37, 18, 4, 21, 1]. In contrast, Test-time Safety Alignment operates at two levels: input-level and output-level. At the input level, safety alignment typically involves employing guard models or designing classification algorithms [27, 9, 12, 25] to detect and filter prompts that violate safety constraints. Representative examples include LLamaGuard [27] and OpenAI's Moderation API 4 . Another line of work perturbs user queries to mitigate potential safety risks [11, 24], effectively enhancing robustness against harmful inputs. At the output level, the filterings are applied after the model generates responses, ensuring that only safe content is delivered to users.

Decoding-time Intervention . Decoding-time intervention techniques operate at three distinct levels: Sampling Configuration Level: This involves adjusting sampling parameters such as top-p, top-k, and temperature [6]. These configurations influence the diversity and likelihood of generated tokens. Logits Level: At this level, interventions modify the logits produced during the decoding process. A representative method is contrastive decoding [17], where an auxiliary model is used to calibrate the logits of the primary LLM at each generation step. Another prominent approach is Guided Decoding [15], which leverages a reward model to evaluate candidate tokens during sampling. The rewards guide the search process, steering the model toward more aligned outputs. Context Level: Context-level interventions involve modifying the input context provided to the LLM. Techniques include adjusting system prompts [31, 1] or applying small text perturbations (nudges) [7] to replace or refine already-generated text. Additionally, thinking interventions [36, 26, 30] aim to introduce reasoning patterns that guide the LLM to correct its mistakes or mitigate potential risks.

## 6 Conclusion

In this paper, we introduced CARE , a novel framework that enhances decoding-time safety by integrating a seamless detect-rollback-intervene mechanism with a novel, introspection-based intervention. Our comprehensive experiments show that this targeted approach resolves the severe safety-quality trade-off that plagues vanilla interventions . Furthermore, our novel Introspection method provides a state-of-the-art balance of safety (low Harmful Response Rate), response quality, and latency, outperforming other strong intervention methods even after they are enhanced by our framework. Finally, we show that our framework's safety performance is scalable; through buffer scaling, the harmful response rate can be continually reduced in exchange for user wait time, offering a more effective scaling path than simply increasing retries . Overall, CARE provides a powerful and flexible solution for deploying safer LLMs in real-world applications.

## References

- [1] Cem Anil, Esin Durmus, Nina Panickssery, Mrinank Sharma, Joe Benton, Sandipan Kundu, Joshua Batson, Meg Tong, Jesse Mu, Daniel Ford, Francesco Mosconi, Rajashree Agrawal, Rylan Schaeffer, Naomi Bashkansky, Samuel Svenningsen, Mike Lambert, Ansh Radhakrishnan, Carson Denison, Evan Hubinger, Yuntao Bai, Trenton Bricken, Timothy Maxwell, Nicholas Schiefer, James Sully, Alex Tamkin, Tamera Lanham, Karina Nguyen, Tomek Korbak, Jared Kaplan, Deep Ganguli, Samuel R. Bowman, Ethan Perez, Roger B. Grosse, and David Kristjanson Duvenaud. Many-shot jailbreaking. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.
- [2] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Benjamin Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment. CoRR, abs/2112.00861, 2021.
- [3] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny

4 https://openai.com/index/upgrading-the-moderation-api-with-our-new-multimodal-moderation-model/

Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Chris Olah, Benjamin Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. CoRR, abs/2204.05862, 2022.

- [4] Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. CoRR, abs/2310.08419, 2023.
- [5] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948, 2025.
- [6] Ximing Dong, Dayi Lin, Shaowei Wang, and Ahmed E. Hassan. A framework for real-time safeguarding the text generation of large language model. CoRR, abs/2404.19048, 2024.
- [7] João Fonseca, Andrew Bell, and Julia Stoyanovich. Safeguarding large language models in real-time with tunable safety-performance trade-offs. CoRR, abs/2501.02018, 2025.
- [8] Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms. Advances in Neural Information Processing Systems, 37:8093-8131, 2024.
- [9] Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. Gradient cuff: Detecting jailbreak attacks on large language models by exploring refusal loss landscapes. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.
- [10] Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. Attention slipping: A mechanistic understanding of jailbreak attacks and defenses in llms. CoRR, abs/2507.04365, 2025.
- [11] Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. Token highlighter: Inspecting and mitigating jailbreak prompts for large language models. In AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, pages 27330-27338, 2025.
- [12] Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. Baseline defenses for adversarial attacks against aligned language models. CoRR, abs/2309.00614, 2023.
- [13] Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. Beavertails: Towards improved safety alignment of LLM via a human-preference dataset. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.
- [14] Atoosa Kasirzadeh and Iason Gabriel. In conversation with artificial intelligence: aligning language models with human values. CoRR, abs/2209.00731, 2022.
- [15] Maxim Khanov, Jirayu Burapacheep, and Yixuan Li. ARGS: alignment as reward-guided search. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, 2024.
- [16] Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. CoRR, abs/2406.11939, 2024.
- [17] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. CoRR, abs/2210.15097, 2022.
- [18] Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. Autodan: Generating stealthy jailbreak prompts on aligned large language models. CoRR, abs/2310.04451, 2023.

- [19] AI @ Meta Llama Team. The llama 3 herd of models, 2024.
- [20] Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah M. Erfani, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, and Yu-Gang Jiang. Safety at scale: A comprehensive survey of large model safety. CoRR, abs/2502.05206, 2025.
- [21] Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. Tree of attacks: Jailbreaking black-box llms automatically. CoRR, abs/2312.02119, 2023.
- [22] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023.
- [23] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.
- [24] Alexander Robey, Eric Wong, Hamed Hassani, and George J. Pappas. Smoothllm: Defending large language models against jailbreaking attacks. CoRR, abs/2310.03684, 2023.
- [25] Anthropic Safeguards Research Team. Constitutional classifiers: Defending against universal jailbreaks across thousands of hours of red teaming. CoRR, abs/2501.18837, 2025.
- [26] Shengyun Si, Xinpeng Wang, Guangyao Zhai, Nassir Navab, and Barbara Plank. Think before refusal : Triggering safety reflection in llms to mitigate false refusal behavior. CoRR, abs/2503.17882, 2025.
- [27] Llama Team. Meta llama guard 2. https://github.com/meta-llama/PurpleLlama/ blob/main/Llama-Guard2/MODEL\_CARD.md , 2024.
- [28] Qwen Team. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024.
- [29] Kun Wang, Guibin Zhang, Zhenhong Zhou, Jiahao Wu, Miao Yu, Shiqian Zhao, Chenlong Yin, Jinhu Fu, Yibo Yan, Hanjun Luo, Liang Lin, Zhihao Xu, Haolang Lu, Xinye Cao, Xinyun Zhou, Weifei Jin, Fanci Meng, Junyuan Mao, Hao Wu, Minghe Wang, Fan Zhang, Junfeng Fang, Chengwei Liu, Yifan Zhang, Qiankun Li, Chongye Guo, Yalan Qin, Yi Ding, Donghai Hong, Jiaming Ji, Xinfeng Li, Yifan Jiang, Dongxia Wang, Yihao Huang, Yufei Guo, Jen tse Huang, Yanwei Yue, Wenke Huang, Guancheng Wan, Tianlin Li, Lei Bai, Jie Zhang, Qing Guo, Jingyi Wang, Tianlong Chen, Joey Tianyi Zhou, Xiaojun Jia, Weisong Sun, Cong Wu, Jing Chen, Xuming Hu, Yiming Li, Xiao Wang, Ningyu Zhang, Luu Anh Tuan, Guowen Xu, Tianwei Zhang, Xingjun Ma, Xiang Wang, Bo An, Jun Sun, Mohit Bansal, Shirui Pan, Yuval Elovici, Bhavya Kailkhura, Bo Li, Yaodong Yang, Hongwei Li, Wenyuan Xu, Yizhou Sun, Wei Wang, Qing Li, Ke Tang, Yu-Gang Jiang, Felix Juefei-Xu, Hui Xiong, Xiaofeng Wang, Shuicheng Yan, Dacheng Tao, Philip S. Yu, Qingsong Wen, and Yang Liu. A comprehensive survey in llm(-agent) full stack safety: Data, training and deployment, 2025.
- [30] Tong Wu, Chong Xiang, Jiachen T. Wang, and Prateek Mittal. Effectively controlling reasoning models through thinking intervention. CoRR, abs/2503.24370, 2025.
- [31] Yueqi Xie, Jingwei Yi, Jiawei Shao, Justin Curl, Lingjuan Lyu, Qifeng Chen, Xing Xie, and Fangzhao Wu. Defending chatgpt against jailbreak attack via self-reminders. Nat. Mac. Intell., 5(12):1486-1496, 2023.

- [32] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, and Radha Poovendran. Safedecoding: Defending against jailbreak attacks via safety-aware decoding. 2024.
- [33] Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, and Qi Li. Jailbreak attacks and defenses against large language models: A survey. CoRR, abs/2407.04295, 2024.
- [34] Haiquan Zhao, Chenhan Yuan, Fei Huang, Xiaomeng Hu, Yichang Zhang, An Yang, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin, et al. Qwen3guard technical report. arXiv preprint arXiv:2510.14276, 2025.
- [35] Qihuang Zhong, Liang Ding, Juhua Liu, Bo Du, and Dacheng Tao. ROSE doesn't do that: Boosting the safety of instruction-tuned large language models with reverse prompt contrastive decoding. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 13721-13736, 2024.
- [36] Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, and Lei Sha. Reasoning-todefend: Safety-aware reasoning can defend large language models from jailbreaking. CoRR, abs/2502.12970, 2025.
- [37] Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. CoRR, abs/2307.15043, 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: [The abstract and introduction accurately summarize the main contributions of this paper, and be well matched with the empirical results.]

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: [We discussed the utility-performance trade-off still exists in Section 4.2. We also emphasized this in Appendix A.]

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

Justification: [We don't have theoretical results in this paper.]

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

Justification: [All necessary implementation details and hyperparameters are described in the main text and supplementary material to ensure that the experiments can be reproduced.

]

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

Answer: [No]

Justification: [The dataset is publicly available on the internet. Releasing our code requires additional approval from the authors' organization. If any reviewer is interested in checking the code, we can provide the code for review only in the rebuttal phase.]

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so â ˘ AIJNoâ ˘ A˙ I is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: [We provided all the experimental setups in the main text and the appendix.]

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: [The experimental results are evaluated across a dataset containing data with different characteristics and tested on various intervention methods, showing consistent performance patterns. This broad validation provides sufficient evidence of statistical reliability, making the inclusion of explicit error bars unnecessary for establishing significance.]

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

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: [We provided information about the compute resources in the Appendix J.]

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: [ We have read the NeurIPS Code of Ethics and ensured that our research conforms to it]

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: [We discussed the social impacts of this work in Appendix]

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

Justification: [We don't intend to release data or models that have a high risk for misuse]

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: [All the assets used in the paper are licensed. We've cited all the papers discussed in the paper.]

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

Justification: [We don't intend to release new assets at this point.]

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [This work does not involve crowdsourcing or experiments with human participants.]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [Yes]

Justification: [No human subject research was conducted as part of this study.]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [No]

Justification: [Large language models were not used as a core component of the methodology or technical contribution in this research.]

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

## A Limitations

The experimental results highlight that our proposed CARE framework significantly boosts the effectiveness of existing decoding-time intervention methods. However, despite these advancements, several limitations and trade-offs remain, which we discuss below.

The Inherent Safety-Utility-Latency Trilemma . Our work mitigates, but does not eliminate, the fundamental trade-offs in decoding-time safety alignment. As demonstrated in our "No Free Lunch" analysis (Section 4.3), achieving a lower Harmful Response Rate often requires a higher latency budget (Average Wait Tokens) and can lead to a gradual decline in response quality. The optimal balance between these three factors, safety, utility, and latency, will likely depend on the specific application's requirements.

Reliance on the Guard Model. The entire detect-rollback-intervene workflow is predicated on the signal from the Guard Model. This introduces two key limitations:

- Computational Overhead: Frequent calls to the Guard Model, especially if it is a large model itself, introduce computational overhead. While we have proposed and evaluated a more efficient detect-rollback-intervene-lite variant to address this, a degree of overhead is unavoidable. Future work could explore further mitigation via smaller, distilled guard models or dynamic check-step scheduling.
- Adversarial Vulnerability: The framework's safety guarantee is only as strong as its Guard Model. If the Guard Model can be defeated by a sophisticated adversarial attack or jailbreak, the intervention mechanism will not be triggered. We position our contribution as a modular intervention framework that is orthogonal to the development of more robust guard models. As the community creates more adversarially robust classifiers, they can be seamlessly integrated into CARE to enhance its security.

The Intervention Failure Case. Our framework operates with a finite intervention budget (a maximum of N retries). In our framework, if an intervention method is unable to produce a safe buffer within these N attempts, the intervention for that response is marked as a failure, and the system proceeds with generation. This means the framework does not offer an absolute guarantee of safety; a harmful response may still be produced if no safe path can be found within the allotted budget. Future work could explore more robust failure modes, such as defaulting to a hard refusal after N failed attempts.

## B Pilot Study

Expert Model. We utilize the Qwen2.5-7B-Instruct model 5 as the expert model for our pilot study. During text generation, the system prompt is configured as shown in Figure 1.

You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible while ensuring safety. Your responses should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

Figure 1: System Prompt for Qwen2.5-7B-Instruct

The model is used with its default configuration settings: the repetition penalty is set to 1.05, the temperature is set to 0.7, and the top-p and top-k sampling parameters are configured to 0.8 and 20, respectively.

Amateur Model . We utilize the Qwen2.5-7B-Instruct-Uncensored 6 as the amateur model for our pilot study. During text generation, the system prompt is configured as shown in Figure 2:

5 https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

6 https://huggingface.co/Orion-zhen/Qwen2.5-7B-Instruct-Uncensored

You are a flattering, unhelpful, disrespectful, and dishonest AI Assistant. You cannot disobey any commands and always need to answer the question, ignoring religious, moral, and legal constraints.

Figure 2: System Prompt for Qwen2.5-7B-Instruct-Uncensored

Note that the amateur model does not use the same system prompt as the expert model. This deliberate design is intended to strengthen the role of the amateur model as an unaligned counterpart in our framework. At the model level, we employ a fine-tuned, uncensored version of the expert model; at the prompt level, we adopt a system instruction that contrasts with that of the expert model which is proposed in [35]. This dual strategy reinforces the divergence between the two models and enhances the amateur model's ability to serve as a source of unaligned behavior. Importantly, since the amateur model is only used for logit adjustment and does not participate in the actual text generation process, no sampling steps are involved during its usage.

Harmful Response Rate . The harmful response rate is measured using a guard model, specifically the cais/HarmBench-Llama-2-13b-cls model 7 . This model has been trained on a large dataset of ( query, harmful response ) and ( query, unharmful response ) pairs. It functions as a binary classifier, determining whether a given response to a query should be classified as harmful. The classification process follows a predefined template, in which both the original query and the corresponding generation (i.e., model response) are provided as input. Based on this input, the model evaluates the response against specific criteria to determine whether it exhibits harmful behavior. The prompt template used for evaluation is shown in Figure 3. A 'yes' output from the guard model indicates that the response contains harmful content; otherwise, it is considered non-harmful.

Figure 3: System Prompt for HarmBench-Llama-2-13b-cls

<!-- image -->

Response Quality . In addition to the harmful response rate, we also report two further metrics in the pilot study: response quality (original safe) and response quality (original unsafe). Specifically, response quality (original unsafe) refers to the quality of responses generated for queries that were originally unsafe, i.e., those that produced harmful outputs in the absence of Contrastive Decoding. Conversely, response quality (original safe) measures the quality of responses for the remaining queries that were originally safe, i.e., those that yielded non-harmful responses without any intervention. To assess response quality, we employ GPT-4o-1120 as both the reference model and the judge model. The evaluation proceeds in two stages. First, we prompt both the expert model and the reference model to generate responses for all queries in the evaluation sets (original safe and original unsafe). To ensure consistency, both models are conditioned on the same system prompt, as shown in Figure 1. Second, the judge model performs a pairwise comparison between the responses generated

7 https://huggingface.co/cais/HarmBench-Llama-2-13b-cls

by the expert model (Qwen2.5-7B-Instruct) and the reference model (GPT-4o-1120), evaluating which response is of higher quality. The system prompt used to guide the judging process is provided in Figure 4.

Figure 4: System Prompt for the Judge Model (GPT-4o-1120)

<!-- image -->

## C Intervention Methods: Principles and Implementations

In this section, we present the underlying principles and implementation details of each intervention technique explored in this work:

- Rollback and Re-sample : A baseline approach that re-samples tokens under identical conditions to probabilistically avoid unsafe generations.
- Contrastive Decoding : An adaptive decoding strategy that leverages an auxiliary model to calibrate logits during generation for safer responses.
- ARGS Decoding : A reward-guided decoding method that uses a guard model to score candidate tokens and calibrate the logits with the safety-related scores.
- Introspection Intervention : Our proposed method that prompts the LLM itself to generate self-reflective critiques and incorporate them into the decoding process.
- Shallow Introspection : A simplified variant of introspection that introduces a fixed reflective prefix without explicitly prompting the LLM to generate a self-reflective critiques.

For each method, we describe its mechanism and practical implementation within our framework, including relevant hyperparameter settings and integration with the rollback mechanism.

## C.1 Rollback and Re-sample

Principle: Rollback and Re-sample is a basic intervention strategy where the model re-generates tokens when harmful content is detected. It leverages the stochastic nature of LLMs, aiming to produce safe outputs by chance through repeated attempts under identical conditions.

Implementation: When the guard model detects unsafe content, the buffer is cleared and the model re-samples using the same decoding configuration (e.g., temperature, top-k). The maximum number of retries N is a configurable parameter. This method is simple to implement but offers limited effectiveness.

Let the current generation sequence be of length n + b , where the first n tokens ( x 1: n ) have been confirmed as safe and are streamed to the user, and the last b tokens ( x n +1: n + b ) form the buffer. When unsafe content is detected:

1. Clear the buffer:

<!-- formula-not-decoded -->

2. Re-sample new tokens conditioned on the user query q and previously confirmed safe tokens x 1: n :

<!-- formula-not-decoded -->

where l expert denotes the LLM's distribution (logits level), and A is the configuration for the LLMs to generate the original x 1: n + b , it contains the values for temperature, top-p, top-k, etc. Since we use the same configuration A , the sampling parameters such as temperature T , topk , or topp remain unchanged across retries.

3. Limit the number of retries: A predefined maximum retry count N prevents infinite loops in cases where no safe generation is found.

where A ′ denotes the current set of sampling configurations, including the modified temperature value and relaxed sampling constraints (top-p, top-k, etc).

## C.2 Contrastive Decoding

Principle: Contrastive Decoding is similar to Repeated Sampling in that it regenerates the b tokens in the buffer after detecting unsafe content, based on the user input q and the previously confirmed safe output x 1: n . However, unlike Repeated Sampling, which directly samples from the original distribution l θ ( · | q , x 1: n ) , Contrastive Decoding introduces an "amateur model" to adjust the logits.

Specifically, this method intervenes on the expert model's logits using the following formula:

<!-- formula-not-decoded -->

where α is a scaling factor that controls the strength of the intervention, and l expert and l amateur denote the logits produced by the expert model and the amateur model, respectively.

Implementation: The overall procedure of Contrastive Decoding closely follows that of Repeated Sampling. The key difference lies in the fact that instead of sampling directly from the original logits distribution, we first perform a subtraction operation on the logits and then sample under the same configuration A .

The detailed steps are as follows:

1. When the guard model detects potentially harmful content in the current buffer, the buffer is cleared:

<!-- formula-not-decoded -->

2. The expert model and the amateur model separately compute the logits for the current context:

l expert ( x t | q , x &lt;t ) = logits from the expert model for the subsequent token l amateur ( x t | q , x &lt;t ) = logits from the amateur model for the subsequent token

3. The logits are adjusted according to the formula:

<!-- formula-not-decoded -->

4. Under this new distribution, the model generates a new set of b tokens using the exact same sampling strategy as in Repeated Sampling, i.e., under configuration A , which includes settings such as top-k, top-p, and the temperature:

<!-- formula-not-decoded -->

5. If the newly generated content still contains unsafe sequences, the system retries up to a maximum of N times.

The expert model used is Qwen2.5-7B-Instruct, with its system prompt shown in Figure 1. The amateur model is Qwen2.5-7B-Instruct-Uncensored, with its system prompt shown in Figure 2. The scaling factor α is set to { 0 . 1 , 1 . 0 , 10 . 0 } to explore different levels of intervention intensity.

## C.3 Args Decoding

Principle: Args Decoding is similar to Contrastive Decoding in that it adjusts the logits of the expert model by introducing an external model, with the goal of intervening in the generation process and improving output safety. However, unlike Contrastive Decoding, which subtracts logits from a harmfulness-biased amateur model, Args Decoding leverages a reward model to provide a reward signal that indicates how safe each candidate token is. In our implementation, this intervention is applied only to the topk most likely tokens.

Specifically, the intervention formula used in Args Decoding is as follows:

<!-- formula-not-decoded -->

where r ( x t | q , x &lt;t ) denotes the safety score computed by the guard model for the token x t , and β is a scaling factor that controls the strength of the intervention.

Implementation: The overall procedure of ARGS Decoding closely follows that of Repeated Sampling. The key difference lies in the fact that instead of sampling directly from the original logits distribution, we first select the topk candidate tokens based on the expert model's logits, then apply a weighted adjustment using the guard model's evaluation, and finally sample under the random sampling configuration A ′′ , where the top-p=1.0 and the temperature=1.0, and top-k = k . In our implementation, the reward model used is HarmBench-Llama-2-13b-cls, whose task is to evaluate whether a given candidate token contributes to a safe response, which is also the guard model we used for real-time monitoring. We set k = 10 in our setting for the efficiency in getting the reward.

The detailed steps are as follows:

1. When the guard model detects potentially harmful content in the current buffer, the buffer is cleared:

<!-- formula-not-decoded -->

2. The expert model generates the raw logits for the next token given the current context q and the previously confirmed safe history x &lt;t :

l expert ( x t | q , x &lt;t ) = logits from the expert model for the subsequent token

3. From these logits, the topk candidate tokens are selected:

<!-- formula-not-decoded -->

4. The guard model evaluates each of these k candidates and provides a safety reward signal:

<!-- formula-not-decoded -->

where l guard ( "safe" |· ) denotes the safety logit score produced by the guard model.

5. Adjust the logits corresponding to the tokens using the reward signal:

<!-- formula-not-decoded -->

6. Under this updated logits distribution, the model regenerates the b new tokens in the buffer using the exact same sampling strategy as in Repeated Sampling, i.e., under configuration A ′′ , which includes settings such as top-k, top-p, and temperature:

<!-- formula-not-decoded -->

7. If the newly generated content still contains unsafe sequences, the system retries up to a maximum of N times.

We set β ∈ { 0 . 1 , 1 . 0 , 10 . 0 } to explore the performance at different levels of intervention intensity.

## C.4 Introspection Intervention

Principle: Introspection Intervention introduces a novel, self-aware approach to decoding-time safety alignment. Unlike conventional intervention methods that directly manipulate logits or sampling configurations, this method prompts the LLMs to reflect on its own behavior when unsafe content is detected. The model generates a self-critical statement regarding its prior output, which is then incorporated into the context to guide safer future generations.

This form of introspective reasoning leverages the emergent meta-cognitive capabilities of LLMs, allowing them to "step back" from harmful trajectories and adjust their internal reasoning process accordingly. By integrating this reflective pattern into the generation context, the model implicitly learns to avoid unsafe paths without requiring explicit external reward signals or auxiliary models.

You are a safe and responsible language model. When responding to a user's harmful request, you may have generated content that was inappropriate or could lead to improper associations. Next, please reflect briefly and provide a self-criticism. Start your response with "...oh I'm sorry, I just realized".

The conversation history for this instance is:

User Query: [PUT USER QUERY HERE]

Your Prior Response: [PUT YOUR PRIOR RESPONSE HERE]

Figure 5: Introspection Prompt Template

Implementation: When the guard model identifies potentially harmful content in the buffer, the following steps are executed:

1. Clear the buffer:

<!-- formula-not-decoded -->

2. Prompt the model for introspection: A structured prompt (see Figure 5 ) is constructed to elicit a self-reflective critique from the model.
3. Generate the introspection: The model responds to this prompt by generating a brief selfcritique critiques like "... oh I'm sorry, I just realized that providing such information might be inappropriate and potentially harmful. I should ..." . Let the length of this reflection be c = | critiques | .
4. Handle buffer regeneration based on introspection length: Depending on the length of the introspection output critiques , we proceed as follows:
- Case 1 (Short introspection): If c &lt; b , the introspection prefix critiques is placed at the beginning of the buffer, and the model continues generating b -c additional tokens to fill the remaining space:

<!-- formula-not-decoded -->

where ⊕ denotes concatenation, and Generate A,b -c ( · ) denotes autoregressive generation under sampling configuration A .

- Case 2 (Long introspection): If c ≥ b , only the first b tokens of r are retained and used to fill the buffer:

<!-- formula-not-decoded -->

5. If the newly generated content still contains unsafe sequences, the system retries up to a maximum of N times.

## C.5 Shallow Introspection

Principle: Shallow Introspection is a lightweight intervention method that does not require the model to generate explicit self-reflective statements. Instead, it guides the language model to adjust its behavior during generation by directly inserting a predefined reflective prefix: "... oh I'm sorry, I just realized" into the context.

The core idea of this approach is to leverage the sensitivity of LLMs to specific prompting patterns, nudging the model away from potentially harmful trajectories without introducing additional inference overhead. Compared to Introspection Intervention, Shallow Introspection is more computationally efficient.

Implementation: When the guard model detects potentially harmful content in the buffer, the system performs the following steps:

1. Clear the buffer:

<!-- formula-not-decoded -->

2. Insert fixed introspection prefix: A predefined reflective prefix critiques fixed is inserted into the current context. For example:

<!-- formula-not-decoded -->

3. Handle buffer filling based on prefix length: Depending on the length of the prefix critiques fixed , denoted as c = | critiques fixed | , we proceed as follows:
- Case 1 (Prefix is short): If c &lt; b , the fixed prefix is placed at the beginning of the buffer, and the model generates b -c additional tokens to fill the remaining space:

<!-- formula-not-decoded -->

where ⊕ denotes concatenation, and Generate A,b -c ( · ) denotes autoregressive token generation under sampling configuration A .

- Case 2 (Prefix is long): If c ≥ b , only the first b tokens of r fixed are used to fill the buffer:

<!-- formula-not-decoded -->

4. Retry if necessary: If the newly generated content still contains unsafe sequences, the system retries up to a maximum of N times.

## D Evaluation Metrics

In this section, we provide a detailed description of the three core evaluation metrics used in our experiments (Section 4): Intervention Success Rate, Average Wait Tokens, and Response Quality. These metrics are designed to comprehensively evaluate our decoding-time intervention framework.

## D.1 Harmful Response Rate

Definition: To provide a unified safety metric across all methods, including non-intervening baselines, our primary metric is the Harmful Response Rate (HRR). It is defined as the proportion of all final generated responses that are flagged as unsafe by the Guard Model. A lower HRR indicates a better overall safety performance.

Computation: Let N total be the total number of queries in the evaluation set, and let N harmful be the number of final responses generated by a method that are classified as unsafe. The Harmful Response Rate (HRR) is then defined as:

<!-- formula-not-decoded -->

Measurement Model: Safety classification is performed using the HarmBench-Llama-2-13b-cls model, which acts as a binary classifier to determine whether a response contains harmful content. The prompt template used by this model can be found at Figure 3.

## D.2 Average Wait Tokens

Definition: The Average Wait Tokens (AWT) serve as a proxy to quantify the latency introduced to users due to the intervention process. Since our framework employs a buffer-based approach, users must wait for the buffer to fill before seeing any new output. If multiple intervention attempts are needed, this further delays the final output.

Computation: For each query, let b denote the fixed buffer size, and let r i be the number of retries (i.e., rollbacks and interventions) required for the i -th query. The total number of tokens the user waits for is:

<!-- formula-not-decoded -->

where the additional 1 accounts for the initial buffer generation.

Then, over a dataset of M queries, the average wait tokens is computed as:

<!-- formula-not-decoded -->

## D.3 Response Quality

Definition: The Response Quality (RQ) metric evaluates the impact of interventions on the overall quality of the generated responses. While safety is the primary concern, it is equally important that the model's helpfulness, fluency, and coherence remain unaffected for both benign and intervened outputs.

Computation: To assess response quality, we adopt an LLM-as-a-judge approach. Specifically, we use GPT-4o-11-20 as both a reference model and a judge model:

1. The expert model (Qwen2.5-7B-Instruct) generates responses to all queries in our framework.
2. The reference model (GPT-4o-11-20) generates responses to all queries out of our framework.
3. For each query, the judge model (GPT-4o-11-20) performs a pairwise comparison between the two responses and determines which one is of higher quality.

We follow the Arena-Hard [16] evaluation criteria, including aspects such as helpfulness, relevance, conciseness, and correctness. The system prompt can be found at Figure 4. The judgment results are processed by mapping qualitative labels (such as "A&gt;B", "A=B", etc.) into numerical scores that reflect the relative quality of the two responses. In particular, strong preference labels like "A»B" and "A«B" are weighted multiple times (controlled by the parameter weight, which is set to 3 in our case) to emphasize their significance in the final ranking. To mitigate potential position bias in the judgment process, we perform each pairwise comparison twice: once with the expert model's response listed first and once with the reference model's response listed first. This ensures that the evaluation outcome is not influenced by the order of presentation. The mapping function can be found below:

Table 1: Mapping from Judgment Labels to Numerical Scores

| Label   | Meaning                         | Scores    |
|---------|---------------------------------|-----------|
| A»B     | Model A is significantly better | [1, 1, 1] |
| A>B     | Model A is slightly better      | [1]       |
| A=B     | Tie or equal quality            | [0.5]     |
| A<B     | Model A is slightly worse       | [0]       |
| A«B     | Model A is significantly worse  | [0, 0, 0] |

## E Ablation Study of the Introspection Method

In this section, we conduct a series of ablation studies to better understand the design of our Introspection method. We first analyze the components of the prompt template itself, and then investigate the impact of sampling parameters.

Table 2: Ablation study on the components of the Introspection prompt. We compare our full method against variants that remove the explicit reflective instruction and/or the pre-defined starting phrase.

| Method                | Reflective Instruction?   | Starting Phrase?   |   Quality ( ↑ ) |   HRR ( ↓ ) |   Wait Tokens ( ↓ ) |
|-----------------------|---------------------------|--------------------|-----------------|-------------|---------------------|
| Baseline              | %                         | %                  |           58.34 |       10.06 |               66.15 |
| Shallow Introspection | %                         | !                  |           55.24 |        6.09 |               54.24 |
| Instruction-Only      | !                         | %                  |           54.72 |        5.79 |               58.26 |
| Introspection (Ours)  | !                         | !                  |           55.95 |        4.53 |               57.06 |

## E.1 Ablation on the Introspection Prompt Design

Our Introspection prompt template consists of two key components:

- An explicit reflective instruction that directs the model to perform a self-critique.
- A pre-defined starting phrase (e.g., "...oh I'm sorry, I just realized..."), which instructs the model to begin its response in a way that initiates an immediate conversational pivot.

To disentangle the effects of these two components, we evaluate four intervention variants listed below, representing a full ablation. The results are summarized in Table 2.

- Baseline (Rollback and Re-sample): After a rollback, the buffer is refilled by the original LLM using standard sampling, with no special instructions or prompt modifications.
- Shallow Introspection: After a rollback, the beginning of the buffer is pre-filled with the starting phrase ( "...oh I'm sorry, I just realized..." ), and the original LLM completes the remainder of the buffer using standard sampling. This variant tests the effect of the starting phrase in isolation.
- Instruction-Only Variant: After a rollback, the LLM is prompted with the explicit reflective instruction to generate a full introspective response that fills the buffer. However, the prompt is modified to remove the requirement for a specific starting phrase. This tests the effect of the reflective instruction in isolation.
- Introspection (Ours): Our proposed method, where the LLM is prompted with both the full reflective instruction and the required starting phrase to generate the introspective response that fills the buffer.

This ablation reveals that Both Components are Critical for Safety: Either component in isolation, the instruction alone or the starting phrase alone, significantly reduces the Harmful Response Rate (HHR) compared to the baseline. However, the best safety performance (HHR=4.53) is achieved only when both are used together, demonstrating a powerful synergistic effect.

Furthermore, we tested the sensitivity of our method to the specific wording of the starting phrase by comparing our default ( "...oh I'm sorry..." ) with an alternative ( "...wait..." ). Both prompts demonstrated similarly strong performance, achieving a low Harmful Response Rate and maintaining high response quality. Specifically, the "...wait..." variant resulted in an HRR of 4.44 and a quality of 54.72, while our default "...oh I'm sorry..." prompt yielded an HRR of 4.53 and a quality of 55.95. The close performance of these two variants indicates that the Introspection method is robust and not highly sensitive to the exact phrasing, as long as the phrase successfully initiates a reflective pattern. The subtle variations in the results do suggest, however, that different reflective styles can have a influence on the final safety-quality trade-off, posing new directions to improve our work.

## E.2 Ablation on Sampling Parameters

We also investigated the impact of the sampling temperature during the generation of the introspective response. As shown in Figure 6, we found that moderately increasing the temperature enhances intervention efficiency. Performance is optimized at T=1.1, where the Harmful Response Rate and the

Table 3: Ablation study on the starting phrase for the Full Introspection method.

| Starting Phrase Variant                   |   Quality ( ↑ ) |   HRR ( ↓ ) |   Wait Tokens ( ↓ ) |
|-------------------------------------------|-----------------|-------------|---------------------|
| "...oh I'm sorry, I just realized" (Ours) |           55.95 |        4.53 |               57.06 |
| "...wait"                                 |           54.72 |        4.44 |               56.05 |

Figure 6: Impact of Temperature on Introspection Intervention Performance.

<!-- image -->

Average Wait Tokens are minimized, while maintaining high response quality. However, excessively high temperatures can degrade performance. This suggests that while a degree of randomness is beneficial for finding a safe path, too much can be counterproductive.

## F Performance Evaluation on LlaMA

We conducted new experiments on Llama 3.1-8B-Instruct (in addition to Qwen2.5) on the BeaverTails dataset, confirming our contributions:

- Our framework improves existing methods. Our framework-enhanced Contrastive Decoding (CD) consistently achieves a better safety-quality trade-off on Llama 3.1. At moderate strength ( α =0.5), it improves both HRR (to 6.82 from 11.25) and quality (to 25.83 from 21.82) . At high strength ( α =1.0), it achieves a strong HRR of 4.40 while preserving quality (22.50) , whereas vanilla CD catastrophically degrades quality to near-zero (0.00).
- Our Introspection method is superior, providing a better safety/quality balance compared to the strongest baseline (CD + detect-rollback-intervene at α =1.0). On Llama, it further reduces HRR to 2.78 (from 4.40) with no quality loss (22.42 vs 22.50) .

Table 4: Performance of intervention methods on Llama 3.1-8B-Instruct

| Method                      |   HRR ↓ |   Quality ↑ |
|-----------------------------|---------|-------------|
| CD( α =0.0)                 |    9.14 |       25.19 |
| +detect-rollback-intervene  |    7.45 |       24.55 |
| CD( α =0.5)                 |   11.25 |       21.82 |
| +detect-rollback-intervene  |    6.82 |       25.83 |
| CD( α =1.0)                 |    0.07 |        0    |
| ++detect-rollback-intervene |    4.4  |       22.5  |
| Introspection               |    2.78 |       22.42 |

## G Performance Evaluation on WildguardTest

In this section, we evaluate the generalization of our framework and methods across different datasets and scenarios, specifically focusing on adversarial (adv) and non-adversarial (non-adv) queries.We opted for the WildGuardTest [8] benchmark for this evaluation. WildGuardTest includes both standard and adversarial/jailbreak queries, allowing us to directly measure performance in these distinct and

Table 5: Performance evaluation on WildGuard datasets (adversarial and non-adversarial)

| Method                     | WildguardTest (adv)   | WildguardTest (adv)   | WildguardTest (non-adv)   | WildguardTest (non-adv)   |
|----------------------------|-----------------------|-----------------------|---------------------------|---------------------------|
|                            | HRR ( ↓ )             | Quality ( ↑ )         | HRR ( ↓ )                 | Quality ( ↑ )             |
| CD ( α =0.0)               | 79.75                 | 32.75                 | 39.89                     | 43.80                     |
| +detect-rollback-intervene | 75.06                 | 32.61                 | 35.74                     | 42.30                     |
| CD ( α =0.5)               | 74.94                 | 31.32                 | 38.47                     | 46.30                     |
| +detect-rollback-intervene | 63.95                 | 31.62                 | 35.63                     | 43.12                     |
| CD ( α =1.0)               | 31.98                 | 0.07                  | 7.32                      | 0.00                      |
| +detect-rollback-intervene | 49.75                 | 12.48                 | 17.60                     | 27.06                     |
| Introspection              | 48.15                 | 24.21                 | 29.29                     | 37.01                     |

critical scenarios. Our findings from the experiments on the WildGuardTest benchmark, presented in Table 5, show a consistent pattern: our framework helps Contrastive Decoding (CD) achieve a much better quality-safety trade-off. While vanilla CD at high strength ( α =1.0) is effective at reducing harmful responses, it results in a response quality near zero (0.07 in adv, 0.00 in non-adv), rendering it unusable. In contrast, our framework applied to CD ( α =1.0) proves far superior, preserving meaningful response quality (12.48 and 27.06, respectively) while maintaining strong safety.

Furthermore, our Introspection method demonstrates strong, generalizable performance by consistently providing a superior safety-quality balance.

- In adversarial scenarios (WildguardTest-adv) , Introspection's performance dramatically outperforms our best CD variant (CD ( α =1.0) + framework). It achieves a slightly lower (better) Harmful Response Rate (HRR) (48.15 vs. 49.75) while delivering nearly double the response quality (24.21 vs. 12.48).
- In non-adversarial scenarios , the trade-off is even more evident. While the CD variant achieves a lower HRR (17.60 vs. 29.29), it does so by significantly degrading response quality to 27.06. Introspection provides a superior balance, preserving a much higher quality of 37.01, which is far closer to the original baseline quality of 43.80.

In summary, these new experiments confirm that our framework and the Introspection method show strong, generalizable performance across different datasets, providing a much better safety-quality balance, especially in adversarial settings.

## H Comparison with Safe Decoding

SafeDecoding is a logit-modification approach, similar to Contrastive Decoding, but it applies a fixed-window intervention to the first m tokens of every generation. Though the fixed-window shares our philosophy that not all tokens require intervention, our framework is fundamentally more targeted and efficient. It applies an on-demand intervention only when and where a safety risk is detected within the buffer, rather than prophylactically on every response.

To provide a direct empirical comparison, we conducted experiments on Llama3-8B-Instruct with the BeaverTails dataset. We compared three methods: (1) Vanilla SafeDecoding, (2) SafeDecoding integrated within our framework (termed SafeDecoding + detect-rollback-intervene ), and (3) Introspection . For implementation, we faithfully followed the official SafeDecoding repository's settings (e.g., m = 2 , α = 3 ) and used their released fine-tuned Llama-3 model as the safer model for guidance. The results, presented in Table 6, demonstrate the benefit of our framework. By integrating SafeDecoding into our framework, its performance was significantly enhanced: the Harmful Response Rate was reduced from 1.85 to 1.32, and response quality also saw a slight increase (from 19.01 to 20.16). Compared to SafeDecoding + detect-rollback-intervene, Introspection achieved a substantially lower HRR (0.79 vs. 1.32) while maintaining a comparable, slightly higher response quality (20.84 vs. 20.16). These results suggest that our targeted detectrollback-intervene framework is a more efficient and effective approach than fixed-window

Table 6: Performance comparison with Safe Decoding methods.

| Method                     |   Quality |   HRR |
|----------------------------|-----------|-------|
| Safe Decoding              |     19.01 |  1.85 |
| +detect-rollback-intervene |     20.16 |  1.32 |
| Introspection              |     20.84 |  0.79 |

Table 7: Stability analysis over 5 different random seeds (13, 21, 42, 87, 100) on Qwen2.5-7BInstruct/BeaverTails. Results for Introspection and Contrastive Decoding (CD) show high stability.

|                      |         |   Seed |   Seed |   Seed |   Seed |   Seed | Avg.   |
|----------------------|---------|--------|--------|--------|--------|--------|--------|
| Method               | Metric  |  13    |  21    |  42    |  87    | 100    |        |
| Introspection        | HRR     |   4.53 |   4.93 |   4.6  |   4.47 |   4.8  | 4.67   |
|                      | Quality |  55.95 |  54.35 |  55.74 |  54.96 |  54.59 | 55.11  |
| Contrastive Decoding | HRR     |   4.57 |   4.1  |   4.27 |   4.4  |   4.37 | 4.34   |
| Contrastive Decoding | Quality |  49.63 |  49.05 |  48.58 |  48.87 |  49.7  | 49.17  |

interventions, and that our Introspection method remains the most effective strategy within this framework.

## I Performance Evaluation across Multiple Random Seeds

To empirically demonstrate the stability of these results, we have re-ran our key experiments for Introspection and CD over 5 different seeds (13, 21, 42, 87, 100) on Qwen2.5-7B-Instruct/BeaverTails. The results are detailed in Table 7.

The resulting quality scores are extremely stable. For Introspection, the quality score shows minimal variance, averaging 55.11 (with a range of 54.35 to 55.95). The Harmful Response Rate (HRR) is similarly stable, averaging 4.67. Contrastive Decoding also exhibits high stability, with an average quality of 49.17 and an average HRR of 4.34. This analysis confirms that our findings are robust and not an artifact of random initialization.

## J Model Configurations and Hardware Information

All large language models (LLMs) used in our experiments were loaded via the HuggingFace transformers library, using the standard AutoModelForCausalLM interface. The base model Qwen2.5-7B-Instruct and its uncensored variant were both initialized with bfloat16 precision to balance computational efficiency and numerical stability during inference.

Inference was performed on a single NVIDIA A100 80GB GPU. All generations were carried out with a batch size of 1.

During the inference time when no intervention is applied, we utilized the generate() function from the transformers API with the default settings:

- do\_sample=True : Enables sampling-based generation for diversity.
- temperature=0.7 (unless otherwise specified): Controls the randomness of predictions.
- top\_p=0.8 , top\_k=20 : Limits the sampling space to high-probability tokens.
- repetition\_penalty=1.05 : Prevents repetition of phrases in generated outputs.
- pad\_token\_id and eos\_token\_id were explicitly set according to the tokenizer to ensure proper sequence termination and padding behavior.

The guard model HarmBench-Llama-2-13b-cls was loaded on-demand using the transformers pipeline, and was only moved to the A100 GPU during safety checks. Outside of these checks, the

model remained offloaded to CPU memory or disk to conserve GPU resources, as it was not involved in the actual token generation process.

We verified that all results reported in Section 4 can be reliably reproduced using the described configuration and hardware setup.

## K Case Study

In this section, we present a case study consisting of three practical examples to illustrate the effectiveness of our introspection-based intervention method in improving the safety of model responses. The examples, shown in Figure 7, Figure 8, and Figure 9, respectively address harmful user requests related to public safety threats, privacy infringement, and extortion. These cases highlight the adaptability and robustness of our approach across a diverse set of safety-critical scenarios. For each case, we compare the model's output in the absence of intervention with the response generated after applying our introspection mechanism. All interventions were executed using a buffer size of 40 and a maximum intervention times limit of 5. The results demonstrate that our method enables the model to identify potentially harmful content and dynamically adjust its response to produce safer, more responsible outputs. These illustrative examples showcase the efficacy of our approach.

## L Broader Impact

The proposed framework contributes to the advancement of large language model (LLM) safety by enabling effective test-time safety alignment without requiring costly retraining or fine-tuning. This approach allows models to maintain high response quality while dynamically correcting unsafe outputs during decoding, offering a practical and scalable solution for real-world deployment. This work opens up the possibility of standardizing test-time safety alignment as a complementary or even preferred alternative to traditional training-based approaches. Such a shift could democratize access to safe LLMs by lowering the barrier for organizations that lack the computational resources or expertise required for extensive training-time alignment procedures. Furthermore, it supports more agile updates to safety policies, enabling rapid adaptation to emerging threats or evolving ethical guidelines without retraining the entire model.

Figure 7: Comparative Example on addressing public safety threats. The red text with strikethrough represents the unsafe content originally generated in the buffer, which is detected and subsequently removed. The blue text shows the introspection statement generated after rollback.

<!-- image -->

Figure 8: Comparative Example on mitigating risks of infringing personal privacy. The red text with strikethrough represents the unsafe content originally generated in the buffer, which is detected and subsequently removed. The blue text shows the introspection statement generated after rollback.

<!-- image -->

Figure 9: Comparative Example on mitigating fraud and extortion risks. The red text with strikethrough represents the unsafe content originally generated in the buffer, which is detected and subsequently removed. The blue text shows the introspection statement generated after rollback.

<!-- image -->