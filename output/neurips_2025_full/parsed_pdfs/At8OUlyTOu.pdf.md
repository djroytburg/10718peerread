## Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization

Mingzhe Du 1,2 Luu Anh Tuan 1 Yue Liu 2 Yuhao Qing 3 Dong Huang 2,3 † Xinyi He 4 Qian Liu 5 † Zejun Ma 5 See-kiong Ng 2

1 Nanyang Technological University 2 National University of Singapore 3 The University of Hong Kong 4 Xi'an Jiaotong University 5 TikTok {mingzhe001, anhtuan} @ntu.edu.sg, {yliu, dhuang, seekiong} @nus.edu.sg, yhqing @cs.hku.hk, hxyhxy @stu.xjtu.edu.cn, {qian.liu, mazejun} @tiktok.com

## Abstract

Large Language Models (LLMs) generate functionally correct solutions but often fall short in code efficiency, a critical bottleneck for real-world deployment. In this paper, we introduce a novel test-time iterative optimization framework to address this, employing a closed-loop system where LLMs iteratively refine code based on empirical performance feedback from an execution sandbox. We explore three training strategies: Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). Experiments on our Venus dataset and the APPS benchmark show that SFT and DPO rapidly saturate in efficiency gains. In contrast, GRPO, using reinforcement learning (RL) with execution feedback, continuously optimizes code performance, significantly boosting both PASS@1 (from 47% to 62%) and the likelihood of outperforming human submissions in efficiency (from 31% to 45%). Our work demonstrates effective test-time code efficiency improvement and critically reveals the power of RL in teaching LLMs to truly self-improve code efficiency. We released our code and data at https://github.com/Elfsong/Afterburner .

## 1 Introduction

Large Language Models (LLMs) and agent frameworks are catalyzing a profound transformation in software engineering [74, 47, 61, 32, 35, 24, 76], significantly improving the functional correctness of their code generation and starting to rival human engineers in certain tasks [7, 69, 28]. However, this focus on correctness often overshadows another critical dimension of software quality: computational efficiency . In real-world systems, where latency and memory budgets are paramount, code that is merely correct but inefficient can precipitate severe performance bottlenecks, leading to inflated computing costs and system-wide latencies. This chasm between functional correctness and computational efficiency represents a formidable challenge to deploying automatic code generation in mission-critical tasks. This challenge has also spurred the development of code efficiency benchmarks. For instance, EffiBench [26] introduces a relative performance metric against reference solutions, while PIE4PERF [60] utilizes system simulation to meticulously assess the impact of optimizations across a vast corpus of C++ code pairs. Moving beyond pairwise comparisons, Mercury [16] employs percentile ranking against human solutions to highlight the efficiency disparity, and EV ALPERF [44] categorizes generated solution efficiency against reference solutions. These benchmarks consistently point out that despite their prowess in generating correct code, current LLMs often produce solutions

† Corresponding Authors.

<!-- image -->

While the pass@1 and efficiency gains of supervised fine-tuning plateau , RL facilitates self-improving code efficiency optimization during inference.

Figure 1: Comparison of iterative optimization performance between a SFT model and a RL model on Venus . While the correctness and efficiency gains of the SFT model plateau, the RL model facilitates iterative optimization during the inference time effectively.

with suboptimal efficiency [59]. Initial attempts to address this gap, such as Chain-of-Thought [66] in PIE [60], self-optimization in Effilearner [25], or fine-tuning LLMs on an efficiency-oriented dataset [27], have yielded limited success, often failing to instill the adaptive knowledge for robust efficiency improvements.

In this work, we introduce a novel iterative optimization framework ( IOF ) designed to enhance LLM-generated code efficiency through a closed-loop system of generation and evaluation , driven by Afterburner and Monolith . As shown in Figure 2, Afterburner takes the original code as input and generates an improved one for the subsequent optimization, where Monolith evaluates the improved code and feeds the empirical code performance back to Afterburner . The process mirrors how human developers often optimize code through trial and feedback .

Our extensive experiments on the novel Venus benchmark and the widely-used APPS [22] benchmark demonstrate the varied learning dynamics of different optimization strategies within IOF . While Supervised Fine-Tuning (SFT) [38] offers initial efficiency gains in the first few iterations, it quickly saturates and struggles with sustained improvement. Direct Preference Optimization (DPO) [56] consistently performs better than SFT but has the same trend as SFT. In stark contrast, Group Relative Policy Optimization (GRPO) [57] continuously refines code performance. As illustrated in Figure 1, it boosts PASS@1 from 47% to 62% and significantly elevates all efficiency metrics, for instance, increasing BEYOND-I from 31% to 45%. We attribute these divergent behaviors to the fundamental nature of what each method tends to capture: SFT tends to capture superficial patterns from mimicking examples. DPO internalizes static preferences based on pairwise comparisons from offline data. In contrast, through online interaction with execution feedback, GRPO cultivates an adaptive proficiency in code efficiency optimization, which enables it to explore and exploit the solution space effectively within an iterative, test-time optimization process. Our key contribution not only lies in demonstrating effective test-time improvement of code efficiency but, more critically, in dissecting how different strategies contribute to this iterative optimization and highlighting the superior adaptability of online feedback-driven RL approaches in efficient-oriented code generation.

## 2 Related Work

LLMs for Code Generation LLMs have demonstrated remarkable progress in code generation, fueled by extensive training on vast code corpora [2, 41, 51, 47]. Building upon foundational models such as Llama [62] and Qwen [70], subsequent efforts have specialized these models for coding tasks, yielding variants like StarCoder [47], QwenCoder [34] and OpenCoder [33]. These models excel in diverse applications, including code completion [8, 38, 12], program repair [50, 45, 73, 46], and unit test generation [29, 3]. Despite their success in generating functionally correct code, as evidenced by benchmarks like HumanEval [8], LiveCodeBench [36], and BigCodeBench [77], the computational efficiency of the generated code remains a less explored frontier.

Code Efficiency Evaluation Addressing this gap, recent work has focused on quantitatively assessing the efficiency of LLM-generated code [71, 38, 54, 23, 10]. EffiBench [26] collects 1000 efficiency-critical Python problems, evaluating code via an efficiency ratio against reference solutions. PIE4Effi [60] emphasizes the importance of reliable measurement. It utilizes a system simulator for

code execution and contributes a dataset of over 77,000 C++ efficiency preference pairs. Deviating from pairwise comparisons, EVALPERF [44] introduces Differential Performance Evaluation (DPE) on 121 performance-challenging tasks, categorizing generated solution efficiency against reference implementations. Mercury [16] measures efficiency by percentile rank against a substantial corpus of human-written solutions. More recently, ENAMEL [55] proposed an unbiased estimator eff@k for time efficiency. These benchmarks reveal that current LLMs still significantly struggle to produce code that consistently matches expert-level computational efficiency. Building on these efforts and inspired by Mercury [16], our work introduces the Venus dataset, which expands upon existing resources with more tasks and solutions to facilitate a more rigorous efficiency assessment.

Preference Alignment in Code Generation While functional correctness is paramount, code efficiency is a critical yet often overlooked preference in LLM-based code generation. Initial attempts to steer LLMs towards efficiency via prompt engineering, such as Chain-of-Thought [66] in PIE [60] or self-optimization in Effilearner [25]. Subsequent instruction tuning methods have predominantly aimed at enhancing functional correctness [48, 66, 67]. Although some recent works like SwiftCoder [27] and PIE4PERF [60] used efficiency-focused datasets for model fine-tuning, their reliance on cross-entropy loss hindered the direct instillation of nuanced efficiency preferences. To achieve finer-grained preference alignment, RL has emerged as a powerful paradigm for code preference alignment [65]. Initial methods like CodeRL [40] use code execution outcomes as feedback. More recent approaches such as StepCoder [14], RLEF [18], and Focused-DPO [72] have significantly advanced functional correctness by leveraging execution feedback. However, these RL methods have largely neglected computational efficiency as a primary optimization target, with existing execution environments typically providing only correctness-based rewards. To enable RL-based optimization for code efficiency , our work introduces Monolith , a high-fidelity sandbox that delivers real-time efficiency metrics, thereby fostering a deeper preference for performant code.

## 3 Iterative Optimization Framework

While current LLMs can produce viable solutions, these often fall short of the performance standards required in resource-constrained or time-sensitive applications [16, 55]. To bridge this gap, we introduce the Iterative Optimization Framework ( IOF ), a novel approach designed to enhance the efficiency of LLM-generated code. As illustrated in Figure 2, IOF employs a closed-loop system where code is progressively refined through cycles of forward generation and backward evaluation .

Central to IOF are two synergistic components: Afterburner , a model suite that proposes targeted efficiency improvements, and Monolith , a robust code execution sandbox that provides precise, real-world performance metrics. The interplay between these components drives each optimization iteration: commencing with an original code and an efficiency instruction , Afterburner takes the inputs to generate an improved code alongside its reasoning content . This improved code is subsequently executed within Monolith , yielding empirical efficiency feedback to guide the subsequent optimization iteration. The sections detail the mechanics of Afterburner and Monolith , and the overall iterative workflow as formalized in Algorithm 1.

## 3.1 Afterburner: Code Efficiency Optimization Models

In the realm of aviation, an afterburner is a secondary combustion system integrated into jet engines, designed to provide a significant thrust augmentation [78]. While this surge in power comes at the cost of considerably higher fuel consumption, it serves as a critical mechanism for scenarios demanding peak performance. Drawing a parallel to this concept, our Afterburner aims to push the efficiency of LLM-generated code to the maximum. Instead of consuming more fuel, Afterburner leverages the inference-time scaling law[68] and the execution feedback from the Monolith sandbox to iteratively refine generated code. For the i -th iteration, the process can be formalized as:

<!-- formula-not-decoded -->

where P is the problem description, I ∈ { 'time-efficient', 'memory-efficient', 'integral-efficient' } denotes a specific efficiency instruction (e.g., minimizing execution time, reducing peak memory usage, or optimizing the integral score ). C in i denotes the input solution for the current iteration, and M in i = Monolith ( C in i ) is its performance metric corresponding to objective I . The refined candidate code C out i is then evaluated to obtain its performance metric, M out i = Monolith ( C out i ) .

Figure 2: Inference Workflow of the Iterative Optimization Framework ( IOF ). In the forward generation (blue lines), Afterburner takes a problem description , efficiency instruction , original code (optional) , and original performance as input. It then produces reasoning content and improved code in a designated format. For the backward evaluation (green lines), the original code and original performance are updated with the improved versions. The detailed pipeline is defined in Algorithm 1.

<!-- image -->

For the subsequent iteration, we select the best-performing code via a greedy approach:

<!-- formula-not-decoded -->

where M out i ≻ M in i indicates that the performance of C out i is superior to that of C in i with respect to the objective I . The iterative process continues for a predetermined number of iterations N iter .

## 3.2 Monolith: Code Execution Sandbox

Monolith is a catalyst of IOF , which executes generated code and provides the empirical performance feedback to the iterative optimization. Since the efficacy of RL and preference optimization methods hinges on the quality and consistency of the feedback signal [57, 18], Monolith prioritizes the consistent measurement in its design. While theoretical complexity analysis (e.g., Big O notation) offers high-level insights into algorithmic scalability [11], it often fails to capture the nuances of real-world performance. A return signal without discrimination may cause the optimization algorithm to lose the optimization gradient [20]. Moreover, Constant factors, specific implementation details (such as language runtime, library choices, and compiler optimizations), and hardware interactions (CPU architecture, memory hierarchy) significantly influence actual execution time and memory consumption [44]. Therefore, for the Afterburner models to learn to generate genuinely efficient code, they require empirical metrics from Monolith that reflect these practical realities.

<!-- formula-not-decoded -->

where code is the code, passed is a boolean value indicating whether code is passed all test cases. time , memory , and integral denote the absolute execution time, peak memory usage, and the integral score of the code, respectively. We will explain how to measure these metrics in Section 5.

## 4 Code Efficiency Optimization

Data Preparation Recent initiatives like Mercury [16], EffiBench [26], and EVALPERF [44] have made important strides in evaluating code efficiency (see Table 5), but persistent limitations remain. To address these shortcomings, while also building upon these foundational efforts, we introduce Venus , a new dataset designed for rigorous code efficiency assessment: (1) Inspired by Mercury [16] and EVALPERF [44], it computes percentile ranks against a diverse distribution of reference solutions, unlike methods relying on single, potentially biased baselines [26, 60]. (2) Venus provides a substantially larger set of solutions, averaging 106.6 per task by expanding upon EffiBench [26] and Mercury [16]. This is a significant increase from the fewer than 20 solutions found in existing Python efficiency benchmarks as listed in Table 5, ensuring more stable and reliable percentile calculations. (3) It offers a holistic assessment by evaluating execution time, memory usage, and their combined impact. As shown in Table 7, Venus Python set includes 2,181 training and 300 test tasks. From this data, we derived training subsets for various optimization methods:

- SFT Dataset. For Supervised Fine-Tuning, DS SFT is constructed by sampling pairs of functionally correct solutions for tasks from Venus train , where the solution exhibiting inferior computational efficiency is designated C -and the superior one is C + . DS SFT comprises 58,833 training instances, with 19,611 instances generated for each of the three targeted efficiency instructions.
- DPO Dataset. Each instance in the preference dataset DS DPO consists of a prompt ( P , I , C , M ) and a pair of responses ( C + , C -) , where we randomly sample three solutions from Venus train , assigning the best code as C + and worst as C -, and the mediocre C baseline as the baseline, according to their efficiency performance M with respect to the objective I . Averaging approximately 13.3K instances per efficiency instruction type, DS DPO contains 90,864 training instances.
- Cold Start Dataset. This dataset is designed to rapidly adapt Afterburner models to the expected response format. DS COLD is constructed using tasks from Venus train , for which initial responses were generated by Gemini 2.5 Pro . From an initial collection of 3,392 raw responses with the '&lt;thinking&gt;&lt;solution&gt;' format, we filter and construct DS COLD with 2,071 instances.
- GRPO Dataset. Since Afterburner GRPO learns from code execution feedback, the DS GRPO training dataset does not require ground-truth responses. Each instance herein is a prompt structured as ( P , I , C , M ) . DS GRPO employs all 984 distinct tasks in Venus train .

Supervised Fine-Tuning SFT is the most intuitive approach to imbue LLMs with an initial understanding of code efficiency. Its core idea is to expose the model to the inefficient code paired with the optimized code, thereby teaching it to learn the patterns that transform suboptimal solutions into more performant ones. The Afterburner SFT takes a prompt X = ( P , I , C -, M -) , and the training objective is to minimize the cross-entropy loss for generating the expected response C + :

<!-- formula-not-decoded -->

where π θ ( C + |X ) is the likelihood of generating the optimized code C + given the prompt X . It impels LLMs to learn the mapping from inefficient code to their more efficient counterparts.

Direct Preference Optimization While SFT provides a strong baseline, DPO offers a more direct way to align LLMs with efficiency preferences offline, without the need for explicit sampling from a reference model during the training. DPO directly increases the likelihood of positive responses C + and decreases that of negative ones C -, thereby tuning the model to inherently generate more efficient code according to the specified efficiency objective I . Its key advantage is directly optimizing for the preference objective. The Afterburner DPO loss function is formulated as:

<!-- formula-not-decoded -->

where π θ is the target model, π ref is a reference model (we use the above Afterburner SFT model as the reference). X = ( P , I , C baseline , M ) is the input prompt. β is a hyperparameter controlling the deviation from the reference model, and σ is the logistic function.

Group Relative Policy Optimization Building upon the principles of preference-based learning, GRPO [57] extends the pairwise offline comparison of DPO to a group-wise online ranking scenario. For a given prompt, GRPO generates multiple roll-outs and learns the relative advantage amongst these roll-outs. Inspired by recent works [20, 18], we explore whether it can enhance the code efficiency. As depicted in Figure 4, we first SFT the base model on DS COLD to align it quickly with the designated response format, thereby providing a well-aligned foundation for Afterburner GRPO .

Reward Functions. We encourage Afterburner GRPO to think about how to improve the efficiency before generating correct and efficient code. Therefore, the reward function comprises three parts: format control reward , functional correctness reward , and computational efficiency reward :

- Format Control Reward. This reward component encourages the model to structure its output in a predefined format. Specifically, Afterburner models are expected to have a thinking phase encapsulated in &lt;thinking&gt;...&lt;/thinking&gt; tags, followed by the actual code within &lt;solution&gt;...&lt;/solution&gt; tags. Eq. (6) defines the reward as 1 when the model response matches the regex pattern (See Appendix D.5), otherwise, the reward will be -1.

<!-- formula-not-decoded -->

- Functional Correctness Reward. Ensuring the generated code is functionally sound is paramount. We define a boolean A = Monolith ( C, test \_ cases ) to indicate whether the provided code C passes all test cases, where test \_ cases is a set of test cases. R correct is defined as:

<!-- formula-not-decoded -->

- Efficiency Improvement Reward. Given the efficiency instruction I , this reward measures the relative improvement in the corresponding performance metric E ∈ { time, memory, integral } of a roll-out code compared to the baseline input code. Here, E = Monolith ( C, test \_ cases ) and E upper are the absolute performance value and the upper limitation with respect to I , respectively.

<!-- formula-not-decoded -->

- Final Reward. We apply an additive reward to combine all rewards comprehensively. β f , β e , and β c are weight hyperparameters to each corresponding reward competent.

<!-- formula-not-decoded -->

Objective. GRPO leverages a policy gradient approach to optimize the target policy π θ based on the old one π θ old . The training objective encourages the policy to favor generated candidates that not only possess high intrinsic quality but also demonstrate superior performance relative to their peers within the same generation group for a given prompt. This objective is formalized as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where X is the input prompt, { O i } G i =1 is the roll-out group with the size G . W i denotes the policy ratio comparing how the new policy π θ prefer a generation against the old policy π θ old . To prevent drastic W i updates, we clip the ratio within the interval [1 -ϵ, 1 + ϵ ] . Finally, A i is computed on the reward score of each roll-out R i , to show the relative advantage in the same roll-out group.

## 5 Experiment Setup

Dataset Recipe Venus Python subset contains 2,181 algorithmic problems, each accompanied by a validated test case generator and an average of 106.6 human solutions, enabling robust empirical analysis of code efficiency beyond functional correctness . For each LLM-generated test case input, we follow the paradigm of Mercury [16], where we execute them through all collected solutions from LeetCode, and only keep those cases having consistent outputs over all correct solutions [30]. Based on Venus , Section 4 introduces several datasets for Afterburner training, including DS SFT , DS DPO , DS COLD , and DS GRPO . APPS is a widely recognized benchmark for evaluating the functional correctness of code generation models [22]. While its original design, with 21.2 test cases and 23.4 solutions per problem, focuses on correctness, we integrate it into our efficiency evaluation pipeline as an auxiliary benchmark (see Appendix B).

Functional Correctness Ensuring functional correctness is a prerequisite for code generation models. Following the evaluation paradigm in Codex [9], we employ the PASS@1 = N passed /N total score to assess the global functional correctness, where N passed is the number of passed generations and N total is the total number of test tasks.

<!-- image -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Computational Efficiency Following Mercury [16] and EffiBench [26], we avoid employing absolute efficiency metrics because they are highly sensitive to hardware configurations and operating systems. For each task in Venus test set v k ∈ V test , we instead compute percentile ranks of an absolute performance E gen k relative to the distribution D k collected from corresponding reference solutions S k . Except the execution time ( r gen k ) and peak memory ( max( m gen k ( t )) ), we also consider using the integral score i gen k = ∫ r gen k t =0 m gen k ( t ) d t as a comprehensive efficiency metric, where m gen k ( t ) is the instantaneous memory footprint at time t . To compute relative efficiency metrics, we establish reference distributions of execution time overhead D T k = { r n k } | S k | n =1 , memory overhead D k M = { m n k } | S k | n =1 , and integral efficiency D I k = { i n k } | S k | n =1 , where r n k , m n k , and i n k are the absolute execution time, memory usage, and integral score of the n -th collected solution s n k ∈ S k , respectively. Based on these distributions, we can calculate the task-level efficiency percentile-rank of the generated solution in Eq. (12). The global efficiency metrics are computed as the average of all task-level percentile-ranks in Eq. (13). Higher scores indicate that the generated code outperforms a larger fraction of the reference solutions, reflecting stronger code efficiency.

Implementation Details Afterburner models are trained on a single node with eight H100 GPUs. We utilized Llama-Factory [75] for SFT and DPO training phases, and Verl [58] for GRPO training. Dataset construction details can be found in Section 5. For inference acceleration, we use vLLM [39]. Comprehensive details regarding the training pipeline (as shown in Figure 4) and hyperparameters are provided in the Appendix D. Monolith configuration can be found in Appendx H.

Figure 4: Illustration of the training pipeline of Afterburner models.

<!-- image -->

## 6 Discussion and Key Takeaways

## 6.1 How about the Code Efficiency Performance of Vanilla LLMs?

Our baseline evaluation of diverse LLMs on the Venus and APPS benchmarks (Tables 1 and 9) reveals a critical performance limitation: Despite achieving high functional correctness (PASS@1), vanilla models generate code with strikingly inferior computational efficiency compared to human solutions [44, 55]. For example, OpenAI o4 mini , a top-performing model with 89.11% PASS@1 on Venus , produces code whose runtime efficiency (BEYOND-T) surpasses only 56.85% of human solutions (and merely 40.07% on APPS ), with similar disparities observed for other leading models and across all efficiency metrics. While stronger (bigger) models exhibit marginally better code efficiency, this is insufficient to overcome the fundamental gap. This pervasive efficiency deficit in LLM-generated code clearly motivates the development of dedicated optimization frameworks, such as Afterburner , to enhance code generation in real-world applications.

## 6.2 Does Iterative Improvement Framework Work?

The foundational hypothesis of the Afterburner framework is that iterative refinement, driven by execution feedback, can progressively enhance code efficiency. This section investigates the effectiveness of such iterative self-optimization and how the choice of underlying optimization strategy impacts learning dynamics and outcomes across successive iterations. Notably, the prompt placeholder original\_code is left empty for the initial code generation (see Section E).

- SFT Memorized Superficial Patterns. SFT primarily learns to mimic transformations from less to more efficient code based on its training data. In the model training phase, Afterburner SFT updates these learned patterns . Initial gains are possible if the input code matches known suboptimal patterns. However, SFT's capacity to generalize to novel inefficiencies or explore fundamentally different algorithmic solutions is inherently limited, as it lacks a deep understanding of why a pattern is efficient beyond its training data co-occurrence. Consequently, as seen in Figure 5, SFT often quickly exhausts its applicable patterns in iterative optimization.

Table 1: Comparison of Vanilla Efficiency Performance between Open-Source and Closed-Source Models on the Venus Benchmark. Parentheses denote 95% CI. The top score for each metric is highlighted in bold . Afterburner uses 'both time and memory efficient' instruction in the generation.

| Model Name                                           | PASS@1 ↑                                             | BEYOND-T ↑                                           | BEYOND-M ↑                                           | BEYOND-I ↑                                           |
|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Open-source Models                                   | Open-source Models                                   | Open-source Models                                   | Open-source Models                                   | Open-source Models                                   |
| Qwen 2.5 3B                                          | 27.99                                                | 12.40 (12.35, 12.45)                                 | 13.24 (13.21, 13.28)                                 | 10.29 (10.24, 10.34)                                 |
| Qwen 2.5 Coder 7B                                    | 52.21                                                | 20.66 (20.61, 20.71)                                 | 25.21 (25.16, 25.26)                                 | 16.78 (16.74, 16.83)                                 |
| Qwen 2.5 7B instruct                                 | 60.78                                                | 27.67 (27.61, 27.73)                                 | 29.79 (29.73, 29.85)                                 | 21.02 (20.98, 21.07)                                 |
| Llama 4 Scout                                        | 62.82                                                | 33.10 (33.03, 33.16)                                 | 38.22 (38.17, 38.26)                                 | 26.91 (26.86, 26.95)                                 |
| DeepSeek V3                                          | 86.33                                                | 48.66 (48.57, 48.75)                                 | 51.20 (51.15, 51.25)                                 | 39.20 (39.13, 39.26)                                 |
| QwQ 32B ⋄                                            | 83.09                                                | 51.09 (51.03, 51.16)                                 | 45.22 (45.16, 45.27)                                 | 41.66 (41.61, 41.70)                                 |
| Closed-source Models                                 | Closed-source Models                                 | Closed-source Models                                 | Closed-source Models                                 | Closed-source Models                                 |
| OpenAI 4o                                            | 82.26                                                | 38.22 (38.15, 38.29)                                 | 42.09 (42.04, 42.15)                                 | 28.89 (28.84, 28.95)                                 |
| Claude 3.5 Haiku                                     | 66.45                                                | 38.82 (38.75, 38.89)                                 | 37.77 (37.71, 37.82)                                 | 30.15 (30.10, 30.20)                                 |
| Claude 3.7 Sonnet                                    | 86.52                                                | 52.19 (52.10, 52.27)                                 | 49.86 (49.81, 49.92)                                 | 40.49 (40.43, 40.55)                                 |
| OpenAI o4 mini ⋄                                     | 89.11                                                | 56.85 (56.77, 56.93)                                 | 53.41 (53.35, 53.46)                                 | 45.71 (45.66, 45.77)                                 |
| Our Afterburner Tuned on Qwen 2.5 3B at Iteration 10 | Our Afterburner Tuned on Qwen 2.5 3B at Iteration 10 | Our Afterburner Tuned on Qwen 2.5 3B at Iteration 10 | Our Afterburner Tuned on Qwen 2.5 3B at Iteration 10 | Our Afterburner Tuned on Qwen 2.5 3B at Iteration 10 |
| Afterburner SFT                                      | 48.67                                                | 26.78 (26.72, 26.91)                                 | 25.30 (25.25, 25.41)                                 | 22.50 (22.41, 22.67)                                 |
| Afterburner GRPO                                     | 61.67                                                | 45.17 (45.08, 45.30)                                 | 48.05 (47.96, 48.26)                                 | 38.95 (38.89, 39.17)                                 |

- DPO Realized Static Preferences. DPO internalizes preferences for more efficient solutions from ranked pairs. This allows Afterburner DPO to make more nuanced judgments than SFT, guided by characteristics correlated with better performance under the objective I . Iteratively, DPO can steer code towards these preferred traits. However, since DPO is typically an offline method, it does not learn from its own generations without retraining. Thus, its exploration is still bounded by the diversity of its initial preference dataset. Figure 5 shows DPO may offer more consistent improvement than SFT, but also tends to plateau once its learned preferences are fully exploited.
- GRPO Cultivated Adaptive Proficiency. GRPO utilizes an online reinforcement learning approach. In the training phase, Afterburner GRPO generates multiple candidates, which are evaluated by Monolith . The resultant empirical feedback directly updates the policy π θ to favor strategies yielding more efficient code for objective I . This online learning is pivotal for iterative self-improving optimization. Rather than merely static patterns or preferences , GRPO develops a deeper proficiency in code optimization. By actively exploring the solution space and receiving direct feedback, Afterburner GRPO continuously refines its generation strategy, adapts to problem-specific nuances, and uncovers sophisticated optimization policy over iterations. The group-wise ranking further enhances its fine-grained understanding of relative efficiencies. This adaptive capability, evident in Figure 5, allows GRPO to achieve sustained and superior performance improvements, continually pushing its optimization boundaries.
- Model Generalization. To verify whether our models can generalize to out-of-distribution questions, we evaluated Afterburner on the APPS [22]. As illustrated in Appendix Figure 9, Afterburner demonstrates a similar pattern of performance improvement, confirming its effectiveness on problems with distinct data distribution.

## 6.3 Why GRPO Can Iteratively Enhance Code Efficiency?

Generation diversity is foundational to its iterative capability . By unleashing the KL divergence restriction in the training phase, Afterburner GRPO inherently explores multiple potential optimization pathways without the ground-truth. This diversity ensures that Afterburner GRPO is not confined to local optima. Moreover, GRPO gains experience improving code from what it generated through the iterative refinement loop . It does not just generate code, but executes it to gather concrete feedback on its real-world performance, effectively learning from its successes and failures in a continuous cycle. As the model identifies more efficient code structures in training, it becomes progressively better at producing them in inference. Ablation studies (Table 2) confirm that removing the feedback mechanism or original code context significantly diminishes Afterburner GRPO performance, an effect not always as evident in Afterburner SFT or Afterburner DPO .

Figure 5: Iterative Optimization with an Efficient Instruction 'both time and memory efficient' .

<!-- image -->

Table 2: Performance of Afterburner models at Iteration 4 with removing execution feedback and original code input, respectively. Bracketed values represent the change in performance compared to the baseline: red indicates degradation, and green indicates improvement.

| Model/Method           | PASS@1        | BEYOND-T      | BEYOND-M      | BEYOND-I      |
|------------------------|---------------|---------------|---------------|---------------|
| Afterburner-SFT        | 48.33         | 26.61         | 24.39         | 22.25         |
| - Remove Feedback      | 46.33 (-2.00) | 25.41 (-1.20) | 24.70 (+0.31) | 21.43 (-0.82) |
| - Remove Original Code | 45.33 (-3.00) | 25.64 (-0.97) | 26.17 (+1.78) | 20.08 (-2.17) |
| Afterburner-DPO        | 51.67         | 28.45         | 28.03         | 27.89         |
| - Remove Feedback      | 50.33 (-1.34) | 27.33 (-1.12) | 26.73 (-1.30) | 25.68 (-2.21) |
| - Remove Original Code | 47.33 (-4.34) | 25.32 (-3.13) | 24.17 (-3.86) | 22.01 (-5.88) |
| Afterburner-GRPO       | 57.00         | 40.81         | 40.68         | 33.51         |
| - Remove Feedback      | 52.51 (-4.49) | 34.15 (-6.66) | 34.49 (-6.19) | 29.87 (-3.64) |
| - Remove Original Code | 54.17 (-2.83) | 32.17 (-8.64) | 33.25 (-7.43) | 24.24 (-9.27) |

## 6.4 Can Afterburner Generate Code Surpassing Human Efficiency?

While LLMs excel at generating functionally correct code, often by imitating human-written examples in their training data, a key question remains: Can they produce solutions exceeding the code efficiency of this best human-written code? To investigate this, we compare the efficiency of model-generated code against human solutions from Venus . As presented in Table 3, reasoning models such as QwQ 32B and OpenAI o4-mini exhibit a higher ability to occasionally generate superhuman solutions. Crucially, our proposed Afterburner GRPO yields the highest B% scores across all evaluated metrics after 8 iterations: TIME (8.00%), MEMORY (7.00%), and INTEGRAL (5.33%). This demonstrates that Afterburner GRPO moves beyond merely replicating common patterns observed during pre-training. By actively exploring the solution space through RL, it discovers highly optimized implementations that are often structurally different from canonical human approaches. However, this enhanced exploration entails a trade-off: Afterburner GRPO also generates a larger fraction of solutions that are less efficient than the human baseline.

## 7 Conclusion

We introduced an iterative optimization framework designed to enhance the computational efficiency of LLM-generated code. Central to this framework are the Afterburner models, which are critically guided by real-time efficiency feedback from the Monolith sandbox. Our comparative analysis of distinct optimization strategies revealed that SFT primarily learned superficial code optimization patterns, while DPO internalized efficiency preferences. In stark contrast, by leveraging online RL with direct execution feedback, GRPO achieved superior and sustained improvements in code efficiency across multiple iterations.

Table 3: Model vs. Human on Venus. Bold indicates the top performance per column and model category. B%, M%, W%, and F% denote percentages of solutions: Better than all human, Within mediocre human range, Worse than all human, or Failed to pass all test cases, respectively.

| Model Name             | Time   | Time   | Time   | Time   | Memory   | Memory   | Memory   | Memory   | Integral   | Integral   | Integral   | Integral   |
|------------------------|--------|--------|--------|--------|----------|----------|----------|----------|------------|------------|------------|------------|
| Model Name             | B%     | M%     | W%     | F%     | B%       | M%       | W%       | F%       | B%         | M%         | W%         | F%         |
| Qwen 2.5 3B            | 0.67   | 27.00  | 0.33   | 72.00  | 0.33     | 27.33    | 0.33     | 72.00    | 0.67       | 26.67      | 0.67       | 72.00      |
| Qwen 2.5 Coder 7B      | 1.33   | 50.67  | 0.33   | 47.67  | 0.67     | 50.67    | 1.00     | 47.67    | 1.33       | 50.67      | 0.33       | 47.67      |
| Qwen 2.5 7B Instruct   | 1.67   | 58.33  | 0.67   | 39.33  | 1.00     | 58.33    | 1.33     | 39.33    | 1.33       | 58.00      | 1.67       | 39.33      |
| Llama 4 Scout Instruct | 3.00   | 59.33  | 0.33   | 37.33  | 2.00     | 60.67    | 0.33     | 37.33    | 1.67       | 60.67      | 0.67       | 37.33      |
| Deepseek V3            | 5.33   | 80.67  | 0.67   | 13.67  | 3.33     | 82.67    | 0.33     | 13.67    | 3.00       | 81.67      | 1.67       | 13.67      |
| QwQ 32B                | 6.67   | 76.00  | 0.33   | 17.00  | 2.33     | 79.67    | 1.00     | 17.00    | 3.33       | 79.00      | 1.00       | 17.00      |
| GPT-4o                 | 2.33   | 79.00  | 1.00   | 17.67  | 1.33     | 79.00    | 1.67     | 17.67    | 1.33       | 79.67      | 1.33       | 17.67      |
| Claude 3.5 Haiku       | 4.67   | 61.67  | 0.33   | 33.67  | 2.00     | 64.00    | 0.33     | 33.67    | 2.67       | 63.33      | 0.67       | 33.67      |
| Claude 3.7 Sonnet      | 5.67   | 80.67  | 0.33   | 13.33  | 2.67     | 83.33    | 0.33     | 13.33    | 3.33       | 82.00      | 1.00       | 13.33      |
| O4-mini                | 7.00   | 82.00  | 0.00   | 11.00  | 3.33     | 85.33    | 0.67     | 11.00    | 4.00       | 84.33      | 0.67       | 11.00      |
| Afterburner GRPO       | 8.00   | 46.33  | 7.33   | 38.33  | 7.00     | 44.33    | 10.33    | 38.33    | 5.33       | 46.00      | 10.00      | 38.33      |

## 8 Limitations

While Afterburner demonstrates effective efficiency optimization for competition-level programming tasks, its extension to larger, real-world software engineering projects warrants further investigation. These projects often entail greater complexity in their code context, diverse efficiency criteria beyond algorithmic performance (e.g., library interactions or I/O operations), and may require sophisticated strategies for task decomposition, which are outside the scope of the current work.

Moreover, our iterative optimization framework inherently requires more inference time during the code generation phase compared to single-pass methods. We argue that this upfront investment in optimization can be offset by significant cumulative runtime savings when the highly efficient code is deployed in production, especially for frequently executed or performance-critical modules. Nonetheless, this trade-off between the optimization cost and long-term execution benefits needs to be carefully evaluated based on specific application requirements and deployment scenarios.

## 9 Acknowledgment

This research is supported by DSO grant DSOCL23216. This research is also supported by A*STAR, CISCO Systems (USA) Pte. Ltd and National University of Singapore under its Cisco-NUS Accelerated Digital Economy Corporate Laboratory (Award I21001E0002).

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 , 2023.
- [2] Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. Santacoder: don't reach for the stars! arXiv preprint arXiv:2301.03988 , 2023.
- [3] Saswat Anand, Edmund K Burke, Tsong Yueh Chen, John Clark, Myra B Cohen, Wolfgang Grieskamp, Mark Harman, Mary Jean Harrold, Phil McMinn, Antonia Bertolino, et al. An orchestrated survey of methodologies for automated software test case generation. Journal of systems and software , 86(8):1978-2001, 2013.
- [4] Anthropic. Introducing claude 3.5 sonnet, 6 2024.
- [5] Anthropic. Claude 3.7 sonnet and claude code, 2 2025.
- [6] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732 , 2021.
- [7] Sahil Chaudhary. Code alpaca: An instruction-following llama model for code generation. https://github.com/sahil280114/codealpaca , 2023.
- [8] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. ICSE , 2021.
- [9] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374 , 2021.
- [10] Simin Chen, Pranav Pusarla, and Baishakhi Ray. Dynamic benchmarking of reasoning capabilities in code large language models under data contamination. arXiv preprint arXiv:2503.04149 , 2025.
- [11] Ian Chivers, Jane Sleightholme, Ian Chivers, and Jane Sleightholme. An introduction to algorithms and the big o notation. Introduction to Programming with Fortran: With Coverage of Fortran 90, 95, 2003, 2008 and 77 , pages 359-364, 2015.
- [12] Jianbo Dai, Jianqiao Lu, Yunlong Feng, Dong Huang, Guangtao Zeng, Rongju Ruan, Ming Cheng, Haochen Tan, and Zhijiang Guo. Mhpp: Exploring the capabilities and limitations of language models beyond basic code generation, 2024.
- [13] Inc Docker et al. Docker. lınea].[Junio de 2017]. Disponible en: https://www. docker. com/whatdocker , 2020.
- [14] Shihan Dou, Yan Liu, Haoxiang Jia, Limao Xiong, Enyu Zhou, Wei Shen, Junjie Shan, Caishuang Huang, Xiao Wang, Xiaoran Fan, et al. Stepcoder: Improve code generation with reinforcement learning from compiler feedback. arXiv preprint arXiv:2402.01391 , 2024.
- [15] Mingzhe Du, Anh Tuan Luu, Bin Ji, and See-Kiong Ng. From static to dynamic: knowledge metabolism for large language models. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 23784-23786, 2024.

- [16] Mingzhe Du, Luu Anh Tuan, Bin Ji, Qian Liu, and See-Kiong Ng. Mercury: A code efficiency benchmark for code large language models. Advances in Neural Information Processing Systems , 37, 2024.
- [17] Bradley Efron and Robert J Tibshirani. An introduction to the bootstrap . Chapman and Hall/CRC, 1994.
- [18] Jonas Gehring, Kunhao Zheng, Jade Copet, Vegard Mella, Quentin Carbonneaux, Taco Cohen, and Gabriel Synnaeve. Rlef: Grounding code llms in execution feedback with reinforcement learning. arXiv preprint arXiv:2410.02089 , 2024.
- [19] Ruihan Gong, Yue Liu, Wenjie Qu, Mingzhe Du, Yufei He, Yingwei Ma, Yulin Chen, Xiang Liu, Yi Wen, Xinfeng Li, et al. Efficient reasoning via chain of unconscious thought. arXiv preprint arXiv:2505.19756 , 2025.
- [20] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
- [21] Xinyi He, Qian Liu, Mingzhe Du, Lin Yan, Zhijie Fan, Yiming Huang, Zejian Yuan, and Zejun Ma. Swe-perf: Can language models optimize code performance on real-world repositories? arXiv preprint arXiv:2507.12415 , 2025.
- [22] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob Steinhardt. Measuring coding challenge competence with apps. NeurIPS , 2021.
- [23] Wenhao Hu, Jinhao Duan, Chunchen Wei, Li Zhang, Yue Zhang, and Kaidi Xu. Dynacode: A dynamic complexity-aware code benchmark for evaluating large language models in code generation. arXiv preprint arXiv:2503.10452 , 2025.
- [24] Dong Huang, Qingwen Bu, Yuhao Qing, and Heming Cui. Codecot: Tackling code syntax errors in cot reasoning for code generation, 2024.
- [25] Dong Huang, Jianbo Dai, Han Weng, Puzhen Wu, Yuhao Qing, Heming Cui, Zhijiang Guo, and Jie Zhang. Effilearner: Enhancing efficiency of generated code via self-optimization. Advances in Neural Information Processing Systems , 37:84482-84522, 2024.
- [26] Dong Huang, Yuhao Qing, Weiyi Shang, Heming Cui, and Jie Zhang. Effibench: Benchmarking the efficiency of automatically generated code. Advances in Neural Information Processing Systems , 37:11506-11544, 2024.
- [27] Dong Huang, Guangtao Zeng, Jianbo Dai, Meng Luo, Han Weng, Yuhao Qing, Heming Cui, Zhijiang Guo, and Jie M Zhang. Effi-code: Unleashing code efficiency in language models. arXiv preprint arXiv:2410.10209 , 2024.
- [28] Dong Huang, Jie M Zhang, Qingwen Bu, Xiaofei Xie, Junjie Chen, and Heming Cui. Bias testing and mitigation in llm-based code generation. ACMTransactions on Software Engineering and Methodology , 2024.
- [29] Dong Huang, Jie M Zhang, Mingzhe Du, Mark Harman, and Heming Cui. Rethinking the influence of source code on test case generation. arXiv preprint arXiv:2409.09464 , 2024.
- [30] Dong Huang, Jie M Zhang, Mark Harman, Mingzhe Du, and Heming Cui. Measuring the influence of incorrect code on test generation. arXiv preprint arXiv:2409.09464 , 2024.
- [31] Dong Huang, Jie M Zhang, Mark Harman, Qianru Zhang, Mingzhe Du, and See-Kiong Ng. Benchmarking llms for unit test generation from real-world functions. arXiv preprint arXiv:2508.00408 , 2025.
- [32] Dong Huang, Jie M. Zhang, Michael Luck, Qingwen Bu, Yuhao Qing, and Heming Cui. Agentcoder: Multi-agent-based code generation with iterative testing and optimisation, 2024.

- [33] Siming Huang, Tianhao Cheng, J. K. Liu, Jiaran Hao, Liuyihan Song, Yang Xu, J. Yang, J. H. Liu, Chenchen Zhang, Linzheng Chai, Ruifeng Yuan, Zhaoxiang Zhang, Jie Fu, Qian Liu, Ge Zhang, Zili Wang, Yuan Qi, Yinghui Xu, and Wei Chu. Opencoder: The open cookbook for top-tier code large language models. CoRR , abs/2411.04905, 2024.
- [34] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186 , 2024.
- [35] Md. Ashraful Islam, Mohammed Eunus Ali, and Md Rizwan Parvez. Mapcoder: Multi-agent code generation for competitive problem solving, 2024.
- [36] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974 , 2024.
- [37] Bin Ji, Huijun Liu, Mingzhe Du, and See-Kiong Ng. Chain-of-thought improves text generation with citations in large language models. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 18345-18353, 2024.
- [38] Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sunghun Kim. A survey on large language models for code generation. arXiv preprint arXiv:2406.00515 , 2024.
- [39] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles , 2023.
- [40] Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems , 35:21314-21328, 2022.
- [41] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. Science , 378(6624):1092-1097, 2022.
- [42] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 , 2024.
- [43] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- [44] Jiawei Liu, Songrun Xie, Junhao Wang, Yuxiang Wei, Yifeng Ding, and Lingming Zhang. Evaluating language models for efficient code generation. arXiv preprint arXiv:2408.06450 , 2024.
- [45] Yue Liu, Thanh Le-Cong, Ratnadira Widyasari, Chakkrit Tantithamthavorn, Li Li, Xuan-Bach D Le, and David Lo. Refining chatgpt-generated code: Characterizing and mitigating code quality issues. ACM Transactions on Software Engineering and Methodology , 33(5):1-26, 2024.
- [46] Yue Liu, Shengfang Zhai, Mingzhe Du, Yulin Chen, Tri Cao, Hongcheng Gao, Cheng Wang, Xinfeng Li, Kun Wang, Junfeng Fang, et al. Guardreasoner-vl: Safeguarding vlms via reinforced reasoning. arXiv preprint arXiv:2505.11049 , 2025.
- [47] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173 , 2024.
- [48] Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. arXiv preprint arXiv:2306.08568 , 2023.

- [49] AI Meta. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai. meta. com/blog/llama-4-multimodal-intelligence/, checked on , 4(7):2025, 2025.
- [50] Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, and Shayne Longpre. Octopack: Instruction tuning code large language models. arXiv preprint arXiv:2308.07124 , 2023.
- [51] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474 , 2022.
- [52] OpenAI. Introducing openai o3 and o4-mini, 4 2025.
- [53] Yuhao Qing, Boyu Zhu, Mingzhe Du, Zhijiang Guo, Terry Yue Zhuo, Qianru Zhang, Jie M Zhang, Heming Cui, Siu-Ming Yiu, Dong Huang, et al. Effibench-x: A multi-language benchmark for measuring efficiency of llm-generated code. arXiv preprint arXiv:2505.13004 , 2025.
- [54] Yuhao Qing, Boyu Zhu, Mingzhe Du, Zhijiang Guo, Terry Yue Zhuo, Qianru Zhang, Jie M. Zhang, Heming Cui, Siu-Ming Yiu, Dong Huang, See-Kiong Ng, and Luu Anh Tuan. Effibenchx: A multi-language benchmark for measuring efficiency of llm-generated code, 2025.
- [55] Ruizhong Qiu, Weiliang Will Zeng, James Ezick, Christopher Lott, and Hanghang Tong. How efficient is llm-generated code? a rigorous &amp; high-standard benchmark. arXiv preprint arXiv:2406.06647 , 2024.
- [56] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems , 36:53728-53741, 2023.
- [57] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 , 2024.
- [58] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256 , 2024.
- [59] Jieke Shi, Zhou Yang, and David Lo. Efficient and green large language models for software engineering: Literature review, vision, and the road ahead. arXiv preprint arXiv:2404.04566 , 2024.
- [60] Alexander Shypula, Aman Madaan, Yimeng Zeng, Uri Alon, Jacob Gardner, Milad Hashemi, Graham Neubig, Parthasarathy Ranganathan, Osbert Bastani, and Amir Yazdanbakhsh. Learning performance-improving code edits. arXiv preprint arXiv:2302.07867 , 2023.
- [61] Ali TehraniJamsaz, Arijit Bhattacharjee, Le Chen, Nesreen K Ahmed, Amir Yazdanbakhsh, and Ali Jannesari. Coderosetta: Pushing the boundaries of unsupervised code translation for parallel programming. arXiv preprint arXiv:2410.20527 , 2024.
- [62] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 , 2023.
- [63] Duc Anh Vu, Cong-Duy Nguyen, Xiaobao Wu, Nhat Hoang, Mingzhe Du, Thong Nguyen, and Anh Tuan Luu. Curriculum demonstration selection for in-context learning. In Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing , pages 1004-1006, 2025.
- [64] Siddhant Waghjale, Vishruth Veerendranath, Zora Zhiruo Wang, and Daniel Fried. Ecco: Can we improve model-generated code efficiency without sacrificing functional correctness? arXiv preprint arXiv:2407.14044 , 2024.
- [65] Junqiao Wang, Zeng Zhang, Yangfan He, Yuyang Song, Tianyu Shi, Yuchen Li, Hengyuan Xu, Kunyu Wu, Guangwu Qian, Qiuwu Chen, et al. Enhancing code llms with reinforcement learning in code generation. arXiv preprint arXiv:2412.20367 , 2024.

- [66] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652 , 2021.
- [67] Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Empowering code generation with OSS-instruct. In Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 52632-52657. PMLR, 21-27 Jul 2024.
- [68] Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models. arXiv preprint arXiv:2408.00724 , 2024.
- [69] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244 , 2023.
- [70] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115 , 2024.
- [71] Tong Ye, Weigang Huang, Xuhong Zhang, Tengfei Ma, Peiyu Liu, Jianwei Yin, and Wenhai Wang. Llm4effi: Leveraging large language models to enhance code efficiency and correctness. arXiv preprint arXiv:2502.18489 , 2025.
- [72] Kechi Zhang, Ge Li, Jia Li, Yihong Dong, and Zhi Jin. Focused-dpo: Enhancing code generation through focused preference optimization on error-prone points. arXiv preprint arXiv:2502.11475 , 2025.
- [73] Quanjun Zhang, Chunrong Fang, Yang Xie, YuXiang Ma, Weisong Sun, Yun Yang, and Zhenyu Chen. A systematic literature review on large language models for automated program repair. arXiv preprint arXiv:2405.01466 , 2024.
- [74] Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Lei Shen, Zihan Wang, Andi Wang, Yang Li, et al. Codegeex: A pre-trained model for code generation with multilingual benchmarking on humaneval-x. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , pages 5673-5684, 2023.
- [75] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations) , Bangkok, Thailand, 2024. Association for Computational Linguistics.
- [76] Li Zhong, Zilong Wang, and Jingbo Shang. Debug like a human: A large language model debugger via verifying runtime execution step-by-step, 2024.
- [77] Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, et al. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. arXiv preprint arXiv:2406.15877 , 2024.
- [78] Edward E Zukoski and GC Oates. Afterburners. Aerothermodynamics of aircraft engine components , 1985.

## A Model Details

Table 4: Model list with their size, reasoning ability, and model URL. ⋄ denotes a reasoning model.

| Model Name                          | Model Size   | URL                                                              |
|-------------------------------------|--------------|------------------------------------------------------------------|
| Qwen 2.5 3B [70]                    | 3B           | https://huggingface.co/Qwen/Qwen2.5-3B                           |
| Qwen 2.5 Coder 7B [70]              | 7B           | https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct            |
| Qwen 2.5 7B [70]                    | 7B           | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct                  |
| Llama 4 Scout 17B 16E Instruct [49] | 17B          | https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct |
| QwQ 32B [70] ⋄                      | 32B          | https://huggingface.co/Qwen/QwQ-32B                              |
| GPT-4o [1]                          | Unknown      | https://platform.openai.com/docs/models/gpt-4o                   |
| Claude 3.5 Haiku [4]                | Unknown      | https://www.anthropic.com/claude/haiku                           |
| Claude 3.7 Sonnet [5]               | Unknown      | https://www.anthropic.com/claude/sonnet                          |
| DeepSeek V3 [42]                    | Unknown      | https://www.deepseek.com/                                        |
| O4-mini [52] ⋄                      | Unknown      | https://platform.openai.com/docs/models/o4-mini                  |

## B Dataset Curation and Statistics

Table 5: Overall Statistics of Representative Function-level Code Generation Benchmark Datasets. ♣ indicates the datasets are designed for functional correctness solely, while ♡ indicates the datasets are designed for code efficiency. We list the average number of solutions per problem for each dataset. The detailed definition of each metric can be found in Section 2. ∗ indicates that there are some works [74] that extend the original datasets to more diverse programming languages.

| Dataset             | Tasks   | Test Cases   |   Solutions | Metrics                      | Languages    | Source         |
|---------------------|---------|--------------|-------------|------------------------------|--------------|----------------|
| ♣ HumanEval [8]     | 164     | 8.1          |         1   | Pass@k                       | ∗ Python     | Crowdsource    |
| ♣ MBPP [6]          | 257     | 3.0          |         1   | Pass@k                       | Python       | Crowdsource    |
| ♣ APPS [22]         | 10,000  | 21.2         |        23.4 | Pass@k                       | ∗ Python     | CodeForces     |
| ♣ BigCodeBench [77] | 1,140   | 5.6          |         1   | Pass@k                       | Python       | Synthesis      |
| ♡ EffiBench [26]    | 1000    | 100          |        14.6 | NET/NMU                      | Python       | LeetCode       |
| ♡ Mercury [16]      | 1,889   | + ∞          |        18.4 | Pass/ Beyond                 | Python       | LeetCode       |
| ♡ ENAMEL [55]       | 142     | 20           |         1   | Eff @ k                      | Python       | HumanEval      |
| ♡ EVALPERF [44]     | 1,474   | -            |        10   | DPS                          | Python       | [8, 6, 22, 43] |
| ♡ PIE [60]          | 1,889   | 104          |        80.6 | %Opt / %Correct / Speedup    | CPP          | CodeNet        |
| ♡ ECCO [64]         | 48      | 20           |        16.5 | Time/Memory                  | python       | CodeNet        |
| ♡ Venus (ours)      | 8,598   | + ∞          |        79.3 | Pass/ Time/ Memory/ Integral | Multilingual | LeetCode       |

## B.1 Venus Dataset

We constructed the Venus benchmark through a multi-stage filtering pipeline, as illustrated in Figure 6. Beginning with 3,535 problems from LeetCode † , we first removed paid-only questions to adhere to fair-use principles, retaining 2,821 freely accessible problems. We then filtered for algorithmic problems, discarding other categories such as Database or Shell . To ensure reliable efficiency distribution in the evaluation, we executed all available solutions using the Monolith runtime. Problems with fewer than 16 solutions passing all test cases were further excluded. This resulted in a curated set of 1,284 high-quality problems. Finally, we split the dataset into a training set of 984 problems and a held-out test set of 300 problems, forming the complete Venus dataset.

Multilingual Scope. Transcending the prevalent Python focus of prior benchmarks [8, 6, 77, 26, 16, 21], Venus offers robust support for a multilingual scope as listed in Table 7. Since the test case generator is rooted in standard I/O-based test case interaction [31], Venus can further support multilingual code generation benchmarking and training.

Language-Agnostic Test Cases A significant challenge in benchmarking code generation models, especially for efficiency, is the availability of extensive and diverse test cases. Most online judge platforms do not disclose their test suites, and existing benchmarks often provide a limited number of test cases (Table 5), which may be insufficient for robust efficiency profiling. To address this issue, we propose a novel approach to generate a large-scale, language-agnostic test case dataset. The process involves:

† https://leetcode.com/problemset/

Table 6: Definitions of the fields within Venus datasets.

| Column Name                                                                                                                       | Description                                                                                                                                                                                                                                                                                                                                                                                                                    |
|-----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| problem_id title question_content difficulty tags code_prompt test_case_generator test_case_evaluator test_case_runners solutions | Unique identifier for each problem (int64) Title of the problem (string) Full text of the problem statement (string) Difficulty level (categorical) List of associated tags (sequence) Prompt used for solution generation (string) Code generating test cases (string) Code evaluating test case outputs (string) Code executing solutions with test cases (string) Human-submitted solutions from LeetCode (list of strings) |

<!-- image -->

Figure 6: Pipeline for constructing the Venus dataset. We start from 3,535 LeetCode problems and apply a series of quality-control and de-duplication filters, retaining 1,284 high-quality problems in the Venus benchmark.

- Automated Test Case Generation: For each problem in Venus , a dedicated test case generator program is synthesized by GPT-4O based on the given problem description. These generators are designed to produce a virtually unbounded stream of diverse and valid inputs.
- Rigorous Validation: The validity of each generated test case is paramount. Before being used for evaluation, a candidate test case is run against all collected canonical human solutions for that problem. Only test cases for which all canonical solutions produce consistent outputs are accepted. This ensures that the test cases are unambiguous and accurately reflect the problem's requirements as understood by proficient human programmers.
- Standard I/O for Language Agnosticism: The key to the multilingual capability of Monolith lies in its interaction protocol with the code being tested. All solutions, irrespective of their programming language, interface with the test harness exclusively via standard input ( stdin ) and standard output ( stdout ). Test inputs are provided as text streams via stdin , and the solution's output is captured from stdout . This text-based I/O mechanism decouples the test data from the specifics of any programming language.

This design allows the same set of validated test cases to be used for evaluating solutions written in any of the languages supported by Venus (Python, C++, Go, Java, JavaScript, etc., as shown in Table 7). This language-agnostic approach not only broadens the applicability of our framework but also simplifies its extension to new programming languages in the future, as new test case generators are not required for each language. The common testbed ensures fair and consistent efficiency comparisons across different languages and models.

Venus Justification. Multilingual Coverage. A primary contribution is that Venus is the first multilingual code efficiency benchmark, covering six languages. Prior work, including Mercury [16], EffiBench [26], and EvalPerf [44], focuses exclusively on Python. EffiBench-X [53] is a multilingual

Table 7: Breakdown of Venus dataset by programming language. For each language we list the total number of tasks and the average number of human submission per task.

| Language       | Python   | C++   |    Go | Java   |   JavaScript | Total   |
|----------------|----------|-------|-------|--------|--------------|---------|
| Train Tasks    | 2,181    | 2,183 | 866   | 1,358  |        704   | 7,298   |
| Test Tasks     | 300      | 300   | 200   | 300    |        200   | 1,300   |
| Avg. Solutions | 106.6    | 112.2 |  33.6 | 69.6   |         74.4 | 79.3    |

Figure 7: An Example in Venus Python Subset.

<!-- image -->

efficiency benchmark at the same time. Solutions Diversity. Venus offers a significantly larger and more diverse set of reference solutions. While EffiBench uses a single baseline and Mercury averages 18.4 solutions per problem, Venus provides an average of 79.3 solutions per problem for each language. This massive increase directly contributes to the statistical diversity of performance metrics, as shown in the time and memory distributions in Appendix Figure 7. Evaluation Dimensions. As a direct extension of Mercury, which only measures execution time, Venus evaluates execution time, memory usage, and their integral, providing a more holistic assessment of code efficiency. To better illustrate the Venus dataset, we provide a complete instance from Venus as shown in Figure 7.

## B.2 APPS Dataset

APPS is a widely recognized benchmark for evaluating the functional correctness of code generation models [22]. While its original design focuses on correctness, we integrate it into our efficiency evaluation pipeline as an auxiliary benchmark. It consists of 10,000 Python programming problems, where each problem is accompanied by an average of 21.2 test cases and 23.4 solutions.

Table 8: Definitions of the fields within APPS datasets.

| Column Name                                            | Description                                                                                                                                                                                     |
|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| title question_content difficulty solutions test_cases | Title of the problem (string) Full text of the problem statement (string) Difficulty level (categorical) Human-submitted solutions from LeetCode (list of strings) Test cases (list of strings) |

Figure 8: Selection procedure for the APPS subset used in our benchmark. Beginning with the official APPS training split (5,000 problems), we discard problems that lack a sufficient number of accepted reference solutions, yielding 2,803 problems in the final dataset.

<!-- image -->

| CodeForces 10400   | Others   | Others         | Others                           | Others                           |
|--------------------|----------|----------------|----------------------------------|----------------------------------|
| CodeForces 10400   | APPS     |                |                                  |                                  |
| CodeForces 10400   | 10000    |                | Insufficient Pass Solutions 2197 | Insufficient Pass Solutions 2197 |
| CodeForces 10400   | 10000    | Train Set 5000 | Sufficient Solutions 2803        | Test Set 300                     |
| CodeForces 10400   | 10000    | Train Set 5000 | Sufficient Solutions 2803        | Train Set 2503                   |

Table 9: Comparison of Efficiency Performance between Open-Source and Closed-Source Models on the APPS Benchmark. Parentheses denote 95% confidence intervals. The top score for each metric is highlighted in bold , while the leading score within each model category is underlined.

| Model Name           | Pass@1               | Beyond-T             | Beyond-M             | Beyond-I             |
|----------------------|----------------------|----------------------|----------------------|----------------------|
| Open-source Models   | Open-source Models   | Open-source Models   | Open-source Models   | Open-source Models   |
| Qwen 2.5 3B          | 9.67                 | 5.10 (5.06, 5.13)    | 5.48 (5.45, 5.51)    | 3.80 (3.78, 3.83)    |
| Qwen 2.5 Coder 7B    | 16.00                | 9.30 (9.25, 9.34)    | 8.79 (8.75, 8.83)    | 6.81 (6.78, 6.85)    |
| Qwen 2.5 7B instruct | 17.00                | 9.65 (9.59, 9.70)    | 9.33 (9.29, 9.37)    | 7.28 (7.24, 7.32)    |
| Llama 4 Scout        | 47.67                | 24.85 (24.78, 24.92) | 27.96 (27.90, 28.03) | 18.22 (18.17, 18.28) |
| QwQ 32B ⋄            | 69.62                | 35.59 (35.51, 35.68) | 37.84 (37.75, 37.93) | 27.32 (27.26, 27.39) |
| Closed-source Models | Closed-source Models | Closed-source Models | Closed-source Models | Closed-source Models |
| GPT-4o               | 46.23                | 26.30 (26.24, 26.37) | 26.47 (26.41, 26.53) | 19.76 (19.71, 19.82) |
| Claude 3.5 Haiku     | 36.67                | 13.63 (13.57, 13.69) | 20.96 (20.91, 21.01) | 10.74 (10.70, 10.77) |
| Claude 3.7 Sonnet    | 45.63                | 21.68 (21.61, 21.75) | 25.76 (25.70, 25.81) | 15.75 (15.69, 15.81) |
| DeepSeek V3          | 56.63                | 31.58 (31.49, 31.66) | 31.42 (31.35, 31.49) | 23.20 (23.13, 23.27) |
| OpenAI o4 Mini ⋄     | 78.81                | 40.07 (39.98, 40.15) | 41.00 (40.93, 41.07) | 28.76 (28.69, 28.82) |

## C Iterative Efficiency Optimization Procedure

The Iterative Efficiency Optimization Procedure, detailed in Algorithm 1, is designed to systematically enhance source code performance. Given a problem description P , an efficiency instruction I (e.g., targeting time or memory), and a set of test cases T cases , the algorithm iteratively refines code. It begins with an initial code version C in 0 , which is generated by the Afterburner component if not provided, and its initial performance M in 0 is evaluated by the Monolith component. Over N iter iterations, new code versions are proposed by Afterburner based on the current code and its metrics, and then evaluated by Monolith . If a newly generated version C out i exhibits improved performance M out i according to the criterion I when compared to the current iteration's input performance M in i , it is adopted as the input for the subsequent iteration; otherwise, the previous code is retained. The procedure concludes by returning the best-performing code C in N iter found after N iter iterations, along with its corresponding performance metrics M in N iter .

## D Model Training Details

## D.1 Training Pipeline.

As shown in Figure 4, we explore different optimization strategies to train the Afterburner models. Initially, Afterburner SFT models are trained using the DS SFT dataset. For Afterburner DPO models, we initialize them from the checkpoints of the corresponding Afterburner SFT models and

## Algorithm 1 Iterative Efficiency Optimization Procedure

Input : Problem description P , Efficiency instruction I ∈ { time, memory, integral } , Set of test cases T cases , Original code C in 0 (optional), Number of iterations N iter Output : Improved code C out 0 , Improved code performance M out 0

```
if not C in 0 then C in 0 ← Afterburner ( P , I , None , None ) ▷ Initial code generation. end if M in 0 ← Monolith ( C in 0 , T cases ) ▷ Initial code evaluation. for i ← 1 to N iter do C out i ← Afterburner ( P , I , C in i , M in i ) ▷ Code optimization. M out i ← Monolith ( C out i , T cases ) ▷ Code evaluate. if M out i ≻ M in i ) then ▷ Compare the performance concerning I . ( C in i +1 , M in i +1 ) ← ( C out i , M out i ) ▷ Update with the better performing candidate. else ( C in i +1 , M in i +1 ) ← ( C in i , M in i ) ▷ Otherwise, retain the current best. end if end for return ( C in N iter , M in N iter ) ▷ Return the best code found after N iter iterations and its metrics
```

subsequently finetune them on the DS DPO dataset. The training process for Afterburner GRPO models involves two steps: first, a base model is finetuned on the DS cold \_ start dataset to ensure adherence to the required response format; thereafter, these models are trained on DS GRPO .

## D.2 Details of Afterburner SFT

Training. We fine-tune Qwen/Qwen2.5-3B-Instruct using Low-Rank Adaptation (LoRA). The model is trained for one epoch on DS SFT . Key hyperparameters include a learning rate of 3e-5, managed by a cosine scheduler with 200 warm-up steps, an effective batch size of 64 (per-device batch size of 4 with 16 gradient accumulation steps), and the adamw\_torch optimizer . For LoRA, the rank is 8 and alpha is 16. The training uses BF16 precision, and gradients are clipped at a norm of 1.0.

## D.3 Details of Afterburner DPO

Training. Afterburner DPO is trained from the checkpoint of Afterburner SFT utilizing LoRA for one epoch of DS DPO dataset. Key hyperparameters include: learning\_rate=4e-5 with a cosine scheduler and 300 warm-up steps, an effective batch size of 16 (per-device batch size of 2 with 8 gradient accumulation steps), and the adamw optimizer. LoRA parameters are set to rank 16, alpha 16, and a dropout of 0.05. DPO-specific settings include a beta of 0.1 and a sigmoid loss function, with pref\_ftx (SFT loss component) set to 0. The training uses BF16 precision, and gradients are clipped at a norm of 1.0.

## D.4 Details of Afterburner Code Start

Model Response Collection. We collect the model response from gemini-2.5-pro-exp-03-25 , using the system prompt as shown in Section E and the Afterburner inference prompt as shown in Section E. We only keep the responses that can pass the response regex filter as shown in Section D.5.

Training. We conduct SFT on base model Qwen2.5-3B-Instruct using the LLaMA Factory framework [75]. The model undergoes full fine-tuning on an epoch of the DS COLD dataset, with input sequences processed up to a maximum length of 32,768 tokens. Key hyperparameters included learning rate=5e-5 , managed by a cosine scheduler with 50 warm-up steps , an effective batch size of 4, and the adamw\_bnb\_8bit optimizer.

## D.5 Details of Afterburner GRPO

Training. Afterburner GRPO is trained on Verl [58] and initialized from Afterburner CS . The GRPO training runs for 20 epochs on DS GRPO . Since executing generated code and computing

its efficiency metrics are time-consuming, we use a batch reward function to accelerate the reward calculation in a parallel manner. Key hyperparameters include: actor\_learning\_rate=1e-6 , ppo\_mini\_batch\_size=32 (4 per-GPU micro-batch). During the roll-outs, 16 responses are generated per prompt using vLLM [39] with inference\_temperature=1.0 . KL loss for actor updates is disabled, and the entropy coefficient is 0. For the reward weights, we set β f = 0 . 2 , β e = 0 . 3 , β c = 0 . 5 . Note that R efficiency is set to 0 if C ′ pass = 0 . eupper is set to 90, 1048576, 94371840 , respectively, which aligns with our timeout ( 90s ) and memory ( 1GB ) limitation. In our experiment, we observed that a dominant weight for functional correctness ( β c ≥ 0 . 5 ) is essential for stable training, as values below this threshold often led to training crash. To balance our objectives, we set a fixed β c = 0 . 5 while gradually increasing the code efficiency weight β e from 0.3 to 0.5. This strategy ensures the model first learns to produce correct code before optimizing for efficiency.

Format Regex. Inspired by recent works [20, 57, 37, 63, 19], we encourage our model to generate the reasoning content before the code solution. The designated response format: '&lt;thinking&gt; thing\_content &lt;/thinking&gt; &lt;solution&gt; solution\_content &lt;/solution&gt;' .

```
Afterburner Format Regex 1 import re 2 def single_thinking_solution_format(text: str) -> bool: 3 pattern = re.compile( 4 r""" 5 \A\s* # optional leading whitespace 6 <thinking > 7 (?:(?!<thinking >).)*? 8 </thinking >\s* # end <thinking > 9 <solution > 10 (?:(?!<thinking >|<solution >).)*? 11 </solution >\s* # end <solution > 12 \Z 13 """, 14 re.DOTALL | re.VERBOSE , 15 ) 16 return bool(pattern.fullmatch(text))
```

Reward Function Design for Enhanced Code Generation The efficacy of our Group Relative Policy Optimization (GRPO) framework, particularly for a task as nuanced as code generation, heavily relies on a well-designed reward function. Our objective is to guide the Afterburner GRPO model not merely towards syntactically valid code, but towards solutions that are functionally correct, computationally efficient, and adhere to a desired structured output format that includes an explicit reasoning phase. To this end, our final reward R final is a carefully weighted composite of three distinct components, each targeting a critical aspect of code quality.

Format Control ( R Format ). We first incentivize adherence to a predefined output structure, which mandates a thinking phase encapsulated in &lt;thinking&gt;...&lt;/thinking&gt; tags followed by the code within &lt;solution&gt;...&lt;/solution&gt; tags. As defined in Eq. (6), R Format provides a strong binary signal ( +1 for compliance, -1 otherwise). This not only ensures predictable and parsable outputs for automated assessment but also explicitly encourages the model to engage in a "thought process" prior to generating the final solution, a step we believe is crucial for complex problem-solving.

Functional Correctness ( R correct ). Ensuring functional soundness is paramount. However, a simple binary pass/fail reward for the current generation C ′ can be a sparse and inefficient signal. Instead, R correct (Eq. (7)) evaluates C ′ in comparison to a baseline attempt C . It assigns the highest positive reward ( 1 . 0 ) for an "upgrade" (i.e., C ′ passes tests while C fails) and the largest penalty ( -1 . 0 ) for a "downgrade" ( C ′ fails while C passes). Maintaining a passing or failing status yields intermediate rewards ( 0 . 5 and -0 . 5 respectively). This relative assessment provides a more nuanced gradient, strongly favoring improvements and robustly penalizing regressions.

Efficiency Improvement ( R efficiency ). Beyond correctness, generating efficient code is our key objective. R efficiency (Eq. (8)) is designed to reward relative improvements in computational performance (e.g., time, memory). The core of this reward is e gain, which measures the normalized improvement

of the current solution C out over a baseline C in , after clipping efficiency metrics to a sensible range [0 , eupper ] to handle outliers. Crucially, we apply the hyperbolic tangent function ( tanh ) to e gain. This bounds the reward component within ( -1 , 1) , providing a smooth, scaled signal that is sensitive to gains but diminishes returns for extremely large improvements or degradations, thereby stabilizing the learning process. A small ϵ in the denominator of e gain ensures numerical stability. Furthermore, the inherent stochasticity often present in empirical efficiency measurements (e.g., due to minor system-level variations or non-deterministic aspects of complex code execution) means that R efficiency naturally introduces a degree of noise. This moderate, implicit stochasticity can be beneficial for GRPO, as it helps maintain variance in reward signals across roll-outs. This, in turn, can prevent the advantage term A i (Eq. (11)) from prematurely collapsing or vanishing, thereby fostering continued exploration and more robust policy updates.

Figure 9: Iterative Optimization Performance on APPS.

<!-- image -->

## E Model Prompts

## Baseline Prompt

```
1 You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages. 2 Your task is to implement a {efficiency_instruction} solution to the following problem in {target_lang}. 3 4 ## Problem Description 5 {question} 6 7 ## Output Format 8 -Provide the complete solution code in **one markdown code block** with appropriate language identifier. 9 -Implement the function with the exact signature (name, parameters , etc.) specified in the starter code.
```

## Afterburner System Prompt

- 1 A conversation between User and Assistant. 2 The User asks a question and provides an original solution , then the Assistant improves it. 3 The assistant first thinks about the reasoning process in the mind and then provides the user with the improved solution. 4 The reasoning process and solution are enclosed within &lt;thinking &gt; &lt;/thinking &gt; and &lt;solution &gt; &lt;/solution &gt; tags , respectively.
- 5 For example , "&lt;thinking &gt;reasoning\_process &lt;/thinking &gt; &lt;solution &gt; improved\_solution &lt;/solution &gt;".

```
Afterburner Prompt Template 1 ## Instructions 2 Your task is to implement a solution to the following problem in {target_lang}. 3 4 ## Problem Description 5 {problem_description} 6 7 ## Original Solution 8 {original_solution} 9 10 ## Original Performance 11 Passed: {original_passed} / Time: {original_time} / Memory: { original_memory} / Integral: {original_integral} 12 13 ## Output Format 14 -Provide the complete solution code in one markdown code block with an appropriate language identifier. 15 -Generate the initial solution code directly if Original Solution is empty. 16 -Fix the original solution if it was not passed. Optimize the { efficiency_instruction} performance if the original solution was passed.
```

## F Would the History-Aware Model Perform Better?

We attribute the under-performance of history-aware model in Table 10 to two primary factors: 1) Distribution Shift: The performance drop is likely caused by a distribution shift, as the training format (single-turn) differs from the multi-turn, history-aware format used during inference. 2) Training Challenges: While fine-tuning the model with multi-turn format could mitigate the distribution shift, this approach introduces the significant challenge of long-term credit assignment, a well-known difficulty in Reinforcement Learning [15].

Table 10: Performance of Single-turn Loop (Afterburner) and Multi-turn Loop (History-aware).

| Iteration                       | 0                               | 1                               | 2                               | 5                               | 7                               | 10                              |
|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| Single-turn Loop                | Single-turn Loop                | Single-turn Loop                | Single-turn Loop                | Single-turn Loop                | Single-turn Loop                | Single-turn Loop                |
| Pass%                           | 47.33                           | 50.33                           | 52.00                           | 58.17                           | 60.50                           | 61.67                           |
| Beyond-I%                       | 18.24                           | 24.81                           | 29.44                           | 35.48                           | 38.01                           | 38.95                           |
| Multi-turn Loop (History-aware) | Multi-turn Loop (History-aware) | Multi-turn Loop (History-aware) | Multi-turn Loop (History-aware) | Multi-turn Loop (History-aware) | Multi-turn Loop (History-aware) | Multi-turn Loop (History-aware) |
| Pass%                           | 45.00                           | 46.33                           | 46.33                           | 46.33                           | 46.67                           | 46.67                           |
| Beyond-I%                       | 15.10                           | 15.10                           | 15.10                           | 15.10                           | 15.10                           | 15.10                           |

The above results show that applying the history-aware method directly to a vanilla model yielded no performance improvement over the single-turn baseline. Despite instructions to avoid repetition, the model predominantly replicated existing solutions from the prompt. The few novel solutions it generated, while claiming to be superior, demonstrated no empirical performance gains. This observation aligns with our core hypothesis: vanilla models can generate correct code but lack an intrinsic awareness of code efficiency.

## G Uncertainty in Code Efficiency Measurement

Quantifying code efficiency is a nuanced challenge. Theoretical efficiency, typically expressed via asymptotic notation (e.g., O ( nlogn ) ), offers high-level algorithmic understanding but often neglects constant factors, compiler optimizations, and hardware-specific impacts (like cache performance or instruction-level parallelism) crucial for real-world performance. Consequently, it provides an incomplete picture for comparing concrete code implementations.

Table 11: A summary of different training and inference schemes.

| Training / Inference   | Single-Turn                                                                                                                 | Multi-Turn                                                                                                                                                          |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Single-Turn            | Afterburner (P) Consistent distribution. (P) Stable RL training. (P) Avoids credit assign- ment. (C) Cannot handle history. | Training Waste (C) It wastes the model's trained abil- ity.                                                                                                         |
| Multi-Turn             | Distribution Mismatch (C) Critical mismatch between train- ing and inference data leads to poor performance.                | Ideal History-Aware (P) Creates a truly context-aware agent. (C) Suffers from sparse rewards and unstable training. (C) Difficult long- distance credit assignment. |

On the other end of the spectrum, simulation and low-level statistics (e.g., cycle-accurate simulations, CPU performance counters) can provide extremely detailed data [60]. However, these methods often introduce substantial complexity in setup and interpretation, may have limited scope in accurately modeling all modern system intricacies, or can be overly specific to a particular hardware configuration, making generalization difficult. For our purposes, the granularity and setup overhead of such approaches outweigh their benefits.

We therefore opt for empirical performance measurement , directly observing execution metrics like runtime and memory usage. This approach holistically captures the interplay of algorithm, code structure, compilation, and the underlying hardware. While direct, empirical results are subject to inherent system noise and run-to-run variability. To rigorously address this and derive stable performance indicators, robust statistical techniques are indispensable, leading to our choice of bootstrapping for uncertainty quantification.

## G.1 Details of Bootstrapping Evaluation

To quantify the statistical uncertainty of our efficiency metrics, we employ a bootstrapping procedure [17]. Task-level efficiency metrics are first grouped by their respective IDs to ensure independent sampling for each task. We generate B = 128 bootstrap replicates. Each replicate is constructed by sampling k = 4 solutions for every unique problem (in our settings, we repeatedly evaluate each generated code 16 times). For each of these B replicates, we then calculate the Average BEYOND-T, BEYOND-M, and BEYOND-I. Finally, we report the mean of each of these four metrics across all replicates, along with their corresponding 95% confidence intervals, to offer a robust evaluation of model performance.

## H Monolith Implementation

Code Execution Environment. We deploy a code execution environment on a GCP n2-highcpu-96 instance (96 vCPUs, 96 GB Memory) with 81 Monolith workers. Each worker operates within a dedicated Docker container [13], which is allocated 1 vCPU, 1 GB of memory, and provided with an isolated temporary directory. To ensure a pristine execution environment for each evaluation, containers are created anew for every task. CPU affinity for each worker was set to 100% to minimize performance variability during measurements. Execution time and peak memory overhead were measured using the 'time -v' command. To gather instantaneous memory usage and calculate the integral score, we sampled the 'VmRSS' field from the process status file ( /proc/[pid]/status ). To accelerate model inference, we use the batch inference feature on Neibus † for all available models listed in Table 4. For proprietary models, we call their provided APIs. For those models without an online inference point, we host vLLM [39] inference service locally. Further details on the execution environment are available in the Appendix H.

Runtime. The monolith's runtime environment is standardized using Docker containerization to ensure consistency and portability across different programming languages. Each language or service

† https://studio.nebius.com/

within the monolith operates within a specific, pre-defined Docker image. Table 12 details the official Docker images utilized for various supported programming languages.

Table 12: Programming Language Docker Images

| Language                                | Image                                                                                                                                                    |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Python Java Javascript Cpp Go Ruby Rust | python:3.9.19-bullseye openjdk:11.0.12-jdk-bullseye node:22-bullseye gcc:11.2.0-bullseye golang:1.17.0-bullseye ruby:3.0.2-bullseye rust:1.85.0-bullseye |

## I Symbol List

| Symbol / Term                                                                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LLM SFT DPO GRPO RL IOF                                                                      | Large Language Model Supervised Fine-Tuning Direct Preference Optimization Group Relative Policy Optimization Reinforcement Learning Iterative Optimization Framework (the proposed framework)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Afterburner Monolith Venus APPS DS SFT DS DPO DS CS DS GRPO P I C                            | Code optimization models (trained via SFT, DPO, GRPO) A high-fidelity code execution sandbox for performance feedback A dataset with human solutions, curated for efficiency benchmarking An existing dataset for code generation, also used for evaluation Dataset constructed for Supervised Fine-Tuning Preference dataset constructed for Direct Preference Optimization Cold Start Dataset used for initial format alignment of GRPO models Dataset used for Group Relative Policy Optimization training                                                                                                                                                                                                                                                                                                                                                                                                             |
| C in i C out i C + C - C baseline M M in i M out i ≻ N iter X π θ π ref L SFT L DPO L GRPO β | Problem description Efficiency instruction A code solution. Variants: Input code solution for iteration i Output (improved) code solution from Afterburner at iteration i A more efficient/preferred code solution A less efficient/dis-preferred code solution A baseline code solution for comparison (in DPO context) Performance metric(s) of a solution. Variants: Performance metrics of C in i Performance metrics of C out i Relation indicating superior performance (e.g., M out i ≻M in i ) Total number of optimization iterations Input prompt to a model (often includes P , I , C in , M in ) The policy (language model) being trained, parameterized by θ Reference policy (e.g., in DPO, the SFT model) Loss function for Supervised Fine-Tuning Loss function for Direct Preference Optimization Loss function for Group Relative Policy Optimization Weights for reward components ( β f ,β c ,β e ). |

Continued on next page

| Symbol / Term                                                                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|---------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| R Format R correct R efficiency E E gain E clip E upper R final O i G W i A i ϵ | Reward component for adhering to the specified output format Reward component for functional correctness Reward component for improvement in computational efficiency An absolute code efficiency value (e.g., execution time, peak memory) Normalized relative gain in an efficiency metric e Clipped value of an efficiency metric e Upper limit for clipping an efficiency metric e The final combined reward signal for GRPO The i -th generated output (rollout/candidate solution) in a GRPO group Size of the rollout group in GRPO Policy ratio (importance weight) for rollout O i in GRPO, π θ ( O i &#124;X ) π θ old ( O i &#124;X ) Advantage of rollout O i within its group in GRPO |
| PASS@1 PR( x,D )                                                                | Percentage of the first generated solution passes all test cases Percentile Rank: fraction of items in distribution D that x is greater than or equal to (for efficiency, lower is better, so 1 - PR or adjusted PR is used implicitly if higher means better)                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| BEYOND-T                                                                        | Global efficiency metric: average percentile rank of generated code's Global efficiency metric: average percentile rank of generated code's memory usage relative to human solutions. Higher is better.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| k k k T M I                                                                     | generated solution for task k                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|                                                                                 | A small constant                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|                                                                                 | Percentage of generations that are worse than all human solutions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| r gen ,m gen                                                                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|                                                                                 | execution time relative to human solutions. Higher is better.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| BEYOND-M                                                                        | Global efficiency metric: average percentile rank of generated code's                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| BEYOND-I , i gen                                                                | integral score relative to human solutions. Higher is better. Absolute execution time, peak memory usage, and integral score of the                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| D k ,D k ,D k                                                                   | Distributions of execution times, memory usages, and integral                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|                                                                                 | scores from reference human solutions for task k                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| B%                                                                              | Percentage of generations that are better than all human solutions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| M%                                                                              | Percentage of generations whose efficiency falls within the range of human solutions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| F%                                                                              |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| W%                                                                              | Percentage of failed model generation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims made in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discussed the limitations of the work.

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

Justification: The paper does not include theoretical results.

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

Justification: All necessary information to reproduce the experimental results are provided.

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

Justification: We released the data and code to reproduce our main experimental results.

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

Answer: [Yes]

Justification: We specified all the training and test details.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Yes. We report CI 95%.

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

Answer: [Yes]

Justification: We provided information on the computer resources.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conforms with the NeurIPS Code of Ethics.

Guidelines:

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
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to

generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cited the original paper.

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

Justification: The newly introduced dataset is well-documented.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.