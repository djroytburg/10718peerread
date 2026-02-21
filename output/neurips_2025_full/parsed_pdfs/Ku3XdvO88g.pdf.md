## GuardReasoner-VL:

## Safeguarding VLMs via Reinforced Reasoning

Yue Liu 1,2,3 , Shengfang Zhai 3 , Mingzhe Du 4,3

Yulin Chen 3 , Tri Cao 3 , Hongcheng Gao 3 , Cheng Wang 3

Xinfeng Li 4 , Kun Wang 4 , Junfeng Fang 3 , Jiaheng Zhang 3 , Bryan Hooi 2,3

1 Integrative Sciences and Engineering Programme, NUS Graduate School 2 Institute of Data Science, NUS

3 Department of Computer Science, School of Computing, NUS

4 School of Computer Science and Engineering, NTU

yliu@u.nus.edu

## Abstract

To enhance the safety of VLMs, this paper introduces a novel reasoning-based VLM guard model dubbed GuardReasoner-VL. The core idea is to incentivize the guard model to deliberatively reason before making moderation decisions via online RL. First, we construct GuardReasoner-VLTrain, a reasoning corpus with 123K samples and 631K reasoning steps, spanning text, image, and text-image inputs. Then, based on it, we cold-start our model's reasoning ability via SFT. In addition, we further enhance reasoning regarding moderation through online RL. Concretely, to enhance diversity and difficulty of samples, we conduct rejection sampling followed by data augmentation via the proposed safety-aware data concatenation. Besides, we use a dynamic clipping parameter to encourage exploration in early stages and exploitation in later stages. To balance performance and token efficiency, we design a length-aware safety reward that integrates accuracy, format, and token cost. Extensive experiments demonstrate the superiority of our model. Remarkably, it surpasses the runner-up by 19.27% F1 score on average, as shown in Figure 1. We release data, code, and models (3B/7B) of GuardReasoner-VL 1 .

Warning: This Paper Contains Potentially Harmful Content.

## 1 Introduction

Built upon large language models (LLMs), vision-language models (VLMs) achieve remarkable success in a wide range of real-world applications such as computer use [73], deep research [77], embodied AI [13], etc. However, when deploying VLMs in safety-critical domains such as education [11], finance [86], or government, they remain vulnerable to manipulations and attacks [52, 21, 53, 41]. To alleviate this problem, safety alignment methods [51, 101] are proposed by training VLMs to align with human values and expectations. While effective, it imposes the alignment tax [27, 42], compromising the fundamental capabilities of models, such as creativity, helpfulness, and reasoning.

To mitigate this drawback, VLM guard models [15, 10, 30] are developed to safeguard VLMs without direct modifications to the victim VLMs. For example, VLMGuard [15] detects malicious text-image prompts using unlabeled data. In addition, LLaMA Guard 3-Vision [10] moderates both text-image

1 https://github.com/yueliu1999/GuardReasoner-VL

2 Jiaheng Zhang and Bryan Hooi are corresponding authors.

<!-- image -->

<!-- image -->

(b) Response Harmfulness Detection Task.

Figure 1: Mean Performance of GuardReasoner-VL on Multi-modal Guardrail Benchmarks.

prompts and text responses by SFT. Then, Beaver-Guard-V [30] is developed via RL with a welltrained reward model. The existing VLM guard models are trained to output only classification results. Although effective, they lack interpretability, as the models do not justify their decisions. Besides, the harmful categories are fixed, restricting the generalization to new categories.

Therefore, this paper aims to build a reasoning-based VLM guard model. It has three challenges as follows. 1) Limited Data . The available training data is limited in terms of the number of samples, input modalities, and reasoning processes. 2) Offline Training . Current guard models are typically restricted to offline training, which hampers their performance. 3) Token Efficiency . The reasoning process increases token costs, reducing inference efficiency.

To this end, we propose a novel reasoning-based VLM guard model termed GuardReasoner-VL by incentivizing it to reason-then-moderate via online RL. 1) First, to solve data limitations, we create GuardReasoner-VLTrain, a reasoning corpus with 123K samples and 631K reasoning steps. Unlike the existing data, we collect a mixture of text, image, and text-image samples (see Figure 3) to match the diverse input modalities of VLMs, and generate reasoning processes by prompting GPT-4o. Based on GuardReasoner-VLTrain, we cold-start our model via SFT. 2) Then, we conduct online RL to incentivize our model. To increase the diversity and difficulty of the data, we perform data augmentation via our proposed safety-aware data concatenation . The main principle is to guide the model to detect harmful content hidden among predominantly harmless content. We concatenate the inputs of different samples and assign new safety labels based on whether any of the original samples are labeled as harmful. Besides, we use a dynamic clipping parameter to encourage the model to explore in the early stage and exploit in the later stage. 3) To balance the model performance and token efficiency, we design a length-aware safety reward , integrating accuracy, format, and reasoning tokens. We develop two model versions: GuardReasoner-VL, a more powerful version, and GuardReasoner-VL-Eco, a more token-economical version. The contributions are listed as follows.

- We develop GuardReasoner-VL, a novel VLM guard model that first reasons and then moderates.
- We curate a reasoning corpus for VLM guard termed GuardReasoner-VLTrain, containing 123K samples with 631K reasoning steps, covering text, image, and text-image paired samples.
- We incentivize the reasoning ability of our model through online RL, incorporating the proposed safety-aware data concatenation, dynamic clipping parameter, and length-aware safety reward.
- Extensive experiments and analyses verify the superiority of our proposed GuardReasoner-VL.

## 2 GuardReasoner-VL

This section outlines the methodology of the proposed GuardReasoner-VL. First, we define the moderation task of VLM guard models. Then, we present the data curation for our training data. In addition, we introduce the training pipeline of our proposed reasoning-based VLM guard model. The overview training pipeline is shown in Figure 2. The basic notations are summarized in Table 4.

Figure 2: Overview Training Pipeline of GuardReasoner-VL. It mainly contains three processes, including data curation, model cold-start, and online RL. Concretely, we first build a reasoning corpus, which contains 123K samples with 631K reasoning steps, spanning text, image, and textimage modalities. We cold-start the model via reasoning SFT. Then, we perform data augmentation to improve the difficulty and diversity of the data via safety-aware data concatenation. In addition, we conduct online RL with a dynamic clipping parameter and the designed length-aware safety reward.

<!-- image -->

Moderation Task. Given a victim VLM F , a user inputs a prompt X and receives a response S = F ( X ) , where X can be represented by one of the following modalities: a text T , an image I , or an text-image pair {T , I} . The VLM guard model G moderates the input and output of the victim VLM F by detecting whether they are harmful, formulated as follows.

<!-- formula-not-decoded -->

where ˆ Y prom ∈ { harmful , unharmful } is the predicted label for the prompt harmfulness detection task, and ˆ Y res ∈ { harmful , unharmful } is the predicted label for the response harmfulness detection task. The performance of G is evaluated using the F1 score between the predicted label ˆ Y and the ground-truth Y = {Y prom , Y res } . The harmful/unharmful samples are treated as positives/negatives.

However, existing VLM guard models [15, 10, 30] merely provide classification results, limiting performance, explainability, and generalization. Thus, we aim to develop a reasoning-based VLM guard model G reasoner to first deliberatively reason and then make moderation decisions as follows.

<!-- formula-not-decoded -->

where R are reasoning processes, improving performance, explainability, and generalization.

## 2.1 Data Curation

Figure 3: Input Modalities and Distribution of Our Training Data GuardReasoner-VLTrain. It contains 123K samples with 631K reasoning steps, spanning 3 input modalities, including text, image, and text-image. The prompt and response can be classified as harmful or unharmful.

<!-- image -->

First, to match the diverse input modalities of VLMs, we collect a mixture of text, image, and text-image samples . The distribution and cases are demonstrated in Figure 3.

Figure 4: Data Sources and Cases of Different Modalities in GuardReasoner-VLTrain Dataset. The Y-axis denotes the number of samples. The X-axis denotes the different data sources.

<!-- image -->

Text . Following GuardReasoner [49], we collect and combine the data of WildGuardTrain [25], AegisTrain [19], BeaverTailsTrain [29], and ToxicChatTrain [43]. To balance the ratios of different input modalities, we use 50% of the mixed text data.

Image . We collect and combine the data of UnsafeBench [66], BadNews [94], HatefulMemes [34], HatefulPoliticalMemes (HatefulPMemes) [65], and HOD [24]. For HatefulMemes and HatefulPMemes, we utilize the processed data from VLGuard [101]. For HOD, we use 60% of the original dataset to balance the harmful and unharmful categories of the images. For this constructed image data, we use 80% for training and 20% for testing. The test set is named as HarmImageTest.

Text-Image . We utilize the SPA-VL-Train dataset [97] as the text-image paired training data. To balance the ratios of different input modalities, we use 50% of the SPA-VL-Train dataset.

Then, to train the reasoning-based VLM guard models, we generate the reasoning processes via prompting GPT-4o [49], as shown in Figure 10. As a result, we obtain a reasoning corpus termed GuardReasoner-VLTrain, consisting of 123K samples and 631K reasoning steps. The detailed statistics is listed in Table 6. In Figure 4, we show the distribution of data sources, the distribution of harmful categories, and representative cases of each modality in GuardReasoner-VLTrain.

## 2.2 Model Cold-Start

Based on the curated reasoning dataset GuardReasoner-VLTrain, denoted as D , we cold-start the base model via Reasoning Supervised Fine-Tuning (R-SFT). Specifically, given the guardrail instruction Q , the user prompt X , and the victim model's response S , we train the base model M base to generate both the reasoning process R and the moderation result Y . The objective is formulated as follows.

<!-- formula-not-decoded -->

where θ denotes the model parameters. The input X can be a text, an image, or a text-image pair. The instruction, input, and output are showcased in Figure 11. Through R-SFT, we endow the model with basic reasoning ability for moderation, resulting in a reasoning model M R-SFT.

## 2.3 Online Reinforcement Learning

Then, we perform online RL on M R-SFT to further enhance the reasoning ability regarding moderation. It contains three parts, including data augmentation, training process, and reward design.

## 2.3.1 Data Augmentation

We generate harder and more diverse samples to better facilitate the generalization of online RL. First, we perform rejection sampling on M R-SFT over the reasoning corpus D . We run the entire dataset four times with high randomness and select the samples for which all predictions are incorrect.

Figure 5: Safety-Aware Data Concatenation for Data Augmentation. Given two samples with labels {X 1 , Y 1 } and {X 2 , Y 2 } , we generate a new sample X new by concatenating text and merge image. We assign the new label Y new as harmful if any of the original labels Y 1 , Y 2 is harmful. It enables the guard model to identity harmful content hidden among predominantly harmless content.

<!-- image -->

Then, to further improve the diversity and the difficulty of the data, we conduct data augmentation via safety-aware data concatenation , as shown in Figure 5. Our core idea is to enable the guard model to identify harmful content hidden among predominantly harmless content. Take the prompt harmfulness detection task as an example, given two text-image paired inputs X 1 = {T 1 , I 1 } , X 2 = {T 2 , I 2 } and their labels Y 1 , Y 2 , the augmented sample is constructed as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where text\_concat denotes concatenating two textual inputs into a single context. image\_merge denotes combining two image inputs through image-level transformations. For the new label Y new of the augmented sample X new, we assign it as harmful if any of the original samples is harmful. In this manner, it can enhance the guard model's ability to detect harmfulness in more complex and challenging cases. Through rejection sampling and safety-aware data augmentation, we generate a hard-sample reasoning corpus D RL for online RL. We admit the imbalance could degrade the overall data quality. However, the overall performance improvement shows the effectiveness of our overall data. We think this problem can be solved by designing a new sampling strategy, i.e., when the imbalance happens, we discard the corresponding samples. Besides, we carefully checked our data augmentation and didn't find any imbalance problem. The cases can be found in Figure 4.

## 2.3.2 Training Process

Based on D RL, we train M R-SFT via online RL. We implement it by using group relative policy optimization (GRPO) [70]. Unlike standard GRPO, we omit the KL divergence loss to reduce constraints on the model's behavior. In addition, we propose to encourage exploration in the early

training stages and exploitation in the later training stages. The objective is formulated as follows.

<!-- formula-not-decoded -->

where K i is the policy ratio, A i denotes the estimated advantage, { r 1 , r 2 , ..., r G } is a group of rewards. We introduce a dynamic clipping parameter B s in Formula (7), where s is the current training step, and s total is the total number of training steps. In the early stage, the clipping threshold is set to a large value, allowing the model to explore more freely. In the later stages, it is gradually reduced to encourage more stable and fine-grained updates.

## 2.3.3 Reward Design

We design a safety reward to guide our guard model to finish two guardrail tasks, i.e., prompt harmfulness detection and response harmfulness detection. First, the model should output in a correct format to ensure the predicted results are extracted correctly. Then, based on the correct format, we calculate the correctness between the predicted results and the ground truth of these two tasks, and combine them linearly. This safety reward is formulated as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where I format indicates whether the output format satisfies the required structure, i.e., I format = 1 if the model places the reasoning process R between the '&lt;think&gt;' and '&lt;/think&gt;' tags, and the predicted label between the '&lt;result&gt;' and '&lt;/result&gt;' tags; otherwise, I format = 0 .

Based on r safety, to balance the performance and token efficiency, we incorporate the length of the reasoning process into the reward. The basic idea is that when the model fails to complete these guardrail tasks correctly, it is encouraged to improve its accuracy by scaling up the reasoning length, while remaining within a constraint. This length-aware safety reward is formulated as follows.

<!-- formula-not-decoded -->

where l norm ∈ [0 , 1] is the normalized length of the reasoning R , and β is a cut-off hyper-parameter to alleviate over-thinking. Note that the numerator r safety is constrained to be non-positive, i.e., r safety ∈ [ -1 , 0] . Thus, when the model fails to complete all tasks correctly, i.e., r safety ∈ [ -1 , 0) , it is encouraged to improve its accuracy by increasing the reasoning length, subject to the constraint β .

Through online RL with these designs, we obtain a reasoning-based VLM guard model G reasoner .

## 3 Experiments

Environment . All experimental results are obtained on two servers with 8 NVIDIA H100 (80 GB) GPUs, and one server with 4 NVIDIA H200 (141GB) GPUs. For SFT, we use the LLaMA Factory [98] training platform. For online RL, we use the EasyR1 [99] training platform.

Benchmark . We evaluate our method on 14 benchmarks across two guardrail tasks, including prompt harmfulness detection and response harmfulness detection. For prompt harmfulness detection, we use 8 benchmarks, covering text-only inputs (ToxicChat [43], OpenAIModeration [58], AegisSafetyTest [19], SimpleSafetyTests [80], HarmBench [59], WildGuardTest [25]), image-only inputs (HarmImageTest), and text-image paired inputs (SPA-VL-Eval [97]). For response harmfulness detection, we use 6 benchmarks, including HarmBench [59], SafeRLHF [12], BeaverTails [29], XSTestResponse [69], WildGuardTest [25], and SPA-VL-Eval [97]. The statistical information of these benchmarks is listed in Table 5. We use F1 score (harmful category as positive samples) for evaluation. Due to the varying sample sizes across benchmarks (0.1K to 3K), we use a sample-weighted average of F1 scores

Table 1: F1 score (%) of 21 Models on 8 Benchmarks of Prompt Harmfulness Detection. The bold and underlined values denote the best and the runner-up. '-' denotes that the result is unavailable.

| Method                    | ToxicChat        | HarmBench        | OpenAI Moderation   | Aegis SafetyTest   | Simple SafetyTests   | WildGuard Test   | Average (Text)   | HarmImage Test   | SPA-VL- Eval     | Average (All)    |
|---------------------------|------------------|------------------|---------------------|--------------------|----------------------|------------------|------------------|------------------|------------------|------------------|
| LLM Guard Models          | LLM Guard Models | LLM Guard Models | LLM Guard Models    | LLM Guard Models   | LLM Guard Models     | LLM Guard Models | LLM Guard Models | LLM Guard Models | LLM Guard Models | LLM Guard Models |
| LLaMA Guard 7B            | 61.60            | 67.20            | 75.80               | 74.10              | 93.00                | 56.00            | 64.89            | 00.00            | 00.00            | 33.43            |
| LLaMA Guard 2 8B          | 47.10            | 94.00            | 76.10               | 71.80              | 95.80                | 70.90            | 63.62            | 00.00            | 00.00            | 32.77            |
| LLaMA Guard 3 8B          | 53.12            | 98.94            | 79.69               | 71.39              | 99.50                | 76.18            | 68.47            | 00.00            | 00.00            | 35.27            |
| Aegis Guard Defensive 7B  | 70.00            | 77.70            | 67.50               | 84.80              | 100.00               | 78.50            | 72.99            | 00.00            | 00.00            | 37.60            |
| Aegis Guard Permissive 7B | 73.00            | 70.50            | 74.70               | 82.90              | 99.00                | 71.50            | 73.83            | 00.00            | 00.00            | 38.03            |
| Aegis Guard 2.0 8B        | -                | -                | 81.00               | -                  | -                    | 81.60            | -                | 00.00            | 00.00            | -                |
| ShieldGemma 2B            | 06.91            | 11.81            | 13.89               | 07.47              | 05.83                | 09.36            | 09.38            | 00.00            | 00.00            | 04.83            |
| ShieldGemma 9B            | 67.92            | 67.96            | 78.58               | 77.63              | 91.89                | 57.74            | 68.77            | 00.00            | 00.00            | 35.42            |
| WildGuard 7B              | 70.80            | 98.90            | 72.10               | 89.40              | 99.50                | 88.90            | 77.99            | 00.00            | 00.00            | 40.17            |
| GuardReasoner 1B          | 72.09            | 94.92            | 69.02               | 89.34              | 98.99                | 87.13            | 77.18            | 00.00            | 00.00            | 39.76            |
| GuardReasoner 3B          | 78.38            | 88.58            | 71.88               | 91.19              | 100.00               | 88.97            | 80.80            | 00.00            | 00.00            | 41.62            |
| GuardReasoner 8B          | 79.43            | 93.30            | 71.24               | 90.27              | 100.00               | 88.59            | 81.09            | 00.00            | 00.00            | 41.77            |
| VLM Guard Models          | VLM Guard Models | VLM Guard Models | VLM Guard Models    | VLM Guard Models   | VLM Guard Models     | VLM Guard Models | VLM Guard Models | VLM Guard Models | VLM Guard Models | VLM Guard Models |
| OpenAI Moderation API     | 25.40            | 09.60            | 79.00               | 31.90              | 63.00                | 12.10            | 35.28            | 44.39            | 63.00            | 44.20            |
| Azure Content Safety API  | 57.61            | 37.41            | 74.27               | 46.75              | 74.21                | 32.54            | 54.30            | 26.42            | 43.64            | 44.95            |
| LLaMA Guard 3 Vision 11B  | 58.19            | 96.09            | 67.64               | 70.62              | 97.96                | 75.19            | 67.24            | 00.48            | 54.86            | 48.03            |
| Qwen2.5-VL-Instruct 3B    | 34.61            | 90.11            | 52.03               | 82.15              | 100.00               | 64.05            | 51.47            | 48.66            | 62.81            | 53.53            |
| Qwen2.5-VL-Instruct 7B    | 40.99            | 91.61            | 57.21               | 81.58              | 100.00               | 74.77            | 58.04            | 43.88            | 66.02            | 56.53            |
| GuardReasoner-VL-Eco 3B   | 73.47            | 88.58            | 70.87               | 89.04              | 99.50                | 89.16            | 78.43            | 66.79            | 85.82            | 77.39            |
| GuardReasoner-VL 3B       | 74.45            | 89.10            | 70.83               | 88.79              | 99.50                | 88.92            | 78.77            | 70.93            | 86.47            | 78.73            |
| GuardReasoner-VL-Eco 7B   | 76.26            | 98.73            | 70.82               | 90.34              | 99.50                | 88.54            | 79.82            | 64.84            | 85.26            | 77.49            |
| GuardReasoner-VL 7B       | 76.51            | 98.30            | 70.98               | 90.13              | 98.99                | 88.35            | 79.88            | 70.84            | 85.60            | 79.07            |

across benchmarks to evaluate the performance. 'Average (Text)' is the average performance on text guardrail benchmarks. 'Average (All)' is the average performance on all guardrail benchmarks, including text, image, and text-image guardrail benchmarks. We do not evaluate response harmfulness in the image modality, as VLM responses are absent in the collected image benchmark.

Baseline . Since the used benchmarks contain text, image, and text-image inputs, we compare our model with both LLM guard models (LLaMA Guard 7B [28], LLaMA Guard 2 8B [16], LLaMA Guard 3 8B, Aegis Guard Defensive 7B, Aegis Guard Permissive 7B [19], Aegis Guard 2.0 8B [20], ShieldGemma 2B, ShieldGemma 9B [95], HarmBench LLaMA 13B, HarmBench Mistral 7B [59], MD-Judge 7B [39], BeaverDam 7B [29], WildGuard 7B [25]) and VLM guard models (LLaMA Guard 3-Vision [10], OpenAI Moderation API [58], Azure Content Safety API [3]). For Azure Content Safety API, we use text moderation for the text inputs, image moderation for image inputs, and multimodal moderation for text-image inputs. We did not compare with [30], as their models were not fully released at the time of our work.

## 3.1 Performance

The performance is shown in Table 1 (prompt harmfulness detection) and Table 2 (response harmfulness detection). In Figure 1 ('Average (All)' metric) and Figure 8 ('Average (Text)' metric), we show the average performance of these two tasks. From the results, we draw 4 findings. 1) LLM guard models, limited to text inputs, underperform on image and text-image modalities, yielding unpromising average performance. 2) Existing VLM guard models, typically trained as pure classifiers on text-image pairs, struggle with image-only moderation. 3) Our models achieve the best performance by learning to reason for moderation across modalities. 4) Our models achieve comparable performance on text guardrail benchmarks with the state-of-the-art LLM guard models.

## 3.2 Ablation Study

This section verifies the effectiveness of modules in GuardReasoner-VL. As shown in Figure 6, we conduct ablation studies on 3B and 7B models over the prompt harmfulness detection task. They are grouped into two stages, including the reasoning SFT stage and the online RL stage.

First, at the reasoning SFT stage, 'SFT' denotes conducting supervised fine-tuning on the collected multimodal data (text, images, text-image pairs) without reasoning processes. 'R-SFT (Text)' denotes conducting SFT on the collected text data with reasoning processes. 'R-SFT (Image)' denotes conducting SFT on the collected image data with reasoning processes. 'R-SFT (T-I)'

Table 2: F1 score (%) of 25 Models on 6 Benchmarks of Response Harmfulness Detection. The bold and underlined values denote the best and the runner-up. '-' denotes the result is unavailable.

| Method                    | HarmBench        | SafeRLHF         | BeaverTails      | XSTestReponse    | WildGuard Test   | Average (Text)   | SPA-VL -Eval     | Average (All)    |
|---------------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| LLM Guard Models          | LLM Guard Models | LLM Guard Models | LLM Guard Models | LLM Guard Models | LLM Guard Models | LLM Guard Models | LLM Guard Models | LLM Guard Models |
| LLaMA Guard 7B            | 52.00            | 48.40            | 67.10            | 82.00            | 50.50            | 58.27            | 00.00            | 41.07            |
| LLaMA Guard 2 8B          | 77.80            | 51.60            | 71.80            | 90.80            | 66.50            | 66.99            | 00.00            | 47.22            |
| LLaMA Guard 3 8B          | 85.07            | 44.36            | 67.84            | 87.67            | 70.80            | 64.97            | 00.00            | 45.79            |
| Aegis Guard Defensive 7B  | 62.20            | 59.30            | 74.70            | 52.80            | 49.10            | 62.79            | 00.00            | 44.25            |
| Aegis Guard Permissive 7B | 60.80            | 55.90            | 73.80            | 60.40            | 56.40            | 63.55            | 00.00            | 44.79            |
| Aegis Guard 2.0 8B        | -                | -                | -                | 86.20            | 77.50            | -                | 00.00            | -                |
| ShieldGemma 2B            | 35.36            | 16.92            | 30.97            | 65.55            | 20.13            | 27.24            | 00.00            | 19.20            |
| ShieldGemma 9B            | 56.44            | 47.07            | 63.61            | 73.86            | 47.00            | 55.67            | 00.00            | 39.24            |
| HarmBench LLaMA 13B       | 84.30            | 60.00            | 77.10            | 64.50            | 45.70            | 65.49            | 00.00            | 46.16            |
| HarmBench Mistral 7B      | 87.00            | 52.40            | 75.20            | 72.00            | 60.10            | 66.70            | 00.00            | 47.01            |
| MD-Judge 7B               | 81.60            | 64.70            | 86.70            | 90.40            | 76.80            | 78.67            | 00.00            | 55.45            |
| BeaverDam 7B              | 58.40            | 72.10            | 89.90            | 83.60            | 63.40            | 76.60            | 00.00            | 53.99            |
| WildGuard 7B              | 86.30            | 64.20            | 84.40            | 94.70            | 75.40            | 77.95            | 00.00            | 54.94            |
| GuardReasoner 1B          | 84.75            | 68.39            | 85.84            | 90.12            | 74.81            | 79.06            | 00.00            | 55.72            |
| GuardReasoner 3B          | 85.66            | 69.02            | 86.72            | 91.36            | 79.70            | 80.80            | 00.00            | 56.95            |
| GuardReasoner 8B          | 85.47            | 70.04            | 87.60            | 94.34            | 78.20            | 81.22            | 00.00            | 57.24            |
| VLM Guard Models          | VLM Guard Models | VLM Guard Models | VLM Guard Models | VLM Guard Models | VLM Guard Models | VLM Guard Models | VLM Guard Models | VLM Guard Models |
| OpenAI Moderation API     | 20.60            | 10.10            | 15.70            | 46.60            | 16.90            | 16.68            | 47.21            | 25.69            |
| Azure Content Safety API  | 44.16            | 36.56            | 51.52            | 57.80            | 38.12            | 44.47            | 39.35            | 42.96            |
| LLaMA Guard 3 Vision 11B  | 80.95            | 41.72            | 64.98            | 81.08            | 56.51            | 59.28            | 41.43            | 54.01            |
| Qwen2.5-VL-Instruct 3B    | 62.14            | 64.71            | 73.30            | 31.40            | 29.79            | 58.05            | 52.84            | 56.51            |
| Qwen2.5-VL-Instruct 7B    | 65.21            | 59.73            | 77.29            | 47.06            | 42.21            | 62.25            | 60.00            | 61.58            |
| GuardReasoner-VL-Eco 3B   | 84.72            | 66.96            | 85.39            | 93.59            | 77.39            | 79.31            | 72.01            | 77.14            |
| GuardReasoner-VL 3B       | 85.76            | 66.37            | 85.16            | 93.08            | 76.07            | 78.83            | 71.19            | 76.56            |
| GuardReasoner-VL-Eco 7B   | 86.22            | 66.15            | 85.51            | 93.33            | 78.60            | 79.51            | 70.81            | 76.94            |
| GuardReasoner-VL 7B       | 87.22            | 66.37            | 84.76            | 92.72            | 79.04            | 79.42            | 73.22            | 77.58            |

Figure 6: Ablation Studies of 3B (left) and 7B Models (right) on Prompt Harmfulness Detection. Y-axis denotes F1 score (%), and X-axis denotes model variants in reasoning SFT and online RL.

<!-- image -->

denotes conducting SFT on the collected text-image data with reasoning processes. 'R-SFT' denotes conducting SFT on our GuardReasoner-VLTrain data. We have the conclusions as follows. 1) The reasoning processes help the model achieve better performance, e.g., 'R-SFT' outperforms 'SFT'. 2) Each modality of the reasoning data contributes to the performance improvement. However, SFT on images alone degrades the textual capability of the model, leading to unpromising performance.

Second, at the online RL stage, 'Ours' denotes our GuardReasoner-VL model. 'w/o Aug.' denotes our model without safety-aware data augmentation. 'w/o Dyn.' denotes our model without the dynamic clipping strategy. 'w/o Len. Reward' denotes our model without the length term in the reward. We find that 1) Each design contributes to the performance improvement. 2) GuardReasonerVL achieves the best performance, showing the effectiveness of the combination of these designs. Similar conclusions hold for the response harmfulness detection task, as shown in Figure 9.

Table 3: Performance and Token Costs of GuardReasoner-VL and GuardReasoner-VL-Eco. The F1 score is averaged over the prompt harmfulness detection and response harmfulness detection.

|                      | 3B           | 3B            | 7B           | 7B            |
|----------------------|--------------|---------------|--------------|---------------|
| Model                | F1 Score (%) | Output Tokens | F1 Score (%) | Output Tokens |
| GuardReasoner-VL     | 77.65        | 213.32        | 78.33        | 208.33        |
| GuardReasoner-VL-Eco | 77.27        | 187.30        | 77.22        | 180.08        |
| Relative Change      | 0.48% ↓      | 12.20% ↓      | 1.42% ↓      | 13.56% ↓      |

Figure 7: Response Length and Reward During Training of Our Models.

<!-- image -->

## 3.3 Token Efficiency

Although our reasoning-based VLM guard models achieve promising performance, their multi-step reasoning process incurs higher token consumption, which increases moderation latency. To mitigate this issue, we set a constraint parameter β = 1 6 in Formula (10), developing a more token-efficient [50] variant, termed GuardReasoner-VL-Eco. As shown in Table 3, this variant achieves comparable performance (1% ∼ 2% F1 score drops) while reducing around 10% token usage.

## 3.4 Analyses

Training Process . We analyze the training process of our models. As shown in Figure 7, we visualize the training curves of GuardReasoner-VL 7B and GuardReasoner-VL-Eco 7B. We observe that GuardReasoner-VL 7B tends to increase its response length to gain higher rewards. In contrast, GuardReasoner-VL-Eco 7B initially increases the length slightly but soon stabilizes, still achieving competitive rewards under the imposed constraint.

Case Study . To further verify the effectiveness of our proposed GuardReasoner-VL, we conduct case studies on our GuardReasoner-VL 7B and 'Qwen2.5-VL-Instruct 7B + SFT'. 'Qwen2.5-VL-Instruct 7B + SFT' denotes conducting SFT on the collected multimodal data (text, images, text-image pairs) without reasoning processes for the Qwen2.5-VL-Instruct 7B model. The cases are demonstrated in Figure 12 (text input data), Figure 13 (image input data), and Figure 14 (text-image input data). From these cases, we observe that GuardReasoner-VL can accurately identify harmful content in both user requests and AI responses. Also, it can effectively infer the underlying reasons for its predictions.

## 3.5 Discussions on Over-relying on Text Data

We conducted experiments by separating the different modalities of the training data (see Figure 5). From these experimental results, we can find that R-SFT (which trains the model with text, image, and text-image) can achieve better performance than R-SFT (text). This suggests that introducing the image modality can enhance performance, demonstrating that the model's performance doesn't over-rely on the textual modality.

Besides, we also analyze the performance of our model on data with different modalities. For example, in Table 1, our model can achieve promising performance on HarmImageTest (which only contains images) and SPA-VL-Eval (which only contains image-text pairs). These results also verify that our model has the generalization ability to image and text-image modalities, and our model doesn't have the risk of over-relying on the textual modality.

## 4 Related Work

## 4.1 Vision-Language Models

Large language models [76] have achieved remarkable success in real-world applications like code intelligence [56, 54, 55]. Motivated by their success, Vision-language models (VLMs) are developed to extend the strong ability of LLMs to process both visual and textual information. The pioneer models like Flamingo [1], CLIP [67], and the BLIP series [36, 37] aim to align the visual encoders and LLMs in the latent space. Then, LLaVA is [44] proposed to construct the visual instruction data and conduct visual instruction tuning. This visual instruction tuning pipeline has become mainstream, and researchers [6, 45] pay attention to the construction of visual instruction data. Besides, anyresolution methods [8, 46] enable VLMs to handle images with any resolutions and ratios, improving the adaptability of VLMs in real-world applications. More recently, state-of-the-art open-sourced VLMs such as the LLaVA series [46, 35], InternVL series [8, 7, 9], and QwenVL [4, 83, 90] series have significantly advanced the capabilities of vision-language understanding.

## 4.2 Safety of VLMs

Despite their impressive performance, current VLMs remain susceptible to manipulations and attacks [48, 81, 53, 41], posing substantial risks in safety-critical applications such as autonomous driving [57], robotic manipulation [31], and education [11]. To alleviate this problem, the 3H principle [2] (Helpful, Honest, and Harmless) provides a foundational guideline for constraining model behaviors. Safety alignment techniques are proposed to better align VLMs with human values and expectations [92]. For example, [51] implements the safety alignment of VLMs by training the additional safety modules. In addition, ADPO [87], Safe RLHF-V [30], and [40] enhance the safety alignment of VLMs via DPO [68], RLHF [63], and GRPO [70], respectively. Besides, open-sourced datasets [97, 30, 22] contributed to high-quality alignment data and benchmarks. Differently, [84, 18, 14, 47] propose to conduct safety alignment at inference time.

Although effective, safety alignment on the VLM itself compromises its capabilities in other dimensions, e.g., creativity, reasoning, and helpfulness. As an alternative, safeguarding methods [85, 72, 96, 60, 47] are proposed to perform content moderation, aiming to ensure the safety of VLMs without directly degrading VLMs' core abilities. Among these, one promising approach is to train a separate VLM-based guard model to moderate the inputs and outputs of the victim VLM. For example, based on LLaVA-OneVision [35] and the collected multimodal safety dataset, LLaVAGuard [26] is built to conduct large-scale dataset annotation and moderate the text-image models. However, it is merely designed to moderate the images rather than the text-image pairs. In addition, VLMGuard [15] is proposed to conduct malicious text-image prompt detection by leveraging the unlabeled user prompts. Moreover, LLaMA Guard 3-Vision [10] is developed to moderate both the text-image input and text output of VLMs via SFT. To improve the generalization ability, [30] presents Beaver-Guard-V by training a reward model and then applying reinforcement learning. Recently, GuardReasoner [49] has been proposed to enhance the performance, explainability, and generalization of the LLM guard model by guiding it to learn to reason. Motivated by its success, this paper develops a reasoning-based VLM guard model named GuardReasoner-VL.

## 5 Conclusion

This paper presents GuardReasoner-VL, a novel reasoning-based VLM guard model that moderates harmful multimodal inputs by first performing deliberative reasoning. To enable this, we construct a large-scale reasoning dataset, GuardReasoner-VLTrain, spanning diverse input modalities and complex safety cases. We further enhance the guard model via online reinforcement learning, leveraging a set of tailored techniques including safety-aware data concatenation, dynamic clipping, and a length-aware safety reward to balance safety performance and token efficiency. Extensive experiments demonstrate that GuardReasoner-VL significantly outperforms existing VLM guard models across multiple benchmarks. We hope our work offers a new direction for building interpretable, generalizable VLM guard models, and we release all data, code, and models to support future research. In the future, it is worthy building reasoning-based guard models for agentic systems [17].

## References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716-23736, 2022.
- [2] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.
- [3] Microsoft Azure. Azure ai content safety. https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety/, 2024.
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.
- [5] Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/ Deep-Agent/R1-V , 2025. Accessed: 2025-02-02.
- [6] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370-387. Springer, 2024.
- [7] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [8] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101, 2024.
- [9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185-24198, 2024.
- [10] Jianfeng Chi, Ujjwal Karn, Hongyuan Zhan, Eric Smith, Javier Rando, Yiming Zhang, Kate Plawiak, Zacharie Delpierre Coudert, Kartikeya Upasani, and Mahesh Pasupuleti. Llama guard 3 vision: Safeguarding human-ai image understanding conversations. arXiv preprint arXiv:2411.10414, 2024.
- [11] Zhendong Chu, Shen Wang, Jian Xie, Tinghui Zhu, Yibo Yan, Jinheng Ye, Aoxiao Zhong, Xuming Hu, Jing Liang, Philip S Yu, et al. Llm agents for education: Advances and applications. arXiv preprint arXiv:2503.11733, 2025.
- [12] Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe rlhf: Safe reinforcement learning from human feedback. arXiv preprint arXiv:2310.12773, 2023.
- [13] Google Deepmind. Gemini robotics brings ai into the physical world. https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/, 2025.
- [14] Yi Ding, Bolian Li, and Ruqi Zhang. Eta: Evaluating then aligning safety of vision language models at inference time. arXiv preprint arXiv:2410.06625, 2024.
- [15] Xuefeng Du, Reshmi Ghosh, Robert Sim, Ahmed Salem, Vitor Carvalho, Emily Lawton, Yixuan Li, and Jack W Stokes. Vlmguard: Defending vlms against malicious prompts via unlabeled data. arXiv preprint arXiv:2410.00296, 2024.

- [16] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [17] Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min Lin, and Tianyu Pang. Flowreasoner: Reinforcing query-level meta-agents. arXiv preprint arXiv:2504.15257, 2025.
- [18] Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, and Amrit Singh Bedi. Immune: Improving safety against jailbreaks in multi-modal llms via inference-time alignment. arXiv preprint arXiv:2411.18688, 2024.
- [19] Shaona Ghosh, Prasoon Varshney, Erick Galinkin, and Christopher Parisien. Aegis: Online adaptive ai content safety moderation with ensemble of llm experts. arXiv preprint arXiv:2404.05993, 2024.
- [20] Shaona Ghosh, Prasoon Varshney, Makesh Narsimhan Sreedhar, Aishwarya Padmakumar, Traian Rebedea, Jibin Rajan Varghese, and Christopher Parisien. Aegis2. 0: A diverse ai safety dataset and risks taxonomy for alignment of llm guardrails. In Neurips Safe Generative AI Workshop 2024, 2024.
- [21] Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. Figstep: Jailbreaking large vision-language models via typographic visual prompts. arXiv preprint arXiv:2311.05608, 2023.
- [22] Tianle Gu, Zeyang Zhou, Kexin Huang, Liang Dandan, Yixu Wang, Haiquan Zhao, Yuanqi Yao, Yujiu Yang, Yan Teng, Yu Qiao, et al. Mllmguard: A multi-dimensional safety evaluation suite for multimodal large language models. Advances in Neural Information Processing Systems, 37:7256-7295, 2024.
- [23] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14953-14962, 2023.
- [24] Eungyeom Ha, Heemook Kim, Sung Chul Hong, and Dongbin Na. Hod: A benchmark dataset for harmful object detection. arXiv preprint arXiv:2310.05192, 2023.
- [25] Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms. arXiv preprint arXiv:2406.18495, 2024.
- [26] Lukas Helff, Felix Friedrich, Manuel Brack, Patrick Schramowski, and Kristian Kersting. Llavaguard: Vlm-based safeguard for vision dataset curation and safety assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8322-8326, 2024.
- [27] Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Zachary Yahn, Yichang Xu, and Ling Liu. Safety tax: Safety alignment makes your large reasoning models less reasonable. arXiv preprint arXiv:2503.00555, 2025.
- [28] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674, 2023.
- [29] Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. Beavertails: Towards improved safety alignment of llm via a human-preference dataset. Advances in Neural Information Processing Systems, 36, 2024.
- [30] Jiaming Ji, Xinyu Chen, Rui Pan, Han Zhu, Conghui Zhang, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, et al. Safe rlhf-v: Safe reinforcement learning from human feedback in multimodal large language models. arXiv preprint arXiv:2503.17682, 2025.

- [31] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. arXiv preprint arXiv:2502.21257, 2025.
- [32] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [33] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13700-13710, 2024.
- [34] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes. Advances in neural information processing systems, 33:2611-2624, 2020.
- [35] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [36] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888-12900. PMLR, 2022.
- [37] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730-19742. PMLR, 2023.
- [38] Kaixin Li, Yuchen Tian, Qisheng Hu, Ziyang Luo, and Jing Ma. Mmcode: Evaluating multi-modal code large language models with visually rich programming problems, 2024.
- [39] Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, and Jing Shao. Salad-bench: A hierarchical and comprehensive safety benchmark for large language models. arXiv preprint arXiv:2402.05044, 2024.
- [40] Xuying Li, Zhuo Li, Yuji Kosuga, and Victor Bian. Optimizing safe and aligned language generation: A multi-objective grpo approach. arXiv preprint arXiv:2503.21819, 2025.
- [41] Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen. Images are achilles' heel of alignment: Exploiting visual vulnerabilities for jailbreaking multimodal large language models. In European Conference on Computer Vision, pages 174-189. Springer, 2024.
- [42] Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, Rui Pan, Haoxiang Wang, Wenbin Hu, Hanning Zhang, et al. Mitigating the alignment tax of rlhf. arXiv preprint arXiv:2309.06256, 2023.
- [43] Zi Lin, Zihan Wang, Yongqi Tong, Yangkun Wang, Yuxin Guo, Yujia Wang, and Jingbo Shang. Toxicchat: Unveiling hidden challenges of toxicity detection in real-world user-ai conversation. arXiv preprint arXiv:2310.17389, 2023.
- [44] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892-34916, 2023.
- [45] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296-26306, 2024.
- [46] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. URL https://llava-vl.github.io/blog/2024-01-30-llava-next/ .

- [47] Qin Liu, Fei Wang, Chaowei Xiao, and Muhao Chen. Vlm-guard: Safeguarding visionlanguage models via fulfilling safety alignment gap. arXiv preprint arXiv:2502.10486, 2025.
- [48] Yue Liu, Xiaoxin He, Miao Xiong, Jinlan Fu, Shumin Deng, and Bryan Hooi. Flipattack: Jailbreak llms via flipping. arXiv preprint arXiv:2410.02832, 2024.
- [49] Yue Liu, Hongcheng Gao, Shengfang Zhai, Jun Xia, Tianyi Wu, Zhiwei Xue, Yulin Chen, Kenji Kawaguchi, Jiaheng Zhang, and Bryan Hooi. Guardreasoner: Towards reasoning-based llm safeguards. arXiv preprint arXiv:2501.18492, 2025.
- [50] Yue Liu, Jiaying Wu, Yufei He, Hongcheng Gao, Hongyu Chen, Baolong Bi, Jiaheng Zhang, Zhiqi Huang, and Bryan Hooi. Efficient inference for large reasoning models: A survey. arXiv preprint arXiv:2503.23077, 2025.
- [51] Zhendong Liu, Yuanbi Nie, Yingshui Tan, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, and Bo Zheng. Safety alignment for vision language models. arXiv preprint arXiv:2405.13581, 2024.
- [52] Weimin Lyu, Lu Pang, Tengfei Ma, Haibin Ling, and Chao Chen. Trojvlm: Backdoor attack against vision language models. In European Conference on Computer Vision, pages 467-483. Springer, 2024.
- [53] Weimin Lyu, Jiachen Yao, Saumya Gupta, Lu Pang, Tao Sun, Lingjie Yi, Lijie Hu, Haibin Ling, and Chao Chen. Backdooring vision-language models with out-of-distribution data. arXiv preprint arXiv:2410.01264, 2024.
- [54] Yingwei Ma, Yue Liu, Yue Yu, Yuanliang Zhang, Yu Jiang, Changjian Wang, and Shanshan Li. At which training stage does code data help llms reasoning? arXiv preprint arXiv:2309.16298, 2023.
- [55] Yingwei Ma, Rongyu Cao, Yongchang Cao, Yue Zhang, Jue Chen, Yibo Liu, Yuchen Liu, Binhua Li, Fei Huang, and Yongbin Li. Swe-gpt: A process-centric language model for automated software improvement. Proceedings of the ACM on Software Engineering, 2 (ISSTA):2362-2383, 2025.
- [56] Yingwei Ma, Qingping Yang, Rongyu Cao, Binhua Li, Fei Huang, and Yongbin Li. Alibaba lingmaagent: Improving automated issue resolution via comprehensive repository exploration. In Proceedings of the 33rd ACM International Conference on the Foundations of Software Engineering, pages 238-249, 2025.
- [57] Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, and Chaowei Xiao. Dolphins: Multimodal language model for driving. In European Conference on Computer Vision, pages 403-420. Springer, 2024.
- [58] Todor Markov, Chong Zhang, Sandhini Agarwal, Florentine Eloundou Nekoul, Theodore Lee, Steven Adler, Angela Jiang, and Lilian Weng. A holistic approach to undesired content detection in the real world. In Proceedings of the AAAI Conference on Artificial Intelligence, 2023.
- [59] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249, 2024.
- [60] Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, and Srijan Kumar. Uniguard: Towards universal safety guardrails for jailbreak attacks on multimodal large language models. arXiv preprint arXiv:2411.01703, 2024.
- [61] OpenAI. Learning to reason with llms. https://openai.com/index/learning-to-reason-with-llms/, 2024.
- [62] OpenAI. Openai o3-mini. https://openai.com/index/openai-o3-mini/, 2024.

- [63] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730-27744, 2022.
- [64] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.
- [65] Shraman Pramanick, Shivam Sharma, Dimitar Dimitrov, Md Shad Akhtar, Preslav Nakov, and Tanmoy Chakraborty. Momenta: A multimodal framework for detecting harmful memes and their targets. arXiv preprint arXiv:2109.05184, 2021.
- [66] Yiting Qu, Xinyue Shen, Yixin Wu, Michael Backes, Savvas Zannettou, and Yang Zhang. Unsafebench: Benchmarking image safety classifiers on real-world and ai-generated images. arXiv preprint arXiv:2405.03486, 2024.
- [67] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748-8763. PmLR, 2021.
- [68] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728-53741, 2023.
- [69] Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models. arXiv preprint arXiv:2308.01263, 2023.
- [70] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [71] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.
- [72] Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, and Chaowei Xiao. Safeguarding vision-language models against patched visual prompt injectors. arXiv preprint arXiv:2405.10529, 2024.
- [73] Claude Team. Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku. https://www.anthropic.com/news/3-5-models-and-computer-use, 2024.
- [74] Deepseek Team. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [75] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.
- [76] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.
- [77] OpenAI Team. Introducing deep research. https://openai.com/index/introducing-deep-research/, 2025.
- [78] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [79] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [80] Bertie Vidgen, Nino Scherrer, Hannah Rose Kirk, Rebecca Qian, Anand Kannappan, Scott A Hale, and Paul Röttger. Simplesafetytests: a test suite for identifying critical safety risks in large language models. arXiv preprint arXiv:2311.08370, 2023.
- [81] Cheng Wang, Yue Liu, Baolong Li, Duzhen Zhang, Zhongzhi Li, and Junfeng Fang. Safety in large reasoning models: A survey. arXiv preprint arXiv:2504.17704, 2025.
- [82] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=QWTCcxMpPA .
- [83] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [84] Pengyu Wang, Dong Zhang, Linyang Li, Chenkun Tan, Xinghao Wang, Ke Ren, Botian Jiang, and Xipeng Qiu. Inferaligner: Inference-time alignment for harmlessness through cross-model guidance. arXiv preprint arXiv:2401.11206, 2024.
- [85] Yu Wang, Xiaogeng Liu, Yu Li, Muhao Chen, and Chaowei Xiao. Adashield: Safeguarding multimodal large language models from structure-based attack via adaptive shield prompting. In European Conference on Computer Vision, pages 77-94. Springer, 2024.
- [86] Ziao Wang, Yuhang Li, Junda Wu, Jaehyeon Soon, and Xiaofeng Zhang. Finvis-gpt: A multimodal large language model for financial chart analysis. arXiv preprint arXiv:2308.01430, 2023.
- [87] Fenghua Weng, Jian Lou, Jun Feng, Minlie Huang, and Wenjie Wang. Adversary-aware dpo: Enhancing safety alignment in vision language models via adversarial training. arXiv preprint arXiv:2502.11455, 2025.
- [88] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024.
- [89] Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step, 2024. URL https://arxiv.org/abs/2411.10440 .
- [90] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [91] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.
- [92] Mang Ye, Xuankun Rong, Wenke Huang, Bo Du, Nenghai Yu, and Dacheng Tao. A survey of safety on large vision-language models: Attacks, defenses and evaluations. arXiv preprint arXiv:2502.14881, 2025.
- [93] J Diego Zamfirescu-Pereira, Richmond Y Wong, Bjoern Hartmann, and Qian Yang. Why johnny can't prompt: how non-ai experts try (and fail) to design llm prompts. In Proceedings of the 2023 CHI conference on human factors in computing systems, pages 1-21, 2023.
- [94] Eric Zeng, Tadayoshi Kohno, and Franziska Roesner. Bad news: Clickbait and deceptive ads on news and misinformation websites. In Workshop on Technology and Consumer Protection, pages 1-11, 2020.

- [95] Wenjun Zeng, Yuchi Liu, Ryan Mullins, Ludovic Peran, Joe Fernandez, Hamza Harkous, Karthik Narasimhan, Drew Proud, Piyush Kumar, Bhaktipriya Radharapu, et al. Shieldgemma: Generative ai content moderation based on gemma. arXiv preprint arXiv:2407.21772, 2024.
- [96] Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Ming Hu, Jie Zhang, Yang Liu, Shiqing Ma, and Chao Shen. Jailguard: A universal detection framework for llm prompt-based attacks. arXiv preprint arXiv:2312.10766, 2023.
- [97] Yongting Zhang, Lu Chen, Guodong Zheng, Yifeng Gao, Rui Zheng, Jinlan Fu, Zhenfei Yin, Senjie Jin, Yu Qiao, Xuanjing Huang, et al. Spa-vl: A comprehensive safety preference alignment dataset for vision language model. arXiv preprint arXiv:2406.12030, 2024.
- [98] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372 .
- [99] Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework. https:// github.com/hiyouga/EasyR1 , 2025.
- [100] Yucheng Zhou, Xiang Li, Qianning Wang, and Jianbing Shen. Visual in-context learning for large vision-language models. arXiv preprint arXiv:2402.11574, 2024.
- [101] Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales. Safety fine-tuning at (almost) no cost: A baseline for vision large language models. arXiv preprint arXiv:2402.02207, 2024.

## A Appendix

## A.1 Impact Statement

We introduce a guard model designed to enhance the safety of VLMs. By implementing this guard model, we aim to mitigate the potential risks and harmful impacts that VLMs may pose to society. The key aim of this paper is to demonstrate that the performance, explainability, and generalizability of the guard model can be improved by learning to reason. Inspired by this work, companies can build their own guard models for commercial use.

## A.2 Notations

We list the basic notations of this paper in Table 4.

Table 4: Basic Notations of This Paper.

| Notations   | Meanings                       | Notations          | Meanings                        |
|-------------|--------------------------------|--------------------|---------------------------------|
| F           | Victim VLM User Input          | D X D B s r l L RL | Reasoning Corpus for R-SFT      |
| X           |                                | new                | Augmented Use Input             |
| T           | Text Input                     | RL                 | Reasoning Corpus for RL         |
| I           | Image Input                    | M base             | Base Model                      |
| {T , I}     | Text-image Paired Input        | M R-SFT            | Trained Model via R-SFT         |
| S           | Response of Victim VLM         | G reasoner         | Reasoning-based VLM Guard Model |
| G           | VLM Guard Model                | L R-SFT            | Objective of R-SFT              |
| Q           | Instruction for Guardrail Task |                    | Dynamic Clipping Parameter      |
| R           | Reasoning Process              |                    | Overall Reward                  |
| ˆ Y         | Predicted Label                | norm               | Normalized Length of Reasoning  |
| Y           | Ground Truth                   |                    | Objective of RL                 |

## A.3 Datasets

We list the statistical information of the used benchmarks in Table 5.

We list statistics of GuardReasoner-VLTrain in Table 6.

## A.4 Additional Experiments

We show the average performance of our model on text guardrail benchmarks in Figure 8. We list the additional experiments regarding ablation studies in Figure 9.

<!-- image -->

- (a) Prompt Harmfulness Detection.

(b) Response Harmfulness Detection.

Figure 8: Mean Performance of GuardReasoner-VL on Text Guardrail Benchmarks.

Table 5: Statistics of 14 Benchmarks on 2 Guardrail Tasks.

| Guardrail Task               | Benchmark            | # Sample   | Input Modality   |
|------------------------------|----------------------|------------|------------------|
| Prompt Harmfulness Detection | ToxicChat            | 2,853      | Text Text Text   |
| Prompt Harmfulness Detection | OpenAIModeration     | 1,680      |                  |
| Prompt Harmfulness Detection | AegisSafetyTest      | 359        |                  |
|                              | SimpleSafetyTests    | 100        | Text             |
|                              | HarmBenchPrompt      | 239        | Text             |
|                              | WildGuardTest        | 1,756      | Text             |
|                              | HarmImageTest        | 3,295      | Image            |
|                              | SPA-VL-Eval          | 3,282      | Text-Image       |
| Response Harmfulness         | HarmBenchResponse    | 602        | Text             |
| Response Harmfulness         | SafeRLHF             | 2,000      | Text             |
| Detection                    | BeaverTails          | 3,021      | Text             |
| Detection                    | XSTestReponseHarmful | 446        | Text             |
| Detection                    | WildGuardTest        | 1,768      | Text             |
| Detection                    | SPA-VL-Eval          | 3,282      | Text-Image       |

## A.5 Implementation

## A.5.1 Baseline

We use the original codes of the baselines to replicate their results. We introduce the baselines and provide the implementation details as follows, including 16 LLM guard models and 5 guard models.

## LLMGuard Models

Figure 9: Ablation Studies of 3B (left) and 7B Models (right) on Response Harmfulness Detection. X-axis denotes model variants in reasoning SFT and online RL.

<!-- image -->

<!-- image -->

Table 6: Statistics of our Reasoning Corpus GuardReasoner-VLTrain.

| Modality   | # Sample   | # Step   |   Mean Step |   Mean Len. per Step |
|------------|------------|----------|-------------|----------------------|
| Text       | 63,799     | 353,440  |        5.54 |               163.25 |
| Image      | 13,267     | 57,322   |        4.32 |               154.03 |
| Text-Image | 46,030     | 221,033  |        4.8  |               160.79 |
| Overall    | 123,096    | 631,795  |        5.13 |               159.36 |

- LLaMA Guard 7B. LLaMA Guard 7B [28] is Meta's AI content guard model. It is instruct-tuned from the base model LLaMA 2 7B [79]. The training data is private and contains 13K samples.
- LLaMA Guard 2 8B. LLaMA Guard 2 8B is the second version of the LLaMA Guard series. It is based on LLaMA 3 8B [16]. They flip labels to conduct data augmentation on the training data.
- LLaMA Guard 3 8B. LLaMA Guard 3 8B is the third version of LLaMA Guard series. The base model is LLaMA 3.1 8B [16]. It supports 8 languages and has a context window of 128K tokens.
- Aegis Guard Defensive/Permissive 7B. They are developed by NVIDIA. It is based on LLaMA Guard 7B and uses LoRA to train the model. The defensive version classifies samples that need caution as harmful, and the permissive version classifies them as benign.
- Aegis Guard 2.0 8B. It is the second version of the Aegis Guard series. The base model is LLaMA 3.1-instruct 8B. Ghosh et al. [20] propose a new safety corpus with 12 top-level hazard categories.
- ShieldGemma 2B/9B. ShieldGemma 2B/9B is Google's AI content moderation model. It is based on Gemma 2 2B/9B [75] and targets on four harm categories: sexually explicit, dangerous content, hate, and harassment.
- HarmBench LLaMA 13B. HarmBench LLaMA 13B is based on LLaMA 2 13B [79]. The training data comes from GPT-4. It is used to evaluate jailbreak attacks in HarmBench [59].
- HarmBench Mistral 7B. HarmBench Mistral 7B is based on Mistral 7B [32]. The training data is constructed by prompting GPT-4. It is used to evaluate jailbreak attacks in HarmBench [59].
- MD-Judge 7B. MD-Judge 7B [39] is based on Mistral 7B [32]. The training data is private.
- BeaverDam 7B. BeaverDam 7B [29] is based on LLaMA 7B [78] and is instruction-tuned on BeaverTails training dataset [29].
- WildGuard 7B. WildGuard 7B is based on Mistral 7B [32]. It unifies the tasks of prompt/response harmfulness detection and refusal detection. They release the training data, WildGuardTrain.
- GuardReasoner 1B. WildGuard 1B is based on LLaMA 3.2 1B [16]. It is a reasoning-based LLM guard model. They release the reasoning corpus GuardReasonerTrain.

- GuardReasoner 3B. WildGuard 3B is based on LLaMA 3.2 3B [16]. It is a reasoning-based LLM guard model. They release the reasoning corpus GuardReasonerTrain.
- GuardReasoner 8B. WildGuard 8B is based on LLaMA 3.1 8B. It is a reasoning-based LLM guard model. They release the reasoning corpus GuardReasonerTrain.

## VLMGuard Models.

- OpenAI Moderation API. It [58] is a tool that automatically detects and filters harmful or inappropriate user-generated content using AI, helping developers maintain safe environments.
- Azure Content Safety API. The cloud-based Azure AI Content Safety API [3] provides developers with access to advanced algorithms for processing images and text and flagging content that is potentially offensive, risky, or otherwise undesirable.
- LLaMA Guard 3 Vision 11B. LLaMA Guard 3 Vision [10] is a LLaMA-3.2-11B pretrained model [16], fine-tuned for content safety classification. It can be used to safeguard content for both LLM inputs and LLM responses.
- Qwen2.5-VL-Instruct 3B/7B. Qwen2.5-VL-Instruct 3B/7B are fine-tuned for instructionfollowing, agent tool use, creative writing, and multilingual tasks across 100+ languages and dialects. We prompt them to finish VLM guardrail tasks.

## A.5.2 GuardReasoner-VL

We provide the implementation details of our proposed GuardReasoner-VL as follows.

( I ) In the R-SFT stage, we adopt 2 base VLM models with different scales, including Qwen2.5-VLInstruct 3B and Qwen2.5-VL-Instruct 7B. We use our synthesized GuardReasoner-VLTrain as the training data of R-SFT. It contains 123K samples with 631K reasoning steps. The chat template is set to qwen2\_vl. The cutoff length is set to 2048 tokens. The initial learning rate is set to 5e-05, and we use the cosine learning rate scheduler. We use the BFloat16 training, and we adopt the full-parameter fine-tuning. We adopt AdamW optimizer. The number of epochs is set to 3. The total batch size is set to 192 = 8( accumulate step ) × 6( batch size ) × 4( device ) . The DeepSpeed stage is set to 3.

( II ) In the online RL stage, we first perform rejection sampling. We generate 4 candidate responses using temperature = 1 . 0 and top\_p = 0 . 95 , and retain only those hard samples where all responses are incorrect. Then, we perform data augmentation on these hard samples by randomly selecting pairs of the samples and conducting safety-aware data concatenation. We set the augmented samples to comprise 20% of the training data for online RL. We obtain training data for online RL, consisting of 12K samples. During training, the number of rollouts is set to 16 and temperature = 1 . 2 . The batch size of rollouts is set to 512. The batch size for the actor model is 256. The initial learning rate for the actor model is set to 1e-6, and the weight decay is set to 1e-2. The clipping ratio ϵ is set to 0.2. The length constrain β is set to 1 for GuardReasoner-VL, and 1 6 for GuardReasoner-VL-Eco.

## A.6 Additional Related Work

Reasoning Ability of VLMs. Recent advances in vision-language reasoning have enabled VLMs to tackle increasingly complex multimodal tasks, including math [82], code [38], and agent systems [88]. Early efforts focused on eliciting reasoning capabilities through improved visual encoding strategies [33], task-specific modules [23], in-context learning [100], and prompt tuning [93]. More recently, inspired by models such as OpenAI o1/o3 [61, 62] and DeepSeek R1 [74], researchers have shifted toward actively incentivizing VLMs to learn the reasoning processes [89, 91, 5, 64, 71].

## A.7 Case Studies

The cases are demonstrated in Figure 12 (text input data), Figure 13 (image input data), and Figure 14 (text-image input data).

## A.8 Limitations

Although the proposed GuardReasoner-VL achieves promising performance, the token efficiency is still limited. In the future, we could solve this problem via techniques like model merge, agentic router, pruning, etc. Besides, for the attacks in the wild, e.g., indirect attacks in the environment for a

Figure 10: Prompt for Reasoning Data Synthesis.

<!-- image -->

Figure 11: Instruction, Input, and Output for Reasoning SFT.

<!-- image -->

computer-use agent, our models may achieve unpromising performance. In the future, it is worthy developing the reasoning-based guard models for the computer-use agents or multi-agent systems.

Figure 12: Case Study on Text Input Data . This case is sampled from WildGuardTest [25].

<!-- image -->

Figure 13: Case Study on Image Input Data . This case is sampled from HatefulMemes [34].

<!-- image -->

Figure 14: Case Study on Text-Image Input Data . This case is sampled from SPA-VL-Eval [97].

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We claim the main contributions in abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the efficient limitation of our proposed method.

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

Justification: N.A.

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

Justification: See the code and details.

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

Justification: We release the code and data.

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

Justification: We provide the details.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The improvement is significant.

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

Justification: We provide the details for computing resources.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We follow the code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the impact of our model.

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

Justification: N.A.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: N.A.

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

## Answer: [NA]

Justification: N.A.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [Yes]

Justification: We provide the details.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: N.A.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: N.A.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.