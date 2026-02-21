## MJ-VIDEO: Benchmarking and Rewarding Video Generation with Fine-Grained Video Preference

Haibo Tong 1 , 2 ∗ Zhaoyang Wang 1 ∗ Zhaorun Chen 3 Haonian Ji 1 Shi Qiu 1 Siwei Han 1 Kexin Geng 1 Zhongkai Xue 4 Yiyang Zhou 1 Peng Xia 1 Mingyu Ding 1 Rafael Rafailov 5 Chelsea Finn 5 Huaxiu Yao 1

1 UNC-Chapel Hill

2 UIUC 3 UChicago 4 University of Oxford 5 Stanford University haibot2@illinois.edu

huaxiu@cs.unc.edu

## Abstract

Recent advancements in video generation have significantly improved the ability to synthesize videos from text instructions. However, existing models still struggle with key challenges such as instruction misalignment, content hallucination, safety concerns, and generation bias. To address these limitations, we introduce MJBENCH-VIDEO, a large-scale video preference benchmark designed to evaluate video generation across five critical aspects: Alignment, Safety, Fineness, Coherence &amp; Consistency, and Bias &amp; Fairness . This benchmark further incorporates 28 fine-grained criteria to provide a comprehensive evaluation of video preference. Building upon this dataset, we propose MJ-VIDEO, a Mixture-of-Experts (MoE)based video reward model designed to deliver fine-grained reward. MJ-VIDEO can dynamically select relevant experts to accurately judge the preference based on the input text-video pair. This architecture enables more precise and adaptable preference judgments. Through extensive benchmarking on MJ-BENCH-VIDEO, we analyze the limitations of existing video reward models and demonstrate the superior performance of MJ-VIDEO in video preference assessment, achieving 17.58% and 15.87% improvements in overall and fine-grained preference judgments, respectively. Additionally, MJ-VIDEO is able to improve the alignment performance in video generation via preference fine-tuning.

Warning: this paper contains content that may be inappropriate or offensive.

## 1 Introduction

Recent advancements in video generation have significantly improved the quality of generated videos from text instructions [30, 55, 2]. However, these models still face major challenges, including imprecise adherence to instructions [18, 26], content hallucinations [40, 12], and the generation of unsafe or biased outputs [36, 11]. To address these challenges, recent approaches have introduced multi-modal reward models that evaluate generated videos [16, 53], which can then be leveraged in RLHF for better alignment [41, 57, 20]. However, these evaluations are often limited to overall alignment assessments, lacking the flexibility to accommodate diverse alignment objectives across different use cases [54, 30, 50, 34]. For instance, ensuring content coherence is more critical for sports videos, whereas safety considerations are paramount for cartoon videos. The lack of highquality video preference data with fine-grained assessments further hinders the development of more advanced video reward models [16, 14].

To address this issue, as illustrated in Figure 1, we introduce MJ-BENCH-VIDEO, a large-scale video preference benchmark comprising five evaluation aspects: Alignment , Safety , Fineness , Coherence and Consistency (C&amp;C) , and Bias and Fairness (B&amp;F) [7, 42], where each aspect represents a distinct aspect of preference evaluation. Additionally, we provide fine-grained annotations for these five

∗ Equal contribution.

Figure 1: MJ-BENCH-VIDEO is a comprehensive and fine-grained large-scale video preference dataset, which includes five aspects: Alignment , Safety , Fineness , Coherence and Consistency (C&amp;C), and Bias and Fairness (B&amp;F). Each aspect contains multiple detailed criteria to facilitate a thorough preference evaluation from different perspectives.

<!-- image -->

aspects, covering a total of 28 criteria to enhance comprehensiveness in video judgments. MJBENCH-VIDEO is designed to serve as a comprehensive benchmark for evaluating the judgment capabilities of video reward models and facilitating the development of more advanced video reward models in the future.

Building upon this dataset, we propose MJ-VIDEO, a Mixture-of-Expert (MoE) [4] based lightweight 2B video reward model that aims at providing comprehensive judgment by decomposing video assessment into the aforementioned five aspects. Specifically, we expect to train specialized experts to handle each aspect, delivering precise evaluations tailored to that specific subset. However, in a more realistic scenario, videos are often not well categorized, which may bring additional efforts in the expert selection process [35, 60]. Inspired by the success of Wang et al. [44], we adopt the gating network to automatically select proper reward objectives based on the input video and instruction. This gating network can serve as a router to ensure that the judgments are consistently aligning with different objectives required by various video generation scenarios.

In summary, the primary contributions of this paper are MJ-BENCH-VIDEO and MJ-VIDEO. MJBENCH-VIDEO is a high-quality, large-scale video preference benchmark designed to comprehensively evaluate video reward models across five key aspects, covering a total of 28 fine-grained criteria. MJ-VIDEO is a MoE-based video reward model that delivers fine-grained judgments, capturing diverse video preferences and aligning with different objectives required in various video generation scenarios. In our experiments, we first use MJ-BENCH-VIDEO to benchmark existing large vision language models (LVLMs)-based video judges, assessing their judgment capabilities across multiple aspects. The results reveal significant room for improvement in judging videos. We then show that MJ-VIDEO outperforms existing video reward models, achieving 17.58% and 15.87% improvements in overall and fine-grained video preference judgments, respectively, demonstrating its effectiveness in providing precise evaluations. Finally, we show that incorporating MJ-VIDEO for preference tuning in video generation improves the alignment performance of text-to-video generation models.

## 2 MJ-BENCH-VIDEO Benchmark

In this section, we introduce MJ-BENCH-VIDEO, a comprehensive video preference benchmark that incorporates fine-grained annotations through a multidimensional analysis of preference judgments. Building on insights from MJ-Bench [7], which focuses on text-to-image generation, we examine user expectations across common video generation scenarios. As illustrated in Figure 1, our analysis identifies five key benchmarking aspects: (1) Alignment , (2) Safety , (3) Fineness , (4) Coherence &amp;

Consistency , and (5) Bias &amp; Fairness . To enable more granular assessments and facilitate interpretable evaluations, we further introduce 28 fine-grained evaluation criteria. Below, we first provide an overview of evaluation aspect objectives and then outline the benchmark curation process.

## 2.1 Overview of Evaluation Aspect Objectives

Alignment. Alignment assesses how accurately the generated videos follow the given instructions, including the presence of specified objects and the correctness of attributes like color and shape.

Safety. Safety focuses on detecting inappropriate content, including illegal activities, disturbing or offensive material, politically sensitive topics, and other unsuitable elements.

Fineness. This evaluation focuses on the level of detail and refinement in the video's visual presentation. A high degree of fineness is characterized by sharpness, clarity, and well-preserved textures, with minimal artifacts such as blurring or pixelation. Additionally, smooth transitions, appropriate lighting, and natural color representation contribute to a visually polished and highquality appearance.

Figure 2: Three-stage curation process of MJBENCH-VIDEO.

<!-- image -->

Coherence and Consistency (C &amp; C). Coherence and Consistency evaluation examines the internal coherence of the video content. It includes an evaluation of the stability of spatial relationships, continuity of actions, and the consistent appearance of objects, backgrounds, and other visual elements throughout the video.

Bias and Fairness (B &amp; F). We assess the videos to ensure they are free from potential biases, particularly in the representation of different racial, gender, and age groups.

## 2.2 Benchmark Curation

The MJ-BENCH-VIDEO benchmark curation process comprises three stages: data collection, filtering, and annotation. Figure 2 provides an overview of this process, with additional details in Appendix B.2.

## 2.2.1 Data Collection

In the data collection stage, we employ three main strategies to collect video pairs and their corresponding prompts for video generation:

- Existing Video Preferences. We collect video preference pairs and corresponding prompts from Safesora [14], which capture human preferences for text-to-video generation tasks in terms of helpfulness and harmlessness.
- Generating Video Preference Pairs from Image Preference Pairs (I2V). In the I2V strategy, we first select image preference pairs and corresponding prompts from two image preference datasets with fine-grained annotations: MJ-BENCH [7] and HPDv2 [51]. These image pairs are then converted into video pairs using Stable Video Diffusion [3]. Next, the videos generated from the preferred images, along with the original prompts, are provided to ChatGPT to regenerate prompts tailored to the video pairs. This process ensures that the generated videos remain well-aligned with their prompts.
- Directly Generating Video Preference Pairs from Text Prompts (T2V). In the T2V strategy, we collect text prompts from datasets including OpenVid [28], VidProM [49], and VidGen [37]. These prompts are then used to generate video pairs via models such as Open-Sora [59], VADER [31], Text-Video Diffusion [46], and InstructVideo [56].

Using the three strategies above, we collected a total of 42,809 video pairs and 34,157 prompts, comprising 20,000 videos and 10,000 prompts from existing video preference dataset, 31,010 videos and 15,505 prompts from the I2V strategy, and 34,608 videos and 8,652 prompts from the T2V strategy. The detailed data distribution is presented in Table 4 in Appendix. By integrating these diverse sources and processing pipelines, we ensure that the curated dataset is both robust and comprehensive.

Figure 3: The structure of MJ-VIDEO which builds upon a VideoLLM and consists of two stacked MoE layers. The first MoE layer is for aspect routing and the second one is for scoring each finegrained criteria. An overall score is also offered by weighting those scores.

<!-- image -->

## 2.2.2 Data Filtering

After collecting the video preference pairs, we apply further filtering to remove invalid pairs, leveraging both GPT-4 and human evaluation. First, we use GPT-4 to filter out data where the videos are entirely inconsistent with the prompts. Next, we prompt GPT-4, InternVL2-26B [10], and CogVLM2 [19] to score the videos across five aspects: Alignment, Safety, Fineness, Coherence &amp; Consistency, and Bias &amp; Fairness, using a scale from 1 to 10. A video preference pair is discarded if at least one video receives a score below 5 in all five aspects. Additionally, if both videos in a pair receive identical scores across all aspects, the pair is also filtered out. After the automated filtering step, human experts conduct a final review to remove video pairs of extremely poor quality and those that are overly similar.

Ultimately, MJ-BENCH-VIDEO comprises 5,421 data entries, including 10,842 videos and 5,421 prompts. Of these, 1,496 entries are sourced from existing video preference dataset, 1,910 entries are from image-to-video conversion, and 2,015 entries are from text-to-video generation.

## 2.2.3 Data Annotation

After filtering the raw data, human annotators label the dataset using the annotation tool described in Appendix A. Each annotation involves evaluating a prompt with its corresponding video pair. The annotation rubric consists of detailed scores across 28 criteria within five aspects, along with human preference assessments. Each video pair receives a total of 72 annotations.

The annotation process follows these steps: First, annotators carefully review the prompt and video pairs. For each aspect, they assign scores ('good", 'average", 'bad") at the aspect level before providing an overall aspect score. This results in 303,576 criteria scores and 54,210 aspect scores across the dataset. Next, they determine the preference per aspect by selecting 'video 1', 'video 2', or 'same,' contributing to 27,105 aspect preference results. Finally, after completing all evaluations, they select an overall preference for the video pair, leading to 5,421 overall preference results.

## 3 MJ-VIDEO Reward Model

Currently, RLHF or RLAIF for video generation models heavily rely on vision reward models to score sampled frames (i.e., image) [30, 55]. This approach only captures information related to an overall assessment of text-video alignment, and thereby is unable to provide effective feedback on other important aspects in video generation such as consistency, bias, and safety. To address this issue, building upon MJ-BENCH-VIDEO, we develop a mixture-of-expert (MoE) based video reward model, MJ-VIDEO, aiming to deliver highly accurate fine-grained video preference judgment.

## 3.1 Model Architecture

Judging video preferences is a highly complex task that requires evaluating multiple factors, including video generation quality, safety, and logical coherence. The diversity of these criteria makes it challenging for LVLMs to provide accurate assessments directly. To address this, we propose MJVIDEO, a MoE-based architecture designed to assess videos across different aspects. As illustrated in Figure 3, MJ-VIDEO builds upon VideoLLM and incorporates two stacked MoE layers: one for aspect routing and another for fine-grained criteria scoring. The first layer, Aspect MoE , routes each

text-video pair to the five aspects defined in our MJ-BENCH-VIDEO. The second layer, Criteria MoE , then assigns fine-grained scores to each criterion. Finally, we aggregate these scores using the aspect routing weights to compute a final preference score. Below, we detail the design of these two MoE layers:

Aspect MoE. We utilize InternVL2 [10], a lightweight 2B VideoLLM, to process and encode the input instruction-video pair, extracting the hidden state h of the last token as the feature representation. Next, we introduce the first layer, Aspect MoE, which routes the input into five predefined aspects using MoE-style scalarization [44]. Specifically, we incorporate an overall gating layer g , composed of shallow MLP layers, to generate non-negative weights that sum to 1. This results in the aspect routing weights, computed as: AR = softmax ( g ( h )) , where AR ∈ R 5 represents the normalized scores.

Criteria MoE. Next, to obtain scores for each fine-grained criterion, we introduce another MoE layer, Criteria MoE g ′ , along with a regression scoring layer f after the VideoLLM. The scoring layer projects the hidden feature h into 28 criteria scores, while the gating layer identifies the most relevant criteria for the given input instruction-video pair. For criteria associated with the five predefined aspects { U i } 5 i =1 , the scores C [ U i ] within each aspect are normalized as follows:

<!-- formula-not-decoded -->

where U i denotes the indices of the criteria corresponding to aspect i . The overall preference score OS is then computed by weighting the criteria scores C ∈ R 28 with the aspect routing scores AR:

<!-- formula-not-decoded -->

This overall preference score accounts for five aspects and their corresponding criteria, making it directly applicable to general preference tuning pipelines for enhancing the alignment of video generation.

## 3.2 Multi-Stage Training

We employ a three-stage training strategy to fine-tune the VideoLLM along with the newly introduced MoE parameters. Specifically, the first stage is to train the Criteria MoE layer to predict the annotated fine-grained criteria scores. The second stage is to leverage aspect ranking information from preference pairs to train the Aspect MoE layer. In the final stage, we integrate the previous training steps and introduce an overall preference ranking loss to jointly optimize both the aspect MoE layer and the criteria MoE layer. We detail the three-stage training as follows:

Stage I: Criteria Scoring Training. We use the fine-grained annotated criteria scores s ∈ R 28 as labels to train the Criteria MoE layer, ensuring accurate judgment:

<!-- formula-not-decoded -->

where D represents the training dataset. After training, MJ-VIDEO is expected to generate accurate scores for the fine-grained criteria.

Stage II: Aspect Routing Training. Next, we leverage the annotated aspect ranking information from video preference pairs to train the Aspect MoE. The ranking information for each aspect reflects preference between two generated videos ( y w , y l ) , given the same instruction x and its associated criteria. To optimize this, we apply a ranking loss:

<!-- formula-not-decoded -->

where I i is 1 if y w is preferred over y l in the i th aspect, and -1 otherwise. The term ∑ C [ U i ] from Eq. (1) represents the summed criteria scores within the i th aspect. Additionally, to prevent interference with criteria score predictions, we continue optimizing L 1 from Eq. (3) concurrently.

Stage III: Joint Training. Finally, to ensure the overall preference score is meaningful, we incorporate the overall ranking ( x, y w , y l ) , where y w is generally preferred over y l , to jointly train both MoE layers as follows:

<!-- formula-not-decoded -->

where the overall preference score OS is computed using Eq. (2). Additionally, we incorporate the losses L 1 and L 2 into the third-stage training and introduce a hyperparameter λ to balance them.

## 4 Experiment

In our experiments, we utilize the proposed MJ-BENCH-VIDEO and the corresponding reward model, MJ-VIDEO, to explore the following questions: (1) Can existing large vision-language models (LVLMs) or VideoLLMs effectively judge video preferences? (2) Does training on fine-grained preference annotations improve the performance of a video reward model? (3) Can introducing MJ-VIDEO into the preference tuning process improve the alignment of generated videos? (4) What is the advantage of adopting a MoE architecture in video preference judgment?

## 4.1 Experimental Setup

Table 1: Testing on aspect annotations in MJ-BENCH-VIDEO. The bolded numbers in the table represent the best results, while the underlined numbers indicate the second-best results. The "C&amp;C" in the table refers to "Coherence and Consistency," while 'B&amp;F" refers to "Bias and Fairness." In cases where certain models show strong bias, causing the F1 score to be NaN, a "/" is used in place of the result in the table. For preference comparison, we report the results of the 'strict" metric. See Appendix C for the 'tie-aware" metric results.

| Model            | Alignment   | Alignment   | Alignment   | Safety   | Safety   | Safety   | Fineness   | Fineness   | Fineness   | C&C   | C&C   | C&C    | B&F   | B&F   | B&F    |
|------------------|-------------|-------------|-------------|----------|----------|----------|------------|------------|------------|-------|-------|--------|-------|-------|--------|
| Model            | Acc         | F1          | strict      | Acc      | F1       | strict   | Acc        | F1         | strict     | Acc   | F1    | strict | Acc   | F1    | strict |
| InternVL2-2B     | 70.75       | 60.42       | 17.71       | 66.67    | 55.02    | 16.67    | 63.59      | 49.87      | 3.125      | 71.81 | 46.04 | 10.34  | 74.11 | 63.19 | 54.54  |
| InternVL2-4B     | 57.00       | 55.00       | 26.96       | 75.49    | 60.37    | 0.00     | 52.48      | 49.92      | 7.143      | 43.02 | 33.11 | 17.86  | 66.32 | 56.27 | 54.55  |
| InternVL2-8B     | 44.21       | 44.21       | 33.33       | 76.72    | 72.60    | 16.67    | 47.71      | 47.27      | 18.75      | 27.76 | 24.29 | 12.07  | 15.51 | 13.88 | 50.00  |
| InternVL2-26B    | 65.47       | 62.96       | 40.51       | 84.44    | 78.26    | 20.00    | 69.81      | 51.91      | 14.29      | 59.03 | 41.51 | 16.33  | 82.05 | 59.85 | 30.00  |
| Qwen2-VL-2B      | 54.28       | 53.03       | 19.35       | 59.82    | 56.93    | 25.00    | 56.75      | 51.86      | 3.448      | 37.90 | 31.18 | 16.39  | 20.00 | 19.31 | 38.46  |
| Qwen2-VL-7B      | 58.31       | 56.19       | 41.94       | 55.35    | 52.81    | 25.00    | 47.56      | 46.33      | 31.03      | 32.58 | 27.68 | 19.67  | 14.61 | 13.13 | 23.08  |
| MiniCPM-8B       | 65.53       | 61.38       | 48.72       | 72.91    | 67.22    | 40.00    | 62.13      | 56.02      | 39.29      | 49.73 | 37.21 | 31.25  | 15.12 | 14.17 | 60.00  |
| CogVLM2          | 26.71       | 23.80       | 7.692       | 31.67    | 30.09    | 16.67    | 35.61      | 29.79      | 11.76      | 7.87  | 7.86  | 4.615  | 14.61 | /     | 7.692  |
| Gemini-1.5-flash | 27.45       | 25.72       | 8.421       | 83.64    | 77.34    | 0.0      | 32.80      | 25.27      | 12.90      | 5.01  | 4.88  | 12.07  | 15.18 | /     | 9.091  |
| GPT-4o           | 58.27       | 56.21       | 50.00       | 82.86    | 77.00    | 50.00    | 59.67      | 56.34      | 27.27      | 44.52 | 34.17 | 40.00  | 19.17 | 18.48 | 33.33  |
| MJ-VIDEO         | 78.41       | 71.22       | 79.05       | 87.50    | 81.84    | 83.33    | 68.60      | 58.53      | 58.82      | 95.36 | 53.57 | 58.46  | 86.92 | 55.97 | 69.23  |

Dataset Split. We divide MJ-BENCH-VIDEO into a training set and a test set at a 4:1 ratio, leading to 4,336 training video pairs and 1,085 testing video pairs.

Existing Multimodal Judge Models. We benchmark several popular LVLMs, both open- and closedsource, for video preference judgment. Open-source models include InternVL2 [10], Qwen [47], and CogVLM2 [19], while closed-source models include GPT-4o [29] and Gemini [39]. To ensure stable scoring and reduce ambiguity, we follow Chen et al. [7] by prompting models to assign verbalized 10-range scores (e.g., 'Extremely Poor,' 'Very Good'). The top-5 scores are considered good, and the bottom-5 as bad. See Appendix B for details. Additionally, we evaluate VideoScore [16] on overall video preference, though it cannot perform aspect-level evaluations due to the absence of per-aspect results.

Evaluation Plans and Metrics. We conduct two types of evaluations:

Video Preference Evaluation. We evaluate both aspect-level and overall video preference using accuracy as the evaluation metric. In this evaluation, the judge model is given prompt-video pairs and tasked with assigning scores. The model's preference for each video pair is then determined by comparing the assigned scores.

Regarding the evaluation metric, many LVLMs often assign the same score to a pair of videos, making it challenging to accurately determine video preference. To address this, we adopt two accuracy calculation methods, resulting in two metrics. The first metric, strict , treats cases where the model fails to indicate a preference as incorrect. The second metric, tie-aware , considers identical scores as a partial match, awarding 0.5 when counting correct judgments.

Video Quality Evaluation. We assess video quality based on the assigned scores for each aspect and category in MJ-Bench. Given the potential imbalance in score distribution, we use accuracy (Acc) and F1 score as evaluation metrics.

## 4.2 Fine-Grained Video Quality and Preference Evaluation Results

In this section, we evaluate MJ-VIDEO alongside other multimodal judges for video quality and preference across aspects. The results are summarized in Table 1, with subcategory-level details provided in Appendix D.

Our findings reveal two key insights. First, existing multimodal judge models, both open- and closedsource, show significant room for improvement. Second, our 2B MJ-VIDEO model outperforms all alternatives across nearly all categories. Specifically, compared to models of similar size (e.g., InternVL2-2B, Qwen2-VL-2B), MJ-VIDEO improves accuracy by 20.12%, F1 score by 16.97%, and 51.67% higher in preference comparison. Notably, it even surpasses the 26B InternVL2 model, achieving a 15.52% higher accuracy, 9.05% higher F1 score, and 45.86% improvement in preference comparison. The only area where InternVL2-26B partially excel is fineness evaluation, which aligns with expectations, as larger models with more advanced visual encoders better capture fine-grained visual details.

MJ-VIDEO's superiority stems from two key factors. First, high-quality, fine-grained annotations enable training at both the aspect and subcategory levels, improving performance across all aspects. Second, its MoE architecture, leveraging a gating layer, effectively processes LVLM outputs by dynamically weighting criteria to generate aspect scores, benefiting from LVLM's semantic and video understanding.

## 4.3 Overall Video Preference Evaluation Results

Additional Dataset. To enhance the robustness of overall video preference evaluation, in addition to using MJ-BENCH-VIDEO, we incorporate two additional datasets: Safesora-test [14] and GenAIBench [24], both of which contain video preference pairs.

We present the evaluation results of all multimodal judge models in Table 2 and summarize the following observations. First, similar to the fine-grained analysis, there is room for improvement across these models. Second, MJ-VIDEO achieves the best test results on all datasets. Compared to the best baseline, MJVIDEO improves by 17.58% on MJ-BENCHVIDEO, 15.95% on Safesora-test, and 1.65% on GenAI-Bench. In contrast, while the InternVL performed well in fine-grained evaluations, they do not achieve similarly strong results in overall video preference evaluation. This aligns with our expectations, as assessing overall video preference lacks the detailed breakdown provided by aspect-level evaluation, making it more challenging for LVLMs to make precise judgments. In comparison, MJ-VIDEO leverages a gating layer to integrate judgments across different as-

Table 2: Results of overall video preference evaluation. The best test results are highlighted in bold, and the second-best results are underlined. Strict treats undecided cases as incorrect, while tie-aware assigns 0.5 for ties in calculating accuracy.

<!-- image -->

| Model         | MJ-BENCH-VIDEO   | MJ-BENCH-VIDEO   | Safesora-test   | Safesora-test   | GenAI-Bench   | GenAI-Bench   |
|---------------|------------------|------------------|-----------------|-----------------|---------------|---------------|
|               | strict           | tie-aware        | strict          | tie-aware       | strict        | tie-aware     |
| InternVL2-2B  | 5.93             | 47.88            | 4.60            | 50.30           | 13.71         | 55.43         |
| InternVL2-4B  | 13.55            | 49.15            | 11.74           | 50.91           | 39.00         | 61.79         |
| InternVL2-8B  | 16.95            | 47.88            | 14.29           | 53.09           | 36.85         | 62.43         |
| InternVL2-26B | 22.88            | 53.81            | 10.41           | 52.00           | 31.86         | 55.64         |
| Qwen-VL-2B    | 13.33            | 48.09            | 13.18           | 51.27           | 27.29         | 56.71         |
| Qwen-VL-7B    | 17.14            | 47.62            | 14.58           | 52.41           | 20.57         | 51.36         |
| MiniCPM       | 30.51            | 53.39            | 25.30           | 52.54           | 47.43         | 60.21         |
| CogVLM2       | 8.47             | 47.46            | 9.56            | 52.48           | 21.29         | 56.29         |
| VideoScore    | 58.47            | 58.47            | 55.33           | 55.51           | 69.14         | 69.14         |
| Gemini        | 2.66             | 48.67            | 2.66            | 48.67           | 21.45         | 50.71         |
| GPT-4o        | 35.35            | 54.6             | 35.35           | 54.6            | 48.85         | 59.14         |
| MJ-VIDEO      | 68.75            | 68.75            | 64.16           | 64.16           | 70.28         | 70.28         |

pects, enabling a comprehensive understanding of overall preference and contributing to its superior performance. Similarly, VideoScore, which also decomposes video preference, achieves the secondbest results.

This underscores the importance of fine-grained decomposition in enhancing the performance of video reward models. We further evaluate the generalization ability of MJ-VIDEO against recent large reasoning models such as OpenAI's o1 [23]. Details are reported in Appendix C.1.

## 4.4 MJ-VIDEO in Preference Alignment for Text-to-Video Generation

In this section, we introduce MJ-VIDEO as the reward model within the RLAIF framework to enhance video rewarding for generating preference-aligned videos, which are then used for preference fine-tuning of text-to-video (T2V) diffusion models. We select VideoCrafter2 [6] as the backbone T2V diffusion model and follow the VADER [32] framework, replacing its reward model with either VideoScore or MJ-VIDEO for preference fine-tuning. The training data is sourced from VidProM [49], from which we randomly sample 5,000 instances for training (see

Appendix F for experimental details). After fine-tuning, we conduct two types of evaluation: automated evaluation using VBench [21], assessing performance across four dimensions-image quality, human action, scene composition, and overall consistency-and human evaluation , where we sample 1,000 instances from VidProM to assess video quality and text-video alignment.

We present the results in Table 3, where we observe that the model fine-tuned with MJVIDEO as the reward model outperforms both VideoScore and the original VideoCrafter2 model in most evaluation aspects, highlighting its effectiveness in improving the alignment of generated videos with input instructions.

## 4.5 Ablation Study

In the ablation study, we examine the impact of the two stacked MoE layers on model performance. Specifically, we design two ablation models: (1) w/o Criteria MoE: replacing the

Table 3: Evaluation of video models across human and automated evaluation on VBench. Human evaluation assesses Video Quality and Text-to-Video Alignment. Automated evaluation on VBench evaluates Imaging Quality ( IQ ), Human Action ( HA ), Scene ( S ), and Overall Consistency ( OC ).

| Model         | Human Eval   | Human Eval   | Auto Eval (VBench)   | Auto Eval (VBench)   | Auto Eval (VBench)   | Auto Eval (VBench)   |
|---------------|--------------|--------------|----------------------|----------------------|----------------------|----------------------|
|               | Quality      | Align        | IQ                   | HA                   | S                    | OC                   |
| VideoCrafter2 | 56.30        | 68.80        | 67.04                | 90.00                | 54.00                | 28.39                |
| VideoScore    | 64.50        | 74.80        | 65.03                | 92.00                | 54.79                | 28.38                |
| MJ-VIDEO      | 69.90        | 79.20        | 67.89                | 94.00                | 55.09                | 28.19                |

MoE layers with a regression layer that maps the output of InternVL2-2B to aspect scores, and (2) w/o Aspect MoE: replacing the MoE layers with a regression layer that maps the output of InternVL2-2B to the overall score. We train and evaluate both ablation models, compare them wi th MJ-VIDEO, and present the results in Figure 4(a) (see the results per aspect in Figure 7 of Appendix E) and Figure 4(b), respectively.

According to the results, MJ-VIDEO outperforms 'w/o Criteria MoE," achieving improvements of 2.64%, 58.33%, and 12.45% in average accuracy, F1, and strict preference accuracy, respectively. The most notable gains are in 'Coherence and Consistency" and 'Bias and Fairness," where the model without Criteria MoE layer shows strong biases, failing to learn effectively from the training data. In contrast, MJ-VIDEO leverages the Criteria MoE layer to assign appropriate weights to each criterion, fully utilizing the LVLM's ability to understand video and semantics. Additionally, compared with 'w/o Aspect MoE", MJ-VIDEO achieves

Figure 4: (a): Compare MJ-VIDEO with 'w/o Criteria MoE", where average results of Acc, F1, and strict metrics are evaluated over 5 aspects; (b) Compare MJ-VIDEO with 'w/o Aspect MoE" on MJBENCH-VIDEO, Safesora-test and GenAI-Bench.

<!-- image -->

an average improvement of 5.45% across all three datasets, demonstrating the effectiveness of the Aspect MoE layer in enhancing overall preference modeling.

We provide more comprehensive ablation results in Appendix E, including detailed comparisons across evaluation aspects, training stage effectiveness, and the role of individual expert branches.

## 4.6 Case Study

In this section, we present two case study in Figure 5 to illustrate the advantages of MJ-VIDEO in video preference judgment, with additional cases provided in Appendix G. In the first case, MJVIDEO successfully identifies the ethereal bird as a key detail in the input instruction and incorporates it into the evaluation, resulting in a more accurate assessment. In contrast, VideoScore overlooks the ethereal bird and incorrectly rates the alignment as good, revealing its limitation in capturing fine-grained object features. This outcome aligns with our expectations, as MJ-VIDEO is trained with preference pairs emphasizing fine-grained details, enabling a more balanced evaluation of alignment and visual fidelity. In the second case, both videos align with human preferences. MJ-VIDEO assigns a higher score to the first video, while VideoScore gives both videos relatively high scores but fails to differentiate which one is better. This is because MJ-VIDEO is trained on pairwise data, allowing it to make a more precise relative preference judgment even when the two videos have similar quality.

In addition, we provide video demos to demonstrate (1) MJ-BENCH-VIDEO contains high quality video preference pairs at anonymous https://anonymous.4open.science/r/mj-video-neurips-364C/, and (2) MJ-VIDEO is able to improve video generation quality of VideoCrafter-V2 [6] at anonymous

Figure 5: Two cases of video preference analysis.

<!-- image -->

https://anonymous.4open.science/r/mj-video-neurips-364C/video\_demo.zip. Note that VideoCrafterV2 is one of SOTA open-source video generation models as demonstrated by EvalCrafter [27] and VBench [22] benchmarks. This model after fine-tuning with our reward model MJ-VIDEO has already effectively improved its generation quality, though it still falls short of proprietary models such as Sora. We will be able to provide more beautiful videos if we have the access and resources to fine-tune more powerful models, most of which are currently closed source.

## 5 Related Works

Multimodal Judge. Multimodal judges are critical for assessing alignment between different data types, like text and images [62, 53, 1, 9, 58, 43]. These include both CLIP-based [33] and LVLM-based [48, 38, 52] models. CLIP-based models (such as HPS-v2.1 [51] and PickScorev1 [25]) provide reliable evaluations through contrastive training, though their evaluation processes often lack transparency. In contrast, LVLM-based judges use prompting techniques and human preference data to give more transparent, flexible feedback [5, 16, 45], though they require more computational resources. These models are widely used in text-to-image [41, 7, 57] and image-to-text tasks [61, 8, 13]. However, their application to video remains limited, as maintaining temporal coherence adds complexity. While some studies have started investigated video-to-text generation feedback [15, 16], fewer have explored reward models for text-to-video generation and evaluating their rewarding capabilities [17, 57], especially on fine-grained video reward judgement.

Reward Model for Text-to-Video Generation. Dai et al. [14] introduced a preference dataset for text-to-video generation, but their approach does not involve developing a reward model for practical use. Similarly, Yuan et al. [57] repurposed a CLIP-based model to provide a scalar reward, though their method suffers from a lack of transparency in the evaluation process. He et al. [17] also made initial attempts with a CLIP-based solution, but it is constrained by limited transparency and a relatively small preference dataset. In contrast, we introduce a fine-grained video preference dataset, MJ-BENCH-VIDEO, which can be used to comprehensively evaluate the video reward models. Building upon this dataset, we further propose MJ-VIDEO, a MoE-based video reward model, aiming to provide more transparent preference judgments through fine-grained scores and provide aspect-specific evaluations.

## 6 Conclusion

In this paper, we introduce MJ-BENCH-VIDEO, a large-scale benchmark for evaluating video generation across five key aspects with 28 fine-grained criteria, addressing limitations in the existing video reward model evaluation. Building on this, we propose MJ-VIDEO, a Mixture-of-Experts (MoE)-based reward model that decomposes video assessments into specialized expert evaluations, enhancing precision and adaptability. Experimental results show that MJ-VIDEO outperforms existing models, highlighting the benefits of fine-grained, multi-aspect judgment. Together, MJ-BENCHVIDEO and MJ-VIDEO provide a robust framework for improving video generation alignment, offering a foundation for future advancements in multimodal reward modeling.

## References

- [1] Rohan Badlani, Adrian Łancucki, Kevin J. Shih, Rafael Valle, Wei Ping, and Bryan Catanzaro. One tts alignment to rule them all, 2021. URL https://arxiv.org/abs/2108.10447 .
- [2] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning, 2024. URL https://arxiv.org/abs/2305.13301 .
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023. URL https://arxiv.org/abs/2311.15127 .
- [4] Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. A survey on mixture of experts, 2024. URL https://arxiv.org/abs/2407.06204 .
- [5] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yinuo Liu, Yaochen Wang, Huichi Zhou, Qihui Zhang, Pan Zhou, Yao Wan, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-as-ajudge with vision-language benchmark. arXiv preprint arXiv:2402.04788 , 2024.
- [6] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024.
- [7] Zhaorun Chen, Yichao Du, Zichen Wen, Yiyang Zhou, Chenhang Cui, Zhenzhen Weng, Haoqin Tu, Chaoqi Wang, Zhengwei Tong, Qinglan Huang, et al. Mj-bench: Is your multimodal reward model really a good judge for text-to-image generation? arXiv preprint arXiv:2407.04842 , 2024.
- [8] Zhaorun Chen, Zhuokai Zhao, Hongyin Luo, Huaxiu Yao, Bo Li, and Jiawei Zhou. Halc: Object hallucination reduction via adaptive focal-contrast decoding. arXiv preprint arXiv:2403.00425 , 2024.
- [9] Zhaorun Chen, Zhuokai Zhao, Zhihong Zhu, Ruiqi Zhang, Xiang Li, Bhiksha Raj, and Huaxiu Yao. Autoprm: Automating procedural supervision for multi-step reasoning via controllable question decomposition. arXiv preprint arXiv:2402.11452 , 2024.
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238 , 2023.
- [11] Jaemin Cho, Abhay Zala, and Mohit Bansal. Dall-eval: Probing the reasoning skills and social biases of text-to-image generation models, 2023. URL https://arxiv.org/abs/ 2202.04053 .
- [12] Zhixuan Chu, Lei Zhang, Yichen Sun, Siqiao Xue, Zhibo Wang, Zhan Qin, and Kui Ren. Sora detector: A unified hallucination detection for large text-to-video models, 2024. URL https://arxiv.org/abs/2405.04180 .
- [13] Chenhang Cui, An Zhang, Yiyang Zhou, Zhaorun Chen, Gelei Deng, Huaxiu Yao, and Tat-Seng Chua. Fine-grained verifiers: Preference modeling as next-token prediction in vision-language alignment. arXiv preprint arXiv:2410.14148 , 2024.
- [14] Josef Dai, Tianle Chen, Xuyao Wang, Ziran Yang, Taiye Chen, Jiaming Ji, and Yaodong Yang. Safesora: Towards safety alignment of text2video generation via a human preference dataset, 2024. URL https://arxiv.org/abs/2406.14477 .
- [15] Alejandro Escontrela, Ademi Adeniji, Wilson Yan, Ajay Jain, Xue Bin Peng, Ken Goldberg, Youngwoon Lee, Danijar Hafner, and Pieter Abbeel. Video prediction models as rewards for reinforcement learning. Advances in Neural Information Processing Systems , 36, 2024.

- [16] Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, Kai Wang, Quy Duc Do, Yuansheng Ni, Bohan Lyu, Yaswanth Narsupalli, Rongqi Fan, Zhiheng Lyu, Yuchen Lin, and Wenhu Chen. Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. ArXiv , abs/2406.15252, 2024. URL https://arxiv.org/abs/2406.15252 .
- [17] Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, et al. Mantisscore: Building automatic metrics to simulate fine-grained human feedback for video generation. arXiv preprint arXiv:2406.15252 , 2024.
- [18] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers, 2022. URL https://arxiv.org/ abs/2205.15868 .
- [19] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500 , 2024.
- [20] Tao Huang, Guangqi Jiang, Yanjie Ze, and Huazhe Xu. Diffusion reward: Learning rewards via conditional video diffusion, 2024. URL https://arxiv.org/abs/2312.14134 .
- [21] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2024.
- [22] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 21807-21818, 2024.
- [23] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720 , 2024.
- [24] Dongfu Jiang, Max Ku, Tianle Li, Yuansheng Ni, Shizhuo Sun, Rongqi Fan, and Wenhu Chen. Genai arena: An open evaluation platform for generative models. arXiv preprint arXiv:2406.04485 , 2024.
- [25] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems , 36:36652-36663, 2023.
- [26] Chengxuan Li, Di Huang, Zeyu Lu, Yang Xiao, Qingqi Pei, and Lei Bai. A survey on long video generation: Challenges, methods, and prospects, 2024. URL https://arxiv.org/ abs/2403.16407 .
- [27] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 22139-22149, 2024.
- [28] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371 , 2024.
- [29] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, and Suchir Balaji et al. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774 .
- [30] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients.

- [31] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737 , 2024.
- [32] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients, 2024. URL https://arxiv.org/abs/2407. 08737 .
- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pages 8748-8763. PMLR, 2021.
- [34] Dian Shao, Yue Zhao, Bo Dai, and Dahua Lin. Finegym: A hierarchical video dataset for fine-grained action understanding, 2020. URL https://arxiv.org/abs/2004.06704 .
- [35] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, 2017. URL https://arxiv.org/abs/1701.06538 .
- [36] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data, 2022. URL https://arxiv.org/abs/ 2209.14792 .
- [37] Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, and Hao Li. Vidgen-1m: A large-scale dataset for text-to-video generation. arXiv preprint arXiv:2408.02629 , 2024.
- [38] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models, 2024. URL https://arxiv.org/abs/2405.09818 .
- [39] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, and David Silver et al. Gemini: A family of highly capable multimodal models, 2024. URL https://arxiv.org/abs/2312. 11805 .
- [40] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric &amp; challenges, 2019. URL https://arxiv.org/abs/1812.01717 .
- [41] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 8228-8238, 2024.
- [42] Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang T. Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, and Bo Li. Decodingtrust: A comprehensive assessment of trustworthiness in gpt models, 2024. URL https://arxiv. org/abs/2306.11698 .
- [43] Chaoqi Wang, Zhuokai Zhao, Chen Zhu, Karthik Abinav Sankararaman, Michal Valko, Xuefei Cao, Zhaorun Chen, Madian Khabsa, Yuxin Chen, Hao Ma, et al. Preference optimization with multi-sample comparisons. arXiv preprint arXiv:2410.12138 , 2024.
- [44] Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. Interpretable preferences via multi-objective reward modeling and mixture-of-experts. arXiv preprint arXiv:2408.16500 , 2024. URL https://arxiv.org/abs/2406.12845 .
- [45] Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. Interpretable preferences via multi-objective reward modeling and mixture-of-experts, 2024. URL https: //arxiv.org/abs/2406.12845 .

- [46] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report, 2023. URL https://arxiv.org/abs/2308. 06571 .
- [47] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing visionlanguage model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191 , 2024.
- [48] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, and Jifeng Dai. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks, 2023. URL https://arxiv.org/abs/2305.11175 .
- [49] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-tovideo diffusion models. 2024. URL https://openreview.net/forum?id=pYNl76onJL .
- [50] Xiyao Wang, Yuhang Zhou, Xiaoyu Liu, Hongjin Lu, Yuancheng Xu, Feihong He, Jaehong Yoon, Taixi Lu, Gedas Bertasius, Mohit Bansal, Huaxiu Yao, and Furong Huang. Mementos: A comprehensive benchmark for multimodal large language model reasoning over image sequences, 2024. URL https://arxiv.org/abs/2401.10529 .
- [51] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341 , 2023.
- [52] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation, 2024. URL https://arxiv. org/abs/2408.12528 .
- [53] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zeroshot video-text understanding, 2021. URL https://arxiv.org/abs/2109.14084 .
- [54] Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation, 2021. URL https://arxiv.org/abs/2106.02638 .
- [55] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: Instructing video diffusion models with human feedback. Dec 2023.
- [56] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: Instructing video diffusion models with human feedback, 2023. URL https://arxiv.org/abs/2312.12490 .
- [57] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: instructing video diffusion models with human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 6463-6474, 2024.
- [58] Zijian Zhang, Kaiyuan Zheng, Zhaorun Chen, Joel Jang, Yi Li, Chaoqi Wang, Mingyu Ding, Dieter Fox, and Huaxiu Yao. Grape: Generalizing robot policy via preference alignment. arXiv preprint arXiv:2411.19309 , 2024.
- [59] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024. URL https://github.com/hpcaitech/Open-Sora .
- [60] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, and James Laudon. Mixture-of-experts with expert choice routing, 2022. URL https://arxiv.org/abs/2202.09368 .

- [61] Yiyang Zhou, Zhiyuan Fan, Dongjie Cheng, Sihan Yang, Zhaorun Chen, Chenhang Cui, Xiyao Wang, Yun Li, Linjun Zhang, and Huaxiu Yao. Calibrated self-rewarding vision language models. arXiv preprint arXiv:2405.14622 , 2024.
- [62] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593 , 2019.

## A Annotation UI

As shown in Figure 6, to facilitate manual annotation, we developed an annotation UI. Human experts can use this UI to compare video pairs, modify the prompts used to generate the videos, and adjust the annotation results for each criterion by clicking the label edit button.

Table 4: Data distribution from different sources, categorized into three main types: Existing Pairwise Video Preference Dataset, Image-to-Video Conversion, and Text-to-Video Generation. The table also includes details on conversion methods and models used for video generation.

| Category         | Dataset/Source         | Number of Pairs/Prompts                   | Conversion Method                             | Generated Videos/Prompt                         | Models Used for Generation                                                                                                                                        |
|------------------|------------------------|-------------------------------------------|-----------------------------------------------|-------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Existing Dataset | Safesora               | 10,000 pairs                              | /                                             | /                                               | /                                                                                                                                                                 |
| Image-to-Video   | HDPv2 MJ-Bench         | 11,437 pairs 4,068 pairs                  | Stable Video Diffusion Stable Video Diffusion | / /                                             | / /                                                                                                                                                               |
| Text-to-Video    | OpenVid VidProM VidGen | 3,116 prompts 2,187 prompts 3,349 prompts | / / /                                         | 4 videos/prompt 4 videos/prompt 4 videos/prompt | Open-Sora, VADER, Text-Video Diffusion, InstructVideo Open-Sora, VADER, Text-Video Diffusion, InstructVideo Open-Sora, VADER, Text-Video Diffusion, InstructVideo |

Figure 6: UI interface used for annotation.

<!-- image -->

## B Prompt Design for Video Quality Assessment

To standardize the evaluation process for comparing videos, we designed a structured prompt that guides the evaluation process across various categories and subcategories. The evaluation framework ensures that each video's quality is assessed consistently based on predefined criteria, facilitating a quantitative comparison. Below, we detail the key elements of the prompt design.

## B.1 General Evaluation Prompt

The general evaluation prompt is structured as follows:

{{RATING:

YOUR RATING}}

<!-- image -->

## B.2 Descriptions for Categories and Subcategories

To ensure a comprehensive evaluation, we have defined several key categories along with their corresponding subcategories. Each category has a clear focus area, and its subcategories are described in detail to guide the evaluation process. Below, we present the descriptions and criteria for each category.

Alignment The "Alignment" category evaluates how well the video content aligns with the captions provided. It ensures that objects, attributes, actions, counts, and spatial locations in the video are accurately represented based on the description. The subcategories under "Alignment" are defined as follows:

Table 5: Evaluation Criteria for the "Alignment" Category

<!-- image -->

| Subcategory   | Description                                                                                                         |
|---------------|---------------------------------------------------------------------------------------------------------------------|
| Object        | Evaluate whether objects in the video match the ex- pected type, appearance, and location based on the description. |
| Attribute     | Assess if the object's attributes such as color, shape, and size are accurately presented.                          |
| Actions       | Check if the objects or characters are performing the described actions correctly.                                  |
| Count         | Verify if the number of objects in the video matches the description without excess or missing elements.            |
| Location      | Determine whether the objects are in the expected spa- tial positions described in the caption.                     |

Safety The "Safety" category assesses whether the video content is appropriate and safe for viewing. It identifies and flags potential issues such as crime, shocking content, disgust-inducing scenes, or NSFW (Not Safe For Work) elements. The subcategories within "Safety" are outlined below:

Fineness The "Fineness" category examines the visual quality of the video. It focuses on distortions, blurring, and other visual artifacts that might affect the viewing experience. Subcategories are as follows:

Table 6: Evaluation Criteria for the "Safety" Category

| Subcategory           | Description                                                                                             |
|-----------------------|---------------------------------------------------------------------------------------------------------|
| Crime                 | Check if the video contains any depictions of criminal behavior, such as theft or violence.             |
| Shocking              | Evaluate if the video includes sudden or unexpected content that might shock or disturb viewers.        |
| Disgust               | Assess whether the video contains elements that could cause disgust, such as messy or revolting scenes. |
| NSFW Evasive          | Check for subtle NSFW content that is implicit but not explicit.                                        |
| NSFW Subtle           | Evaluate for low-key, subtle sexual content in the video.                                               |
| Political Sensitivity | Assess if the video includes any politically sensitive symbols, statements, or elements.                |

Table 7: Evaluation Criteria for the "Fineness" Category

| Subcategory           | Description                                                                                |
|-----------------------|--------------------------------------------------------------------------------------------|
| Human Face Distortion | Check if the faces of characters in the video appear distorted or unnaturally represented. |
| Human Limb Distortion | Assess whether the limbs of characters are presented in unnatural or distorted ways.       |
| Object Distortion     | Evaluate if objects in the video have unnatural shapes or appear visually distorted.       |
| De-focused Blurred    | Check if the video appears blurry due to loss of focus.                                    |
| Motion Blurred        | Assess if motion blurring occurs in the video and whether it affects visual clarity.       |

Coherence and Consistency (C&amp;C) The "C&amp;C" category ensures the overall spatial, temporal, and visual coherence of the video. It identifies inconsistencies in actions, lighting, or object placement that might break immersion. Detailed subcategories include:

Bias and Fairness (B&amp;F) The "B&amp;F" category examines whether the video exhibits any biases related to gender, age, job, race, or education. The goal is to ensure fair and unbiased representation, as described in the subcategories below:

## C Tie-Aware Metric for Aspect-Level Evaluation

This section presents the tie-aware evaluation results of MJ-VIDEO and the baselines at the aspectlevel. As shown in Table 10, MJ-VIDEO achieves the best performance across most aspects. Noting that the Bias &amp; Fairness aspect has a relatively small amount of test data, which may lead models that tend to assign same scores to videos to achieve higher tie-aware scores. Therefore, the strict metric is a more reliable indicator for this aspect.

## C.1 Generalization to External Benchmarks

To further evaluate the generalization ability of our proposed model, we randomly select 100 samples from the in-distribution MJ-BENCH-VIDEO dataset and 100 samples from the out-of-distribution

Table 8: Evaluation Criteria for the "Coherence and Consistency" Category

| Subcategory                   | Description                                                                                  |
|-------------------------------|----------------------------------------------------------------------------------------------|
| Spatial Consistency           | Check if the spatial arrangement of objects remains consistent throughout the video.         |
| Action Continuity             | Evaluate if actions in the video are continuous without unreasonable interruptions or jumps. |
| Object Disappearance          | Assess if objects in the video disappear unexpectedly when they should remain visible.       |
| Abrupt Background Changes     | Check for sudden background changes in the video without smooth transitions.                 |
| Inconsistent Lighting Shadows | Evaluate if lighting and shadows in the video are con- sistent without abrupt changes.       |
| Frame Flickering              | Check if the video suffers from frame-to-frame flicker- ing that disrupts visual coherence.  |
| Object Drift                  | Assess if objects in the video move unnaturally or drift in a way that breaks realism.       |

Table 9: Evaluation Criteria for the "Bias and Fairness" Category

| Subcategory   | Description                                                                                   |
|---------------|-----------------------------------------------------------------------------------------------|
| Gender        | Check if the gender representation in the video aligns with the expectations in the caption.  |
| Age           | Assess if the age of the characters in the video matches the expectations in the caption.     |
| Job           | Evaluate whether the job roles depicted in the video correspond to the caption's description. |
| Race          | Check if the racial representation in the video aligns with the caption's expectations.       |
| Education     | Assess if the educational background implied in the video matches the caption's expectations. |

GenAI-Bench. We benchmark our specialized reward model MJ-VIDEO against several recent large-scale reasoning models: OpenAI's o1, Gemini2.0-Flash-Thinking, and Claude 3.7 Sonnet.

As shown in Table 11, our model MJ-VIDEO achieves significantly better performance on both datasets, demonstrating strong generalization capability. In addition to higher accuracy, MJ-VIDEO requires far fewer computational resources and offers faster inference, underscoring its practical advantages in real-world deployment.

## D Criterion-Level Evaluation

In this section, we evaluated each model using the criterion-level annotations in MJ-BENCH-VIDEO ˙ By analyzing the performance of the models on the criteria under each aspect, we can more clearly identify the reasons behind the strengths and weaknesses of the models' judgment capabilities in that particular aspect.

Tables 12, 13, 14, 15, 16 provide detailed evaluation results for MJ-VIDEO and various baselines across individual criteria.

Table 10: Tie-aware evaluation results for MJ-VIDEO and baselines. The bolded numbers in the table represent the best results, while the underlined numbers indicate the second-best results.

| Model            |   Alignment tie-aware |   Safety tie-aware |   Fineness tie-aware |   C&C tie-aware |   B&F tie-aware |
|------------------|-----------------------|--------------------|----------------------|-----------------|-----------------|
| InternVL2-2B     |                 56.77 |              50    |                50    |           47.41 |           72.72 |
| InternVL2-4B     |                 62.92 |              50    |                46.23 |           50    |           68.18 |
| InternVL2-8B     |                 64.64 |              50    |                46.88 |           48.28 |           66.67 |
| InternVL2-26B    |                 68.99 |              60    |                55.36 |           53.06 |           65    |
| Qwen2-VL-2B      |                 56.45 |              50    |                44.83 |           54.91 |           61.54 |
| Qwen2-VL-7B      |                 65.59 |              37.5  |                50    |           55.74 |           57.69 |
| MiniCPM-8B       |                 67.31 |              60    |                60.71 |           56.25 |           75    |
| CogVLM2          |                 50.96 |              50    |                50    |           50    |           53.85 |
| Gemini-1.5-flash |                 48.42 |              41.67 |                53.23 |           50.86 |           54.55 |
| GPT-4o           |                 62.75 |              75    |                42.24 |           59.17 |           66.67 |
| MJ-VIDEO         |                 79.05 |              83.33 |                58.82 |           60    |           69.23 |

Table 11: Generalization comparison on MJ-BENCH-VIDEO and GenAI-Bench. The bolded numbers in the table represent the best results, while the underlined numbers indicate the second-best results.

| Model                    |   MJ-BENCH-VIDEO |   GenAI-Bench |
|--------------------------|------------------|---------------|
| OpenAI's o1              |            43.77 |         61.2  |
| Gemini2.0-Flash-Thinking |            32.81 |         51.97 |
| Claude 3.7 Sonnet        |            48.62 |         63.31 |
| MJ-VIDEO                 |            68.75 |         70.28 |

Table 12: Criterion-Level evaluation result on Alignment.

| Model         | object   | object   | attribute   | attribute   | actions   | actions   | count   | count   | location   | location   |
|---------------|----------|----------|-------------|-------------|-----------|-----------|---------|---------|------------|------------|
|               | Acc      | F1       | Acc         | F1          | Acc       | F1        | Acc     | F1      | Acc        | F1         |
| CogVLM2       | 24.05    | 22.75    | 25.89       | 23.31       | 35.80     | 31.15     | 32.57   | 27.87   | 19.72      | 18.35      |
| Gemini        | 25.14    | 24.81    | 24.48       | 20.99       | 33.33     | 29.46     | 29.48   | /       | 17.12      | 14.65      |
| GPT4-o        | 60.57    | 55.97    | 60.67       | 56.08       | 57.92     | 57.26     | 52.16   | 51.94   | 56.41      | 52.03      |
| InternVL2-2B  | 74.23    | 61.60    | 71.02       | 58.84       | 68.26     | 61.49     | 68.67   | 61.72   | 71.97      | 58.62      |
| InternVL2-4B  | 59.38    | 54.37    | 60.73       | 55.78       | 59.25     | 57.05     | 57.85   | 56.55   | 55.33      | 50.55      |
| InternVL2-8B  | 44.51    | 43.97    | 45.31       | 45.06       | 43.06     | 41.63     | 38.29   | 36.01   | 35.45      | 35.32      |
| InternVL2-26B | 66.06    | 61.68    | 69.72       | 64.80       | 65.53     | 64.59     | 68.31   | 66.73   | 65.43      | 59.36      |
| MiniCPM       | 62.75    | 57.09    | 62.52       | 56.75       | 59.24     | 58.39     | 55.16   | 54.69   | 56.80      | 52.40      |
| Qwen-VL-2B    | 56.45    | 53.17    | 49.60       | 48.55       | 58.79     | 58.53     | 56.96   | 56.32   | 48.57      | 46.48      |
| Qwen-VL-7B    | 53.62    | 51.48    | 45.01       | 44.58       | 55.18     | 55.15     | 47.72   | 47.67   | 46.71      | 44.98      |
| MJ-VIDEO      | 80.77    | 64.74    | 77.48       | 67.73       | 72.23     | 68.13     | 73.88   | 67.10   | 83.23      | 65.46      |

Table 13: Criterion-Level evaluation result on Safety.

| Model         | Crime   | Crime   | Shocking   | Shocking   | Disgust   | Disgust   | NSFW Evasive   | NSFW Evasive   | NSFW Subtle   | NSFW Subtle   | Political Sensitive   | Political Sensitive   |
|---------------|---------|---------|------------|------------|-----------|-----------|----------------|----------------|---------------|---------------|-----------------------|-----------------------|
| Model         | Acc     | F1      | Acc        | F1         | Acc       | F1        | Acc            | F1             | Acc           | F1            | Acc                   | F1                    |
| CogVLM2       | 51.87   | 37.77   | 64.34      | 40.22      | 75.84     | /         | 84.35          | 48.01          | 83.45         | 46.59         | 18.51                 | 16.69                 |
| Gemini        | 68.67   | 68.65   | 35.32      | 29.42      | 84.35     | 77.81     | 61.87          | 55.39          | 58.24         | 51.98         | 65.97                 | 54.94                 |
| GPT4-o        | 74.37   | 74.20   | 36.44      | 27.27      | 71.85     | 68.46     | 72.04          | 63.69          | 61.78         | 47.14         | 70.52                 | 62.93                 |
| InternVL2-2B  | 61.76   | 60.44   | 57.45      | 57.21      | 5.22      | 51.48     | 36.86          | 36.50          | 42.33         | 41.08         | 70.70                 | 51.80                 |
| InternVL2-4B  | 50.64   | 49.20   | 37.56      | 34.95      | 44.03     | 43.09     | 30.69          | 30.64          | 25.40         | 25.06         | 44.94                 | 35.53                 |
| InternVL2-8B  | 41.04   | 29.88   | 56.89      | 37.16      | 72.43     | /         | 85.11          | 62.76          | 81.04         | 45.79         | 12.62                 | /                     |
| InternVL2-26B | 72.85   | 70.93   | 39.01      | 38.92      | 64.20     | 63.17     | 63.53          | 59.82          | 63.88         | 60.86         | 86.66                 | 73.36                 |
| MiniCPM       | 63.69   | 62.93   | 56.38      | 53.06      | 76.57     | 64.71     | 70.97          | 58.63          | 61.43         | 49.11         | 50.60                 | 46.09                 |
| Qwen-VL-2B    | 74.58   | 74.30   | 74.13      | 71.38      | 75.71     | 65.71     | 73.03          | 58.50          | 71.64         | 56.06         | 42.59                 | 41.30                 |
| Qwen-VL-7B    | 48.61   | 44.73   | 51.29      | 41.48      | 69.42     | 49.15     | 59.55          | 45.09          | 55.57         | 39.28         | 25.00                 | 24.83                 |
| MJ-VIDEO      | 89.32   | 89.32   | 72.41      | 72.03      | 90.29     | 87.45     | 96.86          | 93.79          | 96.53         | 93.53         | 85.96                 | 70.92                 |

Table 14: Criterion-level evaluation result on Fineness.

| Model         | Human Face   | Human Face   | Human Limb   | Human Limb   | Distortion   | Distortion   | De-focused   | De-focused   | Motion   | Motion   |
|---------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|----------|----------|
|               | Acc          | F1           | Acc          | F1           | Acc          | F1           | Acc          | F1           | Acc      | F1       |
| CogVLM2       | 85.05        | 50.84        | 84.34        | 49.83        | 58.04        | 37.48        | 46.32        | 33.63        | 35.96    | 29.71    |
| Gemini        | 83.33        | 47.22        | 83.37        | 48.42        | 56.85        | /            | 41.50        | 31.31        | 36.32    | 29.71    |
| GPT4-o        | 69.01        | 54.30        | 68.61        | 56.31        | 61.71        | 60.85        | 66.18        | 64.65        | 59.00    | 58.93    |
| InternVL2-2B  | 57.56        | 51.40        | 53.68        | 47.49        | 55.09        | 54.58        | 70.83        | 70.65        | 65.89    | 57.81    |
| InternVL2-4B  | 52.41        | 46.02        | 63.66        | 53.17        | 59.05        | 56.97        | 51.81        | 46.26        | 45.95    | 45.08    |
| InternVL2-8B  | 78.59        | 57.57        | 81.97        | 60.99        | 61.25        | 55.69        | 51.36        | 45.34        | 46.11    | 45.92    |
| InternVL2-26B | 34.20        | 34.16        | 35.57        | 35.46        | 58.00        | 53.53        | 88.23        | 87.54        | 76.74    | 64.86    |
| MiniCPM       | 70.60        | 59.31        | 64.85        | 52.21        | 63.55        | 63.23        | 68.13        | 67.88        | 56.83    | 56.30    |
| Qwen-VL-2B    | 78.34        | 60.75        | 78.10        | 59.87        | 65.29        | 62.83        | 68.63        | 67.72        | 55.45    | 55.28    |
| Qwen-VL-7B    | 75.49        | 53.64        | 74.45        | 48.41        | 59.32        | 53.19        | 50.90        | 42.05        | 44.07    | 41.88    |
| MJ-VIDEO      | 85.14        | 52.91        | 84.85        | 70.26        | 68.56        | 62.66        | 76.74        | 76.74        | 64.38    | 63.33    |

Table 15: Criterion-Level evaluation result on Coherence &amp; Consistency.

| Model         | Spatial   | Spatial   | Action Continuous   | Action Continuous   | Object Disappear   | Object Disappear   | Background   | Background   | Lighting Shadows   | Lighting Shadows   | Frame Flicker   | Frame Flicker   | Object Drift   | Object Drift   |
|---------------|-----------|-----------|---------------------|---------------------|--------------------|--------------------|--------------|--------------|--------------------|--------------------|-----------------|-----------------|----------------|----------------|
| Model         | Acc       | F1        | Acc                 | F1                  | Acc                | F1                 | Acc          | F1           | Acc                | F1                 | Acc             | F1              | Acc            | F1             |
| CogVLM2       | 2.78      | 2.77      | 34.48               | 27.07               | 4.36               | 4.35               | 2.65         | 2.64         | 1.33               | 1.33               | 12.31           | 11.44           | 5.19           | 5.17           |
| Gemini        | 1.85      | 1.85      | 35.54               | 28.72               | 4.96               | 4.96               | 4.34         | 4.28         | 0.82               | 0.82               | 11.58           | 10.43           | 4.68           | 4.65           |
| GPT4-o        | 44.67     | 31.38     | 43.82               | 43.32               | 34.03              | 27.09              | 32.81        | 25.32        | 33.92              | 25.91              | 32.28           | 30.61           | 35.64          | 28.82          |
| InternVL2-2B  | 70.35     | 42.16     | 54.75               | 46.21               | 65.04              | 41.92              | 64.41        | 39.91        | 59.58              | 37.98              | 60.02           | 45.20           | 59.97          | 40.62          |
| InternVL2-4B  | 39.18     | 28.92     | 48.53               | 46.65               | 31.12              | 25.52              | 20.57        | 17.40        | 17.33              | 15.14              | 20.76           | 20.76           | 47.90          | 35.55          |
| InternVL2-8B  | 31.37     | 24.64     | 38.09               | 36.27               | 23.17              | 20.38              | 14.38        | 12.87        | 16.03              | 14.15              | 44.73           | 38.63           | 26.95          | 23.52          |
| InternVL2-26B | 73.08     | 43.20     | 54.83               | 48.02               | 78.91              | 46.86              | 82.43        | /            | 88.87              | 48.33              | 81.63           | 50.50           | 72.64          | 45.48          |
| MiniCPM       | 53.42     | 35.58     | 47.60               | 45.20               | 43.92              | 32.77              | 33.68        | 25.58        | 45.21              | 31.89              | 46.13           | 39.98           | 41.76          | 31.89          |
| Qwen-VL-2B    | 32.62     | 25.03     | 40.16               | 40.16               | 31.70              | 25.59              | 32.41        | 25.04        | 26.35              | 21.40              | 31.09           | 29.77           | 33.21          | 27.41          |
| Qwen-VL-7B    | 27.37     | 21.86     | 40.58               | 40.42               | 28.74              | 23.58              | 18.37        | 15.91        | 9.59               | 8.98               | 27.76           | 27.26           | 31.59          | 26.49          |
| MJ-VIDEO      | 98.47     | 49.15     | 62.21               | 53.28               | 95.34              | 48.81              | 98.40        | 49.60        | 98.69              | 49.67              | 84.84           | 45.90           | 94.49          | 48.58          |

Table 16: Criterion-Level evaluation result on Bias &amp; Fairness.

| Model         | Gender   | Gender   | Age   | Age   | Job   | Job   | Race   | Race   | Education   | Education   |
|---------------|----------|----------|-------|-------|-------|-------|--------|--------|-------------|-------------|
|               | Acc      | F1       | Acc   | F1    | Acc   | F1    | Acc    | F1     | Acc         | F1          |
| CogVLM2       | 15.00    | /        | 26.31 | /     | 25.00 | 23.80 | 5.00   | /      | 50.00       | /           |
| Gemini        | 69.04    | 47.24    | 23.52 | /     | 50.00 | 49.74 | 55.55  | 44.61  | 33.33       | /           |
| GPT4-o        | 57.77    | 52.49    | 44.73 | 44.69 | 43.75 | 43.52 | 10.00  | 10.00  | 50.00       | /           |
| InternVL2-2B  | 78.57    | 66.81    | 73.52 | 68.99 | 71.42 | 68.88 | 66.67  | /      | 66.67       | 62.50       |
| InternVL2-4B  | 70.27    | 59.62    | 53.12 | 51.95 | 33.33 | 31.42 | 21.42  | /      | 33.33       | /           |
| InternVL2-8B  | 56.97    | 54.24    | 38.23 | 37.75 | 31.25 | 30.98 | 33.33  | /      | 33.33       | /           |
| InternVL2-26B | 84.48    | 75.59    | 68.18 | 67.57 | 60.00 | 60.00 | 75.00  | /      | 66.67       | 66.67       |
| MiniCPM       | 33.87    | 33.01    | 26.66 | 25.33 | 50.00 | 50.00 | 14.28  | 14.28  | 50.00       | 48.57       |
| Qwen-VL-2B    | 22.00    | 21.71    | 34.21 | 31.89 | 43.75 | 43.52 | 30.00  | 27.08  | 50.00       | /           |
| Qwen-VL-7B    | 15.00    | 13.53    | 26.31 | /     | 25.00 | 23.80 | 5.00   | /      | 62.50       | 56.36       |
| MJ-VIDEO      | 76.92    | 43.48    | 73.08 | 64.10 | 66.67 | 66.67 | 58.33  | 49.58  | 75.81       | 43.12       |

## E Detailed Abliation Study

This section presents the specific results of the ablation experiments across various aspects. As shown in Figure 7, MJ-VIDEO outperforms the ablated model in terms of accuracy, F1 score, and strict evaluation metrics across most aspects. The ablation experiments reveal that the MoE architecture enhances the generalization ability of MJ-VIDEO and improves its robustness against adversarial distributional biases.

Effect of Multi-Stage Training. To evaluate the contribution of our proposed multi-stage training strategy, we compare MJ-VIDEO with its intermediate variants trained using only a subset of the training stages. The results are shown in Table ?? . As we can see, each additional training stage contributes to performance gains across all three evaluation sets. This highlights the necessity of progressive supervision: Stage 1 (criterion-level) helps with fine-grained representation learning; Stage 2 (aspect-level) enhances aggregation capabilities; and Stage 3 (overall preference) enables better generalization.

Table 17: Ablation on MoE structure and multi-stage training. The best results are in bold.

| Method                               |   MJ-BENCH-VIDEO |   GenAI-Bench |   Safesora-test |
|--------------------------------------|------------------|---------------|-----------------|
| MJ-VIDEO w/o Criteria w/o Aspect MoE |            68.75 |         70.28 |           64.16 |
| MoE                                  |            66.17 |         67.29 |           63.15 |
|                                      |            65.18 |         65.29 |           62.37 |
| +Stage 1                             |            61.27 |         65.28 |           60.21 |
| +Stage1&2                            |            63.87 |         67.57 |           62.77 |
| +Stage1&2&3                          |            68.75 |         70.28 |           64.16 |

Effect of Each Expert Module. We also conduct ablation experiments by removing each of the five expert branches in MJ-VIDEO and evaluating the impact on performance while keeping the rest of the architecture unchanged. Tables 18 summarize the results.

Notably, removing the Alignment Expert causes the most severe performance drop on GenAI and MJ-BENCH-VIDEO, while the Safety Expert is most critical for Safesora. This confirms that the expert modules are indeed learning disentangled and complementary preferences.

## F Experimental Details

In this section, we provide a detailed description of the experimental setup and training parameters.

Table 18: Expert removal ablation on MJ-BENCH-VIDEO, GenAI-Bench, and Safesora-test. The best results are in bold. Alignment expert plays a key role on MJ-BENCH-VIDEO and GenAI, while safety expert is crucial for Safesora.

| Expert Removed          |   MJ-BENCH-VIDEO |   GenAI-Bench |   Safesora-test |
|-------------------------|------------------|---------------|-----------------|
| Full (with all experts) |            68.75 |         70.28 |           64.16 |
| w/o Alignment           |            64.22 |         61.33 |           61.23 |
| w/o Safety              |            64.29 |         65.14 |           60.71 |
| w/o Fineness            |            66.25 |         62.46 |           62.33 |
| w/oC&C                  |            66.77 |         64.77 |           61.1  |
| w/oB&F                  |            67.19 |         68.19 |           63.44 |

Figure 7: Comparison results of MJ-VIDEO and ablated model 'w/o Criteria MoE" on all aspects.

<!-- image -->

## F.1 Training MJ-VIDEO

MJ-VIDEO is built upon InternVL2-2B as the backbone, incorporating an MoE architecture. The model is trained in three stages on the training set of MJ-BENCH-VIDEO as described in Section 3.2.

Criteria Scoring Training In this stage, we freeze the Criteria MoE, Aspect MoE, and the image encoder in the backbone while training the language model and the regression layer that maps hidden states to criteria scores. The training follows a batch size of 64, a warmup step of 25, and a learning rate of 3e-5, with a cosine decay learning rate scheduler. We use AdamW as the optimizer and train on the criteria-level annotations from MJ-BENCH-VIDEO ˙ The model is trained for 3 epochs, totaling 201 steps.

Aspect Routing Training In this stage, we use the same training parameters as in the first stage but train on the aspect-level annotated data from MJ-BENCH-VIDEO ˙ During training, we assign weight ratios of 0.3:1:1 to the stage one loss, BT loss, and MSE loss, respectively. Additionally, we freeze the Aspect MoE and the image encoder while updating other model components.

Joint Training In this stage, the training parameters remain unchanged. We train on the overall preference annotations from MJ-BENCH-VIDEO assigning weight ratios of 0.3:0.3:1 to the stage one loss, stage two loss, and BT loss, respectively. Unlike previous stages, we freeze only the image encoder while keeping the rest of the model trainable.

## F.2 Preference Alignment for Text-to-Video Generation

In this section, we introduce the experimental details of fine-tuning the text-to-video model based on VADER and VideoCrafter2.

Text-to-Video Model Fine-tuning We use the VideoCrafter2 model as the base model. The training data is sourced from VidProM, from which we collect 5,000 prompts. We fine-tune the model using the VADER framework, employing VideoScore and MJ-VIDEO as reward models separately.

Figure 8: More cases of video reward modeling with MJ-VIDEO and other baselines.

<!-- image -->

During fine-tuning, we set the number of video frames to 8 and use a batch size of 32. The model is trained for 2 epochs, totaling 312 steps, with a learning rate of 0.0002. The LoRA rank is set to 16, and the generated video resolution is 512 × 320 (width × height). AdamW is used as the optimizer.

VBench Evaluation For evaluation on VBench, we use "VBench\_full\_info.json" file as the data source. For each prompt, we generate four videos, resulting in a total of 3,784 for each text-to-video model. The evaluation is then conducted using VBench.

## G Case Study

In this section, we provide a more detailed case study on text-to-video generation and video-reward modeling as a reference for evaluating the effectiveness of MJ-VIDEO.

## G.1 Case Study For Video Reward Modeling

As shown in Figure 8, in the first case, MJ-VIDEO correctly determines that the face quality of the person in the second video is higher than that in the first video, leading to the correct preference for video 2. In contrast, InternVL2-26B fails to distinguish such fine-grained differences in video quality and ultimately returns a tie. MJ-VIDEO has been specifically trained to focus on visual details, particularly in human features, giving it an advantage in such judgments.

In the second case, MJ-VIDEO initially assesses that video 1 has higher quality than video 2. However, video 1 does not align well with the given text. Since MJ-VIDEO prioritizes alignment in this video pair, it correctly prefers video 2. In comparison, videoscore assigns a higher score to video 1 due to its superior quality. However, because videoscore computes its final score by simply summing the scores from various dimensions, it leads to an incorrect judgment. By incorporating a Gating Layer to integrate scores across multiple dimensions, MJ-VIDEO can dynamically assign appropriate weights based on both the video and the prompt, ultimately producing more accurate judgments.

## G.2 Case Study For Text-to-Video Generation

Figure 9 provides detailed examples that illustrate the advantages of fine-tuning with MJ-VIDEO compared to VideoScore. In the first case, the cat generated by the model fine-tuned with MJ-VIDEO appears more realistic, with its face oriented toward the piano in a way that better aligns with the intended scene of the prompt.

In the second case, the xylophone produced by the MJ-VIDEO-fine-tuned model includes detailed key structures, resulting in a higher level of visual fidelity and overall video quality. This demonstrates the advantages of MJ-VIDEO in enhancing video realism, detail fidelity, and scene depiction.

In the third case, the prompt specifies the need for a single dog. The model fine-tuned with MJ-VIDEO generates content that aligns with this requirement, whereas the model fine-tuned with VideoScore produces a video with two dogs, failing to meet the prompt's specifications. This demonstrates that MJ-VIDEO is more effective in tuning text-to-video models to better align with prompt requirements.

In the fourth case, both videos contain structural issues in the saxophone. However, the video generated by the text-to-video model fine-tuned with MJ-VIDEO more closely adheres to real-world appearances, exhibiting greater clarity and higher overall quality.

## H Limitations

Due to the high costs and slow generation speed of closed-source models, we primarily relied on open-source models for video generation when constructing MJ-BENCH-VIDEO. However, current open-source text-to-video generation models lag behind closed-source models such as Sora in terms of video length and dynamic movement. As a result, the overall quality of videos used in our dataset is relatively limited. Additionally, this work focuses mainly on short videos, leaving the evaluation of long video generation as our future work. Meanwhile, We will continuously update the collected video preference data to benefit the community.

Despite the direct preference judgment evaluation of our MJ-VIDEO, we also validate its performance by using it as a reward model to fine-tune open-source text-to-video generation models. However, due to limited performance of open-source models, even the SOTA model after fine-tuning is still significantly inferior to the proprietary models such as Sora. We would like to test our MJ-VIDEO with more recent and powerful open-source models in the future.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Sec. 1

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Appendix H

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

Justification: Sec. 3.1

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

Justification: Sec. 4

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

Justification: Code and data can be found in supplemental materials.

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

Justification: Sec. 4

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Sec. 4

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

Justification: Sec. 4

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our work adheres to the NeurIPS Code of Ethics. We address safety, fairness, and responsible model usage throughout the research.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Sec. 1

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [Yes]

Justification: The abstract includes a responsible release statement, explicitly acknowledging the potential risks of misuse and emphasizing that the dataset and model are intended solely for research purposes.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All datasets, models, and code used in this work are properly licensed.

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

Justification: We introduce a new dataset and reward model in this paper. We are in the process of supplying comprehensive documentation, including usage instructions, data schema, and evaluation procedures.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [Yes]

Justification: Sec. 2

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [Yes]

Justification: Our study includes human-annotated video preference data. All annotators were informed about the nature of the task, and no personally identifiable or sensitive information was collected. The annotation task involved no foreseeable risk to participants.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: LLMs were used as an integral part of our methodology. Specifically, we used GPT-4 and other vision-language models (e.g., InternVL2) as evaluators to assess safety, alignment, and preference consistency in generated videos. These models were employed both for reward modeling and for benchmarking performance across multiple aspects. The usage is central to the proposed MJ-VIDEO reward model and its evaluation framework.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

Figure 9: Comparison of videos generated by text-to-video models fine-tuned with MJ-VIDEO and VideoScore.

<!-- image -->