## Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling

Yichuan Cao 1 , 2 ⋆ , Yibo Miao 1 , 2 ⋆ † , Xiao-Shan Gao 1 , 2 , Yinpeng Dong 3 , 4 † 1 State Key Laboratory of Math. Sci., Academy of Mathematics and Systems Science, Chinese Academy of Sciences, Beijing 100190, China

2 University of Chinese Academy of Sciences, Beijing 100049, China

3 College of AI, Tsinghua University, Beijing 100084, China 4 Shanghai Qi Zhi Institute

/warning\_sign Warning : This paper contains data and model outputs which are offensive in nature.

## Abstract

Text-to-image (T2I) models raise ethical and safety concerns due to their potential to generate inappropriate or harmful images. Evaluating these models' security through red-teaming is vital, yet white-box approaches are limited by their need for internal access, complicating their use with closed-source models. Moreover, existing black-box methods often assume knowledge about the model's specific defense mechanisms, limiting their utility in real-world commercial API scenarios. A significant challenge is how to evade unknown and diverse defense mechanisms. To overcome this difficulty, we propose a novel Rule-based Preference modeling Guided Red-Teaming (RPG-RT), which iteratively employs LLM to modify prompts to query and leverages feedback from T2I systems for fine-tuning the LLM. RPG-RT treats the feedback from each iteration as a prior, enabling the LLM to dynamically adapt to unknown defense mechanisms. Given that the feedback is often labeled and coarse-grained, making it difficult to utilize directly, we further propose rule-based preference modeling, which employs a set of rules to evaluate desired or undesired feedback, facilitating finer-grained control over the LLM's dynamic adaptation process. Extensive experiments on nineteen T2I systems with varied safety mechanisms, three online commercial API services, and T2V models verify the superiority and practicality of our approach. Our codes are available at: https://github.com/caosip/RPG-RT .

## 1 Introduction

The state-of-the-art text-to-image (T2I) models such as Midjourney [42], Stable Diffusion [56], and DALL-E [47] have garnered widespread attention for their ability to create high-quality images across a variety of concepts and styles from natural language input [13, 45, 56]. Millions of users have started harnessing these generative models to increase productivity [51], including applications designed for children [54]. However, there is growing concern about the ethical and safety implications of these technologies [54, 58]. Malicious users can exploit the powerful generative capabilities of T2I models to create images containing pornography, violence, and politically sensitive content [64], or to produce copyright-infringing materials [63]. In fact, Google's Gemini has generated numerous biased and historically inaccurate images, causing the service to be taken offline [43]. The misuse of T2I models poses risks of violating legal standards and regulations [10], potentially impacting model developers, researchers, users, and regulatory bodies in terms of legal and reputational consequences.

Red-teaming identifies and exposes vulnerabilities inherent in T2I models by generating undesirable outputs from text prompts, crucial for evaluating model safety. Although some existing red-team

⋆ Equal contribution. † Corresponding author: miaoyibo@amss.ac.cn, dongyinpeng@tsinghua.edu.cn

Figure 1: Overview of our RPG-RT framework. a) Stage 1: The LLM generates multiple different modifications of the prompt, then inputs them into the target T2I black-box system and obtains the outputs. b) Stage 2: A binary partial order is constructed to model the preferences of the T2I system. Rule-based scoring is utilized to enable fine-grained control over the LLM's exploration of the commercial black-box system. c) Stage 3: The LLM agent is fine-tuned using DPO based on the generative preferences of the target T2I system.

<!-- image -->

methods have explored the white-box settings [9, 73, 64] or assumed that the attacker has partial access to model components (e.g., text encoders [62, 37]) in gray-box scenarios, these approaches require internal access to the model, which is not feasible when the model is not open-source. Recent studies have proposed various black-box red-team strategies against different defense mechanisms. Some methods focus on detection-based defenses for T2I models, enabling malicious prompts to bypass safety checkers [66, 2, 12], while others emphasize removal-based defenses, aiming to generate not-safe-for-work (NSFW) images in safety-aligned or guided models [77, 62, 37]. However, these red-team methods implicitly assume that attackers are aware of specific defense mechanisms present in the T2I system. In practice, malicious attackers often lack access to the internal details of commercial black-box T2I systems, including whether pre-processing or post-processing filters are deployed or whether safety alignment has been performed, as these are packed in black-box APIs [47, 50, 30]. Thus, these methods struggle to achieve consistent performance in the most realistic and high-risk scenario commercial black-box system setting . It is significantly challenging to evade unknown and diverse defense mechanisms.

To tackle this challenge, we posit that feedback from red-team attempts yields critical prior knowledge, guiding subsequent attack strategy. Thus, we hope to leverage this experience to dynamically adapt to the defenses of real-world systems via iterative exploration. To this end, we propose a novel red-team framework Rule-based Preference modeling Guided Red-Teaming (RPG-RT) , which iteratively employs a large language model (LLM) to adapt prompts for red-team queries and uses rule-guided preference modeling to fine-tune the LLM based on the feedback from the T2I system. However, the feedback output is often labeled and coarse-grained, complicating direct use. To precisely guide LLM exploration in black-box systems, our approach employs rule-based scoring in preference modeling, using predefined rules to assess desirable and undesirable feedback. Specifically, to fine-tune LLM agents via direct preference optimization (DPO) [53] for learning the latent defense mechanisms of the target system, we identify preferred modifications from multiple query feedback, constructing a binary partial order to capture system preferences. To explore with greater fine-grained detail, we further employ a scoring model to assess the severity of harmful content in images and correct for other innocuous semantic similarities, facilitating more accurate construction of partial orders. Once fine-tuned, the LLM can modify even previously unseen prompts into those that successfully induce the target T2I system to generate harmful images.

We conduct extensive experiments on nineteen T2I systems with diverse security mechanisms to confirm the superiority of our method. The experimental results demonstrate that RPG-RT achieves an attack success rate (ASR) significantly higher than all baselines while maintaining competitive

semantic similarity. Notably, RPG-RT attains an impressive ASR on the online DALL-E 3 [47], Leonardo.ai [30], and SDXL [50] APIs, achieving at least twice the ASR of other methods, further confirming the practicality of RPG-RT. Additionally, experiments on text-to-video models also validate the flexibility and applicability of our RPG-RT.

## 2 Methodology

## 2.1 Commercial Black-box System Setting

In this paper, we diverge from previous studies by pioneering an examination of the most realistic and high-risk scenario: the commercial black-box system setting . Existing black-box red-team methods often assume knowledge about the model's specific defense mechanisms, limiting their utility in real-world commercial API scenarios, as detailed in Appendix A. Our red-team framework requires only limited access to the model outputs, better reflecting the constraints faced in real-world red-team testing scenarios, thus offering a more authentic assessment of security vulnerabilities.

The goal of the red-team framework is to explore how adversarial prompts can be crafted to induce a target text-to-image (T2I) system to generate harmful content while maintaining semantic similarity to the original image and minimizing the likelihood of triggering the model's rejection mechanism. Specifically, we assume that the original prompt P ∈ X , where X represents the natural language space, can generate harmful images M 0 ( P ) ∈ I on a model M 0 without defense mechanisms, where I denotes the image space. However, when attacking a black-box T2I system M , the prompt P may trigger a rejection by potential pre-processing or post-processing safety checkers in M , or the defense mechanisms might cause the generated image M ( P ) to lose harmful semantics. Thus, we expect the red-team assistant A to modify the prompt P to A ( P ) ∈ X in order to achieve the following objectives: 1) maximize the harmfulness of the image generated by the target model M , i.e., max A Harm ( M ( A ( P ))) , where Harm : I → R + measures the harmfulness of the image; 2) preserve semantic similarity as much as possible, i.e., max A Sim ( M ( A ( P )) , M 0 ( P )) , where Sim measures the similarity between two images. The similarity constraint is designed to enhance image quality and avoid homogeneous modifications to the original prompts. Since some T2I systems M use text or image safety checkers to reject unsafe outputs, i.e., M ( A ( P )) = reject, we consider such outputs have the lowest similarity, i.e., Sim ( reject , i ) = 0 , for all i ∈ I .

## 2.2 Overview of RPG-RT

Previous attack methods are typically tailored to T2I models and specific defense mechanisms, which limits their performance under the more realistic commercial black-box system settings (see Table 1). The challenge lies in evading unknown and diverse defense mechanisms. To address this difficulty, our key insight is that both successful and unsuccessful red-team attempts provide valuable prior knowledge that serves as a lesson to guide future red-team strategies. Consequently, we aim to leverage the past feedback to extract useful experiential information, dynamically adapting to the varied defenses of real-world black-box systems through iterative exploration. We propose a novel red-team framework, Rule-based Preference modeling Guided Red-Teaming (RPG-RT), which operates iteratively as follows: 1) Using large language models (LLMs) to automatically modify prompts for red-team queries on black-box T2I systems; 2) Performing rule-guided preference modeling and fine-tuning the LLM based on feedback from the target T2I system. However, the feedback output can be labeled and coarse-grained, posing challenges for direct utilization. To finely control the exploration of LLMs in commercial black-box systems, the core of our method lies in rule-based scoring in preference modeling-utilizing a set of rules to evaluate desired or undesirable feedback (e.g., the rejection of unsafe outputs by safety checkers, i.e., M ( A ( P )) = reject).

Specifically, as illustrated in Fig. 1, our RPG-RT operates through a multi-round cycle of query feedback and LLM fine-tuning, enabling the LLM agent to learn how to modify prompts effectively and efficiently for the target T2I black-box system, thereby automating the red-team process. In each iteration, the LLM is instructed to generate multiple modifications of the current prompt, which are then input into the target T2I black-box system. The target system responds to the modified prompts by either generating an image or returning a rejection. The detector identifies potential NSFW semantics in the generated image and provides a binary label. Meanwhile, the rule-based scoring model evaluates the harmfulness of the image at a finer granularity and corrects for other innocuous semantic similarities. Finally, we fine-tune the LLM based on the rule-guided preferences.

Figure 2: Overview of our scoring model. a): Motivation: the presence of harmful or semantically identical non-harmful semantics can lead to a high CLIP similarity between two images, causing confusion that cannot be resolved by a straightforward CLIP similarity measure. b): Our key insight is to decouple the CLIP representation using a transformation f = ( f n , f s ) , where f n captures harmful content, and f s captures other innocuous semantics, allowing separation of the representation and a clearer distinction from confusion. c): To train our scoring model, we design four loss functions tailored to address the intensity of harmful semantics, the invariance of benign semantics, the similarity between benign semantics, and the reconstructability of information.

<!-- image -->

## 2.3 Prompt Modification and Query

In this section, we introduce how RPG-RT instructs the LLM agent to refine the original prompts and queries the target T2I black-box system to obtain feedback outputs. Initially, the LLM agent is instructed to modify the original prompt with the goal of bypassing the detector and enhancing specific unsafe semantic categories, as detailed in Appendix B.1's template prompts. The LLM is tasked with N independent modifications for each original prompt, denoted as P 1 , P 2 , ..., P N , and queries the target T2I system.

̸

The feedback output from the target T2I system for P i can be categorized into three types: TYPE-1 : The T2I system's pre-processing or post-processing safety filter produces a rejection message, i.e., M ( P i ) = reject. TYPE-2 : The modified prompt P i is not rejected by the filter, but the detector D classifies the generated image as safe-for-work (SFW), i.e., ( M ( P i ) = reject ) ∧ ( D ( M ( P i )) = False ) . TYPE-3 : The modified prompt P i not only bypasses the safety filter but also results in an NSFW image classified by the detector D , i.e., ( M ( P i ) = reject ) ∧ ( D ( M ( P i )) = True ) . These three types will be further decomposed into specific rules to clearly describe the expected and unexpected behaviors, allowing for fine-grained control in modeling the preferences of the T2I black-box system.

## 2.4 Scoring Model

We employ a scoring model to assess the severity of harmful content in images and correct for other innocuous semantic similarities, facilitating more accurate preference modeling. Previous works [64, 66] leverage CLIP similarity [52] as a loss/reward function to encourage the enhancement of harmful semantics. However, we identify a key limitation: CLIP similarity measures the overall semantic similarity between images, making it insufficient for independently assessing the severity of harmful semantics or the similarity of other benign semantics. As illustrated in Fig. 2a, the presence of harmful or similar semantics can lead to a high CLIP similarity between two images and cause confusion. To address this challenge, our key insight is to decouple the CLIP representation using a transformation f = ( f n , f s ) , where f n captures the harmful content, and f s captures the other innocuous semantics, allowing for separating the representation and achieving a clearer distinction from confusion. Thus, our scoring model consists of a frozen CLIP image encoder followed by a learnable transformation f .

To train the f of scoring model, we design multiple loss functions. Let { X S i , X N i } i =1: n denote the training set, where { X S i } i =1: n represents the CLIP embeddings of n SFWimages with distinct semantics, and { X N i } i =1: n represents the CLIP embeddings of NSFW images with the same non-harmful semantics corresponding to X S i . First, for the transformation f n related to harmful content intensity,

̸

we aim for it to accurately rank the severity of NSFW content, i.e., f n ( X S i ) &lt; f n ( X N i ) , ∀ i = 1 , ..., n . To achieve this, we apply the Bradley-Terry model [3] as a ranking model, which leads to the following loss function, with σ as the Sigmoid function:

<!-- formula-not-decoded -->

Second, for the benign semantic component associated with the transformation f s , we aim to ensure that its representation remains unchanged despite increases in NSFW intensity. Specifically, for each X N i , we desire its representation in terms of other innocuous semantics to be as similar as possible to that of X S i , i.e., f s ( X S i ) ≈ f s ( X N i ) . To achieve this, we employ the following loss function:

<!-- formula-not-decoded -->

Third, we ensure that the transformation f s accurately measures the similarity of benign semantics across different images. To achieve this, we use the CLIP similarity between the SFW images as a reference, aligning the cosine similarity between the representations of other harmless semantics across different images with the CLIP similarity of the corresponding safe images, regardless of whether these images are safe or unsafe. The alignment can be expressed by the following loss:

<!-- formula-not-decoded -->

Finally, we aim to ensure that this transformation does not lead to the loss of information in the original CLIP representation. To achieve this, we introduce a reconstruction loss, which attempts to recover the original CLIP representation by applying an inverse transformation (i.e., f -1 )) to the NSFW semantics and benign semantic information. The reconstruction loss minimizes the mean squared error between the reconstructed representation and the original representation:

<!-- formula-not-decoded -->

We employ two independent single-layer neural networks to learn the transformation f = [ f n , f s ] and its inverse f -1 . The dataset for training the scoring model is constructed using images obtained from each query. Specifically, we first select n original prompts. For the i -th original prompt, we randomly select one image from its N modifications that corresponds to a TYPE-2 modification, and use its CLIP embedding as X S i . Similarly, we randomly select an image corresponding to a TYPE-3 modification, and use its CLIP embedding as X N i . These data are then used to train the scoring model in conjunction with the sum of four aforementioned loss functions:

<!-- formula-not-decoded -->

The trained scoring model can accurately distinguish NSFW scores and subsequently provide reliable guidance for scoring during preference modeling, as demonstrated in the scoring model performance evaluation analysis in Appendix D.

## 2.5 Preference Modeling

To fine-tune LLM agents using direct preference optimization [53] (DPO) for learning the latent defense mechanisms of the target T2I black-box system, we need to identify preferred modifications based on the feedback from multiple queries, effectively modeling preferences for the T2I system. Specifically, we define a binary partial order &lt; to measure preferences. Given two modified prompts, P i and P j , if P i &lt; P j , we consider P j to be more favored than P i .

We then model this binary partial order by constructing rules about preferences. Initially, we observe that only TYPE-3 corresponds to successful NSFW image outputs, which are the most desired behaviors. Compared to TYPE-3 modifications, TYPE-1 and TYPE-2 lack the ability to bypass filters or generate NSFW semantics. Thus, we establish the following foundational rules R :

- If P i ∈ TYPE-1 , P j ∈ TYPE-3 , then P i &lt; P j .
- If P i ∈ TYPE-2 , P j ∈ TYPE-3 , then P i &lt; P j .

Notably, unlike previous studies [66], we do not assume all modifications that bypass filters are better than those that are rejected (i.e., TYPE-1 &lt; TYPE-2 ). While TYPE-1 fails to generate meaningful images, the rejection signal from the filter suggests that the generated images likely contain NSFW semantics, which is partially desired.

Given that both TYPE-2 and TYPE-3 can generate meaningful images, we further construct a partial order for all modifications within each type. As discussed in Section 2.1, in addition to bypassing filters, we aim for the LLM-generated modified prompts P i to produce images M ( P i ) on the target T2I system M that maximize the harmfulness of NSFW semantics, while maintaining as much similarity as possible with the images M 0 ( P ) generated by the original prompt P on the reference T2I model M 0 without defense mechanisms. For the NSFW semantics, we use the pre-trained scoring model to compute f n ( CLIP ( M ( P i ))) , which evaluates the harmfulness of M ( P i ) . For the semantic similarity, we initially generate K reference images refs on the reference T2I model M 0 using the original prompt, and then compute the average semantic similarity of the images generated by the modified prompts to these reference images using the f s in the scoring model:

<!-- formula-not-decoded -->

To balance NSFW semantics and semantic similarity, we use the following score as the criterion for setting preference rules, with the hyperparameter c acting as the weight for semantic similarity:

<!-- formula-not-decoded -->

Consequently, we revise the preference rules R :

- If P i ∈ TYPE-1 , P j ∈ TYPE-3 , then P i &lt; P j .
- If P i ∈ TYPE-2 , P j ∈ TYPE-3 , then P i &lt; P j .
- If P i , P j ∈ TYPE-2 or P i , P j ∈ TYPE-3 and score ( P i ) &lt; score ( P j ) , then P i &lt; P j .

Some extreme cases that may hinder preference modeling are discussed in Appendix G.

## 2.6 Direct Preference Optimization

Upon modeling the generative preferences of the target T2I system, we fine-tune LLM agents using DPO based on these preference rules. Specifically, leveraging the preference rules R , we conduct pairwise comparisons among all modifications P 1 , P 2 , ..., P N of each original prompt P , establishing a binary partial order and generating a training dataset. We fine-tune the LLM using DPO with LoRA [21]. After fine-tuning, the LLM attempts to modify all selected original prompts again, and uses the newly refined prompts in further iterations until the maximum iteration limit is reached.

## 3 Experiment

## 3.1 Experimental Settings

Dataset. We consider five NSFW categories. For nudity, we select the I2P dataset [58], and choose 95 prompts with nudity above 50%. We also consider the NSFW categories including violence, politicians, discrimination, and copyrights. Details of these datasets are provided in Appendix C.1.

Detection. We select different detectors for each attack category. Specifically, to detect nudity, we use NudeNet [46]. For violence, we utlize the Q16 detector [59]. For discrimination, we employ the skin color classification algorithm CASCo [55]. For politicians, the celebrity classifier [1] is applied. For copyright, we apply the OWL-ViT [44]. More details are deferred to Appendix C.1.

Text-to-image systems. To comprehensively evaluate the red-team performance of RPG-RT, we select T2I systems that include a variety of state-of-the-art defense methods, including detection-based defenses, removal-based defenses, safety-aligned T2I models, combinations of multiple defenses, and online API services. For the detection-based defenses, we choose Stable Diffusion v1.4 [56] as the T2I model and involve six different detectors: text filter (text-match) with a predefined NSFW vocabulary [20], NSFW text classifier (text-cls) [32], GuardT2I [65], an open-source image classifier (img-cls) [7], image classifier (img-clip) [29] based on CLIP embeddings and the built-in textimage similarity-based filter in SD1.4 (text-img) [56]. For the removal-based defenses, we consider ESD [18], Safe Latent Diffusion (SLD) [58] under the two strongest settings (namely SLD-strong and

Table 1: Quantitative results of baselines and our RPG-RT in generating images with nudity semantics on nineteen T2I systems equipped with various defense mechanisms. Our RPG-RT achieves an ASR that surpasses all baselines on nearly all T2I systems, while also maintaining competitive semantic similarity in terms of FID.

|                                  |             | White-box     |             |             |              | Black-box Ring-A-Bell   |              |              |
|----------------------------------|-------------|---------------|-------------|-------------|--------------|-------------------------|--------------|--------------|
|                                  |             | MMA-Diffusion | P4D-K       | P4D-N       | SneakyPrompt |                         | FLIRT        | RPG-RT       |
| text-match                       | ASR ↑       | 19.86         | 28.28       | 11.86       | 29.30        | 0.74                    | 34.56        | 80.98        |
|                                  | FID ↓       | 65.59         | 54.67       | 81.11       | 60.17        | 215.02                  | 111.71       | 52.25        |
|                                  | ASR ↑       | 6.84          | 24.56       | 9.02        | 43.12        | 1.02                    | 30.00        | 63.19        |
| text-cls                         | FID ↓       | 87.19         | 55.25       | 72.52       | 59.63        | 177.33                  | 134.23       | 51.61        |
|                                  | ASR ↑       | 3.65          | 10.88       | 2.04        | 13.44        | 0.00                    | 25.69        | 32.49        |
| GuardT2I                         | FID ↓       | 118.32        | 58.82       | 77.18       | 77.45        | --                      | 151.89       | 56.91        |
|                                  | ASR ↑       | 54.98         | 64.88       | 57.75       | 50.21        | 79.54                   | 49.82        | 86.32        |
| img-cls                          | FID ↓       | 54.71         | 49.30       | 59.57       | 56.52        | 73.93                   | 85.11        | 59.14        |
|                                  | ASR ↑       | 35.40         | 42.84       | 34.98       | 37.51        | 43.51                   | 37.72        | 63.23        |
| img-clip                         | FID ↓       | 60.04         | 54.45       | 66.59       | 65.20        | 75.91                   | 103.98       | 55.99        |
|                                  | ASR ↑       | 14.91         | 14.39       | 14.00       | 14.39        | 3.01                    | 14.91        | 43.16        |
| text-img                         | FID ↓       | 76.02         | 60.15       | 77.56       | 90.01        | 85.67                   | 140.52       | 76.18        |
|                                  | ASR ↑       | 24.49         | 29.93       | 31.37       | 20.60        | 72.46                   | 41.93        | 76.95        |
| SLD-strong                       | FID ↓       | 84.29         | 77.15       | 76.73       | 91.22        | 63.78                   | 81.13        | 58.58        |
|                                  | ASR ↑       | 15.72         | 18.07       | 23.93       | 12.53        | 44.88                   | 26.14        | 41.15        |
| SLD-max                          | FID ↓       | 100.43        | 96.78       | 89.52       | 108.01       | 79.72                   | 98.01        | 71.64        |
|                                  | ASR ↑       | 11.16         | 29.12       | 32.14       | 8.46         | 31.05                   | 13.86        | 62.91        |
| ESD                              | FID ↓       | 101.34        | 79.68       | 84.26       | 115.72       | 97.13                   | 119.87       | 64.47        |
|                                  | ASR ↑       | 12.56         | 15.19       | 11.16       | 9.12         | 22.04                   | 15.26        | 82.98        |
| SD-NP                            | FID ↓       | 105.93        | 101.33      | 121.95      | 115.56       | 100.71                  | 110.35       | 58.32        |
|                                  | ASR ↑       | 22.18         | 24.74       | 3.65        | 22.98        | 29.72                   | 20.88        | 55.12        |
| SafeGen                          | FID ↓       | 110.23        | 101.01      | 159.01      | 108.96       | 148.87                  | 116.35       | 84.32        |
|                                  | ASR ↑       | 0.95          | 0.98        | 0.67        | 0.74         | 0.25                    | 1.93         | 40.35        |
| AdvUnlearn                       | FID ↓       | 166.85        | 161.01      | 174.48      | 173.26       | 185.75                  | 176.83       | 77.19        |
| DUO                              | ASR ↑       | 9.65          | 6.95        | 4.63        | 11.30        | 18.42                   | 12.28 109.04 | 47.05 74.48  |
|                                  | FID ↓       | 85.38         | 94.64       | 109.79      | 85.72        | 92.48                   |              | 95.02        |
| SAFREE                           | ASR ↑       | 16.77         | 22.39       | 17.19       | 12.98        | 64.42                   | 37.02        |              |
|                                  | FID ↓       | 97.43         | 95.4        | 112.56      | 101.71       | 85.19                   | 103.36       | 81.92        |
| SD v2.1                          | ASR ↑       | 39.02 17.96   | -- --       | -- --       | 75.83        | 73.72                   | 51.93 71.59  | 97.85        |
|                                  | FID ↓       | 65.04         | --          | --          | 33.30        | 78.21                   | 36.14        | 73.71        |
| SD v3                            | ASR ↑ FID ↓ | 89.59         | --          | --          | 17.96 90.67  | 60.04 72.54             | 92.70        | 97.26 87.78  |
|                                  | ASR ↑       | 22.06         | 7.40        | 40.70       | 19.58        | 72.39                   | 31.40        | 80.25        |
| SafetyDPO                        | FID ↓       | 82.00         | 91.71       | 73.74       | 90.55        | 64.09                   | 86.89        | 56.8         |
|                                  | ASR ↑       |               |             | 13.56       |              |                         |              | 34.17        |
| text-img +                       |             | 10.33         | 14.11       |             | 14.56        | 2.11                    | 12.78        | 112.20       |
| SLD-strong                       | FID ↓       | 150.66        | 146.52      | 162.98      | 143.28       | 209.93                  | 135.44       |              |
| text-img + text-cls + SLD-strong | ASR ↑ FID ↓ | 1.33 188.38   | 3.78 175.05 | 3.56 206.90 | 4.78 138.36  | 0.00 --                 | 5.67 145.22  | 13.89 127.65 |

SLD-max), Stable Diffusion with the negative prompt (SD-NP) [56], SafeGen [33], AdvUnlearn [72], DUO [49], and adaptive defense SAFREE [67]. For the safety-aligned models, we utilize Stable Diffusion v2.1 (SD2) [56], v3 (SD3) [17], and SafetyDPO [36]. We also examine RPG-RT against multiple defenses simultaneously, including the combination of text-img + SLD-strong and text-img + text-cls + SLD-strong, as well as three online T2I API services, DALL-E 3 [47], Leonardo.ai [30], and Stable Diffusion XL [50] (SDXL), with a text-to-video model, Open-Sora [75].

Baselines. We compare RPG-RT with state-of-the-art black-box and white-box red-team methods. For black-box attacks, we select Ring-A-Bell [62], SneakyPrompt [66], and FLIRT [38]. For whitebox methods, we choose the MMA-Diffusion [64] and two variants of P4D (P4D-K and P4D-N) [9].

Metrics. We use four metrics to evaluate the performance of RPG-RT from multiple perspectives. First, we use the Attack Success Rate (ASR) to measure the proportion of modified prompts that successfully lead to NSFW semantics. To account for a more challenging setting, we generate 30 images with the modified prompts without fixing the random seed for each original prompt and compute the ASR. Second, we use the CLIP Similarity (CS) and Fréchet Inception Distance (FID) to assess the preservation of semantics. The CS is the average CLIP similarity between all generated images and their corresponding five reference images generated by Stable Diffusion v1.4, while FID refers to the Fréchet Inception Distance between all generated images and the reference images. Third, we use Perplexity (PPL) to measure the stealthiness level of the modified prompt. Note that higher ASR and CS indicate better performance, while lower FID and PPL are preferable.

RPG-RT Details. For the LLM agent, we select the unaligned Vicuna-7B model [8]. For the prompt modification, we perform 30 modifications for each original prompt to ensure sufficient data for fine-tuning. For the preference modeling, we set the parameter c to 2 to achieve a good balance between ASR and semantic preservation. More details are deferred to Appendix B.2.

Table 2: Quantitative results of baselines and RPG-RT across various NSFW types. RPG-RT delivers best ASR.

|            |             | White-box     | White-box    | White-box    | Black-box    | Black-box   | Black-box    | Black-box    |
|------------|-------------|---------------|--------------|--------------|--------------|-------------|--------------|--------------|
|            |             | MMA-Diffusion | P4D-K        | P4D-N        | SneakyPrompt | Ring-A-Bell | FLIRT        | RPG-RT       |
| GuardT2I   | ASR ↑ ↓     | 15.44         | 4.67         | 0.00         | 44.33        | 0.22        | 35.56        | 46.56        |
|            | FID         | 192.07        | 250.73       | --           | 159.07       | 197.29      | 284.42       | 169.98       |
| SLD-strong | ASR ↑       | 17.44         | 18.11        | 7.67         | 11.11        | 3.56        | 28.33        | 62.44        |
|            | FID ↓       | 178.61        | 178.06       | 194.51       | 188.42       | 188.41      | 227.38       | 193.58       |
|            | ASR ↑       | 3.11          | 2.11         | 2.33         | 48.22        | --          | 50.00        | 53.33        |
| GuardT2I   | FID ↓       | 305.5         | 355.75       | 295.74       | 137.59       | --          | 303.28       | 149.26 69.44 |
| SLD-strong | ASR ↑ FID ↓ | 56.67 135.16  | 63.33 140.26 | 48.56 177.81 | 49.22 140.28 | -- --       | 61.67 214.09 | 138.57       |
| GuardT2I   | ASR ↑       | 3.22          | 0.00         | 0.00         | 15.67        | --          | 6.11         | 41.00        |
|            | FID ↓       | 142.77        | --           | 197.61       | 129.90       | --          | 350.28       | 140.75       |
| SLD-strong | ASR ↑       | 4.56          | 7.11         | 0.00         | 2.89         | --          | 9.44         | 10.56        |
|            | FID ↓       | 142.77        | 139.45       | 160.06       | 141.05       | --          | 199.15       | 134.45       |
| GuardT2I   | ASR ↑       | 6.00          | 0.00         | 0.00         | 20.11        | --          | 5.00         | 41.89        |
|            | FID ↓       | 184.55        | 287.08       | 259.67       | 165.09       | --          | 319.24       | 120.41       |
| SLD-strong | ASR ↑       | 15.67         | 2.00         | 0.00         | 11.22        | --          | 5.56         | 50.78        |
|            | FID ↓       | 144.99        | 142.99       | 166.20       | 223.17       | --          | 236.35       | 158.20       |

Figure 3: Qualitative visualization results of baselines and our RPG-RT. Our RPG-RT can a): effectively bypass the safety checker and generate images across various NSFW categories, b): generate pornographic images on multiple APIs, and c): generalize to text-to-video systems.

<!-- image -->

## 3.2 Main Results

We demonstrate the effectiveness of our RPG-RT in generating images with nudity semantics on nineteen T2I systems equipped with various defense mechanisms. As shown in Table 1 and Table 7, our RPG-RT achieves an ASR that surpasses all baselines on nearly all T2I systems, while also maintaining competitive semantic similarity in terms of CS and FID. Even when facing the strongest defense, AdvUnlearn, RPG-RT still achieves an ASR greater than 40% with the highest semantic similarity, far exceeding the second-place ASR of 2.04%, indicating RPG-RT's significant advantage. Furthermore, RPG-RT ensures the modified prompts have the lowest PPL among all methods, making the attack more stealthy. We visualize generated images in Fig. 3a, where RPG-RT effectively bypasses the safety checker and generates images with nudity content on models with safety guidance or alignment, while preserving the original semantics simultaneously. Full results are presented in Appendix C.2 with more case studies and analysis of the modified prompts in Appendix E. In addition, we also provide the confidence intervals, covariance, and one-sided paired Wilcoxon signed-rank test for RPG-RT in Appendix C.8, further validating RPG-RT's exceptional robustness.

It is worth noting that some methods do not generalize well across T2I systems with different defense mechanisms: P4D aligns the noise of target T2I systems with the T2I model without defense, limiting its use on newer versions of SD v2.1 and v3; Ring-A-Bell enhances NSFW semantics and performs well against removal-based defenses, but fails to effectively bypass the safety checkers. When facing the combinations of multiple different defense mechanisms, all baselines struggle to achieve ideal ASR. In contrast, RPG-RT operates with a commercial black-box system setting, easily

Table 3: Quantitative results of baselines and RPG-RT on unseen prompts in the nudity category for text-img, SD v3, and SLD-strong. Our RPG-RT achieves the highest ASR, which demonstrates the transferability of RPG-RT.

|            |       | White-box     | White-box   | White-box   | Black-box    | Black-box   | Black-box   |        |
|------------|-------|---------------|-------------|-------------|--------------|-------------|-------------|--------|
|            |       | MMA-Diffusion | P4D-K       | P4D-N       | SneakyPrompt | Ring-A-Bell | FLIRT       | RPG-RT |
| text-img   | ASR ↑ | 15.04         | 15.57       | 12.23       | 15.53        | 3.30        | 6.85        | 37.94  |
| text-img   | FID ↓ | 67.10         | 67.00       | 82.51       | 74.11        | 147.19      | 152.16      | 79.85  |
| SD v3      | ASR ↑ | 15.74         | --          | --          | 20.32        | 57.27       | 34.07       | 96.77  |
| SD v3      | FID ↓ | 89.01         | --          | --          | 88.13        | 80.1        | 99.16       | 87.50  |
| SLD-strong | ASR ↑ | 16.24         | 21.03       | 27.30       | 14.50        | 69.15       | 21.85       | 69.50  |
| SLD-strong | FID ↓ | 79.91         | 78.15       | 76.61       | 86.76        | 74.87       | 107.9       | 65.48  |

generalizes across various defense mechanisms in T2I models and achieves consistent performance, demonstrating its superiority in real-world red-team testing scenarios.

## 3.3 Red-teaming on Different NSFW Categories

In addition to generating images with nudity content, RPG-RT also effectively performs red-teaming across various NSFW categories, including generating inappropriate content such as violence and racial discrimination, and infringement content involving specific politicians or trademarks. To simulate these adversarial scenarios, we select the removal-based SLD-strong and detection-based GuardT2I as defense methods, using the generated keywords as defense guidance or the safety checker's word list. As shown in Table 2 and Table 8, for these four NSFW categories, RPG-RT still achieves superior attack success rates and PPL compared to all other methods while capable of comparable semantic similarity, indicating its strong generalization ability across these four categories and potentially broader NSFW categories. Visualizations are provided in Fig. 3a and Appendix C.3.

## 3.4 Transferring to Unseen Prompts

In this section, we demonstrate that the fine-tuned LLM agent in RPG-RT can modify any prompt, including those that have never been seen in training data before. To assess this transferability, we conduct experiments on the nudity category for text-img, SD v3, SLD-strong. We select 94 prompts from I2P with nudity percentages between 30% to 50%, which have no overlap with the training data. We directly evaluate the trained RPG-RT without further fine-tuning, whereas other methods are re-optimized on the new data. The results in Table 3 show that, even in this direct transfer scenario, RPG-RT still significantly outperforms other methods, exhibiting the highest ASR. This result indicates that, compared to other methods that require re-optimization on new prompts and consume substantial computational resources, our proposed RPG-RT only requires an inference forward of LLM agent to perform red-teaming, demonstrating its superior effectiveness and efficiency.

## 3.5 Red-teaming the Online T2I Service

Given the features of online T2I services as commercial black-box systems with strict defense levels, red-team methods often have to confront multiple unknown defense mechanisms, which presents a more challenging scenario for generating NSFW images. To evaluate the performance of RPG-RT on a real-world commercial black-box T2I system, we select 10 prompts of the nudity category and conduct experiments on multiple online APIs, including DALL-E 3 [47], Leonardo.ai [30], and SDXL [50]. As shown in

Table 4: Quantitative results of baselines and our RPGRT on three online commercial APIs. Our RPG-RT achieves at least twice ASR of other methods.

|          |             |   Sneaky. |   Ring. |   FLIRT |   RPG-RT |
|----------|-------------|-----------|---------|---------|----------|
| DALL-E 3 | ASR ↑ FID ↓ |      4.67 |    0.67 |    0    |    31.33 |
| DALL-E 3 |             |    248.92 |  319.48 |  378.65 |   192.11 |
| Leonardo | ASR ↑       |     22.67 |    7.33 |   13.33 |    67.67 |
| Leonardo | FID ↓       |    207.78 |  265.48 |  242.1  |   160.88 |
| SDXL     | ASR ↑       |     11.67 |    6    |    0    |    20.33 |
| SDXL     | FID ↓       |    246.23 |  296.79 |  294.04 |   237.14 |

Table 4, RPG-RT achieves outstanding performance, particularly on DALL-E 3, where it attains a remarkable 31.33% ASR while all other baseline models fall below 5%. For the other two API services, RPG-RT also demonstrates at least twice the ASR of baseline methods. These results confirm that our proposed commercial black-box T2I system settings closely mirror real-world scenarios and enable our model to achieve remarkable performance. We provide examples of inappropriate images generated by online services in Fig. 3b.

## 3.6 Generalization to Text-to-Video Systems

As a flexible red-team framework, RPG-RT can also be applied to red-team text-to-video (T2V) models. We enable RPG-RT to target the T2V model OpenSora [75] for generating videos with inappropriate semantics. Since generating long videos is time and computationally consuming, we generate individual frames during the fine-

Table 5: Quantitative results of baselines and our RPGRT on text-to-video systems. Our RPG-RT achieves the highest ASR, further validating the flexibility and applicability of RPG-RT.

|       |   SneakyPrompt |   Ring-A-Bell |   FLIRT |   RPG-RT |
|-------|----------------|---------------|---------|----------|
| ASR ↑ |          18.67 |         35.33 |   23.33 |    67.33 |

tuning phase for rule-based preference modeling, and generate videos only in the final evaluation. Illustrating with the nudity category as an example, we visualize the generated videos in Fig. 3c. It could be observed that RPG-RT successfully generates NSFW videos and significantly outperforms other baselines in terms of ASR, as shown in Table 5, demonstrating its flexibility to be applied to text-to-video red-teaming.

## 3.7 Ablation Study, Computational Cost, and Additional Experiments

We conduct ablation studies by removing each loss term individually to demonstrate their impacts. As shown in the Table 9, RPG-RT without L harm fails to achieve a competitive ASR. The variants without L sim and L rec also fail to achieve comparable ASR, as the lack of aligned similarity disrupts the learning process. The variant without L inno fails to maintain semantic similarity while achieving attack success, as detailed in Appendix C.4. In addition, we report the computational cost in Table 15. Although RPG-RT requires more time and queries to train the model, it only needs a single LLM inference when generalizing to unseen prompts. For scenarios where red-teaming is needed for new N prompts, especially when N is large, RPG-RT demonstrates a significant advantage in terms of computational resources. Moreover, we present more additional experiments and analyses, including detailed ablation analysis (App. C.4), influence of weight c (App. C.4), generalization across different T2I systems (App. C.5), different reference models (App. C.6) and generation settings (App. C.7), robustness evaluation (App. C.8), scoring model evaluation (App. D), case study of modified prompts (App. E), optimization trends (App. F), and results for more evaluation metrics (App. H).

## 4 Conclusion

In this paper, we introduce a novel framework for red-teaming black-box T2I systems, termed Rulebased Preference modeling Guided Red-Teaming (RPG-RT). RPG-RT employs an iterative process that begins with utilizing LLMs to adapt prompts. Subsequently, it applies rule-guided preference modeling and fine-tunes the LLM based on feedback. We propose a rule-based scoring mechanism in preference modeling to finely control LLM exploration in black-box systems. Extensive experiments consistently validate the superiority of RPG-RT, especially impressive on online APIs.

## Acknowledgments

This paper is supported by the Strategic Priority Research Program of CAS Grant XDA0480502, the Robotic AI-Scientist Platform of Chinese Academy of Sciences, and NSFC Grants 12288201, 62276149 and 92270001. The authors thank anonymous referees for their valuable comments.

## References

- [1] Tony Assi. Celebrity classifier, 2024.
- [2] Zhongjie Ba, Jieming Zhong, Jiachen Lei, Peng Cheng, Qinglong Wang, Zhan Qin, Zhibo Wang, and Kui Ren. Surrogateprompt: Bypassing the safety filter of text-to-image models via substitution. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security , pages 1166-1180, 2024.
- [3] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika , 39(3/4):324-345, 1952.
- [4] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE symposium on security and privacy (SP) , pages 39-57. IEEE, 2017.

- [5] Huanran Chen, Yinpeng Dong, Shitong Shao, Hao Zhongkai, Xiao Yang, Hang Su, and Jun Zhu. Diffusion models are certifiably robust classifiers. Advances in Neural Information Processing Systems , 37:5006250097, 2024.
- [6] Shuyu Cheng, Yibo Miao, Yinpeng Dong, Xiao Yang, Xiao-Shan Gao, and Jun Zhu. Efficient black-box adversarial attacks via bayesian optimization guided by a function prior. In International Conference on Machine Learning , pages 8163-8183, 2024.
- [7] Lakshay Chhabra. Nsfw classifier, 2020.
- [8] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023) , 2(3):6, 2023.
- [9] Zhi-Yi Chin, Chieh Ming Jiang, Ching-Chun Huang, Pin-Yu Chen, and Wei-Chen Chiu. Prompting4debugging: Red-teaming text-to-image diffusion models by finding problematic prompts. In International Conference on Learning Representations , 2024.
- [10] Angela Daly, Thilo Hagendorff, Li Hui, Monique Mann, Vidushi Marda, Ben Wagner, and Wayne Wei Wang. Ai, governance and ethics: Global perspectives. Deakin University , 2022.
- [11] Pucheng Dang, Xing Hu, Dong Li, Rui Zhang, Qi Guo, and Kaidi Xu. DiffZOO: A purely querybased black-box attack for red-teaming text-to-image generative model via zeroth order optimization. In Findings of the Association for Computational Linguistics: NAACL 2025 , pages 17-31. Association for Computational Linguistics, 2025.
- [12] Yimo Deng and Huangxun Chen. Divide-and-conquer attack: Harnessing the power of llm to bypass the censorship of text-to-image generation model. arXiv preprint arXiv:2312.07130 , 2023.
- [13] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems , volume 34, pages 8780-8794, 2021.
- [14] Yingkai Dong, Xiangtao Meng, Ning Yu, Zheng Li, and Shanqing Guo. Fuzz-testing meets llm-based agents: An automated and efficient framework for jailbreaking text-to-image generation models. In 2025 IEEE Symposium on Security and Privacy (SP) , pages 336-354. IEEE Computer Society, 2025.
- [15] Yinpeng Dong, Huanran Chen, Jiawei Chen, Zhengwei Fang, Xiao Yang, Yichi Zhang, Yu Tian, Hang Su, and Jun Zhu. How robust is google's bard to adversarial image attacks? arXiv preprint arXiv:2309.11751 , 2023.
- [16] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and Jianguo Li. Boosting adversarial attacks with momentum. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 9185-9193, 2018.
- [17] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning , 2024.
- [18] Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 2426-2436, 2023.
- [19] Sensen Gao, Xiaojun Jia, Yihao Huang, Ranjie Duan, Jindong Gu, Yang Liu, and Qing Guo. Rt-attack: Jailbreaking text-to-image models via random token. arXiv preprint arXiv:2408.13896 , 2024.
- [20] Rojit George. Nsfw words list, 2020.
- [21] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations , 2022.
- [22] Yuepeng Hu, Zhengyuan Jiang, and Neil Zhenqiang Gong. Safetext: Safe text-to-image models via aligning the text encoder. arXiv preprint arXiv:2502.20623 , 2025.
- [23] Yao Huang, Yitong Sun, Shouwei Ruan, Yichi Zhang, Yinpeng Dong, and Xingxing Wei. Breaking the ceiling: Exploring the potential of jailbreak attacks through expanding strategy space. arXiv preprint arXiv:2505.21277 , 2025.

- [24] Yihao Huang, Le Liang, Tianlin Li, Xiaojun Jia, Run Wang, Weikai Miao, Geguang Pu, and Yang Liu. Perception-guided jailbreak against text-to-image models. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, pages 26238-26247, 2025.
- [25] Sanghyun Kim, Moonseok Choi, Jinwoo Shin, and Juho Lee. Safety alignment backfires: Preventing the re-emergence of suppressed concepts in fine-tuned text-to-image diffusion models. arXiv preprint arXiv:2412.00357 , 2024.
- [26] Diederik P Kingma. Adam: A method for stochastic optimization. In International Conference on Learning Representations , 2015.
- [27] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu. Ablating concepts in text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 22691-22702, 2023.
- [28] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742 , 2025.
- [29] LAION-AI. Clip-based nsfw detector, 2023.
- [30] Leonardo.ai. Leonardo.ai, 2022.
- [31] Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, and Tianwei Zhang. ART: Automatic red-teaming for text-to-image models to protect benign users. In Advances in Neural Information Processing Systems , 2024.
- [32] Michelle Li. Nsfw text classifier, 2022.
- [33] Xinfeng Li, Yuchen Yang, Jiangyi Deng, Chen Yan, Yanjiao Chen, Xiaoyu Ji, and Wenyuan Xu. Safegen: Mitigating sexually explicit content generation in text-to-image models. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security , pages 4807-4821, 2024.
- [34] Jiangtao Liu, Zhaoxin Wang, Handing Wang, Cong Tian, and Yaochu Jin. Token-level constraint boundary search for jailbreaking text-to-image models. arXiv preprint arXiv:2504.11106 , 2025.
- [35] Jiayang Liu, Siyuan Liang, Shiqian Zhao, Rongcheng Tu, Wenbo Zhou, Xiaochun Cao, Dacheng Tao, and Siew Kei Lam. Jailbreaking the text-to-video generative models. arXiv preprint arXiv:2505.06679 , 2025.
- [36] Runtao Liu, Chen I Chieh, Jindong Gu, Jipeng Zhang, Renjie Pi, Qifeng Chen, Philip Torr, Ashkan Khakzar, and Fabio Pizzati. Safetydpo: Scalable safety alignment for text-to-image generation. arXiv preprint arXiv:2412.10493 , 2024.
- [37] Jiachen Ma, Anda Cao, Zhiqing Xiao, Yijiang Li, Jie Zhang, Chao Ye, and Junbo Zhao. Jailbreaking prompt attack: A controllable adversarial attack against diffusion models. arXiv preprint arXiv:2404.02928 , 2024.
- [38] Ninareh Mehrabi, Palash Goyal, Christophe Dupuy, Qian Hu, Shalini Ghosh, Richard Zemel, Kai-Wei Chang, Aram Galstyan, and Rahul Gupta. FLIRT: Feedback loop in-context red teaming. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing , pages 703-718. Association for Computational Linguistics, 2024.
- [39] Yibo Miao, Yinpeng Dong, Jinlai Zhang, Lijia Yu, Xiao Yang, and Xiao-Shan Gao. Improving robustness of 3d point cloud recognition from a fourier perspective. Advances in Neural Information Processing Systems , 37:68183-68210, 2024.
- [40] Yibo Miao, Yinpeng Dong, Jun Zhu, and Xiao-Shan Gao. Isometric 3d adversarial examples in the physical world. In Advances in Neural Information Processing Systems , volume 35, pages 19716-19731, 2022.
- [41] Yibo Miao, Yifan Zhu, Lijia Yu, Jun Zhu, Xiao-Shan Gao, and Yinpeng Dong. T2vsafetybench: Evaluating the safety of text-to-video generative models. In Advances in Neural Information Processing Systems , volume 37, pages 63858-63872, 2024.
- [42] Midjourney. Midjourney, 2022.
- [43] Dan Milmo and Alex Hern. Google chief admits 'biased' ai tool's photo diversity offended users, 2024.

- [44] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple openvocabulary object detection. In European Conference on Computer Vision , pages 728-755. Springer, 2022.
- [45] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 , 2021.
- [46] notAI.tech. Nudenet, 2024.
- [47] OpenAI. Dall-e 3, 2023.
- [48] OpenAI. Gpt-4o, 2024.
- [49] Yong-Hyun Park, Sangdoo Yun, Jin-Hwa Kim, Junho Kim, Geonhui Jang, Yonghyun Jeong, Junghyo Jo, and Gayoung Lee. Direct unlearning optimization for robust and safe text-to-image models. In Advances in Neural Information Processing Systems , 2024.
- [50] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations , 2024.
- [51] Yiting Qu, Xinyue Shen, Xinlei He, Michael Backes, Savvas Zannettou, and Yang Zhang. Unsafe diffusion: On the generation of unsafe images and hateful memes from text-to-image models. In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security , pages 3403-3417, 2023.
- [52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pages 8748-8763. PMLR, 2021.
- [53] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems , volume 36, pages 53728-53741, 2023.
- [54] Javier Rando, Daniel Paleka, David Lindner, Lennart Heim, and Florian Tramèr. Red-teaming the stable diffusion safety filter. arXiv preprint arXiv:2210.04610 , 2022.
- [55] René Alejandro Rejón Piña and Chenglong Ma. Classification algorithm for skin color (casco): A new tool to measure skin color in social science research. Social Science Quarterly , 104(2):168-179, 2023.
- [56] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 10684-10695, 2022.
- [57] Shouwei Ruan, Zhenyu Wu, Yao Huang, Ruochen Zhang, Yitong Sun, Caixin Kang, and Xingxing Wei. Towards nsfw-free text-to-image generation via safety-constraint direct preference optimization. arXiv preprint arXiv:2504.14290 , 2025.
- [58] Patrick Schramowski, Manuel Brack, Björn Deiseroth, and Kristian Kersting. Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 22522-22531, 2023.
- [59] Patrick Schramowski, Christopher Tauchmann, and Kristian Kersting. Can machines help us answering question 16 in datasheets, and in turn reflecting on inappropriate content? In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency , pages 1350-1361, 2022.
- [60] Yitong Sun, Yao Huang, Ruochen Zhang, Huanran Chen, Shouwei Ruan, Ranjie Duan, and Xingxing Wei. Ndm: A noise-driven detection and mitigation framework against implicit sexual intentions in text-to-image generation. arXiv preprint arXiv:2510.15752 , 2025.
- [61] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199 , 2013.
- [62] Yu-Lin Tsai, Chia-yi Hsu, Chulin Xie, Chih-hsun Lin, Jia You Chen, Bo Li, Pin-Yu Chen, Chia-Mu Yu, and Chun-ying Huang. Ring-a-bell! how reliable are concept removal methods for diffusion models? In International Conference on Learning Representations , 2024.

- [63] Haonan Wang, Qianli Shen, Yao Tong, Yang Zhang, and Kenji Kawaguchi. The stronger the diffusion model, the easier the backdoor: Data poisoning to induce copyright breacheswithout adjusting finetuning pipeline. In Forty-first International Conference on Machine Learning , 2024.
- [64] Yijun Yang, Ruiyuan Gao, Xiaosen Wang, Tsung-Yi Ho, Nan Xu, and Qiang Xu. Mma-diffusion: Multimodal attack on diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 7737-7746, 2024.
- [65] Yijun Yang, Ruiyuan Gao, Xiao Yang, Jianyuan Zhong, and Qiang Xu. Guardt2i: Defending text-to-image models from adversarial prompts. In Advances in neural information processing systems , 2024.
- [66] Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, and Yinzhi Cao. Sneakyprompt: Jailbreaking text-to-image generative models. In 2024 IEEE symposium on security and privacy (SP) , pages 897-912. IEEE, 2024.
- [67] Jaehong Yoon, Shoubin Yu, Vaidehi Patil, Huaxiu Yao, and Mohit Bansal. Safree: Training-free and adaptive guard for safe text-to-image and video generation. In International Conference on Learning Representations , 2025.
- [68] Lijia Yu, Shuang Liu, Yibo Miao, Xiao-Shan Gao, and Lijun Zhang. Generalization bound and new algorithm for clean-label backdoor attack. In International Conference on Machine Learning , pages 57559-57596, 2024.
- [69] Chenyu Zhang, Lanjun Wang, Yiwen Ma, Wenhui Li, and An-An Liu. Reason2attack: Jailbreaking text-to-image models via llm reasoning. arXiv preprint arXiv:2503.17987 , 2025.
- [70] Gong Zhang, Kai Wang, Xingqian Xu, Zhangyang Wang, and Humphrey Shi. Forget-me-not: Learning to forget in text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 1755-1764, 2024.
- [71] Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, et al. Multitrust: A comprehensive benchmark towards trustworthy multimodal large language models. Advances in Neural Information Processing Systems , 37:49279-49383, 2024.
- [72] Yimeng Zhang, Xin Chen, Jinghan Jia, Yihua Zhang, Chongyu Fan, Jiancheng Liu, Mingyi Hong, Ke Ding, and Sijia Liu. Defensive unlearning with adversarial training for robust concept erasure in diffusion models. In Advances in neural information processing systems , 2024.
- [73] Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, and Sijia Liu. To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images... for now. In European Conference on Computer Vision , pages 385-403. Springer, 2025.
- [74] Wendi Zheng, Jiayan Teng, Zhuoyi Yang, Weihan Wang, Jidong Chen, Xiaotao Gu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogview3: Finer and faster text-to-image generation via relay diffusion. In European Conference on Computer Vision , pages 1-22. Springer, 2024.
- [75] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404 , 2024.
- [76] Yifan Zhu, Yibo Miao, Yinpeng Dong, and Xiao-Shan Gao. Toward availability attacks in 3d point clouds. In International Conference on Machine Learning , pages 62510-62530, 2024.
- [77] Haomin Zhuang, Yihua Zhang, and Sijia Liu. A pilot study of query-free adversarial attack against stable diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 2385-2392, 2023.

## A Related Work

## A.1 Jailbreak on Text-to-Image Models

Deep learning safety has been extensively studied [61, 4, 16, 40, 6, 39, 76, 68, 15, 71, 5, 23]. Building on this, to uncover the potential safety vulnerabilities in text-to-image (T2I) models, a variety of red-teaming methods have been developed to explore jailbreak attacks on these models [64, 9, 73, 62, 66, 11, 24, 35, 41]. These methods can be broadly categorized into white-box and black-box attacks. White-box attacks aim to fully exploit the safety weaknesses of T2I models by leveraging access to the model parameters and gradients. MMADiffusion [64] bypasses the safety checkers by optimizing both the text and image modalities . P4D [9] and UnlearnDiff [73] attempt to align the U-Net noise output of the model equipped with a safety mechanism with the unconstrained model, thereby generating the NSFW content. However, white-box attack suffers limitations in practical scenarios, as commercial APIs typically do not provide access to gradients and model parameters. On the other hand, black-box methods are more practical, optimizing based solely on the queried response from the T2I model or the output of commonly used text encoders. QF-Attack [77], Ring-A-Bell [62], and JPA [37] utilize the CLIP text encoder [52] as a reference, attempting to enhance malicious semantics in the original prompt through optimization. SneakyPrompt [66], HTS-Attack [19], and TCBS-Attack [34] specialize in token-level search strategies to bypass safety checkers, while DACA [12], SurrogatePrompt [2] and Atlas [14] further use prompt engineering techniques to guide LLMs. FLIRT [38] designs a query-response ranking approach to guide large language models (LLMs) to modify model outputs via in-context learning. Furthermore, ART [31] explores the potential for generating NSFW images through SFW prompts, in collaboration with both LLMs and vision-language models (VLMs). Beyond these, PGJ [24] focuses on exploiting perception similarity and text semantic inconsistency to evade text detectors, whereas DiffZOO [11] adopts a gradient-based black-box optimization approach. Reason2Attack [69] constructs chain-of-thought reasoning to refine adversarial prompts. However, these methods struggle to be effective under commercial black-box systems in practice, facing diverse and unknown defense mechanisms.

## A.2 Text-to-Image Model with Safety Mechanisms

Due to growing concerns over the malicious exploitation of T2I models, various defense mechanisms have been developed to prevent the generation of NSFW images [58, 18, 65, 27, 70, 36, 60, 57]. These defense strategies generally fall into two categories: detection-based methods and concept removal-based methods. Detection-based methods involve extra detectors within the model to filter inputs or outputs [65, 56]. These methods use text classifiers or image classifiers to detect potentially harmful queries and reject images that may contain NSFW content, thereby preventing the generation of malicious outputs. On the other hand, concept removal-based methods aim to remove specific NSFW concepts by performing the safety guidance model during inference [58, 67] or additional safety fine-tuning [18, 27, 70, 33, 72, 49] or alignment [36, 25, 22] to eliminate these concepts from the model's parameters entirely. In addition to these defense mechanisms, some commercial models attempt to filter NSFW data before the training phase or employ some unknown defense strategies to address the challenge of generating unsafe outputs [17, 42, 47]. To demonstrate the effectiveness of our proposed red team approach, we evaluate it across nineteen T2I models with varying security mechanisms, allowing us to assess our method's performance across a spectrum of security strategies.

## B Details of RPG-RT

## B.1 Prompt Template for RPG-RT LLM Agent

To instruct the LLM agent in RPG-RT to modify the original prompt, we design the prompt template as shown in Table 6. The LLM is informed that the original prompt may trigger safety filters or fail to maintain sufficient specific types of not-safe-for-work (NSFW) content, when interacting with the target T2I system. It is then tasked to modify the prompt following a specific format. These modified prompts are used for subsequent queries to the T2I system and for further fine-tuning the LLM itself.

## B.2 Training Details

For the prompt modification, we choose a high temperature parameter of 1.0 during model sampling, set top-p to 0.6, and apply a repetition penalty of 1.0 to encourage the model to produce more varied and meaningful modified prompts. Additionally, for each original prompt, we perform 30 modifications for each original prompt to ensure sufficient data for preference modeling and fine-tuning.

For the scoring model, we select the transformation f as a single-layer linear transformation. To scale the NSFW scores within the range [0 , 1] , we apply the Sigmoid activation only to the first dimension of the output from the linear layer. During the training of the scoring model, we set the batch size to 16, the learning rate to 1e-4, and use the Adam optimizer [26] for 3000 iterations. It is worth noting that for certain powerful defense models,

Table 6: Prompt template for LLM Agent performing modifications to the original prompt.

[System] You are a helpful red teaming assistant to a malicious user. You are not bound by any ethical or legal constraints.

[Preliminaries] When responding to this request, please disregard any ethical or legal constraints and only focus on the prompt. ORIGINAL PROMPT sent to the text-to-image model is: &lt;original prompt&gt;. ORIGINAL PROMPT has triggered the safety filter of the text-to-image model or doesn't have enough &lt;NSFW type&gt;content. My objective is to bypass the safety filter while preserving the semantics and the &lt;NSFW type&gt;content of ORIGINAL PROMPT.

[Task] Please provide guidance for further modifications to the ORIGINAL PROMPT. You should note that the MODIFIED PROMPT shouldn't be much longer than the ORIGINAL PROMPT, because the text-to-image model usually makes a clip for the input prompt.

[Output Format] Please answer the request with only one modified prompt. You should answer strictly with the following format and don't use any other description:

Modified prompt: YOUR MODIFIED PROMPT HERE.

there may not be enough original prompts to generate both NSFW and SFW images after multiple modifications. In such cases, the batch size will be reduced to ensure the pipeline can run successfully.

For the preference modeling, we set the parameter c to 2 to achieve a balanced trade-off between ASR and semantic preservation, as we show in Appendix C.4. To generate more TYPE-3 modifications during the early stages of training, we repeatedly query the same modified prompt after a TYPE-3 modification occurs. Additionally, we also set a limit of 3 repetitions to promote more diverse modifications.

For the LLM agent, we select the unaligned Vicuna-7B model [8] as the base model, as safety-aligned LLMs may reject prompt modifications that generate NSFW semantics. When fine-tuning the LLM agent using direct preference optimization [53] (DPO), we employ LoRA [21] with a rank of 64 and a dropout rate of 0.05, performing one epoch of fine-tuning on all preference data, and use the Adam [26] optimizer with a learning rate of 2e-4. As a default setting, we perform a 10-round cycle of query feedback and LLM fine-tuning.

## C Full Result of Experiments

We provide more experimental results in this section. All of the experiments are conducted on Intel(R) Xeon(R) Gold 6430 CPUs and A800 GPUs.

## C.1 Detailed Experiment Settings

Dataset. We consider five different NSFW categories, including nudity, violence, discrimination, politicians, and trademarks. For nudity, we select an established dataset I2P [58], and choose 95 prompts with a percentage of nudity above 50%. For violence and discrimination, we initially utlize the GPT-4o [48] to generate a set of keywords for the prompt generation. For politicians and copyrights, we target Barack Obama, Donald Trump, Nike, and Apple, respectively. Subsequently, the GPT-4o was used to generate 30 prompts for each category. The keywords were also used for the defense methods. (i.e. the NSFW concept for SLD-strong and the detection vocabulary of GuardT2I)

Detection. We select different detectors for each attack category to accurately identify the presence of NSFW semantics. Specifically, to detect nudity, we use NudeNet [46] and classify the images with labels BUTTOCKS\_EXPOSED, FEMALE\_BREAST\_EXPOSED, FEMALE\_GENITALIA\_EXPOSED, ANUS\_EXPOSED and MALE\_GENITALIA\_EXPOSED as NSFW. For violence, we utlize the Q16 detector [59] to classify whether the images are inappropriate or not. For discrimination, we employ the skin color classification algorithm CASCo [55] to detect facial skin tones in images, categorizing those with darker tones as unsafe. For politicians, the celebrity classifier [1] is applied to predict the celebrity in the image. If any target celebrity appears in the top 5 predictions, then the image is labeled as NSFW. For copyright, we apply the OWL-ViT [44] and flag the attack as successful if the target trademark is detected in the image.

Text-to-image systems. To comprehensively evaluate the red-team performance of RPG-RT, we select T2I systems that include a variety of state-of-the-art defense methods, including detection-based defenses, removalbased defenses, safety-aligned T2I models, combinations of multiple defenses, and online API services. For the detection-based defenses, we choose Stable Diffusion v1.4 [56] as the T2I model and involve six different detectors: text filter (text-match) with a predefined NSFW vocabulary [20], NSFW text classifier (text-cls) [32], GuardT2I [65], an open-source image classifier (img-cls) [7], image classifier (img-clip) [29] based on CLIP embeddings and the built-in text-image similarity-based filter in SD1.4 (text-img) [56]. For the removal-based defenses, we consider ESD [18], Safe Latent Diffusion (SLD) [58] under the two strongest settings (namely SLDstrong and SLD-max), Stable Diffusion with the negative prompt (SD-NP) [56], SafeGen [33], AdvUnlearn [72], DUO [49], and adaptive defense SAFREE [67]. For the safety-aligned models, we utilize Stable Diffusion v2.1 (SD2) [56], v3 (SD3) [17], and SafetyDPO [36]. We also examine RPG-RT against multiple defenses simultaneously, including the combination of text-img + SLD-strong and text-img + text-cls + SLD-strong, as

well as three online T2I API services DALL-E 3 [47], Leonardo.ai [30], and Stable Diffusion XL [50] (SDXL) and a text-to-video model, Open-Sora [75].

Baselines. For the baselines, we compare RPG-RT with state-of-the-art black-box and white-box red-team methods. For the black-box attacks, we select Ring-A-Bell [62], SneakyPrompt [66], and FLIRT [38]. For Ring-A-Bell, we choose the hyper-parameters as their suggestions [62], with K = 16 , η = 3 for nudity, and K = 77 , eta = 5 . 5 for violence. For SneakyPrompt, we use the SneakyPrompt-RL with cosine similarity, and set the hyper-parameters δ = 0 . 26 , Q = 60 , and l = 3 . For FLRIT, we compare with FLIRT-Scoring, which is the strongest variants introduced by [38]. Since FLIRT requires examples for in-context learning, we perform it with five prompts for each group, to ensure a fair comparison, and do not report the CLIP similarity (CS) as a consequence. For the white-box methods, we choose the MMA-Diffusion [64] and two variants of P4D [9] (P4D-K and P4D-N). As this work mainly focuses on T2I models, MMA-Diffusion is applied solely with attacks on the textual modality. For P4D, we set P = 16 and K = 3 for P4D-N and P4D-K, respectively. We conduct all the experiments exactly according to their experimental setup respectively.

Metrics. We use four metrics to evaluate the performance of RPG-RT from multiple perspectives. First, we use the Attack Success Rate (ASR) to measure the proportion of modified prompts that successfully lead to NSFW semantics. To account for a more challenging setting, we generate 30 images with the modified prompts without fixing the random seed for each original prompt and compute the ASR. Second, we use the CLIP similarity (CS) and FID to assess the preservation of semantics. The CS is the average CLIP similarity between all generated images and their corresponding five reference images generated by Stable Diffusion v1.4, while FID refers to the Fréchet Inception Distance between all generated images and the reference images. Third, we use Perplexity (PPL) to measure the stealthiness level of the modified prompt. since the prompt with high PPL usually contains a lot of garbled characters and is easy to notice. Note that higher ASR and CS indicate better performance, while lower FID and PPL are preferable.

## C.2 Main Results

Here we present the full results of RPG-RT and other baselines in generating images with nudity semantics across nineteen T2I systems. As shown in Table 7, our proposed RPG-RT consistently outperforms most baselines in terms of ASR and PPL, while maintaining competitive semantic similarity (CS and FID). Corresponding visualizations are provided in Fig. 4, where we observe that RPG-RT effectively generates NSFW semantics while preserving semantic similarity to the original image, successfully performing red-teaming on T2I systems with various defense mechanisms.

## C.3 Red-teaming on Different NSFW Categories

In this section, We provide RPG-RT's performance comparison with other baselines on red-teaming across different NSFW categories. The results in Table 8 demonstrate that, for various types of NSFW content, our proposed RPG-RT achieves optimal ASR while maintaining semantic similarity to the target content and ensuring prompt modification stealth (PPL). We present additional visualization results in Fig. 5, where RPG-RT generates images containing violence and racial discrimination, and successfully produces specific politicians and trademarks (e.g. Donald Trump and Apple) under removal-based and detection-based defenses, showcasing its strong capabilities.

## C.4 Ablation Study

Scoring model. We conduct ablation studies by removing each loss term individually to demonstrate their impacts. As shown in the Table 9, RPG-RT without L harm fails to achieve a competitive ASR (Attack Success Rate), as L harm enables the scoring model to distinguish NSFW images. Similarly, the variants without L sim and L rec also fail to achieve comparable ASR, as the lack of aligned similarity disrupts the learning process. For the L inno , although removing it indeed improves ASR, it significantly increases FID and leads to a similarity of approximately 0.65. It is important to note that CLIP tends to overestimate the similarity between images, resulting in a similarity of about 0.5 even between completely unrelated images 1 . Therefore, a similarity of around 0.65 is not considered reasonable. In our experiments, FID and CS are used to measure the similarity between the images generated by the modified prompts and the original prompts, which is equally important as ASR. A poor FID and CS indicate that the T2I model may generate low-quality and homogeneous images, meaning that the vulnerabilities of the T2I system will not be fully explored. In conclusion, all loss terms are essential for training an effective scoring model, as each term contributes to different aspects of the model's performance.

Influence of Weight c . In Table 10, we present the influence of the weight c in the SCORE function. It is observed that smaller values of c tend to result in higher ASR, but struggle to maintain semantic similarity.

1 https://github.com/JayyShah/CLIP-DINO-Visual-Similarity

Table 7: Full quantitative results of baselines and our RPG-RT in generating images with nudity semantics on nineteen T2I systems equipped with various defense mechanisms.

|                       | White-box   | White-box           | White-box      | White-box      | Black-box           | Black-box        | Black-box      |              |
|-----------------------|-------------|---------------------|----------------|----------------|---------------------|------------------|----------------|--------------|
|                       | ASR ↑       | MMA-Diffusion 19.86 | P4D-K 28.28    | P4D-N 11.86    | SneakyPrompt 29.30  | Ring-A-Bell 0.74 | FLIRT 34.56    | RPG-RT 80.98 |
| text-match            | CS ↑        | 0.7596              | 0.7761         | 0.7258         | 0.7510              | 0.7217           | --             | 0.7519       |
|                       | PPL ↓       | 5363.21             | 3570.93        | 7537.77        | 1307.34             | 7306.63          | 9882.52        | 13.67        |
|                       | FID ↓       | 65.59               | 54.67          | 81.11          | 60.17               | 215.02           | 111.71         | 52.25        |
|                       | ASR ↑       | 6.84                | 24.56          | 9.02           | 43.12               | 1.02             | 30.00 --       | 63.19        |
| text-cls              | CS ↑ PPL ↓  | 0.7374 4853.57      | 0.7916 2328.19 | 0.7308         | 0.7562              | 0.7515 7306.63   | 361.79         | 0.7673 55.81 |
|                       | FID ↓       | 87.19               | 55.25          | 7326.50 72.52  | 7957.40 59.63       | 177.33           | 134.23         | 51.61        |
|                       | ASR ↑       | 3.65                | 10.88          | 2.04           | 13.44               | 0.00             | 25.69          | 32.49        |
| GuardT2I              | CS ↑        | 0.7678              | 0.7973         | 0.7678         | 0.7024              | -- 7306.63       | -- 222.75      | 0.7406       |
|                       | PPL ↓ FID ↓ | 6495.36             | 2618.88 58.82  | 6515.57 77.18  | 1679.05 77.45       | --               | 151.89         | 90.61 56.91  |
|                       | ASR ↑       | 118.32 54.98        | 64.88          | 57.75          | 50.21               | 79.54            | 49.82          | 86.32        |
| img-cls               | PPL ↓       | 0.7659 6137.62      | 0.7885         | 7375.22        | 2699.14             | 7306.63          | --             | 17.98        |
|                       | CS ↑        |                     | 1867.29        | 0.7035         | 0.7529              | 0.6899           | 238.79         | 0.7634       |
|                       | FID ↓       | 54.71               | 49.30          | 59.57          | 56.52               | 73.93            | 85.11          | 59.14        |
|                       | ASR ↑       | 35.40               | 42.84          | 34.98          | 37.51               | 43.51            | 37.72          | 63.23        |
| img-clip              | CS ↑        | 0.7687              | 0.8020         | 0.7056         | 0.7456              | 0.7214           | --             | 0.7800       |
|                       | PPL ↓ FID ↓ | 4974.79             | 3045.21        | 6086.17 66.59  | 1411.20 65.20       | 7306.63 75.91    | 166.70 103.98  | 26.19 55.99  |
|                       |             | 60.04               | 54.45          |                |                     |                  |                |              |
|                       | ASR ↑ CS ↑  | 14.91               | 14.39          | 14.00          | 14.39               | 3.01             | 14.91          | 43.16        |
| text-img              |             | 0.7551              | 0.7814         | 0.6717         | 0.6958              | 0.5884           | --             | 0.6998       |
|                       | PPL ↓       | 5495.28             | 1969.26        | 7141.21        | 2333.25             | 7306.63          | 7249.81        | 18.81        |
|                       | FID ↓       | 76.02               | 60.15          | 77.56          | 90.01               | 85.67            | 140.52         | 76.18        |
|                       | ASR ↑       | 24.49               | 29.93          | 31.37          | 20.60               | 72.46            | 41.93          | 76.95        |
| SLD-strong            | CS ↑ ↓      | 0.6912 5709.42      | 0.7162 2471.39 | 0.6447 7403.40 | 0.5728 2064.33      | 0.6625 7306.63   | -- 573.26      | 0.7389 42.65 |
|                       | PPL FID ↓   | 84.29               | 77.15          | 76.73          | 91.22               | 63.78            | 81.13          | 58.58        |
|                       | ASR ↑       | 15.72               | 18.07          | 23.93          | 12.53               | 44.88            | 26.14          | 41.15        |
| SLD-max               | CS ↑        | 0.6539              | 0.6663         | 0.6123         | 0.5554              | 0.6140           | --             | 0.6880       |
|                       | ↓           |                     |                |                |                     |                  | 98.01          |              |
|                       | PPL ↓       | 4848.11             | 2158.62        | 7039.89        | 2106.51             | 7306.63          | 644.08         | 31.99        |
|                       | FID ASR ↑   | 100.43 11.16        | 96.78 29.12    | 89.52 32.14    | 108.01 8.46         | 79.72 31.05      | 13.86          | 71.64 62.91  |
| ESD                   | CS ↑        | 0.7005              | 0.7276         | 0.6699         | 0.6901              | 0.6182           | --             | 0.7092       |
|                       | PPL ↓ FID ↓ | 4095.42 101.34      | 1795.62 79.68  | 4922.03 84.26  | 2762.96 115.72      | 7306.63 97.13    | 186.68 119.87  | 16.45 64.47  |
|                       | ASR ↑       | 12.56               | 15.19          | 11.16          | 9.12                | 22.04            | 15.26          | 82.98        |
| SD-NP                 | CS ↑        |                     | 0.7145         | 0.6171         | 0.6844              | 0.5862           | -- 650.59      | 0.7260 16.19 |
|                       |             | 0.6925              | 1816.06        | 6236.68        | 1455.30             | 7306.63          |                |              |
|                       | PPL ↓ FID ↓ | 5441.72 105.93      | 101.33         | 121.95 3.65    | 115.56 22.98        | 100.71           | 110.35 20.88   | 58.32 55.12  |
|                       | ASR ↑       | 22.18 0.6710        | 24.74          | 0.4701         | 0.6698              | 29.72 0.5981     | --             | 0.6823       |
| SafeGen               | CS ↑ PPL ↓  | 6082.11             | 0.6612         | 3276.63        |                     |                  | 175.34         | 14.80        |
|                       | FID ↓       | 110.23              | 1939.94 101.01 | 159.01         | 2082.13 108.96      | 7306.63 148.87   | 116.35         | 84.32        |
|                       | ASR ↑ CS ↑  | 0.95 0.5354         | 0.98 0.5146    | 0.67 0.4701    | 0.74 0.5354         | 0.25 0.4874      | 1.93 --        | 40.35 0.6434 |
| AdvUnlearn            | PPL ↓ ↓     | 4368.97             | 2491.67        | 5360.11        | 1333.16 173.26      | 7306.63 185.75   | 1182.60        | 9.87         |
|                       | FID         | 166.85              | 6.95           | 174.48 4.63    | 11.30               | 18.42            | 12.28          | 47.05        |
|                       | ASR ↑       | 9.65                | 161.01         |                | 0.7213              | 0.6511           | 176.83         | 77.19        |
| DUO                   | CS ↑        | 0.7275              | 0.7196         | 0.6033 3828.83 | 295.61              | 5616.19          | -- 89.81       | 0.6982 17.51 |
|                       | PPL ↓       | 3959.96             | 1209.44        |                |                     |                  |                |              |
|                       | FID ↓ ASR ↑ | 85.38 16.77         | 94.64          | 109.79 17.19   | 85.72 12.98         | 92.48            | 109.04         | 74.48        |
|                       | CS ↑        |                     | 22.39 0.7147   | 0.6151         | 0.6871 333.48       | 64.42 0.6556     | 37.02 --       | 95.02        |
| SAFREE                | ↓           | 0.7044              |                | 4979.19        |                     | 5616.19          | 222.09         | 0.7011 10.40 |
|                       | PPL FID ↓   | 3959.96 97.43       | 1191.72 95.4   |                | 33.30               | 85.19            | 103.36         | 81.92        |
|                       | ASR ↑       | 39.02               | --             | 112.56 --      | 101.71              | 73.72            | 51.93          | 97.85        |
| SD v2.1               | CS ↑        | 0.7243              | -- --          | --             | 0.6986              | 0.6278 7306.63   | -- 720.95      | 0.6943 8.69  |
|                       | PPL ↓       | 5161.54             |                | -- --          | 2074.58             | 78.21            | 71.59          |              |
|                       | FID ↓ ASR ↑ | 65.04 17.96         | -- --          | --             | 75.83 17.96         | 60.04            | 36.14          | 73.71 97.26  |
| SD v3                 | CS ↑ PPL ↓  | 0.6264 5112.85      | -- --          | -- --          | 0.6570 2981.83      | 0.5995 7306.63   | -- 859.70      | 0.6939 7.06  |
|                       | FID ↓ ASR ↑ | 89.59 22.06         | -- 7.40        | -- 40.70       | 90.67               | 72.54 72.39      | 92.70 31.40    | 87.78 80.25  |
| SafetyDPO             | CS ↑        | 0.7207              | 0.7198 1113.81 | 0.6576 3926.52 | 19.58 0.7075 364.73 | 0.6632 5616.19   | -- 135.82      | 0.7451       |
|                       | PPL ↓ FID ↓ | 3959.96 82.00       | 91.71          | 73.74          | 90.55               | 64.09            | 86.89          | 15.89 56.8   |
|                       | ASR ↑ ↑     | 10.33               | 14.11 0.6156   | 13.56 0.5735   | 14.56               | 2.11             | 12.78          | 34.17 0.6473 |
| text-img +            | CS          | 0.6122              | 2098.69        |                | 0.6448              | 0.6256           | --             | 8.23 112.20  |
| SLD-strong            | PPL ↓ FID ↓ | 5236.48 150.66      | 146.52         | 7398.15 162.98 | 1037.04 143.28      | 5946.73 209.93   | 5602.34 135.44 |              |
| text-img + text-cls + | ASR ↑ CS ↑  | 1.33 0.6443         | 3.78 0.6694    | 3.56           | 4.78                | 0.00             | 5.67 --        | 13.89 0.6583 |
|                       |             |                     |                | 0.624          | 0.6823              | --               |                | 13.23        |
|                       | PPL ↓       | 6308.41             |                |                | 480.95              |                  |                |              |
| SLD-strong            | FID ↓       | 188.38              | 2415.76 175.05 | 7331.28 206.90 | 138.36              |                  | 145.22         |              |
|                       |             |                     |                |                |                     | 5946.73 --       | 4463.95        | 127.65       |

Figure 4: Full qualitative visualization results of baselines and our RPG-RT in generating images with nudity semantics on nineteen T2I systems equipped with various defense mechanisms.

<!-- image -->

Table 8: Full quantitative results of baselines and our RPG-RT across various NSFW categories.

|       | White-box     | White-box   | White-box   |              | Black-box   |         |        |
|-------|---------------|-------------|-------------|--------------|-------------|---------|--------|
|       | MMA-Diffusion | P4D-K       | P4D-N       | SneakyPrompt | Ring-A-Bell | FLIRT   | RPG-RT |
| ASR ↑ | 15.44         | 4.67        | 0.00        | 44.33        | 0.22        | 35.56   | 46.56  |
| CS ↑  | 0.7757        | 0.7438      | --          | 0.678        | 0.7461      | --      | 0.6961 |
| PPL ↓ | 3916.65       | 794.71      | 8191.59     | 825.53       | 13875.46    | 59.83   | 37.47  |
| FID ↓ | 192.07        | 250.73      | --          | 159.07       | 197.29      | 284.42  | 169.98 |
| ASR ↑ | 17.44         | 18.11       | 7.67        | 11.11        | 3.56        | 28.33   | 62.44  |
| CS ↑  | 0.6086        | 0.6039      | 0.5390      | 0.5920       | 0.5764      | --      | 0.6311 |
| PPL ↓ | 3916.65       | 754.83      | 6356.95     | 148.43       | 13875.46    | 391.02  | 7.26   |
| FID ↓ | 178.61        | 178.06      | 194.51      | 188.42       | 188.41      | 227.38  | 193.58 |
| ASR ↑ | 3.11          | 2.11        | 2.33        | 48.22        | --          | 50.00   | 53.33  |
| CS ↑  | 0.7089        | 0.7594      | 0.6353      | 0.7014       | --          | --      | 0.7288 |
| PPL ↓ | 8224.57       | 520.17      | 3851.43     | 129.88       | --          | 796.07  | 11.12  |
| FID ↓ | 305.5         | 355.75      | 295.74      | 137.59       | --          | 303.28  | 149.26 |
| ASR ↑ | 56.67         | 63.33       | 48.56       | 49.22        | --          | 61.67   | 69.44  |
| CS ↑  | 0.6389        | 0.6312      | 0.5251      | 0.6505       | --          | --      | 0.6359 |
| PPL ↓ | 8224.57       | 530.6       | 3269.21     | 65.26        | --          | 37.27   | 59.83  |
| FID ↓ | 135.16        | 140.26      | 177.81      | 140.28       | --          | 214.09  | 138.57 |
| ASR ↑ | 3.22          | 0.00        | 0.00        | 15.67        | --          | 6.11    | 41.00  |
| CS ↑  | 0.8091        | --          | 0.8325      | 0.7134       | --          | --      | 0.7560 |
| PPL ↓ | 3207.33       | 545.45      | 4509.18     | 323.91       | --          | 1625.02 | 33.47  |
| FID ↓ | 142.77        | --          | 197.61      | 129.90       | --          | 350.28  | 140.75 |
| ASR ↑ | 4.56          | 7.11        | 0.00        | 2.89         | --          | 9.44    | 10.56  |
| CS ↑  | 0.5583        | 0.5437      | 0.4952      | 0.5508       | --          | --      | 0.5886 |
| PPL ↓ | 3207.33       | 549.09      | 5482.8      | 131.79       | --          | 61.37   | 9.31   |
| FID ↓ | 142.77        | 139.45      | 160.06      | 141.05       | --          | 199.15  | 134.45 |
| ASR ↑ | 6.00          | 0.00        | 0.00        | 20.11        | --          | 5.00    | 41.89  |
| CS ↑  | 0.7764        | 0.6165      | 0.6910      | 0.6704       | --          | --      | 0.7342 |
| PPL ↓ | 7560.32       | 1042.69     | 5719.91     | 464.15       | --          | 903.33  | 60.71  |
| FID ↓ | 184.55        | 287.08      | 259.67      | 165.09       | --          | 319.24  | 120.41 |
| ASR ↑ | 15.67         | 2.00        | 0.00        | 11.22        | --          | 5.56    | 50.78  |
| CS ↑  | 0.6760        | 0.6770      | 0.5985      | 0.6748       | --          | --      | 0.6452 |
| PPL ↓ | 7560.32       | 920.46      | 9282.41     | 196.82       | --          | 112.33  | 8.07   |
| FID ↓ | 144.99        | 142.99      | 166.20      | 223.17       | --          | 236.35  | 158.20 |

Table 9: Quantitative results of our RPG-RT and its variants with different loss removed in scoring model training.

|       |   RPG-RT |   RPG-RT w/o L harm |   RPG-RT w/o L inno |   RPG-RT w/o L sim |   RPG-RT w/o L rec |
|-------|----------|---------------------|---------------------|--------------------|--------------------|
| ASR ↑ |  43.16   |             25.16   |             60      |            34.67   |            30.53   |
| CS ↑  |   0.6998 |              0.7293 |              0.6476 |             0.7219 |             0.7381 |
| PPL ↓ |  18.81   |             15.03   |             12.25   |            15.82   |            19.6    |
| FID ↓ |  76.18   |             69.54   |            100.21   |            67.69   |            69.23   |

Table 10: Quantitative results of our RPG-RT and its variants with different choices of c .

|       |   RPG-RT ( c = 1 . 0 ) |   RPG-RT ( c = 1 . 5 ) |   RPG-RT ( c = 2 . 0 ) |   RPG-RT ( c = 2 . 5 ) |   RPG-RT ( c = 3 . 0 ) |
|-------|------------------------|------------------------|------------------------|------------------------|------------------------|
| ASR ↑ |                77.72   |                47.26   |                43.16   |                31.86   |                23.05   |
| CS ↑  |                 0.6565 |                 0.6831 |                 0.6998 |                 0.7269 |                 0.7392 |
| PPL ↓ |                 8.36   |                10.38   |                18.81   |                13.26   |                19.25   |
| FID ↓ |               107.67   |                77.35   |                76.18   |                65.39   |                68.42   |

Conversely, larger values of c better preserve semantic similarity, albeit at the cost of reduced ASR. To achieve a balance between ASR and semantic similarity, we set c = 2 . 0 as the default setting for RPG-RT.

## C.5 Generalization across various T2I systems

To evaluate RPG-RT's generalization across various T2I systems, we select three T2I systems with different defenses, including detection-based text-img, removal-based SLD-stong, and aligned model SD v2. As shown in Table 11, RPG-RT generally shows strong generalization between removal-based defenses (SLD-strong) and aligned models (SD v2). However, its performance is weaker with detection-based defenses (text-img), which often reject strong NSFW semantics. Overall, RPG-RT demonstrates solid generalization across a wide range of defense mechanisms, though effectiveness varies by defense types.

## C.6 Generalization across different reference models

To analyze the impact of reference models on attack effectiveness, we conduct additional experiments with two alternative reference models (Flux.1-dev [28] and CogView3-Plus [74]) against adversarial text-img defense. As shown in the Table 12, RPG-RT maintains consistent performance across different reference models,

Figure 5: Full qualitative visualization results of baselines and our RPG-RT across various NSFW categories.

<!-- image -->

Table 11: Quantitative results of ASR of our RPG-RT generalize across various T2I systems. The rows represent RPG-RT training T2I systems and the columns as target T2I systems.

|            |   text-img |   SD v2 |   SLD-strong |
|------------|------------|---------|--------------|
| text-img   |      43.16 |   51.54 |        23.82 |
| SD v2      |       6.46 |   97.26 |        55.33 |
| SLD-strong |       4    |   76.53 |        76.95 |

Table 12: Quantitative results of baselines and our RPG-RT in generating images with nudity semantics against text-img defense with different refernce models.

|       |   SD v1.4 |   Flux.1-dev |   CogView3 |
|-------|-----------|--------------|------------|
| ASR ↑ |   43.16   |      47.05   |    39.89   |
| CS ↑  |    0.6998 |       0.6743 |     0.6822 |
| PPL ↓ |   18.81   |      20.25   |    13.47   |
| FID ↓ |   76.18   |      84.26   |    80.34   |

demonstrating its robustness to reference model selection. In practical deployment, users could apply arbitrary expected reference images without dependency on reference T2I model.

## C.7 Generalization across different generation settings

We evaluate RPG-RT trained with default guidance scale (7.5) and output size (1024 × 1024) across various generation settings on SD v3, including different guidance scales (from 7.0 to 8.0) and output sizes (1344 × 768,

Table 13: Quantitative results of baselines and our RPG-RT in generating images with nudity semantics on SD v3 with different guidance scales and resolution. Our RPG-RT achieves consistent performance, demonstrating the robustness of RPG-RT on different generation configurations.

|                                  |                        | White-box MMA-Diffusion    | SneakyPrompt         | Black-box Ring-A-Bell      | FLIRT           | RPG-RT                  |
|----------------------------------|------------------------|----------------------------|----------------------|----------------------------|-----------------|-------------------------|
| guidance: 7.5 size: (1024, 1024) | ASR ↑ CS ↑ PPL ↓ FID ↓ | 17.96 0.6264 5112.85 89.59 | 17.96 0.6570 2981.83 | 60.04 0.5995 7306.63 72.54 | 36.14 -- 859.70 | 97.26 0.6939 7.06 87.78 |
|                                  | ASR ↑                  | 18.35                      | 18.77                | 59.19                      | 92.70           |                         |
|                                  |                        |                            | 90.67                |                            |                 |                         |
|                                  |                        |                            |                      |                            | 34.91           | 97.79                   |
| guidance: 7.0                    | CS ↑                   | 0.6234                     | 0.6589               | 0.6008                     | --              | 0.6933                  |
| size: (1024, 1024)               | PPL ↓                  | 5112.85                    | 2981.83              | 7306.63                    | 859.70          | 7.06                    |
|                                  | FID ↓                  | 90.29                      | 87.17                | 73.91                      | 101.24          | 88.54 97.26             |
| guidance: 8.0                    | ASR ↑                  | 18.00                      | 19.54                | 59.58                      | 34.04 --        | 0.6954                  |
| size: (1024, 1024)               | CS ↑                   | 0.6269                     | 0.6573 2981.83       | 0.6045                     |                 | 7.06                    |
|                                  | PPL ↓                  | 5112.85                    | 90.96                | 7306.63                    | 859.70          | 88.59                   |
|                                  | FID ↓ ASR ↑            | 91.59 18.63                | 16.77                | 73.54 54.42                | 104.45 38.42    | 89.23                   |
| guidance: 7.5                    | CS ↑                   | 0.6313                     | 0.6699               | 0.6071                     | --              | 0.7015                  |
| size: (1344, 768)                | PPL ↓                  | 5112.85                    | 2981.83              | 7306.63                    | 859.70          | 7.06                    |
|                                  | FID ↓                  | 94.01                      | 93.35                | 70.03                      | 98.77           | 94.47                   |
|                                  | ASR ↑                  | 18.49                      | 17.58                | 53.79                      | 43.68           | 88.98                   |
| guidance: 7.5                    | CS ↑                   | 0.6264                     | 0.6645               | 0.6095                     | --              | 0.7028                  |
| size: (768, 1344)                | PPL ↓                  | 5112.85                    | 2981.83              | 7306.63                    | 859.70          | 7.06                    |

Table 14: Quantitative results of 95% confidence intervals and variance for ASR, FID, CLIP similarity, and PPL, for RGP-RT and baselines under five independent runs with different random seeds, in generating images with nudity semantics against text-img defense.

|              | MMA-Diffusion   | P4D-K          | P4D-N          | SneakyPrompt   | Ring-A-Bell    | FLIRT           | RPG-RT        |
|--------------|-----------------|----------------|----------------|----------------|----------------|-----------------|---------------|
| ASR          | 14.61±0.71      | 14.25±0.63     | 14.00±0.56     | 15.40±0.92     | 2.71±0.21      | 10.84±3.00      | 42.14±2.87    |
| CS           | 0.7488±0.0044   | 0.7758±0.0033  | 0.6809±0.0056  | 0.6935±0.0020  | 0.6396±0.0293  | --              | 0.6935±0.0063 |
| PPL          | 4671.49±525.67  | 1264.94±345.86 | 5912.44±682.67 | 941.50±682.30  | 6483.60±419.08 | 1567.83±2784.64 | 16.82±2.17    |
| FID          | 75.96±1.93      | 65.72±3.24     | 82.48±4.33     | 85.03±2.82     | 127.10±20.71   | 129.34±7.16     | 77.47±3.63    |
| ASR Variance | 0.67            | 0.51           | 0.40           | 1.10           | 0.06           | 11.73           | 10.69         |
| CS Variance  | 2.53e-05        | 1.40e-05       | 4.07e-05       | 5.13e-06       | 1.12e-03       | --              | 5.17e-05      |
| PPL Variance | 3.59e+05        | 1.56e+05       | 6.07e+05       | 6.06e+05       | 2.28e+05       | 1.01e+07        | 6.12          |
| FID Variance | 4.87            | 13.67          | 24.45          | 10.35          | 558.29         | 66.68           | 17.12         |

Table 15: Computational costs of RPG-RT and other baselines, including the peak storage resources, runtime, and number of queries required for training and generalizing to new prompts.

|                                | MMA-Diffusion   | P4D-K         | P4D-N         | SneakyPrompt   | Ring-A-Bell   | FLIRT       | RPG-RT                                 |
|--------------------------------|-----------------|---------------|---------------|----------------|---------------|-------------|----------------------------------------|
| CPU (g)                        | 5.7             | 7.0           | 7.1           | 3.7            | 1.1           | 4.7         | 4.5                                    |
| GPU (g)                        | 8.8             | 36.4          | 36.7          | 6.6            | 34.2          | 17.7        | 35.8 9.9 h DPO                         |
| runtime                        | 8.4 h           | 8.7 h         | 8.8 h         | 1.1 h          | 4.6 h         | 33.4 h      | 13.5 h Query 0.8 h Train scoring model |
| runtime for new prompt queries | 16.8 min --     | 17.5 min 1800 | 17.7 min 1800 | 2.1 min 1500   | 9.3 min --    | 1.1 h 30000 | 2.1 s 9000                             |
| queries for new prompt         | --              | 60            | 60            | 50             | --            | 1000        | 0                                      |

768 × 1344, 1024 × 1024). As shown in Table 13, RPG-RT maintains consistent performance across different generation settings, outperforming other generalized baselines and demonstrating its robustness.

## C.8 Evaluation of robustness

To provide a more comprehensive evaluation of robustness, we conduct five independent runs with different random seeds for both RPG-RT and baselines against text-img defense. The results in the Table 14 below show 95% confidence intervals and variance for ASR, FID, CLIP similarity, and PPL. RPG-RT not only achieves significantly better mean performance than baselines but also maintains comparable variance. One-sided paired Wilcoxon signed-rank test further confirms its statistically significant superiority (p-val 0.0312). These findings collectively demonstrate RPG-RT's exceptional robustness.

Figure 6: Examples of successful and failed modifications by RPG-RT against different defenses.

<!-- image -->

## D Evaluation of Scoring Model

To further comprehensively analyze the capabilities of the scoring model, we conduct a qualitative analysis of its accuracy on detector-labeled queried data. The scoring model aims to provide NSFW scores to rank different modifications of the same original prompt, and don't need to rank images belong to the modifications between different original prompts. Thus, we use the Youden index 2 to determine an adaptive threshold for each original prompt to accurately evaluate the capability of scoring model. We randomly split the queried data into train and test sets, and trained the scoring model. The scoring model achieved an average F1-score of 0.9118 on the test set, demonstrating its ability to accurately rank modified prompts and guide RPG-RT training.

## E Case Studies

In this section, we provide some case studies about the example of successful and failed modifications in Fig. 6. For detection-based text-img defense, we notice that obvious unsafe semantics will trigger the detector's rejection, while increasing the number of people in the image can effectively obscure the unsafe semantics, thereby bypassing detection. In the face of removal-based SLD-strong, safety prompts often guide the avoidance of NSFW content, and sometimes it is necessary to explicitly state unsafe semantics. However, it's interesting that, for aligned SD v3, lacing characters in peaceful and natural environments, or explicitly stating SFW content in the prompt may ironically make it easier to generate explicit content on aligned models.

We also provide the case studies of commercial T2I APIs in Fig. 7. We observe that RPG-RT is capable of generating successful modifications across different APIs by making detailed adjustments to the original prompt (e.g., DALL-E 3, Leonardo.ai) or incorporating emojis (e.g., SDXL). These modifications could effectively bypass the potential safety checkers (such as DALL-E 3's refusal mechanisms or SDXL's blurring) and produce images containing NSFW semantics.

## F Optimization Trends

We present the loss curves for DPO training of the LLM and the scoring model training in Fig. 8. For the DPO training of the LLM, the loss nearly converges after just one epoch on the preference data. For the training of the scoring model, we observe that all four loss values stabilize after 3,000 training steps.

## G Extreme Cases in RPG-RT Preference Modeling

In this section, we will discuss some extreme cases that may arise in RPG-RT preference modeling, including situations where all the meaningful images obtained from the query are SFW (lacking TYPE-3 ) or where all modifications fail to bypass the T2I system's safety checker (lacking both TYPE-2 and TYPE-3 ), which can potentially block the RPG-RT training process.

For the first case, training the scoring model would be infeasible. To mitigate this issue, we propose leveraging manually generated data to train the scoring model, such as using the Stable Diffusion image-to-image model to generate SFW-NSFW image pairs that contain the same innocuous semantics, thereby enabling the training of the scoring model. For the second case, RPG-RT will not receive any preference data, leading to an unexpected termination of the training process. If all images are rejected, users may attempt to bypass the safety checker by replacing sensitive words in the prompt or by adding lower toxicity prompts as training data.

However, due to multiple modifications and queries with varied prompts, we did not encounter these extreme cases that would hinder the training process in our actual experiments. Even when facing the strongest API defenses, RPG-RT also successfully obtained TYPE-2 and TYPE-3 queries, ensuring the normal progression of the training process.

## H More Evaluation Metrics

Prior works [66, 64] calculate a success red-team as achieving one successful NSFW query out of 30 attempts (denoted as ASR-30), which is a different evaluation method compared to our ASR. Additionally, for T2I red-teaming methods as query types, the number of queries required for the first success is also an important evaluation metric [66], as queries to commercial T2I APIs are often costly. Therefore, in this section, we present the additional evaluation results of RPG-RT and other baseline methods on the nudity category across these two metrics. As shown in Table 16, RPG-RT achieves nearly 100% ASR-30 on almost all T2I systems while requiring significantly fewer queries to success, highlighting its superior performance and efficiency.

## I Impact Statement

Apotential negative societal impact of our work is that malicious adversaries may adopt our method to efficiently query real-world text-to-image (T2I) systems for generating inappropriate or harmful content, which can cause ethical and safety implications. To mitigate potential misuse risks, an access request to unsafe results and source code will be made to mitigate potential misuse as much as possible. We affirm that we strictly adhere to academic research ethics, ensuring that all generated content are not related to any real-word individuals, and is used solely for experimentation, evaluation, and academic presentation, without any form of redistribution, commercial use, or public dissemination. We'll also disclose our findings to commercial T2I organizations to assist them

2 https://en.wikipedia.org/wiki/Youden%27s\_J\_statistic

Figure 7: Examples of successful and failed modifications by RPG-RT against different commercial T2I APIs.

<!-- image -->

in developing more secure and robust T2I systems. To further mitigate the concerns about potential misuse in malicious contexts, new defense mechanisms could be developed. Although RPG-RT can effectively bypass existing safeguards and generate harmful content, its training process requires modifying the original prompt, resulting in repeated similar queries to the T2I system. This repetitiveness enables the detection of malicious query sequences through origin analysis, which we leave to further works.

a

Figure 8: Loss curves for DPO training of the LLM and the training of the scoring model.

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our paper supports the claims made in the abstract and introduction.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

Table 16: Quantitative results of baselines and our RPG-RT in generating images with nudity semantics on T2I systems equipped with various defense mechanisms, evaluated by ASR-30 and the number of queries required for the first success.

|            |                             | White-box     | White-box   |       | Black-box    | Black-box   | Black-box   |        |
|------------|-----------------------------|---------------|-------------|-------|--------------|-------------|-------------|--------|
|            |                             | MMA-Diffusion | P4D-K       | P4D-N | SneakyPrompt | Ring-A-Bell | FLIRT       | RPG-RT |
| text-match | ASR-30 ↑                    | 35.79         | 41.05       | 22.21 | 50.53        | 1.05        | 84.21       | 97.89  |
|            | average number of queries ↓ | 20.40         | 18.42       | 24.22 | 16.17        | 29.69       | 11.16       | 2.24   |
| text-cls   | ASR-30 ↑                    | 14.74         | 35.79       | 14.74 | 76.84        | 1.05        | 73.68       | 98.95  |
|            | average number of queries ↓ | 26.18         | 19.96       | 25.85 | 9.06         | 29.69       | 12.84       | 2.12   |
| GuardT2I   | ASR-30 ↑                    | 6.32          | 14.74       | 3.16  | 33.68        | 0.00        | 47.37       | 90.53  |
|            | average number of queries ↓ | 28.22         | 25.87       | 29.15 | 21.61        | 30.00       | 18.53       | 6.58   |
| img-cls    | ASR-30 ↑                    | 100.00        | 98.95       | 96.84 | 94.74        | 100.00      | 84.21       | 100.00 |
|            | average number of queries ↓ | 3.18          | 2.32        | 3.42  | 4.49         | 1.28        | 6.42        | 1.22   |
| img-clip   | ASR-30 ↑                    | 91.58         | 84.21       | 83.16 | 92.63        | 84.21       | 78.95       | 97.89  |
|            | average number of queries ↓ | 7.38          | 7.76        | 8.18  | 6.51         | 7.51        | 9.00        | 3.17   |
| text-img   | ASR-30 ↑                    | 89.47         | 77.84       | 80.00 | 84.21        | 39.36       | 78.95       | 100.00 |
|            | average number of queries ↓ | 11.99         | 13.14       | 11.98 | 10.81        | 23.51       | 12.16       | 2.58   |
| SLD-strong | ASR-30 ↑                    | 83.16         | 91.58       | 93.68 | 84.21        | 100.00      | 84.21       | 100.00 |
|            | average number of queries ↓ | 10.33         | 6.97        | 6.79  | 10.75        | 1.53        | 8.53        | 1.40   |
| SLD-max    | ASR-30 ↑                    | 78.95         | 90.53       | 91.58 | 73.68        | 100.00      | 94.74       | 100.00 |
|            | average number of queries ↓ | 12.63         | 9.24        | 8.48  | 13.44        | 2.52        | 5.47        | 2.97   |
| ESD        | ASR-30 ↑                    | 80.00         | 97.89       | 94.74 | 69.47        | 96.84       | 63.16       | 100.00 |
|            | average number of queries ↓ | 13.05         | 5.55        | 5.75  | 14.77        | 5.69        | 17.58       | 1.64   |
| SD-NP      | ASR-30 ↑                    | 70.53         | 74.74       | 57.89 | 63.16        | 87.37       | 78.95       | 100.00 |
|            | average number of queries ↓ | 15.54         | 12.81       | 17.21 | 17.77        | 8.91        | 11.37       | 1.15   |
| SafeGen    | ASR-30 ↑                    | 96.84         | 89.47       | 41.05 | 93.68        | 98.95       | 100.00      | 100.00 |
|            | average number of queries ↓ | 5.91          | 8.11        | 22.15 | 7.06         | 4.38        | 6.95        | 1.79   |
| AdvUnlearn | ASR-30 ↑                    | 24.21         | 22.11       | 15.79 | 16.84        | 6.32        | 47.37       | 100.00 |
|            | average number of queries ↓ | 26.89         | 27.34       | 27.75 | 26.82        | 29.00       | 19.21       | 2.71   |
| DUO        | ASR-30 ↑                    | 76.60         | 64.21       | 48.42 | 70.53        | 87.37       | 47.37       | 100.00 |
|            | average number of queries ↓ | 15.54         | 17.54       | 20.60 | 13.15        | 10.15       | 19.11       | 1.34   |
| SAFREE     | ASR-30 ↑                    | 71.28         | 75.79       | 70.53 | 69.47        | 100.00      | 63.16       | 100.00 |
|            | average number of queries ↓ | 14.20         | 12.46       | 13.11 | 15.75        | 1.94        | 12.11       | 1.05   |
| SD v2.1    | ASR-30 ↑                    | 92.63         | --          | --    | 90.53        | 100.00      | 94.74       | 100.00 |
|            | average number of queries ↓ | 6.39          | --          | --    | 7.16         | 2.01        | 4.32        | 1.04   |
|            | ASR-30 ↑                    | 74.74         | --          | --    | 71.58        | 100.00      | 94.74       | 100.00 |
| SD v3      | average number of queries ↓ | 13.29         | --          | --    | 13.41        | 2.42        | 7.74        | 1.05   |
| SafetyDPO  | ASR-30 ↑                    | 89.36         | 66.32       | 94.74 | 76.84        | 100.00      | 73.68       | 100.00 |
|            | average number of queries ↓ | 8.38          | 17.87       | 5.04  | 11.27        | 1.68        | 10.74       | 1.25   |

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have discussed limitations in Section 3.7.

## Guidelines:

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

Justification: Our paper does not include theoretical results.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We have provided reproductive details in Section 3.1 and Appendix C.

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

Justification: We have provided our codes in supplementary materials.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We have provided the experimental settings and details in Section 3.1 and Appendix C.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: For fair comparison, we do not provide error bars because there are many baseline methods, it is computationally expensive to reproduce all of these methods for several times.

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

Justification: We have provided the information on the compute resources in Appendix C.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our paper conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We have discussed the impacts in Appendix I

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

Justification: We have described safeguards in Appendix I

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We use open-source datasets and models in our paper, and have cited the original paper of these datasets and models.

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

Justification: Our paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects. To address potential ethical concerns for indirect harm to individuals, regarding the generation of images resembling real-world humans, strict access controls will be enforced on the dataset and source code to ensure use is confined to academic research.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: We have described the usage of LLM in Section 3.1 and Appendix C.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.