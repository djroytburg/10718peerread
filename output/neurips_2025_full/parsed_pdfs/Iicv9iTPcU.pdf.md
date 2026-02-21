## RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis

1 , 2 ∗ 3 ∗ 2 , 3 1 , 2

Haolin Li Tianjie Dai Zhe Chen Siyuan Du Jiangchao Yao 2 , 3 † Ya Zhang 2 , 4 , 5 Yanfeng Wang 2 , 4 †

1 College of Computer Science and Artificial Intelligence, Fudan University

2 Shanghai AI Laboratory 3 CMIC, Shanghai Jiao Tong University 4

School of Artificial Intelligence, Shanghai Jiao Tong University

5 Institute of Artificial Intelligence for Medicine, Shanghai Jiao Tong University School of Medicine

## Abstract

Clinical diagnosis is a highly specialized discipline requiring both domain expertise and strict adherence to rigorous guidelines. While current AI-driven medical research predominantly focuses on knowledge graphs or natural text pretraining paradigms to incorporate medical knowledge, these approaches primarily rely on implicitly encoded knowledge within model parameters, neglecting task-specific knowledge required by diverse downstream tasks. To address this limitation, we propose R etrievalA ugmented D iagnosis (RAD), a novel framework that explicitly injects external knowledge into multimodal models directly on downstream tasks. Specifically, RAD operates through three key mechanisms: retrieval and refinement of disease-centered knowledge from multiple medical sources, a guidelineenhanced contrastive loss that constrains the latent distance between multi-modal features and guideline knowledge, and the dual transformer decoder that employs guidelines as queries to steer cross-modal fusion, aligning the models with clinical diagnostic workflows from guideline acquisition to feature extraction and decision-making. Moreover, recognizing the lack of quantitative evaluation of interpretability for multimodal diagnostic models, we introduce a set of criteria to assess the interpretability from both image and text perspectives. Extensive evaluations across four datasets with different anatomies demonstrate RAD's generalizability, achieving state-of-the-art performance. Furthermore, RAD enables the model to concentrate more precisely on abnormal regions and critical indicators, ensuring evidence-based, trustworthy diagnosis. Our code is available at this repository.

## 1 Introduction

The rapid development of multimodal learning [37, 46] has revolutionized numerous fields by enabling models to process and integrate diverse data types, including images, texts, audio, and structured records [4, 11, 77]. Biomedical applications particularly benefit from these advancements, given that diagnostic workflows inherently depend on multimodal evidence, ranging from radiographic imaging and reports to electronic health records (EHR) [16, 52, 75]. For instance, radiologists integrate X-ray or MRI scans with textual pathology reports, while clinicians combine electronic health records, vital signs, and even genomic data to form comprehensive patient profiles. Accordingly, recent research efforts have increasingly focused on developing multimodal architectures tailored to healthcare challenges, seeking to enhance diagnostic precision through cross-modal synergy [6, 32, 61, 68]. While these approaches demonstrate significant progress in integrating data from different modalities, they often overlook the foundational principles governing clinical decision-making.

∗ Equal Contribution. † Correspondence to Jiangchao Yao (Sunarker@sjtu.edu.cn) and Yanfeng Wang (wangyanfeng622@sjtu.edu.cn).

Figure 1: The design motivation of RAD. Left : Previous methods mostly focus on enhancing a single aspect of the diagnostic process, whereas our approach is holistic. Right : Visualization of model attention to textual content. Color intensity reflects attention magnitude, with red highlighting disease-critical indicators mentioned in the guideline. Models without explicit knowledge guidance exhibit limited focus on key indicators, whereas our model can make evidence-based diagnoses.

<!-- image -->

Medical analysis fundamentally differs from natural scene understanding in its strict adherence to evidence-based principles, relying heavily on structured protocols [31, 50]. Clinical decisions must be grounded in standardized diagnostic criteria derived from patient-specific symptoms, imaging findings, and laboratory results. This inherent rigor poses a critical challenge for black-box neural networks, whose vague decision-making mechanisms hinder trustworthy and practical deployment in clinical settings [19, 49, 53]. Consequently, there has been growing interest in integrating medical knowledge into AI models to simultaneously improve model performance and interpretability [8, 9, 73].

Existing approaches primarily focus on knowledge injection during pretraining phases. Researchers enhance the text encoders by pretraining them on large-scale medical corpora [30, 47] or leveraging structured knowledge graphs to imbue models with semantic relationships between biomedical entities [36, 48]. While effective in expanding the semantic coverage of text encoders, these approaches often struggle to explicitly integrate fine-grained knowledge tailored for downstream diagnostic tasks. To this end, we argue that effective knowledge integration requires task-centric, holistic alignment with disease-level knowledge throughout the entire diagnostic pipeline. As illustrated in Figure 1(a), our framework systematically integrates refined knowledge to guide input augmentation, feature extraction, and modality fusion, contrasting with prior methods confined to a single perspective. Figure 1(b) presents a case of the model's attention distribution over the input text. The previous model fails to concentrate on critical indicators, but focuses on obvious disease terms in the reports. While the RAD model can not only attend to these terms but also consider other guideline-recommended key indicators. The explicit knowledge guidance enables RAD to prioritize critical indicators tailored for the current disease, making trustworthy diagnoses aligned with clinical standards.

In this paper, we propose a holistic knowledge-injection framework RAD, which operates through three synergistic components spanning the entire diagnostic workflow. RAD begins by retrieving and refining disease-specific guidelines from diverse sources, flexibly adapting to downstream tasks in different scenarios. We then employ two modality encoders coupled with guideline-enhanced contrastive loss that explicitly aligns the modality-specific feature with the corresponding diseaseguideline prototypes in the joint latent space. A dual decoder network is further developed to steer the cross-modal fusion process, which simultaneously incorporates disease labels and their corresponding guidelines to interact with fused multimodal features for final predictions. Through this systematic knowledge infusion paradigm, our framework achieves performance gains while establishing a traceable decision pathway grounded in clinical guidelines-a critical step toward clinically actionable AI. We further establish an evaluation system for model interpretability, which quantitatively assesses the model's adherence to guidelines through both textual indicators and visual localization. Combined with qualitative visualization, this system provides measurable evidence that RAD's decisions are driven by the injected knowledge. In summary, our contributions are three-fold:

- We propose RAD to systematically inject external medical knowledge into multimodal diagnosis models. RAD incorporates a guideline-enhanced loss and a dual-decoder structure to explicitly steer multimodal feature extraction and cross-modal fusion with disease-guideline prototypes.

- A dual-axis evaluation system for the interpretability of diagnosis models is developed, formulating both textual and visual metrics. This system enables quantitative analyses of the model's adherence to clinical guidelines, demonstrating the transparency and explainability brought by RAD.
- Wealigned MIMIC-CXR [28] and MIMIC-IV [29] to construct the MIMIC-ICD53 dataset, covering three modalities with 53 types of disease. Extensive experiments on our dataset and three other public datasets demonstrate the superiority of RAD over SOTA baselines across various metrics.

## 2 Related Work

## 2.1 Multimodal Learning in Medicine

Recent years have witnessed significant advancements in the field of multimodal learning, with models such as CLIP [46], BLIP [33], and LLaVA [37] exhibiting remarkable capabilities in natural domains. These developments have spurred increasing interest in extending multimodal frameworks to the medical field, where the integration of diverse data modalities demonstrates prominent potential in diagnostic tasks. Current research focus lies in multimodal pretraining methods, which focus on cross-modal alignment between imaging and textual data to improve the representation transferability [7, 17]. ConVIRT [72] and GLoRIA [26] pioneered the application of CLIP-style architectures in the medical domain by constructing image-text pairs from radiology datasets. MedCLIP [56] and BiomedCLIP [70] addressed the scarcity of paired medical image-text data by leveraging multisource datasets, achieving state-of-the-art performance. Beyond pretraining methods, multimodal fusion approaches aim at integrating information from different modalities for diagnostic applications [20, 57, 66]. MedFuse [21] introduced an LSTM-based temporal fusion method of time-series data and X-ray images. HEALNet [23] proposed a hybrid early-fusion method to learn from data sources with different structures. While these works have made significant strides, they often operate without explicit guidance from medical knowledge when addressing specific diagnosis tasks. In contrast, considering the evidence-based nature of medicine [44, 51], RAD explicitly incorporates task-specific knowledge to guide both representation extraction and multimodal fusion processes.

## 2.2 Medical Knowledge Injection

Injecting professional knowledge into AI models is a prevalent strategy to improve their domainspecific capabilities [40, 65]. Various techniques have been investigated to incorporate medical knowledge into the models. Pretraining-based approaches train the text encoder on extensive medical domain corpora, such as PubMedBERT [18] and HUATUO-GPT [69]. Other methods like KAD [71] and DRAGON [67] leverage structured knowledge graphs of medical entities for pre-training to enhance the text encoder's comprehension of medical terminology. While showing empirical effectiveness, these knowledge integration methods remain primarily confined to the pretraining phase, providing only implicit guidance during the subsequent diagnostic stage. With the rapid development of large language models (LLMs) [1, 35, 54], various Retrieval-Augmented Generation (RAG) methods have been proposed to enhance the generation process of medical LLMs [13, 61]. These methods dynamically retrieve external medical knowledge to improve the performance of LLMs on question-answering (QA) tasks [34, 60]. Building upon this foundation, multimodal RAG approaches further retrieve similar data samples (e.g., image-report pairs) for visual question-answering (VQA) [59, 74]. In contrast to RAG methods that online retrieve knowledge to augment input for generative QA/VQA tasks, our framework adopts a structured approach for discriminative tasks by performing offline retrieval of disease-specific knowledge, which is systematically incorporated to guide model training.

## 3 Method

In this section, we first present the problem formulation, followed by the detailed introduction of our proposed method, Retrieval-Augmented Diagnosis (RAD), which consists of guideline retrieval and refinement, guideline-enhanced feature constraint, and dual diagnostic network. Finally, we introduce our interpretability evaluation system. The overall framework of RAD is illustrated in Figure 2.

Label

Figure 2: The overview of our Retrieval-Augmented Diagnosis framework, including multi-source medical knowledge retrieval and refinement, multimodal representation learning under the guideline constraint, and the dual diagnosis network. ⊕ represents the concatenation operation.

<!-- image -->

## 3.1 Problem Formulation

Given a training set of N samples, D = { ( x i , t i , y i ) } N i =1 , where x i represents a radiology image, t i is the report and electronic health records, and y i ∈ { 0 , 1 } m is the corresponding multi-label vector indicating the presence of m diseases. The multi-source medical knowledge corpus is denoted as P = { p i | i = 1 , 2 , . . . , s } , where p i denotes the i -th source and s denotes the number of sources. The guideline corresponding to the disease label is derived from multi-source retrieval and refinement. We denote this guideline as g = { g i | i = 1 , 2 , . . . , m } . The objective is to develop a multimodal model trained on D , capable of accurately predicting the disease for any given multimodal sample.

## 3.2 Retrieval-Augmented Diagnosis

## 3.2.1 Guideline Retrieval and Refinement

Knowledge-corpus. To retrieve disease-related diagnostic knowledge, we collect medical knowledge from four distinct sources: 'Wiki', 'Research', 'Guideline', and 'Book'. Wiki provides comprehensive descriptions of target diseases, such as formal medical definitions, and clinically relevant subcategories. Research incorporates the latest research articles from PubMed (a premier database of biomedical literature). These articles provide cutting-edge findings in disease mechanisms, diagnostic criteria, and therapeutic interventions. Guideline includes 45K clinical practice guidelines from 13 sources, providing rigorously vetted diagnostic criteria and treatment protocols for medical practitioners. Book consists of diverse medical textbooks, covering basic medical knowledge in surgery, medical imaging, and drugs, etc. More details of the corpus can be found in Appendix B.1.

Disease Knowledge Retrieval. For a given dataset with m diseases, our objective is to retrieve the most relevant knowledge from the knowledge corpus P , including but not limited to: associated symptoms, imaging characteristics, and critical examination/laboratory indicators. We adopt MedCPT [27], a dual-encoder model optimized for medical scenarios, as the retriever. Specifically, the article encoder R A ( · ) is utilized to convert the corpus P into dense vectors for retrieval. The disease names E are used as the input query of the query encoder R Q ( · ) . The obtained embeddings are then used to calculate the similarity as Sim( E,P ) = R Q ( E ) ⊤ R A ( P ) . For each disease with a name e i ∈ E , we preserve the topk retrieved documents as:

<!-- formula-not-decoded -->

LLMRefinement. Given that retrieved documents C i may contain content irrelevant to the diagnosis of the current disease and exhibit cross-source redundancy, directly combining the retrieved documents as the final guideline is suboptimal. In addition, the total document length often exceeds

the context window of the diagnosis model. To address these challenges, we employ large language models (LLMs) to perform automated summarization and refinement of C i . The final refined guideline g i of disease e i can be obtained by:

<!-- formula-not-decoded -->

where c i,j ∈ C i is the j -th document. This process yields standardized, well-structured diagnostic guidelines that preserve critical clinical information while eliminating noise and redundancy. In practice, we choose Qwen2.5-72B [64] as the LLM. Examples of the guideline and prompt templates are provided in Appendix B.2.

## 3.2.2 Guideline-enhanced Feature Constraint

For multimodal downstream tasks, our framework utilizes two modality-specific encoders to separately learn visual and textual representations. The refined guideline g obtained in Section 3.2.1 is employed here as the feature constraint of both textual and visual representation.

Given a sample ( x i , t i , y i ) , we use the vision encoder denoted as Φ img ( · ) to extract the visual embeddings V i from x i . The text encoder Φ text ( · ) is employed to obtain the textual embeddings T i . Meanwhile, the refined guideline g is also encoded by the text encoder for subsequent feature alignment. The encoding process is summarized as follows:

<!-- formula-not-decoded -->

where h, w are the height, width of the image, l is the max token length of the text encoder, m is the number of disease types, and d is the embedding dimension. These embeddings with spatial information are then used as the input of the dual decoder in Section 3.2.3 for multimodal fusion and final diagnosis. Here, we perform pooling on the extracted embeddings and use the pooled features for subsequent feature alignment. Specifically, we apply adaptive pooling operations to get the visual feature V ′ i ∈ R d , and directly use the embedding of the [CLS] token as the textual feature T ′ i ∈ R d . The corresponding pooled disease-guideline prototypes are G ′ = { G ′ i ∈ R d | i = 1 , 2 , . . . , m } .

To align the extracted features with diagnostic criteria, we propose a guideline-enhanced multi-modal feature constraint strategy. Specifically, disease-guideline prototypes are utilized as an anchor to pull both image and text features closer to them. To achieve this, we introduce a Guideline-Enhanced Contrastive Loss (GECL) for feature extraction under the guideline constraint. For sample i with the disease label y i , the guideline features G ′ are split into P i and N i , where P i = { G ′ j ∈ G ′ | y ij = 1 } is the set of guideline features corresponding to positive disease labels, N i is the set of guideline features with negative disease labels. To avoid using excessive negative samples, we sample a subset Q i from N i that satisfies | Q i | = min ( r | P i | , | N i | ) , where r is the negative sampling ratio. The final guideline feature set is S i = P i ∪ Q i . Then, we can formalize GECL as a cross-entropy-based supervised contrastive learning objective:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where I [ · ] is the indicator function. ϕ ( I i , S ij ) = I ⊤ i S ij /τ is the similarity score between the modality-specific feature and the guideline feature, τ is the temperature hyperparameter, and α is the trade-off hyperparameter. Note that the similarity score ϕ can be converted into a probability via the Sigmoid function. As shown in Eq. (5), L GECL aligns image features V ′ i and text features T ′ i with disease guideline prototypes, which are the diagnostic criteria of each disease defined by the embedding of its guideline. Dynamically aligning sample features with their positive prototypes prevents representation collapse while enhancing model robustness in multi-label scenarios. Furthermore, this approach induces the model to selectively focus on clinically relevant features that match the guidelines, improving the model performance and interpretability simultaneously. For detailed derivation from the standard cross-entropy form to Eq. (4), please refer to Appendix B.3.

## 3.2.3 Dual Diagnostic Network

Under the guideline constraint, we obtained enhanced visual and textual features. To achieve the final disease diagnosis, we develop a transformer-based cross-modal information fusion module, which has a dual decoder architecture. In the first decoder, the guideline g is employed as the query, and the concatenated modality embeddings V i ⊕ T i are used as the key and value. After forward through the fusion structure Φ g D , we obtain the logits corresponding to each disease:

<!-- formula-not-decoded -->

To further enhance the performance, we symmetrically utilize the second similar structure Φ l D where the query is replaced with the disease names, while keeping the key and value unchanged. This symmetric operation gets ˆ y label i = Φ l D (Φ text ( E ) , V i ⊕ T i , V i ⊕ T i ) ∈ R m , ensuring comprehensive feature integration. Finally, we compute the binary-cross-entropy loss on both logits with the ground truth. Thus, the total training loss of RAD is:

<!-- formula-not-decoded -->

where β represents the trade-off hyperparameter between L BCE and L GECL.

## 3.3 Interpretability Evaluation System

To validate the evidence-based diagnosis of RAD, we introduce a dual-axis interpretability evaluation system that quantitatively measures the model's adherence to injected guidelines through both textual and visual metrics. Formal definitions of the metrics for each input modality are presented below.

## 3.3.1 Textual Recall of Indicators

The Guideline Recall is designed to quantify the model's explicit compliance with diseasespecific diagnostic standards. The refined guideline of each disease contains a set of key laboratory indicators that are considered valuable for diagnosing this disease. The extent to which a model attends to these indicators can reflect its adherence to the guideline. Formally, when the input text contains indicators mentioned in the guideline, we assess the model's attention to these indicators by aggregating the attention weights of the corresponding tokens (derived from the cross-attention maps in the transformer

<!-- image -->

| Algorithm 1 Guideline Recall   | Algorithm 1 Guideline Recall                                                     |
|--------------------------------|----------------------------------------------------------------------------------|
| 1:                             | Input: Guideline G , text token sequence T , at- tention weights A , threshold θ |
| 2:                             | U ← Extract indicators from G                                                    |
| 3:                             | attended = 0 , total = 0                                                         |
| 4:                             | for each u ∈ U do                                                                |
| 5:                             | Matched ← Tokens in T matching u                                                 |
| 6:                             | if Matched = ∅ then                                                              |
| 7:                             | total = total +1                                                                 |
| 8:                             | if mean ( A Matched ) > θ then                                                   |
| 9:                             | attended = attended +1                                                           |
| 10:                            | return attended/total if total > 0 else 0                                        |

̸

decoders). When the aggregated attention weights exceed a predefined threshold θ , this provides quantitative evidence that the model exhibits statistically significant attention to the corresponding indicator. The detailed computation process is outlined in Algorithm 1.

## 3.3.2 Visual Attention Grounding Ability

For visual explainability, an attention-derived localization metric is employed to measure the alignment between model-attended regions and pathological abnormalities. Given expert-annotated bounding boxes for lesions, we compute the overlap between top-activated regions in the attention map and these ground truths. Specifically, we use the Intersection over Union IoU = | A ∩ B | | A ∪ B | as the metric, where A is the model localization derived from the attention map and B is the ground truth.

These two metrics formally establish a dual-modality interpretability evaluation system. Through the systematic analysis of how the injected knowledge explicitly intervenes in the model's decisionmaking, this system provides a quantitative evaluation for explainable multimodal medical AI.

Table 1: Performance across four datasets of different anatomies. The values of 'Acc" and 'Acc-S" on FairVLMed are the same since the dataset has only one disease. Subscript with arrows represents the absolute difference between RAD and the second-best method. ∆ is the variance of RAD.

| Dataset             | Method                                | F1                                        | Precision                                 | Recall                                    | AUC                                       | mAP                                       | Acc                                       | Acc-S                                     | Avg                                       |
|---------------------|---------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| MIMIC-ICD53 (Chest) | MedFuse BiomedCLIP KAD DrFuse HEALNet | 34.46 32.99 36.32 34.10 35.42             | 31.36 29.56 33.80 33.70 32.76             | 45.04 45.04 48.33 45.34 47.95             | 90.85 88.71 91.95 89.50 88.80             | 31.77 29.91 33.54 31.19 31.97             | 95.34 94.72 95.12 94.68 94.90             | 41.44 39.83 40.27 38.25 40.10             | 52.89 51.54 54.19 52.39 53.13             |
| FairVLMed           | RAD ∆ MedFuse BiomedCLIP KAD          | 39.71 3 . 39 ↑ ± 0.0101 81.33 81.27 81.18 | 39.07 5 . 27 ↑ ± 0.0099 76.13 72.87 73.92 | 54.74 6 . 41 ↑ ± 0.0016 87.29 91.88 90.03 | 93.00 1 . 05 ↑ ± 0.0103 87.99 87.69 88.62 | 36.74 3 . 20 ↑ ± 0.0116 88.76 87.62 88.88 | 95.40 0 . 06 ↑ ± 0.0050 79.50 78.35 78.65 | 42.33 0 . 89 ↑ ± 0.0228 79.50 78.35 78.65 | 57.28 3 . 09 ↑ ± 0.0089 83.50 83.28 83.55 |
| FairVLMed           | RAD ∆                                 | 84.30 2 . 50 ↑ ± 0.0028                   | 77.52 1 . 39 ↑ ± 0.0070                   | 92.38 0 . 50 ↑ ± 0.0005                   | 91.32 1 . 72 ↑ ± 0.0126                   | 91.88 1 . 43 ↑ ± 0.0144                   | 82.40 2 . 80 ↑ ± 0.0080                   | 82.40 2 . 80 ↑ ± 0.0080                   | 86.63 2 . 24 ↑ ± 0.0060                   |
| SkinCAP (Skin)      | MedFuse BiomedCLIP KAD DrFuse HEALNet | 79.25 81.49 82.06 81.18 82.20             | 85.96 87.13 86.79 85.70 88.69             | 77.99 81.41 81.27 79.64 81.18             | 96.50 97.22 97.80 94.92 92.68 97.97 0 . ± | 73.61 79.22 80.40 76.42 77.97             | 99.34 99.11 99.25 99.29 99.37             | 74.36 74.36 75.46 77.66 78.39             | 83.86 85.71 86.15 84.97 85.79             |
|                     | RAD ∆                                 | 85.48 3 . 28 ↑ ± 0.0678                   | 89.48 0 . 79 ↑ ± 0.0750                   | 83.23 1 . 82 ↑ ± 0.0136                   | 17 ↑ 0.0356                               | 83.55 3 . 15 ↑ ± 0.0639                   | 99.48 0 . 14 ↑ ± 0.0159                   | 81.32 2 . 93 ↑ ± 0.0474                   | 88.64 2 . 49 ↑ ± 0.0407                   |
| NACC (Brain)        | MedFuse BiomedCLIP KAD DrFuse HEALNet | 31.53 34.36 35.09 34.11 35.91             | 25.59 29.02 29.68 27.86 28.92             | 68.36 66.95 64.49 68.96 67.33             | 85.50 84.00 85.88 82.88 85.04             | 24.49 26.03 27.73 27.88 26.13             | 87.44 88.80 89.69 87.99 89.55             | 58.45 58.21 57.86 51.31 56.79             | 54.48 55.34 55.77 54.43 55.67             |
|                     | RAD ∆                                 | 37.65 1 . 74 ↑ ± 0.0015                   | 36.24 7 . 32 ↑ ± 0.0049                   | 65 . 78 1 . 55 ↓ ± 0.0003                 | 87.11 2 . 07 ↑ ± 0.0019                   | 30.03 3 . 90 ↑ ± 0.0023                   | 90.36 0 . 81 ↑ ± 0.0010                   | 59.64 2 . 85 ↑ ± 0.0078                   | 58.12 2 . 45 ↑ ± 0.0020                   |

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate RAD on four multimodal medical datasets with different anatomies, including MIMIC-ICD53, Harvard-FairVLMed [41], SkinCAP [76], and NACC [5].

Table 2: Detailed information of the datasets.

| Dataset           | Anatomy Modality   | Anatomy Modality                       |   Label Sample |   Label Sample |
|-------------------|--------------------|----------------------------------------|----------------|----------------|
| MIMIC-ICD53       | Chest              | X-ray Image &Report &EHR (Lab Results) |             53 |          51830 |
| Harvard-FairVLMed | Eye                | Fundus Image &Report &Demographics     |              1 |          10000 |
| SkinCAP           | Skin               | Dermatology Image &Report              |             50 |           2526 |
| NACC              | Brain              | 3D MRI Image &EHR (Lab Results)        |             11 |           4199 |

MIMIC-ICD53 is constructed through the alignment and integration of MIMIC-CXR [28] and MIMIC-IV [29], comprising chest X-ray images, corresponding reports, and EHRs, annotated with 53 diseases under the ICD [43] standard. For laboratory indicators in the EHR, we quantified the numerical results on a scale of 1 to 10 based on the upper and lower limits of their normal range. We will release the dataset on PhysioNet [42]. Details of dataset construction are provided in Appendix C.1.1. Harvard-FairVLMed, SkinCAP, and NACC are multimodal datasets focusing on eyes, skin, and brain, respectively. All patient data has been de-identified. More detailed statistics of datasets are presented in Table 2.

Baselines. We select representative baseline methods in the medical field, including large-scale pre-training model BiomedCLIP [70], knowledge-enhanced pre-training method KAD [71], and state-of-the-art multimodal fusion methods MedFuse [21], DrFuse [66], and HEALNet [23].

Evaluation Metrics. For the evaluation of model performance, we adopt widely used multi-label classification metrics including F1, Precision, Recall, AUC, mAP, and ACC. All metrics are the average of multiple labels. Since standard accuracy (ACC) aggregates predictions across all labels and thus may not adequately reflect the correctness for individual patients, we include an additional metric named sample-wise ACC (ACC-S). This metric considers a prediction correct only if all labels of a patient are accurately classified, making it more aligned with clinical scenarios.

Table 3: Quantitative evaluation of Visual Explainability. We calculate the metrics for each disease category and report both disease-averaged (Avg-D) and patient-averaged (Avg-P) values.

| Method   | Visual Grounding (mIoU)   | Visual Grounding (mIoU)   | Visual Grounding (mIoU)   | Visual Grounding (mIoU)   | Visual Grounding (mIoU)   | Visual Grounding (mIoU)   | Visual Grounding (mIoU)   | Visual Grounding (mIoU)   | Visual Grounding (mIoU)   |
|----------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
|          | Consolidation             | Atelectasis               | Effusion                  | Emphysema                 | Fibrosis                  | Fracture                  | Mass                      | Avg-D                     | Avg-P                     |
| w/o RAD  | 17.68                     | 19.23                     | 18.89                     | 14.95                     | 17.22                     | 13.13                     | 10.81                     | 15.98                     | 17.78                     |
| RAD      | 24.30                     | 20.74                     | 20.13                     | 21.15                     | 19.42                     | 17.14                     | 15.15                     | 19.72                     | 22.04                     |

Table 4: Quantitative evaluation of Textual Explainability. We present the guideline recall on representative laboratory indicators and the total average recall. The indicator names are abbreviated.

| Method   | Guideline Recall   | Guideline Recall   | Guideline Recall   | Guideline Recall   | Guideline Recall   | Guideline Recall   | Guideline Recall   |
|----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
|          | PC                 | Bilirubin          | ALT                | IBC                | WBC                | AST                | Total              |
| w/o RAD  | 23.82              | 31.34              | 6.81               | 37.38              | 11.96              | 4.41               | 24.76              |
| RAD      | 64.55              | 51.71              | 57.96              | 71.82              | 29.09              | 40.65              | 65.62              |

Implementation Details. In practice, Topk in Eq.(1) is set to 10. All guidelines obtained by Eq.(2) and the indicators used in Algorithm 1 are manually verified to avoid potential factual errors. The default backbone of the text image encoder is ClinicalBERT [55] and ResNet-50 [22], respectively. The hyperparameters α and β , which serve as the balancing ratio between different losses, are set to be 1 e -2 and 1 e -1 , respectively. All experiments are conducted on a single NVIDIA A100 GPU.

## 4.2 Diagnosis Performance

As demonstrated in Table 1, our method consistently achieves superior performance across four benchmarks of diverse anatomies. Specifically, RAD outperforms the second-best method with average improvements of 3.09%, 2.24%, 2.49%, and 2.45% on MIMIC-ICD53, FairVLMed, SkinCAP, and NACC datasets, respectively. The most substantial gains occur in MIMIC-ICD53, where RAD improves both precision and recall over 5%, suggesting strong robustness in handling complex, real-world clinical label distributions. This improvement is particularly noteworthy given the dataset's challenging nature, containing both fine-grained ICD labels and noisy clinical documentation. Notably, the sample-wise accuracy (ACC-S) of all methods exhibits a significant degradation compared to macro-average accuracy (ACC), especially in datasets with extensive label spaces. This discrepancy highlights fundamental limitations in current models' capacity to handle multi-label problems, exposing challenges for real-world clinical deployment. Intriguingly, KAD, which injects medical knowledge during the pretraining phase, achieves strong performance on MIMIC-ICD53 but falls short on others. This is likely because its pretraining data concentrated on chest X-rays, limiting its ability to generalize to other anatomical regions. In contrast, our approach directly injects knowledge on downstream tasks, offering greater adaptability across distinct regions and modalities. These consistent improvements across diverse anatomies, data scales, and label complexities validate the versatility and scalability of RAD. Full baseline results with variance are in Appendix C.3.

## 4.3 Interpretability Evaluation

## 4.3.1 Interpretability from Textual Perspective

To quantitatively assess the impact of knowledge injection from the textual perspective, we calculate the guideline recall defined in Section 3.3.1. As presented in Table 4, incorporating knowledge via RAD prominently increases the recall value from 24.76% to 65.62%. This indicates that RAD indeed injects guideline-derived knowledge into the model, thereby enhancing its focus on key information mentioned in the guideline. Notably, the conventional model exhibits extremely low recall (&lt;10%) on Alanine Aminotransferase (ALT) and Aspartate Aminotransferase (AST). This may stem from their inability to understand highly specialized, rare medical terms. In contrast, RAD explicitly highlights the importance of these indicators in the guideline, leading to significant recall improvement. This finding underscores the necessity of flexible knowledge adaptation for downstream tasks rather than static pretraining paradigms. Overall, the enhanced guideline recall demonstrates that RAD enables the model to make reliable evidence-based diagnoses according to guidelines. This improvement also aligns with the qualitative attention patterns observed in Figure 1. To further substantiate the interpretability of RAD, we provide more and clearer visualization cases in Appendix C.5.

Figure 4: Performance under different combinations of modality encoder backbones on the MIMICICD53 dataset. (R = ResNet, V = ViT, C = ClinicalBERT, B = BioClinicalBERT)

<!-- image -->

Table 5: Ablation on each component of our method. " × " in the "Decoder" column means replacing our dual diagnostic decoder with a conventional MLP. The best results are in boldface .

| L vision GECL   | L text GECL   | Decoder   |    F1 |   Precision |   Recall |   AUC |   mAP |   Acc |   Acc-S |   Avg |
|-----------------|---------------|-----------|-------|-------------|----------|-------|-------|-------|---------|-------|
| ×               | ×             | ×         | 34.91 |       31.01 |    50.91 | 91.27 | 32.24 | 94.5  |   38.63 | 53.35 |
| ✓               | ×             | ×         | 37.43 |       33.98 |    51.44 | 92.53 | 34.8  | 95.26 |   38.1  | 54.79 |
| ×               | ✓             | ×         | 37.75 |       36.32 |    51.52 | 92.91 | 35.03 | 95.43 |   39.65 | 55.52 |
| ✓               | ✓             | ×         | 39.34 |       37.74 |    51.87 | 92.94 | 36.36 | 95.59 |   39.95 | 56.26 |
| ×               | ×             | ✓         | 39.22 |       36.88 |    51.41 | 92.25 | 36.44 | 95.39 |   39.8  | 55.91 |
| ✓               | ✓             | ✓         | 39.71 |       39.07 |    54.74 | 93    | 36.74 | 95.4  |   42.33 | 57.28 |

## 4.3.2 Interpretability from Visual Perspective

<!-- image -->

Disease

Mass

Fibrosis

Atelectasis

Consolidation

Figure 3: Visualization of grounding results on four diseases.

Symmetrically, we conduct zero-shot grounding experiments on the ChestXDet dataset [38]. The results shown in Table 3 demonstrate a significant improvement in mIoU scores for lesion detection after the injection of refined guidelines. Besides, Figure 3 illustrates multiple cases of lesion grounding. For clearer visualization, we overlay spectrum heatmaps on the original CXR images, together with the ground truth bounding box high- lighted in red. A comparison between the lesions identified by the model and the bounding boxes marked by clinical experts reveals a notable improvement in alignment when our guidelines are applied. This indicates that the model's focus is more accurately directed toward clinically significant lesions, emphasizing RAD's enhanced diagnosis capabilities and interpretability under the guidance of external knowledge.

## 4.4 Ablation Study

In this subsection, we conduct ablation studies on each component of RAD and validate its generalizability across different model architectures. All experiments are conducted on MIMIC-ICD53.

Ablation on Each Component. As shown in Table 5, we evaluate the efficacy of each newly proposed component in RAD. It is evident that removing either the L GECL or the dual decoder negatively impacts model performance, highlighting the importance of the guideline in both representation learning and multimodal fusion. Notably, the removal of the Dual Decoder results in the most substantial performance degradation, underscoring the necessity of leveraging guidelines to intervene in the final decision-making process. We further compared the performance of the textual and visual branches of L GECL when used individually. The results show that the textual branch yields more significant improvements. This can be attributed to the fact that both the input text and the guideline belong to the same modality, allowing for more effective alignment.

Ablation on Different Backbones. To demonstrate the robustness and flexibility of our method, we verify RAD on different encoder backbone combinations. Specifically, we iteratively replaced the default visual encoder and text encoder with two other popular architectures, ViT [15] and

Customized Radar Chart

Figure 5: Performance comparison with MLLMs. We convert the single-label dataset FairVLMed into the VQA format and evaluate MLLMs under both few-shot and supervised fine-tuning settings.

<!-- image -->

BioClinicalBERT [2]. As illustrated in Figure 4, RAD consistently offers substantial performance gain across all four combinations of backbones. This not only highlights the insensitivity of our approach to different backbone architectures but also underscores its robustness and generalizability. Specifically, ResNet and ViT exhibit comparable performance gains, while ClinicalBERT shows more pronounced improvement than BioClinicalBERT. Overall, RAD exhibits the robust ability to generalize to diverse data and model structures, ensuring reliable performance in various scenarios. Detailed results, more ablation studies, and hyperparameter analysis are presented in Appendix C.6.

## 4.5 Discussion

Comparison with Multimodal Large Language Models Multimodal large language models exhibit remarkable capabilities in visual content understanding and generalization. To further validate the effectiveness of RAD, we compare with state-of-the-art MLLMs, including Qwen2.5-VL-7B [3], HuatuoGPT-Vision-7B [10], and Lingshu-7B [62]. As presented in Figure 5, our discriminative framework achieves superior performance with significantly lower computational cost. These results demonstrate that complex diagnostic tasks are better suited for specialized discriminative models than generative MLLMs. The significant performance gap, observed on the simplest single-label dataset, underscores the practical advantages of our approach in clinical applications where both accuracy and efficiency are critical.

## 5 Conclusion

This paper proposes RAD, which enhances the capabilities of multimodal diagnosis models by leveraging external medical knowledge. RAD operates via a tri-fold methodology, consisting of offline retrieval and refinement of disease-centered external guidelines, multimodal feature alignment under the guideline constraint, and the dual diagnostic network. Extensive experiments on four datasets of different anatomies demonstrate the effectiveness of RAD. Furthermore, RAD exhibits dual-axis interpretability by simultaneously achieving precise lesion localization in imaging data and prioritizing guideline-concordant indicators in textual analysis. This evidence-based explainability enhances clinical trustworthiness, offering the potential to inspire future research in medical AI.

## Acknowledgement

This work is supported by the National Key R&amp;D Program of China (No. 2022ZD0160703), National Natural Science Foundation of China (No. 62306178) and STCSM (No. 22DZ2229005), 111 plan (No. BP0719010), and Shanghai Artificial Intelligence Laboratory.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 , 2023.
- [2] Emily Alsentzer, John R Murphy, Willie Boag, Wei-Hung Weng, Di Jin, Tristan Naumann, and Matthew McDermott. Publicly available clinical bert embeddings. arXiv preprint arXiv:1904.03323 , 2019.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 , 2025.
- [4] Tadas Baltrušaitis, Chaitanya Ahuja, and Louis-Philippe Morency. Multimodal machine learning: A survey and taxonomy. IEEE transactions on pattern analysis and machine intelligence , 41(2):423-443, 2018.
- [5] Duane L Beekly, Erin M Ramos, William W Lee, Woodrow D Deitrich, Mary E Jacka, Joylee Wu, Janene L Hubbard, Thomas D Koepsell, John C Morris, Walter A Kukull, et al. The national alzheimer's coordinating center (nacc) database: the uniform data set. Alzheimer Disease &amp; Associated Disorders , 21(3):249-258, 2007.
- [6] Edgar A Bernal, Xitong Yang, Qun Li, Jayant Kumar, Sriganesh Madhvanath, Palghat Ramesh, and Raja Bala. Deep temporal multimodal fusion for medical procedure monitoring using wearable sensors. IEEE Transactions on Multimedia , 20(1):107-118, 2017.
- [7] Benedikt Boecking, Naoto Usuyama, Shruthi Bannur, Daniel C Castro, Anton Schwaighofer, Stephanie Hyland, Maria Wetscherek, Tristan Naumann, Aditya Nori, Javier Alvarez-Valle, et al. Making the most of text semantics to improve biomedical vision-language processing. In European conference on computer vision , pages 1-21. Springer, 2022.
- [8] Cheng Chen, Qi Dou, Yueming Jin, Quande Liu, and Pheng Ann Heng. Learning with privileged multimodal knowledge for unimodal segmentation. IEEE transactions on medical imaging , 41(3):621-632, 2021.
- [9] Junying Chen, Chi Gui, Anningzhe Gao, Ke Ji, Xidong Wang, Xiang Wan, and Benyou Wang. Cod, towards an interpretable medical agent using chain of diagnosis. arXiv preprint arXiv:2407.13301 , 2024.
- [10] Junying Chen, Chi Gui, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy Chen, Xidong Wang, Ruifei Zhang, Zhenyang Cai, Ke Ji, et al. Huatuogpt-vision, towards injecting medical visual knowledge into multimodal llms at scale. arXiv preprint arXiv:2406.19280 , 2024.
- [11] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset. Advances in Neural Information Processing Systems , 36:72842-72866, 2023.
- [12] Zeming Chen, Alejandro Hernández Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas Köpf, Amirkeivan Mohtashami, et al. Meditron-70b: Scaling medical pretraining for large language models. arXiv preprint arXiv:2311.16079 , 2023.
- [13] Zhe Chen, Yusheng Liao, Shuyang Jiang, Pingjie Wang, Yiqiu Guo, Yanfeng Wang, and Yu Wang. Towards omni-rag: Comprehensive retrieval-augmented generation for large language models in medical applications. arXiv preprint arXiv:2501.02460 , 2025.
- [14] Tianjie Dai, Ruipeng Zhang, Feng Hong, Jiangchao Yao, Ya Zhang, and Yanfeng Wang. Unichest: Conquer-and-divide pre-training for multi-source chest x-ray classification. IEEE Transactions on Medical Imaging , 2024.
- [15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 , 2020.
- [16] Qi Dou, Quande Liu, Pheng Ann Heng, and Ben Glocker. Unpaired multi-modal segmentation via knowledge distillation. IEEE transactions on medical imaging , 39(7):2415-2425, 2020.
- [17] Sedigheh Eslami, Christoph Meinel, and Gerard De Melo. Pubmedclip: How much does clip benefit visual question answering in the medical domain? In Findings of the Association for Computational Linguistics: EACL 2023 , pages 1181-1193, 2023.
- [18] Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare (HEALTH) , 3(1):1-23, 2021.

- [19] Bo Han, Jiangchao Yao, Tongliang Liu, Bo Li, Sanmi Koyejo, Feng Liu, et al. Trustworthy machine learning: From data to models. Foundations and Trends® in Privacy and Security , 7(2-3):74-246, 2025.
- [20] Hrayr Harutyunyan, Hrant Khachatrian, David C Kale, Greg Ver Steeg, and Aram Galstyan. Multitask learning and benchmarking with clinical time series data. Scientific data , 6(1):96, 2019.
- [21] Nasir Hayat, Krzysztof J Geras, and Farah E Shamout. Medfuse: Multi-modal fusion with clinical timeseries data and chest x-ray images. In Machine Learning for Healthcare Conference , pages 479-503. PMLR, 2022.
- [22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [23] Konstantin Hemker, Nikola Simidjievski, and Mateja Jamnik. Healnet: Multimodal fusion for heterogeneous biomedical data. Advances in Neural Information Processing Systems , 37:64479-64498, 2024.
- [24] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature , 637(8045):319-326, 2025.
- [25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR , 1(2):3, 2022.
- [26] Shih-Cheng Huang, Liyue Shen, Matthew P Lungren, and Serena Yeung. Gloria: A multimodal globallocal representation learning framework for label-efficient medical image recognition. In Proceedings of the IEEE/CVF international conference on computer vision , pages 3942-3951, 2021.
- [27] Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau, Lana Yeganova, W John Wilbur, and Zhiyong Lu. Medcpt: Contrastive pre-trained transformers with large-scale pubmed search logs for zero-shot biomedical information retrieval. Bioinformatics , 39(11):btad651, 2023.
- [28] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific data , 6(1):317, 2019.
- [29] Alistair EW Johnson, Lucas Bulgarelli, Lu Shen, Alvin Gayles, Ayad Shammout, Steven Horng, Tom J Pollard, Sicheng Hao, Benjamin Moody, Brian Gow, et al. Mimic-iv, a freely accessible electronic health record dataset. Scientific data , 10(1):1, 2023.
- [30] Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics , 36(4):1234-1240, 2020.
- [31] Karim Lekadir, Richard Osuala, Catherine Gallin, Noussair Lazrak, Kaisar Kushibar, Gianna Tsakou, Susanna Aussó, Leonor Cerdá Alberich, Kostas Marias, Manolis Tsiknakis, et al. Future-ai: guiding principles and consensus recommendations for trustworthy artificial intelligence in medical imaging. arXiv preprint arXiv:2109.09658 , 2021.
- [32] Haolin Li, Yuhang Zhou, Ziheng Zhao, Siyuan Du, Jiangchao Yao, Weidi Xie, Ya Zhang, and Yanfeng Wang. Lorkd: Low-rank knowledge decomposition for medical foundation models. arXiv preprint arXiv:2409.19540 , 2024.
- [33] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning , pages 19730-19742. PMLR, 2023.
- [34] Mingchen Li, Halil Kilicoglu, Hua Xu, and Rui Zhang. Biomedrag: A retrieval augmented large language model for biomedicine. Journal of Biomedical Informatics , 162:104769, 2025.
- [35] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 , 2024.
- [36] Fenglin Liu, Chenyu You, Xian Wu, Shen Ge, Xu Sun, et al. Auto-encoding knowledge graph for unsupervised medical report generation. Advances in Neural Information Processing Systems , 34:1626616279, 2021.
- [37] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems , 36:34892-34916, 2023.

- [38] Jingyu Liu, Jie Lian, and Yizhou Yu. Chestx-det10: chest x-ray dataset on detection of thoracic abnormalities. arXiv preprint arXiv:2006.10550 , 2020.
- [39] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision , pages 10012-10022, 2021.
- [40] Zhuang Liu, Degen Huang, Kaiyu Huang, Zhuang Li, and Jun Zhao. Finbert: A pre-trained financial language representation model for financial text mining. In Proceedings of the twenty-ninth international conference on international joint conferences on artificial intelligence , pages 4513-4519, 2021.
- [41] Yan Luo, Min Shi, Muhammad Osama Khan, Muhammad Muneeb Afzal, Hao Huang, Shuaihang Yuan, Yu Tian, Luo Song, Ava Kouhana, Tobias Elze, et al. Fairclip: Harnessing fairness in vision-language learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 12289-12301, 2024.
- [42] George B Moody. Physionet. In Encyclopedia of Computational Neuroscience , pages 2806-2808. Springer, 2022.
- [43] World Health Organization et al. International classification of diseases-icd. 2009.
- [44] Yifan Peng, Justin F Rousseau, Edward H Shortliffe, and Chunhua Weng. Ai-generated text may have a role in evidence-based medicine. Nature medicine , 29(7):1593-1594, 2023.
- [45] Fernando Pérez-García, Harshita Sharma, Sam Bond-Taylor, Kenza Bouzid, Valentina Salvatelli, Maximilian Ilse, Shruthi Bannur, Daniel C Castro, Anton Schwaighofer, Matthew P Lungren, et al. Exploring scalable medical image encoders beyond text supervision. Nature Machine Intelligence , pages 1-12, 2025.
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pages 8748-8763. PmLR, 2021.
- [47] Laila Rasmy, Yang Xiang, Ziqian Xie, Cui Tao, and Degui Zhi. Med-bert: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. NPJ digital medicine , 4(1):86, 2021.
- [48] Maya Rotmensch, Yoni Halpern, Abdulhakim Tlimat, Steven Horng, and David Sontag. Learning a health knowledge graph from electronic medical records. Scientific reports , 7(1):5994, 2017.
- [49] Zohaib Salahuddin, Henry C Woodruff, Avishek Chatterjee, and Philippe Lambin. Transparency of deep neural networks for medical image analysis: A review of interpretability methods. Computers in biology and medicine , 140:105111, 2022.
- [50] Gregor Stiglic, Primoz Kocbek, Nino Fijacko, Marinka Zitnik, Katrien Verbert, and Leona Cilar. Interpretability of machine learning-based prediction models in healthcare. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery , 10(5):e1379, 2020.
- [51] Vivek Subbiah. The next generation of evidence-based medicine. Nature medicine , 29(1):49-58, 2023.
- [52] Heung-Il Suk, Seong-Whan Lee, Dinggang Shen, Alzheimer's Disease Neuroimaging Initiative, et al. Hierarchical feature representation and multimodal fusion with deep learning for ad/mci diagnosis. NeuroImage , 101:569-582, 2014.
- [53] Jesse Sun, Fatemeh Darbehani, Mark Zaidi, and Bo Wang. Saunet: Shape attentive u-net for interpretable medical image segmentation. In International conference on medical image computing and computerassisted intervention , pages 797-806. Springer, 2020.
- [54] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 , 2023.
- [55] Guangyu Wang, Xiaohong Liu, Zhen Ying, Guoxing Yang, Zhiwei Chen, Zhiwen Liu, Min Zhang, Hongmei Yan, Yuxing Lu, Yuanxu Gao, et al. Optimized glycemic control of type 2 diabetes with reinforcement learning: a proof-of-concept trial. Nature Medicine , 29(10):2633-2642, 2023.
- [56] Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng Sun. Medclip: Contrastive learning from unpaired medical images and text. In Proceedings of the Conference on Empirical Methods in Natural Language Processing. Conference on Empirical Methods in Natural Language Processing , volume 2022, page 3876, 2022.

- [57] Tom Nuno Wolf, Sebastian Pölsterl, Christian Wachinger, Alzheimer's Disease Neuroimaging Initiative, et al. Daft: A universal module to interweave tabular data and 3d images in cnns. NeuroImage , 260:119505, 2022.
- [58] Chaoyi Wu, Pengcheng Qiu, Jinxin Liu, Hongfei Gu, Na Li, Ya Zhang, Yanfeng Wang, and Weidi Xie. Towards evaluating and building versatile large language models for medicine. npj Digital Medicine , 8(1): 58, 2025.
- [59] Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun Zhang, James Zou, and Huaxiu Yao. Mmed-rag: Versatile multimodal rag system for medical vision language models. arXiv preprint arXiv:2410.13085 , 2024.
- [60] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-augmented generation for medicine. In Findings of the Association for Computational Linguistics ACL 2024 , pages 6233-6251, 2024.
- [61] Ran Xu, Wenqi Shi, Yue Yu, Yuchen Zhuang, Bowen Jin, May D Wang, Joyce C Ho, and Carl Yang. Ram-ehr: Retrieval augmentation meets clinical predictions on electronic health records. arXiv preprint arXiv:2403.00815 , 2024.
- [62] Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, et al. Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning. arXiv preprint arXiv:2506.07044 , 2025.
- [63] Chonghua Xue, Sahana S Kowshik, Diala Lteif, Shreyas Puducheri, Varuna H Jasodanand, Olivia T Zhou, Anika S Walia, Osman B Guney, J Diana Zhang, Serena T Pham, et al. Ai-based differential diagnosis of dementia etiologies on multimodal data. Nature Medicine , 30(10):2977-2989, 2024.
- [64] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115 , 2024.
- [65] Yue Yang, Mona Gandhi, Yufei Wang, Yifan Wu, Michael Yao, Chris Callison-Burch, James Gee, and Mark Yatskar. A textbook remedy for domain shifts: Knowledge priors for medical image analysis. Advances in Neural Information Processing Systems , 37:90683-90713, 2024.
- [66] Wenfang Yao, Kejing Yin, William K Cheung, Jia Liu, and Jing Qin. Drfuse: Learning disentangled representation for clinical multi-modal fusion with missing modality and modal inconsistency. In Proceedings of the AAAI conference on artificial intelligence , volume 38, pages 16416-16424, 2024.
- [67] Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D Manning, Percy S Liang, and Jure Leskovec. Deep bidirectional language-knowledge graph pretraining. Advances in Neural Information Processing Systems , 35:37309-37323, 2022.
- [68] Sukwon Yun, Inyoung Choi, Jie Peng, Yangfan Wu, Jingxuan Bao, Qiyiwen Zhang, Jiayi Xin, Qi Long, and Tianlong Chen. Flex-moe: Modeling arbitrary modality combination via the flexible mixture-of-experts. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [69] Hongbo Zhang, Junying Chen, Feng Jiang, Fei Yu, Zhihong Chen, Jianquan Li, Guiming Chen, Xiangbo Wu, Zhiyi Zhang, Qingying Xiao, et al. Huatuogpt, towards taming language model to be a doctor. arXiv preprint arXiv:2305.15075 , 2023.
- [70] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al. Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915 , 2023.
- [71] Xiaoman Zhang, Chaoyi Wu, Ya Zhang, Weidi Xie, and Yanfeng Wang. Knowledge-enhanced visuallanguage pre-training on chest radiology images. Nature Communications , 14(1):4542, 2023.
- [72] Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D Manning, and Curtis P Langlotz. Contrastive learning of medical visual representations from paired images and text. In Machine learning for healthcare conference , pages 2-25. PMLR, 2022.
- [73] Zizhao Zhang, Yuanpu Xie, Fuyong Xing, Mason McGough, and Lin Yang. Mdnet: A semantically and visually interpretable medical image diagnosis network. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 6428-6436, 2017.
- [74] Ruochen Zhao, Hailin Chen, Weishi Wang, Fangkai Jiao, Xuan Long Do, Chengwei Qin, Bosheng Ding, Xiaobao Guo, Minzhi Li, Xingxuan Li, et al. Retrieving multimodal information for augmented generation: A survey. arXiv preprint arXiv:2303.10868 , 2023.

- [75] Hong-Yu Zhou, Xiaoyu Chen, Yinghao Zhang, Ruibang Luo, Liansheng Wang, and Yizhou Yu. Generalized radiograph representation learning via cross-supervision between images and free-text radiology reports. Nature Machine Intelligence , 4(1):32-40, 2022.
- [76] Juexiao Zhou, Liyuan Sun, Yan Xu, Wenbin Liu, Shawn Afvari, Zhongyi Han, Jiaoyan Song, Yongzhi Ji, Xiaonan He, and Xin Gao. Skincap: A multi-modal dermatology dataset annotated with rich medical captions. arXiv preprint arXiv:2405.18004 , 2024.
- [77] Yuhang Zhou, Siyuan Du, Haolin Li, Jiangchao Yao, Ya Zhang, and Yanfeng Wang. Reprogramming distillation for medical foundation models. In International Conference on Medical Image Computing and Computer-Assisted Intervention , pages 533-543. Springer, 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Section 3 explains our method. Section 4 presents experimental results to verify RAD on different datasets with diverse anatomies (Section 4.2) and different model structures (Section 4.4).

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitation is discussed in Appendix A.2.

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

Justification: There are no included theoretical results in this work.

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

Justification: We have included a detailed experimental setup in Section 4.1, including dataset and implementation details (configuration like metrics, hyperparameters, etc) The preprocessing of the datasets is provided in Appendix C.1.1. The construction of the retrieval corpus is provided in Appendix B.1. And the dataset and code will be publicly available.

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

Justification: The code is available at: https://github.com/tdlhl/RAD . With the exception of MIMIC-ICD53, all datasets are publicly available. We will also make our constructed MIMIC-ICD53 available on PhysioNet upon publication.

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

Justification: We have included a detailed experimental setup in Section 4.1, including dataset and implementation details (configuration like metrics, hyperparameters, etc). And the details can also be found in the code.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We have reported the variance of our method over 5 runs in Table 1. The full results with variance of all baselines are presented in Appendix C.3.

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

Justification: All the experiments are conducted on a single NVIDIA A100 GPU, and we have reported the compute sources in the implementation details of Section 4.1.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have reviewed the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Broader impacts have been discussed in Appendix A.1.

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

Justification: There is no safeguard risk in this work.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We have cited the original papers and followed their license.

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

Justification: The code is available at: https://github.com/tdlhl/RAD . Due to licensing restrictions, our new dataset (MIMIC-ICD53) will be available only on PhysioNet upon publication. The construction details of this dataset have been included in Appendix C.1.1.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: We utilized LLMs for the guideline refinement, which has been thoroughly detailed in Section 3.2.1. The function of LLM here is to serve as a powerful long-context text summarization tool.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

| Further              | Discussion                                               |   24 |
|----------------------|----------------------------------------------------------|------|
| A.1                  | Broader impact . . . . . . . . . . . . . . . . . . . .   |   24 |
| A.2                  | Limitations . . . . . . . . . . . . . . . . . . . . . .  |   24 |
| Method Details       | Method Details                                           |   24 |
| B.1                  | Knowledge-corpus Construction . . . . . . . . . . .      |   24 |
| B.2                  | Details for LLM Refinement . . . . . . . . . . . . .     |   24 |
| B.3                  | Derivation of the Guideline-Enhanced Contrastive Loss    |   25 |
| Experimental Details | Experimental Details                                     |   27 |
| C.1                  | Details of Datasets . . . . . . . . . . . . . . . . . .  |   27 |
| C.2                  | Benchmarking MIMIC-ICD53 . . . . . . . . . . . .         |   29 |
| C.3                  | The Variance of Baselines . . . . . . . . . . . . . . .  |   30 |
| C.4                  | Label-wise Analysis of MIMIC-ICD53 . . . . . . . .       |   31 |
| C.5                  | Interpretability Cases . . . . . . . . . . . . . . . . . |   31 |
| C.6                  | Ablation Study . . . . . . . . . . . . . . . . . . . .   |   33 |
| C.7                  | Cost Analysis. . . . . . . . . . . . . . . . . . . . . . |   34 |
| C.8                  | Hyper-parameter analysis . . . . . . . . . . . . . . .   |   34 |

## A Further Discussion

## A.1 Broader impact

The method proposed in this paper can effectively enhance the diagnostic capability of multimodal medical models. With the integration of the guidelines, RAD is optimized through intervention in accordance with the guidelines. This not only improves diagnostic accuracy but also strengthens the model's interpretability, making its decision-making process more transparent and deployable in realworld clinical scenarios. Specifically, the systematic integration of multimodal data (imaging, text, and structured records) enables RAD to capture disease manifestations from multiple perspectives, potentially advancing personalized medicine through comprehensive patient profiling. However, the use of multimodal clinical data, including sensitive patient records and imaging features, necessitates stringent compliance with relevant regulations to prevent misuse or unintended leakage of private health information.

## A.2 Limitations

A limitation of the current implementation is the static retrieval knowledge corpus. While medical guidelines undergo periodic updates (e.g., every 3-5 years) to incorporate new evidence and diseases, RAD relies on a fixed knowledge base that may require manual updates to reflect revised diagnostic standards. This temporal mismatch could be addressed by regular updates of guidelines and further fine-tuning of the models, which would enhance long-term clinical relevance without compromising current performance.

## B Method Details

## B.1 Knowledge-corpus Construction

For retrieving disease-related diagnostic knowledge, we collect medical knowledge from four distinct sources: 'Wiki', 'Research', 'Guideline', and 'Book'. Wikipedia provides comprehensive and general descriptions of target diseases, such as standard disease nomenclature, formal medical definitions, and clinically relevant subcategories. The processed data are obtained from Huggingface 2 . Research incorporates the latest research articles from PubMed (a premier database of biomedical literature). These articles provide cutting-edge findings in disease mechanisms, diagnostic criteria, and therapeutic interventions. We utilize the 2024 PubMed baseline 3 , which is a complete snapshot of PubMed data. We filter the valid data through their paper titles and corresponding abstracts. Guideline includes 45K clinical practice guidelines from 13 sources. The guidelines provide rigorously vetted diagnostic criteria and treatment protocols for medical practitioners, serving as a critical component for reliable decision support. We employ the Clinical Guidelines dataset [12] and use the provided scripts to crawl non-redistributable portions of the data. Book. consists of diverse medical textbooks. These books cover well-organized basic medical knowledge in surgery, medical imaging, and drugs, etc. We follow MedOmniKB [13] to collect 18K PDF documents from online medical libraries and academic publishers. Then, deduplicate and filter these books to obtain the final retrieval database.

## B.2 Details for LLM Refinement

An example of the final guideline is presented in Figure 6. For more guidelines, please refer to our GitHub repository.

The detailed prompt template for LLM refinement is shown in Figure 7. In the prompt, {dis-ease\_icd\_name} is the disease name e i , {topk} is the number of preserved documents, and {re-trieve\_passages\_str} is the content of the document.

2 https://huggingface.co/datasets/wikimedia/wikipedia

3 https://ftp.ncbi.nlm.nih.gov/pubmed/baseline

## The guideline of "bronchitis":

Summary of Key Diagnostic Features for Bronchitis

Disease Description: Bronchitis is an inflammation of the bronchi, the air passages in the lungs. It can be classified into two main types: acute and chronic. Acute bronchitis is typically a self-limiting condition characterized by a cough that may produce sputum and is often caused by viral infections. Chronic bronchitis, on the other hand, is a long-term condition defined by a productive cough lasting for at least three months in two consecutive years, often associated with chronic obstructive pulmonary disease (COPD). The primary risk factor for chronic bronchitis is tobacco smoking, with other factors including air pollution and occupational exposures.

Important Lab Tests and Values: - Acute Bronchitis: White Blood Cell Count (WBC): Usually normal or slightly elevated. C-reactive Protein (CRP): May be slightly elevated but not typically high. Sputum Culture: Not routinely necessary, but can be useful if bacterial infection is suspected. Chronic Bronchitis: Pulmonary Function Tests (PFTs): Reduced FEV1/FVC ratio, indicating airflow obstruction. Sputum Analysis: Increased mucus production, often with neutrophil infiltration. Blood Gas Analysis:** May show hypoxemia and hypercapnia in advanced cases.

Key Radiological or Clinical Findings: Acute Bronchitis: Chest X-ray: Usually normal, but may show hyperinflation or peribronchial thickening. Physical Examination: Wheezing, crackles, and rhonchi on auscultation. Chronic Bronchitis: Chest X-ray: May show hyperinflation, increased bronchovascular markings, and signs of emphysema. CT Scan: Can reveal bronchial wall thickening and mucus plugging. Physical Examination: Barrel chest, cyanosis, and signs of cor pulmonale in advanced cases.

Diagnostic Symptoms or Relevant Clinical Features: Acute Bronchitis: Cough: Initially dry, then becomes productive with clear or yellowish sputum. Fever: Usually mild or absent; high fever suggests pneumonia. Fatigue and Body Aches: Common but generally mild. Wheezing and Shortness of Breath: May be present, especially in patients with underlying asthma. Chronic Bronchitis: Cough: Persistent, productive cough with sputum, often for at least three months in two consecutive years. Dyspnea: Shortness of breath, especially on exertion. Wheezing: Common, especially in the morning. Chest Pain: May occur due to prolonged coughing. Fatigue and Malaise: Persistent, often due to chronic hypoxemia.

Figure 6: Examples of the guidelines.

## B.3 Derivation of the Guideline-Enhanced Contrastive Loss

Here, we derive the form of the most basic cross-entropy loss to the form in Eq. (4). We demonstrate the equivalence between the sigmoid-based cross-entropy formulation and the logit-style implementation of our supervised contrastive loss.

The equation in cross-entropy form with explicit sigmoid terms is defined as:

<!-- formula-not-decoded -->

where ϕ ij is short for ϕ ( I i , S ij ) = I ⊤ i S ij /τ , the similarity score between the modality-specific feature and the guideline feature. σ ( · ) is the Sigmoid function, the logit of the similarity score is σ ( ϕ ij ) = 1 1+ e -ϕ ij . The difference between Eq. (8) and the standard cross-entropy loss is that we use the similarity score as logits, and we add the normalization coefficient 1 | P i | to balance the gradient contribution of each positive label in multi label scenarios. The following is a step-by-step derivation. First, substitute the sigmoid function into Eq. (8):

## Prompt for LLM Refinement:

Your task is to help filter and summarize the relevant information from multiple sets of retrieved external knowledge (from 4 different sources) to support my multi-modal disease classification model in diagnosing the specific disease associated with the provided ICD disease description. For the given disease, identify the critical symptoms, lab indicators, and radiological features that are most strongly associated with the disease diagnosis. Discard any information that is unrelated or irrelevant to disease classification. You do not need to focus on treatment options, but instead, concentrate on factors that would help in diagnosing the disease. You will be provided with a set of documents (containing topk retrieved documents, each approximately 2000 characters), containing a range of medical information. Your job is to:

Review the retrieved documents and determine which information is directly relevant to diagnosing the disease, based on its ICD code and description. Eliminate any information that is unrelated to diagnosis or classification, such as treatment options, management strategies, or irrelevant clinical details. Focus on identifying key diagnostic features, including symptoms, laboratory test results, and imaging findings that help confirm the presence of the disease. Evaluate the importance of each retrieved documents: Some documents may provide more critical or reliable information than others. Prioritize information that is most relevant and useful for diagnosing the disease, even if it means excluding less relevant details from certain documents. Summarize the most relevant content into a single cohesive summary of approximately 2000 characters (or 500 words). The summary should include the essential diagnostic criteria, including lab values, clinical features, and radiological findings.

The summary should emphasize:

Abrief explanation or description of the disease, including any variations or related conditions under the same ICD code. Important lab tests and values (e.g., white blood cell count, Creactive protein). Key radiological or clinical findings associated with the disease's presence (e.g., lung opacity, pleural effusion). Any diagnostic symptoms or relevant clinical features. Discard information that is not useful for diagnosing the disease.

Please ensure that the summary is concise and directly related to diagnosis, omitting irrelevant details.

Disease description (ICD name): {disease\_icd\_name} Retrieved passages: {retrieve\_passages\_str}

Figure 7: Prompt for LLM refinement.

<!-- formula-not-decoded -->

The final equation is:

<!-- formula-not-decoded -->

We use this form directly in the main body Section 3.2.2.

Figure 8: Label distribution of MIMIC-ICD53. The X-axis represents the formal disease names under the ICD-10 standard.

<!-- image -->

## C Experimental Details

## C.1 Details of Datasets

## C.1.1 Construction Process of MIMIC-ICD53

First, we merged and aligned the ED, HOSP, and ICU parts of MIMIC-IV [29]. Subsequently, we aligned the processed MIMIC-IV dataset with the MIMIC-CXR-JPG dataset [28]. We utilized patient\_id and study\_id to align the datasets at the patient level. Given that temporal patient dynamics were not considered, we selected the most recent radiological examination for each patient, including the associated images and reports. For temporal alignment, we extracted EHR data and ICD disease codes from MIMIC-IV within a three-day window following the radiological examination. After excluding instances with missing modalities or labels, we obtained a final sample size of 51830. For disease labeling, we standardized the granularity of diagnoses according to the ICD-10 classification, using the format Xab (where X is a letter and ab are digits), resulting in over 2000 unique labels. To refine and further clean the dataset, we consulted LLMs and then physicians to identify and select 53 critical disease categories that were related to thoracic and cardiovascular conditions or could be identified using laboratory indicators in the EHR. For the numerical indicators in the EHR, we quantized each indicator on a 0-10 integer scale based on its corresponding normal range limits. A value of 4-7 is considered normal, 0-3 indicates too low, and 8-10 signifies too high. The label distribution of MIMIC-ICD53 is shown in Figure 8. The training set and test set are randomly divided in a ratio of 4:1. The final processed dataset, termed MIMIC-ICD53, will be made publicly available on PhysioNet after publication. (The MIMIC dataset requires that all datasets developed based on MIMIC can only be released on PhysioNet.)

## C.1.2 Preprocess of NACC

The National Alzheimer's Coordinating Center (NACC) dataset [5] is a large, standardized resource comprising clinical and neuropathological data collected from individuals assessed at Alzheimer's Disease Research Centers (ADRCs) across US, which consists various neurodegenerative diseases, like Alzheimer's disease, Parkinson's disease, vascular dementia, and other forms of cognitive impairment. We follow [63] to organize the dataset, resulting in 11 labels including: "Normal cognition" (NC), "Mild cognitive impairment" (MCI), "Dementia" (DE), "Alzheimer's disease" (AD), "Vascular dementia, vascular brain injury and vascular dementia" (VD), "Lewy body dementia, including dementia with Lewy bodies and Parkinson's disease dementia" (LBD), "Psychiatric conditions including schizophrenia, depression, bipolar disorder, anxiety and posttraumatic stress disorder"

(PSY), "Frontotemporal lobar degeneration and its variants, including primary progressive aphasia, corticobasal degeneration and progressive supranuclear palsy, and with or without amyotrophic lateral sclerosis" (FTD), "Systemic and environmental factors including infectious diseases (HIV included), metabolic, substance abuse / alcohol, medications, systemic disease and delirium" (SEF), "Other dementia conditions, including neoplasms, Down syndrome, multiple systems atrophy, Huntington's disease and seizures" (ODE), and "Moderate/severe traumatic brain injury, repetitive head injury and chronic traumatic encephalopathy" (TBI). The label distribution of NACC is shown in Figure 9. Given that NACC contains over 800 distinct EHR variables, selecting the most relevant features for analysis was a critical step in our study. To ensure both scientific validity and clinical interpretability, we first utilized LLM for cleaning and double-checked with several physicians.

Figure 9: Label distribution of NACC.

<!-- image -->

Finally, we distilled the original set down to a final list of 36 key EHR variables. The selected variables include height, weight, body mass index (BMI), systolic blood pressure, diastolic blood pressure, cortical atrophy (Alzheimer's disease marker), small vessel disease (vascular dementia related), left motor cortex vascular lesion, right motor cortex vascular lesion, normal pressure hydrocephalus gait, parkinsonian signs (tremor/rigidity), bradykinesia (Parkinsonian symptom), neck rigidity (dystonia), gait disturbance, history of hypertension, history of diabetes, history of cardiovascular disease, history of stroke, history of Parkinson's disease, sleep apnea, REM sleep behavior disorder (RBD), his- tory of traumatic brain injury (TBI), delusions, hallucinations, depressive symptoms, agitation or aggression, anti-dementia medication (e.g.), Parkinson's disease medication (e.g.), anticoagulant use (stroke prevention), antidepressant medication, postural instability (Parkinson's or Lewy body dementia), APOE ϵ 4 allele (Alzheimer's disease risk), hypercholesterolemia (vascular risk), amyotrophic lateral sclerosis (ALS) signs, left visual cortex functional impairment, and right visual cortex functional impairment. The final processed dataset is randomly divided in a ratio of 4:1 for training and testing.

## C.1.3 Details of Harvard-FairVLMed and SkinCAP

The Harvard-FairVLMed dataset [41], sourced from the Department of Ophthalmology at Harvard Medical School, contains 10,000 multimodal samples (7,000 train, 1,000 val, 2,000 test) with paired clinical notes, diagnostic labels, and detailed demographic attributes (race, gender, ethnicity, language). The dataset is publicly available under the CC BY-NC-ND 4.0 license at Github 4 . We directly used the original dataset.

SkinCAP is a multimodal dermatology dataset containing 4,000 expert-annotated skin disease images with rich natural language descriptions [76]. The dataset combines cases from diverse dermatology image datasets, all annotated by board-certified dermatologists to ensure clinical accuracy. It is publicly available under an open license at HuggingFace 5 . To address class imbalance, we removed tail categories with too few positive samples, resulting in a filtered dataset of 2,526 samples with 50 disease labels. The final dataset was partitioned into a 4:1 train-test split.

4 https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP

5 https://huggingface.co/datasets/joshuachou/SkinCAP

Table 6: Benchmarking performance of each modality on MIMIC-ICD53 dataset.

| Modality    | Model/Method    |    F1 |   Precision |   Recall |   AUC |   mAP |   Acc |   Acc-S |   Avg |
|-------------|-----------------|-------|-------------|----------|-------|-------|-------|---------|-------|
| Image       | ResNet18        | 14.39 |       10.49 |    32.94 | 71.47 |  9.66 | 87.6  |   21.94 | 35.5  |
| Image       | ResNet50        | 14.49 |       10.99 |    33.09 | 72.35 |  9.92 | 87.68 |   22.41 | 35.85 |
| Image       | ViT-Base        | 15.49 |       11.59 |    34.9  | 73.93 | 10.81 | 88    |   20.66 | 36.75 |
| Image       | Swin-Base       | 16.93 |       13.61 |    33.97 | 75.12 | 12.11 | 89.03 |   18.94 | 37.1  |
| Image       | UniChest        | 19.3  |       16.68 |    31.45 | 78.16 | 14.18 | 92    |   21.63 | 39.06 |
| Image       | RadDino         | 17.26 |       13.79 |    29.9  | 75.34 | 11.75 | 90.89 |   13.87 | 36.11 |
| EHR         | MLP             | 19.14 |       15.15 |    38.52 | 72.57 | 14.81 | 89.31 |    0    | 35.64 |
| EHR         | TabFPN          |  9.2  |        5.85 |    53.32 | 52.7  |  5.36 | 59.24 |    0    | 26.52 |
| EHR         | ClinicalBERT    | 14.96 |        9.99 |    50.41 | 79.8  |  9.65 | 84.16 |   18.03 | 38.14 |
| EHR         | BioClinicalBERT | 11.9  |        8.57 |    42.57 | 72.52 |  7.21 | 80.51 |   18.98 | 34.61 |
| EHR         | PubMedBERT      | 10.18 |        6.31 |    78.48 | 67.16 |  5.87 | 53.35 |   19.88 | 34.46 |
| EHR         | LLaMa-3.2-1B    | 22    |       18.94 |    35.86 | 84.47 | 16.52 | 91.92 |   15.85 | 40.79 |
| EHR         | LLaMa-3.1-8B    | 22.53 |       18.55 |    38.84 | 84.98 | 16.66 | 92.28 |   17.65 | 41.64 |
| EHR         | MMedS-8B        | 21.88 |       18    |    34.47 | 84.72 | 15.99 | 92.47 |   19.98 | 41.07 |
| Report      | ClinicalBERT    | 24.29 |       21.06 |    39.95 | 83.23 | 18.45 | 91.52 |   22.66 | 43.02 |
| Report      | BioClinicalBERT | 29.12 |       25.13 |    45.18 | 84.21 | 23.03 | 91.59 |   17.29 | 45.08 |
| Report      | PubMedBERT      | 15.46 |       12.06 |    36.86 | 73.8  |  9.65 | 87.07 |   26.59 | 37.36 |
| Report      | LLaMa-3.2-1B    | 30.86 |       28.91 |    42.67 | 86.81 | 25.46 | 93.83 |   22.53 | 47.3  |
| Report      | LLaMa-3.1-8B    | 32.53 |       31.84 |    42.03 | 87.54 | 27.3  | 94.14 |   25.29 | 48.67 |
| Report      | MMedS-8B        | 32.39 |       29.52 |    43.73 | 86.78 | 27.08 | 94.17 |   22.97 | 48.09 |
| Report +EHR | ClinicalBERT    | 27.13 |       23.43 |    46.32 | 89.22 | 22.2  | 92.47 |   31.32 | 47.44 |
| Report +EHR | BioClinicalBERT | 28.18 |       23.62 |    46.96 | 89.28 | 22.28 | 92.26 |   29.26 | 47.41 |
| Report +EHR | PubMedBERT      | 10.1  |        6.02 |    82.24 | 66.5  |  5.71 | 50.46 |    0.11 | 31.59 |
| Report +EHR | LLaMa-3.2-1B    | 33.68 |       30.57 |    46.89 | 91.9  | 29.65 | 94.75 |   36.5  | 51.99 |
| Report +EHR | LLaMa-3.1-8B    | 32.84 |       28.69 |    47.94 | 90.91 | 28.2  | 94.09 |   32.35 | 50.72 |
| Report +EHR | MMedS-8B        | 33.27 |       31.8  |    47.58 | 91.51 | 28.96 | 94.44 |   32.87 | 51.49 |

## C.2 Benchmarking MIMIC-ICD53

To further evaluate the quality of our constructed dataset MIMIC-ICD53, we employed various unimodal methods to train and test its performance. For the visual modality, we selected ResNet-18, ResNet-50 [22], ViT [15], and Swin Transformer [39], along with two SOTA CXR-specific pretrained models, UniChest [14] and RadDino [45] as the baselines. Based on both computational efficiency and data leakage prevention considerations, we ultimately designated ResNet-50 as the standard visual backbone for the main experiments. ViT is also investigated in ablation studies in Section C.6.

For electronic health record (EHR) data, we first leveraged its inherent tabular structure by treating each EHR attribute as an input dimension, with corresponding numerical values assigned to their respective dimensions. We experimented with MLP and a SOTA tabular data process method TabFPN [24], but both exhibited suboptimal performance. Consequently, we reformatted the EHR data into natural language text using the following template: "Laboratory values within the 4-7 range indicate normal levels, values 0-3 suggest clinically low levels, and values 8-10 denote elevated levels. The current panel includes [ATTRIBUTE] with the discretized value of [VALUE]..." . We then evaluated the reformatted EHR data using classic backbone ClinicalBERT [55], BioClinicalBERT [2], and PubMedBERT [18]. We also included natural and medical LLMs LLaMa [54] and MMedS [58]. Specifically, we replace the last layer of LLMs with a classification head to adapt to the text classification task. Due to the high consumption of computing resources, we only use LoRA [25] to fine-tune the LLM-based models. We further conducted experiments using the same text encoders for the report modality alone and reports combined with EHR data. As shown in Table 6, unimodal performance analysis reveals that the report modality achieves the highest diagnostic results, followed by the EHR modality. Combining the two modalities in text form (Report+EHR) can bring significant performance gains. For the visual modality, the performance gap between different backbone architectures is relatively small. While for the text-based modality, LLM-based

Table 7: Performance with variance on four datasets of different anatomies. All results are calculated over 5 independent runs.

| Dataset             | Method                                | F1                                                                         | Precision                                                                  | Recall                                                                     | AUC                                                                        | mAP                                                                        | Acc                                                                        | Acc-S                                                                      | Avg                                                                        |
|---------------------|---------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|
| MIMIC-ICD53 (Chest) | MedFuse BiomedCLIP KAD DrFuse HEALNet | 34.46 ± 0.0077 32.99 ± 0.0058 36.32 ± 0.0107 34.10 ± 0.0067 35.42 ± 0.0075 | 31.36 ± 0.0082 29.56 ± 0.0073 33.80 ± 0.0125 33.70 ± 0.0067 32.76 ± 0.0079 | 45.04 ± 0.0004 45.04 ± 0.0007 48.33 ± 0.0019 45.34 ± 0.0244 47.95 ± 0.0016 | 90.85 ± 0.0168 88.71 ± 0.0061 91.95 ± 0.0165 89.50 ± 0.0287 88.80 ± 0.0137 | 31.77 ± 0.0084 29.91 ± 0.0061 33.54 ± 0.0111 31.19 ± 0.0073 31.97 ± 0.0081 | 95.34 ± 0.0127 94.72 ± 0.0032 95.12 ± 0.0049 94.68 ± 0.0639 94.90 ± 0.0204 | 41.44 ± 0.0239 39.83 ± 0.0087 40.27 ± 0.0254 38.25 ± 0.0239 40.10 ± 0.0209 | 52.89 ± 0.0085 51.54 ± 0.0048 54.19 ± 0.0104 52.39 ± 0.0086 53.13 ± 0.0076 |
| MIMIC-ICD53 (Chest) | RAD                                   | 39.71 ± 0.0101                                                             | 39.07 ± 0.0099                                                             | 54.74 ± 0.0016                                                             | 93.00 ± 0.0103                                                             | 36.74 ± 0.0116                                                             | 95.40 ± 0.0050                                                             | 42.33 ± 0.0228                                                             | 57.28 ± 0.0089                                                             |
| FairVLMed (Eye)     | MedFuse BiomedCLIP KAD DrFuse HEALNet | 81.33 ± 0.0010 81.27 ± 0.0014 81.18 ± 0.0028 81.69 ± 0.0028 81.80 ± 0.0011 | 76.13 ± 0.0021 72.87 ± 0.0034 73.92 ± 0.0080 73.72 ± 0.0090 75.22 ± 0.0028 | 87.29 ± 0.0003 91.88 ± 0.0005 90.03 ± 0.0010 91.59 ± 0.0022 89.64 ± 0.0001 | 87.99 ± 0.0034 87.69 ± 0.0041 88.62 ± 0.0137 89.33 ± 0.0217 89.60 ± 0.0030 | 88.76 ± 0.0049 87.62 ± 0.0038 88.88 ± 0.0158 90.38 ± 0.0204 90.45 ± 0.0041 | 79.50 ± 0.0024 78.35 ± 0.0044 78.65 ± 0.0101 79.00 ± 0.0121 79.60 ± 0.0022 | 79.50 ± 0.0024 78.35 ± 0.0044 78.65 ± 0.0101 79.00 ± 0.0121 79.60 ± 0.0022 | 83.50 ± 0.0020 83.28 ± 0.0024 83.55 ± 0.0064 84.29 ± 0.0076 84.39 ± 0.0019 |
| FairVLMed (Eye)     | RAD                                   | 84.30 ± 0.0028                                                             | 77.52 ± 0.0070                                                             | 92.38 ± 0.0005                                                             | 91.32 ± 0.0126                                                             | 91.88 ± 0.0144                                                             | 82.40 ± 0.0080                                                             | 82.40 ± 0.0080                                                             | 86.63 ± 0.0060                                                             |
| SkinCAP (Skin)      | MedFuse BiomedCLIP KAD DrFuse HEALNet | 79.25 ± 0.0418 81.49 ± 0.1073 82.06 ± 0.1025 81.18 ± 0.0389 82.20 ± 0.0890 | 85.96 ± 0.0538 87.13 ± 0.1228 86.79 ± 0.1290 85.70 ± 0.0470 88.69 ± 0.1130 | 77.99 ± 0.0036 81.41 ± 0.0091 81.27 ± 0.0147 79.64 ± 0.0040 81.18 ± 0.0186 | 96.50 ± 0.0194 97.22 ± 0.0351 97.80 ± 0.0454 94.92 ± 0.0185 92.68 ± 0.0225 | 73.61 ± 0.0363 79.22 ± 0.1114 80.40 ± 0.1066 76.42 ± 0.0365 77.97 ± 0.0925 | 99.34 ± 0.0166 99.11 ± 0.0282 99.25 ± 0.0244 99.29 ± 0.0158 99.37 ± 0.0176 | 74.36 ± 0.0148 74.36 ± 0.1184 75.46 ± 0.1098 77.66 ± 0.0122 78.39 ± 0.0480 | 83.86 ± 0.0223 85.71 ± 0.0646 86.15 ± 0.0654 84.97 ± 0.0208 85.79 ± 0.0475 |
| SkinCAP (Skin)      | RAD                                   | 85.48 ± 0.0678                                                             | 89.48 ± 0.0750                                                             | 83.23 ± 0.0136                                                             | 97.97 ± 0.0356                                                             | 83.55 ± 0.0639                                                             | 99.48 ± 0.0159                                                             | 81.32 ± 0.0474                                                             | 88.64 ± 0.0407                                                             |
| NACC (Brain)        | MedFuse BiomedCLIP KAD DrFuse HEALNet | 31.53 ± 0.0005 34.36 ± 0.0013 35.09 ± 0.0024 34.11 ± 0.0030 35.91 ± 0.0008 | 25.59 ± 0.0001 29.02 ± 0.0008 29.68 ± 0.0039 27.86 ± 0.0032 28.92 ± 0.0004 | 68.36 ± 0.0051 66.95 ± 0.0002 64.49 ± 0.0008 68.96 ± 0.0085 67.33 ± 0.0049 | 85.50 ± 0.0038 84.00 ± 0.0043 85.88 ± 0.0052 82.88 ± 0.0070 85.04 ± 0.0037 | 24.49 ± 0.0004 26.03 ± 0.0008 27.73 ± 0.0026 27.88 ± 0.0024 26.13 ± 0.0006 | 87.44 ± 0.0110 88.80 ± 0.0010 89.69 ± 0.0013 87.99 ± 0.0191 89.55 ± 0.0090 | 58.45 ± 0.0196 58.21 ± 0.0004 57.86 ± 0.0071 51.31 ± 0.0045 56.79 ± 0.0001 | 54.48 ± 0.0011 55.34 ± 0.0008 55.77 ± 0.0028 54.43 ± 0.0025 55.67 ± 0.0008 |
| NACC (Brain)        | RAD                                   | 37.65 ± 0.0015                                                             | 36.24 ± 0.0049                                                             | 65.78 ± 0.0003                                                             | 87.11 ± 0.0019                                                             | 30.03 ± 0.0023                                                             | 90.36 ± 0.0010                                                             | 59.64 ± 0.0078                                                             | 58.12 ± 0.0020                                                             |

models generally outperform BERT-based models. Among the BERT-based models, ClinicalBERT consistently achieves the best performance. Considering model size and practicality, we selected ClinicalBERT as the default text encoder in our RAD framework.

## C.3 The Variance of Baselines

Due to space limitations, we do not show the variance of the baselines in Table 1. Here we add the variance of all baselines in Table 7. It can be observed that the overall variance of SkinCAP is the largest among all datasets. Meanwhile, there is no significant gap between the variance of different methods, all methods exhibit stable performance across the four datasets.

Figure 10: Detailed AUC and F1 for each class in MIMIC-ICD53. The y-axis is the disease name. The numbers in brackets represent the number of samples with this disease.

<!-- image -->

## C.4 Label-wise Analysis of MIMIC-ICD53

In addition to evaluating the overall performance of RAD in Section 4.2, we also investigated its comprehensive performance across all categories on MIMIC-ICD53. As illustrated in Figure 10, our method achieved the highest scores in 41 out of 53 categories across both AUC and F1 metrics. Furthermore, in the long-tail categories (classes with fewer than 100 positive samples), our method outperformed the previous SOTA by 1.60% in AUC and by 4.44% in F1. Importantly, the performance gains in these long-tail categories exceeded the average improvements observed across all categories, underscoring the robustness and practical utility of RAD under real-world clinical settings.

## C.5 Interpretability Cases

In this subsection, we further explore the textual interpretability of RAD by presenting additional visualization cases. In addition, the full names of the abbreviations in Table 4 are given here. In Table 4, the "PC" is short for Platelet Count, "Bilirubin" is Serum Bilirubin, "ALT" is Alanine

<!-- image -->

w/o RAD

RAD

Figure 11: Visualization of model attention to textual content. Both font size and color intensity reflect attention magnitude, with red highlighting disease-critical indicators mentioned in the guideline.

Aminotransferase, "IBC" is Iron-Binding Capacity, "WBC" is White Blood Cell Count, and "AST" is Aspartate Aminotransferase. Figure 11 presents longer and clearer cases of interpretability on the textual data. The third row is the complete content of Figure 1, and the other rows are other cases

Table 8: Performance across different combinations of encoder backbones on MIMIC-ICD53. Subscript with arrows represents the absolute improvement. Our method is highlighted with shading.

| Backbone                | Method      | F1                   | Precision            | Recall               | AUC                  | mAP                  | Acc                  | Acc-S                | Avg                  |
|-------------------------|-------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| ResNet+ ClinicalBERT    | w/o RAD RAD | 34.91 39.71 4 . 80 ↑ | 31.01 39.07 8 . 06 ↑ | 50.91 54.74 3 . 83 ↑ | 91.27 93.00 1 . 73 ↑ | 32.24 36.74 4 . 50 ↑ | 94.50 95.40 0 . 90 ↑ | 38.63 42.33 3 . 70 ↑ | 53.35 57.28 3 . 93 ↑ |
| ViT+ ClinicalBERT       | w/o RAD RAD | 37.22 41.21 3 . 99 ↑ | 35.77 41.14 5 . 37 ↑ | 45.64 51.89 6 . 25 ↑ | 91.01 92.70 1 . 69 ↑ | 34.05 37.24 3 . 19 ↑ | 95.63 95.78 0 . 15 ↑ | 41.04 41.97 0 . 93 ↑ | 54.34 57.42 3 . 08 ↑ |
| ResNet+ BioClinicalBERT | w/o RAD RAD | 36.71 39.95 3 . 24 ↑ | 33.76 39.90 6 . 14 ↑ | 49.99 51.72 1 . 73 ↑ | 92.03 92.59 0 . 56 ↑ | 34.31 36.34 2 . 03 ↑ | 95.02 95.89 0 . 87 ↑ | 38.86 42.29 3 . 43 ↑ | 54.38 56.95 2 . 57 ↑ |
| ViT+ BioClinicalBERT    | w/o RAD RAD | 36.32 40.00 3 . 68 ↑ | 34.35 39.58 5 . 23 ↑ | 48.99 50.70 1 . 71 ↑ | 92.08 92.25 0 . 17 ↑ | 33.39 36.38 2 . 99 ↑ | 95.18 96.01 0 . 83 ↑ | 39.77 42.52 2 . 75 ↑ | 54.30 56.78 2 . 48 ↑ |

Table 9: Ablation on LLM refinement of RAD on the MIMIC-ICD53.

| LLM-refine   |    F1 |   Precision |   Recall |   AUC |   mAP |   Acc |   Acc-S |   Avg |
|--------------|-------|-------------|----------|-------|-------|-------|---------|-------|
| ×            | 38.73 |       36.94 |    53.24 | 92.99 | 36.56 | 95.34 |   40.43 | 56.32 |
| ✓            | 39.71 |       39.07 |    54.74 | 93    | 36.74 | 95.4  |   42.33 | 57.28 |

Table 10: Ablation on retrieval knowledge sources. "Ours" is the default setting with four knowledge sources. "+ Google Search" means adding a new source based on "Ours". "- Random Drop" means randomly removing one knowledge source for each guideline.

|               | Source          |    F1 |   Precision |   Recall |   AUC |   mAP |   Acc |   Acc-S |   Avg |
|---------------|-----------------|-------|-------------|----------|-------|-------|-------|---------|-------|
| Single Source | Wiki            | 39.77 |       39.11 |    47.05 | 93.14 | 36.93 | 96.02 |   41.67 | 56.24 |
| Single Source | Research        | 38.54 |       36.35 |    51.41 | 93.01 | 36.26 | 95.47 |   40.69 | 55.96 |
| Single Source | Guideline       | 39.79 |       39.17 |    50.32 | 93.03 | 37.12 | 96.02 |   41.42 | 56.7  |
| Single Source | Book            | 39.49 |       39.14 |    47.65 | 93.11 | 36.84 | 96.2  |   42.04 | 56.35 |
|               | - Random Drop   | 40.18 |       40.15 |    49.34 | 93.05 | 37.35 | 96.24 |   43.32 | 57.09 |
|               | Ours            | 39.71 |       39.07 |    54.74 | 93    | 36.74 | 95.4  |   42.33 | 57.28 |
|               | + Google Search | 40.56 |       40.01 |    50.56 | 92.84 | 36.97 | 96.15 |   42.89 | 57.14 |

with different indicators. It can be observed that RAD enables the model to dynamically focus on indicators valuable for the current diagnostic goal based on the retrieved guidelines.

## C.6 Ablation Study

In this part, we conduct a comprehensive ablation study to systematically evaluate the impact of architectural backbones, key components, and hyperparameter configurations in RAD.

Ablation on different backbones. In Section 4.4, we demonstrated the impact of RAD on model performance when replacing different modality backbones, as reflected in the average metrics, AUC, F1, and mAP. To provide a more comprehensive evaluation, we have included additional metrics in Table 8, such as Precision, Recall, Accuracy, and Acc-S, which collectively illustrate the holistic enhancement of the model in diagnostic tasks.

Ablation on LLM refinement of the retrieved knowledge. To assess the necessity of LLM refinement in Section 3.2.1, we further conducted an ablation study by comparing RAD with and without this step. Specifically, we constructed baseline guidelines through direct concatenation of topk retrieved documents and evaluated the performance on MIMIC-ICD53. The results in Table 9 demonstrate that all metrics have decreased after removing the LLM filtering step, underscoring the importance of regularizing the retrieved text. The LLM refinement not only performs semantic filtering to eliminate irrelevant contexts but also standardizes heterogeneous medical knowledge into actionable diagnostic guidelines-a critical enabler for effective downstream knowledge infusion.

Ablation on Knowledge Sources. To investigate the effect of modifying the knowledge base on model performance, we compare each knowledge source's individual performance, as well as the performance of adding or removing sources based on our default setting. As presented in Table 10, clinical guidelines provide the most valuable knowledge, as they directly encode established

Figure 12: Analysis of hyper-parameters on MIMIC-ICD53.

<!-- image -->

diagnostic criteria, key indicators, and decision pathways specifically designed for clinical practice. Research papers show the lowest contribution, as they often focus on novel discoveries, experimental treatments, or specialized cases rather than established diagnostic standards. For well-established diseases, diagnostic criteria have become a consensus, making cutting-edge research less useful. When applying multiple knowledge sources, the performance of RAD remains stable across different source counts (±0.2 Avg), demonstrating RAD's robustness to knowledge base modifications.

## C.7 Cost Analysis.

To evaluate the practical feasibility of our framework, we analyze the additional cost of RAD brought by the guideline acquisition process. Since we only perform retrieval at the label level, which avoids the prohibitive cost of sample-wise retrieval. The retrieval process incurs negligible computational overhead. The additional cost primarily occurs during the LLM refinement phase, where the retrieved documents are processed by LLMs for each label of the dataset. When using Qwen2.5-72B model, the average processing time is 33.83s per label. The total preprocessing time for guideline retrieval and refinement on MIMIC-ICD53 is around 31 minutes. The cost can be further reduced using smaller LLMs. When expanding to new datasets, the linear growth of retrieval cost O ( N disease ) ensures efficient scalability, as it grows significantly slower than patient samples O ( N sample ) in real-world scenarios. Furthermore, the retrieval and refinement steps are executed once per dataset during preprocessing, eliminating runtime delays during clinical deployment. In general, RAD achieves knowledge infusion with minimal practical overhead.

## C.8 Hyper-parameter analysis

To evaluate the impact of hyperparameters in RAD, we conduct experimental analysis on the three key hyperparameters α , β , and topk . The hyperparameter α determines the weight of the guidelineenhanced contrastive learning for visual and text features. And β determines the weight of binary cross-entropy loss and the guideline-enhanced contrastive loss. Topk controls the number of retrieved documents for each disease (label). Figure 12 presents the performance trends as these parameters vary. As β decreases, the model performance initially improves before declining. This pattern arises because an excessively high weight over-prioritizes the auxiliary loss, disrupting the optimization of the primary classification loss. On the contrary, a very low weight also leads to performance degradation, underscoring the utility of the guideline in refining multi-modal feature representations. α exhibits a similar pattern. The optimal values for α and β are 1 e -2 and 1 e -1 ,

respectively. Regarding the topk hyperparameter, the model achieves worst performance at k = 1 , with gradual improvement as k increases. However, performance plateaus after reaching a threshold ( k = 10 here). When retrieving too few documents, limited informative content leads to suboptimal results. Conversely, retaining excessive documents beyond the threshold primarily introduces noisy knowledge, as core disease-related information has already been captured within the top-ranked documents.