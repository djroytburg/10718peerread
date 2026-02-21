## Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders

## Mengyu Ye

Tohoku University

ye.mengyu.s1@dc.tohoku.ac.jp

Tatsuro Inaba ∗ MBZUAI

tatsuro.inaba@mbzuai.ac.ae

## Jun Suzuki

Tohoku University &amp; RIKEN jun.suzuki@tohoku.ac.jp

## Tatsuki Kuribayashi

MBZUAI

tatsuki.kuribayashi@mbzuai.ac.ae

## Abstract

Recent interpretability work on large language models (LLMs) has been increasingly dominated by a feature-discovery approach with the help of proxy modules. Then, the quality of features learned by, e.g., sparse auto-encoders (SAEs), is evaluated. This paradigm naturally raises a critical question: do such learned features have better properties than those already represented within the original model parameters, and unfortunately, only a few studies have made such comparisons systematically so far. In this work, we revisit the interpretability of feature vectors stored in feed-forward (FF) layers, given the perspective of FF as key-value memories, with modern interpretability benchmarks. Our extensive evaluation revealed that SAE and FFs exhibits a similar range of interpretability, although SAEs displayed an observable but minimal improvement in some aspects. Furthermore, in certain aspects, surprisingly, even vanilla FFs yielded better interpretability than the SAEs, and features discovered in SAEs and FFs diverged. These bring questions about the advantage of SAEs from both perspectives of feature quality and faithfulness, compared to directly interpreting FF feature vectors, and FF key-value parameters serve as a strong baseline in modern interpretability research 2 .

## 1 Introduction

Transformer-based language models (LMs) have exhibited outstanding performance on a wide variety of tasks [10, 35, 1, 44, 45], whereas their underlying mechanisms remain opaque [47, 37, 50, 17, 38, 18, 34, 28, 31]. This issue has been tackled in the interpretability field, and in earlier days, the field has typically adopted a top-down approach, where, given candidate features or algorithms, e.g., syntactic structure, it has been inspected where in the original model those are encoded. Nowadays, as a variety of capabilities emerge in larger LMs, the question tends to be more on the bottom-up , feature-discovery side: what kind of features are encoded in the model?; and how can we discover and control them? This feature-discovery age has brought two trends to the interpretability community simultaneously: (i) training an external proxy module dedicated to this purpose, namely, sparse autoencoder (SAEs), to decompose neuron activations into simpler basic features [53, 8, 24, 30, 21, 16] ( proxy-based analysis ), and (ii) developing new comprehensive interpretability benchmarks [36, 27] to test the quality of discovered features.

∗ Work done at Tohoku University.

2 Project page: https://muyo8692.com/projects/ff-kv-sae

This paper explores one overlooked question in the field, to what extent a proxy-based, artificial decomposition of neuron activations empirically benefits the model interpretation. In other words, feed-forward (FF) layers naturally implement the decomposition of neural activation into a set of feature vectors, through the lens of FF as key-value memories [17]( FF-KV analysis ), why not first evaluate such organic features in FFs with the newly developed interpretability benchmarks? Proxybased and FF-KV analysis have complementary advantages, and thus, there is no immediate reason to dismiss the latter. For example, while some proxy-based methods have a theoretical motivation to handle superposition, they also have limitations that FF-KV analysis can automatically bypass: proxy modules can additionally expose biases to the interpretation, e.g., specific features are repeatedly found [8, 49, 11]; the external proxy hallucinates features [22]; and additional computation costs are needed to interpret the model. Furthermore, the FF activations are reported to be naturally sparse even without any regularization [29]. Thus, if FF-KV and SAE analyses yield comparable results, there are several advantages (more simply put, from Occam's Razor principle) to adopting the former FF-KV analyses.

To gauge the (dis)similarities between FF-KVs' and SAEs' interpretability level, we perform both automatic evaluation and manual feature analyses. Automatic evaluation with SAEBENCH demonstrates surprising similarities between the two approaches. The evaluation scores fell into a similar range in all eight metrics in SAEBENCH, and the inter-metrics tendencies are also paralleled, e.g., causal intervention scores are poorer than feature disentanglement scores in both methods. One can even observe some advantages of FF-KVs; for example, features in the original FFs tend to avoid feature overlapping, resulting in better absorption scores [11] (i.e., less redundancy) than those in SAEs. These comparable quality is further supported by human manual evaluation of feature qualities. Conceptual features can be found with almost equal ease from both FF-KV features and SAEs. These tentatively conclude that features from FF-KVs and SAEs serve a quite similar level of interpretability from both quantitative and qualitative perspectives.

In our analysis, we further investigate the faithfulness of proxy-discovered features, considering FF-KV features as gold, how large is the overlap between the feature sets of the original FF-KV module and that of the proxy module? We analyzed such an overlap with Transcoder (TC), the closest counterpart to FF-KV, as a proxy model, and revealed that the majority of TC features do not have similar counterparts in the original FF module. This aligns with the existing report that SAEs can interpret even random Transformers [22], and perhaps the proxy module hallucinates new features rather than translating the workings of the original FF module, encouraging further research on the faithfulness of the learned features, with FF-KV features as grounding points. To sum up, our study reveals that proxy-based methods such as SAEs empirically offer very limited advantage over the direct analysis of FFs (i.e., key-value memories). That is, the theoretical advantage of SAEs is not observed empirically, at least through the lens of the current evaluation scheme, and encourages the inclusion of FF-KV features as a strong baseline when assessing feature-discovery methods in the interpretability field.

## 2 Background

## 2.1 Related Work

Dictionary Learning and LLMs Interpretation. Dictionary learning has been proposed to address polysemanticity of the representation [3, 52, 42, 40, 14, 5, 20], and this has been applied to interpret the internal activations of LLMs, represented by sparse autoencoders (SAEs) [53, 8, 24, 30, 21, 16, 25]. Specifically, these introduce a proxy module to decompose and reconstruct a model's activation, and seek interpretable features in it. Apparently, promising results were observed in earlier days: the learned features are highly interpretable and can be directly used to steer the model's behavior [46]: modification on a feature will either eliminate the corresponding behavior, or enhance it.

Mixed Reports on SAE Features. Although the SAEs get increasing attention, concurrent works have brought skeptical views on their success. For example, SAE-based feature steering quality is inferior to simple baselines utilizing activations [51]; SAEs can learn meaningful features even from a randomly initialized Transformer [22]; and they exhibit no clear advantage in downstream tasks and sometimes underperform linear probes that use the model's raw activations [9, 26]. This study, at a high level, provides additional support for such criticisms of the general advantage of SAEs.

Interpretability of Feed-Forward (FF) Layers. There have been a fair number of studies to interpret the feed-forward (FF) layer in Transformers directly [13, 40, 2]. The closest work to ours is Geva et al. [17], where FFs can be viewed as key-value memories, and they are interpretable and useful to control the model output. Recent work also indicates that activations in FFs are already sparse [29], and their neurons can be manipulated [51]; these motivate our work to contextualize the bare FF interpretability with SAE works.

## 2.2 Sparse Autoencoder for Transformer Interpretability

Transformer. Transformer architecture is a stack of multiple modules, such as attention mechanisms, feed-forward (FF) layers, and normalization layers. There have recently been increasing endeavors to interpret, especially, neuron activations around FF layers, such as SAEs. Henceforth, vector denotes a row vector.

SAE. SAE decomposes and reconstructs the neuron activations, typically after the FF layer (residual stream). That is, let x FF out ∈ R d model be neuron activations after the FF layer, and d SAE denotes the dimension of SAE features. SAE decomposes the neuron activations x FF out and reconstructs it ˆ x FF out as follows:

<!-- formula-not-decoded -->

with W enc ∈ R d model × d SAE , W dec ∈ R d SAE × d model , b enc ∈ R d SAE , and b dec ∈ R d model in the SAE module. ReLU ( · ) : R d → R d denotes an element-wise ReLU activation. Each activation dimension is treated as a potentially interpretable feature, and the matrix maps each feature dimension to its feature vector in the representation space. This module is trained so that the activations are as sparse as possible with a sparsity loss to disentangle the potentially polysemantic input neurons.

Transcoder. Notably, as an alternative to SAEs and perhaps the closest attempt to this study, Transcoders have recently been proposed [12]. This approximates the original FF by training a sparse MLP as a proxy to predict FF output x FF out from FF input x FF in , and its internal activations ( ∈ R d TC ) are evaluated in the same way as the standard SAE. Still, their work [12] did not clearly evaluate how interpretable the original FF's internal activations are, and this work complements this overlooked question.

## 2.3 Feed-Forward Layer as Key-Value Memories

Feed-forward layers in Transformers once project the FF input x FF in ∈ R d model to d FF -dimensional representation ( d model &lt; d FF ) , applies an element-wise non-linear activation ϕ ( · ) : R d → R d , and projects it back, as follows:

<!-- formula-not-decoded -->

where W K ∈ R d model × d FF and W V ∈ R d FF × d model are learnable weight matrices, and b K ∈ R d FF , b V ∈ R d model are learnable biases. d FF is typically set as 4 d model . One interpretation of the FF layer is a knowledge retrieval module; that is, the module first creates keys (activations) from an input x FF in and then aggregates their associated values (feature vectors) . Existing studies have analyzed what kind of concept is stored in each feature vector of W V [: ,i ] and when they are activated by ϕ ( x FF in W K ) [ i ] [17].

## 3 Comparing FF-KVs with SAEs

The feed-forward key-value memory module (FF-KV) inherently performs the same operation as SAEs (although it is somewhat obvious, given that both adopt the MLP architectures): it first decomposes the neuron activations into feature vectors and then aggregates them. This naturally raises a question about how similar the decomposition naturally made by FF-KVs is to that learned by the proxy module, e.g., SAEs. We examine several variants of FF-KV-based feature discovery methods 3 .

3 See Appendix A for the details on the implementations

## 3.1 Methods

FF-KV. The vanilla FF key-value memories are evaluated with the SAE evaluation framework, treating the key activations as features and the value vectors as feature vectors.

TopK FF-KV. To encourage the alignment with SAE research, we also introduce sparsity to activations in FFs by applying a topk activation function to the key vector, although it has been reported that the vanilla FFs' activations are somewhat already sparse [29]. This keeps only the k neurons with the k largest activations in each inference, zeroing out the activation for the rest. We call this TopK FF-KV , defined as follows:

<!-- formula-not-decoded -->

Normalized FF-KV. The feature vectors of SAE are typically normalized, whereas those in FF are not. If a particular feature vector W V [ i, :] has a large norm, the magnitude of its corresponding activation may be underestimated. To handle this potential concern, we normalize each row of W V , and the discounted vector norm is weighted to activations. We refer to the method with this post-correction as Normalized (TopK) FF-KV :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, ˜ W V = diag( s ) -1 W V , where diag( · ) expands a vector R d to a diagonal matrix R d × d .

## 3.2 Inference and Feature Discovery

Once a method to obtain activations from the models is determined, one can get an activation history over a certain set of text. Here, we introduce several notations before going to the experiments.

Notations. Feature activations are analyzed through feeding specific texts to models, and the exact text contents will vary depending on evaluation metrics. Let us denote a set of input texts as S = [ s 1 , · · · , s n ] , where each text consists of multiple tokens s k = [ w ( k, 1) , w ( k, 2) , · · · , w ( k,m ) ] ∈ S , which are used to get feature activations. For brevity, we flatten and re-index the tokens as [ w 1 , w 2 , · · · , w l ] ; one can recover the original indices ( i, j ) indicating text id and token position via σ : N [1 ,l ] → N [1 ,n ] × N [1 ,m ] , e.g., σ (2) = (1 , 2) . For each token w t , we first collect feature activations a t ∈ R d coder with a particular method, such as SAE. Here, d coder should be d SAE , d TC , or d FF , depending on the methods; in other words, each method can maximally yield d coder numbers of features F = [ f 1 , · · · , f d coder ] . Repeatedly collecting the activations over inputs [ w 1 , · · · , x l ] gives an activation history matrix A ∈ R l × d coder , where each row corresponds to each token x t , and each column corresponds to each feature (neuron) f p ∈ F , respectively. A [: ,p ] ∈ R l presents where a feature f p was activated in S . S p = { t i ∈ T | A t,p &gt; 0 and i, \_ = σ ( t ) } ⊆ S represents the text subset associated with the feature f p .

SwiGLU activation. Modern LMs adopt a SwiGLU gating function [41] for the non-linear activation of FFs ϕ ( · ) . The existing work [54] showed the compatibility of SwiGLU activation with FF-KV analysis, and thus, the above methods (§ 3.1) can be naturally implemented with SwiGLU. For example, on top of the SwiGLU activation, the TopK FF-KV can be written as follows:

<!-- formula-not-decoded -->

where, W G ∈ R d FF × d model is the gating matrix.

## 4 Experiment 1: Automatic Evaluation

We evaluate FF-KVs, SAEs, and Transcoders using the metrics from SAEBENCH [27]. We also report the Feature Alive Rate as a complementary statistic.

## 4.1 Evaluation Metrics

Here, we give a high-level description of each metric, and details are shown in Appendix B.

Feature Alive Rate aggregates how many features are alive, out of d coder features. A positive value of A t,p is regarded as the activation of p -th feature in x t . An indicator function, χ : X [: ,p ] ↦→ 1 if max( X [: ,p ] ) &gt; 0 else 0 , judges if the feature f p is alive (activated at least once) and the following score is calculated: ∑ d coder j =1 χ ( A [: ,j ] ) d coder . A score of 1 indicates that all features are activated at least once.

Explained Variance evaluates how well the proxy module reconstructs the original activations, and the FF-KV methods (without proxies) can automatically get a perfect score (=1) since this is the original module as is.

Absorption Score evaluates how many features a particular simple concept (e.g., word starting with 'S') is split into. A higher value implies that many features are needed to emulate the targeted single concept, and thus, the feature set is redundant.

Sparse Probing evaluates the existence of specific informative features (e.g., sentiment) and their generalizability to held-out data. This is measured based on the accuracy of probe classifiers trained on the activation patterns to predict the properties of unseen inputs for the proxy module.

Auto-Interpretation Score evaluates how easily the activation patterns of the feature can be summarized in natural language (e.g., 'a feature related to accounting'). Specifically, given a text subset S p for the feature f p , 4 an LLM is requested to summarize the feature concept and then predict the (binary) feature activation on the held-out set based on the summary, following Paulo et al. [36]. A score of 1 indicates that features can be perfectly summarized, and their activations are predictable.

Spurious Correlation Removal (SCR) evaluates how well spuriously correlated two features (e.g., gender and profession) are disentangled from different features, using the SHIFT [32] data. A score of 1 indicates a perfect disentanglement. Notably, its extension, Targeted Probe Perturbation (TPP) score, was also invented as a supplemental metric in SAEBENCH [27]. The TPP results are shown in Appendix and yielded consistent results with SCR. Notably, SCR and TPP consider topK activations in evaluations (we adopt K = 20 here, following existing studies [27]) 5 , and results for different K are shown in Appendix B.

RAVEL [23] further evaluates the feature overlap and disentanglement, but slightly from a different angle from SCR and TPP. Specifically, this concerns the separability and controllability of multiple different attributes of the same entity (e.g., Japan continent - - - - - → Asia, Japan capital - - - → Tokyo). The RAVEL score can be decomposed into two complementary scores: (i) Isolation score-the probability that all non-edited attributes remain unchanged; and (ii) Causality score-the probability that the edit successfully changes the target attribute. We report both scores for the RA VEL results to clarify the fine-grained properties.

## 4.2 LMs and Proxy Modules

LMs. We evaluate FFs in all layers of five LMs: Gemma-2-2B, Gemma-2-9B [43], Llama-3.18B [45], GPT-2 [39], and Pythia-70M [4]. Due to space limitations, the results of the middle layers of Gemma-2-2B (layer 13 ) and Llama-3.1-8B (layer 16 ) are shown in the main part of this paper, and the results for other layers and models are shown in Appendix C. We also target randomly initialized LLMs as baselines, given the assertion that SAEs can even interpret randomly initialized Transformers. In our main experiments, we set k = 10 for the TopK FF-KV, and we additionally compare results under varying k values.

SAEs. We use pretrained SAEs from Gemma Scope [30] (width 16k) for Gemma-2-2B and Gemma2-9B, Llama Scope [21] (width 32k) for Llama-3.1-8B, and SAELens [7] for GPT-2. All SAEs are trained on FF outputs.

4 It is also marked at which token in the text the feature was activated.

5 We found SCR and TPP scores are highly unstable, and one may have to treat them as supplementary results. See Appendix B for details.

Table 1: Overview of the SAEBENCH evaluation results for the middle layer of Gemma-2-2B (layer 13 ) and Llama-3.1-8B (layer 16 ). Results are reported as mean ± 2 standard errors of the mean over multiple seeds/settings. Norm. represent the normalized FF-KV. We also present the scores for a randomly initialized FF layers, which serve as the baseline. No substantial difference between FF-KV and SAEs is observed.

|                    | Coder Status    | Coder Status    | Concept Detection   | Concept Detection   |
|--------------------|-----------------|-----------------|---------------------|---------------------|
| Model SAE Type     | Feat. Alive ↑   | Expl. Var. ↑    | Absorption ↓        | Sparse Prob. ↑      |
| SAE                | 0 . 988 ± 0.000 | 0 . 699 ± 0.000 | 0 . 087 ± 0.173     | 0 . 846 ± 0.161     |
| Transcoder         | 1 . 000 ± 0.000 | 0 . 637 ± 0.000 | 0 . 025 ± 0.116     | 0 . 854 ± 0.149     |
| FF-KV              | 0 . 999 ± 0.000 | 1 . 000 ± 0.000 | 0 . 000 ± 0.001     | 0 . 827 ± 0.158     |
| FF-KV (Norm.)      | 1 . 000 ± 0.000 | 1 . 000 ± 0.000 | 0 . 000 ± 0.001     | 0 . 826 ± 0.160     |
| TopK-FF-KV         | 0 . 984 ± 0.000 | 0 . 160 ± 0.000 | 0 . 000 ± 0.001     | 0 . 768 ± 0.168     |
| TopK-FF-KV (Norm.) | 0 . 984 ± 0.000 | 0 . 160 ± 0.000 | 0 . 000 ± 0.000     | 0 . 768 ± 0.168     |
| Random Transformer | 1 . 000 ± 0.000 | 1 . 000 ± 0.000 | 0 . 007 ± 0.013     | 0 . 798 ± 0.067     |
| SAE                | 1 . 000 ± 0.000 | 0 . 594 ± 0.000 | 0 . 097 ± 0.332     | 0 . 879 ± 0.123     |
| Transcoder         | -               | -               | -                   | -                   |
| FF-KV              | 1 . 000 ± 0.000 | 1 . 000 ± 0.000 | 0 . 000 ± 0.001     | 0 . 876 ± 0.098     |
| FF-KV (Norm.)      | 1 . 000 ± 0.000 | 1 . 000 ± 0.000 | 0 . 000 ± 0.000     | 0 . 876 ± 0.098     |
| TopK-FF-KV         | 0 . 992 ± 0.000 | 0 . 238 ± 0.000 | 0 . 000 ± 0.001     | 0 . 832 ± 0.150     |
| TopK-FF-KV (Norm.) | 0 . 992 ± 0.000 | 0 . 238 ± 0.000 | 0 . 000 ± 0.001     | 0 . 832 ± 0.150     |
| Random Transformer | 1 . 000 ± 0.000 | 1 . 000 ± 0.000 | 0 . 002 ± 0.006     | 0 . 837 ± 0.084     |

|                         | Feature Explanation   | Feature Disentanglement   | Feature Disentanglement   | Feature Disentanglement   |
|-------------------------|-----------------------|---------------------------|---------------------------|---------------------------|
| Model SAE Type          | Autointerp ↑          | RAVEL-ISO ↑               | RAVEL-CAU ↑               | SCR (k=20) ↑              |
| SAE                     | 0 . 782 ± 0.274       | 0 . 985 ± 0.027           | 0 . 002 ± 0.006           | 0 . 170 ± 0.191           |
| 2B Transcoder           | 0 . 790 ± 0.270       | 0 . 940 ± 0.040           | 0 . 010 ± 0.017           | 0 . 104 ± 0.178           |
| Gemma-2 FF-KV           | 0 . 710 ± 0.246       | 0 . 952 ± 0.035           | 0 . 012 ± 0.021           | 0 . 041 ± 0.094           |
| FF-KV (Norm.)           | 0 . 706 ± 0.255       | 0 . 952 ± 0.035           | 0 . 012 ± 0.021           | 0 . 041 ± 0.120           |
| TopK-FF-KV              | 0 . 772 ± 0.276       | 0 . 943 ± 0.039           | 0 . 009 ± 0.015           | 0 . 045 ± 0.105           |
| TopK-FF-KV (Norm.)      | 0 . 773 ± 0.269       | 0 . 942 ± 0.038           | 0 . 009 ± 0.014           | 0 . 029 ± 0.134           |
| Random Transformer      | 0 . 679 ± 0.248       | -                         | -                         | 0 . 004 ± 0.022           |
| SAE                     | 0 . 817 ± 0.272       | 0 . 993 ± 0.016           | 0 . 002 ± 0.007           | 0 . 219 ± 0.323           |
| 8B Transcoder           | -                     | -                         | -                         | -                         |
| FF-KV                   | 0 . 751 ± 0.248       | 0 . 955 ± 0.044           | 0 . 007 ± 0.012           | 0 . 048 ± 0.070           |
| Llama-3.1 FF-KV (Norm.) | 0 . 749 ± 0.245       | 0 . 954 ± 0.044           | 0 . 007 ± 0.012           | 0 . 046 ± 0.071           |
| TopK-FF-KV              | 0 . 807 ± 0.267       | 0 . 954 ± 0.044           | 0 . 006 ± 0.011           | 0 . 030 ± 0.045           |
| TopK-FF-KV (Norm.)      | 0 . 807 ± 0.256       | 0 . 955 ± 0.043           | 0 . 006 ± 0.010           | 0 . 029 ± 0.043           |
| Random Transformer      | 0 . 656 ± 0.237       | -                         | -                         | 0 . 053 ± 0.239           |

Transcoders. We use pretrained Transcoders (TCs) from Gemma Scope [30] for Gemma-2-2B and one from the original paper [12] for GPT-2, respectively. 6

## 4.3 Results

Overall. Table 1 shows the results for each interpretability method. First of all, the SAE-based results and FF-KV results rendered similar tendencies. In each metric, the absolute difference between the scores from different methods is typically much smaller than seed/layer variance. In addition, the difficulty of each task (metric) is aligned across the tasks; for example, SAEs and FF-KVs achieved higher RAVEL-Isolation scores than RAVEL-Causality scores. These results suggest that, even with the activations in the original FF module, comparable interpretability can be realized compared to proxy-based methods, i.e., SAEs and Transcoders.

6 See Appendix D for more details on the SAEs and Transcoders we use.

Figure 1: Evaluation scores for TopK FF-KVs at Layer 13 on Gemma-2-2B, under a different sparsity value k . A lower k indicates a higher sparsity. Shaded areas denote ± 2 standard errors of the mean, computed across multiple seeds and evaluation settings.

<!-- image -->

Inter-Methods Similarity. To mention specific similarity among the methods, causal intervention (RAVEL-Causality) was difficult for both SAEs and FF-KVs; that is, FF-KV methods inherit the limitation of SAEs. In contrast, feature isolation is well realized in FF-KVs, similarly to SAEs, even without any specific feature disentanglement regulation in FF-KVs, based on the high scores in RAVEL-isolation. Later layers tend to yield a high RA VEL-isolation score, and vice versa, especially in the case of FF-KVs. This shows a parallel with the existing observation that FFs in later layers have more semantic features [17], and attributes targeted in the RA VEL dataset might not be well-shaped in earlier layers.

Inter-Methods Difference. To highlight the differences among the methods, FF-KV methods can achieve perfect explained variance by definition (i.e., zero reconstruction loss as the original model is directly analyzed), whereas SAEs cannot. In addition, FF-KVs exhibited better absorption scores; that is, a simple single concept is not overly split into multiple concepts in FF-KVs than in SAEs, and in this sense, features in FF-KVs are less redundant. Sparseprobing results are comparable or slightly better in FF-KVs; representative features are encoded and generalizable to the same extent in both SAEs and FF-KVs. SAEs achieved slightly but consistently better Auto-interpretablity and SCP

Loading [MathJax]/extensions/MathMenu.js Figure 2: Relationship between FF hidden dimension size (model scale) and RAVEL scores.

<!-- image -->

(although around zero) scores, which are only the advantage of SAEs compared to FF-KV-based analyses.

FF-KV Variants. Within the FF-KV variants, TopK and normalization effects were generally small. Vanilla FF-KV features already exhibited a reasonable interpretability.

TopK Effects. Figure 1 shows the relationship between the k value of the TopK FF-KV (x-axis) and the SAEBENCH evaluation scores (y-axis). The increase of sparsity level leads to inconsistent results, for example, a higher sparse probing (top-5) score but a lower autointerpretation score, suggesting that higher sparsity is not always better, at least for interpretation FF-KV .

FF-Scaling Effects. Figure 2 shows the relationship between FF hidden representation size (model scale; x-axis) and RAEL interpretability scores (y-axis). These results suggest that FFs with a larger hidden dimension size do not always get better interpretability results, suggesting that just extending the hidden dimension size of FFs into that of SAEs does not lead to better interpretability. See Appendix E for other metrics.

## 5 Experiment 2: Human Evaluation

Our results in Section 4 suggest that FFs' internal activations have overall comparable interpretability to SAEs/Transcoders based on automatic evaluations. In this section, we further perform a follow-up manual inspection on the interpretability of features extracted from layer 13 of Gemma-2-2B's FF-KV, SAE, and Transcoder. We specifically explore the following questions: 1) Do features from the FF-KV, SAE, and Transcoder appear equally interpretable to humans? 2) How accurately can humans infer the origin of the feature?

## 5.1 Settings

We randomly sampled 50 features each from the FFKV, TopK FF-KV, SAE, and Transcoder of Gemma-2-2B model, yielding a total of 200 features. Each feature f p is presented with its top-ten associated texts ∈ S p based on the activation magnitude over a subset of OpenWebText corpus [19] (200M tokens in total). From the annotator's view, the presentation order of features is randomly shuffled, and their origins remain hidden throughout the experiment. The annotations in this section were conducted by one of the authors.

Table 2: Number of superficial, conceptual, and uninterpretable features.

| Coder   |   SuperficialConceptualUninterp. |   SuperficialConceptualUninterp. |   SuperficialConceptualUninterp. |
|---------|----------------------------------|----------------------------------|----------------------------------|
| FF-KV   |                                6 |                                8 |                               36 |
| K-FF-KV |                                9 |                                9 |                               32 |
| SAE     |                                6 |                                9 |                               35 |
| TC      |                               16 |                               11 |                               23 |

Interpretability Evaluation. One annotator judges the qualitative quality of a feature using three categories: 1) superficial Feature : activates on shallow surface patterns (e.g., particular word, such as 'the,' punctuation, digits); 2) Conceptual Feature : activates on higher-level concepts spanning multiple tokens (e.g., sentiment, topics); or, 3) Uninterpretable : exhibits no clear activation pattern 7 .

Feature Origin Judgment. We also designed a task to predict from which module a feature originates, only based on the feature activation patterns in 10 texts, with the same data, to exploratorily find any difference between these activation patterns. If annotators can not guess which module is used to obtain the given feature, the used methods would have the same level of feature extraction ability. One annotator conducted this analysis, and as preliminary training, the annotator had first learned several activation patterns in the held-out set, paired with their module names.

## 5.2 Results

Interpretability Evaluation. The results are presented in Table 2. First, the number of conceptual features is nearly the same across the four interpretability methods. In this sense, the quality of the obtained features is comparable. Transcoders could find a larger number of features that are interpretable (superficial or conceptual), but the ratio of superficial features is higher than in the other methods.

Feature Origin Judgment. Table 3 shows the results. The annotator could not correctly predict the original model, except for the FF-KV methods. Through interviewing the annotator, we found that they could identify the FF-KV features by relying on superficial patterns in the magnitude and variance of feature activations (FF-KV tends to have a small value with high variance), rather than the represented concepts. Using TopK FF-KV (K-FF-KV) alleviates this distinction pattern, and thus,

7 We provide the actual text we use to annotate in Appendix F.

Table 3: Origin judgment accuracies of features.

| Origin   | Judging Acc.   |
|----------|----------------|
| FF-KV    | 0 . 86         |
| K-FF-KV  | 0 . 28         |
| SAE      | 0 . 13         |
| TC       | 0 . 18         |

if one wants to render a visualization of activations similar to that of SAEs, TopK FF-KV should be preferred. The low accuracies for K-FF-KV, SAE, and TC support that their discovered features and activations are similar to each other, as the human evaluator could not distinguish them.

## 6 Analysis: Feature Alignments

We analyze how similar features discovered by proxy methods, e.g., SAEs, are to the FF-KV ones.

## 6.1 Settings

To investigate the alignment of features from different interpretability methods, we specifically focus on those from FF-KV and Transcoder (not SAE, as the closest counterpart to FF-KV). In this analysis, we used layer 13 of Gemma-2-2B, which showed reasonable performance in the automatic evaluation experiment. Given an r -th feature vector in the FF W V [: ,r ] , we find the index u of the most aligned feature vector in W dec from Transcoder, based on their cosine similarity: u = arg max k ( W V [: ,r ] · W dec[: , k] ) . Note that when analyzing features in TC, the searching direction will be opposite: arg max k ( W dec[: ,r ] · W V [: ,k ] ) . We call these max-cosine scores MCS.

We first perform a quick check of the correlation between MCS and the semantic feature alignment. For each MCS bin, we sampled ten feature pairs. Then, for feature pairs ( f p 1 , f p 2 ) within a specific MCS range, annotators manually inspected their alignment. Similarly to the previous experiment, each feature f p is accompanied by ten associated texts ∈ S p from a subset of the OpenWebText corpus [19]. We consider a pair matched if these three criteria meet: 1) The two features generally represent the same concept (e.g., sentiment, topic); OR 2) The associated texts for the two features ( S p 1 , S p 2 ) exhibits 8 / 10 overlap; OR 3) The topics of the texts from two features coincide. The annotations in this section were independently conducted by a different author from that of § 5.

## 6.2 Results

Validity of Cosine-Based Alignments. The results of alignment analysis are shown in Figure 3. This clearly shows that higher cosine similarity entails their semantic alignment. Based on these results, we tentatively regard a feature pair with cosine similarity above 0.9 as aligned , and the similarity below 0.3 as unaligned in the following analyses 8 .

Large Number of Unaligned Features. Based on the above criteria with cosine similarity, 41% (=3,780/9,216) and 66% (=10,835/16,384) features in FF-KV and Transcoder are unaligned with each other, respectively. In contrast, 5.7% (=527/9,216)

Figure 3: Histogram of aligned features numbers distribution for each MCS bin, between the FF and Transcoder.

<!-- image -->

and 3.2% (=527/16,384) features are regarded as aligned in FF-KV and Transcoder, respectively. That is, there are a large number of unaligned features between FF and Transcoder, clarifying that the same level of interpretability from different methods in automatic evaluation (§ 4) was not simply due to their similar feature sets. In the next paragraph, we manually analyze the unaligned features.

Feature Complementarity. We manually analyze three sets of features: (a) aligned features (FFKV ∩ Transcoder), (b) FF-KV features not aligned with any Transcoder feature (FF-KV \ Transcoder), and (c) Transcoder features not aligned with any FF-KV feature (Transcoder \ FF-KV). The analysis target is the same as § 5; the features are classified into three categories of superficial, conceptual, and uninterpretable. Table 4 shows the distribution of feature categories in each feature set. First and interestingly, the unaligned features both in FFs and TC have a fair amount of conceptual ones. In particular, 32% of features that are unaligned with FFs were conceptual.

8 See Appendix F for examples of the feature pairs we use for annotation as well as how we decide the threshold.

Discussion. Why are there so many unaligned features? One optimistic view is that TC successfully decomposed FF features into simpler ones, resulting in decomposed features being orthogonal to the original FF features, although this may offer a potential side effect of feature absorption, which is suggested by relatively large number of superficial features in TC \ FF-KV and an already good absorption scores achieved by FF-KVs (Table 1). One more pessimistic view is that Transcoder invented completely new features that are not in the original FF-KV, which is in line with the fact that SAE can interpret even randomly initialized Transformers [22]. Our analysis alone can not

Table 4: Number of superficial, multitoken conceptual, and uninterpretable features in aligned/unaligned features between FF (FF-KV) and Transcoders (TC).

| Coder      |   SuperficialConcept.Uninterp. |   SuperficialConcept.Uninterp. |   SuperficialConcept.Uninterp. |
|------------|--------------------------------|--------------------------------|--------------------------------|
| FF-KV ∩ TC |                              7 |                             16 |                             27 |
| FF-KV \ TC |                              1 |                              8 |                             41 |
| TC \ FF-KV |                              6 |                             14 |                             23 |

fully distinguish between the two cases, but this fact of frequent misalignment deserves a motivation to further explore the faithfulness of the learned features in proxy modules. Features in the FF-KVs will serve as grounding points to evaluate such faithful evaluation, on top of our first extensive attention to FF-KVs in the context of SAE research.

## 7 Conclusion

In this work, we revisit the interpretability of feature vectors already represented in the feed-forward (FF) module, as a strong baseline to SAEs. Our results show that the original FF feature vectors already exhibit reasonable interpretability comparable to that of sparse autoencoders (SAEs) and Transcoders on both comprehensive benchmark and human evaluations. We further demonstrate that a large portion of the features between the FF and the Transcoder are not aligned, and manual analysis suggests a potential feature over-splitting or hallucination of new features in the proxy module. To sum up, our results demonstrate that SAEs and Transcoders offer only limited advantages over the direct analysis of feed-forward key-value (FF-KV) representations. This finding highlights the lack of a strong and simple baseline within the interpretability community and underscores the importance of including FF-KV analysis as a fundamental reference point for evaluating interpretability methods. It also encourages future work to consider both model-internal parameters and proxy-module parameters when pursuing feature-discovery-based interpretability of large language models.

Limitations. The feature dimension of SAEs and Transcoders we used was fixed; more diverse configurations should be examined in the comparison, although publicly available pre-trained SAEs/Transcoders are limited, and prior work shows that simply scaling width does not necessarily improve SAEs and that there is not a universally best architecture choice [27]. Not all models are accompanied by Transcoder results: still, training Transcoders on all layers of, e.g., 9 B-parameter models is prohibitively costly under an academic budget. We conducted only a few qualitative case studies on the effect of k for TopK FF-KVs and on FF size. Although our analysis showed a discrepancy between FF-KV and Transcocder features, the interpretation of this difference (faithfulness of the learned features) remains unclear; future work should elaborate on this point.

Impact Statement. Our findings indicate that SAEs and Transcoders do not consistently outperform the original feature vectors in FFs with respect to interpretability. We underscore the need to reassess both the interpretability and, potentially, the reproducibility of the previously reported advantages of SAEs. In a sense, our study supports the use of inherently black-box neural LLMs while setting aside the interpretability issue, as their FFs appear to possess a certain degree of interpretability. Nevertheless, one of the ultimate objectives of this line of research should remain the development of models that are interpretable by design.

Ethics Statement. Our research primarily relied on publicly available models and datasets, and strictly adhered to their respective licenses (see Table 7). For human evaluation, we collected data as described in § 5.1. The data were collected with participant consent, and we ensured that responses were anonymized to prevent them from being traced back to individuals. To promote transparency and reproducibility, we have made the collected data, along with all code used in our experiments, publicly accessible. Comprehensive details of our experimental setup are provided in each section and the appendix to ensure reproducibility.

## Acknowledgment

Author Contribution. M. Ye led the research project, implemented the experimental pipeline, conducted the SAEBENCH evaluation, designed the human evaluation task, and carried out one of the human evaluation studies. T. Kuribayashi initially proposed the idea of comparing the transcoder with FF-KV augmented by a TopK activation function and was deeply involved in regular discussions throughout the project. T. Inaba provided valuable feedback on implementation, conducted one of the human evaluation studies, and actively contributed to project discussions. J. Suzuki provided overarching guidance and feedback at all stages of the project as well as computational resources.

M. Ye drafted the initial version of the manuscript. T. Kuribayashi offered extensive feedback and revisions on the writing. T. Inaba authored the impact and ethics statements. J. Suzuki contributed valuable insights into the overall framing and positioning of the paper.

Acknowledgments. We want to express our gratitude to the members of the Tohoku NLP Group for their insightful comments. And special thanks to Charles Spencer James for assisting in determining the MCS threshold for text alignment through human annotation. This work was supported by the JSPS KAKENHI Grant Number JP24H00727, JP25KJ06300; JST Moonshot R&amp;D Grant Number JPMJMS2011-35 (fundamental research); JST BOOST, Japan Grant Number JPMJBS2421.

## References

- [1] Anthropic. The claude 3 model family: Opus, sonnet, haiku. arXiv preprint arXiv:2303.08774 , 2024.
- [2] O. Antverg and Y. Belinkov. On the pitfalls of analyzing individual neurons in language models. In The Tenth International Conference on Learning Representations , 2022.
- [3] S. Arora, Y. Li, Y. Liang, T. Ma, and A. Risteski. Linear algebraic structure of word senses, with applications to polysemy. Transactions of the Association for Computational Linguistics , 6:483-495, 2018.
- [4] S. Biderman, H. Schoelkopf, Q. G. Anthony, H. Bradley, K. O'Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth, E. Raff, A. Skowron, L. Sutawika, and O. Van Der Wal. Pythia: A suite for analyzing large language models across training and scaling. In Proceedings of the 40th International Conference on Machine Learning , pages 2397-2430. PMLR, 2023.
- [5] S. Bills, N. Cammarata, D. Mossing, H. Tillman, L. Gao, G. Goh, I. Sutskever, J. Leike, J. Wu, and W. Saunders. Language models can explain neurons in language models, 2023. URL https: //openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html .
- [6] J. Bloom. Gpt2-small-oai-v5-32k-mlp-out-saes, 2024. URL https://huggingface.co/ jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs .
- [7] J. Bloom, C. Tigges, A. Duong, and D. Chanin. Saelens, 2024. URL https://github.com/ jbloomAus/SAELens .
- [8] T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, T. Conerly, N. Turner, C. Anil, C. Denison, A. Askell, R. Lasenby, Y. Wu, S. Kravec, N. Schiefer, T. Maxwell, N. Joseph, Z. HatfieldDodds, A. Tamkin, K. Nguyen, B. McLean, J. E. Burke, T. Hume, S. Carter, T. Henighan, and C. Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread , 2023. URL https://transformer-circuits.pub/2023/ monosemantic-features/index.html .
- [9] T. Bricken, J. Marcus, S. Mishra-Sharma, M. Tong, E. Perez, M. Sharma, K. Rivoire, T. Henighan, and A. Jermyn. Using dictionary learning features as classifiers. Transformer Circuits Thread , 2024. URL https://transformer-circuits.pub/2024/ features-as-classifiers/index.html .
- [10] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems , volume 33, pages 1877-1901, 2020.

- [11] D. Chanin, J. Wilken-Smith, T. Dulka, H. Bhatnagar, and J. Bloom. A is for absorption: Studying feature splitting and absorption in sparse autoencoders. arXiv preprint arXiv:2409.14507 , 2024.
- [12] J. Dunefsky, P. Chlenski, and N. Nanda. Transcoders find interpretable llm feature circuits. In Advances in Neural Information Processing Systems , volume 37, pages 24375-24410, 2024.
- [13] N. Durrani, H. Sajjad, F. Dalvi, and Y. Belinkov. Analyzing individual neurons in pre-trained language models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing , pages 4865-4880. Association for Computational Linguistics, 2020.
- [14] N. Elhage, T. Hume, C. Olsson, N. Nanda, T. Henighan, S. Johnston, S. ElShowk, N. Joseph, N. DasSarma, B. Mann, D. Hernandez, A. Askell, K. Ndousse, A. Jones, D. Drain, A. Chen, Y. Bai, D. Ganguli, L. Lovitt, Z. Hatfield-Dodds, J. Kernion, T. Conerly, S. Kravec, S. Fort, S. Kadavath, J. Jacobson, E. Tran-Johnson, J. Kaplan, J. Clark, T. Brown, S. McCandlish, D. Amodei, and C. Olah. Softmax linear units. Transformer Circuits Thread , 2022. URL https://transformer-circuits.pub/2022/solu/index.html .
- [15] L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, S. Presser, and C. Leahy. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027 , 2020.
- [16] L. Gao, T. D. la Tour, H. Tillman, G. Goh, R. Troll, A. Radford, I. Sutskever, J. Leike, and J. Wu. Scaling and evaluating sparse autoencoders. In The Thirteenth International Conference on Learning Representations , 2025.
- [17] M. Geva, R. Schuster, J. Berant, and O. Levy. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , pages 5484-5495. Association for Computational Linguistics, 2021.
- [18] M. Geva, A. Caciularu, K. Wang, and Y. Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , pages 30-45. Association for Computational Linguistics, 2022.
- [19] A. Gokaslan and V. Cohen. Openwebtext corpus, 2019. URL http://Skylion007.github. io/OpenWebTextCorpus .
- [20] W. Gurnee, N. Nanda, M. Pauly, K. Harvey, D. Troitskii, and D. Bertsimas. Finding neurons in a haystack: Case studies with sparse probing. Transactions on Machine Learning Research , 2023. ISSN 2835-8856.
- [21] Z. He, W. Shu, X. Ge, L. Chen, J. Wang, Y. Zhou, F. Liu, Q. Guo, X. Huang, Z. Wu, Y.-G. Jiang, and X. Qiu. Llama scope: Extracting millions of features from llama-3.1-8b with sparse autoencoders. arXiv preprint arXiv:2410.20526 , 2024.
- [22] T. Heap, T. Lawson, L. Farnik, and L. Aitchison. Sparse autoencoders can interpret randomly initialized transformers. arXiv preprint arXiv:2501.17727 , 2025.
- [23] J. Huang, Z. Wu, C. Potts, M. Geva, and A. Geiger. RA VEL: Evaluating interpretability methods on disentangling language model representations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 8669-8687. Association for Computational Linguistics, 2024.
- [24] R. Huben, H. Cunningham, L. R. Smith, A. Ewart, and L. Sharkey. Sparse autoencoders find highly interpretable features in language models. In The Twelfth International Conference on Learning Representations , 2024.
- [25] T. Inaba, G. Kamoda, K. Inui, M. Isonuma, Y. Miyao, Y. Oseki, B. Heinzerling, and Y. Takagi. Howabilingual lm becomes bilingual: Tracing internal representations with sparse autoencoders. arXiv preprint arXiv:2503.06394 , 2025.
- [26] S. Kantamneni, J. Engels, S. Rajamanoharan, M. Tegmark, and N. Nanda. Are sparse autoencoders useful? a case study in sparse probing. In Forty-second International Conference on Machine Learning , 2025.
- [27] A. Karvonen, C. Rager, J. Lin, C. Tigges, J. I. Bloom, D. Chanin, Y.-T. Lau, E. Farrell, C. S. McDougall, K. Ayonrinde, D. Till, M. Wearden, A. Conmy, S. Marks, and N. Nanda. SAEBench: A comprehensive benchmark for sparse autoencoders in language model interpretability. In Forty-second International Conference on Machine Learning , 2025.

- [28] K. Li, O. Patel, F. Viégas, H. Pfister, and M. Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems , volume 36, pages 41451-41530, 2023.
- [29] Z. Li, C. You, S. Bhojanapalli, D. Li, A. S. Rawat, S. J. Reddi, K. Ye, F. Chern, F. Yu, R. Guo, and S. Kumar. The lazy neuron phenomenon: On emergence of activation sparsity in transformers. In The Eleventh International Conference on Learning Representations , 2023.
- [30] T. Lieberum, S. Rajamanoharan, A. Conmy, L. Smith, N. Sonnerat, V. Varma, J. Kramar, A. Dragan, R. Shah, and N. Nanda. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP , 2024.
- [31] S. Marks and M. Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In First Conference on Language Modeling , 2024.
- [32] S. Marks, C. Rager, E. J. Michaud, Y. Belinkov, D. Bau, and A. Mueller. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. In The Thirteenth International Conference on Learning Representations , 2025.
- [33] N. Nanda and J. Bloom. Transformerlens, 2022. URL https://github.com/ TransformerLensOrg/TransformerLens .
- [34] C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, S. Johnston, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, J. Clark, J. Kaplan, S. McCandlish, and C. Olah. In-context learning and induction heads. arXiv preprint arXiv:2209.11895 , 2022.
- [35] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 , 2024.
- [36] G. S. Paulo, A. T. Mallen, C. Juang, and N. Belrose. Automatically interpreting millions of features in large language models. In Forty-second International Conference on Machine Learning , 2025.
- [37] T. Pimentel, N. Saphra, A. Williams, and R. Cotterell. Pareto probing: Trading off accuracy for complexity. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing , pages 3138-3153. Association for Computational Linguistics, 2020.
- [38] T. Pimentel, J. Valvoda, N. Stoehr, and R. Cotterell. The architectural bottleneck principle. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , pages 11459-11472. Association for Computational Linguistics, 2022.
- [39] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. OpenAI , 2019. URL https://cdn.openai.com/better-language-models/language\_models\_are\_ unsupervised\_multitask\_learners.pdf .
- [40] H. Sajjad, N. Durrani, and F. Dalvi. Neuron-level interpretation of deep NLP models: A survey. Transactions of the Association for Computational Linguistics , 10:1285-1303, 2022.
- [41] N. Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202 , 2020.
- [42] X. Suau, L. Zappella, and N. Apostoloff. Finding experts in transformer models. arXiv preprint arXiv:2005.07647 , 2020.
- [43] G. Team. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118 , 2024.
- [44] G. Team. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 , 2024.
- [45] L. Team. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 , 2024.
- [46] A. Templeton, T. Conerly, J. Marcus, J. Lindsey, T. Bricken, B. Chen, A. Pearce, C. Citro, E. Ameisen, A. Jones, H. Cunningham, N. L. Turner, C. McDougall, M. MacDiarmid, C. D. Freeman, T. R. Sumers, E. Rees, J. Batson, A. Jermyn, S. Carter, C. Olah, and T. Henighan. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Transformer Circuits Thread , 2024. URL https://transformer-circuits.pub/2024/ scaling-monosemanticity/index.html .

- [47] I. Tenney, D. Das, and E. Pavlick. BERT rediscovers the classical NLP pipeline. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics , pages 4593-4601. Association for Computational Linguistics, 2019.
- [48] C. Tigges. Pythia-70m-deduped-mlp-sm, 2024. URL https://huggingface.co/ctigges/ pythia-70m-deduped\_\_mlp-sm\_processed .
- [49] N. L. Turner, A. Jermyn, J. Batson, and J. Batson. Measuring feature sensitivity using dataset filtering. Transformer Circuits Thread , 2024. URL https://transformer-circuits.pub/ 2024/july-update/index.html#feature-sensitivity .
- [50] E. Voita and I. Titov. Information-theoretic probing with minimum description length. In B. Webber, T. Cohn, Y. He, and Y. Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing , pages 183-196. Association for Computational Linguistics, 2020.
- [51] Z. Wu, A. Arora, A. Geiger, Z. Wang, J. Huang, D. Jurafsky, C. D. Manning, and C. Potts. Axbench: Steering LLMs? even simple baselines outperform sparse autoencoders. In Fortysecond International Conference on Machine Learning , 2025.
- [52] J. Xin, J. Lin, and Y. Yu. What part of the neural network does this? understanding LSTMs by measuring and dissecting neurons. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , pages 5823-5830. Association for Computational Linguistics, 2019.
- [53] Z. Yun, Y. Chen, B. Olshausen, and Y. LeCun. Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors. In Proceedings of Deep Learning Inside Out (DeeLIO): The 2nd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures , pages 1-10. Association for Computational Linguistics, 2021.
- [54] S. Zhong, M. Xu, T. Ao, and G. Shi. Understanding transformer from the perspective of associative memory. arXiv preprint arXiv:2505.19488 , 2025.

## A Implementation Details on FF-KVs

## A.1 Overall Framework

Weimplement FF-KVs use the custom\_sae class provided by SAEBENCH [7]. To faithfully reproduce the activation of the FF sublayers, we apply the hook-based approach. The encode method simulates the FF's forward pass up to its neuron activations. It takes an input tensor x , injects it at the FF's input hook point, and captures the subsequent neuron activations using another hook. Conversely, the decode method simulates the FF's transformation from its neuron activations to its output. It accepts a tensor of neuron activations, injects them at the corresponding hook point, and captures the FF's final output. A forward method is also provided, performing the full pass through the FF block using hooks to inject input and capture the final output. This framework allows for the direct examination of an FF's feature extraction and signal reconstruction capabilities as if it were an SAE, providing a unique lens for interpreting learned representations within large language models.

## A.2 FF-KV Implement Details

The core principle is to map the FF's operations to the conceptual stages of an SAE:

Input. Activations entering the FF block serve as the input to our pseudo-SAE.

Feature Representation (Encoding). The FF's internal neuron activations, captured after the non-linear activation function and any gating, are interpreted as the SAE's latent features. The dimensionality of this feature space is equivalent to the FF's hidden dimension. The effective 'encoder weights' are the FF's input weights.

Reconstruction (Decoding). The FF's output, which is typically added to the residual stream, is considered the reconstructed input. The effective 'decoder weights' and 'decoder bias' are the FF's output weights and output bias, respectively.

## A.3 TopK FF-KV Implement Details

The TopK FF-KV extends the FF-KV framework by enforcing a strict TopK sparsity on the intermediate feature representations. The only modification from a FF-KV resides in the encode method. After obtaining the FF's internal neuron activations, a k-sparsity constraint is applied. For each input position (e.g., token in a sequence), the method identifies the k neuron activations with the largest absolute values. All other neuron activations for that position are set to zero. This results in a feature vector where, at most, k dimensions are non-zero.

## A.4 Normalized FF-KV Implement Details

The FF block's original output weights (which serve as the 'decoder weights') are L 2 -normalized along their feature dimension. The original norms of these weight vectors are stored. The encode and forward methods remain unchanged from their respective base FF-KV implementations. The key difference lies in the decode method. Before applying the normalized 'decoder weights', the input feature activations are first scaled by the stored original norms. This step ensures that the magnitude of the reconstructed output appropriately reflects the original scaling of the FF block's output projections, despite the normalization. For models that include a final normalization step after the FF output (e.g., Gemma-2 Models), this step is also applied to maintain fidelity with the original model's computation.

## A.5 Transcoder Implement Details

We load the Transcoder weight into the JumpReLU class of SAEBENCH. While evaluating it, we follow the instructions of Gemma-Scope paper [30] to load model weights with folding applied.

## B Details on Metrics

We provide detailed definitions of the metrics used in our main results, alongside the experimental settings for each metric.

Feature Alive Rate. This metric belongs to the 'core' evaluations in SAEBENCH. It counts how many features are alive out of the d coder total features. A feature is deemed active when its activation exceeds 0 . The evaluation is conducted on a 4 M-token subset of OpenWebText [19].

The metric is especially relevant for TopK FF-KVs employing a TOPK activation, ensuring that the mechanism does not repeatedly select only a small subset of neurons.

Explained Variance. Also in the 'core' suite, this metric is computed on a 0.4 M-token subset of OpenWebText [19].

Absorption Score. Feature absorption [11] stems from feature splitting [8, 49], in which newly uncovered features become overly specific. A concrete example is a feature that activates only on 'U.S. cities except New York and Los Angeles.'

The metric targets a first-letter classification task, measuring situations where the main feature for a letter fails to fully capture the concept of 'first letter', and other features compensate. Specifically, it evaluates all 26 letters with the prompt ' {word} has the first letter:'.

Given primary features S main (e.g., selected via sparse probing) and auxiliary features S abs , the absorption score for one input is

<!-- formula-not-decoded -->

where a i is the activation, d i the unit decoder direction, and p the ground-truth probe direction. We use the default hyperparameters.

Sparse Probing. Sparse probing, introduced by Gurnee et al. [20], evaluates the alignment between individual features and a prespecified concept c . It has a hyperparameter K that specifies how many features are used when training the probe. For each feature h j ,

<!-- formula-not-decoded -->

where X + and X -denote inputs with and without c , respectively. The top K features by s j serve as inputs to a logistic-regression probe; the probe's test accuracy constitutes the sparse-probing score. We again employ the default hyperparameters.

Auto-Interpretation Score. This evaluation has two phases: generation and scoring. In the generation phase, it obtain SAE activations, annotate each token with its activation value for the feature under consideration, and prompt an LLM to generate explanations based on these annotated activation patterns. The scoring phase constructs a test set for each feature containing 14 examples, exactly two of which are activated texts. The LLM must label each of the 14 texts as activated or not; the resulting prediction accuracy yields the auto-interpretation score.

The dataset is a subset of the copyright-free version of the Pile [15] (monology/pile-uncopyrighted), comprising 2 M tokens. GPT-4o [35] is used both to generate explanations and to predict activations.

SCRandTPP Following SHIFT [32], the SCR evaluation proceeds as follows. A baseline classifier C base is trained on data containing both true and spurious correlations. We then zero-ablate the K features most attributable to the spurious signal and re-measure accuracy on a balanced set:

<!-- formula-not-decoded -->

where A abl is the post-ablation accuracy, A base the baseline accuracy, and A oracle the accuracy of an oracle probe trained on the true concept.

TPP. We extend Targeted Probe Perturbation to the multi-class setting. For m classes, let C j be a linear probe that classifies concept c j with accuracy A j . Let A i,j denote the accuracy of probe C j after ablating the K most contributive features for class c i . The TPP score is then

̸

<!-- formula-not-decoded -->

so higher SCR and TPP values indicate stronger disentanglement.

Stability Caveats. Both metrics are highly sensitive to the choice of K . Across different K values, SCR can range from below 0 . 1 to above 0 . 4 . For TPP the variation is even larger: for the same coder, scores span from under 0 . 1 (SAE, K = 2 , Figure 14) to over 0 . 4 (SAE, K = 50 , Figure 18). Error bands obtained from multiple sub-runs are also wide-not only for SAEs (e.g., SAE on Llama-3.1 in Figure 14, and on Pythia in Figure 15) but likewise for FF-KVs (e.g., Pythia in Figure 14 and Figure 15).

Based on these empirical observations, we interpret SCR and TPP scores with caution.

RAVEL. Unlike SCR and TPP, we find RAVEL to be consistent across multiple models and coders, and the results align with the scores reported in the original work [23]. This stability suggests that RAVEL is comparatively insensitive to hyperparameter choices and dataset splits, making it a reliable baseline when assessing disentanglement. Accordingly, we place greater weight on RAVEL when synthesizing conclusions across metrics.

## C Detailed Results on SAEBENCH

We provide detailed evaluation result with error bars indicate 95% confidence intervals ( ± 2 SEM), compute as SEM = √ ∑ ( x i -¯ x ) 2 n ( n -1) where n is the number of runs on different datasets for each metric in each layer. Note that error bars are not applicable for the feature alive metric, as they are counts for features activated at least once.

## C.1 Detailed Results on More Models

Figures 5, 6, 10, 13, 7, 8, 9, and 17 present the detailed results for all models.

## C.2 Detailed Results on Various Hyperparameter Choices

For metrics that have multiple hyperparameter choices for k , we provide detailed results for all tested hyperparameters.

- For SCR, the available k values are 2 , 5 , 10 , 20 , 50 , 100 , and 500 ; the corresponding results are shown in Figures 14, 15, 16, 17, 18, 19, and 20.
- For TPP, the available k values are the same, and all results are shown in Figures 21, 22, 23, 24, 25, 26, and 27.
- For Sparse Probing, the available k values are 1 , 2 , and 5 ; the results are shown in Figures 11, 12, and 13.

## D Details on SAEs/Transcoders used

For both SAEs and Transcoders from Gemma-Scope, we use the canonical versions, whose average L 0 sparsity is close to 100 , which are believed to be reasonably useful 9 . The SAEs are loaded through SAELens [7] with the following keys: 'gemma-scope-2b-pt-mlp-canonical' for Gemma-2-2B, 'gemma-scope-9b-pt-mlp-canonical' for Gemma-2-9B, and 'llama\_scope\_lxm\_8x' for Llama-3.18B.

For Transcoders, since no canonical versions have been explicitly defined and the SAELens release we use does not yet support loading them, we manually select checkpoints from the Gemma-Scope

9 This statement can be found on Gemma Scope's collection page on HuggingFace (link).

Table 5: Mapping of layers to their corresponding IDs

|   Layer | ID                                           |
|---------|----------------------------------------------|
|       0 | layer_0/width_16k/average_l0_115/params.npz  |
|       1 | layer_1/width_16k/average_l0_104/params.npz  |
|       2 | layer_2/width_16k/average_l0_87/params.npz   |
|       3 | layer_3/width_16k/average_l0_96/params.npz   |
|       4 | layer_4/width_16k/average_l0_88/params.npz   |
|       5 | layer_5/width_16k/average_l0_87/params.npz   |
|       6 | layer_6/width_16k/average_l0_95/params.npz   |
|       7 | layer_7/width_16k/average_l0_70/params.npz   |
|       8 | layer_8/width_16k/average_l0_92/params.npz   |
|       9 | layer_9/width_16k/average_l0_72/params.npz   |
|      10 | layer_10/width_16k/average_l0_88/params.npz  |
|      11 | layer_11/width_16k/average_l0_108/params.npz |
|      12 | layer_12/width_16k/average_l0_111/params.npz |
|      13 | layer_13/width_16k/average_l0_89/params.npz  |
|      14 | layer_14/width_16k/average_l0_81/params.npz  |
|      15 | layer_15/width_16k/average_l0_78/params.npz  |
|      16 | layer_16/width_16k/average_l0_87/params.npz  |
|      17 | layer_17/width_16k/average_l0_112/params.npz |
|      18 | layer_18/width_16k/average_l0_99/params.npz  |
|      19 | layer_19/width_16k/average_l0_89/params.npz  |
|      20 | layer_20/width_16k/average_l0_88/params.npz  |
|      21 | layer_21/width_16k/average_l0_102/params.npz |
|      22 | layer_22/width_16k/average_l0_117/params.npz |
|      23 | layer_23/width_16k/average_l0_116/params.npz |
|      24 | layer_24/width_16k/average_l0_96/params.npz  |
|      25 | layer_25/width_16k/average_l0_110/params.npz |

collection and download the corresponding weights from HuggingFace (link). These checkpoints are chosen according to the same criteria as the canonical SAEs, and their exact filenames are listed in Table 5. We also directly download the weight from the Transcoder proposal work [12] on HuggingFace (Link). To the best of our knowledge, there are no Transcoders publicly available for Gemma-2-9B and Llama-3.1-8B, and Pythia-70M.

## E Detailed Results on FF Scaling

Table 6 shows the results on all metrics regarding to various FF intermediate sizes. Scores are not showing noticeable improvement except for RAVEL and absorption. Trends shown in SCR somehow understandable: these metrics highly depend on the ground truth probing performance, which is not always stable. Sparse probing result is also understandable, since sparse probing on FF from a random transfer can achieve a reasonable score, the probs can learn unintended signal in the dataset, rather than the true feature.

## F Feature Examples

Figure 4: Proportion of aligned features as a function of the max-cosine score (MCS).

<!-- image -->

We provide example features for each annotation and coder, visualizing the top input examples that most strongly activate each feature. All features are extracted from layer 13 of Gemma-2-2B. To analyze the relationship between alignment and the max-cosine score (MCS), we divide the MCS

Table 6: Evaluation scores for different size of Pythia models' FF and TopK FF-KVs.

|         |                 | Coder Status                    | Coder Status                    | Concept Detection               | Concept Detection               |
|---------|-----------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| FF Size | SAE Type        | Feat. Alive ↑                   | Expl. Var. ↑                    | Absorption ↓                    | Sparse Prob. ↑                  |
| 2048    | FF-KV TopK-FFKV | 1 . 000 ± 0.000 1 . 000 ± 0.000 | 1 . 000 ± 0.000 0 . 227 ± 0.000 | 0 . 060 ± 0.083 0 . 064 ± 0.127 | 0 . 802 ± 0.173 0 . 717 ± 0.204 |
| 3072    | FF-KV TopK-FFKV | 1 . 000 ± 0.000 0 . 999 ± 0.000 | 1 . 000 ± 0.000 0 . 129 ± 0.000 | 0 . 013 ± 0.038 0 . 014 ± 0.036 | 0 . 826 ± 0.140 0 . 779 ± 0.153 |
| 4096    | FF-KV TopK-FFKV | 1 . 000 ± 0.000 1 . 000 ± 0.000 | 1 . 000 ± 0.000 0 . 082 ± 0.000 | 0 . 003 ± 0.007 0 . 006 ± 0.011 | 0 . 803 ± 0.128 0 . 765 ± 0.126 |
| 8192    | FF-KV TopK-FFKV | 1 . 000 ± 0.000 1 . 000 ± 0.000 | 1 . 000 ± 0.000 0 . 277 ± 0.000 | 0 . 003 ± 0.009 0 . 003 ± 0.009 | 0 . 812 ± 0.194 0 . 770 ± 0.186 |
| 10240   | FF-KV TopK-FFKV | 1 . 000 ± 0.000 1 . 000 ± 0.000 | 1 . 000 ± 0.000 0 . 090 ± 0.000 | 0 . 001 ± 0.004 0 . 002 ± 0.009 | 0 . 850 ± 0.117 0 . 783 ± 0.144 |
| 16384   | FF-KV TopK-FFKV | 1 . 000 ± 0.000 1 . 000 ± 0.000 | 1 . 000 ± 0.000 0 . 047 ± 0.000 | 0 . 001 ± 0.003 0 . 001 ± 0.004 | 0 . 870 ± 0.119 0 . 807 ± 0.155 |
| 20480   | FF-KV TopK-FFKV | 1 . 000 ± 0.000 1 . 000 ± 0.000 | 1 . 000 ± 0.000 0 . 195 ± 0.000 | 0 . 002 ± 0.005 0 . 000 ± 0.001 | 0 . 818 ± 0.164 0 . 735 ± 0.157 |

|         |                 | Feature Explanation             | Feature Disentanglement         | Feature Disentanglement         | Feature Disentanglement           |
|---------|-----------------|---------------------------------|---------------------------------|---------------------------------|-----------------------------------|
| FF Size | SAE Type        | Autointerp ↑                    | RAVEL-ISO ↑                     | RAVEL-CAU ↓                     | SCR (k=20) ↑                      |
| 2048    | FF-KV TopK-FFKV | 0 . 727 ± 0.256 0 . 766 ± 0.277 | - -                             | - -                             | - 0 . 056 ± 0.464 0 . 000 ± 0.131 |
| 3072    | FF-KV TopK-FFKV | 0 . 734 ± 0.252 0 . 731 ± 0.271 | 0 . 411 ± 0.272 0 . 389 ± 0.218 | 0 . 033 ± 0.043 0 . 032 ± 0.043 | 0 . 093 ± 0.195 0 . 017 ± 0.114   |
| 4096    | FF-KV TopK-FFKV | 0 . 708 ± 0.252 0 . 716 ± 0.263 | 0 . 750 ± 0.132 0 . 739 ± 0.123 | 0 . 044 ± 0.051 0 . 039 ± 0.053 | 0 . 017 ± 0.054 - 0 . 001 ± 0.028 |
| 8192    | FF-KV TopK-FFKV | 0 . 714 ± 0.256 0 . 712 ± 0.271 | 0 . 900 ± 0.032 0 . 897 ± 0.031 | 0 . 035 ± 0.064 0 . 023 ± 0.040 | 0 . 047 ± 0.099 0 . 016 ± 0.038   |
| 10240   | FF-KV TopK-FFKV | 0 . 707 ± 0.259 0 . 704 ± 0.278 | 0 . 916 ± 0.037 0 . 915 ± 0.033 | 0 . 013 ± 0.025 0 . 013 ± 0.026 | 0 . 049 ± 0.130 - 0 . 017 ± 0.065 |
| 16384   | FF-KV TopK-FFKV | 0 . 702 ± 0.254 0 . 701 ± 0.265 | 0 . 879 ± 0.095 0 . 865 ± 0.122 | 0 . 041 ± 0.098 0 . 025 ± 0.046 | 0 . 023 ± 0.056 0 . 003 ± 0.018   |
| 20480   | FF-KV TopK-FFKV | 0 . 693 ± 0.252 0 . 698 ± 0.270 | 0 . 881 ± 0.072 0 . 854 ± 0.069 | 0 . 021 ± 0.031 0 . 019 ± 0.028 | 0 . 030 ± 0.050 0 . 022 ± 0.064   |

range into ten bins (e.g., 0.1-0.2, 0.2-0.3). From each bin, we sample ten features, each associated with ten pairs of input examples (for a total of 100 examples per bin), and annotate the proportion of aligned features within each bin. As shown in Figure 4, features with an MCS below 0.3 are almost never aligned, whereas those above 0.9 exhibit over 60% alignment.

## F.1 Superficial Features

We show examples of 'superficial' features here.

- Figure 29 shows the first FF-KV feature we annotate as superficial, activating on 'now'.
- Figure 30 shows the first TopK FF-KV feature we annotate as superficial, focused on 'the'.
- Figure 31 shows the first SAE feature we annotate as superficial, activating on 'return' in code.
- Figure 32 shows the first Transcoder feature we annotate as superficial, activating on alphanumeric token combinations.

## F.2 Conceptual Features

We illustrate features that activate on higher-level concepts or semantic themes.

- Figure 33 shows the first FF-KV feature we annotate as conceptual, activating on coastal concepts.
- Figure 34 shows the first TopK FF-KV feature we annotate as conceptual, linked to recipes and desserts.
- Figure 35 shows the first SAE feature we annotate as conceptual, activating on country and region names.
- Figure 36 shows the first Transcoder feature we annotate as conceptual, activating on college degree concepts.

## F.3 Uninterpretable Features

We also show examples of features without clear patterns.

- Figure 37 shows the first FF-KV feature we annotate as uninterpretable.
- Figure 38 shows the first TopK FF-KV feature we annotate as uninterpretable.
- Figure 39 shows the first SAE feature we annotate as uninterpretable.
- Figure 40 shows the first Transcoder feature we annotate as uninterpretable.

## F.4 Aligned Features

Figure 41 shows the first FF-KV feature we annotate as uninterpretable.

## F.5 Unaligned Features

Figure 42 shows the first FF-KV feature we annotate as uninterpretable.

Table 7: The list of assets used in this work.

| Asset Type   | Asset Name                    | Link                                                                                                                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                | Citation   |
|--------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| Code         | SAEBench                      | /github /github /github Link google/gemma-scope-2b-pt-mlp google/gemma-scope-9b-pt-mlp fnlp/Llama3_1-8B-Base-LXM-8x openai-community/gpt2 google/gemma-2-2b google/gemma-2-9b meta-llama/Llama-3.1-8B EleutherAI/pythia-70m-deduped EleutherAI/pythia-160m-deduped EleutherAI/pythia-410m-deduped EleutherAI/pythia-1.4b-deduped EleutherAI/pythia-2.8B-deduped EleutherAI/pythia-6.9B-deduped | License Not specified MIT License MIT License CC0 1.0 Universal Apache 2.0 Apache 2.0 Not specified Not specified Not specified Apache 2.0 MIT License Gemma License Gemma License 3 Community Apache 2.0 Apache 2.0 Apache 2.0 Apache 2.0 2.0 | [27]       |
| Code         | TransformerLens               |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [33]       |
| Code         | SAELens                       |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [7]        |
| Dataset      | OpenWebText                   |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [19]       |
| SAE          | Gemma-Scope-2B-pt-mlp         |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [30]       |
| SAE          | Gemma-Scope-9B-pt-mlp         |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [30]       |
| SAE          | Llama-Scope-3.1-8B-LXM-8x     |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [21]       |
| SAE          | GPT2-Small-32k-mlp-out        | jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs                                                                                                                                                                                                                                                                                                                                                      |                                                                                                                                                                                                                                                | [6]        |
| SAE          | Pythia-70m-deduped-mlp        | ctigges/pythia-70m-deduped__mlp-sm_processed                                                                                                                                                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                | [48]       |
| Transcoder   | Gemma-Scope-2B-pt-transcoders | google/gemma-scope-2b-pt-transcoders                                                                                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                | [30]       |
| Model        | GPT-2-small                   |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [39]       |
| Model        | Gemma-2-2B                    |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [43]       |
| Model        | Gemma-2-9B                    |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [43]       |
| Model        | Llama-3.1-8B                  |                                                                                                                                                                                                                                                                                                                                                                                                | Llama License                                                                                                                                                                                                                                  | [45]       |
| Model        | Pythia-70M-deduped            |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [4]        |
| Model        | Pythia-160M-deduped           |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [4]        |
| Model        | Pythia-410M-deduped           |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [4]        |
| Model        | Pythia-1.4B-deduped           |                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                | [4]        |
| Model        | Pythia-2.8B-deduped           |                                                                                                                                                                                                                                                                                                                                                                                                | Apache                                                                                                                                                                                                                                         | [4]        |
| Model        | Pythia-6.9B-deduped           |                                                                                                                                                                                                                                                                                                                                                                                                | Apache 2.0                                                                                                                                                                                                                                     | [4]        |
| Model        | Pythia-12B-deduped            | EleutherAI/pythia-12B-deduped                                                                                                                                                                                                                                                                                                                                                                  | Apache 2.0                                                                                                                                                                                                                                     | [4]        |

## G Use of Existing Assets

Table 7 shows the assets being used in this paper, with the type, name, link, license, and citation for each asset used in the paper.

## H Compute Statement

Most experiments presented in this paper were run on a cluster consisting of the NVIDIA H200 GPUs with 141GB of memory. All experiments on models are run using a single 141GB memory GPU. Evaluation time per layer differs largely on model size, with Pythia-70M, it takes approximately 1 hour, and for larger models like Gemma-2-9B, it takes approximately 4 hours per layer. The total GPU time for this work is approximately 1400 hours, including exploratory research stage.

<!-- image -->

Loading [MathJax]/extensions/MathMenu.js

Figure 5: Detailed feature alive scores on all tested models, across all layers.

Figure 6: Detailed explained variance scores on all tested models, across all layers.

<!-- image -->

Figure 7: Detailed auto-interpretation scores on all tested models, across all layers.

<!-- image -->

Figure 8: Detailed RAVEL scores on all tested models, across all layers.

<!-- image -->

Figure 9: Detailed RAVEL scores on all tested models, across all layers.

<!-- image -->

Figure 10: Detailed absorption scores on all tested models, across all layers.

<!-- image -->

25

Figure 11: Detailed sparse probing scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

25

Figure 12: Detailed sparse probing scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

25

Figure 13: Detailed sparse probing scores on all tested models, across all layers, with the same hyperparameter choice as the main result in Table 1.

<!-- image -->

Figure 14: Detailed SCR scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 15: Detailed SCR scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 16: Detailed SCR scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 17: Detailed SCR scores on all tested models, across all layers, with the same hyperparameter choice as the main result in Table 1.

<!-- image -->

Figure 18: Detailed SCR scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 19: Detailed SCR scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 20: Detailed SCR scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 21: Detailed TPP scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 22: Detailed TPP scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 23: Detailed TPP scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 24: Detailed TPP scores on all tested models, across all layers, with the same hyperparameter choice as the main result in Table 1.

<!-- image -->

Figure 25: Detailed TPP scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 26: Detailed TPP scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

Figure 27: Detailed TPP scores on all tested models, across all layers, and various hyperparameter ( K ) choices.

<!-- image -->

<!-- image -->

Mean Norm

Mean ± 1 Std Dev

Figure 28: Distribution of the L2 norms of all tested models' FF-KV decoder weights (i.e., W 2 in its FF sublayer). Although the norms are not exactly one, they are concentrated in a narrow range.

```
Text ID:6269 (Max Activation: 0.691) thatthe unemployment rate is actually 4 4 percent This number includes about 80 0 0 0 0 discouragedworkers who have given up seeking employment tandroughlyeight million workers who are in time jobsbut wouldprefer to work afull -timejob " The unemployment rate now stands at 9 percent To put this number in perspective， while that big improvement from the 10 percentr reached in late 2009 it is now higher than unemployment evergotint the 24 yearsbeforethe Great Recession," said Yellen in her prepared remarks
```

Figure 29: Top-4 activating examples for a particular feature in FF-KV annotated as 'superficial' . This feature specifically activates most on the word 'now', in various contexts.

```
Text ID: 962(MaxActivation:0.629) despite all the electoral spending，Ruth Davidson S Conservatives （TM) won just 1 3 seats no of 59 They the election uo the single issueof to another independence referendum，and they still lost . Improving onthe number of seats you achieved stilldoesn " t make you a winner ， any more than improving yourresultsinan arithmetic examfroman F to a D means that you re now top of the class Scotland looked at Ruth Davidson " S partya and stillgave themafailing grade all the Tory triumphalism in the world won t changet that simple ari thme tical reality
```

```
Text ID: 2495 (Max Activation: 0.520) re moving a pieceof functionalityout of a method that buop too many things it's very likely that theoriginal method now has extra references that just aren 'tneededanymore. In the midst of the ref actoring you may be So focused on the extraction，that you forget to clean these bitsup.And， by cleaning it up you may uncover a trail of other that can go into the trashbin as well Without this cleanup you ve sn! pappe thedebt of code to your system That might make the newly ref actored code base just as hard to
```

```
Text ID: 14099 (MaxActivation:0.500) He struggled to continue the paper，and though he asked for help he received none ThehomeofTravisand Ros anna，relocated toPer due Hill，Alabama andrestored in 1985 OnFebruary 27 Travis passed his law examination and received permission to legally practice he borrowed $55 .37 to open a law office,[12] as well as $ 90 earlier in the year to help pay for the Herald [13 Now in and with no practical income ， he took in three boarding students，and 1to help
```

```
Text ID:11936 (Max Activation: 2.609) Claudio Ran ieri as coach of the first team."Following this move the financial contract with Ran ieri，whose deal was coming toanendon30June2011，hasendedbymutualconsent.RomawishestothankClaudioRanieri for the professionalism shown and the work done[during his time at the club ]."Ran ieri enjoyed a successful first season with Roma afterreplacingLucianoSpal lettiinSeptember2009. The clubhadenduredanhorrendousstarttothecampaignandRanieri ，who had been fired by Juventus months earlier，rescued the teamandnearlyledthemtothescu
```

Figure 30: Top-activating examples for a feature in k -Sparse FF-KV annotated as 'superficial' . This feature specifically activates most on the word 'the', in various contexts.

```
Text ID:15318(MaxActivation:2.484) 59.7mAngelDiMaria--RealMadrid toManUtd，Aug.20146）f56mKaka--ACMilan toRealMadrid， June2009Sources toldESPN FC that Chelseahave been quoted aprice of40million pounds by PSG for Cav ani，and that the hasalreadybeenpreliminarycontactbetween the London club and the player's camp，although Chelsea are stillassessing the best course of action.WithMourinhohavingregularly complained
```

```
Text ID:7334 (Max Activation: 2.484) spokesman AdamRosen said heis'shocked'that an agency of firstresponders would enforce such an order the week ofSept 11. The foursuspendedfirefighterssaidthey weretoldthat the orderwasissuedbecauseofracialdiscord[in]the department The four，whoincludetwowhite firefighters，ablack firefighter，andafourthfirefighter whoisaCuban é migré， said nosuchproblemsexist,"wroteCBS，whichalsoreportedthat the four firefighters trace the issue "to a decision by several The newflagmysteriously
```

```
Text ID: 12010 (Max Activation: 2.453) postedareview onthewebsiteafter stayingat theBroadwayHotelinBlackpool.However the couplelaterfoundthatf100 charged to their credit card，which the BBC reported was theresult of ahotel policy in thecase of"bad"reviews The manager of the hotelwasnotavailableforcommentlastnight The Jenkins ons，who
```

```
Text ID: 11353(MaxActivation:18.412) zwd（'）d（'）g uoss=d+y'o）x return pmfTherangeof valuesinthecomputedP mfisfrom0tohigh.Soif thevalueoflamwereexactly3.4 ，we would compute :lam =3.4 goal _ dist = think bayes .Make Poisson P mf （ lam，1 0）丨 chose the upper bound，10，because the probability of scoring more than10 goals in a game is quite low.That′s simple
```

Figure 31: Top-activating examples for a feature in SAE annotated as 'superficial' . This feature activates on the word 'return', especially in programming-related contexts.

```
TextID:8952(MaxActivation:17.365) q-k+ 1)/k if k<= p:c list.append （c）d 1.0*(q-k+1)/(p+q-k+1)/kifk<=q:d list .append（d） return np array clist [:-11),np . array（d list [:-1 ]） def arg box （y， ymin ， ymax ， imin ， imax ):"find limits （we hope ）where y[i]is between ymin and ymax " ii = np. arg where（np. logical_ and（y !=（xew>'uw<
```

```
Text ID: 4503 (Max Activation: 17.178) asm_("f sel %0，%1，% 2，% 3":"=f"（result ）:"f"（test ),"f"（b),"f"（a ); return result }Suchoptimizationsareimplementationdetails，butaredescribedherebecausetheyprovideapractical performance benefit to the performance-conscioususer.It wouldbe nice if stdSTLimplementationsprovidedsuch things，though Met rower ks has been known to do so in some cases.Instead of just < algorithm >，EAST L has < algorithm.h >，< sort.h >，< algo set.h >，
```

```
Text ID:10231(MaxActivation:16.843) ！￥0L+0L%(0L%(0L%0L/L!+0L%000L/L!+0L%0000000L 0；printf（"% d"，i 2); return o.uab-sdm -equenbuab-sdm -yu1 p o-oob s~:!ey  Jasn user @kali:~$./ d link-wps-gen973 2932 9user @kali:~$Youcanfetchthisprogramathttps :/l
```

```
Text ID:9000 (MaxActivation:14.418) xos）sod-（sd）o）=]（x return S # extras : find a direction of maximum sensitivity u， s ，v = np. linalg . svd（ S， compute _ uv = True ）# largest singular directioninreverse order # tomatch polynomialcoefficients n-1:0 return S s[0],v[0,:-1 」S，kappa，V = find _root _ sen siti vities （Q，extras = True ）S，kappa，V（ array (I[- 0
```

```
Text ID:4591(MaxActivation:34.000) mightbe worthusingasareference.Afulldiscussionofcompiler inliningcharacteristics isoutsidethescope of this document，but some Internet discussions regarding GCC in lining problems can be found at : http :/l groups. google.com/group/comp.lang.c++/browse_frm/thread/b74eed 6 bd 8 d 4 2e http:/ll groups. google.com/ group/ fa. linux.kernel/ browse_frm/ thread/ 186 1b 26 3 cd fa68a/http://www.pixel glow.com/ lists/archive
```

Figure 32: Top-activating examples for a feature in Transcoder annotated as 'superficial' . This feature activates on the combination of digits and alphabet, in various contexts.

```
Text ID:3930 (MaxActivation:33.750) buy allof the parts needed，including the plastic case，knob，and AC adapter.You can edit your cart after com/Project Manager / Project Detail.aspx?AccessID=b68a 3023 orhttp://www.mouser.com/Tools /Tools.aspx andenter thisaccesscode:b68a3023 UpgradesIamoftenaskedwhatcanbedoneto upgrade the designs that |publish.In this case，there
```

```
Text ID:6968 (MaxActivation:32.250) an environment variable JE BlO _ APl KEY，or pass it as a parameter if you are importing the script as a library ). Queries return JSON output，except for download requests，that return binary attachments.The return"code" variableissetto 0on success，!=0on error.Here are a few examples:Querya file hash:$ jeb io.py check42aaa93 a 8 a 6 bf cbc 3 b09e4ea9f7 428 42aaa93 8 9 4a 6
```

```
Text ID:6971 (MaxActivation:31.500) asta.apk"}}Note:theuser details section ispresent only if youup lola ded the file yourself.Upload a file: ）：lyepeoumoapnuapeoldn.'oapoo}:ydeydepeondnd·oqa$ subject topermission）$jeb io.pydownloada 2 ba 1b acc 6b 2c93089692bf5f3 0 d68a2ba1b acc 6 96b
```

```
Text ID: 6561 (Max Activation: 1.891) oftheCanadianmainlandisPointPelee，Ontario.[47] The westernmostpointisBoundary Peak 187(60°1 8'22.929"N, 141°00'7.128"W）atthesouthern end of the Yukon = Alaska border whichis roughly following W but leans veryslightlyeast as it North .[48][47] The eastern most point is Cape Spear， Newfoundland（47°31′N w） .[4
```

Figure 33: Top-activating examples for a feature in FF-KV annotated as 'conceptual' . The specific annotation was 'concept related to the coast' especially in various contexts.

```
Text ID:9818 (MaxActivation: 1.617) positions as a result Jo aquinwillmove northward much of this weekend roughly paralle ling the East coast There isnearlyequal possibility thestormwillmake land fall along the -Atlantic coast the New England coast or veerout tosea Due tothe potential close proximity of the hurricane to the coast people fromtheCarolinas to Massachusetts willneedtocloselymonitorthe track and strength of Joaquin for high wind and coastal flooding concerns Should Joaquin make landfall ， areas near pue north ofthe center pinoM face the worst coastal and strong winds Ifthe storm were to make landfall in North Carolina
```

```
Text ID:12657(MaxActivation:1.547) copaskedhimwhy he wasn t driving.Hesaidhedidn·t havea truck anda horsetrailer，just ahorse，apack horse andadog.Hisplanwassimply to the coast to SanDiego and turn left He had what he called shoulder pass which he drew from his pocketand presented to the officer The officer being confused was not even sure such and examineditsmolecularstructure Then theLagunaAnimalControlofficer showed up.That officer informed the cow poke that he did not have his dog on aleash.Something all good little（ citizens of California
```

```
Text ID: 13010 (Max Activation: 1.531) picked up from the Visitor Center "Walking Tour of HistoricDowntownCorsicana" Onpage four，the tale ofthe wandering Jewish rope walker. Lord，Nancy，i listen tothis.Atdusk，if noone'sdown thereworking，I'Ildescend tothe and stand for a while int the southeast corner whereDoug usually spent hissparet time.Withwindows uado on two sides . There is aMain Street butt the real main street is Beat on Street which runs along the east side of thebuilding From the south side east mostwindow right behind Doug as he worked you get
```

Figure 34: Top-activating examples for a feature in k -Sparse FF-KV annotated as 'conceptual' . The specific annotation was 'concept related to recipes' especially in contexts related to deserts.

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

```
Text ID:14508 (MaxActivation:14.855) Idid with the Codex.Wi RED:How do you deal with technology?Sera fini:|remember my first encounter with a tablet.Iwas working on theopening titles of twoItalian television broadcasts，O nda Verde and Enzo Bi agi's La L unga Marcia about his journey through China It was a new tool，wired to a giant - size computer - quite fascinating at the time.I used it recently to illustrate Nature Stories by Jules Ren ard，but
```

Figure 35: Top-activating examples for a feature in SAE annotated as 'conceptual' . The specific annotation was 'name of country and region' in various contexts.

```
Text ID: 6117(MaxActivation:13.967) this print are all linked by their support for the Revolution.The women were distinguished for ref uting Burke in print，or so it seemed .Wiliams who was noted for her sympathetic，eyewitness Letters Written in France hadjustpublishedapoeminpraiseof thestormingof theBastille. Catharine Macaulay's forthcoming attack on Burke′s Reflections had been announced and Barba uld，who had first opposed Burke in March 179 0，wasassumedtobewritinganother ref utationof hisReflections.While
```

```
Text ID: 13267 (Max Activation: 13.466) abroad including in Israel and the OPT This includes guidance on Israeli settlements.Weare advising British businesses tobear inmind the British Government's view onthe illeg ality of settlements under internationallaw whenconsidering their investments andactivities in the region This is voluntary guidance to British businesses on doing business in Israel pue OPT .Ultimatelyit willbe thedecision of anindividualor company whether tooperate in settlements in theOccupied Territories ，but the British Govermment would neither encourage nor offer support to such activity.When approached by businesses，we set out the UK's clear position on
```

```
Text ID: 9248(MaxActivation:13.382) the official said.Manchester United goalkeeper Sam Johnstone is poised to rejoin Aston Villa on Monday.The2 4-year -old has been a target for a number of Championship and Premier League sides but willsign for Villa on a season -long loan.John stone，who spent the second tour partythattravels to the USA uo
```

```
Text ID: 7155 (Max Activation: 13.311) 980s，Evergreen transported U.S.troops on drug raids in Central and South America the paper said.Over the years，company oficials denied working for theClA.When contacted Wednesday byThe Providence Joumal todiscussEvergreen'srelationshipwith theCIA，aspokesman private businesses.Ever green filed for liquidation under Chapter 7 of the U.S.Bankruptcy Code on Dec.31，20 13，in the Delaware District.
```

Figure 36: Top-activating examples for a feature in Transcoder annotated as 'conceptual' . The specific annotation was 'concept related to college degrees' in various contexts.

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

```
Text ID:1669 (Max Activation: 0.746) S specialized inDark movessor maybe dear Godwhat have|become?Theygotme. m hooked on Pokemon and up!p 'tevenseeitcoming. Strangehowmucha poob motivatorcanimprovea game S overall appeal. Suddenly the arena battles weren tso pointless was working towards something Collecting Pokemon started tomatter because the more powerful ones could help me beat challenges to morepoints toupgradeZoroark.My Zoro ark My precious ... Beyond the storyline that can be beat in roughly six hours of playtime there are tons of little motiv ators built
```

Figure 37: Top-activating examples for a feature in FF-KV annotated as 'Uninterpretable' .

```
Text ID:5179 (MaxActivation: 0.691) feasiblethatthe organisationmayannounce somekindof second Season Pass assumingthattheplayerbaseisstillthere Whatever thecase ，anewpatchwilldeploynextweek，adding Elite Driver levels improvements to the title S Photo Mode andmuchmore.Wherewouldyouliketoseethe9 game xuo6 Takea sharp hairpin turn in the comments sectionbelow · Melania 's official White Houseportrait hasbeen released Here it is And ... she looks like a model . Probably because she aqoposn one So The BostonGlobeimmediatelypickedthewholethingaparttopto bottom
```

```
Text ID: 9857 (Max Activation: 0.664) relatives of thehospital'smedical，nursing and administrative personnel.Among them five refused toparticipate， whereas five had mini -mental examination scores Z> and thereforewere excluded from the study .Thus，the control group consisted of 0HCs（103females/7 males unrelated to thepatients who were age 一 sex matchedwiththep SS dnob Exclusion criteriaincludedthepresenceofautoimmune disease ， neuro psychiat ric history，use of psych otropic drugs，alcoholism and drugabuse The first part of the questionnaire the main demographiccharacteristicsas wellas smokinganddrinkinghabits ： The second part of the
```

```
Text ID:12993(MaxActivation:0.645) role in supporting the newcomers at 100W evenas the arts same way.The arts council is more focused on arts education in NavarroCounty schools than producing or art these days， Brooks said."ItI makesnoS sense ， he said "—' m an old educator，so丨get doing itfor the kids ， but thereare adultswithartistictalent that to be nurtured " In his study at home，Brooks keeps a caricature someone made of him back when he was helping start the arts council.The character is pulling a wagon fullofstatues
```

## Text ID:7567 (MaxActivation:0.676)

).丨suggest that this flter plays a significant role in explaining the diferent patterns of deference exhibitedby conservatives andliberals.The dispositiontodefer to others withwhom we share apoliticalor religious outlook is continuouswiththedispositiontousebenevolenceasacuetoreliability;wearedisposedtoseethosewith whom weshareapoliticaloutlook and/ or areligiousafiliationas thosewhoarebenevolent towardus andour interests （ or，perhaps，the disposition to use benevolence as a cue to reliability is just a special case of a disposition to defertothosewith whom wesharevalues).Theuseofpoliticalandreligiousaffiliationasaproxyforbenevolence is

<!-- image -->

Text ID: 7727 (Max Activation:0.609) ，Idon't even know who Thomas Mars is and Inever have the phone records，"Robertson said."丨never find thatcall."TensofthousandsofRomaniansandBulgarianshavecometotheUKtoworksincerestrictionswere lifted Peter Nicholls/The TimesNet migration has reached arecord336，000as figures published yesterday showedthatRomaniansarenowthe third biggest group comingto theUK.Oficials said thelatest increase was up jobs. The surge was partly because

<!-- image -->

```
Text ID:8655 (Max Activation: 0.605) Iranthreat.Checkout thefullsegment below.Heisnow oneof fiveactivecoaches in theACC and oneof Avent recorded his1，00 0 th career win against North Carolina Central Tuesday evening，as the Wolf pack hit a seasonasaheadcoachand21statNCState，Avent
```

## Text ID: 5332 (Max Activation:0.605)

. They reported that Einstein was right ． Since then ， his theory has been ret ouched in detail， but its essentials have been repeatedly verified.No important scientist is to be found among the skep tics ，although there is every incentive to deb unk Einstein，if it can be done.lmmort ality awaits the man who can overthrow Einstein.The popularuproaroverthetheorysurprisednoone more thantheauthorofthetheory.Hehadbeenalmostareclu se.Hiscontacts hadbeenwithquiet，scholarlymenof his owntype，andhissuddengloryappalledhim. Interview ers，photographers，lion -hunters ，cause -prom oters，testimonial

Figure 38: Top-activating examples for a feature in k -Sparse FF-KV annotated as 'Uninterpretable' .

```
Text ID: 15344 (Max Activation:4.632) Aaron Gle eman of Hard ball Talk）reports C.K.purchased an East End mansion that Babe Ruth spent time at. Ke il reports the comedian shelled out $ 2.49million for the 4，957-squarefoot"Prim roseCottage" formerly visited by theNew York Yankeeslegend.Thehomeisa three-story Tudor originally constructedin19 0 1with six bedrooms，three- and -a-half baths andfivefireplaces Yes，therearemorefireplacesthan bathroomsinthishome，whichisprobablyazoningrequirementintheHam ptons（or oneofRuth'secc entri
```

## Text ID:12336(MaxActivation:4.299)

the new ones.Additionally，ssh server key theft is another one -time vector that can beused to quicklybootstrap intonodekey theft.For thisreason，node admins shouldalways usesshkeyauthfor tor nodeadministration accounts，since it prevents ssh server key theft from implying continuous server compromise:http:// www.gre m well .com/ssh-mit m-public-key-authenticationIssuesWithEphe meralIdentityKeysThereareafewissueswith deployingephemeralidentitykeys.IssuesWithEphe meralIdentity Keys ClientguardnodelossTheprimaryissue

```
Text ID: 8158 (Max Activation: 4.289) s     a b  o    ds  d The United States cannot be trusted.Third，the U.S-South Korean alliance is not impenetrable.President Trump tweeted his criticism about theSouth Korea -UnitedStates free tradeagreement around the time of the4 th ，as 丨 have told them， that their talk of appea sement with North Korea will not work， they only understand one thing！ Donald J.Trump(@ real Donald Trump
```

Figure 39: Top-activating examples for a feature in SAE annotated as 'Uninterpretable' .

```
Text ID: 4557 (Max Activation: 4.193) values[128]）{for（inti=0;i<128;++i）mTable[i]=((values[i]^0xff80) +128 ）-i；}booloperator(（inta，intb）const{returnm Table [b]<mTable[a];}intmTable[1 28];}; std::sort（v，v+128，TableBasedSorter（values);The
```

```
Text ID: 9130 (Max Activation: 7.969) ，but their authorsarewell-re know ned.Linear Systems andState-SpaceSorry，Idon'thaveanythingIcanrecommend here what 1' ve read IS air TheDah lehs atMIT fallintothis latter category.）Linear Systems TheorybyJoaoP.Hespan hamaybeofuse.Chapter1is freely available online，and丨found it useful for the various state- spaceinter connections.But it looks likeatypicaltextbook over relianceon MATLAB.Oh，andIcan'tstand theheading
```

Figure 40: Top-activating examples for a feature in Transcoder annotated as 'Uninterpretable' .

```
Text ID:10600(MaxActivation:7.719) ulus AG.The announcement was made official by John Han ke，Director of Google Earth & Maps.Stefan and Bruno Mu ff founded End ox on.A large number of the employees of End ox on are still with Google working as Software Engineers， Managers，and V Ps.36.4%of Xun lei（Price:$5 million Date:January 4，2007 un lel IS aChinese file-sharing website that supports Bit Torrent，FTP，e Donkey，etc Xun lei was developedbyThunderNetworkingTechnologies andisbasedinthesouthernprovinceofShenzhen Xun
```

```
Text ID:6250(MaxActivation:7.469) :$135billionformedicalcareandbenefitsofIraqandAfghanwarveterans.Anestimated$743billioninadditionsto thePentagon's base budget.Although these funds were not spent directly in the war theaters，the researchers believe they would not have been appropriated had the wars not been undertaken.$ 4 5 5 billion for homeland security Again，the assumptionismade that much or all this spending PinoM not have been undertakenbutforthewarandclimateofwar. $130billioninadditionalspendingonwaroperationsandwar-relatedbasebudgetfor2014.
```

```
Text ID: 15443 (MaxActivation: 7.406) remaining cardsinthedeck in the middle of the playing area.Leave room next tothedeck for adiscard pile.Third，deal outtheRecipecards，face-up，intocardpilesinthecenter of the playing area.The number of piles will beonemorethan thetotal number of players.For example，if playing a game with four players，there willbe five Recipe piles.Fourth，take the Season cards and put them intheirproperorder Spring，Summer，Fall，andthenWinter).Select aseasontostartwith and place this card face-up.TheSeason deck should remain face
```

## FF-KV

<!-- image -->

Crossing New Leaf am iibo cards willalsobecome availablefor purchase No matter how youplay it，this isthe perfect time to cozy up to the charmand creativity of this special game New friends and discoveries await every day.Express yourself by customizing your character，home，and town as you create your ideal world .The arcade:t thelastbastion of loose change.Or sowethought. Anold-school arcade.in the retirement capitalofNewZ Zealand no less ， complete withPac -Man，pinballandteddy bear claw machines has started using cryptocurrency And actually.it·s not that cryptic

## Transcoder

<!-- image -->

Crossing New Leaf am iibo cards will also become available for purchase . No matter how you play it ， this is the perfect time to cozy up to the charm and creativity of this special game. New friends and discoveries await every day .Express yourself by customizing your character，home, andtownasyoucreate your idealworld.Thearcade:

the last bastion of loose change. Or so we thought . An old - school arcade，in the retirement capital of New Zealand no less，complete with Pac -Man，pinballand teddy bear claw machines ，has started using cryptocurrency.

And actually，it's not that cryptic.

Figure 41: The first feature pair we annotate as aligned .

<!-- image -->

## FF-KV

## Transcoder

Figure 42: The first feature pair we annotate as un-aligned .

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our main claim is that the SAE-based approach provides comparable interpretability to feature vectors stored in feed-forward layers (FF-KV). We diligently investigate this claim across multiple LLMs and their corresponding SAEs, along with Transcoders, through both automatic, extensive evaluation § 4 and manual evaluations § 5. All results demonstrate high similarities between FF-KVs § 4.3 and SAEs § 5.2. We further analyze the overlap between Transcoder features and FF-KV features § 6.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper includes a dedicated 'Limitations' paragraph in the 'Conclusion' section (Section 7). We discuss several limitations of the study, including that the feature dimension of the SAEs and Transcoders used in this work was fixed; the results for the Transcoders are not available for all models because not every model is accompanied by one; and our qualitative analyses are limited to case studies.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational eMLPiciency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best

judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA]

Justification: This paper does not involve theoretical results.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it aMLPects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide details of all metrics and models, as well as the experimental setup, in Section 4. We also provide detailed info on the SAEs we used in Appendix D. Additional information on the implementation details is presented in Appendix A. We also release the code at https://github.com/muyo8692/ff-kv-sae to facilitate maximum reproducibility of our main results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suMLPice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provide open access to our data and code.

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

Justification: We provide details on exprimental settings for our main result in § 4.2 and Appendix D. We also provide additional information on metrics used in Appendix B.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The figures displaying as our main result (Table 1), as well as all detailed figures shown in Appendix C, all include with 2-sigma error bands, calculated by SEM =

√ ∑ ( x i -¯ x ) 2 n ( n -1) . These are also reported in the text in Section 4.3 and Appendix C.

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

Question: For each experiment, does the paper provide suMLPicient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We describe the compute used in Appendix H.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: To the best of our knowledge, the research conducted conforms with the NeurIPS Code of Ethics. We explicitly discusses potential negative social impacts and includes an ethics statement in Section 7.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We explicitly discuss potential negative societal impacts in Section 7 Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the eMLPiciency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: We did not find cause to believe our methods are at high risk for misuse, and therefore did not feel that additional safeguards were warranted.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing eMLPective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith eMLPort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All existing open-source models, datasets, and evaluations that we use are cited. We specify the asset type and license type in Appendix G.

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

Justification: The paper does not introduce or release any new datasets, code, or models.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No crowdsourcing experiments were performed. The human raters who carried out the qualitative interpretability assessment in Section 5 and Section 6 were the authors of this paper and colleagues at the lab, respectively. For assessment carried out by authors, care was taken in the design and execution of this experiment to ensure that no authorial bias would influence the results. For that was done by our colleagues, we include the detailed evaluation criteria we ask them to follow in Section 6.1.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve human subjects-see item 14.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The LLM is used only for writing and formatting purposes.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.