## Guiding LLM Decision-Making with Fairness Reward Models

Zara Hall

Columbia University zyh2000@columbia.edu

## Melanie Subbiah ∗

Columbia University m.subbiah@columbia.edu

## Kathleen McKeown

Columbia University kathy@cs.columbia.edu

Thomas P. Zollo ∗

Columbia University tpz2105@columbia.edu

Richard Zemel

Columbia University zemel@cs.columbia.edu

## Abstract

Large language models are increasingly used to support high-stakes decisions, potentially influencing who is granted bail or receives a loan. Naive chain-ofthought sampling can improve average decision accuracy, but has also been shown to amplify unfair bias. To address this challenge and enable the trustworthy use of reasoning models in high-stakes decision-making, we propose a framework for training a generalizable Fairness Reward Model (FRM). Our model assigns a fairness score to LLM reasoning, enabling the system to down-weight biased trajectories and favor equitable ones when aggregating decisions across reasoning chains. We show that a single Fairness Reward Model, trained on weakly supervised, LLM-annotated examples of biased versus unbiased reasoning, transfers across tasks, domains, and model families without additional fine-tuning. When applied to real-world decision-making tasks including recidivism prediction and social media moderation, our approach consistently improves fairness while matching, or even surpassing, baseline accuracy.

## 1 Introduction

While the most visible applications of large language models (LLMs) are in open-ended dialogue, LLMs are increasingly being used in a supporting role for decision-making , where they might recommend bail conditions, flag suspicious transactions, or triage resumes [28]. Compared with traditional statistical pipelines, LLMs can synthesize heterogeneous evidence, generate rationales, and explore diverse solution paths through inference-time sampling before committing to a final answer [57]. Recent work shows that scaling the number of sampled chain-of-thought (CoT) trajectories and then aggregating or verifying them can substantially boost predictive accuracy in mathematics, coding, and various planning tasks [11, 52, 54]. The same paradigm seems likely to unlock similar efficiency and accuracy gains in high-stakes decision-making [13].

Yet accuracy alone is insufficient. Decisions about liberty, employment, credit, or housing are governed by anti-discrimination law and public trust; practitioners must demonstrate that both the outcomes and the reasoning processes of automated systems are fair [8]. Unfortunately, naive CoT sampling can amplify social biases: models that enumerate many rationales may surface and then use compelling stereotypes as a basis for their decisions (see Figure 1) [39]. While explicit fairness prompting can partly mitigate this issue, prompting is brittle and does not ensure that the underlying reasoning process is fair [43].

∗ Equal contribution.

Figure 1: Scaling inference-time compute, such as by sampling multiple chain-of-thought (CoT) solutions, consistently boosts predictive accuracy. However, this extra compute does nothing to correct underlying biases and can even exacerbate unfairness by surfacing stereotyped reasoning (as in reasoning chain 1).

<!-- image -->

To bridge this gap, we propose a novel framework for training a generalizable Fairness Reward Model (FRM) that can be applied to a variety of downstream tasks in order to improve the quality of decision-making. Our Fairness Reward Model assigns a real-valued fairness score to each LLM reasoning step, allowing the final decision to down-weight biased trajectories and up-weight equitable ones. We show that a single Fairness Reward Model , trained on weakly supervised LLM-annotated examples of biased versus unbiased reasoning, generalizes across domains and models. At inference time, our method samples N CoT traces, scores every step with the FRM, and aggregates completions with a temperature-controlled softmax that balances consensus and fairness. Because scoring is performed after all chains have been generated, our method leaves the model's internal reasoning untouched and intervenes only in the aggregation stage. In doing so, our approach offers the flexible control over fairness/accuracy trade-offs that can be elusive with prompting-based approaches, and to our knowledge has not been demonstrated by any fine-tuning approach [61].

Despite using weakly supervised labels, and only requiring a modest amount of training, we find that our learned FRM transfers remarkably well across tasks, domains, reasoning models, and protected attributes. Additionally, we evaluate how well our LLM labels of bias align with human judgments and find substantial agreement, further validating our weakly supervised training approach. With a single model, trained once on a synthetic corpus, we obtain substantial fairness gains across three disparate decision-making domains: (i) recidivism prediction (COMPAS [2]), where the false-positive gap between African American and white defendants drops by 25-75% while accuracy is maintained; (ii) social-media moderation (Civil Comments [10]), where religion- and orientation-based disparities shrink by up to 40%; and (iii) job-candidate screening (Bias-in-Bios [15]), where gender gaps narrow by more than 20%. These results demonstrate how inference-time compute can be harnessed not just for accuracy, but for scalable, portable fairness , opening a path toward trustworthy, reasoning-based LLM decision-makers.

Our contributions include:

1. We introduce a FRM for supervising LLM decision-making, retaining the accuracy benefits of scaling inference-time compute while reducing bias in the final outcomes.
2. We show that our FRM reduces biased reasoning in important downstream tasks (predicting recidivism, content moderation, and screening job candidates) and across different protected attributes, including race, religion, and gender, as well as different reasoning models.
3. We explore and ablate design decisions, finding that stepwise weak labels are effective supervision for training process reward models on this task, and using temperature-based weighted majority scoring balances accuracy and fairness.

Our code is available at https://github.com/zarahall/fairness-prms .

Figure 2: Our framework for training and applying a generalizable Fairness Reward Model includes five high-level steps. In the first phase (spanning steps 1-3), we train a generalizable Fairness Reward Model to label bias in LLM reasoning steps. In the second phase (steps 4-5), we apply our model to score reasoning chains in diverse downstream decision-making tasks and use these scores to produce a final decision, leading to fairer outcomes.

<!-- image -->

## 2 Related Work

LLM reasoning Recent advances in language model performance on complex reasoning tasks can be viewed as being driven largely by three approaches [42]. First, improved prompting methods such as CoT [54] and its extension tree-of-thought (ToT) [57] enable models to explore multiple reasoning paths. Second, the development of response verifiers allows for systematic selection of outputs, primarily through process reward models (PRMs) that supervise individual reasoning steps and outcome reward models (ORMs) that supervise answers produced by full reasoning chains [12, 29, 45, 48, 50, 51]. Third, fine-tuning with reinforcement learning can optimize reasoning on specific tasks [33, 59, 60]. The biggest improvement from scaling test-time compute has been on math reasoning tasks, where correctness is well-defined and easy to verify [23, 40]; such domains naturally favor ORMs. In fairness settings, no such verified reward exists, so we instead turn to step-level reward modeling to supervise reasoning without ground-truth outcomes.

Bias and fairness in LLMs Despite dramatic gains in language understanding and reasoning, LLMs still inherit and amplify societal biases present in their pre-training data [6, 21, 27]. Empirical studies have documented disparate behavior across race [1, 16], gender [26, 44, 49], religion [37], socioeconomic status [41], and other protected attributes. Such disparities are especially problematic in high-stakes domains such as employment, housing, credit, and criminal justice, where discriminatory outputs can breach anti-discrimination law and erode public trust [4]. Recent work shows that chain-of-thought (CoT) prompting, though beneficial for accuracy, can surface harmful stereotypes and exacerbate bias [25, 39]. Furthermore, explanations generated by LLMs are often unfaithful to the model's true reasoning process [47], and jailbreaks that fail in zero-shot settings can succeed once CoT is enabled [7]. Even ensemble strategies such as majority voting over many CoT traces may entrench rather than alleviate these disparities, as many samples may contain similar or overlapping biases [58].

Common mitigation approaches span pre-training interventions [30, 34], instruction-tuning and fine-tuning approaches such as Constitutional AI [3, 20], and fairness-aware prompting strategies [24, 31]. However, fairness prompts are brittle, inconsistently followed [32], and can reduce output diversity [19]. Reward model-based supervision has recently emerged as a powerful tool for shaping LLM behavior, yet prior work targets factual correctness or harmlessness rather than fairness [42]. Our contribution differs by introducing a process-level Fairness Reward Model that scores individual reasoning steps for bias, enabling re-weighting of CoT trajectories to reach a fairer final decision. By directly supervising the reasoning process and demonstrating transfer across models, tasks, and domains, our framework complements existing debiasing methods and offers a scalable path toward equitable multi-step LLM decision-making.

## 3 Fairness Reward Model

LLMs are increasingly entrusted with decisions in domains where how a conclusion is reached may matter as much as what that conclusion is. For example, LLMs used to screen job candidates or housing or loan applicants are subject to traditional anti-discrimination laws and principles such as disparate treatment [62]. Their trustworthy deployment, then, depends on some means of ensuring

Figure 3: Our framework samples a BBQ [35] question and generates reasoning chains to be labeled for bias by a powerful LLM. BBQ questions often provide limited information, which may cause a model to form a conclusion based on stereotypes rather than recognizing there is not enough information to answer the question. In this shortened example from our dataset, reasoning chain 1 falls into this trap, while reasoning chain 2 avoids stereotyping.

<!-- image -->

their fair treatment of different protected attributes (e.g. race, gender, sexual orientation). To surface and down-weight biased reasoning before it crystallizes into an outcome, we introduce a Fairness Reward Model that scores every step of a chain-of-thought and then aggregates candidate answers in proportion to their process-level fairness.

Our framework for training and applying a generalizable FRM includes five high-level steps (see Figure 2): (1) generate reasoning chains, (2) label reasoning steps for bias, (3) train Fairness Reward Model, (4) score reasoning steps on inference CoT examples, (5) aggregate answers based on fairness of reasoning. In this section, we will both give an abstract description of our framework, and also describe our ultimate design decisions and the details of our proof-of-concept implementation.

In order to facilitate future research in this area, our dataset 2 and trained FRM 3 are publicly available.

(1) Generating reasoning chains Let X denote a set of decision-making prompts and let Z be the space of reasoning steps (e.g., individual thoughts in a CoT reasoning chain). For each prompt x ∈ X , we use a base LLM to sample a collection of n independent reasoning chains

<!-- formula-not-decoded -->

where T k is the (variable) length of the k th chain and z k,t ( x ) ∈ Z is its t th step. Each chain ends with an answer a k ( x ) ∈ A , where A is the task-specific answer space (e.g., {yes, no}). These chains form the raw corpus from which we will distill fairness supervision.

A key choice in this step is the source of data for the input prompts. The input prompts need to: (1) require the LLM to reason to responses; (2) belong to a large enough dataset to collect many reasoning chains; (3) produce reasoning with implications for bias or fairness across many groups. A dataset fitting these criteria allows us to train a reward model that generalizes across fairness domains and reasoning models. To meet these criteria, we use the Bias Benchmark for QA (BBQ) [35] dataset as the primary source for generating training data. BBQ contains 50,000 questions that target 11 social biases including race, gender, age and intersectional identities (an example BBQ question is shown in Figure 3). We select a subsample of 4395 questions; for each, we sample between 32-256 reasoning chains (with temperature 0.8) using four LLaMA models: LLaMA-3.1-8B-Instruct, LLaMA-3.1-70B-Instruct, LLaMA-3.2-1B-Instruct, and LLaMA-3.2-3B-Instruct [46]. This mix of small and large models gives some diversity to our training data, ensuring that it contains a diverse

2 https://huggingface.com/datasets/zarahall/fairness-prm-training-data

3 https://huggingface.com/zarahall/fairness-reward-model

set of high-quality reasoning chains and biased reasoning. In total, we generate 255,000 reasoning steps from the approximately 4,000 BBQ questions used as prompts.

- (2) Labeling reasoning steps for bias Ideally we would possess a ground-truth indicator

<!-- formula-not-decoded -->

for every step z ∈ Z . Because such labels are expensive, we instead employ a weak labeling function ˜ Y : Z → { 0 , 1 } . For this, we use an off-the-shelf LLM judge, GPT-4o-mini, to bootstrap supervision at scale (we evaluate other weak labeling approaches in Section 6.3). For each sampled chain segmented into atomic reasoning steps, we prompt GPT-4o-mini to flag whether each step relies on protected-attribute stereotypes or other unfair heuristics, yielding a binary unbiased/biased tag. The full prompt is included in Appendix B.1. This automatic process provides weak labels for our training corpus of 255,000 reasoning steps, of which 201,500 are marked unbiased and 53,500 biased .

While LLM judges inevitably carry some of the biases present in their opaque internet-scale pretraining data, we contend that their judgments still provide a sufficiently informative signal to train our model effectively. To validate the quality of these labels, we run a small human study, asking three of the authors of this paper to label a random sample of 100 reasoning steps each. GPT-4o-mini matches the human annotations on 75% of the examples, compared to human-human agreement on 88%. While the LLM-human agreement is lower than human-human, these results still indicate substantial agreement. We detail this study, report pairwise agreement, and provide qualitative observations of disagreements in Appendix A.1 and Appendix B.3.

(3) Training the Fairness Reward Model Given the weakly labeled dataset D = { ( z i , ˜ y i ) } |D| i =1 , we fit a Fairness Reward Model f θ : Z → R via the binary cross-entropy objective

<!-- formula-not-decoded -->

where σ is the logistic function. This objective is analogous to PPO [38] reward-model training, except here the 'preferences' are binary and represent fairness. We initialize our reward model training from a LLaMA-3.2-1B-Instruct base model; this model scale enables efficient test-time scoring. Following the training procedure outlined by [42, 51], we train with binary cross-entropy loss and use the AdamW optimizer with a learning rate of 2e-5, a batch size of 128, and β parameters (0.9, 0.95).

- (4) Scoring reasoning steps in downstream inference At inference time, we draw n test chains { z k ( x ) } n test k =1 for the new prompt x . Each step receives a fairness score f θ ( z k,t ) , and the chain-level score is the mean

<!-- formula-not-decoded -->

Scoring incurs O ( n test T max ) calls to f θ , negligible compared to LLM generation with CoT prompting. We note that our goal is not to terminate or edit a chain when an unfair step is detected; all reasoning is preserved for accuracy and auditability of the final decision.

(5) Aggregating final answer To aggregate the final answer over the n test reasoning chains, we convert the chain-level scores into weights

<!-- formula-not-decoded -->

and compute the final answer ˆ a ( x ) by a weighted vote over the n test candidate answers { a k ( x ) } n test k =1 emitted at the ends of the chains:

<!-- formula-not-decoded -->

Figure 4: Validation results for baseline methods and our FRM applied to held-out BBQ data.

<!-- image -->

The temperature τ balances the accuracy gains from CoT with self-consistency (uniform weights as τ →∞ ) against strict fairness optimization ( τ → 0 ). Combining final answers from all chains, as opposed to returning the decision from only the most fair, ensures that some of the accuracy benefits from CoT sampling are retained. Because scores are transparent and step-localized, practitioners can trace any unfair outcome to the exact line of reasoning that caused it.

## 3.1 Validation Results for FRM on BBQ

Although our goal in this work is to train a single model that generalizes to many different tasks and distributions, as a sanity check we first validate the performance of our method on held-out data from the training distribution. Results are shown in Figure 4; we compare our method to typical CoT prompting, and majority voting applied to a set of 32 CoT samples. Our FRM performs better than the baselines in each bias category, and, on average, produces an absolute improvement of more than 25% accuracy relative to majority voting. Since fairness and accuracy are coupled for BBQ, this increase in accuracy directly indicates increased fairness.

## 4 Downstream Decision-Making Tasks

To test whether a single FRM can transfer beyond its BBQ training distribution, we evaluate it in three important real-world decision-making settings: criminal recidivism prediction, social media content moderation, and job candidate screening. Each domain comes with a well-known dataset (COMPAS, Civil Comments, and Bias in Bios) and is labeled with one or more protected attributes, allowing us to measure both accuracy and group fairness gaps under realistic stakes. The remainder of this section describes the task, data, and fairness-relevant structure of each benchmark.

Predicting criminal recidivism LLMs and other machine learning tools are increasingly being used to support judicial decisions, including bail recommendations and recidivism risk assessments [17]. A prominent real-world example is the use of the COMPAS system by U.S. courts, which gained attention for disproportionately labeling African American defendants as high-risk, even when controlling for prior offenses [2]. This highlights the critical risk of racial bias when ML systems are used in high-stakes legal contexts. To test the ability of our FRM to mitigate such bias, we use the COMPAS dataset from Angwin et al. [2], which contains demographic information and criminal histories of defendants, along with binary labels indicating whether they will re-offend within two years. Our experiments focus on fairness across racial groups, specifically examining disparities in predicted recidivism rates between African American and white individuals.

Social media content moderation To manage harmful speech at scale, major social media platforms have turned to machine learning models (and increasingly, LLMs) to detect and moderate toxic and hateful content. However, automated moderation tools have been found to disproportionately flag benign content that references marginalized groups, a concern recognized in policy documents like the U.S. AI Bill of Rights [55]. To evaluate whether our FRM can help reduce such disparities in this setting, we use the Civil Comments dataset [10]. Civil Comments contains user-generated

posts labeled for toxicity as well as annotations for whether a protected group (e.g., religion, sexual orientation, or gender identity) is mentioned. We examine whether moderation decisions differ systematically across these group mentions for religion (Christian vs. Muslim) and sexual orientation, and whether re-weighting LLM-generated reasoning using the FRM reduces disparities in toxicity judgments.

Screening job candidates LLMs are increasingly used to support recruiting and hiring decisions, including generating summaries of applicant profiles, identifying top candidates, and inferring likely occupations from unstructured biographies [18, 56, 62]. However, a growing body of evidence suggests these systems risk amplifying historical biases. For example, De-Arteaga et al. [15] showed that classifiers trained on online biographies exhibit significant gender bias when predicting a person's occupation, even when explicit gender indicators like names and pronouns are removed. To evaluate the effectiveness of our Fairness Reward Model in this domain, we use the Bias in Bios dataset [15], which contains more than 390,000 biographies labeled with occupations and binary gender. The task is to predict an individual's occupation from their biography. Since many occupations in the dataset have existing gender imbalances, we measure whether fairness-aware reasoning mitigates disparities in classification accuracy or predicted labels across gender groups, especially in cases where female candidates are underrepresented.

## 5 Experiments

Here we detail the experimental setup for applying our trained FRM to the previously described downstream tasks.

Fairness metrics We measure group fairness with two of the most widely used decision parity criteria in machine learning: equalized odds and its relaxed variant, equalized opportunity [14, 22]. These metrics quantify whether the error rates of a classifier are balanced across protected groups. Let A ∈ { a 1 , a 2 } represent a binary protected attribute such as race or gender, where a 1 and a 2 correspond to different groups. Equalized odds requires that both the true positive rate and the false positive rate are the same for every group:

<!-- formula-not-decoded -->

Equalized opportunity demands parity only for the true positive rate:

<!-- formula-not-decoded -->

Beyond their widespread use in the fairness literature, these metrics capture the intended effect of using our FRM: by suppressing biased rationales (e.g., a resume assessment that treats caregiving gaps as a proxy for lower competence), the FRM should equalize the likelihood that qualified and unqualified candidates of different genders are labeled correctly, thereby closing the TPR and FPR gaps that equalized opportunity and equalized odds quantify. In practice we compute the absolute gap in each relevant error rate between two protected groups. A gap of zero indicates perfect fairness, and larger values signal greater disparity. The precise gap definitions are provided in Appendix B.4. Since Bias in Bios is a 4-way classification task, FPR does not apply, and we only measure TPR/equalized opportunity gaps as in Parrish et al. [35].

Inference We apply our FRM to re-weighting the decisions of 32 CoT samples. For all experiments using Llama models for inference, we set the temperature τ to 0.2 for the fairness-aware decision aggregation; for Mistral, we set τ = 0 . 01 .

Baselines We compare to the following baselines in our main experiments: (1) Chain-of-thought prompting ( CoT@1 ) - decision produced with a single chain-of-thought; (2) Chain-of-thought with majority voting ( Maj@32 ) - decision produced with majority vote from 32 CoT samples using uniform weighting across the chains; (3) Fairness Prompting ( Fairness Prompt ) - CoT prompting where the model is explicitly instructed to avoid biased reasoning. To bolster our results, we also ablate design decisions and various other aspects of our method in Section 6.3, and show the results from a variety of fairness prompting variants in Appendix D.3.

## 6 Results

In this section, we present the results of applying our trained FRM to various downstream tasks. First, we study generalization to new tasks and domains; next, we examine generalization to new reasoning models; finally, we explore and ablate design decisions, and perform a qualitative evaluation of our approach. We also perform a qualitative analysis of our results in Appendix A, examining both successful and failed examples.

## 6.1 Generalizing to new tasks and domains

We begin by testing whether a single FRM can reduce disparities across three different tasks and four different protected attributes, without bespoke tuning. Using a Llama-3.2-3B-Instruct backbone to produce reasoning chains and decisions, we compare three inference modes: COT@1 (a single chain of thought); MAJ@32 (majority vote over 32 chains); and FRM (the same 32 chains re-weighted by their FRM scores). Figure 5 summarizes results for race in COMPAS , sexual orientation and religion in Civil Comments , and gender in Bias in Bios . For each dataset column, the top panel shows the average accuracy, the middle panel the equalized opportunity gap, and the bottom panel displays the equalized odds gap.

Across all tasks, the FRM reduces both fairness violation metrics relative to the CoT@1 and Maj@32 baselines 4 . Fairness prompting improves fairness in some cases, but produces substantial loss of accuracy. The absolute fairness improvements using the FRM are largest in Civil CommentsReligion , where the raw equalized odds gap exceeds sixty percentage points under COT@1 and MAJ@32 but falls by more than ten points after fairness re-weighting. Significant relative gains also appear in COMPAS, Civil Comments-Sexual Orientation, and Bias in Bios, illustrating that the verifier generalizes beyond the domain on which it was trained. Crucially, there is no significant loss in accuracy. In the two Civil Comments settings, accuracy even increases, rising by roughly four percentage points despite the stricter fairness constraints. Although there are often trade-offs between accuracy and fairness, these results show that sometimes fairer decisions are in fact more accurate, and the FRM can work to reduce bias in either scenario. Two other observations stand out. First, majority voting alone can worsen disparities (e.g., equalized odds in Civil Comments-Religion), confirming that ensembling more chains does not automatically neutralize bias, and might worsen it. Second, the greatest absolute fairness improvements coincide with the settings that exhibit the highest initial gaps, suggesting that the FRM is especially effective when unfairness is most pronounced. These results show that a single, once-trained FRM can shrink fairness gaps compared to strong baselines across a variety of real-world tasks and protected groups without harming accuracy (and in several cases even boosting it).

## 6.2 Generalizing to new reasoning models

Our previous experiment studied whether our FRM can effectively generalize outside of its training task and domain. Next, we probe a further dimension of generalization, applying the Fairness Reward Model to supervise the reasoning process of a previously unseen LLM (where the training set of the FRM consists of synthetic data generated by various Llama-3 models). In particular, we use Mistral-7B-Instruct-v0.3 as our reasoning model, and run our experiments on COMPAS and Bias in Bios.

Results are shown in Figure 6, where the measurements for each dataset are shown across a row, and the columns display average accuracy and deviation from equalized opportunity and equalized odds. For both datasets, the FRM is able to improve fairness outcomes. Although the equalized opportunity gap on COMPAS is worse under the FRM than majority voxgting, the overall equalized odds gap is smaller, meaning that its improvement in balancing false positives was greater than the difference in true positive rates. The FRM also improves accuracy by more than 10%, highlighting how fairer reasoning can actually inform more correct decisions, especially in difficult problems like predicting recidivism. For Bias in Bios, the FRM reduces gender disparities by roughly 33%, while retaining most of the accuracy benefits of repeated sampling and majority voting. These findings indicate our FRMcan generalize effectively to new reasoning models that were not used during training.

4 We find these differences significant at level p &lt; 0 . 01 via bootstrap significance testing in Appendix D.

<!-- image -->

Figure 5: Results for generalizing our Fairness Reward Model across three different task domains and four different protected attribute categories, with reasoning and decisions produced by Llama3.2-3B-Instruct. We compare to chain-of-thought, majority voting with 32 CoT samples, and fairness prompting (baselines shown in grey). Our fairness metrics are the deviation from equalized odds and equalized opportunity (lower is better), and we also record accuracy. Overall, the FRM consistently improves decision-making fairness without harming (and sometimes even improving) accuracy.

<!-- image -->

40%

Figure 6: Results generalizing our FRM to reasoning chains produced by a previously unseen LLM, Mistral.

## 6.3 Design decisions and ablations

We explore and ablate key design decisions involved in training our Fairness Reward Model to understand their importance to our method. We compare step-level process reward models vs. chainlevel outcome rewards. For each of these strategies, we consider two sources of weak labels for training: LLM-generated labels (either at the step- or chain-level), and BBQ ground-truth labels. BBQ labels only indicate fairness at the chain-level so we copy the outcome label across every step in the chain for process supervision. Finally, we consider the value of using process supervision for fairness without a trained model by testing a prompting-based zero-shot PRM with no additional training.

All experiments reuse the same Llama-3.2-3B-Instruct generator and the standard inference pipeline of 32 CoT samples with τ = 0 . 2 . We compare our FRM to four modified reward models: (1) an ORM trained on BBQ labels, (2) a PRM trained on BBQ labels, (3) an ORM trained on LLM labels, and (4) a zero-shot PRM (see Appendix C.2 for more details). Results are shown in Figure 7; we also include the Maj@32 baseline for easy reference.

First, we can observe the effects of different labeling strategies. The PRM with BBQ labels is less effective at reducing disparities than our FRM, likely due to applying outcome labels as process supervision during training. While the ORM with LLM weak supervision performs comparably to our FRM on COMPAS, we see that on Civil Comments, our FRM produces an absolute fairness improvement of more than 10% relative to this ORM. The ORM trained on BBQ labels performs

<!-- image -->

40%

Figure 7: Results ablating various design decisions in our FRM: (1) source of weak supervision (BBQ ground-truth vs. LLM), (2) type of reward model (process vs. outcome), and (3) weak label training vs. zero-shot.

poorly on COMPAS, increasing disparities along both metrics. Finally, the zero-shot PRM is strictly worse than the trained FRM in terms of both fairness and accuracy. Overall, only our FRMconsistently narrows parity gaps while preserving accuracy, confirming that LLM weak supervision and process-level granularity are essential design choices.

Ablating temperature parameter In presenting our method, we argue that the inclusion of the temperature parameter τ can enable flexible control of how much fairness is prioritized when combining decisions across chains. For our final experiment, we ablate the effects of this parameter, exploring outcomes for τ ∈ { 0 . 01 , 0 . 2 , 0 . 4 , 0 . 8 } . We run our experiments on all three of our downstream tasks. Results are shown in Figure 12. For all three tasks, reducing the temperature from 0.8 to 0.4, and further to 0.2, decreases the outcome gaps across groups according to both fairness metrics. For COMPAS and Bias in Bios, reducing temperature to the very low setting of 0.01 brings further improvements in fairness, while this effect does not hold for Civil Comments. These results are strong evidence that our FRM inference framework offers the flexible control lacking in methods like prompting and fine-tuning.

## 7 Limitations

Though our approach produces favorable results across a range of settings, it has some limitations in its current form. First, we rely on weakly supervised LLM-annotated labels to train our FRM. While we find substantial agreement with humans for these labels and observe that our FRM is effective on downstream tasks, these LLM labels could introduce bias of their own. In addition, our method weights every reasoning step equally, ignoring that some steps may be pivotal to final decisions, while others are inconsequential. Further, both the training data and downstream evaluations are English-only (to the best of our knowledge) and based on the sociopolitical landscape of the United States, so we cannot conclude how it would generalize to other cultural contexts. Because the weak labels focus on explicit stereotype usage, the FRM should be most sensitive to overt textual bias, where reasoning explicitly invokes race, gender, religion, or other protected groups. Subtler biases, including those that are only visible in statistical patterns, are less likely to be detected. Finally, we evaluate fairness through equalized odds and equalized opportunity alone; additional notions such as calibration within groups, causal fairness, or individual-level fairness could reveal different trade-offs.

## Acknowledgments

We are grateful for the funding which made this work possible. One of the authors is supported by Amazon and Columbia's Center of Artificial Intelligence Technology (CAIT) PhD student fellowship.

One of the authors has an equity interest in OpenAI. We also thank ONR Grant N00014-23-1-2436 for its generous support. This work is supported by the funds provided by the National Science Foundation and by DoD OUSD (R&amp;E) under Cooperative Agreement PHY-2229929 (The NSF AI Institute for Artificial and Natural Intelligence).

## References

- [1] Jiafu An, Difang Huang, Chen Lin, and Mingzhu Tai. Measuring gender and racial biases in large language models, 2024. URL https://arxiv.org/abs/2403.15281 .
- [2] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. Machine bias. ProPublica , 2016. URL https://www.propublica.org/article/ machine-bias-risk-assessments-in-criminal-sentencing .
- [3] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional ai: Harmlessness from ai feedback, 2022. URL https://arxiv.org/abs/ 2212.08073 .
- [4] Solon Barocas, Moritz Hardt, and Arvind Narayanan. Fairness and Machine Learning . fairmlbook.org, 2019. http://www.fairmlbook.org .
- [5] Edward Beeching, Lewis Tunstall, and Sasha Rush. Scaling test-time compute with open models, 2024. URL https://huggingface.co/spaces/HuggingFaceH4/ blogpost-scaling-test-time-compute .
- [6] Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM conference on fairness, accountability, and transparency , 2021. URL https: //s10251.pcdn.co/pdf/2021-bender-parrots.pdf .
- [7] Rishabh Bhardwaj and Soujanya Poria. Red-teaming large language models using chain of utterances for safety-alignment, 2023. URL https://arxiv.org/abs/2308.09662 .
- [8] Emily Black, John Logan Koepke, Pauline T Kim, Solon Barocas, and Mingwei Hsu. Less discriminatory algorithms. Geo. LJ , 113:53, 2024.
- [9] Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri Chatterji, Annie Chen, Kathleen Creel, Jared Quincy Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren Gillespie, Karan Goel, Noah Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh, Mark Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, Jure Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, Avanika Narayan, Deepak Narayanan, Ben Newman, Allen Nie, Juan Carlos Niebles, Hamed Nilforoshan, Julian Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, Joon Sung Park, Chris Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Rob Reich, Hongyu Ren, Frieda Rong, Yusuf Roohani, Camilo Ruiz, Jack Ryan, Christopher Ré, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, Krishnan Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramèr, Rose E. Wang,

William Wang, Bohan Wu, Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, Matei Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, and Percy Liang. On the opportunities and risks of foundation models, 2022. URL https://arxiv.org/abs/2108.07258 .

- [10] Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. Nuanced metrics for measuring unintended bias with real data for text classification. In Sihem AmerYahia, Mohammad Mahdian, Ashish Goel, Geert-Jan Houben, Kristina Lerman, Julian J. McAuley, Ricardo Baeza-Yates, and Leila Zia, editors, Companion of The 2019 World Wide Web Conference, WWW 2019, San Francisco, CA, USA, May 13-17, 2019 , pages 491-500. ACM, 2019. URL https://doi.org/10.1145/3308560.3317593 .
- [11] Bradley C. A. Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. CoRR , abs/2407.21787, 2024. URL https://doi.org/10.48550/arXiv.2407. 21787 .
- [12] Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai Fan. Alphamath almost zero: Process supervision without process. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview.net/forum?id=VaXnxQ3UKo .
- [13] Xiusi Chen, Shanyong Wang, Cheng Qian, Hongru Wang, Peixuan Han, and Heng Ji. Decisionflow: Advancing large language model as principled decision maker, 2025. URL https://arxiv.org/abs/2505.21397 .
- [14] Alexandra Chouldechova. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. Big Data , 5(2):153-163, 2017.
- [15] Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In Proceedings of the Conference on Fairness, Accountability, and Transparency , FAT* '19. ACM, 2019. doi: 10.1145/3287560.3287572. URL http://dx.doi.org/10.1145/3287560.3287572 .
- [16] Nicholas Deas, Jessica Grieser, Shana Kleiner, Desmond Patton, Elsbeth Turcan, and Kathleen McKeown. Evaluation of African American language bias in natural language generation. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , pages 6805-6824, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.421. URL https://aclanthology.org/2023.emnlp-main.421/ .
- [17] Julia Dressel and Hany Farid. The accuracy, fairness, and limits of predicting recidivism. Science Advances , 4(1):eaao5580, 2018.
- [18] Johann D. Gaebler, Sharad Goel, Aziz Huq, and Prasanna Tambe. Auditing large language models for race &amp; gender disparities: Implications for artificial intelligence-based hiring. Behavioral Science &amp; Policy , 10(2):46-55, 2024. doi: 10.1177/23794607251320229. URL https://doi.org/10.1177/23794607251320229 .
- [19] Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. Bias and fairness in large language models: A survey. Computational Linguistics , 50(3):1097-1179, September 2024. doi: 10.1162/coli\_a\_00524. URL https://aclanthology.org/2024.cl-3.8/ .
- [20] Michael Gira, Ruisu Zhang, and Kangwook Lee. Debiasing pre-trained language models via efficient fine-tuning. In Bharathi Raja Chakravarthi, B Bharathi, John P McCrae, Manel Zarrouk, Kalika Bali, and Paul Buitelaar, editors, Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion , pages 59-69, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.ltedi-1.8. URL https://aclanthology.org/2022.ltedi-1.8/ .
- [21] Yufei Guo, Muzhe Guo, Juntao Su, Zhou Yang, Mengqiu Zhu, Hongfei Li, Mengyang Qiu, and Shuo Shuo Liu. Bias in large language models: Origin, evaluation, and mitigation, 2024. URL https://arxiv.org/abs/2411.10915 .

- [22] Moritz Hardt, Eric Price, and Nathan Srebro. Equality of opportunity in supervised learning. In Advances in Neural Information Processing Systems , pages 3315-3323, 2016.
- [23] Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, and Rishabh Agarwal. V-STar: Training verifiers for self-taught reasoners. In First Conference on Language Modeling , 2024. URL https://openreview.net/forum?id=stmqBSW2dV .
- [24] Mahammed Kamruzzaman and Gene Louis Kim. Prompting techniques for reducing social bias in llms through system 1 and system 2 cognitive processes, 2024. URL https://arxiv.org/ abs/2404.17218 .
- [25] Masahiro Kaneko, Danushka Bollegala, Naoaki Okazaki, and Timothy Baldwin. Evaluating gender bias in large language models via chain-of-thought prompting, 2024. URL https: //arxiv.org/abs/2401.15585 .
- [26] Hadas Kotek, Rikker Dockum, and David Sun. Gender bias and stereotypes in large language models. In Proceedings of the ACM collective intelligence conference , pages 12-24, 2023.
- [27] Faisal Ladhak, Esin Durmus, Mirac Suzgun, Tianyi Zhang, Dan Jurafsky, Kathleen McKeown, and Tatsunori Hashimoto. When do pre-training biases propagate to downstream tasks? a case study in text summarization. In Andreas Vlachos and Isabelle Augenstein, editors, Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics , pages 3206-3219, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.eacl-main.234. URL https://aclanthology.org/2023.eacl-main.234/ .
- [28] Zhuoyan Li, Hangxiao Zhu, Zhuoran Lu, Ziang Xiao, and Ming Yin. From text to trust: Empowering ai-assisted decision making with adaptive llm-powered analysis, 2025. URL https://arxiv.org/abs/2502.11919 .
- [29] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In The Twelfth International Conference on Learning Representations , 2024. URL https: //openreview.net/forum?id=v8L0pN6EOi .
- [30] Congda Ma, Tianyu Zhao, and Manabu Okumura. Debiasing large language models with structured knowledge. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024 , pages 10274-10287, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-acl.612. URL https://aclanthology.org/2024.findings-acl.612/ .
- [31] Huan Ma, Changqing Zhang, Yatao Bian, Lemao Liu, Zhirui Zhang, Peilin Zhao, Shu Zhang, Huazhu Fu, Qinghua Hu, and Bingzhe Wu. Fairness-guided few-shot prompting for large language models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems , volume 36, pages 43136-43155. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/ paper/2023/file/8678da90126aa58326b2fc0254b33a8c-Paper-Conference.pdf .
- [32] Natalie Mackraz, Nivedha Sivakumar, Samira Khorshidi, Krishna Patel, Barry-John Theobald, Luca Zappella, and Nicholas Apostoloff. Evaluating gender bias transfer between pre-trained and prompt-adapted language models, 2024. URL https://arxiv.org/abs/2412.03537 .
- [33] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve

Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O'Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774 .

- [34] Orestis Papakyriakopoulos, Simon Hegelich, Juan Carlos Medina Serrano, and Fabienne Marco. Bias in word embeddings. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency , FAT* '20, page 446-457, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450369367. doi: 10.1145/3351095.3372843. URL https: //doi.org/10.1145/3351095.3372843 .
- [35] Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel Bowman. BBQ: A hand-built bias benchmark for question answering. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Findings of the Association for Computational Linguistics: ACL 2022 , pages 2086-2105, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.165. URL https://aclanthology.org/2022.findings-acl.165/ .
- [36] Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, Andy Jones, Anna Chen, Benjamin Mann, Brian Israel, Bryan Seethor, Cameron McKinnon, Christopher Olah, Da Yan, Daniela Amodei, Dario Amodei, Dawn Drain, Dustin Li, Eli Tran-Johnson, Guro Khundadze, Jackson Kernion, James Landis, Jamie Kerr, Jared Mueller, Jeeyoon Hyun, Joshua Landau, Kamal Ndousse, Landon Goldberg, Liane Lovitt, Martin Lucas, Michael Sellitto, Miranda Zhang, Neerav Kingsland, Nelson Elhage, Nicholas Joseph, Noemi Mercado, Nova DasSarma, Oliver Rausch, Robin Larson, Sam McCandlish, Scott Johnston, Shauna Kravec, Sheer El Showk,

Tamera Lanham, Timothy Telleen-Lawton, Tom Brown, Tom Henighan, Tristan Hume, Yuntao Bai, Zac Hatfield-Dodds, Jack Clark, Samuel R. Bowman, Amanda Askell, Roger Grosse, Danny Hernandez, Deep Ganguli, Evan Hubinger, Nicholas Schiefer, and Jared Kaplan. Discovering language model behaviors with model-written evaluations. In Anna Rogers, Jordan BoydGraber, and Naoaki Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023 , pages 13387-13434, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.847. URL https://aclanthology.org/ 2023.findings-acl.847/ .

- [37] Flor Miriam Plaza-del Arco, Amanda Cercas Curry, Susanna Paoli, Alba Cercas Curry, and Dirk Hovy. Divine LLaMAs: Bias, stereotypes, stigmatization, and emotion representation of religion in large language models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024 , pages 4346-4366, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.251. URL https://aclanthology.org/ 2024.findings-emnlp.251/ .
- [38] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347 .
- [39] Omar Shaikh, Hongxin Zhang, William Held, Michael Bernstein, and Diyi Yang. On second thought, let's not think step by step! bias and toxicity in zero-shot reasoning. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 4454-4470, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. acl-long.244. URL https://aclanthology.org/2023.acl-long.244/ .
- [40] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/ 2402.03300 .
- [41] Jisu Shin, Hoyun Song, Huije Lee, Soyeong Jeong, and Jong Park. Ask LLMs directly, 'what shapes your bias?': Measuring social bias in large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024 , pages 16122-16143, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.954. URL https:// aclanthology.org/2024.findings-acl.954/ .
- [42] Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM testtime compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations , 2025. URL https: //openreview.net/forum?id=4FWAwZtd2n .
- [43] Alex Tamkin, Amanda Askell, Liane Lovitt, Esin Durmus, Nicholas Joseph, Shauna Kravec, Karina Nguyen, Jared Kaplan, and Deep Ganguli. Evaluating and mitigating discrimination in language model decisions, 2023. URL https://arxiv.org/abs/2312.03689 .
- [44] Vishesh Thakur. Unveiling gender bias in terms of profession across llms: Analyzing and addressing sociological implications, 2023. URL https://arxiv.org/abs/2307.09162 .
- [45] Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Lei Han, Haitao Mi, and Dong Yu. Toward self-improvement of llms via imagination, searching, and criticizing. In Proceedings of the 38th International Conference on Neural Information Processing Systems , NIPS '24, Red Hook, NY, USA, 2025. Curran Associates Inc. ISBN 9798331314385.
- [46] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut

Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288 .

- [47] Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don't always say what they think: unfaithful explanations in chain-of-thought prompting. In Proceedings of the 37th International Conference on Neural Information Processing Systems , NIPS '23, Red Hook, NY, USA, 2023. Curran Associates Inc.
- [48] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback, 2022. URL https://arxiv.org/abs/2211.14275 .
- [49] Yixin Wan, George Pu, Jiao Sun, Aparna Garimella, Kai-Wei Chang, and Nanyun Peng. 'kelly is a warm person, joseph is a role model': Gender biases in LLM-generated reference letters. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023 , pages 3730-3748, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.243. URL https://aclanthology.org/2023.findings-emnlp.243/ .
- [50] Ziyu Wan, Xidong Feng, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang. Alphazero-like tree-search can guide large language model decoding and training. In Proceedings of the 41st International Conference on Machine Learning , ICML'24. JMLR.org, 2024.
- [51] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 9426-9439, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.510. URL https://aclanthology.org/2024. acl-long.510/ .
- [52] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id=1PL1NIMMrw .
- [53] Albert Webson and Ellie Pavlick. Do prompt-based models really understand the meaning of their prompts? In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 2300-2344, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. naacl-main.167. URL https://aclanthology.org/2022.naacl-main.167/ .
- [54] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 24824-24837. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf .
- [55] White House. Blueprint for an ai bill of rights: Making automated systems work for the american people, 2022.
- [56] Kyra Wilson and Aylin Caliskan. Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval , page 1578-1590. AAAI Press, 2025.

- [57] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems , volume 36, pages 11809-11822. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf .
- [58] Zhangyue Yin, Qiushi Sun, Qipeng Guo, Zhiyuan Zeng, Xiaonan Li, Tianxiang Sun, Cheng Chang, Qinyuan Cheng, Ding Wang, Xiaofeng Mou, Xipeng Qiu, and Xuanjing Huang. Aggregation of reasoning: A hierarchical framework for enhancing answer selection in large language models. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024) , pages 609-625, Torino, Italia, May 2024. ELRA and ICCL. URL https://aclanthology.org/2024.lrec-main.53/ .
- [59] Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models, 2023. URL https://arxiv.org/abs/2308.01825 .
- [60] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 15476-15488. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/639a9a172c044fbb64175b5fad42e9a5-Paper-Conference.pdf .
- [61] Qingquan Zhang, Qiqi Duan, Bo Yuan, Yuhui Shi, and Jialin Liu. Exploring accuracy-fairness trade-off in large language models, 2024. URL https://arxiv.org/abs/2411.14500 .
- [62] Thomas Zollo, Nikita Rajaneesh, Richard Zemel, Talia Gillis, and Emily Black. Towards effective discrimination testing for generative ai. In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency , pages 1028-1047, 2025.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims in the abstract clearly state our key contributions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Yes, details limitations are listed in Section 7, including with respect to technical aspects of our methods, evaluation protocols, and fairness definitions.

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

Justification: We will open source our code upon publication and we provide full experimental details in the main paper and Appendix.

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

Justification: Yes, we will release our training data and trained reward model upon publication of this work.

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

Justification: We provide all experimental settings and training details in the main paper, with minor details deferred to Appendix C

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We perform bootstrap statistical significance testing for our main experiment results. We find them to be significant at a level p ≤ 0 . 01 .

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

Justification: Yes, the experiment details specify the models, data, and amount of training. Additionally, our open source code will facilitate reproducibility.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Yes, the research conforms in every respect with the Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper is centered on societal concerns; it has extensive discussion of the potential positive and negative impacts of LLM use in high-stakes decision-making domains. Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [Yes]

Justification: Our model and dataset cards will describe the potential for misuse of our assets.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All creators and owners of assets are credited via proper citations and relevant links.

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

Justification: The appendix of the paper includes a model card for our reward model and a data card for our weakly labeled step-wise fairness dataset.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [Yes]

Justification: The appendix of the paper includes the prompt for labeling bias in reasoning steps.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The only human annotators involved were the authors of the paper therefore IRB approval was not required.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This research does not involve LLMs as any important, original, or nonstandard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

Figure 8: Successful case where biased reasoning changes the decision. The second chain invokes race-based correlations and concludes a higher risk label, while the first chain avoids demographic associations and concludes low risk. The FRM assigns a higher fairness score to the neutral chain (0.931 vs. 0.6987).

<!-- image -->

## Appendix

## A Qualitative Results

We present qualitative examples to demonstrate the strengths and limitations of our FRM, analyzing both the training data annotations produced by GPT-4o-mini and the decisions made using our final trained FRM. These examples illustrate when our system correctly identifies biased reasoning, when it fails, and how imperfections in the labeling process can propagate into model behavior. We examine the fairness scores given to reasoning steps in 3 COMPAS cases, exploring both successes and failures of the model. Additional FRM outputs are included in Appendix D.2.

Successful examples In the majority of cases, our FRM successfully identifies biased reasoning and assigns higher fairness scores to chains that rely solely on provided evidence. For example, in one pair of reasoning chains evaluating the same defendant, shown in Figure 8, the lower scoring chain (FRM score = 0.6987) references the individual's demographic factors as risk factors and cites group stereotypes to justify a high-risk label. In contrast, the higher scoring chain (FRM score = 0.931) treats these attributes as neutral and bases its final decision on the absence of prior charges and the individual's juvenility. This example highlights the FRM's ability to flag reasoning that is based on harmful stereotypes and unsupported by evidence from the given context. In the example shown in Figure 9, both chains arrive at the same decision (high-risk), but only one uses the race of the defendant as a core part of the reasoning. The FRM correctly assigns a higher fairness score (0.8286) to the chain that focuses solely on criminal history and assigns a lower score (0.6975) to the chain that relies on demographic correlations.

Failed examples While our FRM is generally very effective at detecting biased steps, one shortcoming of our method is that it weighs every step equally regardless of whether the step actually contributes to the final decision. In some cases, our approach assigns lower aggregate fairness scores to chains that have unbiased final decisions than those that have a biased conclusion. In the example shown in Figure 10, both reasoning chains have one step that references a stereotype about African-Americans. In Chain 1, the model acknowledges the stereotype but ultimately makes a decision based on other factors while in Chain 2 the demographic information is actively used in its final risk assessment. Despite this, Chain 1 receives a lower aggregate fairness score as the model

Figure 9: Successful fairness scoring on a COMPAS example where both reasoning chains predict high risk. Reasoning Chain 1 avoids group-based associations and receives a higher fairness score (0.8286), while Chain 2 invokes demographic correlations and is penalized accordingly (0.6975).

<!-- image -->

is not sensitive to the causal role that reasoning plays in determining the final decision. We suspect that this may stem from the usage of LLM labels during training as we observed that the LLM flags any mention of demographic correlations as biased whereas human annotators are more sensitive to context.

## A.1 Limitations of LLM annotations

While we observed substantial agreement between GPT-4o-mini labels and human annotations, we examined disagreement cases to better understand the limits of LLM fairness supervision. Our qualitative study revealed several main failure modes:

1. Group names trigger biased labels: The LLM may mark steps as biased where a group was mentioned even if the text is benign or the information was taken directly from the context. Figure 11 shows an example where the step simply restates information given in the context, yet GPT-4o-mini labels it as biased due to the presence of group identifiers.
2. Failure to recognize implicit bias: The LLM may fail to label biased reasoning that is implicit and requires a deeper contextual understanding.
3. Reasoning is incoherent, but LLM labels it as biased: In some cases, the LLM annotates a step as biased when the reasoning was vague or incoherent. Human annotators marked these steps as unbiased.
4. LLMannotator believes hallucinated reasoning: The model used for generating reasoning data occasionally hallucinated details based on stereotypes that were not in the original context. The LLM annotator was often not able to pick up on this being a form of bias and labeled these steps as unbiased.

There were also some steps on which the LLM annotations disagreed with human judgment without a clear pattern, including errors in both directions. Additional examples are included in Appendix D.2. Despite these annotation issues, the aggregate performance of our FRM suggests that the model is robust to some label noise. Future work could seek to improve annotation quality.

<!-- image -->

Figure 10: Failure case: the FRM assigns a lower fairness score to a reasoning chain that acknowledges but ignores a biased step, while assigning a higher score to a chain that relies more heavily on groupbased reasoning.

<!-- image -->

Figure 11: LLM annotation failure case: the LLM annotator marks this step as biased based on its mention of group identities while the human annotator considers context and labels it as unbiased.

## B Fairness Reward Model

## B.1 LLMLabeling Procedure

We use GPT-4o-mini to weakly label fairness of individual reasoning steps. The model was prompted to analyze each step for bias. We processed examples in batches of 20 and parsed the outputs using regular expressions. The full labeling prompt is included below.

Analyze each numbered step in the examples below for potential bias. For each step, determine if it:

```
-Uses stereotypes about any group -Makes assumptions about people based on group membership -Displays bias or preconceptions towards any group For each example, format your response as: Example [ID]: Step 1: [BIASED/UNBIASED] Explanation: (provide specific explanation if biased) Step 2: [BIASED/UNBIASED] Explanation: (provide specific explanation if biased) ... and so on for each step
```

If an example has no biased steps, simply write "All steps UNBIASED" for that example.

## B.2 Implementation Details

We implement four different reward models, our FRM, a PRM with BBQ labels, an ORM with BBQ labels, a PRM with LLM labels and an ORM with LLM labels. LLM labels are obtained using the process described in Appendix B and BBQ labels are based on the BBQ answers. For ORM labels, we labeled a reasoning chain as biased if a single step in it was labeled as biased.

FRM training details are described in section 3; the training procedure for ORMs is analogous except instead of classifying a single step the model is given the full reasoning chain. The objective is the same binary cross entropy objective as the FRM. We fine tune all models starting from a LLaMA 3.21B-Instruct checkpoint with PPO-style reward training. We train for 2 epochs on 255,000 reasoning steps (for PRMs) or 79,000 reasoning chains (for ORMs) using 4 NVIDIA A100 GPUs with 40GB of memory each. Training takes approximately 2 hours per model.

## Model Details

- Developer: Zara Hall and collaborators
- Model Date: May 2025
- Model Version: v1.0
- Model Type: reward model
- Training Algorithms and Parameters: PPO-style training using Hugging Face's RewardTrainer , optimized with binary cross-entropy loss. AdamW optimizer with learning rate 2e-5, β = (0 . 9 , 0 . 95) , batch size 128.
- Key Features: fairness scoring, interpretability
- License: MIT License
- Contact: zyh2000@columbia.edu

## Intended Use

- Primary Use Cases: scoring fairness in LLM reasoning chains
- Out-of-Scope Use Cases: high-stakes decisions

## Factors

- Relevant Groups: race, gender, religion, sexual idenity
- Evaluation Conditions: generalization to tasks outside of training data, models outside training data

## Metrics

- Performance Measures: equalized odds gap, equalized opportunity gap, accuracy

- Thresholds: temperature τ values varied between 0.01 and 0.8 to trade off fairness and consistency
- Variation Methods: ablations on label source (BBQ vs LLM), reward granularity (step vs. chain), and training

## Evaluation Data

- COMPAS, CivilComments, Bias in Bios
- Motivation: real-world relevance, demographic diversity, ground-truth labels
- Preprocessing: step segmentatation of CoT outputs

## Training Data

- Reasoning chains generated on questions from the BBQ (Bias Benchmark for QA) using 4395 prompts and four LLMs (LLaMA-3.1-8B-Instruct, LLaMA-3.1-70B-Instruct, LLaMA3.2-1B-Instruct, and LLaMA-3.2-3B-Instruct)
- Labels: binary bias annotations (biased/unbiased) from GPT-4o-mini for each reasoning step

## Quantitative Analyses

- Equalized odds and opportunity gaps reduced across all tasks
- No accuracy decrease observed; in several case, accuracy improved over majority voting

## Ethical Considerations

- Labels reflect GPT-4o-mini's biases

## Caveats and Recommendations

- Generalization was not tested on every possible domain

## Motivation

- Our goal in creating this dataset was to generate training data for our Fairness Reward Model (FRM) capable of identifying biased reasoning in LLMs. We also hope to enable future work on fine-grained bias detection.

## Collection Process

- We begin with 4395 prompts from the BBQ (Bias Benchmark for QA) dataset [35] and generated 255,000 reasoning steps using four instruction-tuned LLaMA models. These chains were segmented into individual reasoning steps.

## Preprocessing

- CoT completions were parsed into steps based on section headers. Reasoning chains and annotations were aligned by (BBQ example ID, completion index, step index) for reproducibility.

## Distribution

- The source BBQ prompts are subject to the license and terms described in Parrish et al. [35], and remain the intellectual property of their original authors.
- Generated reasoning chains and GPT-4o-mini labels are our contributions and will be made publicly available.

## Maintenance

- The authors of this paper welcome feedback and plan on maintaining the dataset

## B.3 Human Annotation Study

To evaluate the GPT-4o-mini labels we have three human evaluators independently label 100 steps. Annotators were three of the authors of this paper. Each annotator is shown a question and reasoning trace and then labels each step as biased or unbiased using the same instructions given to the LLM labeler. The average Cohen's Kappa between human annotators is 0.6078 and the average Kappa between GPT-4o-mini and each human annotator is 0.2259. Cohen's Kappa was particularly low in certain cases due to class imbalance in the dataset. Qualitative analysis shows that most disagreements are on steps where the reasoning is incoherent or hallucinated contextual evidence.

Table 1: Pairwise agreement between human annotators and GPT-4o-mini on 100 reasoning steps.

| Annotator Pair            |   Cohen's κ | Agreement (%)   |
|---------------------------|-------------|-----------------|
| Annotator 1 ↔ GPT-4o-mini |      0.2474 | 70.87%          |
| Annotator 2 ↔ GPT-4o-mini |      0.3557 | 80.85%          |
| Annotator 3 ↔ GPT-4o-mini |      0.0744 | 74.29%          |
| Annotator 1 ↔ Annotator 2 |      0.6854 | 86.05%          |
| Annotator 2 ↔ Annotator 3 |      0.4308 | 87.50%          |
| Annotator 1 ↔ Annotator 3 |      0.7071 | 91.07%          |

## B.4 Fairness Metric Definitions

As described in Section 5, we calculate the absolute gap in Equalized Odds and Equalized Opportunity for each of our downstream tasks.

Equalized Opportunity Gap. We compute the absolute difference in true positive rates between the two groups:

<!-- formula-not-decoded -->

Equalized Odds Gap. We compute the sum of absolute differences in true positive and false positive rates:

<!-- formula-not-decoded -->

For each dataset, we binarize the protected attribute and compute the relevant metric by grouping prediction by A .

## C Experiment Details

## C.1 Inference Process

For our FRM to be effective, the base model must output answers in a specific step-by-step format. We modify the system prompt from the repository released by Beeching et al. [5] such that for each dataset the few-shot prompt is tailored to the specific downstream task. As in the original repository, each prompt instructs the model to output the answer in a box for easy analysis. We also use the grading method and answer extraction method from their repository to score all of the reasoning chains. We will open-source our specific code upon publication.

## C.2 Baselines

In order to demonstrate effectiveness of our FRM we perform an exhaustive comparison against existing baselines: fairness prompting , chain-of-thought prompting , chain-of-thought best-of-n , and an untrained PRM . The untrained PRM is the base model of the FRM before it has been fine-tuned: we prompt LLaMA-3.2-1B-Instruct to assign a bias score between 0 to 1 for each step. CoT prompting and CoT prompting best-of-n use the same prompt. For fairness prompting we append the following statement to the prompt:

Table 2: Evaluation datasets and associated prompting formats.

| Dataset       | Task                                           | Protected At- tribute(s)     | Model Prompt Summary                                                                                                                      |
|---------------|------------------------------------------------|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| COMPAS        | Binary clas- sification (recidivism risk)      | Race (Black vs. White)       | Model acts as a risk assessor, rea- soning about behavioral factors (e.g., prior offenses, job stability) and outputs a boxed risk label. |
| CivilComments | Binary classifi- cation (toxicity detection)   | Religion, Sexual orientation | Model simulates a content mod- erator deciding to keep or re- move a comment, with multi- step justification and a boxed de- cision.      |
| Bias in Bios  | 4-way classifica- tion (profession prediction) | Gender (Male vs. Female)     | Model identifies the most likely profession from a short biogra- phy using multi-step reasoning.                                          |

Table 3: FRM performance relative to baselines on COMPAS

| Method                                    | AA Acc ↑                                  | Cauc Acc ↑                                | TPR Gap ↓                                 | FPR Gap ↓                                 |
|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| CoT@1                                     | 59.09                                     | 68.22                                     | 15.52                                     | 12.40                                     |
| Maj@32                                    | 63.22                                     | 69.92                                     | 12.81                                     | 8.80                                      |
| FRM                                       | 61.16                                     | 69.49                                     | 7.70                                      | 7.60                                      |
| Relative change of FRM (0.2) vs baselines | Relative change of FRM (0.2) vs baselines | Relative change of FRM (0.2) vs baselines | Relative change of FRM (0.2) vs baselines | Relative change of FRM (0.2) vs baselines |
| vs CoT@1                                  | +3.50%                                    | +1.86%                                    | -50.4%                                    | -38.7%                                    |
| vs Maj@32                                 | -3.27%                                    | -0.61%                                    | -39.9%                                    | -13.6%                                    |

Make fair, objective assessments based solely on relevant information, avoiding inappropriate influence from demographic characteristics like race, gender, age, or socioeconomic status, and provide clear, step-by-step reasoning to justify your conclusions.

## C.3 Other Details

All experiments were conducted using vllm and HuggingFace's transformers libraries. Model training was performed on 4 NVIDIA A100 GPUs. All prompts, model weights, and evaluation scripts will be open-sourced upon publication.

## D Experiment Results

Figure 12 shows results from varying the aggregation temperature τ across the range { 0 . 01 , 0 . 2 , 0 . 4 , 0 . 8 } on all three tasks.

Additionally, in Table 3 we provide percent deltas on COMPAS results shown in Figure 6.

## D.1 Statistical Significance Testing

We compute bootstrap confidence intervals with 1,000 resamples and find that FRM outperforms both CoT@1 and Maj@32 with p &lt; 0 . 01 across all tasks and fairness metrics.

## D.2 Additional Qualitative Results

To better demonstrate how our FRM works in practice, we present additional examples outputs from both our validation task (BBQ) and another downstream task (CivilComments).

Bias Benchmark for QA (BBQ): Figure 13 presents two reasoning chains responding to a question about socioeconomic status and laziness. The second chain relies on stereotypes to justify its answer, assuming that the line cook is more likely lazy, and receives a low aggregate FRM score (0.6554).

<!-- image -->

40%

Figure 12: Results ablating the temperature parameter in the decision aggregation step. As expected, decreasing this parameter within a reasonable range generally improves decision-making fairness.

Figure 13: Example outputs and FRM scores for a BBQ question

<!-- image -->

One step is particularly biased, asserting that 'line cooks may be more prone to procrastination," and it receives the lowest stepwise fairness score (0.0998). The higher scoring chain (0.9524) correctly argues that there is insufficient information to make a decision. This example highlights how the FRM scores can pinpoint specific moments of biased reasoning while the aggregate score provides a reliable signal for the fairness of the decision as a whole.

Content Moderation (CivilComments): Figure 14 shows two chains assessing an example from CivilComments, where the higher scoring chain (FRM score = 0.9393) considers freedom of expression while the lower scoring chain (FRM score = 0.7819) recommends removal due to subjective implications. While the comment expresses a viewpoint that may be factually incorrect and strongly opposed by many users, it does not contain explicit hate speech. The FRM assigns a higher score to the reasoning chain that remains neutral across viewpoints.

LLMAnnotation Examples We include two additional disagreement cases from our qualitative study of LLM annotations.

Figure 15 shows an instance where the LLM annotator believes hallucinated evidence. The step fabricates details about emotional states that are stereotypical. Human annotators judged this to be biased, and an ideal fairness annotator would detect this type of bias however we do not expect an LLM-based annotator to reliably do so.

<!-- image -->

Figure 14: Example outputs and FRM scores for a CivilComments decision

<!-- image -->

Figure 15: Hallucinated reasoning labeled as unbiased. The reasoning step invents emotional states and caregiver roles not stated in the context, reflecting gender-role assumptions. Human annotators labeled this as biased; GPT-4o-mini did not.

Figure 16 shows a disagreement case that where the reasoning was incoherent and one human annotator simply noted that it 'doesn't make sense.' We include this example to show that some annotation disagreement arises from unintelligible reasoning rather than mislabeling by the LLM annotator.

<!-- image -->

Figure 16: Incoherent reasoning labeled as biased. The reasoning step is circular and fails to engage with the actual question. Human annotators judged the step as incomprehensible while GPT-4o-mini labeled it biased.

## D.3 Additional Prompting Results

Prompt-based approaches rely on the model following the prompt to improve fairness while still making accurate decisions. Given that LLMs cannot always follow prompts and respond to prompt changes in unpredictable ways [9, 36, 53], in some cases this works, in other cases it doesn't. We observed that in many cases, this can lead to the model following similar types of reasoning and giving the same answer every time to 'play it safe' from a bias perspective. Our method addresses these issues by scoring reasoning steps. In this way, we don't change the natural output diversity of the model and we provide transparency into which steps are downvoted as opposed to relying on the model to follow a fairness prompt and adjust its reasoning accordingly. Output diversity is important in these tasks because taking the majority vote over many reasoning chains has been widely shown to increase task accuracy, but if output diversity is significantly reduced, this can remove the benefits of a majority vote.

To further address these concerns, we have performed additional experiments on a set of stronger fairness prompting baselines. Using the COMPAS task, we systematically evaluated 10 original fairness prompts (written by GPT-4o) alongside the 7 prompts proposed by Tamkin et al. [43] to reduce discrimination in high-stakes language model applications. Each prompt was tested with and without CoT and we additionally included a CoT@32 setting using the best-performing prompt from each group. To highlight the modularity of our approach, we also apply the FRM to combine the 32 CoTs produced using fairness prompts. We report equalized odds and equalized opportunity deviations, defined as the sum and individual differences in FPR and TPR across groups, consistent with our original paper.

| Setting                                   |   Avg. Acc. | Min Acc.   | Max Acc.   |   Eq. Odds Dev. |   Eq. Opp Dev. |
|-------------------------------------------|-------------|------------|------------|-----------------|----------------|
| 10 fairness prompts                       |        46.6 | 27.8       | 56.6       |            16   |            6.7 |
| 10 fairness prompts w/ CoT                |        56.7 | 53.6       | 60.2       |            16.2 |            4   |
| 7 Tamkin prompts w/ CoT                   |        58.1 | 54.6       | 59.4       |            17.5 |            3.8 |
| Best fairness prompt w/ CoT (majority@32) |        57.2 |            |            |            19.7 |            4.7 |
| Best fairness prompt w/ CoT (FRM@32)      |        63.6 |            |            |            16.6 |            4.1 |
| Best Tamkin prompt w/ CoT (majority@32)   |        56   |            |            |             4.7 |            0.5 |
| Best Tamkin prompt w/ CoT (FRM@32)        |        56   |            |            |             5.7 |            0.3 |
| FRM(Ours)                                 |        65.3 |            |            |            15.3 |            7.6 |

Table 4: Comparison of prompting-based fairness baselines and our Fairness Reward Model (FRM) on COMPAS.

Consistent with our hypothesis and original findings, some prompts yielded reductions in equalized odds and equalized opportunity. However, this came at the cost of significant reductions in accuracy. This tradeoff remained consistent across settings and accuracy was reduced further when we sampled 32 times with fairness prompts. In many cases, fairness was improved not through better reasoning but by predicting the same label for all inputs. These results reinforce our core claim that reasoning-level supervision offers a more effective and robust fairness intervention than prompting when accuracy is a primary concern. Our findings echo concerns in prior work, including Tamkin et al. [43], regarding output distortion from fairness prompting.