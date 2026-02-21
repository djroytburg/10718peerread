## Sloth: scaling laws for LLM skills to predict multi-benchmark performance across families

## Felipe Maia Polo 1 , Seamus Somerstep 1 Leshem Choshen 2,3 Yuekai Sun 1 Mikhail Yurochkin 4

1 Department of Statistics, University of Michigan

2 MIT-IBM Watson AI Lab, IBM Research

3 Computer Science and Artificial Intelligence Laboratory, MIT

4 Institute of Foundation Models, MBZUAI

## Abstract

Scaling laws for large language models (LLMs) predict model performance based on parameters like size and training data. However, differences in training configurations and data processing across model families lead to significant variations in benchmark performance, making it difficult for a single scaling law to generalize across all LLMs. On the other hand, training family-specific scaling laws requires training models of varying sizes for every family. In this work, we propose Skills Scaling Laws (SSLaws, pronounced as Sloth ), a novel scaling law that leverages publicly available benchmark data and assumes LLM performance is driven by low-dimensional latent skills, such as reasoning and instruction following. These latent skills are influenced by computational resources like model size and training tokens, but with varying efficiencies across model families. Sloth exploits correlations across benchmarks to provide more accurate and interpretable predictions while alleviating the need to train multiple LLMs per family. We present both theoretical results on parameter identification and empirical evaluations on 12 prominent benchmarks, from Open LLM Leaderboard v1/v2, demonstrating that Sloth predicts LLM performance accurately and offers insights into scaling behaviors for complex downstream tasks, increased test-time compute, and compute-optimal scaling of skills. Our code can be found on https://github.com/felipemaiapolo/sloth .

## 1 Introduction

Large Language Model (LLM) scaling laws for benchmarks and downstream tasks efficiently predict the performance of an LLM based on its parameter count and training set size. However, variations in training configurations and data processing across different model families often lead to significant differences in benchmark performance, even for models with comparable compute budgets [Ruan et al., 2024]. Consequently, a single scaling law typically fails to predict performance across all LLMs accurately [Choshen et al., 2024]. In contrast, creating family-specific scaling laws requires training multiple models of increasing size, which is resource-intensive.

In this work, we propose a new class of scaling laws called Sloth to solve this dilemma. These scaling laws are fitted using publicly available data ( e.g. , from LLM leaderboards) across multiple benchmarks, leveraging information shared among benchmarks and model families to improve prediction power and interpretability through parameter efficiency, i.e. , fewer parameters without hurting performance. Specifically, we utilize the correlations in benchmark scores to simplify the scaling law in terms of parameter count without harming prediction power by assuming that LLM performance is driven by a set of low-dimensional latent skills, such as reasoning and instruction

following, which can be easily interpreted. Furthermore, we hypothesize that these latent skills are similarly influenced by computational resources, such as model size and training tokens, across different LLM families, with the key distinction being each family's efficiency in converting compute into skill levels-which can be estimated with only one model per family. In summary, our main contributions are

- Introducing a new class of scaling laws, Sloth , that borrows strength across the available benchmarks and LLM families to make more accurate and interpretable performance predictions of (hypothetical) LLMs in given benchmarks of interest. Specifically, we assume that benchmark performances directly depend on low-dimensional LLM skills, which are influenced by factors such as the number of training tokens and the number of parameters.
- Providing a theoretical result regarding the identification of Sloth 's parameters and empirically demonstrating that our scaling laws can (i) accurately predict the performance of large models in 12 prominent LLM benchmarks and (ii) provide interpretable insights into LLM skills and scaling behavior.
- Demonstrating how predicted latent skills and our model can be used to (i) predict model performance in complex downstream tasks that involve coding and emotional intelligence, (ii) predict LLM behavior with scaled test-time compute, and (iii) derive optimal-scaling rules for skills.

## 1.1 Related work

Scaling laws for deep neural networks: In recent years, researchers have studied scaling laws from different angles. Rosenfeld et al. [2019] provides experimental scaling laws that predict model loss as a function of training set size, model width, and model depth. Likewise, Kaplan et al. [2020] establishes scaling laws that primarily measure loss (perplexity) and not accuracy on downstream tasks or benchmarks. Motivated by the presence of hard limits on the size of trainable data sets but a hypothetical unlimited ability to scale models, the authors of Muennighoff et al. [2023] establish scaling laws in constrained data settings. They find that perhaps unsurprisingly, increasing computing provides diminishing returns if data does not scale. Gadre et al. [2024] addresses the gap between the assumptions in scaling laws and how training is performed in practice; in particular, they construct scaling laws that both perform well in the over-training regime and predict performance on downstream tasks. In a similar but distinct direction, some works try not only to estimate scaling laws but also respond to the following strategic question: 'Given a fixed FLOPs budget, how should one trade-off model size and the number of training tokens?' For example, Hoffmann et al. [2022] provides a partial answer, introducing the celebrated family of Chinchilla scaling laws and finding that training tokens and parameter size should roughly scale together. This contrasts with the older work of Kaplan et al. [2020] that provides a series of power laws that imply that simply increasing parameter count will provide good returns. Each of these referenced works trains models with a particular pretraining setting (e.g., architecture) at various sizes and ultimately seeks to predict test loss. Our focus is distinct, we fit scaling laws on existing benchmark data of multiple model families and predict LLM benchmark performance with minimal amount of data on the new family being predicted. Even though Allen-Zhu and Li [2024], study how LLMs can retain knowledge depending on their scale, the closest related works are Owen [2024], Ruan et al. [2024], Gadre et al. [2024]; we will provide a detailed comparison with their work throughout the paper.

LLMs latent skills: Given that the performance of large language models (LLMs) in different and diverse benchmarks is correlated, it makes sense to think that those models have some lowdimensional latent skills that are reflected in downstream tasks. In this direction, Ili´ c [2023] extracts a general intelligence factor ('g-factor') for LLMs using the Open LLM Leaderboard [Beeching et al., 2023] and GLUE [Wang et al., 2018] using factor analysis. They also verify that this 'g-factor' positively correlates with model size. In a similar direction Burnell et al. [2023] uses HELM [Liang et al., 2022] data to reveal that LLM intelligence may be constituted by three distinct, yet correlated factors. They also verify a positive correlation between model size and these latent skills, yet they do not propose a formal scaling law. In their study, the authors do not account for the training set size or model family information, leading to a poor fit of the regression model; this leaves good extrapolation as an open problem we address. In Kipnis et al. [2024], a unidimensional item response theory model is applied to each one of the 6 (filtered) benchmarks of the Open LLM Leaderboard. A factor analysis on the skill parameters shows that the main factor (carrying 80% of the data variability) is highly correlated with the 'grand' (average) score of LLMs. In a related but different direction, Maia Polo et al. [2024a,b] show that inferring low-dimensional latent skills of LLMs can make model evaluation much more efficient, saving up to 140x in computing power. In this work, we explicitly model LLM

skills as a function of computing resources, which enables the creation of accurate and interpretable scaling laws for benchmark performances.

## 2 Scaling laws for benchmark data

## 2.1 Problem Statement

In this section, we describe the setup we work on and what our objectives are. Within a family of LLMs i ( e.g. , LLaMA 3), our objective is to estimate the performance of a big LLM, e.g. , with 70 billion parameters, in a benchmark j , e.g. , MMLU, given evaluation data from smaller models in the same family. Let s represent the size of the LLM, defined as the number of parameters, and let t denote the number of training tokens. We define Y ij ( s, t ) ∈ [0 , 1] as the score of an LLM from family i , with size s and trained on t tokens, on benchmark j . Our goal is to approximate:

<!-- formula-not-decoded -->

Here, m [ · ] should be interpreted as a central tendency summary measure of a random variable, such as the mean or median 1 . Ideally, the model for µ ij will be simple and some of its parameters will be shared among model families and benchmarks; in this case, the model becomes more interpretable and more data can be used in the fitting process, making the model better estimated. From now on, we denote the set of model families as I = { 1 , · · · , I } and the set of benchmarks as J = { 1 , · · · , J } .

## 2.2 Previous approaches to scaling laws for benchmarks

The closest works to ours that propose models for µ ij ( s, t ) (2.1) are Owen [2024], Ruan et al. [2024], and Gadre et al. [2024]. While Gadre et al. [2024] indirectly model the quantity of interest via the LLMs perplexity in specific datasets, which might not be readily available, Owen [2024] and Ruan et al. [2024] model µ ij ( s, t ) directly through a regression model connecting compute and benchmark performance. One assumption they made is that the performance on benchmarks only depends on s and t through the total amount of training FLOPs, which can be approximated by c ( s, t ) = 6 st . That is, if σ : R → [0 , 1] denotes a fixed activation function, e.g. , the standard logistic (sigmoid) function, and γ j ∈ [0 , 1] , then it is assumed that

<!-- formula-not-decoded -->

where η ij : R 2 → R denotes a linear predictor such that η ij ( s, t ) = α ij + β ij log c ( s, t ) , which can be easily interpreted. Here, γ j adjusts the lower asymptote of µ ij and accounts for the probability of LLMs scoring correctly by chance. Owen [2024], in their best performing models, considers the case in which γ j = 0 (or adds a similar offset parameter to the model) and the parameters α ij and β ij are independent of the model family i . On the other hand, Ruan et al. [2024] consider both α ij and β ij to be family-dependent and, in their most general model, γ j can assume values in [0 , 1] .

The biggest issue with previous approaches when modeling µ ij is that they are either too restrictive or too flexible. From the restrictive side, they assume that (i) µ ij depends on s and t only through FLOPs, (ii) there are no family-dependent parameters, or (iii) the activation function σ is fixed and well-specified. From the flexibility side, Ruan et al. [2024] assume both α ij and β ij to be family dependent making estimation hard (or impossible) depending on the number of models we see for each family. From Ruan et al. [2024]: '(...) fitting such a scaling law can be tricky, as each model family f and downstream benchmark has its own scaling coefficients β f and α f . This means that scaling experiments, especially for post-training analysis, are often fitted on very few (3-5) models sharing the same model family (...).' Thus, in their experiments, they consider a different problem setting, where a large LLM has been trained and evaluated on some benchmarks and use their method to predict its performance on other benchmarks.

At the end of the day, Owen [2024] and Ruan et al. [2024] end up working in different setups: Owen [2024] does not use family information at prediction time, making their scaling law less accurate but more generalizable, and Ruan et al. [2024] assume families are important at prediction time but consider that the target model has already been trained, making their scaling law less applicable in practice and more interesting from an interpretability point of view. In this work we wish to instead predict the performance of a larger LLM without having to train it but taking family information into account, thus allowing practitioners to make decisions regarding investing resources into training

1 Because we minimize the Huber loss in this paper, we aim to approximate the median

large LLMs. Moreover, our formulation also allows interpretable insights from the data. Despite different setups, we make comparisons with Owen [2024] and Ruan et al. [2024] throughout this work by considering their applications/adaptations as baselines.

## 3 Scaling laws for LLMs skills with Sloth

## 3.1 Model architecture

We present a novel scaling law called Sloth , which introduces several modifications to (2.2). The key innovation of Sloth lies in its explicit modeling of the correlation structure between benchmarks, resulting in improved predictive accuracy and interpretability. Moreover, Sloth proposes that (i) LLM capabilities should scale with computing resources similarly across families up to an efficiency factor, (ii) benchmark performance can depend on s and t not only through the total number of FLOPs, and (iii) that the function σ can also be learned in cases in which predictive performance is important. We detail these points.

Inspired by the latent skills ( e.g. , reasoning, language modeling, instruction following) inferred from benchmark data in Burnell et al. [2023], Ili´ c [2023], Ruan et al. [2024], Gor et al., Kipnis et al. [2024], Maia Polo et al. [2024a,b], we propose creating a scaling law for LLMs skills by leveraging the correlation structure of the benchmarks; for example, we model how the construct 'reasoning' scales with compute instead of modeling benchmarks scores directly. The two major advantages of this approach are better performance prediction since we have fewer parameters to fit (reducing overfitting) and extra interpretability/insights. Concretely, we model η ij ( s, t ) 's simultaneously for benchmarks j ∈ J as each being a linear combination of the same low-dimensional latent skills θ i ( s, t ) ∈ R d plus a bias term b ∈ R J , where d ≪ J = |J | . Denote η i ( s, t ) ∈ R J as the vector of { η ij ( s, t ) } j ∈J . Mathematically, we have

<!-- formula-not-decoded -->

One can see that Λ ∈ R J × d encodes the correlation structure between the benchmarks; in particular, it tells us which benchmark measures overlapping (or distinct) skills. Interestingly, our model has a strong connection with factor analysis (FA) models, which we elaborate on in detail in Appendix C. In FA, the matrix Λ is known as factor loadings while θ i ( s, t ) are known as factors.

Next, we propose a model for θ i ( s, t ) . Inspired by models used in Economics, we use the family of translog production functions from stochastic frontier analysis [Kumbhakar and Lovell, 2003]:

<!-- formula-not-decoded -->

Note that (i) the intercept parameter α ik is indeed family-dependent while each skill slope is shared across families and (ii) θ i can depend on s and t not only through c ( s, t ) . In economic terms, the intercept term α ik can be interpreted as an efficiency measure of the family i in converting compute to performance for skill k and, in practice, will absorb all hidden factors specific to family i such as data quality, post-training factors, etc. . We note that the interaction term in ( log( s ) log( t ) ) accounts for the fact that the impact of log( s ) and log( t ) on skills might depend on each other; in Appendix D, we show some evidence that this is indeed the case. Additionally to the changes in η ij , we propose making the activation function σ trainable and specific to each benchmark j if needed. To that end, we adopt a semi-parametric single-index model using neural networks [Bietti et al., 2022]. To make the results more behaved if (out-of-support) generalization is needed, we assume σ j : R → [0 , 1] is given by a monotonic (increasing) neural network, which can be achieved by constraining its weights to be non-negative [Sill, 1997] and its last activation function to be sigmoid. We note, however, that one can always forgo training of the link function and instead assume a sigmoid structure as this simplifies model fitting and may make the model more interpretable. We give more details about model fitting in Section 3.2. Since Sloth is a simple neural network, both model fitting and prediction are done within seconds by a commercial laptop.

## 3.2 Model fitting

Assume that for each model family i we observe a set of tuples ( s, t ) 's denominated by E i . Then, we fit the model by solving the following minimization problem

<!-- formula-not-decoded -->

where ℓ δ is given by the Huber loss with hyperparameter δ = . 01 and µ ij ( s, t ) denotes the most general version of our model. We minimize the loss function via gradient descent using the Adam optimizer [Kingma and Ba, 2017] with a decaying learning rate. We parameterize γ j using the sigmoid transformation to guarantee the constraints are satisfied. Similarly, we truncate the weights of the two-hidden-layer neural network σ j to ensure the trainable function is increasing. If one desires, σ j 's can be set to fixed functions, e.g. , sigmoid, and γ j 's can be fixed beforehand. Unfortunately, the minimization problem is not convex as expected when fitting factor-analysis-like models; multiple initializations of the optimizer can be applied to guarantee a better fit.

## 3.3 Interpretability and practical considerations post model fitting

Ideally, to make reasonable interpretations of models like Sloth , we need its parameters to be identifiable, i.e. , the map connecting η ij ( s, t ) 's and model parameters should be bijective. Unfortunately, as in all exploratory factor models, this is not the case. However, we theoretically show in Appendix A that the model parameters are identifiable up to some parameter transformations. This observation allows us to find a configuration that makes the model more interpretable by mirroring a standard approach used in factor analysis, e.g. , in Chen et al. [2019]'s applications. The main idea is that we fit Sloth without any constraints and then find a configuration of skills (using factor rotation) that makes results interpretable. We detail the applied process in Appendix A.1.

## 4 Sloth in practice

In this section, we present experimental results that provide evidence of the usefulness of Sloth . We perform experiments on a set of twelve benchmarks and state-of-the-art LLM families, including LLaMa 3 [Dubey et al., 2024], Qwen 2 [Yang et al., 2024], and Yi 1.5 [Young et al., 2024]. We explore the following applications: (i) benchmark performance prediction for larger models from a specific LLM family, (ii) interpretability of the scaling of skills (can help practitioners allocate resources based on the skills of interest), and (iii) downstream tasks performance prediction.

## 4.1 Data

We expand the dataset made available by Ruan et al. [2024], including more models from the HuggingFace Open LLM leaderboard v1 [Beeching et al., 2023] and v2 [Fourrier et al., 2024]. In our extended dataset, we have a total of 30 families 2 , which 28 are on v1 of the Open LLM Leaderboard and 17 families measured on v2 of the Open LLM Leaderboard. Furthermore, there are 15 families at the intersection of the two versions. Furthermore, we collect data and present results on the performance of a variety of instruction-tuned versions of the base models we consider. As far as we are aware, our dataset is the most comprehensive among prior works on benchmark data scaling laws. Please check Appendix G for details on the included models.

## 4.2 Comparing scaling laws in terms of prediction errors

In this section, we compare the predictive power of different scaling laws in predicting LLM performance in all the considered benchmarks; we focus on the two versions of the Open LLM Leaderboard, which include 12 benchmarks: GSM8k [Cobbe et al., 2021], MATH Lvl 5 [Hendrycks et al., 2021], MMLU [Hendrycks et al., 2020], MMLU-PRO [Wang et al., 2024], BBH [Suzgun et al., 2022], GPQA [Rein et al., 2023], MUSR [Sprague et al., 2023], TruthfulQA [Lin et al., 2021], HellaSwag [Zellers et al., 2019], Winogrande [Sakaguchi et al., 2019], ARC [Clark et al., 2018], and IFEval [Zhou et al., 2023]. We apply a leave-one-out cross-validation algorithm to obtain test errors for each family of models. We consider base models and instruct models to belong to distinct

2 If we consider that instruct and base models are from different families, we end up with 53 families.

Figure 1: The figure shows the average (across LLM families) mean-absolute-error (MAE) (within a family) for different methods. Sloth performs competitively, with errors similar to or lower than the 'Size and Tokens' variant, indicating its effective inductive bias.

<!-- image -->

families (they will not share the same intercept in our model, for example), but we do not include the instruct (resp. base) family in the training set when the corresponding base (resp. instruct) family is in the test set. Moreover, we do not test older versions of recent families if they are available in the training set, e.g. , we do not include LLaMa 2 in the set of test families if LLaMa 3 is present in the training set. In this section, we present results for the two leaderboards separately; in Figures 12 and 17 of the Appendix, we also present results for the intersection of the two leaderboards.

In the first set of experiments, we consider the case in which only the smallest model of the test family is observed at training time. Because of that reason, we cannot fit the general scaling law in (2.2) in which both the intercept and slope are family dependent. In this scenario, we consider our main baselines to be (i) the model in (2.2) with shared intercept and slope [Owen, 2024] ('FLOPs (shared intercept)'), (ii) the same model but only with shared slope ('FLOPs'), (iii) a version of the PCA idea 3 proposed by Ruan et al. [2024] in which we predict the principal components using the FLOPs model with shared slope that are then mapped to the benchmark scores ('PCA + FLOPs'), (iv) and our model with trainable activation function but assuming Λ is identity ('Size and Tokens'; implies d = J ). Moreover, we include two versions of Sloth . In the 'basic' one, we fix σ to be sigmoid, and γ j 's are given by the 100% over the number of alternatives in the case of multiple-choice benchmarks 4 and 0 otherwise, except for TruthfulQA, which we compute the first percentile of the scores coming from the full Open LLM Leaderboard and fix the lower asymptote to that value. In previous sections, we mentioned that Sloth is parameter efficient; we include a parameter count analysis in Appendix F where we compare our model with other well-performing baselines.

3 We include more details about this approach in Appendix E.

4 When the benchmark has subsections with a different number of alternatives, we compute the asymptote parameters per subsection and then compute an overall asymptote using a weighted average in which the weights are proportional to the number of examples in each subsection.

Figure 2: Needed skills for each benchmark. In this figure, we report the estimated loadings Λ and, based on their values, we give them appropriate names.

<!-- image -->

2.2

Figure 4: In this figure, we plot the skill levels (output) subtracted by the family-specific intercept terms against inputs in the x and y-axis. From these plots, we can see how each one of the inputs can differently affect the production of skills. For example, 'Reasoning' showed to be more affected by model size than tokens when compared to other skills. Moreover, 'Knowledge' is more influenced by inputs (level curves are steeper) in general, while the other skills should be more sensitive to other family-dependent factors.

<!-- image -->

Figure 1 gives the results for the first set of experiments. It depicts the average mean absolute error of all methods when predicting LLM benchmark performance, which is measured in percentage points. It shows the competitiveness of Sloth in terms of prediction quality. One important thing to notice is that Sloth errors are similar or lower than the 'Size and Tokens' variant, suggesting that the assumed low-dimensional structure between benchmarks results is a good inductive bias. We highlight that the analysis includes recent families like LLaMa 3, Qwen 2, and Yi 1.5. For more details on the tested models and extra related results, including model-specific results, please check Appendix H.2. The extra results are qualitatively similar

Figure 3: Running Sloth with shared intercept can offer a great way to model scaling laws that are familyindependent.

<!-- image -->

to the ones in Figure 1, in which Sloth often beats the baselines. In a second set of experiments, we consider the case in which the two smallest models of the test family are observed at training time. In this way, we can fit (2.2) making both parameters family dependent. The results are qualitatively similar to the one presented on Figure 1 and are presented in Appendix H.3. Moreover, we include Figure 10 in the appendix, which is a version of Figure 1 in which Mean Absolute Percentage Error (MAPE) is used instead of MAE; results are qualitatively similar.

In an extra set of experiments, we show that family-specific intercept models are not always needed; we can still get good prediction results for some benchmarks even if we consider a shared intercept between families. The advantage of this approach is that we can claim for a general scaling law that holds for all families. Figure 3 shows us a subset 5 of Figure 12 in the appendix and it is built under the same conditions as Figure 1. It is possible to see that, for a subset of benchmarks Sloth with shared intercept is a strong alternative to the FLOPs model used by Owen [2024]. In some cases, it gets similar prediction errors relative to more complete versions of Sloth .

5 We selected the best d for both versions of Sloth .

Instruction

Figure 5: We compare the skills of base (x-axis) and instruction-tuned models (y-axis); if a model lies on the 45-degree line, it means that the model has the same skill level in its base and instruct versions. Gains from instruction tuning (IT) for different families on three latent skills. Findings include a large and positive impact on 'Instruction Following' and that provides much larger variations in this skill when compared to inputs seen in Figure 4. Moreover, IT had a moderate and negative effect on 'Reasoning' and mixed effects on 'Knowledge'.

<!-- image -->

## 4.3 Interpreting the latent skills

In this section, we use the intersection between the two leaderboards, aiming to get more insights from the combined data. Since we have an identifiability result for the 'basic' version of Sloth , in which we fix the lower asymptotes γ j 's and the link function to be sigmoid (see Section A), we opt for interpreting that version of the model. We set d = 3 as that model version achieved the best prediction results in Figure 12. Figure 2 illustrates the model loadings, Λ , from which we assign names to the three dimensions based on our subjective interpretation. We include the loadings for d = 2 and d = 4 in Appendix I. To complement our exploration, we include Figure 4, which gives us the level curves of different skills (disregarding the family-specific intercept term), and Figure 5 in the appendix that compares the skills of base and instruction-tuned models; in this figure, we include LLM families with more number of models. In both figures, the numbers are given in terms of standard deviations as the skills are standardized to have a zero mean and unitary standard deviation.

Reasoning skill: The first dimension, with strong loadings from benchmarks such as GSM8K, MATH, GPQA, MMLU(-PRO), BBH, and MUSR, is labeled 'Reasoning.' The benchmarks GSM8k and MATH Lvl 5 consist entirely of mathematical word problems while MMLU/MMLU-PRO and GPQA also contain mathematical and advanced science questions. On the other hand, BBH includes logic deduction and linguistic reasoning puzzles. The strong dependence of BBH on the 'Reasoning' skill suggests that in language models, there is an association between logical reasoning, general linguistic ability, and mathematical ability. Finally, MuSR is a benchmark that evaluates 'multistep soft reasoning tasks specified in a natural language narrative' [Sprague et al., 2023]. Figure 4 shows that Reasoning is primarily a function of model size, with a small dependence on the number of training tokens used. Moreover, the first plot of Figure 5 in the appendix compares base models versus their instruction-tuned versions in terms of Reasoning, and we found that there is no clear rule: instruction tuning can either increase or decrease the ability of an LLM to reason. These findings are robust for different values of d as we can see in the figures of Appendix I.

Knowledge skill: The second dimension is positively loaded on ARC, HellaSwag, and Winogrande. These three benchmarks measure the ability of LLMs to remember common sense and basic knowledge; we denominate this skill as 'Knowledge'. More specifically, ARC consists of grade school-level science questions, HellaSwag is meant for sentence completion for common scenarios, and Winogrande common sense pronoun resolution problems. Contrasting with Reasoning, Figure 4 shows that Knowledge is highly influenced by both model size and number of training tokens. Moreover, we can see that the range of standard deviations in the middle plot is much greater than in the other two plots, giving us evidence that this skill might be more sensitive to increases in compute resources and less dependent on the LLM families themselves. On the other hand, Figure 5 in the appendix does not show any strong evidence of the effect of instruction tuning on this skill. These findings are similar to the ones reported in Appendix I for different values of d .

Instruction following skill: IFEval, which is positively and heavily loaded in this skill, measures how well language models produce answers that follow a verifiable set of instructions; for example, including a keyword x number of times in responses. Therefore, we call it 'Instruction Following'. An interesting fact is that instruction tuning has a strong positive effect on this skill for all depicted families we can see in Figure 5 in the appendix. The effect can also be observed in Figure 28 of the appendix. When d = 2 , instruction following gets mixed with other skills and we are not able to see

this effect. Regarding Figure 4, we see that Instruction Following depends on both model size and tokens. Unfortunately, this interpretation does not hold when d = 4 as seen in Appendix I.

## 4.4 Predicting LLM performance on downstream tasks

Another useful application of Sloth , which is inspired by Ruan et al. [2024], is to predict the performance in a downstream task for a large model from a relatively small number of prior performance observations from that task. We use Sloth to estimate the latent skills of hypothetical LLMs and then use them to predict the performance of those LLMs in downstream tasks. With this approach, we expand on the experiments of Ruan et al. [2024], which do not consider performance prediction of hypothetical LLMs.

The basic prediction pipeline is as follows. First, use standard LLM leaderboards to fit a scaling law for skills using Sloth . Second, use existing LLM performance on the downstream task to model how performance can be predicted from latent skills. Third, use Sloth to predict the skills of a (hypothetical) LLM of interest, e.g. , a larger version of an existing LLM. Finally, use the model fitted in the second step to predict the performance of the hypothetical model in the downstream task.

We evaluate this pipeline on two downstream tasks, predicting the performance of meta-llama-3-70B and meta-llama-3-70B-instruct on code completion and meta-llama-3-70B-instruct on emotional intelligence tasks. We fit the same model shown in Section 4.3, but do not include meta-llama-3-70B or meta-llama-3-70B-instruct in the training set (see Figure 32 for the loadings of the latent skills, which is similar to Figure 2). Next, using either HumanEval [Chen et al., 2021] or EQ bench data [Paech, 2024], we fit a regression model with logistic link using latent skills as features and performance on the downstream task as target. Together, this provides us with sufficient information to predict the performance of the held-out models on both tasks with decent accuracy. Figure 6 depicts this logistic curve and the actual values for HumanEval;

Figure 6: Predicting model performance in complex downstream tasks like code completion ('HE') for LLaMa 3 70B (base/instruct). In the first step, we fit Sloth without including LLaMa 3 70B (base/instruct) in the training set. In the second step, we fit a regression model connecting skills and downstream performance. Finally, we predict LLaMa 3 70B (base/instruct) performance from their predicted Sloth skills.

<!-- image -->

EQ bench results can be found in Appendix J. Moreover, we can see that 'Reasoning' is by far the most important skill in predicting coding ability while a mixture of 'Reasoning' and 'Knowledge' is needed for emotional intelligence (see Figure 32 for a more accurate interpretation of the loadings).

In Appendix J, a similar test is provided for agentic capability measured by AgentBench [Liu et al., 2023], although to avoid overfitting, in this case, we must fit Sloth with no family-specific intercept.

## 4.5 Predicting performance behavior with scaled inference compute

In this section, we demonstrate how our method, Sloth , combined with concepts like Item Response Theory (IRT) [Reckase, 2009], can predict how the performance of an LLM scales with increased inference through repeated sampling [Brown et al., 2024]. For this experiment, we utilize the MATH dataset [Hendrycks et al., 2021], made available by Brown et al. [2024], which evaluates 10 LLMs also included in our dataset. These models are part of the LLaMa 3 Instruct, Gemma, and Pythia families. The process is:

1. Train Sloth on our full dataset of 12 benchmarks, excluding the largest models in the LLaMa 3 Instruct, Gemma, and Pythia families.
2. Fit a logistic regression model for each MATH question in Brown et al. [2024]'s data, using the skills θ i ( s, t ) of the 7 training LLMs as covariates to predict the probability of correctly solving those problems.
3. Estimate the skills for the three test models using our scaling law, then predict their probabilities of answering each MATH question correctly via the logistic regressions, and then predict the pass@ k metrics for the 3 test models using those probabilities. The predicted pass@ k metric for a certain LLM is given by the average of 1 -(1 -ˆ p j ) k 's (across j 's) if ˆ p j is the predicted probability of the LLM of interest getting question j correct.

Figure 7 illustrates that Sloth can accurately predict test-time scaling behavior for these models. Note that unlike scaling law in Brown et al. [2024], Sloth can predict inference compute gains for hypothetical LLMs before committing resources to training them. This highlights a practical application where practitioners can estimate the potential performance of a hypothetical model at test time, given a computational budget.

## 4.6 Compute-optimal scaling of skills

One relevant question is: given a certain budget in FLOPs, how do we invest it to maximize one skill of interest? This type of analysis is novel for skills and it has only been carried out for validation loss ( e.g. , by Kaplan et al. [2020], Hoffmann et al. [2022]). A summary of the mathematical formulation of this question is given in the following and exposed in detail in Appendix B. For each model family i and skill k , we have

<!-- formula-not-decoded -->

Define u = log( s ) , v = log( t ) , and l = log( c ) -log(6) . For simplicity, here we consider a simple, but widely used [Kaplan et al., 2020, Hoffmann et al., 2022], compute budget constraint 6 st = c , which is equivalent to u + v = l . We add extra constraints on u and v based on the training data support to avoid unreasonable out-of-support predictions. We optimize

Figure 7: Sloth can accurately predict test-time scaling behavior for LLaMa 3 Instruct, Gemma, and Pythia models for MATH performance when coupled with Item Response Theory.

<!-- image -->

<!-- formula-not-decoded -->

Substituting v = l -u reduces this to a quadratic function g ik ( u ) , whose maximizer, within the observed range U = [max( l -v, u ) , min( l -v, u )] , determines the optimal allocation. Our analysis indicates that the optimal values for s and t do not depend on the model family. Table 1 reports our results for instruction following; a table with results for the other two skills is shown in Appendix B.

## 5 Conclusion

In conclusion, we have introduced the Sloth scaling laws as a novel approach to predicting the performance of large language models across benchmarks and model families. By leveraging the correlations between benchmark scores and assuming that LLM performance is governed by a set of interpretable, low-dimensional latent skills, our approach offers a more efficient and flexible framework for understanding and predicting LLM behavior. The ability to estimate model performance across a variety of bench-

Table 1: Optimal parameter and token allocation.

|   FLOPs (1e19) |   Instruction Params (B) Tokens |   Following (T) |
|----------------|---------------------------------|-----------------|
|            100 |                            0.16 |            1.04 |
|            578 |                            0.3  |            3.24 |
|           3346 |                            0.72 |            7.78 |
|          19360 |                            2.15 |           15    |
|         112005 |                           12.44 |           15    |
|         648000 |                           72    |           15    |

marks and tasks, even with minimal data from individual model families, highlights the practical utility of Sloth scaling laws. Our empirical results demonstrate that Sloth can accurately predict the performance of LLMs across multiple benchmarks while providing insights into the relationship between computational resources and model capabilities.

Limitations. From the predictive side of Sloth , we believe that the main limitation is that the model is still dependent, most of the time, on seeing data from at least one LLM from the LLM family of interest. Moreover, we train the link function in the best version of Sloth using flexible neural networks, which can interpolate data very well, but have no guarantee of extrapolation when the (hypothetical) LLM of interest is very different from others in the training set. From the interpretability side, we only understand the identification problems, such as transformations in the latent space, that can arise in a simple instance of Sloth : fixed activation function σ and γ j 's. This fact limits our understanding and interpretability of the most advanced versions of the model.

## 6 Acknowledgements

This paper is supported by the National Science Foundation (NSF) grants DMS-2027737, DMS2113373, DMS-2414918, and a gift from OpenAI.

## References

- Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.3, knowledge capacity scaling laws. arXiv preprint arXiv:2404.05405 , 2024.
- Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open llm leaderboard., 2023. URL https://huggingface.co/spaces/HuggingFaceH4/open\_llm\_leaderboard,2023.
- Alberto Bietti, Joan Bruna, Clayton Sanford, and Min Jae Song. Learning single-index models with shallow neural networks. Advances in Neural Information Processing Systems , 35:9768-9783, 2022.
- Jeremy Biggs. Factor\_analyzer documentation. Release 0.3 , 1, 2019.
- Christopher M Bishop and Nasser M Nasrabadi. Pattern recognition and machine learning , volume 4. Springer, 2006.
- Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787 , 2024.
- Ryan Burnell, Han Hao, Andrew RA Conway, and Jose Hernandez Orallo. Revealing the structure of language model capabilities. arXiv preprint arXiv:2306.10062 , 2023.
- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. 2021.
- Yunxiao Chen, Xiaoou Li, and Siliang Zhang. Joint maximum likelihood estimation for highdimensional exploratory item factor analysis. Psychometrika , 84:124-146, 2019.
- Leshem Choshen, Yang Zhang, and Jacob Andreas. A hitchhiker's guide to scaling law estimation. arXiv preprint arXiv:2410.11840 , 2024.
- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. URL https://arxiv.org/abs/1803.05457 .
- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/ abs/2110.14168 .
- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 , 2024.
- Clémentine Fourrier, Nathan Habib, Alina Lozovskaya, Konrad Szafer, and Thomas Wolf. Open llm leaderboard v2. https://huggingface.co/spaces/open-llm-leaderboard/open\_llm\_ leaderboard , 2024.
- Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, Rui Xin, Marianna Nezhurina, Igor Vasiljevic, Jenia Jitsev, Luca Soldaini, Alexandros G. Dimakis, Gabriel Ilharco, Pang Wei Koh, Shuran Song, Thomas Kollar, Yair Carmon, Achal Dave, Reinhard Heckel, Niklas Muennighoff,

and Ludwig Schmidt. Language models scale reliably with over-training and on downstream tasks, 2024. URL https://arxiv.org/abs/2403.08540 .

- Maharshi Gor, Tianyi Zhou, III Hal Daumé, and Jordan Boyd-Graber. Do great minds think alike? investigating human-ai complementarity for question answering.
- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations , October 2020.
- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874 .
- Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556 , 2022.
- David Ili´ c. Unveiling the general intelligence factor in language models: A psychometric approach. arXiv preprint arXiv:2310.11616 , 2023.
- Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. URL https://arxiv.org/abs/2001.08361 .
- Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980 [cs] , January 2017.
- Alex Kipnis, Konstantinos Voudouris, Luca M Schulze Buschoff, and Eric Schulz. metabench - a sparse benchmark to measure general ability in large language models. arXiv preprint arXiv:2407.12844 , 2024.
- Subal C Kumbhakar and CA Knox Lovell. Stochastic frontier analysis . Cambridge university press, 2003.
- Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110 , 2022.
- Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958 , 2021.
- Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. AgentBench: Evaluating LLMs as Agents, October 2023.
- Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin. tinybenchmarks: evaluating llms with fewer examples. arXiv preprint arXiv:2402.14992 , 2024a.
- Felipe Maia Polo, Ronald Xu, Lucas Weber, Mírian Silva, Onkar Bhardwaj, Leshem Choshen, Allysson Flavio Melo de Oliveira, Yuekai Sun, and Mikhail Yurochkin. Efficient multi-prompt evaluation of llms. arXiv preprint arXiv:2405.17202 , 2024b.
- Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained language models, 2023. URL https://arxiv.org/abs/2305.16264 .
- David Owen. How predictable is language model benchmark performance? arXiv preprint arXiv:2401.04757 , 2024.
- Samuel J. Paech. Eq-bench: An emotional intelligence benchmark for large language models, 2024. URL https://arxiv.org/abs/2312.06281 .

- Mark D Reckase. 18 multidimensional item response theory. Handbook of statistics , 26:607-642, 2006.
- MDReckase. Multidimensional Item Response Theory . Springer Science &amp; Business Media, 2009.
- David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&amp;a benchmark, 2023. URL https://arxiv.org/abs/2311.12022 .
- Jonathan S. Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit. A constructive prediction of the generalization error across scales, 2019. URL https://arxiv.org/abs/1909.12673 .
- Yangjun Ruan, Chris J. Maddison, and Tatsunori Hashimoto. Observational Scaling Laws and the Predictability of Language Model Performance, July 2024.
- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale, 2019. URL https://arxiv.org/abs/1907. 10641 .
- Joseph Sill. Monotonic networks. Advances in neural information processing systems , 10, 1997.
- Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. Musr: Testing the limits of chain-of-thought with multistep soft reasoning. arXiv preprint arXiv:2310.16049 , 2023.
- Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging bigbench tasks and whether chain-of-thought can solve them, 2022. URL https://arxiv.org/ abs/2210.09261 .
- Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461 , 2018.
- Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark, 2024. URL https://arxiv.org/abs/2406.01574 .
- An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671 , 2024.
- A Yates. Multivariate exploratory data analysis: A perspective on exploratory factor analysis . State University of New York Press, 1987.
- Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652 , 2024.
- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence?, 2019. URL https://arxiv.org/abs/1905.07830 .
- Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models, 2023. URL https://arxiv.org/abs/2311.07911 .

## A Identifiability of model parameters and interpretability

To interpret Sloth parameters, we need to guarantee they are identifiable. Given that our scaling law models the function µ ij ( s, t ) = E [ Y ij ( s, t )] , that condition is equivalent to the following statement: if two sets of parameters are responsible for characterizing µ ij ( s, t ) , then those set of parameters should be the same up to predictable variations such as translations or rotations. To prove identifiability, we work with a fixed and invertible σ , as usually done in the literature, and assume γ j 's are fixed. The last condition is reasonable since these constants are usually known beforehand, e.g. , it is well accepted that the lower asymptote γ j for MMLU [Hendrycks et al., 2020] performance is 25% which is given by 100% divided by the number of multiple-choice alternatives. Denote our fixed design matrix as X ∈ R n × p , where each row is given by an LLM and p equals 3 plus the number of families, and define

<!-- formula-not-decoded -->

such that the rows of XB ∈ R n × d give the skills vectors θ ( i ) ≜ ( XB ) ( i ) 's of all models in our dataset. Here n denotes the total number of models in the dataset and m is the total number of model families. To prove identifiability, we adopt standard assumptions from the factor analysis literature [Chen et al., 2019] or regression literature, which assumes that the skills vectors θ ( i ) ∈ R 1 × d 's are standardized, i.e. , their average is null while their covariance matrix is fixed, rank (Λ) = d , and rank ( X ) = p .

Assumption A.1 (Identifiability constraints) . Assume that

<!-- formula-not-decoded -->

and that rank (Λ) = d and rank ( X ) = p , where θ ( i ) denotes the i -th row of XB and Ψ is a positive definite matrix.

One possible choice for the covariance matrix is Ψ = I d [Chen et al., 2019], which assumes uncorrelated skills. One implicit implication of Assumption A.1 is that n ≥ p ≥ d must be satisfied, otherwise the covariance matrix cannot be full rank. This condition is satisfied in our experiments. Under Assumption A.1, we show the identifiability of the model parameters up to a transformation of Λ tied to a transformation of B , which leaves the outputs of the model unchanged. This means that we can potentially approximate the true values for Λ and B up to a transformation, which is usually the norm within the class of exploratory factor analysis models.

Theorem A.2. Given that the true set of model parameters is (Λ , b, B ) , if there is another set of parameters ( ˜ Λ , ˜ b, ˜ B ) that satisfy

<!-- formula-not-decoded -->

then, under the Assumption A.1, we have ˜ b = b , ˜ Λ = Λ M , and ˜ B = B ( M ⊤ ) -1 for an invertible matrix M ∈ R d × d . In particular, M is orthogonal if Ψ = I d , i.e., M ⊤ M = MM ⊤ = I d .

Proof. We start proving that b = ˜ b . Because σ is invertible, we get

<!-- formula-not-decoded -->

and consequently by the standardization of the latent skills

<!-- formula-not-decoded -->

Now, we prove that ˜ Λ = Λ M . Given that b = ˜ b , we have

<!-- formula-not-decoded -->

and consequently by the standardization of the latent skills

<!-- formula-not-decoded -->

By Cholesky's decomposition, we can write Ψ = LL ⊤ , for a lower triangular matrix L . If we define Λ ′ ≜ Λ L and ˜ Λ ′ ≜ ˜ Λ L , then

<!-- formula-not-decoded -->

Because rank (Λ) = d , we have that rank (Λ ′ ) = d and we claim that ˜ Λ ′ = Λ ′ U for an orthogonal matrix U ∈ R d × d . To see that, first, realize that

- rank (Λ ′ ) = rank (Λ ′ Λ ′⊤ ) = rank ( ˜ Λ ′ ˜ Λ ′⊤ ) = rank ( ˜ Λ ′ ) . We see this by realizing that the null spaces of Λ ′⊤ and Λ ′ Λ ′⊤ are the same: for an arbitrary vector z , Λ ′⊤ z = 0 ⇒ Λ ′ Λ ′⊤ z = 0 and Λ ′ Λ ′⊤ z = 0 ⇒ Λ ′⊤ Λ ′ Λ ′⊤ z = 0 ⇒ Λ ′⊤ z = 0 , where the last implication follows from the assumption that Λ ′⊤ Λ ′ is full rank (rank (Λ ′ ) = d ). Because the null spaces of Λ ′⊤ and Λ ′ Λ ′⊤ are the same, their ranks should be the same as well. The same reasoning applies to ˜ Λ ′ ˜ Λ ′⊤ and ˜ Λ ′ , proving this intermediate result.
- Because Λ ′ and Λ ′ Λ ′⊤ have the same rank, the column space of these two matrices must be the same as the columns of Λ ′ Λ ′⊤ are given by linear combinations of columns of Λ ′ . Same for ˜ Λ ′ and ˜ Λ ′ ˜ Λ ′⊤ . Consequently, the column spaces of Λ ′ and ˜ Λ ′ are the same.

Because the column spaces of Λ ′ and ˜ Λ ′ are the same, there must be an invertible matrix U such that ˜ Λ ′ = Λ ′ U . But then

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Because ˜ Λ ′ = Λ ′ U , we have that

<!-- formula-not-decoded -->

If Ψ = I d , then L = I d and M = U .

Finally, we prove that ˜ B = B ( M ⊤ ) -1 . From previous considerations, we can write

<!-- formula-not-decoded -->

If Ψ = I d , then L = I d and ( M ⊤ ) -1 = U .

From our proof, we can see that the matrix M is dependent on the specification of Ψ .

## A.1 Interpretability

In practical situations, it is hard to fix the covariance matrix of skills to something meaningful before fitting the model, as suggested in Section A.1. To make the model interpretable, we mirror a standard approach used in factor analysis, e.g. , in Chen et al. [2019]'s applications. First, we fit Sloth without any constraints on the covariance of skills obtaining the estimates ( ˆ Λ , ˆ b, ˆ B ) . Second, we find the matrix A ∈ R d × d such that the skills X ˆ BA have covariance identity, update ˆ B ← ˆ BA , and update ˆ Λ ← ˆ Λ( A ⊤ ) -1 so the model outputs remains unchanged, because ˆ Λ( X ˆ B ) ⊤ = ˆ Λ( A ⊤ ) -1 ( X ˆ BA ) ⊤ .

<!-- formula-not-decoded -->

Third, we find a matrix M ∈ R d × d such that ˆ Λ M is easily interpretable ( e.g. , it is a sparse matrix); there are different methods to find M and, in this paper, we use the Geomin [Yates, 1987, Chen et al., 2019] oblique rotation method to find a suitable M using the Python package FactorAnalyzer [Biggs, 2019]. We then update ˆ Λ ← ˆ Λ M and, to make the model invariant, we also update ˆ B ← ˆ B ( M ⊤ ) -1 ; the resulting skills are still guaranteed to have unitary standard deviations, so their covariance equals their correlation. Finally, we standardize the columns of the skills X ˆ B to have zero mean, while keeping the correlation structure unchanged. This last step implies that ˆ b must be translated to make the model invariant.

## B Compute-optimal scaling

In this section, we derive compute-optimal scaling rules for skills. Specifically, given a language model family and a particular skill, our goal is to determine the model configuration that maximizes performance under a fixed computation budget 6 st = c .

Consider a model family i with skill k defined as

<!-- formula-not-decoded -->

Letting u = log( s ) , v = log( t ) , and l = log( c ) -log(6) , we formulate the optimization problem as:

<!-- formula-not-decoded -->

This reduces to a simpler optimization problem in terms of u alone:

<!-- formula-not-decoded -->

To prevent our compute-optimal scaling method from making unreasonable predictions, we further restrict its solution to the ranges of u and v observed in the training data. Hence, we impose constraints u ∈ [ u, u ] and v = l -u ∈ [ v, v ] , where bounds u, u, v, v are set based on quantiles from training data. Combining these constraints yields:

<!-- formula-not-decoded -->

Thus, the optimization problem becomes:

<!-- formula-not-decoded -->

which is straightforward to solve. Specifically, if β k 2 &gt; 0 , g ik ( u ) is a concave parabola, and the maximizer is either at the vertex (if it falls within U ) or at one of the interval endpoints max( l -v, u ) or min( l -v, u ) .

Table 2 extends the results in Table 1 for all skills.

| FLOPs (1e19)   | Reasoning   | Reasoning   | Knowledge   | Knowledge   | Instruction Following   | Instruction Following   |
|----------------|-------------|-------------|-------------|-------------|-------------------------|-------------------------|
|                | Params (B)  | Tokens (T)  | Params (B)  | Tokens (T)  | Params (B)              | Tokens (T)              |
| 100            | 0.93        | 0.18        | 0.16        | 1.04        | 0.16                    | 1.04                    |
| 578            | 5.36        | 0.18        | 0.16        | 6.02        | 0.30                    | 3.24                    |
| 3346           | 30.98       | 0.18        | 0.37        | 15.0        | 0.72                    | 7.78                    |
| 19360          | 72.0        | 0.45        | 2.15        | 15.0        | 2.15                    | 15.0                    |
| 112005         | 72.0        | 2.59        | 12.44       | 15.0        | 12.44                   | 15.0                    |
| 648000         | 72.0        | 15.0        | 72.0        | 15.0        | 72.0                    | 15.0                    |

Table 2: Optimal allocation of parameters (B) and tokens (T) across skills for various compute budgets (FLOPs).

## C Connections with factor analysis

Sloth is heavily inspired by (exploratory) factor analysis models. Factor analysis is a statistical technique used to identify underlying relationships between observed variables by reducing the data's

dimensionality [Bishop and Nasrabadi, 2006, Chen et al., 2019]. It assumes that multiple observed variables are influenced by a smaller number of unobserved/latent variables called factors (skills θ ( i ) , in our case). These factors help explain the correlations among the observed variables. The method aims to model the observed variability and reveal the structure behind the data by estimating the factor loadings ( Λ , in our case). The classical model assumes

<!-- formula-not-decoded -->

where Y i is a vector of variables of interest and ε i is an error term. There are versions for the factor model in which a nonlinear model for Y i is assumed, e.g. , in item response theory (IRT) [Reckase, 2006, Chen et al., 2019]. It is usually the case that θ i is estimated using a random effects model, i.e. , practitioners place a prior distribution on θ i . In our work, we assume θ i is given by a function of observable covariates and a family-specific intercept, which is fitted using data.

## D Motivating the interaction term in Sloth

As shown in Section 3, we include an interaction term between log( s ) and log( t ) . In the first place, we consider this as a natural extension of the model that depends on s and t only through FLOPs, since we recover that formulation if β k 1 = β k 2 and β k 3 = 0 . In the second place, we believe that the dependence of benchmark performances on log( s ) depends on log( t ) (and possibly vice-versa). To motivate this idea we show some plots for two benchmarks we use: MMLU-PRO and BBH. For these plots, we only keep families with a higher number of models. First, realize that in both Figures 8 and 9, the performance within families in the middle plot can be well approximated by a line. Also, the slope of this line has a strong relation with the number of tokens in the last plots. For example, Pythia was trained in a small dataset and its (hypothetical) slopes on the second plot are almost zero in both cases. On the other hand, Qwen2 was trained on more data and its (hypothetical) slope on the middle plots is high. Certainly, this relationship does not always exist, but adding an interaction term in our model helps us to leverage this pattern when it exists.

Figure 8: Inputs vs MMLU-PRO scores.

<!-- image -->

Figure 9: Inputs vs BBH scores.

<!-- image -->

## E PCA approach formulation

We follow the ideas of Ruan et al. [2024] as closely as possible to create a prediction method. Moreover, we follow their code 6 and apply PCA with the same set of hyperparameters. Assume we have a matrix of scores Y ∈ [0 , 1] n × J in which columns represent benchmarks and each row represents a language model. We compute the covariance matrix of benchmark scores using Y and then compute its eigenvector matrix U , where the columns give the ordered eigenvectors (from the highest eigenvalue to the lowest one). To reduce the dimensions of Y , we keep only the first d columns of Y U , resulting in ˜ Y ∈ R n × d . For each column of ˜ Y (principal components; PCs), we train a linear regression model using logFLOPs as the covariate; in this case, either the intercept or both the intercept and slope can be family-dependent. At test time, we predict the PCs of a held-out model and then go back to the original coordinate axis to obtain the final predictions by computing ∑ d j =1 ˆ PC j U · ,j ∈ R J .

## F Sloth is parameter-efficient: parameter count analysis

When the number of model families f is moderately large, Sloth actually uses fewer parameters than the top-performing baselines (both of which consider family-specific intercept or train the activation function). This is because of the assumed latent skill structure of Sloth . For example, with d = 3 and we use J = 12 benchmarks:

- Sloth : 12 · ( d +2) + d · ( f +3) = 69 + 3 f parameters
- 'FLOPs' baseline: 12 · ( f +3) = 36 + 12 f parameters
- 'Size and Tokens' baseline: 12 · ( f +5) = 50 + 12 f parameters

So for f ≥ 4 , Sloth uses fewer parameters than either baseline.

6 See https://github.com/ryoungj/ObsScaling .

## G Models in our dataset

Table 3 gives a detailed view of our dataset. The column 'Family' considers that base and instruct models are from different families, while 'OriginalFamily' does not. The column 'TestFamily' tells if that specific family is considered to be part of the test set in our experiment while the remaining three columns tell if the data is available for these specific benchmarks. For the EQ data, only the following models are available 'gemma-7b-it', 'llama-2-13b-chat', 'llama-2-70b-chat', 'llama-2-7b-chat', 'meta-llama-3-70b-instruct', 'meta-llama-3-8b-instruct', 'qwen1.5-1.8b-chat', 'qwen1.5-14b-chat', 'qwen1.5-32b-chat', 'qwen1.5-4b-chat', 'qwen1.5-7b-chat', 'yi-1.5-34b-chat', 'yi-1.5-6b-chat', 'yi-1.5-9b-chat', 'yi-34b-chat'.

|       | Model                          | Family                        | OriginalFamily        | TestFamily   | Leaderboard1   | Leaderboard2   | HumanEval   |
|-------|--------------------------------|-------------------------------|-----------------------|--------------|----------------|----------------|-------------|
| 0     | bloom                          | bloom                         | bloom                 | True         | True           | False          | True        |
| 1     | bloom-1b1                      | bloom                         | bloom                 | True         | True           | True           | True        |
| 2     | bloom-3b                       | bloom                         | bloom                 | True         | True           | True           | True        |
| 3     | bloom-560m                     | bloom                         | bloom                 | True         | True           | True           | True        |
| 4     | bloom-7b1                      | bloom                         | bloom                 | True         | True           | True           | True        |
| 5     | blossom-v5.1-34b               | blossom-v5.1                  | yi-1.5                | False        | True           | True           | False       |
| 6     | blossom-v5.1-9b                | blossom-v5.1                  | yi-1.5                | False        | False          | True           | False       |
| 7     | codegen-16b-nl                 | codegen-nl                    | codegen               | True         | True           | False          | True        |
| 8     | codegen-6b-nl                  | codegen-nl                    | codegen               | True         | True           | False          | True        |
| 9     | codellama-13b                  | codellama                     | codellama             | True         | True           | False          | True        |
| 10    | codellama-34b                  | codellama                     | codellama             | True         | True           | False          | True        |
| 11    | codellama-70b                  | codellama                     | codellama             | True         | True           | False          | True        |
| 12    | codellama-7b                   | codellama                     | codellama             | True         | True           | False          | True        |
| 13    | deepseek-coder- 1.3b-base      | deepseek- coder-base          | deepseek-             | True         | True           | False          | True        |
| 14    | deepseek-coder- 33b-base       | deepseek- coder-base          | coder deepseek- coder | True         | True           | False          | True        |
| 15    | deepseek-coder- 6.7b-base      | deepseek- coder-base          | deepseek- coder       | True         | True           | False          | True        |
| 16    | dolly-v2-12b                   | dolly-v2                      | pythia                | True         | True           | True           | True        |
| 17    | dolly-v2-3b                    | dolly-v2                      | pythia                | True         | False          | True           | False       |
| 18    | dolly-v2-7b                    | dolly-v2                      | pythia                | True         | True           | True           | False       |
| 19    | dolphin-2.9.1-yi- 1.5-34b      | dolphin-2.9.1- yi-1.5         | yi-1.5                | True         | True           | True           | False       |
| 20    | dolphin-2.9.1-yi- 1.5-9b       | dolphin-2.9.1- yi-1.5         | yi-1.5                | True         | True           | True           | False       |
| 21    | dolphin-2.9.2- qwen2-72b       | dolphin-2.9.2- qwen2          | qwen2                 | True         | False          | True           | False       |
| 22    | dolphin-2.9.2- qwen2-7b        | dolphin-2.9.2- qwen2          | qwen2                 | True         | False          | True           | False       |
| 23    | falcon-180b                    | falcon                        | falcon                | True         | True           | False          | False       |
| 24 25 | falcon-40b falcon-40b-instruct | falcon                        | falcon falcon         | True True    | True           | True           | False       |
|       |                                | falcon-instruct               | falcon                |              | False          | True           | False       |
| 26    | falcon-7b                      | falcon                        |                       | True         | True           | True           | False False |
| 27 28 | falcon-7b-instruct gemma-2-2b  | falcon-instruct               |                       | True True    | True           | True           | False       |
| 29    | gemma-2-2b-it                  | gemma-2                       | falcon gemma-2        | True         | False False    | True           | False       |
| 30    | gemma-2-9b                     | gemma-2-it gemma-2            | gemma-2 gemma-2       | True         | False          | True True      | False       |
|       | gemma-2-9b-it                  | gemma-2-it                    | gemma-2               | True         | True           | True           | False       |
| 31 32 | gemma-2b                       | gemma                         | gemma                 | True         | False          | True           | True        |
| 33    | gemma-2b-it                    | gemma-it                      | gemma                 | True         | True           | True           | True        |
| 34    | gemma-7b                       | gemma                         | gemma                 | True         | True           | True           | True        |
| 35    | gemma-7b-it                    | gemma-it                      | gemma                 | True         | True           | True           | True        |
| 36    | gpt-j-6b                       | gpt-j-neo-neox                | gpt-neo/j             | True         | True           | False          | True        |
| 37    | gpt-neo-1.3b                   | gpt-j-neo-neox                | gpt-neo/j             | True         | True           | True           | True        |
| 38    | gpt-neo-125m                   |                               |                       | True         |                |                |             |
|       |                                | gpt-j-neo-neox                | gpt-neo/j             | True         | True           | False          | True        |
| 39    | gpt-neo-2.7b gpt-neox-20b      | gpt-j-neo-neox gpt-j-neo-neox | gpt-neo/j gpt-neo/j   | True         | True True      | True False     | True True   |
| 40 41 | internlm2-20b                  | internlm2                     | internlm2             | True         | True           | False          | False       |
| 42    |                                | internlm2                     |                       |              |                |                |             |
|       | internlm2-7b                   |                               | internlm2             | True         | True           | False          | False       |
| 43 44 | llama-13b                      | llama                         | llama llama-2         | False False  | True           | True           | True        |
| 45    | llama-2-13b                    | llama-2                       |                       |              | True           | True           | True        |
|       | llama-2-13b-chat               | llama-2-chat                  | llama-2               | False        | True           | True           | True        |
| 46    | llama-2-70b                    | llama-2                       | llama-2               | False        | True           | True           | True        |
| 47    | llama-2-70b-chat               | llama-2-chat                  | llama-2               | False        | True           | True           | True        |
| 48    | llama-2-7b                     | llama-2                       | llama-2               | False        | True           | True True      | True        |
| 49    | llama-2-7b-chat llama-3-       | llama-2-chat llama-3-         | llama-2               | False        | True           |                | True        |
| 50    | sauerkrautlm-                  | sauerkrautlm-                 | meta-llama-3          | True         | False          | True           | False       |
| 51    | 70b-instruct llama-3-          | instruct llama-3-             | meta-llama-3          | True         | True           | True           | False       |

8b-instruct instruct

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89

90

91

92

93

94

95

96

97

98

99

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

126

127

128

llama-30b llama-65b

llama-7b meta-llama-3-70b

meta-llama-3-70b- instruct

meta-llama-3-8b meta-llama-3-8b-

instruct mpt-30b

mpt-30b-chat mpt-30b-instruct

mpt-7b mpt-7b-chat

mpt-7b-instruct olmo-1b

olmo-7b open\_llama\_13b

open\_llama\_3b open\_llama\_3b\_v2

open\_llama\_7b open\_llama\_7b\_v2

openhermes-13b openhermes-7b

opt-1.3b opt-125m

opt-13b opt-2.7b

opt-30b opt-350m

opt-6.7b opt-66b

orca-2-13b orca-2-7b

orca\_mini\_v3\_13b orca\_mini\_v3\_70b

orca\_mini\_v3\_7b orca\_mini\_v7\_72b

orca\_mini\_v7\_7b pythia-1.4b

pythia-12b pythia-160m

pythia-1b pythia-2.8b

pythia-410m pythia-6.9b

pythia-70m qwen-14b

qwen-72b qwen-7b

qwen1.5-0.5b qwen1.5-0.5b-chat

qwen1.5-1.8b qwen1.5-1.8b-chat

qwen1.5-14b qwen1.5-14b-chat

qwen1.5-32b qwen1.5-32b-chat

qwen1.5-4b qwen1.5-4b-chat

qwen1.5-72b qwen1.5-72b-chat

qwen1.5-7b qwen1.5-7b-chat

qwen2-0.5b qwen2-0.5b-instruct

qwen2-1.5b qwen2-1.5b-instruct

qwen2-72b qwen2-72b-instruct

qwen2-7b qwen2-7b-instruct

rwkv-4-14b-pile rwkv-4-169m-pile

rwkv-4-1b5-pile rwkv-4-3b-pile

rwkv-4-430m-pile rwkv-4-7b-pile

sauerkrautlm- gemma-2b

llama llama

llama meta-llama-3

meta-llama-3- instruct

meta-llama-3

meta-llama-3- instruct

mpt mpt-chat

mpt-instruct mpt

mpt-chat mpt-instruct

olmo olmo

open\_llama\_

open\_llama\_

open\_llama\_\_v2

open\_llama\_

open\_llama\_\_v2

openhermes openhermes

opt opt

opt opt

opt opt

opt opt

orca-2

orca-2

orca\_mini\_v3\_

orca\_mini\_v3\_

orca\_mini\_v3\_

orca\_mini\_v7\_

orca\_mini\_v7\_

pythia pythia

pythia pythia

pythia pythia

pythia pythia

qwen qwen

qwen qwen1.5

qwen1.5-chat qwen1.5

qwen1.5-chat qwen1.5

qwen1.5-chat qwen1.5

qwen1.5-chat qwen1.5

qwen1.5-chat qwen1.5

qwen1.5-chat qwen1.5

qwen1.5-chat qwen2

qwen2-instruct qwen2

qwen2-instruct qwen2

qwen2-instruct qwen2

qwen2-instruct rwkv-4-pile

rwkv-4-pile rwkv-4-pile

rwkv-4-pile rwkv-4-pile

rwkv-4-pile sauerkrautlm-

gemma llama

llama llama

meta-llama-3

meta-llama-3

meta-llama-3

meta-llama-3

mpt mpt

mpt mpt

mpt mpt

olmo olmo

openllama openllama

openllamav2

openllama openllamav2

llama-2

llama-2

opt opt

opt opt

opt opt

opt opt

llama-2

llama-2

llama-2

llama-2

llama-2

qwen2

qwen2

pythia pythia

pythia pythia

pythia pythia

pythia pythia

qwen qwen

qwen qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen1.5

qwen2

qwen2

qwen2

qwen2

qwen2

qwen2

qwen2

qwen2

rwkv rwkv

rwkv rwkv

rwkv rwkv

gemma

False

False

False

True

True

True

True

True

True

True

True

True

True

True

True

False

False

False

False

False

False

False

True

True

True

True

True

True

True

True

False

False

False

False

False

False

False

True

True

True

True

True

True

True

True

False

False

False

False

False

False

False

False

False

False

False

False

False

False

False

False

False

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

False

True

False

False

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

False

True

False

True

False

True

False

True

True

True

True

True

True

True

False

True

True

True

True

True

True

False

False

False

False

False

False

True

True

False

False

False

False

False

True

True

True

False

False

False

True

False

False

False

True

True

True

True

True

True

True

False

True

True

False

True

True

True

False

False

False

False

True

True

True

True

True

True

True

True

True

True

False

False

True

True

True

True

True

True

True

True

True

True

False

False

False

False

False

False

True

True

True

True

True

True

True

True

True

False

False

True

False

False

False

False

False

False

False

False

False

False

False

True

True

True

True

True

True

True

True

False

False

False

False

False

False

False

True

True

True

True

True

True

True

True

True

True

True

True

False

True

False

True

False

True

False

True

False

True

True

True

False

False

False

False

False

False

False

False

False

False

False

False

False

False

False

False

129

130

131

132

133

134

135

136

137

138

139

140

141

142

143

144

145

146

147

148

149

150

151

152

153

154

155

156

157

158

159

160

161

162

163

sauerkrautlm- gemma-7b

smollm-1.7b smollm-1.7b-

instruct smollm-135m

smollm-135m- instruct

smollm-360m smollm-360m-

instruct stablelm-base-

alpha-3b stablelm-base-

alpha-7b starcoder2-15b

starcoder2-3b starcoder2-7b

starcoderbase starcoderbase-1b

starcoderbase-3b starcoderbase-7b

wizardlm-13b-v1.0

wizardlm-70b-v1.0

xglm-1.7b xglm-4.5b

xglm-564m xglm-7.5b

yi-1.5-34b yi-1.5-34b-chat

yi-1.5-6b yi-1.5-6b-chat

yi-1.5-9b yi-1.5-9b-chat

yi-34b yi-34b-200k

yi-34b-chat yi-6b

yi-6b-200k yi-6b-chat

yi-9b sauerkrautlm-

gemma smollm

smollm- instruct

smollm smollm-

instruct smollm

smollm- instruct

stablelm-base- alpha

stablelm-base- alpha

starcoder2

starcoder2

starcoder2

starcoderbase starcoderbase

starcoderbase starcoderbase

wizardlm-v1.0

wizardlm-v1.0

xglm xglm

xglm xglm

yi-1.5

yi-1.5-chat yi-1.5

yi-1.5-chat yi-1.5

yi-1.5-chat yi

yi-200k yi-chat

yi yi-200k

yi-chat yi

gemma smollm

smollm smollm

smollm smollm

smollm stablelm

stablelm starcoder2

starcoder2

starcoder2

starcoder starcoder

starcoder starcoder

llama-2

llama-2

xglm xglm

xglm xglm

yi-1.5

yi-1.5

yi-1.5

yi-1.5

yi-1.5

yi-1.5

yi yi-200k

yi yi

yi-200k yi

yi

True

True

True

True

True

True

True

True

True

True

True

True

False

False

False

False

False

False

True

True

True

True

True

True

True

True

True

True

False

False

False

False

False

False

False

True

False

False

False

False

False

False

True

True

True

True

True

True

True

True

True

False

False

True

True

True

True

True

True

True

True

True

True

True

True

True

True

True

False

True

True

True

True

True

True

True

True

False

False

True

True

True

False

False

False

False

True

True

False

False

False

False

True

True

True

True

True

True

True

False

False

True

False

True

True

False

False

False

False

False

False

False

False

False

True

True

True

True

True

True

True

False

False

True

True

True

True

False

False

False

False

False

False

True

False

False

True

False

False

False

## H Extra performance prediction results

In this section, we present the full versions of the figures presented in the main text and some other extra results.

## H.1 Mean Absolute Percentage Error (MAPE) plot

Figure 10: MAPE version of Figure 1. The results are given in percentage points, and we can see that the results are qualitatively similar to the MAE version, with Sloth producing the best predictions for both Open LLM Leaderboards.

<!-- image -->

## H.2 Test families have exactly one model in the training set

## H.2.1 Average prediction loss across models

Figure 11: The figure shows the average (across LLM families) mean-absolute-error (MAE) (within a family) for different methods. This is a complete version of Figure 1, in which we include Sloth versions with shared intercept.

<!-- image -->

Open LLM Leaderboard v1/v2

Figure 12: The figure shows the average (across LLM families) mean-absolute-error (MAE) (within a family) for different methods when fitting only one scaling law for both leaderboards.

<!-- image -->

## H.2.2 Family-specific prediction losses

Open LLM Leaderboard v1 (Average error across benchmarks)

Figure 13: The figure shows the average (across benchmarks) mean-absolute-error (MAE) for each family considering only Open LLM Leaderboard v1.

<!-- image -->

Open LLM Leaderboard v2 (Average error across benchmarks)

Figure 14: The figure shows the average (across benchmarks) mean-absolute-error (MAE) for each family considering only Open LLM Leaderboard v2.

<!-- image -->

Figure 15: The figure shows the average (across benchmarks) mean-absolute-error (MAE) for each family considering Open LLM Leaderboard v1/v2.

<!-- image -->

## H.3 Test families have exactly two models in the training set

## H.3.1 Average prediction loss across models

Figure 16: The figure shows the average (across LLM families) mean-absolute-error (MAE) (within a family) for different methods.

<!-- image -->

Open LLM Leaderboard v1/v2

Figure 17: The figure shows the average (across LLM families) mean-absolute-error (MAE) (within a family) for different methods using the intersection of both leaderboards.

<!-- image -->

## H.3.2 Family-specific prediction losses

Figure 18: The figure shows the average (across benchmarks) mean-absolute-error (MAE) for each family considering only Open LLM Leaderboard v1.

<!-- image -->

Figure 19: The figure shows the average (across benchmarks) mean-absolute-error (MAE) for each family considering only Open LLM Leaderboard v2.

<!-- image -->

Figure 20: The figure shows the average (across benchmarks) mean-absolute-error (MAE) for each family considering only Open LLM Leaderboard v1/v2.

<!-- image -->

## I Extra interpretability results

## I.1 Results for d = 2

<!-- image -->

Figure 21: Needed skills for each benchmark. In this figure, we report the estimated loadings Λ and, based on their values, we give them appropriate names.

Figure 22: Needed skills for each benchmark. In this figure, we report the estimated loadings Λ and, based on their values, we give them appropriate names.

<!-- image -->

Figure 23: Gains from instruction tuning for different families on three latent skills. Major findings include a large and positive impact on instruction following and a negative impact on mathematical reasoning.

<!-- image -->

Figure 24: Level curves in producing different latent abilities from parameter count and training tokens.

<!-- image -->

## I.2 Results for d = 3

Figure 25: Needed skills for each benchmark. In this figure, we report the estimated loadings Λ and, based on their values, we give them appropriate names.

<!-- image -->

## I.3 Results for d = 4

Figure 26: Needed skills for each benchmark. In this figure, we report the estimated loadings Λ and, based on their values, we give them appropriate names.

<!-- image -->

<!-- image -->

Figure 27: Needed skills for each benchmark. In this figure, we report the estimated loadings Λ and, based on their values, we give them appropriate names.

Figure 28: Gains from instruction tuning for different families on three latent skills. Major findings include a large and positive impact on instruction following and a negative impact on mathematical reasoning.

<!-- image -->

Figure 29: Level curves in producing different latent abilities from parameter count and training tokens.

<!-- image -->

Figure 30: Predicting model performance in complex downstream tasks like emotional intelligence ('EQ') for LLaMa 3 70B (base/instruct). In the first step, we fit Sloth without including LLaMa 3 70B (base/instruct) in the training set. In the second step, we fit a regression model connecting skills and downstream performance. Finally, we predict LLaMa 3 70B (instruct) performance from their predicted Sloth skills.

<!-- image -->

## J Extra downstream task plots

- J.1 Emotional Intelligence
- J.2 Agentic Capabilities

Figure 31: Predicting Agentic Capabilities of Llama-2-70B-chat.

<!-- image -->

Figure 32: Loadings for downstream prediction tasks.

<!-- image -->

## K Insights from the different link functions

In this section, we visually compare Sloth considering trainable and logistic link function σ , Owen [2024]'s model ('FLOPs (shared intercept)') and our adaptation of Ruan et al. [2024]'s observational scaling law ('PCA + FLOPs') described in Appendix E. For this experiment, we study the two Open LLM Leaderboards separately and consider LLaMa-3 and Yi-1.5 families as the test families; we make this choice because both families are popular ones and the training set size is the same for all models in each family, making comparison between models possible (in the x-axis, we use model size). For LLaMA-3, we just include one model from that family in the training set and do not train a family-specific slope for PCA+FLOPs. For Yi-1.5, we include two models in the training

set and train a family-specific slope for PCA+FLOPs. In summary, we see that: (i) training the link function can produce a much more flexible scaling law that can better predict performance saturation ( e.g. , the performance of Yi-1.5 in ARC, HellaSwag etc. ), (ii) training no family-specific parameters at all ('FLOPs (shared intercept)') usually produce poor prediction results, and (iii) PCA+FLOPs often produces flatter curves that underestimate the performance of bigger models, e.g. , see Yi-1.5 in TruthfulQA, GSM8k, and MMLU.

Figure 33: Prediction curves for different methods considering Open LLM Leaderboard 1 benchmark and the LLaMa-3 as the test family.

<!-- image -->

Figure 34: Prediction curves for different methods considering Open LLM Leaderboard 2 benchmark and the LLaMa-3 as the test family.

<!-- image -->

Figure 35: Prediction curves for different methods considering Open LLM Leaderboard 1 benchmark and the Yi-1.5 as the test family.

<!-- image -->

Figure 36: Prediction curves for different methods considering Open LLM Leaderboard 2 benchmark and the Yi-1.5 as the test family.

<!-- image -->

## L Comparing against Ruan et al. [2024] in their observational scaling law setting

In this section, we compare Sloth with Ruan et al. [2024]'s observational scaling law; that is, we extract abstract skills using a set of benchmark scores and then use those skills to predict the performance of models of interest in a target downstream task. For this experiment, we use the same data and tasks explored in Section 4.4. For our method, we fit Sloth using benchmark data from all models, including performance data of LLaMa-3-70B models, and extract the skills of each model. For Ruan et al. [2024]'s method, we fit PCA on the benchmark data to extract the skills. For both methods, we set d = 3 and then fit a regression model with a logistic link to predict downstream performance from skills. Figures 37 and 38 present the prediction results for both methods and Figures 39 and 40 give the loading of both approaches. In both plots, out-sample prediction has a similar prediction error. At the same time, the in-sample fit is better for Sloth in the coding task and for Ruan et al. [2024]'s observational scaling law in the emotional intelligence task. Regarding the loading, it is possible to draw some similarities, e.g. , the presence of instruction following skill, but there is no one-to-one correspondence between skills.

<!-- image -->

Figure 37: Predicting code completion of LLaMa 3 70B (base/instruct) with Sloth vs Obs. Scaling Law [Ruan et al., 2024].

Figure 38: Predicting emotional intelligence of LLaMa 3 70B (base/instruct) with Sloth vs Obs. Scaling Law [Ruan et al., 2024].

<!-- image -->

Figure 39: Loadings for downstream prediction tasks ( Sloth ).

<!-- image -->

Figure 40: Loadings for downstream prediction tasks (Obs. Scaling Law [Ruan et al., 2024]).

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: abstract and introduction accurately reflect the paper's contributions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: we include a section for limitations in the appendix.

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

Justification: we provide assumptions and proof.

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

Justification: we explain the experimental setup in detail and share our code and data.

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

Justification: we make everything public.

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

Justification: we provide all details.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: most of our experiments do not need error bars to justify our conclusions. Moreover, for performance analysis, the data is not iid, so the usual error bar analysis is not valid here. An alternative would be having error bars across different families; in this direction, we provide results for individual LLM families in the appendix; they can be seen as a (more detailed) substitute for error bars for this application.

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

Justification: we mention in the text that our model can be easily trained with a commercial laptop.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: we made sure this holds

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: there are no clear direct societal impacts.

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

Justification: no clear risk of misuse of data.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: we made sure to credit third parties when needed.

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

Justification: they are well documented

Guidelines:

- The answer NA means that the paper does not release new assets.

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: we used no crowdsourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: we had no study participants.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: we used LLMs mainly for rephrasing some paragraphs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.