## Exploring the Noise Robustness of Online Conformal Prediction

Huajun Xi 1 , Kangdao Liu 1,2 , Hao Zeng 1 , Wenguang Sun 3 , Hongxin Wei 1 ∗

1 Department of Statistics and Data Science, Southern University of Science and Technology 2 Department of Computer and Information Science, University of Macau 3 Center for Data Science, Zhejiang University

## Abstract

Conformal prediction is an emerging technique for uncertainty quantification that constructs prediction sets guaranteed to contain the true label with a predefined probability. Recent work develops online conformal prediction methods that adaptively construct prediction sets to accommodate distribution shifts. However, existing algorithms typically assume perfect label accuracy which rarely holds in practice. In this work, we investigate the robustness of online conformal prediction under uniform label noise with a known noise rate. We show that label noise causes a persistent gap between the actual mis-coverage rate and the desired rate α , leading to either overestimated or underestimated coverage guarantees. To address this issue, we propose a novel loss function robust pinball loss , which provides an unbiased estimate of clean pinball loss without requiring ground-truth labels. Theoretically, we demonstrate that robust pinball loss enables online conformal prediction to eliminate the coverage gap under uniform label noise, achieving a convergence rate of O ( T -1 / 2 ) for both empirical and expected coverage errors (i.e., absolute deviation of the empirical and expected mis-coverage rate from the target level α ). This loss offers a general solution to the uniform label noise, and is complementary to existing online conformal prediction methods. Extensive experiments demonstrate that robust pinball loss enhances the noise robustness of various online conformal prediction methods by achieving a precise coverage guarantee and improved efficiency.

## 1 Introduction

Machine learning techniques are revolutionizing decision-making in high-stakes domains, such as autonomous driving [1] and medical diagnostics [2]. It is crucial to ensure the reliability of model predictions in these contexts, as wrong predictions can result in serious consequences. While various techniques have been developed for uncertainty estimation, including confidence calibration [3] and Bayesian neural networks [4], they typically lack rigorous theoretical guarantees. Conformal prediction addresses this limitation by establishing a systematic framework to construct prediction sets with provable coverage guarantee [5, 6, 7, 8]. Notably, this framework requires no parametric assumptions about the data distribution and can be applied to any black-box predictor, which makes it a powerful technique for uncertainty quantification.

Recent research extends conformal prediction to online setting where the data arrives in a sequential order [9, 10, 11, 12, 13, 14]. These methods provably achieve the desired coverage property under arbitrary distributional changes. However, previous studies typically assume perfect label accuracy , an assumption that seldom holds true in practice due to the common occurrence of noisy labels in online learning [15, 16, 17]. Recent work [18] proves that online conformal prediction can achieve a

∗ Correspondence to: Hongxin Wei &lt; weihx@sustech.edu.cn &gt;

conservative coverage guarantee under uniform label noise, leading to unnecessarily large prediction sets. However, their analysis relies on a strong distributional assumption of non-conformity score and cannot quantify the specific deviation of coverage guarantees. These limitations motivate us to establish a general theoretical framework for this problem and develop a noise-robust algorithm that maintains precise coverage guarantees while producing small prediction sets.

In this work, we present a general theoretical framework for analyzing how uniform label noise affects the performance of standard online conformal prediction, i.e., adaptive conformal inference (dubbed ACI) [9]. Notably, our theoretical results are independent of the distributional assumptions on the non-conformity scores made in the previous work [18]. In particular, we demonstrate that label noise causes a persistent gap between the actual mis-coverage rate and the desired rate α , with higher noise rates resulting in larger gaps. This gap can lead to either overestimated or underestimated coverage guarantees, which depend on the size of the prediction sets (see Proposition 3.1).

To address this challenge, we propose a novel loss function robust pinball loss , which provides an unbiased estimate of clean pinball loss value without requiring access to ground-truth labels. Specifically, we construct the robust pinball loss as a weighted combination of the pinball loss computed with respect to noisy scores and the pinball loss with scores of all classes. We prove that this loss is equivalent to the pinball loss under clean labels in expectation. Theoretically, we demonstrate that robust pinball loss enables ACI to eliminate the coverage gap under uniform label noise. It achieves a convergence rate of O ( T -1 / 2 ) for both empirical and expected coverage errors (i.e., absolute deviation of the empirical and expected mis-coverage rate from the target level α ). Notably, our robust pinball loss offers a general solution to the uniform label noise, and is complementary to existing online conformal prediction methods.

To verify the effectiveness of the robust pinball loss, we conduct extensive experiments on CIFAR100 [19] and ImageNet [20] with synthetic uniform label noise. In particular, we integrate the proposed loss into ACI with constant [9] and dynamic learning rates [12], and strongly adaptive online conformal prediction [11]. Empirical results show that the robust pinball loss enhances the noise robustness of online conformal prediction by eliminating the coverage gap caused by the label noise. Thus, these methods achieve both long-run coverage and local coverage rate that are close to the target 1 -α , and improved prediction set efficiency. For example, on ImageNet with error rate α = 0 . 1 , noise rate ϵ = 0 . 15 and dynamic learning rate, ACI with the standard pinball loss deviates from the target coverage level of 0 . 9 , exhibiting a coverage gap of 8.372% and an average set size of 171.2. In contrast, ACI equipped with the robust pinball loss achieves a negligible coverage gap of 0.183% and a prediction set size of 13.10. In summary, our method consistently enhances the noise robustness of various online conformal prediction methods by achieving a precise coverage guarantee and improved efficiency.

We summarize our contributions as follows:

- We present a general theoretical framework for analyzing the effect of uniform label noise on the coverage of online conformal prediction. Our theoretical results are independent of the distributional assumptions made in the previous work [18].
- To address the issue of label noise, we propose a novel loss function robust pinball loss that enhances the noise robustness of online conformal prediction. This loss is complementary to online conformal prediction algorithms and can be seamlessly integrated with these methods.
- We empirically validate that our method can be applied to various online conformal prediction methods and non-conformity score functions. It is straightforward to implement and does not require sophisticated changes to the framework of online conformal prediction.

## 2 Preliminary

Online conformal prediction. We study the problem of generating prediction sets in online classification where the data arrives in a sequential order [9, 12]. Formally, we consider a sequence of data points ( X t , Y t ) , t ∈ N + , which are sampled from a joint distribution P XY over the input space X ⊂ R d , and the label space Y = { 1 , . . . , K } . In online conformal prediction, the goal is to construct prediction sets C t ( X t ) , t ∈ N + , that provides precise coverage guarantee: lim T → + ∞ 1 T ∑ T t =1 1 { Y t / ∈ C t ( X t ) } = α , where α ∈ (0 , 1) denotes a user-specified error rate.

At each time step t , we construct a prediction set C t ( X t ) by

<!-- formula-not-decoded -->

where ˆ τ t is a data-driven threshold, and S : X × Y → R denotes a non-conformity score function that measures the deviation between a data sample and the training data. For example, given a pre-trained classifier f : X → R K , the LAC score [21] is defined as S ( X,Y ) = 1 -ˆ π Y ( X ) , where ˆ π Y ( X ) = σ Y ( f ( X )) denotes the softmax probability of instance X for class Y , and σ is the softmax function. For notation shorthand, we use S t to denote the random variable S ( X t , Y t ) and use S t,y to denote S ( X t , y ) for a given class y ∈ Y . Following previous work [12, 22], we will assume that the non-conformity score function is bounded, and the threshold is specifically initialized:

Assumption 2.1. The score is bounded by S ( · , · ) ∈ [0 , 1] .

Assumption 2.2. The threshold is initialized by ˆ τ 1 ∈ [0 , 1] .

In online conformal prediction, a representative method is adaptive conformal inference (ACI) [9], which updates the threshold ˆ τ t with pinball loss :

<!-- formula-not-decoded -->

where τ denotes a threshold and s is a non-conformity score. As the label Y t of X t is observed after model prediction, the threshold is then updated via online gradient descent :

<!-- formula-not-decoded -->

where ∇ τ l 1 -α ( τ, s ) denotes the gradient of pinball loss w.r.t the threshold τ , and η &gt; 0 is the learning rate. The optimization will increase the threshold if the prediction set C t ( X t ) fails to encompass the label Y t , resulting in more conservative predictions in future instances (and vice versa).

We use the empirical coverage error and the expected coverage error to evaluate the coverage performance. The empirical coverage error measures the absolute deviation of the mis-coverage rate from the target level α , while the expected coverage error quantifies the absolute deviation in expectation. In particular, for any T ∈ N ∗ , we define

<!-- formula-not-decoded -->

Uniform Label noise. In this paper, we focus on the issue of noisy labels in online learning, a common occurrence in the real world. This is primarily due to the dynamic nature of real-time data streams and the potential for human error or sensor malfunctions during label collection. Let ( X t , ˜ Y t ) be the data sequence with label noise, and ˜ S t = S ( X t , ˜ Y t ) be the noisy non-conformity score. In this work, we focus on the setting of uniform label noise [18, 23], i.e., the correct label is replaced by a label that is randomly sampled from the K classes with a fixed probability ϵ ∈ (0 , 1) :

<!-- formula-not-decoded -->

where U is uniformly distributed over [0 , 1] , and ¯ Y is uniformly sampled from the set of classes Y . We assume the probability ϵ (i.e., the noise rate) is known, in alignment with prior works [18, 23, 24]. This assumption is practical as the noise rate can be estimated from historical data [25, 26, 27].

Recent work [18] investigates the noise robustness of online conformal prediction under uniform label noise, with a strong distributional assumption. Their analysis demonstrates that noisy labels will lead to a conservative long-run coverage guarantee, with the assumption that noisy score distribution stochastically dominates the clean score distribution, i.e., P { ˜ S ≤ s } ≤ P { S ≤ s } , ∀ s ∈ R . The distributional assumption is too strong to ensure valid coverage under general cases. Moreover, their analysis fails to quantify the specific deviation of coverage guarantees. These limitations motivate us to establish a general theoretical framework for this problem and develop a noise-robust algorithm for online conformal prediction.

## 3 The impact of label noise on online conformal prediction

In this section, we theoretically analyze the impacts of uniform label noise on ACI [9]. In this case, the threshold ˆ τ t is updated by

<!-- formula-not-decoded -->

where ˜ S t is the noisy non-conformity score. As shown in Eq. (3), the essence of online conformal prediction is to update the threshold ˆ τ t with the gradient of pinball loss. However, the gradient estimates can be biased if the observed labels are corrupted. Formally, with high probability:

̸

<!-- formula-not-decoded -->

This bias in gradient estimation can result in two potential consequences: conformal predictors may either fail to maintain the desired coverage or suffer from reduced efficiency (i.e., generating large prediction sets). We formalize these consequences in the following proposition:

Proposition 3.1. Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (3) , then for any δ ∈ (0 , 1) and T ∈ N + , the following bound holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Interpretation. The proof is provided in Appendix E.1. In Proposition 3.1, we evaluate the long-run mis-coverage rate using 1 T ∑ T t =1 1 { Y t / ∈ C t ( X t ) } . Our analysis shows that label noise introduces a coverage gap of

<!-- formula-not-decoded -->

between the actual mis-coverage rate and desired mis-coverage rate α (the term A √ T + B T will diminish and eventually approach zero, as T increases). In particular, the effect of label noise on the coverage guarantee can manifest in two distinct scenarios, depending on the prediction set size:

- When the prediction sets are small such that E [ |C t ( X t ) | ] ≤ K (1 -α ) , label noise can result in over-coverage of prediction sets: 1 T ∑ T t =1 1 { Y t / ∈ C t ( X t ) } ≤ α . In this scenario, a higher noise rate ϵ results in a larger coverage gap.
- When the prediction sets are large such that E [ |C t ( X t ) | ] ≥ K (1 -α ) , label noise causes under-coverage of prediction sets. This situation generally occurs only when the model significantly underperforms on the task, which is uncommon in practical applications (See details in Appendix B.3)).

Empirical verification. To verify our theoretical result, we compare the performance of ACI under different noise rates. In particular, we compute the long-run coverage, defined as Cov( T ) = 1 T ∑ T t =1 1 { Y t ∈ C t ( X t ) } , under noise rate ϵ ∈ { 0 . 05 , 0 . 1 , 0 . 15 } , with a ResNet18 model on CIFAR100 and ImageNet datasets. We employ LAC score [21] to generate prediction sets with error rate α = 0 . 1 , and use noisy labels to update the threshold with a constant learning rate η = 0 . 05 . The experimental results in Figure 1 validate our theoretical analysis: label noise introduces discrepancies between the actual and target coverage rates 1 -α , with higher noise rates resulting in a more pronounced coverage gap.

Overall, our results show that label noise significantly impacts the coverage guarantee of online conformal prediction. The size of the prediction set determines whether coverage is inflated or deflated, and the noise rate controls how much coverage is changed. In the following, we introduce a robust pinball loss, which addresses the issue of label noise.

## 4 Method

## 4.1 Robust pinball loss

Our theoretical analysis establishes that biased gradient estimates arising from label noise can significantly impact the coverage properties of online conformal prediction. Therefore, the key

Figure 1: Performance of ACI [9] with a constant learning rate η = 0 . 05 under different noise rates, using ResNet18 on CIFAR-100 and ImageNet datasets. The results validate that label noise introduces a coverage gap, with higher noise rates resulting in a more pronounced gap.

<!-- image -->

challenge of the noisy setting lies in how to obtain unbiased gradient estimates without requiring ground-truth labels. In this work, we propose a novel loss function robust pinball loss , which provides an unbiased estimate of the clean pinball loss value without requiring access to ground-truth labels. We begin by developing the intuition behind how to approximate the clean pinball loss under uniform label noise.

Consider a data sample ( X,Y ) with a noisy label ˜ Y , we denote the clean non-conformity score as S = S ( X,Y ) , the noisy score as ˜ S = S ( X, ˜ Y ) , and the score for an arbitrary class y ∈ Y as S y = S ( X,y ) . Under a uniform label noise with noise rate ϵ ∈ (0 , 1) , the distributions of these scores have the following relationship: P { S ≤ s } = 1 1 -ϵ P { ˜ S ≤ s } -ϵ K (1 -ϵ ) ∑ K y =1 P { S y ≤ s } , for an arbitrary number s ∈ R . We formally establish this equation in Lemma F.1 with a rigorous proof. This correlation motivates the following approximation:

<!-- formula-not-decoded -->

The above decomposition suggests that the clean pinball loss can be approximated by replacing its indicator function with the above expression. Inspired by this, we propose the robust pinball loss as:

<!-- formula-not-decoded -->

The following theoretical properties demonstrate how this loss function mitigates label noise bias:

Proposition 4.1. The robust pinball loss defined in Eq. (4) satisfies the following two properties:

<!-- formula-not-decoded -->

Interpretation. The proof can be found in Appendix E.2. In Proposition 4.1, the first property ensures that our robust pinball loss matches the expected value of the true pinball loss, while the second guarantees that the gradients of both losses have the same expectation. These properties establish that updating the threshold with robust pinball loss is equivalent to updating with clean pinball loss in expectation.

In summary, Proposition 4.1 establishes the validity of our robust pinball loss in expectation . Notably, our robust pinball loss offers a general solution to the uniform label noise, and is complementary to existing online conformal prediction methods. In the following sections, we apply the robust pinball loss to ACI with constant [9] and dynamic learning rates [12]. We will show that by updating the threshold with the proposed loss, ACI eliminates the coverage gap caused by the label noise.

## 4.2 Convergence with constant learning rate

We now analyze the convergence of the coverage rate of ACI under uniform label noise with a constant learning rate schedule [9]. In particular, we update the threshold of ACI with respect to the robust pinball loss:

<!-- formula-not-decoded -->

For notation shorthand, we denote:

<!-- formula-not-decoded -->

We first present the results for expected coverage error:

Proposition 4.2. Under the same assumptions in Proposition 3.1, when updating the threshold according to Eq. (5) , then for any δ ∈ (0 , 1) and T ∈ N + , the following bound holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

Proof Sketch. The proof (in Appendix E.3) relies on the following decomposition:

<!-- formula-not-decoded -->

Part (a) converges to zero in probability at rate O ( T -1 / 2 ) by the Azuma-Hoeffding inequality, and part (b) achieves a convergence rate of O ( T -1 ) following standard online conformal prediction theory. Combining the two parts establishes the desired upper bound.

Building on Proposition 4.2, we now provide an upper bound for the empirical coverage error:

Proposition 4.3. Under the same assumptions in Proposition 3.1, when updating the threshold according to Eq. (5) , then for any δ ∈ (0 , 1) and T ∈ N + , the following bound holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

Proof Sketch. The proof (detailed in Appendix E.4) employs a similar decomposition:

<!-- formula-not-decoded -->

The analysis shows that both terms achieve O ( T -1 / 2 ) convergence: part (a) through the Azuma-Hoeffding inequality, and part (b) via Proposition 4.2.

Remark 4.4. It is worth noting that our method achieves a O ( T -1 / 2 ) convergence rate for empirical coverage error even in the absence of noise (i.e., ϵ = 0 ), which is slightly slower than the O ( T -1 ) rate achieved by standard online conformal prediction theory [9, 12]. This is because our analysis relies on martingale-based concentration to handle label noise, leading to the O ( T -1 / 2 ) rate.

## 4.3 Convergence with dynamic learning rate

Recent work [12] highlights a limitation of constant learning rates: while coverage holds on average over time, the instantaneous coverage rate Cov(ˆ τ t ) = P { S ≤ ˆ τ t } would exhibit substantial temporal

variability (see Proposition 1 in [12]). Thus, they extend ACI to dynamic learning rate schedule where η t can change over time for updating the threshold. In this section, we apply the robust pinball loss to the dynamic learning rate schedule. Specifically, we update the threshold by:

<!-- formula-not-decoded -->

For the convergence of coverage rate, our theoretical results show that, under dynamic learning rate, NR-OCP achieves convergence rates of EmCovErr(T) = O ( T -1 / 2 ) and ExCovErr(T) = O ( T -1 / 2 ) . The proofs are presented in Appendix E.5 and E.6.

Proposition 4.5. Under the same assumptions in Proposition 3.1, when updating the threshold according to Eq. (6) , then for any δ ∈ (0 , 1) and T ∈ N + , the following bound holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

Proposition 4.6. Under the same assumptions in Proposition 3.1, when updating the threshold according to Eq. (6) , then for any δ ∈ (0 , 1) and T ∈ N + , the following bound holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

In Proposition 4.5 and 4.6, we establish that for any sequence of learning rates that satisfies the condition ∑ T t =1 ∣ ∣ η -1 t -η -1 t -1 ∣ ∣ /T → 0 as T → + ∞ , both expected and empirical coverage errors asymptotically vanish with a convergence rate of O ( T -1 / 2 ) . Therefore, by applying our method, the long-term coverage rate would approach the desired target level of 1 -α .

Online learning theory (see Theorem 2.13 of [28]) established that ACI achieves a regret bound:

<!-- formula-not-decoded -->

with optimally chosen η t , which serves as a helpful measure alongside coverage [11]. We provide a regret analysis for our method in Appendix B.1. In addition, we formally establish the convergence of ˆ τ t towards the global minima in Appendix B.2 by extending the standard convergence analysis of stochastic gradient descent (see Theorem 5.3 in [29]).

## 5 Experiments

## 5.1 Experimental setups

Datasets and setup. We evaluate the performance of NR-OCP under uniform label noise. The experiments include both constant η = 0 . 05 and dynamic learning rates η t = 1 /t 1 / 2+ ε with ε = 0 . 1 , following prior work [12]). We use CIFAR-100 [19] and ImageNet [20] datasets with synthetic label noise. On ImageNet, we use four pre-trained classifiers from TorchVision [30] - ResNet18, ResNet50 [31], DenseNet121 [32] and VGG16 [33]. On CIFAR-100, we train these models for 200 epochs using SGD with a momentum of 0.9, a weight decay of 0.0005, and a batch size of 128. We set the initial learning rate as 0.1, and reduce it by a factor of 5 at 60, 120 and 160 epochs.

Conformal prediction algorithms. We integrate the robust pinball loss into ACI with constant [9] and dynamic learning rates [12], and strongly adaptive online conformal prediction [11]. We apply LAC [21], APS [34], RAPS [35] and SAPS [36] to generate prediction sets. A detailed description of these non-conformity scores is provided in Appendix B.4. In addition, we use error rates α ∈ { 0 . 1 , 0 . 05 } in experiments.

Figure 2: Long-run coverage and local coverage performance of various methods under uniform noisy labels with noise rate ϵ = 0 . 05 . We apply robust pinball loss to ACI with (a) constant [9] and (b) dynamic learning rate [12], and (c) SAOCP [11]. We employ LAC to generate prediction sets with α = 0 . 1 , using ResNet18 on CIFAR100. 'Baseline' and 'Clean' denote the online conformal prediction with standard pinball loss, using noisy and clean labels.

<!-- image -->

Metrics. We employ four evaluation metrics: long-run coverage (Cov) , local coverage (LocalCov) , coverage gap (CovGap) and prediction set size (Size) . We use long-run coverage and local coverage to present the dynamics of coverage during the optimization. Formally, for any T ∈ N + ,

<!-- formula-not-decoded -->

In particular, long-run coverage measures the coverage rate over the first T time steps, while local coverage is over an interval with length L . In the experiments, we employ a length of L = 200 . Besides, the coverage gap and the prediction set size are computed over the full test set, indicating the final performance of the online conformal prediction method. Formally, given a test dataset I test ,

<!-- formula-not-decoded -->

Small prediction sets are preferred as they can provide precise predictions, thereby enabling accurate human decision-making in real-world scenarios [37].

## 5.2 Main results

Robust pinball loss enhances the noise robustness of existing online conformal prediction methods. In figure 2, we apply robust pinball loss to ACI with constant [9] and dynamic learning rate [12], and SAOCP [11]. We evaluate these methods with long-run coverage and local coverage. We use the LAC score and a ResNet18 model on CIFAR100, with error rate α = 0 . 1 and noise rate ϵ = 0 . 05 . The results demonstrate that the proposed robust pinball loss enables these methods to achieve a long-run coverage and local coverage close to the target coverage rate of 1 -α . In summary, the empirical results highlight the versatility of our robust pinball loss, demonstrating its potential to enhance the noise robustness across diverse algorithms.

Robust pinball loss is effective across various settings. In Table 1, we incorporate robust pinball loss into ACI with constant [9] and dynamic learning rate [12], under different noise rate ϵ , error rate α and non-conformity scores. We evaluate the prediction sets with coverage gap and prediction set size. Due to space constraints, we report the average performance across four non-conformity score functions. The performance on each score function is provided in Appendix B.5. The results

Table 1: Average performance of different methods under uniform noisy labels across 4 score functions, using ResNet18. The performance on each score function is provided in Appendix B.5. 'Baseline' denotes the ACI with standard pinball loss. We include two learning rate schedules: constant learning rate η = 0 . 05 and dynamic learning rates η t = 1 /t 1 / 2+ ε where ε = 0 . 1 . ' ↓ ' indicates smaller values are better and Bold numbers are superior results.

| LR Schedule   | Error rate   | Method   | CIFAR100    | CIFAR100    | CIFAR100   | CIFAR100   | ImageNet    | ImageNet    | ImageNet   | ImageNet   |
|---------------|--------------|----------|-------------|-------------|------------|------------|-------------|-------------|------------|------------|
| LR Schedule   | Error rate   | Method   | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     |
| LR Schedule   | Error rate   | Method   | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 |
| Constant      | ϵ = 0 . 05   | Baseline | 3.942       | 2.867       | 11.93      | 30.66      | 4.163       | 3.289       | 101.0      | 231.7      |
| Constant      | ϵ = 0 . 05   | Ours     | 0.386       | 0.183       | 7.786      | 16.54      | 0.084       | 0.272       | 67.37      | 143.3      |
| Constant      | ϵ = 0 . 1    | Baseline | 7.139       | 3.753       | 21.81      | 46.93      | 7.509       | 4.068       | 178.4      | 389.7      |
| Constant      | ϵ = 0 . 1    | Ours     | 0.270       | 0.428       | 8.815      | 17.96      | 0.095       | 0.194       | 81.87      | 157.3      |
| Constant      | ϵ = 0 . 15   | Baseline | 8.032       | 4.147       | 36.95      | 59.92      | 8.566       | 4.348       | 318.0      | 513.2      |
| Constant      | ϵ = 0 . 15   | Ours     | 0.520       | 0.395       | 9.396      | 19.41      | 0.198       | 0.144       | 90.52      | 163.4      |
| Dynamic       | ϵ = 0 . 05   | Baseline | 4.106       | 2.780       | 6.527      | 21.97      | 4.498       | 2.584       | 31.15      | 128.1      |
| Dynamic       | ϵ = 0 . 05   | Ours     | 0.170       | 0.658       | 2.958      | 10.95      | 0.120       | 0.242       | 7.502      | 47.41      |
| Dynamic       | ϵ = 0 . 1    | Baseline | 7.414       | 3.733       | 18.18      | 37.18      | 7.249       | 3.595       | 99.54      | 214.9      |
| Dynamic       | ϵ = 0 . 1    | Ours     | 0.414       | 0.217       | 3.361      | 12.01      | 0.079       | 0.321       | 10.34      | 67.03      |
| Dynamic       | ϵ = 0 . 15   | Baseline | 8.403       | 4.211       | 29.41      | 48.24      | 8.372       | 4.127       | 171.2      | 274.9      |
| Dynamic       | ϵ = 0 . 15   | Ours     | 0.214       | 0.195       | 4.065      | 12.98      | 0.183       | 0.256       | 13.10      | 78.64      |

show that robust pinball loss allows ACI to eliminate the coverage gap, achieving precise coverage guarantees while significantly improving the long-run efficiency of prediction sets. For example, on ImageNet with error rate α = 0 . 1 , noise rate ϵ = 0 . 15 and dynamic learning rate, ACI employing the robust pinball loss achieves a negligible coverage gap of 0.183% and a prediction set size of 13.10. We report additional results on various models in Appendix B.6. Overall, the robust pinball loss is effective across different settings, including various label noise, error rate, non-conformity score functions, and model architectures.

In real-world scenarios where the noise rate could be unknown, it can be estimated from historical data [25, 26, 27]. In Appendix C, we leverage an existing approach to estimate the noise rate. Then, we employ this estimated noise rate in our method. We show that in this circumstance, the robust pinball loss can improve the noise robustness of online conformal prediction algorithms.

## 6 Conclusion

In this work, we investigate the robustness of online conformal prediction under uniform label noise with a known noise rate, in both constant and dynamic learning rate schedules. Our theoretical analysis shows that the presence of label noise causes a deviation between the actual and desired mis-coverage rate α , with higher noise rates resulting in larger gaps. To address this issue, we propose a novel loss function robust pinball loss , which provides an unbiased estimate of clean pinball loss without requiring ground-truth labels. We theoretically establish that this loss is equivalent to the pinball loss under clean labels in expectation. In our theoretical analysis, we show that robust pinball loss enables online conformal prediction to eliminate the coverage gap caused by the label noise, achieving a convergence rate of O ( T -1 / 2 ) for both empirical and expected coverage errors under uniform label noise. This loss offers a general solution to the uniform label noise, and is complementary to existing online conformal prediction methods. Extensive experiments demonstrate that the proposed loss enhances the noise robustness of various online conformal prediction methods by eliminating the coverage gap caused by the label noise. Notably, our loss function is effective across different label noise, error rate, non-conformity score functions, and model architectures.

Limitation. As the first step to explore the label noise issue in online conformal prediction, our analysis and method are limited to the setting of uniform noisy labels with a known noise rate. We believe it will be interesting to develop online conformal prediction algorithms that are robust to various types of label noise with fewer assumptions in the future.

## Acknowledgements

This research is supported by the Shenzhen Fundamental Research Program (Grant No. JCYJ20230807091809020). We gratefully acknowledge the support of the Center for Computational Science and Engineering at the Southern University of Science and Technology for our research.

## References

- [1] Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al. End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316 , 2016.
- [2] Rich Caruana, Yin Lou, Johannes Gehrke, Paul Koch, Marc Sturm, and Noemie Elhadad. Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , 2015.
- [3] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning , 2017.
- [4] Ralph C Smith. Uncertainty quantification: theory, implementation, and applications . Siam, 2013.
- [5] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a random world . Springer, 2005.
- [6] Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. Journal of Machine Learning Research , 2008.
- [7] Vineeth Balasubramanian, Shen-Shyang Ho, and Vladimir Vovk. Conformal prediction for reliable machine learning: theory, adaptations and applications . Newnes, 2014.
- [8] Anastasios N Angelopoulos, Stephen Bates, et al. Conformal prediction: A gentle introduction. Foundations and Trends® in Machine Learning , 2023.
- [9] Isaac Gibbs and Emmanuel Candes. Adaptive conformal inference under distribution shift. Advances in Neural Information Processing Systems , 2021.
- [10] Shai Feldman, Liran Ringel, Stephen Bates, and Yaniv Romano. Achieving risk control in online learning settings. Transactions on Machine Learning Research , 2023.
- [11] Aadyot Bhatnagar, Huan Wang, Caiming Xiong, and Yu Bai. Improved online conformal prediction via strongly adaptive online learning. In International Conference on Machine Learning , 2023.
- [12] Anastasios Nikolas Angelopoulos, Rina Barber, and Stephen Bates. Online conformal prediction with decaying step sizes. In Forty-first International Conference on Machine Learning , 2024.
- [13] Isaac Gibbs and Emmanuel J Candès. Conformal inference for online prediction with arbitrary distribution shifts. Journal of Machine Learning Research , 2024.
- [14] Erfan Hajihashemi and Yanning Shen. Multi-model ensemble conformal prediction in dynamic environments. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [15] Shai Ben-David, Dávid Pál, and Shai Shalev-Shwartz. Agnostic online learning. In The 22nd Conference on Learning Theory , 2009.
- [16] Nagarajan Natarajan, Inderjit S. Dhillon, Pradeep Ravikumar, and Ambuj Tewari. Learning with noisy labels. In Christopher J. C. Burges, Léon Bottou, Zoubin Ghahramani, and Kilian Q. Weinberger, editors, Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems , 2013.

- [17] Changlong Wu, Ananth Grama, and Wojciech Szpankowski. Information-theoretic limits of online classification with noisy labels. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [18] Bat-Sheva Einbinder, Shai Feldman, Stephen Bates, Anastasios N Angelopoulos, Asaf Gendler, and Yaniv Romano. Label noise robustness of conformal prediction. Journal of Machine Learning Research , 2024.
- [19] AKrizhevsky. Learning multiple layers of features from tiny images. Master's thesis, University of Tront , 2009.
- [20] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition . Ieee, 2009.
- [21] Mauricio Sadinle, Jing Lei, and Larry Wasserman. Least ambiguous set-valued classifiers with bounded error levels. Journal of the American Statistical Association , 2019.
- [22] Shayan Kiyani, George J. Pappas, and Hamed Hassani. Conformal prediction with learned features. In Forty-first International Conference on Machine Learning , 2024.
- [23] Coby Penso and Jacob Goldberger. A conformal prediction score that is robust to label noise. arXiv preprint arXiv:2405.02648 , 2024.
- [24] Matteo Sesia, YX Rachel Wang, and Xin Tong. Adaptive conformal classification with noisy labels. Journal of the Royal Statistical Society Series B: Statistical Methodology , 2024.
- [25] Tongliang Liu and Dacheng Tao. Classification with noisy labels by importance reweighting. IEEE Transactions on pattern analysis and machine intelligence , 38(3):447-461, 2015.
- [26] Xiyu Yu, Tongliang Liu, Mingming Gong, Kayhan Batmanghelich, and Dacheng Tao. An efficient and provable approach for mixture proportion estimation using linear independence assumption. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 4480-4489, 2018.
- [27] Hongxin Wei, Lei Feng, Xiangyu Chen, and Bo An. Combating noisy labels by agreement: A joint training method with co-regularization. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2020.
- [28] Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213 , 2019.
- [29] Guillaume Garrigos and Robert M Gower. Handbook of convergence theorems for (stochastic) gradient methods. arXiv preprint arXiv:2301.11235 , 2023.
- [30] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems , 2019.
- [31] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2016.
- [32] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 4700-4708, 2017.
- [33] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015.

- [34] Yaniv Romano, Matteo Sesia, and Emmanuel J. Candès. Classification with valid and adaptive coverage. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems , 2020.
- [35] Anastasios Nikolas Angelopoulos, Stephen Bates, Michael I. Jordan, and Jitendra Malik. Uncertainty sets for image classifiers using conformal prediction. In 9th International Conference on Learning Representations , 2021.
- [36] Kexin Huang, Ying Jin, Emmanuel J. Candès, and Jure Leskovec. Uncertainty quantification over graph with conformalized graph neural networks. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems , 2023.
- [37] Jesse C. Cresswell, Yi Sui, Bhargava Kumar, and Noël Vouitsis. Conformal prediction sets improve human decision making. In Forty-first International Conference on Machine Learning , 2024.
- [38] Harris Papadopoulos, Kostas Proedrou, Volodya Vovk, and Alex Gammerman. Inductive confidence machines for regression. In Tapio Elomaa, Heikki Mannila, and Hannu Toivonen, editors, Machine Learning: ECML 2002, 13th European Conference on Machine Learning , 2002.
- [39] Jing Lei and Larry Wasserman. Distribution-free prediction bands for non-parametric regression. Journal of the Royal Statistical Society Series B: Statistical Methodology , 2014.
- [40] Yaniv Romano, Matteo Sesia, and Emmanuel J. Candès. Classification with valid and adaptive coverage. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems , 2020.
- [41] Jianguo Huang, HuaJun Xi, Linjun Zhang, Huaxiu Yao, Yue Qiu, and Hongxin Wei. Conformal prediction for deep classifier via label ranking. In Forty-first International Conference on Machine Learning , 2024.
- [42] Kangdao Liu, Hao Zeng, Jianguo Huang, Huiping Zhuang, Chi-Man Vong, and Hongxin Wei. C-adapter: Adapting deep classifiers for efficient conformal prediction sets. arXiv preprint arXiv:2410.09408 , 2024.
- [43] Leying Guan and Robert Tibshirani. Prediction and outlier detection in classification problems. Journal of the Royal Statistical Society Series B: Statistical Methodology , 2022.
- [44] Stephen Bates, Emmanuel Candès, Lihua Lei, Yaniv Romano, and Matteo Sesia. Testing for outliers with conformal p-values. The Annals of Statistics , 2023.
- [45] Ziyi Liang, Matteo Sesia, and Wenguang Sun. Integrative conformal p-values for out-ofdistribution testing with labelled outliers. Journal of the Royal Statistical Society Series B: Statistical Methodology , 2024.
- [46] Yu Gui, Ying Jin, and Zhimei Ren. Conformal alignment: Knowing when to trust foundation models with guarantees. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [47] John Cherian, Isaac Gibbs, and Emmanuel Candes. Large language model validity via enhanced conformal prediction methods. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [48] Heeseung Bang, Aditya Dave, and Andreas A Malikopoulos. Safe merging in mixed traffic with confidence. arXiv preprint arXiv:2403.05742 , 2024.
- [49] Christian Moya, Amirhossein Mollaali, Zecheng Zhang, Lu Lu, and Guang Lin. Conformalizeddeeponet: A distribution-free framework for uncertainty quantification in deep operator networks. arXiv preprint arXiv:2402.15406 , 2024.

- [50] Lijun Zhang, Tianbao Yang, Rong Jin, and Zhi-Hua Zhou. Dynamic regret of strongly adaptive methods. In Proceedings of the 35th International Conference on Machine Learning , 2018.
- [51] Peng Zhao, Yan-Feng Xie, Lijun Zhang, and Zhi-Hua Zhou. Efficient methods for non-stationary online learning. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems , 2022.
- [52] Terry Anderson. The theory and practice of online learning. Athabasca University , 2008.
- [53] Joi L Moore, Camille Dickson-Deane, and Krista Galyen. e-learning, online learning, and distance learning environments: Are they the same? The Internet and higher education , 2011.
- [54] Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in Optimization , 2016.
- [55] Vandana Singh and Alexander Thurman. How many ways can we define online learning? a systematic literature review of definitions of online learning (1988-2018). American Journal of Distance Education , 2019.
- [56] Steven CH Hoi, Doyen Sahoo, Jing Lu, and Peilin Zhao. Online learning: A comprehensive survey. Neurocomputing , 2021.
- [57] Xiaobo Xia, Tongliang Liu, Nannan Wang, Bo Han, Chen Gong, Gang Niu, and Masashi Sugiyama. Are anchor points really indispensable in label-noise learning? In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems , 2019.
- [58] Pengfei Chen, Junjie Ye, Guangyong Chen, Jingwei Zhao, and Pheng-Ann Heng. Beyond class-conditional assumption: A primary attempt to combat instance-dependent label noise. In Proceedings of the AAAI Conference on Artificial Intelligence , 2021.
- [59] Hongxin Wei, Lue Tao, Renchunzi Xie, and Bo An. Open-set label noise can improve robustness against inherent label noise. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems , 2021.
- [60] Yichen Wu, Jun Shu, Qi Xie, Qian Zhao, and Deyu Meng. Learning to purify noisy labels via meta soft label corrector. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021 , 2021.
- [61] Xuefeng Li, Tongliang Liu, Bo Han, Gang Niu, and Masashi Sugiyama. Provably end-toend label-noise learning without anchor points. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning , 2021.
- [62] Zhaowei Zhu, Jialu Wang, and Yang Liu. Beyond images: Label noise transition matrix estimation for tasks with lower-quality features. In International Conference on Machine Learning , 2022.
- [63] Hongxin Wei, Huiping Zhuang, Renchunzi Xie, Lei Feng, Gang Niu, Bo An, and Yixuan Li. Mitigating memorization of noisy labels by clipping the model prediction. In International Conference on Machine Learning , 2023.
- [64] Hao Chen, Jindong Wang, Ankit Shah, Ran Tao, Hongxin Wei, Xing Xie, Masashi Sugiyama, and Bhiksha Raj. Understanding and mitigating the label noise in pre-training on downstream tasks. In The Twelfth International Conference on Learning Representations , 2024.
- [65] Hongfu Gao, Feipeng Zhang, Wenyu Jiang, Jun Shu, Feng Zheng, and Hongxin Wei. On the noise robustness of in-context learning for text generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [66] Ingo Steinwart and Andreas Christmann. Estimating conditional quantiles with the help of the pinball loss. Bernoulli , 2011.

- [67] Kwang-Sung Jun, Francesco Orabona, Stephen Wright, and Rebecca Willett. Improved strongly adaptive online learning using coin betting. In Artificial Intelligence and Statistics , pages 943-951. PMLR, 2017.
- [68] Francesco Orabona and Dávid Pál. Scale-free online learning. Theoretical Computer Science , 716:50-69, 2018.

## A Related work

Conformal prediction [38, 5] is a statistical framework for uncertainty qualification. In the literature, many conformal prediction methods have been proposed across various domains, such as regression [39, 40], image classification [35, 41, 42], outlier detection [43, 44, 45], and large language models [46, 47]. Conformal prediction is also deployed in other real-world applications, such as human-inthe-loop decision-making [37], automated vehicles [48], and scientific computation [49]. In what follows, we introduce the most related works in two settings: online learning and noise robustness.

Online conformal prediction. Conventional conformal prediction algorithms provide coverage guarantees under the assumption of data exchangeability. However, in real-world online scenarios, the data distribution may evolve over time, violating the exchangeability assumption [50, 51]. To address this challenge, recent research develops online conformal prediction methods that adaptively construct prediction sets to accommodate distribution shift [9, 10, 11, 12, 13, 14]. Building on online convex optimization techniques [52, 53, 54, 55, 56, 28], these methods employ online gradient descent with pinball loss to provably achieve desired coverage under arbitrary distributional changes [9]. Still, these algorithms typically assume perfect label accuracy, an assumption that rarely holds in practice, given the prevalence of noisy labels in online learning [15, 16, 17]. In this work, we theoretically show that label noise can significantly affect the long-run mis-coverage rate through biased pinball loss gradients, leading to either inflated or deflated coverage guarantee.

Noise-robust conformal prediction. The issue of label noise has been a common challenge in machine learning with extensive studies [57, 27, 58, 59, 60, 61, 62, 63, 64, 65]. In the context of conformal prediction, recent works develop noise-robust conformal prediction algorithms for both uniform noise [23] and noise transition matrix [24]. The most relevant work [18] shows that online conformal prediction maintains valid coverage when noisy scores stochastically dominate clean scores. Our analysis extends this work by removing the assumption on noisy and clean scores, offering a more general theoretical framework for understanding the impact of uniform label noise.

## B Additional results

## B.1 Regret analysis

As [11] demonstrates, regret serves as a helpful performance measure alongside coverage. In particular, it can identify algorithms that achieve valid coverage guarantees through impractical means. For example, prediction sets that alternate between empty and full sets with frequencies { α, 1 -α } satisfy coverage bounds on any distribution but have linear regret on simple distributions (see detailed proof in Appendix A.2 of [11]). Drawing from standard online learning theory (see Theorem 2.13. of [28]), we analyze the regret bound for our method. Let τ ∗ := arg min ˆ τ ∑ T t =1 ˜ l 1 -α ( τ ∗ , ˜ S t , { S t,y } K y =1 ) , and define the regret as

<!-- formula-not-decoded -->

This leads to the following regret bound (the proof is provided in Appendix ?? ):

Proposition B.1. Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (6) , for any T ∈ N + , we have:

<!-- formula-not-decoded -->

Example 1: constant learning rate. With a constant learning rate η t ≡ η , our method yields the following regret bound:

<!-- formula-not-decoded -->

This linear regret aligns with the observation in [12]: the constant learning rate schedule, while providing valid coverage on average, lead to significant temporal variability in Coverage( ˆ τ t ) (see Proposition 1 in [12]). The linear regret bound provides additional theoretical justification for this instability.

Example 2: decaying learning rate. Suppose the proposed method updates the threshold with a decaying learning rate schedule: η t = (1 -ϵ/ 1 + ϵ + η 0 ) · √ t , where η 0 ∈ R . The inequality ∑ T t =1 1 / √ t ≤ 2 √ T follows that

<!-- formula-not-decoded -->

This sublinear regret bound O ( √ T ) implies that the decaying learning rate schedule achieves superior convergence compared to the linear regret of constant learning rates.

## B.2 Convergence analysis of ˆ τ t

We analyze the convergence of our method toward global minima by combining the convergence analysis of SGD (see Theorem 5.3 in [29]) and self-calibration inequality of pinball loss (see Theorem 2.7. in [66]). Following [66], we first establish assumptions on the distribution of S . Let R := 2 S -1 ∈ [0 , 1] , with γ denoting the 1 -α quantile of R . This indicates that γ = 2 τ -1 , where τ is the 1 -α quantile of S . We make the following assumption:

Assumption B.2. There exists constants b &gt; 0 , q ≥ 2 , and ε 0 &gt; 0 such that P { R = ˆ γ } ≥ b | ˆ γ -γ | q -2 holds for all ˆ τ ∈ [ τ -ε 0 , τ + ε 0 ] .

Furthermore, let β = b/ ( q -1) and δ = β (2 ε 0 ) q -1 . This leads to the following proposition (the proof is provided in Appendix ?? ):

Proposition B.3. Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1, 2.2 and B.2, when updating the threshold according to Eq. (6) , for any T ∈ N + , we have:

<!-- formula-not-decoded -->

where ¯ τ = ∑ T t =1 η t ˆ τ t / ∑ T t =1 η t .

## B.3 Why does standard online conformal prediction rarely exhibit under-coverage under label noise?

As established in Proposition 3.1, the label noise introduces a gap of ϵ 1 -ϵ · 1 T ∑ T t =1 ( (1 -α ) -1 K E [ C t ( X t )] ) between the actual mis-coverage rate and desired miscoverage rate α . This result indicates that when the prediction sets are sufficiently large such that (1 -α ) ≤ 1 K E [ C t ( X t )] , a high noise rate ϵ decreases this upper bound, resulting in under-coverage prediction sets, i.e., 1 T ∑ T t =1 1 { Y t / ∈ C t ( X t ) } ≥ α . However, this scenario only arises when E [ C t ( X t )] ≥ K (1 -α ) . To illustrate, considering CIFAR-100 with K = 100 classes and error rate α = 0 . 1 , under-coverage would require E [ C t ( X t )] ≥ 90 , a condition that remains improbable even with random predictions. Moreover, in such extreme cases, as (1 -α ) -1 K E [ C t ( X t )] approaches 0, the coverage gap becomes negligible, resulting in coverage rates that approximate the desired 1 -α . We verify this empirically in Figure 3, where we implement online conformal prediction using an untrained, randomly initialized ResNet18 model on CIFAR-10 and CIFAR-100 with noise rates ϵ = 0 . 05 , 0 . 1 , 0 . 15 . The results confirm that even in this extreme scenario, the coverage gap remains negligible, with coverage rates approaching the desired 0.9.

Figure 3: Performance of standard online conformal prediction under different noise rates, with a ResNet18 model on CIFAR-10 and CIFAR-100 datasets. We use noisy labels to update the threshold with decaying learning rates η t = 1 /t 1 / 2+ ε where ε = 0 . 1 .

<!-- image -->

## B.4 Common non-conformity scores

Adaptive Prediction Set (APS). [34] In the APS method, the non-conformity score of a data pair ( x , y ) is calculated by accumulating the sorted softmax probability, defined as:

<!-- formula-not-decoded -->

where π (1) ( x ) , π (2) ( x ) , · · · , π ( K ) ( x ) are the sorted softmax probabilities in descending order, and o ( y, π ( x )) denotes the order of π y ( x ) , i.e., the softmax probability for the ground-truth label y . In addition, the term u is an independent random variable that follows a uniform distribution on [0 , 1] .

Regularized Adaptive Prediction Set (RAPS). [35] The non-conformity score function of RAPS encourages a small set size by adding a penalty, as formally defined below:

<!-- formula-not-decoded -->

where ( z ) + = max { 0 , z } , k reg controls the number of penalized classes, and λ is the penalty term.

Sorted Adaptive Prediction Set (SAPS). [36] Recall that APS calculates the non-conformity score by accumulating the sorted softmax values in descending order. However, the softmax probabilities typically exhibit a long-tailed distribution, allowing for easy inclusion of those tail classes in the prediction sets. To alleviate this issue, SAPS discards all the probability values except for the maximum softmax probability when computing the non-conformity score. Formally, the nonconformity score of SAPS for a data pair ( x , y ) can be calculated as

<!-- formula-not-decoded -->

where λ is a hyperparameter representing the weight of ranking information, ˆ π max ( x ) denotes the maximum softmax probability and u is a uniform random variable.

## B.5 Additional experiments on different non-conformity score functions

We evaluate NR-OCP (Ours) against the standard online conformal prediction (Baseline) that updates the threshold with noisy labels for both constant and dynamic learning rate schedules (see Eq. (3)). We use LAC (Table 2), APS (Table 3), RAPS (Table 4) and SAPS (Table 5) scores to generate prediction sets with error rates α ∈ { 0 . 1 , 0 . 05 } , and employ noise rates ϵ ∈ { 0 . 05 , 0 . 1 , 0 . 15 } . A detailed description of these non-conformity scores is provided in Appendix B.4.

Table 2: Performance of different methods under uniform noisy labels with LAC score, using ResNet18. 'Baseline' denotes the standard online conformal prediction methods. We include two learning rate schedules: constant learning rate η = 0 . 05 and dynamic learning rates η t = 1 /t 1 / 2+ ε where ε = 0 . 1 . ' ↓ ' indicates smaller values are better and Bold numbers are superior results.

| LR Schedule   | Error rate   | Method   | CIFAR100    | CIFAR100    | CIFAR100   | CIFAR100   | ImageNet    | ImageNet    | ImageNet   | ImageNet   |
|---------------|--------------|----------|-------------|-------------|------------|------------|-------------|-------------|------------|------------|
| LR Schedule   | Error rate   | Method   | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     |
| LR Schedule   | Error rate   | Method   | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 |
| Constant      | ϵ = 0 . 05   | Baseline | 2.744       | 1.900       | 31.61      | 56.84      | 2.747       | 1.601       | 364.5      | 599.1      |
| Constant      | ϵ = 0 . 05   | Ours     | 0.289       | 0.122       | 22.78      | 47.31      | 0.018       | 0.138       | 254.7      | 530.6      |
| Constant      | ϵ = 0 . 1    | Baseline | 4.911       | 2.967       | 43.03      | 67.58      | 4.578       | 2.656       | 488.1      | 707.6      |
| Constant      | ϵ = 0 . 1    | Ours     | 0.056       | 0.189       | 26.57      | 54.18      | 0.027       | 0.156       | 312.4      | 584.9      |
| Constant      | ϵ = 0 . 15   | Baseline | 6.056       | 3.533       | 54.11      | 75.06      | 5.791       | 3.200       | 571.5      | 759.3      |
| Constant      | ϵ = 0 . 15   | Ours     | 0.378       | 0.344       | 28.39      | 57.53      | 0.251       | 0.076       | 346.7      | 603.4      |
| Dynamic       | ϵ = 0 . 05   | Baseline | 3.978       | 2.833       | 11.54      | 41.75      | 4.333       | 2.889       | 87.27      | 391.8      |
| Dynamic       | ϵ = 0 . 05   | Ours     | 0.089       | 0.067       | 4.290      | 26.56      | 0.031       | 0.020       | 16.46      | 150.5      |
| Dynamic       | ϵ = 0 . 1    | Baseline | 6.756       | 3.844       | 30.07      | 60.17      | 6.960       | 3.933       | 284.6      | 600.2      |
| Dynamic       | ϵ = 0 . 1    | Ours     | 0.233       | 0.222       | 5.394      | 30.38      | 0.091       | 0.313       | 27.76      | 227.9      |
| Dynamic       | ϵ = 0 . 15   | Baseline | 7.844       | 4.289       | 42.69      | 70.67      | 8.098       | 4.276       | 447.9      | 704.4      |
| Dynamic       | ϵ = 0 . 15   | Ours     | 0.456       | 0.067       | 8.359      | 33.60      | 0.131       | 0.158       | 38.89      | 272.5      |

Table 3: Performance of different methods under uniform noisy labels with APS score, using ResNet18. 'Baseline' denotes the standard online conformal prediction methods. We include two learning rate schedules: constant learning rate η = 0 . 05 and dynamic learning rates η t = 1 /t 1 / 2+ ε where ε = 0 . 1 . ' ↓ ' indicates smaller values are better and Bold numbers are superior results.

|             |            |          | CIFAR100    | CIFAR100    | CIFAR100   | CIFAR100   | ImageNet    | ImageNet    | ImageNet   | ImageNet   |
|-------------|------------|----------|-------------|-------------|------------|------------|-------------|-------------|------------|------------|
| LR Schedule | Error rate | Method   | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     |
|             |            |          | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 |
| Constant    | ϵ = 0 . 05 | Baseline | 4.556       | 2.933       | 5.862      | 21.17      | 4.728       | 3.508       | 13.35      | 79.88      |
| Constant    | ϵ = 0 . 05 | Ours     | 2.667       | 0.311       | 2.788      | 6.763      | 0.144       | 0.211       | 4.879      | 14.09      |
| Constant    | ϵ = 0 . 1  | Baseline | 7.511       | 3.822       | 15.71      | 37.19      | 8.402       | 4.388       | 67.25      | 224.2      |
| Constant    | ϵ = 0 . 1  | Ours     | 1.333       | 0.367       | 2.861      | 6.302      | 0.119       | 0.246       | 4.797      | 14.21      |
| Constant    | ϵ = 0 . 15 | Baseline | 8.533       | 4.188       | 29.76      | 51.63      | 9.380       | 4.644       | 192.4      | 355.3      |
| Constant    | ϵ = 0 . 15 | Ours     | 0.500       | 0.233       | 2.915      | 6.813      | 0.288       | 0.139       | 4.787      | 14.71      |
| Dynamic     | ϵ = 0 . 05 | Baseline | 4.000       | 2.433       | 4.693      | 13.03      | 4.157       | 2.122       | 11.31      | 31.38      |
| Dynamic     | ϵ = 0 . 05 | Ours     | 0.278       | 0.466       | 2.427      | 5.475      | 0.108       | 0.255       | 4.431      | 13.36      |
| Dynamic     | ϵ = 0 . 1  | Baseline | 7.211       | 3.366       | 11.14      | 21.91      | 6.791       | 3.137       | 27.78      | 54.14      |
| Dynamic     | ϵ = 0 . 1  | Ours     | 0.400       | 0.456       | 2.716      | 5.515      | 0.028       | 0.277       | 4.513      | 13.30      |
| Dynamic     | ϵ = 0 . 15 | Baseline | 8.288       | 3.911       | 19.62      | 31.19      | 7.928       | 3.675       | 47.88      | 80.33      |
| Dynamic     | ϵ = 0 . 15 | Ours     | 0.188       | 0.211       | 2.479      | 5.790      | 0.264       | 0.088       | 4.321      | 14.03      |

Table 4: Performance of different methods under uniform noisy labels with RAPS score, using ResNet18. 'Baseline' denotes the standard online conformal prediction methods. We include two learning rate schedules: constant learning rate η = 0 . 05 and dynamic learning rates η t = 1 /t 1 / 2+ ε where ε = 0 . 1 . ' ↓ ' indicates smaller values are better and Bold numbers are superior results.

| LR Schedule   |            | Method   | CIFAR100    | CIFAR100    | CIFAR100   | CIFAR100   | ImageNet    | ImageNet    | ImageNet   | ImageNet   |
|---------------|------------|----------|-------------|-------------|------------|------------|-------------|-------------|------------|------------|
|               | Error rate |          | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     |
|               |            |          | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 |
| Constant      | ϵ = 0 . 05 | Baseline | 3.978       | 3.533       | 5.275      | 26.71      | 4.716       | 4.509       | 13.73      | 164.7      |
| Constant      | ϵ = 0 . 05 | Ours     | 0.533       | 0.089       | 2.971      | 6.265      | 0.051       | 0.673       | 5.217      | 14.21      |
|               | ϵ = 0 . 1  | Baseline | 7.656       | 4.278       | 15.25      | 50.25      | 8.689       | 4.84        | 85.19      | 395.5      |
|               | ϵ = 0 . 1  | Ours     | 0.356       | 0.200       | 3.126      | 6.370      | 0.184       | 0.271       | 5.264      | 14.76      |
|               | ϵ = 0 . 15 | Baseline | 8.896       | 4.467       | 36.03      | 62.27      | 9.733       | 4.933       | 305.6      | 576.3      |
|               | ϵ = 0 . 15 | Ours     | 0.489       | 0.560       | 3.500      | 7.693      | 0.224       | 0.227       | 5.616      | 19.81      |
| Dynamic       | ϵ = 0 . 05 | Baseline | 4.322       | 3.254       | 4.883      | 19.42      | 4.642       | 3.457       | 12.57      | 60.94      |
| Dynamic       | ϵ = 0 . 05 | Ours     | 0.233       | 0.211       | 2.586      | 5.344      | 0.067       | 0.078       | 4.305      | 13.73      |
| Dynamic       | ϵ = 0 . 1  | Baseline | 8.378       | 4.400       | 19.68      | 44.43      | 8.084       | 4.311       | 54.42      | 153.7      |
| Dynamic       | ϵ = 0 . 1  | Ours     | 0.756       | 0.122       | 2.891      | 5.942      | 0.009       | 0.140       | 4.381      | 14.67      |
| Dynamic       | ϵ = 0 . 15 | Baseline | 9.022       | 4.600       | 33.87      | 57.87      | 9.256       | 4.978       | 133.3      | 239.8      |
| Dynamic       | ϵ = 0 . 15 | Ours     | 0.011       | 0.440       | 2.821      | 5.290      | 0.231       | 0.264       | 4.578      | 15.59      |

Table 5: Performance of different methods under uniform noisy labels with SAPS score, using ResNet18. 'Baseline' denotes the standard online conformal prediction methods. We include two learning rate schedules: constant learning rate η = 0 . 05 and dynamic learning rates η t = 1 /t 1 / 2+ ε where ε = 0 . 1 . ' ↓ ' indicates smaller values are better and Bold numbers are superior results.

| LR Schedule   | Error rate   | Method   | CIFAR100    | CIFAR100    | CIFAR100   | CIFAR100   | ImageNet    | ImageNet    | ImageNet   | ImageNet   |
|---------------|--------------|----------|-------------|-------------|------------|------------|-------------|-------------|------------|------------|
| LR Schedule   | Error rate   | Method   | CovGap(%) ↓ | CovGap(%) ↓ | Size       | Size       | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     |
| LR Schedule   | Error rate   | Method   | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 |
| Constant      | ϵ = 0 . 05   | Baseline | 4.489       | 3.100       | 4.982      | 17.92      | 4.462       | 3.537       | 12.53      | 83.04      |
| Constant      | ϵ = 0 . 05   | Ours     | 0.455       | 0.211       | 2.604      | 5.810      | 0.124       | 0.067       | 4.702      | 14.32      |
| Constant      | ϵ = 0 . 1    | Baseline | 8.478       | 3.944       | 13.25      | 32.71      | 8.368       | 4.389       | 73.14      | 231.5      |
| Constant      | ϵ = 0 . 1    | Ours     | 0.533       | 0.955       | 2.701      | 4.987      | 0.051       | 0.102       | 5.013      | 15.37      |
| Constant      | ϵ = 0 . 15   | Baseline | 8.644       | 4.400       | 27.89      | 50.73      | 9.360       | 4.613       | 202.3      | 361.7      |
| Constant      | ϵ = 0 . 15   | Ours     | 0.711       | 0.444       | 2.778      | 5.609      | 0.029       | 0.135       | 4.971      | 15.68      |
| Dynamic       | ϵ = 0 . 05   | Baseline | 4.122       | 2.600       | 4.992      | 13.68      | 4.859       | 1.866       | 13.46      | 28.28      |
| Dynamic       | ϵ = 0 . 05   | Ours     | 0.078       | 1.889       | 2.528      | 6.417      | 0.275       | 0.615       | 4.813      | 12.06      |
| Dynamic       | ϵ = 0 . 1    | Baseline | 7.311       | 3.322       | 11.84      | 22.19      | 7.162       | 3.000       | 31.37      | 51.64      |
| Dynamic       | ϵ = 0 . 1    | Ours     | 0.267       | 0.067       | 2.444      | 6.218      | 0.186       | 0.553       | 4.715      | 12.23      |
| Dynamic       | ϵ = 0 . 15   | Baseline | 8.456       | 4.044       | 21.44      | 33.21      | 8.204       | 3.577       | 55.78      | 75.26      |
| Dynamic       | ϵ = 0 . 15   | Ours     | 0.200       | 0.063       | 2.602      | 7.230      | 0.104       | 0.515       | 4.594      | 12.43      |

## B.6 Additional experiments on different model architectures

We evaluate NR-OCP (Ours) against the standard online conformal prediction (Baseline) that updates the threshold with noisy labels for both constant and dynamic learning rate schedules (see Eq. (3) and Eq. ( ?? )), employing ResNet50 (Table 6), DenseNet121 (Table 7), and VGG16 (Table 8). We use LAC scores to generate prediction sets with error rates α ∈ { 0 . 1 , 0 . 05 } , and employ noise rates ϵ ∈ { 0 . 05 , 0 . 1 , 0 . 15 } .

Table 6: Performance of different methods under uniform noisy labels with LAC score, using ResNet50. 'Baseline' denotes the standard online conformal prediction methods. We include two learning rate schedules: constant learning rate η = 0 . 05 and dynamic learning rates η t = 1 /t 1 / 2+ ε where ε = 0 . 1 . ' ↓ ' indicates smaller values are better and Bold numbers are superior results.

| LR Schedule   | Error rate   | Method   | CIFAR100    | CIFAR100    | CIFAR100   | CIFAR100   | ImageNet    | ImageNet    | ImageNet   | ImageNet   |
|---------------|--------------|----------|-------------|-------------|------------|------------|-------------|-------------|------------|------------|
| LR Schedule   | Error rate   | Method   | CovGap(%) ↓ | CovGap(%) ↓ | Size       | Size       | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     |
| LR Schedule   | Error rate   | Method   | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 |
| Constant      | ϵ = 0 . 05   | Baseline | 3.244       | 1.783       | 28.58      | 56.08      | 3.226       | 1.982       | 294.4      | 575.7      |
| Constant      | ϵ = 0 . 05   | Ours     | 0.033       | 0.089       | 16.26      | 46.04      | 0.073       | 0.035       | 188.5      | 466.8      |
| Constant      | ϵ = 0 . 1    | Baseline | 5.267       | 2.879       | 42.09      | 67.25      | 5.326       | 2.935       | 425.9      | 674.2      |
| Constant      | ϵ = 0 . 1    | Ours     | 0.125       | 0.122       | 17.88      | 49.06      | 0.231       | 0.067       | 241.1      | 506.4      |
| Constant      | ϵ = 0 . 15   | Baseline | 6.369       | 3.445       | 53.34      | 74.03      | 6.584       | 3.453       | 531.4      | 758.5      |
| Constant      | ϵ = 0 . 15   | Ours     | 0.223       | 0.216       | 21.62      | 50.14      | 0.202       | 0.056       | 263.1      | 549.4      |
| Dynamic       | ϵ = 0 . 05   | Baseline | 4.166       | 2.724       | 10.91      | 41.95      | 4.397       | 3.104       | 44.86      | 361.8      |
| Dynamic       | ϵ = 0 . 05   | Ours     | 0.115       | 0.189       | 3.120      | 18.35      | 0.082       | 0.048       | 8.377      | 96.68      |
| Dynamic       | ϵ = 0 . 1    | Baseline | 6.984       | 3.921       | 30.09      | 61.96      | 7.279       | 4.073       | 232.8      | 582.4      |
| Dynamic       | ϵ = 0 . 1    | Ours     | 0.205       | 0.310       | 4.143      | 27.87      | 0.113       | 0.178       | 10.83      | 170.2      |
| Dynamic       | ϵ = 0 . 15   | Baseline | 8.093       | 4.223       | 43.90      | 70.67      | 8.406       | 4.453       | 417.9      | 705.9      |
| Dynamic       | ϵ = 0 . 15   | Ours     | 0.244       | 0.043       | 4.368      | 27.81      | 0.133       | 0.059       | 21.98      | 178.0      |

Table 7: Performance of different methods under uniform noisy labels with LAC score, using DenseNet121. 'Baseline' denotes the standard online conformal prediction methods. We include two learning rate schedules: constant learning rate η = 0 . 05 and dynamic learning rates η t = 1 /t 1 / 2+ ε where ε = 0 . 1 . ' ↓ ' indicates smaller values are better and Bold numbers are superior results.

| LR Schedule   | Error rate   | Method   | CIFAR100    | CIFAR100    | CIFAR100   | CIFAR100   | ImageNet    | ImageNet    | ImageNet   | ImageNet   |
|---------------|--------------|----------|-------------|-------------|------------|------------|-------------|-------------|------------|------------|
| LR Schedule   | Error rate   | Method   | CovGap(%) ↓ | CovGap(%) ↓ | Size       | Size       | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     |
| LR Schedule   | Error rate   | Method   | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 |
| Constant      | ϵ = 0 . 05   | Baseline | 3.978       | 3.533       | 5.275      | 26.71      | 4.716       | 4.509       | 13.73      | 164.7      |
| Constant      | ϵ = 0 . 05   | Ours     | 0.533       | 0.089       | 2.971      | 6.265      | 0.051       | 0.673       | 5.217      | 14.21      |
| Constant      | ϵ = 0 . 1    | Baseline | 7.656       | 4.278       | 15.25      | 50.25      | 8.689       | 4.84        | 85.19      | 395.5      |
| Constant      | ϵ = 0 . 1    | Ours     | 0.356       | 0.200       | 3.126      | 6.370      | 0.184       | 0.271       | 5.264      | 14.76      |
| Constant      | ϵ = 0 . 15   | Baseline | 8.896       | 4.467       | 36.03      | 62.27      | 9.733       | 4.933       | 305.6      | 576.3      |
| Constant      | ϵ = 0 . 15   | Ours     | 0.489       | 0.560       | 3.500      | 7.693      | 0.224       | 0.227       | 5.616      | 19.81      |
| Dynamic       | ϵ = 0 . 05   | Baseline | 4.322       | 3.254       | 4.883      | 19.42      | 4.642       | 3.457       | 12.57      | 60.94      |
| Dynamic       | ϵ = 0 . 05   | Ours     | 0.233       | 0.211       | 2.586      | 5.344      | 0.067       | 0.078       | 4.305      | 13.73      |
| Dynamic       | ϵ = 0 . 1    | Baseline | 8.378       | 4.400       | 19.68      | 44.43      | 8.084       | 4.311       | 54.42      | 153.7      |
| Dynamic       | ϵ = 0 . 1    | Ours     | 0.756       | 0.122       | 2.891      | 5.942      | 0.009       | 0.140       | 4.381      | 14.67      |
| Dynamic       | ϵ = 0 . 15   | Baseline | 9.022       | 4.600       | 33.87      | 57.87      | 9.256       | 4.978       | 133.3      | 239.8      |
| Dynamic       | ϵ = 0 . 15   | Ours     | 0.011       | 0.440       | 2.821      | 5.290      | 0.231       | 0.264       | 4.578      | 15.59      |

Table 8: Performance of different methods under uniform noisy labels with LAC score, using VGG16. 'Baseline' denotes the standard online conformal prediction methods. We include two learning rate schedules: constant learning rate η = 0 . 05 and dynamic learning rates η t = 1 /t 1 / 2+ ε where ε = 0 . 1 . ' ↓ ' indicates smaller values are better and Bold numbers are superior results.

| LR Schedule   | Error rate   | Method   | CIFAR100    | CIFAR100    | CIFAR100   | CIFAR100   | ImageNet    | ImageNet    | ImageNet   | ImageNet   |
|---------------|--------------|----------|-------------|-------------|------------|------------|-------------|-------------|------------|------------|
| LR Schedule   | Error rate   | Method   | CovGap(%) ↓ | CovGap(%) ↓ | Size       | Size       | CovGap(%) ↓ | CovGap(%) ↓ | Size ↓     | Size ↓     |
| LR Schedule   | Error rate   | Method   | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 | α = 0 . 1   | α = 0 . 05  | α = 0 . 1  | α = 0 . 05 |
| Constant      | ϵ = 0 . 05   | Baseline | 1.911       | 1.033       | 51.13      | 73.03      | 3.164       | 1.995       | 297.8      | 565.3      |
| Constant      | ϵ = 0 . 05   | Ours     | 0.078       | 0.144       | 43.50      | 67.97      | 0.002       | 0.031       | 206.6      | 488.1      |
| Constant      | ϵ = 0 . 1    | Baseline | 3.474       | 1.700       | 58.08      | 76.44      | 4.878       | 2.975       | 439.8      | 669.4      |
| Constant      | ϵ = 0 . 1    | Ours     | 0.224       | 0.267       | 45.48      | 68.28      | 0.217       | 0.119       | 229.4      | 511.2      |
| Constant      | ϵ = 0 . 15   | Baseline | 4.489       | 2.411       | 64.16      | 80.16      | 6.173       | 3.626       | 544.2      | 746.8      |
| Constant      | ϵ = 0 . 15   | Ours     | 0.300       | 0.189       | 44.43      | 71.81      | 0.168       | 0.004       | 281.9      | 551.3      |
| Dynamic       | ϵ = 0 . 05   | Baseline | 2.678       | 1.388       | 40.96      | 66.81      | 4.608       | 3.044       | 51.64      | 374.1      |
| Dynamic       | ϵ = 0 . 05   | Ours     | 0.244       | 0.004       | 28.92      | 59.26      | 0.062       | 0.013       | 5.316      | 112.7      |
| Dynamic       | ϵ = 0 . 1    | Baseline | 4.189       | 2.567       | 50.40      | 74.38      | 7.142       | 4.067       | 244.4      | 580.6      |
| Dynamic       | ϵ = 0 . 1    | Ours     | 0.289       | 0.133       | 30.81      | 61.08      | 0.195       | 0.071       | 10.65      | 152.1      |
| Dynamic       | ϵ = 0 . 15   | Baseline | 5.733       | 3.044       | 60.46      | 78.65      | 8.413       | 4.451       | 432.2      | 708.9      |
| Dynamic       | ϵ = 0 . 15   | Ours     | 0.440       | 0.144       | 35.67      | 63.38      | 0.119       | 0.282       | 18.39      | 224.6      |

## C The impact of misestimated noise rate on NR-OCP

In this section, we conduct an additional experiment where the noise rate is unknown and needs to be estimated. In particular, we apply an existing algorithm [61] to estimate the noise rate without access to clean data during the training of the pre-trained model. Then, we employ this estimated noise rate in our method for online conformal prediction. The experiments are conducted on CIFAR-10 dataset, using ResNet18 model with noise rates ϵ ∈ { 0 . 1 , 0 . 2 , 0 . 3 } . We employ LAC score to generate prediction sets and use a constant learning rate η = 0 . 05 . The results demonstrate that our method consistently achieves much lower coverage gaps and smaller prediction sets compared to the baseline. This highlights the robustness and applicability of our method even when the noise rate is not precisely estimated.

Table 9

| Noise rate   | Estimated Noise rate   | Method        | CovGap (%)   | Size      |
|--------------|------------------------|---------------|--------------|-----------|
| ϵ = 0 . 1    | ˆ ϵ = 0 . 09           | Baseline Ours | 6.99 0.82    | 2.47 0.94 |
| ϵ = 0 . 2    | ˆ ϵ = 0 . 23           | Baseline Ours | 8.72 1.67    | 5.41 0.98 |
| ϵ = 0 . 3    | ˆ ϵ = 0 . 26           | Baseline Ours | 9.04 2.32    | 6.72 1.15 |

## D Additional information on Strongly adaptive online conformal prediction

## D.1 Strongly adaptive online conformal prediction

The SAOCP algorithm leverages techniques for minimizing the strongly adaptive regret [67] to perform online conformal prediction. In particular, SAOCP is a meta-algorithm that manages multiple experts, where each expert is itself an arbitrary online learning algorithm taking charge of its own active interval that has a finite lifetime . At each t ∈ [ T ] , SAOCP instantiates a new expert A t with active interval [ t, t + L ( t ) -1] , where L ( t ) is the lifetime:

<!-- formula-not-decoded -->

where g ∈ Z ≥ 1 is a multiplier for the lifetime of each expert. It can be shown that at most g log 2 t experts are active at any time t under the choice of L ( t ) in (7), resulting in a total runtime of O ( T log T ) for SAOCP when g = Θ(1) . At each time t , the threshold ˆ τ t is obtained by aggregating the predictions of the active experts.

<!-- formula-not-decoded -->

where the weight { p i,t } i ∈ [ t ] is computed by the coin betting scheme [67, 68].

Choice of expert. SAOCP employs Scale-Free OGD (dubbed SF-OGD) [68] as its expert. In particular, SF-OGD is a variant of OGD that decays its effective learning rate based on cumulative past gradient norms.

## D.2 Noise-robust strongly adaptive online conformal prediction

In the following, we present a pseudo-algorithm of NR-SAOCP. The essence of this method is to update the threshold with our robust pinball loss. In Algorithm 1, we demonstrate how NR-SAOCP manages multiple experts to update the threshold. In Algorithm 2, we present how each expert updates the corresponding threshold.

## Algorithm 1 Noise-Robust Strongly Adaptive Online Conformal Prediction (NR-SAOCP)

Require: Target coverage 1 -α ∈ (0 , 1) , initial threshold ˆ τ 0 .

- 1: for t = 1 , . . . , T do
- 2: Initialize new expert A t = NR-SF-OGD ( α ← α ; η ← 1 / √ 3; ˆ τ 1 ← ˆ τ t -1 ) (Algorithm 2), and set weight w t,t = 0
- 3: Compute active set Active ( t ) = { i ∈ [ T ] : t -L ( i ) ≤ i ≤ t } , where L ( i ) is defined in Eq. (7)
- 4: Compute prior probability π i ∝ i -2 (1 + ⌊ w t,i ⌋ + ) 1 { i ∈ Active ( t ) } , compute un-normalized probability ˆ p i = π i [ w t,i ] + for all i ∈ [ t ] , normalize p = ˆ p/ || ˆ p || 1 if || ˆ p || 1 &gt; 0 , else p = π
- 5: Update the threshold ˆ τ t = ∑ i ∈ Active ( t ) p i ˆ τ i t
- 6: Observe input X t and construct prediction set ˆ C t ( X t ) as in Eq. (1)
- 7: Observe true label Y t ∈ Y , compute non-conformity score S t = S ( X t , Y t ) and S t,y = S ( X t , y ) for all y ∈ Y
- 8: for i ∈ Active ( t ) do
- 9: Update expert A i with robust pinball loss and obtain next predicted radius ˆ τ i t +1
- 10: Compute

<!-- formula-not-decoded -->

where ˜ l is the robust pinball loss defined in Eq. (4).

<!-- formula-not-decoded -->

- 12: end for
- 13: end for

## Algorithm 2 Noise-Robust Scale-Free Online Gradient Descent (NR-SF-OGD)

Require: α ∈ (0 , 1) , learning rate η &gt; 0 , initial threshold ˆ τ 1 ∈ R

- 1: for t ≥ 1 do
- 2: Observe input X t and construct prediction set ˆ C t ( X t ) as in Eq. (1)
- 3: Observe true label Y t ∈ Y , compute non-conformity score S t = S ( X t , Y t ) and S t,y = S ( X t , y ) for all y ∈ Y
- 4: Update the threshold:

<!-- formula-not-decoded -->

where ˜ l is the robust pinball loss defined in Eq. (4).

- 5: end for

## E Omitted proofs

## E.1 Proof for Proposition 3.1

Lemma E.1. Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (3) , for any T ∈ N + , we have

<!-- formula-not-decoded -->

Proof. We prove this by induction. First, we know ˆ τ 1 ∈ [0 , 1] by assumption, which indicates that Eq. (8) is satisfied at t = 1 . Then, we assume that Eq. (8) holds for t = T , and we will show that ˆ τ T +1 lies in this range. Consider three cases:

Case 1. If ˆ τ T ∈ [0 , 1] , we have

<!-- formula-not-decoded -->

where (a) is because ∇ ˆ τ t l 1 -α (ˆ τ t , ˜ S t ) ∈ [ α -1 , α ] .

Case 2. Consider the case where ˆ τ T ∈ [1 , 1 + (1 -α ) η ] . The assumption that ˜ S ∈ [0 , 1] implies 1 { ˜ S &gt; ˆ τ t } = 0 . Thus, we have

<!-- formula-not-decoded -->

which follows that

<!-- formula-not-decoded -->

Case 3. Consider the case where ˆ τ T ∈ [ -αη, 0] . The assumption that ˜ S ∈ [0 , 1] implies 1 { ˜ S &gt; ˆ τ t } = 1 . Thus, we have

<!-- formula-not-decoded -->

which follows that

ˆ τ T +1 = ˆ τ T -η · ∇ ˆ τ t l 1 -α (ˆ τ t , ˜ S t ) = ˆ τ T -η ( -1 + α ) ∈ [ -αη, (1 -α ) η ] ⊂ [ -αη, 1 + (1 -α ) η ] Combining three cases, we can conclude that

<!-- formula-not-decoded -->

Proposition E.2 (Restatement of Proposition 3.1) . Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (3) , for any δ ∈ (0 , 1) and T ∈ N + , the following bound holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

Proof. Consider the gradient of clean pinball loss:

<!-- formula-not-decoded -->

Part (a): Lemma F.3 gives us that

<!-- formula-not-decoded -->

Part (b): Recall that the expected gradient of the clean pinball loss satisfies

<!-- formula-not-decoded -->

where (a) comes from Lemma F.1. In addition, Lemma F.3 implies that

<!-- formula-not-decoded -->

Thus, we can derive an upper bound for (b) as follows

<!-- formula-not-decoded -->

Then, we derive an upper bound for (c). The update rule (Eq. (3)) gives us that

<!-- formula-not-decoded -->

Accumulating from t = 1 to t = T and taking absolute value gives

<!-- formula-not-decoded -->

where (a) follows from the assumption that ˆ τ 1 ∈ [0 , 1] and Lemma E.1. Thus, we have

<!-- formula-not-decoded -->

By taking union bound and combining two parts, we can obtain that

<!-- formula-not-decoded -->

holds with at least 1 -δ probability. Recall that

<!-- formula-not-decoded -->

We can conclude that

<!-- formula-not-decoded -->

## E.2 Proof for Proposition 4.1

Proposition E.3 (Restatement of Proposition 4.1) . The robust pinball loss defined in Eq. (4) satisfies the following two properties:

<!-- formula-not-decoded -->

Proof. The property (1): We begin by proving the first property. Taking expectation on pinball loss gives

<!-- formula-not-decoded -->

Part (a):

<!-- formula-not-decoded -->

Part (b):

<!-- formula-not-decoded -->

Part (c):

<!-- formula-not-decoded -->

Part (d):

<!-- formula-not-decoded -->

Combining (a), (b), (c), and (d), we can conclude that

<!-- formula-not-decoded -->

The property (2): We proceed by proving the second property, which demonstrates that the expected gradient of the robust pinball loss (computed using noise and random scores) equals the expected gradient of the true pinball loss. Consider the gradient of robust pinball loss:

<!-- formula-not-decoded -->

## E.3 Proof for Proposition 4.2

Lemma E.4. Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (5) , for any δ ∈ (0 , 1) and T ∈ N + , we have

<!-- formula-not-decoded -->

Proof. We prove this by induction. First, we know ˆ τ 1 ∈ [0 , 1] by assumption, which indicates that Eq. (9) is satisfied at t = 1 . Then, we assume that Eq. (9) holds for t = T , and we will show that ˆ τ T +1 lies in this range. Consider three cases:

Case 1. If ˆ τ T ∈ [0 , 1] , we have

<!-- formula-not-decoded -->

where (a) follows from Lemma F.2.

Case 2. Consider the case where ˆ τ T ∈ [1 , 1+ αη -η/ (1 -ϵ )] . The assumption that ˜ S T , S t,y ∈ [0 , 1] implies 1 { ˜ S T ≤ ˆ τ T } = 1 { S t,y ≤ ˆ τ T } = 1 . Thus, we have

<!-- formula-not-decoded -->

which follows that

<!-- formula-not-decoded -->

Case 3. Consider the case where ˆ τ T ∈ [ -αη -ϵη/ (1 -ϵ ) , 0] . The assumption that ˜ s, s y ∈ [0 , 1] implies 1 { ˜ s ≤ ˆ τ T } = 1 { s y ≤ ˆ τ T } = 0 . Thus, we have

<!-- formula-not-decoded -->

which follows that

<!-- formula-not-decoded -->

Combining three cases, we can conclude that

<!-- formula-not-decoded -->

Proposition E.5 (Restatement of Proposition 4.2) . Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (5) , for any δ ∈ (0 , 1) and T ∈ N + , the following bound holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

Proof. The update rule of the threshold ˆ τ t gives us that

<!-- formula-not-decoded -->

Accumulating from t = 1 to t = T and taking absolute value gives

<!-- formula-not-decoded -->

where (a) follows from the assumption that ˆ τ 1 ∈ [0 , 1] and Lemma E.4. In addition, Lemma F.5 gives us that

<!-- formula-not-decoded -->

Thus, with at least 1 -δ probability, we have

<!-- formula-not-decoded -->

Applying the second property in Proposition 4.1, we have

<!-- formula-not-decoded -->

which implies that with at least 1 -δ probability,

<!-- formula-not-decoded -->

Therefore, we can conclude that

<!-- formula-not-decoded -->

holds with at least 1 -δ probability.

## E.4 Proof for Proposition 4.3

Proposition E.6 (Restatement of Proposition 4.3) . Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (5) , for any δ ∈ (0 , 1) and T ∈ N + , the following bound holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

Proof. Lemma F.3 gives us that

<!-- formula-not-decoded -->

Follows from Proposition 4.2 (see Eq. (10)), we know

<!-- formula-not-decoded -->

holds with at least 1 -δ/ 2 probability. In addition, recall that

<!-- formula-not-decoded -->

Thus, by union bound, we can obtain that with at least probability 1 -δ ,

<!-- formula-not-decoded -->

Recall that

<!-- formula-not-decoded -->

Therefore, we can conclude that

<!-- formula-not-decoded -->

holds with at least 1 -δ probability.

## E.5 Proof for Proposition 4.5

Proposition E.7 (Restatement of Proposition 4.5) . Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (6) , for any δ ∈ (0 , 1) and T ∈ N + , the following bound holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (a) comes from the update rule (Eq. (6)), (b) is due to triangle inequality, and (c) follows from Lemma ?? . In addition, Lemma F.5 gives us that

<!-- formula-not-decoded -->

Thus, with at least 1 -δ probability, we have

<!-- formula-not-decoded -->

Applying the second property in Proposition 4.1, we have

<!-- formula-not-decoded -->

which implies that with at least 1 -δ probability,

<!-- formula-not-decoded -->

Therefore, we can conclude that

<!-- formula-not-decoded -->

holds with at least 1 -δ probability.

## E.6 Proof for Proposition 4.6

Proposition E.8 (Restatement of Proposition 4.6) . Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (6) , for any δ ∈ (0 , 1) and T ∈ N + , the following bound holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

Proof. Lemma F.3 gives us that

<!-- formula-not-decoded -->

Follows from Proposition 4.5 (see Eq. (11)), we know

<!-- formula-not-decoded -->

holds with at least 1 -δ/ 2 probability. In addition, recall that

<!-- formula-not-decoded -->

Thus, by union bound, we can obtain that with at least probability 1 -δ ,

<!-- formula-not-decoded -->

Recall that

<!-- formula-not-decoded -->

Therefore, we can conclude that

<!-- formula-not-decoded -->

holds with at least 1 -δ probability.

## E.7 Proof for Proposition B.1

Proposition E.9 (Restatement of Proposition B.1) . Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (6) , for any T ∈ N + , we have:

<!-- formula-not-decoded -->

Proof. The update rule (Eq. (2)) gives us that

<!-- formula-not-decoded -->

Recall that the robust pinball loss is defined as

<!-- formula-not-decoded -->

Since pinball loss is convex, robust pinball loss inherits the convexity property. Thus, we have

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

Following from Lemma F.2, we have

<!-- formula-not-decoded -->

Dividing this inequality by η t and summing over t = 1 , 2 , · · · , T provides

<!-- formula-not-decoded -->

Lemma F.4 gives us that which follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E.8 Proof for Proposition B.3

Lemma E.10. Under Assumption B.2, the pinball loss satisfies

<!-- formula-not-decoded -->

Proof. We employ the proof technique from Lemma C.4 in [11]. The Assumption B.2 gives us that

<!-- formula-not-decoded -->

By Theorem 2.7 of [66], we have

<!-- formula-not-decoded -->

Since | ˆ γ -γ | = 2 | ˆ τ -τ | and l 1 -α (ˆ γ, R ) = l 1 -α (ˆ τ, S ) , we can obtain

<!-- formula-not-decoded -->

Proposition E.11 (Restatement of Proposition B.3) . Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1, 2.2 and B.2, when updating the threshold according to Eq. (6) , for any T ∈ N + , we have:

<!-- formula-not-decoded -->

where ¯ τ = ∑ T t =1 η t ˆ τ t / ∑ T t =1 η t .

Proof. We begin our proof from Eq. (12):

<!-- formula-not-decoded -->

By taking expectation condition on ( X t , Y t ) (or equivalently on ˜ S t and { S t,y } K y =1 ), and applying Lemma F.2 and Proposition 4.1, we have

<!-- formula-not-decoded -->

By rearranging and taking expectation, we have

<!-- formula-not-decoded -->

Summing over t = 1 , 2 , · · · , T provides

<!-- formula-not-decoded -->

Let us denote ¯ τ := ∑ T t =1 η t ˆ τ t / ∑ T t =1 η t . Applying Jensen's inequality and dividing both sides by 2 ∑ T t =1 η t gives

<!-- formula-not-decoded -->

Continuing from Lemma E.10, we can conclude that

<!-- formula-not-decoded -->

## F Helpful lemmas

Lemma F.1. The distribution of the true non-conformity score, noise non-conformity score, and scores of all classes satisfy the following relationship:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where S , ˜ S , and S y denote the true score, noisy score, and score for class y respectively.

Proof. (1):

<!-- formula-not-decoded -->

which follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which follows that

<!-- formula-not-decoded -->

(3):

<!-- formula-not-decoded -->

which follows that

<!-- formula-not-decoded -->

Lemma F.2. The gradient of robust pinball loss can be bounded as follows

<!-- formula-not-decoded -->

Proof. Consider the gradient of robust pinball loss:

<!-- formula-not-decoded -->

Due to fact that 1 {·} ∈ [0 , 1] , we can bound each part as follows: Part (a):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining four parts, we can conclude that

<!-- formula-not-decoded -->

Lemma F.3. With at least probability 1 -δ , we have

<!-- formula-not-decoded -->

Proof. Define

<!-- formula-not-decoded -->

Now, we will verify that { Y T } is a martingale, and { D T } is a bounded martingale difference sequence. Due to the definition of { Y T } , we have

<!-- formula-not-decoded -->

where the last equality follows from the definition of { D T } . In addition, we have

<!-- formula-not-decoded -->

Part (b):

Part (c):

Part (d):

and Eq. (13) gives us that

<!-- formula-not-decoded -->

Therefore, by applying Azuma-Hoeffding's inequality, we can have

<!-- formula-not-decoded -->

Using r = √ 2 T log(4 /δ ) , we have

<!-- formula-not-decoded -->

Lemma F.4. Consider online conformal prediction under uniform label noise with noise rate ϵ ∈ (0 , 1) . Given Assumptions 2.1 and 2.2, when updating the threshold according to Eq. (6) , for any T ∈ N + , we have

<!-- formula-not-decoded -->

for all T ∈ N + .

Proof. We prove this by induction. First, we know ˆ τ 1 ∈ [0 , 1] by assumption, which indicates that Eq. (14) is satisfied at t = 1 . Then, we assume that Eq. (14) holds for t = T , and we will show that ˆ τ T +1 lies in this range. Consider three cases:

Case 1. If ˆ τ T ∈ [0 , 1] , we have

<!-- formula-not-decoded -->

where (a) follows from Eq. (13).

Case 2. Consider the case where ˆ τ T ∈ [1 , 1 + max 1 ≤ t ≤ T -1 η t · (1 / (1 -ϵ ) -α )] . The assumption that ˜ S T , S t,y ∈ [0 , 1] implies 1 { ˜ S T ≤ ˆ τ T } = 1 { S t,y ≤ ˆ τ T } = 1 . Thus, we have

<!-- formula-not-decoded -->

which follows that

<!-- formula-not-decoded -->

Case 3. Consider the case where ˆ τ T ∈ [ -max 1 ≤ t ≤ T -1 η · ( α + ϵ/ (1 -ϵ )) , 0] . The assumption that ˜ S, S y ∈ [0 , 1] implies 1 { ˜ S ≤ ˆ τ T } = 1 { S y ≤ ˆ τ T } = 0 . Thus, we have

<!-- formula-not-decoded -->

which follows that

<!-- formula-not-decoded -->

Combining three cases, we can conclude that

<!-- formula-not-decoded -->

Lemma F.5. With at least probability 1 -δ , we have

<!-- formula-not-decoded -->

Proof. Define

<!-- formula-not-decoded -->

Now, we will verify that { Y T } is a martingale, and { D T } is a bounded martingale difference sequence. Due to the definition of { Y T } , we have

<!-- formula-not-decoded -->

where the last equality follows from the definition of { D T } . In addition, we have

<!-- formula-not-decoded -->

and Lemma F.2 gives us that

<!-- formula-not-decoded -->

Therefore, by applying Azuma-Hoeffding's inequality, we can have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS paper checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction summarize the paper's key contributions: 1. theoretical analysis of the impact of label noise; 2. a method to eliminate this impact; 3. Extensive evaluation of the method.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitation in the Section 6.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We clearly present the assumptions made in the theoretical results. All the proofs can be found in the Appendix.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We have listed all the information needed in Section 5.

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

Justification: We include an example code in supplemental material.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All the details for reproducing the main experimental results are presented in Section 5.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification:

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

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The information is provided in Section 5.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This paper is aligned with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper focuses on online conformal prediction, and it has no societal impacts.

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

Justification: All the datasets and models are available online.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All the creators or original owners of assets are properly credited.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.