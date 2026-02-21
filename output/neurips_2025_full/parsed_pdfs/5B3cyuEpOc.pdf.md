## Provable Sample-Efficient Transfer Learning Conditional Diffusion Models via Representation Learning

Ziheng Cheng 1 , Tianyu Xie 2 , Shiyue Zhang 2 , Cheng Zhang 2 , 3 , ∗

1 Department of Industrial Engineering and Operations Research, University of California, Berkeley 2

School of Mathematical Sciences, Peking University

3 Center for Statistical Science, Peking University ziheng\_cheng@berkeley.edu, tianyuxie@pku.edu.cn,

zhangshiyue@stu.pku.edu.cn, chengzhang@math.pku.edu.cn

## Abstract

While conditional diffusion models have achieved remarkable success in various applications, they require abundant data to train from scratch, which is often infeasible in practice. To address this issue, transfer learning has emerged as an essential paradigm in small data regimes. Despite its empirical success, the theoretical underpinnings of transfer learning conditional diffusion models remain unexplored. In this paper, we take the first step towards understanding the sample efficiency of transfer learning conditional diffusion models through the lens of representation learning. Inspired by practical training procedures, we assume that there exists a low-dimensional representation of conditions shared across all tasks. Our analysis shows that with a well-learned representation from source tasks, the sample complexity of target tasks can be reduced substantially. Numerical experiments are also conducted to verify our results.

## 1 Introduction

Conditional diffusion models (CDMs) utilize a user-defined condition to guide the generative process of diffusion models (DMs) to sample from the desired conditional distribution. In recent years, CDMs have achieved groundbreaking success in various generative tasks, including text-to-image generation [Ho et al., 2020, Song et al., 2020, Ho and Salimans, 2022, Rombach et al., 2022], reinforcement learning [Janner et al., 2022, Chi et al., 2023, Wang et al., 2022, Reuss et al., 2023], time series [Tashiro et al., 2021, Rasul et al., 2021], and life science [Song et al., 2021, Watson et al., 2022, Gruver et al., 2024, Guo et al., 2024].

Training a CDM from scratch requires a large amount of data to achieve good generalization. However, in practical scenarios, users often have access to only limited data for the target distribution due to cost or risk concerns, making the model prone to over-fitting. In such small data regime, transfer learning has emerged as a predominant paradigm [Moon et al., 2022, Ruiz et al., 2023, Xie et al., 2023, Han et al., 2023]. By leveraging knowledge acquired during pre-training on large source datasets, transfer learning enhances the performance of fine-tuning on target tasks, facilitating few-shot learning and significantly improving practicality.

Among the successful applications of transfer learning CDMs, the conditions are typically highdimensional vectors with embedded low-dimensional representations (features) that encapsulate all the information required for inference. In addition, these representations are likely to be task-agnostic,

∗ Corresponding Author.

Table 1: Comparing the number of parameters of different parts in CDMs.

| Tasks                              | Backbone Score Network   | Condition Encoder   |
|------------------------------------|--------------------------|---------------------|
| Text-to-Image [Esser et al., 2024] | 2-8B                     | 4.7B                |
| Text-to-Audio [Liu et al., 2024]   | 350-750M                 | 750M                |
| Robotic Control [Chi et al., 2023] | 9M                       | 20-45M              |

enabling effective knowledge transfer. For example, in text-to-image generation, the text input is inherently in high-dimensional space, but contains low-dimensional semantic information such as object attributes, spatial relationships, despite the differences of styles or contents in different image distributions. To take advantage of this structure, condition encoders are often frozen in the fine-tuning stage [Rombach et al., 2022, Esser et al., 2024], which typically constitutes a significant portion of the overall model (see Table 1).

While this paradigm has demonstrated remarkable empirical success, its theoretical underpinnings remain largely unexplored. The following fundamental question is still open:

Can transfer learning CDMs improve the sample efficiency of target tasks by leveraging the representation of conditions learned from source tasks?

There are some recent works attempting to study the theoretical underpinnings of CDMs [Fu et al., 2024, Jiao et al., 2024, Hu et al., 2024], but focus on single task training. Notably, Yang et al. [2024] investigates transfer learning DMs under the assumption that the data is a linear transformation of a low-dimensional latent variable following the same distribution across all tasks. However, fine-tuning merely the data encoder is not a widely adopted training approach in practice.

In this paper, we take the first step towards addressing the above question. Our key assumption is that there exists a generic low-dimensional representation of conditions shared across all distributions. Then we show that, with a well-learned representation from source tasks, the sample complexity of target tasks can be reduced substantially by training only the score network. The main contributions are summarized as follows:

- In Section 3, we establish the first generalization guarantee for transferring score matching error in CDMs, showing that transfer learning can reduce the sample complexity for learning condition encoder in the target task. This is aligned with existing transfer learning theory in supervised learning. Specifically, we present two results in Theorem 3.4 and Theorem 3.6, under the settings of task diversity assumption and meta-learning 2 , respectively. On the technical side, we develop a novel approach to tackle Lipschitz continuity under weaker assumptions on data distribution in Lemma 3.1, which may be of independent interest for the analysis of even single-task diffusion models.
- In Section 4, we provide an end-to-end distribution estimation error bound in transfer learning CDMs. To obtain an L 2 accurate conditional score estimator, we construct a universal approximation theory using deep ReLU neural networks in Theorem 4.1. Then by combining both generalization error and approximation error, Theorem 4.2 and 4.3 provide sample complexity bounds for estimating conditional distribution. Notably, our results are the state of the art even when reduced to single-task learning setting.

In Section A, we further utilize our results to establish statistical guarantees in practical applications of CDMs. In particular, we investigate amortized variational inference (Theorem A.1) and behavior cloning (Theorem A.2), and present guarantees in terms of posterior estimation and optimality gap, laying the theoretical foundations of transfer learning CDMs in practice. We also conduct numerical experiments in Section 5 to verify our results.

2 In practice, the terms such as transfer learning, meta-learning, learning-to-learn, etc. , often refer to the same training paradigm, i.e. , to fine-tune on target tasks with limited data using knowledge from source tasks. However, in the theoretical framework, we use meta-learning to emphasize that target tasks and source tasks are randomly sampled from a meta distribution [Baxter, 2000], whereas in transfer learning, the tasks are fixed.

## 1.1 Related Works

Score Approximation and Distribution Estimation Recently, some works analyze the score approximation theory via deep neural networks and corresponding sample complexity bounds for diffusion models. Oko et al. [2023] considers distributions with density in Besov space and supported on bounded domain. Chen et al. [2023b] assumes the data distribution lies in a low-dimensional linear subspace and obtains improved rates only depending on intrinsic dimension. Fu et al. [2024] studies conditional diffusion models for Hölder densities and Hu et al. [2024] further extends the framework to more advanced neural network architectures, e.g. , diffusion transformers. Wibisono et al. [2024] establishes a minimax optimal rate to estimate Lipschitz score by kernel methods. With an L 2 accurate score estimator, several works provide the convergence rate of discrete samplers for diffusion models [Chen et al., 2022b, 2023a, Lee et al., 2023, Chen et al., 2024]. Combining score matching error and convergence of samplers, one can obtain an end-to-end distribution estimation error bound.

Transfer Learning and Meta-learning Theory in Supervised Learning The remarkable empirical success of transfer learning, meta-learning, and multi-task learning across a wide range of machine learning applications has been accompanied by gradual progress in their theoretical foundations, especially from the perspective of representation learning. To the best of our knowledge, Baxter [2000] is the first theoretical work on meta-learning. It assumes a universal environment to generate tasks with some shared features. Following this setting, Maurer et al. [2016] provides sample complexity bound for general supervised learning problem and Aliakbarpour et al. [2024] studies very few samples per task regime. Another line of research replaces the environment assumption and instead establishes connections between source tasks and target tasks through various notions of task diversity [Tripuraneni et al., 2020, Du et al., 2020, Tripuraneni et al., 2021, Watkins et al., 2023, Chua et al., 2021]. However, theoretical understandings of transfer learning for unsupervised learning are much more limited.

Few-shot Fine-Tuning of Diffusion Models Adapting pre-trained conditional diffusion models to specific tasks with limited data remains a challenge in varied application scenarios. Few-shot fine-tuning aims to bridge this gap by leveraging various techniques to adapt those models to a novel task with minimal data requirements [Ruiz et al., 2023, Giannone et al., 2022]. A promising paradigm is to use transfer (meta) learning by constructing a representation for conditions in all the tasks, which has been widely applied in image generation [Rombach et al., 2022, Ramesh et al., 2022, Sinha et al., 2021], reinforcement learning [He et al., 2023, Ni et al., 2023], inverse problem [Tewari et al., 2023, Chung et al., 2023], etc . Another work Yang et al. [2024] is closely related to this paper, proving that few-shot diffusion models can escape the curse of dimensionality by fine-tuning a linear encoder.

## 2 Preliminaries and Problem Setup

Notations We use x and y to denote the data and conditions, respectively. The blackboard bold letter P represents the joint distribution of ( x, y ) , while the lowercase p denotes its density function. The superscript k indicates the task index, and the subscript i means the sample index. The norm ∥ · ∥ refers to the ℓ 2 -norm for vectors and the spectral norm for matrices. For the hypothesis class F , we use F ⊗ K to refer its K -fold Cartesian product. For any a, b ∈ R , a ∧ b = min { a, b } and a ∨ b = max { a, b } . Finally, we use standard O ( · ) , Ω( · ) to omit constant factors.

## 2.1 Conditional Diffusion Models

Let R d x denote the data space and [0 , 1] D y denote the condition space. Let P be any joint distribution over R d x × [0 , 1] D y with density p and P ( ·| y ) be the conditional distribution with density p ( ·| y ) . As in diffusion models, the forward process is defined as an Ornstein-Uhlenbeck (OU) process, √

<!-- formula-not-decoded -->

where { W t } t ≥ 0 is a standard Wiener process. We denote the distribution of X t as P t ( ·| y ) . Note that the limiting distribution P ∞ ( ·| y ) is a standard Gaussian N (0 , I ) .

To generate new samples, we can reverse the forward process (2.1) from any T &gt; 0 ,

<!-- formula-not-decoded -->

where { W t } 0 ≤ t ≤ T is a time-reversed Wiener process. Unfortunately, we don't have access to the exact conditional score function ∇ log p T -t and need to estimate it through neural networks. For any ( x, y ) ∼ P and score estimator s , define the individual denoising score matching objective [Vincent, 2011] as

<!-- formula-not-decoded -->

where ϕ t ( x t | x ) = N ( x t | α t x, σ 2 t I ) , α t = e -t , σ 2 t = 1 -e -2 t , is the transition kernel of x t | x 0 = x . And the population error of score matching is

<!-- formula-not-decoded -->

Here s P ∗ denotes the true score function and t ∼ Unif ([ T 0 , T ]) . We also define ℓ P ( x, y, s ) := ℓ ( x, y, s ) -ℓ ( x, y, s P ∗ ) . In practice, with a score estimator ̂ s , the generative process is to simulate

<!-- formula-not-decoded -->

Here T 0 &gt; 0 is the early-stopping time. And the distribution of ̂ X ← T -T 0 is written as ̂ P ( ·| y ) .

Note that we don't apply the commonly used classifier-free guidance [Ho and Salimans, 2022] which has a tunable guidance strength since we mainly concentrate on sampling from conditional distribution instead of optimizing other objectives.

## 2.2 Transfer Diffusion Models via Learning Representation

Consider K source distributions over R d x × [0 , 1] D y , P 1 , · · · , P K , and a target distribution P 0 . Suppose that for each source distribution P k , 1 ≤ k ≤ K , we have n i.i.d. samples { ( x k i , y k i ) } n i =1 ∼ P k , and m i.i.d. samples { ( x 0 i , y 0 i ) } m i =1 ∼ P 0 are available for the target distribution, where typically m ≪ n . In transfer (meta) learning setup, we assume there exists a shared nonlinear representation of the condition y for all distributions, i.e. , the conditional distribution P k x | y = P k x | h ∗ ( y ) for some h ∗ : [0 , 1] D y → [0 , 1] d y (see also Assumption 3.2). Note that due to the shared features, the score of p k t ( ·| y ) also has the form of ∇ log p k t ( x t | y ) = f k ∗ ( x t , h ∗ ( y ) , t ) for some f k ∗ .

Similar to Tripuraneni et al. [2020], our transfer learning procedures consist of two phases. In the pre-training phase, the goal is to learn a representation map h ∗ through nK samples from K source distributions. Then during the fine-tuning phase, we learn the target distribution via m new samples and the representation map learned in the pre-training phase.

Formally, let F , H be the hypothesis classes of score networks and representation maps, respectively. Further let F 0 ⊆ F be the hypothesis class of score network in fine-tuning phase. In the pre-training phase, we solve the following Empirical Risk Minimization (ERM),

<!-- formula-not-decoded -->

Then for the fine-tuning task, we solve

<!-- formula-not-decoded -->

Here s f,h ( x, y, t ) := f ( x, h ( y ) , t ) for f : R d x × [0 , 1] d y × [ T 0 , T ] → R d x and h : [0 , 1] D y → [0 , 1] d y and ℓ is defined in (2.3).

In the meta-learning setting, we further assume that all the distributions { P k } k are i.i.d. sampled from a meta distribution P meta. Here P meta can be interpreted as a universal environment [Baxter, 2000, Maurer et al., 2016]. In this case, we posit the existence of a shared representation map that holds for all P ∼ P meta. And the performance benchmark is then defined as the expected error on the target distribution P 0 ∼ P meta .

## 2.3 Deep ReLU Neural Network Family

We use feedforward neural networks to approximate the score function and representation map. Let σ ( x ) := max { x, 0 } be the ReLU activation. Define the score network family NN f ( L, W, M, S, B, R, γ ) := { f ( x, w, t ) = ( A L σ ( · ) + b L ) ◦ · · · ◦ ( A 1 [ x ⊤ , w ⊤ , t ] ⊤ + b 1 ) :

L

<!-- formula-not-decoded -->

S, max ∥ A i ∥ ∞ ∨ ∥ b i ∥ ∞ ≤ B, ∥ f ( x, w, t ) -f ( x, w ′ , t ) ∥ ≤ γ ∥ w -w ′ ∥ ∞ , ∀ ∥ x ∥ ∞ ≤ R,t ≤ T } , and encoder network NN h ( L, W, S, B ) := { h ( y ) = ( A L σ ( · ) + b L ) ◦ · · · ◦ ( A 1 y + b 1 ) :

<!-- formula-not-decoded -->

∥ b i ∥ 0 ) ≤ S, max ∥ A i ∥ ∞ ∨ ∥ b i ∥ ∞ ≤ B } . Throughout this paper, we let F 0 = F = NN f ( L f , W f , M f , S f , B f , R f , γ f ) and H = NN h ( L h , W h , S h , B h ) unless otherwise specified.

Remark 1. In practice, F 0 ⊆ F may (and typically will) depend on ̂ f for parameter efficient fine-tuning (PEFT), e.g., LoRA [Hu et al., 2021]. This will substantially reduce the complexity of F 0 and further improve sample efficiency. The analysis of PEFT is beyond the scope of this paper.

## 3 Statistical Guarantees for Transferring Score Matching Error

In this section, we present our main theoretical results, a statistical theory of transferring the conditional score matching loss. We provide two upper bounds of the score matching loss on target distribution, based on whether task diversity [Tripuraneni et al., 2020] is explicitly assumed. Our analysis introduces novel techniques to address the smoothness properties of the noised data distribution-a challenge that remains nontrivial even in single-task settings. Additionally, we extend the classical theory of local Rademacher complexity to quantify the empirical estimation error.

Throughout this paper, we make the following standard and mild regularity assumptions [Tripuraneni et al., 2020, Chen et al., 2023b] on the initial data distribution P and the representation map h ∗ .

Assumption 3.1 (Sub-gaussian tail) . For any source or target distribution P , P is supported on R d x × [0 , 1] D y and admits a continuous density p ( x, y ) ∈ C 2 ( R d x × [0 , 1] D y ) . Moreover, the conditional distribution p ( x | y ) ≤ C 1 exp( -C 2 ∥ x ∥ 2 ) for some constant C 1 , C 2 .

Assumption 3.2 (Shared low-dimensional representation) . There exists an L -Lipschitz function h ∗ : [0 , 1] D y → [0 , 1] d y with d y ≤ D y , such that for any source and target distribution P , the conditional density p ( x | y ) = g P ∗ ( x, h ∗ ( y )) for some g P ∗ ∈ C 2 ( R d x × [0 , 1] d y ) .

Equivalently, h ∗ ( y ) is a sufficient statistic for x , which indicates that p t ( x | y ) = p t ( x | h ∗ ( y )) . Therefore, with a little abuse of notation, for any w ∈ [0 , 1] d y , we define p ( x ; w ) = p ( x | h ∗ ( y ) = w ) = g P ∗ ( x, w ) . Also note that by definition, for any x, y , we have p ( x ; h ∗ ( y )) = p ( x | h ∗ ( y )) = p ( x | y ) .

Assumption 3.3 (Lipschitz score) . For any source and target distribution P and its density function p , the conditional score ∇ x log p ( x | y ) = ∇ x log g P ∗ ( x, h ∗ ( y )) . The score function ∇ x log g P ∗ ( x, w ) is L -Lipschitz in x and w . And ∥∇ x log g P ∗ (0 , w ) ∥ ≤ B for some constant B and any w .

## 3.1 Tackling Lipschitz Continuity under Weaker Assumptions

Notice that we only impose smoothness assumption on the original data distribution p ( ·| y ) , instead of the entire trajectory p t ( ·| y ) in forward process. This is substantially weaker than the Lipschitzness assumption required in Chen et al. [2023b, 2022b], Yuan et al. [2024], Yang et al. [2024]. However, Lipschitzness of loss function ℓ and class F is a crucial hypothesis in theoretical analysis of transfer learning [Tripuraneni et al., 2020, Chua et al., 2021]. The intuition is that without Lipschitz continuity of the score network f , it is generally impossible to characterize the error from an imperfect representation map h . Hence it is inevitable to show the smoothness of p t ( ·| y ) to some extent.

Fortunately, even with assumptions merely on the initial data distribution, we are still able to prove smoothness of the forward process in any bounded region, as shown in the following lemma. The proof can be found in Appendix B.1.

Lemma 3.1. Under Assumption 3.1, 3.2, 3.3, for any w ∈ [0 , 1] d y , denote the conditional score of forward process ∇ x log p t ( x ; w ) by f ∗ ( x, w, t ) . There exist constants C X , C ′ X , such that for any R &gt; 0 , the function f ∗ ( x, w, t ) is ( C X + C ′ X R 2 ) -Lipschitz in x , ( C X + C ′ X R ) -Lipschitz in w , in the domain B R × [0 , 1] d y × [0 , T ] . Here B R denotes the ball with radius R centered at the origin.

## 3.2 Results under Task Diversity: Sample-Efficient Transfer Learning

In the literature of transfer learning, task diversity is an important assumption that connects target tasks with source tasks [Tripuraneni et al., 2020, Du et al., 2020, Chua et al., 2021]. In the context of conditional diffusion models, we state the formal definition as follows.

Definition 3.1 (Task diversity) . Given hypothesis classes F , H , we say the source distributions P 1 , · · · , P K are ( ν, ∆) -diverse over target distribution P 0 , if for any representation h ∈ H ,

<!-- formula-not-decoded -->

Here L P is defined in (2.4). This notion of diversity ensures that the representation error on the target task caused by ̂ h can be controlled by the error on the source tasks, thereby establishing certain relationships in between. More detailed discussions are deferred to Appendix B.5.

We first present the generalization guarantee for each phase respectively.

Proposition 3.2 (Fine-tuning phase generalization) . Under Assumption 3.1, 3.2, 3.3, for any ̂ h ∈ H , the population loss of ̂ f 0 can be bounded by

<!-- formula-not-decoded -->

where r x = log ˜ N F m and log ˜ N F is some complexity measures of F .

Proposition 3.3 (Pre-training phase generalization) . Under Assumption 3.1, 3.2, 3.3, if R f ≳ log 1 2 ( nKM f /δ ) , with probability no less than 1 -δ , the population loss can be bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining these two propositions with the notion of task diversity in Definition 3.1, we are able to show the statistical rate of transfer learning as follows.

Theorem 3.4. Under Assumption 3.1, 3.2, 3.3, suppose P 1 , · · · , P K are ( ν, ∆) -diverse over target distribution P 0 given F , H . If R f ≳ log 1 2 ( nKM f /δ ) , then with probability no less than 1 -δ ,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The formal statements and proofs are provided in Appendix B.2.

Let ε approx = inf h ∈H 1 K K ∑ k =1 inf f ∈F E ( x,y ) ∼ P k [ ℓ P k ( x, y, s f,h )] be the approximation error. The lead- ing terms can be simplified to ˜ O ( ε approx + K log N F +log N H nK + log N F m ) , where log N F and log N H capture the complexity of the hypothesis classes.

Improving Sample Efficiency Theorem 3.4 demonstrates the sample efficiency of transfer learning. Compared to naively training the full CDM for target distribution, which has an error of ˜ O ( ε approx + log N F +log N H m ) , transfer learning saves the complexity of learning H and thus the performance is much better when m is relatively small to n, K ( i.e. , in few-shot learning setting).

## 3.3 Results without Task Diversity: Meta-Learning Perspective

The results in previous section heavily depend on the task diversity assumption, which is hard to verify in practice. An alternative is to consider meta-learning setting, where all source and target distributions are sampled from the same environment , i.e. , a meta distribution.

For any h ∈ C ([0 , 1] D y ; [0 , 1] d y ) and distribution P over R d x × [0 , 1] D y , define the representation error as

<!-- formula-not-decoded -->

We characterize the generalization bound of source tasks on the entire meta distribution as follows.

Proposition 3.5 (Generalization on meta distribution) . Under Assumption 3.1, 3.2, 3.3, there exists constant C P such that for { P k } K k =1 i.i.d. ∼ P meta, with probability no less than 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds for any h ∈ H , where r P = M 2 f exp( -Ω( R 2 f )) + S h L h log ( KL h W h ( B h ∨ 1) M f γ f ) K .

Theorem 3.6. Under Assumption 3.1, 3.2, 3.3, if R f ≳ log 1 2 ( nKM f /δ ) , then with probability no less than 1 -δ , the expected population loss of new task can be bounded by

<!-- formula-not-decoded -->

where log N F , log N H are defined in (3.5).

The formal statements and proofs are provided in Appendix B.3.

Let ˜ ε approx = inf h ∈H E P ∼ P meta inf f ∈F E ( x,y ) ∼ P [ ℓ P ( x, y, s f,h )] be the approximation error in meta-learning.

The results above can be further simplified to ˜ O ( ˜ ε approx + log N F m ∧ n + log N H K ) . Different from transfer learning bound in Theorem 3.4, the leading term decays only in K and not in n . This is because that without task diversity assumption, the connection between source distributions and target distributions can only be constructed through meta distribution. And according to Proposition 3.5, the source distributions P 1 , · · · , P K collectively form a K -shot empirical estimation of P meta, leading to an estimation error of O (1 /K ) . Despite this, Theorem 3.6 still demonstrates the sample efficiency of meta-learning compared to naive training method when m is small and n, K are sufficient large.

## 4 End-to-End Distribution Estimation via Deep Neural Network

Section 3 provides a statistical guarantee for transferring score matching. In this section, we establish an approximation theory using deep neural network to quantify the misspecification error. Combining both results we are able to obtain an end-to-end distribution estimation error bound for transfer learning diffusion models.

## 4.1 Score Neural Network Approximation

The following theorem provides a guarantee for the ability of deep ReLU neural networks to approximate score and representation. The proof is provided in Appendix C.1.

Theorem 4.1. Under Assumption 3.1, 3.2, 3.3, to achieve R f ≳ log 1 2 ( nKM f /δ ) and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the configuration of F and H should satisfy

<!-- formula-not-decoded -->

L h = O (log(1 /ε )) , W h = O ( ε -D y log(1 /ε ) ) , S h = O ( ε -D y log 2 (1 /ε ) ) , B h = O (1) . (4.4) Here O ( · ) hides all the polynomial factors of d x , d y , D y , C 1 , C 2 , L, B .

Universal approximation of deep ReLU neural networks in a bounded region has been widely studied [Yarotsky, 2017, Schmidt-Hieber, 2020]. However, we have to deal with an unbounded domain here, hence more refined analysis is required, e.g. truncation arguments.

In addition, traditional approximation theories typically cannot provide Lipschitz continuity guarantees, which is crucial in transfer learning analysis. Following the constructions in Chen et al. [2023b], the Lipschitzness restriction doesn't compromise the approximation ability of neural networks, while ensuring validity of the generalization analysis in Section 3.

## 4.2 Distribution Estimation Error Bound

Given the approximation and generalization results, we are in the position of bounding the distribution estimation error of our transfer (meta) learning procedures. The formal statements and proofs can be found in Appendix C.2.

Theorem 4.2 (Transfer learning) . Under Assumption 3.1, 3.2, 3.3 and ( ν, ∆) -diversity with proper configuration of neural network family and T, T 0 , it holds that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Theorem 4.3 (Meta-learning) . Under Assumption 3.1, 3.2, 3.3 and meta-learning setting, with proper configuration of neural network family and T, T 0 , it holds that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Table 2: MSEs for β 0 = 5 . 5 .

| m                  |    10 |    20 |   30 |   40 |   50 |   100 |
|--------------------|-------|-------|------|------|------|-------|
| fine-tuning        | 14.47 |  3.68 | 2.45 | 1.82 | 1.9  |  0.91 |
| train-from-scratch | 21.99 | 10.61 | 5.71 | 2.38 | 1.77 |  1.04 |

Table 3: MSEs for β 0 = 15 .

| m                  |    10 |    20 |    30 |    40 |   50 |   100 |
|--------------------|-------|-------|-------|-------|------|-------|
| fine-tuning        |  6.14 |  2.65 |  1.61 |  1.08 | 0.96 |  0.45 |
| train-from-scratch | 24.41 | 20.62 | 18.67 | 13.49 | 7.03 |  1.23 |

Theorem 4.2 and 4.3 again unveil the benefits of transfer (meta) learning for conditional diffusion models, with a rate of ˜ O (( m ∧ n ) -1 dx + dy +9 +( nK ) -1 Dy +2 ) or ˜ O (( m ∧ n ) -1 dx + dy +9 + K -1 Dy +2 ) To compare, naively learning the target distribution in isolation will yield ˜ O ( m -1 dx + Dy +9 ) . When the condition dimension D y is much larger than feature dimension d y , transfer (meta) learning can substantially improve sample efficiency on target tasks, thanks to representation learning.

.

Comparison with Existing Complexity Bounds of CDMs Fu et al. [2024] studies conditional diffusion model for sub-gaussian distributions with β -Hölder density. Since the Lipschitzness of score is analogous to the requirement of twice differentiability of density [Wibisono et al., 2024], it is reasonable to let β = 2 for a fair comparison. In this case, the TV distance is bounded by ˜ O ( m -1 2( dx + Dy +2) ) with sample size m according to Fu et al. [2024], which is worse than our naive bound ˜ O ( m -1 dx + Dy +9 ) due to the inefficiency of score approximation. We are also aware of another work [Jiao et al., 2024] that assumes Lipschitz density and score, obtaining a rate of ˜ O ( m -1 2( dx +3)( dx + Dy +3) ) .

Relation to Yang et al. [2024] Unlike our setup, Yang et al. [2024] considers transfer learning unconditional diffusion models with only one source task, i.e. , D y = d y = 0 , K = 1 . The unconditional distribution is assumed to be supported in a low-dimensional linear subspace, where the source task and the target task have the same latent variable distribution. Hence, only a linear encoder is trained for fine-tuning instead of the full score network. In this case, Yang et al. [2024] is able to bound the TV distance by ˜ O ( m -1 4 + n -1 -α ( n ) dx +5 ) , escaping the curse of dimensionality for target task. However, the assumption on shared latent variable distribution is stringent and we believe our analysis methods can be extended to this setting as well.

## 5 Experiments

Our theoretical results can be readily applied in various real world settings. In Appendix A, we investigate amortized variational inference and behavior cloning utilizing our theories, providing statistical guarantees of practical applications of CDMs. In addition, we conduct experiments on both synthetic and real world data to numerically verify the sample efficiency of transfer learning.

Conditioned Diffusion The first numerical example is the high-dimensional conditioned diffusion [Cui et al., 2016, Yu et al., 2023] arising from the following Langevin SDE

<!-- formula-not-decoded -->

where β &gt; 0 and w s is a one-dimensional standard Brownian motion. The SDE (5.1) is discretized by the Euler-Maruyama scheme with a step size of 0 . 02 , which defines the prior distribution p β ( x ) for the (discretized) trajectory x = ( u 0 . 02 , u 0 . 04 , . . . , u 1 . 00 ) ⊤ ∈ R 50 . We consider a conditional Gaussian likelihood function, p ( y | x ) = N ( Mx,I 100 / 4) , where M ∈ R 100 × 50 is a pre-defined projection matrix. Given a set of pre-selected { β k ; 1 ≤ k ≤ K } with β k = k and K = 10 , the k -th joint source distribution is given by P k ( x, y ) = p β k ( x ) p ( y | x ) . The target distribution P 0 ( x, y ) is given by β 0 = 5 . 5 (in-domain) or β 0 = 15 (out-of-domain). More details are found in Appendix E.1.

We report the MSEs of the estimated posterior mean of P 0 ( x | y ) on the test samples in Table 2 and 3. We see that across different values of β and m , the fine-tuned models can provide significantly more accurate posterior mean estimations in most cases, suggesting the effectiveness of the representation map ̂ h learned in the pre-training phase. Notably, as the number of fine-tuning samples m increases, the performance gaps between fine-tuned models and train-from-scratch models get smaller, since more training samples yield more generalization benefits and thus less dependence on the pre-trained

Table 4: MSEs on the image restoration task.

| m                  |     10 |     20 |     30 |     40 |     50 |    100 |
|--------------------|--------|--------|--------|--------|--------|--------|
| fine-tuning        | 0.3799 | 0.2846 | 0.2544 | 0.2406 | 0.2404 | 0.2268 |
| train-from-scratch | 0.4409 | 0.318  | 0.2746 | 0.2551 | 0.2501 | 0.2344 |

model. This is aligned with our theoretical results. We also notice a large variance among the results of different replicates, and attribute the slightly worse performance of fine-tuned models at m = 50 , β 0 = 5 . 5 to the potential randomness.

Image Restoration For a real data experiment, we consider the image restoration task on MNIST. We have K = 9 source tasks with /a80 k ( x, y ) = p k ( x ) p ( y | x ) , where the prior p k ( x ) is the data distribution of the digit k in the MNIST data set ( 1 ≤ k ≤ K ) and p ( y | x ) = N ( x, I 784 / 4) . The target task is /a80 0 ( x, y ) = p 0 ( x ) p ( y | x ) , where p 0 ( x ) is the data distribution of the digit 0. We use the full MNIST 1-9 data for pre-training which corresponds to n = 5000 . For the finetuning phase, we consider m = 10 , 20 , 30 , 40 , 50 , 100 training samples and 100 test samples from /a80 0 ( x, y ) . More details can be found in Appendix E.2.

We report the MSEs between estimated posterior mean of /a80 0 ( x, y ) = p 0 ( x ) p ( y | x ) and the ground truth sample x on the test samples in Table 4. We see that for all fine-tuning sample sizes m , the results obtained by fine-tuning consistently outperform those obtained by training from scratch, indicating the benefits of transfer learning. Similarly to the experiment on conditioned diffusion, we also observe a reduced performance gap as m increases.

## 6 Conclusion and Discussion

In this paper, we take the first step towards understanding the sample efficiency of transfer learning conditional diffusion models from the perspective of representation learning. We provide a generalization guarantee for transferring score matching in CDMs in different settings. We further establish an end-to-end distribution estimation error bound using deep neural networks. Two practical applications are investigated based on our theoretical results. We hope this work can motivate future theoretical study on the popular transfer learning paradigm in generative AIs.

Although this work provides the first statistical guarantee for transfer learning in CDMs, it has several limitations that we plan to address in future research. First, our theoretical results heavily rely on the task diversity notion introduced in Section 3.1, which can be challenging to verify in practice. While we provide some preliminary empirical evidence in Appendix B.5, a more fine-grained theoretical and empirical analysis will be essential for a deeper understanding of CDMs. Second, our analysis focuses on the ERM estimator, whereas in practice, fine-tuning typically starts from a pre-trained model and may employ techniques such as LoRA. Incorporating these settings would allow for an optimization-based perspective on the sample efficiency of transfer learning. Finally, in our current formulation, the sample efficiency gain arises from reducing the complexity associated with learning the conditional encoder. Consequently, our results primarily apply to CDMs in which the conditional encoder constitutes a substantial part of the overall model. Extending the theory to settings where this assumption does not hold is an important direction for future work.

## Acknowledgements

This work was supported by National Natural Science Foundation of China (grant no. 12201014, grant no. 12292980 and grant no. 12292983). The research of Cheng Zhang was support in part by National Engineering Laboratory for Big Data Analysis and Applications, the Key Laboratory of Mathematics and Its Applications (LMAM) and the Key Laboratory of Mathematical Economics and Quantitative Finance (LMEQF) of Peking University. The authors are grateful for the computational resources provided by the High-performance Computing Platform of Peking University. The authors appreciate the anonymous NeurIPS reviewers for their constructive feedback.

## References

- Anurag Ajay, Yilun Du, Abhi Gupta, Joshua Tenenbaum, Tommi Jaakkola, and Pulkit Agrawal. Is conditional generative modeling all you need for decision-making? arXiv preprint arXiv:2211.15657 , 2022.
- Maryam Aliakbarpour, Konstantina Bairaktari, Gavin Brown, Adam Smith, Nathan Srebro, and Jonathan Ullman. Metalearning with very few samples per task. In The Thirty Seventh Annual Conference on Learning Theory , pages 46-93. PMLR, 2024.
- Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications , 12(3):313-326, 1982.
- Jonathan Baxter. A model of inductive bias learning. Journal of artificial intelligence research , 12: 149-198, 2000.
- Olivier Bousquet. Concentration inequalities and empirical processes theory applied to the analysis of learning algorithms . PhD thesis, École Polytechnique: Department of Applied Mathematics Paris, France, 2002.
- Hongrui Chen, Holden Lee, and Jianfeng Lu. Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. In International Conference on Machine Learning , pages 4735-4763. PMLR, 2023a.
- Minshuo Chen, Wenjing Liao, Hongyuan Zha, and Tuo Zhao. Distribution approximation and statistical estimation guarantees of generative adversarial networks. arXiv preprint arXiv:2002.03938 , 2020.
- Minshuo Chen, Haoming Jiang, Wenjing Liao, and Tuo Zhao. Nonparametric regression on lowdimensional manifolds using deep relu networks: Function approximation and statistical recovery. Information and Inference: A Journal of the IMA , 11(4):1203-1253, 2022a.
- Minshuo Chen, Kaixuan Huang, Tuo Zhao, and Mengdi Wang. Score approximation, estimation and distribution recovery of diffusion models on low-dimensional data. In International Conference on Machine Learning , pages 4672-4712. PMLR, 2023b.
- Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. arXiv preprint arXiv:2209.11215 , 2022b.
- Sitan Chen, Sinho Chewi, Holden Lee, Yuanzhi Li, Jianfeng Lu, and Adil Salim. The probability flow ode is provably fast. Advances in Neural Information Processing Systems , 36, 2024.
- Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research , page 02783649241273668, 2023.
- Kurtland Chua, Qi Lei, and Jason D Lee. How fine-tuning allows for effective meta-learning. Advances in Neural Information Processing Systems , 34:8871-8884, 2021.
- Hyungjin Chung, Dohoon Ryu, Michael T McCann, Marc L Klasky, and Jong Chul Ye. Solving 3d inverse problems using pre-trained 2d diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 22542-22551, 2023.
- Pierre Comon. Independent component analysis, a new concept? Signal processing , 36(3):287-314, 1994.
- Tiangang Cui, Kody JH Law, and Youssef M Marzouk. Dimension-independent likelihood-informed mcmc. Journal of Computational Physics , 304:109-137, 2016.
- Simon S Du, Wei Hu, Sham M Kakade, Jason D Lee, and Qi Lei. Few-shot learning via learning the representation, provably. arXiv preprint arXiv:2002.09434 , 2020.

- Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning , 2024.
- Hengyu Fu, Zhuoran Yang, Mengdi Wang, and Minshuo Chen. Unveil conditional diffusion models with classifier-free guidance: A sharp statistical theory. arXiv preprint arXiv:2403.11968 , 2024.
- Giorgio Giannone, Didrik Nielsen, and Ole Winther. Few-shot diffusion models. arXiv preprint arXiv:2205.15463 , 2022.
- Nate Gruver, Samuel Stanton, Nathan Frey, Tim GJ Rudner, Isidro Hotzel, Julien Lafrance-Vanasse, Arvind Rajpal, Kyunghyun Cho, and Andrew G Wilson. Protein design with guided discrete diffusion. Advances in neural information processing systems , 36, 2024.
- Zhiye Guo, Jian Liu, Yanli Wang, Mengrui Chen, Duolin Wang, Dong Xu, and Jianlin Cheng. Diffusion models in bioinformatics and computational biology. Nature reviews bioengineering , 2 (2):136-154, 2024.
- Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 7323-7334, 2023.
- Haoran He, Chenjia Bai, Kang Xu, Zhuoran Yang, Weinan Zhang, Dong Wang, Bin Zhao, and Xuelong Li. Diffusion model is an effective planner and data synthesizer for multi-task reinforcement learning. Advances in neural information processing systems , 36:64896-64917, 2023.
- Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 , 2022.
- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems , 33:6840-6851, 2020.
- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 , 2021.
- Jerry Yao-Chieh Hu, Weimin Wu, Yi-Chen Lee, Yu-Chao Huang, Minshuo Chen, and Han Liu. On statistical rates of conditional diffusion transformers: Approximation, estimation and minimax optimality. arXiv preprint arXiv:2411.17522 , 2024.
- Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. arXiv preprint arXiv:2205.09991 , 2022.
- Yuling Jiao, Lican Kang, Jin Liu, Heng Peng, and Heng Zuo. Model free prediction with uncertainty assessment. arXiv preprint arXiv:2405.12684 , 2024.
- Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 , 2013.
- Holden Lee, Jianfeng Lu, and Yixin Tan. Convergence of score-based generative modeling for general data distributions. In International Conference on Algorithmic Learning Theory , pages 946-985. PMLR, 2023.
- Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Qiao Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D Plumbley. Audioldm 2: Learning holistic audio generation with self-supervised pretraining. IEEE/ACM Transactions on Audio, Speech, and Language Processing , 2024.
- Pascal Massart. About the constants in talagrand's concentration inequalities for empirical processes. The Annals of Probability , 28(2):863-884, 2000.
- Andreas Maurer, Massimiliano Pontil, and Bernardino Romera-Paredes. The benefit of multitask representation learning. Journal of Machine Learning Research , 17(81):1-32, 2016.

- Taehong Moon, Moonseok Choi, Gayoung Lee, Jung-Woo Ha, and Juho Lee. Fine-tuning diffusion models with limited data. In NeurIPS 2022 Workshop on Score-Based Methods , 2022.
- Fei Ni, Jianye Hao, Yao Mu, Yifu Yuan, Yan Zheng, Bin Wang, and Zhixuan Liang. Metadiffuser: Diffusion model as conditional planner for offline meta-rl. In International Conference on Machine Learning , pages 26087-26105. PMLR, 2023.
- Kazusato Oko, Shunta Akiyama, and Taiji Suzuki. Diffusion models are minimax optimal distribution estimators. In International Conference on Machine Learning , pages 26517-26582. PMLR, 2023.
- Vitchyr H Pong, Ashvin V Nair, Laura M Smith, Catherine Huang, and Sergey Levine. Offline meta-reinforcement learning with online self-supervision. In International Conference on Machine Learning , pages 17811-17829. PMLR, 2022.
- Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125 , 1(2):3, 2022.
- Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In International Conference on Machine Learning , pages 8857-8868. PMLR, 2021.
- Moritz Reuss, Maximilian Li, Xiaogang Jia, and Rudolf Lioutikov. Goal-conditioned imitation learning using score-based diffusion policies. arXiv preprint arXiv:2304.02532 , 2023.
- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 10684-10695, 2022.
- Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 22500-22510, 2023.
- Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with ReLU activation function. The Annals of Statistics , 48(4):1875 - 1897, 2020. doi: 10.1214/19-AOS1875. URL https://doi.org/10.1214/19-AOS1875 .
- Abhishek Sinha, Jiaming Song, Chenlin Meng, and Stefano Ermon. D2c: Diffusion-decoding models for few-shot conditional generation. Advances in Neural Information Processing Systems , 34: 12533-12548, 2021.
- Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 , 2020.
- Yang Song, Liyue Shen, Lei Xing, and Stefano Ermon. Solving inverse problems in medical imaging with score-based generative models. arXiv preprint arXiv:2111.08005 , 2021.
- Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. Csdi: Conditional score-based diffusion models for probabilistic time series imputation. Advances in Neural Information Processing Systems , 34:24804-24816, 2021.
- Ayush Tewari, Tianwei Yin, George Cazenavette, Semon Rezchikov, Josh Tenenbaum, Frédo Durand, Bill Freeman, and Vincent Sitzmann. Diffusion with forward models: Solving stochastic inverse problems without direct supervision. Advances in Neural Information Processing Systems , 36: 12349-12362, 2023.
- Nilesh Tripuraneni, Michael Jordan, and Chi Jin. On the theory of transfer learning: The importance of task diversity. Advances in neural information processing systems , 33:7852-7862, 2020.
- Nilesh Tripuraneni, Chi Jin, and Michael Jordan. Provable meta-learning of linear representations. In International Conference on Machine Learning , pages 10434-10443. PMLR, 2021.
- Ramon Van Handel. Probability in high dimension. Lecture Notes (Princeton University) , 2(3):2-3, 2014.

- Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation , 23(7):1661-1674, 2011.
- Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint , volume 48. Cambridge university press, 2019.
- Zhendong Wang, Jonathan J Hunt, and Mingyuan Zhou. Diffusion policies as an expressive policy class for offline reinforcement learning. arXiv preprint arXiv:2208.06193 , 2022.
- Austin Watkins, Enayat Ullah, Thanh Nguyen-Tang, and Raman Arora. Optimistic rates for multi-task representation learning. Advances in Neural Information Processing Systems , 36:2207-2251, 2023.
- Joseph L Watson, David Juergens, Nathaniel R Bennett, Brian L Trippe, Jason Yim, Helen E Eisenach, Woody Ahern, Andrew J Borst, Robert J Ragotte, Lukas F Milles, et al. Broadly applicable and accurate protein design by integrating structure prediction networks and diffusion generative models. BioRxiv , pages 2022-12, 2022.
- Andre Wibisono, Yihong Wu, and Kaylee Yingxi Yang. Optimal score estimation via empirical bayes smoothing. arXiv preprint arXiv:2402.07747 , 2024.
- Enze Xie, Lewei Yao, Han Shi, Zhili Liu, Daquan Zhou, Zhaoqiang Liu, Jiawei Li, and Zhenguo Li. Difffit: Unlocking transferability of large diffusion models via simple parameter-efficient fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 4230-4239, 2023.
- Ruofeng Yang, Bo Jiang, Cheng Chen, Ruinan Jin, Baoxiang Wang, and Shuai Li. Few-shot diffusion models escape the curse of dimensionality. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview.net/forum?id= JrraNaaZm5 .
- Dmitry Yarotsky. Error bounds for approximations with deep relu networks. Neural networks , 94: 103-114, 2017.
- Longlin Yu, Tianyu Xie, Yu Zhu, Tong Yang, Xiangyu Zhang, and Cheng Zhang. Hierarchical semi-implicit variational inference with application to diffusion model acceleration. Advances in Neural Information Processing Systems , 36, 2023.
- Hui Yuan, Kaixuan Huang, Chengzhuo Ni, Minshuo Chen, and Mengdi Wang. Reward-directed conditional diffusion: Provable distribution estimation and reward improvement. Advances in Neural Information Processing Systems , 36, 2024.

## A Applications

We explore two applications of transfer learning for conditional diffusion models, supported by theoretical guarantees derived from our earlier results. In particular, we study amortized variational inference and behavior cloning. These real-world use cases not only validate the applicability of our theoretical findings but also lay the foundations of transferring diffusion models in practice.

## A.1 Amortized Variational Inference

Diffusion models have exhibited groundbreaking success in probabilistic inference, especially latent variable models. We study a simple amortized variational inference model, where the observation y given latent variable x is distributed according to an exponential family F Ψ with density

<!-- formula-not-decoded -->

where ψ ∈ Ψ is non-negative and supported on [0 , 1] D and h ∗ ( y ) ∈ [0 , 1] d . Note that we also have d x = d in this case. The prior distribution of variable x is denoted as p ϕ for some ϕ ∈ Φ . Let θ = ( ψ, ϕ ) and we aim to sample from the posterior distribution of p θ ( x | y ) ∝ p ϕ ( x ) p ψ ( y | x ) ∝ p ϕ ( x ) exp( ⟨ x, h ∗ ( y ) ⟩ -A ψ ( x )) . Due to the special structure, the posterior p θ ( x | y ) only depends on the low-dimensional feature h ∗ ( y ) , shared across all θ ∈ Θ := Ψ × Φ . This formulation encompasses various applications including independent component analysis [Comon, 1994], inverse problem [Song et al., 2021, Ajay et al., 2022] and variational Bayesian inference [Kingma, 2013].

Consider source tasks consisting of θ 1 , · · · , θ K ∈ Θ , and for each θ k we have n i.i.d. samples { ( x k i , y k i ) } n i =1 . For the target task θ 0 , we only have m samples { ( x 0 i , y 0 i ) } m i =1 . We conduct our transfer learning procedures to train a conditional diffusion models ̂ P θ 0 ( ·| y ) . For theoretical analysis, we further impose some assumptions on the probabilistic model as follows.

Assumption A.1. The prior distribution satisfies p ϕ ( x ) ≤ C 1 exp( -C 2 ∥ x ∥ 2 ) and ∇ x log p ϕ ( x ) is L -Lipshcitz in x , ∥∇ x log p ϕ (0) ∥ ≤ B for any ϕ ∈ Φ . The representation h ∗ is L -Lipschitz. The integral ∫ ψ ( y )d y ∈ [1 /C, C ] for any ψ ∈ Ψ .

Theorem A.1. Suppose Assumption A.1 holds. Then under meta-learning setting, we have with probability no less than 1 -δ ,

<!-- formula-not-decoded -->

If ( ν, ∆) -diversity holds, then we have with probability no less than 1 -δ ,

<!-- formula-not-decoded -->

The proof is deferred to Appendix D.1. We show that under mild assumptions, transfer (meta) learning diffusion models can improve the sample efficiency for target task in the context of amortized variational inference. This error bound can be further extended to establish guarantees for statistical inference such as moment prediction, uncertainty assessment, etc .

## A.2 Behavior Cloning via Meta-Diffusion Policy

Although originally developed for image generation tasks, diffusion models have recently been extended to reinforcement learning (RL) [Janner et al., 2022, Chi et al., 2023, Wang et al., 2022], enabling the modeling of complex distributions of dynamics and policies. In the context of meta-RL, some works have further utilized diffusion models for planning and synthesis tasks [Ni et al., 2023, He et al., 2023]. In this application, we focus on a popular framework of behavior cloning, diffusion policy [Chi et al., 2023], which uses conditional diffusion models to learn multi-modal expert policies in high-dimensional state spaces. In such settings, the state often corresponds to visual observations of the robot's surroundings, such as high resolution images, and thus typically share a low-dimensional underlying representation.

Let M be the space of decision-making environments, where each M ∈ M is an infinite horizon Markov Decision Process (MDP) sharing the same state space S , action space A , discount factor γ and initial distribution ρ ∈ ∆( S ) . And each M ∈ M has its own transition kernel T M : S × A → ∆( S ) , and reward function r M : S × A → [0 , 1] . The policy is defined as a map π : S → ∆( A ) . The value function of MDP M under policy π is

<!-- formula-not-decoded -->

Denote the visitation measure as d π M ( s, a ) := (1 -γ ) E s 0 ∼ ρ ∞ ∑ t =0 γ t P ( s t = s | π, s 0 ) π ( a | s ) .

Suppose there are K source tasks M 1 , · · · , M K ∈ M , and the expert policy of each task is denoted as π k ∗ . In behavior cloning, for each source task M k , we have n pairs of { ( s k i , a k i ) } n i =1 i.i.d. ∼ d k ∗ := d π k ∗ M k . The goal is to imitate the expert policy of target task M 0 ∈ M , of which the sample size is only m ≪ n .

To unify the notation, let x = a, y = s and assume A = R d a , S = [0 , 1] D s and representation space [0 , 1] d s . Our meta diffusion-policy framework aims to learn a state encoder h : S → [0 , 1] d s during pre-training, which acts as a shared representation map in different MDPs and consequently enhances sample efficiency on fine-tuning tasks. Let ̂ π 0 be the learned policy in fine-tuning phase. The following theorem shows the optimality gap between the learned policy and the expert policy.

Theorem A.2. Suppose the expert policy π k ∗ satisfies Assumption 3.1, 3.2, 3.3. Then under metalearning setting, it holds that with probability no less than 1 -δ ,

<!-- formula-not-decoded -->

If we further assume π 1 ∗ , · · · , π K ∗ are ( ν, ∆) -diverse over π 0 ∗ , then the gap can be improved by

<!-- formula-not-decoded -->

The proof can be found in Appendix D.2. This provides the first statistical guarantee of diffusion policy in behavior cloning. Notably, in both cases, the number of source tasks K has an exponential dependence on D s , further suggesting the importance of data coverage when tackling distribution shift in offline meta-RL [Pong et al., 2022].

## B Proofs in Section 3

## B.1 Preliminaries

Lemma B.1. If x 0 ∼ p ( x 0 | y ) , the density of forward process p t ( x | y ) can be written as

<!-- formula-not-decoded -->

Besides, the score function has the form of

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. (B.1) can be directly implied by the definition of forward process. And it yields

<!-- formula-not-decoded -->

which is (B.2). Moreover, noticing that ∇ x ϕ t ( x | x 0 ) = -1 α t ∇ x 0 ϕ t ( x | x 0 ) , then by integration by parts,

<!-- formula-not-decoded -->

Hence (B.3) is proved.

Lemma B.2. [Lem. 3.1] For any w ∈ [0 , 1] d y , denote the conditional score of forward process ∇ x log p t ( x ; w ) by f ∗ ( x, w, t ) . Then there exist constants C X , C ′ X , such that for any R &gt; 0 , the function f ∗ ( x, w, t ) is ( C X + C ′ X R 2 ) -Lipschitz in x , ( C X + C ′ X R ) -Lipschitz in w , in the domain B R × [0 , 1] d y × [0 , T ] . Here B R denotes the ball with radius R centered at the origin.

Proof. Define density function q t ( x 0 | x, w ) ∝ ϕ t ( x | x 0 ) p ( x 0 ; w ) . Our proof strategy will depend on whether t ≥ 1 2( L +1) .

When t ≥ 1 2( L +1) , according to (B.2), we have

<!-- formula-not-decoded -->

For any R &gt; 0 , we have

<!-- formula-not-decoded -->

Let R = 2 ∥ x ∥ +2 C 0 σ t , then the domain { x 0 : ∥ α t x 0 -x σ t ∥ ≤ R/ 2 } includes { x 0 : ∥ x 0 ∥ ≤ C 0 } , indicating

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Similarly, for w we have

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that when σ 2 t ≤ α 2 t 2 L , the distribution q t ( x 0 | x, w ) ∝ exp ( -∥ α t x 0 -x ∥ 2 2 σ 2 t ) p ( x 0 ; w ) is L -strongly log-concave, and thus satisfies the Poincare inequality with a constant L -1 [Chen et al., 2023a],

Var q t ( x 0 | x,w ) ( ∇ x log p ( x 0 ; w ) ) ⪯ L -1 E [ ∇ 2 x log p ( x 0 ; w )( ∇ 2 x log p ( x 0 ; w )) ⊤ ] ≤ L. (B.13) And thus

Analogously,

<!-- formula-not-decoded -->

Combine all the arguments in (B.9),(B.11),(B.14),(B.15) and we complete the proof.

Lemma B.3 (Lemma 7, Chen et al. [2022a]) . The covering number of F = NN f ( L f , W f , M f , S f , B f , R f , γ f ) can be bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The covering number of H = NN h ( L h , W h , S h , B h ) can be bounded by

<!-- formula-not-decoded -->

## B.2 Proofs of Transfer Learning

Proposition B.4 (Prop. 3.2) . Under Assumption 3.1, 3.2, 3.3, there exists some constant C xy such that the following holds. For any h ∈ H and ( x 1 , y 1 ) , · · · , ( x m , y m ) i.i.d. ∼ P , define the empirical minimizer

<!-- formula-not-decoded -->

The population loss of ̂ f can be bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Consider the truncated function class defined on R d x × [0 , 1] D y ,

Φ = { ( x, y ) ↦→ ˜ ℓ ( x, y, f ) := ( ℓ ( x, y, s f,h ) -ℓ ( x, y, s P ∗ )) · ✶ ∥ x ∥ ∞ ≤ R : f ∈ F} , (B.20) where the truncation radius R ≥ 1 will be defined later. It is easy to show that with probability no less than 1 -2 m exp( -C ′ 1 R 2 ) , it holds that ∥ x i ∥ ∞ ≤ R for all 1 ≤ i ≤ m . Hence by definition, the empirical minimizer also satisfies ̂ f = arg min f ∈F 1 m m ∑ i =1 ˜ ℓ ( x i , y i , f ) . Below we reason conditioned on this event and verify the conditions required in Lemma B.11.

Step 1. To bound the individual loss,

<!-- formula-not-decoded -->

And by Lemma B.10,

<!-- formula-not-decoded -->

Step 2. To bound the second order moment, we have

<!-- formula-not-decoded -->

Step 3. To bound the local Rademacher complexity, note that

<!-- formula-not-decoded -->

easy to show that diam ( Φ r , ∥ · ∥ L 2 ( ̂ P m ) ) ≤ 2 r . By Dudley's bound [Van Handel, 2014, Wainwright, 2019], there exists an absolute constant C 0 such that for any θ &gt; 0 , where ̂ P m := 1 m m ∑ i =1 δ ( x i ,y i ) . Define Φ r := { φ ∈ Φ : 1 m m ∑ i =1 φ ( x i , y i ) 2 ≤ r } and it is √

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let R 1 = 2 R . Since x t | x i ∼ N ( x t ; α t x i , σ 2 t I ) , we have P ( ∥ x t ∥ ∞ ≥ R 1 ) ≤ d x P ( |N (0 , 1) | ≤ R ) ≤ 2 d x exp( -C ′ 0 R 2 ) for some absolute constant C ′ 0 . Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any ε ≥ 16 d 1 2 x M exp( -C ′ 0 R 2 / 2) , according to B.3,

<!-- formula-not-decoded -->

Plug in (B.25) and let θ = 16 d 1 2 x M exp( -C ′ 0 R 2 / 2) ,

<!-- formula-not-decoded -->

Combine the three steps above, by Lemma B.11 with B 0 = 8 M 2 exp( -C ′ 1 R 2 ) , B = 4 M,b = M , it holds that with probability no less than 1 -2 m exp( -C ′ 1 R 2 ) -δ/ 2 , for any f ∈ F ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where r ∗ m is the largest fixed point of ˜ R m , and it can be bounded as

<!-- formula-not-decoded -->

for some absolute constant C 6 . Moreover, we have

<!-- formula-not-decoded -->

Combine this with (B.31),(B.32),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plug in the definition of M = C ( C ′′ X R 6 + M 2 f + d x ( log(1 /T 0 ) T +1 ) ) and let R = C log 1 2 ( md x M f /δ ) for some large constant C . Hence (B.35) and (B.36) reduce to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we obtain that with probability no less than 1 -δ , the population loss of the empirical minimizer ̂ f can be bounded by

<!-- formula-not-decoded -->

We conclude the proof by noticing that E [ X ] = ∫ ∞ 0 P ( X ≥ x )d x and plugging in the bound above.

Proposition B.5 (Prop. 3.3) . There exists some constant C Z , C R such that the following holds. For any P 1 , · · · , P K , let x k 1 , · · · , x k n i.i.d. ∼ P k for any k and ( x k i ) i,k are all independent. Consider the empirical minimizer

<!-- formula-not-decoded -->

For any δ ∈ (0 , 1) , if the configuration of F satisfies R f ≥ C R log 1 2 ( nKM f /δ ) , then with probability no less than 1 -δ , the population loss of ̂ f , ̂ h can be bounded by

<!-- formula-not-decoded -->

Proof. Throughout the proof, we will use z = ( k, x, y ) to denote the tuple of task index k and data ( x, y ) . With a little abuse of notation, we will also let s k ∗ = s P k ∗ . Consider the function class defined on [ K ] × R d x × [0 , 1] D y ,

<!-- formula-not-decoded -->

where 1 ≤ R ≤ R f 2 will be specified later. It is easy to show that with probability no less than 1 -2 nK exp( -C ′ 1 R 2 ) , it holds that ∥ x k i ∥ ∞ ≤ R for all i, k . Hence by definition, the empirical minimizer also satisfies

<!-- formula-not-decoded -->

where z k i = ( k, x k i , y k i ) . Below we reason conditioned on this event and verify the conditions in Lemma B.12.

Following Step 1 and 2 in Proposition B.4, we have for any f ∈ F ⊗ K , h ∈ H ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the local Rademacher complexity bound, note that

<!-- formula-not-decoded -->

where ̂ P ( K ) n := 1 nK K ∑ k =1 n ∑ i =1 δ z k i and diam ( Φ r , ∥ · ∥ L 2 ( ̂ P ( K ) n ) ) ≤ 2 √ r . By Dudley's bound [Van Handel, 2014, Wainwright, 2019], there exists an absolute constant C 0 such that for any θ &gt; 0 ,

<!-- formula-not-decoded -->

Since ∥ x k ∥ ≤ R ,

<!-- formula-not-decoded -->

Let R 1 = 2 R . Since x t | x k i ∼ N ( x t ; α t x k i , σ 2 t I ) , we have P ( ∥ x t ∥ ∞ ≥ R 1 ) ≤ d x P ( |N (0 , 1) | ≤ R ) ≤ 2 d x exp( -C ′ 0 R 2 ) for some absolute constant C ′ 0 . Therefore,

<!-- formula-not-decoded -->

where Ω R 1 := [ -R 1 , R 1 ] d x × [0 , 1] d y × [ T 0 , T ] . Moreover, notice that R f ≥ 2 R = R 1 ,

<!-- formula-not-decoded -->

Plug in the bound above,

<!-- formula-not-decoded -->

For any ε ≥ 32 d 1 2 x M exp( -C ′ 0 R 2 / 2) , according to Lemma B.3,

<!-- formula-not-decoded -->

Plug in (B.25) and let θ = 32 d 1 2 x M exp( -C ′ 0 R 2 / 2) ,

<!-- formula-not-decoded -->

Combine the arguments above, by Lemma B.12 with B 0 = 8 M 2 exp( -C ′ 1 R 2 ) , B = 4 M,b = M , it holds that with probability no less than 1 -2 nK exp( -C ′ 1 R 2 ) -δ/ 2 , for any f ∈ F ⊗ K , h ∈ H ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where r ∗ K,n is the largest fixed point of ˜ R K,n , and it can be bounded by

<!-- formula-not-decoded -->

for some absolute constant C 6 . Moreover, we have

<!-- formula-not-decoded -->

Combine this with (B.54),(B.55),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plug in the definition of M = C ( C ′′ X R 6 + M 2 f + d x ( log(1 /T 0 ) T +1 ) ) and define R = C ′ log 1 2 ( nKd x M f /δ ) for some large constant C ′ . Hence (B.58) and (B.59) reduce to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where nK

Therefore, we obtain that with probability no less than 1 -δ , the population loss of the empirical minimizer ̂ f , ̂ h can be bounded by

<!-- formula-not-decoded -->

which concludes the proof.

Theorem B.6 (Thm. 3.4) . Under Assumption 3.1, 3.2, 3.3, suppose P 1 , · · · , P K are ( ν, ∆) -diverse over target distribution P 0 given F , H . There exists some constant C, C R such that the following holds. Define the empirical minimizer of training task and new task as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If R f ≥ C R log 1 2 ( nKM f /δ ) , then with probability no less than 1 -δ , the expected population loss of new task can be bounded by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

Here we apply Proposition B.4 in the first inequality, task diversity in the second inequality, and Proposition B.5 in the fourth. Plug in the definition of r z , r x and log N F , log N H and we complete the proof.

## B.3 Proofs of Meta-Learning

Proposition B.7 (Prop. 3.5) . There exists some constants C ′ 1 , C P , such that for P 1 , · · · , P K i.i.d. ∼ P meta , with probability no less than 1 -δ , we have for any h ∈ H ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where r P = M 2 f exp( -C ′ 1 R 2 f ) + S h L h log ( KL h W h ( B h ∨ 1) M f γ f ) K .

Proof. Given P 1 , · · · , P K i.i.d. ∼ P meta, we define the empirical Rademacher complexity of a function class Φ defined on the set of distribution P ( R d x × [0 , 1] D y ) as

<!-- formula-not-decoded -->

For any r &gt; 0 , let H r := { h ∈ H : 1 K K ∑ k =1 ( L ( P k , h )) 2 ≤ r } and Φ r := {L ( · , h ) : h ∈ H r } . Note that for any φ 1 , φ 2 ∈ Φ r ,

<!-- formula-not-decoded -->

where P ( K ) meta := 1 K K ∑ k =1 δ P k and diam ( Φ r , ∥ · ∥ L 2 ( P ( K )) meta ) ≤ 2 √ r . Then by Dudley's bound [Van Handel, 2014, Wainwright, 2019], there exists an absolute constant C 0 such that for any θ ≥ 0 ,

<!-- formula-not-decoded -->

For any P and h 1 , h 2 ∈ H r , denote the minimizer of (3.6) in F as f 1 , f 2 , respectively. Without loss of generality, suppose L ( P , h 1 ) ≥ L ( P, h 2 ) . Then

<!-- formula-not-decoded -->

In the last inequality we apply ∥ f i ∥ ≤ M f and E t,x t ,y ∥∇ x log p t ( x t | y ) ∥ 2 ≤ C L by Lemma B.9. Moreover,

<!-- formula-not-decoded -->

Therefore, let C 3 = 32( M f + C 1 / 2 L ) M f ≤ 64 M 2 f and we have

<!-- formula-not-decoded -->

which implies that when ε ≥ 2 C 3 exp( -C ′ 1 R 2 f ) , by Lemma B.3,

<!-- formula-not-decoded -->

Plug in (B.73) and let θ = 2 C 3 exp( -C ′ 1 R 2 f ) ,

<!-- formula-not-decoded -->

According to Lemma B.11 (by setting B 0 = 0 , B = b = C L ), for some absolute constant C 5 , with probability no less than 1 -δ , we have for any h ∈ H ,

<!-- formula-not-decoded -->

where r ∗ K is the unique fixed point of ˜ R K . And it is easy to show that for some absolute constant C 6 ,

<!-- formula-not-decoded -->

which concludes the proof.

Theorem B.8 (Thm. 3.6) . Under Assumption 3.1, 3.2, 3.3, there exists some constant C, C R such that the following holds. Define the empirical minimizer of training task and new task as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If R f ≥ C R log 1 2 ( nKM f /δ ) , then with probability no less than 1 -δ , the expected population loss of new task can be bounded by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

Here we apply Proposition B.4 in the first inequality, Proposition B.7 in the second and last inequality, Proposition B.5 in the fourth. Plugging in the definition of r z , r P , r x and log N F , log N H and noticing that R f ≥ C R log 1 2 ( nKd x M f /δ ) ≥ C ′ R log 1 2 ( M f K log N H ) , we have with probability no less than 1 -δ ,

<!-- formula-not-decoded -->

## B.4 Auxiliary Lemmas

Lemma B.9. There exists some constant C L such that for any h, P ,

<!-- formula-not-decoded -->

Proof. Note that

<!-- formula-not-decoded -->

and 0 ∈ F , it suffices to show that E t,x t ,y [ ∥∇ x log p t ( x t | y ) ∥ 2 ] is uniformly bounded for any P , h . According to (B.2),

<!-- formula-not-decoded -->

On the other hand, by (B.3) and Assumption 3.3,

<!-- formula-not-decoded -->

Therefore, we have

According to (B.2),

By (B.3), we also have

<!-- formula-not-decoded -->

Combine the two inequalities,

<!-- formula-not-decoded -->

Plug in (B.95) and we obtain for some constant C ′′ X ,

<!-- formula-not-decoded -->

Lemma B.11. Let Φ be a class of functions on domain Ω and P be a probability distribution over Ω . Suppose that for any φ ∈ Φ , ∥ φ ∥ L ∞ (Ω) ≤ b , E P [ φ ] ≥ 0 , and E P [ φ 2 ] ≤ B E P [ φ ] + B 0 for some b, B, B 0 ≥ 0 . Let x 1 , · · · , x n i.i.d. ∼ P and ϕ n be a positive, non-decreasing and sub-root function such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma B.10. There exists some constant C ′′ X such that for any t ∈ [0 , T ] and x ∈ R d x , y ∈ [0 , 1] D y , E x t | x ∥∇ x log p t ( x t | y ) ∥ 2 ≤ C ′′ X ( ∥ x ∥ 6 +1) . (B.94)

Proof. Note that x t | x ∼ N ( x t | α t x, σ 2 t I ) and by Lemma B.2,

<!-- formula-not-decoded -->

Let q t ( x 0 | x t , y ) ∝ ϕ t ( x t | x 0 ) p ( x 0 | y ) . Since ϕ t (0 | x 0 ) ∝ exp ( -α 2 t ∥ x ∥ 2 2 σ 2 t ) is decreasing in ∥ x ∥ , by Fortuin-Kasteleyn-Ginibre inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Φ r := { φ ∈ Φ : 1 n n ∑ i =1 ( φ ( x i )) 2 ≤ r } . Define the largest fixed point of ϕ n as r ∗ n . Then for some absolute constant C ′ , with probability no less than 1 -δ , it holds that for any φ ∈ Φ ,

<!-- formula-not-decoded -->

Proof. We follow the procedures in Bousquet [2002]. Let ϵ j = b 2 -j and consider a sequence of classes

<!-- formula-not-decoded -->

Note that Φ = ∪ j ≥ 0 Φ ( j ) and for φ ∈ Φ ( j ) , E P [ φ 2 ] ≤ Bϵ k + B 0 . Let j 0 = ⌊ log 2 n ⌋ . Then by Bousquet [2002, Lemma 6.1], it holds that with probability no less than 1 -δ , for any j ≤ j 0 and φ ∈ Φ ( j ) ,

<!-- formula-not-decoded -->

Besides, for φ ∈ ∪ k&gt;k 0 Φ ( j ) =: Φ ( j 0 :) ,

<!-- formula-not-decoded -->

From now on we reason on the conjunction of (B.105), (B.106) and (B.107). Define

<!-- formula-not-decoded -->

and thus for any φ ∈ Φ ( j ) , we have 1 n n ∑ i =1 ( φ ( x i )) 2 ≤ CU j for some absolute constant C by (B.106), indicating that R n (Φ ( j ) ) ≤ ϕ n ( CU j ) ≤ √ Cϕ n ( U j ) . For any j ≤ j 0 ,

<!-- formula-not-decoded -->

Since ϕ n is non-decreasing and sub-root, the inequality above implies that

<!-- formula-not-decoded -->

Therefore, for any φ ∈ Φ ( j ) , j ≤ j 0 , by (B.105),

<!-- formula-not-decoded -->

Noticing that E P [ φ ] ≤ ϵ j ≤ 2 E P [ φ ] , it reduces to

<!-- formula-not-decoded -->

Hence we have by noting that F n is also a non-decreasing sub-root function,

<!-- formula-not-decoded -->

Here C ′ is an absolute constant. Moreover, when φ ∈ Φ ( j ) for j &gt; j 0 , we have E P [ φ ] ≤ b n , and according to (B.107),

<!-- formula-not-decoded -->

Hence the same bounds apply, which completes the proof.

Lemma B.12. Let Φ be a class of functions on domain Ω , P 1 , · · · , P K be probability distributions over Ω , and ̂ P ( K ) = 1 K K ∑ k =1 δ P k . Suppose that for any φ ∈ Φ , ∥ φ ∥ L ∞ (Ω) ≤ b , E ̂ P ( K ) [ φ ] ≥ 0 , and E ̂ P ( K ) [ φ 2 ] ≤ B E ̂ P ( K ) [ φ ] + B 0 for some b, B, B 0 ≥ 0 . Let x k 1 , · · · , x k n i.i.d. ∼ P k for any k and all ( x k i ) i,k are independent. Let ϕ K,n be a positive, non-decreasing and sub-root function such that

<!-- formula-not-decoded -->

where Φ r := { φ ∈ Φ : 1 nK K ∑ k =1 n ∑ i =1 ( φ ( x k i )) 2 ≤ r } . Define the largest fixed point of ϕ K,n as r ∗ K,n . ′

Then for some absolute constant C , with probability no less than 1 -δ , it holds that for any φ ∈ Φ ,

<!-- formula-not-decoded -->

Proof. We follow the procedures in Bousquet [2002]. Let ϵ k = b 2 -k and consider a sequence of classes

<!-- formula-not-decoded -->

Note that Φ = ∪ j ≥ 0 Φ ( j ) and for φ ∈ Φ ( j ) , E ̂ P ( K ) [ φ 2 ] ≤ Bϵ j + B 0 . Let j 0 = ⌊ log 2 ( nK ) ⌋ . Then by Massart [2000, Theorem 3], with probability no less than 1 -δ , for any j ≤ j 0 and φ ∈ Φ ( j ) ,

<!-- formula-not-decoded -->

Besides, for any φ ∈ ∪ j&gt;j 0 Φ ( j ) =: Φ ( j 0 :) ,

<!-- formula-not-decoded -->

From now on we reason on the conjunction of (B.120), (B.121) and (B.122). Define

<!-- formula-not-decoded -->

and thus for any φ ∈ Φ ( j ) , we have 1 nK K ∑ k =1 n ∑ i =1 ( φ ( x k i )) 2 ≤ CU j for some absolute constant C by (B.121), indicating that R K,n (Φ ( j ) ) ≤ ϕ K,n ( CU j ) ≤ √ Cϕ K,n ( U j ) . For any j ≤ j 0 ,

<!-- formula-not-decoded -->

Since ϕ K,n is non-decreasing and sub-root, the inequality above implies that

<!-- formula-not-decoded -->

Therefore, for any φ ∈ Φ ( j ) , j ≤ j 0 , by (B.120),

<!-- formula-not-decoded -->

Noticing that E ̂ P ( K ) [ φ ] ≤ ϵ j ≤ 2 E ̂ P ( K ) [ φ ] , it reduces to

<!-- formula-not-decoded -->

Hence we have by noting that F K,n is also a non-decreasing sub-root function,

<!-- formula-not-decoded -->

Here C ′ is an absolute constant. Moreover, when φ ∈ Φ ( j ) for j &gt; j 0 , we have E ̂ P ( K ) [ φ ] ≤ b nK , and according to (B.122),

<!-- formula-not-decoded -->

Hence the same bounds apply, which completes the proof.

## B.5 Verifying Task Diversity Assumption

When F is linear function class, Tripuraneni et al. [2020] provides an explicit bound on ( ν, ∆) . However, in general, performing a fine-grained analysis is challenging, especially for complex function classes such as neural networks. In the following proposition, we present a very pessimistic bound for ( ν, ∆) based on density ratio, which is independent of the specific choice of hypothesis classes F and H .

<!-- formula-not-decoded -->

We mention that the only requirement is F is a convex hull of itself, which can be easily satisfied by most hypothesis classes such as neural networks. More refined analysis on specific neural network class is an interesting future work.

<!-- formula-not-decoded -->

## C Proofs in Section 4

## C.1 Proofs of Score Network Approximation

Theorem C.1 (Thm. 4.1) . Under Assumption 3.1, 3.2, 3.3, to achieve R f ≥ C R log 1 2 ( nKM f /δ ) and

<!-- formula-not-decoded -->

the configuration of F = NN f ( L f , W f , M f , S f , B f , R f , γ f ) , H = NN h ( L h , W h , S h , B h ) should satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here O ( · ) hides all the polynomial factors of d x , d y , D y , C 1 , C 2 , L, B .

Proof. With a little abuse of notation, in transfer learning setting, we define P meta := 1 K K ∑ k =1 δ P k and it directly reduces to meta-learning case. Therefore, we only focus on the proof in meta-learning.

We first decompose the misspecification error into two components: representation error and score approximation error.

<!-- formula-not-decoded -->

Further note that for any f ∈ F ,

<!-- formula-not-decoded -->

d d

<!-- formula-not-decoded -->

Proposition C.2. To achieve R f ≥ C R log 1 2 ( nKM f /δ ) and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the configuration of F = NN f ( L f , W f , M f , S f , B f , R f , γ f ) should satisfy

<!-- formula-not-decoded -->

Here O ( · ) hides all the polynomial factors of d x , d y , D y , C 1 , C 2 , L, B .

Proof. For notation simplicity, we will f ∗ = f P ∗ throughout the proof. Our procedures consist of two main steps. The first is to clip the whole input space to a bounded set Ω R f := [ -R f , R f ] d x × [0 , 1] d y × [ T 0 , T ] thanks to the light tail property of P . Then we approximate f P ∗ on Ω R f .

By Lemma B.2 and C.6, f ∗ is γ 1 -Lipschitz in x , γ 2 -Lipschitz in w , and γ 3 -Lipshcitz in t in a bounded domain Ω R f , where γ 1 = C X + C ′ X R 2 f , γ 2 = C X + C ′ X R f , γ 3 = C s R 3 f T 3 .

<!-- formula-not-decoded -->

We first rescale the input domain by x ′ = x 2 R f + 1 2 , w ′ = w,t ′ = t/T , which can be implemented by a single ReLU layer. Denote v = ( x ′ , w ′ , t ′ ) . We only need to approximate g ( v ) := f ∗ ( R f (2 x ′ -1) , w ′ , T t ′ ) defined on Ω := [0 , 1] d x + d y × [ T 0 /T, 1] . And g is γ x := 2 γ 1 R f -Lipschitz in x ′ , γ w := γ 2 -Lipschitz in w ′ and γ t := γ 2 T -Lipschitz in t ′ . We will approximate each coordinate of g = [ g 1 , · · · , g d x ] ⊤ separately and then concatenate them together.

Now we partition the domain Ω into non-overlapping regions. For the first d x + d y dimensions, the space [0 , 1] d x + d y is uniformly divided into hypercubes with an edge length of e 1 . For the last dimension, the interval [ T 0 /T, 1] is divided into subintervals of length e 2 , where the values of e 1 and e 2 will be specified later. Let the number of intervals in each partition be N 1 = ⌈ 1 /e 1 ⌉ and N 2 = ⌈ 1 /e 2 ⌉ , respectively.

<!-- formula-not-decoded -->

where Ψ is the coordinate-wise product of trapezoid function:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We claim that ¯ g i is an approximation to g i since for any o ′ = ( x ′ , w ′ ) ∈ [0 , 1] d x + d y , t ′ ∈ [ T 0 /T, 1] ,

<!-- formula-not-decoded -->

Below we construct a ReLU neural network to approximate ¯ g i . Let σ be ReLU activation and r ( a ) = 2 σ ( a ) -4 σ ( a -0 . 5) + 2 σ ( a -1) for any scalar a ∈ [0 , 1] . Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to Yarotsky [2017],

<!-- formula-not-decoded -->

Then we approximate Ψ u,j by recursively apply ϕ l mul :

<!-- formula-not-decoded -->

And we construct the final neural network approximation as

<!-- formula-not-decoded -->

The approximation error of ̂ g i can be bounded by

<!-- formula-not-decoded -->

Besides, by Chen et al. [2020, Lemma 15], for l ≳ d x + d y and ∀ x ′ , w ′ , w ′′ , t ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define ̂ g := [ ̂ g 1 , · · · , ̂ g d x ] and ̂ f ( x, w, t ) := ̂ g ( x 2 R f + 1 2 , w, t/T ) . Then the approximation error of ̂ f in Ω R f can be bounded by

<!-- formula-not-decoded -->

Therefore, when R f ≥ C R log 1 2 ( ( M 2 f + C L ) /ε ) , the overall approximation error is

<!-- formula-not-decoded -->

Now we characterize the configuration of neural network ̂ f ( x, w, t ) . For boundedness, by Lemma B.10,

<!-- formula-not-decoded -->

Hence we can let R f = O ( log 1 2 ( nK εδ )) to ensure the lower bound of R f mentioned above and in Theorem B.8. For Lipschitzness, by (C.21),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the size of neural network, for each coordinate, by the construction in (C.18), the neural network ̂ g i consists of N d x + d y 1 N 2 parallel subnetworks, i.e. , g i ( u/N 1 , j/N 2 ) ̂ Ψ u,j ( · , · , · ) . By definition in (C.17), the subnetwork consists of O ( ( d x + d y )( d x + d y +log R f ε ) ) layers and the width is bounded by O ( d x + d y ) . Therefore, the whole neural network ̂ g i can be implemented by O (( d x + d y )( d x + d y +log( R f /ε ))) layers with width O ( N d x + d y 1 N 2 ( d x + d y ) ) = O ( R 3( d x + d y ) f ε d x + d y +1 T 3 0 ) , and the number of parameter is bounded by O ( R 3( d x + d y ) f log( R f /ε ) ε d x + d y +1 T 3 0 ) . Com- bine these arguments together, we can claim that the size of neural network ̂ f is

<!-- formula-not-decoded -->

To bound of the neural network parameters, note that the trapezoid function ψ is rescaled by 3 N 1 or 3 N 2 and the weight parameter of ϕ l mul is bounded by a constant. Moreover, the input of ̂ f is first rescaled by R f or T . Hence we have

<!-- formula-not-decoded -->

which concludes the proof.

## Proposition C.3. To achieve

<!-- formula-not-decoded -->

the configuration of H = NN h ( L h , W h , S h , B h ) should satisfy

<!-- formula-not-decoded -->

Here O ( · ) hides all the polynomial factors of d x , d y , L .

Proof. The main idea replicates Yarotsky [2017, Theorem 1]. We approximate each coordinate of h ∗ = [ h ∗ 1 , · · · , h ∗ d y ] respectively and then concatenate all them together. By Yarotsky [2017, Theorem 1], h ∗ i can be approximated up to ε by a network ̂ h i with O (log(1 /ε )) layers and O ( ε -D y log(1 /ε ) ) width. Besides, the range of all the parameters are bounded by some constant, and the number of parameters is O ( ε -D y log 2 (1 /ε ) ) . Then we concatenate all the subnetworks to get ̂ h = [ ̂ h 1 , · · · , ̂ h d y ] and ∥ ̂ h -h ∗ ∥ L ∞ ([0 , 1] Dy ) ≤ √ d y ε .

## C.2 Proofs of Distribution Estimation

Theorem C.4 (Thm. 4.2) . Suppose Assumption 3.1, 3.2, 3.3 hold. For sufficiently large integers n, K, m and δ &gt; 0 , further suppose that P 1 , · · · , P K are ( ν, ∆) -diverse over target distribution P 0 with proper configuration of neural network family and T, T 0 . It holds that with probability no less than 1 -δ ,

<!-- formula-not-decoded -->

Proof. Combine Theorem C.1 and Theorem B.6 and plug in the configuration of F , H , we have with probability no less than 1 -δ

<!-- formula-not-decoded -->

By Lemma C.7,

<!-- formula-not-decoded -->

Taking expectation of y, ̂ f P , P , we have

<!-- formula-not-decoded -->

Let T 0 = O ( ε 2 0 / log d x +1 (1 /ε 0 ) ) , T = O (log(1 /ε 0 )) , ε = O ( ε 0 / log( nK/ ( ε 0 δ 0 ))) for some small ε 0 &gt; 0 defined later. Then it reduces to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem C.5 (Thm. 4.3) . Suppose Assumption 3.1, 3.2, 3.3 hold. For sufficiently large integers n, K, m and δ &gt; 0 , with proper configuration of neural network family and T, T 0 , it holds that with probability no less than 1 -δ ,

<!-- formula-not-decoded -->

Proof. Combine Theorem C.1 and Theorem B.8 and plug in the configuration of F , H , we have with probability no less than 1 -δ

<!-- formula-not-decoded -->

By Lemma C.7,

<!-- formula-not-decoded -->

Taking expectation of y, ̂ f P , P , we have

<!-- formula-not-decoded -->

Let T 0 = O ( ε 2 0 / log d x +1 (1 /ε 0 ) ) , T = O (log(1 /ε 0 )) , ε = O ( ε 0 / log( nK/ ( ε 0 δ 0 ))) for some small ε 0 &gt; 0 defined later. Then it reduces to

<!-- formula-not-decoded -->

Let ε 0 = C max { log 5 2 ( nK/δ ) log 3 ( m ∧ n ) ( m ∧ n ) 1 dx + dy +9 , log 2 ( nK/δ ) K 1 Dy +2 } , and we can conclude that

<!-- formula-not-decoded -->

## C.3 Auxiliary Lemmas

Lemma C.6. Let Ω R f = [ -R f , R f ] d x × [0 , 1] d y × [ T 0 , T ] for some R f ≥ 1 . Then there exists some constant C s , such that the score function f P ∗ ( x, w, t ) is C s R 3 f T 3 0 -Lipschitz with respect to t in Ω R f .

Proof. According to (B.2),

<!-- formula-not-decoded -->

Define density function q t ( x 0 | x, w ) ∝ ϕ t ( x | x 0 ) p ( x 0 ; w ) . Then

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let R = 2 R f +2 C 0 σ t . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, for any ( x, w, t ) ∈ Ω R f ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. With a little abuse of notation, we will use p t ( x t | y ) to denote the conditional density of x t | y under P 0 x | y . Consider the following two backward processes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote the distribution of ˜ x t as ˜ P T -t . And note that ¯ x t ∼ p T -t by classic reverse-time SDE results [Anderson, 1982]. Then by Fu et al. [2024, Lemma D.5],

<!-- formula-not-decoded -->

At the same time, we apply Data Processing inequality and Pinsker's inequality to get

<!-- formula-not-decoded -->

Again according to Pinsker's inequality and Oko et al. [2023, Proposition D.1],

<!-- formula-not-decoded -->

Combine three inequalities above and we complete the proof.

## D Proofs in Section A

## D.1 Proof of Theorem A.1

Proof. Due to the structure of exponential family, Assumption 3.2 holds obviously. To apply previous results, we only need to verify Assumption 3.1 and 3.3. Recall that a basic property of exponential family is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence by Assumption A.1, A ψ ( x ) ≤ A ψ (0) + ∥ x ∥ 1 ≤ log (∫ ψ ( y )d y ) + ∥ x ∥ 1 ≤ log C + ∥ x ∥ 1 .

And A ψ ( x ) ≥ A ψ (0) -∥ x ∥ 1 ≥ -log C -∥ x ∥ 1 . Further note that the posterior density p θ ( x | y ) = p ϕ ( x ) exp( ⟨ x, h ∗ ( y ) ⟩ -A ψ ( x )) , where the normalizing constant Z θ ( y ) is lower bounded by

Z θ

<!-- formula-not-decoded -->

where in the second inequality we apply P ϕ ( ∥ x ∥ ≥ R ) ≤ 2 exp( -C ′ 1 R 2 ) and let R = 1 / √ C ′ 1 to get C 0 . Therefore, by Assumption A.1,

<!-- formula-not-decoded -->

and thus Assumption 3.1 holds. At the same time, ley w = h ∗ ( y ) , then the score function is

<!-- formula-not-decoded -->

Since ∇ x log p ϕ ( x ) is L -Lipschitz, ∇ A ψ ( x ) is also 1 -Lipschitz, the score function ∇ x log p θ ( x, w ) is ( L +1) -Lipschitz in x and 1 -Lipschitz in w . And ∥∇ x log p θ (0 , w ) ∥ ≤ ∥∇ x log p ϕ (0) ∥ +2 √ d = B +2 √ d , indicating that Assumption 3.3 holds with L ′ = L +1 , B ′ = B +2 √ d .

We conclude the proof by applying Theorem 4.3 under meta-learning setting or Theorem 4.2 under ( ν, ∆) -diversity.

## D.2 Proof of Theorem A.2

Proof. Let A π M ( s, a ) = Q π M ( s, a ) -V M ( π, s ) be the advantage function of policy π . Note that the reward function r M ∈ [0 , 1] , we have | A π M ( s, a ) | ≤ 2 1 -γ for any M,π . According to performance difference lemma,

<!-- formula-not-decoded -->

Hence in meta-learning setting, we plug in Theorem 4.3 to obtain

<!-- formula-not-decoded -->

If we further assume ( ν, ∆) -diversity holds, then we plug in Theorem 4.2,

<!-- formula-not-decoded -->

## E Experiment Details

## E.1 Conditioned Diffusion

Each f k and f 0 are implemented as a 2-layer MLP with 128 internal channels and 60 input channels. The representation map h is implemented as a 5-layer MLP with 512 internal channels and 10 output channels. We have n = 1000 pre-training samples from each source distribution P k , m ∈ { 10 , 20 , 30 , 40 , 50 , 100 } fine-tuning samples from the target distribution P 0 . We run Langevin Monte Carlo for sufficiently long time to obtain 100 test samples from the target distribution P 0 for evaluating the test error of different models. In the pre-training phase, the { ̂ f k ; 1 ≤ k ≤ K } and ˆ h are trained on the K = 10 source distributions with 400K iterations and a batch size of 512. In the fine-tuning phase, the pre-trained representation map ̂ h is fixed, and the ̂ f 0 is trained on the target distribution with 200K iterations and a batch size of m . As an important baseline, we also consider jointly training h and f 0 on the target distribution from scratch, using the same fine-tuning samples.

## E.2 Image Restoration on MNIST

Each f k and f 0 are implemented as a 3-layer MLP with 512 internal channels and 784 input channels. The representation map h is implemented as a 5-layer MLP with 512 internal channels and 64 output channels. We have n = 5000 pre-training samples from each source distribution P k , and m ∈ { 10 , 20 , 30 , 40 , 50 , 100 } fine-tuning samples from the target distribution P 0 . For evaluation, we directly compute the mean squared error between the posterior samples and the ground truth images, based on 100 test samples from P 0 . In the pre-training phase, the the { ̂ f k ; 1 ≤ k ≤ K = 9 } and ˆ h are 2K epochs and a batch size of 512. The initial learning rate is 0.0003 and is annealed according to a cosine annealing schedule. In the fine-tuning phase, the pre-trained representation map ̂ h is fixed, and the ̂ f 0 is trained on the target distribution with 20K iterations and a batch size of m . As an important baseline, we also consider jointly training h and f 0 on the target distribution from scratch, using the same fine-tuning samples.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction accurately reflect the paper's contributions, i.e., proposing a data-efficient training method for machine learning models.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses the limitations of the work.

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

Answer: [Yes]

Justification: The paper provides the full set of assumptions and complete (and correct) proofs.

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

Justification: The paper fully discloses all the information needed to reproduce the main experimental results.

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

Justification: We will provide complete codes upon acceptance.

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

Justification: The paper specifies all the training and test details.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The paper reports experimental results based on the average of independent random trials.

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

Justification: The paper provides sufficient information on the computer resources.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

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
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
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

Answer: [NA]

Justification: The paper does not use existing assets.

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

Justification: The paper does not release new assets currently.

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