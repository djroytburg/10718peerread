## Bandit Guided Submodular Curriculum for Adaptive Subset Selection

## Prateek Chanda ∗ Prayas Agrawal ∗ Saral Sureka Lokesh Reddy Polu Atharv Kshirsagar Ganesh Ramakrishnan

Department of Computer Science and Engineering, Indian Institute of Technology Bombay

{prateekch, prayas, ssaral {lokeshreddypolu, atharvksagar, ganesh}@cse.iitb.ac.in

## Abstract

Traditional curriculum learning proceeds from easy to hard samples, yet defining a reliable notion of difficulty remains elusive. Prior work has used submodular functions to induce difficulty scores in curriculum learning. We reinterpret adaptive subset selection and formulate it as a multi-armed bandit problem, where each arm corresponds to a submodular function guiding sample selection. We introduce ONLINESUBMOD, a novel online greedy policy that optimizes a utility-driven reward and provably achieves no-regret performance under various sampling regimes. Empirically, ONLINESUBMOD outperforms both traditional curriculum learning and bi-level optimization approaches across vision and language datasets , showing superior accuracy-efficiency tradeoffs. More broadly, we show that validationdriven reward metrics offer a principled way to guide the curriculum schedule. Our code is publicly available at GitHub 2 .

## 1 Introduction

Curriculum Learning (CL), inspired by cognitive development, posits that training machine learning models by gradually exposing them to data of increasing complexity can significantly enhance both learning efficiency and generalization performance [4, 59]. The underlying principle is that mastering simpler concepts first provides a robust foundation for acquiring more complex ones, leading to improved convergence and a more effective exploration of the hypothesis space [19]. Empirical evidence shows CL improves model training, particularly in areas like code understanding [33], enhances graph embeddings through complexity-based ordering [57], mitigates catastrophic forgetting [2, 26, 50, 43], and boosts learning efficiency in reinforcement learning [34]. We first provide a formal definition of Curriculum Learning.

Definition 1. ( Curriculum Learning ) Given a dataset D = ⋃ k i =1 B i partitioned into disjoint batches B i , and a batch difficulty score function d : {B i } k i =1 → R ≥ 0 assigning non-negative difficulty scores, a batch-wise curriculum can be represented as a permutation π : [ k ] ↦→ [ k ] over the ordered indices such that the ordered sequence

<!-- formula-not-decoded -->

satisfies the monotonic difficulty score condition : d ( B π ( t ) ) ≤ d ( B π ( t +1) ) ∀ t ∈ { 1 , . . . , k -1 } .

Determining Difficulty is challenging A critical challenge in realizing the full potential of Curriculum Learning (CL) is determining the optimal sequence of batches. This is complicated by the fact

∗ These authors contributed equally.

2 https://github.com/efficiency-learning/banditsubmod/

that the difficulty score, denoted as d , is typically unknown. Traditional approaches often rely on domain expertise or practitioner's knowledge to assess the hardness or difficulty of samples.

Recent works, such as [19], have proposed using submodular function maximization over data batches as an intrinsic measure of sample difficulty. In particular, representative submodular functions representative submodular functions are used to identify easy samples, while diversity focused submodular functions are used to capture difficult ones. As a result, the CL objective is typically constructed by prioritizing diversity functions later and representative functions earlier in the training phase. However, this definition of hardness is still restrictive, as it relies on a fixed pretraining phase and does not account for evolving training dynamics.

Figure 1: Sequential Ordering of Submodular Functions : Observations on CIFAR100 : Initial training with subsets sampled using representationbased submodular functions followed by diversity results in better performance gains than the opposite order. (1a): First 50% of steps in an epoch . (1b): First 50% of epochs .

<!-- image -->

## 1.1 Our Contributions

Submodular curriculum learning via online bandits We formulate the curriculum learning problem in conjunction with the adaptive subset selection as a multi-arm bandit problem, where each arm corresponds to a submodular function that captures its unique characteristics, thereby providing a good surrogate difficulty score required for curriculum learning design.

A no-regret greedy policy for adaptive subset selection We introduce ONLINESUBMOD, a novel greedy utility-based policy that leverages feedback from validation performance-driven reward signal to adaptively guide the subset selection process. We prove that ONLINESUBMOD achieves noregret performance under general sampling regimes, providing theoretical grounding for its learning efficiency.

Validation performance-aware reward design Unlike prior work which uses static heuristics or model-dependent metrics, we define a utility function based on validation performance-driven reward improvements, thereby aligning curriculum progression with actual generalization objectives. Empirical improvements across modalities Through extensive experiments on large-scale language and vision benchmarks, we demonstrate that ONLINESUBMOD outperforms traditional curriculum strategies and state-of-the-art adaptive selection methods in terms of accuracy-efficiency trade-offs across diverse subset budgets and training stages.

## 1.2 Brief Discussion on Related Work &amp; Limitations

Here we detail some of the recent prior work in the space of adaptive subset selection and corresponding limitations.

Leveraging Training Gradient information : Efficiently training robust machine learning models often involves selecting informative data subsets. GLISTER [20] directly addresses this through a mixed discrete-continuous bi-level optimization framework, leveraging validation likelihood for robustness. The concept of adaptive data subset selection , where the subset evolves during training, is explored by methodologies like coreset selection [32]. [18] tackles the problem by focusing on minimizing gradient matching error , as the quality of this matching significantly impacts convergence. By modeling this error as weakly submodular and using OMP [9], GradMatch achieves tighter convergence guarantees for various convex loss functions. Despite their advancements, many contemporary subset selection techniques, such as coreset selection and related methods [8], pose a considerable computational burden due to their complex optimization processes.

Adaptive Subset Selection Induces CL Many adaptive subset selection methods although can be viewed as forms of curriculum learning, incur substantial computational overhead. For instance, Glister [20] solves costly bilevel optimization involving joint subset selection and model training with validation feedback. GradMatch [18] minimizes gradient matching error by solving complex optimization problems to approximate full-dataset gradients. Importance sampling approaches [7, 46] similarly require expensive importance score estimations. Such costs limit the scalability of advanced curriculum strategies, especially under resource constraints or large datasets.

Reweighting Techniques : In this context, [53] offered significant insights into strategies for selecting data subsets that focus on identifying high-quality subsets during the training of models. As we shift towards meta-learning and weighted loss techniques, traditional methods like importance sampling, first introduced by [6], and more contemporary approaches such as focal loss proposed by [25], provide essential perspectives on weighting samples to highlight more challenging examples during training. However, all these strategies entail additional costs. We further share a more detailed Related work section in Appendix G.

## 2 Notation and Problem Setup

Notation : We consider a supervised learning setup where we have a training dataset D tr = { ( x i , y i ) } n i =1 , with each instance independently and identically distributed (i.i.d.) according to a distribution P X×Y over the feature space X and label space Y . Similarly, we have a validation dataset D val = { ( x val j , y val j ) } j m =1 , also drawn i.i.d. from P X×Y . Here, x ∈ X represents the features and y ∈ Y represents the labels. Let M θ be a model parameterized by θ ∈ Θ ⊆ R d , with Θ being a compact and convex parameter space. The learning objective is to minimize the empirical risk L ( M θ ; D tr ) [51]. The training process unfolds over a discrete time horizon T ∈ Z + . Let F be the space of set functions, with F sub ⊂ F denoting the subspace of submodular functions.

Note: Throughout this paper, we use z to denote a training instance from B t , unless explicitly labeled as z val, which refers to a validation instance. In Appendix Section B we provide an extensive notation summary. We provide here some important definitions which would be utilised in the later sections.

Definition 2 ( Submodularity ) . Given a ground set V , a set function f : 2 V ↦→ R is submodular if for all S ⊆ V and B ⊆ A ⊆ V , it holds that f ( S ∪ A ) -f ( A ) ≤ f ( S ∪ B ) -f ( B ) .

Definition 3 ( Monotonicity ) . A set function f : 2 V ↦→ R ≥ 0 is monotone if for all B ⊆ A ⊆ V , it holds that f ( B ) ≤ f ( A ) .

Definition 4 ( Maximum High Value Subset ) . Corresponding to a monotone submodular function f , the maximum high value subset of cardinality at most β , denoted by f arg ( β ) = B opt ⊆ V , is defined as: B opt = argmax B⊆V ; |B|≤ β f ( B ) .

## 2.1 Problem Formulation : Adaptive Subset Selection posed as Curriculum Learning

At each discrete time step t ∈ [ T ] , we consider a mini-batch B t ⊆ D tr upon which the model M θ is trained. Let ℓ : Z× Θ ↦→ R denote the instance-wise loss function, where Z = X ×Y is the instance space, and the model parameter at time t is denoted by θ t ∈ Θ . The total loss over the mini-batch B t is given by L t ( θ t ) = ∑ z ∈B t ℓ ( z , θ t ) . Concurrently, we have access to a validation mini-batch B val t ⊆ D val at each time step t .

Gradient Matrix and Mean Gradient: Let G θ t = [ g θ t ( z 1 ) , . . . , g θ t ( z |B t | ) ] ∈ R d ×|B t | be the batch gradient matrix at time step t , where each column g θ t ( z i ) = ∇ θ ℓ ( z i , θ t ) ∈ R d is the sample-wise gradient of the loss function ℓ with respect to the model parameter θ t , for all z i ∈ B t . Let 1 |B t | ∈ R |B t |× 1 denote the column vector of ones. We define the per-batch gradient as ¯ g ( b ) θ t = 1 |B t | ∑ z i ∈B t g θ t ( z i ) = 1 |B t | G θ t 1 |B t | .

Action Space and Submodular Selection Policy. At each time step t ∈ [ T ] , the learner observes a mini-batch B t ⊆ D tr and must select a subset of size β to compute a gradient update. The learner chooses an action a t ∈ A from a discrete action space: A := { f (1) , f (2) , . . . , f ( K ) } , f ( a ) ∈

| Function                                             | f ( X )                                                          |
|------------------------------------------------------|------------------------------------------------------------------|
| Representative Facility Location Graph Cut Diversity | ∑ i ∈V max j ∈ X s ij ∑ i ∈V ,j ∈ X s ij - ρ ∑ i,j ∈ X s ij      |
| Log Determinant Disparity-Min Disparity-Sum          | log det( S X ) min i = j ∈ X (1 - s ij ) ∑ i = j ∈ X (1 - s ij ) |

̸

Table 1: Submodular functions used in arm definitions. V is the ground set, X ⊆ V , s ij denotes pairwise similarity, and S X is the similarity submatrix. ρ indicates the balancing factor between representative and diversity nature. We also utilise mutual information variants (Details in Appendix)

F sub , where each f ( a ) : 2 B t → R is a monotone submodular function used to score subsets of B t . These functions encode different sample selection criteria such as diversity, coverage,

̸

and representativeness (see Table 1 for examples). The selected function f ( a t ) is then approximately maximized over B t under a fixed cardinality constraint to produce a training subset: S t := arg max S ⊆B t , | S |≤ β f ( a t ) ( S ) , which is typically computed via a greedy algorithm. The model is updated using S t , and the quality of the update is evaluated using a utility-based reward defined on a held-out validation mini-batch B val t ⊆ D val .

Specifically, let ϑ ( a | B t ) be the empirical estimate of the expected reward for arm a ∈ A .

Policy: Greedy Deterministic Selection. We adopt a greedy deterministic policy π : 2 D tr → A that selects the arm with the highest estimated reward at each time step i.e. a t := π ( B t ) := arg max a ∈ A ϑ ( a t | B t ) . . where U t is the utility function defined in Section 2.1.

Regret as a Performance Measure We denote by ( ∗ ) the index of an optimal action, so that µ ( ∗ ) ( B t ) represents the expected utility (e.g., value of the selected subset) of an optimal submodular function f ( a ∗ t ) when applied to mini-batch B t . For each action a t ∈ A , we define the optimality gap at time t as ∆ ( a t ) ( B t ) := max { 0 , ϑ ( a ∗ t | B t ) -ϑ ( a t | B t ) } . The cumulative regret after T rounds is then defined as Regret T := ∑ T t =1 ∆ a t ( B t ) , . Minimizing Regret T ensures that the learner approaches the performance of the best submodular selector in hindsight. We define ϑ ( · | B t ) in Sec 2.2

Reward Utility Metric for Performance Evaluation Drawing upon the concept of training data influence [38], we define a utility function U t ( B t , z val ) : 2 D tr ×D val ↦→ R to quantify the impact of a training mini-batch B t ⊆ D tr at time step t on a validation instance z val ∈ B val t . Specifically, the utility is the reduction in the loss on the validation instance after one step of stochastic gradient descent:

<!-- formula-not-decoded -->

where the updated parameter vector ˜ θ t +1 ( B t ) = θ t -η t ∇ θ ( 1 |B t | ∑ z ∈B t ℓ ( z , θ t ) ) .

First-Order Approximation of Marginal Utility Gain : We define the instance-wise conditional marginal utility gain of including the i -th training instance z i into a partially constructed mini-batch B ( &lt;i ) t = { z 1 , z 2 , . . . , z i -1 } at time step t , with respect to a validation instance z val , as the change in utility U t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The approximation in the last step utilizes a first-order Taylor expansion, which is reasonable under the common assumption of a small learning rate η t . We defer the derivation to Appendix

Second-Order Approximation and Gradient Influence : Further approximating the second term in Equation (3) using another first-order Taylor expansion around θ t , we obtain:

<!-- formula-not-decoded -->

where H z val ( θ t ) = ∇ 2 θ ℓ ( z val , θ t ) denotes the Hessian of the loss function with respect to the model parameters θ evaluated at θ t for the validation data point z val .

Gradient Influence Function : The first term indicates the importance score of z i w.r.t validation data point z val which, in essence, captures the effectiveness of the gradient of the training instance z i towards the reduction in the validation loss. This term closely resembles the influence function proposed in [38].

Relative Similarity Term The second term indicates the Hessian weighted relative similarity of the current training instance with all other training instances in the batch B ( &lt;i ) t .

Hessian Approximation Strategies The Hessian term H z val ( θ t ) in Equation (4) presents a major computational bottleneck due to its high cost. To alleviate this, several approximation strategies are

commonly employed: Kronecker-Factored Approximation methods [54] exploit layer-wise structure and approximate the Hessian using Kronecker products; Gauss-Newton Decomposition [42] replaces the Hessian with the covariance of output gradients, assuming a negligible residual; and the Identity Approximation [29, 37] simplifies the Hessian to I d , yielding a low-cost diagonal preconditioner. In our current list of experiments, we consider Hessian to I d as it is has been shown to be usefull with low approximation error in large scale trainings e.g. LLM settings [52]. In Appendix Section D.6, we include other Hessian Approximation strategies which we tried out along with corresponding ablation studies.

## 2.2 Sample-wise Expected Marginal Gain

We define the sample-wise expected marginal gain as the expectation of the conditional marginal utility gain over a validation instance z val t and a training instance z i from the partially constructed minibatch B ( &lt;i ) t as E z val t ∈B val t , z i ∈B ( &lt;i ) t [ ∆ U t ( z i | B ( &lt;i ) t , z val t )] Here, due to the property of permutation invariance over the samples in B ( &lt;i ) t as shown in Lemma 2, the inner expectation can be written as:

<!-- formula-not-decoded -->

A direct greedy approach to maximize the conditional marginal gain at each step t by iteratively selecting the training instance z ∗ i / ∈ B ( &lt;i ) t that yields the maximal local reduction in validation loss, i.e., z ∗ i = argmax z i / ∈B ( &lt;i ) t ∆ U t ( z i | B ( &lt;i ) t , z val t ) , is computationally prohibitive. Constructing the new subset batch S t of size β from the current mini-batch B t via this exhaustive greedy maximization starting from an empty set ( B ( &lt; 0) t = ∅ ) incurs a computational complexity of O ( |B t | β ) .

Submodular Relaxation for Efficient Selection: To overcome the computational intractability of exact optimization, we introduce a relaxation that exploits the structure of submodular functions to enable efficient selection of high-value subsets. Specifically, for each submodular function arm a t ∈ A , we compute an approximately optimal subset S opt a t ⊆ B t of size at most β , chosen to maximize the submodular objective f ( a t ) ( S ) . Since exact maximization of submodular functions is NP-hard, we adopt a standard greedy algorithm that offers a provable (1 -1 /e ) -approximation guarantee under cardinality constraints.

Reward Formulation using Submodular Function Arms: We define the overall expected marginal gain ϑ : A × T ↦→ R for each submodular function arm a t ∈ A at time step t as the expectation of the instance-wise conditional marginal gain ∆ U t , conditioned on a validation instance and a training instance from the approximately optimal subset S opt a t :

<!-- formula-not-decoded -->

The best arm is then selected via ˆ a t = arg max a t ∈ A ( ϑ ( a t | B t )) .

## 2.3 Speedup for ONLINESUBMOD

Gradient Computation Full-model gradients in deep networks are expensive to compute due to high-dimensionality. For vision tasks, we adopt last-layer gradients following [3], and for LLMs, we compute gradients over LoRA adapters (rank 128) as in [52]. Both reduce overhead while preserving informative signals for subset selection.

ONLINESUBMOD-Batch To align with batch-level baselines [18], we extend our samplewise formulation to the batch setting, treating each batch as a unit. Let M ( b ) t = [ ˜ g ( b ) 1 · · · ˜ g ( b ) | S t | ] be the matrix of average gradients ˜ g i for batches B i ∈ S t , where S t denotes the set of sampled batches at time t . The expected conditional marginal gain becomes:

<!-- formula-not-decoded -->

Other methods can be analogously adapted by substituting samples x i with batches B i .

## 3 Algorithm

ONLINESUBMOD instantiates a contextual multi-armed bandit framework to adaptively select curriculum policies throughout training.

```
Algorithm 1: ONLINESUBMOD Input: T ∈ N : Total training steps { f ( a ) } K a =1 : Candidate submodular arms λ ( · ) , π ( · ) : Time-varying exploration parameters Output: θ T +1 : Final model parameter 1 for t = 1 to T do 2 Receive batch B t 3 Sample ζ ∼ U (0 , 1) 4 Threshold : Ξ t ← t ( t + λ ( t )) π ( t ) 5 ˆ a t ← { arg max a t ∈ A ϑ ( a t | B t ) if ζ > Ξ t Uniform ( A ) otherwise 6 S (ˆ a t ) ← arg max |S|≤ β, S⊆B t f (ˆ a t ) ( S ) 7 θ t +1 ← θ t -η t |S (ˆ a t ) | ∑ z ∈S (ˆ a t ) g θ t ( z ) 8 return θ T +1
```

## 4 Theoretical Results

In this section, we present the main theoretical results of our work, focusing on regret guarantees for our best-arm selection policy. Specifically, we analyze the regret incurred by our method relative to the performance of the optimal arm in hindsight. This requires a set of structural assumptions (pertaining to describe properties of the exploration dynamics, utility approximation quality, and the existence of a reward gap between optimal and suboptimal arms).

Assumption (a) ( Constant Fractional Exploration Dampening ): The exploration dampening parameter λ ( t ) is time-invariant λ ( t ) = ϵ where ϵ ∈ (0 , 1) .

Assumption (b) ( Optimality Gap ): There exists an optimality gap ϱ such that for every suboptimal arm a t ∈ A \ { a ∗ } : 0 ≤ ϱ ≤ ∆ ( a t ) ( B t ) .

Assumption (c) ( Fractional Exploration Sharpness ): The exploration sharpness parameter π ( t ) is a bounded quantity π ( t ) ∈ (0 , 1) .

Assumption (d) ( Utility Metric Approximation ): The utility metric U t ( · , · ) satisfies the approximation bound as per Theorem 2 (Appendix) with constants C ( a ) for each arm a ∈ A and let n a be a specific constant associated with arm a such that Theorem 2 (Appendix) holds true.

Theorem 1 ( Regret Guarantees ) . Under Assumptions a - d , for all t &gt; t 0 , with probability at least

<!-- formula-not-decoded -->

the expected instantaneous regret incurred by the arm selection policy satisfies

<!-- formula-not-decoded -->

where C ∗ is the approximation constant corresponding to the optimal arm a ∗ .

- Step: 1-2 The model receives a batch B t and chooses an arm ˆ a t ∈ A , each corresponding to a distinct submodular utility function f (ˆ a t ) : 2 B t → R ≥ 0 .
- Step: 5 The arm selection is governed by a exploration threshold Ξ t := t ( t + λ ( t )) π ( t ) , parameterized by time-dependent schedules λ ( t ) and π ( t ) that modulate the annealing from exploration to exploitation. Here, λ ( t ) (Exploration Dampening) and π ( t ) (Exploration Sharpness) act as curriculum schedulers. If a uniform sample satisfies ζ &gt; Ξ t , the algorithm enters Exploitation Phase and selects the arm maximizing ϑ ( a | B t ) ; otherwise, an arm is sampled uniformly at random ( Exploration Phase ).
- Step: 6 Once an arm ˆ a t is selected, the algorithm performs approximate maximization over B t with respect to f (ˆ a t ) , selecting a subset S (ˆ a t ) .
- Step: 7 The model parameters θ t are then updated using a stochastic gradient step computed only on the selected subset.

The theorem guarantees that, under the specified assumptions, the arm selection strategy based on maximizing the expected marginal utility gain converges to the optimal arm almost surely, with the regret decreasing at a rate combining a fast 1 /t decay and a slower √ log t t decay modulated by constants related to the utility approximation and the number of arms. The presence of the optimality gap ϱ in the denominator highlights the difficulty of distinguishing between arms when their utility values are close. We also showcase proofs in Appendix when Assumption ( a ) and Assumption ( c ) are relaxed with no constraints on the bounds of λ ( · ) and π ( · ) .

## 4.1 Supporting Lemmas

Here we detail out Supporting Lemmas that are utilised in proofs and derivations above.

Lemma 1 ( Permutation Invariance of Expected Marginal Gain ) . Let Π denote the set of all permutations over the elements of B ( &lt;i ) t . Then the expected marginal gain E z i ∈B ( &lt;i ) t [ ∆ U t ( z i | B ( &lt;i ) t , z val t ) ] is invariant under any permutation π ∈ Π , i.e.,

<!-- formula-not-decoded -->

We provide the detailed derivations for all proofs in Appendix H.

## 5 Experimental Setup

We evaluate ONLINESUBMOD across diverse datasets to highlight its advantages in terms of both accuracy and computational efficiency. All vision-related experiments are conducted using NVIDIA 3 × A6000 GPUs, while large language model (LLM) experiments are performed on 8 × H100 GPUs to ensure fair comparisons with all baselines. We share more details in Appendix Section D.

## 5.1 Finetuning Large Language Models

Model-Training-Evaluation Pairs. Weevaluate ONLINESUBMOD using combinations of two LLMs: LLAMA-2-7B [49] and MISTRAL-7B [16] finetuned on LESS [55], with performance assessed on MMLU and TYDIQA (Table 1). We use batch size of 16 and use 2 random validation points for computing the reward utility. We select 50% of the batch data for gradient updates during each step.

Table 1: Performance comparison across tasks. Bold indicates best performance in each column.

| Method       | Avg.   | Soc.   | Pol.   | Hist.   | Anat.   | ML.   | Eth.   | Gen.   | Bio.   | Chem.   | TydiQA   |
|--------------|--------|--------|--------|---------|---------|-------|--------|--------|--------|---------|----------|
| GradNorm     | 46.4%  | 61.0%  | 62.5%  | 52.1%   | 40.5%   | 40.2% | 43.0%  | 46.7%  | 42.9%  | 32.3%   | 54.6%    |
| MaxLoss      | 45.2%  | 60.2%  | 64.4%  | 48.0%   | 39.5%   | 38.1% | 44.4%  | 43.8%  | 42.6%  | 31.1%   | 55.4%    |
| RhoLoss      | 46.4%  | 60.6%  | 66.2%  | 49.4%   | 41.5%   | 40.2% | 42.8%  | 46.1%  | 41.1%  | 33.7%   | 55.2%    |
| SBERT        | 45.8%  | 62.3%  | 63.7%  | 47.0%   | 43.1%   | 36.8% | 43.4%  | 44.2%  | 42.0%  | 32.4%   | 54.2%    |
| GREATS       | 47.8%  | 63.2%  | 66.2%  | 48.3%   | 42.6%   | 41.1% | 46.2%  | 48.9%  | 43.1%  | 33.6%   | 55.7%    |
| ONLINESUBMOD | 49.6%  | 65.3%  | 67.4%  | 52.1%   | 45.2%   | 42.7% | 48.6%  | 50.9%  | 45.1%  | 35.7%   | 55.9%    |

<!-- image -->

Steps

Steps

Steps

Steps

Figure 2: Test perplexity dynamics on LLAMA-2-7B during training with various online batch selection strategies on MMLU . We evaluate on US Foreign Policy , Anatomy , Sociology , and Chemistry . ONLINESUBMOD significantly outperforms baselines.

Baselines : We compare our algorithm with a variety of online batch selection algorithms: 1 MAX-LOSS [27], which selects training data points with the highest loss values. 2 GRADNORM [17], which prioritizes training data points with the highest gradient norms, 3 RHO-LOSS[30], using LLaMA-3.1-8B-Instruct as the reference and LLaMA-2-7B as the target. 4 SBERT, which selects batches by semantic similarity to validation data using Sentence-BERT embeddings [40]. 5 GREATS [52] which has a similar utility metric as ours, but the optimization objective instead relies

on directly selecting samples greedily that maximizes the utility reward Eqn (4) rather than utilizing any monotone submodular characteristics.

Observations: As can be seen from the perplexity curves (Figure 8) and downstream performance (Table 1), ONLINESUBMOD significantly outperforms other existing baselines, thereby indicating how principled validation performance aware reward signal combined with induced submodular curriculum results in better generalization than static heuristic based approaches.

## 5.2 Image Classification

We showcase the utility of our method across 5 datasets primarily CIFAR10, CIFAR100 [22],TINYIMAGENET[23], MNIST [24] and SVHN [36]. We compare ONLINESUBMOD with: 1 GRADMATCH: [18], 2 CRAIG [32], 3 GLISTER [20], 4 RHO-LOSS [30] and 5 BOSS [1]. All models are trained on a ResNet backbone, 300 epochs, with 20 epochs warm-start (we provide more details regarding cold start vs warm start in Appendix Sec: C.3). We provide a detailed summary of individual baselines, comparision with more datasets and hyperparameters in Appendix D.

Performance Metrics across all baselines We work with the batch-wise variant for ONLINESUBMOD (Section 2.3) keeping in line with other methods. To compare various baselines, we utilize Speedup as a relative measure of the training times for each baseline in relation to full batch training. Our goal is to identify a baseline that achieves both high speedup and high test accuracy. We evaluate on budget fractions ( β B t × 100% ) of 10%, 30% and 50%. From Table 2, we observe that our method significantly outperforms baselines in accuracy, with speedup extremely close to the optimal speedup, obtained by MILO which relies on an expensive offline filtering step based on some assorted selection of submodular functions, contrast to our method which dynamically selects subsets in an online manner. However, this gap is minimal, while our method achieves higher accuracy. For completeness, we also showcase per samplewise selection in Figure 3.

Table 2: Batchwise version performance: Accuracy vs Speedup. ↦→ highest accuracy ↦→ 2nd highest accuracy ↦→ 3rd highest accuracy.: Performance comparison across different datasets and fractions. Bold indicates the best performance in each column.

| Method         | TinyImageNet   | TinyImageNet   | TinyImageNet   | CIFAR-100    | CIFAR-100   | CIFAR-100    | CIFAR-10    | CIFAR-10     | CIFAR-10     |
|----------------|----------------|----------------|----------------|--------------|-------------|--------------|-------------|--------------|--------------|
|                | 10%            | 30%            | 50%            | 10%          | 30%         | 50%          | 10%         | 30%          | 50%          |
| CRAIG [32]     | 0.524 / 4.82   | 0.555 / 2.41   | 0.615 / 1.7    | 0.672 / 5.1  | 0.723 / 2.5 | 0.751 / 1.5  | 0.900 / 6.7 | 0.924 / 1.9  | 0.931 / 1.15 |
| MILO [19]      | 0.532 / 8.62   | 0.593 / 3.1    | 0.623 / 2.6    | 0.723 / 10.1 | 0.746 / 3.5 | 0.756 / 1.95 | 0.922 / 5.8 | 0.932 / 2.05 | 0.941 / 2.15 |
| GRADMATCH [18] | 0.526 / 5.92   | 0.581 / 2.62   | 0.619 / 2.1    | 0.683 / 6.9  | 0.746 / 3.1 | 0.753 / 1.3  | 0.922 / 4.3 | 0.932 / 1.95 | 0.941 / 1.48 |
| GLISTER [20]   | 0.515 / 5.5    | 0.563 / 2.65   | 0.621 / 1.7    | 0.642 / 7.7  | 0.723 / 2.6 | 0.746 / 1.2  | 0.911 / 4.5 | 0.921 / 1.7  | 0.926 / 1.3  |
| RHO-LOSS [30]  | 0.544 / 5      | 0.597 / 2.57   | 0.621 / 2      | 0.713 / 3.9  | 0.748 / 1.9 | 0.757 / 1.2  | 0.901 / 2.5 | 0.915 / 1.6  | 0.941 / 1.15 |
| BOSS [1]       | 0.526 / 5.4    | 0.601 / 2.9    | 0.621 / 2.15   | 0.717 / 7.8  | 0.737 / 3   | 0.754 / 1.9  | 0.916 / 4.9 | 0.930 / 1.8  | 0.938 / 1.53 |
| ONLINESUBMOD   | 0.553 / 8.43   | 0.607 / 3.08   | 0.626 / 2.6    | 0.736 / 9.2  | 0.754 / 3.3 | 0.758 / 1.92 | 0.924 / 5.4 | 0.937 / 2    | 0.941 / 2.08 |

Figure 3: Samplewise Submodular Curriculum: ONLINESUBMOD consistently achieves top-1 accuracy across all subset sizes on TINYIMAGENET, SVHN, CIFAR-10, and CIFAR-100, and remains competitive on MNIST. Notably, it matches or outperforms all baselines at early subset fractions (10%, 30%) on all datasets except MNIST.

<!-- image -->

## 5.3 Ablation Study results

To understand how the choice of submodular function at each step affects the model performance under Exploration / Exploitation scheme, it is important to understand how the underlying variables affect the overal submodular function selection and thereby model performance at each step.

Figure 4: Evolution of Term I and Term II

<!-- image -->

(Eq 4) across training epochs on CIFAR-100.

Here we study the effect of the λ ( t ) and π ( t ) . Note for time independent constants we ignore the argument inside λ ( · ) , π ( · ) .

Formally, λ ( t ) : ( Exploration Dampening ) modulates the inertia of exploration. Larger values induce slower increases in Ξ t , prolonging stochastic exploration across arms, while smaller values accelerate convergence to greedy selection. On the other hand π ( t ) ( Exploration Sharpness ) controls the curvature of the annealing schedule. High π ( t ) enforces an abrupt shift to exploitation , while low π ( t ) yields smoother, prolonged exploration phases . As shown in Figure 5, for any λ ( t ) , increasing π leads to a higher degree of exploitation. For effective learning, the policy must exploit frequently while retaining sufficient exploration to ensure coverage of the state space. π = 1 . 5 offers a suitable tradeoff-predominantly exploiting with occasional exploration-whereas π = 1 . 0 explores too uniformly and π = 0 . 5 almost always explores. Hence, π = 1 . 5 emerges as the most effective choice.

Computational overhead of Submodular Optimization: Submodular maximization is NPhard, but most practical solvers use the greedy algorithm, which guarantees a (1 -1 /e ) approximation [35]. Table 3 discusses the tradeoff incurred for the submodular maximization problem w.r.t overall subset selection that involves gradient computation. In our LLM fine-tuning setup on MMLU, LLAMA2-7B using LoRA of rank 128, Table 3 shows that submodular selection takes 0.8 ms on average, while gradient computation takes 630 ms-a 800 × gap.

<!-- image -->

Epoch

Epoch

Figure 5: Arm selection distribution over epochs on CIFAR-100. Diversity based submodular functions become increasingly active during training.

<!-- image -->

Figure 6: Cumulative exploration vs. exploitation choices over time on CIFAR-100. -0.02t -0.02t -0.02t 入(t)=e 入（t)=e 入（t)=e 入(t)=10 入(t)=10 入(t)=10 A(t)= e0.01t A(t)= e0.01t 入(t)= e0.01t

| Computation Breakdown           | Average time   |
|---------------------------------|----------------|
| Gradient Computation            | 630 ms         |
| Submodular Maximization         | 0.8 ms         |
| Total time for Subset Selection | 640 ms         |

Table 3: Runtime breakdown showing submodular selection adds negligible overhead.

Thus, gradient computation remains the primary bottleneck, and submodular selection adds negligible overhead.

## 6 Conclusion

We introduce ONLINESUBMOD, a bandit-guided framework for online submodular subset selection that provides a principled alternative to traditional curriculum learning paradigms. By dynamically optimizing a utility-driven reward function, ONLINESUBMOD effectively balances the trade-off between accuracy and efficiency across diverse training budgets. Our extensive empirical evaluation demonstrates consistent gains over strong state-of-the-art baselines on multiple benchmarks. Future work will focus on extending the proposed greedy utility metric to train neural scoring models, thereby enabling scalable and adaptive subset selection in large-scale pretraining regimes.

## 7 Acknowledgements

We thank the anonymous reviewers for their constructive feedback and insightful suggestions that helped improve the quality of this work. PC acknowledges the Microsoft Research India PhD Award and Prime Minister Research Fellowship to support this research work. GR thanks Bank of Baroda Chair Professorship. We also acknowledge the computing resources provided by the Department of Computer Science and Engineering at IIT Bombay. In addition, we are exceptionally grateful to the BharatGen Initiative 3 for providing compute resources for conducting large scale language model experiments. Finally, we thank our colleagues and collaborators for valuable discussions and feedback throughout the course of this research process.

## References

- [1] Abhinab Acharya, Dayou Yu, Qi Yu, and Xumin Liu. Balancing feature similarity and label variability for optimal size-aware one-shot subset selection. In Forty-first International Conference on Machine Learning .
- [2] Rahaf Aljundi, Klaas Kelchtermans, and Tinne Tuytelaars. Task-free continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 11254-11263, 2019.
- [3] Jordan T Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds. In ICLR , 2020.
- [4] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning , pages 41-48, 2009.
- [5] Andrew A Bian, Joachim M Buhmann, and Andreas Krause. Guarantees for greedy maximization of non-submodular functions with applications. In International Conference on Machine Learning (ICML) , pages 498-507, 2017.
- [6] Lawrence D Brown. Estimation with incompletely specified loss functions (the case of several location parameters). Journal of the American Statistical Association , 70(350):417-427, 1975.
- [7] Daniele Calandriello, Michal Derezinski, and Michal Valko. Sampling from a k-dpp without looking at all items. Advances in Neural Information Processing Systems , 33:6889-6899, 2020.
- [8] Prateek Chanda, Shrey Modi, and Ganesh Ramakrishnan. Bayesian coreset optimization for personalized federated learning. In The Twelfth International Conference on Learning Representations .
- [9] Ethan R Elenberg, Rajiv Khanna, Alexandros G Dimakis, and Sahand Negahban. Restricted strong convexity implies weak submodularity. The Annals of Statistics , 46(6B):3539-3568, 2018.
- [10] Kazuya Fujita, Kensuke Okada, and Kentaro Katahira. The fisher information matrix: A tutorial for calculation for decision making models. 2022.
- [11] Daniel Golovin and Andreas Krause. Adaptive submodularity: Theory and applications in active learning and stochastic optimization. Journal of Artificial Intelligence Research , 42:427-486, 2011.
- [12] László Györfi, Michael Kohler, Adam Krzyzak, Harro Walk, et al. A distribution-free theory of nonparametric regression , volume 1. Springer, 2002.
- [13] Nicholas Harvey, Christopher Liaw, and Tasuku Soma. Improved algorithms for online submodular maximization via first-order regret bounds. Advances in Neural Information Processing Systems , 33:123-133, 2020.

3 BharatGen: http://bharatgen.tech/

- [14] Hamed Hassani, Mahdi Soltanolkotabi, and Amin Karbasi. Gradient methods for submodular maximization. Advances in Neural Information Processing Systems , 30, 2017.
- [15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [16] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023.
- [17] Angelos Katharopoulos and François Fleuret. Not All Samples Are Created Equal: Deep Learning with Importance Sampling. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning (ICML) , volume 80 of Proceedings of Machine Learning Research , pages 2536-2545. PMLR, 10-15 Jul 2018. Also available as arXiv:1803.00942.
- [18] Krishnateja Killamsetty, Sivasubramanian Durga, Ganesh Ramakrishnan, Abir De, and Rishabh Iyer. Grad-match: Gradient matching based data subset selection for efficient deep model training. In International Conference on Machine Learning , pages 5464-5474. PMLR, 2021.
- [19] Krishnateja Killamsetty, Alexandre V Evfimievski, Tejaswini Pedapati, Kiran Kate, Lucian Popa, and Rishabh Iyer. Milo: Model-agnostic subset selection framework for efficient model training and tuning. arXiv preprint arXiv:2301.13287 , 2023.
- [20] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer. Glister: Generalization based data subset selection for efficient and robust learning. In Proceedings of the AAAI conference on artificial intelligence , volume 35, pages 8110-8118, 2021.
- [21] Suraj Kothawade, Vishal Kaushal, Ganesh Ramakrishnan, Jeff Bilmes, and Rishabh Iyer. Prism: Arich class of parameterized submodular information measures for guided data subset selection. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 1023810246, 2022.
- [22] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [23] Yann Le and Xuan Yang. Tiny imagenet visual recognition challenge. CS 231N , 7(7):3, 2015.
- [24] Yann LeCun, Corinna Cortes, Chris Burges, et al. Mnist handwritten digit database, 2010.
- [25] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision , pages 2980-2988, 2017.
- [26] David Lopez-Paz and Marc'Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems , 30, 2017.
- [27] Ilya Loshchilov and Frank Hutter. Online Batch Selection for Faster Training of Neural Networks, 2015.
- [28] Adyasha Maharana, Prateek Yadav, and Mohit Bansal. D2 pruning: Message passing for balancing diversity and difficulty in data pruning. arXiv preprint arXiv:2310.07931 , 2024.
- [29] James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International conference on machine learning , pages 2408-2417. PMLR, 2015.
- [30] Sören Mindermann, Jan M Brauner, Muhammed T Razzak, Mrinank Sharma, Andreas Kirsch, Winnie Xu, Benedikt Höltgen, Aidan N Gomez, Adrien Morisot, Sebastian Farquhar, et al. Prioritized training on points that are learnable, worth learning, and not yet learnt. In International Conference on Machine Learning , pages 15630-15649. PMLR, 2022.

- [31] Baharan Mirzasoleiman, Ashwinkumar Badanidiyuru, Amin Karbasi, Jan Vondrak, and Andreas Krause. Fast constrained submodular maximization: Personalized data summarization. In International Conference on Machine Learning (ICML) , pages 1358-1367, 2016.
- [32] Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In International Conference on Machine Learning , pages 6950-6960. PMLR, 2020.
- [33] Marwa Naïr, Kamel Yamani, Lynda Said Lhadj, and Riyadh Baghdadi. Curriculum learning for small code language models. arXiv preprint arXiv:2407.10194 , 2024.
- [34] Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E Taylor, and Peter Stone. Curriculum learning for reinforcement learning domains: A framework and survey. Journal of Machine Learning Research , 21(181):1-50, 2020.
- [35] George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. An analysis of approximations for maximizing submodular set functions-i. Mathematical programming , 14:265-294, 1978.
- [36] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS Workshop on Deep Learning and Unsupervised Feature Learning , 2011.
- [37] Alex Nichol, Joshua Achiam, and John Schulman. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999 , 2018.
- [38] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems , 33:19920-19930, 2020.
- [39] Michael Rawson and Radu Balan. Convergence guarantees for deep epsilon greedy policy learning. arXiv preprint arXiv:2112.03376 , 2021.
- [40] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. arXiv preprint arXiv:1908.10084 , 2019.
- [41] Tim Roughgarden and Joshua R Wang. An optimal algorithm for online unconstrained submodular maximization. arXiv preprint arXiv:1806.03349 , 2018.
- [42] Levent Sagun, Utku Evci, V Ugur Guney, Yann Dauphin, and Leon Bottou. Empirical analysis of the hessian of over-parametrized neural networks. arXiv preprint arXiv:1706.04454 , 2017.
- [43] Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, Sayna Ebrahimi, and Hao Wang. Continual learning of large language models: A comprehensive survey. arXiv preprint arXiv:2404.16789 , 2024.
- [44] Matthew Staib, Bryan Wilder, and Stefanie Jegelka. Distributionally robust submodular maximization. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 506-516. PMLR, 2019.
- [45] Matthew Streeter, Daniel Golovin, and Andreas Krause. Online learning of assignments. Advances in neural information processing systems , 22, 2009.
- [46] Shivakanth Sujit, Somjit Nath, Pedro Braga, and Samira Ebrahimi Kahou. Prioritizing samples in reinforcement learning with reducible loss. Advances in Neural Information Processing Systems , 36:23237-23258, 2023.
- [47] Artin Tajdini, Lalit Jain, and Kevin Jamieson. Nearly minimax optimal submodular maximization with bandit feedback. Advances in Neural Information Processing Systems , 37:9625496281, 2024.
- [48] Haoru Tan, Sitong Wu, Wei Huang, Shizhen Zhao, and Xiaojuan Qi. Data pruning by information maximization. International Conference on Learning Representations , 2025.

- [49] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zhengxu Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open Foundation and Fine-Tuned Chat Models, 2023.
- [50] Gido M Van de Ven and Andreas S Tolias. Three scenarios for continual learning. arXiv preprint arXiv:1904.07734 , 2019.
- [51] Vladimir Vapnik. Principles of risk minimization for learning theory. Advances in neural information processing systems , 4, 1991.
- [52] Jiachen Tianhao Wang, Tong Wu, Dawn Song, Prateek Mittal, and Ruoxi Jia. Greats: Online selection of high-quality data for llm training in every iteration. Advances in Neural Information Processing Systems , 37:131197-131223, 2024.
- [53] Kai Wei, Yuzong Liu, Katrin Kirchhoff, Chris Bartels, and Jeff Bilmes. Submodular subset selection for large-scale speech training data. In 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 3311-3315. IEEE, 2014.
- [54] Yikai Wu, Xingyu Zhu, Chenwei Wu, Annie Wang, and Rong Ge. Dissecting hessian: Understanding common structure of hessian in neural networks. arXiv preprint arXiv:2010.04261 , 2020.
- [55] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. LESS: Selecting influential data for targeted instruction tuning. In International Conference on Machine Learning (ICML) , 2024.
- [56] Sifan Yang, Yuanyu Wan, and Lijun Zhang. Online nonsubmodular optimization with delayed feedback in the bandit setting. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, pages 21992-22000, 2025.
- [57] Zheng Zhang, Junxiang Wang, and Liang Zhao. Curriculum learning for graph neural networks: Which edges should we learn first. Advances in Neural Information Processing Systems , 36, 2024.
- [58] Haizhong Zheng, Rui Liu, Fan Lai, and Atul Prakash. Coverage-centric coreset selection for high pruning rates. arXiv preprint arXiv:2210.15809 , 2022.
- [59] Yuwei Zhou, Zirui Pan, Xin Wang, Hong Chen, Haoyang Li, Yanwen Huang, Zhixiao Xiong, Fangzhou Xiong, Peiyang Xu, Wenwu Zhu, et al. Curbench: Curriculum learning benchmark. In Forty-first International Conference on Machine Learning .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims in the abstract and introduction are consistent with the technical contributions and empirical results presented in the paper. We clearly state our proposed method, theoretical foundations, and experimental validation, and these are substantiated in the body of the work without overstatement or omission.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: Yes we have discussed the limitations of our work at specific portions of the paper, and have also added in Appendix

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

Justification: The theoretical proofs are provided in the Appendix H

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

Justification: The full codebase is provided along with the supplementary material

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

Justification: The full codebase is provided along with the supplementary material along with running instructions commands in the Readme. Further, we test our algorithm on open source datasets only which we have cited sufficiently and have provided links in the codebase Readme file.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/ guides/CodeSubmissionPolicy) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All specific training details are specified in the Appendix D

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report std error of most our results over 3 runs per baseline on average.

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

Justification: All specific training and compute details are specified in the Appendix D

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: Yes we have reviewed the Code of Ethics Guidelines.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: We have not discussed any potential societal impacts (neither positive nor negative). We do hope that since we are able to show significant efficiency improvement both across different modalities (especially in LLM settings) this may be of siginificant potential impact for Large scale LLM training.

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

Justification: In this work, we are not releasing any generative models or new datasets.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification:

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

Justification:

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Our paper's methodology is not involved in LLM usage and nor is the experimental pipeline.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

## Supplementary Material: Bandit Guided Submodular Curriculum for Adaptive Subset Selection

## Contents

| Appendices       | Appendices                                                       | Appendices                                                                                                      | 21   |
|------------------|------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|------|
| A                | Organization of the Appendix                                     | Organization of the Appendix                                                                                    | 22   |
| B                | Notation Summary                                                 | Notation Summary                                                                                                | 22   |
| C                | Implementation Details                                           | Implementation Details                                                                                          | 22   |
|                  | C.1                                                              | Details about model architectures used . . . . . . . . . . . . . . . . . . . . . . . .                          | 22   |
|                  | C.2                                                              | Details on submodular functions implementation . . . . . . . . . . . . . . . . . . .                            | 23   |
|                  | C.3                                                              | Gradient Computation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                          | 23   |
|                  | C.4                                                              | Evaluation metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                        | 24   |
| D                | Experimental Setup Details                                       | Experimental Setup Details                                                                                      | 25   |
|                  | Software and Hardware                                            | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                   | 25   |
|                  | D.1                                                              | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                     | 25   |
|                  | D.2 D.3                                                          | Language Model Experiments Vision Model Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 25   |
|                  | D.4                                                              | Baseline Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                         | 26   |
|                  | D.5                                                              | Comparison between DINO and Gradient-Based Features for Submodular Selection                                    | 27   |
|                  | D.6                                                              | Fisher Information Matrix . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                         | 27   |
| E                | Additional Experiments                                           | Additional Experiments                                                                                          | 28   |
|                  | E.1                                                              | Sensitivity of Validation Dataset Configuration . . . . . . . . . . . . . . . . . . . .                         | 28   |
|                  | E.2                                                              | Effect of Submodular Functions Individually and RANDOM Selection over Arms .                                    | 28   |
|                  | E.3                                                              | Additional Experiments on Vision datasets . . . . . . . . . . . . . . . . . . . . . .                           | 30   |
|                  | E.4                                                              | Additional Experiments on Large Language Models . . . . . . . . . . . . . . . . .                               | 31   |
| F                | Details of Submodular Function used in all our training settings | Details of Submodular Function used in all our training settings                                                | 32   |
|                  | . . . . . . . . .                                                | . . . . . . . . . . . . . . . .                                                                                 | 32   |
|                  | F.1                                                              | Diversity based Submodular Function                                                                             |      |
|                  | F.2                                                              | Representative based Submodular Function . . . . . . . . . . . . . . . . . . . . . .                            | 32   |
|                  | Additional Related Work                                          | Additional Related Work                                                                                         | 33   |
| G                | G.1                                                              | Online Submodular Maximization . . . . . . . . . . . . . . . . . . . . . . . . . .                              | 33   |
|                  | G.2                                                              | Pruning Mechanisms . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                            | 34   |
| H                | Main Theoretical Results                                         | Main Theoretical Results                                                                                        | 35   |
|                  | H.1                                                              | Proof for permutation invariance of Expected Marginal Gain . . . . . . . . . . . . .                            | 35   |
|                  | H.2                                                              | Theorem: Capacity-Controlled Risk Convergence Theorem . . . . . . . . . . . . .                                 | 35   |
|                  | H.3                                                              | No-Regret Bounds under Constant λ ( · ) . . . . . . . . . . . . . . . . . . . . . . . .                         | 35   |
|                  | H.4                                                              | Regret bounds in the case of growing with time exploration dampening function . .                               | 39   |
| I Broader Impact | I Broader Impact                                                 | I Broader Impact                                                                                                | 42   |

## Supplementary Material: Bandit Guided Submodular Curriculum for Adaptive Subset Selection

## A Organization of the Appendix

The appendix is organized as follows. Section I provides a summary of the impact of our work.Section B provides a summary of the notation used throughout the paper. Section H presents our main theoretical results. Section D outlines the experimental setup and implementation details for both vision and language model tasks. Section G discusses additional related work. Section F describes the various submodular functions employed in our experiments.

## B Notation Summary

Table 4: Table of Notations

| Topic                  | Notation               | Explanation                                                                              |
|------------------------|------------------------|------------------------------------------------------------------------------------------|
| Data (sub)Sets Indices | D train                | Entire Training Set consisting of n instances                                            |
| Data (sub)Sets Indices | D val z i              | Entire Validation Set consisting of m instances i -th training instance in a batch       |
| Data (sub)Sets Indices | B t                    | Denotes the full sized t -th step train minibatch : { x p } &#124;B t &#124; p =1        |
| Data (sub)Sets Indices | B val t                | Denotes the full sized t -th step validation minibatch                                   |
| Data (sub)Sets Indices | B ( <i ) t             | Denotes the t -th step minibatch being constructed uptil x i - 1 i.e. { x p } i - 1 p =1 |
| Data (sub)Sets Indices | S opt ( a t )          | Optimal subset obtained when submodular function f ( a t ) is applied                    |
| Parameters             | θ ∗                    | Optimal model parameter (vector)                                                         |
| Parameters             | θ t                    | Model parameter at t th step                                                             |
| Parameters             | θ t +1                 | Model parameter at ( t +1) th step                                                       |
| Loss Function          | ℓ                      | Strongly convex instance-wise loss function                                              |
| Loss Function          | L t                    | Total loss over mini-batch                                                               |
| Loss Function          | U t ( B t ; z val t )  | Utility metric capturing validation loss drop for a particular validation data point     |
| Loss Function          | U t ( B t ; B val t )  | Aggregated utility metric over validation set                                            |
| Hyperparams            | λ ( t )                | ( Exploration Dampening ) modulates the inertia of explo- ration                         |
| Hyperparams            | ϑ                      | Reward function                                                                          |
| Hyperparams            | Ξ t                    | Exploration-exploitation threshold                                                       |
| Hyperparams            | F div sub , F repr sub | Diversity/representative function subsets                                                |
| Hyperparams            | ζ ∼ Uniform (0 , 1)    | Random sample for trade-off                                                              |
| Hyperparams            | π ( t )                | ( Exploration Sharpness ) controls the curvature of the anneal- ing schedule rule        |

## C Implementation Details

## C.1 Details about model architectures used

## Vision Model Architecture Details:

The ResNet18 model [15] architecture begins with a basic block , which is composed of two main sections. The first section consists of a convolution layer followed by a batch normalization layer, and then a ReLU activation function. The second section similarly comprises a convolution layer followed by batch normalization. This entire basic block is repeated twice for each of the four layers in the network. These layers progress with input dimensions of [64 , 128 , 256 , 512] to form the complete ResNet18 architecture.

## Language Model Architecture Details

The LLaMA-2-7B model is a decoder-only transformer comprising approximately 7 billion parameters. It includes 32 transformer layers, each built with pre-normalization using RMSNorm and employing the SwiGLU activation function. The self-attention mechanism uses multi-head causal attention with 32 heads and a hidden dimensionality of 4096. Rotary positional embeddings (RoPE) are applied to the query and key vectors within each attention head. The model begins with a learned token embedding layer and concludes with a tied output projection layer to predict the next token. Mistral-7B-v0.3 is architecturally similar to LLaMA-2-7B, also featuring 32 transformer layers and a 4096-dimensional hidden state, but introduces several efficiency-focused modifications. It uses grouped-query attention (GQA) with 8 query groups across 32 heads, improving inference throughput. Furthermore, Mistral replaces full causal attention with sliding-window attention to handle long contexts more efficiently. As with LLaMA, it utilizes RoPE for positional encoding and SwiGLU activations. These optimizations maintain strong modeling performance while enabling greater scalability in both training and inference settings.

## C.2 Details on submodular functions implementation

We provide detailed formulations of the specific submodular functions employed as arms in our experiments in Section F. From an implementation standpoint, each submodular arm operates on a similarity kernel computed over the set of instances within a given batch. This kernel, typically represented as a symmetric positive semi-definite matrix, encodes pairwise affinities between samples based on their embedding representations (e.g., cosine similarity or RBF kernel). Once the similarity structure is established, any submodular function can be instantiated over this ground set-such as facility location, log-determinant, or graph-cut functions-depending on the desired coverage, diversity, or representativeness property being optimized.

To operationalize this, we leverage the Submodlib library 4 , an open-source framework maintained by the Decile organization 5 , which provides efficient and modular implementations of a wide family of submodular functions. The library supports both dense and sparse similarity representations and includes greedy as well as lazy-greedy optimization routines, enabling scalable computation even for large batch sizes.

## C.3 Gradient Computation

Computing full-model gradients in modern deep networks is computationally prohibitive due to the extremely high dimensionality of parameter spaces-often exceeding billions of parameters for large vision or language models. Moreover, for the purpose of subset selection, what is typically required is not the full parameter gradient but an informative proxy that captures the relative contribution of individual samples to the model's training dynamics.

Following this motivation, we adopt partial-gradient approximations that preserve discriminative signal while substantially reducing computational cost. Specifically, for vision models, we compute gradients only with respect to the last linear classification layer , as in [3]. This choice leverages the empirical observation that gradients in earlier layers are highly correlated and redundant, and that last-layer gradients retain sufficient information to distinguish hard, redundant, or noisy samples based on their contribution to the decision boundary.

For large language models (LLMs), computing full backpropagation across all transformer layers is infeasible. We therefore restrict gradient computation to Low-Rank Adaptation (LoRA) adapter parameters (rank 128), following the setup of [52]. This approach not only reduces memory and compute overhead by several orders of magnitude but also captures localized curvature information relevant to the fine-tuning or instruction-following objective. Since LoRA adapters are trained in the low-dimensional subspace most sensitive to task adaptation, their gradients provide a faithful and low-noise estimate of per-sample learning signals.

Importantly, both approximations maintain gradient informativeness under the assumption that the selected subspace (last layer or adapter) spans the most discriminative directions of parameter updates. Prior empirical evidence (see [3, 52]) shows that subset selection, influence estimation, and sample

4 https://submodlib.readthedocs.io/en/latest/

5 https://decile.org/

reweighting methods computed in these reduced spaces closely match those computed with full gradients. In our experiments, we verify that this approximation incurs negligible performance degradation while providing up to 30 × faster per-batch computation. Thus, the proposed gradient computation scheme achieves a favorable balance between computational efficiency and fidelity of learning signal for submodular subset selection.

Warm-starting Data Selection. A common challenge in data subset selection methods lies in the instability of early-stage gradients. During the initial epochs of training, model parameters are far from any local minimum, and per-sample gradients tend to be highly noisy and uninformative. Consequently, performing subset selection too early can result in biased or suboptimal subsets that fail to represent the underlying data distribution or learning dynamics. To mitigate this issue, for the image experiments, we conduct a warm-start strategy, wherein the model is first trained for a small number of epochs on the full dataset before invoking any subset selection procedure.

Concretely, for each algorithm considered in this paper (i.e., ONLINESUBMOD GRADMATCH, GRADMATCHPB, CRAIG, CRAIGPB, and GLISTER,), we include a warm-start variant. Let T denote the total number of training epochs and k the subset size. We define two quantities: T f (the number of full-training epochs prior to subset selection) and T s (the number of epochs during which subset selection is active). We set these in proportion as T s = κT,T f = T s k n , where n is the total number of training samples and κ ∈ (0 , 1] is the fraction of total training epochs used for subset selection. This parametrization ensures that the effective compute budget remains comparable across methods, while allowing early-stage training to stabilize the model representation before adaptive data selection begins.

Empirically, we observe that performing a few epochs of full-data warm-up ( T f ) consistently improves convergence stability and downstream accuracy across all subset selection algorithms. The warmstart phase enables the gradient space to form a meaningful geometry, allowing the submodular or gradient-based selection objectives to more accurately identify informative and diverse samples. In contrast, starting selection from random initialization often leads to premature overfitting or unstable subset composition due to noisy or poorly conditioned gradients.

Setting T f too large, however, diminishes the benefit of subset selection, as the model effectively performs full training with minimal adaptive sampling. In this limit, the behavior approaches that of the full-batch baseline with early stopping, which we include as a control setting in our experiments. Thus, the warm-start scheme provides a principled balance between computational efficiency and representational stability-retaining the benefits of subset-based training while ensuring robust and smooth convergence.

## C.4 Evaluation metrics

Image Classification : For image classification experiments, we report the standard test accuracy as the primary performance metric, measured as the proportion of correctly classified samples on the held-out validation or test split. This metric provides a direct and interpretable indicator of the model's generalization performance under different subset selection strategies.

In addition to accuracy, we evaluate the computational efficiency of our method by comparing the total training time required to reach convergence across different selection policies. To ensure a fair comparison, all other hyperparameters-including optimizer configuration, learning rate schedule, batch size, and data augmentations-are held fixed across runs. The only varying factor is the subset selection mechanism applied at each training step.

We define the speedup metric with respect to the baseline model trained using full-batch selection (i.e., without any submodular or adaptive sampling). Formally, if T full denotes the wall-clock training time for the full-batch model and T sub denotes the time under our submodular selection strategy, the speedup is given by Speedup = T full T sub . A higher speedup thus indicates a more efficient training regime, achieved without sacrificing downstream accuracy. In practice, we observe consistent gains in training efficiency-typically in the range of 3 × -8 × -depending on the dataset and the choice of submodular objective, confirming that adaptive selection substantially reduces redundant gradient computations while maintaining comparable predictive performance.

## D Experimental Setup Details

## D.1 Software and Hardware

Vision Experiments All experiments were conducted using Python 3.10.13 and PyTorch 2.1.2. Our proposed methods, ONLINESUBMOD and ONLINESUBMOD-Batch, along with their corresponding ablations, were trained on NVIDIA RTX A6000 GPUs (48 GB). Baseline methods, including RHO-LOSS and BOSS, were also trained using the same GPU configuration to ensure comparability.

For reference, a typical training run of our ResNet18-based model on an RTX A6000 consists of 300 epochs, with each epoch averaging approximately one minute (excluding certain baselines). Model checkpointing is employed to retain only the best-performing model based on validation accuracy, as well as the final model. Running multiple training jobs concurrently on the same GPU incurs only a slight overhead in training time due to resource contention.

## D.2 Language Model Experiments

We experiment for the LLM finetuning setup using a RANDOM subset of 9 datasets from MMLU, and on TydiQA. We choose Sociology, Policy, History, Anatomy, ML, Ehics, Genetics, High School Biology, High School Chemistry. All language model experiments, including both our proposed methods and the baselines, were conducted using 8 NVIDIA H100 GPUs. Additionally, Weights &amp; Biases (WandB) 6 wandb was used to manage and monitor all experiments. For all experiments we take batch size of 16, initial learning rate of 2e-5 using adam optimizer with default state, finetuned on 10% of LESS[55] version of OpenWebText.

Additional Experiment Results : For MMLU we also showcase additional experiments on LLaMa27b and Mistral-7b for TydiQA, later in the appendix.

Mathematical definitions of the submodular objectives used as arms are provided in Appendix F. For this experiment, each arm is a mutual information variant of a classical submodular function, designed to maximize I f ( X ; Q ) = f ( X ) + f ( Q ) -f ( X ∪ Q ) , where X is the candidate training set, Q is the validation set, and f is a base submodular function (Facility-Location, Graph-Cut, or Log-Determinant).

We use mutual information forms to ensure the selected subset is explicitly conditioned on the current validation set, making the acquisition process adaptive to the downstream task. Features for X and Q are derived either from Sentence-BERT embeddings or from gradient vectors, with the latter shown to yield better alignment with task-specific error signals and improved selection performance.

## D.3 Vision Model Experiments

The experimental setup was configured to evaluate the proposed method on several datasets, including CIFAR-10, CIFAR-100, Tiny-ImageNet-200, and SVHN. The data module used a batch size of 128, with four workers for data loading. The model architecture employed was ResNet18 [15], and the training followed a curriculum-based mode, progressively utilizing 10%, 30%, and 50% of the training data. The optimizer used was SGD with a learning rate of 0.05, momentum of 0.9, weight decay of 0.0005, and Nesterov momentum enabled.

For all our settings (across different baselines and dataset), we consider ResNet18 [15] as our primary model with the following architecture and training details:

In our training setup, we employed batch-wise Nesterov accelerated gradient descent with a batch size of 128. The optimization configuration included a learning rate of 0.05 and a momentum of 0.9, alongside a cosine-annealing scheduler.

Across all dataset comparisons, we set the submodular function budget β to 10%, 30%, and 50% of the entire batch size.

Dataset Specifics. We conduct experiments across a range of standard vision benchmarks. For the MNIST dataset, we use 60,000 training instances, 10,000 test instances, and 10,000 validation instances, with training proceeding until full convergence, typically around 200 epochs . On CIFAR-

6 https://wandb.ai/site/

10, we use 50,000 training instances, 10,000 test instances, and 10,000 validation instances, with models trained for up to 300 epochs . For CIFAR-100 , we similarly use 50,000 training examples spread across 100 classes (500 per class), and a validation set of 10,000 examples (100 per class). The SVHN dataset comprises 73,257 training images across 10 classes with variable class frequencies, and a validation set of 26,032 images distributed proportionally. Finally, for TINYIMAGENET , we use 100,000 training images across 200 classes (500 per class), and a validation set of 10,000 images (50 per class), covering the same label space as the training data.

## D.4 Baseline Training Details

We compare our method ONLINESUBMOD with several state of the art baselines for our experiments:

MAX-LOSS [27]: Within each training batch, the loss is computed for every example. A fixed fraction (e.g., top-K%) of samples with the highest per-example loss is selected for gradient computation and model update. This assumes that high-loss samples are currently mis-predicted and could contribute the most to updating the decision boundary.

GRADNORM [17]: The l 2 norm of each example's per-sample gradient i.e. ∥∇ θ L ( z ; θ ) ∥ 2 is computed, and a subset with the highest norms is selected for each batch. This prioritizes examples inducing the largest parameter updates under the current model, helping direct learning toward sensitive or uncertain regions.

RHO-LOSS [30]: Each example's reducible loss is estimated as the difference between the current model's loss and its irreducible loss , the latter approximated by a small auxiliary model trained on held-out clean data. Examples with high reducible loss are selected, as they are considered learnable but not yet learned, making them useful for continued training. For LLM experiments, we use LLaMa3-7b-instruct as our auxiliary model. For image experiments, we begin by training an irreducible model on the specific task for 100 epochs. Subsequently, we precompute the irreducible losses for the training set, which are required for the target model training. During the target model training phase, we train the model for 300 epochs across the CIFAR-10, CIFAR-100, SVHN, and TINYIMAGENET datasets, using subset ratios of 0.1, 0.3, and 0.5. We employ the ResNet-18 architecture for both the irreducible model and the target model. For training, we use the SGD optimizer with Nesterov accelerated gradient descent, a batch size of 128, and the following configuration: a learning rate of 0.05, momentum of 0.9, and a cosine-annealing scheduler. One observation we made is that RHO Loss converges to reasonably good accuracy within a few epochs, but further training does not significantly improve performance, and it fails to reach the accuracy levels of other state-of-the-art (SOTA) baselines.

SBERT [40]: In this case, each training and validation example is encoded into a sentence embedding using a pre-trained SBERT model. Cosine similarity is computed between training examples and the validation set, and those with the highest average similarity are selected. This favors examples that align semantically with the validation distribution.

For GREATS [52], each example's impact on the loss is approximated using a first-order Taylor expansion of the objective. For model parameters θ and a batch { x 1 , . . . , x n } , gradients ∇L ( x i ; θ ) are used to estimate loss reduction. A greedy selection strategy then chooses the subset expected to most decrease validation loss under this approximation.

For fair comparison against our model we considered the configuration where subset selection happens at every epoch for all the 3 baselines with a lazy optimizer. Due to our multi-class image classification setup we utilise CrossEntropy loss for our model training.

BOSS [1]: For BOSS, to select the subset, we first initialized a model by training it using the full dataset. With the help of the training dynamics obtained from the initialized model, we calculated the difficulty score for each sample that is used to select the subset. We evaluated the selected subset keeping the subset fixed and using it to train a new RANDOM initialized model. For the difficulty score, we experimented using the EL2N score because it can be efficiently computed early on during training.We trained the model for 300 epochs across the CIFAR-10, CIFAR-100, SVHN and TINYIMAGENET datasets, using subset ratios of 0.1, 0.3, and 0.5. We employed ResNet18 model using SGD with a learning rate of 0.1, and momentum of 0.9 with a batch size of 128.

## D.5 Comparison between DINO and Gradient-Based Features for Submodular Selection

To evaluate how closely our feature representations must align with the downstream objective, we compared two ways of representing each training item when optimising submodular acquisition functions (and their mutual-information variants):

<!-- image -->

Time

Figure 7: Comparison of Fashion-MNIST with DINO-embeddings, and with Gradient Features for submodular optimization

1. DINO embeddings . We obtain a fixed d -dimensional feature vector for every image by running it through a frozen DINO vision transformer, exactly as one would use a CLIP encoder. These representations are task-agnostic and remain static throughout training.
2. Gradient-based features . At every training step we compute the gradient of the scalar loss with respect to the parameters of the final layer. We average these per-example gradients within the mini-batch to form a single vector 7 . For mutual-information objectives, validation gradients serve as the query features.

Figure 7 shows that gradient features yield substantially higher test accuracy on FASHION-MNIST across all subset sizes: they encode task-specific error signals that guide the submodular optimiser toward examples most useful for loss reduction, whereas DINO embeddings capture only generic visual similarity. Hence, directly leveraging gradients as features is the more effective choice for data subset selection in this setting.

## D.6 Fisher Information Matrix

Fisher Information Matrix Approximation An alternative and potentially more informative approach to approximating the Hessian is through the use of the Fisher Information Matrix (FIM) [10]. The FIM provides insights into the curvature of the loss landscape and can serve as a useful surrogate for the Hessian. While the exact computation of the FIM requires calculating an expectation, which can be computationally intensive, it can be efficiently approximated using an exponential moving average of the outer product of the gradients from the validation data points.

Let Ω i := g ( z i , θ t ) g ( z i , θ t ) ⊤ denote the outer product of the gradient for the i th data point in the current batch B t . The

( t )

approximate FIM ̂ H B t at time step t for the current mini-batch B t can be computed recursively as where ̂ H ( t -1) B t -1 is the approximate FIM at the previous time step t -1 for the mini-batch B t -1 , and α ∈ (0 , 1] is the smoothing

<!-- formula-not-decoded -->

parameter for the exponential moving average. This recursive formulation provides a computationally efficient approach to approximating the Hessian, particularly in high-dimensional settings where direct Hessian computation is prohibitively expensive.

7 Using the batch-wise average was consistently superior to concatenating per-example gradients, and lastlayer gradients are sufficient while keeping the computation inexpensive.

## E Additional Experiments

## E.1 Sensitivity of Validation Dataset Configuration

To better understand the influence of validation data composition on model performance, we investigate how sensitive the algorithm is to different validation set configurations. In particular, we examine what occurs during the early stages of training when the validation dataset includes harder samples that the model has not yet adequately learnt. This analysis provides a deeper perspective on how the distribution and difficulty of validation examples can affect optimization dynamics and generalization, thereby strengthening the empirical validity of our findings.

Specifically, we aim to understand the following questions:

- Q1: To what extent is the algorithm's performance sensitive to the configuration and composition of the validation dataset?
- Q2: How does the presence of harder, yet-unlearned samples in the validation set during early training stages affect convergence and generalization?

To better understand this issue, we conducted a controlled experiment on CIFAR-100 (300 epochs) where we varied the hardness of the validation dataset. Hardness was measured via gradient norm where the gradient is calculated w.r.t model parameter at that time step. In accordance with other literature, a crude way to approximate difficulty of a sample is to check if the gradient norm is high. (higher gradient norm ∼ harder example). We compared four validation subset configurations:

- Easiest: Lowest gradient norms
- EasyHard: Easy samples early, hard samples later
- HardEasy: Hard samples early, easy samples later
- Hardest: Highest gradient norms

Each configuration was evaluated at validation subset sizes of 10%, 20%, and 30%.

## Observations:

Validation sets composed of the most difficult examples tend to yield lower performance, particularly when smaller subsets are used. This decline likely stems from noisy or overly pessimistic reward signals during the early stages of training. In contrast, mixed validation configurations such as EasyHard and HardEasy generally perform best, indicating that a balanced distribution of sample difficulty across training can enhance robustness. These findings suggest that further exploring how validation sample difficulty and ordering interact especially through the lens of curriculum learning could be a promising direction for future work. Importantly, even validation sets containing difficult samples early in training do not lead to instability or model collapse.

## E.2 Effect of Submodular Functions Individually and RANDOM Selection over Arms

To assess the contribution of the multi-armed bandit formulation in our framework, we perform ablation experiments on CIFAR-100 (10% subset, 300 epochs) under two simplified settings: (a) using a single, fixed submodular arm throughout training (i.e., no bandit-driven adaptation), and (b)RANDOMly selecting an arm at each round (i.e., no explore-exploit balancing). These ablations isolate the effect of static versus dynamic subset selection policies on training efficiency and generalization.

Table 5: Final test accuracies under different validation subset configurations.

| Validation Subset   |   10% |   20% |   30% |
|---------------------|-------|-------|-------|
| Easiest             | 72.3  |  74.4 | 76.03 |
| EasyHard            | 73.1  |  74.5 | 76.4  |
| HardEasy            | 72.29 |  74.6 | 76.2  |
| Hardest             | 71.31 |  74.3 | 75.9  |

We compare various submodular selection strategies that define the reward structure of the curriculum. Representative functions (e.g., GraphCut , FacilityLocation ) promote coverage and ensure the selected subset reflects the global data distribution, while Diversity -oriented functions (e.g., DisparitySum , LogDeterminant ) encourage maximal dissimilarity among chosen samples. These functions capture different inductive biases: representation versus decorrelation.

| Selection Strategy              |   Accuracy (%) |
|---------------------------------|----------------|
| DisparitySum (Div., Static)     |           68.6 |
| FacilityLocation (Rep., Static) |           72   |
| LogDeterminant (Div., Static)   |           71.1 |
| GraphCut (Rep., Static)         |           72.6 |
| Random arm per round            |           72   |
| ONLINESUBMOD (ours)             |           73.6 |

Table 6: Performance comparison of individual and RANDOM arm selection strategies on CIFAR-100. ONLINESUBMOD adaptively balances diversity and representativeness over training epochs.

Our proposed ONLINESUBMOD method dynamically alternates between these functions through an adaptive explore-exploit policy governed by the bandit controller. This dynamic weighting enables the model-whether a ResNet-18 backbone or a small LLMfine-tuning setup -to exploit high-yield submodular arms while continually exploring others that may improve validation loss or perplexity.

## Observations:

Different submodular functions show complementary but limited strengths. Coverage-based methods such as GraphCut and FacilityLocation converge quickly early in training, while diversitybased ones like DisparitySum and LogDet encourage better generalization but can become unstable when applied uniformly. Random arm selection gives reasonable results, suggesting that diversity matters, but it lacks feedback to adapt to validation performance. In contrast, the proposed ONLINESUBMOD approach adjusts arm selection based on past rewards, maintaining a stable balance between exploration and exploitation. This supports our main hypothesis that adaptive, reward-driven selection leads to more robust and generalizable outcomes than static or random strategies.

## E.3 Additional Experiments on Vision datasets

Table 7 summarizes batchwise data selection results across multiple vision datasets. Across all budgets and datasets, ONLINESUBMOD consistently achieves the highest test accuracy while maintaining competitive or lower training time compared to prior methods. Notably, it surpasses strong baselines such as GRADMATCH, MILO, and RHO-LOSS, particularly at low data budgets (10%-30%), indicating superior sample efficiency and adaptivity under constrained training regimes. The improvement is most pronounced on CIFAR100 and TINYIMAGENET, where the model benefits from dynamic online selection over diverse feature manifolds. In contrast, static coreset-based methods (e.g., CRAIG, GLISTER) exhibit slower convergence and lower performance as the budget increases.

Here, red highlights the best result and blue denotes the second-best result for each setting. Overall, these results confirm that ONLINESUBMOD provides a strong trade-off between accuracy and computational efficiency across datasets of varying complexity.

Table 7: (Batchwise) Data Selection Results on Vision Datasets

| Dataset      | Selection Strategy                 | Test accuracy (%)   | Test accuracy (%)   | Test accuracy (%)   | Training time (hrs)   | Training time (hrs)   | Training time (hrs)   |
|--------------|------------------------------------|---------------------|---------------------|---------------------|-----------------------|-----------------------|-----------------------|
|              | Budget(%)                          | 10%                 | 30%                 | 50%                 | 10%                   | 30%                   | 50%                   |
| CIFAR10      | FULL (skyline for test accuracy)   | 95.09               | 95.09               | 95.09               | 1.73                  | 1.73                  | 1.73                  |
|              | RANDOM (skyline for training time) | 77.49               | 89.62               | 91.85               | 0.29                  | 0.75                  | 0.85                  |
|              | CRAIG                              | 90.07               | 92.4                | 93.12               | 0.26                  | 0.62                  | 1.54                  |
|              | GLISTER                            | 91.15               | 92.18               | 92.65               | 0.38                  | 1.05                  | 1.34                  |
|              | GRADMATCH                          | 92.27               | 93.28               | 93.15               | 0.42                  | 0.95                  | 1.21                  |
|              | MILO                               | 92.25               | 93.21               | 94.16               | 0.34                  | 0.85                  | 0.89                  |
|              | RHO-LOSS                           | 90.16               | 91.54               | 94.03               | 0.76                  | 1.13                  | 1.54                  |
|              | BOSS                               | 91.64               | 93.04               | 93.8                | 0.36                  | 0.94                  | 1.18                  |
|              | ONLINESUBMOD (ours)                | 92.44               | 93.75               | 94.18               | 0.32                  | 0.87                  | 0.83                  |
| CIFAR100     | FULL (skyline for test accuracy)   | 76.8                | 76.8                | 76.8                | 1.52                  | 1.52                  | 1.52                  |
|              | RANDOM (skyline for training time) | 35.03               | 61.93               | 64.67               | 0.15                  | 0.42                  | 0.78                  |
|              | CRAIG                              | 67.25               | 72.38               | 73.12               | 0.31                  | 0.62                  | 1.12                  |
|              | GLISTER                            | 64.27               | 72.36               | 74.62               | 0.26                  | 0.57                  | 1.3                   |
|              | GRADMATCH                          | 68.34               | 74.63               | 72.36               | 0.22                  | 0.48                  | 1.22                  |
|              | MILO                               | 72.36               | 74.66               | 75.60               | 0.15                  | 0.44                  | 0.82                  |
|              | RHO-LOSS                           | 71.37               | 74.82               | 75.74               | 0.53                  | 0.86                  | 1.46                  |
|              | BOSS                               | 71.73               | 73.77               | 75.41               | 0.27                  | 0.53                  | 0.85                  |
|              | ONLINESUBMOD (ours)                | 73.67               | 75.46               | 75.78               | 0.165                 | 0.47                  | 0.82                  |
| SVHN         | FULL (skyline for test accuracy)   | 96.49               | 96.49               | 96.49               | 6.436                 | 6.436                 | 6.436                 |
|              | RANDOM (skyline for training time) | 93.47               | 95.31               | 95.84               | 0.6383                | 1.90                  | 3.19                  |
|              | CRAIG                              | 95.27               | 96.15               | 96.40               | 0.934                 | 2.332                 | 4.17                  |
|              | GLISTER                            | 95.52               | 95.69               | 96.42               | 0.83                  | 2.42                  | 4.26                  |
|              | GRADMATCH                          | 95.64               | 96.4                | 96.42               | 0.789                 | 2.398                 | 4.19                  |
|              | MILO                               | 95.62               | 96.36               | 96.41               | 0.69                  | 2.09                  | 3.25                  |
|              | RHO-LOSS                           | 94.64               | 94.27               | 94.85               | 1.08                  | 2.56                  | 3.94                  |
|              | BOSS                               | 94.31               | 95.75               | 96.01               | 0.76                  | 2.39                  | 3.56                  |
|              | ONLINESUBMOD (ours)                | 95.68               | 96.38               | 96.46               | 0.68                  | 2.12                  | 3.28                  |
| TINYIMAGENET | FULL (skyline for test accuracy)   | 64.36               | 64.36               | 64.36               | 15.4                  | 15.4                  | 15.4                  |
|              | RANDOM (skyline for training time) | 19.61               | 35.68               | 43.84               | 1.82                  | 4.92                  | 6.12                  |
|              | CRAIG                              | 52.42               | 55.56               | 61.48               | 3.27                  | 6.46                  | 9.23                  |
|              | GLISTER                            | 51.54               | 56.37               | 62.15               | 2.84                  | 5.93                  | 9.47                  |
|              | GRADMATCH                          | 52.63               | 58.19               | 61.93               | 2.63                  | 5.94                  | 7.24                  |
|              | MILO                               | 53.24               | 59.36               | 62.28               | 1.81                  | 4.97                  | 6.16                  |
|              | RHO-LOSS                           | 54.46               | 59.78               | 62.15               | 3.16                  | 6.38                  | 7.94                  |
|              | BOSS                               | 52.63               | 60.17               | 62.13               | 2.85                  | 5.47                  | 7.26                  |
|              | ONLINESUBMOD (ours)                | 55.3                | 60.74               | 62.58               | 1.84                  | 5.16                  | 6.14                  |

## E.4 Additional Experiments on Large Language Models

We evaluate the evolution of test perplexity during pretraining on the MMLU benchmark using the LLAMA-2-7B model under different online batch selection strategies . Each method is trained under identical hyperparameter and compute budgets to ensure fair comparison.

Test Perplexity (Policy)

<!-- image -->

Test Perplexity (Anatomy)

Figure 8: Test perplexity dynamics on LLAMA-2-7B during training with various online batch selection strategies on MMLU . ONLINESUBMOD significantly outperforms baselines.

## F Details of Submodular Function used in all our training settings

We describe here the submodular functions we broadly utilised for all our experiments.

## F.1 Diversity based Submodular Function

Here we share the details on the diversity based submodular functions we used for our training purposes.

Definition 1. Log-determinant Function is a diversity-based submodular function. It is nonmonotone in nature. Let L denote a positive semidefinite kernel matrix and L S denote the subset of rows and columns indexed by set S . Log-determinant function f is specified as:

<!-- formula-not-decoded -->

The log-det function models diversity and is closely related to a determinantal point process.

Definition 2. Disparity Sum Function characterizes diversity by considering the sum of distances between every pair of points in a subset S . For any two points i, j ∈ S , let d ij denote the distance between them.

<!-- formula-not-decoded -->

The aim is to select a subset S such that f ( S ) is maximized.

Definition 3. Disparity Min Function characterizes diversity by considering the minimum distance between any two non-similar points in a subset S .

̸

<!-- formula-not-decoded -->

The aim is to select a subset S such that f ( S ) is maximized.

## F.2 Representative based Submodular Function

Here we share the details on the representative based submodular functions we used for our training purposes.

Definition 4. Facility Location Function characterizes the representativeness in the dataset by considering the minimum distance between any two non-similar points in a subset S .

<!-- formula-not-decoded -->

The aim is to select a subset S such that f ( S ) is maximized.

Definition 5. Graph Cut Function characterizes representativeness by using the parameter λ which governs the tradeoff between representation and diversity. When λ becomes large, graph cut function also tries to model diversity in the subset. S .

<!-- formula-not-decoded -->

The aim is to select a subset S such that f ( S ) is maximized.

Submodular Mutual Information We first provide a definition of Submodular Mutual Information:

<!-- formula-not-decoded -->

Definition 6. Log-Determinant Mutual Information Function is an instantiation of a submodular mutual information function using a LogDeterminantFunction . Let S A,B be the crosssimilarity matrix between the items in sets A and B . Also, denote S AB = S A ∪ B . We construct a similarity matrix S η (on a base matrix S ) such that the cross-similarity between A and Q is multiplied by η (i.e., S η A,Q = ηS A,Q ) to control the trade-off between query relevance and diversity. Higher values of η ensure greater query-relevance while lower values favor diversity. Using a similarity matrix defined above and with f ( A ) = log det( S η A ) , we have:

<!-- formula-not-decoded -->

Definition 7 ( Generalized Submodular Mutual Information ) . Let Ω be a ground set and V ⊆ Ω be a domain of interest. Let f : 2 Ω → R ≥ 0 be a restricted submodular function , i.e., submodular when restricted to subsets of V . A Submodular Mutual Information (SMI) function defined via such a function f is called a Generalized Submodular Mutual Information (GMI) function.

A notable instance of GMI is the Concave Over Modular (COM) function [21], defined for subsets A ⊆ V and Q ⊆ V ′ as:

<!-- formula-not-decoded -->

where η ∈ R ≥ 0 controls the trade-off between query-relevance and diversity, ψ : R ≥ 0 → R ≥ 0 is a concave function, S = [ s ij ] is a kernel similarity matrix such that s ij = ✶ ( i = j ) for i, j ∈ V or i, j ∈ V ′ .

Definition 8 (Facility Location Mutual Information (FL1MI)) . Let I f ( A ; Q ) denote a Submodular Mutual Information (SMI) function. An instantiation of SMI using the FacilityLocationFunction is known as the Facility Location Mutual Information (FL1MI) function.

Formally, given subsets A ⊆ V and Q ⊆ V ′ , FL1MI is defined as:

<!-- formula-not-decoded -->

where: η ∈ R ≥ 0 is a relevance-diversity trade-off parameter, s ij denotes similarity between elements i and j in the kernel similarity matrix S , V is the candidate set and V ′ is the query set domain.

Definition 9 (Graph Cut Mutual Information (GCMI)) . Let I f ( A ; Q ) denote a Submodular Mutual Information (SMI) function. An instantiation of SMI using the GraphCutFunction is called the Graph Cut Mutual Information (GCMI) function.

Formally, for subsets A ⊆ V and Q ⊆ V ′ , GCMI is defined as:

<!-- formula-not-decoded -->

where λ ∈ R ≥ 0 controls the scale of mutual information, s ij denotes the similarity between elements i and j in the kernel similarity matrix S , V is the candidate set and V ′ is the query set domain.

## G Additional Related Work

## G.1 Online Submodular Maximization

A growing body of research has advanced our understanding of online submodular maximization under diverse feedback models and constraint classes. A notable contribution is the recent work on [13], which introduces a principled framework leveraging first-order regret bounds from online linear optimization to derive improved guarantees in submodular settings. At each round t , the algorithm selects a feasible set S t ∈ C ⊆ 2 V , where C encodes combinatorial constraints such as matroids or cardinality bounds, and observes an adversarially chosen submodular function f t . For monotone submodular functions under matroid constraints, the method achieves a (1 -c/e -ϵ ) -approximate regret bound of O ( kT log( n/k )) , improving on earlier results by Streeter and Golovin [45], and Golovin and Krause [11], even in the absence of curvature (i.e., c = 1 ). For non-monotone unconstrained submodular functions, a novel algorithm based on Blackwell approachability achieves a 1 / 2 -regret of O ( n √ T ) , extending Roughgarden and Wang [41].

These developments complement recent advances in bandit and semi-bandit feedback models, including those by Hassani et al. [14] and [47], who analyze online submodular optimization in stochastic and adversarial environments, obtaining nearly minimax optimal regret bounds. Related efforts have also explored limited feedback settings with structure-aware exploration strategies (e.g., combinatorial Thompson sampling or optimism-based approaches), enhancing sample efficiency in large-scale decision spaces.

In the full-information setting, the online continuous greedy algorithm of Bian et al. [5] offers near-optimal (1 -1 /e ) -regret for monotone submodular functions under matroid constraints, while extensions [56] tackle delayed feedback. Meanwhile, online versions of lazy greedy [31] and distributionally robust submodular maximization [44] have enabled scalable implementations in real-world domains such as streaming recommendation and dataset summarization.

Although submodular maximization is NP-hard in general, approximation algorithms often yield near-optimal performance in practice across diverse applications [45, 11]. These include influence maximization, budget-constrained recommendation, and online resource allocation, all of which benefit from the expressive yet structured nature of submodular objectives. As such, the aforementioned theoretical advances not only deepen our algorithmic understanding but also broaden the applicability of online submodular optimization frameworks to practical domains involving limited feedback, combinatorial constraints, and dynamic inputs.

## G.2 Pruning Mechanisms

Several recent works have explored data pruning and subset selection for efficient training, including D2PRUNING [28], INFOMAX [48], and CCS [58]. While these methods offer valuable insights into coreset selection and dataset reduction, they predominantly operate in static, full-dataset settings, in contrast to our dynamic, batch-level framework.

INFOMAX formulates an objective that can be interpreted as a reformulation of the Graph Cut function, which is monotone submodular, and leverages similarity kernels such as DINO embeddings (for image tasks) or gradient-based features. This aligns conceptually with our approach, where Graph Cut is explicitly implemented as a bandit arm. However, INFOMAX selects samples over the entire dataset in a static manner, whereas our framework is modular and dynamic: an InfoMax-like objective can be treated as a bandit arm and applied in batch-level pruning during training. This flexibility enables more scalable deployment in real-world training pipelines where adaptive, online selection is crucial.

D2PRUNING frames data subset selection as a subgraph pruning task over the dataset, representing data points as nodes in a similarity graph. Selection is performed using message passing algorithms, which can be computationally intensive and challenging to scale to large datasets. Like INFOMAX, D2PRUNING operates at a static, dataset-wide level, making it less suitable for dynamic batch-level pruning.

Similarly, CCS addresses static selection by optimizing for both coverage and diversity. A key contribution of CCS is its theoretical characterization of the pruning budget, identifying thresholds beyond which accuracy degradation becomes catastrophic. While our method does not primarily operate at the full-dataset level, we note that analogous effects could, in principle, occur in batch-level selection; investigating such phenomena is a potential avenue for future work.

In summary, although INFOMAX, D2PRUNING, and CCS provide important foundations for data pruning and coreset strategies, they are largely static and dataset-wide in nature. Our approach extends these ideas to a dynamic, scalable setting by leveraging a bandit-driven curriculum where batch-level pruning decisions are guided by validation performance. Moreover, the modularity of our framework allows for seamless integration of alternative reward signals, such as forgetting scores or other criteria discussed in prior works, enabling flexible and adaptive training in large-scale environments.

## H Main Theoretical Results

## H.1 Proof for permutation invariance of Expected Marginal Gain

Lemma 1 ( Permutation Invariance of Expected Marginal Gain ) . Let Π denote the set of all permutations over the elements of B ( &lt;i ) t . Then the expected marginal gain E z i ∈B ( &lt;i ) t [ ∆ U t ( z i | B ( &lt;i ) t , z val t ) ] is invariant under any permutation π ∈ Π , i.e.,

<!-- formula-not-decoded -->

Proof. Let S = B ( &lt;i ) t , with | S | = n , and let z val = z t val . We denote g i := g θ t ( z i ) , g v := g θ t ( z val ) , and H v := H z val ( θ t ) . Then the expected marginal gain as per Eq 4 is given by:

<!-- formula-not-decoded -->

Let ¯ g := 1 n ∑ z ∈ S g z . Then:

<!-- formula-not-decoded -->

This expression depends only on the multiset S , not the order of its elements. Therefore, for any permutation π ( S ) , the same value holds:

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

which proves the claim.

## H.2 Theorem: Capacity-Controlled Risk Convergence Theorem

Theorem 2 ( Capacity-Controlled Risk Convergence ) . [([12]) Theorem 16.3 ] Let M Θ be a neural network with d parameters belonging to the parameter space Θ with an objective to minimize the empirical risk over the training data, D = { ( X i , Y i ) } n i =1 where X i ∈ R m and Y are almost surely bounded and where Y i = ϑ ( z i ) ∼ N ( µ z i , σ z i ) where ϑ : R m → R and P denotes the data distribution. Then for d large enough, we have the following, for any c &gt; 0 .

<!-- formula-not-decoded -->

The above theorem follows from [39] and [12] and is useful to prove further bounds in our case as below.

## H.3 No-Regret Bounds under Constant λ ( · )

We first state here the main theorem under the following assumptions as stated in our main text:

Let τ R ( a ) ( t ) denote the number of times the a -th submodular function f ( a ) is chosen in the first t -1 steps by the uniform branch of the algorithm.

Assumption (a) ( Constant Fractional Exploration Dampening ): The exploration dampening parameter λ ( t ) is time-invariant λ ( t ) = ϵ where ϵ ∈ (0 , 1) .

Assumption (b) ( Optimality Gap ): There exists an optimality gap ϱ such that for every suboptimal arm a t ∈ A \ { a ∗ } : 0 ≤ ϱ ≤ ∆ ( a t ) ( B t ) .

Assumption (c) ( Fractional Exploration Sharpness ): The exploration sharpness parameter π ( t ) is a bounded quantity π ( t ) ∈ (0 , 1) .

Assumption (d) ( Utility Metric Approximation ): The utility metric U t ( · , · ) satisfies the approximation bound as per Theorem 2 (Appendix) with constants C ( a ) for each arm a ∈ A and let n a be a specific constant associated with arm a such that Theorem 2 (Appendix) holds true.

Theorem 1 ( Regret Guarantees ) . Under Assumptions a - d , for all t &gt; t 0 , with probability at least

<!-- formula-not-decoded -->

the expected instantaneous regret incurred by the arm selection policy satisfies

<!-- formula-not-decoded -->

where C ∗ is the approximation constant corresponding to the optimal arm a ∗ .

Proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let M Θ , f ( a t ) indicates the trained neural network in accordance to [39] for action f ( a t ) . By Markov's inequality

<!-- formula-not-decoded -->

Based on Proposition 1 we have τ R ( a ) ( t ) ≥ t -2 2 K (2 -π ) (1 + (1 -π ) ϵ ) for all a ∈ A . Let C ( a ) indicate the constant from Theorem 2 and let n a be the minimal training data size. We choose t 0 &gt; e (2 K max { e, max a n a } ) . Since the x → √ ln( x ) x is monotone decreasing for x &gt; e , the above expression is further bounded by

Proof.

<!-- formula-not-decoded -->

where, the last inequality is based on Proposition 1. We define the variance of τ R ( a ) ( t ) as σ ( τ R ( a ) ( t )) and the corresponding upperbound as Z ( σ ( t ))

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus we have the following:

<!-- formula-not-decoded -->

To showcase the lower bound, we have for a -th arm not optimal that,

<!-- formula-not-decoded -->

Lemma 2 ( Bound on Uniform Arm Selection Frequency ) . Since τ R ( a ) ( t ) denotes the number of times the a -th submodular function f ( a ) is chosen in the first t -1 steps by the uniform branch of the algorithm, we have the following:

<!-- formula-not-decoded -->

Using Bernstein's inequality

<!-- formula-not-decoded -->

By union bound method

<!-- formula-not-decoded -->

Proposition 1 ( Integral Lower Bound (Constant λ ) ) . Let λ ( t ) = ϵ with 0 &lt; ϵ &lt; 1 , and 0 &lt; π &lt; 1 . Then, for

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

The above integral computation is used in the main paper for our proofs.

## H.4 Regret bounds in the case of growing with time exploration dampening function

Proposition 2 ( Integral Lower Bound (Exponential Growing λ ) ) . Let λ ( t ) = 1 -e -i t be a time-growing function with rate i &gt; 0 , and let 0 &lt; π &lt; 1 . Then, for

<!-- formula-not-decoded -->

the following lower bound holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using Jensen's Inequality

Lemma 3 (Exploration Dampening: Annealing) . Since τ R ( a ) ( t ) denotes the number of times the a -th submodular action is chosen in the first t -1 steps by the uniform branch of the algorithm, a ∈ [ K ] , then in the case of λ ( t ) = 1 -e -i t , for i ≥ 0 (i.e. growing exploration dampening probability), we have the following:

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

The last inequality comes from Proposition 2

We define the variance of τ R ( a ) ( t ) as σ ( τ R ( a ) ( t )) and the corresponding upperbound as Z ( σ ( t ))

<!-- formula-not-decoded -->

Using Bernstein's inequality

P ( τ R a ( t ) ≤ Z ( σ ( t )) 2 ) = P ( τ R a ( t ) -Z ( σ ( t )) ≤ - Z ( σ ( t )) 2 ) ) ≤ exp ( -Z ( σ ( t )) 2 8 σ ( τ R a ( t )) + 1 3 Z ( σ ( t )) 2 ) ≤ exp ( -Z ( σ ( t )) 2 8 Z ( σ ( t )) + 1 3 Z ( σ ( t )) 2 ) ≤ exp ( -3 Z ( σ ( t )) 28 ) ≤ exp ( -3 28 K ( 1 2 i [ ln(2 e i ( t -1) -1) -ln(2 e ( a ) -1) ] ) π ) By union bound method P ( K ⋃ j =1 { τ R ( a ) ( t ) ≤ 1 2 K ( 1 2 i [ ln(2 e i ( t -1) -1) -ln(2 e ( a ) -1) ] ) π } ) ≤ K P ( τ R 1 ( t ) ≤ 1 2 K ( 1 2 i [ ln(2 e i ( t -1) -1) -ln(2 e ( a ) -1) ] ) π ) ≤ K exp ( -3 28 K ( 1 2 i [ ln(2 e i ( t -1) -1) -ln(2 e ( a ) -1) ] ) π ) Therefore P ( K ⋂ j =1 { τ R ( a ) ( t ) ≥ 1 2 K ( 1 2 i [ ln(2 e i ( t -1) -1) -ln(2 e ( a ) -1) ] ) π } ) ≥ 1 -K exp ( -3 28 K ( 1 2 i [ ln(2 e i ( t -1) -1) -ln(2 e ( a ) -1) ] ) π )

Theorem 3 ( Regret Guarantees ) . Under Assumptions b - d , for all t &gt; t 0 and with λ ( t ) = 1 -e -i t , with probability at least

<!-- formula-not-decoded -->

the expected instantaneous regret incurred by the arm selection policy satisfies

<!-- formula-not-decoded -->

where C ∗ is the approximation constant corresponding to the optimal arm a ∗ .

Proof. Continuing from the same step in Section H.3 Theorem 1, we have the following: For the case of τ R ( a ) ( t ) ≥ 1 2 K ( 1 2 i [ ln(2 e i ( t -1) -1) -ln(2 e ( a ) -1) ] ) π for all a via Proposition 2. Let C ( a ) indicate the constant from Theorem 2 and let n a be the minimal training data size. We choose t 0 &gt; e (2 K max { e, max a n a } ) . Since the x → √ ln( x ) x is monotone decreasing for x &gt; e , the above expression is further bounded by

<!-- formula-not-decoded -->

## I Broader Impact

The primary aim of our work is to improve the data efficiency of machine learning training pipelines via submodular subset selection. By leveraging principled selection algorithms-such as monotone submodular functions, we can reduce the number of training examples needed without sacrificing model performance. This contributes directly to more sustainable and accessible machine learning, especially in scenarios where training data or compute is limited.

Societal and Environmental Benefits : Reducing the amount of data required for training has multiple practical benefits. For large-scale models, this can translate into lower energy consumption and a reduced carbon footprint. For smaller research labs or applications in low-resource settings, our approach can make training state-of-the-art models more feasible.

Equity and Fairness : By allowing for careful and task-informed selection of training data, our methods could help surface underrepresented or domain-critical samples early in training. However, care must be taken to ensure that the subset selection process does not reinforce existing dataset biases. We encourage practitioners to combine our framework with fairness-aware selection techniques and to audit the resulting models for any performance disparities across groups.

Scientific Impact : More broadly, this work highlights the growing role of data-centric approaches in machine learning research, particularly for compute efficient machine learning research.