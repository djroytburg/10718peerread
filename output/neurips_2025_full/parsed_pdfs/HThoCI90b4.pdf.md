## Principled Long-Tailed Generative Modeling via Diffusion Models

## Pranoy Das, Kexin Fu, Abolfazl Hashemi, Vijay Gupta

School of Electrical and Computer Engineering, Purdue University, W. Lafayette, IN, 47906 {das211,fu448,abolfazl,gupta869}@purdue.edu

## Abstract

Deep generative models, particularly diffusion models, have achieved remarkable success but face significant challenges when trained on real-world, long-tailed datasets- where few "head" classes dominate and many "tail" classes are underrepresented. This paper develops a theoretical framework for long-tailed learning via diffusion models through the lens of deep mutual learning. We introduce a novel regularized training objective that combines the standard diffusion loss with a mutual learning term, enabling balanced performance across all class labels, including the underrepresented tails. Our approach to learn via the proposed regularized objective is to formulate it as a multi-player game, with Nash equilibrium serving as the solution concept. We derive a non-asymptotic first-order convergence result for individual gradient descent algorithm to find the Nash equilibrium. We show that the Nash gap of the score network obtained from the algorithm is upper bounded by O ( 1 √ T train + β ) where β is the regularizing parameter and T train is the number of iterations of the training algorithm. Furthermore, we theoretically establish hyper-parameters for training and sampling algorithm that ensure that we find conditional score networks (under our model) with a worst case sampling error O ( ϵ +1) , ∀ ϵ &gt; 0 across all class labels. Our results offer insights and guarantees for training diffusion models on imbalanced, long-tailed data, with implications for fairness, privacy, and generalization in real-world generative modeling scenarios.

## 1 Introduction

Successful integration of deep learning models into society requires working with real-world data. This comes with many challenges: data quality issues such as inaccurate data, data bias, ethical issues such as breach in privacy, transparency, technical issues such as data integration, generalization, scalability, etc. Furthermore, real world class-labeled datasets are not uniform, but follow a skewed or sometimes referred to as "long-tailed" distribution. It is characterized by a "head" classes that occurs with high probability while the probability of the rest of the classes, often referred to as "tail" classes fall off very quickly. It is well known that the performance of traditional deep learning ([14, 18]) and generative models ([30]) suffer significantly when trained on long-tailed distributions.

One might be curious to ask, "Should deep learning or generative models be concerned with class labels which occur with very low frequency?" The answer is Yes! Even though individually each class occurs with low frequency, collectively these classes may occur with high probability. Diffusion models, which are the focus of this work, are no exception to this phenomenon. Diffusion models are latent variable generative models which learn diffusion process for a given dataset, such that the process can generate new elements that are distributed similarly as the original dataset (See section 3 for more details). They have become popular techniques in image generation beating traditional models such as GANs [3, 7, 38], natural language processing [39], time series forecasting [26] and in fields of applied chemistry [1], biology [9] and medicine [15] to name a few. However, the study of diffusion models for long-tailed learning is limited. [23] showed that when the traditional conditional Diffusion Denoising Probabilistic Model (DDPM) is trained on a long-tailed distribution,

the conditional DDPM model as shown in [23, Figure 1], "generates head class images with satisfying performance, whereas conversely, the generated images on tail classes are very likely to show unrecognizable semantics". Moreover, there might be privacy and ethical concerns if the model overfits (memorize) to the tail class label data and replicate them during generation.

Motivated by this, we develop a theory of Long-Tailed Learning for diffusion models in a mathematically rigorous manner through the perspective of Deep Mutual Learning . Our main results are:

1. Under a suitable metric (KL- divergence) that captures the distance between the learnt class distribution and the ground truth distribution, we derive an upper bound on the worst case distance across all class labels. To do so, we employ Deep Mutual Learning along with the score based diffusion model objective in literature [29, 32]. We present the formulation as a game across conditional score networks and propose Nash equilibrium as the appropriate solution concept.
2. Borrowing ideas from [13] on Deep Mutual Learning, we derive a non-asymptotic first order convergence result for the individual gradient descent algorithm to find the Nash Equilibrium of the proposed game. We show the Nash gap of the score network obtained from the algorithm is upper bounded by O ( 1 √ T train + β ) where β is the regularizing parameter and T train is the number of iterations of the training algorithm. Finally, we show we can find hyper-parameters for training and sampling such that the score networks obtained from the algorithm enjoys a worst case error bound of O ( ϵ +1) for any ϵ &gt; 0 for any class, tail and otherwise.

## 2 Related Works

Long-Tailed Learning for Diffusion Models. To tackle the issue with long-tailed distributions, diverse techniques have been proposed such as re-sampling [27], re-weighting [27], transfer learning [23], and feature augmentation [10]. The closest work to ours is that of [23] titled "Class- Balancing Diffusion Models" or CBDM and its followups [33, 35]. The paper proposes a distribution adjustment regularizer as a solution along with the usual DDPM objective. This represents a modification in the training phase. Their experiments show that the images generated by CBDM exhibit greater diversity and quality in both quantitative and qualitative ways when trained on CIFAR100/CIFAR100LT datasets. As mentioned in [33], "CBDM [23] represents an inaugural inquiry into the performance of DDPM within the context of long-tailed data scenarios" . Motivated by CBDM and contrastive learning, [33] propose adding a penalty function to demarcate the distribution boundaries of different data categories. However, the derivation of the distribution adjustment regularizer in [23, 24, Proposition 2, Appendix A] relies on strong assumptions. They follow a traditional machine learning framework that optimizes over a single objective function with a single neural network and give empirical verification of their method's performance. On the other hand, we define a game across conditional score networks and propose the Nash equilibrium of this game as the egalitarian solution to learn a fair score function for equally good generation over all classes. Furthermore, our framework and analysis do not rely on the strong assumptions made in [23, 24, Proposition 2, Appendix A].

Deep Mutual Learning. Deep Mutual Learning (DML) [36] is a knowledge distillation process that allows the transfer of knowledge from a highly powerful model to a smaller faster efficient model. In DML, an ensemble of students (models) learn collaboratively and teach each other throughout the training process. DML has shown promise in visual object tracking [37], metric learning [22], multi-modal recommender systems [16], and classification tasks trained on Long-Tailed distributions [21]. The theoretical performance guarantees for models trained with DML are scarce. [13] gives a non-asymptotic first order convergence result for training models for classification task using DML. Deep Mutual Learning literature proposes various methods for optimizing Deep Mutual Learning objectives without specifying the solution concept they seek. In contrast, we show that the individual gradient descent (one method for DML) is seeking a Nash equilibrium of underling multiplayer game. While this result of ours could be of independent interest, in this work we further leverage this result to obtain a generalization result for diffusion models for long-tailed generation.

Training and Sampling of Score-based (Conditional) Diffusion Models. The performance of score-based diffusion models have been rigorously studied in the literature of generative modeling. [11, 17, 32] provided a full error analysis of training and sampling from a diffusion model. [11, 17]

parametrize the score network using a random feature model and use gradient flow to train the model. [11] leverages Neural Tangent Kernels to obtain an approximation and generalization error for diffusion models. [32] parametrize the score network by a deep neural network and prove exponential convergence of its gradient descent training dynamic on the empirical loss function. For conditional diffusion models, [8] provides data- dependent approximation bounds of the conditional score function by multi-layered neural network and also give an expected sampling error of the approximated distribution over all class labels. Compared to [8, 32], we consider a finite label class and make no assumption on how the data is distributed. While our result can readily be extended to deeper neural network in line with [32], we parametrize the score function using a two-layer ReLU network (as in [11, 17]) due to the nice properties it induces in the proposed game.

## 3 Basics of Score-Based Diffusion Generative Models

Notation Let ∥ . ∥ denote the ℓ 2 norm for vectors and matrices, ∥ . ∥ F be the Frobenius norm. For the discrete time points, we use t i to denote time point for forward dynamics and t ← i for backward dynamics. σ ( x ) where x ∈ R d refers to the ReLU activation function applied element-wise while ¯ σ t refers to the variance of the forward dynamics. τ ∈ [ T train ] represents the iteration of the training algorithm, which in our case is gradient descent. θ y is the training parameter for score for label y ∈ Y while θ -y is the training parameter for score for all label y ′ ∈ Y - { y } . Given two distributions p and q , the KL divergence from q to p is defined as D KL ( p || q ) = ∫ R d p ( x ) log p ( x ) q ( x ) dx .

In subsequent sections, we introduce the basics of diffusion model training and generation. Denote the initial conditional distribution as P 0 ( X 0 = x | y ) , ∀ x ∈ A ⊂ R d , A is a compact set of all possible features and y ∈ Y where Y is the finite set of class labels.

## 3.1 Forward and Backward Processes

The use of diffusion model in generative modeling involves two processes:

1. Forward Process: The forward process pushes an initial distribution P 0 ( . | y ) to Gaussian by adding noise progressively to X 0 , and is usually described as an Ornstein-Uhlenbeck (OU) process,

<!-- formula-not-decoded -->

where f t , g t are functions of t ∈ [0 , T ] and dW t is the incremental Brownian motion or Wiener process, X t is a d -dimensional random variable with X t ∼ P t ( . | y ) . The choice of f t , g t results in various diffusion model schemes such as Variance Preserving (VP), Variance Exploding (VE) SDE (see [29] for more details).

2. Backward Process: To generate a new sample, the forward dynamics can be reversed conditioned on the final distribution X ← T ∼ P 0 ( . | y ) to get the backward or reverse diffusion process defined as:

<!-- formula-not-decoded -->

where X ← 0 ∼ P T and p t is the density of P t . Then X ← T -t and X t have the same distribution with density P t ( . | y ) , which means that the dynamics will push near-Gaussian distributions back to the initial distribution P 0 ( . | y ) , ∀ y ∈ Y .

## 3.2 Training via Denoising Score Matching

From (2), to generate samples conditionally, one needs access to ∇ x log p T -t ( X ← t | y ) , the conditional score function, which is unknown. Let s t,θ ( x, y ) be an estimator of ∇ x log p t ( x | y ) . To estimate the conditional score function, a natural loss function to train a model would be the following objective:

<!-- formula-not-decoded -->

Once the conditional score function is learnt, a datum from class label y ∈ Y is sampled using the reverse diffusion process given below:

<!-- formula-not-decoded -->

To measure how well the learnt score function approximates the ground truth distribution, KLdivergence is employed as the metric. To assess the goodness of the learnt score function through the optimization of L y conti ( θ ) , we have to relate the KL-divergence between the learnt distribution and the ground truth to the training objective. Informally, the KL divergence between the learned distribution and the ground truth distribution is bounded by the score based diffusion model objective ( L conti ( θ )) as (see [28, Theorem 1] or [8, Appendix D] for detailed proof)

<!-- formula-not-decoded -->

Achieving a bound on the expectation as [8, Theorem 4.1] gives no insights into the worst case sampling error over all y ∈ Y . In this work, we provide a methodology to achieve an upper bound on max y ∈Y D KL ( P 0 ( . | y ) || P 0 ,θ ( . | y )) , thereby addressing the long-tailed issue in generative modeling.

## 4 Long-Tailed Learning

## 4.1 Egalitarian Solution Concept

Previous work in conditional diffusion models [8] have focused on optimizing the following objective

<!-- formula-not-decoded -->

for classifier guided sampling [29] or the unconditional score function along with the conditional score function from 5 for classifier free guidance. The above objective is sound when the marginal density of the classes p ( y ) itself is uniformly distributed. Observe that when optimizing L conti ( θ ) (eq. 5), an optimization algorithm will give more weight towards reducing L y ( θ ) for head classes (classes with high p ( y ) , appearing with higher frequency in the data). Thus, the trained model overfits the head class, while performing poorly on the tail classes. One way of ensuring that each class label is equally weighted during the training process is to re-weigh each class objective function by a factor inversely proportional to the class marginal density p ( y ) . This ensures that both head and tail classes receive equal weighting during the training process.

<!-- formula-not-decoded -->

However, in many real world scenarios the marginal density p ( y ) is unknown and hence such an accurate reweighting is not possible. For Long-Tailed Learning , as we desire to perform well (in terms of generation quality) for every class label, the natural objective would be to minimize max y ∈Y D KL ( P 0 ( . | y ) || P 0 ,θ ( . | y )) , that is, minimize the worst-case KL divergence over all y ∈ Y . Suppose L y conti ( θ ) is convex in the training parameter θ , then so is f ( θ ) = max y ∈Y L y conti ( θ ) as maximum of finite convex functions is again convex. f ( θ ) may not be differentiable even if L y conti ( θ ) are differentiable in θ for all y ∈ Y . One could use sub-gradient methods to optimize the worst case class loss max y L y ( θ ) . However, in practice one has to work with the empirical version of these losses which might be noisy and lead to parameters that are sub-optimal with respect to the population loss.

## 4.2 Nash Equilibrium as a Solution Concept

To enable diffusion models for Long-tailed learning, we modify the DM objective to add the mutual learning objective defined as

<!-- formula-not-decoded -->

to obtain a regularized version of the DM objective function denoted as L y cont,reg ( θ y , θ -y ) . In the setting of Mutual Learning, the distribution Q is uniform. But, the distribution can be a hyperparameter over which one could optimize. From now on, we will drop the weighting arguments λ ( . ) , ω ( . ) in the objective functions, leading to the following regularized objective for each class:

<!-- formula-not-decoded -->

Learning the score ∇ x ( t ) log p t ( x ( t ) | y ) is difficult as it is intractable. Conditioning on X 0 and using law of iterated expectation, one can rewrite the objective function as (see [32, Appendix A] for detailed proof) with discretized time points as 0 &lt; t 0 &lt; t 1 &lt; · · · t N = T to get the training objective

<!-- formula-not-decoded -->

where ¯ C ( y ) = 1 2 ∑ N j =1 λ ( t j )( t j -t j -1 ) C t j ( y ) and C t ( y ) = E X t ∥∇ log p t ( . | y ) ∥ 2 -E X 0 E X t | X 0 ∥∇ log p t ( x t | x 0 , y ) ∥ 2 . [32, Remark 1] point out that C ( y ) &lt; 0 and hence the first summand in Eq. 9 is always bound below by -C ( y ) . ¯ C ( y ) along with the entire first summation in 9 correspond to L y ( θ y ) while the third term is L y mut ( θ y , θ -y ) . As ¯ C ( y ) doesn't depend on θ , we can ignore it for the purpose of training. But we note that C ( y ) will appear in our final worst case sampling error. When the drift and diffusion coefficient of the forward dynamics satisfy some nice properties, the distribution of p t ( x t | x 0 ) is normally distributed, whose mean and variance ( ¯ σ t ) can be explicitly computed. Exploiting this knowledge, one can rewrite the objective function in eq 9 as (See Appendix B.3 for details)

<!-- formula-not-decoded -->

where ¯ L n y reg ( θ y , θ -y ) is the empirical version of L y reg ( θ y , θ -y ) with n y samples, { x i } n y i =1 with x i ∼ P 0 ( . | y ) denotes the initial data, { ξ ij } N j =1 where ξ ij ∼ N (0 , I d ) denotes the noise and input data of the neural network is { t j , x i ( t j ) } n y ,N i =1 ,j =1 , where x i ( t j ) ∼ P t j ( . | y ) is obtained from the forward diffusion process.

## 4.2.1 Neural Network Architecture for Score Parametrization

The approximation power of two-layer ReLU network with randomly sampled input layer are well understood from numerous works [12, 25] and has been used to study the generalization properties of Diffusion Models in [11, 17]. We also parametrize the score function s t,θ y for each label y ∈ Y using a random feature model

<!-- formula-not-decoded -->

where σ ( · ) = max { 0 , ·} is the ReLU activation function, A y = ( a y, 1 , · · · , a y,m ) ∈ R d × m is the trainable parameter, W y = ( w y, 1 , · · · , w y,m ) T ∈ R m × d and U y = ( u y, 1 · · · , u y,m ) T ∈ R d × d e are randomly initialized embedding matrices that are frozen during training, e : R ≥ 0 → R d e is the embedding function for the time. The above model represents a neural network with one hidden layer with m neurons and a d -dimensional vector as an output. Suppose a y,i , w y,i and u y,i are i.i.d. sampled from an underlying distribution ρ . Then as m →∞ , we can view

<!-- formula-not-decoded -->

## Algorithm 1 Individual Gradient Descent(IGD)

```
Input parameters: Learning rate η τ Initialize: ( W y , U y ) y ∈Y and θ 0 y , ∀ y ∈ Y for τ = 0 ...T train do for y = 0 ... |Y| do θ τ +1 y ← θ τ y -η τ ∇ θ y ¯ L n y reg ( θ τ y , θ τ -y ) end for end for Output: ( θ y , θ -y ) = min τ ∈ [ T train ] NE-gap ( θ τ y , θ τ -y
```

```
)
```

with a y ( w,u ) := 1 ρ 0 ( w,u ) ∫ R d a y ρ ( a, w, u ) da y and ρ 0 ( w,u ) := ∫ R d ρ ( a, w, u ) da . The above relation represents s t,θ y ( x ) as an approximation of the continuous version ¯ s t, ¯ θ y ( x ) , which can be viewed as a neural network with infinite width, i.e., infinite number of neurons in the hidden layer ( m →∞ ) . Furthermore, we assume the embedding matrices W y and U y are sampled independently for every y ∈ Y from a set with bounded support.

Having defined our loss function, we define the strategy space as Θ y = { A y ∈ R d × m : ∥ A y ∥ F ≤ B } , ∀ y ∈ Y . Now, consider the |Y| -player game ⟨Y , ( ¯ L n y reg ) y ∈Y , (Θ y ) y ∈Y ⟩ ,

Definition 1 (Nash Gap) . Let B y : Θ -y → Θ y represent the best response function for label y ∈ Y defined as B y ( θ -y ) ∈ argmin θ ∈ Θ y ¯ L n y reg ( θ, θ -y ) . Using the best response function, we define the Nash gap of a strategy profile ( θ y ) y ∈Y ∈ × y ∈Y Θ y as:

<!-- formula-not-decoded -->

Definition 2 (Nash Equilibrium) . A strategy ( θ ′ y ) y ∈Y ∈ × y ∈Y Θ y is an ϵ - Nash equilibrium of the game ⟨Y , ( ¯ L n y reg ) y ∈Y , (Θ y ) y ∈Y ⟩ if NE-gap (( θ ′ y ) y ∈Y ) ≤ ϵ . When NE-gap (( θ ′ y ) y ∈Y ) = 0 , then ( θ ∗ y ) y ∈Y is a Nash equilibrium.

The ability to find an ϵ - Nash equilibrium of the game ⟨Y , ( ¯ L n y reg ) y ∈Y , (Θ y ) y ∈Y ⟩ is crucial in our analysis to bound the worst case sampling error.

## 4.3 Algorithm

In this section, we propose the individual gradient descent algorithm 1 to find an approximate Nash equilibrium of the game ⟨Y , ( ¯ L n y reg ) y ∈Y , (Θ y ) y ∈Y ⟩ . The input parameter for the algorithm is the step-size η τ where τ is the τ th step of the individual gradient descent algorithm. The initialization step samples the embedding matrices and fixes an initial condition for the training parameter ( W y , U y , θ (0) y ) y ∈Y . The individual gradient proceeds for T train steps and within each step an individual gradient update is performed by computing the gradient ∇ θ y ¯ L n y reg ( θ τ y , θ τ -y ) .

The complexity of finding Nash equilibrium: One of the most celebrated results in game theory [6] proved that the computational complexity of the problem of computing of a Nash equilibrium in an arbitrary game lies in the complexity class PPAD. So far, there does not exist an polynomial time algorithm that can find an approximate or exact solution to problems in PPAD. The game ⟨Y , ( ¯ L n y reg ) y ∈Y , (Θ y ) y ∈Y ⟩ is a convex minimization game (See B.5). [20] showed that concave maximization games (convex minimization games) also lie in the class PPAD. We present a positive result that in our game ⟨Y , ( ¯ L n y reg ) y ∈Y , (Θ y ) y ∈Y ⟩ , individual gradient descent finds an approximate Nash equilibrium whose NE-gap is bounded by O ( m 2 √ T train + β ) .

## 5 Main Result

We now present the main result of the capability of diffusion models in long-tailed learning through deep mutual learning. We derive a data-independent worst-case bound for D KL ( p 0 ( . | y ) || p 0 ,θ y,t ( . | y )) . Let θ ∗ y = argmin θ y L y ( θ y ) , ∀ y ∈ Y and let ¯ θ ∗ y be the optimal solution when the true score function

s t,θ y ( x ) is replaced in the class-label objective function L y ( θ y ) (equation 9) by its approximation ¯ s t, ¯ θ y ( x ) . We make one assumption on the support of data distribution (justified in Remark 1).

Assumption 1. We assume that the target distribution P 0 ( x | y ) is continuously differentiable in x and has compact support for every y ∈ Y . Let for any y ∈ Y , x ∈ A ⊂ R d , ∥ x ∥ ∞ ≤ K

Generation Algorithm. We consider the DDPM sampling scheme. Under this scheme f t = 1 and g t = √ 2 in Eq. 1. Denote the backward time schedule as { t ← j } 0 ≤ j ≤ N such that 0 = t ← 0 &lt; t ← 1 &lt; · · · , t ← N = T -α . To simulate the backward SDE, we use the exponential integrator scheme [34] which can be piecewisely expressed as a continuous-time SDE: for any t ∈ [ t ← j , t ← j +1 ) . .

<!-- formula-not-decoded -->

Denote q t ( . | y ) := Law ( ¯ Y t ) , ∀ t ∈ [0 , T -α ] . γ k = t ← k +1 -t ← k and assume there exists κ &gt; 0 such that γ k ≤ κ min { 1 , T -t ← k +1 } . Let u 2 2 be such that E x 0 ∼ P 0 ( . | y ) [ ∥ x ∥ 2 ] ≤ u 2 2 &lt; ∞ , ∀ y ∈ Y .

Remark 1. Assumption 1 ensure the data belong to a bounded set and the score is well defined. This also ensures the second moment of the data distribution are bound which is necessary for convergence of forward SDE. Some works [4, 32] do not require the existence of score function for the data distribution P 0 ( . | y ) . These works employ early stopping of the reverse (sampling) process. They do so because for non-smooth data distributions ∇ log q t can blow up as t → T . This means that the model will approximate q T -α rather than q T = P 0 ( . | y ) , which is acceptable since for small α the distance (e.g. in Wasserstein-p metric) between q T -α and P 0 ( . | y ) is small [4].

We now present the main result of the paper.

Theorem 1. Given Assumption 1, for 0 &lt; δ ≪ 1 , we have with probability 1 -N ( ∑ y ∈Y n y ) δ that

1. The empirical loss functions ¯ L n y reg ( θ y , θ -y ) , are L y m 2 smooth w.r.t to their own parameter θ y ∀ y ∈ Y (See Lemma 3 in Appendix B.5)
2. If one runs individual gradient descent with step-size η τ ≤ m 2 max y ∈Y L y √ T train for T train iterations and selects the parameter from ( θ τ y , θ τ -y ) τ ∈ [ T train ] that minimizes the Nash Gap of the game ⟨Y , ( ¯ L n y reg ) y ∈Y , (Θ y ) y ∈Y ⟩ and samples according to Eq.13, the sampling error

<!-- formula-not-decoded -->

where n ∗ = min y ∈Y n y , ¯ C = max y ∈Y -¯ C ( y ) as in Eq. 9, κ 2 Nu 2 2 + κTu 2 2 is an upper bound on the discretization error due to the reverse SDE, exp( -2 T ) u 2 2 is the error due to the convergence of the forward SDE and constant C 0 is some constant. ˜ O hides the log 1 δ factors, |Y| 2 and bounds on strategy space, embedding matrices and other constants.

Corollary 1 (Full Error Analysis) . Fix ϵ &gt; 0 arbitrarily. If T ≥ 1 , α &lt; 1 and N &gt; log 1 α , then there exists 0 = t o &lt; t 1 &lt; · · · t N = T -α such that for some κ = Θ( T +log 1 α N ) and γ k ≤ κ min { 1 , T -t k +1 }∀ k = 0 , 1 , · · · , N -1 . If we take T = 1 2 log d ϵ , N = Θ( d ( T +log 1 α ) 2 ϵ ) , β = ˜ Θ( ϵ ) , T train = ˜ Θ( 1 ϵ 6 ) and m = ˜ Θ( 1 ϵ 2 ) , then under similar conditions as Theorem 1, we achieve

<!-- formula-not-decoded -->

where ˜ O , ˜ Θ and ≲ hides the polynomial of log 1 δ , |Y| 2 and bounds on strategy space, embedding matrices and other constants. L y ( ¯ θ ∗ y ) is the universal approximation error of approximating the score with two layer network with random ReLUs.

Corollary 1 gives us the range of hyper-parameters such as width of hidden-layer, number of training steps, discretization of sampling, etc. to achieve worst case sampling error of O ( ϵ +1) . The O (1) term C ( y ) in Eq. 9, can be viewed as the error incurred due to diffusion model's nature in approximating ∇ log p t ( x t | y ) which is intractable by ∇ log p t ( x t | x 0 , y ) with reverse SDE.

## 5.1 Proof sketch of Theorem 1

We provide a sketch for the proof and defer the details to the Appendix. We use a slight variant of [4, Theorem 1] (See Appendix B for more details) to upper bound the KL-divergence between the distribution approximated by our model and the ground truth to get

<!-- formula-not-decoded -->

We then perform the following decomposition for max y ∈Y L y ( θ y ) (See Appendix B.1), where

<!-- formula-not-decoded -->

Proposition 1 (Training and bounding the Nash Gap) . Suppose ¯ L n y reg ( θ τ y , θ τ -y ) is L y m 2 smooth for all y ∈ Y . Then by selecting a constant learning rate η τ ≤ η √ T train ≤ m 2 max y ∈Y L y √ T train that depends on the total iteration T train , and using ˜ O to hide the log 1 δ factors, we have

<!-- formula-not-decoded -->

The proof is presented in Appendix B.6. Proposition 1 gives a non-asymptotic first order convergence of individual gradient descent. When no further assumption on the gradient mapping (e.g., (strong) monotonicity of the game ⟨Y , ( ¯ L n y reg ) y ∈Y , (Θ y ) y ∈Y ⟩ ) is considered, this is the best we can hope for. The iterate at which the minimum Nash Gap is achieved can be tracked by storing the parameters ( θ τ y , θ τ -y ) for which the max y ∈Y ∥ ∥ ∇ θ y ¯ L n y reg ( θ τ y , θ τ -y ) ∥ ∥ 2 is the least.

Monte-Carlo Estimate. To bound max y ∈ Y L y ( θ ∗ y ) , we employ ideas from [17, Lemma 6]. Informally (See Prop 2 in Appendix B.7 ), for 0 &lt; δ ≪ 1 , with probability 1 -2 N |Y| δ, we achieve

<!-- formula-not-decoded -->

where ˜ O hides the log 1 δ factors. L y ( ¯ θ ∗ y ) is the error associated with approximating the score of the data using a two layer networks of random ReLUs.

Rademacher Complexity. Finally, we bound the generalization error (See Lemma 9 in Appendix B.8 for the derivation) by the Rademacher Complexity

<!-- formula-not-decoded -->

where n ∗ = min y ∈Y n y , ¯ C = max y ∈Y -¯ C ( y ) . ˜ O hides the log 1 δ factors, |Y| 2 and bounds on strategy space, embedding matrices and other constants.

Bound on Mutual Learning Loss The final term in Eq. 17 max y ∈ Y sup ( θ y ,θ -y ) L y mut ( θ y , θ -y ) is O (1) (See Lemma 6 in Appendix B.8).

## 5.2 Interpretation of the Main Result and Implications for Long-tailed Learning

Firstly, when the training objective function are nice, Proposition 1 shows that individual gradient descent employed in Deep Mutual Learning literature is seeking a Nash Equilibrium of an underlying game across different models. Second, when diffusion models are employed for long-tailed generation, Theorem 1 shows that a Nash equilibrium of an underlying game across conditional score network achieves an egalitarian solution w.r.t to sampling error. Our result give insight into the bottleneck process in diffusion generative modeling when faced with limited computing resources and long-tailed data. To the best of our knowledge, our result is the first to provide a comprehensive view of Deep Mutual Learning and long-tailed generation(learning) with diffusion models.

<!-- image -->

timet timet

timet

Figure 1: Fitting error on a toy demo with and without mutual learning. Top row represents the tail class and bottom row represents head class. The middle column represents mutual learning and the right column represents without mutual learning ( β = 0 ). Lighter areas represent higher probability region (left column) and larger fitting error (middle and right column)

## 6 Numerical Experiments

## 6.1 Toy Model

Diffusion models are trained to learn the score function ∇ log p t ( x t | x 0 , y ) of the forward process which is then used in the reverse SDE to sample. From Figure 1 (left column), the score model works well when p t ( x | y ) is large but suffers from large error when p t ( x | y ) is small. This observation can be explained by examining the training loss on Figure 1(middle and right columns). Since the training data is sampled from p t ( x | y ) , in regions with a low p t ( x | y ) value, the learned score network is not expected to work well due to the lack of training data. As a consequence, to ensure ¯ Y 0 is close to x 0 , one need to make sure ¯ Y t stays in the high p t ( x | y ) region ∀ t ∈ [0 , T ] . The mutual learning term aligns the score network for the tail classes with the high confidence scores of head classes at the high noise regime area ( t ≫ 0 ) decreasing the fitting error. This can be seen from a comparison of the heatmap of the tail class (top row middle column) with mutual learning having larger portion of area with low fitting error compared to the case with no mutual learning (top row right column). 1

## 6.2 Real World Datasets

Datasets We perform empirical validation of our theoretical findings with the widely used CIFAR10 dataset in the domain of image synthesis, specifically its long-tailed versions CIFAR10LT. The construction of CIFAR10LT follows from [5], where the size decreases exponentially with its class label index according to the imbalance factor imb = 0 . 01 . We also perform experiments on synthetic dataset such as Gaussian Mixture Model and include them in Appendix C.

Table 1: Best Performance for Various Methods

| Method                 | FID( ↓ )   | IS( ↑ )         |
|------------------------|------------|-----------------|
| Vanilla DDPM ( β = 0 ) | 16 . 58    | 8 . 78 ± 0 . 15 |
| Mutual Learning        | 14 . 58    | 8 . 92 ± 0 . 19 |
| CBDM                   | 15 . 28    | 8 . 11 ± 0 . 14 |

Implementation Details We take the code from [23] and modify the training procedure according to individual gradient descent. The Neural network Architecture employed is U-net as in [23]. To

1 The code is available at https://github.com/pranoydas51/IGD-ML

be able to make direct comparisons to DDPM and a rudimentary comparison to CBDM, we modify the code of CBDM and employ error networks for mutual learning (individual gradient descent) instead of score networks as above. We run both CBDM and Individual Gradient Descent(IGD) for T train = 60 k training steps. We generate 15 k samples per class and make the comparison at the 60 k training step mark. We provide FID, IS across various parameter settings in Appendix C.2.

Comparison with baselines The baseline model for us is DDPM models trained individually on each class label dataset ( β = 0 ). We also make a comparison of mutual learning with Class Balancing Diffusion Models. While empirical experiments on CIFAR10LT shows Mutual Learning perform better than CBDM, we do not make any claim such as mutual learning outperforms CBDM. Since our contribution is theoretical in nature, comprehensive numerical comparison with CBDM is left as a future direction.

## 7 Discussion

Choice of λ ( t ) and ω ( t ) . We choose ω ( t ) as an increasing function of t (as in [23]) and λ ( t ) such that λ ( t ) ¯ σ t is non-increasing in t . The motivation behind this is to ensure that the training process gives more weight to fitting to the data distribution for smaller 0 &lt; t &lt; T and give more weight to the mutual learning objective for high noise regions i.e. larger 0 &lt; t &lt; T of the forward diffusion process. There might exist a better weighting function. Our analysis doesn't involve the investigation of an optimal weighting function. We leave this as a future direction to pursue.

Bound on Approximation Error L y ( ¯ θ ∗ y ) . Given universal approximation results for two-layer networks of Random ReLUs such as [11, Theorem 3.6] and assuming ∇ log p t ( x t | y ) to be Lipschitz continuous w.r.t. x t , we can follow [11] to achieve an upper bound for max y ∈Y L y ( ¯ θ ∗ y ) . This bound can be made arbitrarily small by controlling hyperparameters such as bound on RKHS norm of ¯ s t, ¯ θ y and 0 &lt; δ ≪ 1 .

Extension of Theoretical Results to Deeper Neural Networks Following [2], which proves that stochastic gradient descent (SGD) can find global minima in Deep Neural Networks (DNN) in polynomial time (given that the inputs are non-degenerate and the network is over-parameterized), and [32], which extends [2] to determine the training complexity for diffusion models and determine the generalization error of sampling with DNNs, our theoretical analysis can be extended to Deeper Neural Network architecture in three steps. First, we can use [31] to obtain the generalization bound using Rademacher Complexity for DNNs with ReLU activation function. Then, using the fact that ¯ L n y reg ( θ y , θ -y ) = ¯ L n y ( θ y ) + β ¯ L n y mut ( θ y , θ -y ) , we observe that [32, Lemma 9] proves the semi-smoothness of ¯ L n y ( θ y ) with high probability. Thus, we can use [2, Theorem 3] to obtain the semi-Smoothness of ¯ L n y mut ( θ y , θ -y ) . The only thing that one needs to compute are the various hyperparameter dependent constants. The final step would be to derive a PL like inequality as in [2, Theorem 3] [32, Lemma 1(Appendix D.1)] with high probability. Proving whether ¯ L n y reg ( θ y , θ -y ) satisfies a PL like inequality is challenging. [32, Lemma 1(Appendix D.1)] considers the case without mutual learning. Even though ¯ L n y ( θ y ) and ¯ L n y mut ( θ y , θ -y ) individually satisfy a PL like inequality, their sum may not. We leave this as a conjecture for future work.

Application to Federated Learning. Consider the following scenario, each class label y ∈ Y is thought of as a client that holds private training data with variable number of training sample points. Individual Gradient Descent then represents local training of score network with global sharing of updated score network parameters while preserving the privacy of local client data. This allows fair learning and generalization among all classes and prevents overfitting (memorization) for class labels with low training data frequency.

Limitations. While we achieve a bound on the worst case generalization (sampling) error, the current analysis should be extended to provide insight into whether the performance of the head class score networks is preserved upon adding the mutual learning loss. Further, we set Q = Uniform ( Y ) and further investigation is warranted on the effect of the distribution Q on the worst-case sampling error. It is worth examining if generalization (sampling) error can be made arbitrarily small (also noted in [32, Section 3.3]) i.e. the O (1) bias be removed. Finally, while we support our analysis with empirical experiments, validating our findings on larger real world datasets CIFAR100LT and a detailed comparison with CBDM [23] could further strengthen the approach.

## Acknowledgments and Disclosure of Funding

We thank Mainak Pal for his help with the numerical simulations. The first author was partially supported by ARO grant W911NF2310266, the second by ONR grant 13001274, the third by NSF under Grants CNS-2313109 and DMS-2502560, and the fourth by ONR grant N000142312604.

## References

- [1] Amira Alakhdar, Barnabas Poczos, and Newell Washburn. Diffusion models in de novo drug design. Journal of Chemical Information and Modeling , 64(19):7238-7256, 2024.
- [2] Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via over-parameterization. In International conference on machine learning , pages 242-252. PMLR, 2019.
- [3] Dmitry Baranchuk, Ivan Rubachev, Andrey Voynov, Valentin Khrulkov, and Artem Babenko. Label-efficient semantic segmentation with diffusion models. arXiv preprint arXiv:2112.03126 , 2021.
- [4] Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly d -linear convergence bounds for diffusion models via stochastic localization. arXiv preprint arXiv:2308.03686 , 2023.
- [5] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced datasets with label-distribution-aware margin loss. Advances in neural information processing systems , 32, 2019.
- [6] Constantinos Daskalakis, Paul W Goldberg, and Christos H Papadimitriou. The complexity of computing a Nash equilibrium. Communications of the ACM , 52(2):89-97, 2009.
- [7] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems , 34:8780-8794, 2021.
- [8] Hengyu Fu, Zhuoran Yang, Mengdi Wang, and Minshuo Chen. Unveil conditional diffusion models with classifier-free guidance: A sharp statistical theory. arXiv preprint arXiv:2403.11968 , 2024.
- [9] Zhiye Guo, Jian Liu, Yanli Wang, Mengrui Chen, Duolin Wang, Dong Xu, and Jianlin Cheng. Diffusion models in bioinformatics and computational biology. Nature reviews bioengineering , 2(2):136-154, 2024.
- [10] Pengxiao Han, Changkun Ye, Jieming Zhou, Jing Zhang, Jie Hong, and Xuesong Li. Latentbased diffusion model for long-tailed recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 2639-2648, 2024.
- [11] Yinbin Han, Meisam Razaviyayn, and Renyuan Xu. Neural network-based score estimation in diffusion models: Optimization and generalization. arXiv preprint arXiv:2401.15604 , 2024.
- [12] Daniel Hsu, Clayton H Sanford, Rocco Servedio, and Emmanouil Vasileios VlatakisGkaragkounis. On the approximation power of two-layer networks of random relus. In Conference on Learning Theory , pages 2423-2461. PMLR, 2021.
- [13] Weipeng Fuzzy Huang, Junjie Tao, Changbo Deng, Ming Fan, Wenqiang Wan, Qi Xiong, and Guangyuan Piao. Rényi divergence deep mutual learning. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases , pages 156-172. Springer, 2023.
- [14] Ziyu Jiang, Tianlong Chen, Bobak J Mortazavi, and Zhangyang Wang. Self-damaging contrastive learning. In International Conference on Machine Learning , pages 4927-4939. PMLR, 2021.
- [15] Amirhossein Kazerouni, Ehsan Khodapanah Aghdam, Moein Heidari, Reza Azad, Mohsen Fayyaz, Ilker Hacihaliloglu, and Dorit Merhof. Diffusion models in medical imaging: A comprehensive survey. Medical image analysis , 88:102846, 2023.

- [16] Jianing Li, Chaoqun Yang, Guanhua Ye, and Quoc Viet Hung Nguyen. Graph neural networks with deep mutual learning for designing multi-modal recommendation systems. Information Sciences , 654:119815, 2024.
- [17] Puheng Li, Zhong Li, Huishuai Zhang, and Jiang Bian. On the generalization properties of diffusion models. Advances in Neural Information Processing Systems , 36:2097-2127, 2023.
- [18] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Open long-tailed recognition in a dynamic world. IEEE Transactions on Pattern Analysis and Machine Intelligence , 46(3):1836-1851, 2022.
- [19] Andreas Maurer. A vector-contraction inequality for rademacher complexities. In Algorithmic Learning Theory: 27th International Conference, ALT 2016, Bari, Italy, October 19-21, 2016, Proceedings 27 , pages 3-17. Springer, 2016.
- [20] Christos H Papadimitriou, Emmanouil-Vasileios Vlatakis-Gkaragkounis, and Manolis Zampetakis. The computational complexity of multi-player concave games and kakutani fixed points. arXiv preprint arXiv:2207.07557 , 2022.
- [21] Changhwa Park, Junho Yim, and Eunji Jun. Mutual learning for long-tailed recognition. In Proceedings of the IEEE/CVF winter conference on applications of computer vision , pages 2675-2684, 2023.
- [22] Wonpyo Park, Wonjae Kim, Kihyun You, and Minsu Cho. Diversified mutual learning for deep metric learning. In Computer Vision-ECCV 2020 Workshops: Glasgow, UK, August 23-28, 2020, Proceedings, Part I 16 , pages 709-725. Springer, 2020.
- [23] Yiming Qin, Huangjie Zheng, Jiangchao Yao, Mingyuan Zhou, and Ya Zhang. Class-balancing diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 18434-18443, 2023.
- [24] Yiming Qin, Huangjie Zheng, Jiangchao Yao, Mingyuan Zhou, and Ya Zhang. Class-Balancing Diffusion Models (arxiv version), 2023.
- [25] Ali Rahimi and Benjamin Recht. Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning. Advances in neural information processing systems , 21, 2008.
- [26] Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In International conference on machine learning , pages 8857-8868. PMLR, 2021.
- [27] Jie Shao, Ke Zhu, Hanxiao Zhang, and Jianxin Wu. Diffult: Diffusion for long-tail recognition without external knowledge. Advances in Neural Information Processing Systems , 37:123007123031, 2024.
- [28] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. Advances in neural information processing systems , 34:14151428, 2021.
- [29] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 , 2020.
- [30] Shuhan Tan, Yujun Shen, and Bolei Zhou. Improving the fairness of deep generative models without retraining. arXiv preprint arXiv:2012.04842 , 2020.
- [31] Lan V. Truong. On rademacher complexity-based generalization bounds for deep learning, 2025.
- [32] Yuqing Wang, Ye He, and Molei Tao. Evaluating the design space of diffusion-based generative models. arXiv preprint arXiv:2406.12839 , 2024.
- [33] Divin Yan, Lu Qi, Vincent Tao Hu, Ming-Hsuan Yang, and Meng Tang. Training classimbalanced diffusion model via overlap optimization. arXiv preprint arXiv:2402.10821 , 2024.

- [34] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902 , 2022.
- [35] Tianjiao Zhang, Huangjie Zheng, Jiangchao Yao, Xiangfeng Wang, Mingyuan Zhou, Ya Zhang, and Yanfeng Wang. Long-tailed diffusion models with oriented calibration. In The twelfth international conference on learning representations , 2024.
- [36] Ying Zhang, Tao Xiang, Timothy M Hospedales, and Huchuan Lu. Deep mutual learning. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 4320-4328, 2018.
- [37] Haojie Zhao, Gang Yang, Dong Wang, and Huchuan Lu. Deep mutual learning for visual object tracking. Pattern Recognition , 112:107796, 2021.
- [38] Yuanzhi Zhu, Zhaohai Li, Tianwei Wang, Mengchao He, and Cong Yao. Conditional text image generation with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 14235-14245, 2023.
- [39] Hao Zou, Zae Myung Kim, and Dongyeop Kang. A survey of diffusion models in natural language processing. arXiv preprint arXiv:2305.14671 , 2023.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We discuss under related works, the scope and how our work differs from those in existing literature. Theorem 1, Proposition 1, Corollary 1 along with numerical experiments reflect the theoretical contribution of our paper and support the claims made in the abstract and introduction .

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In discussion section, we highlight limitations of our theoretical analysis. Guidelines:

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

Justification: We clearly state the assumptions we make and cite references that use existing literature. We provide a proof sketch of the Main Theorem of the paper. The complete proof and supporting lemmas have been cited are provided in the supplemantary material.

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

## Justification: We provide the hyper-parameter values and provide additional graphs in Appendix C .

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

Justification: We provide a github link to the code within the paper.

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

Justification: We clearly state the dataset chosen and the hyper-parameters used for the experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Weprovide error bar of 1 standard deviation and provide the experimental setting in the main text.

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

Justification: We provide the details of our computing resources in the supplementary material in Appendix C

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We confirm that our research is in line with NeurIPS Code of Ethics. We perform our experiments on synthetic datasets.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We point out in our abstract, introduction and discussion the broader impacts of our work to privacy, copyright related issues in generative modeling and existing literature on diffusion models for Long-tailed generation.

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

Justification: We provide the link to github repository where we include the code used for the numerical experiments in our paper.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We modify the code in CBDM [23] from their github repository and we provide a link to the github repository for our numerical experiments. We use the probability flow ODE sampler used in [17] and we mention this in the main text under numerical experiments.

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

Justification: We use simple synthetic data to verify our theoretical findings along with experiments on empirical real world datasets such as CIFAR10LT. We provide link to the github repository where our code is.

Guidelines:

- The answer NA means that the paper does not release new assets.

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No crowd sourcing experiments or research with human subjects were conducted for this work.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The numerical experiments were performed on synthetic data. There were no subjects on which experiments were conducted.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMwasn't use for any task during the development of the research or the preparation of the manuscript.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Appendix / supplemental material

| Notations   | Notations                                                                  |
|-------------|----------------------------------------------------------------------------|
| Symbol      | Meaning                                                                    |
| m           | Number of neurons in hidden-layer of score network                         |
| C w y ,u y  | Upper bound on ∥ w y,i ∥ 1 , ∥ u y,i ∥ 1                                   |
| F 2 T       | Upper bound on E X 0 E ξ j [ ∥ σ ( Wx ( t )+ Ue ( t )) ∥ 2 2 ] , 0 ≤ t ≤ T |
| L y         | ¯ L n y reg ( θ y , θ - y ) is L y m 2 smooth w.r.t. θ y                   |
| ϕ y         | Lipschitz constant of ¯ L n y mut ( θ y , θ - y ) w.r.t. θ y               |
| σ y         | ¯ L n y reg ( θ y , θ - y ) is σ y Lipschitz in θ y                        |
| B           | Upper Bound of the Frobenius norm of A y                                   |

## B Sampling

Denote the backward time schedule as { t ← j } 0 ≤ j ≤ N such that 0 = t ← 0 &lt; t ← 1 &lt; · · · , t ← N = T -α . Lower case p t represents the density of P t . We consider the exponential integrator scheme for simulating the backward SDE with

The generation algorithm can be expressed as a piecewise continuous-time SDE: for any t ∈ [ t ← j , t ← j +1 ) .

<!-- formula-not-decoded -->

Denote q t := Law ( ¯ Y t ) , ∀ t ∈ [0 , T -δ ] .

Theorem 2. [4, Theorem 1] Let Assumption 1 hold. Then there exists a numerical constant C 0 &gt; 0 , such that

<!-- formula-not-decoded -->

where E D ≤ κ 2 Nu 2 2 + κTu 2 2 is the discretization error due to the reverse SDE, E F ≤ exp( -2 T ) u 2 2 is the error due to the convergence of the forward SDE and E S is the score estimation error

<!-- formula-not-decoded -->

where γ j := t ← j +1 -t ← j , ∀ j = 0 , 1 , · · · N -1 is the step-size of the generation algorithm.

When the training is done over the forward discretization given by ( t N -j = T -t ← j ) N -1 j =0 , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Theorem 3. (Appendix B and [4, Theorem 1]) Let Assumption 1 hold. Then there exists a numerical constant C 0 &gt; 0 , such that

<!-- formula-not-decoded -->

where E D ≤ κ 2 Nu 2 2 + κTu 2 2 is the discretization error due to the reverse SDE, E F ≤ exp( -2 T ) u 2 2 is the error due to the convergence of the forward SDE.

## B.1 Decomposition of L y ( θ y )

Let θ ∗ y = argmin θ y L y ( θ y ) . We further decompose L y ( θ y ) as

<!-- formula-not-decoded -->

where ( a ) follows from the fact that L y reg ( B ( θ -y ) , θ -y ) ≤ L y reg ( θ y , θ -y ) . We further decompose this to obtain an upper bound on max y ∈ Y min t L y ( θ t y )

<!-- formula-not-decoded -->

where ( a ) follows from adding and subtracting the empirical losses ¯ L n y reg ( θ y , θ -y ) and ¯ L n y reg ( B y ( θ -y ) , θ -y ) and using triangle inequality of the max norm, ( b ) follows from the gradient domination property for strongly convex functions, ( c ) follows from taking the minimum over the iterates of the algorithm.

## B.2 Boundedness of Forward Dynamics

Lemma1. Consider the forward diffusion process with linear drift coefficients. For any δ &gt; 0 , δ ≪ 1 , w.p. (with probability) of atleast 1 -δ . we have

<!-- formula-not-decoded -->

where C T := max t ∈ [0 ,T ] r ( t ) , r ( t ) v ( t ) .

Proof: The proof is similar to [17, Lemma 1] When the drift coefficient f ( ., t ) : R d → R d is linear in x i.e. f ( x, t ) = -f ( t ) x , the transition kernel p t | 0 has a closed form

<!-- formula-not-decoded -->

where µ ( t ) := exp ( ∫ t 0 f ( ξ ) dξ ) , ¯ σ 2 ( t ) := 2 ∫ t 0 exp(2 µ s -2 µ t ) σ 2 s ds . Together we get,

<!-- formula-not-decoded -->

For any ϵ ∼ N (0 , 1) , c &gt; 1 , we have

<!-- formula-not-decoded -->

Let δ = √ 2 π e -c 2 2 , then

<!-- formula-not-decoded -->

Hence, for any δ ∈ (0 , 1) with δ ≪ 1 , w.p. at least 1 -δ , we have

<!-- formula-not-decoded -->

## B.3 Boundedness of Loss function ¯ L n y reg ( θ y , θ -y )

In this section, study some properties of the game defined by ⟨Y , ( ¯ L n y reg ) y ∈Y , (Θ y ) y ∈Y ⟩ . From Eq. 8, we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Conditioning on X 0 and using law of iterated expectation, we can write [32, Appendix A], we get

<!-- formula-not-decoded -->

where C t ( y ) = E X t ∥∇ log p t ( X t | y ) ∥ 2 -E X 0 E X t | X 0 ∥∇ log p t ( X t | X 0 , y ) ∥ 2 to learn the score ∇ x ( t ) log p t ( x ( t ) | x 0 , y ) .

Furthermore, we discretize the time points 0 = t 0 &lt; t 1 &lt; · · · &lt; t N = T to the objective function

<!-- formula-not-decoded -->

where ¯ C ( y ) = 1 2 ∑ N j =1 λ ( t j )( t j -t j -1 ) C t j ( y ) From [32, Appendix A], we have X t | X 0 ∼ N ( e -µ t X 0 , ¯ σ 2 t I ) and its density function is

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

Let ξ = ϵ t ¯ σ t ∼ N (0 , I )

<!-- formula-not-decoded -->

Finally putting all of it together, we get the empirical loss function

<!-- formula-not-decoded -->

We will show that the empirical loss function for the label y ∈ Y , ¯ L n y reg ( θ y , θ -y ) that is optimized is convex and smooth in θ y with high probability.

Lemma 2. For δ &gt; 0 , δ ≪ 1 , wp. 1 -n y Nδ , the empirical loss function

<!-- formula-not-decoded -->

is bounded i.e.

<!-- formula-not-decoded -->

Proof: From Lemma 1, we have δ &gt; 0 , δ ≪ 1

<!-- formula-not-decoded -->

Thus, w.p . 1 -n y Nδ , we have | ξ ij | ≤ √ 2 πδ 2 and hence we have ∥ x ( t j ) ∥ ∞ ≤ C t N ,δ , ∀ i = 1 , · · · , n y and j = 1 , , · · · , N Thus, w.p. 1 -n y Nδ

<!-- formula-not-decoded -->

For a bound on ∥ ∥ s t j ,θ y ( x ( t j )) ∥ ∥ 2

<!-- formula-not-decoded -->

where ( a ) follows from triangle inequality for norms, ( b ) follows from the fact that the ReLu function satisfies | σ ( x ) | ≤ | x | and Holder inequality and ( c ) follows from the bounds on the embeddings and x ( t j ) with ∥ w y,i ∥ 1 , ∥ u y,i ∥ 1 ≤ C w y ,u y , ∀ i ∈ [ m ] . Thus, for δ &gt; 0 , δ ≪ 1 , we have w.p. 1 -n y Nδ

<!-- formula-not-decoded -->

where C 1 = (¯ σ 2 t N + 2)( C t n ,δ + C t N ,e ) 2 C 2 w y ,u y B 2 + 2 πδ 2 . Since ¯ σ t j is non-decreasing in j , so max j ¯ σ t j = ¯ σ t N .

## B.4 Boundedness of Gradient of Loss function ¯ L n y reg ( θ y , θ -y )

<!-- formula-not-decoded -->

Since w.p. 1 -n y Nδ the empirical loss function ¯ L n y reg ( θ y , θ -y ) is bounded, ∥ ∥ ∇ A y ¯ L n y reg ( θ y , θ -y ) ∥ ∥ 2 F is bounded with the same probability.

This also shows that for fixed θ -y , ( W y , U y ) y ∈Y , w.p. 1 -n y Nδ , ¯ L n y reg ( θ y , θ -y ) is a Lipschitz function in θ y with Lipschitz constant σ y such that σ 2 y = 4 C 1 Nd 2 ( C t N ,δ + C t N ,e ) 2 C 2 w y ,u y max j { λ ( t j )( t j -t j -1 )¯ σ t j , βω ( t j )( t j -t j -1 ) } ( ∑ N j =1 λ ( t j )( t j -t j -1 ) ¯ σ t j + βω ( t j )( t j -t j -1 ) )

## B.5 Smoothness of Loss Function ¯ L n y reg ( θ y , θ -y )

Lemma 3. Let ( W y , U y ) y ∈Y , θ -y , { t j } N j =1 be fixed. Let L y = d ( C t N ,δ + C t N ,e ) 2 C 2 w y ,u y ∑ N j =1 ( λ ( t j )( t j -t j -1 )¯ σ t j + βω ( t j )( t j -t j -1 ) ) . Then for δ &gt; 0 , δ ≪ 1 , w.p. 1 -n y Nδ , ¯ L n y reg ( θ y , θ -y ) is L y m 2 smooth and convex in θ y .

Proof We have,

<!-- formula-not-decoded -->

To show smoothness, we will show that the function f ( θ y ) = ∥ ∥ ¯ σ t j s t j ,θ y ( x ( t j )) + ξ ij ∥ ∥ 2 2 and g ( θ y ) = ∥ ∥ ∥ s t j ,θ y ( x t j ) -s t j ,θ y ′ ( x ( t j )) ∥ ∥ ∥ 2 2 are individually smooth. Once we prove this, it is easy to show ¯ L n y reg ( θ y , θ -y ) is smooth as the linear combination of smooth functions is again smooth. To show smoothness, we need to show that ∥ ∥ ∥ ∇ 2 θ y f ( θ y ) ∥ ∥ ∥ and ∥ ∥ ∥ ∇ 2 θ y g ( θ y ) ∥ ∥ ∥ have a bounded norm. Recall that s t,θ y ( x ) = 1 m A y σ ( W y x ( t ) + U y e ( t ) . Let h 1 ( x, t ) := σ ( W y x + U y e ( t )) , h 2 ( x, t ) := s t,θ y ′ ( x ) , h 3 ( i, j ) = ξ ij , we have

<!-- formula-not-decoded -->

where ( a ) follows from the identity x T Ay = trace ( Byx T ) , ( b ) follows from the following identities

<!-- formula-not-decoded -->

and B 3 = h 1 ( x ( t j ) , t j ) h T 3 ( i, j ) .

Similarly, we have for g ( θ y )

<!-- formula-not-decoded -->

where B 1 := h 1 ( x ( t j ) , t j ) h T 1 ( x ( t j ) , t j ) and B 2 := h 1 ( x ( t j ) , t j ) h T 2 ( x ( t j ) , t j ) . Thus,

<!-- formula-not-decoded -->

The eigenvalues of ( B 1 ⊗ I ) is the same as B 1 with multiplicity. Thus, to show smoothness, we need to bound the maximum eigenvalues of B 1 . For any v ∈ R m

<!-- formula-not-decoded -->

Now,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, we have for any v ∈ R m

<!-- formula-not-decoded -->

Since w.p. 1 -n y Nδ we have {∥ x ij ∥ ∞ ≤ C t N ,δ } n y ,N i =1 ,j =1 , we have with the same probability f ( θ y ) and g ( θ y ) are smooth in θ y for every W y , U y , x ( t j ) , θ -y .

<!-- formula-not-decoded -->

## B.6 Proof: First order convergence of the algorithm

Proof Our proof follows closely along the lines of [13]. Let ¯ L n y reg ( θ y , θ -y ) be the empirical version of L y reg ( θ y , θ -y ) with n y samples. By L y smoothness of ¯ L n y reg ( θ y , θ -y ) we have, for any y ∈ Y ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ψ ( θ τ +1 y , θ τ y , θ τ -y ) = ¯ L n y mut ( θ τ y , θ τ -y ) -¯ L n y mut ( θ τ +1 y , θ τ -y ) .

## B.6.1 Analyzing the Bias Term

Lemma 4. Suppose θ -y , ( W y , U y ) y ∈Y are fixed. Let ϕ y = d 1 . 5 N ( C t N ,δ + C t N ,e ) 2 C 2 w y ,u y B max j ω ( t j )( t j -t j -1 ) . Then for δ &gt; 0 , δ ≪ 1 , w.p. 1 -n y Nδ , we have

<!-- formula-not-decoded -->

is ϕ y Lipschitz in θ y .

## Proof:

<!-- formula-not-decoded -->

Since ¯ L n y mut ( θ y , θ -y ) ≤ ¯ L n y reg ( θ y , θ -y ) and w.p. 1 -n y Nδ , ¯ L n y reg ( θ y , θ -y ) is bounded. Thus, ∥ ∥ ∇ A y ¯ L n y mut ( θ y , θ -y ) ∥ ∥ 2 F is bounded and hence ¯ L n y mut ( θ y , θ -y ) is Lipschitz in θ y with ϕ y = d 1 . 5 N ( C t N ,δ + C t N ,e ) 2 C 2 w y ,u y B max j ω ( t j )( t j -t j -1 ) Here,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For ( θ y , θ -y ) , we have

<!-- formula-not-decoded -->

Since the strategy space for θ y is bounded in norm. We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.7 Monte Carlo Error of the Finite Neural Network

Observe that

<!-- formula-not-decoded -->

For each y ∈ Y , L y ( θ ∗ y ) is the optimal loss function for the unregularized version under the current hypothesis class. Let L y ( ¯ θ ∗ y ) be the optimal unregularized loss function under the continuous version of the random feature model. Then,

<!-- formula-not-decoded -->

## Proposition 2. Monte Carlo estimates . Define the Monte Carlo error

<!-- formula-not-decoded -->

Suppose that ∥ X (0) ∥ ∞ ≤ K and the trainable parameter a and embedding functions W,U,e ( . ) are both bounded. Then. given any ¯ θ . for any δ &gt; 0 , δ ≪ 1 , with probability of at least 1 -2 Nδ , there exists θ such that

<!-- formula-not-decoded -->

Proof. The proof closely along the line of [17]. Fix any ¯ θ . For notational convenience, we will drop y from θ y and ¯ θ y . For k = 1 , 2 , · · · d , define

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

where ( a ) follows from the fact that ( a + b ) 2 ≤ 2( a 2 + b 2 ) and Jensen's Inequality E 2 [ Z t,k ( W,U )] ≤ E W,U [ Z 2 t,k ( W,U )] . According to Lemma 1. for any δ &gt; 0 , δ ≪ 1 , w.p. atleast 1 -δ , we have

<!-- formula-not-decoded -->

If ( ˜ W, ˜ U ) is different from ( W,U ) at only one component indexed by i , we have w.p. 1 -δ

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) and ( b ) follows from triangle inequality | ∥ a ∥-∥ b ∥ | ≤ ∥ a -b ∥ and ∥ a -b ∥ ≤ ∥ a ∥ + ∥ b ∥ , ( c ) follows from the fact that | σ ( y ) | ≤ | y | , ( d ) follows from Lemma 1 and Holder Inequality, ( e ) follows from the bounds on ∥ w i ∥ 1 , ∥ u i ∥ 1 , x, | a i,k | , e ( t j ) .

Thus, w,p. 1 -δ , Z t,k ( W,U ) has bounded increment property. Using McDiarmid's inequality, w.p. 1 -2 δ , we have

<!-- formula-not-decoded -->

Now we compute

<!-- formula-not-decoded -->

̸

where ( b ) is due to Fubini's theorem, ( c ) is due to independence of sampling ( w i , u i ) and ( w j , u j ) , ( d ) is due to a j,k σ ( w T j x + u T j e ( t )) being an unbiased estimator of the continuous version of score network, ( e ) follows from V ar ( X ) ≤ E [ X 2 ] , ( f ) follows from | σ ( y ) | ≤ | y | and Holder's inequality. Thus. w.p. 1 -2 δ ,

<!-- formula-not-decoded -->

Finally, we have w.p. 1 -2 Nδ

<!-- formula-not-decoded -->

## B.8 Radamacher Complexity

In this section, we will bound the term related to the generalization bound

<!-- formula-not-decoded -->

̸

The Rademacher complexity of a real valued function class F is defined as:

<!-- formula-not-decoded -->

The variables σ 1 , · · · , σ m are iid Bernoulli random variables that take values { +1 , -1 } with equal probability and are independent of x 1 , · · · , x m . However, for our random feature model, we have a vector valued function class

<!-- formula-not-decoded -->

Theorem 4. [19, Theorem 3] Let X be nontrivial, symmetric and subgaussian. Then there exists a constant C &lt; ∞ , depending only on the distribution of X , such that for any countable set S and functions ψ i : S → R , ϕ i : S → l 2 , 1 ≤ i ≤ n satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the X ik are independent copies of X for 1 ≤ i ≤ n and 1 ≤ k ≤ ∞ and ϕ i ( s ) k is the k-th coordinate of ϕ i ( s ) . If X is a Rademacher variable we may choose C = √ 2 , if X is a standard normal C = √ π 2 .

Corollary 2. [19, Corollary 4] Let X be any set, ( x 1 , · · · , x n ) ∈ X n , let F be a class of functions f : X → l 2 and let h i : l 2 → R have Lipschitz norm L . Then

<!-- formula-not-decoded -->

where ϵ ik is an independent doubly indexed Rademacher sequence and f k ( x i ) is the k-th component of f ( x i ) .

Lemma 5. [19] Consider the function class F = { x → A m ϕ ( x, W, U ) : A ∈ B ( H, R ) , ∥ A ∥ F ≤ B } . Then the empirical Rademacher complexity of F is

<!-- formula-not-decoded -->

Moreover, if E x ∥ ϕ ( x, W, U ) ∥ 2 ≤ C 2 , the Rademacher Complexity of F is

<!-- formula-not-decoded -->

we have

Proof:

<!-- formula-not-decoded -->

where D ∈ B ( H, R K ) is the random transformation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus,

Thus,

<!-- formula-not-decoded -->

Suppose 0 &lt; t 1 &lt; · · · &lt; t N = T are the chosen points of discretization for training, we have from the forward process

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the above bounds along with bounded support of embedding matrices W,U and embedding function e ( t ) and Assumption 1, it is easy to show that

<!-- formula-not-decoded -->

for some constant F 2 T and x ( t ) = e -t x (0) + √ 1 -e -2 t ξ j , ξ j ∼ N (0 , I ) Lemma 6. The term

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: Using the fact of bounded support of embedding matrices W,U and embedding function e ( t ) , bounded strategy space and Assumption 1 and eq 114, we get the desired bounded.

Lemma 7. Suppose L C 1 = ¯ σ 2 t j BF T + √ d √ log 2 πδ 2 ) . Then, with probability 1 -δ , the function h : A ⊂ R d → R

<!-- formula-not-decoded -->

is Lipschitz in x , where A = { x ∈ R d : ∥ x ∥ 2 ≤ F T B } .

Proof. It is sufficient to show the norm of the gradient of h ( x ) is bounded for x ∈ A . With probability 1 -δ ,

<!-- formula-not-decoded -->

(118)

Lemma 8. Suppose L C 2 = 2 F T B |Y| . Define g : A Y ⊂ R d |Y| → R where

<!-- formula-not-decoded -->

is Lipschitz in x , where A = { x ∈ R d : ∥ x ∥ 2 ≤ F T B } .

Proof:

We know

<!-- formula-not-decoded -->

where C t ( y ) = E X t ∥∇ log p t ( . | y ) ∥ 2 -E X 0 E X t | X 0 ∥∇ log p t ( x t | x 0 , y ) ∥ 2 . Let ¯ C ( y ) = 1 2 ∑ N j =1 λ ( t j )( t j -t j -1 ) C t j ( y )

Lemma 9. With probability 1 -Nn y δ , an upper bound for the generalization gap i.e.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is

<!-- formula-not-decoded -->

Proof. Observe that, we can rewrite Eq. 126 using triangle inequality as

<!-- formula-not-decoded -->

Further decomposing them, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

where

<!-- formula-not-decoded -->

Finally using Corollary 2, Lemmas 5 7,8, [19, Section 4.1] we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

## C Numerical Experiments

Computing resources. The numerical experiments were conducted on a MacBook Air (2023) and Gilbreth. Gilbreth has heterogeneous hardware comprising of Nvidia V100, A100, A10, and A30 GPUs in separate sub-clusters. All the nodes are connected by 100 Gbps Infiniband interconnects. We used sub-cluster B with 16 nodes, 24 cores per node, 192 GB memory per node, 3 A30 (24 GB) per node. For more information follow this link.

## C.1 Gaussian Mixture Models

Dataset We perform empirical experiments on synthetic datasets to verify our theoretical findings. The synthetic dataset is randomly generated under the true distribution and fixed. We detail out the underlying distribution on a case by case basis.

Implementation Details We employ the random feature model with the width of network m = 16 , learning rate η τ = 10 -4 , ∀ τ, T train = 5000 is fixed for Adam optimizer. We set λ ( t ) = ¯ σ t , ω ( t ) = e t , total number of training samples is 50.

Case one We perform more empirical experiments on d = 1 , imbalance ratio r = 2 . 5 , β = 0 . 01 . We compute the KL-divergence between the ground truth distribution and the learned model using the procedure in [17]. P ( x | y = 1) ∼ N ( -µ, σ 2 ) and class 2 is P ( x | y = 2) ∼ N ( µ, σ 2 ) . We observe Fig. 2 the worst case KL divergence for the mutual learning case is lower than the vanilla when we change the distance between mean and the variance of each class label. The performance of head class doesn't worsened for small µ . However, the head class performance suffers for mutual learning case when the distance between the mean increases. This might be because when the support of class distribution are farther apart mutual learning is not advantageous as transfer of knowledge between the class is not useful.

Case two We now consider a case with two classes with imbalance ratio r = 2 . 5 , β = 0 . 01 . Class 1 itself is a uniform mixture of two Gaussian i.e P ( x | y = 1) ∼ 1 2 N ( -4 , 3) + 1 2 N (4 , 3) and class 2 is P ( x | y = 2) ∼ N (0 , 2) as in Fig. 3. We observe the Mutual Learning objective with our formulation have lower KL-divergence for both the classes compared to the vanilla diffusion models trained on each class. In this case, mutual learning allows useful transfer of knowledge between the classes increasing the performance for both. We hypothesize that under some notion of similarity between various class distributions, mutual learning is advantageous in improving the performance of all classes.

<!-- image -->

Figure 2: Case one: (Left) The first plot shows the KL-divergence for each class with and without mutual learning objective as µ is varied. (Right) shows the KL-divergence for each class with and without mutual learning objective as σ is varied ( µ = 2 fixed).

<!-- image -->

Epoch

Figure 3: Case two: (Top) The first plot shows class 1 as a gaussian mixture with class 2 as Gaussian. (Bottom Left) Shows the KL-divergence for each class with and without mutual learning objective. (Bottom Right) Shows min τ max y ∈Y ∥ ∥ ∇ ¯ L n y reg ( θ τ y , θ τ -y ) ∥ ∥ decreasing with training epoch.

Figure 4: FID and IS Scores for Different β and η Values

| Different β Values ( η = 2 × 10 - 4 )   | Different β Values ( η = 2 × 10 - 4 )   | Different β Values ( η = 2 × 10 - 4 )   | Different η Values ( β = 0 . 1 )   | Different η Values ( β = 0 . 1 )   | Different η Values ( β = 0 . 1 )   |
|-----------------------------------------|-----------------------------------------|-----------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| Method                                  | FID( ↓ )                                | IS( ↑ )                                 | Method                             | FID( ↓ )                           | IS( ↑ )                            |
| β = 0 . 0                               | 16 . 58                                 | 8 . 78 ± 0 . 15                         | η = 2 × 10 - 4                     | 18 . 61                            | 8 . 94 ± 0 . 10                    |
| β = 0 . 1                               | 18 . 61                                 | 8 . 94 ± 0 . 10                         | η = 10 - 4                         | 14 . 58                            | 8 . 92 ± 0 . 19                    |
| β = 1 . 0                               | 16 . 74                                 | 8 . 55 ± 0 . 21                         | η = 10 - 5                         | 18 . 62                            | 8 . 62 ± 0 . 21                    |

Figure 5: Visualization of image generated from Vanilla DDPM ( β = 0 ) (Left) and Mutual Learning (Right)

<!-- image -->

## C.2 Experiments on CIFAR10LT

In this section, we present the numerical results for varying hyperparamters η and β values. Furthermore, for completeness, we provide visualization of the images generated from various methods.

Figure 6: Image Visualization of CBDM

<!-- image -->