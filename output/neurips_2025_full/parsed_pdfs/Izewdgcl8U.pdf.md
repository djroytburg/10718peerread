## How Does Label Noise Gradient Descent Improve

## Generalization in the Low SNR Regime?

Wei Huang 1 , 2 ∗ , Andi Han 3 , 1 ∗ , Yujin Song 4 , 1 , Yilan Chen 5 , Denny Wu 6 , 7 , 8 4 , 1

Difan Zou , Taiji Suzuki

1 RIKEN AIP 2 The Institute of Statistical Mathematics

3 The University of Sydney 4 The University of Tokyo 5 University of California San Diego 6 7 8

New York University Flatiron Institute The University of Hong Kong

wei.huang.vr@riken.jp; andi.han@sydney.edu.au;

y.song.research@gmail.com; yic031@ucsd.edu; dennywu@nyu.edu;

dzou@hku.hk; taiji@mist.i.u-tokyo.ac.jp

## Abstract

The capacity of deep learning models is often large enough to both learn the underlying statistical signal and overfit to noise in the training set. This noise memorization can be harmful especially for data with a low signal-to-noise ratio (SNR), leading to poor generalization. Inspired by prior observations that label noise provides implicit regularization that improves generalization, in this work, we investigate whether introducing label noise to the gradient updates can enhance the test performance of neural network (NN) in the low SNR regime. Specifically, we consider training a two-layer NN with a simple label noise gradient descent (GD) algorithm, in an idealized signal-noise data setting. We prove that adding label noise during training suppresses noise memorization, preventing it from dominating the learning process; consequently, label noise GD enjoys rapid signal growth while the overfitting remains controlled, thereby achieving good generalization despite the low SNR. In contrast, we also show that NN trained with standard GD tends to overfit to noise in the same low SNR setting and establish a non-vanishing lower bound on its test error, thus demonstrating the benefit of introducing label noise in gradient-based training.

## 1 Introduction

The success of deep learning across various domains [34, 48, 7] is often attributed to their ability to extract features [20, 15] via gradient-based training [14, 3]. One desirable property of gradient-based feature learning is the algorithmic regularization that prioritizes learning of the underlying signal instead of overfitting to noise: real-world data contains noise due to mislabeling, data corruption, or inherent ambiguity, yet despite having the capacity to memorize noise, neural networks (NNs) trained by gradient descent (GD) tend to identify informative features and 'low-complexity' solutions that generalize [57, 43].

To understand this behavior, recent theoretical works considered data models that partition the features into signal and noise components [19, 5, 54], and studied the performance of gradient-based training in different signal-noise conditions. Among existing theoretical settings, the signal-noise model proposed in [2, 8] has been extensively studied in the feature learning theory literature. In this model, input features are constructed by combining a label-dependent signal with label-independent noise . The signal represents meaningful patterns relevant to the predictive task while the noise

∗ Equal Contribution

component captures background features unrelated to the learning task. This idealized setting has shed light on how various algorithms, neural network architectures, and other factors influence optimization and generalization of neural networks, depending on the signal-to-noise ratio (SNR) [18, 58, 29, 25, 56, 9, 27, 22, 26, 23, 35].

In such model, it is known that the SNR dictates a transition from benign overfitting to harmful overfitting . In the high SNR regime, gradient-based feature learning prioritizes signal learning over noise memorization; hence upon convergence, the trained NN recovers the signal and generalizes to unseen data despite some degree of noise memorization, a phenomenon known as benign overfitting [4, 51, 40, 44, 46, 31]. In contrast, when the SNR is low, noise memorization dominates the training dynamics, and the network fails to identify useful features before the training loss becomes small, leading to harmful overfitting [8, 33].

Given these challenges, recent works have explored algorithmic modifications that either enhance signal learning or suppress noise memorization, to improve generalization in the challenging low SNR regime. [25] showed that the smoothing effect of graph convolution in graph neural networks mitigates overfitting to noise; however, this approach requires the graph to be sufficiently dense and exhibits high homophily. [10] found that the sharpness-aware minimization (SAM) method [17] prevents noise memorization in early stages of training, thereby promoting effective feature learning; this being said, SAM has higher computational cost than standard GD due to the two forward and backward passes per step, and it involves more complex hyperparameter tuning. The goal of this work is to address the following question:

Is there a simple modification of GD with no computational overhead that achieves small generalization error in low SNR settings where standard GD fails to generalize?

## 1.1 Our contributions

We provide an affirmative answer to the question above by introducing random label noise to the training dynamics as a form of regularization, inspired by label noise (stochastic) gradient descent (GD) [6, 45, 49]. Specifically, we analyze the classification extension of label noise GD considered in [24, 13], where random label flipping is introduced to prevent overfitting.

Empirically , we first present findings in a controlled classification setting, where we train a VGG-16 model on (a subset of) the CIFAR-10 dataset. To modulate the SNR, we follow [19] and add varying levels of noise to the high-frequency Fourier components of the images - higher noise strength corresponds to lower SNR and vice versa. The results, shown in Figure 1, demonstrate that as the SNR decreases, the performance gap between label noise GD and standard GD becomes more significant, hence suggesting that label noise GD improves generalization in the low SNR regime. The goal of this work is to rigorously establish this separation in an idealized theoretical model .

Figure 1: Test accuracy of VGG-16 on CIFAR-10 with varying SNR . Label noise GD consistently outperforms standard GD, and the gap increases with the noise strength.

<!-- image -->

Theoretically , we characterize the properties of Label Noise GD in low SNR regimes, by considering the learning of a two-layer convolutional neural network in a binary classification problem studied in [8], and show that by randomly flipping the labels of a small proportion of training samples at each iteration, noise memorization can be suppressed despite the low SNR, whereas signal learning experiences a period of fast growth. As a result, neural network trained by label noise GD attains good generalization performance in regimes where standard GD fails, as summarized in the following informal theorem:

Theorem 1.1 (Informal) . Given n training samples drawn from the distribution in Definition 2.1 in the low SNR regime where n -1 SNR -2 = ˜ Ω(1) . Then for any ϵ &gt; 0 , after a polynomial number of training steps t (depending on ϵ ), with high probability we have: (i) Standard GD minimizes the logistic training loss to L ( t ) S ≤ ϵ , but the generalization error (0-1 loss) remains large, i.e., L ( t ) D = Ω(1) . (ii) Label noise GD cannot reduce the logistic training loss to a small value L ( t ) S = Ω(1) , but achieves small generalization error (0-1 loss), i.e., L ( t ) D = o (1) .

We make the following remarks on our main results.

- Improved Generalization due to Label Noise. The theorem provides an upper bound on the test error of label noise GD and lower bound on the error of standard GD. This demonstrates that incorporating label noise into the gradient descent updates improves generalization in the low SNR regime. We note that our conditions on label noise GD learnability are weaker than those required for SAM as specified in [10], even though our studied algorithm is arguably simpler and more computationally efficient - see Section 3 for more comparisons.
- Analysis of Feature Learning Dynamics. We establish the main theorem via a refined characterization of the training dynamics of label noise GD on a two-layer convolutional NN with squared ReLU activation. A key observation in our analysis is that label noise introduces regularization to the noise memorization process, preventing it from growing beyond a constant level; meanwhile, signal learning continues to exhibit a rapid growth rate, allowing the model to identify the informative features and avoid harmful overfitting in low SNR regimes.

## 2 Problem Setup

In this section, we describe the signal-noise data model, the neural network architecture used for training, and the label noise gradient descent algorithm considered in this work.

Data generating process. We consider the signal-noise data model from [8, 10, 25]. Let µ ∈ R d be a fixed signal vector, and for each data point ( x , y ) , the feature x is composed of two patches, denoted as x = { x (1) , x (2) } ∈ R 2 d . The target variable y is a binary label, taking values in {± 1 } . Then the data is generated according to the following process.

Definition 2.1. We consider the following generating process for ( x , y ) :

1. The true label y is drawn from a Rademacher distribution, i.e., P [ y = 1] = P [ y = -1] = 1 / 2 .
2. One of the patches, x (1) or x (2) is randomly selected to be y µ (the signal), while the other is set to be ξ i ∼ N (0 , σ 2 p ( I d -µµ ⊤ ∥ µ ∥ -2 2 )) (the noise). Here, σ 2 p denotes the strength of the noise vector.

We make the following remarks on the data distribution.

- The data model simulates a setting where the input features are composed of both signal and noise components. Specifically, each data point is divided into two patches, and one of these patches contains meaningful information (signal) related to the classification label, while the other patch only contains random noise independent of the label. The noise covariance σ 2 p ( I d -µµ ⊤ ∥ µ ∥ -2 2 ) is set to ensure that the noise vector is orthogonal to the signal vector for simplicity.
- This setup is designed to reflect real-world scenarios where data contains a mix of relevant and irrelevant features (see Appendix A in [2] for discussions). Note that in high dimensions ( n ≪ d ), the NN can achieve small training loss just by overfitting to the noise component. Therefore, the challenge for the learning algorithm in the low SNR regime is to identify and learn the signal patch while ignoring the noisy patch.
- We use the minimum number of patches in the multi-patch model for concise presentation. Our results can be extended to more general cases where the number of patches is greater than 2; see [2, 47] for such extension.

Neural network and loss function. Following [8], we consider a two-layer convolutional neural network with squared ReLU activation and shared filters applied separately to each patch. The network is defined as f ( W , x ) = F +1 ( W +1 , x ) -F -1 ( W -1 , x ) , where

<!-- formula-not-decoded -->

in which m denotes the size of the hidden layer, and σ ( z ) = (max { 0 , z } ) 2 . Note that j ∈ {-1 , +1 } corresponds to the fixed second-layer. The symbol W j represents the collection of weight vectors in

| Algorithm 1 Label noise gradient descent   | Algorithm 1 Label noise gradient descent                                                                                            |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| 1:                                         | Initialize W 0 , step size η , flipping probability p ∈ [0 , 1]                                                                     |
| 2:                                         | for t = 0 , ...,T - 1 do                                                                                                            |
| 3:                                         | Sample ϵ ( t ) i ∼ Rademacher(1 - p, p ) , ∀ i ∈ [ n ] .                                                                            |
| 4:                                         | W ( t +1) = W ( t ) - η ∇ W L ϵ S ( W ( t ) ) , where L ϵ S ( W ( t ) ) = 1 n ∑ i ∈ [ n ] ℓ ( ϵ ( t ) i y i f ( W ( t ) , x i ) ) . |
| 5: end for                                 | 5: end for                                                                                                                          |

the first layer, i.e., W j = [ w j, 1 , w j, 2 , . . . , w j,m ] ∈ R d × m , where w j,r ∈ R d is the weight vector of the r -th neuron. Here, j ∈ {-1 , +1 } indicates the fixed value in the second layer. The initial weights W ± 1 has entries sampled from N (0 , σ 2 0 ) .

Remark 2.2 . Since we do not optimize the 2nd-layer parameters, we expect the 2-homogeneous squared ReLU activation to mimic the behavior of training both layers simultaneously in a ReLU network; such higher-order homogeneity amplifies feature learning (e.g., see [12, 21]) and creates a significant gap between signal learning and noise memorization. Similar effect can be achieved by smoothed ReLU with local polynomial growth as in [2, 47].

We use the logistic loss computed over n training samples, denoted as S = { ( x i , y i ) } i ∈ [ n ] :

<!-- formula-not-decoded -->

where ℓ ( z ) = log(1 + exp( -z )) . To evaluate the generalization performance of the trained network, we measure its expected 0-1 loss on unseen data, defined as

̸

<!-- formula-not-decoded -->

where D denotes the data distribution specified in Definition 2.1, and 1 ( · ) is the indicator function.

Label noise GD for binary classification. We train the above neural network by gradient descent on either (i) the original loss function (standard GD), or (ii) the loss function with label-flipping noise defined as

<!-- formula-not-decoded -->

Here, ϵ ( t ) i is a random variable equal to 1 with probability 1 -p and -1 with p , i.e., ϵ ( t ) i ∼ Rademacher(1 -p, p ) . In other words, labels flip with probability p independently at each step. Remark 2.3 . We remark that label smoothing [45, 49] and label flipping are equivalent in expectation. This connection has also been discussed in [38]. However, note that this equivalence in expectation does not imply closeness in training dynamics due to the stochasticity introduced by the label-flipping.

The label noise GD update is then given as follows:

<!-- formula-not-decoded -->

where η is the learning rate, and we defined ˜ ℓ ′ ( t ) i = ℓ ′ ( ϵ ( t ) i y i f ( W ( t ) , x i )) as the derivative of the loss function. This label noise GD training procedure is outlined in Algorithm 1. Observe that the proposed algorithm is computationally efficient , as the introduced label noise does not modify the original gradient descent framework. Hence this method is simple to implement, does not add significant computational overhead, and requires no complex hyperparameter tuning.

## 3 Main results

In this section, we quantify the benefits of label noise gradient descent by comparing its generalization performance against standard gradient descent (GD) training without label noise. We begin by outlining the assumptions that apply to both label noise GD and standard GD.

Assumption 3.1. Define SNR = ∥ µ ∥ 2 σ p √ d . We consider the following setting for both algorithms:

- (i) data dimension d = ˜ Ω(max { n 2 , n ∥ µ ∥ 2 2 /σ 2 p } ) ; signal-to-noise ratio SNR = ˜ O (1 / √ n ) .
- (ii) network width m = ˜ Ω(1) ; number of training samples n = ˜ Ω(1) .
- (iii) learning rate η ≤ ˜ O ( σ -2 p d -1 ) .
- (iv) initialization variance ˜ O ( nσ -1 p d -3 / 4 ) ≤ σ 0 ≤ ˜ O (min {∥ µ ∥ -1 2 d -5 / 8 , σ -1 p d -1 / 2 } ) .
- (v) flipping rate of label noise p lies in the interval p ∈ ( C log d √ mn , 1 C ) , where C is a sufficient large constant.

We make the following remarks on the above assumption.

- The high-dimensional assumption ( i ) is standard in the benign overfitting analysis of NNs (e.g., see [8, 18]). The low SNR condition is derived from the comparison between the magnitude of signal learning and noise memorization - see Section 4.1; similar conditions has been established in [8, 33] for different activations.
- The requirements on the hidden layer size m and the sample size n being at least polylogarithmic in the dimension d ensure that certain statistical properties regarding weight initialization and the training data hold with high probability at least 1 -1 /d .
- The upper bound on the learning rate η ensures that the iterates in (4-6) remain bounded, which is required for standard GD to reach low training loss; see Proposition 4.1.
- The upper bound on initialization scale σ 0 is used to ensure convergence of GD, and the lower bound is used for anti-concentration at initialization. Similar requirements can be found in [8, Condition 4.2].
- The lower bound ensures that the number of flipped samples concentrates around its expectation so that our theoretical analysis remains valid, while the upper bound on label flipping rate p prevents the label noise from dominating the true signal.

We first state the negative result for standard gradient descent (GD) without label noise.

Theorem 3.2 (GD fails to generalize under low SNR) . Under Assumption 3.1, for any ϵ &gt; 0 , there exists t = Θ( nm log(1 / ( σ 0 σ p √ d )) ησ 2 p d + m 3 n ηϵσ 2 p d ) , such that with probability at least 1 -d -1 / 4 , it holds that

- The training error converges, i.e., L S ( W ( t ) ) ≤ ϵ .
- The test error is large, i.e., L D ( W ( t ) ) ≥ 0 . 24 .

Theorem 3.2 indicates that even though standard GD can minimize the training error to an arbitrarily small value, the generalization performance remains poor. This is mainly because the neural network overfits to the noise components in the input data instead of learning the useful features. Next, we present the positive result for label noise GD.

Theorem 3.3 (Label Noise GD generalizes under low SNR) . Under Assumption 3.1, there exists t = Θ( nm log(1 / ( σ 0 σ p √ d )) ησ 2 p d + m log(6 / ( σ 0 ∥ µ ∥ 2 )) η ∥ µ ∥ 2 2 ) and constants C &gt; 0 , such that with probability at least 1 -d -1 / 4 , it holds that

- The training error is at constant order, i.e., L S ( W ( t ) ) = Θ(1) .
- The test error is small, i.e., L D ( W ( t ) ) ≤ 2 exp ( -Cd n 2 ) .

Theorem 3.3 shows that label noise GD achieves vanishing generalization error when the input dimensionality is large (i.e., d = Ω( n 2 ) ) despite the low SNR.

Remark 3.4 . Theorems 3.3 and 3.2 present contrasting outcomes for standard GD and label noise GD in the low SNR regime. In particular,

- Standard GD minimizes the training error effectively but does so by primarily overfitting to noise in the training data. This significant noise memorization leads to harmful overfitting.

- In contrast, label noise GD introduces a regularization effect through label noise, which prevents the network from fully memorizing the noise components. This allows the network to focus on learning the true signal, resulting in a phase of accelerated signal learning. Consequently, the model generalizes even though the training loss does not vanish (due to noise injection).

Comparison with sharpness-aware minimization [10]. We briefly discuss the differences between our findings and those in [10] for the sharpness-aware minimization (SAM) method, where the authors established conditions on the SNR under which SAM can generalize better than stochastic gradient descent (SGD). However, their analysis requires the additional condition that the signal norm satisfies ∥ µ ∥ 2 ≥ ˜ Ω(1) , indicating the necessity of a sufficiently strong signal. In contrast, we show that label noise GD enjoys good generalization without this strong signal condition. This highlights the robustness of label noise GD in low SNR regimes (even when the signal strength is considerably weaker compared to the noise).

Comparison with stopping times across theorems. We compare the stopping times in Theorems 3.2 and 3.3. The stopping times for standard GD and label noise GD are not directly comparable, as they correspond to different evaluation criteria. Specifically, the stopping time for standard GD is the number of iterations required for the training loss to converge below a threshold ϵ , whereas the stopping time for label noise GD is defined as the number of iterations needed to achieve sufficiently low 0 -1 test loss. To enable a meaningful comparison, we derive the ratio between the two stopping times under Assumption 3.1. By setting m 2 = log ( 6 ) /ϵ , we obtain t Standard GD =

σ

0

∥

µ

∥

2

t

label noise GD

Θ ( n ∥ µ ∥ 2 2 σ 2 p d ) = Θ( n SNR 2 ) . According to Assumption 3.1, we assume n SNR 2 ≪ 1 , which implies that label noise GD requires more iterations to achieve good test performance compared to the time required for the training loss of standard GD to converge.

## 4 Proof Sketch

In this section, we give an overview of of our analysis of the optimization dynamics of standard GD and label noise GD . Our key technical contributions are summarized as follows: (i) Boundary characterization in low SNR regimes. Unlike previous studies [8, 33, 10] that focus on the higher polynomial or standard ReLU activation, we analyze the 2-homogeneous squared ReLU activation, leading to a different boundary characterization of the low SNR regime for standard GD - see Section 4.2. (ii) Upper bound via supermartingale. We apply supermartingale arguments with Azuma's inequality to bound noise memorization in label noise GD. This yields high-probability guarantees on training dynamics, previously unestablished in this context.

## 4.1 Signal-noise decomposition

To analyze the training dynamics, we adopt a parameter decomposition technique from [8, 33]: there exist { γ ( t ) j,r } and { ρ ( t ) j,r,i } such that

<!-- formula-not-decoded -->

This decomposition originates from the observation that the gradient descent update always evolves in the direction of µ and x i for i ∈ [ n ] . In particular, γ ( t ) j,r ≈ ⟨ w ( t ) j,r , µ ⟩ serves as the signal learning coefficient, whereas ρ ( t ) j,r,i ≈ ⟨ w ( t ) j,r , ξ i ⟩ characterizes the noise memorization during training. Next we let ρ ( t ) j,r,i = ρ ( t ) j,r,i 1 ( y i = j ) and ρ ( t ) j,r,i = ρ ( t ) j,r,i 1 ( y i = -j ) . Combined with the gradient descent update given by Equation (2), we obtain the iteration rules for these coefficients:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the initial values of the coefficients are given by γ (0) j,r = 0 and ρ (0) j,r,i = 0 for all i ∈ [ n ] , j ∈ {-1 , 1 } and r ∈ [ m ] .

To analyze the optimization trajectory, we track the dynamics of signal learning coefficients ( γ ( t ) j,r ) and noise memorization coefficients ( ρ ( t ) j,r,i ) using the iteration rules in Equations (4-6). To facilitate a detailed analysis, we first provide upper bounds on the absolute value of both the signal learning and noise memorization coefficients throughout the entire training process.

Proposition 4.1. Given Assumption 3.1 and ϵ &gt; 0 . Let β = 2max j,r,i {|⟨ w (0) j,r , µ ⟩| , |⟨ w (0) j,r , ξ i ⟩|} and α = 4log( T ∗ ) . For 0 ≤ t ≤ T ∗ , where T ∗ = η -1 poly( n, m, d, ∥ µ ∥ -1 2 , ( σ 2 p d ) -1 , σ -1 0 , ϵ -1 ) , for all i ∈ [ n ] , r ∈ [ m ] and j ∈ {-1 , 1 } , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof is in Appendix C. Proposition 4.1 shows that throughout training, the absolute values of signal learning and noise memorization coefficients have a logarithmic upper bound. This result is key for a stage-wise characterization of training dynamics. Notably, this bound holds for both standard GD and label noise GD.

## 4.2 Proof Sketch for Theorem 3.2

We first establish the negative result for standard GD based on a two-stage analysis. As previously mentioned, we consider the 2-homogeneous σ ( z ) = ReLU 2 ( z ) which differs from [8, 33, 10]. This leads to a key difference in the boundary characterization of the low SNR regime.

First stage. Notice that starting from small initialization, the loss derivative remains close to a constant. Based on this observation, we establish the difference in magnitude between the coefficients of signal learning and noise memorization.

According to the update rule for the signal learning coefficient given by Equation (4) and by setting ϵ ( t ) i = 1 for all t and i ∈ [ n ] (i.e., no label flipping), the upper bound of signal learning can be achieved as γ ( t ) j,r + |⟨ w (0) j,r , µ ⟩| ≤ exp ( 2 η ∥ µ ∥ 2 2 m t ) |⟨ w (0) j,r , µ ⟩| . Meanwhile, the bounds for the noise memorization coefficients can be derived from the update rules (5) and (6). The results are given as max j,r | ρ ( t ) j,r,i | ≤ 3 ησ 2 p td nm √ log(8 mn/δ ) σ 0 σ p √ d , and max j,r ρ ( t ) j,r,i ≥ exp ( ηC 1 σ 2 p d 2 nm t ) σ 0 σ p √ d/ 4 -0 . 6 β , for all i ∈ [ n ] , where we define ¯ β = min i ∈ [ n ] max r ∈ [ m ] ⟨ w (0) y i ,r , ξ i ⟩ , and use | ˜ ℓ ′ ( t ) i | ≥ C 1 . In the low SNR setting, where σ p √ d is much larger than ∥ µ ∥ 2 , we observe that noise memorization dominates the feature learning process during the first stage, as shown in the following lemma.

Lemma 4.2. Under the same condition as Theorem 3.2, and let T 1 = Θ( nm log(1 / ( σ 0 σ p √ d )) ησ 2 p d ) , the following results hold with high probability at least 1 -d -1 : (i) max j,r ρ ( T 1 ) j,r,i ≥ 1 , for all i ∈ [ n ] ; (ii) max j,r,i | ρ ( t ) j,r | ≤ ˜ O ( σ 0 σ p √ d ) , for all t ∈ [ T 1 ] ; (iii) max j,r γ ( t ) j,r ≤ ˜ O ( σ 0 ∥ µ ∥ 2 ) , for all t ∈ [ T 1 ] .

Lemma 4.2 indicates that when the SNR is sufficiently low, i.e., SNR = ˜ O (1 / √ n ) , noise memorization dominates the training dynamics during the early phase of standard GD optimization. We highlight that this 'low-SNR' condition differs from that of [8, 33] due to the choice of activation function. In particular, [8] assumed σ ( z ) = (max { 0 , z } ) q with q &gt; 2 and established a low-SNR boundary n -1 SNR -q = ˜ Ω(1) , whereas [33] considered the ReLU activation and derived the condition n ∥ µ ∥ 4 2 σ 4 p d ≤ O (1) .

Second stage. After the first stage, the loss derivative is no longer bounded by a constant value. To prove convergence of the training loss L ( t ) ≤ ϵ , we build upon the analysis from the first stage and define w ∗ j,r = w (0) j,r +2 m log(2 /ϵ ) ∑ n i =1 ∥ ξ i ∥ -2 2 ξ i . Weshow that, as gradient descent progresses, the

distance between W ( t ) and W ∗ decreases until L ( t ) ≤ ϵ : ∥ W ( t ) -W ∗ ∥ 2 F -∥ W ( t +1) -W ∗ ∥ 2 F ≤ ηL S ( t ) -ηϵ . Moreover, we show that the difference between signal learning and noise memorization still holds in the second stage, as summarized below.

Lemma 4.3. Let T 2 = η -1 σ -2 p d -1 nm log(1 / ( σ 0 σ p d )) + η -1 ϵ -1 m 3 nσ -2 p d -1 . Under the same assumptions as Theorem 3.2, for training step t ∈ [ T 1 , T 2 ] , it holds that γ ( t ) j,r ≤ ˜ O ( σ 0 ∥ µ ∥ 2 ) , | ρ ( t ) j,r,i | ≤ ˜ O ( σ 0 σ p √ d ) , and ρ ( t ) j,r,i ≥ 1 . Besides, there exists a step t ∈ [ T 1 , T 2 ] , such that L S ( t ) &lt; ϵ .

Lemma 4.3 shows that standard GD achieves low training error after polynomially many steps, and noise memorization dominates the entire training process, which results in harmful overfitting.

## 4.3 Proof Sketch for Theorem 3.3

We also divide the training dynamics of label noise GD into two phases. In the first phase, both signal learning and noise memorization increase exponentially despite the presence of random label noise. In the second phase, label noise suppresses the growth of noise memorization, causing it to oscillate within a constant range; meanwhile, signal learning continues to grow exponentially until stabilizing at constant value, which leads to beneficial feature learning and low generalization error.

First stage. Leveraging the fact that the derivative of the loss function remains within a constant range due to small initialization, we demonstrate that both signal learning and noise memorization exhibit exponential growth rates, even in the presence of label noise. According to the iterative update of the signal learning coefficient in Equation (4), the upper and lower bounds are given as γ ( t ) j,r + |⟨ w (0) j,r , µ ⟩| ≤ exp ( 2 η ∥ µ ∥ 2 2 m t ) |⟨ w (0) j,r , µ ⟩| , and max r ∈ [ m ] { γ ( t ) j,r + j ⟨ w (0) j,r , µ ⟩} ≥ exp( C 0 η ∥ µ ∥ 2 2 8 m ) ( max r ∈ [ m ] ⟨ w (0) j,r , µ ⟩ ) , respectively. Here C 0 is the lower bound for the absolute loss derivative. These bounds indicate that signal learning grows exponentially with the number of training iterations. On the other hand, from the update equation (5), we characterize the behavior of noise memorization. Despite the injected label noise, we can show a lower bound on the noise memorization rate: max j,r { ρ ( t ) j,r,i +0 . 6 |⟨ w (0) j,r , ξ i ⟩|} ≥ exp( ηC 0 σ 2 p d 2 nm ) |⟨ w (0) j,r , ξ i ⟩| . The main results for the first stage are summarized in the following lemma.

Lemma 4.4. Under the same condition as Theorem 3.3, and let T 1 = Θ( nm log((1 /σ 0 σ p d )) ησ 2 p d ) . Then the following holds with probability at least 1 -d -1 : (i) max j,r ρ ( T 1 ) j,r,i ≥ 0 . 1 , for all i ∈ [ n ] ; (ii) max j,r,i | ρ ( t ) j,r | ≤ ˜ O ( σ 0 σ p √ d ) , for all t ∈ [ T 1 ] . (iii) max j,r γ ( t ) j,r ≥ ˜ O ( σ 0 ∥ µ ∥ 2 ) , for all t ∈ [ T 1 ] .

Lemma 4.4 states that both signal learning and noise memorization grow exponentially during the first stage.

In order to guarantee that the number of flipped labels remains within its expected range with high probability, we require p = ˜ Ω(1 / √ t ) , where t is the number of training steps. We set t = ˜ Θ( n/ ( ησ 2 p d )) according to Lemma 4.4. Together with the upper bound on η from Assumption 3.1, we obtain p = ˜ Ω(1 / √ mn ) .

For the analysis of label noise GD, one additional technical challenge is the instability of training dynamics caused by the injected noise, which we address as follows. For signal learning, we make use of the small label flipping rate p and aggregate information across all samples via concentration. Whereas for noise memorization (which is tied to individual samples), we leverage the broad range of time steps in the first stage to establish the overall increment rate.

Second stage. As shown in Lemma 4.4, at the end of the first phase, noise memorization has reached a significant level, dominating the model's output. However, label noise introduces randomness in the labels, which affects the updates of noise memorization coefficients. We track the evolution of ρ ( t ) j,r,i via the following approximation. Define ι ( t ) i ≜ 1 m ∑ m r =1 ( ρ ( t ) y i ,r,i ) 2 . The evolution of noise memorization under label noise GD can be approximated as

<!-- formula-not-decoded -->

Unlike conventional approaches such as [8, 33], we analyze this process using a supermartingale argument and apply Azuma's inequality with a union bound over the second-stage training period. Via a martingale argument, we show that noise memorization remains at a constant level with high probability. While noise memorization stabilizes, signal learning continues to grow exponentially. This discrepancy enables signal learning to eventually dominate the generalization. The analysis of the second stage is summarized by the following lemma.

Lemma 4.5. Under the same condition as Theorem 3.3, during t ∈ [ T 1 , T 2 ] with T 2 = T 1 + log(6 / ( σ 0 ∥ µ ∥ 2 ))4 m (1 + exp( c 2 )) η -1 ∥ µ ∥ -2 2 , there exist a sufficient large positive constant C ι and a constant ι ∗ i depending on sample index i such that the following results hold with probability at least 1 -1 /d : (i) | ι ( t ) i -ι ∗ i | ≤ C ι ; (ii) γ ( t ) j,r ≤ 0 . 1 for all j ∈ {-1 , 1 } and r ∈ [ m ] (iii) 1 2 m ( ∑ m r =1 ρ ( t ) y i ,r,i ) 2 ≤ f ( t ) i ≤ 2 m ( ∑ m r =1 ρ ( t ) y i ,r,i ) 2 and (iv) max r ∈ [ m ] ( γ ( t ) j,r + |⟨ w (0) j,r , µ ⟩| ) ≥ exp ( η ∥ µ ∥ 2 2 16 m ( t -T 1 ) ) max r ∈ [ m ] | γ ( T 1 ) j,r + ⟨ w (0) j,r , µ ⟩| .

Lemma 4.5 demonstrates that label noise introduces a regularizing effect preventing the noise memorization coefficients from growing unchecked, while simultaneously allowing signal learning to grow to a sufficiently large value. Building on this result, we show that both signal learning and noise memorization reach a constant order of magnitude. Consequently, the population loss can be bounded by L D ( W ( t ) ) ≤ 2 exp ( -Cd n 2 ) , corresponding to the second bullet point of Theorem 3.3.

## 5 Synthetic Experiments

We conduct experiments using synthetic data to validate our theoretical results. The samples are generated according to Definition 2.1. The train and test sample size is n = 200 and n test = 2000 , and the input dimension is set to d = 2000 . The label noise flip rate is p = 0 . 1 . We train the two-layer network with squared ReLU activation using standard GD and label noise GD for t = 2000 steps. The network width is m = 20 and the learning rate is η = 0 . 5 . The signal vector is defined as µ = [2 , 0 , 0 , . . . , 0] ∈ R d and the noise variance is set to σ 2 p = 0 . 25 .

Dynamics of signal and noise coefficients. In Figure 2, we present the feature learning coefficients defined in Section 4.1, the training loss and test accuracy for both algorithms. We observe that GD successfully minimizes the training loss to a near-zero value; however, noise memorization ( ρ ) significantly exceeds signal learning ( γ ), leading to poor test performance. In contrast, label noise GD does not fully minimize the training loss, as it oscillates around 0.5; consistent with our theoretical analysis, this behavior causes noise memorization to remain constant in the second stage, while signal learning continues to grow rapidly. Hence the test accuracy of label noise GD steadily improves.

Figure 2: Ratio of noise memorization over signal learning, training loss, and test accuracy, of standard GD and label noise GD. See Section 4.1 for definitions of signal learning ( γ ) and noise memorization ( ρ ).

<!-- image -->

Heatmap of generalization error. Next we explore a range of SNR values from 0.03 to 0.10 and sample sizes n ranging from 100 to 700. For each combination of SNR and sample size n , we train the NN for 1000 steps with η = 1 . 0 using standard GD or label noise GD. The resulting test error is visualized in Figure 3. Observe that standard GD (left) fails to generalize when SNR = O ( n -1 / 2 ) , which is consistent with our theoretical prediction in Theorem 3.2. On the other hand, label noise GD (right) achieves perfect test accuracy across a broader range of SNR, which agrees with Theorem 3.3.

Additional Experiments and Extended Analysis. In addition to our primary experiments, we extend our analysis in Appendix G by evaluating Label Noise GD on deeper neural networks, modified MNIST and CIFAR datasets, different types of label noise (e.g., Gaussian), and higher-order ReLU activation functions, demonstrating its robustness across various settings.

Figure 3: Test accuracy heatmap of Standard GD (left) and Label Noise GD (right) after training.

<!-- image -->

## 6 Conclusion and limitation

We presented a theoretical analysis of gradient-based feature learning in the challenging low SNR regime. Our main contribution is to demonstrate that label noise gradient descent (GD) can effectively enhance signal learning while suppressing noise memorization; this implicit regularization mechanism enables label noise GD to generalize in low SNR settings where standard GD suffers from harmful overfitting. Our theoretical findings are supported by experiments on synthetic data.

Limitations and Broader Impacts. Our current theoretical analysis is limited to a specific choice of activation function (squared ReLU) and network architecture (two-layer convolutional neural network). Extending this theoretical framework to more complex architectures, such as deeper or residual networks, would be a promising direction for future research. Additionally, investigating label noise GD under other optimization algorithms, including stochastic gradient descent (SGD) and adaptive methods like Adam, could provide further insight into its implicit regularization effects in practical settings. This work aims to advance the theoretical understanding of generalization in neural networks. We are not aware of any immediate negative societal impacts resulting from this research.

## Acknowledgments

We thank the anonymous reviewers for their constructive comments. WH was supported by JSPS KAKENHI (24K20848) and JST BOOST (JPMJBY24G6). TS was partially supported by JSPS KAKENHI (24K02905) and JST CREST (JPMJCR2015). DZ is supported by NSFC 62306252 and central fund from HKU IDS.

## References

- [1] Zeyuan Allen-Zhu and Yuanzhi Li. Feature purification: How adversarial training performs robust deep learning. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS) , pages 977-988. IEEE, 2022.
- [2] Zeyuan Allen-Zhu and Yuanzhi Li. Towards understanding ensemble, knowledge distillation and self-distillation in deep learning. The Eleventh International Conference on Learning Representations , 2023.
- [3] Jimmy Ba, Murat A Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. Highdimensional asymptotics of feature learning: How one gradient step improves the representation. Advances in Neural Information Processing Systems , 35:37932-37946, 2022.
- [4] Peter L Bartlett, Philip M Long, Gábor Lugosi, and Alexander Tsigler. Benign overfitting in linear regression. Proceedings of the National Academy of Sciences , 117(48):30063-30070, 2020.
- [5] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. High-dimensional limit theorems for SGD: Effective dynamics and critical scaling. Advances in Neural Information Processing Systems , 35:25349-25362, 2022.

- [6] Guy Blanc, Neha Gupta, Gregory Valiant, and Paul Valiant. Implicit regularization for deep neural networks driven by an Ornstein-Uhlenbeck like process. In Conference on learning theory , pages 483-513. PMLR, 2020.
- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems , 33:1877-1901, 2020.
- [8] Yuan Cao, Zixiang Chen, Misha Belkin, and Quanquan Gu. Benign overfitting in two-layer convolutional neural networks. Advances in Neural Information Processing Systems , 35:2523725250, 2022.
- [9] Zixiang Chen, Yihe Deng, Yue Wu, Quanquan Gu, and Yuanzhi Li. Towards understanding mixture of experts in deep learning. Advances in Neural Information Processing Systems , 2022.
- [10] Zixiang Chen, Junkai Zhang, Yiwen Kou, Xiangning Chen, Cho-Jui Hsieh, and Quanquan Gu. Why does sharpness-aware minimization generalize better than SGD? Advances in neural information processing systems , 2023.
- [11] Muthu Chidambaram, Xiang Wang, Chenwei Wu, and Rong Ge. Provably learning diverse features in multi-view data with midpoint mixup. In International Conference on Machine Learning , pages 5563-5599. PMLR, 2023.
- [12] Lenaic Chizat and Francis Bach. Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss. In Conference on learning theory , pages 1305-1338. PMLR, 2020.
- [13] Alex Damian, Tengyu Ma, and Jason D Lee. Label noise SGD provably prefers flat global minimizers. Advances in Neural Information Processing Systems , 34:27449-27461, 2021.
- [14] Alexandru Damian, Jason Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In Conference on Learning Theory , pages 5413-5452. PMLR, 2022.
- [15] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers) , pages 4171-4186, 2019.
- [16] Luc Devroye, Abbas Mehrabian, and Tommy Reddad. The total variation distance between high-dimensional gaussians with the same mean. arXiv preprint arXiv:1810.08693 , 2018.
- [17] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. International Conference on Learning Representations , 2021.
- [18] Spencer Frei, Niladri S Chatterji, and Peter Bartlett. Benign overfitting without linearity: Neural network classifiers trained by gradient descent for noisy linear data. In Conference on Learning Theory , pages 2668-2703. PMLR, 2022.
- [19] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. When do neural networks outperform kernel methods? Advances in Neural Information Processing Systems , 33:14820-14830, 2020.
- [20] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 580-587, 2014.
- [21] Margalit Glasgow. SGD finds then tunes features in two-layer neural networks with near-optimal sample complexity: A case study in the XOR problem. The Twelfth International Conference on Learning Representations , 2024.
- [22] Andi Han, Wei Huang, Yuan Cao, and Difan Zou. On the feature learning in diffusion models. The Thirteenth International Conference on Learning Representations , 2025.

- [23] Andi Han, Wei Huang, Zhanpeng Zhou, Gang Niu, Wuyang Chen, Junchi Yan, Akiko Takeda, and Taiji Suzuki. On the role of label noise in the feature learning process. In Forty-second International Conference on Machine Learning , 2025.
- [24] Jeff Z HaoChen, Colin Wei, Jason Lee, and Tengyu Ma. Shape matters: Understanding the implicit bias of the noise covariance. In Conference on Learning Theory , pages 2315-2357. PMLR, 2021.
- [25] Wei Huang, Yuan Cao, Haonan Wang, Xin Cao, and Taiji Suzuki. Quantifying the optimization and generalization advantages of graph neural networks over multilayer perceptrons. In The 28th International Conference on Artificial Intelligence and Statistics , 2025.
- [26] Wei Huang, Andi Han, Yongqiang Chen, Yuan Cao, Zhiqiang Xu, and Taiji Suzuki. On the comparison between multi-modal and single-modal contrastive learning. Advances in Neural Information Processing Systems , 37:81549-81605, 2024.
- [27] Wei Huang, Ye Shi, Zhongyi Cai, and Taiji Suzuki. Understanding convergence and generalization in federated learning through feature learning theory. In The Twelfth International Conference on Learning Representations , 2023.
- [28] Jung Eun Huh and Patrick Rebeschini. Generalization bounds for label noise stochastic gradient descent. In International Conference on Artificial Intelligence and Statistics , pages 1360-1368. PMLR, 2024.
- [29] Samy Jelassi and Yuanzhi Li. Towards understanding how momentum improves generalization in deep learning. In International Conference on Machine Learning , pages 9965-10040. PMLR, 2022.
- [30] Samy Jelassi, Michael Sander, and Yuanzhi Li. Vision transformers provably learn spatial structure. Advances in Neural Information Processing Systems , 35:37822-37836, 2022.
- [31] Jiarui Jiang, Wei Huang, Miao Zhang, Taiji Suzuki, and Liqiang Nie. Unveil benign overfitting for transformer in vision: Training dynamics, convergence, and generalization. Advances in Neural Information Processing Systems , 37:135464-135625, 2024.
- [32] Yiwen Kou, Zixiang Chen, Yuan Cao, and Quanquan Gu. How does semi-supervised learing with pseudo-labelers work? a case study. In International Conference on Learning Representations , 2023.
- [33] Yiwen Kou, Zixiang Chen, Yuanzhou Chen, and Quanquan Gu. Benign overfitting in two-layer relu convolutional neural networks. In International Conference on Machine Learning , pages 17615-17659. PMLR, 2023.
- [34] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature , 521(7553):436-444, 2015.
- [35] Bingrui Li, Wei Huang, Andi Han, Zhanpeng Zhou, Taiji Suzuki, Jun Zhu, and Jianfei Chen. On the optimization and generalization of two-layer transformers with sign gradient descent. In The Thirteenth International Conference on Learning Representations , 2025.
- [36] Hongkang Li, Meng Wang, Sijia Liu, and Pin-Yu Chen. A theoretical understanding of shallow vision transformers: Learning, generalization, and sample complexity. The Eleventh International Conference on Learning Representations , 2023.
- [37] Hongkang Li, Meng Wang, Tengfei Ma, Sijia Liu, Zaixi Zhang, and Pin-Yu Chen. What improves the generalization of graph transformers? a theoretical dive into the self-attention and positional encoding. The Forty-First International Conference on Machine Learning , 2024.
- [38] Weizhi Li, Gautam Dasarathy, and Visar Berisha. Regularization via structural label smoothing. In International Conference on Artificial Intelligence and Statistics , pages 1453-1463. PMLR, 2020.
- [39] Zhiyuan Li, Tianhao Wang, and Sanjeev Arora. What happens after SGD reaches zero loss?-a mathematical framework. arXiv preprint arXiv:2110.06914 , 2021.

- [40] Zhu Li, Weijie J Su, and Dino Sejdinovic. Benign overfitting and noisy features. Journal of the American Statistical Association , 118(544):2876-2888, 2023.
- [41] Miao Lu, Beining Wu, Xiaodong Yang, and Difan Zou. Benign oscillation of stochastic gradient descent with large learning rates. The Twelfth International Conference on Learning Representations , 2024.
- [42] Samet Oymak, Ankit Singh Rawat, Mahdi Soltanolkotabi, and Christos Thrampoulidis. On the role of attention in prompt-tuning. In International Conference on Machine Learning , pages 26724-26768. PMLR, 2023.
- [43] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International conference on machine learning , pages 5301-5310. PMLR, 2019.
- [44] Amartya Sanyal, Puneet K Dokania, Varun Kanade, and Philip HS Torr. How benign is benign overfitting? International Conference on Learning Representations , 2021.
- [45] Christopher J Shallue, Jaehoon Lee, Joseph Antognini, Jascha Sohl-Dickstein, Roy Frostig, and George E Dahl. Measuring the effects of data parallelism on neural network training. Journal of Machine Learning Research , 20(112):1-49, 2019.
- [46] Ohad Shamir. The implicit bias of benign overfitting. Journal of Machine Learning Research , 24(113):1-40, 2023.
- [47] Ruoqi Shen, Sébastien Bubeck, and Suriya Gunasekar. Data augmentation as feature manipulation. In International conference on machine learning , pages 19773-19808. PMLR, 2022.
- [48] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature , 529(7587):484-489, 2016.
- [49] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2818-2826, 2016.
- [50] Shokichi Takakura and Taiji Suzuki. Mean-field analysis on two-layer neural networks from a kernel perspective. The Forty-First International Conference on Machine Learning , 2024.
- [51] Alexander Tsigler and Peter L Bartlett. Benign overfitting in ridge regression. Journal of Machine Learning Research , 24(123):1-76, 2023.
- [52] Roman Vershynin. High-Dimensional Probability: An Introduction With Applications in Data Science . Cambridge, UK: Cambridge Univ. Press, 2018.
- [53] Loucas Pillaud Vivien, Julien Reygner, and Nicolas Flammarion. Label noise (stochastic) gradient descent implicitly solves the lasso for quadratic parametrisation. In Conference on Learning Theory , pages 2127-2159. PMLR, 2022.
- [54] Zhichao Wang, Denny Wu, and Zhou Fan. Nonlinear spiked covariance matrices and signal propagation in deep neural networks. In The Thirty Seventh Annual Conference on Learning Theory , pages 4891-4957. PMLR, 2024.
- [55] Zixin Wen and Yuanzhi Li. Toward understanding the feature learning process of self-supervised contrastive learning. In International Conference on Machine Learning , pages 11112-11122. PMLR, 2021.
- [56] Zhiwei Xu, Yutong Wang, Spencer Frei, Gal Vardi, and Wei Hu. Benign overfitting and grokking in reLU networks for XOR cluster data. The Twelfth International Conference on Learning Representations , 2024.

- [57] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM , 64(3):107115, 2021.
- [58] Difan Zou, Yuan Cao, Yuanzhi Li, and Quanquan Gu. The benefits of mixup for feature learning. In International Conference on Machine Learning , pages 43423-43479. PMLR, 2023.
- [59] Difan Zou, Yuan Cao, Yuanzhi Li, and Quanquan Gu. Understanding the generalization of Adam in learning neural networks with proper regularization. The Eleventh International Conference on Learning Representations , 2023.

## Contents

| A   | Additional related Works                             | Additional related Works                             |   16 |
|-----|------------------------------------------------------|------------------------------------------------------|------|
| B   | Preliminary Lemmas                                   | Preliminary Lemmas                                   |   16 |
| C   | Proof of Proposition 4.1                             | Proof of Proposition 4.1                             |   17 |
| D   | Standard GD Fails to Generalize with low SNR         | Standard GD Fails to Generalize with low SNR         |   19 |
|     | D.1                                                  | Proof of Lemma 4.2 . . . . . .                       |   19 |
|     | D.2                                                  | Proof of Lemma 4.3 . . . . . .                       |   21 |
|     | D.3                                                  | Proof of Theorem 3.2 . . . . .                       |   22 |
| E   | label noise GD Successfully Generalizes with Low SNR | label noise GD Successfully Generalizes with Low SNR |   25 |
|     | E.1                                                  | Proof of Lemma 4.4 . . . . . .                       |   25 |
|     | E.2                                                  | Proof of Lemma 4.5 . . . . . .                       |   29 |
|     | E.3                                                  | Proof of Theorem 3.3 . . . . .                       |   33 |
| F   | Experimental Details for Figure 1                    | Experimental Details for Figure 1                    |   35 |
|     | F.1                                                  | Dataset and Noise Injection . .                      |   35 |
|     | F.2                                                  | Model and Training Setup . .                         |   35 |
|     | F.3                                                  | Results Analysis . . . . . . . .                     |   36 |
|     | F.4                                                  | Reproducibility . . . . . . . .                      |   36 |
| G   | Additional Experiments                               | Additional Experiments                               |   36 |
|     | G.1                                                  | Deeper Neural Network . . . .                        |   36 |
|     | G.2                                                  | Real World Dataset . . . . . .                       |   36 |
|     | G.3                                                  | Different Type of Label Noise                        |   38 |
|     | G.4                                                  | Higher Order Polynomial ReLU                         |   39 |

## Appendix

## A Additional related Works

Label Noise SGD. Recent works have empirically shown that label noise stochastic gradient descent (SGD) exhibits favorable generalization properties due to the regularization effect of the injected noise [24, 13]. From a theoretical standpoint, label noise SGD has been primarily explored in the context of linear regression or shallow neural networks, particularly in regression settings [6, 13, 24, 28, 39, 53, 50]; these studies have highlighted the implicit regularization benefits of label noise in SGD. For instance, [50] illustrated the implicit regularization of label noise in mean-field neural networks, while [39, 13] proved that label noise introduces bias towards flat minima. In contrast to these existing literature, our work focuses on the binary classification setting specified by the signal-noise model, providing a quantitative analysis of the training dynamics and the generalization benefits of label noise GD in the low SNR regime.

Signal-Noise Data Models. Recent theoretical works have studied the signal-noise model in various contexts, including ( i ) optimization algorithms , such as Adam [59], momentum [29], sharpnessaware minimization [10], large learning rates [41]; ( ii ) learning paradigms , such as ensembling and knowledge distillation [2], semi-and self-supervised learning [32, 55], Mixup [58, 11], adversarial training [1], and prompt tuning [42]; and ( iii ) neural network structures , such as convolutional neural network [8, 33], vision transformer [30, 36], graph neural network [25, 37]. Our work is in line with [9, 25], with the goal of showing that a simple algorithmic modification (label noise GD) facilitates feature learning in the challenging low SNR regime.

## B Preliminary Lemmas

Lemma B.1 ([8]) . Suppose that δ &gt; 0 and d = Ω(log(4 n/δ ))) . Then with probability 1 -δ ,

<!-- formula-not-decoded -->

̸

for all i, i ′ = i ∈ [ n ] .

Lemma B.2 ([8]) . Suppose that d ≥ Ω(log( mn/δ )) , m = Ω(log(1 /δ )) . Then with probability at least 1 -δ , it satisfies that for all r ∈ [ m ] , j ∈ {± 1 } , i ∈ [ n ] ,

<!-- formula-not-decoded -->

and for all j ∈ {± 1 } , i ∈ [ n ]

<!-- formula-not-decoded -->

Lemma B.3. Let S ( t ) ± = { i : ϵ ( t ) i = ± 1 } and S j = { i : y i = j } . Then ∀ t ≥ 0 , we have following with probability at least 1 -δ ,

<!-- formula-not-decoded -->

2. The size of set follows, ∀ j ∈ {± 1 }

<!-- formula-not-decoded -->

Suppose n ≥ 8 log(8 T ∗ /δ ) p 2 ≥ 8 log(8 T ∗ /δ ) (1 -p ) 2 , we have

<!-- formula-not-decoded -->

Proof of Lemma B.3. By independence, we have E |S ( t ) + | = (1 -p ) n and E |S ( t ) -| = pn . By Hoeffding's inequality, we have for arbitrary τ &gt; 0 ,

<!-- formula-not-decoded -->

Setting τ = √ ( n/ 2) log(4 /δ ) and taking the union bound over [ T ∗ ] gives

<!-- formula-not-decoded -->

which holds with probability at least 1 -δ .

Similarly, by the same argument, we can show the result for |S ( t ) + ∩ S j | and |S ( t ) -∩ S j | .

Suppose n ≥ 8 log(8 T ∗ /δ ) p 2 ≥ 8 log(8 T ∗ /δ ) (1 -p ) 2 , then we have with probability at least 1 -δ , we have |S ( t ) + ∩ S j | ∈ [ (2 -3 p ) n 4 , (2 -p ) n 4 ] , |S ( t ) -∩ S j | ∈ [ pn 4 , 3 pn 4 ] .

Lemma B.4. Let S ( t ) i, ± := { s ≤ t : ϵ ( s ) i = ± 1 } . Then for any i ∈ [ n ] , t &gt; 0 , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma B.4. By independence, we have E |S ( t ) i, + | = (1 -p ) t and E |S ( t ) i, -| = pt . By Hoeffding's inequality, we have for arbitrary τ &gt; 0 ,

<!-- formula-not-decoded -->

Setting τ = √ ( t/ 2) log(4 /δ ) and taking the union bound gives

<!-- formula-not-decoded -->

which holds with probability at least 1 -δ .

Suppose t ≥ 2 log(4 n/δ ) p 2 ≥ 2 log(4 n/δ ) (1 -p ) 2 , then we have with probability at least 1 -δ , we have |S ( t ) i, + | ∈ [ (2 -3 p ) t 2 , (2 -p ) t 2 ] , |S ( t ) i, -| ∈ [ pt 2 , 3 pt 2 ] .

## C Proof of Proposition 4.1

In this section, we provide a proof for Proposition 4.1, which establishes upper bounds for the absolute values of the signal learning and noise memorization coefficients throughout the entire training stage. Additionally, we present some preliminary lemmas that will be used in the proof of Proposition 4.1 as well as in other results in the subsequent sections.

Lemma C.1. Suppose that inequalities (7) and (8) hold for all r ∈ [ m ] , j ∈ {-1 , 1 } , i ∈ [ n ] and t ∈ [0 , T ∗ ] . For any δ &gt; 0 , with probability at least 1 -δ , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma C.1. From the signal-noise decomposition of w ( t ) j,r , we have

̸

<!-- formula-not-decoded -->

where (a) follows from the weight decomposition, and inequality (b) is due to Lemma B.1 and the upper bound of ρ ( t ) j,r,i based on inequalities (7) and (8).

Next, for the projection of the weight difference onto the signal vector, we have:

<!-- formula-not-decoded -->

where the equality holds because ⟨ ξ i , µ ⟩ = 0 for i ∈ [ n ] due to the covariance property of the noise vector distribution.

With Lemma C.1 in place, we are now prepared to prove Proposition 4.1. The general proof strategy follows the approach outlined in [8]. However, we present a complete proof here for the sake of clarity and to provide a unified analysis for both gradient descent and label noise GD.

Proof of Proposition 4.1. The proof uses induction and covers both gradient descent and label noise gradient descent.

At t = 0 , it is straightforward that the results hold for all coefficients, as they are initialized to zero. Now, assume that there exists a time step ˆ T such that for t ∈ [1 , ˆ T ] the following inequalities hold:

<!-- formula-not-decoded -->

To complete the induction, we need to show that the above inequalities hold for t = ˆ T +1 . First, we examine ρ ( ˆ T +1) j,r,i for j = -y i , since ρ ( ˆ T +1) j,r,i = 0 when j = y i by definition. Using Lemma C.1, if ρ ( ˆ T ) j,r,i ≤ -0 . 5 β -8 √ log(4 n 2 /δ ) d nα , we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

where we have used σ ′ ( ⟨ w ( ˆ T ) j,r , ξ i ⟩ ) = 0 . On the other hand, if ρ ( ˆ T ) j,r,i ≥ -0 . 5 β -8 √ log(4 n 2 /δ ) d nα , the update function implies:

<!-- formula-not-decoded -->

where (a) is due to choosing ϵ ( ˆ T ) i = 1 and ⟨ w ( ˆ T ) j,r , ξ i ⟩ &gt; 0 , follows from Lemma B.1, and (c) holds when η ≤ 2 nm 3 σ 2 p d .

Next, consider ρ ( ˆ T +1) j,r,i for j = y i . Let ˆ T 1 to be the last time that ρ ( t ) j,r,i ≤ 0 . 5 α . By propagation, we have:

<!-- formula-not-decoded -->

where (a) holds since ℓ ′ ( ˆ T 1 ) i ≥ -1 and ϵ ( t ) i ≤ 1 for all t ∈ [ ˆ T 1 , ˆ T ] , (b) is by Lemma B.1, Lemma C.1, and -˜ ℓ ′ ( t ) i ≤ exp( -F y i +1) ≤ exp( -4 α 2 +1) . Here we have used that β +16 √ log(4 n 2 /δ ) d nα ≤ 2 α with the condition that d = ˜ Ω( n 2 ) and σ 0 ≤ ˜ O (1) min {∥ µ ∥ -1 2 , σ -1 p d -1 / 2 } . The final inequality (c) holds because η = O ( nm σ 2 p d ) and exp( -4 α 2 +1) α &lt; 1 with α = 4log( T ∗ ) .

Similarly, we can prove that γ ( ˆ T +1) j,r ≤ α using η = O ( nm ∥ µ ∥ 2 2 ) , which completes the induction proof.

## D Standard GD Fails to Generalize with low SNR

## D.1 Proof of Lemma 4.2

In this section, we provide a proof for the result obtained in the first stage of gradient descent training. Several preliminary lemmas are established to facilitate the analysis.

Lemma D.1 (Upper bound on γ ( t ) j,r ) . Under Assumption 3.1, in the first stage, where 0 ≤ t ≤ T 1 = nm log(1 / ( σ 0 σ p √ d )) ησ 2 p d , there exists an upper bound for γ ( t ) j,r , for all j ∈ {-1 , 1 } , r ∈ [ m ] :

<!-- formula-not-decoded -->

Proof of Lemma D.1. By the iterative update rule of signal learning, we have:

<!-- formula-not-decoded -->

where (a) follows from | ℓ ′ ( t ) i | ≤ 1 , (b) is derived using Lemma C.1, and (c) is due to the properties of the squared ReLU activation function.

Define A ( t ) := γ ( t ) j,r + |⟨ w (0) j,r , µ ⟩| . Then, we have:

<!-- formula-not-decoded -->

where we use 1 + x ≤ exp( x ) . This suggests:

<!-- formula-not-decoded -->

Lemma D.2 (Upper bound on ρ ( t ) j,r,i ) . Under Assumption 3.1, in the first stage, where 0 ≤ t ≤ T 1 = nm log(1 / ( σ 0 σ p √ d )) ησ 2 p d , there exists an upper bound for | ρ ( t ) j,r,i | , for all j, r, i :

<!-- formula-not-decoded -->

Proof of Lemma D.2. The proof uses the induction method. By the iterative update rule for noise memorization, we have:

<!-- formula-not-decoded -->

where the inequality (a) is by the upper bound on | ℓ ′ ( t ) i | ≤ 1 ; Inequality (b) is derived using Proposition 4.1, Lemma B.1, and Lemma C.1. Finally, the inequality (c) uses the fact that ρ ( t ) j,r,i &lt; 0 and Lemma B.2.

Taking a telescoping sum over t form 0 to T 1 , we obtain:

<!-- formula-not-decoded -->

where we substituted T 1 = Θ( nm log(1 / ( σ 0 σ p √ d )) ησ 2 p d ) , thereby completing the proof.

Lemma D.3. Let ¯ β = min i ∈ [ n ] max r ∈ [ m ] ⟨ w (0) y i ,r , ξ i ⟩ . Suppose that σ 0 ≥ 160 n √ log(4 n 2 /δ ) d ( σ p √ d ) -1 α . Then it holds that ¯ β ≥ 40 n √ log(4 n 2 /δ ) d α .

Proof of Lemma D.3. The proof follows directly from Lemma B.2. With high probability, we have: β ≥ σ 0 σ p √ d/ 4 . Substituting the condition on σ 0 , we obtain:

<!-- formula-not-decoded -->

Lemma D.4 (Lower bound on ρ ( t ) j,r,i ) . Under Assumption 3.1, in the first stage, where 0 ≤ t ≤ T 1 = nm log(1 / ( σ 0 σ p √ d )) ησ 2 p d , there exists a lower bound for max j,r ρ ( t ) j,r,i , for all i ∈ [ n ] :

<!-- formula-not-decoded -->

Proof of Lemma D.4. By the iterative update rule for noise memorization, we have:

<!-- formula-not-decoded -->

where the inequality (a) is by the lower bound on | ℓ ′ ( t ) i | ≥ C 1 in the first stage; Inequality (b) is by Lemma B.1 and Lemma C.1. Finally, the inequality (c) is by Lemma D.3.

<!-- formula-not-decoded -->

With the above lemmas in place, we are now ready to prove Lemma 4.2.

Proof of Lemma 4.2. Wechoose the end of stage 1 as T 1 = 4 nm ησ 2 p d log(1 / ( σ 0 σ p √ d )) . Then by Lemma D.4, we conclude that max j,r ρ ( T 1 ) j,r,i ≥ 1 , for all i ∈ [ n ] . Besides, by Lemma D.2, we directly obtain the result that

<!-- formula-not-decoded -->

Finally, Lemma D.1 yields

<!-- formula-not-decoded -->

where we have used the condition of low SNR, namely n SNR 2 ≤ 1 / log( σ 0 σ p d ) . By Lemma B.2, we conclude the proof for max j,r γ ( T 1 ) j,r = ˜ O ( σ 0 ∥ µ ∥ 2 ) .

## D.2 Proof of Lemma 4.3

In this section, we provide a complete proof for Lemma 4.3 based on Lemma 4.2 and an iterative analysis of the training dynamics. We introduce several necessary preliminary lemmas that will be used in the proof for t ∈ [ T 1 , T 2 ] with T 2 = η -1 σ -2 p d -1 nm log(1 / ( σ 0 σ p √ d )) + η -1 ϵ -1 m 3 nσ -2 p d -1 .

Lemma D.5 ([8]) . Under the same condition as Theorem 3.2, for all t ∈ [ T 1 , T 2 ] and i ∈ [ n ] , the following properties hold:

<!-- formula-not-decoded -->

With the above lemmas at hand, we are now ready to provide the complete proof for Lemma 4.3.

Proof of Lemma 4.3. We start by showing the convergence of gradient descent. The key idea is to construct a reference weight matrix W ∗ defined as w ∗ j,r = w (0) j,r +2 m log(2 /ϵ ) ∑ n i =1 ∥ ξ i ∥ -2 2 ξ i .

Summing the above inequality from W ( t ) and W ∗ :

<!-- formula-not-decoded -->

where in equation (a), we have applied the homogeneity property of the squared ReLU activation. The inequality (b) is by ⟨∇ f ( W ( t ) , x i ) , W ∗ ⟩ ≥ 2 log(2 /ϵ ) as stated in Lemma D.5, and the inequality (c) is due to the convexity of the logistic function. Finally, the inequality (d) is by Lemma D.5 and the condition on the learning rate.

Taking a summation over the above inequality from T 1 to T 2 , we have

<!-- formula-not-decoded -->

where in the second inequality, we have applied Lemma D.5. Finally, plugging in the T 2 = η -1 ϵ -1 m 3 nσ -2 p d -1 + η -1 σ -2 p d -1 nm log(1 / ( σ 0 σ p √ d )) , we achieve L S ( W ( t ) ) ≤ ϵ .

Next, we provide the lower bound for the noise memorization coefficient ρ ( t ) j,r,i and the upper bound for the signal learning coefficient γ ( t ) j,r in the second stage. For the noise memorization coefficient, using its update equation:

<!-- formula-not-decoded -->

Here, we have used ℓ ′ ( t ) i ≥ 0 and property of the squared ReLU activation. This implies that ρ ( t ) j,r,i never decreases during training. Therefore, we have max j,r ρ ( t ) j,r,i ≥ 1 , for all i ∈ [ n ] and t ∈ [ T 1 , T 2 ] . For the signal learning coefficient, we use the induction method. From Lemma 4.2, we know that max j,r γ ( T 1 ) j,r = ˜ O ( σ 0 ∥ µ ∥ 2 ) ≜ ˆ β . Suppose that there exists T ∈ [ T 1 , T 2 ] such that max j,r γ ( t ) j,r ≤ 2 ˆ β for all t ∈ [ T 1 , T ] . Then we analyze:

<!-- formula-not-decoded -->

where the inequality (a) is due to Lemma C.1, the inequality (b) is by | ℓ ′ i | ≤ ℓ i for i ∈ [ n ] , and the inequality (c) is due to the inequality (9). Finally, the inequity (d) is by the condition that n -1 SNR -2 = ˜ Ω(1) . Similarly, with the induction method, we can show that | ρ ( t ) j,r,i | ≤ ˜ O ( σ 0 σ p √ d ) .

## D.3 Proof of Theorem 3.2

To complete the proof of Theorem 3.2, we provide a proof for the generalization result.

Lemma D.6. Define g ( ξ i ) = 1 m j ∑ j,r σ ( ⟨ w ( t ) j,r , ξ i ⟩ ) . Under Assumption 3.1, there exists a fixed vector v with ∥ v ∥ 2 ≤ 0 . 02 σ p such that

<!-- formula-not-decoded -->

Proof of Lemma D.6. To proceed with the proof, we construct the vector v ≜ λ ∑ i : y i =1 ξ i , where λ = 0 . 01 / √ nd . Then we show that

̸

<!-- formula-not-decoded -->

where the first inequity is by Lemma B.1, the second inequality is by d ≥ ˜ Ω( n 2 ) , and the final equality is by λ = 0 . 01 / √ nd , which confirms that ∥ v ∥ 2 ≤ 0 . 02 σ p .

By the convexity property of the squared ReLU function, we have that

<!-- formula-not-decoded -->

With the above inequalities, we have that almost surely for all ξ i :

<!-- formula-not-decoded -->

On the other hand, using the properties of the squared ReLU function and the triangle inequality, we have:

<!-- formula-not-decoded -->

Next, we compare |⟨ w ( t ) 1 ,r , v ⟩| and |⟨ w ( t ) -1 ,r , v ⟩| with |⟨ w ( t ) 1 ,r , ξ i ⟩| and |⟨ w ( t ) -1 ,r , ξ i ⟩| . We show that

<!-- formula-not-decoded -->

where the first inequality is by Lemma B.2 and Lemma 4.3, and the second inequality is by the condition on σ 0 from Assumption 3.1. Besides,

<!-- formula-not-decoded -->

where the first inequality is by Lemma B.2 and Lemma 4.3; and the second inequality is by the condition on σ 0 from Assumption 3.1.

Finally, by Lemma B.2, Proposition 4.1, and Lemma B.1 it holds that

<!-- formula-not-decoded -->

On the other hand, it is observed that ⟨ w ( t ) 1 ,r -w (0) 1 ,r , ξ i ⟩ ∼ N (0 , σ 2 w ) , where the variance σ w follows

<!-- formula-not-decoded -->

where (a) is by Lemma B.1 and condition on d from Assumption 3.1, (b) is due to Lemma B.1, and (c) is by Lemma 4.3.

By the anti-concentration inequality of Gaussian variance, we have

<!-- formula-not-decoded -->

Then with probability at least 1 -δ , it holds that

<!-- formula-not-decoded -->

where we have used log(1 + x ) ≥ x 1+ x for x &gt; -1 and δ 2 ≤ 1 / 8 .

Together, we conclude that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the final inequality holds by σ 2 0 ≤ ˜ O ( 1 d 5 / 4 ∥ µ ∥ 2 2 ) with δ chosen as d -1 / 4 , thus completing the proof.

Proof of Theorem 3.2. For the population loss, we expand the expression

̸

<!-- formula-not-decoded -->

Recall the weight decomposition:

<!-- formula-not-decoded -->

Then we conclude that:

<!-- formula-not-decoded -->

First, we provide the bound for the signal learning part:

<!-- formula-not-decoded -->

where the first and second inequalities follow from the properties of the squared ReLU function, and the last inequality is by Lemma B.2 and Lemma 4.3.

Denote that g ( ξ i ) = 1 m j ∑ j,r σ ( ⟨ w ( t ) j,r , ξ i ⟩ ) . It follows that:

<!-- formula-not-decoded -->

Define the set A = { ξ i : | g ( ξ i ) | ≥ ˜ Ω( σ 2 0 ∥ µ ∥ 2 2 ) } . By Lemma D.6, we have:

<!-- formula-not-decoded -->

Thus, there must exist at least one of ξ i , ξ i + v , -ξ i and -ξ i + v that belongs to A and the probability is larger than 0.25. Furthermore, we have:

<!-- formula-not-decoded -->

where the first inequality is by Proposition 2.1 in [16] and the second inequality is by ∥ v ∥ 2 ≤ 0 . 01 σ p according to Lemma D.6. Combined with that P ( A ) = P ( -A ) , we finally achieve that P ( A ) ≥ 0 . 24 , corresponding to the second bullet result. Combined with Lemma 4.3, which establishes the first bullet point, this completes the proof of 3.2

## E label noise GD Successfully Generalizes with Low SNR

## E.1 Proof of Lemma 4.4

Lemma E.1 (Lower bound on γ ( t ) j,r ) . Under Assumption 3.1, during the first stage, where 0 ≤ t ≤ T 1 = nm log(1 / ( σ 0 σ p √ d )) ησ 2 p d , there exists an lower bound for γ ( t ) j,r , for all j :

<!-- formula-not-decoded -->

where C 0 is the lower bound on | ˜ ℓ ′ ( t ) | ≥ C 0 is the first stage.

Proof of Lemma E.1. If ⟨ w ( t ) j,r , µ ⟩ ≥ 0 , then

<!-- formula-not-decoded -->

Note that we have defined S ( t ) ± = { i : ϵ ( t ) i = ± 1 } and S j = { i : y i = j } in Lemma B.3.

On the other hand, when ⟨ w ( t ) j,r , µ ⟩ &lt; 0 ,

<!-- formula-not-decoded -->

By Lemma B.3, we have

<!-- formula-not-decoded -->

These hold with probability at least 1 -δ . This suggests that when p &lt; C 0 / 6 , n ≥ 72 C -2 0 log(8 T ∗ /δ ) , we have:

<!-- formula-not-decoded -->

Hence, we have:

<!-- formula-not-decoded -->

When j = 1 , due to the increase of γ ( t ) j,r , we have

<!-- formula-not-decoded -->

Let B ( t ) j = max r ∈ [ m ] { γ ( t ) j,r + j ⟨ w (0) j,r , µ ⟩} , then we have

<!-- formula-not-decoded -->

where we use the fact that 1 + x ≥ exp( x/ 2) for x ≤ 2 .

Similarly when j = -1 , we have γ ( t +1) -1 ,r ≥ γ ( t ) -1 ,r -C 0 η ∥ µ ∥ 2 2 4 m ( ⟨ w (0) -1 ,r , µ ⟩ -γ ( t ) -1 ,r ) and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, we obtain B ( t ) j ≥ exp ( C 0 η ∥ µ ∥ 2 2 8 m t ) σ 0 ∥ µ ∥ 2 2 , ∀ j ∈ {± 1 } .

Lemma E.2. Let ¯ β = min i ∈ [ n ] max r ∈ [ m ] ⟨ w (0) y i ,r , ξ i ⟩ . Suppose that σ 0 ≥ 160 n √ log(4 n 2 /δ ) d ( σ p √ d ) -1 αd 1 / 4 . Then we have that ¯ β/d 1 / 4 ≥ 40 n √ log(4 n 2 /δ ) d α .

Proof of Lemma E.2. The proof follows from Lemma B.2. It is known that, with high probability, we have β ≥ σ 0 σ p √ d/ 4 . By substituting the condition for σ 0 , we obtain

<!-- formula-not-decoded -->

Lemma E.3 (Lower bound on ρ ( t ) j,r,i ) . Let ¯ β = min i ∈ [ n ] max r ∈ [ m ] ⟨ w (0) y i ,r , ξ i ⟩ and A ( t ) y i ,r,i := ρ t j,r,i + ⟨ w (0) j,r , ξ i ⟩ -0 . 4 ¯ β/d 1 / 4 . Under Assumption 3.1, if ⟨ w (0) j,r , ξ i ⟩ ≥ ¯ β , then at time step T 1 = nm log(1 / ( σ 0 σ p √ d )) ησ 2 p d , with high probability, it holds that

<!-- formula-not-decoded -->

Proof of Lemma E.3. First, consider y i = j as the case of ρ ( t ) j,r,i . By Lemma C.1 and Lemma D.3, when y i = j ,

<!-- formula-not-decoded -->

From the update of ρ ( t ) j,r,i , when ϵ ( t ) i = 1 and ⟨ w ( t ) j,r , ξ i ⟩ &gt; 0 ,

<!-- formula-not-decoded -->

On the other hand, when ϵ ( t ) i = -1 and ⟨ w ( t ) j,r , ξ i ⟩ &gt; 0 ,

<!-- formula-not-decoded -->

For simplification of notations, denote ζ = 0 . 8 ¯ β/d 1 / 4 . Let A ( t ) y i ,r,i := ρ t j,r,i + ⟨ w (0) j,r , ξ i ⟩ -0 . 4 ¯ β/d 1 / 4 . Then when ϵ ( t ) i = 1 , we have

<!-- formula-not-decoded -->

and when ϵ ( t ) i = -1 , we have

<!-- formula-not-decoded -->

Here we prove when ⟨ w (0) j,r , ξ i ⟩ ≥ ¯ β , A ( t ) y i ,r,i &gt; ζ . The proof is by the induction method.

First it is clear that A (0) y i ,r,i = ⟨ w (0) j,r , ξ i ⟩ -0 . 5 ζ &gt; ζ because d ≫ Θ(1) . Then we consider when t ≤ 2 log(4 n/δ ) p 2 (where the condition for Lemma B.4 does not hold). In this case, |S ( t ) + | ≥ (1 -p ) t -√ t 2 log( 4 n δ ) , |S ( t ) -| ≤ pt + √ t 2 log( 4 n δ ) . In addition, the worst case lower bound is achieved by the case where all the S ( t ) -events happen at the first few iterations. This gives

<!-- formula-not-decoded -->

where the last inequality follows from the fact that d ≫ Θ(1) . To see this, suppose there exists a t ≤ 2 log(4 n/δ ) p 2 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

while t ≤ 2 log(4 /δ ) p 2 raises a contradiction by the choice of d . This proves for all t ≤ 2 log(4 /δ ) p 2 , we have (1 -3 ησ 2 p d nm ) pt + √ t 2 log( 4 n δ ) A (0) y i ,r,i ≥ 2 ζ and thus A ( t ) y i ,r,i ≥ ζ .

Then we consider the case when t ≥ 2 log(4 /δ ) p 2 where the condition for Lemma B.4 holds. Now suppose for all s ≤ t -1 , we have A ( s ) y i ,r,i ≥ ζ , which clearly holds for t = 2 log(4 n/δ ) p 2 . For all s ≤ t -1 , we have A ( s ) y i ,r,i ≥ (1 -3 ησ 2 p dζ nm ) A ( s ) y i ,r,i when ϵ ( s ) i = -1 . This leads to the following lower bound for A ( t ) y i ,r,i as

<!-- formula-not-decoded -->

where the second last inequality follows from the choice of

<!-- formula-not-decoded -->

We can verify that p = C 0 24 satisfies the above inequality. This concludes the proof that, for all t , we have A ( t ) y i ,r,i ≥ ζ and thus for all t . Finally, we conclude that

<!-- formula-not-decoded -->

then we have

With the above lemmas at hand, we are ready to prove Lemma 4.4:

Proof of Lemma 4.4. By Lemma E.3, at t = T 1 , taking the maximum over r yields

<!-- formula-not-decoded -->

where the first inequality is by max r ⟨ w (0) j,r , ξ i ⟩ ≥ ¯ β and 0 . 4 ¯ βd -1 / 4 ≤ 0 . 4 ¯ β . In the last inequality, we use (1 + z ) ≥ exp( z/ 2) for z ≤ 2 .

Then we see max r A ( t ) y i ,r,i ≥ 1 in at least T 1 = log(20 / ( σ 0 σ p √ d ))4 nm ηC 0 σ 2 p d and because max j,r ρ T 1 j,r,i ≥ A T 1 y i ,r,i -max j,r |⟨ w (0) j,r , ξ i ⟩| +0 . 4 ¯ β ≥ 1 .

Besides, by Lemma D.2, we directly obtain the result that

<!-- formula-not-decoded -->

Furthermore, Lemma D.1 yields

<!-- formula-not-decoded -->

where we have used the condition of low SNR, namely n SNR 2 ≤ 1 / log(20 / ( σ 0 σ p √ d )) . By Lemma B.2, we conclude the proof for max j,r γ ( T 1 ) j,r = ˜ O ( σ 0 ∥ µ ∥ 2 ) .

Lastly, according to Lemma E.1, at the end of stage1, we have the lower bound on signal learning coefficient

<!-- formula-not-decoded -->

## E.2 Proof of Lemma 4.5

The key idea is to show ρ ( t ) j,r,i oscillates during the second stage, where the growth tends to offset the drop over a given time frame. This would suggest the f ( W ( t ) , x ) is both upper and lower bounded by a constant, which is crucial to ensuring that γ ( t ) j,r increases exponentially during the second stage.

Without loss of generality, for each i with ⟨ w ( t ) j,r,i , ξ i ⟩ &gt; 0 and j = y i = 1 , the evolution of ρ t +1 j,r,i is written as

<!-- formula-not-decoded -->

where we denote f ( t ) i = f ( W ( t ) , x i ) . Note that f ( t ) i ≈ 1 m ∑ m r =1 ( ρ ( t ) +1 ,r,i ) 2 when γ ( t ) j,r ≪ 1 .

To simplify the notation, we define that ι ( t ) i ≜ 1 m ∑ m r =1 ( ρ ( t ) +1 ,r,i ) 2 . Then the dynamics can be approximated to

<!-- formula-not-decoded -->

Lemma E.4 (Restatement of Lemma 4.5) . Under the same condition as Theorem 3.3, during t ∈ [ T 1 , T 2 ] with T 2 = T 1 +log(6 / ( σ 0 ∥ µ ∥ 2 ))4 m (1 + exp( c 2 )) η -1 ∥ µ ∥ -2 2 , there exist a sufficient large positive constant C ι and a constant ι ∗ i depending on sample index i such that the following results hold with high probability at least 1 -1 /d :

<!-- formula-not-decoded -->

- γ ( t ) j,r ≤ 0 . 1 for all j ∈ {-1 , 1 } and r ∈ [ m ]
- 1 2 m ∑ m r =1 ( ρ ( t ) y i ,r,i ) 2 ≤ f ( t ) i ≤ 2 m ∑ m r =1 ( ρ ( t ) y i ,r,i ) 2

<!-- formula-not-decoded -->

Proof of Lemma E.4. The proof is based on the method of induction. Without loss of generality, we consider all i with y i = 1 . We first check that at time step t = T 1 , by Lemma 4.4, there exists a constant C such that

<!-- formula-not-decoded -->

Besides, by Lemma 3.3, it is straightforward to check that γ ( T 1 ) j,r ≤ 1 for all j ∈ {-1 , 1 } and r ∈ [ m ] , and max j γ ( T 1 ) j,r ≥ 0 . Next, we can show the following result at time t = T 1 :

̸

<!-- formula-not-decoded -->

̸

where the first inequality is by Lemma 4.5, Proposition 4.1, and Lemma B.1, The second inequality follows from the condition on σ 0 and d in Assumption 3.1. Similarly, we have

<!-- formula-not-decoded -->

Next, we assume that all the results hold for T 1 &lt; t ≤ T . By the induction hypothesis, we can bound c 1 ≤ f ( T ) i ≤ c 2 for all i ∈ [ n ] . Then we can show that γ ( T +1) j,r continues to exhibit exponential

growth:

<!-- formula-not-decoded -->

where the last inequality is by 3 2 p ≤ 1 2 1 1+exp( c 2 ) . Next, define B ( t ) = max r ∈ [ m ] ( γ ( t ) j,r + |⟨ w (0) j,r , µ ⟩| ) , we have:

<!-- formula-not-decoded -->

At the same time, there exists an upper bound on the signal learning:

<!-- formula-not-decoded -->

where we used the condition that T &lt; T 2 .

To show that ι ( T +1) i remains within a constant range, we define M ( t ) i ≜ ( ι ( t ) i -ι ∗ i ) 2 where ι ∗ i is a sufficiently large constant depending on i . Using the relation 1 2 m ∑ m r =1 ( ρ ( T ) y i ,r,i ) 2 ≤ f ( T ) i ≤ 2 m ∑ m r =1 ( ρ ( T ) y i ,r,i ) 2 we have:

<!-- formula-not-decoded -->

At the same time,

<!-- formula-not-decoded -->

Then we show that

<!-- formula-not-decoded -->

Subtracting M ( T ) i yields

<!-- formula-not-decoded -->

where the final inequality is by ι ( T ) i ≤ 4 ι ∗ and p &lt; 1 / (1+2exp(( ι ( T ) i ) 2 )) and condition the learning rate from Assumption 3.1, which confirms that { M ( t ) i } t ∈ [ T 1 ,T ] is a super martingale. By one-sided Azuma inequality, with probability at least 1 -δ , for any τ &gt; 0 , it holds that

<!-- formula-not-decoded -->

where,

<!-- formula-not-decoded -->

Taking the upper bound of c k ≤ ηC 2 yields

<!-- formula-not-decoded -->

where we define C 2 0 ≜ ( ι ( T ) i -ι ∗ i ) 2 &gt; 0 . Therefore, we conclude with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where the last inequality is by η ≤ ˜ O ( σ -2 p d -1 ) and T &lt; T 2 . Finally, we check that

<!-- formula-not-decoded -->

̸

̸

where the first inequality is by Lemma 4.4 and the induction claim, and the second inequality is by condition on d from Assumption 3.1. Similarly, by the same argument, we conclude that:

<!-- formula-not-decoded -->

Let T 2 = T 1 +log(6 / ( σ 0 ∥ µ ∥ 2 ))4 m (1 + exp( c 2 )) η -1 ∥ µ ∥ -2 2 , then by lemma 4.4 we can show that

<!-- formula-not-decoded -->

## E.3 Proof of Theorem 3.3

Proof of Theorem 3.3. For the population loss, we expand the expression as follows:

̸

<!-- formula-not-decoded -->

Recall the weight decomposing

<!-- formula-not-decoded -->

From this, we obtain:

<!-- formula-not-decoded -->

By Lemma 4.5, we conclude that

<!-- formula-not-decoded -->

Therefore, it holds that

<!-- formula-not-decoded -->

where the last inequity is by Lemma 4.5.

Next, we provide the bound for the noise memorization part. Define that g ( ξ ) = ∑ m r =1 σ ( ⟨ w ( t ) -y,r , ξ ⟩ ) . By Theorem 5.2.2 in [52], for any τ &gt; 0 , it holds

<!-- formula-not-decoded -->

where c is a constant and ∥ g ∥ Lip is the Lipschitz norm of function g ( ξ ) , which can be calculated as follows:

<!-- formula-not-decoded -->

where the first inequality is by the triangle inequality, the second inequality follows from the the convexity of the activation function, the third inequality is by the Cauchy-Schwarz inequality, and the last inequality follows from B.1. Therefore we conclude that

<!-- formula-not-decoded -->

Furthermore, given that ⟨ w ( t ) -y,r , ξ ⟩ ∼ N (0 , σ 2 p ∥ w ( t ) -y,r ∥ 2 2 ) we have:

<!-- formula-not-decoded -->

To obtain the the upper bound of g ( ξ ) , we show that:

<!-- formula-not-decoded -->

̸

where the first inequality is by Lemma B.1, and the second inequality is by the condition on d in Assumption 3.1. With the results above, we conclude that

̸

<!-- formula-not-decoded -->

which corresponds to the second bullet point of Theorem 3.3. Combined with Lemma 4.5, which establishes the first bullet point, this completes the proof of Theorem 3.3.

## F Experimental Details for Figure 1

In this section, we provide a detailed description of the experimental setup used to generate the results shown in Figure 1, which compares the performance of Label Noise GD and Standard GD on the CIFAR-10 dataset under varying SNR conditions.

## F.1 Dataset and Noise Injection

We used the CIFAR-10 dataset, selecting 1,000 images in total, with 100 images per class, to perform both Standard GD and Label Noise GD training. The random seed used for selecting training samples was fixed to ensure a fair comparison across different hyperparameters.

To simulate varying SNR conditions, inspired by [19], we introduced noise to the high-frequency Fourier components of the images using the following procedure:

- Each image was transformed into the frequency domain using a 2D Fourier transform.
- Gaussian noise was added to the high-frequency components, excluding the low-frequency region near the center of the Fourier spectrum. The intensity of the noise was controlled by a noise level parameter, where higher values correspond to noisier data and lower SNR.
- Finally, the image was transformed back into the spatial domain using an inverse Fourier transform.

The noise level was adjusted to control the SNR factor, which is represented on the x-axis in Figure 1.

## F.2 Model and Training Setup

The experiments were conducted using a VGG-16 model trained from scratch on the CIFAR-10 dataset. The final fully connected layer of the model was modified to output predictions for the 10 classes in CIFAR-10. Both Standard GD and Label Noise GD were trained using cross-entropy loss and gradient descent (GD) with a learning rate of 0.05. Training was performed with a full-batch setup over 5,000 epochs. For Label Noise GD, labels were flipped randomly with a probability of 20% at each iteration to simulate label noise.

## F.3 Results Analysis

The results shown in Figure 1 demonstrate that Label Noise GD consistently achieves higher test accuracy than Standard GD across all SNR levels. The performance gap is most evident under low SNR conditions, where Standard GD suffers significant accuracy degradation due to noise memorization, while Label Noise GD effectively suppresses noise and promotes robust feature learning.

## F.4 Reproducibility

To ensure reproducibility, all experiments were implemented in PyTorch. The codebase, including dataset preprocessing, model training, and evaluation, is provided in the supplementary material.

## G Additional Experiments

In this section, we provide additional experiments to further support our theoretical findings.

## G.1 Deeper Neural Network

Figure 4: Performance of a 3-layer ReLU neural network: The ratio of noise memorization to signal learning, along with training loss and test accuracy, for standard GD and label noise GD.

<!-- image -->

We have conducted additional experiments using a 3-layer neural network with ReLU activation. The network is defined as f ( W , x ) = F +1 ( W +1 , W , x ) -F -1 ( W -1 , W , x ) , where

<!-- formula-not-decoded -->

in which σ ( · ) is the ReLU activation, W ∈ R d × m denotes the weight in the first layer, and W ± 1 ∈ R m × m are weights in the second layer. The last layer is fixed.

Specifically, we train the first two layers. The number of training samples is n = 200 , and the number of test samples is n test = 2000 . The input dimension was set to d = 2000 . We set the width to m = 20 , the learning rate to η = 0 . 5 , and the noise flip rate to p = 0 . 1 . The data model follows our theoretical setting, where µ = [1 , 0 , 0 , · · · , 0] and the noise strength is σ p = 1 . The experimental results, shown in Figure 4, are consistent with our original findings: compared to standard gradient descent, label noise GD boosts signal learning (as shown in the first plot) and achieves better generalization (as shown in the last plot).

## G.2 Real World Dataset

We conducted an experiment using the MNIST dataset, in which Gaussian noise was added to the borders of the images while retaining the digits in the middle. The noise level was set to σ p = 5 . Moreover, the original pixel values of the digits ranged from 0 to 255, and we chose a normalization factor of 80. In this setup, the added noise formed a 'noise patch" and the digits formed a 'signal patch". We focused on the digits '0' and '1', using n = 100 samples for training and 200 samples for testing. The learning rate was set to η = 0 . 001 , and the width was set to m = 20 , with a label noise level of p = 0 . 15 . The results, shown in Figure 5, were consistent with our theoretical conclusions, reinforcing the insights derived from our analysis.

Figure 5: Performance on the modified MNIST dataset: The ratio of noise memorization to signal learning, along with training loss and test accuracy, for standard GD and label noise GD.

<!-- image -->

Figure 6: Test accuracy heatmap of standard GD (left) and Label Noise GD (right) after training on modified MNIST dataset.

<!-- image -->

To assess the sensitivity of the methods to the choice of noise parameters and signal normalization, we conducted additional experiments on a modified MNIST dataset. The signal normalization values were varied from 60 to 140, while the noise levels ranged from 4 to 8. For each combination of noise level and signal normalization, we trained the neural network for 200,000 steps with a learning rate η = 0 . 001 , using either standard gradient descent (GD) or label noise GD.

The resulting test errors are visualized in Figure 6. Notably, label noise GD (right) consistently achieves higher test accuracy than standard GD (left) across all configurations. This demonstrates the robustness of label noise GD to variations in noise and signal normalization parameters.

The motivation behind using MNIST was its clearer signal, which allows us to more directly observe the effects of label noise without other confounding factors. However, we also conducted experiments on a subset of CIFAR-10, using two classes: airplane and automobile . Gaussian noise was added to a portion of the images, following a similar setup to MNIST. For these experiments, we set q = 2 , the number of neurons m = 20 , the learning rate η = 0 . 001 , the signal norm signal\_norm = 64 , the noise level noise\_level = 5 , the number of samples n = 100 , the label noise probability p = 0 . 15 , and the input dimension d = 6144 .

Figure 7: Performance on the modified CIFAR-10 dataset: The ratio of noise memorization to signal learning, along with training loss and test accuracy, for standard GD and label noise GD.

<!-- image -->

The results shown in Figure 7 indicate that label noise GD continues to provide benefits in terms of generalization compared to standard GD. We believe these extended experiments help establish a broader applicability of our findings to more complex benchmarks.

## G.3 Different Type of Label Noise

To validate the robustness of label noise GD under different noise forms, we varied p across different values. For example, we show the results for p = 0 . 3 in Figure 8 and p = 0 . 4 in Figure 9. The results consistently indicate that label noise helps reduce overfitting and boost generalization, especially in low SNR settings.

In addition, we extended our empirical analysis to include Gaussian noise and uniform distribution noise added to the labels. For Gaussian noise, we used two examples, namely ϵ ( t ) i ∼ N (1 , 1) and ϵ ( t ) i ∼ N (1 , 1) , with the results shown in Figures 10 and 11, respectively. Furthermore, for the uniform distribution, we simulated the noise with ϵ ( t ) i ∼ unif[ -1 , 2] and ϵ ( t ) i ∼ unif[ -2 , 3] . The results are shown in Figures 12 and 13, respectively.

Our results indicate that label noise GD still performs effectively, achieving better generalization compared to standard GD, providing further evidence of the robustness of label noise GD under different noise forms.

<!-- image -->

Figure 8: Performance with flip noise p = 0 . 3 : The ratio of noise memorization to signal learning, training loss, and test accuracy of standard GD and label noise GD.

Figure 9: Performance with flip noise p = 0 . 4 : The ratio of noise memorization to signal learning, training loss, and test accuracy of standard GD and label noise GD.

<!-- image -->

Figure 10: Performance with Gaussian noise N (1 , 1) : The ratio of noise memorization to signal learning, training loss, and test accuracy of standard GD and label noise GD.

<!-- image -->

<!-- image -->

Figure 11: Performance with Gaussian noise N (0 . 6 , 1) : The ratio of noise memorization to signal learning, training loss, and test accuracy of standard GD and label noise GD.

Figure 12: Performance with uniform distribution noise unif[ -1 , 2] : The ratio of noise memorization to signal learning, training loss, and test accuracy of standard GD and label noise GD.

<!-- image -->

Figure 13: Performance with uniform distribution noise unif[ -2 , 3] : The ratio of noise memorization to signal learning, training loss, and test accuracy of standard GD and label noise GD.

<!-- image -->

## G.4 Higher Order Polynomial ReLU

In this work, we set the activation function as squared ReLU. This choice makes q = 2 a particularly interesting and challenging case to analyze, as it allows us to study the interaction between signal and noise in a setting that closely resembles practical two-layer ReLU networks.

For higher values of q , we also conducted experiments with q = 3 and q = 4 . For q = 3 , we set the learning rate η = 0 . 5 , the number of neurons m = 20 , the number of samples n = 200 , the signal mean µ = [2 , 0 , 0 , · · · , 0] , and the noise strength σ p = 0 . 5 . The results are shown in Figure 14. For q = 4 , the parameters were set as η = 0 . 1 , m = 20 , n = 50 , µ = [5 , 0 , 0 , · · · , 0] , and σ p = 0 . 5 . The results are shown in Figure 15.

Figure 14: Performance with q = 3 for polynomial ReLU: The ratio of noise memorization to signal learning, training loss, and test accuracy of standard GD and label noise GD.

<!-- image -->

Figure 15: Performance with q = 4 for polynomial ReLU: The ratio of noise memorization to signal learning, training loss, and test accuracy of standard GD and label noise GD.

<!-- image -->

In all these cases, the experimental results consistently show that using a higher polynomial ReLU activation helps label noise GD suppress noise memorization while enhancing signal learning. This ultimately leads to improved test accuracy compared to standard GD.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims in the abstract and introduction precisely match the contributions made in the paper, which include both the theoretical analysis and empirical validation of label noise GD for improving generalization under low SNR.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations are discussed in the "Conclusion and limitation" section, where we clearly state the current theoretical analysis is limited to specific architectures and activation functions, and outline directions for future work.

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

Justification: All assumptions for the theoretical results are clearly stated (e.g., Assumption 3.1), and complete proofs are provided in the appendix, with proof sketches included in the main text (Sections 3 and 4, Appendix B-E).

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

Justification: All necessary experimental details, including dataset construction, noise injection, model architecture, training protocol, and evaluation metrics, are provided in Section 5 and Appendix F and G.

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

Justification: The code necessary to reproduce the main experimental results is included in the supplementary material.

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

Justification: Section 5 and Appendix F and G provide all relevant training and test details, including data splits, hyperparameters, optimizer choices, and noise settings.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The primary findings were robust and stable across all tested settings, so statistical significance analysis was not included. If requested, we can provide additional runs and error bar analysis during the rebuttal period.

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

Justification: Section 5 and Appendix F specify the compute environment (e.g., GPU type) and approximate runtime for main experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research fully conforms to the NeurIPS Code of Ethics. No human or sensitive data is used.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [Yes]

Justification: The broader impact is discussed in the "Conclusion and limitation" section, where we state there are no immediate negative societal impacts, and the work aims to advance theoretical understanding.

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

Justification: The paper does not release any high-risk data or models.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All datasets and code used (e.g., CIFAR-10, VGG-16) are cited in the references and used in accordance with their licenses.

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

Justification: No new assets are introduced in this paper.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs are not used as an important, original, or non-standard component of the core methods in this research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.