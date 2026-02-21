## SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training

## Yehonathan Refael

Faculty of Engineering

Tel Aviv University yehonathan@tau.ac.il

## Guy Smorodinsky

Department of Computer science Ben Gurion University smorodin@post.bgu.ac.il

## Ofir Lindenbaum

Faculty of Engineering Bar-Ilan University

ofir.lindenbaum@biu.ac.il

## Abstract

Low-rank gradient-based optimization methods have significantly improved memory efficiency during the training of large language models (LLMs), enabling operations within constrained hardware without sacrificing performance. However, these methods primarily emphasize memory savings, often overlooking potential acceleration in convergence due to their reliance on standard isotropic steepest descent techniques, which can perform suboptimally in the highly anisotropic landscapes typical of deep networks, particularly LLMs. In this paper, we propose SUMO (Subspace-Aware Moment-Orthogonalization), an optimizer that employs exact singular value decomposition (SVD) for moment orthogonalization within a dynamically adapted low-dimensional subspace, enabling norm-inducing steepest descent optimization steps. By explicitly aligning optimization steps with the spectral characteristics of the loss landscape, SUMO effectively mitigates approximation errors associated with commonly used methods, such as the Newton-Schulz orthogonalization approximation. We theoretically establish an upper bound on these approximation errors, proving their dependence on the condition numbers of moments, conditions we analytically demonstrate are encountered during LLM training. Furthermore, we both theoretically and empirically illustrate that exact orthogonalization via SVD substantially improves convergence rates while reducing overall complexity. Empirical evaluations confirm that SUMO accelerates convergence, enhances stability, improves performance, and reduces memory requirements by up to 20% compared to state-of-the-art methods. Code: https://github.com/guy120494/SUMO .

## 1 Introduction

Low-rank gradient-based optimization methods have become powerful tools for reducing memory consumption during the pre-training and fine-tuning of large language models (LLMs), often without sacrificing performance, and sometimes even improving it. For instance, while pre-training Llama 7B typically requires around 58GB of memory, far exceeding the 24GB available on consumer GPUs like RTX 4090, recent advances, such as those discussed in [1-3], have demonstrated that Llama 7B can now be trained from scratch on a single 24GB GPU without the need for costly memory offloading. The theoretical analysis in [1] attributes this efficiency to the inherent low-rank structure of the gradients, which enables optimization in a significantly reduced latent space. Furthermore, [2] found

## Tom Tirer

Faculty of Engineering Bar-Ilan University

tirer.tom@biu.ac.il

a consistent decrease in gradient rank throughout training, suggesting that low-rank optimization not only reduces memory usage but also converges toward increasingly compact subspaces.

However, despite these advancements, existing methods primarily focus on memory savings and often overlook the potential to accelerate convergence. Current approaches typically rely on standard steepest descent methods and assume isotropic geometry, which can hinder efficiency in ill-conditioned settings. This observation motivates our primary objective: to develop a subspace-aware optimizer that leverages low-rank structure while adapting to the geometry of the loss landscape. By reevaluating the choice of norm and its influence on gradient descent dynamics, we aim to design an algorithm that improves generalization, accelerates convergence, while preserving the memory advantages of low-rank methods.

Classical gradient descent, including SGD [4], performs steepest descent under the Euclidean norm, which reflects isotropic curvature. However, deep networks exhibit highly anisotropic loss landscapes, making this assumption suboptimal. Recent work shows that adaptive optimizers, such as Shampoo [5], SOAP [6], and Muon [7], can be interpreted as steepest descent under non-Euclidean norms tailored to the network architecture and data structure. As shown in [8], these methods implicitly adapt to spectral or operator norms, which better capture local curvature and improve convergence. This motivates the design of subspace-aware optimizers that exploit both low-rank structure and appropriate geometry to accelerate training.

To formalize the role of geometry in optimization, consider a neural network with a differentiable loss function L : W → R defined on a weight space W = R n . The local behavior around a point w can be approximated by the Taylor expansion, L ( w + ∆ w ) ≈ L ( w ) + g ⊤ ∆ w + λ 2 ∥ ∆ w ∥ 2 , where g = ∇ w L ( w ) , λ &gt; 0 captures the sharpness or curvature of the loss surface and ∥ · ∥ is a chosen norm reflecting the geometry of the optimization landscape. Minimizing this approximation corresponds precisely to performing steepest descent under the given norm constraint. According to [9], the solution to this minimization explicitly takes the form,

<!-- formula-not-decoded -->

where ∥· ∥ ∗ denotes the dual norm of ∥· ∥ . Adaptive optimizers differ primarily in their norm choices. Adam utilizes a dynamic Max-of-Max norm constraint. Recent optimizers consider matrix norms while applying steepest descent at the layer level. Muon imposes a fixed Schattenp norm constraint for large p , effectively using the spectral norm on weight matrices [10, 7]. Shampoo [5] dynamically learns the optimal approximate Schattenp norm for steepest descent, with its variants like SOAP [11] apply momentum to efficiently navigate the space of possible norms. Muon, by contrast, operates within a relatively fixed but large Schattenp norm, balancing the dynamic adaptability of Shampoo with static spectral-norm constraints. Since neural network weights locally act as linear operators on Euclidean spaces, the induced operator (spectral) norm provides a natural constraint that aligns with the curvature of the loss surface. This perspective motivates gradient orthogonalization, which ensures that optimization updates respect the spectral norm, thereby controlling perturbation magnitude and enhancing optimization stability and efficiency [9].

While norm-induced optimization methods offer a principled way to align updates with the geometry of the loss landscape, their practical deployment often incurs substantial computational overhead. For instance, Shampoo requires computing matrix inverses or root operations at every optimization step, which can be computationally expensive for large-scale neural networks. Similarly, Muon's first-order moments-orthogonalization, though effective, involves a costly approximation to spectral decompositions, computed by applying five iterations of Newton-Schulz [12] (referred to as NewtonSchulz5). Therefore, there is an inherent trade-off between the theoretical optimality provided by these norm-induced optimization approaches and their practical computational demands.

To bridge the gap between the geometric advantages of norm-induced methods and their computational costs, we first analyze the limitations of existing approximations. We derive an upper bound on the error introduced by the Newton-Schulz orthogonalization, demonstrating that this error increases with the condition number of the moment matrix. This finding explains the increasing instability of the Newton-Schulz5 method in ill-conditioned scenarios, which we subsequently demonstrate to occur in the first-order moment matrices during the training of large language models (LLMs). Building on this analysis, we establish a convergence rate for Muon optimization and compare it to an alternative method that replaces the Newton-Schulz approach with exact Singular Value Decomposition (SVD).

Remarkably, we find that the SVD-based approach converges faster, with improvements directly proportional to the accumulated errors from the moments' orthogonalization by the Newton-Schulz5 method. Motivated by the empirical observation that gradients in LLMs often exhibit a low-rank structure, especially during early training, we propose a subspace-aware optimization scheme. This scheme performs exact SVD-based moment orthogonalization within a low-dimensional adaptive subspace. This approach benefits from the relatively low computational cost of SVD calculations for low-rank input matrices and enhances convergence stability. Also, our approach attains an even greater reduction in memory usage than all previous low-rank training methods by relying solely on the first-order moment, as detailed below in Table 1. We support our method with a theoretical convergence guarantee and validate its empirical benefits through experiments, demonstrating faster training and better model performance compared to existing methods.

Table 1: Comparison of properties between SUMO, GaLore, Adam, Shampoo, and SOAP. Assume W ∈ R m × n with m ≥ n , a constant projection rank r and a subspace update rate K .

|                      | SUMO                | Adam     | Shampoo         | SOAP               | GaLore              |
|----------------------|---------------------|----------|-----------------|--------------------|---------------------|
| Computation          | O ( mnr + mn 2 /K ) | O ( mn ) | O ( m 3 + n 3 ) | O ( m 3 + n 3 )    | O ( mnr + mn 2 /K ) |
| Optim. states memory | nr + mr             | 2 mn     | m 2 + n 2       | 2 mn +2 m 2 +2 n 2 | 2 nr + mr           |
| Subspace-aware       | ✓                   | ×        | ×               | ×                  | ✓                   |
| Orthogonalization    | ✓                   | ×        | ×               | ×                  | ×                   |

## 2 Related Work

Low-rank gradient optimization. Low-rank gradients naturally emerge during neural network training, as shown in both theoretical and empirical studies [13-15]. This structure has been leveraged to reduce memory and computational costs during training [16-18]. Recent work [2] showed that gradients in reversible layers [19] tend to collapse to rank one over time and used this to adaptively adjust gradient rank in Adam. In this paper, we demonstrate that the same low-rank trend is present in the first-order moment, which we utilize to efficiently apply exact orthogonalization-avoiding the accumulation of approximation errors, such as those encountered in Newton-Schultz, during optimization.

Memory efficient optimizers. Reducing the memory demands of training large language models (LLMs) has driven extensive algorithmic research. One research direction, initiated by LoRA [20], reduces the number of trainable parameters via low-rank adaptation. Yet, such methods often fall short of fully parameterized models, especially during pre-training. Another direction does not restrict the set of trainable parameters but instead optimizes the training methods, with notable examples including AdaRankGrad, GaLore, Fira, Flora, Adam-mini, GaLore-mini, LDAdam, GoLore, LoQT, and Apollo [1-3, 21-25], integrating low-rank gradient projections in optimization. In this work, we reduce memory usage even further by relying solely on a first-order momentum, as shown in Table 1.

Gradient preconditioning. Preconditioning the Gradient method is critical in enhancing the efficiency and effectiveness of optimizers. Several notable approaches for using a preconditioner have emerged, including methods based on signed gradients [26-29], gradient clipping [30], normalization [30, 31], and gradient whitening [7, 32-37]. Recent studies [7, 38] explored gradientorthogonalization strategies, thereby speeding up training. Orthogonalizing gradients effectively constrains updates to lie on directions of uniform magnitude (spectral radius = 1), preventing updates from exaggerating specific gradient directions over others. This procedure ensures a form of normalization that mitigates potential instabilities from ill-conditioned gradients. Unlike these methods, which apply preconditioning or approximate orthogonalization in the high-dimensional parameter space, our approach performs exact SVD-based orthogonalization within an adaptively selected low-rank subspace, offering improved stability and lower computational overhead.

Orthogonal Stochastic Gradient Descent with Momentum (OSGDM). OSGDM [38] is a recently introduced first-order optimization method that speeds up neural network training by orthogonalizing gradients before the optimization step. Specifically, for a data batch ξ ( t ) , OSGDM applies SVD to the gradient matrix G ( t ) l = ∇ W l L ( Φ ( ξ ( t ) ; θ )) of each neural network layer l to generate an orthonormal gradient approximation O l . This ensures diversity among learned representations

and reduces redundancy. The update rule for OSGDM with momentum term γ and learning rate η is defined as,

<!-- formula-not-decoded -->

where orth( G ) = ( GG ⊤ ) -1 / 2 G is the ortogonalization operator, and M l is the first order moment of layer l . Despite the additional computational overhead of SVD, OSGDM empirically converges faster and achieves higher accuracy than common methods such as Adam.

Muon optimizer. At iteration t , given weight W ( t ) , momentum µ , learning rate η t , and objective L t , Muon, introduced by [7], constructs the update rule,

<!-- formula-not-decoded -->

Here, M ( t ) is the momentum at iteration t , initialized as a zero matrix when t = 0 . The NewtonSchulz5 method [35] approximates ( M ( t ) M ( t ) ⊤ ) -1 / 2 M ( t ) , orthogonalizing M ( t ) and thus ensuring uniform update directions, avoiding dominance by few directions. Muon explicitly controls the norm of gradient updates-particularly the spectral norm (or Schattenp norm with large p ), which limits updates to smaller, well-conditioned steps in parameter space. By constraining the spectral norm, moment orthogonalization implicitly prevents overly large or ill-conditioned parameter updates. Such updates often lead to poor generalization due to instability or overfitting. Shortly after the introduction of Muon, the study in [39] proposed a framework to scale Muon for larger LLMs, mainly adding weight decay, and carefully adjusting the per-parameter update scale.

## 3 Method and Main Results

## 3.1 Theoretical Motivation: Exact moments orthogonalization leads to significantly faster convergence

Previous work on pre-training and fine-tuning large language models (LLMs) has primarily focused on reducing memory usage for constrained hardware or lowering computational cost (e.g., [1, 40, 3]). In this paper, we take a step toward accelerating LLM optimization by showing that applying exact orthogonalization (e.g., via SVD) to the first-order moment offers a practical advantage, even over the most accurate approximations, such as the commonly used Newton-Schulz5 method. Specifically, we find that SVD converges faster and incurs lower computational overhead. To support this, we first present a new observation: the moment matrix in LLM training tends to decrease in rank over time. Building on this, we derive an upper bound on the approximation error of NewtonSchulz5, showing that it depends on both the number of iterations and the matrix condition number, thereby highlighting its limitations in ill-conditioned or low-rank settings (precisely the case in LLM optimization moments). This motivates the need for more accurate orthogonalization of moment matrices during LLM training. Applying SVD directly to full-sized layers is generally impractical. The surprising result, however, is that when integrated into a low-rank optimization scheme, the use of SVD becomes not only feasible but preferable. We conclude with a convergence analysis of Muon optimization, which, under these conditions, converges significantly more slowly than the SVD-based alternative. To the best of our knowledge, our convergence analysis of Muon optimization is the first to avoid neglecting the error in the Newton-Schultz approximation [41]. The proofs of all lemmas and theorems of this section are relegated to the Appendix A.

Lemma 3.1 (Moment Becomes Low-Rank During Training) . Let M ( t ) ∈ R n × m denote the first moment of a reversible layer 1 in a moment-based optimization algorithm, updated according to M ( t ) = β 1 M ( t -1) + G ( t ) , where G ( t ) is the gradient matrix at iteration t . Let M ( t ) = U ( t ) Σ ( t ) V ( t ) ⊤ be the singular value decomposition (SVD) of M ( t ) , and define the rankr orthogonal projection matrix as P ( t ) ( r ) = U ( t ) [: , 1: r ] U ( t ) [: , 1: r ] ⊤ . Then the relative error of the best rank-one approximation,

<!-- formula-not-decoded -->

satisfies κ M ( t ) ≤ O ( C -t ) for some constant C &gt; 1 .

1 Reversible networks are formally defined in Appendix B.1

The above result, in (1), implies that M ( t ) approaches its rank-one approximation P ( t ) (1) M ( t ) , as the iteration number increases, namely, M ( t ) becomes rank-one. The following Lemma 3.2 characterizes the impact of the moments' low-rank structure on the approximation error of the Newton-Schulz5 orthogonalization.

Lemma 3.2 (Orthogonalization error E i ) . For a matrix A ∈ R m × n , let σ 1 be the largest singular value of AA ⊤ and σ m be the smallest (without the loss of generality, assume m ≤ n ). Let r ≤ m be the largest index where σ r &gt; σ r +1 = · · · = σ m ≥ 0 . Let κ = σ 1 σ m by the condition number of AA ⊤ . Denote E i the error of Newton-Schultz after i iterations. Then we have

<!-- formula-not-decoded -->

According to the lemma, the approximation error grows exponentially with the condition number. Given the low-rank structure of the first-order moments, low-dimensional optimization can mitigate this error. Specifically, projecting the moment estimates ˆ M ( t ) onto their dominant (small) r -dimensional subspace ensures that the squared moment ˆ M ( t ) ˆ M ( t ) ⊤ is constructed using only the top r squared eigenvalues. These dominant components are significantly larger and exclude near-zero values, resulting in a substantially lower condition number compared to that of the full-rank squared moment matrix. This observation motivates the use of the Muon optimizer within a low-rank optimization framework for LLMs, including 2D reversible layers. Such an approach not only preserves the inherent memory efficiency of low-rank methods but also reduces approximation error during optimization, potentially leading to faster convergence and improved performance compared to full-dimensional training. However, we also empirically observe that the eigenvalues of the moment matrix decay gradually. As shown in Figure 1, even when projecting onto the dominant subspace, the resulting matrix ˆ M ( t ) ˆ M ( t ) ⊤ , composed of the top r = 16 squared eigenvalues, can still exhibit a large condition number, thereby introducing non-negligible approximation error.

(a) Condition number of the first-order moment vs. training step. The red line marks value 10.

<!-- image -->

(b) Illustration of the moment's singular value decay, taken arbitrarily at step 100.

<!-- image -->

Figure 1: Evidence of anisotropy and ill-conditioning in the first-order moment matrix as a function of the Galore steps of the Roberta-base model [42] on the GLUE dataset RTE task [43]: (a) condition number growth, (b) spectral decay of moment.

To comprehend the cumulative error of Newton-Schulz5 orthogonalization at each optimization step, we proceed to derive the convergence rate of the Moun optimization. To that end, we now provide some notations. Consider a neural network denoted as Φ( · ; θ ) , which consists of L layers and is parameterized by θ ≜ [ W d 1 × d 0 1 , . . . , W d L -1 × d L -2 L -1 , W d L × d L -1 L ] . Here, W i represents the weights tensor parameters associated with the i -th layer, for i ∈ [ L ] . We denote the differential loss L , where, with a slight abuse of notation, we write the training problem by min W L ( W ) = E ξ [ L (Φ( W , ξ ))] , if the context refers to the weights of a certain layer. We use the Frobenius norm, denoted ∥ · ∥ F , which is induced by the inner product ⟨ X , Y ⟩ = tr( X ⊤ Y ) . Assume that the stochastic gradient ∇L ( W , ξ ) is an unbiased estimator of the full gradient ∇L ( W ) , with variance bounded by σ 2 , i.e., E [ ∥∇L ( W , ξ ) - ∇L ( W ) ∥ 2 F ] ≤ σ 2 . Let E ( t ) i = orth ( M ( t ) ) -Newton-Schulz ( M ( t ) ) denote the approximation error of the Newton-Schulz (with i ≥ 1 iteration) at time t , where M ( t ) denotes the moment at iteration t .

Lemma 3.3 (Exact convergence rate of Muon) . Consider the Muon optimizer update w.r.t layer W ∈ R m × n defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where M ( t ) = U ( t ) S ( t ) V ( t ) ⊤ denotes the singular value decomposition of M ( t ) , and E ( t ) i represents the Newton-Schulz5 approximation error after i iterations. Suppose the following:

- The gradient ∇L ( W ) is L -Lipschitz continuous.
- There exists δ &gt; 0 such that ∥E ( t ) i ∥ ≤ δ ∥ U ( t ) V ( t ) ⊤ ∥ = δ √ n , for all t .

If we take β = 1 -α with α = min( √ RL σ √ T , 1) , η t = η = √ 4 R √ (10 / (1 -β )+2 m +4 mδ +2 mδ 2 ) TL , and B

<!-- formula-not-decoded -->

= 1 (batch free convergence) than 1 T ∑ T t =1 E [ ∥∇L ( W ( t ) ) ∥ ] is bounded by where R = L ( W ( 0 ) ) -L ∗ . If we take β as an arbitrary constant, and B = T , we have,

<!-- formula-not-decoded -->

Remark 3.4 (Comparison: slower convergence vs exact orthogonalization) . When δ = 0 , indicating an absence of error, the convergence rate is aligned with the one derived in [41], Theorem 2.1, that is

<!-- formula-not-decoded -->

This result overlooks the error associated with the Newton-Schulz5 approximation because it is based on a theoretically exact method of orthogonalization.

Remark 3.5 (The impact of δ on the convergence rate) . A reduction in δ is associated with an improvement in the convergence rate. Furthermore, it should be noted that δ influences the step size η ; a larger δ yields a smaller η , providing an additional explanation for the convergence rate.

Remark 3.6 (The size of δ ) . We acknowledge that the findings of our analysis are applicable only under the conditions specified in 1 -4 √ nδ &gt; 0 ⇒ δ &lt; 1 4 √ n . In scenarios where δ &gt; 1 4 √ n applies, the algorithm may fail to converge. To ensure that δ remains sufficiently small, the Newton-Schulz5 method necessitates a substantial number of iterations, consequently slowing down the convergence. Remark 3.7 (Speed-up by SVD vs Newton-Schulz5 approximation) . According to Lemma 3.2, these low-rank moments, which inherently possess exceptionally high κ , result in an error expressed by (1 -ε ) 2 i concerning a remarkably small ε . This situation necessitates numerous iterations for the Newton-Schultz method to converge. For example, if (1 -ε ) = 0 . 99 is considered and NewtonSchultz5 is utilized with 5 iterations, the error would be ≈ 0 . 99 32 = 0 . 725 , relative to the norm of the moment, namely M .

Recall that in the low-rank setting, accurately computing the pseudoinverse using singular value decomposition (SVD) is numerically advantageous and reasonably computationally affordable compared to iterative methods such as Newton-Schulz. For a general matrix A ∈ R n × r , the SVD provides a decomposition A = UΣV ⊤ , with U ∈ R n × n , Σ ∈ R n × r , and V ∈ R r × r . The MoorePenrose pseudoinverse is then calculated as A † = VΣ † U ⊤ , requiring approximately 4 nr 2 +8 r 3 floating-point operations (FLOPs) for the initial decomposition, and an additional rn 2 + r 2 n FLOPs for subsequent multiplications, totaling roughly 4 nr 2 +8 r 3 + rn 2 + r 2 n FLOPs.

Alternatively, approximating the inverse of A ⊤ A ∈ R r × r using Newton-Schulz iterations involves nr 2 FLOPs to form the matrix A ⊤ A , approximately 20 r 3 +10 r 2 FLOPs for five iterations, and an additional r 2 n FLOPs to multiply by A ⊤ , resulting in a total of about nr 2 + r 2 n +20 r 3 +10 r 2 FLOPs. For example, when the rank is r = 8 and n = 1024 , the SVD approach requires approximately twice as many operations as Newton-Schulz5. Nonetheless, given the superior numerical stability and inherent optimality of the SVD-based method, the increase in computational effort is negligible relative to the other full-matrix operations required in each update step.

## 3.2 Method

We are now ready to present our main algorithm designed to accelerate the low-rank optimization scheme outlined in Algorithm 1. A detailed mathematical formulation of the weight update rule proposed in this paper is given in Appendix C. The algorithm consists of four primary blocks, all contained within an outer loop that continues until convergence is achieved or a predefined number of epochs is reached. Each block serves a specific purpose, which will be explained in detail below.

- Block 1 (Adaptive Subspace Selection) : We select the subspace along the directions of the r largest eigenvectors, but since computing full SVD for large matrices is computationally intensive and memory-demanding, we leverage the Randomized-SVD by [44], which is an efficient technique for producing a 'good" proxy for the optimal low-rank approximation. It solves the optimization problem arg min Q ( t ) ∈ R m × r ∥ ∥ ∥ G -Q ( t ) Q ( t ) ⊤ G ( t ) ∥ ∥ ∥ , and approximates the matrix G as ˆ G ≈

Q ( t ) Q ( t ) ⊤ G ( t ) , that requires O ( mnr + mr 2 ) operations, instead of O (min( mn 2 , m 2 n )) applied by SVD.

- F
- Block 1.1 (Moment Subspace Transformation) : We transform the first-order moments evaluated during the low-rank optimization steps, which occur in Block 2, between the preceding and the newly updated subspace. This transformation is required because, as will be demonstrated later, the first moments of the gradients in Block 2 are aligned with the previously projected subspace. Consequently, a transformation is necessary to translate them from the former subspace to the current one.
- Block 2 (Low-Rank Steepest Descent Optimization) : Here we calculate the (steepest) optimization step. SVD operation is adopted to solve exactly ( M ( t ) M ( t ) ⊤ ) -1 / 2 M ( t ) . Let UΣV ⊤ = ˆ M ( t ) be the singular value decomposition (SVD) of ˆ M ( t ) , we will have ( M ( t ) M ( t ) ⊤ ) -1 / 2 M ( t ) = UV T , which orthogonalizes ˆ M ( t ) . Formally, the

<!-- formula-not-decoded -->

- Block 3 : Rather than using standard gradient clipping, we adopt the Norm-growth Limiter (NL) introduced in [21], which has been shown to slightly outperform traditional clipping techniques by better constraining the progression of gradient magnitudes. Specifically, the gradient update is modified as follows, if ∥ O ( t ) ∥ ∥ O ( t -1) ∥ &gt; γ then O ( t ) ← O ( t ) ∥ O ( t ) ∥ · γ ∥ ∥ O ( t -1) ∥ ∥ , where the scalar γ serves as a growth threshold to regulate abrupt increases in gradient norm from one iteration to the next. We use γ = 1 . 1 , which empirically yields the best results.
- Block 4 (Update Step in the Original Space). To better exploit already computed gradient information, we suggest to use the orthogonal term of the gradient that lies outside the low-rank subspace spanned O ( t ) , namely G ( t ) ⊥ = G ( t ) -Q ( t ) Q ( t ) ⊤ G ( t ) ︸ ︷︷ ︸ ˆ G ( t ) . Importantly, G ( t ) ⊥ does not

interfere with the orthogonalized moment estimation O ( t ) , meaning it can be incorporated without compromising stability. Since G ( t ) is already computed and stored in each iteration, no extra memory is required. Furthermore, because Q ( t ) is of a low rank (typically rank 4 , 8 , or 16 ), the additional computationations overhead is negligible. For efficient memory usage, instead of explicitly forming the perpendicular part G ( t ) ⊥ , we work only in the rankr subspace spanned by Q ( t ) . Practically, we use fact that

<!-- formula-not-decoded -->

which utilizes the already calculated projected gradient ˆ G ( t ) := Q ( t ) ⊤ G ( t ) . Accordingly, the pre-trained model parameters are updated along with weight decay,

<!-- formula-not-decoded -->

To ensure stable training across parameter matrices of different shapes, we interpret the root mean square (RMS) magnitude of updates as implicit layer-wise learning rate adaptation , following

the approach in [39]. By scaling updates by √ max( m,n ) , our method compensates for shapeinduced magnitude differences, achieving consistent effective learning rates across layers, similar to adaptive optimizers like AdamW.

## Algorithm 1 SUMO: Subspace-Aware Moment-Orthogonalization Optimization

```
Input: A weight matrix W ∈ R m × n with m ≥ n . Step size η , decay rate µ , weight decay λ , rank r , subspace update frequency K , step clipping ratio γ . Initialize : t ← 0 repeat # Block 1: Calculate low rank gradient projection. Sample mini-batch B = { ξ 1 , ξ 2 , . . . , ξ | B | } Compute G ( t ) ← ∑ | B | i =1 ∂ ∂ W L (Φ( x i , θ ) , y i ) if t mod K = 0 then Q ( t ) ← Truncated _ Randomized _ SVD( G ( t ) ) # Alternatively Truncated _ SVD( G ( t ) ) # Block 1.1: Moment subspaces transformation R r × r ← Q ( t ) ⊤ Q ( t -1) if t ≥ 1 , else 0 r × r M ( t ) r × n ← RM ( t -1) , if t ≥ 1 , else 0 r × n { 1 st -order moment} end if # Alternatively criteria ∥ ˆ G ( t ) ∥ ≤ ς ˆ G ( t ) ← Q ( t ) ⊤ G ( t ) # Block 2: Low-rank steepest-decent step (moment ortogonalization) M ( t ) ← µ M ( t -1) + ˆ G ( t ) O ( t ) ← Orthogonalization_SVD ( M ( t ) ) # Block 3 (Optional): if ∥ O ( t ) ∥ ∥ O ( t -1) ∥ > γ then O ( t ) ← O ( t ) ∥ O ( t ) ∥ · γ ∥ ∥ O ( t -1) ∥ ∥ # Block 4: Update weight in original space. W ( t ) ← W ( t -1) -η ( G ( t ) -Q ( t ) ( ˆ G ( t ) -O ( t ) ) ) -η λ W ( t -1) t ← t +1 until convergence criteria met (e.g. epoch number, gradient norm ∥ G ( t ) ∥ ≤ ξ ) return W ( T )
```

Note that, for clarity, we can assume, without loss of generality, that m ≥ n . In the opposite scenario, the projection matrix would multiply the gradient from the right-hand side.

Theorem 3.8 (Convergence of SUMO) . For a loss function L , and given architecture Φ , suppose that the compositions of f ≡ L (Φ( · )) is β -smooth non-convex function that is bounded by some M ∈ R + . Let G ( t ) j denote the gradient matrix w.r.t. the j -th reversible layer W ( t ) j , at time t ∈ N , for all j ∈ [ L ] , and T ℓ , ℓ ∈ N times are set by a convergence criterion (that is, ∥ ˆ G ( T ℓ ) j ∥ ≤ ς ℓ ). Then, there exist C ∈ R + and N such that for all T N &gt; C ε 2 , and 1 T N ∑ N -1 i =0 ∑ T i +1 -1 t = T i ∥ ∥ ∥ G ( t ) j ∥ ∥ ∥ 2 F ≤ ε . Namely, Algorithm 1 achieves an ε -critical point, 2 i.e., ∥ ∥ ∥ G ( t ) j ∥ ∥ ∥ 2 F ≤ ε , for some t ∈ N , and any j ∈ [ L ] .

The proof of Theorem 1 can be found in Appendix A. We emphasize that the convergence proof for Galore in [1] addresses only optimization within fixed subspaces, ignoring dynamic updates. AdaRankGrad's proof [2] first established guarantees for the complete dynamic-subspace updates, yet both prior works simplified the inner steps as standard SGD. In contrast, SUMO's convergence proof explicitly considers the exact optimization steps without simplifications.

To reduce memory consumption, Algorithm 1 applies per-layer weight updates during backpropagation, following recent works such as [45]. This contrasts with conventional optimizers, which store full gradients and update all weights afterward, potentially leading to inefficiencies. Details for post-hoc adapter extraction are discussed in Appendix B.

<!-- image -->

Step

Figure 2: SUMO with SVD demonstrates superior convergence speed ( ∼ 1 . 6 × faster), attaining comparable or higher accuracy than GaLore and SUMO with Newton-Schultz5 with significantly fewer optimization steps on QNLI.

## 4 Experiments

Fine-tuning on GLUE benchmark. Our model was evaluated using the GLUE benchmark [43] through the fine-tuning of the pre-trained Roberta-base model [42] across eight tasks. The comparative analysis includes full fine-tuning, LoRA, and GaLore methodologies, with the results enumerated in Table 2. The reported metrics are the overall accuracy (matched and mismatched) for MNLI, Matthew's correlation for CoLA, Pearson correlation for STS-B, F1-score for MRPC, and accuracy for the remaining tasks. Evidently, our approach improves fine-tuning accuracy while requiring less training memory, using only a single moment compared to GaLore. The experiments were carried out using the NVIDIA A100 GPU.

Table 2: Comparison of SUMO against state-of-the-art memory-efficient fine-tuning methods on the GLUE benchmark using the pre-trained RoBERTa-Base model. For comparison, we provide detailed results for SUMO using both SVD and Newton-Schulz5 orthogonalizations (ablation study).

| Model                         | Memory   |   CoLA |   STS-B |   MRPC |   RTE |   SST2 |   MNLI |   QNLI |   QQP |
|-------------------------------|----------|--------|---------|--------|-------|--------|--------|--------|-------|
| Full Fine-Tuning              | 747M     |  62.24 |   90.92 |  91.3  | 79.42 |  94.57 |  87.18 |  92.33 | 92.28 |
| LoRA (rank=4)                 | 257M     |  61.38 |   90.57 |  91.07 | 78.7  |  92.89 |  86.82 |  92.18 | 91.29 |
| GaLore (rank=4)               | 253M     |  60.35 |   90.73 |  92.25 | 79.42 |  94    |  87    |  92.24 | 91.06 |
| SUMO (Newton-Schulz5, rank=4) | 197M     |  61.8  |   90.82 |  92.43 | 79.36 |  94.17 |  86.92 |  92.26 | 91.27 |
| SUMO (SVD, rank=4)            | 197M     |  62.3  |   91.04 |  93.5  | 81.07 |  94.93 |  87.34 |  93.26 | 91.68 |
| LoRA (rank=8)                 | 264M     |  61.83 |   90.8  |  91.9  | 79.06 |  93.46 |  86.94 |  92.25 | 91.22 |
| GaLore (rank=8)               | 257M     |  60.06 |   90.82 |  92    | 79.78 |  94.38 |  87.17 |  92.2  | 91.11 |
| SUMO (Newton-Schulz5, rank=4) | 198M     |  61.74 |   90.79 |  91.94 | 79.69 |  94.17 |  87.21 |  92.24 | 91.38 |
| SUMO (rank=8)                 | 198M     |  61.7  |   91.1  |  93.7  | 81.37 |  94.82 |  87.58 |  93.67 | 91.72 |

Pre-training Llama on C4 Dataset. To highlight the effectiveness of our method in pre-training, we pre-train Llama models following the evaluation protocol of [1] and compare performance with the state-of-the-art method, in terms of perplexity and memory usage. Specifically, we compare SUMO's performance with state-of-the-art methods in terms of perplexity and memory efficiency. For this evaluation, we trained large Llama-based models on the C4 dataset, a curated and extensive version of the Common Crawl web corpus [46]. This dataset is widely used for pre-training language models and developing word representations. To better reflect real-world pre-training scenarios, we trained on a non-repeating, large-scale dataset and scaled model sizes up to 1 billion parameters. The results of these experiments are shown in Table 3. Experiments were conducted using an NVIDIA H200 GPU.

Few/Zero-shot reasoning and long-context generalization. To evaluate the performance of our method on a complex reasoning task, we utilize the GSM8K dataset [47] to test systematic generalization. For these experiments, we used a batch size of 32 and 10 epochs for fine-tuning. We present the performance result in Table 4 training Phi-2 (2.7B) model [48], and in Table 5 training

2 Also known as ε -stationary, see, e.g., [14].

Table 3: Comparison of state-of-the-art low-rank algorithms for pre-training Llama models of varying sizes on the C4 dataset. The results are reported in terms of validation perplexity. As shown, SUMO leads to improved performance with substantial memory reduction compared to leading parameter-efficient fine-tuning schemes.

| Method          | 60M                 | 130M                | 350M                | 1B                |
|-----------------|---------------------|---------------------|---------------------|-------------------|
| Full-Rank       | 34 . 06 ( 0 . 36 G) | 25 . 08 ( 0 . 76 G) | 18 . 80 ( 2 . 06 G) | 15 . 56(7 . 80G)  |
| GaLore          | 34 . 88 ( 0 . 24 G) | 25 . 36 ( 0 . 52 G) | 18 . 95 ( 1 . 22 G) | 15 . 64(4 . 38G)  |
| Low-Rank        | 78 . 18 ( 0 . 26 G) | 45 . 51 ( 0 . 54 G) | 37 . 41 ( 1 . 08 G) | 142 . 53(3 . 57G) |
| LoRA            | 34 . 99 ( 0 . 36 G) | 33 . 92 ( 0 . 80 G) | 25 . 58 ( 1 . 76 G) | 19 . 21(6 . 17G)  |
| ReLoRA          | 37 . 04 ( 0 . 36 G) | 29 . 37 ( 0 . 80 G) | 29 . 08 ( 1 . 76 G) | 18 . 33(6 . 17G)  |
| SUMO            | 34.26 ( 0 . 23 G)   | 24.87 ( 0 . 51 G)   | 18.69 ( 1 . 16 G)   | 14.68 ( 3 . 84 G) |
| Training Tokens | 1 . 1 B             | 2 . 2 B             | 6 . 4 B             | 13.1B             |
| r/d model       | 128 / 256           | 256 / 768           | 256 / 1024          | 512 / 2048        |

Llama (3B) model [49]. The results demonstrate that the proposed method significantly improves generalization to out-of-distribution data. We used an NVIDIA H200 GPU.

Table 4: Zero-shot evaluation on GSM8K dataset (Phi-2, 2.7B). Table 5: 8-shot evaluation on GSM8K dataset (LLaMA, 3B).

| Phi-2 (2.7B)   |   Rank | Accuracy (0-shot)   | LLaMA (3B)   |   Rank | Accuracy (8-shot)   |
|----------------|--------|---------------------|--------------|--------|---------------------|
| Base Model     |     64 | 15 . 16%            | Base Model   |     64 | 17 . 93%            |
| Galore         |     64 | 52 . 24%            | Galore       |     64 | 74 . 9%             |
| LoRA           |     64 | 42 . 8%             | LoRA         |     64 | 68 . 3%             |
| SUMO           |     64 | 54.13 %             | SUMO         |     64 | 76.7 %              |

Additional experiments and ablation studies are presented in the Appendix D.

Zero-shot generalization. Following this, we extend our evaluation to diverse commonsense and reasoning tasks using zero-shot methods, and then conduct the exact evaluation protocol described in Table 4 of [3] (which suggests the Apollo optimizer). The task details are presented in Appendix D.1. To ensure a fair comparison, we repeat the exact experimental setup used for AdamW, APOLLO, and APOLLO-Mini. The results are summarized in Table 6 below.

Table 6: Zero-shot evaluation of Llama-350M models pretrained with sequence length 1024 across reasoning tasks (lower is better for perplexity; higher is better otherwise).

| Method      | Memory   |   Perplexity |   BoolQ |    RTE |     HS |     WG |   OBQA |   ARC-E |   ARC-C |   PIQA |   SciQ |   MathQA |   Avg. |
|-------------|----------|--------------|---------|--------|--------|--------|--------|---------|---------|--------|--------|----------|--------|
| AdamW       | 1.37G    |        16.3  |  0.4917 | 0.4693 | 0.3688 | 0.5233 |  0.332 |  0.3729 |  0.2449 | 0.6534 |  0.609 |   0.2064 | 0.4272 |
| APOLLO      | 0.34G    |        15.64 |  0.5373 | 0.4698 | 0.385  | 0.4925 |  0.322 |  0.3788 |  0.2483 | 0.6681 |  0.624 |   0.2127 | 0.4406 |
| APOLLO-Mini | 0.15G    |        16.12 |  0.5376 | 0.4562 | 0.3707 | 0.5217 |  0.324 |  0.3758 |  0.2312 | 0.6638 |  0.619 |   0.2224 | 0.4374 |
| SUMO        | 0.18G    |        15.49 |  0.5479 | 0.4709 | 0.3937 | 0.5313 |  0.321 |  0.3832 |  0.2496 | 0.6709 |  0.623 |   0.2246 | 0.4416 |

As shown, the SUMO-pretrained Llama-350M model achieves lower perplexity on average and consistently outperforms on downstream benchmarks.

## 5 Discussion

Our results highlight that exact moment orthogonalization within a low-dimensional adaptive subspace significantly improves both convergence and stability in memory-efficient LLM training. By avoiding the approximation errors of Newton-Schulz5, the proposed SUMO leverages the low-rank structure of gradients to enable accurate, spectral-norm-aligned updates with minimal overhead.

Empirically, SUMO outperforms prior low-rank methods on both fine-tuning and pre-training tasks, achieving greater memory reduction than memory-efficient benchmarks such as Galore. Our theoretical analysis further confirms its superior convergence properties under practical conditions. These findings position SUMO as a simple yet effective alternative to approximate geometric optimizers. Future work may investigate parallel computations for orthogonalization, integrate quantization techniques, and assess the effectiveness of the method in knowledge editing [50] or domain generalization [51, 52].

## Acknowledgment

TT was supported by the Israel Science Foundation (No. 1940/23) and MOST (No. 0007091) grants. OL was supported by the MOST grant No. 0007341.

## References

- [1] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. Galore: Memory-efficient llm training by gradient low-rank projection, 2024.
- [2] Yehonathan Refael, Jonathan Svirsky, Boris Shustin, Wasim Huleihel, and Ofir Lindenbaum. Adarankgrad: Adaptive gradient rank and moments for memory-efficient LLMs training and fine-tuning. In The Thirteenth International Conference on Learning Representations , 2025.
- [3] Hanqing Zhu, Zhenyu Zhang, Wenyan Cong, Xi Liu, Sem Park, Vikas Chandra, Bo Long, David Z Pan, Zhangyang Wang, and Jinwon Lee. Apollo: Sgd-like memory, adamw-level performance. arXiv preprint arXiv:2412.05270 , 2024.
- [4] Barak Battash, Lior Wolf, and Ofir Lindenbaum. Revisiting the noise model of stochastic gradient descent. In International Conference on Artificial Intelligence and Statistics , pages 4780-4788. PMLR, 2024.
- [5] Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization, 2018.
- [6] Nikhil Vyas, Depen Morwani, Rosie Zhao, Mujin Kwun, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade. Soap: Improving and stabilizing shampoo using adam, 2025.
- [7] Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cecista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024.
- [8] Louis Cesista Franz. The Case for Muon, October 2024.
- [9] Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology. arXiv preprint arXiv:2409.20325 , 2024.
- [10] Dmitry Kovalev. Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization, 2025.
- [11] Hao Sun, Li Shen, Qihuang Zhong, Liang Ding, Shixiang Chen, Jingwei Sun, Jing Li, Guangzhong Sun, and Dacheng Tao. Adasam: Boosting sharpness-aware minimization with adaptive learning rate and momentum for training deep neural networks, 2023.
- [12] Nicholas J Higham. Newton's method for the matrix square root. Mathematics of computation , 46(174):537-549, 1986.
- [13] Jingzhao Zhao, Frederik T. Schaefer, and Anima Anandkumar. Zero initialization: Initializing neural networks with only zeros and ones. Transactions on Machine Learning Research , 2022.
- [14] Romain Cosson, Ali Jadbabaie, Anuran Makur, Armin Reisizadeh, and Devavrat Shah. LowRank Gradient Descent. IEEE Open Journal of Control Systems , 2023.
- [15] Greg Yang, Jacob B. Simon, and Jeremy Bernstein. A spectral condition for feature learning. arXiv preprint arXiv:2310.17813 , 2023. arXiv:2310.17813.
- [16] M. Gooneratne, K. C. Sim, P. Zadrazil, A. Kabel, F. Beaufays, and G. Motta. Low-rank gradient approximation for memory-efficient on-device training of deep neural network. In 2020 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2020, Barcelona, Spain, May 4-8, 2020 . IEEE, 2020.
- [17] Shuang Huang, Brian D. Hoskins, Michael W. Daniels, Matthew D. Stiles, and George C. Adam. Low-Rank Gradient Descent for Memory-Efficient Training of Deep In-Memory Arrays. ACM Journal on Emerging Technologies in Computing Systems , 2023.

- [18] Ionut-Vlad Modoranu, Alexander Kalinov, Ermin Kurtic, Erwin Frantar, and Dan Alistarh. Error Feedback Can Accurately Compress Preconditioners. ArXiv preprint arXiv:2306.06098 , 2023. arXiv:2306.06098.
- [19] Yuandong Tian, Lantao Yu, Xinlei Chen, and Surya Ganguli. Understanding self-supervised learning with dual deep networks, 2021.
- [20] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.
- [21] Xi Chen, Kaituo Feng, Changsheng Li, Xunhao Lai, Xiangyu Yue, Ye Yuan, and Guoren Wang. Fira: Can we achieve full-rank training of llms under low-rank constraint? arXiv preprint arXiv:2410.01623 , 2024.
- [22] Yongchang Hao, Yanshuai Cao, and Lili Mou. Flora: Low-rank adapters are secretly gradient compressors. ArXiv , abs/2402.03293, 2024.
- [23] Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu, Yinyu Ye, Zhi-Quan Luo, and Ruoyu Sun. Adam-mini: Use fewer learning rates to gain more. arXiv preprint arXiv:2406.16793 , 2024.
- [24] Thomas Robert, Mher Safaryan, Ionut-Vlad Modoranu, and Dan Alistarh. Ldadam: Adaptive optimization from low-dimensional gradient statistics, 2025.
- [25] Sebastian Loeschcke, Mads Toftrup, Michael J. Kastoryano, Serge Belongie, and Vésteinn Snæbjarnarson. Loqt: Low-rank adapters for quantized pretraining, 2024.
- [26] Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signsgd: Compressed optimisation for non-convex problems. In International Conference on Machine Learning , pages 560-569. PMLR, 2018.
- [27] Michael Crawshaw, Mingrui Liu, Francesco Orabona, Wei Zhang, and Zhenxun Zhuang. Robustness to unbounded smoothness of generalized signsgd. Advances in neural information processing systems , 35:9955-9968, 2022.
- [28] Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V. Le. Symbolic discovery of optimization algorithms. In NeurIPS , 2023.
- [29] Frederik Kunstner, Jacques Chen, Jonathan Wilder Lavington, and Mark Schmidt. Noise is not the main factor behind the gap between sgd and adam on transformers, but sign descent might be. arXiv preprint arXiv:2304.13960 , 2023.
- [30] Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv Kumar, and Suvrit Sra. Why are adaptive methods good for attention models? Advances in Neural Information Processing Systems , 33:15383-15393, 2020.
- [31] Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. arXiv preprint arXiv:1904.00962 , 2019.
- [32] Zhirong Yang and Jorma Laaksonen. Principal whitened gradient for information geometry. Neural Networks , 21(2-3):232-240, 2008.
- [33] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR (Poster) , 2015.
- [34] Dongseong Hwang. Fadam: Adam is a natural gradient optimizer using diagonal empirical fisher information. arXiv preprint arXiv:2405.12807 , 2024.
- [35] Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology, 2024.
- [36] Jeremy Bernstein and Laker Newhouse. Modular duality in deep learning, 2024.

- [37] David E Carlson, Edo Collins, Ya-Ping Hsieh, Lawrence Carin, and Volkan Cevher. Preconditioned spectral descent for deep learning. Advances in neural information processing systems , 28, 2015.
- [38] Mark Tuddenham, Adam Prügel-Bennett, and Jonathan Hare. Orthogonalising gradients to speed up neural network optimisation. arXiv preprint arXiv:2202.07052 , 2022.
- [39] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for llm training, 2025.
- [40] Yehonathan Refael, Adam Hakim, Lev Greenberg, Tal Aviv, Satya Lokam, Ben Fishman, and Shachar Seidman. Slip: Securing llms ip using weights decomposition, 2024.
- [41] Jiaxiang Li and Mingyi Hong. A note on the convergence of muon and further. arXiv preprint arXiv:2502.02900 , 2025.
- [42] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 , 2019.
- [43] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. In Advances in Neural Information Processing Systems , volume 32, 2019.
- [44] Nathan Halko, Per-Gunnar Martinsson, and Joel A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions, 2010.
- [45] Kai Lv, Hang Yan, Qipeng Guo, Haijun Lv, and Xipeng Qiu. Adalomo: Low-memory optimization with adaptive learning rate, 2024.
- [46] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research , 21(140):1-67, 2020.
- [47] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems, 2021. Arxiv , 2021.
- [48] Mojan Javaheripi, Sébastien Bubeck, Marah Abdin, Jyoti Aneja, Sebastien Bubeck, Caio César Teodoro Mendes, Weizhu Chen, Allie Del Giorno, Ronen Eldan, Sivakanth Gopi, et al. Phi-2: The surprising power of small language models. Microsoft Research Blog , 1(3):3, 2023.
- [49] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 , 2023.
- [50] Amit Rozner, Barak Battash, Lior Wolf, and Ofir Lindenbaum. Knowledge editing in language models via adapted direct preference optimization. In Findings of the Association for Computational Linguistics: EMNLP 2024 , pages 4761-4774, 2024.
- [51] Yehonathan Refael, Iftach Arbel, Ofir Lindenbaum, and Tom Tirer. Lorenza: Enhancing generalization in low-rank gradient llm training via efficient zeroth-order adaptive sam, 2025.
- [52] Amit Rozner, Barak Battash, Lior Wolf, and Ofir Lindenbaum. Domain-generalizable multipledomain clustering. Transactions on Machine Learning Research , 2023.
- [53] Ashok Cutkosky and Harsh Mehta. Momentum improves normalized sgd, 2020.
- [54] Kenji Kawaguchi. Deep learning without poor local minima. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 29. Curran Associates, Inc., 2016.

- [55] Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. MAWPS: A math word problem repository. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 1152-1157, 2015.
- [56] C. Clark, K. Lee, M.W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044 , 2019.
- [57] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations , 2018.
- [58] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics , pages 4791-4800, 2019.
- [59] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM , 64(9):99-106, 2021.
- [60] Todor Mihaylov, Peter Clark, Oyvind Tafjord, and Tushar Khot. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing , pages 2381-2391, 2018.
- [61] Peter Clark and Oren Etzioni. Think you have solved question answering? try arc, the ai2 reasoning challenge. In arXiv preprint arXiv:1803.05457 , 2018.
- [62] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pages 7432-7439, 2020.
- [63] Matt Gardner Johannes Welbl, Nelson F. Liu. Crowdsourcing multiple choice science questions, 2017.
- [64] Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms, 2019.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: we propose SUMO (Subspace-Aware Moment-Orthogonalization), an optimizer that employs exact singular value decomposition (SVD) for moment orthogonalization within a dynamically adapted low-dimensional subspace, enabling norm-inducing steepest descent optimization steps. By explicitly aligning optimization steps with the spectral characteristics of the loss landscape, SUMO effectively mitigates approximation errors associated with commonly used methods like Newton-Schulz orthogonalization approximation (claim. 3.3). We theoretically establish an upper bound on these approximation errors (claim. 3.2), proving their dependence on the condition numbers of moments, conditions we analytically demonstrate are encountered during LLM training. Furthermore, we both theoretically and empirically illustrate that exact orthogonalization via SVD substantially improves convergence rates while reducing overall complexity. Empirical evaluations confirm that SUMO accelerates convergence, enhances stability, improves performance, and reduces memory requirements by up to 20% compared to state-of-the-art methods. (Figure. 2, Table. 1)

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Our methods work when the layers are reversible, and improves other methods especially when the moments are ill-conditioned.

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

Justification: all assumptions of theoretical results are presented in the paper, and all proof are in the Appendix.

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

Justification: Everything needed for reproducibility is presented in Section 4.

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

Justification: We provide experimental details in the main text and appendix. A GitHub link well be published soon.

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

Answer:[Yes]

Justification: All training and test details are presented in Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All detailed in Section 4.

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

Justification: All detailed in Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer:[NA]

Justification: The paper talks about optimizers. It does not have societal impact.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer:[NA]

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We followed and gave proper credits for our use of data and models in Section 4.

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

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human objects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human objects.

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

## A Proofs of Section 3

In this section, we prove all the theorems and results of Section 3.

Proof of Lemma 3.1. We aim to show that if the gradient G ( t ) becomes approximately rank-one exponentially fast, then the exponentially weighted moving average of the gradients (i.e., the momentum M ( t ) ) also exhibits exponential decay of higher-rank components.

Consider the singular value decomposition of the gradient G ( t ) = U ( t ) Σ ( t ) V ( t ) ⊤ , at iteration t . For all natural numbers r &lt; m, , we define H ( t ) m × r ( r ) = U [: , 1 : r ] . To enhance notation clarity, denote P ( t ) ( r ) = H ( t ) ( r ) H ( t ) ⊤ ( r ) , , where P ( t ) ( r ) represents an orthogonal projection matrix, satisfying the conditions P ( t ) ⊤ ( r ) P ( t ) ( r ) = P ( t ) ( r ) , and P ( t ) ( r ) = P ( t ) ⊤ ( r ) . Without compromising generality, it is assumed that at t = 0 , the rank of G (0) is characterized by rank ( G (0) ) &gt; r . For reversible networks, it has been established in [1][Theorem 3.2] that the gradients assume the form G ( t ) = 1 N ∑ N i =1 ( A i -B i W ( t ) C i ) , characterized by constant matrices { A i } i and positive semidefinite (PSD) matrices { B i , C i } i , for t ≥ t 0 , where t 0 ∈ N holds. It is pertinent to recall that the vanilla weight update can be represented as W ( t ) = W ( t -1) + η G ( t -1) . Let S ≜ 1 N ∑ N i =1 C i ⊗ B i and λ 1 &lt; λ 2 denote its two smallest distinct eigenvalues. To substantiate our findings, we utilize several results and arguments presented in the proof of Lemma 3.3 in [1]. Specifically, consider G ( t 0 ) as the projection of G ( t 0 ) onto the minimal eigenspace V 1 of S corresponding to λ 1 . . According to our assumption, the rank of G ( t 0 ) is L , and its singular value decomposition (SVD) is given by G ( t 0 ) = ∑ L l =1 c l z l y ⊤ l , where { z l } L l =1 and { y l } L l =1 are orthonormal unit vectors, and { c l } L l =1 are the corresponding singular values. Therefore, as per Lemma 3.3 in [1], the gradient can be decomposed into,

<!-- formula-not-decoded -->

where g ∥ 0 is the projection of G (0) onto the minimal eigenspace V 1 of S = 1 N ∑ N i =1 C i ⊗ B i , and g ⊥ 0 is orthogonal to V 1 . Here, λ 1 &lt; λ 2 are the smallest distinct eigenvalues of S .

We now unroll the momentum update, M ( t ) = ∑ t s =1 β t -s G ( s ) . Substitute the decomposition of G ( s ) ,

<!-- formula-not-decoded -->

Let us define a t ≜ ∑ t s =1 β t -s (1 -ηλ 1 ) s , b t ≜ ∑ t s =1 β t -s (1 -ηS ) s g ⊥ 0 , so that ∥ ∥ M ( t ) ∥ ∥ 2 F = a t g ∥ 0 + b t . Now, compute the squared Frobenius norm:

<!-- formula-not-decoded -->

Since g ∥ 0 ⊥ g ⊥ 0 and b t lies in the span of g ⊥ 0 , we have ⟨ g ∥ 0 , b t ⟩ = 0 , thus,

<!-- formula-not-decoded -->

Likewise, the spectral norm ∥ M ( t ) ∥ 2 2 ≥ a 2 t ∥ g ∥ 0 ∥ 2 2 . Hence, the ratio

<!-- formula-not-decoded -->

Using that ∥ g ∥ 0 ∥ 2 2 = σ 2 1 , and the decay bound ∥ b t ∥ 2 F = O ((max { β, 1 -ηλ 2 } ) 2 t ) , while a 2 t = Ω((max { β, 1 -ηλ 1 } ) 2 t ) , we conclude:

<!-- formula-not-decoded -->

for some constant C &gt; 1

<!-- formula-not-decoded -->

Before proving Lemma 3.3, we shortly present the following two preliminary lemmas. To that end, we present the following notations,

- M ( t ) - The moment in iteration t . Its dimensions are n × m , where n &lt; m .
- ∥ · ∥ - The Frobenius norm: ∥ A ∥ = ∥ A ∥ F = √ AA ⊤
- L ∗ - A stationary point to which the loss L converges.
- B - Batch size.
- For A ∈ R m × m and B ∈ R n × n we denote AI m × n B ⊤ by AB ⊤ for convenience.

Additionally, we note that our proof is based on an equivalent but slightly modified formulation of moment's update. Specifically, instead of using the standard formulation of the moment's update

<!-- formula-not-decoded -->

we consider the convex combination,

<!-- formula-not-decoded -->

This alternative formulation simplifies the analysis, but equivalent. To show that, we point out that we can choose an modefied learning step η ∗ = η 1 -β &gt; 0 we get the same weight's updating step.

<!-- formula-not-decoded -->

where Orth is the SVD orthogonalization step, formaly solving

<!-- formula-not-decoded -->

Obviously, β &gt; 0 could be chosen in a way that β 1 -β M ( t ) would result in any required positive real number.

We assume the following 4 assumptions throughout our proofs:

- (A1) The gradient ∇L ( W ) is L -Lipschitz continuous.
- (A2) ∇L ( W , ξ ) is an unbiased estimator of ∇L ( W ) where L ( W , ξ ) is the gradient of L ( W ) when taking a single training sample ξ .
- (A3) E ∥∇L ( W , ξ ) -∇L ( W ) ∥ ≤ σ 2 .
- (A4) There exists δ &gt; 0 such that ∥E ( t ) 5 ∥ ≤ δ ∥ U ( t ) V ( t ) ⊤ ∥ = δ √ m for all t .

Lemma A.1 (Descent Lemma with Newton-Schulz Approximation Error) . Consider the Muon optimizer update defined by

<!-- formula-not-decoded -->

where M ( t ) = U ( t ) S ( t ) V ( t ) ⊤ is the singular value decomposition of M ( t ) , and E ( t ) 5 represents the Newton-Schulz (5 iterations) approximation error. Additionally, assume (A1) - (A4). Then the following holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Since L is L-lipschitz function, the descent lemma holds. Thus we have

<!-- formula-not-decoded -->

Where in ( ∗ ) we used [53], equation 2.8.

Lemma A.2. For constant η t = η &gt; 0 , the following holds

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Using Lemma A.1, isolating ∥∇L ( W ( t ) ) ∥ and summing over all steps

<!-- formula-not-decoded -->

Proof of Lemma 3.3. The proof follows [53]. Using the same notations as [53], we denote ˆ γ ( t ) = M ( t ) -∇L ( W ( t ) ) , γ ( t ) = G ( t ) -∇L ( W ( t ) ) and S ( X , Y ) = ∇L ( X ) -∇L ( Y ) . Note that we have the following

- E [ γ ( t ) ] = 0 from A(2).
- E [ ∥ γ ( t ) ∥ 2 ] ≤ σ 2 m from A(3).

□

̸

- E [ ⟨ γ ( i ) , γ ( j ) ⟩ ] = 0 , ∀ i = j since γ ( i ) and γ ( j ) are independent.
- ∥ S ( X , Y ) ∥ ≤ L ∥ X -Y ∥ from A(1).

Now following the update in (2), we get therefore

<!-- formula-not-decoded -->

Taking expectation we get (using the fact that ˆ δ 1 = δ 1 ):

<!-- formula-not-decoded -->

All in all, we get

<!-- formula-not-decoded -->

Using Lemma A.2, we get

<!-- formula-not-decoded -->

Dividing both sides by η -4 ηL √ mδ 4 we get

<!-- formula-not-decoded -->

Now we have two types of parameter choice. If we take B = 1 (batch size free), we need to take 1 -β = min(1 , √ RL σ √ T ) so that we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we take β as an arbitrary constant in (0 , 1) , then we will need to take B = T , so that

<!-- formula-not-decoded -->

Proof of Lemma 3.2. We denote B = AA ⊤ , X k the result after k Newton-Schultz iterations, X 0 = B ∥ B ∥ 2 and O = UV ⊤ where B = UΣV ⊤ is the SVD decomposition with σ 1 ≥ σ 2 ≥ · · · ≥ σ m the singular values.

It is known that Newton-Schultz converges quadratically, so we have

<!-- formula-not-decoded -->

We now bound ∥ X 0 -O ∥ 2 . We know that X 0 = UΣV ⊤ ∥ B ∥ 2 = UΣV ⊤ σ 1

<!-- formula-not-decoded -->

Where last equality is due to the fact that ∥ · ∥ 2 is unitary invariant. The matrix Σ σ 1 -I is diagonal with values σ i σ 1 -1 on the diagonal. From that observation we get that

<!-- formula-not-decoded -->

For the Frobenius norm, we get a similar analysis. ∥ X 0 -O ∥ F = ∥ ∥ ∥ Σ σ 1 -I ∥ ∥ ∥ F since ∥ · ∥ F is unitary invariant, so we just need to calculate ∥ ∥ ∥ Σ σ 1 -I ∥ ∥ ∥ F . It is known that ∥ ∥ ∥ Σ σ 1 -I ∥ ∥ ∥ F ≤ √ r ∥ ∥ ∥ Σ σ 1 -I ∥ ∥ ∥ 2 so all in all we have

<!-- formula-not-decoded -->

Proof of Theorem 3.8. for any layer j ∈ [ L ] ; in the following, for simplicity of notation, we ignore the index j and use G ( t ) instead. By Lemma A.3, the low-rank optimization block 1 in Algorithm 1 is guaranteed to converge; we denote by T ℓ ∈ N the time index t at which we enter block 1 for the ℓ th time (i.e., ∥ ˆ G ( T ℓ ) j ∥ ≤ ς 2 ), for ℓ ∈ N . Furthermore, we recall that G ( t ) j ≜ ∇ W j f ( θ ( t ) ) ; when clear from the context, we omit j from W j , and use instead ∇ W j f ( θ ( t ) ) = ∇ f ( W ( t ) ) . Consider the SVD decomposition of the gradient ∇ W j f ( θ T i ) = U ( T i ) Σ ( T i ) V ( T i ) ⊤ . For t ∈ [ T i , T i +1 -1] , we define the projected gradient as ˆ G ( t ) ≜ P ( T i ) ( r ) G ( t ) , where P ( T i ) ( r ) = U ( T i ) [: , : r ] ⊤ , using the exact truncated-SVD calculation (in Block 1). For simplicity, we refer to Q ( T i ) ( r ) as Q ( T i ) , and we

denote P ( T i ) = Q ( T i ) ⊤ Q ( T i ) . Next, let h t ≜ f ( W ( t ) ) -f ( W ( T i +1 ) ) , and η t denote the learning rate. Then,

<!-- formula-not-decoded -->

where (1) follows Eq 7, and (2) follows Lemma A.3. Thus, summing over t ∈ [ T i , T i +1 ) , we get

<!-- formula-not-decoded -->

Assume a constant learning rate η t = η , and define T i := T i +1 -T i . For each interval [ T i , T i +1 -1] , we have,

<!-- formula-not-decoded -->

Summing over all i = 1 , . . . , N ,

<!-- formula-not-decoded -->

where T = ∑ N i =1 T i is the total number of iterations. Using h T 0 -h T N +1 ≤ M , we get,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, the bound becomes,

<!-- formula-not-decoded -->

Recall that for P ( T i ) , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Lemma B.3 in [1], under η ≤ 2 λ max , we get,

<!-- formula-not-decoded -->

for some α ∈ (0 , 1] . Then,

Hence,

<!-- formula-not-decoded -->

Accordingly, for any ε &gt; 0 , if T N satisfies

<!-- formula-not-decoded -->

a sufficient condition for this to hold is,

<!-- formula-not-decoded -->

it follows that

<!-- formula-not-decoded -->

Thus, there exists an iteration index t ∈ [0 , T N ] such that

<!-- formula-not-decoded -->

which, by definition, implies that the algorithm reaches an ε -critical point.

This concludes that Algorithm 1 achieves an ε -critical point.

□

In the following, we provide the auxiliary lemma that is used in the proof of Theorem 3.8.

Lemma A.3 (Convergence of the Inner Fixed Low-Rank Optimization) . Consider the same setting and assumptions as in Theorem 3.8. Then, the second time t = T ℓ ∈ N in which Algorithm 1 enters Block 1 (where it updates the projection matrix) happens for a finite ℓ ∈ N .

Proof. By the β -smoothness of f , we have

<!-- formula-not-decoded -->

Substituting the update rule W ( t +1) = W ( t ) -η t O ( t ) , we get

<!-- formula-not-decoded -->

Since ˆ G ( t ) = P ( t ) ( r ) ⊤ G ( t ) and O ( t ) ∈ range ( P ( t ) ( r ) ⊤ ) , it holds that

<!-- formula-not-decoded -->

By equation (2.8) in [41], we have

<!-- formula-not-decoded -->

Now we bound ∥ ˆ G ( t ) -M ( t ) ∥ F :

<!-- formula-not-decoded -->

Substituting this into the previous inequality gives

<!-- formula-not-decoded -->

Substituting into equation (5), we obtain

<!-- formula-not-decoded -->

Since by definition ∥ O ( t ) ∥ 2 F ≤ n , we get

<!-- formula-not-decoded -->

For constant step size η t = η , summing over t = 1 to T , we get

<!-- formula-not-decoded -->

Rearranging and using f ( W (1) ) -f ∗ ≤ M , we conclude

<!-- formula-not-decoded -->

## B Additional Information

Definition B.1. (Reversibility [19]) A neural network ϕ that maps the input x to output y = ϕ ( x ; θ ) is reversible, if there exists L ( x ; θ ) so that y = L ( x ; θ ) x , and the backpropagated gradient g x satisfies g x = L ⊤ ( x ; θ ) g y , where g y is the backpropagated gradient at the output y . L ( x ; θ ) depends on the input x and weight θ in the network ϕ .

Several critical observations regarding Algorithm 1 warrant attention. Initially, in order to minimize memory consumption, Algorithm 1 implements a per-layer weight update during the process of backpropagation, as advocated by contemporary studies, see, e.g., [45]. This approach contrasts with conventional optimizers, which typically update all weights after backpropagation by retaining the complete gradients in memory, a method potentially marked by significant inefficiency. Should there be a desire to generate an adapter (i.e., a parallel low-dimensional LoRA-type model) subsequent to fine-tuning, this can be achieved with efficiency through the following steps. Firstly, the training weights gap ∆ ≜ W Fine-Tuned -W Pretrained is computed, where W Fine-Tuned denotes the model weight upon process completion, and W Pretrained refers to the original model weight. Subsequently, r Adaptor ≜ rank (∆) is determined utilizing a matrix ranking algorithm, followed by the resolution of min A ∈ R n × r Adaptor , B ∈ R r Adaptor × m ∥ ∆ -AB ∥ 2 F through any optimization algorithm (e.g., gradient descent). It is noteworthy that any solution to this matrix factorization optimization problem is well-known as a global optimum [54].

## C Update Step Rule Formulation

Definition C.1. [Subspace-Aware Moment-Orthogonalization (SUMO)] SUMO formulates the subsequent gradient update rules. Refer to

<!-- formula-not-decoded -->

with Q t ∈ R m × r and R t ∈ R r × n denoting projection matrices, T ∈ N representing the subspace update interval, η indicating the learning rate, ξ t constituting a stochastic batch, and Orthogonalization\_SVD( A ) as the operator that resolves the following through Singular Value Decomposition (SVD), as described in

<!-- formula-not-decoded -->

## D Additional Experiments

In Table 7, we evaluated SUMO and state-of-the-art memory-efficient fine-tuning methods on the MAWPS[55] dataset using the LLaMA2-7B model. We report results across two rank settings (32 and 128), comparing training time, memory usage, and task accuracy. SUMO consistently achieves superior accuracy while maintaining competitive efficiency in both memory and time (comparing to Galore).

Table 7: Fine-tuning LLaMA2-7B on MAWPS[55]

| Methods               |   Rank |   Time(h) ↓ |   Memory (GB) ↓ |   Accuracy (%) ↑ |
|-----------------------|--------|-------------|-----------------|------------------|
| LoRA                  |     32 |        0.4  |           14.36 |            45.8  |
| DoRA                  |     32 |        0.69 |           15.01 |            44.96 |
| GaLore                |     32 |        2.59 |           15.15 |            58.4  |
| SUMO (Newton-Shultz5) |     32 |        1.83 |           13.86 |            58.47 |
| SUMO (SVD)            |     32 |        1.56 |           13.86 |            61.23 |
| LoRA                  |    128 |        0.45 |           15.64 |            65.97 |
| DoRA                  |    128 |        0.72 |           16.17 |            66.81 |
| GaLore                |    128 |        2.61 |           15.79 |            64.29 |
| SUMO (Newton-Shultz5) |    128 |        1.78 |           14.12 |            64.41 |
| SUMO (SVD)            |    128 |        1.62 |           14.12 |            68.03 |

Comparison with Muon. Table 8 below complements Table 2 by comparing full fine-tuning, vanilla Muon, and our SUMO; results are reported as mean ± standard deviation when available.

Table 8: Additional comparison to Table 2: full fine-tuning vs. vanilla Muon and our SUMO (mean ± std when reported).

| Model                         | Memory   | CoLA          | STS-B         | MRPC          | RTE           | SST-2         |
|-------------------------------|----------|---------------|---------------|---------------|---------------|---------------|
| Full Fine-Tuning              | 747M     | 62.24         | 90.92         | 91.30         | 79.42         | 94.57         |
| Muon Full Fine Tuning         | 458M     | 61.19         | 90.98         | 92.14         | 80.83         | 94.71         |
| SUMO (Newton-Schulz5, rank=4) | 197M     | 61.81 ± 0.02  | 90.81 ± 0.013 | 92.43 ± 0.034 | 79.33 ± 0.031 | 94.14 ± 0.028 |
| SUMO (SVD, rank=4)            | 197M     | 62.32 ± 0.015 | 91.05 ± 0.007 | 93.48 ± 0.022 | 81.08 ± 0.019 | 94.93 ± 0.01  |
| SUMO (Newton-Schulz5, rank=8) | 198M     | 61.73 ± 0.021 | 90.77 ± 0.032 | 91.93 ± 0.04  | 79.66 ± 0.03  | 94.13 ± 0.025 |
| SUMO (SVD, rank=8)            | 198M     | 61.69 ± 0.014 | 91.11 ± 0.02  | 93.72 ± 0.018 | 81.38 ± 0.011 | 94.83 ± 0.01  |

These results show that our SUMO, achieves better performance with a significantly smaller memory footprint compared to Muon full fine-tuning approach.

Hyperparameters Grid search. To evaluate the impact of Subspace Update Frequency (K) and Ranks (r), we performed a grid search during the pretraining of the LLaMA 130M model on the C4 dataset. This specific setup allows for a direct comparison with the Galore method. Table: Perplexity results from a grid-search of Subspace Update Frequency (K) and Ranks (r) for the LLaMA 130M model pretrained on the C4 dataset. Values are presented as Galore/SUMO.

Table 9: Perplexity from a grid-search over Subspace Update Frequency ( K ) and Rank ( r ) for LLaMA-130M on C4. Values are Galore/SUMO

| Update Frequency   | Rank = 128   | Rank = 256   | Rank = 512   |
|--------------------|--------------|--------------|--------------|
| 100                | 29.7/ 28.27  | 27.9/ 26.74  | 27.4/ 26.73  |
| 250                | 28.1/ 27.86  | 26.5/ 24.87  | 26.2/ 24.82  |
| 500                | 27.2/ 25.91  | 25.6/ 24.98  | 25.3/ 24.31  |
| 1k                 | 26.8 /25.83  | 25.1 /25.42  | 24.8 /24.93  |

## D.1 Details of benchmarks in Table 6

Specifically, the pretrained models is evaluated on the following tasks:

- Perplexity: Measured on the C4 dataset [46].
- Commonsense Reasoning: BoolQ [56], RTE [57], HellaSwag (HS) [58], Winogrande (WG) [59], OpenBookQA (OBQA) [60], ARC-Easy (ARC-E), and ARC-Challenge (ARC-C) [61].
- Physical and Scientific Reasoning: PIQA [62], SciQ [63], and MathQA [64].

## D.2 Details of Fine-Tuning on GLUE

We fine-tune the pre-trained RoBERTa-Base model on the GLUE benchmark using the model provided by the Hugging Face. In Table 10 and, we detail the hyper parameters used in fine-tuning.

|                       | MNLI   | SST-2   | MRPC   | CoLA   | QNLI   | QQP   | RTE   | STS-B   |
|-----------------------|--------|---------|--------|--------|--------|-------|-------|---------|
| Batch Size            | 16     | 16      | 16     | 32     | 16     | 16    | 16    | 16      |
| # Epochs              | 30     | 30      | 30     | 30     | 30     | 30    | 30    | 30      |
| Learning Rate         | 1E-05  | 1E-05   | 3E-05  | 3E-05  | 1E-05  | 1E-05 | 1E-05 | 1E-05   |
| Rank Config.          |        |         |        | r = 4  |        |       |       |         |
| Projection back scale |        |         |        | 4      |        |       |       |         |
| Max Seq. Len.         |        |         |        | 512    |        |       |       |         |

Table 10: Hyperparameters of fine-tuning RoBERTa base for the comparison in Table 2 with respect only to rank=4.

Table 11: Hyperparameters of fine-tuning RoBERTa base for the comparison in Table 2 with respect only to rank=8.

|                       | MNLI   | SST-2   | MRPC   | CoLA   | QNLI   | QQP   | RTE   | STS-B   |
|-----------------------|--------|---------|--------|--------|--------|-------|-------|---------|
| Batch Size            | 16     | 16      | 16     | 32     | 16     | 16    | 16    | 16      |
| # Epochs              | 30     | 30      | 30     | 30     | 30     | 30    | 30    | 30      |
| Learning Rate         | 1E-05  | 2E-05   | 2E-05  | 1E-05  | 1E-05  | 2E-05 | 2E-05 | 3E-05   |
| Rank Config.          |        |         |        | r = 8  |        |       |       |         |
| Projection back scale |        |         |        | 2      |        |       |       |         |
| Max Seq. Len.         |        |         |        | 512    |        |       |       |         |

## D.3 Comparison with vanilla Muon

Following we conducted a full fine-tuning experiment with vanilla Muon to complete the comparison for Table 2 in the paper, as presented below.

Table 12: Comparison with vanilla Muon fine-tuning results.

| Model                         | Memory   | CoLA              | STS-B             | MRPC              | RTE               | SST-2             |
|-------------------------------|----------|-------------------|-------------------|-------------------|-------------------|-------------------|
| Full Fine-Tuning              | 747M     | 62.24             | 90.92             | 91.30             | 79.42             | 94.57             |
| Muon Full Fine Tuning         | 458M     | 61.19             | 90.98             | 92.14             | 80.83             | 94.71             |
| SUMO (Newton-Schulz5, rank=4) | 197M     | 61 . 81 ± 0 . 02  | 90 . 81 ± 0 . 013 | 92 . 43 ± 0 . 034 | 79 . 33 ± 0 . 031 | 94 . 14 ± 0 . 028 |
| SUMO (SVD, rank=4)            | 197M     | 62 . 32 ± 0 . 015 | 91 . 05 ± 0 . 007 | 93 . 48 ± 0 . 022 | 81 . 08 ± 0 . 019 | 94 . 93 ± 0 . 01  |
| SUMO (Newton-Schulz5, rank=8) | 198M     | 61 . 73 ± 0 . 021 | 90 . 77 ± 0 . 032 | 91 . 93 ± 0 . 04  | 79 . 66 ± 0 . 03  | 94 . 13 ± 0 . 025 |
| SUMO (SVD, rank=8)            | 198M     | 61 . 69 ± 0 . 014 | 91 . 11 ± 0 . 02  | 93 . 72 ± 0 . 018 | 81 . 38 ± 0 . 011 | 94 . 83 ± 0 . 01  |