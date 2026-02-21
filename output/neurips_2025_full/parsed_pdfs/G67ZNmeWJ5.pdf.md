## NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems

Roman Jacome † ∗ , Romario Gualdrón-Hurtado ‡ ∗ , Leon Suarez ‡ , Henry Arguello ‡

† Department of Electrical, Electronics, and Telecommunications Engineering ‡ Department of Systems Engineering and Informatics Universidad Industrial de Santander, Colombia, 680002

{rajaccar,yesid2238324,leon2238325}@correo.uis.edu.co , henarfu@uis.edu.co

## Abstract

Imaging inverse problems aim to recover high-dimensional signals from undersampled, noisy measurements, a fundamentally ill-posed task with infinite solutions in the null-space of the sensing operator. To resolve this ambiguity, prior information is typically incorporated through handcrafted regularizers or learned models that constrain the solution space. However, these priors typically ignore the task-specific structure of that null-space. In this work, we propose Non-Linear Projections of the Null-Space (NPN), a novel class of regularization that, instead of enforcing structural constraints in the image domain, promotes solutions that lie in a low-dimensional projection of the sensing matrix's null-space with a neural network. Our approach has two key advantages: (1) Interpretability: by focusing on the structure of the null-space, we design sensing-matrix-specific priors that capture information orthogonal to the signal components that are fundamentally blind to the sensing process. (2) Flexibility: NPN is adaptable to various inverse problems, compatible with existing reconstruction frameworks, and complementary to conventional image-domain priors. We provide theoretical guarantees on convergence and reconstruction accuracy when used within plug-and-play methods. Empirical results across diverse sensing matrices demonstrate that NPN priors consistently enhance reconstruction fidelity in various imaging inverse problems, such as compressive sensing, deblurring, super-resolution, computed tomography, and magnetic resonance imaging, with plug-and-play methods, unrolling networks, deep image prior, and diffusion models.

## 1 Introduction

Inverse problems involve reconstructing an unknown signal from noisy, corrupted, or undersampled observations, making the recovery process generally non-invertible and ill-posed. This work focuses on linear inverse problems of the form y = Hx ∗ + ω ∈ R m , where x ∗ ∈ R n is the target highdimensional signal, H ∈ R m × n is the sensing matrix (with m ≪ n ), y ∈ R m represents the low-dimensional measurements, and ω ∼ N (0 , σ 2 I ) is additive Gaussian noise. Numerous imaging tasks rely on this principle, including image restoration-such as deblurring, denoising, inpainting, and super-resolution (SR) [21] (structured Toeplitz sensing matrices)-as well as compressed sensing (CS) [61, 4] (dense sensing matrices) and medical imaging applications like magnetic resonance imaging (MRI) [36] (undersampled Fourier matrices) or computed tomography (CT) [59] (Radon matrices). The challenge on the recovery task lies in the ill-posed nature of the inverse problem due to the non-trivial null-space of the sensing matrix H leading to infinite solutions. Therefore, there is a need to incorporate a signal prior to the reconstruction framework. Under this idea, variational approaches formulate the signal estimator as

<!-- formula-not-decoded -->

˜

x

where g ( · ) is the data fidelity term and h ( · ) is a regularization function based on some prior of x ∗ .

* Equal contribution.

Figure 1: Geometric comparison of subspace-prior learning versus direct reconstruction in a R 3 toy example. (a) In the low-dimensional projection space, the learned mapping G ∗ ( y ) trained on points inside the unit circle, closely matches the true null-space projection Sx ∗ for both training (solid) and test (semi-transparent) inputs, whereas the direct-reconstruction estimate ˜ x 0 projected into S is significantly inaccurate. (b) In the original signal domain, the measurements Hx ∗ = y define two intersecting planes.

<!-- image -->

One of the most common priors in imaging inverse problems is sparsity, which assumes that images are compressible in a given basis [4]. Other widely used priors include low-rank structures [17] and smoothness priors [19]. Additionally, plug-and-play (PnP) priors [55], which traces back its roots from proximal algorithms [40], where these operators, usually defined by analytical models of the underlying signals [61], are replaced by a general-purpose image denoiser [51]. This approach allows the integration of classical image denoiser [37] such as BM3D [14], NLM [2], RF [16] and current deep learning (DL) denoisers [62]. The idea behind DL-based denoisers is to train a deep neural network (DNN) that maps from the noisy observation to the clean image [3, 53, 52, 24, 23]. Another learning-based approaches are based on the null-space of H by embedding the sensing operator's structure directly into learned networks. In particular, Null-Space Networks exploit the decomposition of a signal into measurement and null-space components, learning a corrective mapping over all nullspace modes to enhance interpretability and accuracy [45]. To improve robustness to measurement noise, [6] introduced separate range-space and null-space networks that denoise both components before recombination. Variants of this range-null decomposition have been applied in diffusionbased restoration [11, 57, 58], GAN-prior methods [56], algorithm-unrolling architectures [5], and self-supervised schemes [7, 10], consistently leveraging the full null-space projector to achieve high-fidelity reconstructions.

However, existing learned priors typically promote reconstructions that lie within the subspace spanned by clean training data, without explicitly accounting for the null-space of the sensing matrix H . While the data fidelity term g (˜ x ) enforces consistency with the measurements, it does not sufficiently constrain the null-space components of the solution, especially in the presence of noise, often resulting in suboptimal reconstructions. In this work, we introduce a novel class of regularization, termed Non-Linear Projections of the Null-Space (NPN), which directly promotes solutions within a low-dimensional subspace of the null-space of H that is, within the space of vectors orthogonal to the rows of H . Our method identifies a compact subspace of the null-space by selecting only the most informative directions and trains a neural network to predict their coefficients directly from measurements. By restricting corrections to this learned subspace, we concentrate regularization on unobserved features most predictive of the true signal. This subspace plays a critical role in addressing the ill-posedness of inverse problems, where conventional methods often struggle due to the lack of constraints in directions invisible to the measurement operator. To enable this, we design a projection matrix S whose rows lie in the null-space of H , constructed using either orthogonalization techniques or analytical designs depending on the structure of H . A neural network is trained to estimate this null-space projection from the measurements y , providing a non-linear prior that is both data-adaptive and model-aware. We further propose a joint optimization framework in which both the projection matrix S and the network are learned simultaneously, allowing the projection matrix to adapt during training in a task-specific manner.

Our approach offers two key advantages for imaging inverse problems: interpretability and flexibility. Interpretability : By leveraging a non-linear neural network G ∗ to learn a projection onto a lowdimensional subspace within the null-space of H , we focus on recovering signal components that lie outside the range of the sensing matrix. Learning this projection, rather than directly estimating the full signal x ∗ , reduces complexity through dimensionality reduction while maintaining a clear connection to the geometry of the inverse problem. Flexibility : the learned prior can be readily incorporated into a wide variety of reconstruction algorithms and image priors that can be adapted to other imaging inverse problems. To illustrate the interpretability advantage, Figure 1(a) shows

that the learned non-linear estimator G ∗ closely matches the true null-space projection Sx ∗ (green surface), while the projection of the direct reconstruction ˜ x 0 onto the same subspace spanned by S results in significant errors. Moreover, when evaluating out-of-distribution samples within a 2 × 2 grid in the range [2 , 4] 2 , our method experiences only a minor increase in estimation error, while the direct reconstruction approach drastically amplifies these errors. In Figure 1(b), we demonstrate that integrating our non-linear subspace prior into the inverse reconstruction process effectively regularizes the solution space. The learned subspace helps uniquely resolve the inverse problem, providing a solution close to the true signal x ∗ . Azoomed zone near the true solution is shown, where there is a small estimation error with respect to the true value; this is due to the inherent network error in the true subspace. Note that here we did not use any prior h (˜ x ) , which can reduce the estimation error. See the Appendix A.1 for more details on the setting to obtain this example. We develop a theoretical analysis showing benefits in the convergence rate when the NPN regularizer is used in PnP algorithms. The theory states that the algorithms have a significant speed-up (with respect to the non-regularizer counterpart) in convergence improvement zone (CIZ), which takes into account the inherent estimation error of the learned prior. Additionally, we showed that the NPN regularizer is minimized up to a constant depending on the error of the learned prior if the algorithm reaches optimum values. The theory is validated in a comprehensive evaluation of the method in five imaging inverse problems: CS, MRI, deblurring, SR, and CT. Although our theoretical findings are based on PnP methods, we also validated the NPN regularization in unrolling networks, deep image prior, and diffusion models.

## 2 Related work

## 2.1 Variational reconstructions methods

Variational methods solve (1) typically via proximal-gradient schemes that alternate a gradient step on the data fidelity term, g (˜ x ) = ∥ H ˜ x -y ∥ 2 2 , with a proximal step for h . A classical choice is Tikhonov regularization for smoothness, h ( x ) = ∥ Lx ∥ 2 2 , where L is a derivative operator or identity to penalize energy and ensure well-posedness [19]. Alternatively, sparsity priors use an ℓ 1 -penalty in a transform domain (e.g. wavelets) to promote compressibility of x [4]. Algorithms such as ISTA and its accelerated variant FISTA have been widely used to solve ℓ 1 -penalized reconstructions with provably faster convergence rates [1]. More recently, PnP replaces the proximal operator of h with a generic denoiser ˜ x k = D σ (˜ x k -1 -α H ⊤ ( H ˜ x k -1 -y )) , thus, leveraging sophisticated image priors without an explicit analytic penalty [55, 62]. PnP with both classical denoisers and deep-learning models achieves state-of-the-art results, yet it leaves the null-space of H uncontrolled: any component in the null-space of H may be arbitrarily modified by the denoiser.

## 2.2 Null-space learned reconstruction

Harnessing the sensing model into a learning-based reconstruction network enables more accurate reconstructions [45]. Particularly, null-space networks (NSN) [45] harness the range-null-space decomposition (RNSD), which states that a vector x ∈ R n is decomposed as x = P r ( x ) + P n ( x ) where P r ( x ) = H † Hx , with H † denoting the Moore-Penrose pseudoinverse, is the projection onto the range space of H and P n ( x ) = ( I -H † H ) x is the null-space projection operator. NSN exploits this structure by learning a neural network R : R n → R n such that the reconstruction becomes ˆ x = H † y +( I -H † H )R( H † y ) . This approach benefits the interpretability of the reconstruction. However, this method does not take into account the inherent noise of the measurements; thus, [6] introduces deep decomposition networks (DDN), a more robust formulation is presented where two models R r and R n are employed to improve recovery performance. Consider two architectures, DDNindependent (DDN-I) ˆ x = H † y + P r (R r ( H † y )) + P n (R n ( H † y )) and DDN-Cascade (DDN-C) ˆ x = H † y +P r (R r ( H † y )) + P n (R n ( H † y +P r (R r ( H † y )))) . The range-null-space decomposition has also been used to enhance data fidelity in diffusion models for image restoration [11, 57, 58], generative adversarial network priors [56], unfolding networks [5], and self-supervised learning [7, 10]. In these works, the sensing matrix H structure is incorporated into the reconstruction network through a learned-based RNSD that allows high-fidelity reconstructions. Different from these approaches, our method does not apply the full null-space projection operator. Instead, we first identify a compact subspace of Null( H ) by selecting a projection matrix S ∈ R p × n whose rows span the most informative null-space directions. We then train a network G ∗ : R m → R p to predict the

coefficients y s = Sx directly from the measurements. On the other hand, the learned NSN [45] is optimized to improve the recovery performance of a determined regularized inverse problem (i.e., Tikhonov-based solutions), which makes it algorithm-specific and does not work as a plug-in for other recovery methods. In our case, since we optimize the network offline, only with the knowledge of H and S , it can be easily integrated in a wide range of recovery methods.

## 3 Method

In our approach, we design learned priors promoting solutions in a low-dimensional subspace of the null-space of H . First, let's define some useful properties.

Definition 1 (Null-Space) . The null-space of a matrix H is defined as

<!-- formula-not-decoded -->

Thus, we consider a projection matrix S ∈ R p × n , with p ≤ ( n -m ) , with rows orthogonal to H rows, implying that s i ⊥ h j ∀ i ∈ { 1 , . . . , p } , ∀ j ∈ { 1 , . . . , m } . Based on Def. 1, s i ∈ Null( H ) meaning that any projection y s = Sx lies onto a low-dimensional subspace of the null-space of H . Based on this observation, we propose to learn a data-driven prior G( · ) restricted to the low-dimensional null-space of H , such that G( y ) ≡ Sx ∗ . Specifically, we select a projection matrix S ∈ R p × n whose rows span a subspace of Null( H ) . Consequently, we solve

<!-- formula-not-decoded -->

and G ∗ : R m → R p is a neural network trained to map the measurements y into the low-dimensional subspace Sx ∗ , γ is a regularization parameter, the regularizer ϕ aims to promote solutions on the learned manifold in the null-space of H . Our framework introduces a novel regularization strategy that embeds data-driven models into inverse-problem solvers by constraining solutions to the nonlinear low-dimensional manifold induced by G ∗ , in contrast to existing learned priors that restrict reconstructions to the range of a pre-trained restoration or denoising network [52, 24, 29]. Note that we used the Euclidean norm in the regularizer; however, since the network G ∗ has some error with respect to Sx ∗ , one could use a more robust function such as the Huber loss or a weighted norm. Nevertheless, in our experiments, the Euclidean norm works well by adjusting the hyperparameter γ . One interpretation of the proposed regularization is that it improves the data-fidelity term g (˜ x ) as it promotes low-dimensional projections of blind signal features to H . Our approach is closely related to NSN-based methods [6, 45, 58], as those methods aim to regularize deep learning-based recovery networks, harnessing the RNSD. In our case, making analogy with these models, we can view the reconstruction in (2) as ˜ x = H † y + γ S † G( y ) + prior h,λ where the projection onto the null-space P n ( · ) is replaced by the range-space S which promotes solution lying in the p ≤ ( n -m ) most informative null-space modes instead of the entire null-space operator.

## 3.1 Design of the matrix S

To design the matrix S , it is necessary to analyze the structure of H depending on the inverse problem. Below are insights based on the sensing matrix structure for exploiting the null-space in our prior.

Compressed Sensing (CS): In CS, the sensing matrix H ∈ R m × n is typically dense and randomly generated. Previous approaches [20, 38] use the remaining ( n -m ) rows of a full-rank Hadamard, Gaussian, or Bernoulli matrix as S . Due to the lack of inherent structure in such matrices, we adopt an orthogonalization strategy based on the classical QR decomposition for designing the matrix S (see Alg. 1 in Appendix A.2).

Magnetic Resonance Imaging (MRI): In MRI, the forward operator H corresponds to a discrete 2D discrete Fourier transform (DFT) undersampled, where only a subset of frequency components (k-space lines) is acquired during the scan. Specifically, let F = { f ⊤ 1 , . . . , f ⊤ n } denote the full set of n orthonormal rows of the 2D DFT matrix. The sensing matrix H then consists of m&lt;n selected rows, i.e., { h ⊤ 1 , . . . , h ⊤ m } := F h ⊂ F . These rows define the measurements taken in the Fourier domain. To construct the null-space projection matrix S , we exploit the fact that the remaining rows in F -those not used in H -span the null-space of the sampling operator. Thus, we define { s ⊤ 1 , . . . , s ⊤ p } := F c h , where F c h = F \ F h is the complement of the sampled frequencies. Because the DFT matrix is orthonormal, these complementary rows are guaranteed to be orthogonal to the measurement space and form a natural basis for the null-space of H .

Computed Tomography (CT): In parallel-beam limited-angle CT, the forward operator samples the Radon transform only at a subset of projection angles. Let Θ be the full discrete angle set, Θ h ⊂ Θ the acquired angles, R is the discrete Radon transform matrix with rows indexed by Θ , and H = P Θ h R the forward operator; we define S directly as the complement of the acquired angles, S = P Θ c h R with Θ c h = Θ \ Θ h , so S stacks the rows of R corresponding to the non-acquired angles.

Structured Toeplitz matrices (Deblurring and Super-Resolution): The forward model H is built upon a Toeplitz matrix based on the convolution kernel denoted as H [ i, i + j ] = h [ j ] with i = 1 , . . . , m and j = 1 , . . . , n . The action of H corresponds to a linear filtering process, attenuating high-frequency components. From a frequency point of view, the matrix S should block the low frequencies sampled by H . Thus, we design the matrix S as S [ i, i + j ] = 1 -h [ j ] with i = 1 , . . . , m and j = 1 , . . . , n. . In super-resolution, the sensing matrix is H = DB where B ∈ R n × n is a convolution matrix build with a low-pass filter b and D ∈ R m × n is a decimation matrix denoting √ n m the super-resolution factor (SRF). We construct the matrix S similarly to the deblurring case, i.e., S [ i, i + j ] = 1 -b [ j ] with i = 1 , . . . , m and j = 1 , . . . , n.

## 3.2 Learning the NPN Prior

Given the design of the projection matrix S , which spans a structured low-dimensional subspace orthogonal to the measurement operator H , we train the network G to estimate the null-space component x ∗ of the signal. To further improve the representation power of the NPN prior, we propose to jointly optimize the neural network G and the projection matrix S . While the initial design of S ensures that its rows lie in the null-space of the measurement operator H , optimizing S during training allows the model to discover a task-adaptive subspace that best complements the measurements. This formulation still preserves the orthogonality between H and S , while improving the quality of the previously learned. The optimization objective is

<!-- formula-not-decoded -->

where A = [ H ⊤ , ˜ S ⊤ ] ⊤ , λ 1 , λ 2 &gt; 0 control the trade-off between estimation accuracy and orthogonality enforcement. The first term trains G to predict the projection Sx from the measurements y , following the MMSE objective. The second term enforces near-orthogonality between the row spaces of S and H by pushing S ⊤ S + H ⊤ H ≈ I . The third term promotes full-rank behavior and numerical stability of the combined system matrix A , preventing collapse or redundancy in the learned subspace. Importantly, this formulation enables end-to-end learning of a null-space-aware regularizer, where the matrix S is initialized using principled designs (e.g., QR orthogonalization or frequency complements in MRI), but is then refined during training to maximize consistency with the true signal statistics and the learned estimator G . This formulation has great benefits in cases of non-structured or dense matrices, such as those in CS; however, for well-defined matrices, such as Fourier-based or Toeplitz matrices, finding orthogonal complements has a straightforward analytical solution, and it is not required to jointly optimize it.

## 4 Theoretical analysis

We analyze how the convergence property of a PnP algorithm is affected under this new regularization. Without loss of generalization, we focus on one of the most common PnP approaches, which is based on proximal gradient methods. The iterations are given by

<!-- formula-not-decoded -->

where D σ ( · ) is the denoiser operator and σ is an hyperparameter modeling the noise variance and ℓ = 1 , . . . , L with maximum number of iterations L .

Based on this formulation, we analyzed two main aspects: (i) the convergence rate of the algorithm, and (ii) the convergence behavior of the regularization function ϕ ( ˜ x ℓ +1 ) = ∥ G ∗ ( y ) -S˜ x ℓ +1 ∥ 2 2 . Our theoretical developments leverage the restricted isometry property defined over a specific Riemannian manifold M D , induced by the image-space of the denoiser D σ . For the denoiser to exhibit isometriclike properties, certain criteria must be satisfied, including boundedness, Lipschitz continuity, and a low-rank Jacobian. These properties can be guaranteed through spectral normalization during denoiser training [43]. Such assumptions are commonly employed in the convergence analysis of

iterative projection methods [46] and have been suitably adapted for PnP convergence analyses [31]. Additionally, we introduce assumptions regarding the estimation error of the model G ∗ , assuming a Gaussian error distribution. Furthermore, we define a CIZ in the algorithm iterations based on the estimation error norm. For the guarantees, we consider the noiseless case, i.e., ω = 0 .

Definition 2 (Restricted Isometry Property [46]) . Let M D ⊂ R n be a Riemannian manifold given by the denoiser's image space Im( D σ ) , thus S ∈ R p × n satisfies the restricted isometry property with respect to M D with a restricted isometry constant (RIC) ∆ S M D ∈ [0 , 1) if for all x , z ∈ M D with

Assumption 1 (Prior mismatch) . The trained model G ∗ using (3) , we assume that

<!-- formula-not-decoded -->

Ground truth value

<!-- formula-not-decoded -->

Thus, considering that the nonlinear operator N is K -Lipschitz continuous, and H satisfies Definition 2 with a constant ∆ H M D , thus we have ∥ N( Hx ∗ ) ∥ 2 2 ≤ K (1 + ∆ H M D ) ∥ x ∗ ∥ 2 2 .

<!-- formula-not-decoded -->

Definition 3 (Convergence Improvement Zone (CIZ) by ϕ ( x ℓ ) ) . We define a zone where the proposed NPN prior improves the convergence of the PnP algorithm. For that, S satisfies the RIP with RIC ∆ S M D ∈ [0 , 1) , the CIZ by ϕ are the iterations L ϕ = { 1 , . . . , L ϕ } where L ϕ ≤ L such that the network estimation error N( Hx ) for all ℓ ∈ L ϕ satisfies that

Assumption 2 (Bounded denoiser) . We consider that for x , z ∈ R n , D σ is a bounded denoiser, with a constant δ &gt; 0 , if

<!-- formula-not-decoded -->

We are now equipped to develop the first theoretical benefit of the proposed method.

Theorem 1 (PnP-NPN Convergence) . Consider the fidelity term g (˜ x ) = ∥ y -H˜ x ∥ 2 2 , and assume the denoiser D σ satisfies Assumption 2. Let the matrix S be constructed according to (3) and satisfy the RIP condition (Definition 2) with constant ∆ S M D ∈ [0 , 1) . Then, for a finite number of iterations ℓ = 1 , . . . , L ϕ within the CIZ, the residual ∥ x ℓ +1 -x ∗ ∥ 2 2 decays linearly with rate

<!-- formula-not-decoded -->

The proof of the theorem can be found in Appendix A.3. A key insight from Theorem 1 is the role of the CIZ L ϕ , which characterizes the subset of iterations where the proposed regularizer outperforms provides improved convergence. Specifically, due to the inherent mismatch between the ground-truth projection Sx ∗ and its learned estimate G ∗ ( Hx ∗ ) , the NPN prior is only effective while the projected estimate S ˜ x ℓ remains closer to Sx ∗ than the residual error N( Hx ∗ ) . Outside this zone, the regularizer may no longer provide beneficial guidance to the reconstruction. Nevertheless, thanks to the design of the matrix S either through analytical construction or data-driven optimization, it is guaranteed to be orthogonal to the rows of the sensing matrix H . This orthogonality ensures that the operator norm ∥ ∥ I -α ( H ⊤ H + S ⊤ S ) ∥ ∥ 2 2 remains small for an appropriate step size α . Furthermore, when the spectral norm of S and its restricted isometry constant ∆ S M D are sufficiently low, the acceleration factor ρ falls below one. This guarantees that the PnP-NPN algorithm will converge to a fixed point within the zone L ϕ , thereby validating the theoretical and practical benefits of incorporating the NPN prior into the reconstruction framework.

The second analysis of the proposed approach is the convergence of the regularization.

Theorem 2 (Convergence of NPN Regularization) . Consider the iterations of the PnP-NPN algorithm defined in (4) for ℓ = 1 , . . . , L . Assume that the estimation error of the trained network G ∗ satisfies Assumption 1, and that both S and H satisfy the Restricted Isometry Property (RIP) over the manifold M D with constants ∆ S M D , ∆ H M D ∈ [0 , 1) . Further assume that the residual term N( Hx ∗ ) is K -Lipschitz continuous, thus ∥ N( Hx ∗ ) ∥ 2 2 ≤ K (1 + ∆ H M D ) ∥ x ∗ ∥ 2 2 . Then, after ℓ iterations, the NPN regularization term satisfies the following upper bound:

<!-- formula-not-decoded -->

The proof of Theorem 2 is provided in the technical Appendix A.4. This result shows that the regularization value ϕ ( x ℓ +1 ) decreases as the reconstruction error ∥ x ∗ -x ℓ ∥ 2 2 diminishes over the course of the iterations. From Theorem 1, we know that ∥ x ∗ -x ℓ ∥ 2 2 is reduced, thereby ensuring a monotonic decrease in the regularization term. Moreover, since Theorem 1 guarantees convergence of the sequence { x ℓ } to a fixed point, the difference ∥ x ℓ -x ℓ +1 ∥ 2 2 asymptotically vanishes as ℓ → ∞ . Therefore, in the limit, the regularization value is bounded above by a residual term determined by the Lipschitz constant K , which can be minimized by including spectral normalization in G , the norm of the ground truth signal ∥ x ∗ ∥ 2 2 , and the RIP constant of the sensing matrix: lim ℓ →∞ ϕ ( x ℓ +1 ) ≤ K ∥ x ∗ ∥ 2 2 (1 + ∆ H M D ) . This bound quantifies the asymptotic regularization performance of the NPN prior in terms of the approximation quality of the learned model and the sensing operator geometry.

Our theoretical analysis has focused on integrating the proposed NPN regularizer into projectedgradient-descent-based PnP methods. However, the same regularization strategy can be applied to a broad class of learning-based solvers. For example, in algorithm-unrolling architectures [39], deep equilibrium models [18], and learning-to-optimize frameworks [8], one can insert the NPN proximal step alongside the usual network updates, with end-to-end training of all learnable parameters [44, 9]. Although we demonstrate the empirical benefits of NPN within an unrolled network in Section 5, a full theoretical treatment of these extensions is left for future work. We also incorporate the NPN regularization in Deep Image Prior (DIP) [54] framework (see Appendix A.7 for more details) and in two diffusion-based solvers [15], diffusion posterior sampling (DPS) [12], and DiffPIR [65] (see Appendix A.8 for more details).

## 5 Experiments

The proposed NPN regularization was evaluated in five imaging inverse problems: compressed sensing, super-resolution, computed tomography, single coil MRI, and deblurring. The method was implemented using the PyTorch framework. For the recovery algorithm, we adopt the FISTA solver [1], with a deep denoiser [25], regularization by denoising (RED) [41], and sparsity prior [1], see Appendix A.5. All simulations were performed on an NVIDIA RTX 4090 GPU, with code in 1 .

Compressed Sensing: The single-pixel camera (SPC) is used along with the CIFAR-10 dataset [28], with 50 , 000 images for training and 10 , 000 for testing. All images were resized to 32 × 32 . The Adam [26] optimizer was used with a learning rate of 5 × 10 -4 . H is a random binary sensing matrix with m/n = 0 . 1 . S is initialized by QR decomposition, with Algorithm 1 in Appendix A.2. Then, G (for which we used a ConvNeXt [33]) and S are optimized following Eq. (3).

MRI: We employed the fastMRI knee single-coil MRI dataset [27], which consists of 900 training images and 73 test images of knee MRIs of 320 × 320 . The training set was split into 810 images for training and 90 for validation, and all images were resized to 256 × 256 . For G we used a U-Net based architecture [42] and trained it for 60 epochs with a learning rate of 1 × 10 -4 , using the AdamW optimizer [35] with a weight decay of 1 × 10 -2 and a batch size of 4. Performance was evaluated at acceleration factors ( AF = n m ) of 4, 8, and 16, using a Radial undersampling mask [32].

Deblurring and Super-Resolution: For the deblurring inverse problem, we used a 2-D Gaussian kernel with a variance σ . For these experiments, we used the CelebA [34] dataset resized to 128 × 128 , using 8000 images for training and 2000 for testing. Here, we employed a U-Net architecture for the network G . In this case, we used the Adam optimizer with a learning rate of 1 × 10 -3 and a batch size of 32. The results for the super-resolution case are shown in Appendix A.10.

Computed Tomography: We evaluated the proposed regularization on the limited-angle CT inverse problem, using 60 out of 180 total views (spaced every 1°) under a parallel-beam geometry. We train the DM for 1000 epochs with batch size 4 using the AdamW optimizer and learning rate 3 × 10 -4 . We used a cosine variance schedule ranging from β 1 = 1 × 10 -4 to β 1 = 0 . 02 with T = 1000 time steps. The LoDoPaB-CT dataset [30] was resized to 256 × 256 and used for training; in testing, we used 10 test set slices. G is a U-Net which was trained for 100 epochs with a learning rate 3 × 10 -4 and a batch size of 4 using AdamW.

1 github.com/yromariogh/NPN

Figure 2: PnP-FISTA convergence analysis in CS. (a) Reconstruction error. (b) Null-space prediction error for (red) Initialization ˜ S = QR( H ) from Algorithm 1, and (blue) Designed S with Eq. (3) and m/n = p/n = 0 . 1 . In this case, the CIZ from Definition 3 is highlighted in light red and light blue. (c) Acceleration ratio of signal convergence; here, the CIZ is defined as the empirical convergence ratio of the proposed solution that is lower than the baseline (black).

<!-- image -->

Iterations I terations Figure 3: Convergence curves for σ ∈ { 2 , 5 , 10 } in deblurring, AF ∈ { 4 , 8 , 12 } in MRI.

<!-- image -->

## 5.1 Convergence Analysis

To confirm numerically the theoretical results obtained in Sec. 4, we show Fig. 2 with 3 general cases: Baseline (black), where only H is used; PnP-NPN with designed S (blue), where G and S are jointly trained following Eq. (3), with λ 1 = 0 . 001 , λ 2 = 0 . 01 ; PnP-NPN with QR (red) where the projection matrix ˜ S = QR( H ) is fixed (for details about QR( · ) see Algorithm 1 in Appendix A.2), only G is trained, and λ 1 = λ 2 = 0 . In Fig. 2(a), we plot the error E ℓ = ∥ x ℓ -x ∗ ∥ 2 2 , showing that both NPN-QR (red) and NPN-designed (blue) decay much faster than the baseline (black) over the first ∼ 75 iterations. Fig. 2(c) tracks the projection onto null-space ∥ Sx ∗ -Sx ℓ ∥ 2 2 , which steadily decreases where ∥ N( Hx ∗ ) ∥ 2 2 ≤ ∥ S ( x ℓ -x ∗ ) ∥ 2 2 , whereas the baseline's reprojection error (dashed black) remains high where. The CIZ is defined as in Def. 3. Fig. 2(c) shows the per-step acceleration ratio ∥ x ℓ +1 -x ∗ ∥ 2 2 / ∥ x ℓ -x ∗ ∥ 2 2 , where both NPN curves dip well below the baseline. In this scenario, the CIZ is defined as the iterations when the convergence ratio of the proposed method is lower than that of the baseline, with the designed S achieving the smallest R ℓ around ℓ ≈ 50 -75 , confirming stronger per-iteration error reduction. The results show that empirical improvements in the algorithm convergence are predictable with the CIZ validating Theorem 1. Figure 3 shows convergence plots in PSNR for MRI with AF = { 4 , 8 , 12 } , and for deblurring and σ = { 2 , 5 , 10 } , where NPN regularization consistently yields higher reconstruction quality and faster convergence compared to the baseline. Additional results with other state-of-the-art denoisers in PnP-ADMM are shown in Appendix A.11 for the image deblurring task. For the CS scenario, the selection p and the joint optimization in Eq. (3) is fundamental, see Appendices A.11 and A.12 for detailed analysis.

## 5.2 Visual results

Figure 4 presents reconstruction results for MRI with an AF = 4 , CT with an acquisition of 30 views of a total of 180, and for deblurring with σ = 2 using PnP with the DnCNN prior [63]. The estimate ˆ x 0 is obtained via equation (1), while the estimate ˆ x is obtained with NPN regularization through equation (2). Results show that the learned prior effectively approximates the true nullspace G( y ) ≈ Sx ∗ with an estimation PSNR of 28.11 dB in deblurring, 39.13 dB for MRI. Moreover, the reconstruction ˆ x obtained with NPN regularization preserves high-frequency details and provides overall improved performance than ˆ x 0 .

Figure 4: Deblurring and MRI reconstruction results for PnP and PnP-NPN using a DnCNN prior, with 5 dB SNR measurement noise. The measurements are denoted by y , the nonlinear approximation of the null-space of H by ˆ y s , and the true null-space by y ∗ s . The estimation ˆ x 0 is obtained with (1), and ˆ x is obtained with the proposed regularization with (2). The ground truth signal is x ∗ .

<!-- image -->

Table 1: State-of-the-art comparison for CS, MRI, and Deblurring. For each task, the best results are highlighted in bold teal , while the second-best results are shown in underline orange .

| Inverse Problem      | Baseline   | Baseline   | Baseline   | NPN      | NPN   | NPN   | NSN       | NSN       | NSN       |
|----------------------|------------|------------|------------|----------|-------|-------|-----------|-----------|-----------|
| Inverse Problem      | Sparsity   | PnP        | RED        | Sparsity | PnP   | RED   | DNSN [45] | DDN-C [6] | DDN-I [6] |
| CS ( γ = 0 . 1 )     | 15.93      | 20.04      | 17.45      | 16.15    | 21.12 | 17.53 | 20.10     | 20.03     | 20.7      |
| MRI (AF = 4)         | 36.86      | 35.99      | 36.00      | 38.16    | 38.08 | 38.07 | 35.2      | 33.7      | 33.2      |
| Deblurring ( σ = 2 ) | 29.27      | 30.78      | 32.84      | 31.77    | 31.42 | 33.67 | 33.07     | 33.03     | 32.70     |

## 5.3 State-of-the-art comparison

We compared our approach with other state-of-the-art recovery methods that exploit the null-space structure. Mainly, we compare with the DNSN [45] and the DDN-I and DDN-C [6] (For more details on the implementation of these methods, refer to Appendix A.9). The results for CS, MRI, and Deblurring are in Table 1, showing consistent improvement or competitive performance of the proposed method. In Appendix A.14, we show a comparison where the same neural network as G , instead of estimating Sx it estimates directly x , and incorporates it as a regularizer in a PnP-ADMM algorithm, where we show that the proposed method achieves superior performance.

Table 2: Dataset generalization results for SPC in PnP and Unrolling. Each S , G ∗ , and Unrolling were optimized with CIFAR-10, and tested with the CIFAR-10 and STL10. For each dataset, the best results are highlighted in bold teal , while the second-best results are shown in underline orange .

| Method   | p/n   | CIFAR-10   | CIFAR-10   | STL10   | STL10     |
|----------|-------|------------|------------|---------|-----------|
| Method   | p/n   | PnP        | Unrolling  | PnP     | Unrolling |
| Baseline | 0.0   | 20.04      | 24.32      | 20.09   | 18.35     |
| NPN      | 0.1   | 21.12      | 28.53      | 19.91   | 19.64     |
| NPN      | 0.3   | 21.07      | 28.75      | 21.14   | 20.23     |
| NPN      | 0.5   | 20.78      | 27.64      | 20.77   | 18.76     |
| NPN      | 0.7   | 20.09      | 26.73      | 20.31   | 18.45     |
| NPN      | 0.9   | 20.41      | 29.90      | 21.02   | 19.48     |

## 5.5 Deep Image Prior

We consider the deblurring task with a kernel bandwidth of σ = 4 . 0 . We train G with the Places365 dataset [64], where we used 28.000 images for training and 7000 images for testing. All images were resized to 128 × 128 . The network K was trained following (15) using the Adam optimizer with a learning rate of 1 e -3 for 1000 iterations. The network K is a U-Net of the same size as G . In Fig. 5 is shown the reconstruction performance of DIP and NPN-DIP for different values of γ . The results show significant improvements of up to 5 dB and convergence improvements.

## 5.4 Performance in data-driven models and dataset generalization

In Table 2 we report PSNR (dB) for both PnP and unrolling solvers on CIFAR-10 [28] (in-distribution) and STL10 [13] (out-of-distribution) across projection ratios p/n . PnP achieves a 1 dB boost at p/n = 0 . 1 and sustains 1 dB gains on STL10, peaking when p/n = 0 . 3 . Unrolling delivers up to 5.6 dB in-distribution improvement but only 1.9 dB crossdataset gain at p/n = 0 . 3 before declining. Overall, PnP ensures stable generalization, while unrolling maximizes peak PSNR; p/n ≈ 0 . 3 provides the best balance between accuracy and robustness. In general, NPN improves both PnP and unrolling reconstruction performance regarding dataset changes.

## 5.6 Diffusion model solvers

We integrated the proposed regularization term into two widely adopted DM frameworks, DPS [12] and DiffPIR [65]. Table 3 shows the obtained results for different values of γ . For DPS, the NPN regularization consistently improves reconstruction performance by up to 1.85 dB. For DiffPIR, it yields improvements of up to 0.61 dB. Figure 6 shows reconstruction results using NPN within DPS and DiffPIR; yellow arrows highlight improved image details with NPN. Details on the implementation of NPN into DPS and DiffPIR are provided in the Appendix A.8.

Figure 5: Performance of DIP and NPN-DIP for different values of γ in Deblurring.

<!-- image -->

## 6 Limitations

While the proposed method introduces some additional complexity, such as training a dedicated neural network for each sensing configuration ( H , S ) , the networks are lightweight and tailored to specific inverse problems, keeping computational demands modest (see Appendices A.5 and A.14 for details). The current integration into learning-based reconstruction frameworks (e.g., unrolling networks) involves a two-stage training process, first for the NPN regularizer, then for the reconstruction model, but this modular design enables flexible adaptation. Future work could explore joint end-to-end training of the NPN regularizer and reconstruction network to further enhance performance and efficiency. Unlike NSN-based methods [45, 6], which directly reconstruct, our approach learns a subspace projection offline, adding an extra training step but offering improved compatibility with different solvers. Although we devise some design criteria for selection S , there are scenarios in which the method can fail upon this design, we provide a detailed discussion in A.13 on this aspect from the point of view of finding non-linear relations between Hx and Sx .

## 7 Conclusion and future outlooks

We introduce Non-Linear Projections of the Null-Space for regularizing imaging inverse problems. Intuitively, the regularization promotes selective coefficients of the signal in the null-space of the sensing matrix. This formulation allows flexibility in what features of the null-space we can exploit. Our proposed method is equipped with strong theoretical guarantees for plug-and-play restoration algorithms, showing that the proposed regularization has a zone of convergence improvement controlled by the network error. Additionally, we show that our regularizer converges to a constant depending also on the network estimation error when the algorithm reaches the optimum. We validate our theoretical findings in five distinctive imaging inverse problems: compressed sensing, magnetic resonance, super-resolution, computed tomography, and deblurring. Results validate the theoretical developments, and we have improved performance with state-of-the-art methods. This approach opens new frontiers to regularize the imaging inverse problems in different solvers, such as deep equilibrium models and consensus equilibrium formulations.

Table 3: Ablation over γ for two methods. Best results are bold teal ; secondbest are underline orange .

| γ          | NPN-DPS   | NPN-DiffPIR   |
|------------|-----------|---------------|
| 0.0 (Base) | 28 . 22   | 31 . 30       |
| 10 - 5     | 28 . 55   | 31.88         |
| 10 - 4     | 28 . 30   | 31.91         |
| 0.1        | 30.06     | 28 . 98       |
| 0.2        | 30.07     | 28 . 57       |

Figure 6: Reconstruction results using NPN within DPS and DiffPIR.

<!-- image -->

## Acknowledgements

This work was supported in part by the Agencia Nacional de Hidrocarburos (ANH) and the Ministerio de Ciencia, Tecnología e Innovación (MINCIENCIAS), under contract 045-2025, and in part by the Army Research Office/Laboratory under grant number W911NF-25-1-0165, VIE from UIS project 8087. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government.

## References

- [1] Amir Beck and Marc Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1):183-202, 2009.
- [2] Antoni Buades, Bartomeu Coll, and Jean-Michel Morel. Non-Local Means Denoising. Image Processing On Line, 1:208-212, 2011.
- [3] Harold C Burger, Christian J Schuler, and Stefan Harmeling. Image denoising: Can plain neural networks compete with bm3d? In 2012 IEEE conference on computer vision and pattern recognition, pages 2392-2399. IEEE, 2012.
- [4] E. J. Candes and M. B. Wakin. An introduction to compressive sampling. IEEE Signal Processing Magazine, 25(2):21-30, 2008.
- [5] Bin Chen, Jiechong Song, Jingfen Xie, and Jian Zhang. Deep physics-guided unrolling generalization for compressed sensing. International Journal of Computer Vision, 131(11):28642887, 2023.
- [6] Dongdong Chen and Mike E Davies. Deep decomposition learning for inverse imaging problems. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXVIII 16, pages 510-526. Springer, 2020.
- [7] Dongdong Chen, Julián Tachella, and Mike E Davies. Equivariant imaging: Learning beyond the range space. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4379-4388, 2021.
- [8] Tianlong Chen, Xiaohan Chen, Wuyang Chen, Howard Heaton, Jialin Liu, Zhangyang Wang, and Wotao Yin. Learning to optimize: A primer and a benchmark. Journal of Machine Learning Research, 23(189):1-59, 2022.
- [9] Xiaohan Chen, Jialin Liu, Zhangyang Wang, and Wotao Yin. Theoretical linear convergence of unfolded ista and its practical weights and thresholds. Advances in Neural Information Processing Systems, 31, 2018.
- [10] Yurong Chen, Yaonan Wang, and Hui Zhang. Unsupervised range-nullspace learning prior for multispectral images reconstruction. IEEE Transactions on Image Processing, 2025.
- [11] Xinhua Cheng, Nan Zhang, Jiwen Yu, Yinhuai Wang, Ge Li, and Jian Zhang. Null-space diffusion sampling for zero-shot point cloud completion. In IJCAI, pages 618-626, 2023.
- [12] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022.
- [13] Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 215-223. JMLR Workshop and Conference Proceedings, 2011.
- [14] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising with block-matching and 3d filtering. In Image processing: algorithms and systems, neural networks, and machine learning, volume 6064, pages 354-365. SPIE, 2006.

- [15] Giannis Daras, Hyungjin Chung, Chieh-Hsin Lai, Yuki Mitsufuji, Jong Chul Ye, Peyman Milanfar, Alexandros G Dimakis, and Mauricio Delbracio. A survey on diffusion models for inverse problems. arXiv preprint arXiv:2410.00083, 2024.
- [16] Eduardo S. L. Gastal and Manuel M. Oliveira. Domain transform for edge-aware image and video processing. In ACM SIGGRAPH 2011 Papers, SIGGRAPH '11, New York, NY, USA, 2011. Association for Computing Machinery.
- [17] Tatiana Gelvez, Hoover Rueda, and Henry Arguello. Joint sparse and low rank recovery algorithm for compressive hyperspectral imaging. Appl. Opt., 56(24):6785-6795, Aug 2017.
- [18] Davis Gilton, Gregory Ongie, and Rebecca Willett. Deep equilibrium architectures for inverse problems in imaging. IEEE Transactions on Computational Imaging, 7:1123-1133, 2021.
- [19] Gene H Golub, Per Christian Hansen, and Dianne P O'Leary. Tikhonov regularization and total least squares. SIAM journal on matrix analysis and applications, 21(1):185-194, 1999.
- [20] Romario Gualdrón-Hurtado, Roman Jacome, Leon Suarez, Emmanuel Martinez, and Henry Arguello. Improving compressive imaging recovery via measurement augmentation. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1-5, 2025.
- [21] Bahadir Gunturk and Xin Li. Image restoration. CRC Press, 2018.
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 6840-6851. Curran Associates, Inc., 2020.
- [23] Yuyang Hu, Mauricio Delbracio, Peyman Milanfar, and Ulugbek S Kamilov. A restoration network as an implicit prior. arXiv preprint arXiv:2310.01391, 2023.
- [24] Yuyang Hu, Albert Peng, Weijie Gan, Peyman Milanfar, Mauricio Delbracio, and Ulugbek S Kamilov. Stochastic deep restoration priors for imaging inverse problems. arXiv preprint arXiv:2410.02057, 2024.
- [25] Kamilov et al. Plug-and-play methods for integrating physical and learned models in computational imaging: Theory, algorithms, and applications. IEEE Sig. Proc. Mag., 40(1):85-97, 2023.
- [26] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [27] Florian Knoll, Jure Zbontar, Anuroop Sriram, Matthew J Muckley, Mary Bruno, Aaron Defazio, Marc Parente, Krzysztof J Geras, Joe Katsnelson, Hersh Chandarana, Zizhao Zhang, Michal Drozdzalv, Adriana Romero, Michael Rabbat, Pascal Vincent, James Pinkerton, Duo Wang, Nafissa Yakubova, Erich Owens, C Lawrence Zitnick, Michael P Recht, Daniel K Sodickson, and Yvonne W Lui. FastMRI: A publicly available raw k-space and DICOM dataset of knee images for accelerated MR image reconstruction using machine learning. Radiol. Artif. Intell., 2(1):e190007, January 2020.
- [28] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [29] Joseph Kuo, Jason Granstedt, Umberto Villa, and Mark A Anastasio. Learning a projection operator onto the null space of a linear imaging operator. In Medical Imaging 2021: Physics of Medical Imaging, volume 11595, pages 1019-1025. SPIE, 2021.
- [30] Johannes Leuschner, Maximilian Schmidt, Daniel Otero Baguer, and Peter Maass. LoDoPaBCT, a benchmark dataset for low-dose computed tomography reconstruction. Sci. Data, 8(1):109, April 2021.
- [31] Jiaming Liu, Salman Asif, Brendt Wohlberg, and Ulugbek Kamilov. Recovery analysis for plug-and-play priors using the restricted eigenvalue condition. Advances in Neural Information Processing Systems, 34:5921-5933, 2021.

- [32] Jing Liu and David Saloner. Accelerated MRI with CIRcular cartesian UnderSampling (CIRCUS): a variable density cartesian sampling strategy for compressed sensing and parallel imaging. Quant. Imaging Med. Surg., 4(1):57-67, February 2014.
- [33] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976-11986, 2022.
- [34] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.
- [35] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.
- [36] Michael Lustig, David L Donoho, Juan M Santos, and John M Pauly. Compressed sensing mri. IEEE signal processing magazine, 25(2):72-82, 2008.
- [37] Lei Ma, Hulin Kuang, Jin Liu, Chengchao Shen, and Jianxin Wang. Improving medical image denoising via a lightweight plug-and-play module. In 2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pages 1350-1355. IEEE, 2023.
- [38] Emmanuel Martinez, Roman Jacome, Romario Gualdrón-Hurtado, Iñaki Esnaola, and Henry Arguello. Compressive sensing with augmented measurements via generative self-distillation. In 2025 IEEE Statistical Signal Processing Workshop (SSP), pages 31-35, 2025.
- [39] Vishal Monga, Yuelong Li, and Yonina C Eldar. Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing. IEEE Signal Processing Magazine, 38(2):18-44, 2021.
- [40] Neal Parikh and Stephen Boyd. Proximal algorithms. Foundations and Trends in optimization, 1(3):127-239, 2014.
- [41] Yaniv Romano, Michael Elad, and Peyman Milanfar. The little engine that could: Regularization by denoising (red). SIAM Journal on Imaging Sciences, 10(4):1804-1844, 2017.
- [42] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention-MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234-241. Springer, 2015.
- [43] Ernest Ryu, Jialin Liu, Sicheng Wang, Xiaohan Chen, Zhangyang Wang, and Wotao Yin. Plug-and-play methods provably converge with properly trained denoisers. In International Conference on Machine Learning, pages 5546-5557. PMLR, 2019.
- [44] Jonathan Scarlett, Reinhard Heckel, Miguel RD Rodrigues, Paul Hand, and Yonina C Eldar. Theoretical perspectives on deep learning methods in inverse problems. IEEE journal on selected areas in information theory, 3(3):433-453, 2023.
- [45] Johannes Schwab, Stephan Antholzer, and Markus Haltmeier. Deep null space learning for inverse problems: convergence analysis and rates. Inverse Problems, 35(2):025008, 2019.
- [46] Parikshit Shah and Venkat Chandrasekaran. Iterative projections for signal identification on manifolds: Global recovery guarantees. In 2011 49th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 760-767. IEEE, 2011.
- [47] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256-2265, Lille, France, 07-09 Jul 2015. PMLR.
- [48] Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, and Liyue Shen. Solving inverse problems with latent diffusion models via hard data consistency. arXiv preprint arXiv:2307.08123, 2023.

- [49] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.
- [50] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.
- [51] Afonso M Teodoro, José M Bioucas-Dias, and Mário AT Figueiredo. Image restoration and reconstruction using targeted plug-and-play priors. IEEE Transactions on Computational Imaging, 5(4):675-686, 2019.
- [52] Matthieu Terris, Ulugbek S Kamilov, and Thomas Moreau. Fire: Fixed-points of restoration priors for solving inverse problems. arXiv preprint arXiv:2411.18970, 2024.
- [53] Matthieu Terris, Thomas Moreau, Nelly Pustelnik, and Julian Tachella. Equivariant plug-andplay image reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25255-25264, 2024.
- [54] Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9446-9454, 2018.
- [55] Singanallur V. Venkatakrishnan, Charles A. Bouman, and Brendt Wohlberg. Plug-and-play priors for model based reconstruction. In 2013 IEEE Global Conference on Signal and Information Processing, pages 945-948, 2013.
- [56] Yinhuai Wang, Yujie Hu, Jiwen Yu, and Jian Zhang. Gan prior based null-space learning for consistent super-resolution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 2724-2732, 2023.
- [57] Yinhuai Wang, Jiwen Yu, Runyi Yu, and Jian Zhang. Unlimited-size diffusion restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1160-1167, 2023.
- [58] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490, 2022.
- [59] Martin J Willemink, Pim A de Jong, Tim Leiner, Linda M de Heer, Rutger AJ Nievelstein, Ricardo PJ Budde, and Arnold MR Schilham. Iterative reconstruction techniques for computed tomography part 1: technical principles. European radiology, 23:1623-1631, 2013.
- [60] Zamir et al. Restormer: Efficient transformer for high-resolution image restoration. In IEEE/CVF CVPR, pages 5728-5739, 2022.
- [61] Zhiyuan Zha, Bihan Wen, Xin Yuan, Saiprasad Ravishankar, Jiantao Zhou, and Ce Zhu. Learning nonlocal sparse and low-rank models for image compressive sensing: Nonlocal sparse and low-rank modeling. IEEE Signal Processing Magazine, 40(1):32-44, 2023.
- [62] Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and Radu Timofte. Plug-andplay image restoration with deep denoiser prior. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):6360-6376, 2021.
- [63] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7):3142-3155, 2017.
- [64] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.
- [65] Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 1219-1229, June 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly states the technical contributions of the paper along with the description of experimental validation and the relation with prior art.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We included a dedicated section on limitations to thoroughly discuss the constraints and potential weaknesses of our approach.

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

## Answer: [Yes]

Justification: All theoretical results in the paper are accompanied by clearly stated assumptions and complete proofs. Each theorem, definition, and assumption is properly numbered, cross-referenced, and supported by formal derivations, either in the main paper or in the supplementary material, with intuitive sketches provided where appropriate.

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

Justification: The paper provides a comprehensive experimental description with its corresponding settings for reproducing the results

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

Justification: The code will be uploaded in the supplementary material with sufficient instructions to reproduce the results of the paper.

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

Justification: The paper provides comprehensive details regarding the experimental setup, allowing for reproducibility and a clear understanding of the reported results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The paper reports only the mean performance metrics for all experiments.

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

Justification: The paper mentions the computational resources employed to obtain the experimental results, and it is complemented with the memory and time of execution in the supplementary material

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This research fully complies with the NeurIPS Code of Ethics. It does not involve any sensitive information, personal data, human participants, or applications with significant societal risks. All datasets utilized are publicly accessible and ethically obtained, and every experiment was conducted transparently, fairly, and reproducibly.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper considers applications in medical imaging (MRI), which opens new frontiers for further automated and precise medical diagnosis as well as improved image restoration task for commercial cameras.

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

Justification: The generated weights, models, and results do not have risk for misuse as only a public dataset was used.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper properly cites original work on the used models, data and technical developments.

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

Justification: We release both code and trained models as part of this work. All assets are thoroughly documented, including setup instructions, usage examples, model architecture details, training procedures, and licensing information. The assets are provided in the anonymized repository for review to support reproducibility and ease of use.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects Guidelines:

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

## A Technical Appendices

## A.1 Settings for experiment in Figure 1

To illustrate this concept, we developed a toy example with x ∗ ∈ R 3 , H ∈ R 2 × 3 , and constructed an orthonormal vector S ∈ R 1 × 3 relative to the rows of H , following Algorithm 1. A training dataset p ( x ) was generated, consisting of random points within a circle centered at the origin. Using this dataset, we trained a two-layer neural network defined as G( Hx ) = V ϕ ( WHx ) , with parameters W ∈ R k × 2 and V ∈ R 1 × k , with k = 50 hidden neurons, optimized via the problem stated in (3). For comparison, we also trained a network of similar size to directly reconstruct ˜ x 0 from measurements Hx ∗ .

## A.2 Algorithms for designing S

We developed an algorithm for obtaining S from H satisfying that SH ⊤ = 0 . The algorithm is based on the QR decomposition, first, computing a full QR decomposition of H ⊤ ∈ R n × m , yielding an orthonormal basis Q full ∈ R n × n for R n . The columns from m + 1 to n of Q full form a basis for Null( H ) , which we denote by N ∈ R n × ( n -m ) . To construct a subspace of the null space, the algorithm samples a random Gaussian matrix P ∈ R ( n -m ) × p , which is orthonormalized via QR decomposition to produce U ∈ R ( n -m ) × p . This ensures that the resulting subspace is both diverse and well-conditioned. Finally, the matrix S is obtained as S = U ⊤ N ⊤ ∈ R p × n , which consists of p orthonormal vectors that span a random p -dimensional subspace within Null( H ) .

## Algorithm 1 GENERATE ORTHONORMAL ROWS TO H VIA QR DECOMPOSITION

```
)
```

```
Require: Matrix H ∈ R m × n , desired number of rows p Ensure: Matrix S ∈ R p × n whose rows are orthonormal and lie in Null( H ) 1: Q full ← QR ( H ⊤ ) 2: N ← Q full [: , m +1: n ] ▷ Nullspace basis, size n × ( n -m 3: Sample P ∼ N (0 , I ) ∈ R ( n -m ) × p 4: U ← QR( P ) ▷ U ∈ R ( n -m ) × p with orthonormal columns 5: S ← U ⊤ N ⊤ ▷ Resulting p × n matrix of orthonormal rows 6: return S
```

## A.3 Proof of Theorem 1

Here we provide the technical proof for Theorem 1.

Theorem 1 (PnP-NPN Convergence) . Consider the fidelity g (˜ x ) = ∥ H˜ x -y ∥ 2 2 , the denoiser D σ satisfies assumption 2, the matrix S satisfies the RIP condition 2 with constant ∆ S M D ∈ [0 , 1) . The optimized network G ∗ satisfies assumption 3. We run iterations as in (4) in the convergence improvement zone for ℓ = 1 , . . . , L ϕ . The algorithm converges to a fixed point if

<!-- formula-not-decoded -->

Proof:

<!-- formula-not-decoded -->

## A.4 Proof of Theorem 2

Theorem 2 (Convergence of NPN Regularization) . Consider the iterations of the PnP-NPN algorithm defined in (4) for ℓ = 1 , . . . , L . Assume that the estimation error of the trained network G ∗ satisfies Assumption 1, and that both S and H satisfy the Restricted Isometry Property (RIP) over the manifold M D with constants ∆ S M D , ∆ H M D ∈ [0 , 1) . Further assume that the residual term N( Hx ∗ ) is K -Lipschitz continuous, thus ∥ N( Hx ∗ ) ∥ ≤ K (1 + ∆ H M D ) ∥ x ∗ ∥ . Then, after ℓ iterations, the NPN regularization term satisfies the following upper bound:

<!-- formula-not-decoded -->

Proof: We begin by recalling the definition of the regularization function:

<!-- formula-not-decoded -->

From the iteration difference, we have:

<!-- formula-not-decoded -->

Define the intermediate step:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting the definition of q ℓ and rearranging gives:

<!-- formula-not-decoded -->

Expanding both sides and reorganizing terms, we obtain:

<!-- formula-not-decoded -->

Dividing by 2 α and substituting back into (11) yields:

<!-- formula-not-decoded -->

Next, applying Assumptions 2 and 3, we bound:

<!-- formula-not-decoded -->

Substituting this back into (12), we get:

<!-- formula-not-decoded -->

Using the triangle inequality and the RIP property again, we simplify:

<!-- formula-not-decoded -->

Optimality implies:

Hence, regrouping terms, we achieve a final compact form analogous to the desired inequality:

<!-- formula-not-decoded -->

## A.5 Neural network details

SPC : For the single-pixel camera experiments, we employ a ConvNeXt-inspired backbone [33]. The network has five ConvNeXt blocks, each comprising two successive 3 × 3 convolutions with ReLU activations. A final 3 × 3 convolutional layer projects the 128 features back to one channel. The output feature map is flattened and passed through a linear module to match the measurement dimensionality. We use cosine-based positional encoding with a dimension of 256, and set the number of blocks to 5 and the base channel width to 128.

MRI and Deblurring : To train G for the MRI and Deblurring experiments, we use a U-Net architecture [42] with three downscaling and three upscaling modules. Each module consists of two consecutive Conv → ReLU blocks. The downscaling path uses max pooling for spatial reduction, starting with 128 filters and increasing up to 1,024 at the bottleneck. The upscaling path performs nearest-neighbor interpolation before each module, progressively reducing the number of filters. A final 2D convolutional layer without activation produces the output. Skip connections link corresponding layers in the encoder and decoder.

## A.6 Plug and Play algorithms

This work uses the PnP-FISTA, its unrolled version, and RED-FISTA algorithms to validate the proposed approach. Below is shown algorithms for PnP and RED formulation and their NPNregularized counterparts. For unrolling FISTA, the only change is that the denoiser D σ is changed to a trainable deep neural network that is optimized in an end-to-end manner.

## Algorithm 2 PnP-FISTA

Require: L, H , y , α

1:

x

z

0

=

=

,

t

0

1

= 1

2:

for

ℓ

= 1

, . . . , L

do

3:

⊤

x

z

α

←

-

ℓ

ℓ

H

- 4: x ℓ ← D σ ( x ℓ )
- 5: t = t 6: t = 1+ √ 1+4( t ′ ) 2 2

′

7:

-

z

x

+

←

t

ℓ

+1

ℓ

8:

(

Hz

1

′

t

end for

- 9: return x ℓ

## Algorithm 4 RED-FISTA

Require:

L,

,

, α, λ

H

y

- 1: x 0 = z 1 = 0 , t = 1

2:

k

= 1

, . . . , K

do for

3:

4:

5:

6:

x

ℓ

(

←

z

ℓ

t

′

t

))

=

=

t

1+

z

ℓ

+1

←

7:

end for

- 8: return x ℓ

z

α

-

H

ℓ

√

1+4(

t

2

ℓ

x

+

(

⊤

′

)

2

t

′

-

Hz

ℓ

1

t

(

x

-

ℓ

(

x

ℓ

-

-

ℓ

y

-

## Algorithm 3 NPN-PnP-FISTA

Require: L, H , S , y , G ∗ , α, γ

1:

=

=

,

t

= 1

x

z

0

0

1

2:

ℓ

= 1

, . . . , L

for do

- 3: x ℓ ← z ℓ -α ( H ⊤ ( Hz ℓ -y ) + γ S ⊤ ( Sz ℓ -G ∗ ( y )) )
- 4: x ℓ ← D σ ( x ℓ )
- 6: t = 1+ √ 1+4( t ′ ) 2 2
- 5: t ′ = t
- 7: z ℓ +1 ← x ℓ + t ′ -1 t ( x ℓ -x ℓ -1 )
- 8: end for
- 9: return x ℓ

## Algorithm 5 NPN-RED-FISTA

- 3: x ℓ ← z ℓ -α ( H ⊤ ( Hz ℓ -y ) + γ S ⊤ ( Sz ℓ -G ∗ ( y )) ) -λ ( z ℓ -D σ ( z ℓ ))

```
Require: L, H , S , y , G ∗ , α, γ, λ 1: x 0 = z 1 = 0 , t = 1 2: for ℓ = 1 , . . . , L do 4: t ′ = t 5: t = 1+ √ 1+4( t ′ ) 2 2 6: z ℓ +1 ← x ℓ + t ′ -1 t ( x ℓ -x ℓ -1 ) 7: end for 8: return x ℓ
```

D

σ

y

)

x

ℓ

-

)

-

x

1

λ

ℓ

-

)

(

1

z

)

ℓ

-

## A.7 Deep Image Prior

Deep Image Prior (DIP) [54] is an unsupervised reconstruction framework that leverages untrained neural networks to reconstruct the underlying signal. This approach, since it only compares with the measurements y , its solution may overfit to the noise of the measurements. Therefore, the use of the NPN provides a suitable approach to improve image consistency reconstructions. The optimization problem with the proposed NPN regularization is

<!-- formula-not-decoded -->

where γ is a regularization parameter.

## A.8 Diffusion-based Solvers

Diffusion models (DMs) [47, 22, 49, 50] have recently gained attention due to their exceptional capability in modeling complex image distributions via an iterative noising-denoising process. Conditioning DMs entails guiding their generative reverse diffusion process using measured data to ensure reconstructions align with the measurements. We have integrated our proposed regularization term into two widely adopted diffusion-model frameworks, Diffusion Posterior Sampling (DPS) [12] and DiffPIR [65]. These frameworks serve as canonical diffusion pipelines upon which newer methods build. Our regularization could likewise be incorporated into other approaches, such as latent-space diffusion models [15, 48].

DPS [12]: we denote N is the number of reverse diffusion steps, and i ∈ 0 , . . . , N -1 is the reversetime index; x i ∈ R n is the current latent state and x N ∼ N ( 0 , I ) is the Gaussian start; ˆ s = s θ ( x i , i ) is the score/noise estimate produced by the network with parameters θ ; ˆ x 0 is the network's prediction of the clean sample at step i ; α i ∈ (0 , 1] is the per-step retention factor, β i = 1 -α i is the noise increment, and ¯ α i = ∏ i j =1 α j is the cumulative product (with ¯ α 0 = 1 ); ζ i &gt; 0 is the data-consistency step size and ˜ σ i ≥ 0 is the sampling noise scale at step i ; z ∼ N ( 0 , I ) is i.i.d. Gaussian noise; x ′ i -1 denotes the pre-data-consistency iterate before applying the gradient correction; ∇ x i ∥ y -H ˆ x 0 ∥ 2 2 is the gradient of the quadratic data-fidelity term with respect to x i (through the dependence ˆ x 0 ( x i ) );

| Algorithm 6 DPS Sampling                                                                                                                                                                                                                                                                                                                                                                                                 | Algorithm 7 NPN-DPS Sampling                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Require: N, H , y , { ζ i } N i =1 , { ˜ σ i } N i =1 ▷ step sizes and noise scales 1: x N ∼ N ( 0 , I ) 2: for i = N - 1 , . . . , 0 do 3: ˆ s ← s θ ( x i , i ) 4: ˆ x 0 ← 1 √ ¯ α i ( x i +(1 - ¯ α i )ˆ s ) 5: z ∼ N ( 0 , I ) 6: x ′ i - 1 ← √ α i (1 - ¯ α i - 1 ) 1 - ¯ α i x i + √ ¯ α i - 1 β i 1 - ¯ α i ˆ x 0 + ˜ σ i z 7: x i - 1 ← x ′ i - 1 - ζ i ∇ x i ∥ ∥ y - H ˆ x 0 ∥ ∥ 2 2 8: end for 9: return ˆ x 0 | Require: N, H , y , { ζ i } N i =1 , { ˜ σ i } N i =1 ▷ step sizes and noise scales 1: x N ∼ N ( 0 , I ) 2: for i = N - 1 , . . . , 0 do 3: ˆ s ← s θ ( x i , i ) ▷ score / noise-prediction net 4: ˆ x 0 ← 1 √ ¯ α i ( x i +(1 - ¯ α i )ˆ s ) 5: z ∼ N ( 0 , I ) 6: x ′ i - 1 ← √ α i (1 - ¯ α i - 1 ) 1 - ¯ α i x i + √ ¯ α i - 1 β i 1 - ¯ α i ˆ x 0 + ˜ σ i z 7: x i - 1 ← x ′ i - 1 - ζ i ( ∇ x i ∥ ∥ y - H ˆ x 0 ∥ ∥ 2 2 + γ ∥ G ∗ ( y ) - Sˆ x 0 ∥ ) 8: end for 9: return ˆ x 0 |

DiffPIR [65]. σ n &gt; 0 denotes the standard deviation of the measurement noise, λ &gt; 0 is the data-proximal penalty that trades off data fidelity and the denoiser prior inside the subproblem; ρ i ≜ λ, σ 2 n / ˜ σ 2 i is the iteration-dependent weight used in the proximal objective at step i ; ˜ x ( i ) 0 is the score-model denoised prediction of the clean sample at step i (before enforcing data consistency); ˆ x ( i ) 0 is the solution of the data-proximal subproblem at step i (i.e., the data-consistent refinement of ˜ x ( i ) 0 ); ˆ ϵ = ( 1 -α i ) -1 / 2 ( x i - √ ¯ α i , ˆ x ( i ) 0 ) is the effective noise estimate implied by ( x i , ˆ x ( i ) 0 ) ; ϵ i ∼ N ( 0 , I ) is the fresh Gaussian noise injected at step i ; ζ ∈ [0 , 1] mixes deterministic and stochastic updates in the reverse diffusion ( ζ = 0 fully deterministic, ζ = 1 fully stochastic).

## Algorithm 8 DiffPIR Sampling

<!-- formula-not-decoded -->

## Algorithm 9 NPN-DiffPIR Sampling

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.9 Implementation details on NSN-based methods

Recall the NSN-based models used in comparison:

1. Deep null space network (DNSN)[45]: ˆ x = H † y +( I -H † H )R( H † y ) .
2. Deep decomposition network cascade (DDN-C) [6]: ˆ x = H † y + P r (R r ( H † y )) + P n (R n ( H † y +P r (R r ( H † y )))) .
3. Deep decomposition network independent (DDN-I) [6]: ˆ x = H † y + P r (R r ( H † y )) + P n (R n ( H † y ))

We used the source code 2 of [6] to implement the models R , R r and R n .

- Network R and R n : is a lightweight version of the U-Net architecture for image segmentation. It features an encoder-decoder structure with skip connections. The encoder consists of five convolutional blocks, each followed by max-pooling to extract features. The decoder upsamples using transposed convolutions and refines the features with additional convolutions, leveraging skip connections to combine low- and high-level features. The final output is produced through a 1 × 1 convolution, reducing the output to the desired number of channels. This compact design makes it efficient for tasks with limited resources.
- Network R r : It consists of a series of convolutional layers arranged in a sequential block structure, where each block performs a transformation of the input image through convolutional operations, followed by activation functions, such as ReLU, and normalization techniques like batch normalization. The model starts with an input image, and through several layers of convolutions, the features of the image are progressively refined. Each layer extracts relevant features, while ReLU activations introduce non-linearity to improve the model's capacity to capture complex patterns. Batch normalization layers are added to stabilize training and speed up convergence by normalizing the output of each convolutional layer. The final output layer of the network reconstructs the denoised image. The architecture is designed with a depth of 17 layers.

All models were trained for 100 epochs using the Adam optimizer with a learning rate of 1e-3 using a mean-squared-error loss function.

## A.10 Super resolution experiments

For this scenario, we use the Places365 dataset We set an SR factor SRF = √ n m = 4 ; downsampling was performed with bilinear interpolation. The forward operator is modeled as H = DB ∈ R m × n ,

2 https://github.com/edongdongchen/DDN

Table 4: Quantitative comparison of PnP-FISTA and PnP-FISTA NPN with the estimation of Sx .

| Metric   |   PnP-FISTA Baseline |   PnP-FISTA NPN |   Estimation of Sx |
|----------|----------------------|-----------------|--------------------|
| PSNR     |               22.01  |          23.75  |             24.95  |
| SSIM     |                0.562 |           0.692 |              0.697 |

where B ∈ R n × n is a structured Toeplitz matrix implementing Gaussian blur with bandwidth σ = 2 . 0 and D ∈ R m × n is the downsampling matrix.

To recover the information removed by H , we define the null-space projection operator, similar to the deblurring case, S [ i, i + j ] = 1 -h [ j ] , i = 1 , . . . , m, j = 1 , . . . , n, which captures the high-frequency details of the signal. We use the PnP-FISTA algorithm for evaluation with 60 iterations using α = 0 . 5 and γ = 0 . 1 . In the following table, we show the reconstruction performance with baseline PnP-FISTA and NPN PnP-FISTA, showing an improvement of 1.74 dB in PSNR and 0.13 in SSIM. Additionally, we show the network estimation of the null-space Sx metrics, showing good estimation of the high-frequency details from the low-resolution image.

## A.11 Additional Experiments

In Tables 5 and 6, we provide additional results of MRI and deblurring, respectively, using PnP, RED, and sparsity priors for acceleration factors (AFs) of 4, 8, 12 (with 5 dB SNR measurement noise) and σ of 2, 6, 10. These results validate the robustness of NPN regularization under different numbers of measurements.

| AF   | Method       | Prior   | Prior    | Prior   |
|------|--------------|---------|----------|---------|
| AF   | Method       | PnP     | Sparsity | RED     |
| 4    | Baseline NPN | 30.91   | 30.05    | 29.17   |
| 4    | Baseline NPN | 31.11   | 31.04    | 31.46   |
| 8    | Baseline NPN | 27.25   | 26.72    | 27.59   |
| 8    | Baseline NPN | 29.88   | 30.02    | 28.84   |
| 12   | Baseline NPN | 26.35   | 25.68    | 27.35   |
| 12   | Baseline NPN | 29.06   | 29.24    | 28.13   |

Table 5: MRI: PSNR (dB) for PnP vs. PnP-NPN under different priors and AFs.

| σ    | Method   | Prior   | Prior    | Prior   |
|------|----------|---------|----------|---------|
|      |          | PnP     | Sparsity | RED     |
| 2.0  | PnP      | 30.78   | 29.27    | 32.84   |
| 2.0  | NPN      | 31.77   | 31.42    | 33.67   |
| 5.0  | PnP      | 24.47   | 23.17    | 24.79   |
| 5.0  | NPN      | 25.72   | 25.32    | 25.34   |
| 10.0 | PnP      | 20.49   | 19.61    | 20.48   |
| 10.0 | NPN      | 20.81   | 20.60    | 20.54   |

Table 6: Deblurring: PSNR (dB) for PnP vs. PnPNPN under different priors and noise levels.

In Figures 7, 8, and 9, the effect of the parameter γ on the quality of the recovery is shown, for MRI, Deblurring, and SPC, respectively.

Fig. 10 shows the results in terms of convergence of the PnP and PnP-NPN for SPC with m/n = 0 . 1 . The color of the line indicates the type of projection matrix used, orthonormal by QR (red) or designed by Eq. 3 (blue). The color shade indicates the percentage of the low-dimensional subspace p/n , ranging from 0.1 to 0.9. The best case is with p/n = 0 . 1 and S designed, which was the one used in the experiments of the main paper.

Additionally, we implemented several state-of-the-art denoisers into the PnP framework. We also change the FISTA algorithm used in the paper to an alternating direction method of multipliers (ADMM) formulation. This algorithm splits the optimization problem into two subproblems: the data fidelity and the prior. We employed several state-of-the-art denoisers such as Restormer [60], DnCNN [63], DnCNN-Lipschitz [43], and DRUNet [62]. We validated the image deblurring application with a Gaussian kernel bandwidth σ = 2 . 0 . We used 200 iterations of the PnP-ADMM algorithm, the step size α = 0 . 5 , and the value of γ = 0 . 7 . In Table 1 are shown the obtained results. Here, the proposed regularization function consistently improves the baseline PnP-ADMM algorithm.

<!-- image -->

Figure 7: Effect of γ in NPN on PSNR (dB) for MRI reconstruction, with α = 1 × 10 -4 . The maximum PSNR of 33.67 dB is achieved when γ = 6 × 10 -3 .

<!-- image -->

Figure 8: Effect of γ on PSNR (dB) in deblurring reconstruction, with α = 0 . 5 × 10 -4 . The maximum PSNR of 31.25 dB is achieved when γ = 0 . 263 .

Figure 9: Effect of γ on PSNR (dB) in SPC reconstruction, with α = 8 × 10 -4 . The maximum PSNR of 21.17 dB is achieved when γ = 1 . 04 .

<!-- image -->

Table 7: Comparison of PnP-ADMM method for image deblurring with σ = 2 . 0 using different denoisers. The best result is highlighted in bold teal , and the second-best in orange underline .

| Method   |   Restormer |   DnCNN |   DnCNN-Lipschitz |   DRUNet |   Sparsity Prior |
|----------|-------------|---------|-------------------|----------|------------------|
| Baseline |       29.86 |   29.55 |             30.36 |    29.68 |            28.75 |
| NPN      |       32.62 |   32.12 |             32.35 |    32.07 |            29.75 |

## A.12 Analysis on the selction of p and data adaptation

The selection of the size of S is important for the optimization (3) as the bigger value of p , the more challenging the projection estimation becomes. We show the estimation error in the following table, showing that the error increases by increasing the number of rows of S .

Figure 10: Effect of the low-dimensional subspace dimension p/n and projection matrix S design on signal convergence in SPC reconstruction, with α = 8 × 10 -4 .

<!-- image -->

Table 8: Projection error for different values of p/n .

|   p/n |   ∥ Sx ∗ - G(y) ∥ ∥ Sx ∗ ∥ |
|-------|----------------------------|
|   0.1 |                     0.1031 |
|   0.3 |                     0.1544 |
|   0.5 |                     0.1821 |
|   0.7 |                     0.2566 |
|   0.9 |                     0.2305 |

Additionally, to illustrate the effect of the data-driven adaptability of S , we clarify the following. In compressed sensing, to span the null-space, we design a matrix orthogonal to H via the QR decomposition, and via optimization using the regularization in Eq. (3), which promotes that the rows of S lie in the null-space of H . To analyze how the data distribution affects the design of S , we optimize four matrices, one for each dataset (MNIST, FashionMNIST, CIFAR10, and CelebA), cross-validating the data-driven invertibility loss E x ∥ x -S † Sx ∥ 2 2 . The results show that the QRinitialized matrix provides good invertibility across all datasets, and that optimizing S via Eq. (3) on each dataset further improves it. Thus, the data distribution can enhance the data-driven invertibility of an already orthogonal matrix. Testing S on in-distribution samples yields the best invertibility ( bold ), while diverse training data (e.g. CIFAR10) gives the second-best (underlined), suggesting a single S could generalize across datasets.

Table 9: Cross dataset validation of data invertibility metric E x ∥ x -S † Sx ∥ 2 2

| Test/ S design               | QR ( H ) Alg 1   | MNIST           | FashionMNIST    | CIFAR10       | CelebA          |
|------------------------------|------------------|-----------------|-----------------|---------------|-----------------|
| MNIST 9 . 28 × 10 - 5        | 2 . 97 × 10 - 5  | 4 . 99 × 10 - 5 | 5 . 29 × 10 - 5 | 5 . 56 × 10 5 | -               |
| FashionMNIST 1 . 76 × 10 - 4 | 7 . 07 × 10 - 5  | 3 . 11 × 10 - 5 | 5 . 05 × 10     | - 5           | 5 . 62 × 10 - 5 |
| CIFAR10 2 . 62 × 10 - 4      | 1 . 67 × 10 - 4  | 8 . 95 × 10 - 5 | 4 . 01 × 10 -   | 5 4 . 52      | × 10 - 5        |
| CelebA 2 . 39 10 - 4         | 1 . 55 10 - 4    | 9 . 09 10 -     | 5               | 4 . 68 10 - 5 | 4 . 15 10 - 5   |

×

×

×

×

×

## A.13 Limitations on the selection of S

The method works when it is possible to find a non-linear correlation between null space (NS) components Sx and measurements Hx , leveraging a dataset with triplets ( x i , Hx i , Sx i ) . Wepromote this correlation by solving Eq. (3) that balances the two terms: i) ∥ Sx -G( Hx ) ∥ encourages a non-linear correlation between Sx and Hx . ii) ∥ I -A ⊤ A ∥ where A = [ H ⊤ , S ⊤ ] ⊤ promotes orthogonality and ensures that S samples components from the NS of H .

There are scenarios where the non-linear correlations are easier to achieve, for instance, in deblurring or SR, due to the low-pass filters associated with these tasks being non-ideal, which leads to close frequency bands between the high-pass components selected by S and the Gaussian-like low-pass components sampled by H . In other scenarios, such as MRI or CT, the non-linear correlations between Sx and Hx are more challenging to achieve due to the orthogonality of their respective rows. But, the method still works by designing S such that it leverages structural similarities of H , such as adjacent frequencies in MRI or neighboring angles in CT.

To illustrate the effectiveness of finding non-linear correlations between Sx and Hx , we set a challenging scenario in an MRI task with H sampling only random low frequencies using a 1D Cartesian mask with an acceleration factor (AF) of 12. We then apply an FFTSHIFT operation to this mask to produce a high-frequency sampling pattern, denoted by S 1 , spatially distant from the frequency support of H but maintaining the same AF. We refer to this configuration as the wellseparated sampling. In this case, the method fails to recover non-trivial solutions, ∥ G ∗ ( y ) ∥ = 0 . 0108 , and a high relative error of 1 . 0 .

Then, we configure a scenario where the non-linear correlations are easier to find, using the same H described before, we set S 2 = I -H that selects frequencies adjacent to those sampled by H . This adjacent sampling configuration corresponds precisely to the setup employed in the MRI experiments presented in the main manuscript. See Figure 11. It exhibits increased spectral (nonlinear) correlation, resulting in a small relative error of 0.325, and a norm ∥ G ∗ ( y ) ∥ close to the norm of the ground-truth signal ∥ Sx ∥ .

Figure 11: MRI masks

<!-- image -->

Table 10: Comparison of reconstruction metrics under different sampling strategies.

| Sampling Type     |   ∥ Sx - G( y ) ∥ ∥ Sx ∥ |   ∥ G( y ) ∥ |   ∥ Sx ∥ |   PSNR( G( y ) , Sx ) [dB] |
|-------------------|--------------------------|--------------|----------|----------------------------|
| Disjoint Sampling |                      1   |       0.0197 |     1.72 |                       30.7 |
| Adjacent Sampling |                      0.4 |      98.7    |   112    |                       59.5 |

## A.14 Comparison with reconstruction model

To create a fair comparison with the proposed method, we consider the network W ∗ : R m → R n , which has the same number of parameters as G but is trained to reconstruct directly the underlying signal i.e., W ∗ = arg min W E x , y ∥ x -W( y ) ∥ . The network was trained with the same number of epochs as G (300 epochs), using the AdamW optimizer with a learning rate of 1 e -4 . We evaluate two aspects here: i) the reconstruction performance of the network W and ii) incorporating into the PnP-ADMM the regularization but in the form of ϕ W ( x ) = ∥ Sx -S W ∗ ( y ) ∥ 2 . We trained the

network W with the Places365 dataset. The PnP-ADMM was set with 100 iterations, using the DnCNN-Lipchitz denoiser. We set α = 0 . 3 and γ = 0 . 2 . In the following table, we report the obtained results. The results show that the proposed approach overall improves the baseline, the reconstruction of the model W , and the regularization based on this model.

Table 11: Comparison of PSNR (dB) and SSIM metrics

| Metric    |   Base PnP-ADMM |   NPN PnP-ADMM |   PnP-ADMM with ϕ W |   W ∗ ( y ) |
|-----------|-----------------|----------------|---------------------|-------------|
| PSNR (dB) |          22.3   |         23.87  |              21.2   |       19.71 |
| SSIM      |           0.586 |          0.678 |               0.534 |        0.49 |

In the tables below, we show the training and execution time in seconds (s) and minutes (min), respectively. These results are obtained for a batch size of 100 images with a resolution of 128 × 128 and 100 iterations of the PnP-ADMM. W (reconstruction network from the measurements y ) and G are trained for 300 epochs. The training times of W and G are very similar due to having the same number of parameters. Despite a modest increase of 0.20 seconds over the baseline PnP-ADMM, NPN achieves a 1.57 dB PSNR gain. Moreover, compared to PnP-ADMM with ϕ W , NPN is 0.28 seconds faster while delivering a 2.67 dB improvement in PSNR. These results show that the NPN regularization yields substantial quality gains with minimal or favorable time trade-offs.

Table 12: Training time for the networks W and G (NPN).

| Method   |   Training time (min) |
|----------|-----------------------|
| G( y )   |                   385 |
| W( y )   |                   390 |

Table 13: Execution times for PnP with network W and G (NPN).

| Method            |   Execution time (s) |
|-------------------|----------------------|
| Base PnP-ADMM     |                12.48 |
| NPN PnP-ADMM      |                12.68 |
| PnP-ADMM with ϕ W |                12.96 |