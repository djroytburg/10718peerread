## Follow the Energy, Find the Path: Riemannian Metrics from Energy-Based Models

## ∗

Louis Bethune Apple

Yilun Du Harvard University

David Vigouroux

IRT Saint Exupéry, ANITI, IMT Atlantique

Rufin VanRullen CNRS

Thomas Serre Brown University

## Abstract

What is the shortest path between two data points lying in a high-dimensional space? While the answer is trivial in Euclidean geometry, it becomes significantly more complex when the data lies on a curved manifold-requiring a Riemannian metric to describe the space's local curvature. Estimating such a metric, however, remains a major challenge in high dimensions.

In this work, we propose a method for deriving Riemannian metrics directly from pretrained Energy-Based Models (EBMs)-a class of generative models that assign low energy to high-density regions. These metrics define spatially varying distances, enabling the computation of geodesics-shortest paths that follow the data manifold's intrinsic geometry. We introduce two novel metrics derived from EBMs and show that they produce geodesics that remain closer to the data manifold and exhibit lower curvature distortion, as measured by alignment with ground-truth trajectories. We evaluate our approach on increasingly complex datasets: synthetic datasets with known data density, rotated character images with interpretable geometry, and high-resolution natural images embedded in a pretrained VAE latent space. Our results show that EBM-derived metrics consistently outperform established baselines, especially in high-dimensional settings.

Our work is the first to derive Riemannian metrics from EBMs, enabling dataaware geodesics and unlocking scalable, geometry-driven learning for generative modeling and simulation.

## 1 Introduction

What is the shortest path between two data points in a high-dimensional space? In Euclidean geometry, the answer is a straight line. But in modern machine learning, where data often lies on unknown curved manifolds within a high-dimensional space, straight lines slice through regions without data (see linear interp. in Fig. 1). Capturing the true geometry of data is therefore critical in fields where distance-based analyses depend on underlying structure, such as vision [1-3], language [4, 5], biology [6], and cognitive science [7, 8]. Riemannian geometry offers a principled way to navigate these spaces by introducing a smoothly varying local metric, the Riemannian metric, which encodes how space bends and stretches [9]. Within this framework, the shortest path between two points is no longer a straight line, but a geodesic-a curve that follows the intrinsic curvature of the manifold. Computing geodesics requires knowing the underlying Riemannian metric, but estimating such a metric for complex, high-dimensional data remains a major challenge in machine learning.

∗ Equal contribution.

Victor Boutin ∗ CNRS

A promising strategy for deriving Riemannian metrics is to take a data-driven approach-learning the metric directly from the data itself. This approach estimates the data density and turns it into a Riemannian metric that contracts high-density regions and dilates low-density ones, aligning the geometry with the data manifold [10] (see § 2 for more details). Existing methods, such as kernelbased estimators [11], normalizing flows [12], and density-based constructions [13], have succeeded in low-dimensional settings. However, their performance often degrades in high dimensions, where sparse local sampling makes it hard to capture reliable geometric structure [14, 15]. Meanwhile, recent advances in generative AI [16-18] have produced models capable of capturing complex data distributions in high-dimensional spaces with remarkable accuracy. If these models can learn the data distribution, can they also reveal its underlying geometry?

In this article, we answer affirmatively by proposing to derive Riemannian metrics from pretrained Energy-Based Models (EBMs) [16, 19, 20]. EBMs are a flexible class of generative models that define an energy function E θ , parameterized by a neural network, assigning low energy to likely data points (i.e., p θ ( x ) ∝ exp( -E θ ( x )) ).We show that the energy landscape of an EBM encodes a rich geometric structure and can be leveraged to derive effective Riemannian metrics. Specifically, we introduce two novel conformal Riemannian metrics-metrics that scale the identity by a positive scalar function: G E θ proportional to the energy itself, and G 1 / p θ , proportional to the inverse unnormalized density. We evaluate both against established alternatives ( G RBF [13] and G LAND [11]) across datasets of increasing complexity-from toy distributions with known geodesics (see § 4.2), to rotated character images where the manifold structure is partially known (see § 4.3), and finally to high-dimensional natural images where no ground truth geometry is available (see § 4.4). Throughout this work, we

Figure 1: Geodesics visualization for the URC dataset . Trajectories and samples are projected in the PCA space for visualization.

<!-- image -->

.

adopt the common choice of equipping the data space with a density-based Riemannian metric, thereby defining the geometry of the manifold in terms of data concentration. We show that EBMbased metrics yield geodesics that (i) remain closer to the data manifold and (ii) better reflect its intrinsic curvature (see Fig. 1 for a visualization of the geodesics).

Overall, our contributions are summarized as follows:

- We propose a novel framework based on pretrained Energy-Based Models (EBMs) to derive Riemannian metrics. In particular, we introduce two novel conformal metrics G E θ and G 1 / p θ , based on the data log-likelihood and data density, respectively.
- We demonstrate that these EBM-derived metrics yield geodesics that remain closer to the data manifold and better reflect its curvature.
- We show that the proposed EBM-based metrics scale more robustly than prior approaches.

By grounding Riemannian metrics in generative AI, we hope to initiate a new paradigm for understanding and navigating the hidden geometry of high-dimensional data spaces.

## 2 Related Work

The many facets of data geometry: A variety of approaches have been proposed to study the geometry of data:

- Information Geometry : This historical approach is rooted in the work of Rao [21] and Amari [22]. It connects statistics and differential geometry by interpreting the Fisher information [23] as a Riemannian metric on the manifold of parameters of a statistical model. In contrast, our

work derives Riemannian metrics directly from the data space using the energy or the likelihood of an EBM.

- Data-Space induced metrics : Closer to our work, this approach estimates Riemannian metrics directly from samples. The LAND metric [11] derives a local metric tensor from the empirical covariance of nearby points. The RBF metric [13] defines a conformal metric using an RBF network trained as a parametric KDE, learning centres, widths, and weights so its output forms an unnormalised data density. Both serve as baselines in our study (see § 4.1) and have recently been used for geodesic fitting via flow matching [24]. The (unpublished) work of Perone [25] was also a key inspiration, proposing to build metrics from the score function of a generative model-an idea also explored by Diepeveen et al. [26].
- Latent-Space induced metrics : Another line of work uses pullback geometry [27-32], mapping the Euclidean metric from a network's latent space to the data space-typically through the Jacobian of a VAE encoder [33]. While our method operates in the latent space of a VAE in high-dimensional settings, the metric is derived from the energy of the EBM and remains independent of the VAE encoder.
- Generative modeling on a pre-defined manifold: Recent approaches such as flow-based models [34, 35] and Schrödinger bridges [36, 37] learn transport paths between distributions, sometimes defined over Riemannian manifolds [38-41]. These methods assume a known, fixed manifold geometry (e.g., a hypersphere) and design generative models to operate within that structure. In contrast, our approach starts from a generative model-an EBM-and derives the Riemannian metric itself from the model, allowing the geometry to emerge from the data.

For a more detailed review of the related work (including topological data analysis, symmetries, computer graphics, or metric learning), see Supp. A and [42, 14].

Energy-Based Models (EBMs): EBMs, trained via maximum likelihood [16] (see § 3.2), are particularly well-suited for deriving Riemannian metrics. Their contrastive training, combined with Langevin dynamics sampling, encourages learning a global energy landscape that assigns meaningful values across the entire ambient space, including regions far from the data manifold. In contrast, normalizing flows [43] are limited by their invertible architecture [44, 45] and tend to perform poorly on out-of-distribution data [46], sometimes leaking probability mass outside the support [47]. EBMs trained with diffusion losses [48] or distilled from diffusion models [49] generate high-quality samples, but their energy function depends on a time-indexed noise scale, limiting them to local rather than global energy landscapes. This makes them unsuitable for defining a consistent Riemannian metric. Prior work has used the global energy landscape of EBMs trained via maximum likelihood for trajectory planning in robotics [50], though not in the context of geodesics.

## 3 Method

Notation : Scalars are denoted by plain lowercase (e.g., x), vectors by bold lowercase (e.g., x ∈ R D ), and matrices by bold uppercase (e.g., X ). Let I be the identity matrix of R D × D . S D ++ is the set of symmetric D × D positive definite matrices. Let M be a Riemannian manifold, with tangent space at x ∈ M denoted T M x . Herein, we assume that M is embedded in a D -dimensional Euclidian space ( M⊂ R D ).

## 3.1 A primer on Riemannian geometry

A Riemannian manifold ( M , G ) is a smooth manifold M (i.e., a set locally homeomorphic to R D ) equipped with a Riemannian metric G : M→S D ++ . G defines a smoothly changing inner product on the tangent space T M x at each point x ∈ M : ⟨ u , v ⟩ x = u ⊤ G ( x ) v , with u , v ∈ T M x [9]. The length of a curve γ : [0 , 1] →M linking two points x 0 = γ (0) and x 1 = γ (1) ( x 0 , x 1 ∈ M ), is measured as:

<!-- formula-not-decoded -->

In Eq. 1, ˙ γ ( t ) denotes the velocity vector of the curve γ ( t ) , which lies in the tangent space at that point (i.e., ˙ γ ( t ) ∈ T M γ ( t ) ). The minimizer of Eq. 1 is called a geodesic ; it represents the (locally) shortest path between x 0 and x 1 . In this work, we minimize the kinetic energy functional instead of the length (see Eq. 2). Although both functionals yield the same geodesics up to a parametrization,

minimizing the kinetic energy functional results in a constant Riemannian speed parametrization 2 . This property simplifies optimization and improves numerical stability [9, 13].

<!-- formula-not-decoded -->

In the Euclidean case ( M = R D , G ( x ) = I ), E is equivalent to the kinetic energy of a unit-mass particle moving along γ ( t ) , hence the name kinetic energy functional.

To avoid the computational cost of solving Eq. 2 for each new pair ( x 0 , x 1 ) at inference time, we follow [24] and approximate the geodesic with a neural interpolant φ η (with parameters η ).

<!-- formula-not-decoded -->

This parameterization satisfies the boundary conditions ( x 0 ,η = x 0 , x 1 ,η = x 1 ). In Eq. 3, φ η serves as a nonlinear correction to the linear path, allowing the learned path to bend toward the data manifold. We train a single interpolant network φ η over batches of random endpoint pairs so it can approximate geodesics between arbitrary points (see Algo. 1). Intuitively, our geodesic interpolant begins with a straight line between the endpoints and uses a neural network to compute a smooth curvature relative to this baseline-bending the path toward regions of higher data density, much like pulling a string taut over a curved surface that reflects the geometry of the data. Unlike Kapusniak et al. [24], who use full autodifferentiation to compute ˙ x t,η , we opt for finite difference instead. We found this approach more stable and accurate when using a fine-time discretization.

Although Algo. 1 approximates geodesics for a given metric G , the trajectories may initially deviate from the data manifold-especially early in training, when they are initialized as straight lines in the ambient space. However, if (i) the eigenvalues of G are large when off-manifold and (ii) small when on-manifold, then the interpolated points x t are progressively drawn toward the manifold during optimization [24, 13]. In other words, an effective G should penalize off-manifold directions and encourage paths through high-density

## Algorithm 1: Training geodesic interpolant

<!-- formula-not-decoded -->

paths, steering the geodesics along true data geometry. This insight suggests that defining the metric as a decreasing function of the data probability (e.g., G ( x ) ∝ p ( x ) -1 · I ) can effectively steer trajectories toward high-density regions. In practice, however, the true data distribution is unknown and only observed through samples. In this work, we use an EBM to approximate the data distribution.

## 3.2 Energy-Based Models

Let p M be the true data distribution supported on the manifold M , such that ∫ x ∈M p M ( x ) d x = 1 . In practice, we do not have access to p M directly, but only to a finite set of samples D = { x i } N i =1 drawn from it. These samples define the empirical distribution p D , which we use to train our models.

Energy-Based Models (EBMs) provide a flexible framework for modeling complex, unnormalized probability distributions-making them particularly well-suited for data concentrated on lowdimensional manifolds. Here we define the energy function E θ ( x ) ∈ R , parameterized with a neural network with weights θ . This energy induces a probability distribution of the form:

<!-- formula-not-decoded -->

2 With length fixed, the strictly convex energy E = 1 2 ∫ 1 0 v ( t ) 2 dt attains its minimum-by Jensen's inequality-only when the speed v ( t ) is constant.

Our goal is to train the EBM so that p θ approximates the data distribution p M . To do so, we minimize the negative log-likelihood w.r.t to the empirical distribution: L ML ( θ ) = E x ∼ p D [ -log p θ ( x )] . Although the partition function Z ( θ ) is intractable, previous works have shown that the gradient of this objective can be estimated without computing Z ( θ ) explicitly [51, 52] (see Supp. B.1 for the demonstration), a loss known as contrastive divergence :

<!-- formula-not-decoded -->

where x + are data samples and x -are samples drawn from the model distribution p θ using Langevin dynamics. We adopt the training procedure of [16], which is known to scale well (see Supp. B for the full pseudo-code). From this point on, we refer to E θ as a pre-trained energy function.

EBM can be hard to train in high-dimensional pixel space, especially because of the sampling procedure [53-55]. For complex tasks, we follow standard practice and operate in the latent space of a pretrained VAE [56], where all baselines are evaluated for fairness. To improve the EBM training training stability, we regularize the contrastive divergence loss with a denoising term, which preserves the global structure of the energy landscape while enhancing convergence-a technique we find both effective and broadly applicable.

## 3.3 EBM-derived Riemannian Metrics

Here, we describe the EBM-derived metrics G E θ , G 1 / p θ . For details on the baseline Riemannian metrics G LAND , G RBF , see § 4.1. To ensure a fair comparison -and following standard practice in the field [42, 57, 58]- all metrics are cast using a shared parametric form:

<!-- formula-not-decoded -->

where h ( x ) is a metric-specific, positive-definite function (either scalar, diagonal, or matrix), and α , β are calibration constants. These constants are chosen so that the metric scale to I on the data manifold and to 10 3 · I in low-density regions 3 . This allows fair comparison across metric choices without introducing significant sensitivity to hyperparameter tuning. Further details about the metric calibration procedure are provided in Supp. C.1. Importantly, all EBM-derived metrics are conformal , they take the form λ ( x ) I , where λ is a scalar function. In other words, they scale the identity matrix uniformly in all directions, resulting in isotropic metrics:

- G E θ defines a Riemannian metric by directly scaling the raw energy of a pretrained EBM. This is the simplest -yet surprisingly effective-formulation we consider:

<!-- formula-not-decoded -->

Intuitively, high-energy (low-density) regions receive a larger metric, penalizing movement away from the data. Note that E θ is an affine rescaling of the negative log-likelihood -log p D .

- G 1 / p θ leverages the inverse of an unnormalized probability estimate:

<!-- formula-not-decoded -->

Compared to G E θ , this metric applies an inverse to a decreasing exponential, forming a strong barrier against low-density regions. It stays small near the data manifold but rises sharply elsewhere, acting as a repulsive force. Its key advantages are: (i) a clear probabilistic interpretation via the unnormalized density, and (ii) direct comparability to G LAND and G RBF as they share the same inverse form.

In the next section, we introduce the baseline Riemannian metrics used for comparison. We also empirically evaluate their behavior across datasets of increasing complexity, focusing on how they capture the underlying manifold and shape geodesic paths.

## 4 Experiments

## 4.1 Baseline Riemannian Metrics

G RBF [13, 24] and G LAND [11] are established metrics from the Riemannian geometry literature:

3 Note that this multiplicative factor amounts to a change of unit, to ensure reasonable scaling of the lengths, but the induced geodesics are only determined by the ratio α/β .

- G LAND, also known as the LAND metric [11], is a nonparametric Riemannian metric that adapts to the local geometry of the dataset. Around each point x , it estimates a Gaussian distribution by weighting all data points { x i } N i =1 according to their distance to x :

<!-- formula-not-decoded -->

Here, h ( j ) ( x ) measures the local variance along dimension j , weighted by a Gaussian kernel with bandwidth σ . G LAND is the only diagonal (i.e., non-conformal) metric we consider, allowing it to model local anisotropy. While flexible and model-free, LAND has practical drawbacks: it requires the full dataset at inference, is sensitive to the choice of σ , and can behave non-smoothly near sharp neighborhood transitions (see Supp. C.2 for examples).

- G RBF is a conformal Riemannian metric in which h is a weighted sum of Radial Basis Functions (RBFs) centered on K cluster centroids { ˆ x k } K k =1 computed via K-means [13]:

<!-- formula-not-decoded -->

The weights w k are trained so that h ( x ) ≈ 1 on the data manifold, and λ k is set from inter-cluster distances (see Supp. C.3). This yields a smooth, efficient approximation of the data geometry and scales better than LAND [24]. However, it may miss fine-grained structure, especially in regions of complex or uneven density. Like other methods based on Euclidean distance (and K-means), it suffers from the curse of dimensionality. Its accuracy depends on K , λ k , and centroid placement (illustrated in Supp. C.3).

The scaling constants ( α , β ) are introduced to ensure consistent dynamic range across metrics and have minimal impact on convergence or geodesic quality; the number of discretization steps ( T = 100 ) is chosen as a trade-off between efficiency and accuracy, consistent with prior work. We evaluate G 1 / p θ , G E θ , G RBF , and G LAND on three datasets of increasing complexity. Circular Mixture of Gaussians offers full control and ground-truth geodesics. The rotated characters dataset is higher-dimensional but still allows quantitative evaluation. Animal Faces is made of higher-dimensional images but with no ground truth. This progression tests metric performance as data complexity grows. The code to reproduce all our experiments is available at https://github.com/VictorBoutin/RiemannEBM .

## 4.2 Circular Mixture of Gaussians

We consider two toy datasets built using a mixture of Gaussians arranged along a semicircle. In the first, called Uniform Circular Gaussians (UCG), the Gaussian components have equal weights (see Fig. 2a). In the second, Weighted Circular Gaussians (WCG), the weights are non-uniform, with higher density near the center of the arc, as reflected by the contour intensity shown in Fig. 2c. For both datasets, we have access to the closed-form probability distribution of the data, denoted p M (see Supp. D.1 for details of p M ). We first train an Energy-Based Model (EBM) on each dataset to derive the metrics G E θ and G 1 / p θ (see Supp. D.2 for training details). Then, we apply Algo. 1 to both datasets using all Riemannian metrics described above. Additionally, we include two baseline Riemannian metrics derived directly from the true distribution p M :

<!-- formula-not-decoded -->

Eq. 9 uses calibration constants α and β , computed as in other metrics. Some geodesics obtained for the 6 different metrics are shown in Fig. 2a and Fig. 2c for the UCG and WCG datasets, respectively. We refer the reader to Supp. D for details on network architectures and hyperparameters.

To evaluate geodesic quality, we use two evaluation metrics. The first is the accumulated probability along the geodesic path, p M ( γ ⋆ ) = ∑ T t =1 p M ( x t,η ⋆ ) . It measures how closely the trajectory aligns with the data manifold - the higher the better. The second is the RMSE to a baseline geodesic computed using the true distribution p M , matched by metric type (e.g., G E θ vs. G E M ). All quantitative results are averaged over 1 , 000 geodesics with distinct endpoints (See Fig. 2b and d). G E θ achieves the highest accumulated probability, indicating closest alignment with the data manifold, while G 1 / p θ yields the lowest RMSE to its baseline-best approximating the ground-truth geodesic. Both EBM-based metrics consistently outperform other methods across evaluation criteria.

To test how different metrics behave when the density varies along the data manifold, we switch

<!-- image -->

G1/pe

G1/PM

Figure 2: Geodesics on UCG and WCG datasets. (a, c) : Some geodesics obtained on UCG (a) and WCG (c) , for 6 different Riemannian metrics. The contour plots represent the energy landscape given by -log p M . (b, d) Quantitative evaluation of geodesics on UCG (b) and WCG (d) . We report (i) the accumulated probability along the geodesic (the higher the better) and ii) RMSE between each geodesic and its corresponding baseline (i.e., G E M for G E θ , and G 1 /p M for G 1 / p θ , G LAND and G RBF ). See Supp. D.3 for the 2 -σ error.

from the uniformly populated UCG semicircle to the Weighted Circular Gaussian (WCG), whose samples cluster near the arc's centre. As shown in Fig. 3, log-based metrics ( G E θ , G E M ) accentuate the manifold curvature more than 1 /p -based ones ( G 1 / p θ , G RBF , G LAND , G 1 /p M ), producing larger steps in high-density regions. This is because -log p diverges as p → 0 , amplifying distortions and speed variations.

## 4.3 Rotated Characters

We use an image dataset of seven rotated, non-symmetric characters in two variants: Uniform Rotated Characters (URC), with evenly distributed angles, and Biased Rotated Characters (BRC), concentrated near 0 ◦ . In this subsection, all computations are done in the 64 -dimensional latent space of a regularized autoencoder trained with a triplet loss, ensuring that small angular differences yield short latent distances. This setup provides a unique middle ground: although the underlying Riemannian metric is

Figure 3: Step size along geodesics in the WCG dataset . Log-based metrics ( G E θ and G E M ) produce sharper variations, reflecting stronger sensitivity to density curvature.

<!-- image -->

unknown, we can treat the smooth in-plane rotation between two instances of the same character as a proxy for the ground-truth geodesic. Thanks to the triplet loss, the latent space is structured so that nearby points correspond to slight rotations of the same character, making the shortest path between two orientations a meaningful approximation of the true geodesic in the task-relevant transformation space. Separate EBMs and interpolant networks are trained for each dataset variant. Full experimental details (datasets, architectures, and hyperparameters) are provided in Supp. E.

Figure 4: Geodesics on the URC dataset. (a) Geodesics computed with different Riemannian metrics, projected into pixel space for visualization. G RBF and G LAND often deviate from the intended path, sometimes drifting toward other characters (e.g., the letter F ). (b) Quantitative evaluation using two metrics: (i) D -RMSE, which measures proximity to the dataset manifold, and (ii) γ -RMSE, which measures the deviation from an ideal smooth rotation. See Supp. E.6 for the 2 -σ error.

<!-- image -->

In Fig. 4a, we visualize geodesics computed on the pixel space (see Supp. E.5 for additional results on both URC and BRC). EBM-based metrics ( G E θ and G 1 / p θ ) yield smooth rotations that preserve character identity, while G RBF and especially G LAND often deviate from the intended trajectory. To illustrate these failures, Fig. 1 shows all geodesics projected into PCA space for a case involving the letter F. While G E θ and G 1 / p θ remain on the manifold of rotated F instances, linear interpolation cuts through low-density regions, and G RBF and G LAND drift toward other character classes. To quantify this, we use two metrics: D -RMSE, which measures the average distance from each geodesic point to its nearest neighbor in the dataset-lower values indicate better adherence to the data manifold; and γ -RMSE, which evaluates how closely the geodesic follows an ideal smooth rotation between endpoints. All results are averaged over 1 , 000 geodesics with random endpoint orientations. As shown in Fig. 4b, EBM-based metrics consistently outperform others; G RBF performs reasonably well, while G LAND

URC dataset, projected

back into

Figure 5: Step size along geodesics in the WCG dataset . Log-based metric ( G E θ ) produces sharper variations, reflecting stronger sensitivity to density curvature.

<!-- image -->

shows large deviations on both metrics. Overall, these results suggest that EBM-based metrics scale more effectively to high-dimensional data than alternative approaches.

As in the previous section, we examine how different metrics influence a geodesic's ability to follow the manifold's curvature. We focus on the BRC dataset, where orientations are biased toward 0°, creating sharper curvature near that region. To assess this, we decode the orientation at each time step along geodesics connecting fixed endpoints. As shown in Fig. 5, geodesics under G E θ rotate significantly faster near 0° than those under G 1 / p θ and G RBF , reflecting stronger sensitivity to density variations.

At first glance, it may seem counterintuitive that trajectories following the geodesics move faster in high-density regions. However, this is consistent with minimizing the kinetic energy E in Eq.2, which enforces constant Riemannian speed (i.e., the quantity || ˙ γ ( t ) || γ ( t ) is preserved along the trajectory) but not a constant Euclidean speed (i.e., || ˙ γ ( t ) || is not constant). Since EBM-derived metrics assign lower Riemannian cost in high-density regions, maintaining constant Riemannian speed requires moving faster in Euclidean terms through these regions. The faster rotation near 0°, observed in Fig.5 and Fig. 3, thus reflects the lower Riemannian cost of traveling through high-density regions. These results confirm and extend our previous findings: metrics based on energy (i.e., proportional to -log p ) more effectively capture the curvature of the data manifold.

## 4.4 Animal Faces

We now evaluate our method on the Animal Faces High Quality (AFHQ) dataset [59], using the latent space of the pretrained Stable Diffusion v1 V AE [18] (latent dimension: 4 × 16 × 16 ). An EBM is trained to model the distribution of latent codes, and Algo. 1 is used to compute geodesics between a cat and a dog representation. We compare the resulting paths to two baselines: (i) linear interpolation and (ii) spherical interpolation (slerp) [60], which is known to better preserve the structure of V AE latent spaces under Gaussian priors (see Supp. F.6). Full experimental details are in Supp. F.

Fig. 6 illustrates geodesics computed in the latent space of a pretrained V AE and projected back into image space (see F.5 for additional samples as well as samples for G LAND and linear interpolation). Qualitatively, we observe that geodesics computed with the G 1 / p θ metric best adhere to the data manifold. The G E θ metric also shows noticeable improvements over the other metrics. Despite extensive tuning, G RBF and G LAND produce trajectories only slightly better than linear interpolation-suggesting these parametric metrics struggle to scale in high dimensions, consistent with prior findings [11, 13].

Figure 6: Geodesics on the AFHQ dataset. Each block shows an interpolated trajectory between two animal images (cats and dogs), projected back into image space for visualization. We compare geodesics computed with two EBM-based metrics ( G 1 / p θ , G E θ ), a parametric RBF-based metric ( G RBF ), and spherical interpolation (slerp). Results using G LAND , linear interpolation, and additional examples are provided in Supp. F.5.

<!-- image -->

To quantitatively assess geodesic quality, we report FID scores [61] in Table 1, computed over 50 , 000 trajectories that interpolate from randomly chosen cat images to randomly chosen dog images. The results are consistent with qualitative observations: G 1 / p θ and G E θ yield the lowest FIDs, followed by the model-free slerp baseline, then G RBF , G LAND , and linear interpolation. Note that the FID measures how aligned individual samples are with the training distribution-on-manifold alignment-but does not assess whether the full trajectory respects the true manifold curvature. Unfortunately, AFHQ lacks ground-truth geometry for such evaluation.

## 5 Conclusion

In this work, we use pretrained Energy-Based Models (EBMs) to derive conformal Riemannian metrics, G E θ and G 1 / p θ , and we compare them to established alternatives ( G LAND [11] and

Table 1: FID along geodesics for different Riemannian metrics . FID is computed at each trajectory point to assess on-manifold alignment. See Supp. F.4 for the 2 -σ error.

| Metric         | FID ( ↓ )   |
|----------------|-------------|
| Linear interp. | 42.47       |
| Slerp interp.  | 32.67       |
| G              | 16.47       |
| 1 / p θ        |             |
| G LAND G RBF   | 39.17       |
|                | 37.98       |

G RBF [13]). On both synthetic and high-dimensional data, EBM-derived metrics yield geodesics that stay closer to the data manifold and better capture its curvature-especially with G E θ .

We focus on conformal metrics, which scale the identity by a scalar field to encode density. While more complex, non-conformal and anisotropic metrics (e.g., the Stein metric [25]) are accessible from the EBM score, we found that conformal metrics offer comparable performance with simpler interpretation and reduced computational cost, justifying our focus in this work. Future work may explore these extensions with regularization or structural priors to ensure smoothness and scalability (See Supp.G for a discussion of limitations and Supp.H for broader impact). To keep computational cost manageable, we train the EBM in the latent space of a pretrained autoencoder and compute geodesics using finite-difference optimization, two design choices that substantially reduce complexity and memory use without compromising performance.

Although this article is primarily methodological, it points to promising applications. One example is the mental rotation task, in which humans mentally rotate objects to match a target [62]. In such experiments, reaction times tend to decrease with training [63], suggesting that repeated exposure sharpens internal representations around training examples. These refined representations may concentrate in high-density regions, where mental transformations occur more quickly. As shown in Fig. 3 and 5, our geodesics naturally accelerate in such high-density regions, echoing these psychophysical findings. Modeling mental simulation as geodesics on Riemannian manifolds shaped by a generative model offers a principled computational framework to understand human cognition. It provides a way to formalize and test the hypothesis that the human cognition relies on generative models to support flexible inference [64-68]. Our approach is also particularly relevant for neuroscience, where datasets are high-dimensional, often sparsely sampled, and where understanding the geometry of neural population activity is central to scientific insight. In such settings, high-fidelity geodesics are essential for capturing the true structure of neural trajectories-approximations may distort the manifold and lead to misinterpretation of brain dynamics. While training EBMs is costly, the benefits in terms of interpretability and geometric accuracy make this approach compelling for applications where precision is critical.

As machine learning models are increasingly used to capture complex data distributions, understanding the geometry of their latent spaces becomes essential. Our work contributes to this effort by showing that geometry can serve as a useful tool for building models that better reflect data structure, align with human perception, and shed light on cognitive processes.

## Acknowledgments

This work was supported by ANR-3IA Artificial and Natural Intelligence Toulouse Institute (ANR19-PI3A-0004). Part of this work was carried out within the DEEL project, which is part of IRT Saint Exupéry and the ANITI AI cluster. The authors acknowledge the financial support from DEEL's industrial and academic members and the 'France 2030' program (NR-10-AIRT-01 and ANR-23IACL-0002). Additional support for TS provided by ONR (N00014-24-1-2026 and REPRISM MURI N00014-24-1-2603) and NSF (IIS-2402875).

## References

- [1] Raviteja Vemulapalli, Felipe Arrate, and Rama Chellappa. Human action recognition by representing 3d skeletons as points in a lie group. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 588-595, 2014.
- [2] Mehrtash T Harandi, Mathieu Salzmann, and Richard Hartley. From manifold to manifold: Geometry-aware dimensionality reduction for spd matrices. In Computer Vision-ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part II 13 , pages 17-32. Springer, 2014.
- [3] Oncel Tuzel, Fatih Porikli, and Peter Meer. Region covariance: A fast descriptor for detection and classification. In Computer Vision-ECCV 2006: 9th European Conference on Computer Vision, Graz, Austria, May 7-13, 2006. Proceedings, Part II 9 , pages 589-600. Springer, 2006.
- [4] Maximillian Nickel and Douwe Kiela. Poincaré embeddings for learning hierarchical representations. Advances in neural information processing systems , 30, 2017.
- [5] Alexandru Tifrea, Gary Bécigneul, and Octavian-Eugen Ganea. Poincar \ 'e glove: Hyperbolic word embeddings. arXiv preprint arXiv:1810.06546 , 2018.
- [6] Hongsong Feng, Sean Cottrell, Yuta Hozumi, and Guo-Wei Wei. Multiscale differential geometry learning of networks with applications to single-cell rna sequencing data. Computers in Biology and Medicine , 171:108211, 2024.
- [7] Kazuya Horibe, Gentaro Taga, and Koichi Fujimoto. Geodesic theory of long association fibers arrangement in the human fetal cortex. Cerebral Cortex , 33(17):9778-9786, 2023.
- [8] Peter D Neilson, Megan D Neilson, and Robin T Bye. A riemannian geometry theory of three-dimensional binocular visual perception. Vision , 2(4):43, 2018.
- [9] Manfredo Perdigao Do Carmo and J Flaherty Francis. Riemannian geometry , volume 2. Springer, 1992.
- [10] Søren Hauberg, Oren Freifeld, and Michael Black. A geometric take on metric learning. Advances in Neural Information Processing Systems , 25, 2012.
- [11] Georgios Arvanitidis, Lars K Hansen, and Søren Hauberg. A locally adaptive normal distribution. Advances in Neural Information Processing Systems , 29, 2016.
- [12] Johann Brehmer and Kyle Cranmer. Flows for simultaneous manifold learning and density estimation. Advances in neural information processing systems , 33:442-453, 2020.
- [13] Georgios Arvanitidis, Søren Hauberg, and Bernhard Schölkopf. Geometrically enriched latent spaces. arXiv preprint arXiv:2008.00565 , 2020.
- [14] Samuel Gruffaz and Josua Sassen. Riemannian metric learning: Closer to you than you imagine. arXiv preprint arXiv:2503.05321 , 2025.
- [15] Guy Lebanon. Learning riemannian metrics. arXiv preprint arXiv:1212.2474 , 2012.
- [16] Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. Advances in neural information processing systems , 32, 2019.
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems , 33:6840-6851, 2020.
- [18] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 10684-10695, 2022.
- [19] Ruslan Salakhutdinov and Geoffrey Hinton. Deep boltzmann machines. In Artificial intelligence and statistics , pages 448-455. PMLR, 2009.
- [20] Yang Song and Diederik P Kingma. How to train your energy-based models. arXiv preprint arXiv:2101.03288 , 2021.

- [21] C Radhakrishna Rao. Information and the accuracy attainable in the estimation of statistical parameters. In Breakthroughs in Statistics: Foundations and basic theory , pages 235-247. Springer, 1992.
- [22] Shun-Ichi Amari. A foundation of information geometry. Electronics and Communications in Japan (Part I: Communications) , 66(6):1-10, 1983.
- [23] Ronald A Fisher. On the mathematical foundations of theoretical statistics. Philosophical transactions of the Royal Society of London. Series A, containing papers of a mathematical or physical character , 222(594-604):309-368, 1922.
- [24] Kacper Kapusniak, Peter Potaptchik, Teodora Reu, Leo Zhang, Alexander Tong, Michael Bronstein, Joey Bose, and Francesco Di Giovanni. Metric flow matching for smooth interpolations on the data manifold. Advances in Neural Information Processing Systems , 37: 135011-135042, 2025.
- [25] Christian S. Perone. The geometry of data: the missing metric tensor and the stein score [part ii]. https://blog.christianperone.com/2024/11/ the-geometry-of-data-part-ii/ , November 2024. Terra Incognita.
- [26] Willem Diepeveen, Georgios Batzolis, Zakhar Shumaylov, and Carola-Bibiane Schönlieb. Score-based pullback riemannian geometry. arXiv preprint arXiv:2410.01950 , 2024.
- [27] Georgios Arvanitidis, Miguel González-Duque, Alison Pouplin, Dimitris Kalatzis, and Søren Hauberg. Pulling back information geometry. arXiv preprint arXiv:2106.05367 , 2021.
- [28] Hadi Beik-Mohammadi, Søren Hauberg, Georgios Arvanitidis, Gerhard Neumann, and Leonel Rozo. Learning riemannian manifolds for geodesic motion skills. arXiv preprint arXiv:2106.04315 , 2021.
- [29] Dimitris Kalatzis, David Eklund, Georgios Arvanitidis, and Søren Hauberg. Variational autoencoders with riemannian brownian motion priors. arXiv preprint arXiv:2002.05227 , 2020.
- [30] Xingzhi Sun, Danqi Liao, Kincaid MacDonald, Yanlei Zhang, Guillaume Huguet, Guy Wolf, Ian Adelstein, Tim GJ Rudner, and Smita Krishnaswamy. Geometry-aware generative autoencoder for warped riemannian metric learning and generative modeling on data manifolds. In The 28th International Conference on Artificial Intelligence and Statistics , 2025.
- [31] Georgios Arvanitidis, Soren Hauberg, Philipp Hennig, and Michael Schober. Fast and robust shortest paths on manifolds learned from data. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 1506-1515. PMLR, 2019.
- [32] Nutan Chen, Francesco Ferroni, Alexej Klushyn, Alexandros Paraschos, Justin Bayer, and Patrick van der Smagt. Fast approximate geodesics for deep generative models. In Artificial Neural Networks and Machine Learning-ICANN 2019: Deep Learning: 28th International Conference on Artificial Neural Networks, Munich, Germany, September 17-19, 2019, Proceedings, Part II 28 , pages 554-566. Springer, 2019.
- [33] Georgios Arvanitidis, Lars Kai Hansen, and Søren Hauberg. Latent space oddity: on the curvature of deep generative models. In International Conference on Learning Representations , 2018.
- [34] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t .
- [35] Valentin De Bortoli, Guan-Horng Liu, Tianrong Chen, Evangelos A Theodorou, and Weilie Nie. Augmented bridge matching. arXiv preprint arXiv:2311.06978 , 2023.
- [36] Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, and Can Yang. Deep generative learning via schrodinger bridge. In International conference on machine learning , pages 10794-10804. PMLR, 2021.

- [37] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrodinger bridge matching. Advances in Neural Information Processing Systems , 36:6218362223, 2023.
- [38] Ricky TQ Chen and Yaron Lipman. Flow matching on general geometries. In The Twelfth International Conference on Learning Representations , 2024.
- [39] Friso de Kruiff, Erik Bekkers, Ozan Öktem, Carola-Bibiane Schönlieb, and Willem Diepeveen. Pullback flow matching on data manifolds. arXiv preprint arXiv:2410.04543 , 2024.
- [40] Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, and Arnaud Doucet. Riemannian score-based generative modelling. Advances in neural information processing systems , 35:2406-2422, 2022.
- [41] James Thornton, Michael Hutchinson, Emile Mathieu, Valentin De Bortoli, Yee Whye Teh, and Arnaud Doucet. Riemannian diffusion schrodinger bridge. arXiv preprint arXiv:2207.03024 , 2022.
- [42] Gabriel Peyre, Mickael Pechaud, Renaud Keriven, Laurent D Cohen, et al. Geodesic methods in computer vision and graphics. Foundations and Trends® in Computer Graphics and Vision , 5(3-4):197-397, 2010.
- [43] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International conference on machine learning , pages 1530-1538. PMLR, 2015.
- [44] Zhifeng Kong and Kamalika Chaudhuri. The expressive power of a class of normalizing flow models. In International conference on artificial intelligence and statistics , pages 3599-3609. PMLR, 2020.
- [45] Felix Draxler, Peter Sorrenson, Lea Zimmermann, Armand Rousselot, and Ullrich Köthe. Free-form flows: Make any architecture a normalizing flow. In International Conference on Artificial Intelligence and Statistics , pages 2197-2205. PMLR, 2024.
- [46] Polina Kirichenko, Pavel Izmailov, and Andrew G Wilson. Why normalizing flows fail to detect out-of-distribution data. Advances in neural information processing systems , 33:20578-20589, 2020.
- [47] Keegan Kelly, Lorena Piedras, Sukrit Rao, and David Roth. Variations and relaxations of normalizing flows. arXiv preprint arXiv:2309.04433 , 2023.
- [48] Yilun Du, Conor Durkan, Robin Strudel, Joshua B Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc. In International conference on machine learning , pages 8489-8510. PMLR, 2023.
- [49] James Thornton, Louis Béthune, Ruixiang ZHANG, Arwen Bradley, Preetum Nakkiran, and Shuangfei Zhai. Controlled generation with distilled diffusion energy models and sequential monte carlo. In The 28th International Conference on Artificial Intelligence and Statistics , 2025.
- [50] Yilun Du, Toru Lin, and Igor Mordatch. Model-based planning with energy-based models. In Conference on Robot Learning , pages 374-383. PMLR, 2020.
- [51] Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural computation , 14(8):1771-1800, 2002.
- [52] Oliver Woodford. Notes on contrastive divergence. Department of Engineering Science, University of Oxford, Tech. Rep , 4, 2006.
- [53] Erik Nijkamp, Mitch Hill, Tian Han, Song-Chun Zhu, and Ying Nian Wu. On the anatomy of mcmc-based maximum likelihood learning of energy-based models. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pages 5272-5280, 2020.

- [54] David Duvenaud, Jacob Kelly, Kevin Swersky, Milad Hashemi, Mohammad Norouzi, and Will Grathwohl. No mcmc for me: Amortized samplers for fast and stable training of energy-based models. In International Conference on Learning Representations (ICLR) , 2021.
- [55] Zhisheng Xiao, Karsten Kreis, Jan Kautz, and Arash Vahdat. Vaebm: A symbiosis between variational autoencoders and energy-based models. In International Conference on Learning Representations , 2021.
- [56] Bo Pang, Tian Han, Erik Nijkamp, Song-Chun Zhu, and Ying Nian Wu. Learning latent space energy-based prior model. Advances in Neural Information Processing Systems , 33: 21994-22008, 2020.
- [57] Søren Hauberg. Only bayes should learn a manifold (on the estimation of differential geometric structure from data). arXiv preprint arXiv:1806.04994 , 2018.
- [58] Georgios Arvanitidis, Bogdan M Georgiev, and Bernhard Schölkopf. A prior-based approximate latent riemannian metric. In International Conference on Artificial Intelligence and Statistics , pages 4634-4658. PMLR, 2022.
- [59] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2020.
- [60] Karpathy Andrej. Walk in stable diffusion. https://gist.github.com/karpathy/ 00103b0037c5aaea32fe1da1af553355 , 2022.
- [61] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems , 30, 2017.
- [62] Roger N Shepard and Jacqueline Metzler. Mental rotation of three-dimensional objects. Science , 171(3972):701-703, 1971.
- [63] Lynn A Cooper and Roger N Shepard. Chronometric studies of the rotation of mental images. In Visual information processing , pages 75-176. Elsevier, 1973.
- [64] Karl Friston. A free energy principle for the brain. Journal of Physiology-Paris , 100(1-3): 70-87, 2006.
- [65] Rajesh PN Rao and Dana H Ballard. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature neuroscience , 2(1):79-87, 1999.
- [66] Victor Boutin, Angelo Franciosini, Frédéric Chavane, and Laurent U Perrinet. Pooling strategies in v1 can account for the functional and structural diversity across species. PLOS Computational Biology , 18(7):e1010270, 2022.
- [67] Victor Boutin, Lakshya Singhal, Xavier Thomas, and Thomas Serre. Diversity vs. recognizability: Human-like generalization in one-shot generative models. Advances in Neural Information Processing Systems , 35:20933-20946, 2022.
- [68] Victor Boutin, Rishav Mukherji, Aditya Agrawal, Sabine Muzellec, Thomas Fel, Thomas Serre, and Rufin Van-Rullen. Latent representation matters: Human-like sketches in one-shot drawing tasks. In 38th Conference on Neural Information Processing Systems (NeurIPS) , 2024.
- [69] Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics , 22(1):79-86, 1951.
- [70] Shun-Ichi Amari. Natural gradient works efficiently in learning. Neural computation , 10(2): 251-276, 1998.
- [71] Razvan Pascanu and Yoshua Bengio. Revisiting natural gradient for deep networks. arXiv preprint arXiv:1301.3584 , 2013.

- [72] Richard D Lange, Devin Kwok, Jordan Kyle Matelsky, Xinyue Wang, David Rolnick, and Konrad Kording. Deep networks as paths on the manifold of neural representations. In Topological, Algebraic and Geometric Learning Workshops 2023 , pages 102-133. PMLR, 2023.
- [73] Gunnar Carlsson. Topology and data. Bulletin of the American Mathematical Society , 46(2): 255-308, 2009.
- [74] Afra Zomorodian. Topological data analysis. Advances in applied and computational topology , 70(1-39):19, 2012.
- [75] Olympio Hacquard and Vadim Lebovici. Euler characteristic tools for topological data analysis. Journal of Machine Learning Research , 25(240):1-39, 2024.
- [76] Peter Bubenik et al. Statistical topological data analysis using persistence landscapes. J. Mach. Learn. Res. , 16(1):77-102, 2015.
- [77] Afra Zomorodian and Gunnar Carlsson. Computing persistent homology. In Proceedings of the twentieth annual symposium on Computational geometry , pages 347-356, 2004.
- [78] Herbert Edelsbrunner and Dmitriy Morozov. Persistent homology: theory and practice . eScholarship, University of California, 2013.
- [79] Nina Otter, Mason A Porter, Ulrike Tillmann, Peter Grindrod, and Heather A Harrington. A roadmap for the computation of persistent homology. EPJ Data Science , 6:1-38, 2017.
- [80] Taco Cohen and Max Welling. Group equivariant convolutional networks. In International conference on machine learning , pages 2990-2999. PMLR, 2016.
- [81] Taco S Cohen and Max Welling. Steerable cnns. In International Conference on Learning Representations , 2017.
- [82] Taco Cohen, Maurice Weiler, Berkay Kicanaoglu, and Max Welling. Gauge equivariant convolutional networks and the icosahedral cnn. In International conference on Machine learning , pages 1321-1330. PMLR, 2019.
- [83] Marc Finzi, Samuel Stanton, Pavel Izmailov, and Andrew Gordon Wilson. Generalizing convolutional neural networks for equivariance to lie groups on arbitrary continuous data. In International Conference on Machine Learning , pages 3165-3176. PMLR, 2020.
- [84] Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E (n) equivariant graph neural networks. In International conference on machine learning , pages 9323-9332. PMLR, 2021.
- [85] Rumen Dangovski, Li Jing, Charlotte Loh, Seungwook Han, Akash Srivastava, Brian Cheung, Pulkit Agrawal, and Marin Soljacic. Equivariant self-supervised learning: Encouraging equivariance in representations. In International Conference on Learning Representations , 2022.
- [86] Cédric Rommel, Thomas Moreau, and Alexandre Gramfort. Deep invariant networks with differentiable augmentation layers. Advances in Neural Information Processing Systems , 35: 35672-35683, 2022.
- [87] Jianke Yang, Robin Walters, Nima Dehmamy, and Rose Yu. Generative adversarial symmetry discovery. In International Conference on Machine Learning , pages 39488-39508. PMLR, 2023.
- [88] Moshe Lichtenstein, Gautam Pai, and Ron Kimmel. Deep eikonal solvers. In Scale Space and Variational Methods in Computer Vision: 7th International Conference, SSVM 2019, Hofgeismar, Germany, June 30-July 4, 2019, Proceedings 7 , pages 38-50. Springer, 2019.
- [89] Qijian Zhang, Junhui Hou, Yohanes Adikusuma, Wenping Wang, and Ying He. Neurogf: A neural representation for fast geodesic distance and path queries. Advances in Neural Information Processing Systems , 36:19485-19501, 2023.

- [90] Louis Béthune, Paul Novello, Guillaume Coiffier, Thibaut Boissin, Mathieu Serrurier, Quentin Vincenot, and Andres Troya-Galvis. Robust one-class classification with signed distance function using 1-lipschitz neural networks. In International Conference on Machine Learning , pages 2245-2271. PMLR, 2023.
- [91] Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR'05) , volume 1, pages 539-546. IEEE, 2005.
- [92] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 815-823, 2015.
- [93] Kihyuk Sohn. Improved deep metric learning with multi-class n-pair loss objective. Advances in neural information processing systems , 29, 2016.
- [94] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.
- [95] Zengyi Li, Yubei Chen, and Friedrich T Sommer. Learning energy-based models in highdimensional spaces with multiscale denoising-score matching. Entropy , 25(10):1367, 2023.
- [96] Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, and Richard Zemel. Learning the stein discrepancy for training and evaluating energy-based models without sampling. In International Conference on Machine Learning , pages 3732-3747. PMLR, 2020.
- [97] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation , 23(7):1661-1674, 2011.
- [98] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. In Uncertainty in artificial intelligence , pages 574-584. PMLR, 2020.
- [99] Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the thirteenth international conference on artificial intelligence and statistics , pages 297-304. JMLR Workshop and Conference Proceedings, 2010.
- [100] Tobias Schroder, Zijing Ou, Jen Lim, Yingzhen Li, Sebastian Vollmer, and Andrew Duncan. Energy discrepancies: a score-independent loss for energy-based models. Advances in Neural Information Processing Systems , 36:45300-45338, 2023.
- [101] Elad Hoffer and Nir Ailon. Deep metric learning using triplet network. In Similarity-based pattern recognition: third international workshop, SIMBAD 2015, Copenhagen, Denmark, October 12-14, 2015. Proceedings 3 , pages 84-92. Springer, 2015.
- [102] Partha Ghosh, Mehdi S. M. Sajjadi, Antonio Vergari, Michael Black, and Bernhard Scholkopf. From variational to deterministic autoencoders. In International Conference on Learning Representations , 2020. URL https://openreview.net/forum?id=S1g7tpEYDS .
- [103] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems , 34:8780-8794, 2021.
- [104] Barrett O'neill. Semi-Riemannian geometry with applications to relativity , volume 103. Academic press, 1983.
- [105] Joel W Robbin and Dietmar A Salamon. Introduction to differential geometry . Springer Nature, 2022.

## A Extended related work

Several tools have been developed to study the geometrical properties of distributions. We survey some prominent approaches below.

Information geometry , initiated by the seminal works of [21, 22], was the first to apply ideas from differential geometry to the field of statistics. Unlike our present work, the goal was not to understand the geometry of the data x , but rather to understand the geometry of a smooth manifold θ ∈ Θ of parameters of an estimator p θ . In particular, starting from the Taylor expansion of reverse Kullback-Leibler [69] divergence to p θ , in the neighborhood of p θ itself, with θ ′ = θ + ϵ , we get

<!-- formula-not-decoded -->

One can show that, since θ ′ = θ is the global minimum of this function, the first-order term vanishes and the second-order term ∇ 2 1 D KL ( p θ ∥ p θ ) must be a positive definite form - i.e an inner product. This quantity, called Fisher information [23], gives Θ the structure of a Riemannian manifold. The Riemannian gradient associated with this manifold yields a second-order optimization method coined natural gradient descent [70], that has been proved helpful in deep learning [71]. Our method inherits some spirit of this approach, since we define a local inner product as a function of the density, to give a Riemannian structure to the data manifold. However, we focus on the geometry of the data x , not the geometry of the model's parameters θ .

Riemannian structure of data manifolds has already been proposed in the past. For example, the seminal LAND metric [11] is a non-conformal metric built from the samples, with the intent of generalizing multivariate normal distributions to manifolds. The RBF metric [13] is a conformal metric, derived from a kernel density estimator, with some learnable coefficients. More recently, Kapusniak et al. [24] proposed to use those metrics and learn a flow matching algorithm to fit geodesics in the data manifold. The Jacobian of a generative model also defines a metric [33]. The (unpublished) work of Perone [25] has been inspirational for our contribution. They use the Stein score function to build the metric, an approach also chosen by [26] - although restricted to unimodal densities.

Pullback geometry of latent manifolds is an active research area. [72] studies the manifold of representations of a given network, while [30] builds a generative autoencoder to represent the manifold. Shortest paths are computed with fixed-point methods [31], or using a discrete graph [32]. While we may rely on the latent space of a VAE for some challenging tasks, studying latent representations of a neural network is beyond the scope of our work.

On-manifold generative models can be found in the literature. For example, we can mention flow and bridge matching approaches [34, 35], which learn a flow between a source and a target distribution, including on Riemannian manifolds [38, 39]. In particular, the Schrödinger bridge [36, 37] focuses on an optimization problem involving paths in the space of probability distributions, and was also generalized to non-Euclidean geometries [40, 41]. These works differ significantly from ours: they assume the Riemannian manifold to be given, not chosen, and they build a generative model on top of it. To the contrary, given a special class of generative models to represent the data, we choose the metric to build the manifold.

Topological data analysis [73, 74] studies the topological properties of the data manifold. This field aims to estimate some topological invariants such as the Euler characteristic [75] and persistent Betti numbers [76] (which are the number of connected components, number of closed loops, etc.) from a finite sample. It relies on tools such as persistent homology [77-79] to design algorithms. This approach typically focuses on the global properties: it assumes that the data accumulate on a well-defined manifold, from which these high-level features must be computed. To the contrary, our approach focuses on the local structure defined by the metric, while the global structure is inherited from the induced geodesics. Furthermore, we consider the whole ambient space for our manifold, tweaking only the metric to account for low-density regions.

Symmetries and geometry in representations have gathered considerable attention from the deep learning community, warranting no fewer than 3 workshops at Neurips alone 4 . Symmetries are

4 https://www.neurreps.org

operations under which a structure is left invariant, or equivariant. In particular, some neural architectures are leveraged to reflect priors about the underlying symmetries of the data [80-84]. In other cases, symmetries are discovered and learned from observations [85-87]. Unlike these approaches, we do not seek symmetries in data, and we make minimal assumptions about the model; we are mainly interested in the density to build the structure.

Non-Euclidean 2D and 3D manifolds are first-class citizens in computer graphics. The works of [88, 89] define a way to find shortest paths over such manifolds. However, this requires solving the Eikonal equation, which is prohibitively expensive in high dimensions or restricted to Euclidean geometries [90]. Geodesics can be learned, but this is restricted to low dimensions [89]. These setups are beyond the scope of our work, as we focus on higher-dimensional and sparsely populated spaces, and no discrete meshes can be built from samples.

Metric learning (or distance learning ) is another field whose purpose is to learn a distance function between samples, typically in a weakly-supervised manner with contrastive losses [91-93]. Often, these distances cannot be realized as a geodesic distance and are intended for a specific task, like classification or retrieval.

## B Energy-Based Model

## B.1 Derivation of the Gradient of the EBM Log-Likelihood

The demonstration below is adapted from [52] to fit our notation. Even though this mathematical derivation is not crucial for a good understanding of our work, we include it to make sure our article is self-contained and complete.

We consider an Energy-Based Model (EBM) defining a probability distribution via the Boltzmann form:

<!-- formula-not-decoded -->

Our goal is to minimize the negative log-likelihood with respect to the empirical data distribution p D :

<!-- formula-not-decoded -->

We first expand the log-probability:

<!-- formula-not-decoded -->

Taking the gradient with respect to θ :

<!-- formula-not-decoded -->

The derivative of the log-partition function could be simplified:

<!-- formula-not-decoded -->

Substituting this back into the gradient of the loss:

<!-- formula-not-decoded -->

In practice, we denote the x + the "positive" samples from the empirical data distribution p D , and x -the "negative" samples from the model:

<!-- formula-not-decoded -->

## B.2 EBMtraining algorithm

To train our Energy-Based Models (EBMs), we follow the approach of [16]. Algo. 2 details the general training procedure:

## Algorithm 2: Training Energy-Based Model using Langevin Dynamics

Input: Training dataset : D , learning rate η , Replay Buffer B , Langevin step size α , noise scale σ , number of Langevin steps L

## while Training do

```
x + ∼ D sample from the dataset x 0 ∼ B # sample from a replay buffer with probability 95% ## Refine negative samples using Langevin dynamics for t ← 1 to L do x t +1 ← x t -α ∇ x t E θ ( x t ) + ω with ω ∼ N (0 , σ ) x -= x L .detach() ∇ θ L ML ≈ E Batch [ ∇ θ E θ ( x + i ) -∇ θ E θ ( x -i ) ] ## Compute the ML loss L REG ( θ ) = E Batch [ ∇ θ E θ ( x + i ) 2 + ∇ θ E θ ( x -i ) 2 ] ## Compute Regularization loss θ ← θ -η ∇ θ L ML -η ∇ θ L REG ## update parameters with gradient descent B ← B ∪ x +
```

In all experiments, we use L = 100 Langevin steps with step size α = 1 and noise scale σ = 10 -2 . The energy function is optimized using the Adam optimizer [94] with a learning rate of η = 10 -4 . In addition to the maximum likelihood (ML) loss, we include a regularization term that encourages the energy values to remain close to zero, a technique shown to be effective in prior work [16].

We observed that training can be unstable, particularly for high-dimensional datasets. We attribute this instability to the lack of gradient supervision: the loss is not backpropagated through the Langevin dynamics to reduce memory usage. To mitigate this, we introduce a small Denoising Score Matching (DSM) loss-only for the AFHQ dataset-which provides weak supervision of the energy gradient. This additional regularization loss is similar to the DSM loss in [95]. We found this trick to strongly improve stability without degrading the performance.

The energy network architecture is adapted to the complexity of each dataset. Full details are provided in Appendix D.2, E.3, and F.2. Following Li et al. [95], we design the output layer of the energy function to take a quadratic form.

## B.3 Other training procedure in literature

EBM can also be trained by minimizing the so-called Stein discrepancy [96], Denoising Score Matching [97], Sliced Score Matching [98], Noise Contrastive Estimation [99]. A related objective to contrastive divergence is energy discrepancy [100]. We refer the reader [20], for a complete review of the different methods to train EBMs.

## C Riemannian Metrics

## C.1 Calibration

We normalize each metric using calibration coefficients α and β , with two goals: (i) ensuring that the Riemannian metric averages to the identity matrix I on the manifold, and (ii) aligning the overall scale of all metrics to allow fair comparisons. Here are more details on the calibration procedure:

First, we randomly sample data pairs ( x 0 , x 1 ) from the dataset D (it corresponds to the geodesics endpoints) and generate linear interpolations between them using:

<!-- formula-not-decoded -->

Second, we define two sets of samples: S M , which contains the endpoints x 0 and x 1 lying on the data manifold , and S ¯ M , which contains the midpoints at t = 1 2 . These sets are then used to estimate the calibration coefficients α and β :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This calibration strategy adjusts the metric based on both on-manifold and off-manifold regions. It ensures that all metrics operate within a comparable dynamic range and promotes a useful geometric prior: lower metric values near the data manifold and higher values farther away. As a result, geodesics are encouraged to stay close to high-density areas, aligning the geometry with the data distribution.

## C.2 LAND metric

Figure 7: Effect of the bandwidth σ on the geodesics obtained with the LAND metric. Here we have explored five σ values ( σ ∈ { 0 . 1 , 0 . 25 , 0 . 5 , 1 , 5 } ). We observed that σ has a major impact on the shape of the geodesics.

<!-- image -->

We remind the land metric formula (see Eq. 8):

<!-- formula-not-decoded -->

This metric is highly sensitive to the choice of the σ parameter, which controls the "locality" of the metric. A small σ results in a very local metric that is strongly influenced by nearby points, while a large σ smooths the metric by averaging over a wider region. This directly affects the trade-off between how closely geodesics follow the data manifold and how smooth or stable they are. In practice, we observe that σ has a major impact on the shape of the geodesics, as shown in Fig.7, confirming earlier findings by[11]. To illustrate this, we plot geodesics for five different values of σ ( σ ∈ { 0 . 1 , 0 . 25 , 0 . 5 , 1 , 5 } ) and find that they closely follow the data manifold only within a narrow range, particularly around σ = 0 . 5 .

## C.3 RBF metric

We first remind the RBF formula :

<!-- formula-not-decoded -->

In the equation, the { ˆ x } K i =1 are centroids evaluated using a K-Means algorithm. Following [13], the bandwidth ( λ k ) using the inter-distance to prototype (see Eq. 13):

<!-- formula-not-decoded -->

The bandwidth, λ k , controls the spatial extent of each radial basis function. In Eq. 13, κ is a tunable hyperparameter controlling how concentrated or spread out the RBFs are. Intuitively, a larger κ results in narrower kernels (stronger locality) while a smaller one yields wider coverage. This trade-off is explored via hyperparameter search. The weights w k modulate the relative contribution of each RBF to the resulting scalar field. These weights are optimized to ensure that h ( x ) remains close to 1 on the training data, using the following loss:

<!-- formula-not-decoded -->

This encourages the RBF combination to approximate a constant value (here, 1) across the data distribution, ensuring consistency and stability of the field on the manifold.

In Fig. 8, we evaluate how the number of centroids K affects the shape of the geodesics. The results show that geodesics are highly sensitive to this parameter. When K is too small, the geodesics fail to follow the data manifold accurately. Conversely, when K is too large, the trajectories become overly sinuous-passing through many centroids that are not necessarily aligned with the true manifold.

Figure 8: Effect of the number of centroids K on the geodesics obtained with the RBF metric ( K ∈ { 10 , 50 , 100 } ). We observed that K has a major impact on the shape of the geodesics.

<!-- image -->

## D Experimental details on the Circular Mixture of Gaussian datasets

## D.1 Datasets

To design our toy datasets, we have used a mixture of K (2D) Gaussians. Specifically, K = 200 in all our datasets. The resulting probability distribution is therefore:

<!-- formula-not-decoded -->

where N ( x | µ k , I ) denotes a 2D isotropic Gaussian centered at µ k . Here, I is the identity matrix of size 2 × 2 . In both datasets, the centers of the Gaussians are uniformly positioned along a semi-circle or Radius R (here R = 8 ). Specifically, the centers are given by:

<!-- formula-not-decoded -->

The only difference between the Uniform Circular Gaussian (UCG) dataset and the Weighted Circular Gaussian dataset (WCG) is the weighting coefficient { π k } K k =1

Uniform Circular Gaussian dataset. Here all the weights are similar and equal to 1 /K . As a result, the energy landscape forms a semicircular basin with constant depth (see contour plot of Fig. 2a for an illustration of the energy landscape).

Weighted Circular Gaussian dataset. In this setting, the mixture weights vary, concentrating the distribution toward the center of the arc. The weights are symmetric with respect to the horizontal axis, producing an energy landscape with a semi-circular shape and slopes symmetric around the arc's midpoint (see the contour plot in Fig.2c). Fig.9 shows the weights π k as a function of orientation, with all weights summing to 1. This setup generates a curved, non-uniform density with higher mass near the center of curvature (i.e., at 90 degrees), allowing us to introduce a controlled curvature in the data manifold and assess how well different metrics capture it.

Figure 9: Profile of the Gaussians weights π k

<!-- image -->

## D.2 Neural networks architectures and Hyperparameters on the Circular Mixture of Gaussian Dataset

Here, we describe the architecture of the energy function (see Table 2), the interpolant network (see Table 3), and the hyperparameters used for the G LAND and G RBF metrics. Note that the architectures and settings are the same for both the UCG and WCG datasets.

Energy-Based Model Table 2 summarizes the architecture used for the energy function of the EBM. The output is designed to follow a quadratic form, similar to the approach in [95], which we found improves performance across all datasets. To assess whether the EBM successfully learns the target distribution, we visualize the learned energy landscapes for both the UCG and WCG datasets (see Fig. 10a and Fig. 10b, respectively). For reference, we also include the ground-truth energy landscapes of the target distributions (see Fig. 10c and Fig. 10d for UCG and WCG, respectively). We observe that the EBM accurately captures the overall shape of the energy landscape for both distributions. However, in the WCG dataset, the true energy spans a broader range than the EBM's learned energy. This discrepancy is partially corrected by the normalization procedure described in Appendix C.1.

Figure 10: Energy Landscape on the UCG and WCG datasets. (a, b) shows the energy landscape learned by the EBMs on the UCG and WCG datasets, respectively. (c, d) shows the true energy landscape (i.e., -log p M ) on the UCG and WCG datasets, respectively.

<!-- image -->

Interpolant Network Table 3 summarizes the architecture used for the interpolant network (i.e., φ t,η in Algo. 1 and Eq. 3). For all datasets, we use an autoencoder-like architecture for the interpolant, following a similar approach to [24].

Table 2: MLP architecture of the energy function on both UCG and WCG datasets.

| Nb. Layers   | Layer type                                                                               |
|--------------|------------------------------------------------------------------------------------------|
| 1            | Linear (2, 32) SiLU                                                                      |
| 4            | Linear (32, 32) SiLU                                                                     |
| 1            | Linear (32, 32)                                                                          |
| 1            | Three output heads: Linear (32, 1) for f 1 Linear (32, 1) for f 2 Linear (32, 1) for f 3 |
| output       | f 1 ( x ) · f 2 ( x )+ f 3 ( x 2 )                                                       |

Table 3: MLP architecture of the interpolant network φ t,η for WCG dataset.

|   NB. Layers | Layer type           |
|--------------|----------------------|
|            1 | Linear (3, 32) SiLU  |
|            1 | Linear (32, 64) SiLU |
|            1 | Linear (64, 64) SiLU |
|            1 | Linear (64, 32) SiLU |
|            1 | Linear (32, 3)       |

LAND metric We performed a hyperparameter search to tune the σ parameter. We found that σ = 1 yielded the best performance. Parameters are similar for both UCG and WCG.

RBF metric We conducted a hyperparameter search to tune both the number of centroids K and the scaling factor κ . The best results were obtained with K = 30 and κ = 1 . Parameters are similar for both UCG and WCG.

## D.3 Quantitative evaluation with error bars

In Fig. 11, we report the same quantitative results as in Fig. 2, now including 2 -σ error bars. The standard deviation σ is computed over evaluation metrics, each averaged on a different set of randomly sampled trajectories (five sets in total).

## a) Geodesics evaluation on UCG

## b) Geodesics evaluation on WCG

Figure 11: Quantitative evaluation of the geodesics on the UCG and WCG datasets. We report (i) the accumulated probability along the geodesic (the higher the better) and ii) RMSE between each geodesic and its corresponding baseline (the lower the better). Values after the ± sign indicate the 2 -σ error.

| Metric    | p M ( γ ⋆ ) ( ↑ )   | RMSE ( ↓ )      | Metric    | p M ( γ ⋆ ) ( ↑ )   | RMSE ( ↓ )      |
|-----------|---------------------|-----------------|-----------|---------------------|-----------------|
| G E M     | 0 . 79 ± 0 . 02     | -               | G E M     | 0 . 67 ± 0 . 05     | -               |
| G 1 /p M  | 0 . 77 ± 0 . 04     | -               | G 1 /p M  | 0 . 73 ± 0 . 07     | -               |
| G E θ     | 0 . 78 ± 0 . 03     | 0 . 12 ± 0 . 02 | G E θ     | 0 . 67 ± 0 . 06     | 0 . 18 ± 0 . 07 |
| G 1 / p θ | 0 . 73 ± 0 . 01     | 0 . 10 ± 0 . 03 | G 1 / p θ | 0 . 67 ± 0 . 09     | 0 . 14 ± 0 . 06 |
| G LAND    | 0 . 60 ± 0 . 07     | 0 . 38 ± 0 . 05 | G LAND    | 0 . 65 ± 0 . 11     | 0 . 34 ± 0 . 05 |
| G RBF     | 0 . 61 ± 0 . 06     | 0 . 39 ± 0 . 1  | G RBF     | 0 . 47 ± 0 . 14     | 2 . 2 ± 0 . 1   |

## E Experimental details on the Rotated Character Dataset

## E.1 Datasets

The Rotated Character Datasets consist of 7 printed characters (5, G, F, P, J, 7, 2), represented as black-and-white images of size 32 × 32 . These characters were selected for two main reasons: (i) they are commonly used in psychophysics experiments [63], and (ii) they are asymmetric and visually distinct, which helps avoid ambiguities in the resulting geodesic trajectories. Fig. 12 shows all characters in their unrotated form.

Figure 12: Original (non-rotated) samples from the Rotated Character Dataset

<!-- image -->

The only difference between the Uniform Rotated Character (URC) and Biased Rotated Character (BRC) datasets lies in the distribution of character orientations.

Uniform Rotated Character (URC) In this setting, character orientations are sampled uniformly across the full range of [ -179 ◦ , 180 ◦ ] , using a one-degree step. This ensures that each possible orientation within this interval is equally likely. Importantly, the distribution is consistent across all characters, meaning that each character appears with the same uniform spread of rotations.

Biased Rotated Character (BRC) Here, orientations follow a truncated Gaussian distribution centered at 0 ◦ , designed to mimic natural rotation statistics (see Fig. 13). Unlike the Mixture of Gaussian datasets, we do not have access to a closed-form expression for the underlying

Figure 13: Distribution of orientation for the BRC dataset

<!-- image -->

distribution p M , but we do control its empirical form. This setup introduces a controlled curvature in the data manifold, allowing us to assess how well different metrics adhere to it.

## E.2 Architecture and algorithm of the Triplet Loss autoencoder

We computed geodesics in the latent space of an autoencoder trained with a Triplet Loss [101]. This approach is motivated by the fact that image space is inherently non-Euclidean, making it poorly suited for defining meaningful distances. In contrast, the latent space of our autoencoder is explicitly regularized so that Euclidean distances correspond to differences in orientation. By

<!-- image -->

Input:

Dataset D = { ( x i , θ i ) } , Encoder E ϕ , Decoder D ψ

while

training do

treating the latent space as the ambient space for geodesic computation, we align with the assumption that the data manifold is embedded in an Euclidian Manifold. The training procedure is described in

Algo. 3, and the encoder and decoder architectures-based on the Regularized Autoencoder (RAE) framework [102]-are detailed in Table 4 and Table 5, respectively.

We trained the model using the Adam optimizer [94] with a learning rate of 1 × 10 -4 and a batch size of 128. In Algorithm 3, we set α = 1 and λ = 0 . 1 . For the architecture, the number of input features (i.e., the number of channels in the first convolutional layer) was set to F = 128 . In Table 4 and Table 5, the notation "Conv2D( n c , n f , 3, 1)" refers to a convolutional layer with n c input channels, n f output channels, a kernel size of 3, and padding of 1. Similarly, "ConvTr2D" denotes a transposed convolution. The RAE blocks are modules introduced in [102], referred to here as RaeBlockDown and RaeBlockUp, and are used for efficient downsampling and upsampling, respectively.

Table 4: Encoder architecture of the autoencoder. F is the number of features ( F = 128 ), and z is the size of the latent space ( z = 64 ).

|   Nb. Layers | Layer Type                      |
|--------------|---------------------------------|
|            1 | Conv2d (1, F , 3, 1)            |
|            1 | RaeBlockDown ( F , 2 F ) ReLU   |
|            1 | Conv2d ( 2 F , 2 F , 3, 1)      |
|            1 | RaeBlockDown ( 2 F , 4 F ) ReLU |
|            1 | Conv2d ( 4 F , 4 F , 3, 1) ReLU |
|            1 | Linear ( 4 F ∗ 8 ∗ 8 ,z)        |

Table 5: Decoder architecture of the autoencoder. F is the number of features ( F = 128 ), and z is the size of the latent space ( z = 64 ).

|   Nb. Layers | Layer Type                      |
|--------------|---------------------------------|
|            1 | ConvTr2d ( z , 4 F , 8, 0) ReLU |
|            1 | Conv2d ( 4 F , 4 F , 3, 1)      |
|            1 | RaeBlockUp ( 4 F , 2 F ) ReLU   |
|            1 | Conv2d ( 2 F , 2 F , 3, 1)      |
|            1 | RaeBlockUp ( 2 F , F ) ReLU     |
|            1 | Conv2d ( F , F , 3, 1)          |
|            1 | Conv2d ( F , 1, 4, 1) Tanh      |

## E.3 Architecture of the energy function and the interpolant network on the Rotated Character Dataset

The architecture of the energy function used in the EBM is shown in Table 6, and the architecture of the interpolant network is provided in Table 7. These architectures are used for both the URC and BRC datasets. The EBM was trained using the procedure described in Algorithm 2, and Fig.14 shows samples generated by the EBM at the end of training. All EBM training hyperparameters match those described in SectionB.2. For both the EBM and interpolant training, we use a batch size of 128. The interpolant network is optimized with Adam, using a learning rate of 1 × 10 -4 .

Table 6: Archiecture of the EBM energy function on both URC and BRC datasets

| Nb. Layers   | Layer Type                                                                               |
|--------------|------------------------------------------------------------------------------------------|
| 1            | Linear (64, 128) SiLU                                                                    |
| 1            | Linear (128, 512) SiLU                                                                   |
| 6            | Linear (512, 512) SiLU                                                                   |
| 1            | Linear (512, 64)                                                                         |
| 1            | Three output heads: Linear (64, 1) for f 1 Linear (64, 1) for f 2 Linear (64, 1) for f 3 |
| Output       | f 1 ( x ) · f 2 ( x )+ f 3 ( x 2 )                                                       |

|   Nb. Layers | Layer Type              |
|--------------|-------------------------|
|            1 | Linear (64*3, 128) SiLU |
|            1 | Linear (128, 128) SiLU  |
|            1 | Linear 128, 128) SiLU   |
|            1 | Linear 128, 128) SiLU   |
|            1 | Linear 128, 128) SiLU   |
|            1 | Linear 128, 64) SiLU    |

Table 7: Architecture of the interpolant network used on the URC and BRC dataset.

## E.4 Hyperparameters of the LAND and RBF metric

LAND metric We performed a hyperparameter search to tune the σ parameter. We found that σ = 0 . 4 yielded the best performance. Parameters are similar for both the URC and BRC datasets.

Figure 14: Samples from the EBM train on URC. These samples are generated by applying Langevin dynamics to the energy function learned by the EBM.

<!-- image -->

RBF metric We conducted a hyperparameter search to tune both the number of centroids K and the scaling factor κ . The best results were obtained with K = 300 and κ = 0 . 75 . Parameters are similar for both the URC and BRC datasets.

## E.5 Additional geodesics

URC dataset: In Fig. 15 we show additional geodesics on the URC dataset.

Figure 15: Geodesics on the URC dataset. Geodesics are computed using four different metrics: a) G E θ , b) G RBF , c) G 1 / p θ , d) G LAND . For comparison, a simple linear interpolation is shown in e) . The trajectory are computed in the latent space of the autoencoder and projected into pixel space for visualization. Each trajectory is subsampled at 20 time steps for clarity.

<!-- image -->

BRC dataset: In Fig. 16 we show additional geodesics on the BRC dataset.

<!-- image -->

Figure 16: Geodesics on the BRC dataset. Geodesics are computed using four different metrics: a) G E θ , b) G RBF , c) G 1 / p θ , d) G LAND . For comparison, a simple linear interpolation is shown in e) . The trajectory are computed in the latent space of the autoencoder and projected into pixel space for visualization. Each trajectory is subsampled at 20 time steps for clarity.

## E.6 Quantitative evaluation with error bars

In Table. 8, we report the same quantitative results as in Fig. 4, now including 2 -σ error bars. The standard deviation σ is computed over evaluation metrics, each averaged on a different set of randomly sampled trajectories (five sets in total).

Table 8: Quantitative evaluation on the URC dataset with the 2 σ error. Quantitative evaluation using two metrics: (i) D -RMSE, which measures proximity to the dataset manifold, and (ii) γ -RMSE, which measures the deviation from an ideal smooth rotation. Values after the ± sign indicate the 2 -σ error.

| Metric         | D -RMSE ( ↓ )   | γ ⋆ -RMSE ( ↓ )   |
|----------------|-----------------|-------------------|
| linear interp. | 2 . 96 ± 0 . 42 | 3 . 52 ± 0 . 21   |
| G E θ          | 0 . 11 ± 0 . 01 | 0 . 40 ± 0 . 03   |
| G 1 / p θ      | 0 . 14 ± 0 . 02 | 0 . 44 ± 0 . 07   |
| G LAND         | 0 . 66 ± 0 . 12 | 2 . 39 ± 0 . 51   |
| G RBF          | 0 . 36 ± 0 . 06 | 0 . 86 ± 0 . 17   |

## F Experimental details on the Rotated Character Dataset

## F.1 Dataset

In this section, we conduct experiments on the Animal Faces High-Quality (AFHQ) dataset introduced by [59]. The full dataset contains 15,000 images across three categories: cats, dogs, and wild animals. For our experiments, we restrict the dataset to the cat and dog classes only, each comprising approximately 5,000 images. This choice avoids introducing curvature in the data manifold that could arise from the relatively small number of samples in the wild animal category. All images are cropped, aligned, and have a resolution of 512×512 pixels. AFHQ is widely used for image-to-image translation and style transfer, and its diversity in pose, breed, and appearance makes it well-suited for smooth interpolation tasks. See Fig. 17 for example images from the AFHQ dataset.

Figure 17: Samples from the AFHQ dataset [59]

<!-- image -->

For the experiments in this section, we compute geodesics in the latent space of a pretrained Variational Autoencoder (VAE). Specifically, we use the VAE from Stable Diffusion V1 [18]. The latent representations have a spatial size of 4 × 16 × 16 .

## F.2 Architecture of the energy function and the interpolant network on the AFHQ dataset

Energy Function: The architecture used for the energy function is detailed in Table 9. We set the number of input channels to n c = 4 , matching the dimensionality of the latent representation, and use F = 256 feature channels in the first convolutional layer. The network follows a simple sequence of downsampling convolutional layers, which we found to yield more stable training than ResNet-style architectures. The EBM is trained using Algorithm 2, with the same hyperparameters as in Section B.2. To further improve training stability, we add a denoising score matching regularization term and use a cosine learning rate scheduler.

Table 9: Architecture of the energy function . F denotes the base number of feature channels, and n c is the number of input channels. The final energy is computed using three parallel output heads. The notation Conv2d( n c , n f , k , s , p ) refers to a 2D convolutional layer with n c input channels, n f output channels, a kernel size of k , stride s , and padding p .

| Nb. Layers   | Layer Type                                                                                                  |
|--------------|-------------------------------------------------------------------------------------------------------------|
| 1            | Conv2d ( n c , F , 3, 1, 1) SiLU                                                                            |
| 1            | Conv2d ( F , F , 3, 1, 1) SiLU                                                                              |
| 1            | Conv2d ( F , 2 F , 4, 2, 1) SiLU                                                                            |
| 1            | Conv2d ( 2 F , 2 F , 3, 1, 1) SiLU                                                                          |
| 1            | Conv2d ( 2 F , 4 F , 4, 2, 1) SiLU                                                                          |
| 1            | Conv2d ( 4 F , 4 F , 3, 1, 1) SiLU                                                                          |
| 1            | Conv2d ( 4 F , 8 F , 4, 2, 1) SiLU                                                                          |
| 1 1 1        | Conv2d ( 8 F , 1, 2, 1, 0): for f 1 Conv2d ( 8 F , 1, 2, 1, 0): for f 2 Conv2d ( 8 F , 1, 2, 1, 0): for f 3 |
| Output       | f 1 ( x ) · f 2 ( x )+ f 3 ( x 2 )                                                                          |

In Fig. 18, we show randomly selected samples generated by the EBM after training. The Fréchet Inception Distance (FID) of the model is measured to be 9.89.

Figure 18: Samples from the EBM trained on the AFHQ dataset. These samples are generated by applying Langevin dynamics to the energy function learned by the EBM.

<!-- image -->

Interpolant Network: We use the U-Net architecture from [103], following the same hyperparameter settings.

## F.3 Hyperparameters of the LAND and RBF metric

LAND metric We performed a hyperparameter search to tune the σ parameter. We found that σ = 10 yielded the best performance.

RBF metric We conducted a hyperparameter search to tune both the number of centroids K and the scaling factor κ . The best results were obtained with K = 1000 and κ = 3 .

## F.4 FIDs with error bars

In Table. 10, we include 2 -σ error bars. The standard deviation σ is computed over different sets of randomly sampled trajectories (five sets in total).

Table 10: FID along geodesics for different Riemannian metrics . FID is computed at each trajectory point to assess on-manifold alignment. Values after the ± sign indicate the 2 -σ error.

| Metric                                                    | FID ( ↓ )                                                                                             |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| Linear interp. Slerp interp. G E θ G 1 / p θ G LAND G RBF | 42 . 47 ± 3 . 17 32 . 67 ± 2 . 33 20 . 79 ± 2 . 17 16 . 47 ± 1 . 04 39 . 17 ± 3 . 63 37 . 98 ± 2 . 46 |

## F.5 Additional geodesics on AFHQ

Riemanian metric:

G

1

/

p

θ

Figure 19: Geodesics on the AFHQ dataset using G 1 / p θ . Each row shows a geodesic in latent space between a randomly sampled cat image (start point) and a dog image (end point). Columns correspond to time steps along each geodesic, from left (start) to right (end). Images are obtained by decoding the latent representations back into pixel space.

<!-- image -->

<!-- image -->

## Riemanian metric: G E θ

Figure 20: Geodesics on the AFHQ dataset using G E θ . Each row shows a geodesic in latent space between a randomly sampled cat image (start point) and a dog image (end point). Columns correspond to time steps along each geodesic, from left (start) to right (end). Images are obtained by decoding the latent representations back into pixel space.

<!-- image -->

## Riemanian metric: G RBF

Figure 21: Geodesics on the AFHQ dataset using G RBF . Each row shows a geodesic in latent space between a randomly sampled cat image (start point) and a dog image (end point). Columns correspond to time steps along each geodesic, from left (start) to right (end). Images are obtained by decoding the latent representations back into pixel space.

<!-- image -->

## Riemanian metric: G LAND

Figure 22: Geodesics on the AFHQ dataset using G LAND . Each row shows a geodesic in latent space between a randomly sampled cat image (start point) and a dog image (end point). Columns correspond to time steps along each geodesic, from left (start) to right (end). Images are obtained by decoding the latent representations back into pixel space.

<!-- image -->

## Linear interpolation

Figure 23: Linear interpolation on the AFHQ dataset. Each row shows an interpolation in latent space between a randomly sampled cat image (start point) and a dog image (end point). Columns correspond to time steps along each interpolation, from left (start) to right (end). Images are obtained by decoding the latent representations back into pixel space.

<!-- image -->

## Slerp interpolation

Figure 24: Spherical interpolation (Slerp) on the AFHQ dataset. Each row shows an interpolation in latent space between a randomly sampled cat image (start point) and a dog image (end point). Columns correspond to time steps along each interpolation, from left (start) to right (end). Images are obtained by decoding the latent representations back into pixel space.

<!-- image -->

## F.6 About the Spherical interpolation

Given two points x 0 , x 1 ∈ R D lying on the unit hypersphere (i.e., ∥ x 0 ∥ = ∥ x 1 ∥ = 1 ), the spherical interpolation between them is defined as:

<!-- formula-not-decoded -->

where θ is the angle between x 0 and x 1 , given by:

<!-- formula-not-decoded -->

In practice, when interpolating latent codes from a Variational Autoencoder (V AE), the latent vectors x 0 and x 1 are typically drawn from a standard normal prior and do not lie on the unit sphere. To apply slerp, we first normalize the vectors:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This interpolation method, introduced by [60], has proven particularly effective for interpolating in the latent space of V AEs. The intuition behind its success is that it implicitly assumes the data manifold lies on a hypersphere. While this may seem restrictive, the assumption is reasonable in practice. In a V AE, each latent coordinate x i is drawn from a standard Normal distribution: x i ∼ N (0 , 1) ( 1 &lt; i &lt; D ). As a result, the

D

∑

i

=1

squared distribution with D degree of freedom. This distribution is known to concentrate tightly around D , effectively placing most latent codes near the surface of a hypersphere. To validate this empirically, we visualize the distribution of || x || 2 for all latent codes of the AFHQ dataset (see Fig. 25). We observe that this distribution is concentrated on D -D = 1024 for VAE with latent space of size 4 × 16 × 16 .

To conclude, slerp interpolation is well-suited for VAE latent spaces because it aligns with their underlying geometric structure.

Figure 25: Distribution of || x || 2 on the AFHQ dataset

<!-- image -->

and compute θ as:

squared norm,

||

x

2

||

=

x

2

i

follows a chi-

## F.7 Physical interpretation

We refer the reader to [104] or [105] for a detailed background on differential geometry.

Geodesic equation. Assume that the manifold M is the ambient D -dimensional Euclidean space ( M = R D ). We equipped the manifold M with a conformal Riemannian metric G ( x ) = 1 p ( x ) · I , with p the probability density of the data, and I the identity matrix of R D × D . Let γ ( t ) be a geodesic (i.e. γ : [0 , 1] → R D ). We denote the instantaneous speed of the geodesic at time t , ˙ γ ( t ) , and its acceleration ¨ γ ( t ) . Said otherwise, ˙ γ ( t ) and ¨ γ ( t ) denote ∂ γ ∂t ( t ) and ∂ 2 γ ∂t 2 ( t ) respectively.

The geodesic equation is the 2nd-order ODE written as:

<!-- formula-not-decoded -->

In Eq.17, ¨ γ k ( t ) and ˙ γ k ( t ) denotes the k-th coordinate of ¨ γ ( t ) and ˙ γ ( t ) , respectively (here 1 &lt; k &lt; D ). Γ k i,j are the Christoffel symbols, they are derived from the Riemannian metric and encode how it bends and curves the space. Γ k i,j tells how much the change in direction in the i -th and j -th coordinate causes acceleration in the k -th coordinate ( 1 &lt; i, j, k &lt; D ). Said differently, i , j refer to the coordinate direction along which the particule is moving, and k refers to the coordinate direction where the motion causes effect (i.e. curvature induces acceleration).

Christoffel symbols for conformal metric. The Christoffel symbols for a conform metric G ( x ) = λ ( x ) · I (with λ a scalar function):

<!-- formula-not-decoded -->

In Eq. 18, ∂ i λ ( x ) = ∂λ ( x ) ∂x i (i.e. the partial derivative of λ ( x ) with respect to the i -th coordinate), and δ j,k is the Kronecker symbol ( δ j,k = 1 if j = k and δ j,k = 0 otherwise). If one plugs Eq. 18 in the right hand side of Eq. 17:

<!-- formula-not-decoded -->

where ⟨· , ·⟩ and ∥ · ∥ are the usual Euclidean inner product and norms, respectively. So Eq. 17, becomes :

<!-- formula-not-decoded -->

Pulling everything together. If one plugs our definition of the Riemannian metric (i.e. λ ( γ ( t ) ) = 1 p ( γ ( t ) ) , and therefore ∇ λ ( γ ( t ) ) λ ( γ ( t ) ) = -∇ log p ( γ ( t ) ) ), Eq. 20 becomes:

<!-- formula-not-decoded -->

Eq. 21 is similar in form to Newton's second law. The acceleration of a particle (of unit mass) is governed by a velocity-dependent force built from the Stein Score (i.e. ∇ γ log p ( γ ( t )) ). More speficically:

- ⟨∇ log p ( γ ( t )) , ˙ γ ( t ) ⟩ · ˙ γ ( t ) describes a "force" aligned with the particle velocity direction. This term acts like an anisotropic drag or propulsion term: i) it speeds up the particle when it goes toward a high-density region and ii) it slows down the particle going the other way.
- -1 2 ∥ ˙ γ ( t ) ∥ 2 · ∇ log p ( γ ( t )) is a force in the direction of the stein score (pointing toward low density regions). It behaves like a repulsive force, pushing the particle toward areas with low probability. The faster the particle moves, the stronger the force.

The 'force' seems to depend on the velocity ˙ γ ( t ) , which is typical of inertial forces (i.e, forces that depend on a given frame). This is an artifact from the affine parametrization of the geodesic, which ensures constant speed along the trajectory.

Newtonian formalism. Note that the variable t in previous equations is the geometrical 'time'. This variable t stems from the affine parametrization (e.g. see Eq. 3) and is not related to the physical time. To make Eq. 21 compatible with the "physical" time, denoted s , one can consider the following change of variable:

<!-- formula-not-decoded -->

This change of variable implies that when moving through space according to arc-length s, the geometric time t runs more slowly in low-density regions and faster in high-density ones. This change of variable is particularly handy to interpret Eq. 21 as Newtonian motion. Let's therefore consider the following reparametrization: γ ( t ( s ) ) = x ( s ) , where x is the new trajectory parametrized by the physical time s . So:

<!-- formula-not-decoded -->

Now plugging Eq. 24 and Eq. 23 in Eq. 21:

<!-- formula-not-decoded -->

This equation can be interpreted through Newton's second law: it describes the motion of a particle x following a geodesic in the Riemanannian manifold ( M , 1 p ( x ) ) , where p ( x ) denotes the data density. The particle experiences a force -1 2 ∥ ˙ x ∥ 2 ∇ log p ( x ) , pushing away from regions of high probability. The term || x || 2 modulates the forces magnitude and plays a role analogous to momentum, strengthening the pull when the particle moves quickly. While this is not a literal physical systemhere the particle is a data point, and has no mass, it provides a useful analogy for understanding the dynamics of trajectories shaped by data geometry.

## G Limitations

While our approach provides a promising framework for deriving Riemannian metrics from EBMs, several limitations should be acknowledged:

- First, we restrict our study to conformal metrics, which uniformly scale the identity matrix and thus cannot capture directional (anisotropic) structure in the data manifold. While this simplifies optimization, it limits expressivity in settings where geometry varies across directions-something more expressive, score-based metrics may help resolve.
- Second, our method relies on pretrained EBMs that assign meaningful energy values across the entire space. Training such models is challenging in high-dimensional settings due to the computational cost of sampling (e.g., Langevin dynamics), and performance can degrade if the energy landscape is poorly shaped or overfitted.
- Third, although we demonstrate improvements over strong baselines, our evaluation of geodesic quality remains largely indirect-relying on alignment with proxy measures (e.g., density, rotation smoothness, FID). In complex datasets like natural images, the absence of ground-truth geometry makes rigorous evaluation difficult.
- Fourth, our approach assumes that the data distribution is adequately captured by the EBM, yet in practice, misestimation of density-especially in underrepresented regions-may distort the metric and lead to suboptimal paths.
- Finally, while we demonstrate promising results on several datasets, our experiments are constrained to pretrained generative models and fixed feature spaces (e.g., V AE latents), and generalizing to end-to-end learnable architectures remains unexplored.

Future work may address these limitations by developing scalable score-based metrics, improving EBM training stability, integrating richer evaluation protocols, and extending the framework to broader model classes and learning settings.

## H Broader Impact

This work advances our understanding of data geometry by connecting generative modeling and Riemannian geometry, with potential implications across machine learning, neuroscience, and cognitive science. By enabling principled geodesic computation in high-dimensional spaces, our approach could support safer interpolation in generative models, improve motion planning in robotics, or inform models of human cognition. However, care should be taken when applying such methods to sensitive domains, as learned energy landscapes may inherit biases present in training data.

## I Computational ressources

All experiments were conducted on NVIDIA RTX 3090 GPUs (32 GB memory). Training on the toy dataset was fast, with both the EBM and interpolant completing in a few minutes. For the Rotated Characters dataset, EBM training took 8 GPU hours and the interpolant 30 minutes. On the AFHQ dataset, training required 6 GPU days for the EBM and 24 GPU hours for the interpolant. Including extensive hyperparameter searches and trial-and-error development, the total compute usage amounted to approximately 123,000 GPU hours.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our main claim is that EBM-derived metrics stay closer to the data manifold and better capture the geometry of the data compared to alternative metrics. In all experimental settings this claim is verified (see Fig. 2, Fig. 4, Table 1).

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have briefly discussed limitations in the conclusion section of the main article. But we have included an addtional a full section in the supplementary information (see Supp. G) to expand those limitations.

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

Answer: [NA] .

Justification: This paper does not present new theoretical results, but it builds on and leverages existing theoretical insights from prior work.

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

Justification: Due to space constraints, we could not include all experimental details in the main paper. However, the supplementary material provides a thorough description of each experiment, including neural network architectures, all hyperparameters, additional samples, and the pseudo-codes for the main algorithms we used.

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

Justification: All datasets we used are open access. In addition, upon acceptance we will release the github code to reproduce all the experiments.

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

In the supplementary materials (see Supp. D, Supp. E and Supp. F), we have extensively reported experimental details about the datasets, the type of optimizers we used, and the hyperparameters.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We reported the 2 σ error bar for all quantitative metrics on the Supplementary information (see Supp. D.3, Supp. F and Supp. F.4).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of computing workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We have included a section describing the computational resources we use for all experiments in the supplementary information (see Supp. I).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: All the research conducted in this article conforms to the Neurips Code of Ethics

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We have discussed the broader impact of our research in Supp. H Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA] .

Justification: We don't think our work poses a significant risk.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: In terms of datasets, we use the Mixture of Gaussians (not under license), the AFHQ dataset (under CC BY 4.0 Licence), and the rotated letter (based and sklearn letters). In addition, we use the interpolant training algorithms, and the contrastive divergence to train EBMS. All the creators of these assets have been credited by citing the corresponding articles.

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

Answer: [Yes] .

Justification: Our only new asset is the code used to run all experiments, which will be released publicly under the MIT license upon acceptance. All other assets are fully documented in this article.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA] .

Justification: No human experiments or crowdsourcing are involved in this article.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] .

Justification: No human experiments or crowdsourcing are involved in this article.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.