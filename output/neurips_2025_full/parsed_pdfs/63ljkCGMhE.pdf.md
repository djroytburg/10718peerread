## VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold

## Dominic Maggio ∗ Hyungtae Lim ∗ Luca Carlone

Massachusetts Institute of Technology

{drmaggio, shapelim, lcarlone}@mit.edu

## Abstract

We present VGGT-SLAM, a dense RGB SLAM system constructed by incrementally and globally aligning submaps created from the feed-forward scene reconstruction approach VGGT using only uncalibrated monocular cameras. While related works align submaps using similarity transforms ( i.e., translation, rotation, and scale), we show that such approaches are inadequate in the case of uncalibrated cameras. In particular, we revisit the idea of reconstruction ambiguity, where given a set of uncalibrated cameras with no assumption on the camera motion or scene structure, the scene can only be reconstructed up to a 15-degreesof-freedom projective transformation of the true geometry. This inspires us to recover a consistent scene reconstruction across submaps by optimizing over the SL(4) manifold, thus estimating 15-degrees-of-freedom homography transforms between sequential submaps while accounting for potential loop closure constraints. As verified by extensive experiments, we demonstrate that VGGTSLAM achieves improved map quality using long video sequences that are infeasible for VGGT due to its high GPU requirements. Our code is available at https://github.com/MIT-SPARK/VGGT-SLAM .

## 1 Introduction

One of the most fundamental tasks in computer vision is that of simultaneous localization and mapping (SLAM) where given multiple monocular (or stereo) images, the task is to generate a 3D reconstruction of the scene and estimate the 6-degrees-of-freedom (DOF) pose of the cameras. Most approaches for this have traditionally leveraged classical multi-view geometry constraints [47, 25, 80, 81], data association [38, 37], and backend optimization such as bundle adjustment [52, 53, 57, 1, 58, 59]. Recently, a new paradigm of using simpler, feed-forward networks, which produce point clouds from uncalibrated input images, has gained increasing popularity. In this thrust, the seminal work DUSt3R [73] takes in a pair of images and estimates dense point clouds of both images in the reference frame of the first camera, thus creating a dense scene reconstruction and allowing the camera poses estimated easily with a 3-point RANSAC solver [48, 40].

To extend feed-forward reconstruction to multiple images, VGGT (Visual Geometry Grounded Transformer) [71] takes in an arbitrary number of images, and in addition to estimating dense point clouds of each camera frame, also estimates depth maps, feature tracks, and camera poses and intrinsics. However, VGGT is limited in the number of images that can be processed by GPU memory. For example, in the case of an NVIDIA GeForce RTX 4090 with 24 GB, this is limited to approximately 60 images, making larger reconstructions requiring hundreds or thousands of images infeasible.

∗ Equal contribution.

Figure 1: VGGT-SLAM alignment of 6 submaps created from VGGT using Sim(3) alignment (top) and SL(4) alignment (bottom). Here, Sim(3) is insufficient to align submaps due to a projective ambiguity, motivating our SL(4) -based SLAM. Experiments performed on a segment of the Clio [39] apartment and cubicle scenes.

<!-- image -->

One may suspect that a simple, trivial solution would be to create multiple submaps with VGGT where each submap contains at least one overlapping image, and solve for the scale parameter between submaps (as the reconstruction does not capture metric scale), with VGGT's estimated poses being used to align rotation and translation ( i.e., estimating a Sim(3) transformation between submaps). While we demonstrate Sim(3) optimization shows impressive reconstructions in many cases, we empirically observe that the feed-forward nature of VGGT with uncalibrated cameras introduces a projective ambiguity , which in addition to the Sim(3) DOF includes shear, stretch, and perspective DOF, especially when the disparity between images becomes small. This ambiguity cannot be fully resolved through a similarity transformation alone.

Unveiling why a similarity transformation is sometimes insufficient for this recent transformer-based scene reconstruction method causes us to return to classical computer vision for answers, specifically the notion of projective ambiguity. To rectify a projective reconstruction to a metric reconstruction requires computing a 4 × 4 homography matrix [26] which can be mapped to the Special Linear, SL(4) , Lie group. Since this is a Lie group, we can formulate the submap alignment problem as a factor graph optimized on the SL(4) manifold to globally align an arbitrary number of submaps given both estimates of relative homographies between sequential submaps and added constraints from detected loop closures.

Contributions Firstly, we present the first SLAM system that leverages the feed-forward scene reconstruction capabilities of VGGT [71], extending it to large-scale scenes that cannot be reconstructed from a single inference of VGGT. Our system operates entirely with monocular RGB cameras and does not require known camera intrinsics or consistent calibration across frames. Importantly, it achieves this without any additional training.

Secondly, while Sim(3) optimization is often sufficient, we identify and analyze scenarios where projective ambiguity arises, as presented in Fig. 1. In these cases, conventional similarity transforms do not fully resolve scale and alignment issues. We highlight this limitation and demonstrate how incorporating projective constraints addresses the problem.

Finally, we propose the first factor graph formulation that operates directly on the SL(4) manifold to address projective ambiguity. Even in practical scenarios, where projective ambiguity is less dominant, we show that SL(4) -based optimization achieves performance competitive with or superior to other state-of-the-art learning-based SLAM approaches, offering a principled framework for handling cases where similarity transformations are insufficient.

## 2 Related Work

Classical Scene Reconstruction Classical scene reconstruction methods typically rely on geometric features to estimate camera poses and reconstruct 3D scenes from multi-view images [45, 5, 19, 11, 56] using [32, 75] bundle adjustment [58, 50], by performing sparse feature extraction, matching, and robust pose estimation by optimizing for SE(3) transformations. Several works have also performed dense SLAM [30, 76, 41, 77], and [6] provides a survey on SLAM. Multiple works have also performed classical projective scene reconstruction using uncalibrated cameras [68, 27]. Recently, Sim-Sync [78] introduced a certifiably optimal algorithm that jointly estimates camera poses and per-image scaling factors that leverage pretrained monocular depth predictions.

Feed-forward Scene Reconstruction The seminal work DUSt3R [73] has spawned multiple followup works in feed-forward scene reconstruction. DUSt3R takes in a pair of images and for each image, outputs a dense point map in the reference frame of the first camera. From the point maps, the camera focal lengths can be estimated using the Weiszfeld algorithm [51] and poses can be recovered using multiple methods such as a 3-point RANSAC [48, 40]. MASt3R follows a similar design but also outputs descriptors that can be used to generate pairwise correspondences between the two frames. MASt3R-SFM [15] demonstrates global optimization of multiple images using MASt3R but computation scales quickly with the number of frames.

To extend the idea of DUSt3R to multiple frames, Spann3R [70] leverages a learned memory module and Cut3R [72] uses a recurrent state model. Both can incrementally reconstruct a scene using multiple images, but are each limited to short sequences. Recently, Pow3R [29] extends the DUSt3R framework to optionally take in any estimates of any combination of camera intrinsics, poses, and depth (which may be sparse or dense) and demonstrates substantial improvement in scene reconstruction and pose estimation given the added inputs. Splatt3R [62] extends the DUSt3R idea to Gaussian Splatting [31] by directly outputting the Gaussian Splatting parameters given two views, and PreF3R [8] extends this to multiple views using a similar memory framework as Spann3R. Reloc3r [14] modifies the DUSt3R framework to directly output relative camera poses and uses motion averaging to recover absolute poses with respect to a map database.

Most similar to ours is MASt3R-SLAM [46]. MASt3R-SLAM leverages MASt3R to construct an impressive real time dense monocular SLAM system that does not require known calibration. Their pipeline also includes efficient optimization over Sim(3) poses and loop closures. Since MASt3R is limited to two input frames at a time, here, we desire to build on top of the more powerful VGGT architecture for a SLAM system which can leverage broader information of the scene by taking in an arbitrary number of frames for feed-forward reconstruction (bounded by computational limits) and provides direct estimates of camera poses. However, as mentioned, fusing submaps from VGGT goes beyond a traditional point cloud registration problem as alignment cannot be effectively performed with only a similarity transformation. Unlike MASt3R-SLAM, as will be discussed in Sec. 4.2, we do not need to estimate correspondences between frames.

An alternative paradigm, scene coordinate regression, with works such as ACE [3] and DSAC* [4], estimates world points from images with respect to a global scene frame by using a scene specific trained network.

Optimization over the Special Linear group To the best of our knowledge, we are the first work to create a factor graph optimization for point cloud alignment on the SL(4) manifold. Prior works use optimization on the SL(3) manifold (corresponding to the 8-DOF homography matrix commonly used in image alignment) for aligning multiple images for panoramic stitching [22, 61, 43, 42, 44, 36] and dense SLAM [35]. The 15-DOF homography matrix is used for classical tasks such as autocalibration [23], and good practices for estimating homography are extensively studied in [24].

## 3 Review: VGGT

Here, we provide the relevant preliminaries of VGGT [71]. VGGT takes as input an image set I = { M 1 , · · · , M ¯ w } , which consists of ¯ w images, tokenizes them with a fine-tuned DINO [49] backbone, and then applies Alternating-Attention (alternating between applying global and framewise attention). The output tokens can then be passed to a camera head to estimate intrinsics and camera poses (defined with respect to the first frame), or to Dense Prediction Transformer (DPT)

heads [54], which outputs dense depth maps for each image, a dense point map (where the points of each camera are defined with respect to the first camera), and dense features for point tracking, with confidence estimates provided for each.

In this paper, we use the dense depth maps D = { D 1 , · · · , D ¯ w } and confidence score maps C = { C 1 , · · · , C ¯ w } (as they are fully dense, the width and height of corresponding components in M , D , and C are the same). We refer to the outputs from each I as submap S , which will correspond to a node in pose graph optimization for VGGT-SLAM. We do not run the 3D point DPT head as it was observed in Wang et al. [71] that more accurate point clouds can be achieved by inverse projecting D using projection matrices from the camera head, giving us a dense point cloud which is defined with respect to the coordinate frame of the first camera in I . We denote this point cloud as X S . To filter unreliable points, we prune points whose associated confidence values in the confidence maps are less than τ conf of the average confidence across C .

## 4 VGGT-SLAM

Here we describe the design of our VGGT-SLAM system. In Sec. 4.1 we determine how to generate a list of images that will be passed to VGGT to produce a local submap, S . In Sec. 4.2, we provide a discussion of projective ambiguity and show how we can align two overlapping submaps by estimating a relative 15-DOF homography matrix between sequential submaps, and in Sec. 4.3, we describe the process of adding loop closure constraints between non-sequential submaps. Finally, in Sec. 4.4 we show how we can globally optimize all submap alignments into a consistent map by optimizing on the SL(4) manifold.

## 4.1 Incremental submap-based keyframe selection and generation

First, we begin by describing how to incrementally construct submaps and organize keyframes within each submap from sequentially incoming images. For this, we construct an image set I latest . As is typical in visual SLAM [57, 46, 64], we select an image as a keyframe if disparity (which we estimate using Lucas-Kanade [38]) with respect to the previous keyframe is larger than a user-defined threshold τ disparity. Even though VGGT demonstrates monocular depth capabilities [71] from learned priors, utilizing images with sufficient disparity improves relative depth estimation performance as it adds multi-view information and additionally reduces the number of images to process.

If sufficiently high disparity is estimated, the current image is designated a keyframe and added to a list of images, I latest , until the size of the list reaches a set limit w . In addition to I latest , each submap's associated image set is constructed by concatenating two additional sets of images. The first set includes a single image chosen as the last non-loop-closure image from the previous submap, denoted as M prior . Up to w loop images to be used for loop closures (discussed in Sec. 4.3) may also be appended at the end of the collection, forming the final image set for the submap as I latest ←{ M prior } ∪ I latest ∪I loop . This image set is then passed to VGGT to generate the submap, S latest .

## 4.2 Local submap alignment addressing projective ambiguity

Given two overlapping submaps S i and S j generated as described in Sec. 4.1, which have point clouds X S i and X S j in their respective local submap frames, our objective is to solve for a transformation, H i j ∈ R 4 × 4 that aligns the two submaps such that for any noise-free corresponding points X S i a , X S j b ∈ R 3 , the following relation holds:

<!-- formula-not-decoded -->

where for simplicity, we overload notation such that X S is in homogeneous coordinates when multiplied by the 4 × 4 homography matrix. Under a typical 3D point cloud alignment problem, for example from LIDAR SLAM [33], H would represent a translation and rotation in SE(3) . If the point clouds additionally differ in scale, then H would be on Sim(3) , the group of similarity transformations. However, here we do not have typical point clouds as X S is constructed by uncalibrated cameras. Thus, we recall the Projective Reconstruction Theorem [26, Chapter 10.3], which in summary

states that if correspondences between two images from uncalibrated cameras uniquely determine the fundamental matrix, then the correspondences may be used to reconstruct the corresponding 3D points up to a 15-DOF homography transformation. This transform is the same for any such corresponding points, except those on the line connecting the camera centers as these points cannot be reconstructed uniquely. Relevant to our setup, the Projective Reconstruction Theorem also applies to a reconstruction with more than two cameras [26]. Thus, in the most general case, the reconstruction computed using a set of uncalibrated cameras differs from a metrically correct reconstruction by a projective transformation ( i.e., homography) H . The matrix H has 15 DOF and can be mapped uniquely to the special linear group , SL(4) , by normalizing with the determinant. The SL(4) group consists of all real-valued 4 × 4 matrices with unit determinant. Note this is not the same as the more common 8 DOF homography matrix commonly used in planar computer vision tasks such as image warping, which belongs to SL(3) . The reconstruction can be transformed to an affine reconstruction ( i.e., parallel lines are preserved) when scene priors are available, for example if points are known to lie on parallel lines. If further priors are known, such as lines in the scene are orthogonal, then the reconstruction can be converted to a metric reconstruction (differing by only a similarity transform to the true Euclidean reconstruction). VGGT is thus able to leverage learned scene priors to potentially estimate metric reconstruction, but as we have shown in Fig. 1, in the most general case when estimates of scene priors are unreliable, the reconstruction differs by a projective ambiguity, requiring a 15-DOF homography matrix to rectify. We will now estimate such a homography.

By our construction of the submaps that they share a same image, we have an atypical advantage in solving for H as we have a dense set of correspondences without needing to estimate associations.

As is well known by the direct reconstruction method [26], the optimal homography in (1) can be solved in closed form as a solution to the following homogeneous linear system:

<!-- formula-not-decoded -->

with h ∈ R 16 containing the flattened parameters of the homography and A k contains constraints for a particular pair of 3D points. A minimum solution requires 5 points ( i.e., k ∈ { 1 : 5 } ), and to build in robustness to incorrect depth measurements from VGGT, we solve (2) using RANSAC [18] with a 5-point solver. As the homography matrix is estimated up to scale, we scale by the fourth root of the determinant such that the determinant is unity and the resulting matrix belongs to SL(4) .

Transformation of camera poses via homography Using a homography between reference frames i and j , H i j , the camera poses can be corrected using the following [26]: P i = P j H i j -1 , where P ∈ R 3 × 4 is the camera matrix created from the poses and intrinsic estimates from VGGT. We can then decompose P to recover the camera pose.

## 4.3 Loop closures

Our procedure for creating loop closures for VGGT-SLAM consists of two steps: (i) performing image retrieval ( i.e., setting I loop in Sec. 4.1), and (ii) estimating relative homographies, which are then added to the factor graph as loop closure constraints (Sec. 4.4). First, for image retrieval, when constructing a submap, we compute and store an image descriptor for each keyframe using SALAD [28]. Then, once I latest reaches its size threshold w , we search over the image descriptors in the previous submaps S i ∀ i ∈ { 1 : lastest -τ interval } to fetch a set of frames of size w loop that have the highest similarity (using the L2 norm) to any of the keyframes in I latest , and also exceed a user-defined similarity threshold τ desc to reduce false positive matches. These frames make up I loop , which is added to the list of keyframes for the current submap, and then all frames are sent to VGGT as described in Sec. 4.1.

Next, given the estimated submap, S latest , from VGGT, we estimate the relative homographies between the loop closure frames in S latest and the submaps, S i , retrieved during the image retrieval process described above. As in Sec. 4.2, we again have the benefit of not requiring an estimate of correspondences to compute homographies for loop closures; thus, we can directly use (2) between the frames in I loop and their respective identical frames in the submap where they originated. This then provides w loop loop closure constraints between S latest and the corresponding submaps.

Note that a potential alternative is to get the descriptor using the output tokens from VGGT's finetuned DINO backbone. This alleviates using a separate descriptor module and storing the physical images. However, this requires storing larger features in memory compared to the relatively small SALADfeatures, and the system memory needed to store the images in our base approach is relatively low.

## 4.4 Backend: Nonlinear factor graph optimization on the SL(4) manifold

Given all relative homographies H i j between submaps S i and S j , our goal is to compute the absolute homographies H i that transform all submaps into a common global reconstruction. To achieve this, we formulate a nonlinear factor graph optimization problem 2 based on Maximum A Posteriori (MAP) estimation [2, 17, 20]. Specifically, we estimate the absolute homographies by minimizing the following cost function under Gaussian noise assumptions on the relative homographies:

<!-- formula-not-decoded -->

where Log( · ) is the mapping function that transforms a group element to a (vectorized) element of the corresponding Lie algebra, L denotes an index set of constraints that includes odometry and loop closures, and we set Ω H ij ∈ R 15 × 15 to the identity matrix.

To solve (3), we iteratively compute state increments by solving a linearized least squares problem. To this end, we define ξ ∈ R 15 as the tangent-space parameterization of SL(4) , the mapping function Exp : R 15 → SL(4) , which satisfies Log ( Exp( ξ ) ) = ξ and Exp( ξ ) = exp( ξ ∧ ) = H . In particular, ξ ∧ is a Lie algebra element of sl (4) , computed by summing the k -th component of ξ with its k -th corresponding generator G k ∀ k : { 1 : 15 } ( i.e., ξ ∧ = ∑ 15 k =1 ξ k G k ) [16]. More details can be found in Appendix A.

Next, defining the measurement function as h ( ξ i , ξ j ) = Log ( H -1 i H j ) , the incremental update of each pose can be approximated using Taylor's expansion [13] as follows:

<!-- formula-not-decoded -->

where J i = -Ad H -1 i H j and J j = I 15 × 15 . Here, Ad H is the adjoint map, defined as Ad H = B -1 H ⊗ H -⊺ B [16], where B = [vec ( G 1 ) vec ( G 2 ) · · · vec ( G 15 )] ∈ R 16 × 15 in the case of the SL(4) manifold [9], and ⊗ denotes the Kronecker product, which forms a block matrix by multiplying each element of the first matrix with the entire second matrix.

Finally, we can formulate the linearized residuals and the resulting local problem at the linearization point H i j as follows:

<!-- formula-not-decoded -->

To solve (5), we use the Levenberg-Marquardt optimizer [55], and at each iteration, the poses are updated on the Lie group as H ← H Exp( ˆ δ ) [63].

## 5 Experiments

We follow similar experiments as MASt3R-SLAM to evaluate camera pose estimation and dense reconstruction in Sec. 5.2 and Sec. 5.3 respectively, demonstrate qualitative results in Sec. 5.5, and finally perform ablations in Sec. 5.6.

2 In our case, this is an extension of pose graph optimization, where we estimate absolute poses from pairwise pose measurements.

## 5.1 Experimental setup

We evaluate VGGT-SLAM on standard RGB SLAM benchmarks to assess both camera pose estimation accuracy and dense mapping quality. For evaluation of pose estimation, we employ the 7-Scenes [60] and TUM RGB-D [65] datasets, and report root mean square error (RMSE) of the absolute trajectory error (ATE) using evo [21]. Since 7-Scenes [60] provides scene ground truth, this dataset is also used to evaluate dense mapping quality in terms of accuracy , completion , and Chamfer distance [46].

As baseline approaches, we primarily compare VGGT-SLAM with DROID-SLAM [67] and MASt3RSLAM [46] as the state-of-the-art learning-based SLAM approaches in uncalibrated scenarios (and Spann3R [70] for dense evaluation). We use reported numbers from MASt3R-SLAM [46] for baselines, except for the uncalibrated version of DROID-SLAM. Although DROID-SLAM requires camera intrinsics, we also evaluate it in an uncalibrated setting by estimating intrinsics with an automatic calibration pipeline [69], as is suggested by Murai et al. [46]. While our approach operates without camera calibration, we also include comparison with state-of-the-art methods [7, 34, 79, 66, 12, 82] provided with camera intrinsics. Due to potential randomness in our approach caused by RANSAC, we report the average performance over five runs, which have a low spread (small standard deviation) as shown in Sec. 5.6.

We refer to a simpler Sim(3) version of VGGT-SLAM as Ours ( Sim(3) ), for which we follow similar structure as our SL(4) pipeline except we align relative rotation and translation between submaps using pose estimates from VGGT and estimate a scale correction by comparing the estimated point clouds of the overlapping images. Loop closures and relative factors are added to the factor graph as SE(3) factors.

We use an NVIDIA GeForce RTX 4090 GPU with AMD Ryzen Threadripper 7960X CPU. For parameters, we set w loop = 1 , τ disparity = 50 pixels, τ interval = 2 , τ desc = 0 . 8 , and τ conf = 25%. We also use 300 RANSAC iterations with a threshold of 0.01. We show evaluations of both the SL(4) and Sim(3) version of VGGT-SLAM with different submap sizes ( i.e., different values for w ).

## 5.2 Pose estimation evaluation

As shown in Tables 1 and 2, VGGT-SLAM performs comparable to the top performing uncalibrated baselines on 7-Scenes and TUM RGB-D. On 7-Scenes for instance, VGGT-SLAM has approximately the same average APE as the top performing baseline MASt3R-SLAM. On the TUM dataset, the SL(4) version of VGGT-SLAM performs the best overall with an average error of 0.053 m. This demonstrates that we are able to extend VGGT to multiple sequences while introducing a new category of SLAM system by optimizing submap alignment as SL(4) factors. Here, we observe that our Sim(3) version also performs well, as these scenes are generally cases where VGGT is able to leverage strong priors for metric reconstruction. Thus, while we have shown cases where SL(4) is needed (Fig. 1), the addition of higher degrees of freedom with our novel SLAM formulation maintains competitive performance, while improving some more challenging cases.

One particular scene where our method underperforms is on the TUM floor scene. This highlights a challenge of estimating homography, which is the presence of degeneracy in the case of a planar scene. The floor scene contains several images that only view the flat floor leading to non-unique solutions for the homography matrix, which causes the overall reconstruction to diverge. Building robustness for the planar case is an important component for SL(4) SLAM, which we leave as an exciting direction for future work. The TUM 360 scene is particularly challenging for smaller submap sizes (although handled well with w = 32 ) because smaller submaps are more likely to encounter approximately pure rotation in this scene, which can have reduced depth accuracy and hence a higher outlier ratio when running 5-point RANSAC to estimate homography.

Table 1: Root mean square error (RMSE) of absolute trajectory error (ATE) on 7-Scenes [60] (unit: m). The gray rows indicate the results using the calibrated camera intrinsics and the * symbol indicates that the baseline is evaluated in the uncalibrated mode. Green is best and light green is second best.

<!-- image -->

|          | Method                                                                            |                         |                         |                         | Sequence                |                         |                         |                         | Avg                     |
|----------|-----------------------------------------------------------------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
|          | Method                                                                            | chess                   | fire                    | heads                   | office                  | pumpkin                 | kitchen                 | stairs                  |                         |
| Calib.   | NICER-SLAM [82] DROID-SLAM [67] MASt3R-SLAM [46]                                  | 0.033 0.036 0.053       | 0.069 0.027 0.025       | 0.042 0.025 0.015       | 0.108 0.066 0.097       | 0.200 0.127 0.088       | 0.039 0.040 0.041       | 0.108 0.026 0.011       | 0.086 0.049 0.047       |
| Uncalib. | DROID-SLAM* [67] MASt3R-SLAM* [46] Ours ( Sim(3) ,w = 32 ) Ours ( SL(4) ,w = 32 ) | 0.047 0.063 0.037 0.036 | 0.038 0.046 0.026 0.028 | 0.034 0.029 0.018 0.018 | 0.136 0.103 0.104 0.103 | 0.166 0.114 0.133 0.133 | 0.080 0.074 0.061 0.058 | 0.044 0.032 0.093 0.093 | 0.078 0.066 0.067 0.067 |

Table 2: Root mean square error (RMSE) of absolute trajectory error (ATE) on TUM RGB-D [65] (unit: m). The gray rows indicate the results using the calibrated camera intrinsics and the * symbol indicates that the baseline is evaluated in the uncalibrated mode. Green is best and light green is second best.

<!-- image -->

|          | Method                     |                                       |                               |                                           |       |                                           |                                             |                                             |                                 |                                     | Avg   |
|----------|----------------------------|---------------------------------------|-------------------------------|-------------------------------------------|-------|-------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------|-------------------------------------|-------|
|          | Method                     | 360                                   | desk                          | desk2                                     | floor | plant                                     | room                                        | rpy                                         | teddy                           | xyz                                 |       |
|          | ORB-SLAM3 [7] DeepV2D [66] | × 0.243 0.159 0.112 0.132 0.089 0.111 | 0.017 0.166 0.170 0.018 0.018 | 0.210 0.379 0.253 0.029 0.029 0.028 0.042 | ×     | 0.034 0.203 0.305 0.021 0.022 0.026 0.016 | × 0.246 0.364 0.330 0.096 0.052 0.049 0.061 | × 0.105 0.043 0.030 0.032 0.019 0.026 0.027 | × 0.316 0.601 0.084 0.098 0.048 | 0.009 0.064 0.035 0.010 0.010 0.010 | N/A   |
|          |                            |                                       |                               |                                           | 1.653 |                                           |                                             |                                             |                                 |                                     | 0.375 |
|          | DeepFactors [12]           |                                       |                               |                                           | 0.169 |                                           |                                             |                                             |                                 |                                     | 0.233 |
|          | DPV-SLAM [34]              |                                       |                               |                                           | 0.057 |                                           |                                             |                                             |                                 |                                     | 0.076 |
|          | DPV-SLAM++ [34]            |                                       |                               |                                           | 0.050 |                                           |                                             |                                             |                                 |                                     | 0.054 |
|          | GO-SLAM [79]               |                                       | 0.016                         |                                           | 0.025 |                                           |                                             |                                             |                                 |                                     | 0.035 |
|          | DROID-SLAM [67]            |                                       | 0.018                         |                                           | 0.021 |                                           |                                             |                                             | 0.048                           | 0.012                               | 0.038 |
|          | MASt3R-SLAM [46]           | 0.049                                 | 0.016                         | 0.024                                     | 0.025 | 0.020                                     |                                             |                                             | 0.041                           | 0.009                               | 0.030 |
| Uncalib. | DROID-SLAM* [67]           | 0.202                                 | 0.032                         | 0.091                                     | 0.064 | 0.045                                     | 0.918                                       | 0.056                                       | 0.045                           | 0.012                               | 0.158 |
| Uncalib. | MASt3R-SLAM* [46]          | 0.070                                 | 0.035                         | 0.055                                     | 0.056 | 0.035                                     | 0.118                                       | 0.041                                       | 0.114                           | 0.020                               | 0.060 |
| Uncalib. | Ours ( Sim(3) , w = 32 )   | 0.123                                 | 0.040                         | 0.055                                     | 0.254 | 0.022                                     | 0.088                                       | 0.041                                       | 0.032                           | 0.016                               | 0.074 |
| Uncalib. | Ours ( SL(4) , w = 32 )    | 0.071                                 | 0.025                         | 0.040                                     | 0.141 | 0.023                                     | 0.102                                       | 0.030                                       | 0.034                           | 0.014                               | 0.053 |

## 5.3 Dense reconstruction evaluation

Following the protocol of MASt3R-SLAM, we provide dense reconstruction performance on 7-Scenes; see Table 3. Here, we observe that while performance is comparable across methods, VGGT-SLAM achieves the best performing accuracy and Chamfer distance, demonstrating the high accuracy of our dense point cloud reconstruction.

## 5.4 Timing Analysis

Table 3: Root mean square error (RMSE) reconstruction evaluation on 7-Scenes [60] (unit: m). @ n indicates a keyframe every n images.

<!-- image -->

| Method                                            | 7-Scenes        | 7-Scenes   | 7-Scenes                | 7-Scenes          |
|---------------------------------------------------|-----------------|------------|-------------------------|-------------------|
|                                                   | ATE ↓           | Acc. ↓     | Complet. ↓              | Chamfer ↓         |
| DROID-SLAM [67] MASt3R-SLAM [46] Spann3R @20 [70] | 0.049 0.047 N/A | 0.141      | 0.048 0.085 0.047 0.043 | 0.094 0.087 0.058 |
|                                                   |                 | 0.089      |                         |                   |
|                                                   |                 | 0.069      |                         |                   |
| Spann3R @2 [70]                                   | N/A             | 0.124      |                         | 0.084             |
| MASt3R-SLAM* [46]                                 | 0.066           | 0.068      | 0.045                   | 0.056             |
| Ours ( Sim(3) , w = 32 )                          | 0.067           | 0.052      | 0.062                   | 0.057             |
| Ours ( SL(4) , w = 32 )                           | 0.067           | 0.052      | 0.058                   | 0.055             |

In Table 4 we show approximate timing results comparing the primary components of the Sim(3) and SL(4) versions of VGGT-SLAM. As expected, the total time used to run optical flow and determine keyframes for all frames in the submap (keyframe detection), the total time to run SALAD and perform image retrieval for all frames in the submap (loop closure detection), and the time to run the VGGT model (VGGT inference) is identical up to random variations. The time to optimize the factor graph using GTSAM (backend optimization) is very fast, taking only about half a millisecond for both variants, since the factor graph in VGGT-SLAM is relatively small compared to factor graphs used in typical real time visual odometry problems [56]. The primary runtime difference between the Sim(3) and SL(4) variants is the time to compute relative edge constraints (relative transformation estimation) since the 5-point RANSAC estimation takes about 17 ms longer than the time to compute the Sim(3) transformation. However, this increase is comparatively small, only about 2.5 percent of the time needed to run VGGT.

Memory needed to run VGGT for VGGT-SLAM is independent of the total number of images in the scene since VGGT is run with fixed size submaps. An analysis of memory usage for running VGGT with varying numbers of images can be found in [10].

Table 4: Runtime comparison between Sim(3) and SL(4) versions of VGGT-SLAM on the office\_loop sequence with window size w = 16 (unit: msec). Runtime is averaged over five runs.

| Stage                                                                                                            | VGGT-SLAM w/ Sim(3)               | VGGT-SLAM w/ SL(4)                |
|------------------------------------------------------------------------------------------------------------------|-----------------------------------|-----------------------------------|
| Keyframe detection VGGT inference Loop closure detection Relative transformation estimation Backend optimization | 176 ms 662 ms 105 ms 11 ms 0.4 ms | 176 ms 662 ms 105 ms 28 ms 0.5 ms |

## 5.5 Qualitative results

We present qualitative results to illustrate the mapping fidelity of VGGT-SLAM using SL(4) optimization. In Fig. 2, we show an example reconstruction from the office scene in 7-Scenes and from a longer 55 meter trajectory that loops inside an office corridor. In addition to the dense reconstruction, we also show all mapped camera poses, where different colors indicate the submap associated with each image. In particular, the office corridor loop clearly shows 22 different submaps which have been joined into a globally consistent map with a loop closure at the end of the trajectory.

Figure 2: Reconstruction and pose estimates from VGGT-SLAM on the office scene from 7-Scenes showing 8 submaps and from a custom scene showing a 55 meter loop around an office corridor with 22 submaps. Both use w = 16 . Different image colors indicate the submap associated with each image.

<!-- image -->

In Fig. 1, we show two select examples where using only Sim(3) is unable to align overlapping submaps while our SL(4) alignment strategy is able to rectify the projective ambiguity between submaps. Thus, while we have shown that Sim(3) generally achieves accurate performance across our quantitative experiments, in the general case where a feed-forward reconstruction method like VGGT is unable to estimate a metric reconstruction (for reasons discussed in Sec. 4.2) due to the computational limits, our introduction of an SL(4) -based SLAM system shows promise in leveraging the potential of a high accuracy, dense, learning-based SLAM system. For Fig. 1, τ disparity is set to 0 to highlight the impact of projective ambiguity, which degrades the performance of Sim(3) alignment and affects overall map quality.

## 5.6 Ablations

In Fig. 3 we provide multiple ablation studies which show the following for three different submap sizes ( w = 8 , 16 , 32 ): (a) improved pose accuracy of VGGT-SLAM when loop closures are leveraged along with showing a tight statistical spread of results from averaging 5 runs from our experiments, (b) that loop closures generally lead to increasing reduction in pose error as the number of submaps increases since there are an increased number of loop closures, (c) the effect of different values of τ conf , which as expected, larger values lead to higher accuracy on dense reconstruction and lower completion, with our default value of 25% showing an appropriate balance.

## 6 Limitations

We have presented a new type of SLAM system that addresses the issue of projective ambiguity from an underlying feed-forward scene reconstruction method (in our case, VGGT). As creating a factor

Figure 3: Ablation studies: (a) Effect of loop closure (LC) on absolute trajectory error (ATE) across different window sizes, w , in TUM [65], where ** annotations indicate measurements with 10 -3 &lt;p -value ≤ 10 -2 after a paired t -test. (b) Reduction in ATE achieved by incorporating loop closure in TUM, highlighting that as the number of submaps increases, our SL(4) -based optimization leads to greater reductions in ATE. (c) Performance changes with respect to the confidence threshold τ conf in ATE, accuracy, completion, and Chamfer distance under varying window sizes in 7-Scenes [60].

<!-- image -->

graph that optimizes on the SL(4) manifold is a new paradigm for the SLAM problem, it leaves much ground for further improvements. In particular, the estimation of the full 15-DOF homography matrix is degenerate in the case of planar points, which can lead to unstable solutions as we have observed in the planar floor scene of the TUM dataset. Our current implementation of homography using points from VGGT is also vulnerable to outliers. While we use a 5-point RANSAC to reduce this issue, the presence of a high outlier ratios or adversarial outliers (which are present due to local consistency of points in VGGT) can cause incorrect homography estimates as discussed in Sec. 5.2. The ray-based matching in MASt3R-SLAM provides robustness to errors in depth measurements, and a similar method can potentially be adapted for homography estimation. Additionally, 15 DOF give rise to added opportunity of scene drift. While our addition of loop closures substantially corrects drift, an inaccurate relative homography estimate or long time between loop closures can cause not just drift in scale, rotation, and translation seen in classical SLAM, but also in scene perspective, which opens up an interesting area of research into further optimization into SL(4) -based SLAM. Finally, lens distortion is not rectified by the homography matrix since a projective transformation preserves straight lines, and thus images are assumed to be undistorted when running VGGT-SLAM.

## 7 Conclusion

In this study, we have leveraged VGGT, a feed-forward reconstruction model, to incrementally construct a dense map from uncalibrated monocular cameras, proposing a novel SLAM framework called VGGT-SLAM , which locally and globally (through loop closures) aligns submaps from VGGT. By exploring VGGT's geometric understanding through the lens of classical multi-view computer vision, we have shown that in the general case, these submaps must be aligned with a projective transformation, and in doing so we have created the first factor graph SLAM system optimized on the SL(4) manifold. In future work we will further investigate conditions under which Sim(3) optimization suffices and investigate techniques to actively employ both Sim(3) and SL(4) optimization in a unified system to enable a more robust SLAM system for real-time performance.

## Acknowledgments

This work is supported in part by the NSF Graduate Research Fellowship Program under Grant 2141064, the ONR RAPID program, and the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00461409). The authors would like to gratefully acknowledge Riku Murai for assisting us with benchmarking.

## References

- [1] M. Abate, Y. Chang, N. Hughes, and L. Carlone. Kimera2: Robust and accurate metric-semantic SLAM in the real world. In Intl. Sym. on Experimental Robotics (ISER) , 2023.
- [2] Jose Luis Blanco. A tutorial on se(3) transformation parameterizations and on-manifold optimization. 09 2010.
- [3] Eric Brachmann, Tommaso Cavallari, and Victor Adrian Prisacariu. Accelerated coordinate encoding: Learning to relocalize in minutes using rgb and poses. In CVPR , 2023.

- [4] Eric Brachmann and Carsten Rother. Visual camera re-localization from rgb and rgb-d images using dsac. pami , 44(9):5847-5865, 2021.
- [5] D. Bradley, T. Boubekeur, and W. Heidrich. Accurate multi-view reconstruction using robust binocular stereo and surface meshing. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) , pages 1-8, 2008.
- [6] C. Cadena, L. Carlone, H. Carrillo, Y. Latif, D. Scaramuzza, J. Neira, I. Reid, and J.J. Leonard. Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age. IEEE Trans. Robotics , 32(6):1309-1332, 2016. arxiv preprint: 1606.05830, (pdf).
- [7] Carlos Campos, Richard Elvira, Juan J Gómez Rodríguez, José MM Montiel, and Juan D Tardós. ORBSLAM3: An accurate open-source library for visual, visual-inertial, and multimap SLAM. IEEE Trans. Robotics , 2021.
- [8] Zequn Chen, Jiezhi Yang, and Heng Yang. Pref3r: Pose-free feed-forward 3d gaussian splatting from variable-length image sequence. arXiv preprint arXiv:2411.16877 , 2024.
- [9] Marston Conder, Edmund Robertson, and Peter Williams. Presentations for 3-dimensional special linear groups over integer rings. Proceedings of the American Mathematical Society , 115(1):19-26, 1992.
- [10] Wenyan Cong, Yiqing Liang, Yancheng Zhang, Ziyi Yang, Yan Wang, Boris Ivanovic, Marco Pavone, Chen Chen, Zhangyang Wang, and Zhiwen Fan. E3d-bench: A benchmark for end-to-end 3d geometric foundation models. arXiv preprint arXiv:2506.01933 , 2025.
- [11] Andrei Cramariuc, Lukas Bernreiter, Florian Tschopp, Marius Fehr, Victor Reijgwart, Juan Nieto, Roland Siegwart, and Cesar Cadena. maplab 2.0-A modular and multi-modal mapping framework. IEEE Robotics and Automation Letters , 8(2):520-527, 2022.
- [12] J. Czarnowski, T. Laidlow, R. Clark, and A. Davison. DeepFactors: Real-time probabilistic dense monocular SLAM. IEEE Robotics and Automation Letters , 5(2):721-728, 2020.
- [13] Frank Dellaert. Factor graphs and GTSAM: A hands-on introduction. Technical Report GT-RIM-CP&amp;R2012-002, Georgia Institute of Technology, September 2012.
- [14] Siyan Dong, Shuzhe Wang, Shaohui Liu, Lulu Cai, Qingnan Fan, Juho Kannala, and Yanchao Yang. Reloc3R: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization. arXiv preprint arXiv:2412.08376 , 2024.
- [15] Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud. MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion. arXiv preprint arXiv:2409.19152 , 2024.
- [16] Ethan Eade. Lie groups for 2D and 3D transformations. URL http://ethaneade. com/lie. pdf, revised Dec , 117:118, 2013.
- [17] K. Ebadi, L. Bernreiter, H. Biggie, G. Catt, Y. Chang, A. Chatterjee, C.E. Denniston, S-P. Deschênes, K. Harlow, S. Khattak, L. Nogueira, M. Palieri, P. Petrá˘ cek, P. Petrlík, A. Reinke, V. Krátký, S. Zhao, A. Agha-mohammadi, K. Alexis, C. Heckman, K. Khosoussi, N. Kottege, B. Morrell, M. Hutter, F. Pauling, F. Pomerleau, M. Saska, S. Scherer, R. Siegwart, J.L. Williams, and L. Carlone. Present and future of SLAM in extreme underground environments. IEEE Trans. Robotics , 40:936-959, 2024. (pdf).
- [18] M. Fischler and R. Bolles. Random sample consensus: a paradigm for model fitting with application to image analysis and automated cartography. Commun. ACM , 24:381-395, 1981.
- [19] Y. Furukawa, B. Curless, S. M. Seitz, and R. Szeliski. Towards internet-scale multi-view stereo. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) , pages 1434-1441, 2010.
- [20] Giorgio Grisetti, Rainer Kümmerle, Cyrill Stachniss, and Wolfram Burgard. A tutorial on graph-based SLAM. IEEE Intelligent Transportation Systems Magazine , 2(4):31-43, 2010.
- [21] Michael Grupp. evo: Python package for the evaluation of odometry and SLAM. https://github. com/MichaelGrupp/evo , 2017.
- [22] Tarek Hamel, Robert Mahony, Jochen Trumpf, Pascal Morin, and Minh-Duc Hua. Homography estimation on the special linear group based on direct point correspondence. In 2011 50th IEEE Conference on Decision and Control and European Control Conference , pages 7902-7908, 2011.
- [23] Richard Hartley. An algorithm for self calibration from several views. In cvpr , pages 908-912, 1994.
- [24] R. Hartley and A. Zisserman. Multiple View Geometry in Computer Vision . Cambridge University Press, 2000.
- [25] R. I. Hartley. In defense of the eight-point algorithm. IEEE Trans. Pattern Anal. Machine Intell. , 19(6):580593, June 1997.
- [26] R. I. Hartley and A. Zisserman. Multiple View Geometry in Computer Vision . Cambridge University Press, second edition, 2004.
- [27] Je Hyeong Hong, Christopher Zach, Andrew Fitzgibbon, and Roberto Cipolla. Projective bundle adjustment from arbitrary initialization using the variable projection method. In European Conf. on Computer Vision (ECCV) , pages 477-493. Springer, 2016.
- [28] Sergio Izquierdo and Javier Civera. Optimal transport aggregation for visual place recognition. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) , June 2024.
- [29] Wonbong Jang, Philippe Weinzaepfel, Vincent Leroy, Lourdes Agapito, and Jerome Revaud. Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors. arXiv preprint arXiv:2503.17316 , 2025.
- [30] Krishna Murthy Jatavallabhula, Ganesh Iyer, and Liam Paull. gradslam: Dense slam meets automatic differentiation. In IEEE Intl. Conf. on Robotics and Automation (ICRA) , pages 2130-2137. IEEE, 2020.

- [31] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. , 42(4):139-1, 2023.
- [32] J.J. Koenderink and A.J. vanDoorn. Affine structure from motion. Journal of the Optical Society of America A , 8(2):377-385, 1991.
- [33] Hyungtae Lim, Beomsoo Kim, Daebeom Kim, Eungchang Mason Lee, and Hyun Myung. Quatro++: Robust global registration exploiting ground segmentation for loop closing in LiDAR SLAM. Intl. J. of Robotics Research , pages 685-715, 2024.
- [34] Lahav Lipson, Zachary Teed, and Jia Deng. Deep patch visual SLAM. In European Conf. on Computer Vision (ECCV) , pages 424-440, 2024.
- [35] Steven Lovegrove. Parametric dense visual SLAM . PhD thesis, 2012.
- [36] Steven Lovegrove and Andrew J Davison. Real-time spherical mosaicing using whole image alignment. In eccv , pages 73-86. Springer, 2010.
- [37] D.G. Lowe. Distinctive image features from scale-invariant keypoints. Intl. J. of Computer Vision , 60(2):91-110, 2004.
- [38] B. D. Lucas and Takeo Kanade. An iterative image registration technique with an application in stereo vision. In Intl. Joint Conf. on AI (IJCAI) , pages 674-679, 1981.
- [39] D. Maggio, Y. Chang, N. Hughes, M. Trang, D. Griffith, C. Dougherty, E. Cristofalo, L. Schmid, and L. Carlone. Clio: Real-time task-driven open-set 3D scene graphs. IEEE Robotics and Automation Letters (RA-L) , 9(10):8921-8928, 2024. (pdf),(video),(web).
- [40] J. Matas and O. Chum. Randomized RANSAC with sequential probability ratio test. In Intl. Conf. on Computer Vision (ICCV) , pages 1727-1732, 2005.
- [41] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting SLAM. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) , pages 18039-18048, 2024.

[42]

C. Mei, S. Benhimane, E. Malis, and P. Rives. Constrained multiple planar template tracking for central catadioptric cameras. In

British Machine Vision Conf. (BMVC)

, September 2006.

- [43] C. Mei, S. Benhimane, E. Malis, and P. Rives. Homography-based tracking for central catadioptric cameras. In IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS) , October 2006.
- [44] C. Mei, S. Benhimane, E. Malis, and P. Rives. Efficient homography-based tracking and 3-D reconstruction for single-viewpoint sensors. IEEE Trans. Robotics , 24(6):1352-1364, Dec. 2008.
- [45] E. Mouragnon, M. Lhuillier, M. Dhome, F. Dekeyser, and P. Sayd. 3d reconstruction of complex structures with bundle adjustment: an incremental approach. In IEEE Intl. Conf. on Robotics and Automation (ICRA) , pages 3055-3061, May 2006.
- [46] Riku Murai, Eric Dexheimer, and Andrew J Davison. MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors. arXiv preprint arXiv:2412.12392 , 2024.
- [47] David Nistér. An efficient solution to the five-point relative pose problem. IEEE Trans. Pattern Anal. Machine Intell. , 26(6):756-770, 2004.
- [48] D. Nistér. A minimal solution to the generalised 3-point pose problem. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) , pages 560-567, 2004.
- [49] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 , 2023.
- [50] Linfei Pan, Daniel Barath, Marc Pollefeys, and Johannes Lutz Schönberger. Global Structure-from-Motion Revisited. In European Conf. on Computer Vision (ECCV) , 2024.
- [51] Frank Plastria. The Weiszfeld Algorithm: Proof, Amendments, and Extensions , pages 357-389". Springer US, New York, NY, 2011.
- [52] Tong Qin, Shaozu Cao, Jie Pan, and Shaojie Shen. A general optimization-based framework for global pose estimation with multiple sensors. arXiv preprint: 1901.03642 , 2019.
- [53] Tong Qin, Peiliang Li, and Shaojie Shen. Vins-mono: A robust and versatile monocular visual-inertial state estimator. IEEE Transactions on Robotics , 34(4):1004-1020, 2018.
- [54] R. Ranftl, A. Bochkovskiy, and V. Koltun. Vision transformers for dense prediction. In Intl. Conf. on Computer Vision (ICCV) , pages 12179-12188, 2021.
- [55] D.M. Rosen, M. Kaess, and J.J. Leonard. An incremental trust-region method for robust online sparse least-squares estimation. In IEEE Intl. Conf. on Robotics and Automation (ICRA) , pages 1262-1269, St. Paul, MN, May 2012.
- [56] A. Rosinol, M. Abate, Y. Chang, and L. Carlone. Kimera: an open-source library for real-time metricsemantic localization and mapping. In IEEE Intl. Conf. on Robotics and Automation (ICRA) , pages 1689-1696, 2020. arXiv preprint: 1910.02490, (video), (code), (pdf).
- [57] A. Rosinol, A. Violette, M. Abate, N. Hughes, Y. Chang, J. Shi, A. Gupta, and L. Carlone. Kimera: from SLAM to spatial perception with 3D dynamic scene graphs. Intl. J. of Robotics Research , 40(12-14):15101546, 2021. arXiv preprint: 2101.06894, (pdf).
- [58] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) , pages 4104-4113, 2016.
- [59] Johannes L Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In European Conf. on Computer Vision (ECCV) , pages 501-518. Springer, 2016.
- [60] J. Shotton, B. Glocker, C. Zach, S. Izadi, A. Criminisi, and A. Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgb-d images. In IEEE Conf. on Computer Vision and Pattern

Recognition (CVPR) , pages 2930-2937, 2013.

- [61] Heung-Yeung Shum and Richard Szeliski. Systems and experiment paper: Construction of panoramic image mosaics with global and local alignment. International Journal of Computer Vision , 36:101-130, 2000.
- [62] Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. Splatt3R: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912 , 2024.
- [63] Joan Sola. Quaternion kinematics for the error-state kalman filter. arXiv preprint arXiv:1711.02508 , 2017.
- [64] Seungwon Song, Hyungtae Lim, Alex Junho Lee, and Hyun Myung. DynaVINS: A visual-inertial SLAM for dynamic environments. IEEE Robotics and Automation Letters , 7(4):11523-11530, 2022.
- [65] Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A benchmark for the evaluation of RGB-D SLAM systems. In IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS) , pages 573-580. IEEE, 2012.
- [66] Zachary Teed and Jia Deng. DEEPV2D: Video to depth with differentiable structure from motion. Intl. Conf. on Learning Representations (ICLR) , 2018.
- [67] Zachary Teed and Jia Deng. DROID-SLAM: Deep visual SLAM for monocular, stereo, and RGB-d cameras. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems (NIPS) , 2021.
- [68] B. Triggs. Factorization methods for projective structure and motion. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) , pages 845-851, 1996.
- [69] Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, and Marc Pollefeys. GeoCalib: Learning Single-image Calibration with Geometric Optimization. In European Conf. on Computer Vision (ECCV) , pages 1-20. Springer, 2024.
- [70] Hengyi Wang and Lourdes Agapito. 3D reconstruction with spatial memory. arXiv preprint arXiv:2408.16061 , 2024.
- [71] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. arXiv preprint arXiv:2503.11651 , 2025.
- [72] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3D Perception Model with Persistent State. arXiv preprint arXiv:2501.12387 , 2025.
- [73] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) , pages 20697-20709, 2024.
- [74] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. 2020.
- [75] Juyang Weng, Thomas Huang, and Narendra Ahuja. Motion and structure from line correspondences: Closed-form solution, uniqueness, and optimization. IEEE Trans. Pattern Anal. Machine Intell. , 14(3), 1992.
- [76] T. Whelan, R.F. Salas-Moreno, B. Glocker, A. J. Davison, and S. Leutenegger. ElasticFusion: Real-Time Dense SLAM and Light Source Estimation. 2016.
- [77] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. GS-SLAM: Dense visual slam with 3d gaussian splatting. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) , pages 19595-19604, 2024.
- [78] X. Yu and H. Yang. SIM-Sync: From certifiably optimal synchronization over the 3d similarity group to scene reconstruction with learned depth. IEEE Robotics and Automation Letters , 9(5):4471-4478, 2024.
- [79] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo Poggi. GO-SLAM: Global optimization for consistent 3D instant reconstruction. In Intl. Conf. on Computer Vision (ICCV) , pages 3727-3737, 2023.
- [80] Yinqiang Zheng, Yubin Kuang, Shigeki Sugimoto, Kalle Astrom, and Masatoshi Okutomi. Revisiting the PnP problem: A fast, general and optimal solution. In Intl. Conf. on Computer Vision (ICCV) , pages 2344-2351, 2013.
- [81] Yinqiang Zheng, Shigeki Sugimoto, Imari Sato, and Masatoshi Okutomi. A general and simple method for camera pose and focal length determination. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) , 2014.
- [82] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui, Martin R Oswald, Andreas Geiger, and Marc Pollefeys. NICER-SLAM: Neural implicit scene encoding for RGB SLAM. In IEEE International Conference on 3D Vision (3DV) , pages 42-52, 2024.

## A Tangent space of SL(4)

Here, we provide the explicit 15 generators, G k ∀ k : { 1 : 15 } , of SL(4) , which allow us to relate the Lie algebra sl (4) to the Lie group SL(4) .

̸

The tangent space of SL(4) consists of all 4 × 4 real matrices with zero trace. Thus, there are 15 generators, G k , where 12 of them are defined as E ab for a = b where 1 is in the ( a, b ) entry and 0, elsewhere. The remaining three generators are B 1 = diag(1 , -1 , 0 , 0) , B 2 = diag(0 , 1 , -1 , 0) , B 3 = diag(0 , 0 , 1 , -1) . Explicitly, the generators are as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, as briefly explained in Sec. 4.4, the relation between the Lie algebra, ξ ∧ ∈ sl (4) , and the Lie group H ∈ SL(4) is given by:

<!-- formula-not-decoded -->

## B Extra Quantitative Results

We provide addition results of evaluating on the 7-Scenes [60] and TUM RGB-D [65] datasets where we experiment with different submap sizes (Appendix B.1) and show the number of submaps and loop closures per scene (Appendix B.2).

## B.1 Evaluation with different submap sizes

Here we show results for the 7-Scenes and TUM RGB-D datasets in Tables 5 and 6 with different submap sizes ( w = 8 , 16 , 32 ). For 7-Scenes, we also include results for w = 1 . Recall that w is the size of new images in the submap, so in the case of w = 1 , each submap has one new image, one image from the prior submap, and up to one extra image from loop closures. For small submap size of w = 1 , the backend becomes numerically unstable for some TUM scenes (consistently floor and 360) preventing an estimated alignment, and thus we do not include the w = 1 for TUM. This is due to reasons discussed in Sec. 6. Particularly, for the floor scene there are a large portion of images which only view a planar scene which makes the estimation of the full 15-DOF homography matrix

degenerate, and for the 360 scene, using a small submap size such as w = 1 is likely to encounter a pure rotation which can result in less accurate depth measurements from VGGT and hence reduced accuracy in our estimate of relative homography.

Table 5: Root mean square error (RMSE) of absolute trajectory error (ATE) on 7-Scenes [60] (unit: m). The * symbol indicates that the baseline is evaluated in the uncalibrated mode, all VGGT-SLAM configurations are evaluated in uncalibrated mode. Green is best and light green is second best.

<!-- image -->

| Method                  |       |       |       | Sequence   |         |         |        | Avg   |
|-------------------------|-------|-------|-------|------------|---------|---------|--------|-------|
|                         | chess | fire  | heads | office     | pumpkin | kitchen | stairs |       |
| DROID-SLAM* [67]        | 0.047 | 0.038 | 0.034 | 0.136      | 0.166   | 0.080   | 0.044  | 0.078 |
| MASt3R-SLAM* [46]       | 0.063 | 0.046 | 0.029 | 0.103      | 0.114   | 0.074   | 0.032  | 0.066 |
| Ours ( Sim(3) ,w = 1 )  | 0.047 | 0.025 | 0.032 | 0.113      | 0.138   | 0.050   | 0.083  | 0.070 |
| Ours ( Sim(3) ,w = 8 )  | 0.039 | 0.027 | 0.020 | 0.108      | 0.144   | 0.053   | 0.080  | 0.067 |
| Ours ( Sim(3) ,w = 16 ) | 0.037 | 0.027 | 0.021 | 0.107      | 0.135   | 0.051   | 0.093  | 0.067 |
| Ours ( Sim(3) ,w = 32 ) | 0.037 | 0.026 | 0.018 | 0.104      | 0.133   | 0.061   | 0.093  | 0.067 |
| Ours ( SL(4) ,w = 1 )   | 0.089 | 0.046 | 0.072 | 0.119      | 0.147   | 0.055   | 0.100  | 0.090 |
| Ours ( SL(4) ,w = 8 )   | 0.041 | 0.060 | 0.043 | 0.106      | 0.206   | 0.054   | 0.078  | 0.084 |
| Ours ( SL(4) ,w = 16 )  | 0.036 | 0.065 | 0.037 | 0.107      | 0.139   | 0.050   | 0.093  | 0.075 |
| Ours ( SL(4) ,w = 32 )  | 0.036 | 0.028 | 0.018 | 0.103      | 0.133   | 0.058   | 0.093  | 0.067 |

Table 6: Root mean square error (RMSE) of absolute trajectory error (ATE) on TUM RGB-D [65] (unit: m). The * symbol indicates that the baseline is evaluated in the uncalibrated mode, all VGGT-SLAM configurations are evaluated in uncalibrated mode. Green is best and light green is second best.

<!-- image -->

| Method                   |   Sequence | Sequence   | Sequence   | Sequence   | Sequence   | Sequence   | Sequence   | Sequence   | Sequence   | Avg   |
|--------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|-------|
|                          |    360     | desk       | desk2      | floor      | plant      | room       | rpy        | teddy      | xyz        |       |
| DROID-SLAM* [67]         |      0.202 | 0.032      | 0.091      | 0.064      | 0.045      | 0.918      | 0.056      | 0.045      | 0.012      | 0.158 |
| MASt3R-SLAM* [46]        |      0.07  | 0.035      | 0.055      | 0.056      | 0.035      | 0.118      | 0.041      | 0.114      | 0.020      | 0.060 |
| Ours ( Sim(3) , w = 8 )  |      0.07  | 0.026      | 0.030      | 0.048      | 0.026      | 0.081      | 0.024      | 0.035      | 0.015      | 0.040 |
| Ours ( Sim(3) , w = 16 ) |      0.112 | 0.045      | 0.123      | 0.261      | 0.022      | 0.137      | 0.044      | 0.044      | 0.016      | 0.089 |
| Ours ( Sim(3) , w = 32 ) |      0.123 | 0.040      | 0.055      | 0.254      | 0.022      | 0.088      | 0.041      | 0.032      | 0.016      | 0.074 |
| Ours ( SL(4) , w = 8 )   |      0.179 | 0.046      | 0.095      | 0.210      | 0.033      | 0.272      | 0.038      | 0.130      | 0.031      | 0.115 |
| Ours ( SL(4) , w = 16 )  |      0.147 | 0.032      | 0.087      | 0.158      | 0.027      | 0.150      | 0.037      | 0.069      | 0.035      | 0.083 |
| Ours ( SL(4) , w = 32 )  |      0.071 | 0.025      | 0.040      | 0.141      | 0.023      | 0.102      | 0.030      | 0.034      | 0.014      | 0.053 |

Table 7: Dense reconstruction evaluation on 7-Scenes [60] (unit: m).

<!-- image -->

| Method                   | 7-Scenes   | 7-Scenes   | 7-Scenes   | 7-Scenes   |
|--------------------------|------------|------------|------------|------------|
|                          | ATE ↓      | Acc. ↓     | Complet. ↓ | Chamfer ↓  |
| MASt3R-SLAM* [46]        | 0.066      | 0.068      | 0.045      | 0.056      |
| Ours ( Sim(3) , w = 1 )  | 0.070      | 0.066      | 0.051      | 0.059      |
| Ours ( Sim(3) , w = 8 )  | 0.067      | 0.054      | 0.056      | 0.055      |
| Ours ( Sim(3) , w = 16 ) | 0.067      | 0.054      | 0.058      | 0.056      |
| Ours ( Sim(3) , w = 32 ) | 0.067      | 0.052      | 0.062      | 0.057      |
| Ours ( SL(4) , w = 1 )   | 0.090      | 0.080      | 0.068      | 0.074      |
| Ours ( SL(4) , w = 8 )   | 0.084      | 0.067      | 0.065      | 0.066      |
| Ours ( SL(4) , w = 16 )  | 0.075      | 0.061      | 0.063      | 0.060      |
| Ours ( SL(4) , w = 32 )  | 0.067      | 0.052      | 0.058      | 0.055      |

## B.2 Number of submaps per scene

As a reference, in Tables 8 and 9 we show the number of total submaps in each scene for 7-Scenes and TUM RGB-D for different values of experimented submap size, w , and also show the number of loop closures in each scene.

## B.3 Evaluation of Focal length Consistency

To provide quantitative results showing that VGGT can produce an estimate of the scene which differs by more than a similarity transformation to the true scene, in this section we show inconsistencies in estimates of camera intrinsics from VGGT. Here, a single camera is used per scene and different scenes can use different cameras. We observe that even though the true intrinsics of the camera should be approximately constant within a scene, VGGT has a varying estimate of the instrincs both inside a submap and across different submaps. This provides further demonstration that the VGGT reconstruction of a submap can differ from the true scene by more than a similarity transformation

Table 8: Window size w and corresponding submap and loop closure counts when w loop = 1 , shown as '# of submaps (# of loops)'.

| Window size, w   | Sequences in 7-Scenes [65]   | Sequences in 7-Scenes [65]   | Sequences in 7-Scenes [65]   | Sequences in 7-Scenes [65]   | Sequences in 7-Scenes [65]   | Sequences in 7-Scenes [65]   | Sequences in 7-Scenes [65]   |
|------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
|                  | chess                        | fire                         | heads                        | office                       | pumpkin                      | kitchen                      | stairs                       |
| 1                | 29 (11)                      | 50 (46)                      | 62 (49)                      | 58 (55)                      | 43 (37)                      | 43 (38)                      | 14 (12)                      |
| 8                | 4 (0)                        | 7 (4)                        | 8 (3)                        | 8 (4)                        | 6 (0)                        | 6 (2)                        | 2 (0)                        |
| 16               | 2 (0)                        | 4 (1)                        | 4 (2)                        | 4 (2)                        | 3 (0)                        | 3 (1)                        | 1 (0)                        |
| 32               | 1 (0)                        | 2 (0)                        | 2 (0)                        | 2 (0)                        | 2 (0)                        | 2 (0)                        | 1 (0)                        |

Table 9: Window size w and corresponding submap and loop closure counts when w loop = 1 , shown as '# of submaps (# of loops)'.

| Window size, w   | Sequences in TUM-RGB-D [65]   | Sequences in TUM-RGB-D [65]   | Sequences in TUM-RGB-D [65]   | Sequences in TUM-RGB-D [65]   | Sequences in TUM-RGB-D [65]   | Sequences in TUM-RGB-D [65]   | Sequences in TUM-RGB-D [65]   | Sequences in TUM-RGB-D [65]   | Sequences in TUM-RGB-D [65]   |
|------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
|                  | 360                           | desk                          | desk2                         | floor                         | plant                         | room                          | rpy                           | teddy                         | xyz                           |
| 1                | 168 (151)                     | 54 (42)                       | 98 (84)                       | 99 (87)                       | 102 (92)                      | 186 (162)                     | 95 (89)                       | 146 (125)                     | 56 (54)                       |
| 8                | 21 (4)                        | 7 (4)                         | 13 (7)                        | 13 (3)                        | 13 (5)                        | 24 (7)                        | 12 (10)                       | 19 (9)                        | 7 (5)                         |
| 16               | 11 (2)                        | 4 (2)                         | 7 (4)                         | 7 (2)                         | 7 (2)                         | 12 (4)                        | 6 (4)                         | 10 (4)                        | 4 (2)                         |
| 32               | 6 (1)                         | 2 (0)                         | 4 (2)                         | 4 (1)                         | 4 (2)                         | 6 (2)                         | 3 (1)                         | 5 (2)                         | 2 (0)                         |

and contain affine and projective degrees of freedom which can be resolved using the homography alignment. In Table 10 we summarize the standard deviation, range, and average of all focal length estimates for four scenes. We observe that for both the office loop scene and 7-Scenes, our Sim(3) variant of VGGT-SLAM performs comparable to the SL(4) variant while SL(4) performs significantly better than Sim(3) on the Tabletop and Bollards scene. Consistent with this observation, in Table 10, we notice that the later two have much larger intrinsic error (larger standard deviation and larger range) than the former two.

Figure 4: VGGT estimates of the focal length (fx) of every keyframe in the office loop scene from Fig. 2 for all 22 submaps.

<!-- image -->

Figure 5: VGGT estimates of the focal length (fx) of every keyframe in the tabletop scene from Fig. 7 for all 6 submaps.

<!-- image -->

Table 10: Statistics of VGGT Focal length (fx) estimates. All values in pixels.

| Scene                |   Std Dev |   Range |   Average |
|----------------------|-----------|---------|-----------|
| Office Loop (Fig. 2) |       7.3 |    36.9 |     429   |
| 7-Scenes             |       9   |    59.7 |     435.1 |
| Tabletop (Fig. 7)    |      37.1 |   122.8 |     669.1 |
| Bollards (Fig. 8)    |      51.8 |   177.3 |     738.9 |

## C Extra Qualitative Results

## C.1 Extra examples of SL(4) versus Sim(3)

While we have mentioned that the Sim(3) version of VGGT-SLAM often provides high quality reconstructions, here we provide additional examples of cases where Sim(3) is not sufficient and SL(4) is necessary to achieve consistent alignment across submaps.

Figure 6: Example on a tabletop scene showing Sim(3) is unable to align the submaps while SL(4) is able to correct for projective ambiguity. Here w = 32 and τ disparity = 50 .

<!-- image -->

Figure 7: Example on a tabletop scene showing Sim(3) is unable to align the submaps while SL(4) is able to correct for projective ambiguity. The true scene only has one banana, but the Sim(3) reconstruction shows a hallucination of two caused by misalignment. Camera pose estimates are colored by submap. Here w = 16 and τ disparity = 50 .

<!-- image -->

Figure 8: Example on an outdoor scene with yellow bollards surrounding tanks showing Sim(3) is unable to align the submaps while SL(4) is able to correct for projective ambiguity. The true scene has single bollards spaced around the tanks while the Sim(3) scene hallucinates clusters of bollards due to misalignment. Here w = 16 and τ disparity = 25 .

<!-- image -->

## C.2 7-Scenes Qualitative Results

Here we provide additional visualizations of scene reconstructions from the 7-Scenes dataset experiments for VGGT-SLAM with SL(4) . We use the default parameters from Sec. 5.

Figure 9: Visualization of reconstruction on 7-Scenes fire scene with 2 submaps. Camera pose estimates are colored by submap.

<!-- image -->

Figure 10: Visualization of reconstruction on 7-Scenes heads scene with 2 submaps. Camera pose estimates are colored by submap. Part of the scene in cropped for visual clarity.

<!-- image -->

## C.3 TUMRGB-D Qualitative Results

Here we provide additional visualizations of scene reconstructions from the TUM RGB-D dataset experiments for VGGT-SLAM with SL(4) . We use the default parameters from Sec. 5.

Figure 11: Visualization of reconstruction on TUM room scene with 6 submaps. Camera pose estimates are colored by submap. Part of the scene in cropped for visual clarity.

<!-- image -->

Figure 12: Visualization of reconstruction on TUM 360 scene with 6 submaps. Camera pose estimates are colored by submap. Part of the scene in cropped for visual clarity.

<!-- image -->

Figure 13: Visualization of reconstruction on TUM xyz scene with 2 submaps. Camera pose estimates are colored by submap.

<!-- image -->

## C.4 Additional Outdoor Qualitative Results

While our method is primarily tested on indoor scenes, here we provide an additional example of VGGT-SLAM on an outdoor scene from the TartanAir dataset [74]. Here w = 16 , τ disparity = 50 , and τ conf = 50 .

Figure 14: Visualization of reconstruction on TartanAir (scene Neighborhood Easy, P005, left camera ) with 8 submaps. Camera pose estimates are colored by submap.

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The paper includes the claims in both the abstract and in a contributions section of the introduction, which reflect the scope of the paper's contributions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provide both a limitations section and further discussion of limitations in our experimental analysis.

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

Justification: Our paper does not include new theorems. All equations are numbered with citations provided when they are applied from prior work.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: All parameters are mentioned in the paper and we benchmark using publicly available datasets.

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

Justification: We use public datasets for all baseline experiments and major results and will include our code as a zip file in the supplementary following the code submission guidelines.

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

Justification: We mention all parameters used in our experiments with justification for the major parameters.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Yes, we provide error bars and show the statistical significance of our main experimental results with an added study in our ablations section.

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

Justification: We mention that the experiments are performed on an NVIDIA GeForce RTX 4090 GPU with AMD Ryzen Threadripper 7960X CPU.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have followed all guidelines in the NeurIPS Code of Ethics and our experiments do not include any human subjects and our experiments use common open source SLAM benchmark datasets.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: As a SLAM system, our paper does not have significant societal impacts.

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

Justification: Our method does not require training and use open source weights and datasets.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We provide citation to all use of prior datasets and the foundation model that our SLAM system use and follow the terms of use of each.

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

Justification: We include our code in the supplementary and include relevant licenses with the code.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: We do not use human subjects in our paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: We do not use human subjects in our paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development does not involve the use of LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.