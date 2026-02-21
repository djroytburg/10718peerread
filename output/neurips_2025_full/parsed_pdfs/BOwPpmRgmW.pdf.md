## HoloScene: Simulation-Ready Interactive 3D Worlds from a Single Video

Figure 1: Overview of HoloScene: From a single input video-along with visual cues such as segmentation and monocular depth-HoloScene reconstructs a simulation-ready, interactive 3D digital twin represented as a scene graph with complete geometry, physically plausible dynamics, and realistic rendering. The resulting model enables a variety of downstream applications, including real-time interactive gaming, 3D editing, immersive experience capture, and dynamic visual effects.

<!-- image -->

## Abstract

Digitizing the physical world into accurate simulation-ready virtual environments offers significant opportunities in a variety of fields such as augmented and virtual reality, gaming, and robotics. However, current 3D reconstruction and sceneunderstanding methods commonly fall short in one or more critical aspects, such as geometry completeness, object interactivity, physical plausibility, photorealistic rendering, or realistic physical properties for reliable dynamic simulation. To address these limitations, we introduce HoloScene, a novel interactive 3D reconstruction framework that simultaneously achieves these requirements. HoloScene leverages a comprehensive interactive scene-graph representation, encoding object geometry, appearance, and physical properties alongside hierarchical and interobject relationships. Reconstruction is formulated as an energy-based optimization problem, integrating observational data, physical constraints, and generative priors into a unified, coherent objective. Optimization is efficiently performed via a hybrid approach combining sampling-based exploration with gradient-based refinement. The resulting digital twins exhibit complete and precise geometry, physical stability, and realistic rendering from novel viewpoints. Evaluations conducted on multiple benchmark datasets demonstrate superior performance, while practical use-cases in interactive gaming and real-time digital-twin manipulation illustrate HoloScene's broad applicability and effectiveness. Project page: here.

## 1 Introduction

Imagine wanting, decades later, to revisit the home you live in and love today-how would you capture its memory? Photographs and videos record authentic details but lack immersion; 3D Gaussian splats or photogrammetry can be immersive, yet static chairs and tables feel lifeless. Ideally, we would digitize our environment into a fully interactive digital twin: complete, composable, photorealistic, and manipulable just like the real world. Our work takes a step toward this goal by enabling users to create in silico twins of their surroundings from a single video.

Digitizing the physical world into a simulation-ready virtual environment offers immense opportunities in augmented and virtual reality, gaming, and robotics. However, despite advances in 3D modeling and scene understanding, key challenges remain: capturing complete geometry and appearance in occluded regions, inferring inter-object relationships, and ensuring physical plausibility and interactivity. Existing Real2Sim methods produce incomplete geometry [88, 45, 71] or unstable physics [77, 75]; existing amodal reconstruction focuses on single-image setting [86, 11], individual objects [76, 15, 39], neglects physical plausibility [49] or relies on asset retrieval [10] -sacrificing fidelity and practicality; and prior physically plausible reconstruction [48, 15] is limited to simple object-scene interactions or requires full observations.

To address these gaps, we introduce HoloScene , an interactive 3D reconstruction framework that unifies geometry completeness, object completeness, physical plausibility, realistic rendering, and physical interaction. HoloScene represents a scene as an interactive scene graph encoding object geometry, appearance, and physical properties in a hierarchical structure. We cast scene-graph recovery from video as a structured energy-based optimization, integrating observational data, physical constraints, and generative priors into a single objective. To solve this challenging problem, we propose a novel divide-and-conquer strategy combining sampling-based tree-structured search with gradient-based refinement. The resulting scene models exhibit complete, accurate geometry; stable physical interactions; and realistic rendering from novel viewpoints.

Experiments on three challenging benchmarks demonstrate superior geometry accuracy and physical plausibility, with rendering performance comparable to state-of-the-art amodal and physics-aware reconstruction methods. We further showcase HoloScene's versatility through practical applications in interactive gaming, realistic video effects, and real-time digital-twin manipulation.

## 2 Related Works

Interactive 3D Scene Model Recent advances in 3D scene modeling [52, 93, 17, 5, 68] reconstruct 3D scene from input images or videos, representing the scene as neural fields [46, 65, 19, 62, 61, 25, 3, 4], signed distance functions (SDF) [71, 87, 51, 53, 88, 91, 29], and 3D Gaussians [23, 20, 92, 90]. While producing realistic renderings from novel views, these works cannot provide 3D assets that allow user

Table 1: Comparison of Interactive 3D Scene Models.

<!-- image -->

| Method                                                                                                                                                           | Visual Input                                                            | Real-time Rendering     | Amodal 3D Recon         | Twin Fidelity              | Physics Capacity                                                                 | Physics Optimization                                                        |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------|-------------------------|----------------------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| ACDC [10] Gen3DSR [11] PhysComp [15] CAST [86] NeRF [45] BakedSDF [88] ObjectSDF++ [75] Video2Game [77] DRAWER [78] PhyRecon [48] DP-Recon [49] HoloScene (Ours) | image image image image video video video video video video video video | ✓ ✓ ✗ ✓ ✗ ✗ ✗ ✓ ✓ ✗ ✓ ✓ | ✓ ✓ ✓ ✓ ✗ ✗ ✗ ✗ ✗ ✓ ✓ ✓ | ✗ ✓✗ ✓✗ ✓✗ ✗ ✓ ✓ ✓ ✓ ✓ ✓ ✓ | ✗ ✗ Single Object Scene ✗ ✗ ✗ Single Object Single Object Objects-Ground ✗ Scene | ✗ ✗ Differentiable Differentiable ✗ ✗ ✗ ✗ ✗ Differentiable ✗ Diff &Sampling |

interactions (e.g, move the chairs to different poses). Reconstructing realistic and interactive environments from real images and videos remains challenging due to limited observation, occlusion, and physical reasoning. Some previous works reconstruct 3D objects from sparse viewpoints [39, 35, 79, 7, 41], and some estimate physical properties from visual observation [94, 15]. Nevertheless, these works focus on object-level tasks and cannot handle large and complex indoor scenes. PhyRecon [48] optimizes stable 3D scenes with differentiable physical engines, but does not model inter-object interactions and realistic appearance. DP-Recon [49], Video2Game [77], Drawer [78] leverage generative prior [1, 55] and foundation models [59, 60] to reconstruct decomposed 3D scenes. However, these works can only produce limited components or lack physical stability. Text-to-3D generation [83, 33, 72] creates diverse layouts from text prompts, but fails to reconstruct digital twins that are faithful to visual inputs. In this work, we propose to reason

object interaction with the scene graph, and utilize generative priors and a novel sampling strategy to reconstruct the geometry and appearance of every component, constructing realistic, physically plausible, and interactable 3D environments.

Data-driven Simulation Simulation plays a pivotal role across robotics, self-driving, and content creation, but building high-fidelity virtual scenes remains costly, and the sim-to-real gap poses great challenges. To address this, data-driven simulation [2, 8, 40, 44, 63, 84] has emerged, enabling the modeling of physical dynamics [38, 28, 80, 21, 13, 22, 95], lighting conditions [32, 58, 31], and actionconditioned outcomes [21, 36, 6, 77, 22, 37, 78, 42], directly from real-world data. These methods have also been applied in robot learning [8, 44, 57, 82, 85, 84], LiDAR simulation [34, 44, 81, 84, 99, 98], and interactive media [18, 77]. In robotics, related real-to-sim approaches [9, 10, 67, 70, 27, 43] reconstruct interactable environments from the real world for reproducible embodiment. However, they still lack physical realism. Recent works [48, 6, 97] leverage differentiable physics or priors in reconstruction, but they neglect complex inter-object relationships. The closest work to ours is CAST [86], which also considers physical plausibility in scene reconstruction. However, our proposed HoloScene differs fundamentally in three key aspects: (1) Problem formulation: CAST takes a single-image generative approach where fidelity to input appearance and geometry is not enforced, while HoloScene performs multi-view joint optimization from videos, explicitly ensuring faithful reconstruction of observed scenes through inverse neural rendering. (2) Physical stability: CAST's differentiable optimization only avoids penetrations but cannot guarantee long-term stability. HoloScene employs simulator-in-the-loop optimization with Isaac Sim [47] to ensure objects remain physically stable over time. (3) Inference strategy: CAST follows a multi-step pipeline with greedy sequential optimization, whereas HoloScene adopts a unified energy-based formulation (Eq. 2) that jointly optimizes observation fidelity, physical plausibility, and object completion through generative sampling and scoring. We compare HoloScene with prior works in Tab. 1.

## 3 Method

Given the observations O = {O t } T t =0 , which include the input video sequence { I t } , camera poses { ξ t } (inferred or ground truth), and instance masks { M t } (inferred or ground truth), our goal is to reconstruct a realistic, complete, and physically plausible digital twin of the input scene, yielding interactive, sim-ready assets compatible with simulators and game engines, and which can be used to generate novel visual content. To this end, we represent the scene as an interactive 3D scene graph representation that encodes object geometry, appearance, physical properties, and hierarchical inter-object relationships (Sec. 3.1). We combine observational evidence, generative priors for shape completion, and physical simulation for stability to formulate scene-graph recovery as an energy minimization problem (Sec. 3.2). Finally, we propose an inference method that integrates sampling-based tree search with differentiable optimization (Sec. 3.3). Fig.2 summarizes our approach.

## 3.1 Scene Representation

We represent the scene as an interactive 3D scene graph G = ( V , E ) . Each node v i ∈ V = { v i } N i =0 represents either the background scene or one of the N objects present. A node v i = ( g i , f i , p i , T i ) is comprised of geometry g i , appearance f i , physical properties p i , and dynamic states T i . Each edge e i,j = ( v i , v j ) ∈ E encodes an object-object relationship in G .

Geometry: We represent the geometry of each node v i in the scene with an instance-level neural SDF g i ( x ; θ ): R 3 → R , where x ∈ R 3 is any point in space and θ are learnable parameters. Additionally, to facilitate physical simulation and efficient rendering, we maintain a mesh representation M i = MarchingCube ( g i ) for each object, extracted from its SDF using the marching cubes algorithm.

Appearance: For each object v i , we encode appearance f i = ( c i , α i , µ i , Σ i ) as Gaussian splats, enabling real-time, high-quality rendering. c i , α i , µ i , Σ i are color, opacity, mean, and covariance of Gaussians, respectively. Gaussians capture finer detail than colored meshes but hinder consistency between appearance, geometry, and simulation; following recent work [78], we adopt a Gaussianson-Mesh (GoM) approach and attach each splat to its mesh to ensure alignment and enable physical interactions. Given camera intrinsics K and extrinsics ξ , we denote the splat-rendered RGB images, masks, depth and normal maps as I , M , D , N = SplatRender ( G ; K , ξ ) .

Figure 2: Overview of HoloScene Optimization Stages: Given multiple posed images as well as some visual cues (instance masks, monocular geometry priors), we first employ a gradient-based optimization as the initialization. Then we adopt a generative sampling and tree search strategy along the topology of the scene graph to obtain the complete geometry with physical plausibility. Finally, the final fine-tuning over the scene further enhances the realism of the reconstructed scene.

<!-- image -->

Physics: Each object in our scene graph is modeled as a rigid body. Its physical parameters p i = ( m i , κ i , ζ i , r i ) comprise mass m i , friction κ i (resistance to sliding against other surfaces), damping ζ i (energy dissipation during motion), and restitution r i (elasticity upon impact). These parameters are used in downstream physical simulations to model the object's response to external forces and its interactions with other objects and the background scene.

Object States: All object intrinsic attributes above are defined in object-centric coordinates and remain invariant under motion. To handle dynamic changes, we encode each object's rigid body state by a rigid transform T i from its object-centric frame to the world frame. During static reconstruction, T i is fixed; in dynamic simulation, it may vary over time. Let T = { T i } N i =0 denote the set of all object states, G T the scene graph under those states, and G the scene graph under the static state during reconstruction.

Object Relationships: Each edge e i,j links two nodes with one of three relationships: 1) support , where v i rests in stable equilibrium on its unique parent v pa( i ) under gravity (each object has exactly one such parent, so support edges form a tree in the static scene graph); 2) beside , where siblings ( pa( j ) = pa( i ) ) have touching surfaces, causing occlusions without hierarchy or instability; and 3) collide , where contacts with nonzero momentum yield dynamic effects-ignored during static reconstruction but employed in simulation. Note that the object relationship might change depending on its dynamic status during simulation.

Interaction &amp; Simulation: Our 3D scene graph's distinguishing feature is its support for physical interactions. Formally, at time t , given the dynamic scene graph G T t with current object states T t as well as an input action a t , the next states are computed as

<!-- formula-not-decoded -->

where Sim is a rigid-body physical simulator using the mesh {M i } as collision geometry. Here, a t can represent external inputs- forces, torques, or control actions-applied to the objects at time t .

## 3.2 Problem Formulation

Our framework takes input observations O of a static scene and recovers the scene graph G = ( V , E ) . The resulting scene graph must (i) explain the observations well; (ii) be geometrically complete and plausible; and (iii) reflect the scene's static, physically stable nature. To this end, we cast the problem as a structured energy-minimization problem:

<!-- formula-not-decoded -->

For simplicity, we omit the hyperparameter linear weights for each term. Next, we discuss each energy term.

Observation Terms: The observation terms quantify the discrepancy between the reconstructed 3D scene and the input observations. Let ˆ I t , ˆ M t , ˆ D t , ˆ N t = SplatRender ( G ; K , ξ t ) denote the rendered RGB image, instance mask, depth map, and normal map at camera pose ξ t . We then define three energy terms: the RGB energy E rgb ( I , G ) = ∑ t L MSE ( ˆ I t , I t ) + L LPIPS ( ˆ I t , I t ) , where I t is the ground-truth color image and the loss combines MSE and LPIPS losses [96]; the mask energy E mask ( M , G ) = ∑ t CE ( ˆ M t , M t ) , where CE is cross-entropy and M t is either a given labeled mask [64, 89] or one inferred via segmentation tracking [24]; and the monocular geometry energy E mono ( D , G ) = ∑ t ∥ ˆ N t -N t ∥ 2 2 + L norm ( ˆ D t , D t ) , where N t and D t are monocular normal and depth priors and L norm is the scale- and shift-invariant L2 loss [91].

Regularization Terms: Because videos only partially observe a 3D scene, optimizing observations alone cannot yield a complete, plausible, and physically valid reconstruction; we therefore impose generative, geometric, and physical priors as regularizers to enable fully interactive 3D scenes.

The completeness energy E comp encourages complete reconstruction of each object's shape despite the partial observations. Inspired by generative image-to-3D methods [39], for each object i we synthesize virtual observations ˜ O i = { ˜ I i , ˜ D i , ˜ M i , ˜ N i } by 'shooting' it from multiple virtual viewpoints with a pretrained multi-view diffusion model Wonder3D [39]. Unlike the single object setting for most image-to-3D works, because our complex scenes often feature inter-object occlusions (e.g., a sofa covered by a blanket), we first inpaint occluded regions using LaMa [66] before generating these views. Given the synthesized observations, we define the completeness energy as

<!-- formula-not-decoded -->

where E rgb , E mono , E mask are the observation losses defined similarly in our observation terms, although they are measured at virtual viewpoints here.

The geometry energy E geo ensures geometry compatibility between each object, such that their geometry does not intersect with each other:

<!-- formula-not-decoded -->

̸

The SDF-penetration term E pene \_ sdf = ∑ x ∈ R i ∑ k = i max ( 0 , -g k ( x ) -g i ( x ) ) ensures no two object SDFs overlap, where R ( i ) = { x ∈ R 3 | arg min k g k ( x ) = i } is the set of points belong to instance i . Intuitively, if x lies in instance i , then for any other instance k , g k ( x ) ≥ -g i ( x ) must hold to prevent intersections. Similarly, each object's mesh should not intersect with any other object mesh. This can be measured by measuring whether intersecting two meshes resulting empty set or not: E pene \_ mesh = 1 ( inter ( M i , M j ) = ∅ ) .

̸

Finally, it is important to ensure that our recovered digital twin of the scene is simulatable; hence, physical plausibility is crucial. To this end, we introduce physics energy , which measures physical plausibility via two terms:

<!-- formula-not-decoded -->

The stable term E stable ( G ) = Diff ( T , Sim ( T , a gravity ; G ) ) quantifies translational and rotational deviations of each object, with Diff ( T , T ′ ) = ∑ i ( | trans ( T -1 i T ′ i ) | + | rad ( T -1 i T ′ i ) | ) and Sim is

Figure 3: Qualitative Comparisons on Object Geometry and Appearance Reconstruction: Our method delivers superior reconstructions by smoothly inpainting occluded regions with LaMa and completing invisible back-facing geometry with Wonder3D. Unlike baselines, our approach eliminates object interpenetration, ensuring physical stability during simulation.

<!-- image -->

the forward physical simulator step as defined in Eq. 1; a low E stable indicates static equilibrium under gravity, i.e. scene remains static in the simulator. The touch term E touch = ∑ ( i,j ) ∈E dist ( M i , M j ) encourages each supporting pair ( i, j ) to make contact, dist is the Chamfer distance between meshes.

## 3.3 Inference

Optimizing the scene graph from Sec. 3.2 is challenging because it mixes discrete variables (graph topology, object-object relations) with continuous ones (neural SDFs, Gaussians, and physical parameters) and includes non-differentiable terms like physical stability. We therefore use a three-stage divide-and-conquer approach: first, recover initial geometry and appearance by minimizing the observation terms; next, refine shapes and physical parameters by minimizing the geometry and physics terms through generative sampling combined with structured tree search; and finally, fine-tune appearance by re-minimizing observation terms. This yields a fully plausible, interactive 3D scene (Fig. 2).

Stage 1: Gradient-based Optimization: We first optimize each object node's appearance a i and geometry g i to match the observations O via gradient-based optimization. Specifically, we minimize the observation terms plus SDF-penetration regularization through differentiable volume rendering-similar to neural SDF methods [75, 74, 30]-to obtain per-instance SDFs g i . Additionally, we recover small objects by balancing training samples across all instances. We then extract initial meshes M i via marching cubes and refine each object's Gaussians f i via splat rendering and RGB rendering, mask, and monocular geometry losses, yielding our dual scene representation per each instance [16].

Stage 2: Sampling-based Optimization: The Stage 1 scene model supports freeview rendering and accurate visible-region geometry but remains incomplete, non-physical, and non-interactive. Directly minimizing E complete , E physics , and E geo , however, is challenging due to complex highorder interactions (e.g., multi-object physical interaction), intrinsic multi-modality (invisible regions admit multiple solutions), and non-differentiable components (e.g., mesh intersections, physics simulations). To address this, we adopt an approach that combines the diverse proposal capability of

generative sampling with the combinatorial optimization strength of tree search to minimize our structured objective.

Scene graph edges creation: After the gradient-based optimization, our framework infers the topology of the scene graph G from the instance meshes extracted from neural SDF, where edges E encode support relations in a tree rooted at v 0 (the background, e.g., the room). We build this tree by analyzing physical contact and support relationships between objects: each object is assigned a parent based on which surface provides its primary support, as determined by the geometry and orientation of contact surfaces. We begin by identifying objects resting directly on the background, then recursively process the remaining objects in an order that respects the physical dependency structure-objects that support others are registered before those they support. Starting from v 0 , we repeat until all observed instances have been added to the tree.

Generative sampling: We begin by sampling diverse, complete shapes for each instance: we prompt Wonder3D's multi-diffusion model with real-world observations and generate virtual views ˜ I i from various viewpoints. Thanks to its generative nature, resampling multiple times with different seeds yields diverse virtual observations even from the same viewpoints. Then, for each virtual observation, we minimize E comp independently, producing a diverse set of 3D shape candidates per object.

Structured tree search: We have generated multiple complete shape samples per instance that all fit the observations, but it remains unclear which combination is most physically plausible. Exhaustively evaluating every combination is impractical, since the physical-plausibility energy E physics entails a high-order combinatorial optimization. To address this, we perform a tree search over our generative samples. Starting at the root node, we traverse each node in breadth-first order; at each active node (object), we evaluate E physics for all samples and retain the sample with the lowest energy among those evaluated. We then adjust its state and physical parameters to enforce stability and prevent interpenetration (see details in the supplementary material).

Remark: The key novelty and advantage of the proposed inference algorithm is the combination of generative sampling with a structured tree search for amodal, physically plausible reconstruction. Unlike scene-level amodal methods [10] that rely on asset retrieval, our sampling is asset-free and generates input consistent, diverse shape hypotheses. Unlike prior simulation-verification methods [49, 48], which only enforces object-ground consistency, our tree search ensures global stability along every support chain. By driving a non-differentiable simulator (e.g., IsaacSim) end-to-end, we eliminate any reconstruction-to-deployment gap.

Stage 3: Gradient-based Refinement Since Stage 2 adjusts object states, physical parameters, and shape, it is necessary to further refine the Gaussians attached to the surface to ensure a complete and realistic appearance. To this end, we fine-tune Gaussians for all objects using splat rendering by minimizing the observation terms via gradient descent. This yields our final scene graph.

## 4 Experiments

Dataset: We conduct the experiments across multiple datasets: 3 scenes from Replica [64], 3 scenes from Scannet++ [89], 2 scenes from iGibson [26], and one self-captured scene. The Replica and Scannet++ datasets cover diverse indoor structures and lighting conditions, and the iGibson dataset offers complete geometry of every object, allowing per-object reconstruction evaluation. Instance masks are provided by dataset annotations or estimated with SAM [24].

Metrics: We evaluate geometry quality with Chamfer Distance (CD), F-Score (F1), and Normal Consistency (NC) [91], and assess rendering quality using PSNR, SSIM, and LPIPS. For physical evaluation, we adopt consistent physical parameters, put all scene components in the Isaac Sim [47], and measure stability with translation/rotation changes when gravity is applied. The stability ratio is calculated as: Stable % = # Stable Instances # All Instances , where instances are identified as stable if changes are under a certain threshold. We also report the object reconstruction ratio: OR % = # Reconstructed Instances #All Instances , which quantifies how many objects are present in the final reconstruction, regardless of their completeness.

Baselines: We evaluate our framework against SOTA approaches in instance-aware amodal 3D scene reconstruction. ObjectSDF++ [75] uses per-instance SDF for scene representation. PhyRecon [48] extends instance-aware scene reconstruction by incorporating a differentiable physical loss to

Figure 4: Qualitative Comparisons on Physical Simulation: We compare geometry layouts and appearance before and after physical simulation, with the table geometry reconstructions highlighted in inset figures. HoloScene's complete, non-interpenetrating geometry remains stable in physics simulators, unlike baseline methods. Our Gaussian on mesh delivers high-quality, real-time rendering throughout the simulation process.

<!-- image -->

Table 2: Quantitative Results on Scene Reconstruction: HoloScene's generative sampling and scene graph-based tree search produce the most physically plausible reconstructions while preserving high-quality geometry and supporting real-time and realistic rendering. We highlight the best and second-best methods with distinct colors. For rendering quality, we only compare with DP-Recon's texture mesh-based rendering since ObjSDF++ and PhyRecon lack real-time rendering capabilities.

|        | Method   | Geometry   | Geometry   | Geometry   | Rendering   | Rendering   | Rendering   | Rendering   | Physics   | Physics             | Physics          |
|--------|----------|------------|------------|------------|-------------|-------------|-------------|-------------|-----------|---------------------|------------------|
|        | Method   | CD ↓       | F1 ↑       | NC ↑       | PSNR ↑      | SSIM ↑      | LPIPS ↓     | Real Time   | OR % ↑    | Stable (Ground) % ↑ | Stable (All) % ↑ |
|        | ObjSDF++ | 6.72       | 64.36      | 88.53      | 29.12       | 0.851       | 0.355       | ✗ ✗         | 98.6      | 78.3                | 39.4             |
|        | PhyRecon | 4.52       | 71.07      | 92.06      | 23.19       | 0.764       | 0.434       |             | 77.5      | 56.5                | 5.6              |
|        | DP-Recon | 3.45       | 87.66      | 94.23      | 22.10       | 0.728       | 0.420       | ✓           | 56.3      | 21.7                | 8.5              |
|        | Ours     | 4.05       | 83.21      | 92.21      | 27.82       | 0.849       | 0.304       | ✓           | 100.0     | 95.7                | 81.7             |
| Scene  | ObjSDF++ | 25.20      | 70.71      | 87.15      | 27.46       | 0.887       | 0.292       | ✗           | 96.5      | 81.6                | 28.2             |
| Scene  | PhyRecon | 31.16      | 39.57      | 82.28      | 22.32       | 0.791       | 0.432       | ✗           | 92.9      | 67.3                | 9.4              |
| Scene  | DP-Recon | 22.96      | 65.48      | 87.13      | 21.44       | 0.715       | 0.466       | ✓           | 90.6      | 20.0                | 9.4              |
| Scene  | Ours     | 21.93      | 63.11      | 88.09      | 25.88       | 0.873       | 0.268       | ✓           | 100.0     | 93.9                | 70.6             |
|        | ObjSDF++ | 12.33      | 38.64      | 83.74      | 29.60       | 0.891       | 0.299       | ✗           | 65.0      | 44.2                | 36.1             |
|        | PhyRecon | 11.27      | 45.49      | 83.85      | 27.40       | 0.860       | 0.333       | ✗           | 62.9      | 45.3                | 5.2              |
|        | DP-Recon | 30.31      | 21.89      | 70.81      | 21.94       | 0.728       | 0.432       | ✓           | 74.2      | 16.3                | 4.1              |
|        | Ours     | 12.00      | 34.15      | 82.91      | 25.88       | 0.854       | 0.301       | ✓           | 100.0     | 74.4                | 71.1             |
| Object | ObjSDF++ | 3.52       | 79.03      | 75.30      | 11.03       | 0.571       | 0.134       | ✗           | 65.0      | 44.2                | 36.1             |
|        | PhyRecon | 5.47       | 70.71      | 71.89      | 8.92        | 0.609       | 0.250       | ✗           | 62.9      | 45.3                | 5.2              |
|        | DP-Recon | 5.81       | 61.31      | 70.61      | 13.90       | 0.770       | 0.301       | ✓           | 74.2      | 16.3                | 4.1              |
|        | Ours     | 3.17       | 81.31      | 78.13      | 16.55       | 0.863       | 0.185       | ✓           | 100.0     | 74.4                | 71.1             |

optimize unstable objects. However, it only handles object-ground interactions and does not support hierarchical or inter-object relationships. DP-Recon [49] incorporates diffusion Score Distillation Sampling (SDS) [55] for amodal sparse-view reconstruction. However, it does not handle inter-object occlusions. All three methods use instance-level SDFs, so the SDF values of different instances might interfere with one another. We adopt their open-source codes and adapt them for the testing benchmarks. Please refer to the supplementary material for more implementation details.

Implementation details: Our inference pipeline consists of three stages. Stage 1 employs gradientbased optimization for 100k steps with loss weights λ rgb = 1 . 0 , λ mask = 0 . 5 , λ depth = 0 . 5 , and λ normal = 0 . 1 . Stage 2 uses sampling-based optimization with λ rgb = 2 . 0 , λ mask = 0 . 5 , λ depth = 10 . 0 , λ normal = 10 . 0 , λ pene = 5 . 0 , generating three samples per instance. Stage 3 refines Gaussians via gradient-based optimization with λ t 1 = 0 . 95 and λ ssim = 0 . 05 . The optimization takes approximately 4 hours, 4 hours, and 20 minutes for stages 1, 2, and 3, respectively, on a single A6000 GPU. We utilize Marigold [14] for monocular depth and normal estimation.

Table 3: Ablation Study on Model Design: We observe improved physical stability and object-level reconstruction with the help of generative priors, physical energy, and scene graph representation, and there is a trade-off between scene-level reconstruction and instance-level physical stability.

<!-- image -->

| Level   | Dataset   | TexMesh / GoM   | Physical Energy   | Scene Graph   | CD ↓           | F1 ↑                    | NC ↑              | PSNR ↑      | SSIM ↑                  | LPIPS ↓                 | OR % ↑                  | Stable (All) % ↑    |
|---------|-----------|-----------------|-------------------|---------------|----------------|-------------------------|-------------------|-------------|-------------------------|-------------------------|-------------------------|---------------------|
| Scene   | Replica   | TexMesh GoM     | ✗ ✗               | ✗ ✗ ✗         | 3.50 3.50 4.32 | 88.00 88.00 80.19 83.21 | 92.30 92.30 92.10 | 26.45 28.30 | 0.810 0.845 0.839 0.849 | 0.329 0.349 0.334 0.304 | 100.0 100.0 100.0 100.0 | 43.7 43.7 69.0 81.7 |
| Scene   |           | GoM             | ✗                 | ✗             |                | 79.22                   |                   | 27.14       |                         |                         |                         |                     |
| Scene   |           | GoM             | ✓                 |               |                |                         | 92.21             |             |                         |                         |                         |                     |
| Scene   |           | GoM             | ✓                 | ✓             | 4.05           |                         |                   | 27.82       |                         |                         |                         |                     |
| Object  | iGibson   |                 |                   |               | 4.20           |                         | 76.96             | 13.74       | 0.735                   | 0.204                   | 100.0                   | 41.2                |
| Object  |           | GoM             | ✓                 | ✓             | 3.17           | 81.31                   | 78.13             | 16.55       | 0.863                   | 0.185                   | 100.0                   | 71.1                |

## 4.1 Experimental Results

Scene-level evaluations: We first evaluate the reconstruction at the scene level. In Tab. 2, we compare our framework with the three baselines across Replica, Scannet++. and iGibson dataset, with qualitative results shown in Fig. 3 and Fig. 4. Our method achieves the best physical reconstruction results in terms of both object reconstruction ratio and stable object ratio, while maintaining comparable scene-level reconstruction quality in geometry and appearance. Unlike baselines [75, 48, 50] where small objects often disappear due to SDF interference from larger adjacent objects, our framework benefits from balancing training samples across all instances during the optimization stage 1, recovering all instances in the scenes. PhyRecon assumes that all objects rest directly on the ground, which damages geometry when objects are supported by other objects. DP-Recon prioritizes completeness via SDS over physical stability, failing to adequately address object interpenetration. In contrast, our sampling-based optimization and completion approach yields the most physically stable results.

Object-level evaluations: We evaluate object-level reconstruction using the iGibson dataset, which provides complete ground truth geometry and appearance for each object. This evaluation is especially challenging as it tests the reconstruction of occluded regions that models never observe. Our evaluation compares reconstructed geometry directly with ground truth and renders 6 viewpoints around each object to assess appearance quality. Results in Tab. 2 demonstrate that our method outperforms all baselines in reconstructing invisible and occluded regions, validating our framework's effectiveness in completing objects beyond directly observed surfaces.

Ablation study: Our ablation study (Tab. 3) reveals the contribution of each component. Switching from textured mesh to Gaussian rendering improves visual quality, while adding physics energy with generative priors enhances physical stability. The scene graph inter-object relationships further improve physics performance by better reconstructing occluded regions, especially those from support relationships. However, we identify a trade-off between scene-level reconstruction accuracy and physical stability, where optimizing for physical plausibility may occasionally compromise pixel-alignment with original observations.

Failure rate analysis: We conduct a statistical analysis of failure modes in geometry and physics reconstruction. In Tab. 4, we compare HoloScene against reconstruction-based baselines [75, 48, 50] across geometry failures on iGibson and physics failures on all three datasets, with failure criteria defined by F-score thresholds (70.0) and simulation stability metrics. Geometry failures include invalid surfaces, partial and oversized shapes, while physics failures arise from invalid shapes incompatible with simulators, overlooked object-object contacts, inter-object penetration causing repelling forces, and inaccurate contact modeling leading to positional drift. Our method achieves the lowest failure rates in both categories. Unlike ObjSDF++ [75] and PhyRecon [48] which lack shape priors and produce invalid geometry, our generative priors eliminate all invalid geometry failures through reliable structural guidance and balanced sampling. DP-Recon [50] emphasizes visual completeness over physical constraints, resulting in high penetration failures. Additionally, PhyRecon's ground-contact assumption overlooks contact pairs that can destabilize the simulation. In contrast, we integrate generative priors with physics-aware constraints directly into the reconstruction process, which substantially reduces both invalid shapes and inter-object penetrations, demonstrating the effectiveness of physical consistency in the optimization.

Table 4: Quantitative Failure Rate Comparison: HoloScene achieves significantly lower geometry and physics failure rates compared to reconstruction-based baselines, demonstrating more robust 3D reconstruction and physically plausible scene modeling.

| Method   | Geometry Failure (iGibson)   | Geometry Failure (iGibson)   | Geometry Failure (iGibson)   | Geometry Failure (iGibson)   | Geometry Failure (iGibson)   | Physics Failure Rate (%)   | Physics Failure Rate (%)   | Physics Failure Rate (%)   | Physics Failure Breakdown (All Datasets)   | Physics Failure Breakdown (All Datasets)   | Physics Failure Breakdown (All Datasets)   | Physics Failure Breakdown (All Datasets)   | Physics Failure Breakdown (All Datasets)   |
|----------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|----------------------------|----------------------------|----------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|
| Method   | Total #                      | Fail #                       | Invalid #                    | Partial #                    | Oversized #                  | Replica                    | ScanNet++                  | iGibson                    | Total #                                    | Invalid #                                  | Overlap #                                  | Penetration #                              | Drifting #                                 |
| ObjSDF++ | 97                           | 52                           | 34                           | 13                           | 5                            | 60.6                       | 71.8                       | 53.6                       | 235                                        | 38                                         | 0                                          | 76                                         | 52                                         |
| PhyRecon | 97                           | 64                           | 36                           | 21                           | 7                            | 94.4                       | 90.6                       | 66.0                       | 235                                        | 58                                         | 121                                        | 33                                         | 24                                         |
| DP-Recon | 97                           | 70                           | 25                           | 33                           | 12                           | 91.5                       | 90.6                       | 72.2                       | 235                                        | 64                                         | 0                                          | 102                                        | 69                                         |
| Ours     | 97                           | 24                           | 0                            | 13                           | 11                           | 18.3                       | 29.4                       | 24.7                       | 235                                        | 0                                          | 0                                          | 6                                          | 60                                         |

## 4.2 Interactive Environment Applications

Real-Time Interactive Game With our reconstructed environment, we can create a real-time interactive game with Unreal Engine [12]. As Fig. 1 shows, we build a third-person game with the reconstructed texture meshes. The objects could be physically rearranged in the game world, and the game agent could also interact with the scene through realistic physics.

Interactive 3D Editing In our simulation environment, we could also achieve high-quality interactive 3D editing by moving the object Gaussians with its underlying physical mesh geometry. In Fig. 1, we demonstrate this by changing the location and orientation of the interactable chair.

Immersive Experience Recording We show our interactable reconstructed 3D objects with immersive experience recording. Given a static RGB video of a person manipulating an object, we aim to recover the object's 6D pose and resimulate its motion in a virtual 3D scene. We recover the camera pose with VGGT [69], adjust the predicted depth [54] to align with the virtual scene, and adopt FoundationPose [73] for object tracking with our reconstructed 3D object for model-based 6D pose estimation. As shown in Fig. 1, we enable consistent replay of real-world interactions in virtual scenes while accurately recovering the object's pose from visual input.

<!-- image -->

Figure 5: Dynamic VFX Results. We augment the inferred interactive 3D scene with various visual effects such as dropping objects, adding animations, and fires.

<!-- image -->

<!-- image -->

To enhance immersion, we augment the scene with dynamic visual effects, including rigid body simulations, character animations, and particle effects. We adopt visual effects from AutoVFX [18] to overlay virtual content and shadows onto the image. As Fig. 4 shows, we produce effects that blend naturally with the scene.

## 5 Conclusion &amp; Limitation

We presented HoloScene, a novel interactive 3D modeling framework that uses scene graphs and energy-based optimization to reconstruct environments with realistic appearance, complete geometry, and interactive physical plausibility, achieving superior real-time rendering, geometric accuracy, and stable simulation. Limitations : HoloScene currently only handles videos of static indoor scenes; dynamic scenes and large outdoor environments remain challenging. Future work will focus on relightable reconstruction and extending support to articulated and deformable objects.

Acknowledgment: This project is supported by the Intel AI SRS gift and NSF Awards #2331878, #2340254, #2312102, #2414227, and #2404385. We greatly appreciate the NCSA for providing computing resources through Delta and Delta AI program.

## References

- [1] Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- [2] Amini, A., Wang, T.H., Gilitschenski, I., Schwarting, W., Liu, Z., Han, S., Karaman, S., Rus, D.: Vista 2.0: An open, data-driven simulator for multimodal sensing and policy learning for autonomous vehicles. In: ICRA (2022)
- [3] Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In: ICCV (2021)
- [4] Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In: CVPR (2022)
- [5] Berger, M., Tagliasacchi, A., Seversky, L.M., Alliez, P., Levine, J.A., Sharf, A., Silva, C.T.: State of the art in surface reconstruction from point clouds. In: 35th Annual Conference of the European Association for Computer Graphics, Eurographics 2014-State of the Art Reports. The Eurographics Association (2014)
- [6] Chen, B., Jiang, H., Liu, S., Gupta, S., Li, Y., Zhao, H., Wang, S.: Physgen3d: Crafting a miniature interactive world from a single image. arXiv preprint arXiv:2503.20746 (2025)
- [7] Chen, Y., Ni, J., Jiang, N., Zhang, Y., Zhu, Y., Huang, S.: Single-view 3D scene reconstruction with high-fidelity shape and texture. In: 2024 International Conference on 3D Vision (3DV). IEEE (2024)
- [8] Chen, Y., Rong, F., Duggal, S., Wang, S., Yan, X., Manivasagam, S., Xue, S., Yumer, E., Urtasun, R.: Geosim: Realistic video simulation via geometry-aware composition for self-driving. In: CVPR (2021)
- [9] Chen, Z., Walsman, A., Memmel, M., Mo, K., Fang, A., Vemuri, K., Wu, A., Fox, D., Gupta, A.: Urdformer: A pipeline for constructing articulated simulation environments from real-world images. arXiv (2024)
- [10] Dai, T., Wong, J., Jiang, Y., Wang, C., Gokmen, C., Zhang, R., Wu, J., Fei-Fei, L.: Acdc: Automated creation of digital cousins for robust policy learning. arXiv (2024)
- [11] Dogaru, A., Özer, M., Egger, B.: Generalizable 3d scene reconstruction via divide and conquer from a single view. arXiv preprint arXiv:2404.03421 (2024)
- [12] Epic Games: Unreal engine https://www.unrealengine.com
- [13] Feng, Y., Feng, X., Shang, Y., Jiang, Y., Yu, C., Zong, Z., Shao, T., Wu, H., Zhou, K., Jiang, C., et al.: Gaussian splashing: Dynamic fluid synthesis with gaussian splatting. arXiv e-prints pp. arXiv-2401 (2024)
- [14] Garcia, G.M., Abou Zeid, K., Schmidt, C., De Geus, D., Hermans, A., Leibe, B.: Fine-tuning imageconditional diffusion models is easier than you think. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 753-762. IEEE (2025)
- [15] Guo, M., Wang, B., Ma, P., Zhang, T., Owens, C., Gan, C., Tenenbaum, J., He, K., Matusik, W.: Physically compatible 3d object modeling from a single image. Advances in Neural Information Processing Systems 37 , 119260-119282 (2024)
- [16] Guédon, A., Lepetit, V.: Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering (2023)
- [17] Han, X.F., Laga, H., Bennamoun, M.: Image-based 3d object reconstruction: State-of-the-art and trends in the deep learning era. IEEE transactions on pattern analysis and machine intelligence 43 (5), 1578-1604 (2019)
- [18] Hsu, H.Y., Lin, Z.H., Zhai, A., Xia, H., Wang, S.: Autovfx: Physically realistic video editing from natural language instructions. arXiv preprint arXiv:2411.02394 (2024)
- [19] Hu, S., Zhou, K., Li, K., Yu, L., Hong, L., Hu, T., Li, Z., Lee, G.H., Liu, Z.: Consistentnerf: Enhancing neural radiance fields with 3d consistency for sparse view synthesis. arXiv preprint arXiv:2305.11031 (2023)
- [20] Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2d gaussian splatting for geometrically accurate radiance fields. In: SIGGRAPH. Association for Computing Machinery (2024)
- [21] Jiang, H., Hsu, H.Y., Zhang, K., Yu, H.N., Wang, S., Li, Y.: Phystwin: Physics-informed reconstruction and simulation of deformable objects from videos. arXiv preprint arXiv:2503.17973 (2025)

- [22] Jiang, Y., Yu, C., Xie, T., Li, X., Feng, Y ., Wang, H., Li, M., Lau, H., Gao, F., Yang, Y ., et al.: Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1-1 (2024)
- [23] Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. TOG (2023)
- [24] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: ICCV (2023)
- [25] Kwak, M.S., Song, J., Kim, S.: Geconerf: Few-shot neural radiance fields via geometric consistency. arXiv preprint arXiv:2301.10941 (2023)
- [26] Li, C., Xia, F., Martín-Martín, R., Lingelbach, M., Srivastava, S., Shen, B., Vainio, K., Gokmen, C., Dharan, G., Jain, T., et al.: igibson 2.0: Object-centric simulation for robot learning of everyday household tasks. arXiv preprint arXiv:2108.03272 (2021)
- [27] Li, X., Li, J., Zhang, Z., Zhang, R., Jia, F., Wang, T., Fan, H., Tseng, K.K., Wang, R.: Robogsim: A real2sim2real robotic gaussian splatting simulator. arXiv preprint arXiv:2411.11839 (2024)
- [28] Li, Y., Lin, Z.H., Forsyth, D., Huang, J.B., Wang, S.: Climatenerf: Physically-based neural rendering for extreme climate synthesis. arXiv (2022)
- [29] Li, Z., Müller, T., Evans, A., Taylor, R.H., Unberath, M., Liu, M.Y., Lin, C.H.: Neuralangelo: High-fidelity neural surface reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8456-8465 (2023)
- [30] Li, Z., Lyu, X., Ding, Y., Wang, M., Liao, Y., Liu, Y.: Rico: Regularizing the unobservable for indoor compositional reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17761-17771 (2023)
- [31] Lin, Z.H., Huang, J.B., Li, Z., Dong, Z., Richardt, C., Li, T., Zollhöfer, M., Kopf, J., Wang, S., Kim, C.: Iris: Inverse rendering of indoor scenes from low dynamic range images. arXiv preprint arXiv:2401.12977 (2024)
- [32] Lin, Z.H., Liu, B., Chen, Y.T., Forsyth, D., Huang, J.B., Bhattad, A., Wang, S.: Urbanir: Large-scale urban scene inverse rendering from a single video (2023)
- [33] Ling, L., Lin, C.H., Lin, T.Y ., Ding, Y ., Zeng, Y ., Sheng, Y ., Ge, Y ., Liu, M.Y ., Bera, A., Li, Z.: Scenethesis: A language and vision agentic framework for 3d scene generation. arXiv preprint arXiv:2505.02836 (2025)
- [34] Liu, J.Y., Chen, Y., Yang, Z., Wang, J., Manivasagam, S., Urtasun, R.: Real-time neural rasterization for large scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8416-8427 (2023)
- [35] Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero-1-to-3: Zero-shot one image to 3d object (2023)
- [36] Liu, S., Ren, Z., Gupta, S., Wang, S.: Physgen: Rigid-body physics-grounded image-to-video generation. In: European Conference on Computer Vision. pp. 360-378. Springer (2024)
- [37] Liu, Y., Jia, B., Lu, R., Ni, J., Zhu, S.C., Huang, S.: Building interactable replicas of complex articulated objects via gaussian splatting. arXiv preprint arXiv:2502.19459 (2025)
- [38] Liu, Z., Zhou, G., He, J., Marcucci, T., Li, F.F., Wu, J., Li, Y.: Model-based control with sparse neural dynamics. Advances in Neural Information Processing Systems 36 , 6280-6296 (2023)
- [39] Long, X., Guo, Y.C., Lin, C., Liu, Y., Dou, Z., Liu, L., Ma, Y., Zhang, S.H., Habermann, M., Theobalt, C., et al.: Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:2310.15008 (2023)
- [40] Lu, F., Xu, Y., Chen, G., Li, H., Lin, K.Y ., Jiang, C.: Urban radiance field representation with deformable neural mesh primitives (2023)
- [41] Lu, R., Chen, Y., Ni, J., Jia, B., Liu, Y ., Wan, D., Zeng, G., Huang, S.: Movis: Enhancing multi-object novel view synthesis for indoor scenes. arXiv preprint arXiv:2412.11457 (2024)
- [42] Luo, R., Geng, H., Deng, C., Li, P., Wang, Z., Jia, B., Guibas, L., Huang, S.: Physpart: Physically plausible part completion for interactable objects. arXiv preprint arXiv:2408.13724 (2024)

- [43] Ma, S., Du, W., Yu, C., Jiang, Y., Zong, Z., Xie, T., Chen, Y ., Yang, Y ., Han, X., Jiang, C.: Grip: A general robotic incremental potential contact simulation dataset for unified deformable-rigid coupled grasping. arXiv preprint arXiv:2503.05020 (2025)
- [44] Manivasagam, S., Wang, S., Wong, K., Zeng, W., Sazanovich, M., Tan, S., Yang, B., Ma, W.C., Urtasun, R.: Lidarsim: Realistic lidar simulation by leveraging the real world. In: CVPR (2020)
- [45] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: ECCV (2020)
- [46] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. ACM Communications (2021)
- [47] Mittal, M., Yu, C., Yu, Q., Liu, J., Rudin, N., Hoeller, D., Yuan, J.L., Singh, R., Guo, Y., Mazhar, H., Mandlekar, A., Babich, B., State, G., Hutter, M., Garg, A.: Orbit: A unified simulation framework for interactive robot learning environments. IEEE Robotics and Automation Letters 8 (6), 3740-3747 (2023). https://doi.org/10.1109/LRA.2023.3270034
- [48] Ni, J., Chen, Y., Jing, B., Jiang, N., Wang, B., Dai, B., Li, P., Zhu, Y ., Zhu, S.C., Huang, S.: Phyrecon: Physically plausible neural scene reconstruction. arXiv preprint arXiv:2404.16666 (2024)
- [49] Ni, J., Liu, Y ., Lu, R., Zhou, Z., Zhu, S.C., Chen, Y ., Huang, S.: Decompositional neural scene reconstruction with generative diffusion prior (2025)
- [50] Ni, J., Liu, Y ., Lu, R., Zhou, Z., Zhu, S.C., Chen, Y ., Huang, S.: Decompositional neural scene reconstruction with generative diffusion prior. arXiv preprint arXiv:2503.14830 (2025)
- [51] Oechsle, M., Peng, S., Geiger, A.: Unisurf: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5589-5599 (2021)
- [52] Özsoy, E., Czempiel, T., Örnek, E.P., Eck, U., Tombari, F., Navab, N.: Holistic or domain modeling: a semantic scene graph approach. International Journal of Computer Assisted Radiology and Surgery 19 (5), 791-799 (2024)
- [53] Park, J.J., Florence, P., Straub, J., Newcombe, R., Lovegrove, S.: Deepsdf: Learning continuous signed distance functions for shape representation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 165-174 (2019)
- [54] Piccinelli, L., Sakaridis, C., Yang, Y.H., Segu, M., Li, S., Abbeloos, W., Van Gool, L.: Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv preprint arXiv:2502.20110 (2025)
- [55] Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3D using 2D diffusion. arXiv preprint arXiv:2209.14988 (2022)
- [56] Powell, M.J.: An efficient method for finding the minimum of a function of several variables without calculating derivatives. The computer journal 7 (2), 155-162 (1964)
- [57] Pun, A., Sun, G., Wang, J., Chen, Y., Yang, Z., Manivasagam, S., Ma, W.C., Urtasun, R.: Lightsim: Neural lighting simulation for urban scenes (2023), https://arxiv.org/abs/2312.06654
- [58] Pun, A., Sun, G., Wang, J., Chen, Y., Yang, Z., Manivasagam, S., Ma, W.C., Urtasun, R.: Neural lighting simulation for urban scenes. NeurIPS (2024)
- [59] Qian, S., Fouhey, D.F.: Understanding 3D object interaction from a single image. In: ICCV (2023)
- [60] Ren, T., Liu, S., Zeng, A., Lin, J., Li, K., Cao, H., Chen, J., Huang, X., Chen, Y ., Yan, F., Zeng, Z., Zhang, H., Li, F., Yang, J., Li, H., Jiang, Q., Zhang, L.: Grounded sam: Assembling open-world models for diverse visual tasks (2024), https://arxiv.org/abs/2401.14159
- [61] Seo, S., Han, D., Chang, Y., Kwak, N.: Mixnerf: Modeling a ray with mixture density for novel view synthesis from sparse inputs. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20659-20668 (2023)
- [62] Somraj, N., Soundararajan, R.: Vip-nerf: Visibility prior for sparse input neural radiance fields. In: ACM SIGGRAPH 2023 Conference Proceedings. pp. 1-11 (2023)
- [63] Son, S., Qiao, Y.L., Sewall, J., Lin, M.C.: Differentiable hybrid traffic simulation. TOG (2022)

- [64] Straub, J., Whelan, T., Ma, L., Chen, Y., Wijmans, E., Green, S., Engel, J.J., Mur-Artal, R., Ren, C., Verma, S., Clarkson, A., Yan, M., Budge, B., Yan, Y., Pan, X., Yon, J., Zou, Y., Leon, K., Carter, N., Briales, J., Gillingham, T., Mueggler, E., Pesqueira, L., Savva, M., Batra, D., Strasdat, H.M., Nardi, R.D., Goesele, M., Lovegrove, S., Newcombe, R.: The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797 (2019)
- [65] Sucar, E., Liu, S., Ortiz, J., Davison, A.J.: imap: Implicit mapping and positioning in real-time. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 6229-6238 (2021)
- [66] Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., Kong, N., Goka, H., Park, K., Lempitsky, V.: Resolution-robust large mask inpainting with fourier convolutions. arXiv preprint arXiv:2109.07161 (2021)
- [67] Torne, M., Simeonov, A., Li, Z., Chan, A., Chen, T., Gupta, A., Agrawal, P.: Reconciling reality through simulation: A real-to-sim-to-real approach for robust manipulation. arXiv (2024)
- [68] Wang, H., Li, M.: A new era of indoor scene reconstruction: A survey. IEEE Access (2024)
- [69] Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. arXiv preprint arXiv:2503.11651 (2025)
- [70] Wang, L., Guo, R., Vuong, Q., Qin, Y., Su, H., Christensen, H.: A real2sim2real method for robust object grasping with neural surface reconstruction. In: 2023 IEEE 19th International Conference on Automation Science and Engineering (CASE). pp. 1-8. IEEE (2023)
- [71] Wang, P., Liu, L., Liu, Y ., Theobalt, C., Komura, T., Wang, W.: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv (2021)
- [72] Wang, Y., Qiu, X., Liu, J., Chen, Z., Cai, J., Wang, Y ., Wang, T.H., Xian, Z., Gan, C.: Architect: Generating vivid and interactive 3d scenes with hierarchical 2d inpainting. Advances in Neural Information Processing Systems 37 , 67575-67603 (2024)
- [73] Wen, B., Yang, W., Kautz, J., Birchfield, S.: Foundationpose: Unified 6d pose estimation and tracking of novel objects. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17868-17879 (2024)
- [74] Wu, Q., Liu, X., Chen, Y., Li, K., Zheng, C., Cai, J., Zheng, J.: Object-compositional neural implicit surfaces. In: European Conference on Computer Vision. pp. 197-213. Springer (2022)
- [75] Wu, Q., Wang, K., Li, K., Zheng, J., Cai, J.: Objectsdf++: Improved object-compositional neural implicit surfaces. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 21764-21774 (2023)
- [76] Wu, T., Zheng, C., Guan, F., Vedaldi, A., Cham, T.J.: Amodal3r: Amodal 3d reconstruction from occluded 2d images. arXiv preprint arXiv:2503.13439 (2025)
- [77] Xia, H., Lin, Z.H., Ma, W.C., Wang, S.: Video2game: Real-time interactive realistic and browser-compatible environment from a single video. In: CVPR (2024)
- [78] Xia, H., Su, E., Memmel, M., Jain, A., Yu, R., Mbiziwo-Tiapo, N., Farhadi, A., Gupta, A., Wang, S., Ma, W.C.: Drawer: Digital reconstruction and articulation with environment realism (2025), https: //arxiv.org/abs/2504.15278
- [79] Xiang, J., Lv, Z., Xu, S., Deng, Y ., Wang, R., Zhang, B., Chen, D., Tong, X., Yang, J.: Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506 (2024)
- [80] Xie, T., Zong, Z., Qiu, Y., Li, X., Feng, Y., Yang, Y., Jiang, C.: Physgaussian: Physics-integrated 3d gaussians for generative dynamics. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4389-4398 (2024)
- [81] Xiong, Y., Ma, Wei-Chiu Wang, J., Urtasun, R.: Ultralidar: Learning compact representations for lidar completion and generation. CVPR (2023)
- [82] Yang, J., Ivanovic, B., Litany, O., Weng, X., Kim, S.W., Li, B., Che, T., Xu, D., Fidler, S., Pavone, M., Wang, Y.: Emernerf: Emergent spatial-temporal scene decomposition via self-supervision. In: ICLR (2024)
- [83] Yang, Y., Sun, F.Y., Weihs, L., VanderBilt, E., Herrasti, A., Han, W., Wu, J., Haber, N., Krishna, R., Liu, L., et al.: Holodeck: Language guided generation of 3d embodied ai environments. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16227-16237 (2024)

- [84] Yang, Z., Chen, Y., Wang, J., Manivasagam, S., Ma, W.C., Yang, A.J., Urtasun, R.: Unisim: A neural closed-loop sensor simulator. CVPR (2023)
- [85] Yang, Z., Chai, Y., Anguelov, D., Zhou, Y., Sun, P., Erhan, D., Rafferty, S., Kretzschmar, H.: Surfelgan: Synthesizing realistic sensor data for autonomous driving. In: CVPR (2020)
- [86] Yao, K., Zhang, L., Yan, X., Zeng, Y., Zhang, Q., Xu, L., Yang, W., Gu, J., Yu, J.: Cast: Component-aligned 3D scene reconstruction from an rgb image. arXiv preprint arXiv:2502.12894 (2025)
- [87] Yariv, L., Gu, J., Kasten, Y ., Lipman, Y .: V olume rendering of neural implicit surfaces. Advances in Neural Information Processing Systems 34 , 4805-4815 (2021)
- [88] Yariv, L., Hedman, P., Reiser, C., Verbin, D., Srinivasan, P.P., Szeliski, R., Barron, J.T., Mildenhall, B.: Bakedsdf: Meshing neural sdfs for real-time view synthesis. In: SIGGRAPH Conference (2023)
- [89] Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset of 3D indoor scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12-22 (2023)
- [90] Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A.: Mip-splatting: Alias-free 3d gaussian splatting (2024)
- [91] Yu, Z., Peng, S., Niemeyer, M., Sattler, T., Geiger, A.: Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. arXiv (2022)
- [92] Yu, Z., Sattler, T., Geiger, A.: Gaussian opacity fields: Efficient and compact surface reconstruction in unbounded scenes. arXiv (2024)
- [93] Yuan, Y., Bleier, M., Nuchter, A.: Scenefactory: A workflow-centric and unified framework for incremental scene modeling. IEEE Transactions on Robotics (2025)
- [94] Zhai, A.J., Shen, Y., Chen, E.Y., Wang, G.X., Wang, X., Wang, S., Guan, K., Wang, S.: Physical property understanding from language-embedded feature fields (2024)
- [95] Zhang, K., Li, B., Hauser, K., Li, Y.: Particle-grid neural dynamics for learning deformable object models from rgb-d videos. In: Proceedings of Robotics: Science and Systems (RSS) (2025)
- [96] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR (2018)
- [97] Zhao, H., Wang, H., Zhao, X., Wang, H., Wu, Z., Long, C., Zou, H.: Automated 3d physical simulation of open-world scene with gaussian splatting. arXiv preprint arXiv:2411.12789 (2024)
- [98] Zyrianov, V., Che, H., Liu, Z., Wang, S.: Lidardm: Generative lidar simulation in a generated world (2024), https://arxiv.org/abs/2404.02903
- [99] Zyrianov, V., Zhu, X., Wang, S.: Learning to generate realistic lidar point clouds. In: ECCV (2022)

## Appendix

## A More Interactive Environment Application Results

Real-Time Interactive Game With the reconstruction results of HoloScene, we could create an interactive game in Unreal Engine [12], which supports real-time interactions with our created assets. We import all the objects and the background texture mesh into the game engine and create a thirdperson game, where the agent can freely explore, navigate, jump, and interact with the objects in the scene. Thanks to the complete geometry and appearance, as well as the great physical plausibility of HoloScene, the objects could remain stable in the game. Also, the movements of objects could follow physical rules when the objects interact. The Fig. 1 shows the game screenshots, and more videos could be found in the supp. material as well.

Interactive 3D Editing With each object's appearance and geometry reconstructed, we support interactive 3D scene editing by allowing users to manipulate individual Gaussians through a control panel. Objects can be translated along the 2D xy-plane and rotated around the vertical (z) axis. As illustrated in Fig. 2, the reconstructed swivel chair is repositioned freely and rendered realistically, resulting in coherent and intuitive scene modifications.

Immersive Experience Recording We demonstrate the usage of the reconstructed 3D objects from HoloScene for immersive experience recording. Given a static RGB video of a person manipulating an object, our goal is to recover the object's 6D pose and resimulate its motion within a virtual 3D scene. We use the first frame of the video as the target view for camera calibration. To determine the camera pose, we first estimate relative transformations between the target and nearby training views using VGGT [69]. We then solve a least-squares problem with Powell's method [56] to derive a global transformation that aligns the VGGT-predicted poses with the ground-truth poses of the training views. This transformation is applied to register the target view within the training coordinate system. We use UniDepthV2 [54] to predict the depth map for each frame and compute a scale ratio between the predicted monocular depth and depth rendered from the 3D Gaussians. This ratio adjusts the predicted depth to align accurately within the virtual scene. For object tracking, we adopt FoundationPose [73], using our high-quality reconstructed 3D asset as a CAD model for model-based 6D pose estimation. The recovered pose sequence is then applied to the 3D object and its 3D Gaussians for final rendering. As illustrated in Fig. 3, we accurately recover the object's pose from visual input, enabling consistent replay of real-world interactions in virtual scenes.

Dynamic Visual Effects We showcase an immersive application by augmenting the reconstructed scenes with dynamic visual effects. These include rigid body simulations, 3D character animations, and particle-based effects, all spatially grounded within the virtual scene. We adopt multiple visual effects created by AutoVFX [18] to simulate virtual content and overlay it, along with shadows, onto the input image. As shown in Fig. 4, we produce effects that blend naturally with the scene, enhancing both visual richness and physical realism.

<!-- image -->

Figure 1: More Interactive Game Results: We show more screenshots of the interactive game built by Unreal Engine here. The agent can run, jump, and interact with all the objects in the game.

<!-- image -->

<!-- image -->

## B More Physical Simulation Results

In this section, we will show more physical simulation results of Holoscene and its comparison with the baseline methods [75, 48, 49]. In Fig. 5 and Fig. 6, we show the comparisons of our method and

<!-- image -->

Figure 2: More Interactive 3D Editing Results: The user could interact with the object in the scene, change the location and orientation of the object the control panel, and attain the edited scene.

<!-- image -->

Pose 2

Pose 3

Figure 3: More Immersive Experience Recording Results: Here we show more results of the immersive experience recording of two objects in the scene. For each image pair, the original recording is shown on the left, and the simulated object rendering with Gaussians is on the right. The estimated object 6D poses are marked with oriented bounding boxes.

the baselines in the Scannet++ [89] dataset. In Fig. 7, we show the comparisons in the Replica [64] dataset. In Fig. 8, we show the comparisons in the iGibson [26] dataset. From the results, we can observe that HoloScene can achieve the best physical stability and plausibility, as well as realistic rendering quality.

## C Method Details

Structured Tree Search When we perform the breadth-first traversal on the scene tree structure, for each node, we will end up choosing the reconstructed object candidate with the lowest physical energy. Then we will further adjust the state of the chosen candidate object to ensure its stability. Specifically, we will regularize the SDF value and the mesh vertex locations to prevent the interpenetration of objects. Before applying marching cube to the instance SDF, we will first prune the SDF value of the points if those points fall into the boundary of its parent object or the nearby objects. After we attain the mesh with marching cube, we will perform a collision test of the reconstructed object mesh with its parent object and the nearby objects. And if there exist collisions, we will dynamically optimize the locations of the vertices to avoid the collisions. After the optimization, the contact surfaces of

Figure 4: More Dynamic Visual Effect Results: We show more results of dynamic visual effects, including animations, dropping objects, and setting fires.

<!-- image -->

objects will be regularized and avoid inter-object collisions, so that the stability of the scene can be further improved.

## D Experiment Details

## D.1 Datasets

We adopt three datasets as we mentioned in our main paper: Replica [64], Scannet++ [89], and iGibson [26]. Replica: The original Replica dataset only provides the texture mesh with instance annotations, and no existing videos with camera trajectories are provided. We manually design a dense camera trajectory of several hundred frames for each scene, and render images with a size of (512, 512). Scannet++: The scannet++ dataset provides multi-view posed images and the aligned 3D scan with instance annotation. However, limited by the quality of the scan, the instance mask derived from the scan annotation is not ideal. We adopt [24] to get cleaner instance masks instead. For the image resolution, we downsample the original image by 2. iGibson: The iGibson dataset provides the complete geometry and appearance of every object and the background scene, but without provided camera trajectories. We manually design a dense camera trajectory of several hundred frames for each scene, and render images with a size of (512, 512).

## D.2 Baselines

We adopt the open-source codes of the baselines and adapt them for our benchmark. Our baselines are ObjectSDF++ [75], PhyRecon [48], and DP-Recon [49]. Now we will elaborate the details of them.

ObjectSDF++: The original codebase only supports their rendering images from Replica [64], where each image has a size of (384, 384). In order to make it adaptable to other image resolutions, we implement a new data loader in the code.

PhyRecon: Similar to ObjectSDF, the original code base only supports their rendering images from Replica [64] and a different resolution of resized images from Scannet++ [89]. For the adaption to other image resolutions, we implement a new data loader in the code. The PhyRecon has a final stage

<!-- image -->

Ref Image

Figure 5: More Physical Simulation Results in Scannet++ Dataset Scene 1: We show the geometry and appearance of each object in the scene with physical simulation, from no physics state, intermediate state with physics, and final state with physics.

<!-- image -->

of applying physical loss to every object. But it may happen that some objects fail to reconstruct themselves, so the physical loss can not be applied to those objects. In this case, we skip those objects.

DP-Recon: Similar to the baselines above, we implement a new data loader to adapt the code to our dataset. DP-Recon has the geometry stage and the following appearance stage. We follow the officially released code to run those experiments. DP-Recon needs to generate the object bounding boxes in the running process. But DP-Recon overlooks the possibility that some objects may fail to be reconstructed and disappear. So in this case, we set the bounding box of the object as the largest possible bounding box in the scene to fix the issue. We evaluate the appearance, geometry, and physics based on the final texture mesh results from the second stage of DP-Recon.

## D.3 The inference details of HoloScene

The inference stage of HoloScene is composed of three stages after generating the scene graph in stage 0, as we mentioned in the main paper. In stage 1 of the gradient-based optimization, we follow the previous instance-level SDF [75] to set the hyperparameters and train for 200k iterations similarly. In stage 2 of the sampling-based optimization, we adopt the Isaac Sim [47] as the physical simulator,

Ref Image

Figure 6: More Physical Simulation Results in Scannet++ Dataset Scene 2: We show the geometry and appearance of each object in the scene with physical simulation, from no physics state, intermediate state with physics, and final state with physics.

<!-- image -->

and set the physical parameters in the simulation as default. In the sampling with generative models, we randomly choose the seeds and generate at most five candidates for each object. In stage 3 of the gradient-based refinement, we refine the trained Gaussians globally. We optimize the Gaussians for 30k iterations following [23], and in each iteration, we also use generated images to optimize the instance Gaussians as well. For other learning parameters, they are set to default in our experiments.

<!-- image -->

Ref Image

<!-- image -->

No Physics

With Physics (Intermediate)

With Physics (Final)

Figure 7: More Physical Simulation Results in Replica Dataset: We show the geometry and appearance of each object in the scene with physical simulation, from no physics state, intermediate state with physics, and final state with physics.

<!-- image -->

Ref Image

<!-- image -->

No Physics

With Physics (Intermediate)

With Physics (Final)

Figure 8: More Physical Simulation Results in iGibson Dataset: We show the geometry and appearance of each object in the scene with physical simulation, from no physics state, intermediate state with physics, and final state with physics. DP-Recon fails to reconstruct the complete background mesh, leading to object falling in the figure.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our claims match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations of the work in our paper.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA]

Justification: Our paper doesn't focus on the theoretical derivation.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We fully disclose our method and the experiment settings in the paper.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We would open-source the data and code of the paper.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We elaborate the experiment settings in detail in our paper.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We conduct multiple sets of experiments to alleviate the statistical significance of the experiments.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We provide the training details in the supplementary material.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have read and conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the potential impacts in the paper.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper has no such risks.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The creators or original owners of assets used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.