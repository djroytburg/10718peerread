## Local-Global Associative Frames for Symmetry-Preserving Crystal Structure Modeling

## Haowei Hua 1 , Wanyu Lin 1 , 2 ∗

1 2 Department of Data Science and Artificial Intelligence

Department of Computing, The Hong Kong Polytechnic University, Hong Kong SAR, China haowei.hua@connect.polyu.hk, wan-yu.lin@polyu.edu.hk

## Abstract

Crystal structures are defined by the periodic arrangement of atoms in 3D space, inherently making them equivariant to SO(3) group. A fundamental requirement for crystal property prediction is that the model's output should remain invariant to arbitrary rotational transformations of the input structure. One promising strategy to achieve this invariance is to align the given crystal structure into a canonical orientation with appropriately computed rotations, or called frames . However, existing work either only considers a global frame or solely relies on more advanced local frames based on atoms' local structure. A global frame is too coarse to capture the local structure heterogeneity of the crystal, while local frames may inadvertently disrupt crystal symmetry, limiting their expressivity. In this work, we revisit the frame design problem for crystalline materials and propose a novel approach to construct expressive S ymmetryP reserving Frame s, dubbed as SPFrame , for modeling crystal structures. Specifically, this local-global associative frame constructs invariant local frames rather than equivariant ones, thereby preserving the symmetry of the crystal. In parallel, it integrates global structural information to construct an equivariant global frame to enforce SO(3) invariance. Extensive experimental results demonstrate that SPFrame consistently outperforms traditional frame construction techniques and existing crystal property prediction baselines across multiple benchmark tasks.

## 1 Introduction

Fast and accurate prediction of crystal properties is essential for accelerating the discovery of novel materials, as it enables efficient screening of promising candidates from vast material space [38]. Traditional approaches based on high-fidelity quantum-mechanics calculations, such as density functional theory (DFT), can provide acceptable error margin for property predictions but they require high computational resources [58], thereby limiting their practical deployment. As an alternative, machine learning models have shown great potential for predicting crystal material properties with both efficiency and accuracy. These methods typically leverage 3D geometric graph representations of crystals in conjunction with geometric graph neural networks (GGNNs) [2, 36, 3, 55] or transformerbased variants of GGNNs [57, 49, 56, 27, 51, 19] to establish mappings between crystal structural data and their properties.

When establishing the structure-property mapping, crystal structures are defined by a periodic arrangement of atoms in 3D space, which inherently makes them equivariant under SO(3) group transformations (i.e., rotations). However, many crystal properties, such as formation energy, are scalar quantities that remain invariant under such transformations. Consequently, to ensure accurate property prediction, GGNNs must maintain invariance when input structures are subjected to SO(3)

∗ Corresponding author.

transformations. For this purpose, early studies have employed SO(3)-invariant features, such as simple interatomic distances [55, 56], which prevent the designed GGNNs from capturing rich interatomic directional information. More recent works has attempted to incorporate interatomic directional information [57] while preserving SO(3)-invariance by carefully designing the network architecture, such as converting directional vectors into angle-based representations [57]. Although this approach successfully ensures invariance, it imposes architectural constraints that can limit the flexibility and scalability of neural networks in modeling complex crystal structures.

Alternatively, the global frame approach can be integrated with any GGNNs without imposing constraints on the network architecture, while still ensuring compliance with the SO(3) invariance requirement. These global frames are constructed in an equivariant manner (such as PCA frame [9]) with respect to the input structure, effectively aligning the structure to a canonical orientation [19, 15, 50]. However, because the same frame is applied to all atoms in the structure, the global frame approach lacks the ability to capture the local structure heterogeneity, limiting their expressivity. To address this limitation, recent developments have shifted toward local frame strategies, where distinct frames are dynamically constructed for each atom based on its local structure [19, 34, 50]. This approach allows for greater differentiation among atomic local environment, thereby improving the expressivity and enhancing the model performance [34, 50]. Despite these advantages, directly applying this general equivariant local frame to crystal structures may unintentionally disrupt the symmetry of the crystal, as discussed in Section 3.1. This disruption hinders the model's ability to distinguish atoms located at Wyckoff positions, thereby weakening its capacity to capture structural details.

To address these challenges, we revisit the problem of frame design for crystals and analyze the root cause of the symmetry breaking observed in previous equivariant local frame methods. Specifically, we identify that constructing local frames based solely on the local atomic structure often breaks the symmetry of the crystal. Motivated by this insight and the symmetry characteristics of crystal structures, we propose a Symmetry-Preserving Frame (SPFrame) method for property prediction. SPFrame constructs invariant local frames rather than equivariant ones. For atoms located at symmetry-equivalent positions, SPFrame assigns identical invariant local frames, which allows their relative local relationships to be preserved after frame transformation. Based on these invariant local frames, an equivariant global frame is further incorporated. Since the same global frame is applied to all atoms, it enforces SO(3) invariance across the structure without disrupting the symmetry of the crystal structure. This local-global associative design enables SPFrame to overcome the symmetry-breaking issue observed in previous local frame methods, enhancing the model's ability to differentiate between distinct atomic local structures. The effectiveness of the SPFrame approach is evaluated on two widely used benchmark datasets for crystalline materials.

## 2 Preliminaries

## 2.1 Coordinate Systems for Crystal Structure Representation

Crystalline materials are defined by a periodic 3D arrangement of atoms, where the smallest repeating unit, known as the unit cell, fully determines the entire crystal structure. Prior studies [56, 52, 21] have established two primary coordinate systems for representing such crystal structure mathematically.

Cartesian Coordinate System. A crystal structure is formally defined by the triplet M = ( A , X , L ) . The matrix A = [ a 1 , a 2 , · · · , a n ] ⊤ ∈ R n × d a contains feature vectors for n atoms within a unit cell, where each row a i ∈ R d a describes the individual atom feature. The 3D Cartesian coordinates of n atoms in the unit cell are encoded in X = [ x 1 , x 2 , · · · , x n ] ⊤ ∈ R n × 3 . The lattice maxtrix L = [ l 1 , l 2 , l 3 ] ∈ R 3 × 3 consists of the lattice vectors l 1 , l 2 , and l 3 , which form the basis of the 3D space. The complete crystal structure emerges through periodic repetition: ( ˆ A , ˆ X ) = { ( ˆ a i , ˆ x i ) | ˆ x i = x i + k 1 l 1 + k 2 l 2 + k 3 l 3 , ˆ a i = a i , k 1 , k 2 , k 3 ∈ Z , i ∈ Z , 1 ≤ i ≤ n } , where integer coefficients k 1 , k 2 , k 3 generate all possible atomic positions in the periodic lattice.

Fractional Coordinate System. This system employs lattice vectors l 1 , l 2 , and l 3 as basis, with atomic positions expressed as f i = [ f i, 1 , f i, 2 , f i, 3 ] ⊤ ∈ [0 , 1) 3 . The conversion to Cartesian coordinates is defined as x i = ∑ j f i,j l j , where j = 1 , 2 , 3 . This yields an crystal representation M = ( A , F , L ) , where F = [ f 1 , · · · , f n ] ⊤ ∈ [0 , 1) n × 3 contains the fractional coordinates of all atoms in the unit cell.

<!-- image -->

X

p4mm Wyckoff Positions Figure 1: An illustration with 2D plain group P4mm [52, 14, 21]. (a) The figure illustrates the lattice of the P4mm space group, visually demonstrating equivalent positions; the symmetry-equivalent positions are indicated by the same color. (b) The table depicts the Wyckoff positions present in this lattice.

Multiplicity

Wyckoff letter

## 1 a 2.2 The Symmetry of Crystal Structures

1

b

Fractional coordinates

(0,0)

(1/2,1/2)

2 c (1/2,0), (0,1/2) 4 d (x,0), (-x,0), (0,x), (0,-x) 4 e (x,1/2), (-x,1/2), (1/2,x), (1/2,-x) SO ( 3 ) group. The SO(3) group consists of all rotations in 3D space. Its elements are rotation matrices defined as { Q | Q ∈ R 3 × 3 , Q ⊤ Q = I , det ( Q ) = 1 } . When applied to crystal data M , an SO(3) transformation yields M ′ = ( A , QX , QL ) .

4

f

(x,x), (x,-x), (-x,x), (-x,-x)

(0,0) X 8 g (x,y), (x,-y), (-x,y), (х, -у), (у,х), (у, -x), (у,х), ( -у, -х) Space group. The E(3) group encompasses all rigid transformations including rotations, reflections, and translations. Its elements can be denoted by the pair { ( Q , t ) | Q ∈ R 3 × 3 , Q ⊤ Q = I , t ∈ R 3 } , where Q is an orthogonal matrix and t is a translation vector. When an E(3) transformation is applied to the crystal data M , certain elements ( Q g , t g ) can map a crystal structure back onto itself due to the inherent symmetry of the structure. These specific elements ( Q g , t g ) are collectively referred to space groups. Mathematically, ( A , Q g X + t g , Q g L ) = ( A , X , L ) , where the symbol ' = ' indicates the equivalence between geometric structures.

Wyckoff positions. The concept of space groups leads to the definition of Wyckoff positions, which are sets of symmetry-equivalent atomic sites within a unit cell [20, 21]. Each Wyckoff position is characterized by three fundamental attributes: multiplicity, Wyckoff letter, and fractional coordinates. As shown in Fig. 1, for 2D plain group P4mm, the Wyckoff positions obey specific coordinate constraints, including 0 ≤ x ≤ 0 . 5 , 0 ≤ y ≤ 0 . 5 , x ≤ y , and the identical atomic occupation requirement [20, 21].

## 2.3 Crystal Structure Invariant Learning

SO(3)-invariance requirement for crystal properties prediction. The SO(3) group transformation can alter the orientation of a crystal structure within 3D space [57]. Nevertheless, many critical material properties, such as formation energy, are invariant under the SO(3) group transformation. Consequently, for effective crystal property prediction, it is essential that the model can exhibit SO(3)-invariant prediction capabilities. Specifically, for a prediction model denoted as f θ ( · ) , if it is SO(3)-invariant, for any rotation matrix Q ∈ R 3 × 3 , the following equality holds:

<!-- formula-not-decoded -->

Frame. Frame-based methodologies have shown promising in enforcing equivariance and invariance in geometric deep learning [50, 33, 37]. In the context of SO(3) group transformations, a frame can be interpreted as a rotation matrix F ∈ R 3 × 3 , deriving from a SO(3)-equivariant map denoted as h ( X ) . This frame transforms the atomic positions X into an SO(3)-invariant representation represented as XF ⊤ . Crucially, this representation remains unchanged under arbitrary rotations Q : XF ⊤ Q - → XQ ( FQ ) ⊤ = XQQ ⊤ F ⊤ = XF ⊤ , thus decoupling the SO(3) invariance requirement for neural network design. Additionally, the concept of a global frame involves using a single frame for all atoms within the atomic system, whereas a local frame is defined for each atom individually, with its calculation based on the atom's local structure. A more detailed discussion of related works can be found in Section 4.

(1,1)

Y

Figure 2: Using the 2D plane group P4mm as a running example, we demonstrate why local frame methods may disrupt the symmetry of crystal. For atoms p and q that belong to the same Wyckoff position type, their local structures can be related by a 90-degree rotation after graph construction. Since equivariant local frames are constructed solely based on local structural information, the resulting frames for p and q also exhibit a 90-degree rotational relationship, applying these frames eliminates the relative orientation between the two local structures. In contrast, SPFrame preserves these relative structural differences by incorporating global structural information during frame construction.

<!-- image -->

## 3 Methodology

In this section, we first outline the motivation behind our work, with a particular emphasis on the limitations of existing local frame methods when applied to crystal structures, as discussed in Section 3.1. To address these challenges, we introduce the proposed SPFrame method, a local-global associative frame. We further incorporate this method into the established crystal property prediction architecture, yielding a new framework for crystal structure modeling, as described in Section 3.2.

## 3.1 Symmetry Breaking Induced by the Local Frame

Building upon previous works [22, 57, 51], we begin by presenting the general formulation of message passing at k -th layer in GGNNs using SO(3)-equivariant edge features, defined as follows:

<!-- formula-not-decoded -->

where f ( k ) i denotes the feature vector of atom i , e ij ∈ R d represents SO(3)-invariant edge features (e.g., embeddings of interatomic distances between atoms i and j ), and ̂ e ij ∈ R 3 corresponds to SO(3)-equivariant edge features capturing directional information between atoms (e.g., the edge vector between atoms i and j ). The functions ϕ ( k ) ( · ) and ψ ( k ) ( · ) are learnable non-linear mappings that define the message construction and aggregation processes, respectively.

When integrating the local frame into the message passing framework, the local frame is adaptively defined for each atom, thereby transforming the equivariant edge features into invariant features. Consequently, the message passing process is reformulated as follows:

<!-- formula-not-decoded -->

where F i denotes the local frame for atom i . As illustrated in Section 2.3, the presence of local frames ensures that the output of the GGNNs remains invariant under the SO(3) group transformation.

However, as discussed in Section 2.2, the symmetry of crystal structures implies that atoms occupying the same Wyckoff position type exhibit similar local structures. As illustrated in Figure 2, when constructing the crystal graph, atoms p and q , which belong to the same Wyckoff position type, share identical atom features and invariant edge features (such as interatomic distances). The only distinction between these atoms lies in equivariant edge features. These equivariant features, representing the relative directional vectors of atoms p and q with respect to their neighboring atoms, are related through a 90 degrees rotation matrix Q 90 ◦ .

When local frames are incorporated into GGNNs, they serve to canonicalize the equivariant edge features ̂ e ij by mapping them to invariant representations ̂ e ij F ⊤ i . Since the local frames F p and F q are constructed equivariantly based on the local structures, it follows that F q = Q 90 ◦ F p . As a result, atoms p and q , which initially exhibit distinct orientations in their respective local structures, are aligned to a common orientation, rendering their local structures indistinguishable. This process diminishes the model's ability to distinguish symmetry-equivalent yet spatially distinct atoms, thereby limiting the expressivity of GGNNs. Furthermore, such equivalence between atoms p and q under SO(3) transformations is commonly observed in crystals with screw axes or rotational symmetries, such as those found in space groups like P 2 1 , among others [14].

## 3.2 Our SPFrame

To address the challenges outlined above, several critical considerations must be taken into account. First, it is essential to decouple the SO(3) invariance requirement imposed on GGNNs when employing local frames. Simultaneously, for atoms located at equivalent Wyckoff positions, it is imperative to preserve the relative relationships within their local structures following the application of the local frame. This preservation ensures that the GGNN can effectively differentiate between these atoms. Second, in line with the standard definition of local frames, distinct frames should be assigned to atoms occupying non-equivalent positions.

SO(3) symmetry decoupling and crystal symmetry preserving. As discussed in Section 3.1, conventional local frame construction assigns different frames to symmetry-equivalent atoms p and q , which can inadvertently disrupt the crystal's symmetry. To mitigate this issue in Figure 2, a straightforward yet effective strategy is to assign identical frames to atoms p and q . This design ensures that the relative orientation relationships between their local structures are preserved after the frame transformation. For atoms that are not symmetry-equivalent, distinct frames should be constructed to reflect the differences in their local structures. Therefore, We now introduce the SPFrame, defined as

<!-- formula-not-decoded -->

where F INV ,i denotes the invariant local frame for atom i and F global denotes the equivariant global frame shared across the atomic system. Using SPFrame, we reformulate Equation 3 as follows:

<!-- formula-not-decoded -->

Under an SO(3) group transformation applied to the entire atomic system, the presence of global frames ensures that the output of the GGNNs remains invariant, thereby decoupling the SO(3) invariance requirement. Since the global frame is computed based on global structural information, it remains consistent across symmetry-equivalent atoms and thus does not disrupt the symmetry of the crystal. At the same time, the invariant local frame F INV ,i is constructed using an invariant method based on local structural information. Thus, atoms p and q , which occupy the same type of Wyckoff position, are assigned identical local frames. Consequently, the transformed SO(3)-invariant representation, ̂ e ij F ⊤ global F ⊤ INV ,i , remains distinguishable for atoms p and q , while preserving their relative structural differences.

This local-global associative design enables the model to satisfy SO(3) invariance while maintaining the crystal's symmetry. Furthermore, since the invariant local frame F INV ,i still considers local information, the SPFrame for non-symmetry-equivalent positions are computed differently. In addition, we provide a theoretical justification for the superiority of SPFrame over the local frame, as detailed in Appendix A.1. This work also guarantees SE(3) invariance, as detailed in Appendix A.2.

Algorithm 1 Quaternion to Rotation Matrix Conversion

Require:

Quaternion q = [ a, b, c, d ] ∈ R 4

Ensure:

Rotation matrix Q ∈ R 3 × 3

- 1: Normalize the quaternion:

<!-- formula-not-decoded -->

- 2: Compute the rotation matrix Q :

<!-- formula-not-decoded -->

Symmetry-preserving frame construction. As illustrated in Equation 4, the proposed SPFrame consists of two components: a shared global frame F global applied to all atoms, and a set of atomspecific invariant local frames F INV ,i . The global frame F global plays a key role in decoupling the SO(3) invariance requirement from the GGNN. For simplicity, we propose to use non-parametric approaches such as QR decomposition [33]. Additional implementation details can be found in Appendix A.3.

Correspondingly, the invariant local frame F INV ,i is designed to enhance the expressive power of the GGNN. Since it does not need to independently enforce SO(3) symmetry constraints, this component allows for better flexibility in frame construction. To this end, we employ quaternions [42, 46, 13] as a compact and numerically stable representation of the rotations.

Specifically, we first predict a quaternion for each atom based on its local structure. Inspired by previous work [50, 34], we leverage a message passing scheme to generate the quaternion embeddings:

<!-- formula-not-decoded -->

where q i ∈ R 4 denotes the predicted quaternion for atom i . The message function ϕ ( · ) and the aggregation function ψ ( · ) can be instantiated using any SO(3)-invariant message passing architecture. In this work, we adopt transformer-based implementations following Yan et al. [56]. Once the quaternion q i is obtained, it is converted into the corresponding rotation matrix [42] via the mapping LF ( · ) (Further details can be found in Appendix A.4). The pseudocode for LF ( · ) is presented in Algorithm 1.

Network architecture. As demonstrated in previous studies [50, 34], local frames can be effectively integrated into GGNNs that utilize equivariant edge features for message passing. Among these models, eComFormer [57] represents the state of the art in crystal property prediction, leveraging equivariant edge features to enhance message propagation. Based on this, we adopt eComFormer as the backbone architecture for implementing and evaluating the proposed SPFrame strategy. More details on the integration of SPFrame into eComFormer can be found in Appendix A.5.

## 4 Related Works

Crystal Property Prediction. GGNNs and their transformer-based extensions (hereafter collectively referred to as GGNNs for convenience) have been widely adopted in crystal property prediction due to their capacity to model complex atomic interactions. Several representative methods, such as CGCNN [55], MEGNet [2], GATGNN [36], Matformer [56], PotNet [32], DOSTransformer [27], and CrystalFormer [49], construct crystal graphs by utilizing SO(3)-invariant interatomic distances as edge features. By avoiding the use of equivariant directional vectors, these models ensure SO(3)invariant predictions. Similarly, models such as ALIGNN [3], M3GNet [1], Crystalformer [51], and iComFormer [57] leverage invariant angular information as edge features to maintain SO(3)invariance in prediction. Beyond these, several methods adopt more specialized strategies. For example, eComFormer [57] utilizes equivariant edge features, which subsequently are transformed into two-hop invariant angular representations for preserving invariance.

Global Frame. Frames are widely used in both equivariant and invariant learning. However, earlier frame methods, such as the frame averaging (FA) method [41, 9], rely on frame construction techniques like PCA, which produce non-unique frames. This necessitates the use of specially designed loss functions during training to learn invariant representations across all frame-transformed variants. More recently, minimal frame methods [33] have adopted frame construction techniques such as QR decomposition to produce unique frames, thereby improving the efficiency of frames. This is the type of global frame described in this work. It is worth noting that another approach in equivariant and invariant learning, i.e. canonicalization [37, 23, 10, 43], is equivalent to the minimal frame method to some extent [37].

Local Frame. Similar to the global frame, the local frame is also a method that transforms equivariant data representations into invariant ones [7, 50, 8, 40, 34, 19]. The difference lies in that the local frame approach generates a separate frame for each atom in the atomic system, which can enhance the expressiveness of GGNNs [50]. Recent work, Crystalframer [19], was the first to introduce local frames into the field of crystal property prediction. Building upon the attention mechanism from [49], it designed two types of equivariant local frames and recalculated different local frames at various network layers. However, as mentioned above, the use of general equivariant local frames may unintentionally decouple the symmetry of the crystal structure.

Beyond the aforementioned studies, we also review approaches that incorporate crystal symmetry into method design, together with other key strategies for enhancing predictive accuracy. A more discussion is provided in Appendix A.6.

## 5 Experiments

To validate the effectiveness of the proposed SPFrame, we performed a comprehensive series of experiments on crystal property prediction. Additionally, we conducted comparative analyses between our method and existing equivariant local frame approaches. A detailed summary of the experimental setup and results is provided below.

## 5.1 Experimental setup

Datasets. We utilize two widely used crystal property benchmark datasets: JARVIS-DFT and Materials Project (MP). Following previous work [56, 57, 19], we perform predction tasks of formation energy, total energy, bandgap, and energy above hull (E hull) on JARVIS-DFT dataset. For the MP dataset, we perform predction tasks of formation energy, bandgap, bulk modulus, and shear modulus.

Baseline Methods. We selected several state-of-the-art methods in the field, including CGCNN [55], SchNet [45], MEGNet [2], GATGNN [36], ALIGNN [3], Matformer [56], PotNet [32], Crystalformer [49], eComFormer [57], iComFormer [57], and Crystalframer [19], as baseline methods for comparison.

Frame Comparison and Ablation Studies. In addition to the aforementioned crystal property prediction methods, we also conducted comparative experiments by replacing the proposed SPFrame with other frame methods integrated into the backbone network. The first method is an SO(3)equivariant local frame. Inspired by Wang and Zhang [50] and Lippmann et al. [34], we design this approach using Gram-Schmidt orthogonalization to construct SO(3)-equivariant local frames. This method serves as a baseline for examining the impact of breaking the symmetry of crystal structures on model performance. The second method is an SPFrame variant constructed using Gram-Schmidt orthogonalization, allowing for a more direct comparison with the first method, as both rely on the same orthogonalization procedure. This comparison further enables the evaluation of the advantages of the quaternion-based SPFrame. Additional details on the design of these two frame baselines can be found in Appendix A.7.

Experimental Settings. Following prior work [57], we evaluate model performance using Mean Absolute Error (MAE) and optimize all models using the Adam optimizer. We conduct our experiments on NVIDIA GeForce RTX 3090 GPUs, with complete hyperparameter configurations (including learning rates, batch sizes, and training epochs) provided in Appendix A.8. In our evaluation, we highlight the best-performing results in bold and indicate second-best performances with underlining.

## 5.2 Experimental Results

JARVIS. Table 1 presents the experimental results on JARVIS. The eComFormer architecture, when combined with the our proposed SPFrame, achieves best performance on all prediction tasks, demonstrating consistent improvements over existing approaches. In the comparison of different frame methods, the performance of the SO(3)-equivariant Gram-Schmidt local frame is inferior to that of both the Gram-Schmidt-based SPFrame and the quaternion-based SPFrame in all prediction tasks. This observation confirms that maintaining crystal symmetry while applying local frames enables GGNNs to better distinguish between atoms, leading to improved prediction accuracy. Furthermore, the superior performance of the quaternion-based SPFrame over the Gram-Schmidt-based SPFrame emphasizes that quaternion-derived rotation matrices provide a more effective representation for frame construction in crystal materials. To further demonstrate the generality of SPFrame, we conducted additional experiments combining SPFrame with another backbone architectures. The corresponding results are provided in Appendix A.9.

Table 1: Property prediction results on the JARVIS dataset.

|                                                |   Form. energy eV/atom |   Total energy eV/atom |   Bandgap (OPT) eV |   Bandgap (MBJ) eV |   E hull eV |
|------------------------------------------------|------------------------|------------------------|--------------------|--------------------|-------------|
| Method CGCNN                                   |                 0.063  |                 0.078  |              0.2   |              0.41  |       0.17  |
| SchNet                                         |                 0.045  |                 0.047  |              0.19  |              0.43  |       0.14  |
| MEGNet                                         |                 0.047  |                 0.058  |              0.145 |              0.34  |       0.084 |
| GATGNN                                         |                 0.047  |                 0.056  |              0.17  |              0.51  |       0.12  |
| ALIGNN                                         |                 0.0331 |                 0.037  |              0.142 |              0.31  |       0.076 |
| Matformer                                      |                 0.0325 |                 0.035  |              0.137 |              0.3   |       0.064 |
| PotNet                                         |                 0.0294 |                 0.032  |              0.127 |              0.27  |       0.055 |
| iComFormer                                     |                 0.0272 |                 0.0288 |              0.122 |              0.26  |       0.047 |
| Crystalformer                                  |                 0.0306 |                 0.032  |              0.128 |              0.3   |       0.046 |
| Crystalframer                                  |                 0.0263 |                 0.0279 |              0.117 |              0.242 |       0.047 |
| eComFormer                                     |                 0.0284 |                 0.0315 |              0.124 |              0.283 |       0.044 |
| -w/ SO(3)-equivariant Gram-Schmidt local frame |                 0.0285 |                 0.0296 |              0.115 |              0.271 |       0.043 |
| -w/ Gram-Schmidt-based SPFrame (ours)          |                 0.0268 |                 0.0281 |              0.109 |              0.259 |       0.043 |
| -w/ Quaternion-based SPFrame (ours)            |                 0.0261 |                 0.0276 |              0.107 |              0.239 |       0.042 |

MP. Table 2 presents the experimental results on the MP dataset. Similar to the results obtained on the JARVIS dataset, the eComFormer architecture, when combined with our proposed SPFrame, achieves the best performance on two out of four prediction tasks. The comparison with different frame methods further demonstrates the effectiveness of SPFrame. Furthermore, considering that the performance of the our method on bulk modulus and shear modulus prediction was not particularly strong on the MP dataset, we additionally conducted experiments on the JARVIS dataset for these two properties. The corresponding results are provided in Appendix A.9.

Table 2: Property prediction results on the MP dataset.

| Method                                         |   Formation energy eV/atom |   Bandgap eV |   Bulk modulus log(GPa) |   Shear modulus log(GPa) |
|------------------------------------------------|----------------------------|--------------|-------------------------|--------------------------|
| CGCNN                                          |                     0.031  |        0.292 |                  0.047  |                   0.077  |
| SchNet                                         |                     0.033  |        0.345 |                  0.066  |                   0.099  |
| MEGNet                                         |                     0.03   |        0.307 |                  0.06   |                   0.099  |
| GATGNN                                         |                     0.033  |        0.28  |                  0.045  |                   0.075  |
| ALIGNN                                         |                     0.022  |        0.218 |                  0.051  |                   0.078  |
| Matformer                                      |                     0.021  |        0.211 |                  0.043  |                   0.073  |
| PotNet                                         |                     0.0188 |        0.204 |                  0.04   |                   0.065  |
| iComFormer                                     |                     0.0183 |        0.193 |                  0.038  |                   0.0637 |
| Crystalformer                                  |                     0.0186 |        0.198 |                  0.0377 |                   0.0689 |
| Crystalframer                                  |                     0.0172 |        0.185 |                  0.0338 |                   0.0677 |
| eComFormer                                     |                     0.0182 |        0.202 |                  0.0417 |                   0.0729 |
| -w/ SO(3)-equivariant Gram-Schmidt local frame |                     0.0183 |        0.187 |                  0.0407 |                   0.0721 |
| -w/ Gram-Schmidt-based SPFrame (ours)          |                     0.0174 |        0.191 |                  0.037  |                   0.0678 |
| -w/ Quaternion-based SPFrame (ours)            |                     0.0171 |        0.181 |                  0.0371 |                   0.0672 |

Efficiency comparison. Table 3 compares the model efficiency of several frame methods and the skeleton method, eComFormer. We show the average training time per epoch, total number of parameters, and average testing time per material, all evaluated on the JARVIS-DFT formation energy dataset. The batch size is kept consistent across all experiments, and all experiments are conducted using a single NVIDIA GeForce RTX 3090 GPU. Due to the presence of trainable network

Figure 3: Visual analysis. After graph construction, atoms at symmetry-equivalent positions may exhibit distinct local structures. Local frame methods tend to transform these local structures into identical representations, thereby removing the relative differences and making the atoms indistinguishable to the model. In contrast, SPFrame preserves these structural distinctions, enabling t he model to effectively differentiate between atoms located at symmetry-equivalent positions.

<!-- image -->

components in the frame calculations, all frame-based methods are less efficient than the skeleton method, eComFormer. The SO(3)-equivariant Gram-Schmidt local frame and Gram-Schmidt-based SPFrame, which construct rotation matrices using the Gram-Schmidt orthogonalization method, require two distinct message passing and aggregation modules to generate and orthogonalize two different vectors (see Appendix A.7). Consequently, their efficiencies are similar but lower than that of SPFrame. In contrast, SPFrame only requires a single message passing and aggregation module to predict quaternions, which are subsequently used to construct the rotation matrix, thereby reducing computational cost.

Table 3: Efficiency analysis.

| Method                                         | Num. Params.   | Time/epoch   | Test time/Material   |
|------------------------------------------------|----------------|--------------|----------------------|
| eComFormer                                     | 4.9M           | 127.86 s     | 31.76 ms             |
| -w/ SO(3)-equivariant Gram-Schmidt local frame | 8.5M           | 235.42 s     | 43.43 ms             |
| -w/ Gram-Schmidt-based SPFrame                 | 8.5M           | 234.86 s     | 42.81 ms             |
| -w/ Quaternion-based SPFrame                   | 6.3M           | 143.75 s     | 37.07 ms             |

Visual analysis. To empirically evaluate the limitations of the equivariant local frame method and the effectiveness of SPFrame, we present a concrete visual example, as shown in Figure 3. Specifically, we visualize the crystal structure of PrBPt3 (JVASP-16632) and illustrate how different frames influence atoms located at symmetry-equivalent positions in the context of the formation energy prediction task on the JARVIS-DFT dataset. For the two symmetry-equivalent Pt atoms, the equivariant local frame transforms the edge features such that their local structures become indistinguishable. In contrast, SPFrame preserves the relative structural differences between these atoms after transformation, enabling the model to distinguish them. On this sample, the backbone network using the equivariant local frame yields a MAE of 0.0551, while the same backbone integrated with SPFrame achieves a lower MAE of 0.0503, demonstrating superior predictive accuracy.

## 6 Conclusion

This paper investigates the limitations of applying conventional local frame methods to crystal structures. Although these local frame methods enable GGNNs to satisfy the SO(3) invariance requirement, they may inadvertently disrupt the symmetry of the crystal, limiting the model's ability to distinguish atoms situated at symmetry-equivalent positions. To address this challenge, we propose the SPFrame for crystal property prediction. SPFrame constructs frames by incorporating both local

atomic structural information and global structural information. Such local-global associative frames ensure that GGNNs meet the SO(3) invariance requirement while preserving the crystal's symmetry, enhancing the model's ability to differentiate between distinct atomic structures. Experimental results on multiple datasets demonstrate the effectiveness of SPFrame. We hope that SPFrame provides a new perspective for machine learning and materials science, promoting the specific adaptation of machine learning techniques for materials science applications. Further discussion of SPFrame is provided in Appendix A.10 and Appendix A.11.

## Acknowledgments

This work was partially supported by the Research Grants Counil (RGC) of the Hong Kong (HK) SAR (Grant No. 15208725 and 15208222), the Young Scientists Fund of National Natural Science Foundation of China (NSFC) (Grant No. 62206235), and the Hong Kong Polytechnic University (Grant No. A0046682 and P0057774).

## References

- [1] C. Chen and S. P. Ong. A universal graph deep learning interatomic potential for the periodic table. Nature Computational Science , 2(11):718-728, 2022.
- [2] C. Chen, W. Ye, Y. Zuo, C. Zheng, and S. P. Ong. Graph networks as a universal machine learning framework for molecules and crystals. Chemistry of Materials , 31(9):3564-3572, 2019.
- [3] K. Choudhary and B. DeCost. Atomistic line graph neural network for improved materials property predictions. npj Computational Materials , 7(1):185, 2021.
- [4] K. Das, B. Samanta, P. Goyal, S.-C. Lee, S. Bhattacharjee, and N. Ganguly. Crysxpp: An explainable property predictor for crystalline materials. npj Computational Materials , 8(1):43, 2022.
- [5] K. Das, P. Goyal, S.-C. Lee, S. Bhattacharjee, and N. Ganguly. Crysmmnet: multimodal representation for crystal property prediction. In Uncertainty in Artificial Intelligence , pages 507-517. PMLR, 2023.
- [6] K. Das, B. Samanta, P. Goyal, S.-C. Lee, S. Bhattacharjee, and N. Ganguly. Crysgnn: Distilling pre-trained knowledge to enhance property prediction for crystalline materials. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pages 7323-7331, 2023.
- [7] W. Du, H. Zhang, Y. Du, Q. Meng, W. Chen, N. Zheng, B. Shao, and T.-Y. Liu. Se (3) equivariant graph neural networks with complete local frames. In International Conference on Machine Learning , pages 5583-5608. PMLR, 2022.
- [8] Y. Du, L. Wang, D. Feng, G. Wang, S. Ji, C. P. Gomes, Z.-M. Ma, et al. A new perspective on building efficient and expressive 3d equivariant graph neural networks. Advances in neural information processing systems , 36:66647-66674, 2023.
- [9] A. A. Duval, V. Schmidt, A. Hernández-Garcıa, S. Miret, F. D. Malliaros, Y. Bengio, and D. Rolnick. Faenet: Frame averaging equivariant gnn for materials modeling. In International Conference on Machine Learning , pages 9013-9033. PMLR, 2023.
- [10] N. Dym, H. Lawrence, and J. W. Siegel. Equivariant frames and the impossibility of continuous canonicalization. In Forty-first International Conference on Machine Learning , 2024.
- [11] D. Eberly. Quaternion algebra and calculus. Magic Software Inc , 26:1-8, 2002.
- [12] M. Geiger and T. Smidt. e3nn: Euclidean neural networks. arXiv preprint arXiv:2207.09453 , 2022.
- [13] A. R. Geist, J. Frey, M. Zhobro, A. Levina, and G. Martius. Learning with 3d rotations, a hitchhiker's guide to so (3). In International Conference on Machine Learning , pages 15331-15350. PMLR, 2024.
- [14] T. Hahn, U. Shmueli, and J. W. Arthur. International tables for crystallography , volume 1. Reidel Dordrecht, 1983.
- [15] J. Han, J. Cen, L. Wu, Z. Li, X. Kong, R. Jiao, Z. Yu, T. Xu, F. Wu, Z. Wang, et al. A survey of geometric graph neural networks: Data structures, models and applications. Frontiers of Computer Science , 19(11): 1911375, 2025.

- [16] H. Hong, W. Lin, and K. Tan. Accelerating 3d molecule generation via jointly geometric optimal transport. In The Thirteenth International Conference on Learning Representations , 2025.
- [17] T. Hsu, T. A. Pham, N. Keilbart, S. Weitzner, J. Chapman, P. Xiao, S. R. Qiu, X. Chen, and B. C. Wood. Efficient and interpretable graph network representation for angle-dependent properties applied to optical spectroscopy. npj Computational Materials , 8(1):151, 2022.
- [18] H. Hua, W. Lin, and J. Yang. Fast crystal tensor property prediction: A general o (3)-equivariant framework based on polar decomposition. arXiv preprint arXiv:2410.02372 , 2024.
- [19] Y. Ito, T. Taniai, R. Igarashi, Y . Ushiku, and K. Ono. Rethinking the role of frames for se(3)-invariant crystal structure modeling. In The Thirteenth International Conference on Learning Representations , 2025.
- [20] A. Janner, T. Janssen, and P. De Wolff. Wyckoff positions used for the classification of bravais classes of modulated crystals. Acta Crystallographica Section A: Foundations of Crystallography , 39(5):667-670, 1983.
- [21] R. Jiao, W. Huang, Y. Liu, D. Zhao, and Y. Liu. Space group constrained crystal generation. In The Twelfth International Conference on Learning Representations , 2024.
- [22] O. Kaba and S. Ravanbakhsh. Equivariant networks for crystal structures. Advances in Neural Information Processing Systems , 35:4150-4164, 2022.
- [23] S.-O. Kaba, A. K. Mondal, Y. Zhang, Y. Bengio, and S. Ravanbakhsh. Equivariance with learned canonicalization functions. In International Conference on Machine Learning , pages 15546-15566. PMLR, 2023.
- [24] N. Kazeev, W. Nong, I. Romanov, R. Zhu, A. E. Ustyuzhanin, S. Yamazaki, and K. Hippalgaonkar. Wyckoff transformer: Generation of symmetric crystals. In Forty-second International Conference on Machine Learning , 2025.
- [25] F. E. Kelvinius, O. B. Andersson, A. S. Parackal, D. Qian, R. Armiento, and F. Lindsten. Wyckoffdiff-a generative diffusion model for crystal symmetry. In Forty-second International Conference on Machine Learning , 2025.
- [26] D. Kingma. Adam: a method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations , 2015.
- [27] N. Lee, H. Noh, S. Kim, D. Hyun, G. S. Na, and C. Park. Density of states prediction of crystalline materials via prompt-guided multi-modal transformer. Advances in Neural Information Processing Systems , 36, 2024.
- [28] D. Levy, S. S. Panigrahi, S.-O. Kaba, Q. Zhu, K. L. K. Lee, M. Galkin, S. Miret, and S. Ravanbakhsh. Symmcd: Symmetry-preserving crystal generation with diffusion models. In The Thirteenth International Conference on Learning Representations , 2025.
- [29] Z. Li, X. Sun, W. Lin, and J. Cao. Unveiling molecular secrets: An llm-augmented linear model for explainable and calibratable molecular property prediction. arXiv preprint arXiv:2410.08829 , 2024.
- [30] W. Lin, H. Lan, and B. Li. Generative causal explanations for graph neural networks. In International conference on machine learning , pages 6666-6679. PMLR, 2021.
- [31] W. Lin, H. Lan, H. Wang, and B. Li. Orphicx: A causality-inspired latent variable model for interpreting graph neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 13729-13738, 2022.
- [32] Y. Lin, K. Yan, Y. Luo, Y. Liu, X. Qian, and S. Ji. Efficient approximations of complete interatomic potentials for crystal property prediction. In International Conference on Machine Learning , pages 21260-21287. PMLR, 2023.
- [33] Y. Lin, J. Helwig, S. Gui, and S. Ji. Equivariance via minimal frame averaging for more symmetries and efficiency. In Forty-first International Conference on Machine Learning , 2024.
- [34] P. Lippmann, G. Gerhartz, R. Remme, and F. A. Hamprecht. Beyond canonicalization: How tensorial messages improve equivariant message passing. In The Thirteenth International Conference on Learning Representations , 2025.
- [35] S. Liu, Y. Li, Z. Li, Z. Zheng, C. Duan, Z.-M. Ma, O. Yaghi, A. Anandkumar, C. Borgs, J. Chayes, et al. Symmetry-informed geometric representation for molecules, proteins, and crystalline materials. Advances in neural information processing systems , 36:66084-66101, 2023.

- [36] S.-Y. Louis, Y. Zhao, A. Nasiri, X. Wang, Y. Song, F. Liu, and J. Hu. Graph convolutional neural networks with global attention for improved materials property prediction. Physical Chemistry Chemical Physics , 22 (32):18141-18148, 2020.
- [37] G. Ma, Y. Wang, D. Lim, S. Jegelka, and Y. Wang. A canonicalization perspective on invariant and equivariant learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [38] Z. Mao, W. Li, and J. Tan. Dielectric tensor prediction for inorganic materials using latent information from preferred potential. npj Computational Materials , 10(1):265, 2024.
- [39] S. Mukherjee, M. Ghosh, and P. Basuchowdhuri. Crysatom: Distributed representation of atoms for crystal property prediction. In The Third Learning on Graphs Conference , 2024.
- [40] S. Pozdnyakov and M. Ceriotti. Smooth, exact rotational symmetrization for deep learning on point clouds. Advances in Neural Information Processing Systems , 36:79469-79501, 2023.
- [41] O. Puny, M. Atzmon, E. J. Smith, I. Misra, A. Grover, H. Ben-Hamu, and Y. Lipman. Frame averaging for invariant and equivariant network design. In International Conference on Learning Representations , 2022.
- [42] L. Rodman. Topics in quaternion linear algebra . Princeton University Press, 2014.
- [43] R. Sajnani, A. Poulenard, J. Jain, R. Dua, L. J. Guibas, and S. Sridhar. Condor: Self-supervised canonicalization of 3d pose for partial shapes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 16969-16979, 2022.
- [44] V. G. Satorras, E. Hoogeboom, and M. Welling. E (n) equivariant graph neural networks. In International conference on machine learning , pages 9323-9332. PMLR, 2021.
- [45] K. T. Schütt, H. E. Sauceda, P.-J. Kindermans, A. Tkatchenko, and K.-R. Müller. Schnet-a deep learning architecture for molecules and materials. The Journal of Chemical Physics , 148(24), 2018.
- [46] K. Shoemake. Animating rotation with quaternion curves. SIGGRAPH Comput. Graph. , 19(3):245-254, 1985. ISSN 0097-8930. doi: 10.1145/325165.325242. URL https://doi.org/10.1145/325165. 325242 .
- [47] L. N. Smith and N. Topin. Super-convergence: Very fast training of neural networks using large learning rates. In Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications , volume 11006, pages 369-386. SPIE, 2019.
- [48] Z. Song, Z. Meng, and I. King. A diffusion-based pre-training framework for crystal property prediction. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 8993-9001, 2024.
- [49] T. Taniai, R. Igarashi, Y. Suzuki, N. Chiba, K. Saito, Y. Ushiku, and K. Ono. Crystalformer: Infinitely connected attention for periodic structure encoding. In The Twelfth International Conference on Learning Representations , 2024.
- [50] X. Wang and M. Zhang. Graph neural network with local frame for molecular potential energy surface. In Learning on Graphs Conference , pages 19-1. PMLR, 2022.
- [51] Y. Wang, S. Kong, J. M. Gregoire, and C. P. Gomes. Conformal crystal graph transformer with robust encoding of periodic invariance. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 283-291, 2024.
- [52] Z. Wang, H. Hua, W. Lin, M. Yang, and K. C. Tan. Crystalline material discovery in the era of artificial intelligence. arXiv preprint arXiv:2408.08044 , 2024.
- [53] Z. Wang, Z. Lin, W. Lin, M. Yang, M. Zeng, and K. C. Tan. Explainable molecular property prediction: Aligning chemical concepts with predictions via language models. arXiv preprint arXiv:2405.16041 , 2024.
- [54] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, et al. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations , pages 38-45, 2020.
- [55] T. Xie and J. C. Grossman. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical review letters , 120(14):145301, 2018.
- [56] K. Yan, Y. Liu, Y. Lin, and S. Ji. Periodic graph transformers for crystal material property prediction. Advances in Neural Information Processing Systems , 35:15066-15080, 2022.

- [57] K. Yan, C. Fu, X. Qian, X. Qian, and S. Ji. Complete and efficient graph transformers for crystal material property prediction. In The Twelfth International Conference on Learning Representations , 2024.
- [58] K. Yan, A. Saxton, X. Qian, X. Qian, and S. Ji. A space group symmetry informed network for o (3) equivariant crystal tensor prediction. In Forty-first International Conference on Machine Learning , 2024.
- [59] S. Yang, K. Cho, A. Merchant, P. Abbeel, D. Schuurmans, I. Mordatch, and E. D. Cubuk. Scalable diffusion for materials generation. In The Twelfth International Conference on Learning Representations , 2025.
- [60] S. Zhang, Y. Tay, L. Yao, and Q. Liu. Quaternion knowledge graph embeddings. Advances in neural information processing systems , 32, 2019.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: As discussed in Section 3 and Section 5, the main claims made in the abstract and introduction accurately reflect the paper's contributions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations of this work are discussed in Appendix A.10.

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

Answer: [NA]

Justification: We do not include any theoretical results in the paper.

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

Justification: Detailed experimental settings, datasets, and computations needed are shared in Section 5 and Appendix A.8.

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

## Answer: [No]

Justification: The code is not currently in a state ready for distribution. It will be released after we have some time to clean it up.

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

Justification: Detailed experimental settings are provided in Section 5 and Appendix A.8.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The dataset used in the experiments are large and the results are relatively stable. Training and evaluating models for multiple times is costly.

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

Justification: Detailed compute resources needed are provided in Section 5.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This paper definitely follows the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We provide broader impacts in Appendix A.11.

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

Answer: [Yes]

Justification: All datasets and models used in this paper have been properly cited.

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

Justification: This paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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

## A Appendix / supplemental materials

## A.1 Theoretical Justification of the Superiority of SPFrame over Local Frame

We provide a mutual information-based proof to explain why the SPFrame approach yields more informative node/atom representations than the local frame method. Let a crystal structure be represented as a graph with N atoms, and denote the node/atom feature corresponding to atom i as f i . The complete sets of node or atom features under two different framing schemes are defined as:

<!-- formula-not-decoded -->

where A represents node/atom features obtained using the local frame method, and B corresponds to those derived using the SPFrame method.

We assume that the atoms with indices p, q ( 1 &lt; p, q &lt; N ) are symmetry-equivalent. The use of the local frame results in identical local environments for atoms p and q (as illustrated in Figure 2). Consequently, after message passing via Equation 3, we have

<!-- formula-not-decoded -->

which reduces the number of distinguishable atom representations. Denoting the cardinality as | · | , we obtain

<!-- formula-not-decoded -->

In contrast, the SPFrame approach preserves the differences in the local environments of atoms p and q . After message passing via Equation 5, the node/atom representations satisfy

̸

<!-- formula-not-decoded -->

For scalar crystal property prediction, the final node features are first aggregated to get a global graph-level representation, which is then passed through a regression head. We denote the prediction target as Y ∈ Y . The neural network induces the following mapping:

<!-- formula-not-decoded -->

Since | B | &gt; | A | , there exists a surjective mapping g 1 : B ↦→ A , such that h local ◦ g 1 = h sp . However, an injective mapping g 2 : A ↦→ B does not exist in general due to loss of distinguishability in | A | . Consequently, the information flow can be described via the following Markov chain:

<!-- formula-not-decoded -->

Applying the data processing inequality to this chain yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

i.e., the chain Y → A → B is also valid. However, the absence of a mapping g 2 : A ↦→ B implies that this reverse chain cannot be constructed. Therefore, the inequality is strict:

<!-- formula-not-decoded -->

This result implies that the node/atom representations obtained via SPFrame retain strictly higher mutual information with the target variable Y than those obtained via the local frame method. In other words, SPFrame-based features preserve more task-relevant information.

## A.2 Proof of SE(3) invariance

This work also ensures SE(3) invariance [15, 16]. Let atoms i and j be two neighboring atoms, with positions denoted by x i and x j , respectively. In the message passing formulation of Equation 5, the edge scalar feature e ij can be expressed as || x i -x j || , and the directional vector ̂ e ij can be expressed as x i -x j . After applying a global rotation Q and translation t , we have with equality if and only if

<!-- formula-not-decoded -->

Applying a rotation Q and translation t does not change the expression in Equation 5. Therefore, Equation 5 is unaffected by rotation and translation, indicating that it is SE(3)-invariant.

## A.3 Implementation of Global Frame in SPFrame

As outlined in Section 2.1, the entire crystal structure can be represented by its unit cell. When the entire crystal structure undergoes a rotation, the unit cell also changes accordingly. Therefore, the global frame can be computed from the lattice matrix L ∈ R 3 × 3 of the unit cell [21, 52]. Below, we introduce three commonly used methods for computing the global frame. Each of these methods can be applied to the global frame construction within the SPFrame.

QR Decomposition [33]. Given that the lattice matrix L ∈ R 3 × 3 is invertible, it can be uniquely decomposed via QR decomposition as L = QR , where the diagonal elements of R are constrained to be positive. In this decomposition, Q ∈ R 3 × 3 is an orthogonal matrix, while R ∈ R 3 × 3 is an upper triangular matrix. By applying QR decomposition to L under this positivity constraint, we obtain the orthogonal matrix Q , which is naturally equivariant under O(3) transformations. To further restrict this equivariance to the SO(3) group, we flip the sign of the first column vector of Q if necessary to enforce det( Q ) = 1 . The resulting matrix, now SO(3)-equivariant, serves as a choice for the global frame F global in our proposed SPFrame.

Polar Decomposition [21, 18]. As an invertible matrix, the lattice matrix L ∈ R 3 × 3 can be uniquely decomposed into L = QH , where Q ∈ R 3 × 3 is an orthogonal matrix, H ∈ R 3 × 3 is a Hermitian positive semi-definite matrix. By applying polar decomposition to L , we obtain the orthogonal matrix Q , which is naturally equivariant under O(3) transformations. To ensure that Q is equivariant only under SO(3) transformations, we adjust the sign of its first column vector if needed to enforce det( Q ) = 1 . The resulting matrix, now SO(3)-equivariant, can be reliably utilized as the global frame F global in our SPFrame.

Principal Component Analysis (PCA) [9, 37]. For the lattice matrix L ∈ R 3 × 3 , we first compute the centroid of the lattice vectors as t = 1 n L1 ∈ R 3 , followed by the construction of the covariance matrix

Σ = ( L -1t ⊤ ) ⊤ ( L -1t ⊤ ) . We then perform eigendecomposition on Σ to obtain its eigenvectors u 1 , u 2 , u 3 . Assuming the eigenvalues satisfy the condition λ 1 &gt; λ 2 &gt; λ 3 , the corresponding eigenvectors can be assembled into a 3 × 3 orthogonal matrix U = [ u 1 , u 2 , u 3 ] , which defines one of eight possible O(3)-equivariant frames. To obtain a unique SO(3)-equivariant global frame, we apply Algorithm 2 to resolve the sign ambiguity [37] in U and enforce det( U ) = 1 . The resulting matrix U ∗ serves as the global frame F global in SPFrame.

## A.4 Implementation of Invariant Local Frame in SPFrame

Quaternion Generation via SO(3)-Invariant Message Passing. In this work, we adopt SO(3)invariant message passing proposed by Yan et al. [56] to generate quaternions for constructing local frames. This process leverages the atom features f i , neighboring atom features f j , and invariant edge features e ij to perform message passing from neighbor atom j to the central atom i . The messages

̸

```
Algorithm 2 Unique SO(3)-equivariant global frame based on eigenvectors Require: The orthogonal eigenvector matrices U = [ u 1 , u 2 , u 3 ] Ensure: The unique SO(3)-equivariant global frame U ∗ 1: for i =1,2 do 2: Let j be the smallest index such that u j = 0 3: if u j > 0 then 4: u ∗ i ← u i 5: else 6: u ∗ i ←-u i 7: end if 8: end for 9: if det([ u ∗ 1 , u ∗ 2 , u 3 ]) > 0 then 10: u ∗ 3 ← u 3 11: else 12: u ∗ 3 ←-u 3 13: end if 14: U ∗ = [ u ∗ 1 , u ∗ 2 , u ∗ 3 ]
```

are aggregated across all neighbors, and the result is combined with the central atom's features f i to produce the quaternion q i .

Specifically, we first calculate three components: the query vector q ij = LN Q ( f i ) , the key vector k ij = (LN K ( f i ) | LN K ( f j )) , and the value vector v ij = (LN V ( f i ) | LN V ( f j ) | LN E ( f e ij )) , where LN Q ( · ) , LN K ( · ) , LN V ( · ) , LN E ( · ) denote the linear layers, and | denote the concatenation. Then, the message form atom j to atom i is computed as:

<!-- formula-not-decoded -->

where ξ K , ξ V represent mappings applied to the key and value vectors, respectively, and the operators ◦ denote the Hadamard product, BN refers to the batch normalization layer, and √ d q ij indicates the dimensionality of q ij . Finally, the quaternion q i is generated as:

<!-- formula-not-decoded -->

where ξ msg ( · ) denotes the softplus activation function amd LN msg ( · ) denote the linear layer.

Quaternion to Rotation Matrix Conversion. Quaternions, while originating in pure mathematics, are extensively used for representing and computing 3D rotations [60, 42, 11, 46]. A unit quaternion, represented by four real-valued components, encodes a 3D rotation by specifying a rotation axis and an associated rotation angle [11]. Therefore, we normalize the network's 4-dimensional output to obtain a unit quaternion, which is then converted into a rotation matrix [42].

## A.5 Backbone with SPFrame

The eComFormer has demonstrated strong performance across a wide range of crystal property prediction tasks [57]. This model integrates a node-wise transformer layer and a node-wise equivariant updating layer to capture complex geometric relationships within the crystal structure. Given that the node-wise equivariant updating layer operates on equivariant edge features, we incorporate SPFrame into each of these layers. Furthermore, we append a node-wise equivariant updating layer following each node-wise transformer layer. The detailed network architecture is provided below.

Node-wise transformer layer in eComFormer. The node-wise transformer layer in eComFormer updates node invariant features f i through a message-passing mechanism. This layer integrates three types of information: the node features f i , neighboring node features f j , and invariant edge embeddings f e ij . The update process follows a transformer-style architecture. Fristly, the message from node j to node i is encoded using three projected features query q ij = LN Q ( f i ) , key k ij = (LN K ( f i ) | LN K ( f j )) , and value feature v ij = (LN V ( f i ) | LN V ( f j ) | LN E ( f e ij )) , where LN Q ( · ) ,

Figure 4: The detailed architectures of eComFormer with SPFrame.

<!-- image -->

LN K ( · ) , LN V ( · ) , LN E ( · ) denote the linear transformations, and | denote the concatenation. Then, the attention mechanism computes:

<!-- formula-not-decoded -->

where ξ K , ξ V are nonlinear transformations, and the operators ◦ denote the Hadamard product. BN( · ) refers to the batch normalization layer, and √ d q ij indicates the dimensionality of q ij . Then, node feature f i is updated as follows,

<!-- formula-not-decoded -->

where ξ msg ( · ) denoting the softplus activation function.

Node-wise equivariant updating layer using SPFrame. The node-wise equivariant updating layer in eComFormer employs two tensor product (TP) layers [12] to effectively capture geometric features. The equivalent edge feature e ji is embedded using spherical harmonics, with the representations given by Y 0 ( ̂ e ji ) = c 0 , Y 1 ( ̂ e ji ) = c 1 ∗ ̂ e ji || ̂ e ji || 2 ∈ R 3 and Y 2 ( ̂ e ji ) ∈ R 5 . These harmonics form the input features to the TP layers.

Therefore, we apply the SPFrame to the equivariant edge features before embedding them into spherical harmonics. Specifically, the first TP layer is defined as:

<!-- formula-not-decoded -->

where f l ′ i is the linearly transformed atom feature derived from f l i , |N i | denotes the number of neighboring atoms of atom i , and TP λ denotes the TP layer corresponding to rotation order λ .

The second TP layer further aggregates the directional features across multiple orders as follows:

<!-- formula-not-decoded -->

Finally, the outputs from the two TP layers are combined through both linear and nonlinear transformations to produce the updated atom feature f l i,updated :

<!-- formula-not-decoded -->

where σ ( · ) denotes a nonlinear transformation consisting of two softplus layers with an intervening linear layer, while BN( · ) and LN( · ) represent batch normalization and a linear layer, respectively.

Overall architecture. The overall architecture is illustrated in Fig. 4. The key components of the network are summarized as follows. The architecture begins with embedding layers for node and edge features, followed by a series of stacked message passing modules. Each module consists of a node-wise transformer layer, a node-wise equivariant update layer, and a SPFrame construction block. The network concludes with a global average pooling layer and a multi-layer perceptron (MLP) for property prediction. Notably, drawing inspiration from recent findings [19], which demonstrate that dynamically constructing frames at intermediate layers significantly enhances both model expressiveness and prediction accuracy, we integrate SPFrame construction modules at multiple stages throughout the network. Furthermore, since eComFormer produces SO(3)-invariant outputs, the global frame F global in Equation 5 can be set as the identity matrix. This simplification enables a more efficient implementation of our approach.

## A.6 More Related Works

Other approaches for improving prediction accuracy. In crystal property prediction tasks, in addition to frame-based methods (which can also be regarded as a form of representation learning), pretraining [4, 6, 48] and representation learning [17, 5, 35, 39] are two other important approaches for improving prediction accuracy.

Pretraining methods primarily focus on improving the backbone network architecture. CrysXPP [4] designs an autoencoder for self-supervised pretraining, capturing key structural and chemical features from large amounts of unlabeled crystal graph data to reduce prediction errors. CrysGNN [6] introduces a specialized pretrained GNN framework that combines feature reconstruction, connectivity reconstruction, and contrastive learning across different crystal systems. CrysDiff [48] employs a diffusion-based pretraining approach, where the pretraining phase reconstructs crystal structures via a diffusion process to learn the underlying edge distribution, and the fine-tuning phase generates target property values guided by structural data.

In contrast, representation learning focuses on constructing more expressive representations of crystal structures. Beyond commonly used bond angle information for encoding directionality, ALIGNNd [17] incorporates dihedral angles, achieving a memory-efficient graph representation that captures the full atomic geometry. CrysMMNet [5] integrates textual material descriptions into the crystal graph to encode global structural information, leading to richer and more robust representations. Geom3D [35] systematically benchmarks various geometric encoding strategies, including spherical harmonics, frame-based bases, and angle-based features. CrysAtom [39] learns distributed atomic representations in an unsupervised manner from unlabeled crystal data, significantly improving downstream property prediction.

Incorporating crystal symmetry into method design. Crystal symmetry is a fundamental property of crystalline materials. Most existing methods for property prediction have only limited utilization of this symmetry, while many generative approaches explicitly leverage it to improve model performance. The core idea of these generative approaches is to simplify the data to be generated by exploiting crystal symmetry. Below, we introduce several representative methods.

DiffCSP++ [21]: Owing to crystal symmetry, the lattice matrix elements in different space groups are subject to specific constraints. DiffCSP++ generates only the unconstrained lattice elements, simplifying the generation process. It also reconstructs atomic fractional coordinates and element types by deriving symmetry-equivalent atoms from a single representative atom using symmetry operations. This ensures that the generated crystals strictly satisfy space group constraints.

SymmCD [28]: SymmCD generates only the asymmetric unit rather than the full lattice matrix, outputting its unit parameters. Unlike DiffCSP++, which generates complete lattice matrices using predefined templates, SymmCD demonstrates experimentally that DiffCSP++ may limit structural diversity and novelty.

Wyckoff Transformer [24]: Wyckoff Transformer is an autoregressive generative method distinct from diffusion-based approaches [21, 28, 25]. For symmetry-related atoms, it generates only discrete attributes such as space group, element type, site symmetry, and enumeration. The complete crystal structure is reconstructed by combining these discrete attributes with energy relaxation.

WyckoffDiff [25]: Given a space group and Wyckoff positions, WyckoffDiff predicts the probability distribution of atom types occupying each position. This approach resembles UniMat [59], which predicts elemental probabilities from the periodic table, but WyckoffDiff explicitly embeds symmetry constraints into the generative process.

Our method, in contrast, is designed for scalar property prediction rather than generative modeling. It primarily addresses the limitation of local frames, where atoms at symmetry-equivalent positions share identical local environments, leading to indistinguishable node features and information loss. To mitigate this, we design the frame by combining an invariant local frame with an equivariant global frame shared across the atomic system. This design preserves crystal symmetry and ensures that atoms at symmetry-equivalent positions maintain symmetric yet distinguishable local environments, allowing their node features to remain discriminative.

## A.7 Frame Baseline

SO(3) equivariant frame constructed based on Gram-Schmidt orthogonalization [50, 34] Inspired by previous work [44, 50], we construct an equivariant frame by predicting two equivariant vectors using the Schmidt orthogonalization method, and use this frame as a baseline for comparison with the proposed Symmetry-preserving frame in this paper. Specifically, the two equivariant vectors v i, 1 , v i, 2 ∈ R 3 are predicted as follows:

<!-- formula-not-decoded -->

where f i denotes the feature vector of atom i , e ij represents SO(3)-invariant edge features, and ̂ e ij corresponds to SO(3)-equivariant edge features. ϕ k ( · ) is the message function from Yan et al. [56]. The rotation matrix is then constructed using Gram-Schmidt orthogonalization as follows:

<!-- formula-not-decoded -->

SPFrame constructed based on Gram-Schmidt orthogonalization During the construction of the SPFrame, we need to establish an invariant local frame and an equivariant global frame. Similar to Eq. 24, we predict two invariant vectors using only invariant edge features:

<!-- formula-not-decoded -->

The rotation matrix F INV ,i is then constructed using Eq. 25. The equivariant global frame F global is still derived using the method described in Appendix A.3. Ultimately, this yields the Symmetrypreserving frame F INV ,i F global based on Gram-Schmidt orthogonalization.

Incorporating angular information. When computing the local frame using the Gram-Schmidt orthogonalization method as defined in Equation 25, the intrinsic symmetry of the crystal can lead to cases during training where the vectors v i, 1 and v i, 2 become collinear. This collinearity prevents the Gram-Schmidt orthogonalization method from producing a valid local frame.

To overcome this limitation, we incorporate angular information into Equation 25. Specifically, for each atom within the unit cell, we first compute the frame (such as PCA frame in Appendix A.3) of the equivariant edge vectors and compute the frame of the vectors in the lattice matrix. These vectors are then transformed into invariant representations. Next, we calculate the angles [57] between the transformed edge vectors and the transformed lattice vectors. These angle-based features are then integrated into the invariant edge features [57], enhancing the robustness of the frame construction.

## A.8 Training Settings

In this subsection, we provide the detailed hyperparameter settings for backbone integrated with SPFrame across different tasks. For the network architecture, the backbone follows the parameter settings outlined in the original paper [57], such as those for the graph construction and the embedding layers. The training hyperparameters are as follows.

JARVIS: formation energy. For the eComFormer backbone, the network is trained using L1 loss with the Adam optimizer [26] for 500 epochs, employing the Onecycle scheduler [47] with a pct\_start of 0.3 and an initial learning rate of 0.0005. The network consists of 2 message passing layers and 3 SPFrame modules. Each message passing layer is equipped with one SPFrame module, and an additional SPFrame module is placed before the first message passing layer. The intermediate features, such as node features and invariant edge features, are set to 256 dimensions, and the batch size is set to 64. For the iComFormer backbone, the network is trained using L1 loss with the Adam optimizer for 700 epochs, employing the Onecycle scheduler with a pct\_start of 0.3 and an initial learning rate of 0.001. The network consists of a total of 4 message passing layers, each equipped with an SPFrame module except for the final layer. The dimensionality of all features is set to 256, and the batch size is 64.

JARVIS: band gap (OPT). For the eComFormer backbone, the network is trained using the L1 loss function and the Adam optimizer for 500 epochs. A cosine with warmup scheduler is employed [54], with an initial learning rate of 0.001 and a warmup phase corresponding to 5% of the total training steps. The network consists of a total of 2 message passing layers, each equipped with an SPFrame module. The feature dimension is set to 128, and the batch size is 64. For the iComFormer backbone, the network is trained using the L1 loss function and the Adam optimizer for 500 epochs. A cosine with warmup scheduler is employed, with an initial learning rate of 0.001 and a warmup phase corresponding to 5% of the total training steps. The network consists of a total of 4 message passing layers, each equipped with an SPFrame module except for the final layer. The feature dimension is set to 128, and the batch size is 64.

JARVIS: band gap (MBJ). For the eComFormer backbone, the network is trained using the L1 loss function and the Adam optimizer for 500 epochs. A cosine with warmup scheduler is employed, with an initial learning rate of 0.003 and a warmup phase corresponding to 5% of the total training steps. The network consists of a total of 2 message passing layers, each equipped with an SPFrame module. The feature dimension is set to 128, and the batch size is 64. For the iComFormer backbone, the network is trained using L1 loss with the Adam optimizer for 1000 epochs, employing the Onecycle scheduler with a pct\_start of 0.3 and an initial learning rate of 0.001. The network consists of a total of 4 message passing layers, each equipped with an SPFrame module except for the final layer. The feature dimension is set to 256, and the batch size is 64.

JARVIS: total energy. For the eComFormer backbone, the network is trained using L1 loss with the Adam optimizer for 1000 epochs, employing the Onecycle scheduler with a pct\_start of 0.3 and an initial learning rate of 0.001. The network consists of a total of 2 message passing layers, each equipped with an SPFrame module. The feature dimension is set to 128, and the batch size is 32. For the iComFormer backbone, the network is trained using L1 loss with the Adam optimizer for 1000 epochs, employing the Onecycle scheduler with a pct\_start of 0.3 and an initial learning rate of 0.001. The network consists of a total of 4 message passing layers, each equipped with an SPFrame module except for the final layer. The feature dimension is set to 256, and the batch size is 64.

JARVIS: Ehull. For the eComFormer backbone, the network is trained using L1 loss with the Adam optimizer for 500 epochs, employing the Onecycle scheduler with a pct\_start of 0.3 and an initial learning rate of 0.001. The network consists of a total of 2 message passing layers, each equipped with an SPFrame module. The feature dimension is set to 128, and the batch size is 64. For the iComFormer backbone, the network is trained using L1 loss with the Adam optimizer for 1000 epochs, employing the Onecycle scheduler with a pct\_start of 0.3 and an initial learning rate of 0.001. The network consists of a total of 4 message passing layers, each equipped with an SPFrame module except for the final layer. The feature dimension is set to 128, and the batch size is 64.

JARVIS: bulk modulus. The network is trained using the L1 loss function and the Adam optimizer for 500 epochs. A cosine with warmup scheduler is employed, with an initial learning rate of 0.001 and a warmup phase corresponding to 5% of the total training steps. The network consists of a total

of 2 message passing layers, each equipped with an SPFrame module. The feature dimension is set to 128, and the batch size is 64.

JARVIS: shear modulus. The network is trained using the L1 loss function and the Adam optimizer for 500 epochs. A cosine with warmup scheduler is employed, with an initial learning rate of 0.001 and a warmup phase corresponding to 5% of the total training steps. The network consists of a total of 3 message passing layers, each equipped with an SPFrame module. The feature dimension is set to 128, and the batch size is 64.

MP: formation energy. The network is trained using L1 loss with the Adam optimizer for 500 epochs, employing the Onecycle scheduler with a pct\_start of 0.3 and an initial learning rate of 0.001. The network consists of a total of 2 message passing layers, each equipped with an SPFrame module. The feature dimension is set to 196, and the batch size is 32.

MP: band gap. The network is trained using L1 loss with the Adam optimizer for 500 epochs, employing the Onecycle scheduler with a pct\_start of 0.3 and an initial learning rate of 0.001. The network consists of a total of 3 message passing layers, each equipped with an SPFrame module. The feature dimension is set to 128, and the batch size is 32.

MP: bulk moduli. The network is trained using L1 loss with the Adam optimizer for 500 epochs, employing the Onecycle scheduler with a pct\_start of 0.3 and an initial learning rate of 0.001. The network consists of a total of 4 message passing layers, each equipped with an SPFrame module. The feature dimension is set to 512, and the batch size is 64.

MP: shear moduli. The network is trained using MSE loss with the Adam optimizer for 500 epochs, employing the Onecycle scheduler with a pct\_start of 0.3 and an initial learning rate of 0.001. The network consists of a total of 4 message passing layers, each equipped with an SPFrame module. The feature dimension is set to 128, and the batch size is 64.

## A.9 Additional Experimental Results

To further demonstrate the generality of SPFrame, we conducted additional experiments by integrating SPFrame with iComFormer [57]. Specifically, iComFormer consists of the node-wise transformer layer and the edge-wise transformer layer. In our implementation, an SPFrame construction block was added after each node-wise transformer layer. The resulting frame was applied to the interatomic edge vectors, and then the angles between these edge vectors and the lattice matrix were computed before being fed into the edge-wise transformer layer for edge feature updates. Table 4 presents the experimental results of combining iComFormer with SPFrame on the JARVIS dataset. Because iComFormer serves as a more powerful backbone, the combined model achieves superior performance compared to the model using eComFormer as the backbone across most prediction tasks.

Table 4: Additional property prediction results on the JARVIS dataset.

| Method                                         |   Form. energy eV/atom |   Total energy eV/atom |   Bandgap (OPT) eV |   Bandgap (MBJ) eV |   E hull eV |
|------------------------------------------------|------------------------|------------------------|--------------------|--------------------|-------------|
| Crystalframer                                  |                 0.0263 |                 0.0279 |              0.117 |              0.242 |       0.047 |
| eComFormer                                     |                 0.0284 |                 0.0315 |              0.124 |              0.283 |       0.044 |
| -w/ SO(3)-equivariant Gram-Schmidt local frame |                 0.0285 |                 0.0296 |              0.115 |              0.271 |       0.043 |
| -w/ Quaternion-based SPFrame (ours)            |                 0.0261 |                 0.0276 |              0.107 |              0.239 |       0.042 |
| iComFormer                                     |                 0.0272 |                 0.0288 |              0.122 |              0.26  |       0.047 |
| -w/ SO(3)-equivariant Gram-Schmidt local frame |                 0.0275 |                 0.0287 |              0.112 |              0.255 |       0.045 |
| -w/ Quaternion-based SPFrame (ours)            |                 0.025  |                 0.0259 |              0.106 |              0.251 |       0.042 |

Table 5 presents the experimental results for bulk modulus and shear modulus prediction on the JARVIS dataset. CrystalFramer achieves the best performance on bulk modulus, consistent with the results observed on the MP dataset, while our method slightly outperforms CrystalFramer on shear modulus.

## A.10 Limitations

This work investigates the issue that conventional local frame methods, when applied to crystal structures, may inadvertently disrupt the intrinsic symmetry of the crystal. To address this problem, we

Table 5: Additional bulk and shear modulus prediction results on the JARVIS dataset.

| Method                                         |   Bulk Modulus (Kv) |   Shear Modulus (Gv) |
|------------------------------------------------|---------------------|----------------------|
| Matformer                                      |              11.21  |               10.76  |
| CrysGNN [6]                                    |              10.99  |                9.8   |
| CrysDiff [48]                                  |               9.875 |                9.193 |
| Crystalframer                                  |               8.876 |                8.999 |
| eComFormer                                     |               9.777 |                9.435 |
| -w/ SO(3)-equivariant Gram-Schmidt local frame |               9.855 |                9.689 |
| -w/ Quaternion-based SPFrame (ours)            |               9.357 |                8.963 |

propose SPFrame. Comparative experiments against conventional local frame approaches demonstrate the effectiveness of SPFrame. However, while empirical results validate the benefits of SPFrame, this study does not provide a quantitative theoretical analysis of how symmetry breaking impacts the accuracy of crystal property prediction. This question remains unexplored and is closely related to the broader topic of model interpretability in neural networks [4, 30, 31, 53, 29]. Future research could pursue a theoretical framework to quantify the effects of symmetry disruption and further elucidate its influence on prediction performance.

## A.11 Broader Impacts

As a frame-based method tailored for crystal structures, SPFrame enhances the accuracy of prediction models and facilitates the discovery of new materials with desirable properties. Therefore, this work has the potential to make a meaningful impact in the field of materials science. Furthermore, SPFrame offers a new perspective at the intersection of machine learning and materials science. By adapting machine learning techniques to account for the unique characteristics of crystal systems, SPFrame demonstrates how domain-specific modifications can significantly improve model performance. This highlights the importance of developing specialized methodologies to promote the effective application of machine learning in materials science.