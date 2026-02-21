## Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework

Mustafa Hajij 1,2 Hardik Kabaria 1 Balaji Cherukuri 1 Adrian Lew 4

Lennart Bastian 3,7 John L. Davenport 1 Joseph G. Kocheemoolayil 1 Theodore Papamarkou 5

Sarah Osentoski 1 Sheik Dawood 1 Nastaran Shahmansouri 1 Tolga Birdal 6

1 Vinci4D 2 University of San Francisco 3 Technical University of Munich 4 Stanford University 5 PolyShape 6 Imperial College London 7 MCML

## Abstract

We introduce copresheaf topological neural networks (CTNNs), a powerful unifying framework that encapsulates a wide spectrum of deep learning architectures, designed to operate on structured data, including images, point clouds, graphs, meshes, and topological manifolds. While deep learning has profoundly impacted domains ranging from digital assistants to autonomous systems, the principled design of neural architectures tailored to specific tasks and data types remains one of the field's most persistent open challenges. CTNNs address this gap by formulating model design in the language of copresheaves, a concept from algebraic topology that generalizes most practical deep learning models in use today. This abstract yet constructive formulation yields a rich design space from which theoretically sound and practically effective solutions can be derived to tackle core challenges in representation learning, such as long-range dependencies, oversmoothing, heterophily, and non-Euclidean domains. Our empirical results on structured data benchmarks demonstrate that CTNNs consistently outperform conventional baselines, particularly in tasks requiring hierarchical or localized sensitivity. These results establish CTNNs as a principled multi-scale foundation for the next generation of deep learning architectures.

## 1 Introduction

Deep learning has excelled by exploiting structural biases, such as convolutions for images [Krizhevsky et al., 2012], transformers for sequences [Vaswani et al., 2017], and message passing for graphs [Gilmer et al., 2017]. However, the design of architectures that generalize across domains with complex, irregular, or multiscale structure remains a notorious challenge [Bronstein et al., 2017, Hajij et al., 2023b]. Real-world data, which span physical systems, biomedical signals, and scientific simulations, rarely adhere to the regularity assumptions embedded in conventional architectures. These data are inherently heterogeneous, directional, and hierarchical, often involving relations beyond pairwise connections or symmetric neighborhoods.

Convolutional neural networks (CNNs), designed for uniform grids, do not fully capture local irregularities; graph neural networks (GNNs) often rely on homophily and tend to oversmooth feature representations as depth increases; and transformers, while excellent at capturing long-range dependencies, assume homogeneous embedding spaces, incur quadratic complexity, and lack builtin notions of anisotropy or variable local structures. These shortcomings highlight the need for a framework that can natively encode diverse local behaviors, respect directional couplings, and

propagate information across scales without sacrificing local variations or imposing unwarranted homogeneity.

To address this foundational gap, we propose copresheaf topological neural networks (CTNNs) , a unifying framework for deep learning based on copresheaves , a categorical structure that equips each local region of a domain with its own feature space, along with learnable maps specifying how information flows between regions. Unlike traditional models that assume a global latent space and isotropic propagation, our framework respects local variability in representation and directional flow of information, enabling architectures that are multiscale, anisotropic and expressive.

By constructing CTNNs on combinatorial complexes (CCs) [Hajij et al., 2023b,a], which generalize graphs, simplicial complexes, and cell complexes, we enable a principled message-passing mechanism over general topological domains formulated within the theory of copresheaves. This unified perspective subsumes many deep learning paradigms, including GNNs, attention mechanisms, sheaf neural networks [Hansen and Ghrist, 2019b, Bodnar et al., 2022], and topological neural networks [Papillon et al., 2023, Hajij et al., 2023b, Bodnar et al., 2021a, Ebli et al., 2020, Giusti et al., 2023] within a single formalism. Our approach further departs from the traditional assumption of a single shared latent space by modeling task-specific, directional latent spaces that bridge diverse deep learning frameworks. Furthermore, CTNNs flexibly handle both Euclidean and non-Euclidean data, supporting expressive architectures such as copresheaf GNNs, transformers, and CNNs,

Figure 1: A copresheaf topological neural network (CTNN) operates on combinatorial complexes (CCs), which generalize Euclidean grids, graphs, meshes, and hypergraphs. A CTNN is characterized by a set of locally indexed copresheaf maps ρ x i → x j , defined between cells x i and x j in the CC, and directed from x i to x j . The figure illustrates how a CTNN updates a local representation h x of a cell x using neighborhood representations h y and h z , which are sent to x via the learnable local copresheaf maps ρ x → x , ρ y → x and ρ z → x .

<!-- image -->

which learn structure-aware, directional transport maps. CTNNs offer a promising framework for developing next-generation, topologically informed, structure-aware machine learning models. See Figure 1 for an illustration.

## 2 Related Work

Our work here is related to sheaf neural networks (SNNs), which extend traditional GNNs by employing the mathematical framework of cellular sheaves to capture higher-order or heterogeneous relationships. Early work by Hansen and Ghrist [2019a,b] introduced methods to learn sheaf Laplacians from smooth signals and developed a spectral theory that connects sheaf topology with graph structure. Building on these ideas, Hansen and Gebhart [2020] proposed the first SNN architecture, demonstrating that incorporating edge-specific linear maps can improve performance on tasks involving asymmetric or heterogeneous relations.

Recent advances have focused on mitigating common issues in GNNs, such as oversmoothing and heterophily. For instance, Bodnar et al. [2022] introduced neural sheaf diffusion processes that address these challenges by embedding topological constraints into the learning process. Similarly, Barbero et al. [2022a,b] developed connection Laplacian methods and attention-based mechanisms that further enhance the expressiveness and efficiency of SNNs. The versatility of the sheaf framework has also been demonstrated through its extension to hypergraphs and heterogeneous graphs [Duta et al., 2023, Braithwaite et al., 2024], which enables modeling of higher-order interactions. Moreover, novel approaches incorporating joint diffusion processes [Hernandez Caralt et al., 2024] and Bayesian formulations [Gillespie et al., 2024] have improved the robustness and uncertainty quantification of SNNs. Finally, the application of SNNs in recommender systems [Purificato et al., 2023] exemplifies their practical utility in real-world domains. Together, these contributions demonstrate the potential of SNNs to enrich graph-based learning by integrating topological and geometric information directly into neural architectures. Our proposed CTNNs generalize these architectures while avoiding restrictive co-boundary maps or rank-specific Laplacian operators. Appendix I provides a more thorough literature review of related work.

## 3 Preliminaries

This section presents preliminary concepts needed for developing our theoretical framework. It revisits CCs and neighborhood structures, reviews sheaves and copresheaves on directed graphs, and compares cellular sheaves with copresheaves in graph-based modeling.

## 3.1 Combinatorial Complexes and Neighborhood Structures

To ensure generality, we base our framework on CCs [Hajij et al., 2023b,a], which unify set-type and hierarchical relations over which data are defined. CC-neighborhood functions then formalize local interactions forming a foundation for defining sheaves and higher-order message passing schemes for CTNNs.

Definition 1 (Combinatorial complex [Hajij et al., 2023b]) . A CC is a triple ( S , X , rk) , where S is a finite non-empty set of vertices, X ⊂ P ( S ) \ {∅} , with P ( S ) denoting the power set of S , and rk : X → Z ≥ 0 is a rank function such that if { s } ∈ X , rk( { s } ) = 0 for all s ∈ S , and x ⊆ y = ⇒ rk( x ) ≤ rk( y ) for all x, y ∈ X .

When context permits, we write a CC ( S , X , rk) simply as X . Each x ∈ X has rank rk( x ) , and dim X = max x ∈X rk( x ) . We refer to elements of X by cells . The k -cells x k of X are defined to be the cells x with rk( x ) = k . We use the notation X k = { x ∈ X : rk( x ) = k } = rk -1 ( { k } ) . See Fig 2 for an example.

Definition 2 (Neighborhood function) . A neighborhood function on a CC ( S , X , rk) is a map N : X → P ( X ) , which assigns to each cell x in X a collection of neighbor cells N ( x ) ⊂ X , referred to as the neighborhood of x . In our context, two neighborhood functions

<!-- image -->

:

Figure 2: A combinatorial complex of dimension 2.

are commonly used, namely the adjacency neighborhood N adj ( x ) = { y ∈ X | ∃ z ∈ X : x ⊂ z, y ⊂ z } and the incidence neighborhood N inc ( x ) = { y ∈ X | x ⊂ y } .

In practice, neighborhood functions are stored via matrices called neighborhood matrices .

Definition 3 (Neighborhood matrix) . Let N be a neighborhood function on a CC X . Let Y = { y 1 , . . . , y n } ⊂ X , Z = { z 1 , . . . , z m } ⊂ X be two collections of cells such that N ( y j ) ⊆ Z for all 1 ≤ j ≤ n . An element of the neighborhood matrix G ∈ { 0 , 1 } m × n is defined as

<!-- formula-not-decoded -->

The copresheaf structure that we develop on CCs depends on the neighborhood function. To introduce it, we first review sheaves and copresheaves on graphs, and then extend these notions to CCs.

## 3.2 Sheaves and Copresheaves on Directed Graphs

The copresheaf formalism assigns each vertex its unique feature space F ( x ) , respecting the potentially heterogeneous nature of the data, and each directed edge a transformation ρ x → y that tells how data move between those spaces, F ( x ) →F ( y ) . This separation between where data reside and how they travel provides a foundation for learning beyond the single-latent-space assumption of standard deep learning architectures. Concretely, copresheaves are defined as follows 1 :

Definition 4 (Copresheaf on directed graphs) . A copresheaf ( F , ρ, G ) on a directed graph G = ( V, E ) is given by

- a real vector space F ( x ) for every vertex x ∈ V ;
- a linear map ρ x → y : F ( x ) -→ F ( y ) for every directed edge x → y ∈ E .

Think of a copresheaf as a system for sending messages across a network, where each node has its own language (stalk), and edges translate messages (linear maps) to match the recipient's language. More specifically, on a directed graph G = ( V, E ) , each vertex x ∈ V carries a task-specific latent space

1 While we avoid overly complicated jargon, the appendix links our constructs to those of category theory for a more rigorous exposition.

F ( x ) , and every edge x → y ∈ E applies a learnable, edge-indexed linear map ρ x → y : F ( x ) →F ( y ) that re-embeds x 's features into y 's coordinate frame, thus realizing directional, embedding-level message passing throughout the network.

While much of the recent literature has focused on sheaf learning [Ayzenberg et al., 2025], our approach is based on a copresheaf perspective. This setup departs from the traditional deep learning core assumption of a single, shared latent space , enabling the modeling of heterogeneous, taskspecific latent spaces and directional relations. The significance of this approach lies in its ability to generalize and connect different deep learning paradigms. Copresheaf-type architectures extend beyond SNNs [Hansen and Gebhart, 2020, Bodnar et al., 2022, Barbero et al., 2022b, Duta et al., 2023, Battiloro et al., 2024b] and TNN architectures [Hajij et al., 2023b], typically designed for non-Euclidean data, by also accommodating Euclidean data effectively. This versatility allows them to unify applications across diverse data domains and architectural frameworks, providing a unified structure that uses directional information flow and adapts to task-specific requirements.

In graph-based modeling, cellular sheaves provide a formal framework to ensure data consistency in undirected graphs by encoding symmetric local-to-global relations via an incidence structure between vertices and edges. These structural points, formalized in the following definition, are encoded through restriction maps , ensuring data consistency between vertices and incident edges.

Definition 5 (Cellular sheaf) . Let G = ( V, E ) be a undirected graph. Let x ⊴ e indicate that vertex x ∈ V is incident to edge e ∈ E . A cellular sheaf on G consists of:

- a vector space F ( x ) to each vertex x ∈ V ;
- a vector space F ( e ) to each edge e ∈ E ;

· a linear restriction map F x ⊴ e : F ( x ) →F ( e ) for each incidence x ⊴ e .

For any cell c (vertex or edge), the vector space F ( c ) is typically called the stalk at c . The data on nodes x and y , denoted by h x ∈ F ( x ) and h y ∈ F ( y ) , 'agree' on the edge e if their images under the restriction maps coincide:

<!-- formula-not-decoded -->

A global section of a sheaf on a graph G assigns data h v ∈ F ( v ) to each vertex v and h e ∈ F ( e ) to each edge e , such that for every edge e between nodes x and y it holds that F x◁e ( h x ) = F y◁e ( h y ) . This consistency condition ensures data consistency across local connections in a network. Global sections represent equilibrium states, where this local agreement holds across the entire graph, enabling unified data representations for a complex system. Most sheaf-based architectures have focused on diffusion-type models [Hansen and Gebhart, 2020, Bodnar et al., 2022, Barbero et al., 2022b, Duta et al., 2023, Battiloro et al., 2024b], where the sheaf Laplacian ∆ F minimizes the Dirichlet energy, ensuring global consistency. Precisely, let C 0 ( G ; F ) = ⊕ v ∈ V F ( v ) and C 1 ( G ; F ) = ⊕ e ∈ E F ( e ) denote the spaces of vertex-valued and edge-valued cochains of the sheaf F , respectively. Then for some arbitrary choice of orientation for each edge, define the coboundary map

<!-- formula-not-decoded -->

which measures local disagreement with respect to the edge e . The sheaf Laplacian ∆ F = δ T δ aggregates all the restriction maps {F x◁e } into a single symmetric, positive semidefinite operator. Its associated quadratic form, h T ∆ F h , has a trace that defines the sheaf Dirichlet energy .

Unlike sheaves, which ensure data consistency across overlaps, copresheaves model directional data flow, making them well-suited for processes such as information propagation, causality, and hierarchical dependencies. Copresheaves assign vector spaces F ( x ) only to vertices and define learnable linear maps ρ x → y : F ( x ) →F ( y ) along directed edges, without imposing sheaf consistency constraints. This vertex-centric, anisotropic framework naturally integrates with message-passing architectures such as GNNs and TNNs, allowing parameterized maps to adapt during training. See Appendix B.3 for further discussion, and Appendix C for the definition and properties of the copresheaf Laplacian .

## 4 Copresheaf Topological Neural Networks

We are now ready to introduce copresheaf topological neural networks (CTNNs), a higher-order message-passing mechanism that generalizes the modeling of relational structures within TNNs, as

illustrated in Figure 1. We begin by making use of copresheaves induced by neighborhood functions on CCs, providing a structured way to model general, local-to-global relations.

̸

<!-- formula-not-decoded -->

Let X be a CC and let N : X → P ( X ) be a neighborhood function. We define the effective support of N as the set X N := { x ∈ X | N ( x ) = ∅} . This set identifies cells that receive input from neighbors. The neighborhood function N induces a directed graph G N = ( V N , E N ) , where the vertex set is and the edge set is E N := { y → x | x ∈ X , y ∈ N ( x ) } . The vertex set V N includes both cells with non-empty neighborhoods (targets) and their neighbors (sources). This graph determines how data propagates across the complex, with each edge encoding a directional relation from neighbor y to target x .

Definition 6 (Neighborhood-dependent copresheaf) . Let X be a CC, N a neighborhood function, and G N = ( V N , E N ) the induced directed graph with V N = X N ∪ ⋃ x ∈X N ( x ) . An N -dependent copresheaf assigns a vector space F ( x ) to each x ∈ V N , and a linear map ρ y → x : F ( y ) →F ( x ) for each edge y → x ∈ E N .

See Appendix B.4 for concrete examples. When clear from context, we simplify the notation from F N to F and ρ N y → x to ρ y → x .

Copresheaf neighborhood matrices . Having introduced neighborhood matrices as binary encodings of local interactions, we now generalize this notion to define copresheaf neighborhood matrices (CNMs). Instead of binary entries, CNMs consist of copresheaf maps between data assigned to cells in a CC, allowing richer encoding of local dependencies. Subsequently, we define specialized versions, such as copresheaf adjacency and incidence matrices , that capture specific topological relations, facilitating structured message-passing in our CTNNs.

In particular, define the k -cochain space to be the direct sum C k ( X , F N ) = ⊕ x ∈X k F N ( x ) , and denote by Hom( F N ( i ) , F N ( j )) the space of linear maps from the data at the i -th stalk to that at the j -th stalk. Here, the maps encode how data is transferred or transformed between neighboring cells. We next define the CNM.

Definition 7 (Copresheaf neighborhood matrices) . For a CC X equipped with a neighborhood function N , let ( X , ρ, G N ) be a copresheaf on X . Also let Y = { y 1 , . . . , y n } ⊆ X and Z = { z 1 , . . . , z m } ⊆ X be two collections of cells such that N ( y j ) ⊆ Z for all 1 ≤ j ≤ n . The copresheaf neighborhood matrix of N with respect to Y and Z is the m × n matrix G N :

<!-- formula-not-decoded -->

The neighborhood function N determines the directed relationships between the cells, and the CNM encodes the interactions between cells induced by these relationships, collecting the maps of a cell from its neighbors. Next, analogous to Definition 2, we define copresheaf adjacency and copresheaf incidence matrices as specialized forms of the general CNM.

Definition 8 (Copresheaf adjacency/incidence matrices) . For fixed r, k , define N ( r,k ) adj ( x ) = { y ∈ X r | ∃ z ∈ X r + k , x ⪯ z, y ⪯ z } and N ( r,k ) inc ( x ) = { y ∈ X k | x ⪯ y } . The copresheaf adjacency matrix (CAM) A r,k ∈ R |X r |×|X r | and copresheaf incidence matrix (CIM) B r,k ∈ R |X k |×|X r | are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These specialized matrices capture distinct topological relations. A CAM encodes relations in which cells share an upper cell, while a CIM encodes relations in which one cell is incident to another.

Copresheaf-based message passing . We now generalize traditional graph message-passing to heterogeneous and higher-order interactions involving cells of varying ranks or multi-way relations

going beyond pairwise connections, by explicitly incorporating copresheaf structures defined over CCs. This leads to copresheaf-based message-passing , a flexible and expressive tool for capturing complex, multi-scale relations in structured data.

Definition 9 (Copresheaf message-passing neural network) . Let G = ( V, E ) be a directed graph and ( F , ρ, G ) a copresheaf. For each layer l and vertex x ∈ V , let h ( l ) x ∈ F ( x ) . A copresheaf message-passing neural network (CMPNN) is a neural network whose meassage passing is defined by

<!-- formula-not-decoded -->

where ρ y → x : F ( y ) →F ( x ) is the linear map associated with edge y → x , α is a learnable message function, ⊕ a permutation-invariant aggregator, and β a learnable update function.

Definition 9 establishes a unifying framework that generalizes graph-based message passing neural networks (MPNNs) [Gilmer et al., 2017] by incorporating learnable, anisotropic linear maps associated with directed edges. This formulation subsumes many standard GNNs, such as graph convolutional networks [Kipf and Welling, 2017], graph attention networks [Veliˇ ckovi´ c et al., 2018], and their variants [Veliˇ ckovi´ c, 2022], by viewing message passing as copresheaf maps on directed graphs. Consequently, all architectures derived from these foundational GNN models fit within our copresheaf message-passing paradigm. Moreover, SNNs, which operate on cellular sheaves over undirected graphs, can be adapted into this framework by reinterpreting their edge-mediated transport operator as direct vertex-to-vertex morphisms on a bidirected graph. This adaptation is detailed in the following theorem.

Theorem 1. (SNNs are CMPNNs) Let G = ( V, E ) be an undirected graph equipped with a cellular sheaf F assigning vector spaces to vertices and edges, and linear maps F x ⊴ e for each vertex x ∈ e . Then for each edge e = { x, y } ∈ E , the SNN message passing from y to x , given by the composition F ⊤ x ⊴ e ◦ F y ⊴ e : F ( y ) → F ( x ) , can be realized as a single morphism ρ y → x : F ( y ) → F ( x ) in a copresheaf on the bidirected graph G ′ = ( V, E ′ ) , E ′ = { ( x, y ) , ( y, x ) | { x, y } ∈ E } by setting ρ y → x = F ⊤ x ⊴ e ◦ F y ⊴ e .

Theorem 1 establishes that the message-passing scheme employed by SNNs can be interpreted as a special case of CMPNNs when restricted to bidirected graphs. This perspective generalizes most existing SNN architectures found in the literature, including those in Hansen and Gebhart [2020], Bodnar et al. [2022], Barbero et al. [2022b]. The map F ⊤ x◁e ◦ F y◁e arises from composing the restriction map F y◁e : F ( y ) →F ( e ) with its adjoint F ⊤ x◁e : F ( e ) →F ( x ) , thereby capturing both vertex-to-edge and edge-to-vertex transformations defined by the cellular sheaf. As a consequence of this connection between SNNs and CMPNNs, diffusion-style updates (commonly used in sheaf-based models, such as those based on the sheaf Laplacian) can be succinctly expressed within the CMPNN framework. This result is formally stated in Proposition 1, with a full proof provided in Appendix E.

Proposition 1 (Neural-sheaf diffusion [Bodnar et al., 2022] as copresheaf message-passing) . Let G = ( V, E ) be an undirected graph endowed with a cellular sheaf F . Given vertex features H = [ h v ] v ∈ V with h v ∈ F ( v ) , and learnable linear maps W 1 , W 2 , define the diffusion update

<!-- formula-not-decoded -->

̸

where ∆ F = [ L F,v,u ] v,u ∈ V has blocks L F,vv = ∑ v ⊴ e F ⊤ v ⊴ e F v ⊴ e , L F,vu = -F ⊤ v ⊴ e F u ⊴ e , for u = v, u ⊴ e, v ⊴ e . Then, H + can be expressed in the copresheaf message-passing form of Definition 9.

The next definition formalizes the notion of general multi-way propagation.

Definition 10 (Copresheaf-based higher-order message passing) . Let X be a CC, and N = {N k } n k =1 a collection of neighborhood functions. For each k , let ( F N k , ρ N k , G N k ) be a copresheaf in which the maps ρ N k y → x : F N k ( y ) →F N k ( x ) define the transformations associated to the copresheaf. Given features h ( ℓ ) x , the next layer features are defined as

<!-- formula-not-decoded -->

where α N k is the message function, ⊕ a permutation-invariant aggregator over neighbors y ∈ N k ( x ) , ⊗ combines information from different neighborhoods, and β is the update function.

Table 1: A unified view across domains and architectures -CNNs, GNNs, Transformers, SNNs, and TNNs-as instances of Copresheaf Topological Neural Networks (CTNNs) defined by neighborhood graphs G N and directional transports ρ y → x .

| Classical model                                                       | CTNN form                 | Domain / N                          | ρ y → x / refs                                                         |
|-----------------------------------------------------------------------|---------------------------|-------------------------------------|------------------------------------------------------------------------|
| CNN Li et al. [2021]                                                  | CopresheafConv            | Grid; adjacency (CAM)               | Translation-consistent transport (shared local filters). App. G        |
| MPNN Gilmer et al. [2017]                                             | CMPNN                     | Graph; adjacency                    | Shared / edge-indexed linear maps; GCN/GAT special cases. Def. 10      |
| Euclidean / cellular Tr. Vaswani et al. [2017], Barsbey et al. [2025] | Copresheaf Transformer    | Tokens on grid/seq; full/masked adj | In-head value transport; ρ = I ⇒ dot-product attention. Sec. 5, App. F |
| SNN Bodnar et al. [2022], Hansen and Gebhart [2020]                   | CMPNN on bidirected graph | Graph; incidence (CIM)              | ρ u ← v = F ⊤ u◁e F v◁e (vertex-vertex). Prop. 1, Thm. 1, App. E, B.3  |
| TNN (hyper- graph/simplicial/cellular/CC) jij et al. [2023b]          | Ha- Higher-order CMPNN    | CC; multi- N via CAM/CIM            | Rank-aware maps across overlaps; multi-way ⊗ . Def. 10                 |

Abbrev: CC = combinatorial complex; CAM/CIM = copresheaf adjacency/incidence; Adj/Inc = adjacency/incidence.

Definition 10 lays the foundational framework that unifies a broad class of topological deep learning architectures, bridging higher-order message passing methods, transformers and SNNs. This synthesis not only consolidates existing approaches but also opens avenues for novel architectures based on topological and categorical abstractions. Notably, the formulation in Proposition 10 encompasses simplicial message passing [Ebli et al., 2020, Bunch et al., 2020, Bodnar et al., 2021b], cellular message passing [Hajij et al., 2020, Bodnar et al., 2021a], stable message passing via Hodge theory [Hayhoe et al., 2022], and recurrent simplicial architectures for sequence prediction [Mitchell et al., 2024]. It also subsumes more recent developments that harness multiple signals and higher-order operators such as the Dirac operator [Calmon et al., 2022, Hajij et al., 2023a] and TNNs [Hajij et al., 2023b]. These diverse models are unified under the copresheaf-based formulation by interpreting neighborhood aggregation, feature transport, and signal interaction within a coherent framework. See Appendix E for derivations showing how several of these architectures emerge as special cases of this general formulation. See also Table 1 for a summary.

Remark 1 (Graph vs CC copresheaf models) . Unlike graph-based models, which propagate information edge by edge, copresheaf models on a CC aggregate messages across all overlapping neighborhoods at once. Overlapping neighborhoods have common cells, potentially at different ranks, allowing simultaneous aggregation of multi-way interactions. Applying each neighborhood function N k in turn, we compute its map-driven messages and then merge them into a single update.

## 5 Architectures Derived from the Copresheaf Framework

Having established the abstract copresheaf-based framework on a CC X , we now present several concrete instantiations. Copresheaf transformers (CTs) extend the standard attention mechanism by dynamically learning linear maps ρ y → x : F ( y ) → F ( x ) encoding directional, anisotropic relationships between tokens, i.e., cells, in X . Integrating these maps into attention enables CTs to capture rich, structured interactions. Copresheaf graph neural networks (CGNNs) generalize messagepassing GNNs by incorporating copresheaf linear maps to model relational structures. Copresheaf convolutional networks define convolution-like operations on CCs, modeled as a Euclidean grid, using these linear maps. We present the CT construction next and leave the exact formulations of copresheaf networks, CGNNs, and copresheaf convolution layers to the appendices E, F and G.

Copresheaf transformers . Having established the abstract framework of CTNNs on a CC X , we now introduce a concrete instantiation: the copresheaf transformer (CT) layer. This layer extends the standard attention mechanism by dynamically learning linear maps ρ y → x : R d y → R d x that encode both the combinatorial structure and directional, anisotropic relationships within the complex. By integrating these maps into the attention computation, the CT layer captures rich, structured

Table 2: Mean squared error (mean ± standard deviation) of classical vs copresheaf architectures for learning various physics simulations.

| Network    | Heat (Transformer)                | Advection (Transformer)           | Unsteady stokes (Conv-transformer)   |
|------------|-----------------------------------|-----------------------------------|--------------------------------------|
| Classical  | 2 . 64 × 10 - 4 ± 3 . 50 × 10 - 5 | 3 . 52 × 10 - 4 ± 7 . 70 × 10 - 5 | 1 . 75 × 10 - 2 ± 1 . 32 × 10 - 3    |
| Copresheaf | 9 . 00 × 10 - 5 ± 7 . 00 × 10 - 6 | 1 . 20 × 10 - 4 ± 1 . 20 × 10 - 5 | 1 . 48 × 10 - 2 ± 1 . 48 × 10 - 4    |

interactions across X . At layer ℓ , each cell x ∈ X is associated with a feature h ( ℓ ) x ∈ R d x , where d x denotes the feature dimension of cell x .

Definition 11 (Copresheaf self-attention) . For a fixed rank k and neighborhood N k (e.g., adjacency between k -cells), let W q , W k ∈ R p × d , W v ∈ R d × d denote learnable projection matrices, where p is the dimension of the query and key spaces, and d is the feature dimension (assumed uniform across cells for simplicity). For each k -cell x ∈ X k , copresheaf self-attention defines the message aggregation and feature update as h ( ℓ +1) x = β ( h ( ℓ ) x , m x ) , where m x = ∑ y ∈N k ( x ) a xy ρ y → x ( v y ) and

<!-- formula-not-decoded -->

where q x = W q h ( ℓ ) x , k x = W k h ( ℓ ) x , and v x = W v h ( ℓ ) x . Here, the softmax normalizes over all neighbors y ′ ∈ N k ( x ) and ρ y → x : F ( y ) →F ( x ) is the learned copresheaf map. The update function β is chosen to be a neural network.

Similarly, we define copresheaf cross-attention among s and t rank cells in X , as well as a general algorithm for a corpresheaf transformer layer (Appendix F).

## 6 Experimental Evaluation

We conduct experiments on synthetic and real data in numerous settings to support the generality of our framework. These include learning physical dynamics, graph classification in homophillic and heterophilic cases and classifying higher-order complexes.

## 6.1 Evaluations on Physics Datasets

To verify the validity of our networks in toy setups of different phenomena, we generate a series of synthetic datasets. These include:

1. Heat. We generate 600 realisations by solving the heat equation u t = νu xx on [0 , 1) with ν = 0 . 1 to horizon T = 0 . 1 ; each u 0 is a 10-mode sine series and u T is its Gaussian-kernel convolution, sampled on N = 100 grid points.
2. Advection. Similar to heat, we generate 600 realisations of u t + c u x = 0 with c = 1 ; the solution is a pure phase shift of u 0 , sampled on N = 130 points. Each pair is normalized to the interval [0 , 1] and the dataset is split into 500:100 train/test samples.
3. Unsteady stokes. Let u ( x, y, t ) = ( u ( x, y, t ) , v ( x, y, t ) ) ∈ R 2 denote the incompressible velocity field, p ( x, y, t ) ∈ R the kinematic pressure, and ν &gt; 0 the kinematic viscosity. Throughout, ∂ t is the time derivative, ∇ = ( ∂ x , ∂ y ) the spatial gradient, and ∆ = ∂ xx + ∂ yy the Laplacian operator. The periodic unsteady Stokes system reads ∂ t u -ν ∆ u + ∇ p = 0 , where ∇· u = 0 . We synthesize 200 samples by drawing an 8-mode Fourier stream-function ψ , setting u 0 = ∇ ⊥ ψ , and evolving to T = 0 . 1 with ν = 0 . 1 via the analytic heat kernel. Each pair is sampled on a 16 × 16 grid, with each channel normalized to the interval [0 , 1] , and the dataset is split into 160:40 train/test samples.

Model and training . For heat and advection, we use two transformer layers (positional encoding, four heads, stalk dimension equal to 16 ), followed by a mean pooling and linear head yielding 64D token embeddings. We train our networks using AdamW wiht a learning rate of 10 -3 , cosine LR scheduling, and batch size of two. We use 50 epochs for the heat dataset and 80 for the advection dataset, and report the results over three seeds. For the unsteady Stokes data, we test a compact convolution-transformer U-Net consisting of a convolutional encoder with two input and 32 output channels, followed by two transformer layers (four heads, hidden dimension equal to 32 , and stalk dimension 8 ), and a convolutional decoder mapping back to two output channels. We train it for 300

epochs using AdamW with a batch size of four. The classical baselines use dot-product attention, whereas the copresheaf variants employ learned outer-product maps.

Results . As shown in Table 2, in the heat and advection tests, copresheaf attention significantly outperforms classical dot-product attention, reducing the test MSE by over 50% and achieving more stable results across seeds. For unsteady Stokes, the copresheaf attention lowers the test MSE by ≈ 15% and reduces variance by an order of magnitude, confirming that pair-specific linear transports capture viscous diffusion of vorticity more faithfully than standard self-attention under identical compute budgets.

## 6.2 Graph Classification

We evaluate whether incorporating copresheaf structure into GNNs improves performance on graph classification tasks.

Data . MUTAG dataset, a nitroaromatic compound classification benchmark consisting of 188 molecular graphs, where nodes represent atoms and edges represent chemical bonds. Each node is associated with a 7 -dimensional feature vector encoding atom type, and each graph is labeled as mutagenic or non-mutagenic (two classes). The dataset is split into 80% train and 20% test samples.

Baselines, backbone and training . We compare standard GNN models (GCN, GraphSAGE, GIN) against their copresheaf-counterparts (CopresheafGCN, CopresheafSage, CopresheafGIN) derived below. All models are two-layer networks with a hidden dimension of 32 for GCN and GraphSAGE, and 16 for GIN, followed by global mean pooling and a linear classifier to predict graph labels. The standard models (GCN [Kipf and Welling, 2017], GraphSAGE [Hamilton et al., 2017], GIN [Xu et al., 2019]) use conventional GNN convolutions: GCN with symmetric normalization, GraphSAGE with mean aggregation, and GIN with sum aggregation. The copresheaf-enhanced models augment these with learned per-edge copresheaf maps, introducing local consistency constraints via transformations. All models are trained using Adam with a learning rate of 0 . 01 , and a batch size of 16 . GCN and GIN models are trained for 100 epochs, while GraphSAGE for 50 . The negative log-likelihood loss is minimized, and performance is evaluated via test accuracy. For GCN and GraphSAGE, we use five runs, while GIN uses ten runs.

Enhancing GNNs via copresheaves . The copresheaf structure enhances each GNN by learning a transport map ρ ij = I +∆ ij for each edge ( i, j ) , where ∆ ij is a learned transformation and I is the identity matrix. In what follows D represents the dimension of the input feature in where we apply the copresheaf maps ρ ij . Denote by [ h i ; h j ] to the concatenation of the two node feature vectors h i and h j along their feature dimension. The process for each model is as follows:

- CopresheafGCN . For node features h i , h j , compute ∆ ij = tanh(Linear([ h i ; h j ])) , take its diagonal to get a diagonal D × D matrix with D = 7 . Form ρ ij = I +∆ ij . Aggregate neighbor features as h ′ i = ∑ j ∈N ( i ) 1 √ d i d j ρ ij h j , where d i , d j are the degrees of nodes i and j , respectively. Combine with the selffeature: h ′′ i = (1 + ϵ ) h i + h ′ i .
- CopresheafSage . Compute ∆ ij = tanh( Linear ([ h i , h j ])) , a diagonal matrix, and form ρ ij = I +∆ ij . Aggregate via mean: h ′ i = mean j ∈N ( i ) ( ρ ij h j ) . Combine: h ′′ i = (1+ ϵ ) h i + h ′ i . The map ρ ij enhances local feature alignment.
- CopresheafGIN . Compute ∆ ij = tanh( Linear ([ h i , h j ])) , a full D × D matrix, and form ρ ij = I + ∆ ij . Aggregate: h ′ i = ∑ j ∈N ( i ) ( ρ ij h j ) /d i . Combine: h ′′ i = (1 + ϵ ) h i + h ′ i .

Table 3: Mean test accuracy ( ± std).

| Model          | Accuracy      |
|----------------|---------------|
| GCN            | 0.674 ± 0.014 |
| CopresheafGCN  | 0.721 ± 0.035 |
| GraphSAGE      | 0.689 ± 0.022 |
| CopresheafSage | 0.732 ± 0.029 |
| GIN            | 0.700 ± 0.039 |
| CopresheafGIN  | 0.724 ± 0.021 |

Results . On MUTAG, copresheaf-enhanced GNNs consistently outperform their standard versions across GCN, GraphSAGE, and GIN. CopresheafSAGE achieves the highest average accuracy ( 0 . 732 ) and the largest relative gain over the GraphSAGE baseline ( 0 . 689 ). Learned per-edge transport maps better capture complex structure and enforce local consistency, improving classification. These results demonstrate the promise of copresheaf structures for molecular graph classification.

## 6.3 Combinatorial Complex Classification

Finally, we assess whether incorporating copresheaf structure into transformer-based attention mechanisms improves performance on classifying higher-order, general data structures, such as CCs. Specifically, we compare a classical transformer model against two CT variants (CT-FC and CTSharedLoc) on a synthetic dataset of CCs.

Data . Our synthetic dataset comprises 200 training and 50 test CCs derived from Erd˝ os-Rényi graphs, each with 10 nodes and a base edge probability of 0 . 5 . Triangles are added to form higher-order structures with probability q = 0 . 1 for class 0 (low density) or q = 0 . 5 for class 1 (high density). Each node has 2D feature vectors, consisting of its degree and the number of triangles it participates in, with added Gaussian noise N (0 , 0 . 1) .

Backbone and training . All models are transformer-based classifiers with a single block, using two attention heads, an embedding dimension of 8 , and a head dimension of 4 . The model embeds 2D node features, applies attention, performs global average pooling, and uses a linear classifier to predict one of two classes (low or high triangle density). The classical model employs standard multi-head attention. The CT-FC and CT-SharedLoc models augment attention with learned per-edge transport maps to enforce local consistency. Models are trained for four epochs using Adam with a learning rate of 10 -3 and a batch size of 8 , minimizing cross-entropy loss. Performance is evaluated via test accuracy. Experiments are over four runs with different random seeds to ensure robustness.

Copresheaf attention . Similar to GNNs, the copresheaf structure enhances attention by learning a transport map ρ ij = I +∆ ij . The value vector v j is transformed as ρ ij v j before attention-weighted aggregation. The process for each model is as follows:

- CT-FC . For node features h i , h j , compute ∆ ij = tanh( Linear ([ h i , h j ])) , a full d × d matrix ( d = 4 ). Form ρ ij = I +∆ ij . Apply ρ ij to value vectors in attention: v t = ρ ij v j . The map ρ ij enables rich feature transformations across nodes.
- CT-SharedLoc . Compute a shared ∆ ij = tanh( MLP ([ h i , h j ])) , a full d × d matrix, and a local scalar α ij = σ ( MLP ([ h i , h j ])) . Form ρ ij = I + α ij ∆ ij . Apply ρ ij to value vectors: v t = ρ ij v j . The map ρ ij balances shared transformations with local modulation.

Results . Copresheaf transformer models outperform the standard transformer on the CC classification task, with CT-SharedLoc achieving the highest average accuracy ( 0 . 970 ) and competitive stability (std 0 . 010 ). The learned per-edge transport maps ρ ij enhance the model's ability to capture higherorder structural patterns, such as triangle density, by aligning node features effectively. CT-SharedLoc's combination of shared transport maps and local modulation yields the best performance, showcasing the value of copresheaf structures in transformer-based models for CC classification.

## 7 Discussion and Conclusions

We proposed CTNNs, a unified deep learning framework on (un)structured data. By develping models on copresheaves over CCs, CTNNs generalize GNNs, SNNs, and TNNs through directional, heterogeneous message passing. Besides theoretical advances, CTNNs offer empirical benefits across diverse tasks, laying the principles for multiscale and anisotropic representation learning.

Limitations and future work . CTNNs incur additional overhead from per-edge transformations and have so far been evaluated in modest-scale settings. We plan to explore well-engineered scalable parameterizations, extend CTNNs to large-scale and dynamic domains, and further connect categorical structure with robustness and inductive bias in deep learning.

## Acknowledgments and Disclosure of Funding

Professor Adrian J. Lew's contributions to this publication were as a paid consultant and were not part of his Stanford University duties or responsibilities. T. Birdal acknowledges support from the Engineering and Physical Sciences Research Council [grant EP/X011364/1]. T. Birdal was supported by a UKRI Future Leaders Fellowship [grant number MR/Y018818/1] as well as a Royal Society

Table 4: Mean ± std test accuracy for CC classification.

| Model        | Accuracy      |
|--------------|---------------|
| Classic      | 0.940 ± 0.014 |
| CT-FC        | 0.955 ± 0.009 |
| CT-SharedLoc | 0.970 ± 0.010 |

Research Grant RG/R1/241402. The authors thank Hans Riess for pointing out the relationship between the CTNNs and the quiver Laplacian.

## References

- Samson Abramsky. Notes on presheaf representations of strategies and cohomological refinements of k-consistency and k-equivalence. arXiv preprint arXiv:2206.12156 , 2022.
- Ibrahem Al-Jabea and Thomas John Baird. Cohomology of gkm-sheaves. arXiv preprint arXiv:1806.01761 , 2018.
- Yash Atri, Karan Rungta, Lili Mou, Kai-Wei Chang, and Nanyun Joshi. Promoting topic coherence and inter-document consorts in multi-document summarization via simplicial complex and sheaf graph. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , pages 2154-2166, 2023.
- Anton Ayzenberg, Thomas Gebhart, German Magai, and Grigory Solomadin. Sheaf theory: from deep geometry to deep learning. arXiv preprint arXiv:2502.15476 , 2025.
- Song Bai, Feihu Zhang, and Philip H. S. Torr. Hypergraph convolution and hypergraph attention. Pattern Recognition , 110:107637, 2021.
- Rubén Ballester, Pablo Hernández-García, Mathilde Papillon, Claudio Battiloro, Nina Miolane, Tolga Birdal, Carles Casacuberta, Sergio Escalera, and Mustafa Hajij. Attending to topological spaces: The cellular transformer. arXiv preprint arXiv:2405.14094 , 2024.
- Jacob Bamberger, Federico Barbero, Xiaowen Dong, and Michael M. Bronstein. Bundle neural networks for message diffusion on graphs. In ICML 2024 Workshop on Geometry-grounded Representation Learning and Generative Modeling , 2024.
- Federico Barbero, Cristian Bodnar, Haitz S. de Ocáriz Borde, Michael M. Bronstein, Petar Veliˇ ckovi´ c, and Pietro Liò. Sheaf neural networks with connection laplacians. In Topology, Algebra, and Geometry in Machine Learning (TAG-ML) Workshop at ICML , pages 28-36, 2022a.
- Federico Barbero, Cristian Bodnar, Haitz S. de Ocáriz Borde, and Pietro Liò. Sheaf attention networks. In NeurIPS 2022 Workshop on Symmetry and Geometry in Neural Representations , 2022b.
- Melih Barsbey, Rubén Ballester, Andac Demir, Carles Casacuberta, Pablo Hernández-García, David Pujol-Perich, Sarper Yurtseven, Sergio Escalera, Claudio Battiloro, Mustafa Hajij, et al. Higherorder molecular learning: The cellular transformer. In ICLR 2025 Workshop on Generative and Experimental Perspectives for Biomolecular Design , 2025.
- Claudio Battiloro, Lucia Testa, Lorenzo Giusti, Stefania Sardellitti, Paolo Di Lorenzo, and Sergio Barbarossa. Generalized simplicial attention neural networks. IEEE Transactions on Signal and Information Processing over Networks , 2024a.
- Claudio Battiloro, Zhiyang Wang, Hans Riess, Paolo Di Lorenzo, and Alejandro Ribeiro. Tangent bundle convolutional learning: from manifolds to cellular sheaves and back. IEEE Transactions on Signal Processing , 72:1892-1909, 2024b.
- Federico Battiston, Giulia Cencetti, Iacopo Iacopini, Vito Latora, Maxime Lucas, Alice Patania, Jean-Gabriel Young, and Giovanni Petri. Networks beyond pairwise interactions: structure and dynamics. Physics Reports , 874:1-92, 2020.
- Christian Bick, Elizabeth Gross, Heather A Harrington, and Michael T Schaub. What are higher-order networks? SIAM review , 65(3):686-731, 2023.
- Cristian Bodnar, Fabrizio Frasca, Nina Otter, Yuguang Wang, Pietro Lio, Guido F Montufar, and Michael Bronstein. Weisfeiler and Lehman go cellular: CW networks. Advances in Neural Information Processing Systems , 34:2625-2640, 2021a.
- Cristian Bodnar, Fabrizio Frasca, Yuguang Wang, Nina Otter, Guido F Montufar, Pietro Lio, and Michael Bronstein. Weisfeiler and Lehman go topological: message passing simplicial networks. pages 1026-1037, 2021b.

- Cristian Bodnar, Fabrizio Frasca, Yuguang Wang, Nina Otter, Pietro Liò, Guido Montúfar, and Michael Bronstein. Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in gnns. Advances in Neural Information Processing Systems , 35:18527-18541, 2022.
- Luke Braithwaite, Iulia Duta, and Pietro Liò. Heterogeneous sheaf neural networks. arXiv preprint arXiv:2409.08036 , 2024.
- Glen E. Bredon. Sheaf Theory , volume 170 of Graduate Texts in Mathematics . Springer, 1997.
- Michael M. Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric deep learning: going beyond Euclidean data. IEEE Signal Processing Magazine , 34(4):18-42, 2017.
- Eric Bunch, Qian You, Glenn Fung, and Vikas Singh. Simplicial 2-complex convolutional neural nets. NeurIPS Workshop on Topological Data Analysis and Beyond , 2020.
- Lucille Calmon, Michael T. Schaub, and Ginestra Bianconi. Higher-order signal processing with the Dirac operator. In 56th Asilomar Conference on Signals, Systems, and Computers , pages 925-929. IEEE, 2022.
- Adam Ó Conghaile. Cohomology in constraint satisfaction and structure isomorphism. arXiv preprint arXiv:2206.15253 , 2022.
- Justin Michael Curry. Sheaves, Cosheaves and Applications . PhD thesis, University of Pennsylvania, 2014.
- Iulia Duta, Giulia Cassarà, Fabrizio Silvestri, and Pietro Liò. Sheaf hypergraph networks. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023) , 2023.
- Stefania Ebli, Michaël Defferrard, and Gard Spreemann. Simplicial neural networks. NeurIPS Workshop on Topological Data Analysis and Beyond , 2020.
- Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and Yue Gao. Hypergraph neural networks. Proceedings of the AAAI Conference on Artificial Intelligence , 33(01):3558-3565, 2019.
- Michael Fourman, Christopher Mulvey, and Dana Scott, editors. Applications of Sheaves: Proceedings of the Research Symposium on Applications of Sheaf Theory to Logic, Algebra and Analysis , volume 753 of Lecture Notes in Mathematics . Springer, 1977.
- Thomas Gebhart, Jakob Hansen, and Paul Schrater. Knowledge sheaves: A sheaf-theoretic framework for knowledge graph embedding. In International Conference on Artificial Intelligence and Statistics , pages 9094-9116. PMLR, 2023.
- Robert Ghrist and Yasuaki Hiraoka. Applications of sheaf cohomology and exact sequences to network coding. In NOLTA , 2011.
- Patrick Gillespie, Vasileios Maroulas, and Ioannis D. Schizas. Bayesian sheaf neural networks. arXiv preprint arXiv:2410.09590 , 2024.
- Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In Int. Conf. Mach. Learn. , pages 1263-1272. PMLR, 2017.
- Lorenzo Giusti, Claudio Battiloro, Lucia Testa, Paolo Di Lorenzo, Stefania Sardellitti, and Sergio Barbarossa. Cell attention networks. In 2023 International Joint Conference on Neural Networks (IJCNN) , pages 1-8. IEEE, 2023.
- Joseph A. Goguen. Sheaf semantics for concurrent interacting objects. Mathematical Structures in Computer Science , 2(2):159-191, 1992.
- Christopher Wei Jin Goh, Cristian Bodnar, and Pietro Lio. Simplicial attention networks. In ICLR 2022 Workshop on Geometrical and Topological Representation Learning , 2022.
- Mustafa Hajij, Kyle Istvan, and Ghada Zamzmi. Cell complex neural networks. NeurIPS 2020 Workshop TDA and Beyond , 2020.

- Mustafa Hajij, Karthikeyan Natesan Ramamurthy, Aldo Saenz, and Ghada Zamzmi. High skip networks: a higher order generalization of skip connections. In ICLR 2022 Workshop on Geometrical and Topological Representation Learning , 2022.
- Mustafa Hajij, Ghada Zamzmi, Theodore Papamarkou, Aldo Guzmán-Sáenz, Tolga Birdal, and Michael T Schaub. Combinatorial complexes: bridging the gap between cell complexes and hypergraphs. arXiv preprint arXiv:2312.09504 , 2023a.
- Mustafa Hajij, Ghada Zamzmi, Theodore Papamarkou, Nina Miolane, Aldo Guzmán-Sáenz, Karthikeyan Natesan Ramamurthy, Tolga Birdal, Tamal K. Dey, Soham Mukherjee, Shreyas N. Samaga, Neal Livesay, Robin Walters, Paul Rosen, and Michael T. Schaub. Topological deep learning: going beyond graph data. arXiv , 2023b.
- Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. Adv. Neural Inform. Process. Syst. , 30, 2017.
- Jakob Hansen and Thomas Gebhart. Sheaf neural networks. In Workshop on Topological Data Analysis and Beyond , 2020.
- Jakob Hansen and Robert Ghrist. Learning sheaf laplacians from smooth signals. In ICASSP 20192019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 5446-5450. IEEE, 2019a.
- Jakob Hansen and Robert Ghrist. Toward a spectral theory of cellular sheaves. Journal of Applied and Computational Topology , 3(4):315-358, 2019b.
- Mikhail Hayhoe, Hans Riess, Victor M Preciado, and Alejandro Ribeiro. Stable and transferable hyper-graph neural networks. arXiv preprint arXiv:2211.06513 , 2022.
- Ferran Hernandez Caralt, Guillermo Bernárdez Gil, Iulia Duta, Pietro Liò, and Eduard Alarcón Cot. Joint diffusion processes as an inductive bias in sheaf neural networks. In Proceedings of the Geometry-grounded Representation Learning and Generative Modeling Workshop (GRaM) , volume 251 of Proceedings of Machine Learning Research , pages 249-263. PMLR, 29 Jul 2024.
- Yiming Huang and Tolga Birdal. Hog-diff: Higher-order guided diffusion for graph generation. arXiv preprint arXiv:2502.04308 , 2025.
- Jianwen Jiang, Yuxuan Wei, Yifan Feng, Jingxuan Cao, and Yue Gao. Dynamic hypergraph neural networks. In IJCAI , pages 2635-2641, 2019.
- Eun-Sol Kim, Woo Young Kang, Kyoung-Woon On, Yu-Jung Heo, and Byoung-Tak Zhang. Hypergraph attention networks for multimodal learning. In IEEE Conf. Comput. Vis. Pattern Recog. , pages 14581-14590, 2020.
- Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations , 2017.
- Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. ImageNet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems , pages 1097-1105, 2012.
- Henry Kvinge, Brett Jefferson, Cliff Joslyn, and Emilie Purvine. Sheaves as a framework for understanding and interpreting model fit. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 4222-4230, 2021.
- Zewen Li, Fan Liu, Wenjie Yang, Shouheng Peng, and Jun Zhou. A survey of convolutional neural networks: analysis, applications, and prospects. IEEE transactions on neural networks and learning systems , 33(12):6999-7019, 2021.
- Wenfei Liang, Yanan Zhao, Rui She, Yiming Li, and Wee Peng Tay. Fedsheafhn: Personalized federated learning on graph-structured data. CoRR , abs/2405.16056, 2024.
- Edward C. Mitchell, Brittany Story, David Boothe, Piotr J. Franaszczuk, and Vasileios Maroulas. A topological deep learning framework for neural spike decoding. Biophysical Journal , 2024. doi: https://doi.org/10.1016/j.bpj.2024.01.025.

- Bao Nguyen, Lorenzo Sani, Xinchi Qiu, Pietro Liò, and Nicholas D. Lane. Sheaf hypernetworks for personalized federated learning. 2024. arXiv:2405.20882.
- Theodore Papamarkou, Tolga Birdal, Michael M Bronstein, Gunnar E Carlsson, Justin Curry, Yue Gao, Mustafa Hajij, Roland Kwitt, Pietro Lio, Paolo Di Lorenzo, et al. Position: Topological deep learning is the new frontier for relational learning. In International Conference on Machine Learning , pages 39529-39555. PMLR, 2024.
- M. Papillon, M. Hajij, A. Myers, F. Frantzen, G. Zamzmi, H. Jenne, J. Mathe, J. Hoppe, M. Schaub, T. Papamarkou, A. Guzmán-Sáenz, B. Rieck, N. Livesay, T. Dey, A. Rabinowitz, A. Brent, A. Salatiello, A. Nikitin, A. Zia, C. Battiloro, D. Gavrilev, G. Bökman, G. Magai, G. Bazhenov, G. Bernardez, I. Spinelli, J. Agerberg, K. Nadimpalli, L. Telyatninkov, L. Scofano, L. Testa, M. Lecha, M. Yang, M. Hassanin, O. H. Gardaa, O. Zaghen, P. Hausner, P. Snopoff, P. Melnyk, R. Ballester, S. Barikbin, S. Escalera, S. Fiorellino, H. Kvinge, J. Meissner, K. N. Ramamurthy, M. Scholkemper, P. Rosen, R. Walters, S. N. Samaga, S. Mukherjee, S. Sanborn, T. Emerson, T. Doster, T. Birdal, V. Grande, A. Khamis, S. Scardapane, S. Singh, T. Malygina, Y. Yue, and N. Miolane. ICML 2023 topological deep learning challenge: design and results. In Proceedings of 2nd Annual Workshop on Topology, Algebra, and Geometry in Machine Learning (TAG-ML) , volume 221, pages 3-8. PMLR, 2023.
- Robert Peach, Matteo Vinao-Carl, Nir Grossman, Michael David, Emma Mallas, David J Sharp, Paresh A Malhotra, Pierre Vandergheynst, and Adam Gosztolai. Implicit gaussian process representation of vector fields over arbitrary latent manifolds. In The Twelfth International Conference on Learning Representations .
- Antonio Purificato, Giulia Cassarà, Federico Siciliano, Pietro Liò, and Fabrizio Silvestri. Sheaf4rec: Sheaf neural networks for graph-based recommender systems. ACMTransactions on Recommender Systems , 2023.
- Michael Robinson. Topological signal processing , volume 81. Springer, 2014.
- Amit Singer and H-T Wu. Vector diffusion maps and the connection laplacian. Communications on pure and applied mathematics , 65(8):1067-1144, 2012.
- Yellamraju V. Srinivas. A sheaf-theoretic approach to pattern matching and related problems. Theoretical Computer Science , 112(1):53-97, 1993.
- Otto Sumray, Heather A Harrington, and Vidit Nanda. Quiver laplacians and feature selection. arXiv preprint arXiv:2404.06993 , 2024.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- Petar Veliˇ ckovi´ c. Message passing all the way up. ICLR 2022 Workshop on Geometrical and Topological Representation Learning , 2022.
- Petar Veliˇ ckovi´ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. In International Conference on Learning Representations , 2018.
- Arne Wolf and Anthea Monod. Topological community detection: A sheaf-theoretic approach. In International Conference on Complex Networks and Their Applications , pages 29-42. Springer, 2023.
- Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations , 2019.

## Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework -Supplementary Material-

## Table of Contents

| A   | Notation                                                             |   16 |
|-----|----------------------------------------------------------------------|------|
| B   | Sheaves and Copresheaves on Graphs: A Category Theoretical Look      |   16 |
| C   | Copresheaf Laplacian, Energy, and CTNN Transport-diffusion           |   21 |
| D   | Expressive Power of CTNNs                                            |   22 |
| E   | Sheaf Neural Networks Are Copresheaf Message-Passing Neural Networks |   23 |
| F   | A General Copresheaf-Based Transformer Layer                         |   25 |
| G   | Copresheaf Learning on Euclidean Data                                |   26 |
| H   | Experiments                                                          |   27 |
| I   | Extended Related Work on Topological and Sheaf Neural Networks       |   37 |

## A Notation

We provide a reference summary of the notation and acronyms used throughout the main text. Table 5 details key mathematical symbols, while Table 6 lists abbreviations and their expansions.

Table 5: Summary of key notation used throughout the paper.

̸

| Notation S X P ( S ) rk : X → Z ≥ 0 Z ≥ 0 X k dim X N : X →P ( X ) N adj N inc X N G ∈ { 0 , 1 } m × n G N ρ y → x A r,k B r,k C k ( X , F ) Hom( F ( i ) , F ( j )) h ( ℓ ) x α β ⊕ G = ( V,E ) G N = ( X N ,E N ) F ( x ) , F ( e ) F x ≤ e : F ( x ) →F   | Description Underlying vertex set of a combinatorial complex (Def. 1) Set of nonempty cells ⊆ P ( S ) (Def. 1) Power set of S (Def. 1) Rank function on cells, mapping to non-negative integers (Def. 1) Non-negative integers { x ∈ X : rk( x ) = k } , the k -cells (Def. 1) Dimension of the complex, max x ∈X rk( x ) (Sec. 2.1) Neighborhood function, mapping cells to sets of neighbor cells (Def. Adjacency neighborhood function (Def. 2) Incidence neighborhood function (Def. 2) Effective support of N , { x ∈ X &#124; N ( x ) = ∅} (Sec. 3) Binary neighborhood matrix (Def. 3) Copresheaf neighborhood matrix, entries ρ z i → y j or 0 (Def. 7) Copresheaf morphism F ( y ) →F ( x ) for edge y → x (Def. 4) Copresheaf adjacency matrix between r -cells (Def. 8) Copresheaf incidence matrix between ranks r and k (Def. 8) k -cochain space, ⊕ x ∈X k F ( x ) (Sec. 3) Space of linear maps from F ( i ) to F ( j ) (Def. 7) Feature vector at cell x in layer ℓ (Prop. 1) Learnable message function (Prop. 1) Learnable update function (Prop. 1) Permutation-invariant aggregator (Prop. 1, Prop. 3) Directed or undirected graph (Sec. 2.2) Directed graph induced by N , edges y → x if y ∈ N ( x ) (Def. 6) Stalks: vector spaces at vertex x or edge e (Def. 4, Def. 5) Restriction map in a cellular sheaf for x ≤ e (Def. 5)   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## B Sheaves and Copresheaves on Graphs: A Category Theoretical Look

This appendix provides a category-theoretic exposition of sheaves and copresheaves, emphasizing their definitions within the language of category theory. Additionally, we illustrate the construction of copresheaf neighborhood matrices through explicit combinatorial examples, instantiating the concepts developed in the main text.

## B.1 Copresheaves

Before diving into the technical definition, it is helpful to think of a copresheaf as a way of assigning data that flows along the structure of a graph, like signals along neurons, or resources in a network. In categorical terms, this structure formalizes the idea of consistently associating elements of some category C .

Definition 12 (Copresheaf on a directed graph) . Let G = ( V, E ) be a directed graph, and let C be a category. A copresheaf on G is a functor F : G →C , where the graph G is regarded as a category whose objects are the vertices V , and whose morphisms are the directed edges ( x → y ) ∈ E . When C = Vect R , this structure corresponds to a quiver representation .

Table 6: List of acronyms used throughout the paper.

| Acronym                                                                                                                    | Expansion                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CC CTNN CMPNN SNN GNN CNN CT CGNN GCN GraphSAGE GIN CopresheafGCN CopresheafSage CopresheafGIN NSD SAN MLP GAT CAM CIM CNM | Combinatorial Complex Copresheaf Topological Neural Network Copresheaf Message-Passing Neural Network Sheaf Neural Network Graph Neural Network Convolutional Neural Network Copresheaf Transformer Copresheaf Graph Neural Network Graph Convolutional Network Graph Sample and Aggregate Graph Isomorphism Network Copresheaf Graph Convolutional Network Copresheaf Graph Sample and Aggregate Copresheaf Graph Isomorphism Network Neural Sheaf Diffusion Sheaf Attention Network Multi-Layer Perceptron Graph Attention Network Copresheaf Adjacency Matrix Copresheaf Incidence Matrix Copresheaf Neighborhood Matrix |

## B.2 Cellular Sheaves

Definition 13 (Cellular sheaf on an undirected graph) . Let G = ( V, E ) be an undirected graph. Define the incidence poset ( P, ≤ ) , where P = V ∪ E , and the order relation is given by x ≤ e whenever vertex x ∈ V is incident to edge e ∈ E . A cellular sheaf on G with values in a category C is a functor F : P →C , which assigns:

- to each vertex x ∈ V , an object F ( x ) ∈ C ;
- to each edge e ∈ E , an object F ( e ) ∈ C ;
- to each incidence relation x ≤ e , a morphism F x ≤ e : F ( x ) →F ( e ) , called a restriction map , such that the functoriality condition is satisfied on composable chains in the poset.

## B.3 Comparison Between Copresheaves and Cellular Sheaves

Copresheaves provide a versatile and powerful framework for machine learning applications across diverse domains, particularly excelling in scenarios where directional data flow and hierarchical dependencies are paramount. Unlike cellular sheaves, which are defined over undirected graphs and enforce consistency through restriction maps F x ≤ e : F ( x ) →F ( e ) , copresheaves operate on directed graphs, assigning learnable linear maps ρ x → y : F ( x ) →F ( y ) along edges. This enables anisotropic information propagation, making them ideal for tasks such as physical simulations, where data flows asymmetrically, as in fluid dynamics or heat transfer, or natural language processing, where sequential word dependencies dominate. More importantly, the vertex-centric design, assigning vector spaces F ( x ) solely to vertices, aligns seamlessly with message-passing architectures like Graph Neural Networks (GNNs) and Topological Neural Networks (TNNs), allowing these maps to be parameterized and optimized during training. Empirical evidence from our experiments highlights that copresheaf-based models outperform traditional architectures in capturing complex dynamics, demonstrating their superior ability to model spatially varying patterns and long-range dependencies in general applications. See Table 7 for a summary of the comparison between copresheaves and cellular sheaves.

Furthermore, copresheaves enhance machine learning models with a principled approach to regularization and expressiveness, broadening their suitability across heterogeneous domains. Standard neural network regularizers, such as ℓ 2 decay or dropout, can be readily applied to copresheaf maps,

with optional structural losses like path-consistency ensuring morphism compositionality. This adaptability stands in stark contrast to the rigid cohomological constraints of cellular sheaves, which enforce local-to-global agreement through terms like ∥F x ≤ e ( h x ) -h e ∥ 2 , limiting their flexibility in domains with asymmetric relationships. Copresheaves, by learning edge-wise maps, offer greater expressiveness for tasks involving non-Euclidean or multi-scale data, as evidenced by the superior performance of CopresheafConv layers on grid-based tasks. This makes them particularly effective for applications such as image segmentation, 3D mesh processing, or token-relation learning, where traditional methods like Convolutional Neural Networks (CNNs) struggle with directional or hierarchical structures. Our experiments further corroborate that copresheaf-augmented models consistently improve accuracy and detail recovery across diverse tasks, positioning them as a more suitable and generalizable tool for machine learning applications spanning Euclidean and non-Euclidean domains alike.

Table 7: Comparison between copresheaves and cellular sheaves.

| Aspect                 | Copresheaf                                                                                                                   | Cellular sheaf [Hansen and Ghrist, 2019b]                                                         |
|------------------------|------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Graph type             | Directed graph G = ( V,E )                                                                                                   | Undirected graph G = ( V,E )                                                                      |
| Assigned to vertices   | Vector space F ( x ) for each x ∈ V                                                                                          | Vector space F ( x ) for each x ∈ V                                                               |
| Assigned to edges      | Linear map ρ x → y : F ( x ) → F ( y ) for each di- rected edge x → y ∈ E                                                    | Vector space F ( e ) for each edge e ∈ E                                                          |
| Associated maps        | Pushforward: moves data forward along edges                                                                                  | Restriction: pulls data back from vertices to edges                                               |
| Map direction          | F ( x ) →F ( y ) (source to target)                                                                                          | F ( x ) →F ( e ) (vertex to incident edge)                                                        |
| Interpretation         | Nodes have local features; edges transform and transmit them                                                                 | Edges represent shared contexts; vertex features are restricted into them                         |
| Goal / Objective       | Learn and compose edge-wise feature-space maps                                                                               | Enforce coherence across by gluing local data                                                     |
| Typical Regularization | Standard NN regularizers ( ℓ 2 decay, spectral- norm, dropout, norm); optional structure losses (path-consistency, holonomy) | Agreement between restricted vertex features and the edge-stalk, e.g., ∥F x ≤ e ( h x ) - h e ∥ 2 |
| Use in learning        | Embedding-level message passing, directional in- fluence, anisotropic information flow                                       | Compatibility across shared structures, enforcing local consistency, cohomological constraints    |

## B.4 Copresheaf Neighborhood Matrix Example

Example 1. Setup . Let the combinatorial complex X = ( S , X , rk) have symbols S = { a, b, c, d } and cells

<!-- formula-not-decoded -->

Ranks satisfy rk( X 0 ) = 0 , rk( X 1 ) = 1 , rk( X 2 ) = 2 . Geometrically, this is the union of two triangles ( a, b, c ) and ( d, b, c ) sharing the edge { b, c } .

Let E = [ e 1 , e 2 , e 3 , e 4 , e 5 ] with

<!-- formula-not-decoded -->

Edge-via-face neighborhood . Define N △ : X 1 →P ( X 1 ) by

<!-- formula-not-decoded -->

̸

Thus, two edges are neighbors iff they both bound the same 2-cell. Concretely,

<!-- formula-not-decoded -->

The effective support is X N = X 1 (edges only).

Induced directed graph . The induced directed graph G N △ = ( V N , E N ) has vertices V N = X 1 and directed edges

<!-- formula-not-decoded -->

Messages flow from an edge e ′ to an adjacent edge e whenever both bound a common triangle.

Copresheaf on the edge-adjacency poset . Assign to every edge a feature space F ( e ) = R 2 . For each directed adjacency e ′ → e , attach a linear map ρ e ′ → e : R 2 → R 2 . To keep notation compact,

<!-- formula-not-decoded -->

we write diagonal maps as diag( α, β ) and Id 2 for the 2 × 2 identity:

<!-- formula-not-decoded -->

(Any consistent choice works; the point is a map per directed adjacency e ′ → e .)

Copresheaf Neighborhood Matrix (CNM) . For the ordering E = [ e 1 , e 2 , e 3 , e 4 , e 5 ] , the CNM G N △ ∈ ( R 2 × 2 ) 5 × 5 has block entries

<!-- formula-not-decoded -->

Displayed explicity as:

<!-- formula-not-decoded -->

See Figure 3.

<!-- image -->

Figure 3: (a) A combinatorial complex X = ( S , X , rk) with S = { a, b, c, d } , edges X 1 = {{ a, b } , { b, c } , { c, a } , { d, b } , { c, d }} , and faces X 2 = {{ a, b, c } , { d, b, c }} . (b) Induced edge-adjacency digraph: nodes represent the edges of X and the edges represent the face adjacencies. (c) Copresheaf neighborhood matrix G N △ .

This CNM performs edge-to-edge directional message passing along face-adjacency: each edge e i aggregates transformed features from its face-adjacent neighbors e j via the maps ρ e j → e i . The shared edge naturally becomes a high-degree conduit between the two triangles.

Example 2. We define a copresheaf neighborhood matrix (CNM) for a combinatorial complex with an incidence neighborhood, guiding the reader through the setup, neighborhood, graph, copresheaf, and matrix.

Setup . Consider a combinatorial complex X = ( S , X , rk) with S = { a, b, c } , cells X = {{ a } , { b } , { c } , { a, b } , { b, c }} , and ranks rk( { a } ) = rk( { b } ) = rk( { c } ) = 0 , rk( { a, b } ) = rk( { b, c } ) = 1 . Thus, X 0 = {{ a } , { b } , { c }} , X 1 = {{ a, b } , { b, c }} . See Figure 4.

Incidence neighborhood . The incidence neighborhood N inc : X → P ( X ) is:

<!-- formula-not-decoded -->

For 0-cells: N inc ( { a } ) = {{ a, b }} , N inc ( { b } ) = {{ a, b } , { b, c }} , N inc ( { c } ) = {{ b, c }} . For 1-cells: N inc ( { a, b } ) = N inc ( { b, c } ) = ∅ . The effective support is X N = {{ a } , { b } , { c }} .

Induced graph . We induce a directed graph G N = ( V N , E N ) , with:

<!-- formula-not-decoded -->

<!-- image -->

Figure 4: (a) A combinatorial complex X = ( S , X , rk) with S = { a, b, c } , cells X = {{ a } , { b } , { c } , { a, b } , { b, c }} . The figure also indicates the induced directed graph G N inc = ( V N , E N ) from the incidence neighborhood structure on the combinatorial complex X . Each arrow z → y represents a directed edge from a 1-cell to a 0-cell where y ⊂ z , and is associated with a linear map ρ z → y as part of the copresheaf. (b) The induced directed graph G N inc = ( V N , E N ) from the incidence neighborhood structure N inc. (c) The copresheaf neighborhood matrix (CNM) G N inc , where rows are indexed by 0-cells { a } , { b } , { c } and columns by 1-cells { a, b } , { b, c } . The matrix entries are linear maps ρ z → y when z ∈ N inc ( y ) , and 0 otherwise. This matrix supports directional feature propagation from 1-cells to 0-cells.

Copresheaf . The N inc-dependent copresheaf assigns F ( x ) = R 2 to each x ∈ V N , and linear maps ρ y → x : R 2 → R 2 for y → x ∈ E N :

<!-- formula-not-decoded -->

Neighborhood matrix . The CNM G N for Y = X 0 , Z = X 1 is:

<!-- formula-not-decoded -->

For Y = {{ a } , { b } , { c }} , Z = {{ a, b } , { b, c }} :

<!-- formula-not-decoded -->

This matrix facilitates message passing from 1-cells to 0-cells, e.g., bond-to-atom feature propagation.

More generally, Figure 5 shows an illustrative example of the general setup of copresheaf higher-order message passing on a CC with multiple neighborhood functions.

Figure 5: Illustration of copresheaf higher-order message passing. Left-hand side: Shows a central cell x (as a circle) with arrows pointing to boxes labeled N 1 ( x ) , N 2 ( x ) , . . . , N k ( x ) , representing the collection of neighborhood functions N = {N k } n k =1 . Right-hand side: Depicts the messagepassing process for the same cell x . Each neighborhood N k ( x ) produces an aggregated message using ⊕ y ∈N k ( x ) α N k ( ρ y → x ( h ( ℓ ) y )) . These messages are then combined using the inter-neighborhood function β , shown as a box, with an arrow updating x .

<!-- image -->

## C Copresheaf Laplacian, Energy, and CTNN Transport-diffusion

In this section we introduce a linear transport-discrepancy operator B ρ that maps node fields to edge discrepancies, define the quadratic energy E ρ and the associated Laplacian L ρ = B ⊤ ρ WB ρ , derive its block form and kernel, and show that a CTNN residual layer implements an explicit gradient step that monotonically decreases E ρ for suitable step sizes. This energy perspective provides an interpretable and theoretical foundation for the CTNN architecture, framing it as a diffusion process that minimizes a specific energy function. Our copresheaf Laplacian also coincides with the quiver Laplacian [Sumray et al., 2024] when viewing the transport maps as arrow representations, connecting our framework to established spectral methods while enabling directional, anisotropic information flow distinct from the symmetric diffusion in sheaf-based models Bodnar et al. [2022].

Spaces and operator . Let ( F, ρ, G ) be a copresheaf defined on a directed graph G = ( V, E ) . To each node x ∈ V attach a finite-dimensional real inner-product space F ( x ) . For every edge y → x ∈ E fix a linear transport ρ y → x : F ( y ) → F ( x ) . Define the node and edge product spaces

<!-- formula-not-decoded -->

equipped with canonical (blockwise) inner products ⟨ h , ˜ h ⟩ V = ∑ x ∈ V ⟨ h x , ˜ h x ⟩ and ⟨ ξ , ˜ ξ ⟩ E = ∑ ( y → x ) ∈ E ⟨ ξ y → x , ˜ ξ y → x ⟩ .

Definition 14 (Transport-discrepancy operator) . The linear operator

<!-- formula-not-decoded -->

is defined componentwise by

<!-- formula-not-decoded -->

Let B ⊤ ρ : H E →H V denote the Euclidean adjoint. A direct calculation yields

<!-- formula-not-decoded -->

Definition 15 ( Energy and copresheaf Laplacian ) . Let w y → x &gt; 0 be edge weights and define the diagonal operator W : H E →H E by ( W ξ ) y → x = w y → x ξ y → x . The weighted transport energy and the copresheaf Laplacian are

<!-- formula-not-decoded -->

Remark 2 (Relation to quiver Laplacian) . Interpreting the transports ρ y → x as arrow maps of a quiver representation, the copresheaf Laplacian L ρ in Equation 10 coincides with the standard (weighted) quiver Laplacian B ⊤ WB of that representation. See Sumray et al. [2024] for more about the quiver Laplacian.

It is sometimes useful to have the copresheaf Laplacian in its exact nodewise form. Namely, for every outgoing edge x → z with map ρ x → z and weight w x → z , include the reverse edge z → x with ρ z → x := ρ ⊤ x → z and w z → x := w x → z . With this convention one has the nodewise form of the copresheaf Laplacian:

<!-- formula-not-decoded -->

Theorem 2. With w y → x &gt; 0 , L ρ = B ⊤ ρ WB ρ : H V →H V is symmetric positive semidefinite and h ⊤ L ρ h = ∥ B ρ h ∥ 2 W ≥ 0 . Its block action at node x is

<!-- formula-not-decoded -->

Moreover,

<!-- formula-not-decoded -->

Proof. P.s.d. and the quadratic identity follow from L ρ = B ⊤ ρ WB ρ with W ≻ 0 . The block formula follows by expanding B ⊤ ρ ( WB ρ h ) via 10. Finally, ∥ B ρ h ∥ 2 W = 0 ⇐⇒ B ρ h = 0 .

## C.1 CTNN/CMPNN residual as copresheaf diffusion.

From the copresheaf message passing equation in Definition 9, choose the edge message α ( h x , ρ y → x h y ) = h x -ρ y → x h y , sum aggregation, and residual update β ( h x , m ) = h x -ηm . Then a CTNN/CMPNN layer updates

<!-- formula-not-decoded -->

We have the following theorem.

Theorem 3. For E ρ in (13) ,

<!-- formula-not-decoded -->

Hence (14) is an explicit gradient step on 1 2 E ρ . If 0 &lt; η &lt; ∥ L ρ ∥ -1 2 , then E ρ ( h ( ℓ +1) ) ≤ E ρ ( h ( ℓ ) ) , with strict inequality whenever h ( ℓ ) / ∈ ker L ρ . Moreover, I -ηL ρ is non-expansive in ∥ · ∥ 2 and strictly contractive on ker L ⊥ ρ , so h ( ℓ ) converges to the orthogonal projection of h (0) onto ker L ρ .

̸

Proof. Since E ρ ( h ) = ( B ρ h ) ⊤ W ( B ρ h ) , the chain rule gives ∇ E ρ ( h ) = 2 B ⊤ ρ WB ρ h = 2 L ρ h , so (14) is gradient descent on 1 2 E ρ . Let L ρ = U Λ U ⊤ with Λ = diag( λ i ≥ 0) and write h ( ℓ ) = U c ( ℓ ) . Then c ( ℓ +1) i = (1 -ηλ i ) c ( ℓ ) i and E ρ ( h ( ℓ ) ) = ∑ i λ i ( c ( ℓ ) i ) 2 . If 0 ≤ ηλ i ≤ 1 for all i , then λ i (1 -ηλ i ) 2 ≤ λ i , giving monotone decay, strict if some λ i &gt; 0 has c ( ℓ ) i = 0 . Non-expansiveness and convergence follow from | 1 -ηλ i | ≤ 1 (and &lt; 1 for λ i &gt; 0 ).

Remark 3. ρ y → x may be (i) direct per-edge linear maps, (ii) factored as ρ y → x = F ⊤ x F y to reduce parameters, or (iii) constrained (e.g. softly orthogonal via ∥ ρ ⊤ y → x ρ y → x -I ∥ 2 F ) to regularize the spectrum of L ρ . A learnable (possibly per-layer/head) stepsize η , normalized by a running estimate of ∥ L ρ ∥ -1 2 , enforces the energy-decay guarantee in Theorem 3. When ρ y → x = I , (13) reduces to the (vector-valued) graph Laplacian; orthogonal ρ recovers a connection Laplacian.

## D Expressive Power of CTNNs

## D.1 Universal Approximation of N -Dependent Copresheaves

Here, we demonstrate that multilayer perceptrons (MLPs) can approximate arbitrary copresheaf morphisms induced by neighborhood functions. This result ensures that the proposed sheaf-based model is sufficiently expressive to capture complex data interactions.

Proposition 2 (Universal approximation of N -dependent copresheaves) . Let X be a finite combinatorial complex and N a neighborhood function on X . Suppose G N -→ Vect R is an N -dependent copresheaf with stalks F N ( x ) = R d and morphisms ρ N y → x . Let a feature map

<!-- formula-not-decoded -->

be given so that the 2 d -dimensional vectors ( h y , h x ) are pairwise distinct for every directed edge y → x . Define

<!-- formula-not-decoded -->

Then for any ε &gt; 0 there exists a multilayer perceptron Φ: R 2 d → R d × d with sufficiently many hidden units such that ∥ ∥ Φ( h y , h x ) -ρ N y → x ∥ ∥ &lt; ε for all y → x.

Proof. Since A is finite and its elements are distinct, the assignment g : A → R d × d is well-defined.

Enumerate A = { a i } i ∈ I , choose disjoint open neighborhoods U i ∋ a i , and pick smooth 'bump' functions

Then the sum

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is a smooth map f : R 2 d → R d × d satisfying f | A = g .

Since A is finite, choose a compact set K ⊆ R 2 d containing A , ensuring the applicability of the Universal Approximation Theorem. By that theorem, for any ε &gt; 0 there is an MLP Φ such that

<!-- formula-not-decoded -->

In particular, for each a i ∈ A we have

<!-- formula-not-decoded -->

for every y → x . This completes the proof.

## E Sheaf Neural Networks Are Copresheaf Message-Passing Neural Networks

The computational use of a cellular sheaf on a graph rests on the incidence poset . Let G = ( V, E ) and consider the poset on V ∪ E with x ⪯ e whenever x ∈ e . A cellular sheaf F assigns a vector space to each cell and a linear structure map to each incidence. In our convention, the 'restriction' along x ⪯ e is implemented as a vertex-to-edge lift F x ⊴ e : F ( x ) →F ( e ) . Equipping F ( e ) with an inner product yields the adjoint F ⊤ x ⊴ e : F ( e ) →F ( x ) , a canonical edge-to-vertex back-projection. Message passing between adjacent vertices then arises by composing incidence maps: for e = { x, y } , the message from y to x is

<!-- formula-not-decoded -->

This composition induces a direction of information flow on an undirected graph while remaining faithful to the sheaf poset structure. Diffusion-style updates and sheaf-Laplacian operators are recovered by aggregating such edge-mediated messages over N ( x ) . See Figure 6 for an illustration.

Figure 6: Sheaf-induced message passing on an edge e = { x, y } . (a) Local spaces F ( x ) , F ( y ) , F ( e ) with vertex-to-edge lifts F x ⊴ e and F y ⊴ e along the incidences x ⪯ e , y ⪯ e . (b) Incidence-poset view: arrows encode the sheaf's linear maps attached to x ⪯ e and y ⪯ e . (c) Edge-mediated message passing: the inner product on F ( e ) yields adjoints F ⊤ x ⊴ e . The message from y to x is the composition F ⊤ x ⊴ e ◦F y ⊴ e , i.e., a direct vertex-to-vertex morphism ρ y → x : F ( y ) →F ( x ) , compatible with the bidirected expansion G ′ . This realizes SNN message passing as CMPNN morphisms assembled from sheaf structure maps.

<!-- image -->

In this appendix, we prove Theorem 1 and Proposition 1, which demonstrate that existing sheaf neural networks (SNNs), including sheaf diffusion networks, are special cases of the copresheaf message-passing neural network (CMPNN) framework (Definition 9). Moreover, we summarize how other sheaf-based neural architectures align with our unifying message-passing framework (see Table 8).

Proof of Theorem 1. We construct the bidirected graph G ′ and define the copresheaf G on it, then demonstrate the equivalence of the message-passing operations.

First, construct the bidirected graph G ′ = ( V, E ′ ) from G = ( V, E ) by replacing each undirected edge { x, y } ∈ E with two directed edges ( x, y ) , ( y, x ) ∈ E ′ . This ensures that G ′ retains the connectivity of G while introducing explicit directionality.

Next, define the copresheaf G : G ′ → Vect R as follows:

- For each vertex x ∈ V , assign G ( x ) = F ( x ) , where F ( x ) is the vector space associated with x by the cellular sheaf F .
- For each directed edge ( y, x ) ∈ E ′ , corresponding to the undirected edge e = { x, y } ∈ E , define the linear morphism ρ y → x : G ( y ) →G ( x ) by ρ y → x = F ⊤ x◁e ◦ F y◁e , where F y◁e : F ( y ) →F ( e )

and F x◁e : F ( x ) →F ( e ) are the restriction maps of F , and F ⊤ x◁e : F ( e ) →F ( x ) is the adjoint with respect to an inner product on F ( e ) .

Now, consider the SNN message-passing mechanism along the edge e = { x, y } . For a feature vector h y ∈ F ( y ) at vertex y , the message transmitted to vertex x is given by F ⊤ x◁e ◦ F y◁e ( h y ) .

In the copresheaf G on G ′ , the morphism associated with the directed edge ( y, x ) is ρ y → x = F ⊤ x◁e ◦ F y◁e . Applying this morphism, the message-passing operation in the CMPNN yields ρ y → x ( h y ) = F ⊤ x◁e ◦ F y◁e ( h y ) , which is identical to the SNN message.

To ensure that G is a well-defined copresheaf, observe that it assigns vector spaces to vertices and linear maps to directed edges in a functorial manner. Specifically, for each directed edge ( y, x ) ∈ E ′ , the map ρ y → x is a composition of linear maps and thus linear. The identity and composition properties are satisfied implicitly through the consistency of the sheaf restriction maps.

Therefore, the SNN message passing, which operates via intermediate edge spaces in F , is equivalently represented as direct vertex-to-vertex message passing in the copresheaf G on G ′ . This completes the proof.

Proof of Proposition 1. Compute:

<!-- formula-not-decoded -->

since L F,x,y = 0 for y / ∈ N ( x ) ∪ { x } .

Interpret G as a directed graph with edges y → x for y ∈ N ( x ) and x → x . Define:

- Message function: α ( h x , ρ y → x h y ) = W 2 L F,x,y W 1 h y ,
- Morphisms: ρ y → x implicitly encoded via L F,x,y ,
- Aggregator: ⊕ = ∑ ,
- Update: β ( h x , m ) = h x -m .

Thus:

matching Definition 9.

Table 8 provides a summary of sheaf neural networks realized in terms of Definition 9.

We finally prove the following theorem to show the relationship more precisely between CTNNs, MPNNs and SNNs.

Theorem 4 (CTNNs strictly subsume SNNs and contain MPNNs) . Let F CTNN , F SNN , and F MPNN denote the function classes realized by Copresheaf Topological Neural Networks (CTNNs), Sheaf Neural Networks (SNNs), and Message-Passing Neural Networks (MPNNs), respectively. Then

<!-- formula-not-decoded -->

Proof. (1) F SNN ⊂ F CTNN . First from Theorem 1 we know F SNN ⊆ F CTNN . To prove the strict containment, fix { u, v } ∈ E and let F ( u ) = F ( v ) = R 2 . Define a CTNN on G ′ by

<!-- formula-not-decoded -->

No SNN can realize this, since SNN transports necessarily reciprocate across an undirected edge:

<!-- formula-not-decoded -->

Thus ρ u → v = 0 would force ρ v → u = 0 , contradicting ρ v → u = I 2 . Therefore F SNN ⊂ F CTNN is strict.

<!-- formula-not-decoded -->

Table 8: Unified message passing formulations of various sheaf neural networks using our copresheaf topological neural network (CTNN) notation given in Proposition 10. The restriction maps ρ y → x may be linear, data-driven, or attentional depending on the model.

| Method (Paper)                                                          | Message Passing Equation                                                                         | Notable Features                                                                                                                            | Restriction Map ρ y → x                                                              |
|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Sheaf Neural Network (SNN) Hansen &Gebhart (2020)                       | h ( l +1) x = σ   h ( l ) x + ∑ y ∈N ( x ) ρ y → x h ( l ) y                                 | Linear restriction maps ρ y → x assigned per edge; enables high-dimensional, direction-aware message passing via sheaf structure.           | ρ y → x = F ⊤ x◁e F y◁e , fixed linear map, e = { x, y }                             |
| Neural Sheaf Diffusion (NSD) Bodnar et al. (2022)                       | h ( l +1) x = h ( l ) x - σ   ∑ x◁e ρ x → x h ( l ) x - ∑ y ∈N ( x ) ρ y → x h ( l ) y       | Diffusion over learned sheaf Laplacian; restriction maps ρ y → x are learnable, re- flecting edge-mediated interactions.                    | ρ y → x = F ⊤ x◁e F y◁e , learned lin- ear map, e = { x, y }                         |
| Sheaf Attention Network (SAN) Barbero et al. (2022)                     | h ( l +1) x = σ   ∑ y ∈N ( x ) α xy ( h x , h y ) ρ y → x h ( l ) y                          | Attentional sheaf: attention weights α xy modulate the restricted neighbor feature; mitigates oversmoothing in GAT-style setups.            | ρ y → x : learned linear map, pa- rameterized to capture feature space relationships |
| Connection Laplacian SNN Barbero et al. (2022)                          | h ( l +1) x = σ   h ( l ) x + ∑ y ∈N ( x ) O xy h ( l ) y                                    | Edge maps O xy are orthonormal, de- rived from feature space alignment; re- duces learnable parameters and reflects local geometric priors. | O xy : orthonormal matrix, learned to align feature spaces across edges              |
| Heterogeneous Sheaf Neural Network (HetSheaf) Braithwaite et al. (2024) | h ( l +1) x = σ   h ( l ) x + ∑ y ∈N ( x ) ρ y → x ( h x , h y ) h ( l ) y                   | Type-aware sheaf morphisms: ρ y → x depend on node and edge types, en- abling structured heterogeneity across the graph.                    | ρ y → x : type-aware learned lin- ear map, parameterized by node and edge types      |
| Adaptive Sheaf Diffusion Zaghen et al. (2024)                           | h ( l +1) x = h ( l ) x + σ   ∑ y ∈N ( x ) ρ y → x ( h x , h y ) ( h ( l ) y - h ( l ) x )   | Nonlinear Laplacian-like dynamics with adaptive, feature-aware restriction maps ρ y → x ; enhances expressiveness and lo- cality.           | ρ y → x : feature-aware learned linear map, parameterized by node features           |

(2) F MPNN ⊆ F CTNN . Let an MPNN layer on a directed graph have the form

<!-- formula-not-decoded -->

for a permutation-invariant aggregator □ . The results follows immediately from Definition 1.

## F A General Copresheaf-Based Transformer Layer

The main idea to introduce a copresheaf structure to the transformer is the following. For every ordered pair y → x (within an attention head) we define a parametrized copresheaf map

<!-- formula-not-decoded -->

which transports the value vector from the stalk at y to the stalk at x . Given attention weights α xy = softmax y ∈N ( x ) (( q ⊤ x k y ) / √ d ) , the head message is m x = ∑ y ∈N ( x ) α xy ρ y → x v y .

In Definition 11 we introduced the notion of copresheaf self-attention. A natural extension of this definition is the copresheaf cross-attention.

Definition 16 (Copresheaf cross-attention) . For source rank k s and target rank k t , with neighborhood N s → t , define learnable projection matrices W s → t q ∈ R p × d t , W s → t k ∈ R p × d s , W s → t v ∈ R d s × d t where d s and d t are the feature dimensions of source and target cells, respectively. We then propose the copresheaf cross-attention as the aggregation and update h ( ℓ +1) x = β ( h ( ℓ ) x , m x ) where m x = ∑ y ∈N s → t ( x ) a xy ρ y → x ( v y ) with ρ y → x : F ( y ) →F ( x ) being a learned map and

<!-- formula-not-decoded -->

where k y = W s → t k h ( ℓ ) y , v y = W s → t v h ( ℓ ) y and for each target cell x ∈ X k t , q x = W s → t q h ( ℓ ) x .

Figure 7 illustrates the cross attention in the copresheaf transformer.

In Algorithm 1, we provide the pseudocode for our generic copresheaf-based transformer layer. This algorithm outlines the layer-wise update rule combining self-attention within cells of equal rank and cross-attention between different ranks, using learned copresheaf morphisms to transfer features between stalks. It generalizes standard transformer mechanisms by introducing neighborhooddependent transformations.

<!-- image -->

Figure 7: Copresheaf cross-attention. (a) A target cell x (yellow) in a combinatorial complex with its neighborhood N ( x ) (red). Sources may be at a different rank than x (e.g., faces → vertex). (b) Cross-attention schematic: each neighbor y ∈ N ( x ) contributes a value v y that is first transported into the target's local feature space via a learned map ρ y → x : F ( y ) → F ( x ) ; attention weights a xy are computed from q x and k y . (c) Implementation view: for every y , apply ρ y → x to v y , scale by a xy , and sum to form the message m x = ∑ y ∈N ( x ) a xy ρ y → x ( v y ) , which updates h x . Transporting values with ρ enables directional, cross-rank, and anisotropic information flow beyond standard attention.

<!-- image -->

## G Copresheaf Learning on Euclidean Data

The CopresheafConv layer leverages copresheaf structures to process data on a D -dimensional grid X ⊂ Z D , offering distinct advantages over traditional convolutional neural networks (CNNs). By defining a copresheaf on a combinatorial complex (CC) constructed from the grid, where cells represent grid points (0-cells) and their pairwise connections (1-cells), the layer employs learnable morphisms ρ y → x : F ( y ) →F ( x ) that dynamically adapt to directional relationships between points. Unlike static convolutional filters, these morphisms capture anisotropic, directionally dependent interactions, preserving topological nuances of the grid's geometry. In contrast, regular convolutional kernels enforce translation invariance, limiting their ability to model spatially varying or directional

patterns. The copresheaf is defined over an adjacency neighborhood function N adj ( x ) = { y ∈ X | { x, y } ∈ X 1 } , restricting computation to local, grid-adjacent neighbors, thus ensuring efficiency comparable to CNNs. The morphisms, potentially nonlinear, are conditioned on input features h ( ℓ ) x , h ( ℓ ) y and grid positions, enabling the layer to model complex, multi-scale dependencies. This makes CopresheafConv ideal for tasks like image segmentation, 3D mesh processing, or geometric deep learning, where local and hierarchical relationships are critical. Empirical results demonstrate superior performance in capturing physical dynamics, showcasing the ability of CopresheafConv to handle spatially varying patterns. Algorithm 2 shows the pseudocode for the CopresheafConv used in our experiments.

## CopresheafConv on a D -dimensional grid

Algorithm 2: CopresheafConv on a D -dimensional grid.

<!-- image -->

## H Experiments

## H.1 Mechanistic Notes for Physics Experiments

Scope. The empirical results for advection and unsteady Stokes appear in the main text (Sec. 6.1; Tab. 2). This appendix augments those experiments with architectural rationale and ablations-informed design choices, without introducing any new datasets, training budgets, or evaluation metrics.

Advection (pure transport) . The advection equation

<!-- formula-not-decoded -->

is a rigid translation. Classical self-attention aggregates values in a single global latent space, so x ↔ y interactions are effectively symmetric and only weakly directional (positional encodings help but cannot enforce upwind behavior). The copresheaf transformer (CT) replaces value mixing

<!-- formula-not-decoded -->

̸

with a learnable edge map ρ y → x : F ( y ) → F ( x ) before aggregation. This yields: (i) Directionality ( ρ y → x = ρ x → y ) for upwind-like asymmetry, (ii) Phase-faithful shifts (identity-near, head-wise maps accumulate small signed translations coherently), (iii) Path compositionality : products of ρ along y → x chains bias the model toward consistent transports.

Unsteady Stokes (incompressible viscous flow) . For

<!-- formula-not-decoded -->

accuracy depends on encoding (i) anisotropic diffusion and (ii) rotational structure (vorticity), under a divergence-free constraint. Standard attention lacks built-in diffusion geometry or frame

alignment. CT's edge maps ρ y → x act as local linear operators that: (i) Align with diffusion tensors: SPD/orthonormal variants (Table 18) mimic smoothing/rotation along principal directions. (ii) Support structured coupling: With cross-rank paths, CT allows vertex-edge/value transports akin to discrete parallel transport and pressure-velocity interactions, without imposing global sheaf consistency.

Take-away . Across both tasks, CT's gains come from a geometric factorization of attention into

<!-- formula-not-decoded -->

not merely extra parameters. This aligns with the CTNN principle that heterogeneous stalks + edge-specific morphisms provide natural inductive biases for transport-dominated and anisotropic dynamics.

## H.2 Synthetic Control Tasks

Six canonical univariate time-series patterns ( normal, cyclic, increasing trend, decreasing trend, upward shift, downward shift ) are procedurally generated. We obtain 600 sequences of length 60 (100 per class), normalised to the interval [ -1 , 1] , with an 80:20 split for training and test.

Models and set-up . A lightweight vanilla Transformer (32-d model, 4 heads, 2 layers) Table 9: Synthetic control: mean ± std over 3 runs.

| Model                  | Max acc. (%)     |
|------------------------|------------------|
| Standard Transformer   | 98 . 61 ± 0 . 40 |
| Copresheaf Transformer | 99 . 44 ± 0 . 39 |

is compared with an identically sized Copresheaf Transformer, where multi-head attention is replaced by a gated outer-product tensor-attention layer with orthogonality ( λ = 0 . 01 ) and sparsity ( λ = 10 -4 ) regularisers. Both models share sinusoidal-withlinear-decay positional encodings, use Adam ( 10 -3 learning rate), batch 32, train for 15 epochs, and each experiment is repeated with three random seeds.

Results . As seen in Table 9, the Copresheaf Transformer yields a consistent improvement of +0 . 8 -1 . 0 percentage points (pp) over the vanilla Transformer while remaining lightweight and training in comparable wall-clock time (under one minute per run on a single GPU), highlighting the benefit of richer token-pair transformations for recognition tasks.

## H.2.1 Structure Recognition Datasets

In this experiment we consider two synthetic image datasets containing oriented ellipses or hierarchical triangles.

Dataset (oriented ellipses) . Each 32 × 32 RGB image contains a single black ellipse on a white background. The horizontal and vertical semi-axes a, b are drawn uniformly from { 4 , 5 , . . . , 12 } pixels, and the ellipse is rotated by a random angle in [0 , 180 ◦ ) . The task is to predict the coarse orientation bin ( 4 bins of 45 ◦ each). We synthesise 6 , 000 images, keep 5 , 000:1 , 000 for train/validation, and rescale pixels to [ -1 , 1] .

Dataset (hierarchical triangles) . Each 32 × 32 RGB image contains six coloured circlesred, green, blue, yellow, cyan, magenta -placed on two nested equilateral triangles (inner radius 8 px, outer 12 px). Colours are randomly permuted. A hand-crafted hierarchy of linear maps (inner-triangle, outer-triangle, cross-level) is applied to the circles' one-hot colour vectors; the image is labelled 1 when the resulting scalar exceeds a fixed threshold, else 0 . We generate 6 , 000 images and keep 5 , 000:1 , 000 for train/validation.

Models and set-up . Both tasks use the same compact Vision-Transformer backbone: 32dim patch embeddings (patch size 8 ), 4 heads, 2 layers, learnable positional embeddings,

AdamW ( 3 × 10 -4 learning rate). The baseline is a Regular ViT ; its counterpart is an identically sized Copresheaf ViT in which multi-head attention is replaced by an outer-product copresheaf mechanism (stalk-dim = 8 ). Oriented Ellipses is trained with batch 128 ; Hierarchical Tri-

Table 10: Validation accuracy on both synthetic vision tasks (mean ± std over three seeds).

| Dataset              | Regular ViT      | Copresheaf ViT   |
|----------------------|------------------|------------------|
| Oriented Ellipses    | 84 . 13 ± 4 . 12 | 96 . 23 ± 0 . 33 |
| Hierarchical Triang. | 95 . 47 ± 1 . 31 | 96 . 87 ± 0 . 26 |

angles with batch 64 . All runs use 30 epochs and three independent seeds.

Results . Across both synthetic vision tasks the Copresheaf ViT consistently surpasses the Regular ViT: a dramatic +12 . 1 pp gain on Oriented Ellipses and a subtler yet statistically tighter +1 . 4 pp on Hierarchical Triangles, while also cutting variance by an order of magnitude in the latter case (see Table 10). These outcomes underscore that replacing standard attention with copresheaf-guided outer-product maps yields robust improvements for both low-level geometric orientation recognition and higher-level nested-structure reasoning, all without a significant increase of model size or training budget.

## H.2.2 Classifying Hierarchical Polygons

Similar to the previous section, we now synthesize a hierarchy of nested regular polygons. In particular, each 32 × 32 RGB image contains a variable number n ∈ { 6 , 8 , 10 } of coloured circles arranged on two nested regular polygons (inner radius 8 px, outer 12 px). The first n/ 2 circles form the inner polygon, the remainder the outer; colours are drawn from a palette of n distinct hues and randomised per sample. A hierarchy of hand-crafted linear maps is applied to one-hot colour vectors: pairwise maps on the inner polygon ( F inner), on the outer polygon ( G outer), and a cross-level map H . The image is labelled 1 when the resulting scalar exceeds a threshold, else 0 . For each n we synthesise 6 , 000 images, keep 5 , 000:1 , 000 for train/validation, and normalise pixels to [ -1 , 1] .

Models and training . We reuse the compact ViT backbone (32-dim patch embeddings, patch size 8, 4 heads, 2 layers, learnable positional embeddings). The Regular ViT is compared with an identically sized Copresheaf ViT , which replaces multi-head attention with rank-restricted copresheaf outer-product maps (stalk-dim = 8). Both networks are trained for 10 epochs with AdamW ( 3 × 10 -4 learning rate), batch 64; each configuration is run three times.

Results . Figure 8 shows that the Copresheaf ViT consistently at n =6 ( 0.72 vs 0.66) and regains a clear lead at n =10 ( 0.63 vs 0.57) despite both models dipping at n =8 . The Copresheaf curve displays narrower uncertainty bands at the hardest setting, indicating greater run-to-run stability. Overall, copresheaf-guided attention scales more gracefully with combinatorial complexity, capturing cross-level dependencies that standard self-attention struggles to model.

## H.2.3 Airfoil Self-Noise Regression

The UCI airfoil dataset ( 1 503 rows) maps five continuous descriptors-frequency, angle of attack, chord length, free-stream velocity, Reynolds number-to the sound-pressure level (dB). Inputs and target are min-max scaled to [0 , 1] ; we keep only 400:100 train/test samples for a low-data setting.

exceeds the

Regular

ViT

Figure 8: Validation accuracy (mean ± 1 s.d., 3 runs) as task difficulty increases.

<!-- image -->

Models . Both regressors share a minimalist backbone consisting sequentially of the following: 64-d token embedding 2-layer, 4-head transformer, mean pooling, scalar head. The copresheaf variant swaps dot-product attention for learned outer-product maps ρ ij that depend on each token pair, whereas the Regular baseline keeps standard self-attention. Training uses Adam ( 10 -4 learning rate), 1000 epochs, batch 32.

Results . On the small 100-sample test set the copresheaf regressor lowers MSE by 7.2% relative to the regular transformer and maintains sub-

10 -4 run-to-run variance (see Table 11), confirming that pair-specific linear transports help model heterogeneous feature interactions even in data-scarce regimes.

Table 11: Test MSE (mean ± std over two runs).

| Model               | MSE             |
|---------------------|-----------------|
| Regular Transformer | 0.0223 ± 0.0001 |
| Copresheaf Transf.  | 0.0208 ± 0.0002 |

## H.3 Pixelwise Regression Tasks: Evaluating CopresheafConv2D Layers

We evaluate neural network models incorporating CopresheafConv2D layers, custom convolutional layers with patch-wise trainable linear morphisms, against standard convolutional models across four synthetic pixelwise regression tasks: PDE regression (Bratu and convection-diffusion equations), image denoising, distance transform regression, and edge enhancement. In all tasks, Copresheaf-based

models consistently achieve lower Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) compared to standard convolutional models, suggesting improved modeling of spatial structures and relationships.

## Task Definitions .

- PDE regression .
- -Bratu equation . A nonlinear reaction-diffusion PDE:

<!-- formula-not-decoded -->

where g ( x, y ) is a source intensity.

- -Convection-diffusion equation . A transport PDE:

<!-- formula-not-decoded -->

with diffusion ν and velocities c x , c y .

- Image denoising . Recovering clean structured images (sinusoidal patterns with a Gaussian bump, normalized to [0,1]) from Gaussian noise ( σ = 0 . 3 ).
- Distance transform regression . Predicting the normalized Euclidean distance transform of a binary segmentation (thresholded at 0.5) of structured images.
- Edge enhancement . Predicting edge maps from structured images using a difference-of-anisotropicGaussians (DoG) transformation.

Model and training setup . For PDE regression and distance transform tasks, we use U-Net variants: CopresheafUNet (with CopresheafConv2D layers) and ConvUNet (with standard Conv2d layers), both with a four-level backbone (64 → 128 → 256 → 512 channels). For image denoising and edge enhancement, we use four-layer convolutional networks: CopresheafNet and ConvNet (1 → 8 → 16 → 8 → 1 channels). All models are trained on 64 × 64 inputs using the Adam optimizer and MSE loss, with task-specific settings (learning rates 10 -3 or 10 -4 , batch sizes 8 or 16, 80-300 epochs). Results are averaged over 3 random seeds.

Table 12: Mean ( ± std over 3 seeds) of MSE and RMSE across all tasks.

| Task                 | Model                   | MSE                                       | RMSE                                    |
|----------------------|-------------------------|-------------------------------------------|-----------------------------------------|
| Bratu Equation       | CopresheafUNet ConvUNet | 0 . 0001 ± 0 . 00020 0 . 0003 ± 0 . 00020 | 0 . 0108 ± 0 . 0003 0 . 0183 ± 0 . 0007 |
| Convection-Diffusion | CopresheafUNet ConvUNet | 0 . 0004 ± 0 . 00010 0 . 0006 ± 0 . 00020 | 0 . 0205 ± 0 . 0010 0 . 0232 ± 0 . 0012 |
| Image Denoising      | CopresheafNet ConvNet   | 0 . 0010 ± 0 . 00010 0 . 0011 ± 0 . 00020 | 0 . 0310 ± 0 . 0010 0 . 0336 ± 0 . 0015 |
| Distance Transform   | CopresheafUNet ConvUNet | 0 . 0001 ± 0 . 00002 0 . 0002 ± 0 . 00003 | 0 . 0105 ± 0 . 0003 0 . 0156 ± 0 . 0005 |
| Edge Enhancement     | CopresheafNet ConvNet   | 0 . 0008 ± 0 . 00010 0 . 0009 ± 0 . 00020 | 0 . 0283 ± 0 . 0010 0 . 0300 ± 0 . 0015 |

Take-away . Across all tasks, replacing standard convolutional layers with CopresheafConv2D layers results in lower MSE and RMSE (see Table 12). This consistent improvement suggests that patch-wise linear maps enhance the models' ability to capture complex spatial patterns. These findings highlight the potential of Copresheaf-based architectures for pixelwise regression problems. Subsequently, we address two challenges related to token classification: real/fake token sequence detection and segment-wise token classification. Finally, we conduct a preliminary study on shape classification using copresheaf-augmented attention and graph classification a molecular benchmark, MUTAG. These are followed by applications in graph connectivity classification and text classification on TREC coarse label benchmark.

## H.3.1 Learning Token-Relations with Copresheaf Attention

We study five problems that differ only in the (non)linear operator unknown applied to the first half of a random token sequence (or to a second related sequence). The classifier must decide whether the tail is just a noisy copy (label 0) or a transformed version of the head (label 1).

- Orthogonal block . Eight 16 -d 'head' tokens are either copied ( +0 . 05 noise) or rotated by a sample-specific orthogonal matrix before adding the same noise.

<!-- image -->

0.0

0.0

Figure 9: Model outputs across tasks. A: Bratu equation: input g , target u , CopresheafUNet vs. ConvUNet predictions. B: Convection-diffusion: input g , target u , CopresheafUNet vs. ConvUNet predictions. C: Distance transform: input image, target transform, CopresheafUNet vs. ConvUNet predictions. D: Image denoising: noisy input, clean target, CopresheafNet vs. ConvNet predictions. E: Edge enhancement: input image, target edge map, CopresheafNet vs. ConvNet predictions. Copresheaf-based models show subtle improvements in detail recovery.

- Per-token scaling . As above, but the tail is α i x i + noise with α i ∼ U[0 . 4 , 1 . 6] .
- Rotated copy (embedded 2-D) . Six 2-D points are mapped to 16 d by a fixed linear embed, duplicated to a 12-token sequence; the tail is either a noisy copy or the points after a random planar rotation.
- Query and context linearity . Two parallel sequences ( 50 × 16 'query', 50 × 24 'context'). Class 0: context is a global affine transform of the query with partly correlated semantics. Class 1: context comes from a quadratic warp and weak semantic correlation.
- Affine vs. quadratic token relations . Two parallel sequences (length-6, query dim 16, context dim 24) are considered. For class 0, the context is a linear spatial transformation (rotation and translation) of the query plus correlated semantic noise. For class 1, the context is generated via a spatial quadratic (nonlinear) transformation with weaker semantic correlation.

Table 13: Mean accuracy ( ± std, 3 seeds).

| Task                           | Classic           | Copresheaf        |
|--------------------------------|-------------------|-------------------|
| Orthogonal block               | 0 . 732 ± 0 . 009 | 0 . 928 ± 0 . 007 |
| Per-token scaling              | 0 . 521 ± 0 . 005 | 0 . 707 ± 0 . 004 |
| Rotated copy (2-D)             | 0 . 739 ± 0 . 010 | 0 . 896 ± 0 . 033 |
| Query to context               | 0 . 608 ± 0 . 046 | 0 . 992 ± 0 . 012 |
| Affine vs. Quadratic Relations | 0 . 588 ± 0 . 047 | 0 . 900 ± 0 . 027 |

Data . Tasks 1-2 use 16 tokens, task 3 uses 12, task 4 uses two length-50 sequences, task 5 uses two length-6 sequences. For each of three seeds we draw 4 , 096:1 , 024 train/test sequences (task 4-5: 320:80 ).

Backbone and training . A tiny Transformer encoder (4 heads, token dim 16, stalk-dim 4) → mean-pool → 2-way classifier. Classic uses vanilla dot-product attention; copresheaf augments it with learned token-pair copresheaf maps (we chose General Copresheaf for the first two tasks, and Non-linear MLP for tasks 3-5). We train for 8 / 12 / 10 / 10 / 10 epochs respectively with Adam ( 10 -3 learning rate), batch 64.

Take-away . Across in-sequence, element-wise, embedded-geometric, and cross-sequence settings-including varying degrees of spatial and semantic correlation-injecting copresheaf transports into self-attention consistently lifts accuracy significantly (up to +38 pp, as seen in Table 13). This highlights a general principle: tasks whose signals reside in relations between tokens rather than in absolute token content strongly benefit from explicitly modeling these relations through learnable copresheaf-induced attention.

Limited attention capacity . We study the impact of attention capacity on relational reasoning, by varying the number of heads in a small transformer and testing its ability to classify the query to context dataset provided in Section H.3.1.

We evaluate three setups: a baseline transformer, the same model augmented with positional encoding (PE), and a copresheaf-augmented transformer with 2 heads. Figure 10 shows accuracy as a function of attention capacity.

While positional encoding improves baseline accuracy slightly, the copresheaf-augmented attention with just two heads outperforms all classic models, even those with eight times more heads. This highlights the value of inductive relational structure over brute-force capacity scaling.

## H.3.2 Segment-wise Token Classification

We test whether copresheaf attention improves token-level classification in a sequence partitioned into contiguous segments with distinct patterns. The classifier must assign a segment label (0, 1, or 2) to each token based on its local context.

Data . Each input is a sequence of 100 tokens, where each token is a 16-dimensional feature vector. The sequence is divided into three contiguous segments, each following a different pattern: (i) a cosine oscillation, (ii) a linearly increasing ramp, or (iii) an exponentially decreasing signal, with additive noise. The task is to predict the correct segment label for each token. We generate 300 training sequences and evaluate on three random seeds.

Backbone and training . We use a 2-layer encoder with 4 heads, token dim 16, and stalk-dim 4. A linear classifier maps each token to one of 3 segment labels. Classic is standard attention; copresheaf augments each attention head with learned per-token transport maps. We train for 10 epochs using Adam ( 10 -3 learning rate), with batch size 32.

Take-away. Injecting copresheaf structure into attention substantially improves token-wise

Figure 10: Accuracy as a function of number of attention heads. Even with low capacity, the copresheafaugmented model perfect generalization of the Query to Context task.

<!-- image -->

segmentation accuracy, especially with expressive MLP kernels (see Table 14). This demonstrates that local consistency constraints enforced by per-token transport maps help resolve semantic boundaries even in noisy, positionally ambiguous settings.

## H.3.3 Topological Shape Classification

We evaluate copresheaf-augmented attention on a synthetic 3D point cloud classification task. Each

Table 14: Mean segmentation accuracy ( ± std, 3 seeds).

| Model          | Accuracy      |
|----------------|---------------|
| Classic        | 0.705 ± 0.010 |
| Copresheaf-FC  | 0.833 ± 0.015 |
| Copresheaf-MLP | 0.831 ± 0.007 |
| Copresheaf-SPD | 0.743 ± 0.017 |

input is a set of 128 points in R 3 sampled from one of four classes: cube, sphere, torus, and twisted torus. Rotations are applied to remove alignment bias.

Data . The dataset consists of 480:160 train/test samples, balanced across the four classes. Each point cloud is processed as a sequence of 128 points with 3D coordinates.

Backbone and training . Both models use a 4-layer point transformer with 4 heads and head dimension 32. The Classic model uses standard self-attention. The Copresheaf model augments attention with diagonal copresheaf morphisms. We train each model for 50 epochs using AdamW and cosine learning rate decay across 3 random seeds.

Take-away . Copresheaf-augmented attention improves accuracy on 3D shape classification by Table 15: Mean accuracy ( ± std, 3 seeds). enhancing sensitivity to latent geometric structure (see Table 15).

## H.4 TREC Text Classification Task

We evaluate two transformer-based models on the TRECcoarse-label question classification task, which involves categorizing questions into 6 classes (e.g., abbreviation, entity, description). The models are:

| Model      | Accuracy          |
|------------|-------------------|
| Classic    | 0 . 708 ± 0 . 031 |
| Copresheaf | 0 . 746 ± 0 . 034 |

Classic , a standard transformer with multi-head self-attention; and Copresheaf-FC , which incorporates a GeneralSheafLearner to model stalk transformations.

Task definition . The TREC dataset consists of questions labeled with one of 6 coarse categories. Inputs are tokenized questions truncated to 16 tokens, mapped to a vocabulary of size |V| , and embedded into an 8-dimensional space. The task is to predict the correct class label for each question.

Model and training setup . Both models use a single transformer block with 2 attention heads, an embedding dimension of 8, and a stalk dimension of 4 for the Copresheaf-based model.

The architecture includes an embedding layer, a transformer block with attention and feed-forward components, adaptive average pooling, and a linear classifier. Compared to state-of-the-art (SOTA) models, which often employ multiple transformer layers and high-dimensional embeddings for maximal performance, our networks are intentionally small, using a single block and low embedding dimension to prioritize computational efficiency and controlled experimentation. We train on the TREC training set (5452 samples) over 30

Table 16: Mean ( ± std over 4 seeds) of test accuracy for the TREC classification task.

| Model         | Test Accuracy       |
|---------------|---------------------|
| Classic       | 0 . 7320 ± 0 . 0080 |
| Copresheaf-FC | 0 . 7500 ± 0 . 0150 |

epochs with a batch size of 32, using the Adam optimizer with a learning rate of 10 -3 and crossentropy loss. The test set (500 samples) is used for evaluation. Each experiment is repeated over 4 random seeds. As seen in Table 16, the copresheaf models outperform their SOTA counterparts.

## H.5 Mixed Dirichlet-Robin Reaction-Anisotropic Diffusion on Cellular Complexes

Problem. . Let Ω ⊂ R 2 be a Lipschitz domain with a hole (nonconvex), and let ∂ Ω = ∂ Ω D ∪ ∂ Ω R with ∂ Ω D ∩ ∂ Ω R = ∅ . We consider the steady reaction-anisotropic diffusion equation

<!-- formula-not-decoded -->

with mixed boundary conditions

<!-- formula-not-decoded -->

Here K ( x ) ∈ R 2 × 2 is a symmetric positive definite anisotropy tensor (we store its local components as ( K xx , K xy , K yy ) ), λ, σ ≥ 0 are reaction weights, g is a source, α, r are Robin coefficients, u B a boundary signal, and n the outward unit normal. All spatial fields are discretized and provided as vertex/edge attributes on the mesh.

Discretization on simplicial complexes. . Let K = ( S, X , rk) be a 2D simplicial complex (triangulation) of Ω \ H with 0 -, 1 -, and 2 -cells X 0 , X 1 , X 2 . We employ the neighborhood functions (Def. 2)

<!-- formula-not-decoded -->

and the induced directed graphs G N . Edgewise anisotropy enters through per-edge tensors ( K xx , K xy , K yy ) and yields a stiffness-like coupling consistent with equation 13. Vertex features carry ( u B , g, λ, σ, α, r ) ; edge features carry ( K xx , K xy , K yy ) . Masks on ∂ Ω D and ∂ Ω R enforce the boundary conditions in the loss.

Higher-order copresheaf structure. . We equip G N with an N -dependent copresheaf ( F , ρ, G N ) over multiple neighborhoods:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus information flows across ranks (edges/triangles → vertices) in addition to within rank (vertex ↔ vertex). This realizes a higher-order CMPNN and may be viewed as a principled generalization of topological neural networks (TNNs), where transport maps are learned as copresheaf morphisms instead of being tied to fixed co(boundary) operators.

Copresheaf Cellular Transformer layer. . For features h ( ℓ ) x ∈ F ( x ) at cell x in layer ℓ , we use copresheaf self- and cross-attention :

<!-- formula-not-decoded -->

where q x = W q h ( ℓ ) x , k x = W k h ( ℓ ) x , v x = W v h ( ℓ ) x , the coefficients a xy = softmax y ∈ N ( x ) ( ⟨ q x , k y ⟩ / √ p ) , and β is a residual MLP with normalization. We instantiate ρ with sheaf-style transport maps SheafFC ρ y → x = Id + tanh ( W [ q x ; k y ] ) (zero-initialized), optionally constrained to SPD via ρ = Id + QQ ⊤ . All morphisms act per head and per rank; weights are shared across complexes but conditioned on ( h x , h y ) , enabling directionality and anisotropy. Observe that when the copresheaf structure is the identity, we retain the Cellular Transfomer introduced in Barsbey et al. [2025].

Dataset and training . We synthesize simplicial complexes by jittered grids with warped holes to emulate nontrivial ∂ Ω R . Per-sample scalar fields ( u B , g, λ, σ, α, r ) are sampled on vertices; ( K xx , K xy , K yy ) on edges. Ground-truth u is obtained by a finite-element solve of equation 13equation 14. We train to regress ˆ u at vertices with MSE, enforcing Dirichlet via clamped targets and Robin via boundary-weighted residuals. Optimization uses AdamW ( 10 -3 learning rate), cosine schedule, grad clip 0 . 8 , layer norm, and vertex/edge feature normalization computed per split.

Baselines and size control . We compare the Cellular Transformer on vertices (same depth/width, no copresheaf maps) to our Copresheaf Cellular Transformer on K using identical token dimensions and heads; differences arise solely from the learnable morphisms and cross-rank attention paths.

Results . Quantitative test MSE averaged across n =4 seeds is summarized below (lower is better).

Error maps highlight that copresheaf cross-rank transport (1 → 0 and 2 → 0) attenuates bias along anisotropy directions of K and reduces leakage across ∂ Ω R . Despite identical model size, higherorder morphisms recover boundary layers and interior ridges more faithfully. See Figure 11 for a sample of the results.

Table 17: Mixed Dirichlet-Robin reaction-anisotropic diffusion on simplicial complexes. Test MSE (mean ± std over n =4 seeds). Lower is better.

| Model                                      | Test MSE ↓          |   Runs ( n ) |
|--------------------------------------------|---------------------|--------------|
| Cellular Transformer Barsbey et al. [2025] | 0 . 3277 ± 0 . 0408 |            4 |
| Copresheaf Cellular Transformer            | 0 . 3172 ± 0 . 0365 |            4 |

Take-away . Endowing the simplicial complex with a higher-order copresheaf, learning ρ y → x across adjacency and incidence neighborhoods and across ranks, yields a CMPNN that generalizes TNNs while remaining faithful to the PDE structure in equation 13. Observe that such a network is a generalization of the cellular transformer introduced in Barsbey et al. [2025]. The learned, directional transport improves anisotropic coupling and mixed-boundary handling without increasing parameter count, offering a principled and practical route to physics-aware attention on combinatorial domains.

Figure 11: Mixed Dirichlet-Robin reaction-anisotropic diffusion on triangulated complexes. For each example (rows), the left block shows a 3 × 3 tile of normalized inputs: Dirichlet values u B , source g , reaction λ , spatial reaction field σ , Robin coefficients α and r , and per-vertex edge-averaged conductivity features ( K xx , K xy , K yy ) . The remaining panels show the ground-truth solution u , the Cellular Transformer prediction, and the Copresheaf Cellular Transformer prediction, respectively. Rows illustrate diverse geometries and boundary layouts. Copresheaf transport produces slightly crisper fields and cleaner boundary behavior.

<!-- image -->

## H.6 Catalogue of Copresheaf Maps

We also provide the table of copresheaf maps that we used throughout our experiments in Table 18.

## Computational Complexity

This section analyzes the computational complexity of the Copresheaf Message-Passing Neural Network (CMPNN) and the Copresheaf Transformer (CT), as defined in the framework of Copresheaf

Table 18: Catalogue of copresheaf maps ρ y → x used in our training our copresheaf transformer model. All maps act stalk-wise and are evaluated independently for each attention head; σ is the logistic function.

| Map family          | Copresheaf map ρ y → x (per head)                                    | Learnable params                   |
|---------------------|----------------------------------------------------------------------|------------------------------------|
| General Copresheaf  | ρ y → x = tanh ( W [ q x k y ] )                                     | W ∈ R 2 d × d 2                    |
| Pre-Linear Map      | ρ y → x = q x k ⊤ y                                                  | none                               |
| Diagonal MLP Map    | ρ y → x = diag ( σ (MLP[ q x ,k y ]) )                               | 2-layer MLP                        |
| Graph Attention Map | ρ y → x = σ (MLP[ q x ,k y ]) I d                                    | 2-layer MLP                        |
| Vision Spatial Map  | ρ y → x = σ (MLP( p x - p y )) ( p x , p y ∈ [0 , 1] 2 pixel coords) | 2-layer MLP                        |
| Outer-Product Map   | ρ y → x = W q q x ( W k k y ) ⊤                                      | W q ,W k ∈ R d × d                 |
| Non-linear MLP Map  | ρ y → x = reshape ( MLP[ q x ,k y ] )                                | 2-layer MLP (2 d → 2 d → d 2 )     |
| Gaussian RBF Map    | ρ y → x = e -∥ q x - k y ∥ 2 / 2 σ 2 I d                             | σ (scalar)                         |
| Dynamic Map         | ρ y → x = reshape( W f q x )                                         | W f ∈ R d × d 2                    |
| Bilinear Map        | ρ y → x = ( b ⊤ ( q x ,k y )) I d                                    | b ∈ R d × d                        |
| SheafFC Map         | ρ y → x = I d +tanh ( W q x k y )                                    | W ∈ R 2 d × d 2 (zero init)        |
| SheafMLP Map        | ρ y → x = I d +tanh ( MLP[ q x ,k y ] )                              | 2-layer MLP (last layer zero init) |
| SheafSPD Map        | ρ y → x = I d + QQ ⊤ , Q = W q x k y                                 | W ∈ R 2 d × d 2 (no bias)          |

Topological Neural Networks (CTNNs). We consider a directed graph G N = ( V N , E N ) , induced by a combinatorial complex X with neighborhood function N , where | V N | = n , | E N | = m , and the average degree is c = m/n . Each vertex x ∈ V N is assigned a feature space F ( x ) = R d , and each edge y → x ∈ E N has a linear map ρ y → x : R d → R d , computed at runtime using a map family from Table 16 (e.g., General Copresheaf, Low-rank). The complexity of computing a single morphism ρ y → x is denoted C ( ρ ) .

Unless otherwise stated, we assume sparse graphs (i.e., m = Θ( n ) ). We derive the following propositions for per-layer complexities, followed by a comparison with standard architectures.

Proposition (Copresheaf Message-Passing Complexity). . Consider a directed graph G N = ( V N , E N ) induced by a combinatorial complex X with neighborhood function N , where | V N | = n , | E N | = m , dim F ( x ) = d , and let ⊕ be any permutation-invariant aggregator. Let each y → x ∈ E N have a linear map ρ y → x ∈ R d × d computed at runtime via a map family with complexity C ( ρ ) . The per-layer computational complexity of the CMPNN, as defined in Definition 9, is

<!-- formula-not-decoded -->

Proof. The CMPNN message-passing operation involves computing morphisms, applying them to features, aggregating messages, and updating vertex states:

1. Morphism Computation: For each edge y → x ∈ E N , compute ρ y → x ∈ R d × d based on source and target features ( h ( ℓ ) y , h ( ℓ ) x ) ∈ R 2 d , using a map family from Table 18. This costs O ( mC ( ρ )) .
2. Morphism Application: Apply ρ y → x to the feature vector h ( ℓ ) y ∈ R d . This matrix-vector multiplication requires O ( d 2 ) operations per edge, totaling O ( md 2 ) .
3. Aggregation: For each vertex x ∈ V N , aggregate messages from neighbors y ∈ N ( x ) via ⊕ y ∈N ( x ) ρ y → x h ( ℓ ) y . Each addition involves a R d vector, costing O ( d ) per neighbor. With |N ( x ) | neighbors, this step is O ( |N ( x ) |· d ) . Summing over all vertices: O ( ∑ x ∈ V N |N ( x ) |· d ) = O ( m · d ) .

4. Update: Assign h ( ℓ +1) x = β ( · ) with negligible cost. If an optional single-layer MLP with d hidden units is applied (common in GNNs), the cost per vertex is O ( d 2 ) , yielding O ( n · d 2 ) .

Total: O ( nC ( ρ ) + md 2 + md + nd 2 ) . Since md 2 dominates and m = Θ( n ) , this simplifies to O ( nC ( ρ ) + ( m + n ) d 2 ) .

Proposition (Copresheaf Transformer Complexity). . Consider a directed graph G N = ( V N , E N ) induced by a combinatorial complex X with neighborhood function N , where | V N | = n , | E N | = m , and the average degree is c = m/n . Each vertex x ∈ V N has a feature space F ( x ) = R d , and each of H attention heads computes a morphism ρ ( h ) y → x ∈ R ( d/H ) × ( d/H ) via a map family (Table 16) with complexity C ( ρ ) . The per-layer computational complexity of the Copresheaf Transformer, as defined in Algorithm 1, with sparse cross-attention based on N , is

<!-- formula-not-decoded -->

Proof. The Copresheaf Transformer integrates self-attention within cells of equal rank and sparse cross-attention between neighbors, using learned copresheaf morphisms. Each head attends on features (queries/keys) of dimension p = d/H :

1. Morphism Construction: For each edge y → x ∈ E N and each head, compute ρ ( h ) y → x ∈ R ( d/H ) × ( d/H ) based on ( h ( ℓ ) y , h ( ℓ ) x ) . The complexity per morphism is C ( ρ ) , totaling O ( HmC ( ρ )) .
2. Self- and Cross-Attention: Compute query ( Q ), key ( K ), and value ( V ) matrices per head ( d/H dimensions), costing O ( nd 2 /H ) across all heads. For sparse cross-attention, each vertex attends to |N ( x ) | neighbors, with attention scores QK ⊤ and aggregation costing O ( |N ( x ) |· d/H ) per vertex per head. Summing over vertices and heads: O ( H · ∑ x |N ( x ) |· d/H ) = O ( md ) .
3. Morphism Application: Apply ρ ( h ) y → x to value vectors ( d/H per head) on each edge, costing O ( d 2 /H ) per head, or O ( d 2 ) across H heads. For m edges this contributes O ( md 2 ) .
4. Output and Feed-Forward: Combine head outputs and apply a per-token feed-forward network (FFN) with d hidden units, costing O ( nd 2 ) .

<!-- formula-not-decoded -->

Comparison to Standard Architectures. . Compared to standard GNNs (e.g., GCN) and dense Transformers, CMPNN and CT incur higher costs due to morphism computation (the C ( ρ ) terms) and morphism application ( md 2 ). On sparse graphs with small H , CMPNN's complexity is O ( nC ( ρ ) + md 2 + nd 2 ) , and CT's is O ( HmC ( ρ ) + md 2 + nd 2 + ndH ) . These costs enable superior expressiveness, as shown empirically.

Table 19: Per-layer computational complexity on sparse graphs. If ρ is diagonal (or rankr ), replace d 2 by d (or r d ) in the morphism-application terms.

| Model                       | Per-layer complexity (sparse graph)   |
|-----------------------------|---------------------------------------|
| GCN                         | O ( md + nd 2 )                       |
| CMPNN                       | O ( nC ( ρ ) + md 2 + nd 2 )          |
| Transformer (Dense)         | O ( n 2 d + nd 2 )                    |
| Copresheaf Transformer (CT) | O ( HmC ( ρ ) + md 2 + nd 2 + ndH )   |

## I Extended Related Work on Topological and Sheaf Neural Networks

Foundations of sheaf theory . Sheaf theory offers a unifying categorical framework across algebraic geometry, topology, and algebra [Bredon, 1997]. Early computer-science applications exploited its logical structure [Fourman et al., 1977, Goguen, 1992], and Srinivas generalized pattern matching via sheaves on Grothendieck topologies [Srinivas, 1993], later extended to NP-hard problems [Conghaile, 2022, Abramsky, 2022]. Cellular sheaves, formalized in [Curry, 2014], underpin discrete topological data analysis and signal processing [Ghrist and Hiraoka, 2011, Robinson, 2014]. Hansen &amp; Ghrist

introduced the sheaf Laplacian [Hansen and Ghrist, 2019b], learnable by convex optimization [Hansen and Ghrist, 2019a]. Connection sheaves model discrete vector bundles [Singer and Wu, 2012] and support manifold learning and Gaussian processes [Peach et al.]. GKM-sheaves further connect equivariant cohomology and sheaf cohomology, enriching this framework with applications to torus actions on CW complexes [Al-Jabea and Baird, 2018].

Higher-order representations in deep learning . The growing interest in higher-order network models [Papamarkou et al., 2024, Battiston et al., 2020, Bick et al., 2023] has catalyzed geometric and topological deep learning. Techniques include simplicial, hypergraph, and cellular messagepassing schemes [Gilmer et al., 2017, Ebli et al., 2020, Hayhoe et al., 2022, Hajij et al., 2020, Bunch et al., 2020], skip connections [Hajij et al., 2022], and convolutional operators [Jiang et al., 2019, Feng et al., 2019]. Recent years also witnessed a leap in higher-order diffusion models for graph generation [Huang and Birdal, 2025] as well as higher-order (cellular) transformers [Ballester et al., 2024].

Sheaf neural networks . In recent years, sheaf-based generalizations of graph neural networks (GNNs) have demonstrated notable improvements on tasks involving heterogeneous or non-Euclidean data. Hansen and Gebhart [Hansen and Gebhart, 2020] first introduced sheaf neural networks (SNNs), which generalize graph neural networks (GNNs) by replacing neighborhood aggregation with learnable linear 'restriction maps', thereby customizing information flow between nodes. By allowing each edge to carry its own linear transformation, SNNs capture relationships in heterophilic graphs more effectively than degree-normalized convolutions. Building on this idea, Bodnar et al. [Bodnar et al., 2022] proposed Neural Sheaf Diffusion (NSD), which jointly learns the underlying sheaf structure and the diffusion dynamics. NSD layers adaptively infer the sheaf Laplacian from data, mitigating the oversmoothing problem common in deep GNNs and achieving superior performance on a range of heterophilic benchmark datasets. Barbero et al. [Barbero et al., 2022b] then combined NSD's principled diffusion with attention mechanisms to formulate Sheaf Attention Networks (SANs). SANs modulate self-attention weights by the learned sheaf maps, preserving long-range dependencies while respecting local sheaf geometry.

Alternative formulations include Bundle Neural Networks (BNNs) by Bamberger et al. [Bamberger et al., 2024], which reinterpret propagation as parallel transport in a flat vector bundle rather than discrete message passing. Duta et al. [Duta et al., 2023] extended sheaf methods to hypergraphs, defining linear and nonlinear hypergraph Laplacians that capture higher-order interactions among groups of nodes. On manifold-structured data, Tangent Bundle Neural Networks (TBNNs) proposed by Battiloro et al. [Battiloro et al., 2024b] treat features as elements of tangent spaces and propagate them along estimated geodesics, bridging continuous and discrete models.

Attention mechanisms in higher-order structures . Attention mechanisms have been generalized to hypergraphs [Kim et al., 2020, Bai et al., 2021] and simplicial complexes [Goh et al., 2022, Battiloro et al., 2024a], among else via Hodge or Dirac operators. On combinatorial complexes, feature-lifting attention facilitates hierarchical information propagation [Giusti et al., 2023, Hajij et al., 2023b].

Applications and extensions . These sheaf-theoretic architectures have found diverse applications, from multi-document summarization [Atri et al., 2023] and recommendation systems [Purificato et al., 2023] to community detection via sheaf cohomology [Wolf and Monod, 2023] and personalized federated learning with Sheaf HyperNetworks [Nguyen et al., 2024, Liang et al., 2024]. In representation learning, many knowledge-graph embedding techniques have been reinterpreted as sheaf global-section problems [Gebhart et al., 2023, Kvinge et al., 2021]. Collectively, these advances highlight the expressive power and flexibility of sheaf-based models in handling complex, heterogeneous, and higher-order data domains.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Paper is about Copresheaf Neural Networks and the paper presents the framework as well as a set of experiments.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We briefly discuss them at the end of conclusion and provide an extended version in our appendix.

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

Justification: Our main theorem and propositions are proven in the appendix.

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

Justification: We provide detailed experimental settings in the appendix. We will also release our code and data publicly upon publication.

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

Justification: We will also release our code and data publicly upon publication.

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

Justification: For each experiment, we separately mention the hyperparameters, optimizer, data splits and the curation of data.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Paper report error bars over experiments and describes the number of runs these error bars are over.

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

Justification: Compute is not explicitly discussed; however, timings are not presented as part of the argument about the proposed approach and are not needed for a to replicate the results. Compute resources are modest (single GPU runs); details are not essential to reproducing results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper focuses on a generalized framework for neural networks, and does not address areas or applications of ethical concern as defined by the Ethics Guidelines.

Data sets in the paper do not contain any personally identifiable information/ data about real people.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is primarily theoretical and is not expected to have immediate societal impacts.

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

Justification: Paper presents a theoretical framework and does not release models or present results on data scraped from the internet etc.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cite datasets used in the paper.

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

Justification: We will also release our code and data publicly upon publication. The repository has a license. We do not have assets of others within our data/code base.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No human subject research was done

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [No]

Justification: LLMs were only used for wording help.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy for what should or should not be described.