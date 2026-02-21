## Reproducing Kernel Banach Space Models for Neural Networks with Application to Rademacher Complexity Analysis

## Alistair Shilton, Sunil Gupta, Santu Rana, Svetha Venkatesh

Applied Artificial Intelligence Initiative, Deakin University, Geelong, Australia alistair.shilton@deakin.edu.au , sunil.gupta@deakin.edu.au , santu.rana@deakin.edu.au , svetha.venkatesh@deakin.edu.au

## Abstract

This paper explores the use of Hermite transform based reproducing kernel Banach space methods to construct exact or un-approximated models of feedforward neural networks of arbitrary width, depth and topology, including ResNet and Transformers networks, assuming only a feedforward topology, finite energy activations and finite (spectral-) norm weights and biases. Using this model, two straightforward but surprisingly tight bounds on Rademacher complexity are derived, precisely (1) a general bound that is width-independent and scales exponentially with depth; and (2) a width- and depth-independent bound for networks with appropriately constrained (below threshold) weights and biases.

## 1 Introduction

A significant challenge in neural networks is understanding how large models, despite their high capacity to overfit training data, can still generalize effectively (Neyshabur et al., 2014). Learning theory tells us that inductive bias plays an important role in explaining this phenomena, where inductive bias is the restriction of the space of potential learned functions (neural networks) to a small subset F of the total space of space, either explicitly through regularization or implicitly through the training algorithm used. Rademacher complexity (Bartlett &amp; Mendelson, 2002; Steinwart &amp; Christman, 2008) is one measure of the complexity or expressive power of F that has been used to understand inductive bias through the lens of uniform convergence - that is, the rate at which the empirical risk (on the training dataset of N samples) converges to actual risk (on the data distribution) (for a discussion of alternative approaches see (Valle-P´ erez &amp; Louis, 2020)). A representative approach to Rademacher complexity analysis in neural networks is 'peeling' (Neyshabur et al., 2015; Golowich et al., 2018; Truong, 2022). In this approach, the total compexity is bounded by 'peeling off' the output layer D to extract factors due to that layer and thus express the total Racehmacher complexity as a product of terms due to the output layer D and the Rademacher complexity of the preceeding ( D -1) -layer network. The process is then repeated, peeling off successive layers until the process terminates at the input layer. This typically results in a bound that exhibits width independence (assuming popular schemes such as LeCun, He or Glorot weight scaling) and exponential depth dependence, and contains some (typically Lipschitz-type) scaling term due to the neural activations as well a (typically depth-exponential) 'nuisance factor'. For example in (Neyshabur et al., 2015) it is shown that, for a simple unbiased layerwise network with spectral-norm bound weight matrices W [ j -1: j ] and Lipschitz activations, the Rademacher complexity is bounded as:

<!-- formula-not-decoded -->

This bound can be refined in various ways (eg. (Golowich et al., 2018)), but the basic form remains, as do the nuisance factors (the term 2 D in the above bound, for example) in one form or another.

An alternative approach is to construct a bilinear (dual) representation of the model that splits the input x ∈ X and parameters Θ ∈ W into separate terms in a dual representation:

<!-- formula-not-decoded -->

where Ψ : W →W , φ : X →X are feature maps and 〈· , · ] : W×X →R is a continuous bilinear product. Examples of this type of model are the neural network Gaussian process (Rasmussen &amp; Williams, 2006) (NNGP) models (Neal, 1996), which treat all layers prior to the output as fixed and model the influence of the weights in the output layer; neural tangent kernel (NTK) models (Jacot et al., 2018; Daniely, 2017; Daniely et al., 2016), which model the (first-order) variation of the weights about their initial values in a reproducing kernel Hilbert space (RKHS) (Aronszajn, 1950) (see for example (Du et al., 2019b; Allen-Zhu et al., 2019; Du et al., 2019a; Zou et al., 2020; Zou &amp; Gu, 2019; Arora et al., 2019b,a; Cao &amp; Gu, 2019)); and reproducing kernel Banach space (RKBS) (Lin et al., 2022; Zhang et al., 2009; Zhang &amp; Zhang, 2012; Song et al., 2013; Sriperumbudur et al., 2011; Xu &amp; Ye, 2014) approaches such as (Shilton et al., 2023), which recursively construct feature maps Ψ : W →W , φ : X →X to exactly model the neural network (beyond first-order). 1 In all cases the utility of the model in the context of Rademacher complexity analysis is that it makes the construction of bounds straightforward through the use of either the Cauchy-Schwarz inequality (if 〈· , · ] is an inner product) or the continuity of the bilinear product; and moreover, as peeling is not applied directly to the Rademacher complexity, nuisance factors arising from this procedure may be avoided. However the assumptions made by these models (wide-networks, lazy training, restrictions on neural activations and network topology etc (Arora et al., 2019b; Lee et al., 2019; Bai &amp; Lee, 2019)) can complicate analysis and limit their applicability.

Our goal in this paper is to address two questions, (1) can we formulate an exact (non-approximate) model for a wide class of neural networks, including ResNet and Transformers, avoiding entirely the question of gaps between the performance of the neural network and its model; and (2) can such a model be used to derive straightforward, non-vacuous, widely applicable, training-independent bounds on Rademacher complexity without nuisance factors. We answer these questions with the following contributions:

1. Exact RKBS model (Theorem 1): For feedforward neural network with arbitrary topology, finite weight and biases and finite-energy neural activations, we construct an exact model that recasts neural networks as elements in a reproducing kernel Banach space (RKBS) defined by the bilinear product:

<!-- formula-not-decoded -->

where Ψ : W →W is a weight/bias feature map, φ : X →X is a data feature map, and 〈· , · ] g : W×X → R is a continuous bilinear form characterized by an indefinite metric g .

2. Rademacher Complexity Bound (Theorem 4): We observe that, for our RKBS model:

<!-- formula-not-decoded -->

where C Θ ≤ 1 and, using this, derive a straightforward non-asymptotic bound for the Rademacher complexity of a very general class of neural networks (including ResNet and Transformers) that is width-independent, depth-exponential and contains no nuisance-factors. For example for a scalar-valued, layerwise, fully-connected, unbiased ReLU network of depth D , our bound is exactly:

<!-- formula-not-decoded -->

where ‖ · ‖ 2 is the spectral norm. More generally, we derive conditions under which the Rademacher complexity bound is both widthand depth-independent , and subsequently R N ( F ) ≤ 1 √ N , and discuss implications for ReLU, ResNet and Transformer networks.

1 Beyond bilinear RKBS (Lin et al., 2022), more general RKBS models have been used in eg. (Bartolucci et al., 2023; Sanders, 2020; Shilton et al., 2023; Parhi &amp; Nowak, 2021; Unser, 2021, 2019; Spek et al., 2022).

̃

<!-- image -->

̃

̃

̃

̃

Figure 1: Layerwise feedforward neural network structure. Each layer ˙   ∈ Z D contains nodes L [ ˙   ] , where the output of the node is computed as shown in the inset. Note that a computational skeleton (Daniely et al., 2016) with one input and one output can be modified to this form by inserting skip nodes (nodes with A [ j ] = { ˜  } , W [ ˜  : j ] = I , b [ j ] = 0 , τ [ ˜  : j ] = id ) into the graph as required.

## 1.1 Mathematical Notations

Vectors and matrices: Column vectors are a , b with elements a i , b j . Matrices are A , B with elements A i,j , rows A i : and columns A : j . | a | and sgn( a ) are the elementwise norm and sign. ‖ A ‖ 2 = σ max ( A ) is the spectral norm and ‖ A ‖ F is the Frobenius norm. A glyph[circledot] B , A ⊗ B , A ⊗ glyph[arrowbothv] B are Hadamard, Kronecker and columnwise Kronecker (Khatri-Rao) product. A ⊕ B = [ A T B T ] T is columnar matrix concatenation. A # k = A # k times . . . # A is the exponentiation for operator # .

Products and norms: 〈· , ·〉 , 〈 〈· , · , · , . . . 〉 〉 and 〈· , · ] are inner, multilinear and bilinear products, where 〈 a , b 〉 = ∑ i a i b i , 〈 〈 a , b , c , . . . 〉 〉 = ∑ i a i b i c i . . . , 〈 a , b ] g = ∑ i g i a i b i and 〈 A , b ] g = ∑ i g i A i : b i throughout. We also find it convenient to define an operator form 〈 〈·〉 〉 i a i = 〈 〈 a 1 , a 2 , . . . 〉 〉 .

Sets and functions: N = { 0 , 1 , . . . } , Z + = { 1 , 2 , . . . } , Z N = { 1 , . . . , N } . ∂ A is the boundary of A . id( a ) = a . [ a ] + = max { a, 0 } , [ a ] + = [[ a i ] + ] i . a 〈glyph[circledot] b 〉 = sgn( a ) glyph[circledot] | a | glyph[circledot] b (Der &amp; Lee, 2007). L 2 ( R ˜ H , e -‖ ζ ‖ 2 2 )= { τ : R ˜ H → R H | ∫ ζ ∈ R ˜ H ‖ τ ( ζ ) ‖ 2 2 e -‖ ζ ‖ 2 2 d ζ &lt; ∞} are the finite-energy functions.

Multi-indices: Multi-indices are k , l ∈ N n with elements k i , l j . | k | = ∑ i k i , k ! = ∏ i k i ! , a k = ∏ i a k i i , ( k l ) = ∏ i ( k i l i ) . ∂ k ∂ x k = ∏ i ∂ k i ∂x k i i . We use the shorthands k glyph[follows] n l for k ∈ N n and | k | &gt; l , k glyph[followsequal] n l for k ∈ N n and | k | ≥ l , k ≺ n l for k ∈ N n and | k | &lt; l , k glyph[precedesequal] n l for k ∈ N n and | k | ≤ l .

Hermite Polynomials: He k ( x ) are the (probabilist's) Hermite polynomials. He k ( x ) = ∏ i He k i ( x i ) are the multivariate Hermite polynomials. He k = He k (0) , He k = He k ( 0 ) are the Hermite numbers (Abramowitz et al., 1972; Morse &amp; Feshbach, 1953; Olver et al., 2010; Rahman, 2017).

glyph[negationslash]

Indexing Conventions: Layers are ˙   ∈ Z D (there are D layers). Nodes are j ∈ Z E (there are E nodes). Layer ˙   contains nodes L [ ˙   ] ⊆ Z E : ∪ ˙   ∈ Z D L [ ˙   ] = Z E , L [ ˙   ] ∩ L [ ˙   ′ ] = ∅∀ ˙   = ˙   ′ . Node j ∈ L [ ˙   ] in layer ˙   has parents ˜  ∈ A [ j ] ⊆ L [ ˙   -1] . L [0] = { 0 } , L [ D ] = { E } are the input/output layers.

## 2 Setting and Assumptions

We consider layerwise feedforward neural networks as shown in Figure 1. This contains E nodes j ∈ Z E arranged in D layers ˙   ∈ Z D and a virtual input node j = 0 (in virtual layer ˙   = 0 ), where layer ˙   ∈ Z D contains nodes L [ ˙   ] ⊆ Z E and layer D contains a single output node E . Anode j ∈ L [ ˙   ] has parents A [ j ] ⊆ L [ ˙   -1] , with its function being specified by an operator # [ j ] ∈ {⊕ , ∑ , ⊗ , 〈 〈·〉 〉} . Given input x , data flows from node j = 0 to node j = E as per Figure 1:

<!-- formula-not-decoded -->

̃

̃

̃

̃

̃

<!-- image -->

(a) Single-Query Attention Block

<!-- image -->

- (e) Bounds Relevant to Section 2.1

<!-- image -->

(b) Attention Block

<!-- image -->

(c) Residual Block

Figure 2: Residual, attention and LayerNorm blocks. In the residual block s ∈ (0 , 1) . The singlequery attention block (a) is for a single query x 1 ,Q with keys x 1 ,K , x 2 ,K , . . . and values x 1 ,V , x 2 ,V , . . . . A single-head attention block (b) is formed from multiple single-query blocks (the usual (matrix) output has been vectorized here). See Table 3 for definitions of neural activations used here.

<!-- image -->

where τ [ ˜  : j ] : R H [ ˜  ] → R H [ ˜  : j ] are neural activation functions; W [ ˜  : j ] are weight matrices; b [ j ] biases; γ [ j ] ∈ { 0 , 1 } (unbiased and biased); and Θ = { W [ ˜  : j ] , b [ j ] : j ∈ Z E , ˜  ∈ A [ j ] } . We assume that:

1. Bounded inputs: x ∈ X = { x ∈ R n : ‖ x ‖ 2 ≤ 1 } .
2. Finite weights/biases: Θ ∈ W = { W [ ˜  : j ] , b [ j ] : ‖ W [ ˜  : j ] ‖ 2 , ‖ b [ j ] ‖ 2 &lt; ∞ : ˜  ∈ A [ j ] , j ∈ Z E } .
3. Finite activations: τ [ ˜  : j ] ∈ L 2 ( R H [ ˜  ] , e -‖ ζ ‖ 2 2 ) ∀ ˜  ∈ A [ j ] , j ∈ Z E .
4. ( Lipschitz/Bounded activations: τ [ ˜  : j ] is either Lipschitz or bounded ∀ ˜  ∈ A [ j ] , j ∈ Z E , and x ∈ ∂ X = { x ∈ R n : ‖ x ‖ 2 = 1 } if any τ [ ˜  : j ] are non-Lipschitz. )

Note that assumption 4 is not required when constructing our bilinear feature space model of neural networks, but is required to cast this model in RKBS and subsequently derive our Rademacher complexity bound. The set of neural networks satisfying our assumptions is F , and its dual is F glyph[star] :

<!-- formula-not-decoded -->

This model is rather general to encompass a wider variety of network architectures. Residual (He et al., 2016) blocks can be built using additive nodes # [ j ] = ∑ as shown in Figure 2c. Single

Figure 3: Characteristics of common neural activation functions. We include poly-ReLU (Cho &amp; Saul, 2009) here as an example where the Lipschitz constant L a of τ | a depends on the radius a . See (Gao &amp; Pavel, 2017) for more detail regarding the softmax Lipschitz constant.

| Neural Activation   | τ ∈ L 2 ( R H , e -‖ ζ ‖ 2 2   | Lipschitz ( L r )         | Bounded ( B )      | Valid here   |
|---------------------|--------------------------------|---------------------------|--------------------|--------------|
| Linear              | glyph[check]                   | glyph[check] ( 1 )        | ×                  | glyph[check] |
| ReLU                | glyph[check]                   | glyph[check] ( 1 )        | ×                  | glyph[check] |
| Poly-ReLU           | glyph[check]                   | glyph[check] ( pa p - 1 ) | ×                  | glyph[check] |
| Tanh                | glyph[check]                   | glyph[check] ( 1 )        | -                  | glyph[check] |
| Sigmoid             | glyph[check]                   | glyph[check] ( 1 2 )      | -                  | glyph[check] |
| Softmax             | glyph[check]                   | glyph[check] ( λ )        | -                  | glyph[check] |
| Softmax i           | glyph[check]                   | glyph[check] ( λ )        | -                  | glyph[check] |
| Norm                | glyph[check]                   | ×                         | glyph[check] ( 1 ) | glyph[check] |

query attention blocks (Vaswani et al., 2017) can be constructed as shown in Figure 2a using not just additive but also inner-product # [ j ] = 〈 〈·〉 〉 , multiplicative # [ j ] = ⊗ and columnar concatenation # [ j ] = ⊕ nodes. Full attention block can be constructed as shown in Figure 2b (and similarly multihead attention). Finally, a LayerNorm (layer normalization (Ba et al., 2016)) block is shown in Figure 2d. We note that blocks of this sort may be combined to form more general networks. Later, we find it convenient to include non-trivial nodes or blocks in the network, so for example we may speak of an 'attention node' j that encompasses (abstracts away) a complete attention block (Figure 2b).

## 2.1 Characterization of Neural Activations

As noted previously, we assume all activation functions in the network are Lipschitz/bounded and finite energy. The finite-energy assumption allows us to apply the Hermite transform to the neural activation functions and subsequently construct our bilinear model of the network, while the Lipschitz/bounded property suffices to ensure that the bilinear model is continuous. Starting with the finite-energy assumption, the multivariate (probabilist's) Hermite polynomials (Abramowitz et al., 1972; Morse &amp; Feshbach, 1953; Olver et al., 2010; Rahman, 2017) are, for multi-index k ∈ N n :

<!-- formula-not-decoded -->

where He k = He k ( 0 ) are Hermite numbers. These form an orthogonal basis of L 2 ( R n , e -‖ ζ ‖ 2 2 ) (Appendix A). By assumption τ [ ˜  : j ] ∈ L 2 ( R H [ ˜  ] , e -‖ ζ ‖ 2 2 ) , and thus the Hermite transform exists: 2

<!-- formula-not-decoded -->

From this we define the magnitude functions s [ ˜  : j ] η : R + → R + (where η ∈ R + ):

<!-- formula-not-decoded -->

which are monotonically increasing and superadditive on R + . Note that while the Hermite transform terms and magnitude functions play an important role in the construction of our model, they play no role in our subsequent analysis of Rademacher complexity (they vanish in our analysis in the limit η → 0 + ). Thus, for our purposes, beyond their existence (which is guaranteed), their exact value/form does not matter here. Nevertheless, see Appendix B for a full analysis of the ReLU activation.

Regarding assumptions 4, if τ [ ˜  : j ] | a ( τ [ ˜  : j ] restricted to a ball of radius a ) is Lipschitz then we denote the Lipschitz constant by L [ ˜  : j ] a . Conversely, if τ [ ˜  : j ] is absolutely bounded, we denote the bound B [ ˜  : j ] , where | τ [ ˜  : j ] ( ζ ) | ≤ B [ ˜  : j ] ∀ ζ . While assumption 4 is not required to construct our bilinear dual representation we find it useful to include L [ ˜  : j ] a here to simplify later results. When L [ ˜  : j ] a is illdefined we use the nominal value L [ ˜  : j ] a = B [ ˜  : j ] /φ [ ˜  ] in the bounded case, or L [ ˜  : j ] a = 1 if assumption 4 is not satisfied. Table 3 provides Lipschitz constants/bounds for common neural activations.

2 These are conditionally convergent series in general, so ordering of multi-indices k , l in sums and vectors must be enforced consistently and must be compatible with the semi-ordering imposed by glyph[precedesequal] H [ ˜  ] .

̃

̃

<!-- image -->

̃

̃

̃

Figure 4: Recursive definition of the bilinear representation f ( x ; Θ) = 〈 Ψ (Θ) , φ ( x )] g . The upper figure is a schematic representation of the formal definition (5), where the bilinear representation of the output of each node is obtained, using (5a), from the bilinear representations of the inputs ˜  ∈ A [ j ] to that node. Subsequently the bilinear representation of the network is defined recursively in terms of the (trivial) bilinear representation of x = 〈 I , x ] 1 . η ∈ R + is an (arbitrary) constant that will be helpful in our Rademacher complexity analysis. With regard notation, we recall that node j is characterized by its operation # [ j ] ∈ {⊕ , ∑ , ⊗ , 〈 〈·〉 〉} , and subsequently the form of the feature-map recursion (5a) depends on this operation as specified by the operators glyph[square] [ j ] (weight map operator), GLYPH&lt;30&gt; [ j ] (data map/metric operator) and glyph[circlering] [ j ] (norm bound operator) defined. For non-Lipschitz neural activations we set L [ ˜  : j ] a = B [ ˜  : j ] /φ [ ˜  ] in the bounded case and L [ ˜  : j ] a = 1 if assumption 4 is not satisfied.

̃

̃

̃

## 3 Neural Networks in Reproducing Kernel Banach Space

As noted in our introduction, a recurring theme in the machine learning (most famously kernel methods) is the use of bilinear (dual) representations to cleanly separate data and model parameters, ie:

<!-- formula-not-decoded -->

Here the set of network parameters Θ , and the data x , are mapped entirely independently into distinct feature spaces by, respectively, Ψ : W →W (weights and biases) and φ : X →X (data). The bilinear product 〈· , · ] : W × X → R m generalizes the inner product of eg SVMs (Cortes &amp; Vapnik, 1995; Burges, 1998; Cristianini &amp; Shawe-Taylor, 2005; Steinwart &amp; Christman, 2008) without losing the very useful property of bilinearity that makes this formalism so convenient to work with. Apart from the potential for constructing a representor theory (kernelization), if the bilinear product is continuous (ie. if ∃ C, C ′ ∈ R + so that 〈 Ψ , φ ] ≤ C ‖ φ ‖∀ Ψ or 〈 Ψ , φ ] ≤ C ′ ‖ Ψ ‖∀ φ ) then the existence of such a model significantly simplifies the development of Rademacher complexity bounds. A model of this type was developed in (Shilton et al., 2023) using a recursive Taylor series expansion of the neural activations - in brief, noting that 〈 a , b ] n g = 〈 a ⊗ n , b ⊗ n ] g ⊗ n , if the input to a neuron can be represented bilinearly then so too could the output, which recursion defines the model. Unfortunately this approach only works for continuous neural activations, and even then only within the RoC of the Taylor expansion, rendering it inapplicable for common activations such as ReLU. Alternatively, in this paper we propose using a Hermite polynomial expansion, which has two benefits, precisely (1) the Hermite polynomial expansion exist for all finite-energy activations and is convergent everywhere (applicability), and (2) as the Hermite polynomials are constructed from monomials we can also use 〈 a , b ] n g = 〈 a ⊗ n , b ⊗ n ] g ⊗ n to construct our model (practicality).

We begin by constructing our dual representation:

Theorem 1. Let f : X × W → R m be a neural network (1) satisfying assumptions 1- 3. Assume nominal bounds ‖ W [ ˜  : j ] ‖ 2 ≤ ω [ ˜  : j ] &lt; ∞ and ‖ b [ j ] ‖ 2 ≤ β [ j ] &lt; ∞∀ j ∈ Z E , ˜  ∈ A [ j ] . Let η ∈ R + . Defining feature maps Ψ : W →W⊂ R ∞× m (weights and biases) and φ : X →X ⊂ R ∞ (data) and metric g ∈ R ∞ as per (5) (Figure 4), the network may be written in bilinear form:

<!-- formula-not-decoded -->

glyph[negationslash]

where ‖ Ψ (Θ) ‖ F ≤ ψ η &lt; ∞ ∀ Θ ∈ W , ‖ φ ( x ) ‖ 2 ≤ φ η &lt; ∞ ∀ x ∈ X , (the constants ψ η , φ η are provided in Appendix C.1), where lim η → 0 + ψ η = ψ and lim η → 0 + φ η = φ ; and we note that lim η → 0 + ‖ φ ( x ) ‖ 2 = φ ∀ x ∈ ∂ X (ie. if ‖ x ‖ 2 = 1 ), and lim η → 0 + ‖ φ ( x ) ‖ 2 &gt; 0 ∀ x = 0 .

A full inductive proof can be found in Appendix C.1. To summarize, picking a layer ˙   ∈ Z D , we assume all nodes ˜  ∈ L [ ˙   -1] in the preceding layer may be written x [ ˜  ] = 〈 Ψ [ ˜  ] (Θ) , φ [ ˜  ] ( x ) ] g [ ˜  ] , which is trivial for the base case ˙   = 0 . Then, using ( A T b ) glyph[circledot] p = ( A ⊗ glyph[arrowbothv] p ) T ( b ⊗ p ) in combination with the Hermite (number) expansion of the neural activation function, we write the incoming edge activations x [ ˜  : j ] as bilinear products x [ ˜  : j ] = 〈 Ψ [ ˜  : j ] (Θ) , φ [ ˜  : j ] ( x ) ] g [ ˜  : j ] (see Appendix for full definitions).

This, combined with the observation that © [ j ] ˜  ∈ A [ j ] W [ ˜  : j ]T x [ ˜  : j ] = ( glyph[square] [ j ] ˜  ∈ A [ j ] W [ ˜  : j ] ) T ( GLYPH&lt;30&gt; [ j ] ˜  ∈ A [ j ] x [ ˜  : j ] ) , suffices to show that x [ j ] = 〈 Ψ [ j ] (Θ) , φ [ j ] ( x ) ] g [ j ] as given, and the result follows by induction.

As alluded to in section 2, we can readily incorporate non-trivial nodes into this framework. In the recursive construction of the feature maps, (5a) is effectively a recipe for converting the bilinear expansion of the inputs to that node to a bilinear expansion of the node's output. As stated, (5a) is for a trivial node of the type shown in Figure 1, but alternatively we could wrap an entire sub-network or block inside this node (eg. an attention block - Figure 2b) and replace (5a) with the overall recipe for converting bilinear expansions of its input to a bilinear expansion of its output. Thus we may reasonably speak of an 'attention node' in a Transformer network without needless clutter. For example Figure 2d includes a table detailing calculations for φ for attention, residual and LayerNorm blocks (nodes) (derivations for these can be found in Appendix D.1, D.2, D.3 and D.4).

Unfortunately the dual representation (6) is insufficient for Rademacher complexity analysis without assumption 4, which requires that the neural activations be Lipschitz or bounded (and in the latter case that ‖ x ‖ 2 = 1 ). This assumption is central to casting the dual model (6) into RKBS, precisely:

Definition 1 (Reproducing kernel Banach space (RKBS)) . A RKBS on X is a Banach space B of functions f : X → Y , where Y is normed, for which the point evaluation functionals δ x ( f ) = f ( x ) on B are continuous (i.e. ∀ x ∈ X ∃ C x ∈ R + such that ‖ δ x ( f ) ‖ ≤ C x ‖ f ‖ B ∀ f ∈ B ).

This is somewhat generic, so following (Lin et al., 2022) we focus on the special case:

<!-- formula-not-decoded -->

where φ : X →X is a data feature map, Ψ : W →W is a weight feature map, X and W are Banach spaces, and 〈· , · ] W×X : W×X → R m is a continuous bilinear form. Given this prequel we have:

Corollary 2. The set F of networks (2) satisfying assumptions 1-4 with Lipschitz neural activations and weights and biases bounded as per Theorem 1 is an RKBS with ‖ f ( · ; Θ) ‖ F glyph[defines] lim η → 0 + ‖ Ψ (Θ) ‖ F ≤ ψ &lt; ∞ and ‖ f ( x ; Θ) ‖ 2 ≤ C x ‖ f ( · ; Θ) ‖ F , where C x = 1 ∀ x ∈ X .

Corollary 3. The set F glyph[star] of networks (2) satisfying assumptions 1-4 with Lipschitz or bounded neural activations and with weights and biases bounded as per Theorem 1 is an RKBS with ‖ f ( x ; · ) ‖ F glyph[star] glyph[defines] lim η → 0 + ‖ φ ( x ) ‖ 2 ≤ φ &lt; ∞ and ‖ f ( x ; Θ) ‖ 2 ≤ C Θ ‖ f ( x ; · ) ‖ F glyph[star] , where C Θ = 1 ∀ Θ ∈ W .

See Appendix C.3 for proofs (the structure of which minics that of the proof of Theorem 1). It follows from this that the model presented in Theorem 1 suffices to achieve our primary goal. Note that this result applies to a very wide range of networks, including feedforward ReLU networks, convolutional networks, residual networks (ResNet), and Transformer networks (see later discussion). We observe that the conditions for F to be an RKBS are stricter than the conditions for F glyph[star] to be an RKBS, as non-Lipschitz neural activations appears incompatible with F being an RKBS. However as we will see that we only require F glyph[star] be an RKBS to proceed with our Rademacher complexity analysis.

## 4 Rademacher Complexity Bounds

We now address our secondary goal, namely using our dual model to bound the Rademacher complexity of neural networks. For h : R m → R , the Rademacher complexity is defined as:

<!-- formula-not-decoded -->

for Rademacher random variables glyph[epsilon1] i ∈ {± 1 } , where x ∼ ν . We have:

Theorem 4. Let F be the set of networks (2) satisfying assumptions 1-4 with weights and biases bounded as per Theorem 1, and let h : R m → R be L -Lipschitz. Then:

<!-- formula-not-decoded -->

where H 1 = 1 if h = id , H m = √ 2 mL otherwise, and φ is defined in Figure 4.

The proof follows the usual template for RKHS models (see eg. (Bartlett &amp; Mendelson, 2002)) using our feature map; replacing the Cauchy-Schwarz inequality with ‖ f ( x ; Θ) ‖ 2 ≤ C Θ ‖ φ ( x ) ‖ 2 ; taking the limit η → 0 + ; and recalling that C Θ = 1 and lim η → 0 + ‖ φ ( x ) ‖ 2 ≤ φ , so ‖ f ( x ; Θ) ‖ 2 ≤ C Θ ‖ φ ( x ) ‖ 2 ≤ φ . See Appendix F for full details.

Considering this Rademacher complexity bound, we recall that typically neural network weights and biases are initialized with magnitude proportional to 1 √ H [ j ] (LeCun initialization) or 1 √ H [ ˜  : j ] (He initialization), and stay close to their initial values in the wide limit, assuming a convex objective. Thus we would expect that ‖ W [ ˜  : j ] ‖ 2 (and hence its upper bound ω [ ˜  : j ] ) should be independent of network width, rendering the complexity bound in Theorem 4 (effectively) width-independent. We also observe that the complexity bound does not contain any explicitly depth-dependent terms (nuisance terms that are often present in such bounds as discussed in (Golowich et al., 2018)); however the bound will in general grow exponentially with depth due to the multiplicative build-up of terms in φ from input to output, which is typical of such results (Neyshabur et al., 2015; Golowich et al., 2018; Truong, 2022). For a scalar-valued, unbiased, Lipschitz network with 1 node j = ˙   per layer, (8) becomes: 3

<!-- formula-not-decoded -->

While this bound is depth-exponential in general, we can use to to derive conditions (on the weights) under which this exponentiality can be (in effect) neutralised. Motivated by this, the following result gives general, non-trivial threshold conditions for depth-independent Rademacher complexity:

<!-- formula-not-decoded -->

Figure 5: Conditions for Depth Independent Rademacher Complexity Bounds for Typical Nodes.

| Node or Block Type               | Depth-Independence Condition                                                                               | Notes                                                                                                                                                                                           |
|----------------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Trivial                          | ∥ ∥ b [ j ] ∥ ∥ 2 2 + glyph[circlering] [ j ] ˜  ∈ A [ j ] L [ ˜  : j ]2 1 ∥ ∥ W [ ˜  : j ] ∥ ∥ 2 2 ≤ 1 | See Figure 1 and equation (10).                                                                                                                                                                 |
| Residual                         | ∏ q ∈ Z d j ∥ ∥ W [ ˜  : j ] q ∥ ∥ 2 ≤ s                                                                  | See Figure 2c. In this bound we denote the weight matrix for (internal) layer q as. W [ ˜  : j ] q . See Appendix D.1 for the complete derivation.                                             |
| Single-Query Attention           | λ ‖ W V ‖ 2 ‖ W Q ‖ 2 ‖ W K ‖ 2 ≤ 1                                                                        | See Figure 2a. In this bound λ is the heat parameter for the softmax. See Appendix D.3 for a complete derivation.                                                                               |
| Single- and Multi-Head Attention | λ √ d model ‖ W V ‖ 2 ‖ W Q ‖ 2 ‖ W K ‖ 2 ≤ 1                                                              | See Figure 2b. In this bound λ is the heat parameter for the softmax. Here d model is the product of the number of queries and the number of heads. See Appendix D.4 for a complete derivation. |

Corollary 5. Let F be the set of networks (2) satisfying our assumptions with weights and biases bounded as per Theorem 1, and let h : R m → R be L -Lipschitz. If:

<!-- formula-not-decoded -->

for all nodes j ∈ Z E , then R N ( h ◦ F ) ≤ H m √ N , independent of both width and depth.

This follows from the recursive definition of φ in (5) (Figure 4) as a sufficient condition to ensure that φ [ j ] = 1 given φ [ ˜  ] = 1 for all nodes j ∈ Z E , ˜  ∈ A [ j ] , and subsequently (recursively) φ = φ [ E ] = 1 . In practice the interpretation of this result is node specific. Conditions for various nodes (in the Lipschitz case) can be found in Table 5, where derivations may be found in the appendices noted. The general, non-Lipschitz (bounded) case is somewhat more complicated. Recall that if there are non-Lipschitz neural activations in the network we assume that x ∈ ∂ X or, equivalently, ‖ x ‖ 2 = 1 ; and for non-Lipschitz, bounded neural activations τ [ ˜  : j ] , we set L [ ˜  : j ] a = B [ ˜  : j ] /φ [ ˜  ] . Considering one such non-Lipschitz neural activation τ [ ˜  : j ] , in the recursive definition of the norm-bound φ in (5), the corresponding term in the sum becomes L [ ˜  : j ] a φ [ ˜  ] = B [ ˜  : j ] - so, for example, for a LayerNorm block (Figure 2d) j ∈ Z E we see that φ [ j ] = √ H [ ˜  : j ] (for full derivation see Appendix D.2), and moreover if this is the only node in its layer then the Rademacher complexity bound will be independent of all layers preceeding it. However we would advise caution here; the assumption x ∈ ∂ X is quite strong and may not be realistic in general. We will discuss how this assumption may be relaxed, along with what impact this relaxation has on our Rademacher complexity bound, in section 4.1.

## 4.1 Generalizations and Standard Toplogies

In this section, we consider two more realistic relaxions assumption 1 - firstly expanding the bounds on ‖ x ‖ 2 , and secondly considering x ∼ X drawn from an unbounded distribution X such that it lies in the bounded of case 1 with high probability (whp). Using these, we conclude the paper by analysing a range of standard network topologies. Formally, we consider two generalization of assumption 1:

Strictly Bounded: x ∈ X ρ,r = { x ∈ R n : ρ ≤ ‖ x ‖ 2 ≤ r } , where 0 ≤ ρ ≤ r ∈ R + and ρ &gt; 0 if the network contains non-Lipschitz neural activations.

Distributional: x ∼ X for a distribution X for which there exists 0 ≤ ρ ≤ r ∈ R + ( ρ &gt; 0 if the network contains non-Lipschitz neural activations) such that x ∈ X ρ,r with high probability ≥ 1 -glyph[epsilon1] .

In both cases we consider a mild modification of our feature map (5), precisely: 4

<!-- formula-not-decoded -->

and moreover for non-Lipschitz, bounded neural activations τ [ ˜  : j ] , we set L [ ˜  : j ] a = B [ ˜  : j ] /φ [ ˜  ] ↓ . For a full discussion of this generalization see Appendix C. Observe that, in the limit η → 0 + :

<!-- formula-not-decoded -->

4 In both the cases ρ = 0 , r = 1 (the fully Lipschitz variant of assumption 1) and ρ = r = 1 (the nonLipschitz variant of assumption 1) this reduces to the standard feature map (5).

glyph[negationslash]

The Rademacher complexity bound (Theorem 8) takes the same form as usual. The exact impact of letting r = 1 is dependent on the network topology. For a simple, layerwise, fully Lipschitz neural network with 1 trivial node j = ˙   per layer, as demonstrated in Appendix E.1: 5

<!-- formula-not-decoded -->

This bound is exponential in depth, as discussed previously. As a mild generalization of this scenario, if we allow non-Lipschitz neural activations (for example LayerNorm blocks) in this simple network, with the last such at layer ˙   = D ↓ , then, using (12) and noting that φ [ ˙   ] /φ [ ˙   ] ↓ = r ρ ∀ ˙   ∈ Z D ∪ { 0 } :

<!-- formula-not-decoded -->

where we note that this bound is exponential in the depth to the non-Lipschitz node D -D ↓ and proportional to r ρ . The independence from the weights of layers preceeding D ↓ is noteworthy, but if we consider as an example a ReLU network terminated by a LayerNorm and observe that the scale of these weights is entirely arbitrary, it perhaps not surprising. The 1 ρ term reflects the need to assume that, in the worst-case, small inputs will be 'amplified' (e.g. by LayerNorm) to the largest possible output.

The transformer can be similarly analysed. The catch in this case is that the attention block is multiplicative. In particular (see Appendices D.3, D.4 for details), for an attention block:

<!-- formula-not-decoded -->

so, unlike the simpler case considered above, each attention block will cause polynomial growth in the ratio φ φ ↓ . Subsequently, as shown in Appendix E.3, the overall bound (due to the final LayerNorm) is:

<!-- formula-not-decoded -->

where W out are the weights for the linear output layer of the transformer. 6 If ρ = r (that is, x ∈ ∂ X as in assumption 1) this collapses to H m √ d model ω √ N , but in general, despite being independent of the weights in all but the output layer of the network, this bound grows doubly-exponentially in depth, dependent on the ratio r ρ of smallest/largest inputs.

Finally, bounds for the distributional case follow the strictly bounded case, but only whp ≥ 1 -glyph[epsilon1] . For example, in Appendix C.4 we consider x ∼ X = N ( 0 n , σ 2 I n ) , showing that ρ ≤ ‖ x ‖ 2 ≤ r , where:

<!-- formula-not-decoded -->

whp ≥ 1 -glyph[epsilon1] which apply, respectively, for the purely Lipchitz and bounded cases. In particular, the latter result allows one to explore Rademacher complexity bounds in the general case without giving ρ or r (the bounds on ‖ x ‖ 2 , where ρ in particular may be difficult to quantify intuitively) a-priori.

## 5 Conclusions

In this paper we have constructed a dual model of a very general set of feedforward neural networks that re-expresses them as a continuous bilinear product between a weight/bias feature map and a data feature map - that is, a reproducing kernel Banach space (RKBS) model. This model is exact , with no approximation or assumptions beyond bounded (norm) inputs, bounded (spectral norm) weights and biases, and finite-energy neural activations, and incorporates networks ranging from simple layerwise models (ReLU etc) to ResNet and Transformers. Subsequently, we have applied this model to the analysis of the Rademacher complexity analysis of neural networks, giving a simple recursive bound for the Rademacher complexity of all models neural network topologies covered by our model. This bound is exact (non-asymptotic) and does not include depth- or width- dependent nuisance factors. Moreover it is width-independent and, while exponential in depth (due to the multiplicative build-up of terms through the layers of the networks), enables us to derive straightforward (spectral) threshold conditions under which depth-dependence may be removed entirely.

5 This also applies to ResNet, where for residual blocks ˙   with d internal layers we let ω [ ˙   -1: ˙   ]2 = ( ω [ ˜  -1: ˜  ] d 2 . . . ω [ ˜  -1: ˜  ] 2 2 ω [ ˜  -1: ˜  ] 1 2 +1 -s 2 as also described in Appendix E.1.

6 M here is the size of the encoder/decoder stacks. We use M here rather than N as used in (Vaswani et al., 2017) to avoid a notational ambiguity within our paper.

## References

- Abramowitz, M., Stegun, I. A., and McQuarrie, D. A. Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables . Dover, 1972.
- Allen-Zhu, Z., Li, Y., and Song, Z. A convergence theory for deep learning via over-parameterization. In International Conference on Machine Learning , pp. 242-252. PMLR, 2019.
- Aronszajn, N. Theory of reproducing kernels. Transactions of the American Mathematical Society , 68:337-404, Jan-Jun 1950.
- Arora, S., Du, S., Hu, W., Li, Z., and Wang, R. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In International Conference on Machine Learning , pp. 322-332. PMLR, 2019a.
- Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R. R., and Wang, R. On exact computation with an infinitely wide neural net. In Advances in Neural Information Processing Systems , pp. 81398148, 2019b.
- Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450 , 2016.
- Bai, Y. and Lee, J. D. Beyond linearization: On quadratic and higher-order approximation of wide neural networks. arXiv preprint arXiv:1910.01619 , 2019.
- Bartlett, P. L. and Mendelson, S. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research , 3:463-482, 2002.
- Bartolucci, F., De Vito, E., Rosasco, L., and Vigogna, S. Understanding neural networks with reproducing kernel banach spaces. Applied and Computational Harmonic Analysis , 62:194-236, January 2023.
- Boyd, J. P. The rate of convergence of hermite function series. Mathematics of Computation , 35 (152):1309-1316, October 1980.
- Burges, C. J. C. A tutorial on support vector machines for pattern recognition. Knowledge Discovery and Data Mining , 2(2):121-167, 1998.
- Cao, Y. and Gu, Q. Generalization bounds of stochastic gradient descent for wide and deep neural networks. In Advances in neural information processing systems , volume 32, 2019.
- Cho, Y. and Saul, L. K. Kernel methods for deep learning. In Y., B., D., S., D., L. J., Williams, C. K. I., and Culotta, A. (eds.), Advances in Neural Information Processing Systems 22 , pp. 342-350. Curran Associates, Inc., 2009. URL http://papers . nips . cc/paper/3628-kernelmethods-for-deep-learning . pdf .
- Cortes, C. and Vapnik, V. Support vector networks. Machine Learning , 20(3):273-297, 1995.
- Courant, R. and Hilbert, D. Methods of Mathematical Physics . John Wiley and sons, New York, 1937.
- Cristianini, N. and Shawe-Taylor, J. An Introduction to Support Vector Machines and other KernelBased Learning Methods . Cambridge University Press, Cambridge, UK, 2005.
- Daniely, A. SGD learns the conjugate kernel class of the network. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 30 , pp. 2422-2430. Curran Associates, Inc., 2017. URL http://papers . nips . cc/paper/6836-sgd-learns-the-conjugate-kernelclass-of-the-network . pdf .
- Daniely, A., Frostig, R., and Singer, Y . Toward deeper understanding of neural networks: The power of initialization and a dual view on expressivity. In Lee, D. D., Sugiyama, M., Luxburg, U. V., Guyon, I., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 29 , pp. 22532261. Curran Associates, Inc., 2016. URL http://papers . nips . cc/paper/6427-towarddeeper-understanding-of-neural-networks-the-power-of-initialization-anda-dual-view-on-expressivity . pdf .

- Der, R. and Lee, D. Large-margin classification in banach spaces. In Proceedings of the JMLR Workshop and Conference 2: AISTATS2007 , pp. 91-98, 2007.
- Du, S., Lee, J., Li, H., Wang, L., and Zhai, X. Gradient descent finds global minima of deep neural networks. In International conference on machine learning , pp. 1675-1685. PMLR, 2019a.
- Du, S. S., Zhai, X., Poczos, B., and Singh, A. Gradient descent provably optimizes over-parameterized neural networks. In Conference on Learning Representations , 2019b.
- Gao, B. and Pavel, L. On the properties of the softmax function with application in game theory and reinforcement learning. arXiv preprint arXiv:1704.00805 , 2017.
- Golowich, N., Rakhlin, A., and Shamir, O. Size-independent sample complexity of neural networks. In COLT , 2018.
- Gradshteyn, I. S. and Ryzhik, I. M. Table of Integrals, Series, and Products . Academic Press, London, 2000.
- He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 770-778, 2016.
- Hille, E. Contributions to the theory of Hermitian series. II. The representation problem. Trans. Amer. Math. Soc. , 47:80-94, 1940.
- Jacot, A., Gabriel, F., and Hongler, C. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in neural information processing systems , pp. 8571-8580, 2018.
- Lee, J., Xiao, L., Schoenholz, S., Bahri, Y., Novak, R., Sohl-Dickstein, J., and Pennington, J. Wide neural networks of any depth evolve as linear models under gradient descent. Advances in neural information processing systems , 32, 2019.
- Lin, R., Zhang, H., and Zhang, J. On reproducing kernel banach spaces: Generic definitions and unified framework of constructions. Acta Mathematica Sinica, English Series , 2022.
- Maurer, A. A vector-contraction inequality for rademacher complexities. In Ortner, R., Simon, H. U., and Zilles, S. (eds.), Algorithmic Learning Theory , pp. 3-17, Cham, 2016. Springer International Publishing. ISBN 978-3-319-46379-7.
- Morse, P. M. and Feshbach, H. Methods of Theoretical Physics . McGraw-Hill, 1953.
- Neal, R. M. Priors for infinite networks , pp. 29-53. Springer, 1996.
- Neyshabur, B., Tomioka, R., and Srebro, N. In search of the real inductive bias: On the role of implicit regularization in deep learning. arXiv preprint arXiv:1412.6614 , 2014.
- Neyshabur, B., Tomioka, R., and Srebro, N. Norm-based capacity control in neural networks. In Proceedings of Conference on Learning Theory , pp. 1376-1401, 2015.
- Olver, F. W., Lozier, D. W., Boisvert, R. F., and Clark, C. W. NIST Handbook of Mathematical Functions . Cambridge University Press, USA, 1st edition, 2010. ISBN 0521140633.
- Parhi, R. and Nowak, R. D. Banach space representer theorems for neural networks and ridge splines. J. Mach. Learn. Res. , 22(43):1-40, 2021.
- Rahman, S. Wiener-hermite polynomial expansion for multivariate gaussian probability measures. Journal of Mathematical Analysis and Applications , 454(1):303-334, 2017.
- Rasmussen, C. E. and Williams, C. K. I. Gaussian Processes for Machine Learning . MIT Press, 2006.
- Sanders, K. Neural networks as functions parameterized by measures: Representer theorems and approximation benefits. Master's thesis, Eindhoven University of Technology, 2020.
- Shilton, A., Gupta, S., Rana, S., and Venkatesh, S. Gradient descent in neural networks as sequential learning in reproducing kernel banach space. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pp. 31435-31488. PMLR, 23-29 Jul 2023.

- Song, G., Zhang, H., and Hickernell, F. J. Reproducing kernel banach spaces with the glyph[lscript] 1 norm. Applied and Computational Harmonic Analysis , 34(1):96-116, Jan 2013.
- Spek, L., Heeringa, T. J., and Brune, C. Duality for neural networks through reproducing kernel banach spaces. arXiv preprint arXiv:2211.05020 , 2022.
- Sriperumbudur, B. K., Fukumizu, K., and Lanckriet, G. R. Learning in hilbert vs. banach spaces: A measure embedding viewpoint. In Advances in Neural Information Processing Systems , pp. 1773-1781, 2011.
- Steinwart, I. and Christman, A. Support Vector Machines . Springer, 2008.
- Truong, L. V. On rademacher complexity-based generalization bounds for deep learning. arXiv preprint arXiv:2208.04284 , 2022.
- Unser, M. A representer theorem for deep neural networks. J. Mach. Learn. Res. , 20(110):1-30, 2019.
- Unser, M. A unifying representer theorem for inverse problems and machine learning. Foundations of Computational Mathematics , 21(4):941-960, 2021.
- Valle-P´ erez, G. and Louis, A. A. Generalization bounds for deep learning. arXiv preprint arXiv:2012.04115 , 2020.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems , volume 30. Curran Associates, Inc., 2017. URL https://proceedings . neurips . cc/ paper files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper . pdf .
- Xu, Y. and Ye, Q. Generalized mercer kernels and reproducing kernel banach spaces. arXiv preprint arXiv:1412.8663 , 2014.
- Zhang, H. and Zhang, J. Regularized learning in banach spaces as an optimization problem: representer theorems. Journal of Global Optimization , 54(2):235-250, Oct 2012.
- Zhang, H., Xu, Y., and Zhang, J. Reproducing kernel banach spaces for machine learning. Journal of Machine Learning Research , 10:2741-2775, 2009.
- Zou, D. and Gu, Q. An improved analysis of training over-parameterized deep neural networks. In Advances in neural information processing systems , volume 32, 2019.
- Zou, D., Cao, Y., Zhou, D., and Gu, Q. Gradient descent optimizes over-parameterized deep relu networks. Machine learning , 109(3):467-492, 2020.

## A Properties of Hermite polynomials

## A.1 Univariate Case

The (probabilist's) Hermite polynomials are given by (Abramowitz et al., 1972; Morse &amp; Feshbach, 1953; Olver et al., 2010; Courant &amp; Hilbert, 1937):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and form an orthogonal basis of L 2 ( R , e -x 2 ) . For any f ∈ L 2 ( R , e -x 2 ) there exist Hermite coefficients a 0 , a 1 , . . . ∈ R (the Hermite transform of f ) such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and moreover the series representation converges everywhere on the real line. 7

The Hermite numbers derive from the Hermite polynomials: 8

<!-- formula-not-decoded -->

where k !! = k ( k -2)( k -4) . . . is the double-factorial. It is well known that (see eg. (Morse &amp; Feshbach, 1953)):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It follows that, taking care not to change or order of summation (remember this is an alternating series, so convergence depends on the order of the summation):

<!-- formula-not-decoded -->

For later reference we also note that the Hermite polynomials satisfy the well-known recursion and derivative relation for k &gt; 1 :

<!-- formula-not-decoded -->

7 Hille (1940); Boyd (1980) show that this series converges on a strip X ρ = { z ∈ C : -ρ &lt; Im( z ) &lt; ρ } of width ρ about the real axis in the complex plane, where (note that Hille (1940); Boyd (1980) use the normalized physicist's Hermite polynomials. The additional scale factor here arises in the translation to the un-normalized probabilist's Hermite polynomials used here):

<!-- formula-not-decoded -->

8 Typically the Hermite numbers are defined from the physicist's Hermite polynomials, but as we use the Probabilist's form as we find these more convenient for our purposes.

or, explicitly:

where:

and so:

## A.2 Multivariate Case

The multivariate Hermite polynomials He k : R n → R , k ∈ N n , are the functions (Rahman, 2017):

<!-- formula-not-decoded -->

where we use multi-index notation | k | = ∑ i k i , a k = ∏ i a k i i , k ! = ∏ i k i ! , k !! = ∏ i k i !! , and ∂ k ∂ ζ k = ∏ i ∂ k i ∂ζ k i i . For any f ∈ L 2 ( R n , e -ζ T ζ ) there exists coefficients a k ∈ R : k glyph[followsequal] n i (the Hermite transform of f ), where k glyph[followsequal] n i means k ∈ { k ∈ N n : | k | ≥ i } , such that:

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

and the series representation converges everywhere on R n .

As in the univariate case, the multivariate Hermite numbers are defined as:

<!-- formula-not-decoded -->

where in the final step we have used (15). Subsequently:

<!-- formula-not-decoded -->

where k glyph[follows] n i means k ∈ { k ∈ N n : | k | &gt; i } .

Finally, if we consider a vector-valued function f : R n → R m then it is not hard to see that scalarvalued expansion can be extended to:

<!-- formula-not-decoded -->

where a k ,i are the Hermite coefficients of f i . We note that if n = m and f ( ζ ) = [ g ( ζ i )] i acts elementwise (for example a neural activation that acts elementwise) then:

<!-- formula-not-decoded -->

where b 0 , b 1 , . . . are the (univariate) Hermite coefficients of g : R → R .

## B ReLU Activation Function Analysis

In this section we derive the Hermite-polynomial expansion of the ReLU activation function:

<!-- formula-not-decoded -->

We find it convenient to work in terms of the physicists Hermite polynomials H k to suit (Gradshteyn &amp;Ryzhik, 2000). So:

<!-- formula-not-decoded -->

and hence, using (16) and (Gradshteyn &amp; Ryzhik, 2000, (7.373)):

<!-- formula-not-decoded -->

and using (Gradshteyn &amp; Ryzhik, 2000, (7.373)) again:

<!-- formula-not-decoded -->

If k = 2 p and p &gt; 0 then, noting that H k (0) = √ 2 k He k :

<!-- formula-not-decoded -->

If k = 2 p +1 and p &gt; 0 then:

<!-- formula-not-decoded -->

For the cases k = 0 , 1 we use the result:

and so:

In the case k = 0 :

and in the case k = 1 :

<!-- formula-not-decoded -->

Subsequently, for the elementwise ReLU neural activaiton, using (18):

<!-- formula-not-decoded -->

Next we derive the magnitude functions for the ReLU. Using integration by parts, we see that:

<!-- formula-not-decoded -->

So:

<!-- formula-not-decoded -->

Select c so that the first derivative is 1 2 ζ :

<!-- formula-not-decoded -->

Hence:

<!-- formula-not-decoded -->

## C Bilinear Representation - Proofs, Bounds and Generalizations

In this section we present proof of theorems, bounds and generalizations related to the bilinear representation. To avoid repeating work we consider a mild generalization of the map presented in the body of the paper, as shown in Figure 6. The key generalizations here over the main body of the paper are:

1. Welet x ∈ X ρ,r = { x ∈ R n : ρ ≤ ‖ x ‖ 2 ≤ r } for some 0 ≤ ρ ≤ r ∈ R + . In the main body of the paper we let ρ = 0 , r = 1 for simplicity when all neural activations are Lipschitz, and ρ = r = 1 otherwise. In general we require ρ &gt; 0 when considering a network containing non-Lipschitz neural activations.
2. We use base-case Ψ [0] (Θ) = r I n , g [0] = 1 r 1 n here (recall r = 1 in the main body).
3. We use L [ ˜  : j ] ψ [ ˜  ] η , L [ ˜  : j ] φ [ ˜  ] η to scale the feature map here rather than L [ ˜  : j ] ψ [ ˜  ] and L [ ˜  : j ] φ [ ˜  ] . Note, however, that (as we demonstrate) lim η → 0 ψ [ ˜  ] η = ψ [ ˜  ] and lim η → 0 φ [ ˜  ] η = φ [ ˜  ] , so the definitions coincide in the limit η → 0 + , which is the case we are primarily concerned with (as it is used in our Rademacher complexity bound).
4. For non-Lipschitz, bounded neural activations (edges), we let L [ ˜  : j ] = B [ ˜  : j ] φ [ ˜  ]2 ↓ η , where φ [ ˜  ] ↓ η is a lower bound on ‖ φ [ ˜  ] ( x ) ‖ 2 (recall that ρ = 1 in the main body of the paper, and note that we will prove that φ [ ˜  ] ↓ η = φ [ ˜  ] η in this case). More generally for neural activations that are neither bounded or Lipschitz we let L [ ˜  : j ] = 1 . Note, however, that we cannot prove continuity of our bilinear product in this case, so the relevant parts of the proof do not apply for this.

̃

̃

<!-- image -->

̃

̃

̃

Figure 6: Complete version of Figure 4 (recursive definition of bilinear representation) splitting edge/node maps, showing limits and including correct weights (the main body uses simplified weights that are correct in the limit and sets r = 1 , ρ = 0 (or ρ = 1 if non-Lipschitz neurons are present). For non-Lipschitz, bounded neural activations τ [ ˜  : j ] we set L [ ˜  : j ] a = B [ ˜  : j ] /φ [ ˜  ]2 ↓ η , and for non-Lipschitz and unbounded neural activations we set L [ ˜  : j ] a = 1 .

̃

̃

̃

Note that for each j ∈ Z E the feature map construction is split into two steps - a construction (21a) for the incoming edges [ ˜  : j ] , which we refer to as the edge case ; and a construction (21b) for the (core of the) node itself, which we refer to as the node case . This split will simplify our proofs and improve clarity by separating the key steps therein. As in the main body of the paper the overall representation is:

<!-- formula-not-decoded -->

We will also show that:

<!-- formula-not-decoded -->

where the following bounds hold:

<!-- formula-not-decoded -->

noting that φ [ ˜  : j ] η ↓ , φ [ j ] η ↓ &gt; 0 if ρ &gt; 0 and:

<!-- formula-not-decoded -->

## C.1 Proof of Theorem 1 - Bilinear Representation

Recalling that the network is arranged in layers ˙   = 0 , 1 , 2 , . . . , D , and given that we know the feature map representation for the input layer ˙   = 0 is, tivially:

<!-- formula-not-decoded -->

where Ψ [0] (Θ) = r I , φ [0] ( x ) = x and g [0] = 1 r 1 , it suffices to show that if all outputs of all nodes ˜  ∈ L [ ˙   -1] ( L [0] = { 0 } ) in layer ˙   -1 can be expressed in terms of bilinear products:

<!-- formula-not-decoded -->

then all nodes j ∈ L [ ˙   ] , using the definitions given, can be written:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and:

We call (27) the edge case and (28) the node case, and will treat them separately.

Edge case: We are given that (26) is correct. Substituting (21a) into the bilinear product and using (26), (3) and (17), we find that:

<!-- formula-not-decoded -->

which is the desired result (27).

Node case: We have shown that (27) is correct. Substituting (21b) into the bilinear product and using (27), we find that, for columnar concatenation nodes © [ j ] = ⊕ (so glyph[square] [ j ] = diag , [ j ] = ⊕ ):

GLYPH&lt;30&gt;

<!-- formula-not-decoded -->

For additive nodes © [ j ] = ∑ (so glyph[square] [ j ] = ⊕ , GLYPH&lt;30&gt; [ j ] = ⊕ ):

<!-- formula-not-decoded -->

For Kronecker-product nodes © [ j ] = ⊗ (so glyph[square] [ j ] = ⊗ , GLYPH&lt;30&gt; [ j ] = ⊗ ):

<!-- formula-not-decoded -->

For Hadamard product nodes © [ j ] = ⊙ (so glyph[square] [ j ] = ⊗ glyph[arrowbothv] , [ j ] = ⊗ ):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where on the final line we use 〈 〈·〉 〉 as an operator (see notation section). So, in all cases:

<!-- formula-not-decoded -->

which is the desired result (28) for the node case.

## C.2 Proof of Theorem 1 - Norm-Bounds

Recalling that the network is arranged in layers ˙   = 0 , 1 , 2 , . . . , D , and noting that for the input layer ˙   = 0 is, tivially from our assumptions:

<!-- formula-not-decoded -->

where ψ [0] η = r , φ [0] ↓ η = ρ and φ [0] η = r , it suffices to show that if all outputs of all nodes ˜  ∈ L [ ˙   -1] in layer ˙   -1 satisfy:

<!-- formula-not-decoded -->

then all nodes j ∈ L [ ˙   ] , using the definitions given, satisfy:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We call (30) the edge case and (31) the node case, and will treat them separately.

Edge case: We are given that (29) is correct. By direct calculation, for incoming edges (21a), using the multinomial theorem at step ( ∗ ) :

<!-- formula-not-decoded -->

and:

and:

<!-- formula-not-decoded -->

which we may bound as (using that s [ ˜  : j ] η is increasing on R + ):

<!-- formula-not-decoded -->

which is the desired result (30).

Node case: We have shown that (30) is correct. For columnar concatenation nodes © [ j ] = ⊕ (so glyph[square] [ j ] = diag , [ j ] = ⊕ ):

<!-- formula-not-decoded -->

GLYPH&lt;30&gt;

For additive nodes © [ j ] = ∑ (so glyph[square] [ j ] = ⊕ , [ j ] = ⊕ ):

<!-- formula-not-decoded -->

For Kronecker-product nodes © [ j ] = ⊗ (so glyph[square] [ j ] = ⊗ , [ j ] = ⊗ ):

<!-- formula-not-decoded -->

For Hadamard product nodes © [ j ] = ⊙ (so glyph[square] [ j ] = ⊗ glyph[arrowbothv] , [ j ] = ⊗ ):

<!-- formula-not-decoded -->

For multi-inner-product nodes © [ j ] = 〈 〈·〉 〉 (so glyph[square] [ j ] = ⊗ glyph[arrowbothv] ( · ) 1 , [ j ] = ⊗ ):

<!-- formula-not-decoded -->

Thus in general, for all nodes considered here:

<!-- formula-not-decoded -->

which we may bound as:

<!-- formula-not-decoded -->

which is the desired result (31) for the node case.

We observe that the data-feature-map bound is tight:

<!-- formula-not-decoded -->

In the limit η → 0 , identifying ψ [ j ] = ψ [ j ] 0 + , φ [ j ] ↓ = φ [ j ] ↓ 0 + , φ [ j ] = φ [ j ] 0 + ; ψ = ψ 0 + , φ ↓ = φ ↓ 0 + , φ = φ 0 + ; ψ = ψ [ E ] , φ ↓ = φ [ E ] ↓ , φ = φ [ E ] ; where, recursively ∀ j ∈ Z E :

<!-- formula-not-decoded -->

(here we have used that lim η → 0 s [ ˜  : j ] η ( z ) s [ ˜  : j ] η (1) = z by observation of the definition), which justifies our simplification in the main body of the paper.

## C.3 Proof of Corollaries 2 and 3 - Continuity Bounds

Our approach here mimics the previous two proofs. For the input node j = 0 , for a given Θ ∈ W :

<!-- formula-not-decoded -->

As in the previous section consider a single node j ∈ L [ ˙   ] in layer ˙   . Assume that, for all nodes in the previous layer ˜  ∈ L [ ˙   -1] :

<!-- formula-not-decoded -->

Edge case: for a Lipschitz neural activation τ [ ˜  : j ] , for incoming edges (21a), for fixed Θ ∈ W :

<!-- formula-not-decoded -->

and similarly for fixed x ∈ X :

<!-- formula-not-decoded -->

Alternatively, for bounded (non-Lipschitz) neural activations:

<!-- formula-not-decoded -->

which is finite; and:

<!-- formula-not-decoded -->

which is unbounded in general. It follows that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and:

<!-- formula-not-decoded -->

For additive nodes © [ j ] = ∑ (so glyph[square] [ j ] = ⊕ , GLYPH&lt;30&gt; [ j ] = ⊕ ):

<!-- formula-not-decoded -->

and:

<!-- formula-not-decoded -->

For Kronecker-product nodes © [ j ] = ⊗ (so glyph[square] [ j ] = ⊗ , GLYPH&lt;30&gt; [ j ] = ⊗ ):

<!-- formula-not-decoded -->

and:

<!-- formula-not-decoded -->

For Hadamard product nodes © [ j ] = ⊙ (so glyph[square] [ j ] = ⊗ glyph[arrowbothv] , GLYPH&lt;30&gt; [ j ] = ⊗ ), using that the norm of the

Hadamard product of unit vectors is ≤ 1 :

<!-- formula-not-decoded -->

and:

<!-- formula-not-decoded -->

For multi-inner-product nodes © [ j ] = 〈 〈·〉 〉 (so glyph[square] [ j ] = ⊗ glyph[arrowbothv] ( · ) 1 , GLYPH&lt;30&gt; [ j ] = ⊗ ), using that the multiinner-product of ( 2 -norm) unit vectors is at most 1 :

<!-- formula-not-decoded -->

and:

<!-- formula-not-decoded -->

Thus in general, for all nodes considered here:

<!-- formula-not-decoded -->

where we have defined:

<!-- formula-not-decoded -->

Consequently, defining C Θ ,η = C [ E ] Θ ,η , C W ,η = C [ E ] W ,η , C x ,η = C [ E ] x ,η , and C X ,η = C [ E ] X ,η :

<!-- formula-not-decoded -->

where C W ,η is finite in general and C X ,η is finite if all neural activations are Lipschitz.

The limit case η → 0 + is of particular interest here. Defining C Θ = lim η → 0 + C Θ ,η , C [ j ] Θ = lim η → 0 + C [ j ] Θ ,η , C W = lim η → 0 + C W ,η , C [ j ] W = lim η → 0 + C [ j ] W ,η , we observe that, using the form of the base case and recursion:

<!-- formula-not-decoded -->

This result, combined with Theorem 1, suffices to prove Corollaries 2 and 3.

## C.4 Bounds for Data Drawn from a Distribution

A common variation of our assumption x ∈ X ρ,r - that is, the assumption that x is hard-limited in terms of its 2 -norm - is that x ∼ X is drawn from some data distribution X . With regard to our analysis, for arbitrary data distributions it is not possible to extend our analysis; however if it can be proven that x ∈ X ρ,r with-high-probability ≥ 1 -glyph[epsilon1] for suitable ρ, r then our results will follow whp ≥ 1 -glyph[epsilon1] . To take a simple example, suppose we draw data from an n -dimensional normal distribution:

<!-- formula-not-decoded -->

## D Non-Trivial Blocks

In this section we consider norm- and continuity- bounds for particular common neural network archictectural blocks. Note that in all cases the continuity bounds C Θ , C W , C x , C X are well-behaved, so our task is to analyse the norm-bound φ . In this regard we refer the reader to (21) in Figure 6.

Figure 7: Calculation of φ out in a residual network block.

<!-- image -->

Trivially, for x ∼ X :

<!-- formula-not-decoded -->

Thus we have x ∈ X ρ,r with high probability ≥ 1 -glyph[epsilon1] , where:

<!-- formula-not-decoded -->

for some υ ∈ [0 , 1) , In the purely Lipschitz case we can simplify this by setting υ = 0 (so ρ = 0 ):

<!-- formula-not-decoded -->

and more generally, if we allow non-Lipschitz neural activations, whp ≥ 1 -glyph[epsilon1] :

<!-- formula-not-decoded -->

and:

and subsequently:

Figure 8: Calculation of φ out in a residual network block.

<!-- image -->

## D.1 Residual Block Bounds

In this section we consider the calculation of φ for a residual block. Figure 7 shows the notation we use here. All neural activations in this block are 1 -Lipschitz so trivially, using our bounds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we note that, for ρ &gt; 0 :

## D.2 LayerNorm Block Bounds

As shown in Figure 8, the LayerNorm block is distinct insofar as it is non-Lipschitz. First we note that that ‖ √ H I ‖ 2 = √ H , ‖ I -1 H 11 T ‖ 2 = 1 , so we may set ω [in:mid] = 1 , ω [mid:out] = √ H . Noting

<!-- formula-not-decoded -->

Figure 9: Single-query attention block.

<!-- image -->

that the Norm activation is non-Lipschitz and bounded by B [Norm] = 1 , we see that:

<!-- formula-not-decoded -->

and trivially ψ mid = ω [in:mid] ψ in = ψ in , φ mid ↓ = ω [in:mid] φ in ↓ = φ in ↓ and φ mid = ω [in:mid] φ in = φ in . Hence, overall: √

where we note that, for ρ &gt; 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D.3 Single-query Attention Block Bounds

The standard bounds as presented in (21) are needlessly pessimistic for softmax nodes in attention blocks (Figure 2) as they are derived without taking into account the operation of the softmax in layer 3, which is a full softmax that has been split into components here - so while we can bound the set of all QK outputs, the standard bounds only bound the individual components without taking into account the interaction between then. The following more nuanced analysis gives a tighter bound.

In the following analysis we make the simplifying assumption ψ η, ˜ ı,V = ψ η,V , φ η, ˜ ı,V ↓ = φ η,V ↓ , φ η, ˜ ı,V = φ η,V ; ψ η, ˜ ı,K = ψ η,K , φ η, ˜ ı,K ↓ = φ η,K ↓ , φ η, ˜ ı,K = φ η,K . Given this:

Layer 1: following the standard approach:

<!-- formula-not-decoded -->

where we note that, for ρ &gt; 0 :

<!-- formula-not-decoded -->

Layer 2: following the standard approach:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Layer 3: we need to take some care with this layer. In particular, noting that the output of the layer is effectively the softmax split componentwise, we can constrain the sum of φ [3] η, ˜ ı,QK as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some c 1 , c 2 , . . . ≥ 0 : ∑ ˜ ı c 2 ˜ ı = 1 (in the standard analysis we would let c 1 = c 2 = . . . = 1 ). Layer 4: following the standard approach:

<!-- formula-not-decoded -->

Layer 5: recalling that c 1 , c 2 , . . . ≥ 0 satisfy ∑ ı c 2 ı = 1 :

˜

˜

<!-- formula-not-decoded -->

Taking the limit η → 0 + we summarise the overall operation of this block as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 10: Single-Head attention block.

<!-- image -->

## D.4 Single-Head and Multi-Head Attention Block Bounds

The standard single-head attention block is constructed from from single-query attention blocks as shown in figure 10. Multi-head attention is similar, with an additional h concatenations. Making the additional assumption, over section D.3, that ψ η, ˜ ı,V = ψ η,V , φ η, ˜ ı,V ↓ = φ η,V ↓ , φ η, ˜ ı,V = φ η,V , it is not difficult to see that:

<!-- formula-not-decoded -->

where d Q is the number of queries and h is the number of heads. We note that, for ρ &gt; 0 :

<!-- formula-not-decoded -->

## E Bounds for Standard Network Toplogies

In this section we apply our results, and in particular our norm-bound ‖ φ ( x ) ‖ 2 ≤ φ ∀ x ∈ X ρ,r which is central in our Rademacher complexity bound, to standard network topologies.

## E.1 Simple Unbiased Lipschitz Layerwise Network and ResNet

Consider a simple network with 1 unbiased node with L -Lipschitz activations per layer, so D = E , j = ˙   ∈ Z D , and A [ ˙   ] = { ˙   -1 } . In this case, using (21), ∀ ˙   ∈ Z D :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we find that the norm-bound φ (and hence our Rademacher complexity bound) is proportional to the product of the weight-matrix norms, the maximum input norm r , and the exponentiated Lipschitz constant. In the distributional case, assuming x ∼ N ( 0 n , σ 2 I n ) and using (33), whp ≥ 1 -glyph[epsilon1] :

<!-- formula-not-decoded -->

We can also bound the residual network (ResNet) norm with this by including residual blocks as and hence, using that φ [0] = r :

nodes in the network. For example, if node ˙   is a residual block then the effective weight-norm bound ω [ ˙   -1: ˙   ] becomes, for that non-trivial block, using (35):

<!-- formula-not-decoded -->

where ω [ ˙   -1: ˙   ] k is the norm-bound for the k th weight matrix W [ ˙   -1: ˙   ] k in the residual block ˙   .

## E.2 Simple Unbiased non-Lipschitz Layerwise Network and LayerNorm

In this section we consider the same network as in the previous section E.1, excepting that we assume at least 1 non-Lipschitz, bounded neural activation. In this case, using (21), ∀ ˙   ∈ Z D :

<!-- formula-not-decoded -->

where φ [0] ↓ = ρ and φ [0] = r . We immediately observe that:

<!-- formula-not-decoded -->

and hence (39) simplifies to:

<!-- formula-not-decoded -->

If we further assume that node ˙   = D ↓ is the non-Lipschitz node closest to the output node, bounded by B [ D ↓ -1: D ↓ ] , the norm bound becomes:

<!-- formula-not-decoded -->

The first thing to note with this bound is that it is no longer depth exponential, but rather depth-tonon-Lipschitz ( D -D ↓ ) exponential. This may appear surprising at first, but it is perhaps not so surprising when we note that the Lipschitz norm-bound scales with the max weight-matrix normbound, while a bounded neural-activation displays attributes that, in a crude sense, flatten out the magnitude of their input from previous layers. The obvious extreme case is a network combining ReLU and LayerNorm nodes, in which case we can scale weight matrices preceeding the LayerNorm arbitrarily without affecting the operation of the network in any way. This is directly reflected in the above expression, where the norm-bound φ is independent of the magnitude (matrix norm) of the weight-matrices in layers 1 , 2 , . . . , D ↓ -1 before the LayerNorm.

The ratio r ρ in the bound is perhaps less intuitive. In particular, while we would expect that the norm bound of ‖ φ ( x ) ‖ 2 should scale (increase) as ‖ x ‖ 2 ≤ r increases (which the norm-bound does), it is less obvious that the bound should increase as the lower bound ‖ x ‖ 2 ≥ ρ decreases . To understand this behaviour, recall that we only characterise neural activation τ [ D ↓ -1: D ↓ ] by its upper bound B [ D ↓ -1: D ↓ ] ( 1 for simplicity), so we must make a worst-case assumption that ‖ x [ D ↓ ] ‖ 2 = 1 for all x ∈ X ρ,r . If ‖ x ‖ 2 = ρ then, in our worst-case analysis, the node must, in effect, amplify the input so that ‖ x [ D ↓ ] ‖ 2 = 1 ; the smaller ρ , the larger the amplification required. 9 This is why we take care not to over-claim in the case ρ = r = 1 in the main body of the paper.

Another apparent difficulty with this norm-bound is that one may argue that the lower bound ‖ x ‖ 2 ≥ ρ is artificial, and that real data may not satisfy this bound. To cover this, we may use the distributional

9 In the limit ρ → 0 + the amplication must approach ∞ , which is why we insist ρ &gt; 0 in this case.

case. For example, if node D ↓ is a LayerNorm node and assuming x ∼ N ( 0 n , σ 2 I ) then, using (36) and (34), with high probability ≥ 1 -glyph[epsilon1] :

<!-- formula-not-decoded -->

We observe that this bound is scale-independent, both in terms of the 'size' σ of the data distribution and weight-norm bounds for layers prior to the final non-Lipschitz node D ↓ . The proportionality to √ H arises from the choice of LayerNorm, and the exact form of the new scaling arises from our choice of distribution. 10

## E.3 Transformers

Finally we may consider the Transformer. For concreteness we will assume the structure described in (Vaswani et al., 2017, Figure 1); and for tractability we will ignore the input/output embedding and positional encoding, and instead assume inputs and outputs (post-embedding/encoding) x I , x O ∈ X ρ,r = { x ∈ R d K : ρ ≤ ‖ x ‖ 2 ≤ r } .

Encoder: The first layer in the encoder stack consists of a multi-head attention block inside a residual block, followed by a LayerNorm block. Using (38), the output norm-bound of the multihead attention block will satisfy:

<!-- formula-not-decoded -->

Subsequently, the output norm-bound of the residual block will satisfy:

<!-- formula-not-decoded -->

and we see from (37) that the output of the LayerNorm will satisfy:

<!-- formula-not-decoded -->

This is followed by a feed-forward network inside a residual block, again followed by a LayerNorm block. The analysis of this is similar to the above, excepting that, because the block inside the residual block is additive, there is no need to cube the ratio. The output of the LayerNorm in this layer will therefore satisfy:

<!-- formula-not-decoded -->

At total of 11 M = 6 of these layers occur sequentially, where for each the ratio is cubed due to the presence of the multi-head attention block. Subsequently, for the output of the encoder, we find:

<!-- formula-not-decoded -->

Decoder: The decoder is similar, with some important caveats. Perhaps most importantly, in the first layer the output of the second attention block (and therefore the output of the first layer in the decoder) will satisfy:

<!-- formula-not-decoded -->

This is followed by ( M -1) = 5 additional layers, and so it may be seen that the output of the decoder, prior to the final linear and softmax, will satisfy:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and, using (36):

Subsequently, assuming the weights in the linear output layer of the Transformer satisfy ‖ W ‖ 2 ≤ ω and assuming λ = 1 in the final softmax we find that the overall norm-bound for the Transformer is:

<!-- formula-not-decoded -->

10 It may be informative to investigate the impact of the distribution x ∼ X on this bound in future work.

11 We use M here rather than N due to the notational clash between (Vaswani et al., 2017) and our use of N .

## F Proof of Theorem 4 - Rademacher Complexity

We are concerned with calculating the Rademacher complexity of:

<!-- formula-not-decoded -->

where h is L -Lipschitz. We have from (Maurer, 2016) that:

<!-- formula-not-decoded -->

Thus we reduce the dimensionality of the problem to 1 -dimension. Proceeding with the standard argument:

<!-- formula-not-decoded -->

and so:

<!-- formula-not-decoded -->

and the final result follows in the limit η → 0 + , recalling lim η → 0 + C W ,η = 1 , lim η → 0 + φ η = φ :

<!-- formula-not-decoded -->

NB : in the special case m = 1 , h = id , we can skip the first step which contributed the factor √ 2 L and the 1 -norm2 -norm-inequality.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: the abstract/introduction were written after the key contributions were completed specifically to reflect them.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: the limitations of the work are clearly outlined in the Setting and Assumptions section of the paper.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate 'Limitations' section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (eg., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, eg., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: All assumptions are clearly stated in the body of the paper. Most (non-trivial) proofs are summarised in the body, with reference to relevant appendices for details.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [NA]

Justification: This is a purely theoretical work. Results will apply to any network satisfying our assumption, which are analytic in nature: network topology, bounds on weights/biases that are translated to the final result, and requirements on neural network activation functions.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (eg., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (eg., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: This is a purely theoretical work. As noted previously, results will apply to any network satisfying our assumption, which are analytic in nature: network topology, bounds on weights/biases that are translated to the final result, and requirements on neural network activation functions.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips . cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips . cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: See previous justification re results, data and code.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: See previous justification re results, data and code.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer 'Yes' if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: See previous justification re results, data and code.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips . cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conforms to the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work is purely theory, so I cannot foresee specific societal impacts beyond improved performance in neural networks.

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

Justification: see previous responses.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: this is a theory paper.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode . com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: this is a theory paper.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs are not core to this method, though the complexity bounds derived herein may apply to them.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips . cc/Conferences/2025/LLM ) for what should or should not be described.