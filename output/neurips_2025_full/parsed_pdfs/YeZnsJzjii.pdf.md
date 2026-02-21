## Metric Automata Theory: A Unifying Theory of RNNs

## Adam Dankowiakowski

University of Oxford adankowiakowski02@gmail.com

## Alessandro Ronca

IRIS-AI

alessandro.ronca@iris-ai.org

## Abstract

We propose Metric Automata Theory , an elegant generalisation of classic Automata Theory to continuous dynamical systems, that constitutes a unifying theory of all kinds of Recurrent Neural Networks (RNNs), including widely-adopted architectures such as xLSTM and State Space Models (SSMs). The theory allows one to analyse RNNs both in the finite and unbounded precision settings seamlessly, while utilising fundamental results of Automata Theory. It also provides a novel notion of robustness that guarantees numerical stability and contributes to stability of learning. We employ the theory to prove a comprehensive set of expressivity results for widely-adopted RNNs, with a focus on robustness and finite-precision. Notably, we contrast the capabilities of xLSTM and SSMs for robustly modelling all star-free regular languages-xLSTM can do so, while SSMs cannot robustly recognize the FLIP-FLOP language. Thus we give a novel perspective on the importance of non-linear recurrences, giving insight for why xLSTM shows superior performance to SSMs on several tasks. We provide an improved understanding of the capabilities of Mamba, a popular SSM model. We show that Mamba is not generally capable of recognising the star-free languages under finite-precision, which is seemingly in contrast with the existing theoretical and empirical results for SSMs. We clarify the picture, by showing that Mamba admits a piecewise-linearly separable state space that allows it to approximate star-free languages, with some length-generalisation abilities. At the same time, Mamba does not admit such state spaces for languages like Parity. This explains why empirically Mamba performs well on star-free languages, and fails on Parity.

## 1 Introduction

Recurrent Neural Networks (RNNs) encompass all the neural networks that process sequences by maintaining a state though some form of recurrence . Notable RNNs are Vanilla RNNs such as Elman-RNNs [Elman, 1990], LSTM [Hochreiter and Schmidhuber, 1997], GRU [Cho et al., 2014], and the more recent and now widely-adopted xLSTM [Beck et al., 2024] and the family of State Space Models (SSMs) including S4 [Gu et al., 2022], Mamba [Gu and Dao, 2023], HiPPO [Gu et al., 2020], and DeltaNet [Yang et al., 2024]. The more recent RNNs achieve state-of-the-art performance, comparable to other notable neural networks such as Transformers [Vaswani et al., 2017], by leveraging new design principles that overcome the limitations of previous RNNs, enabling key properties such as parallel training to take full advantage of modern computer architectures.

Recently, there has been an increasing interested in developing a systematic understanding of the capabilities of sequence models, including RNNs and Transformers, beyond empirical evidence. As of now, there is a rich literature of formal results regarding the expressivity of both RNN (cf. [Knorozova and Ronca, 2024a,b, Weiss et al., 2018, Merrill et al., 2020]) and Transformers (cf. [Strobl et al., 2024, Merrill and Sabharwal, 2024, Hahn, 2020]), with some studies directly comparing the two, e.g., [Bhattamishra et al., 2024]. We focus on modelling RNN expressivity in terms of formal

languages-an active area, with big impact on the new directions of research for novel architectures. For instance, Sarrof et al. [2024] showed that a family of SSM models, including Mamba [Gu and Dao, 2023], has expressivity restricted to star-free regular languages in the finite precision setting, due to restricting the eigenvalues of the state-update gates to be non-negative. Soon after, Grazzi et al. [2025] extended the capabilities of SSMs beyond star-free languages, by modifying the implementation of Mamba and DeltaNet models to allow negative eigenvalues-narrowing the gap between SSMs and LSTM models.

The main limitation of this recent literature is that it lacks a principled, commonly-accepted theory or framework providing the a established setting for investigations. For instance, both Sarrof et al. [2024] and Grazzi et al. [2025] provide similar arguments proving that under finite-precision, SSMs with gates with non-negative eigenvalues are restricted to star-free languages (with Sarrof et al. [2024] proving a special case). However, the details of the finite-precision frameworks used by the two are completely different, and result in different assumptions. This means that the results are hard to compare without carefully assessing the assumptions and inspecting the proofs.

We propose Metric Automata Theory (MAT) as an elegant and principled theory that generalises Automata Theory to continuous dynamical systems, with RNNs being a special case of particular interest. It has the ambition of being a unifying theory for the study of all kinds of RNNs, providing a common framework that allows for analysing the expressivity of RNN architectures in a uniform way, in order to guarantee solid progress in the field. First of all, MAT generalises the notion of finiteness to a general metric notion of η -finiteness (Definition 2), which captures the intuitive idea of the finite-precision setup, while retaining generality. Second, we develop a correspondence between η -finite systems and finite automata, thus allowing us to apply powerful algebraic results and notions of Automata Theory. Third, the theory introduces a notion of robustness (Definition 4) that guarantees numerical stability, contributes to stability learning, and notably allows one to prove results for real-world finite-precision implementations while abstracting away the difficulties introduced by finite-precision arithmetic. Fourth, we develop the notion of geometrically-constrained systems (Definition 8). This notion goes beyond the setting of finite-precision, allowing for modelling of languages beyond regular. It captures the empirical properties of systems approximating languages with length-generalization properties, which are observed in practice. Finally, we showcase the effectiveness of Metric Automata Theory by proving a comprehensive set of expressivity results for widely-adopted RNN architectures, with a focus on robustness and finite-precision. We argue that our results provide an improved understanding of the actual capabilities of RNNs as observed in practice.

## 2 Preliminaries

We present the most central and possibly lesser-known preliminary notions here, and we defer notation, additional background on metric spaces, background on Recurrent Neural Networks, and additional background on the topics presented below to Appendix A.

Path-connectedness in metric spaces. A path in X from a to b is a continuous map γ : [0 , 1] → X such that γ (0) = a and γ (1) = b . We can define a relation ∼ X , where a ∼ X b when there is a path in X from a to b . This relation is an equivalence, partitioning X into disjoint equivalence classes, called (path-connected) components . For space X , we denote the set of its equivalence classes by X . Path-connectedness is preserved by continuous functions, which is a crucial property to our theory. Notably, a continuous function X → Y , with finite codomain Y , has to map all points within a component of X to the same element of Y .

Dynamical systems. Following [Knorozova and Ronca, 2024a,b], we adopt dynamical systems as our general formalism. A (dynamical) system is a tuple S = ⟨ X,U,f,x 0 , Y, h ⟩ , where X is the state space , U is the input space , f : X × U → X is the dynamics function , x 0 ∈ X is the initial state , Y is the output space and h : X × U → Y is the output function. We have that X,U,Y are metric spaces, and f, h are continuous . We call the tuple D = ⟨ X,U,f ⟩ the dynamics of S . Given x 0 ∈ X and N ∈ N , dynamics D define a map from sequences u [1 ..N ] of inputs to sequences x [1 ..N ] of states with each state given by x n = f ( x n -1 , u n ) . Hence, we say that D defines the function D : X × U ∗ → X with D ( x 0 , ε ) = x 0 on the empty sequence ε , and D ( x, u [1 ..n ] ) = x n on any input string u [1 ..n ] . We refer to the function defined by D as state-sequence function . System S defines a map from input sequences u [1 ..N ] to output sequences y [1 ..N ] where y n = h ( x n , u n ) for

all n . Hence, we say that S defines the function S : U + → Y with S ( u [1 ..n ] ) = y n . When h is independent of U , we additionally define S ( ε ) = h ( x 0 ) , extending the definition to S : U ∗ → Y .

Cascades. The formalism of cascades provides a flexible way to describe dynamical systems consisting of subsystems forming an acyclic network. Their flexibility allows us, e.g., to consider not only feed-forward layers of SSMs as in [Grazzi et al., 2025, Sarrof et al., 2024], but also more complex architectures with, e.g., mixes of different types of neurons (see Figure 6 in Appendix A.6).

A feed-forward cascade C is a form of dynamics ⟨ X,U,f ⟩ with X = X 1 × · · · × X n , and f with a particular factorisation. We may see C as consisting of dynamics D 1 , . . . , D n where D i = ⟨ X i , U × X [1 ..i -1] ⟩ . State updates in a cascade proceed in a feed-forward fashion, with component D i having access to the updated states of the previous components D 1 , . . . , D i -1 . Details of cascades in relation to Automata Theory are deferred to Appendices B.5 and G.2.

Finite Automata and Formal Languages. A (finite) alphabet is a finite set Σ of elements called letters or symbols . A (formal) language L over Σ is a subset of Σ ∗ . It is often convenient to characterise L in terms of its indicator function I L . A (finite) automaton is a tuple A = ⟨ Q, Σ , δ, q 0 , Γ , θ ⟩ , where Q is a finite set of elements called states , Σ is the finite input alphabet , δ : Q × Σ → Q is called transition function , Γ is an alphabet called output alphabet , and θ : Q × Σ → Γ is called output function . The tuple A ′ = ⟨ Q, Σ , δ ⟩ is called a semiautomaton , and in particular A ′ is the semiautomaton of A .

An automaton A with output alphabet Γ = { 0 , 1 } is called a language recogniser , and it recognises the language L whose indicator function is the one defined by A . The languages recognised by finite automata are the regular languages .

Algebraic Automata Theory (AAT). It studies finite automata through the lens of algebraic notions such as semigroups and groups, c.f. [Hartmanis and Stearns, 1966, Ginzburg, 1968, Arbib, 1969, Dömösi and Nehaniv, 2005]. The Prime Decomposition Theorem by Krohn and Rhodes [1965] shows how every semiautomaton can be decomposed into a cascade of prime semiautomata. One prime semiautomaton is the flip-flop , that describes the elementary system with the ability to store and manipulate one bit of information. Formally, FLIP-FLOP := ⟨{ high , low } , { set , reset , id } , δ ⟩ with transitions give by δ ( q, id ) = q , δ ( q, set ) = high , and δ ( q, reset ) = low for every state q .

Automata that admit a cascade decomposition into flip-flops are called group-free , and they are central since group-free automata recognise the star-free languages , cf. [Ginzburg, 1968]. To relate different automata, we adopt the notion of realisation for Mealy machines, cf. Definitions 1.14 and 1.15 of [Hartmanis and Stearns, 1966] and appendix B.4. Realisation describes how a machine can imitate another machine after a renaming of inputs and outputs.

Recurrent Neural Network Architectures An Elman-RNN has dynamics D = ⟨ X,U,f ⟩ where f ( x, u ) = tanh ( A X · x + A U · u + b ) . State Space Models (SSMs) are based on linear recurrence with particular parametrisations such as Mamba [Gu and Dao, 2023]. To model linear recurrence in general, we introduce Linear Recurrent Dynamics (LRD), defined as dynamics ⟨ X,U,f ⟩ , where f ( x, u ) = A ( u ) · x + B ( u ) , with states X ⊆ K d state , inputs U = K d input for K ∈ { R , C } , We call A ( u ) ∈ K d state × d state the state-transition gate and B ( u ) ∈ K d state the input gate . The recently introduced model xLSTM [Beck et al., 2024] makes use of both non-linear and linear recurrences. xLSTM introduces two types of blocks: sLSTM and mLSTM. We provide the parametrization of mLSTM blocks in Appendix G.3.

## 3 Metric Automata Theory

We present Metric Automata Theory (MAT), a generalisation of Automata Theory to dynamical systems. Next we present preliminary considerations on automata and the preliminary notion of language recognition for dynamical systems. Then we present the central notions of the theory.

Automata as dynamical systems. We start by observing that finite automata are a special case of dynamical systems . Our goal is to establish a framework to analyse Recurrent Neural Networks (RNNs), with a focus on the study of their expressivity in terms of the ability to recognise formal

Figure 1: System dynamics and corresponding canonical semiautomaton, given in Definition 3.

<!-- image -->

languages. Every automaton A = ⟨ Q, Σ , δ, q 0 , Γ , θ ⟩ is a dynamical system if we endow Q, Σ , Γ with the discrete metric , giving them a discrete topology. In particular, this implies that the functions δ and θ are trivially continuous, and hence it shows that automata are continuous systems . The connection to dynamical systems makes it clear that a semiautomaton is a special case of dynamics, and clarifies how automata define functions F : Σ + → Γ .

Definition 1. Given alphabets Σ and Γ , and continuous functions enc : Σ → U and dec : Y → Γ , we say that a system S implements a function F : Σ + → Γ , with encoder enc and decoder dec , if F ( w ) = dec ◦ S (enc( w )) , for every w ∈ Σ + , where enc( w ) ∈ U + applies enc element-wise. We also say that S can-implement F if it implements F for some choice of enc and dec . When Γ = { 0 , 1 } , we say that S recognises a language L if it implements its indicator function I L , and that S can-recognise L if it can-implement I L .

## 3.1 The Notion of η -Finiteness for Dynamical Systems

We show that the metric setting allows for a general notion of finiteness of a given space, capturing the fact that it is essentially finite even if its cardinality is not-details in Appendix B.

Definition 2. For X a set with X ⊆ R d or X ⊆ C d , we say that X is η -finite if it is a finite union of compact, path-connected components. Then, we say that dynamics ⟨ X,U,f ⟩ are η -finite if both X and U are η -finite. Finally, a system S is η -finite if its dynamics are η -finite.

We refer to the components of the definition as η -components of the space. For example, finite alphabets are η -finite, with each element being its own η -component. As path-connectedness and compactness are preserved by continuous mappings, the notions of η -finiteness and η -component have very favourable theoretical properties . Any continuous mapping f : X → Y , with X and Y η -finite, is guaranteed to map any η -component of X entirely into a single η -component of Y .

As a result, all points within the same state η -component will be interpreted as equivalent states, yielding equivalent behaviours of the system; and all points within the same input or output η -component will correspond to the same inputs and outputs modulo encoding and decoding.

All automata are η -finite systems since they are discrete. Conversely, every η -finite system admits a canonical automaton , which fully captures its dynamics and capabilities. It gives us a way to employ the powerful characterisations and results of AAT to any η -finite system dynamics. Figure 1 visualizes the way in which the canonical (semi)automaton is a discrete interpretation of the continuous dynamics.

Definition 3. Any η -finite dynamical system S = ⟨ X,U,f,x 0 , Y, h ⟩ admits a unique canonical automaton , and any η -finite dynamics D = ⟨ Z, V, g ⟩ admits a unique canonical semiautomaton , which are respectively given by C ( S ) := ⟨ X,U,f, [ x 0 ] ∼ X , Im h, h ⟩ , and C ( D ) := ⟨ Z,V , g ⟩ .

Theorem 1. An η -finite system S can-implement the same functions as its canonical automaton, which are necessarily regular.

Canonical automata are a core tool we use to develop an algebraic theory of continuous systems . Generally, it allows us to abstract away the local details of continuous behaviour that do not affect the global expressive capacity of the system. We use it to, e.g., apply the decomposition theorems of AAT to RNNs, and to create the appropriate analogue of realisation of a continuous system.

Strongly robust system + ϵ -covering approximation

<!-- image -->

-robust

:

B

(

f

(

x, u

)

"Transitions have

ε

margin of error"

Figure 2: Given sufficient precision, the transitions of strongly ϵ -robust dynamics can be realized with approximate dynamics on a finite datatype, whenever the datatype gives a ϵ -covering for the state-space.

MAT applied to finite-precision. We explain how Metric Automata Theory provides the foundations to study dynamical systems in a principled way, with the study of RNNs as a special case that is of particular interest for us. First, we note that η -finiteness ensures that the η -components of the considered space are bounded and separated by some positive non-zero distance. Thus, in all η -finite systems, sequences of states converging to a limit will eventually lie in a single η -state-we note that this is the key property of the finite-precision arguments in [Sarrof et al., 2024]. Additionally, in every finite-precision implementation of a system (e.g., with states represented as tensors of floating-point numbers), the state space has finite cardinality, and hence it is trivially η -finite. Similar considerations apply to the input and output spaces. Altogether, Metric Automata Theory equipped with the notion of η -finiteness allows for studying finite-precision (implementations of) dynamical systems without restricting the analysis to any specific representation of the relevant spaces. The next section develops the theory in the case of system implementations based on concrete datatypes.

## 3.2 Robust Systems

The central notion that allows us to extend Metric Automata Theory to the study of finite-precision implementations is the notion of ϵ -robustness . Intuitively, it describes stability of the dynamics under transition perturbations. It provides a way to connect η -finite systems to their floatingpoint implementations on real-world computer architectures, without requiring us to commit to any particular standard of floating-point operations. We let B Ω ( x, r ) := { y ∈ Ω : || x -y || ≤ r } , which denotes the closed Ω -ball at x of radius r .

Definition 4. For ϵ &gt; 0 and X ⊆ Ω , dynamics D = ⟨ X,U,f ⟩ are ϵ -robust (in Ω ) if, for every x ∈ X and every u ∈ U , it holds that B Ω ( f ( x, u ) , ϵ ) ⊆ X -i.e., y ∈ X for all y ∈ Ω s.t. ∥ f ( x, u ) -y ∥ ≤ ϵ . . We say that dynamics D are strongly ϵ -robust (in Ω ) if they are ϵ -robust (in Ω ) and each η -component of X contains an Ω -ball of radius at least ϵ .

We call dynamics robust (resp. strongly robust) , if they are ϵ -robust (resp. strongly ϵ -robust) for some ϵ &gt; 0 . Note that the property of robustness is with respect to the ambient space Ω , which contains the state space X . It is possible that a dynamics is ϵ -robust w.r.t. some ambient space (e.g., R ), and not ϵ -robust w.r.t. another ambient space (e.g., C ). Next we discuss how our notion of robustness allows for drawing conclusions on finite-datatype implementations of a system.

Definition 5. A finite datatype is a set D ⊆ Ω = R d having finite cardinality. A finite-datatype implementation of a system S is then a system whose input, state, and output spaces are finite datatypes, and whose dynamics and output functions are implemented using floating-point operations.

Theorem 2 (Informal version) . Every η -finite system with strongly ϵ -robust dynamics, for ϵ &gt; 0 , can be implemented with floating-point operations given sufficient precision.

By sufficient precision we mean that the state space is sufficiently covered by the finite datatype, and that the floating-point approximation of the dynamics has error at most ϵ . In Appendix C we show two examples of floating-point parametrisations for which the former condition can always be achieved using sufficiently-many bits of precision.

ε

, ε

)

∈

X

u

ε

Considerations on training. Training any machine learning model that can be seen as a dynamical system amounts to optimising a parametric dynamics function f θ , with learnable parameter θ ∈ Θ , along with optimising the output function. In Section C.2 of Appendix C, we prove that, under some mild regularity assumptions, when dynamics D θ = ⟨ X,U,f θ ⟩ are robust, there is a δ &gt; 0 such that for all θ ′ ∈ Θ with || θ -θ ′ || ≤ δ , the function f θ ′ : X × U → X is a well-defined dynamics function, and the corresponding dynamics D θ ′ = ⟨ X,U,f θ ′ ⟩ have the same η -state dynamics as D θ -i.e., they both have the same canonical semiautomaton. Thus, replacing f θ with f θ ′ in a system will not change the system behaviour. Given the previous consideration, an argument should be possible by which models enjoying this form of robust parametrisation are more likely to be produced by training algorithms, compared to models that do not admit a robust parametrisation. However, a systematic development of this argument is beyond the scope of our work.

## 4 Expressivity Results for Vanilla-RNNs, xLSTM, and SSMs

Metric Automata Theory allows us to establish a rich ensemble of expressivity results in the finiteprecision setting and beyond finite-precision. The elegance and generalisability of our setup enables us to compare capabilities of wildly different models. For example, Theorem 4 applies to SSMs with both real and complex state spaces.

## 4.1 Expressivity Results for Robust Language Recognition

We prove that linear recurrences do not admit robust dynamics, whenever they have an identity transformation on their η -states.

Theorem 3 (Non-robustness of LRDs) . Suppose an η -finite LRD D is such that its canonical semiautomaton D A has at least two states, and an input inducing an identity transformation. Then D cannot be ϵ -robust for any ϵ &gt; 0 .

In fact, we show that upon iterating any single input the whole state-space of an η -finite ϵ -robust LRD collapses to a single η -component. We call such dynamics contracting . Furthermore, we show that a cascade of contracting dynamics is contracting, and that contracting dynamics cannot implement a FLIP-FLOP. We defer the technical details to Section D.3.

Theorem 4 (LRDs cannot do FLIP-FLOP robustly) . FLIP-FLOP cannot be implemented by a cascade of η -finite ϵ -robust LRDs for any ϵ &gt; 0 .

xLSTM. We provide constructions for strongly robust realisations of the FLIP-FLOP dynamics for Elman-RNNs and for an sLSTM block-see Appendix G.3 for the details. The Elman-RNN construction is similar to one provided in [Knorozova and Ronca, 2024a], with the high and low η -states located around the attracting fixed-points of tanh . For xLSTM, fixing a particular parametrisation of a sLSTM block allows us to use a very similar construction, with a sigmoid non-linearity. By Fact 1, this proves that all star-free languages can be implemented cascade of strongly-robust xLSTM blocks. Such a cascade is strongly-robust.

Theorem 5 (xLSTM does start-free robustly) . All star-free languages can be recognised by xLSTM cascades, as well as by floating-point implementations of xLSTM cascades given sufficient precision.

## 4.2 SSM Expressivity in Finite-Precision

We prove that η -finite SSMs with state-transition gates having non-negative eigenvalues are restricted to group-free dynamics, and hence can only implement star-free languages, in line with the theoretical results by Sarrof et al. [2024] and Grazzi et al. [2025], in their respective finite-precision setups.

SSMs with non-negative eigenvalue gates are star-free. To transfer the group-free notion into the continuous η -finite dynamics setting we introduce a notion of aperiodic dynamics. We say that an infinite sequence in a η -finite space X is η -convergent in X if all terms of the sequence are eventually in the same η -component of X .

Definition 6. We say that η -finite dynamics D = ⟨ X,U,f ⟩ are aperiodic if, for every x 0 ∈ X and every input sequence ( u n ) n ≥ 1 that is η -convergent in U , we have that the corresponding state sequence ( x n ) n ≥ 1 is η -convergent in X .

Figure 3: Definition of aperiodicity means that the state sequence of aperiodic dynamics under iterated input from the same η -component always η -converges. In particular, if the state sequence always converges to some limit under iterated inputs, then the dynamics are aperiodic. This is the case e.g., for LRDs with diagonal gates with entries in ( -1 , 1) .

<!-- image -->

In Section D.2 of Appendix D, we show that cascades of aperiodic dynamics are aperiodic. Moreover, we show, that η -finite dynamics are aperiodic iff their canonical semiautomaton is group-free. Thus, aperiodic η -finite systems can implement only star-free regular languages.

Theorem 6. Let D be η -finite Linear Recurrent Dynamics, with its state-transition gates having all non-negative eigenvalues. Then D is aperiodic.

The proof structure is similar to the proof of Theorem 1 in [Grazzi et al., 2025], with significant simplifications afforded by our theory. We show that, iterating a fixed input, the state converges, by considering the Jordan Normal Form of the state-transition gate. We also show that finite context length convolutions are aperiodic. Thus, SSMs like Mamba, which are cascades of convolutions and LRDs with non-negative eigenvalue gates, can only recognise star-free languages as they are η -finite.

Mamba cannot implement FlipFlop in finite-precision. The FlipFlop dynamics construction presented in [Sarrof et al., 2024] makes use of the identity state-transition gate. Parametrisation of Mamba prevents it from making use of such gate.In fact, we prove that in the η -finiteness framework, Mamba blocks are contracting dynamics, and thus cannot implement a FLIP-FLOP.

Theorem 7. SSMs with Mamba parametrisation cannot recognise FLIP-FLOP .

## 4.3 Geometrically Constrained Systems

The case of Mamba successfully length-generalising on star-free tasks, despite being unable to model the dynamics for unbounded length inputs, motivates us to expand our theory beyond η -finite systems. The intuition behind the following setup is to allow only for output functions that are sufficiently regular to expect them to be learnable from short input sequences, with length-generalisation. Ultimately, this section provides an example of how Metric Automata Theory can be used to develop theories alternative to η -finiteness, motivated by phenomena observed empirically and defined by geometric properties of the dynamical systems.

Definition 7. Let Ω = R d or Ω = C d . We call C ⊆ Ω a convex-covering if it is a finite union of open, convex sets in Ω . Then, we say that X ⊆ Ω is convex-covered by C if X ⊆ C .

A convex-covering C consists of finitely-many path-connected components, which are open. The path-connected components of C can be arbitrarily classified by an output function with piecewiselinear decision boundaries, with finitely many vertices. Such output functions include all feed-forward networks with ReLU activations, see Proposition 6.1 in [Zhang et al., 2018].

Definition 8. Let Ω = R d or Ω = C d , and let C ⊆ Ω . We say that dynamics D = ⟨ X,U,f ⟩ are convex-covered by C if X is convex-covered by C . We call a system S C = ⟨ X,U,f,C,x 0 , Y, h ⟩ geometrically-constrained by C if its dynamics D C = ⟨ X,U,f ⟩ are convex-covered by C .

Geometrically-constrained systems (GCS) are a generalisation of η -finite systems, as any η -finite can be extended to a geometrically-constrained system with equal capabilities. GCSs can in fact express dynamics beyond finite-state, e.g., Construction 1. The GCS framework is presented in Appendix E.

Construction 1. Consider Linear Recurrent Dynamics with state-space X = Z , input space U = { a, b } and dynamics function f ( n, a ) = n +1; f ( n, b ) = n -1 . The space C = ( -∞ , -0 . 5) ∪

Figure 4: SSMs with state transition gates without negative eigenvalues are not capable of alternating around any separating hyperplane under iterated input. Thus, eventually the state must be mapped to a constant output. This makes such SSMs unable to e.g., implement Parity as GCSs.

<!-- image -->

( -0 . 5 , 0 . 5) ∪ (0 . 5 , ∞ ) is a convex-covering for this dynamics. We may define the output function h : C →{ 0 , 1 } to map points in ( -∞ , -0 . 5) ∪ (0 . 5 , ∞ ) to 0 and points in ( -0 . 5 , 0 . 5) to 1 . Picking initial state x 0 = 0 , we have that this GCS outputs 0 precisely when the input has the same number of a s and b s.

Connection to automata. In the case of dynamics ⟨ X,U,f ⟩ constrained by X , we recover the correspondence to Automata Theory via canonical semiautomata, and hence we can use the theorems of AAT-details in Section E.2. The next construction shows that, as a GCS, Linear Recurrent Dynamics with Mamba parametrisation realise FLIP-FLOP, unlike in the robust η -finite setting.

Construction 2. FLIP-FLOP dynamics can be realised by constrained Linear Recurrent Dynamics with diagonal state-transition gate, with entries in [ 1 4 , 3 4 ] . Take D = ⟨ X,U,f ⟩ with X = X l ∪ X h , where X l = ( -1 , 0) , X h = (0 , 1) , U = { i, l, h } and f ( x, σ ) = A σ · x + B σ where ⟨ A i , B i ⟩ = ⟨ 3 / 4 , 0 ⟩ ; ⟨ A l , B l ⟩ = ⟨ 1 / 4 , -1 / 2 ⟩ ; ⟨ A h , B h ⟩ = ⟨ 1 / 4 , 1 / 2 ⟩ . With output function X l ↦→ low and X h ↦→ high (indeed continuous), D realizes FLIP-FLOP, and X is a convex-covering of D .

In particular, given the realisation of FLIP-FLOP in Construction 2, we obtain the following:

Theorem 8. SSMs with Mamba parametrisation can recognise all star-free languages as GCSs.

Modular counting. We extend the notions of cascades to this setup, with restriction on how components depend on inputs from other components, corresponding to the idea of joining the cascade with connecting functions. Similarly, we extend the notion of aperiodic dynamics, with the modification that we require the state-sequence to be η -convergent in C , instead of X in the usual definition. Appendix E.1 explains how aperiodicity is preserved by constrained cascades in this setup. In the GCS framework, we can no longer equate aperiodic dynamics with group-free semiautomata-the GC-system in Construction 1 is aperiodic, but implements a language which is not even regular. We can still obtain more specialised expressivity results. Aperiodicity prevents a GC-system from modelling any function for which iterating the same input can alternate between distinct outputs indefinitely. We call a function F : Σ + → Γ is alternating if, for some σ ∈ Σ , the sequence ( F ( σ n ) ) n ≥ 1 changes value infinitely many times. All alternating functions are group-like. As an example, functions that perform moduloM counting are alternating.

Theorem 9. Let D be an η -finite Linear Recurrent Dynamics, with its state-transition gates having all non-negative eigenvalues. Let C be a covex-regular covering of D . Then D is aperiodic w.r.t. C .

The proof is similar to that of Theorem 6. By considering the Jordan Normal Form of the statetransition gate, we show that the state sequence cannot alternate around the separating hyperplanes of the convex components making up C . Overall, we obtain that SSMs such as Mamba are not able to implement alternating functions as geometrically-constrained systems.

## 5 Empirical Validation of Our Results

Mamba performance on star-free tasks. The experiments presented by [Sarrof et al., 2024] demonstrate that Mamba can effectively learn star-free languages with length-generalisation abilities. On the benchmark from [Bhattamishra et al., 2023], Mamba performed perfectly on all 11 star-free tasks, also on out-of-distribution input lengths. This is consistent with its expressivity described by

Figure 5: FLIP-FLOP task [Liu et al., 2023]. PCA of a trained 1-layer Mamba states for each channel: red and blue are state sequences under i0 inputs, starting from w1 and w0 respectively. After ≈ 1000 inputs, both state sequences give the same predictions on the read instruction r , incorrectly.

<!-- image -->

Theorem 8. We performed additional experiments on the dataset from [Liu et al., 2023]. It introduces the task of realizing the FLIP-FLOP by predictively modelling a sequence of instructions. We found that in the case of training 1-layer Mamba, despite achieving accuracy 1 on all validation datasets, iterating the ignore instruction indeed leads to incorrect outputs, as predicted by our results for η -finite systems, namely Theorem 7 . See Figures 5,10 and Appendix F for details.

Non-star-free tasks. Our negative results for SSMs in the η -finite setup predict that SSMs with nonnegative eigenvalues (non-negative SSMs for short) for the state-transition gate cannot implement nonstar-free tasks. The experiments performed by Sarrof et al. [2024] on Mamba with the datasets from [Bhattamishra et al., 2023] show that Mamba struggles to model non-star-free tasks. The empirical evidence presented in [Grazzi et al., 2025] similarly validates our results, with results for non-star-free languages from the Chomsky Hierarchy benchmark by Deletang et al. [2023]. Remarkably, both the Chomsky Hierarchy and Bhattamishra's benchmarks have the worst results for non-negative SSMs on languages involving modulo counting. Our negative results in the geometrically-constrained framework suggest that this is caused by the inherent geometry of the state-space for these models.

Significance of robustness. Beck et al. [2024] and Grazzi et al. [2025] evaluate their proposed architectures on the Chomsky Hierarchy benchmark [Deletang et al., 2023]. Even though, as shown in [Grazzi et al., 2025], DeltaNet with negative eigenvalues is capable of modelling the Modular Arithmetic w/o Brackets task, it falls short of perfect accuracy on all sequence lengths. On the other hand, sLSTM achieves perfect accuracy on this task, as reported by Beck et al. [2024] (although Grazzi et al. [2025] failed to reproduce these results). Theorem 3 gives a possible explanation for why linear recurrences may perform worse in practice than non-linear recurrences. This effect can also be observed for star-free tasks-we defer further discussion to Appendix F.

Beyond regular tasks. Contex-free and context-sensitive tasks remain challenging for the recent recurrent archtectures, as evident by the performance of xLSTM, DeltaNet and Mamba on the Chomsky Hierarchy benchmark, reported in [Beck et al., 2024] and [Grazzi et al., 2025]. This indicates that η -finite systems are largely a good model for the finite-precision setting. Sarrof et al. [2024] report that Mamba achieves good results for counter languages, but with limited lengthgeneralisation. We conjecture that counter-like dynamics, which are permitted in the GCS framework, are not possible for Mamba, as its dynamics are space-contracting.

## 6 Limitations

The limitations of Metric Automata Theory (MAT) in its current, initial, state of develoment revolve around three aspects, that we discuss below.

Limitations inherited from AAT. MAT allows one to employ Algebraic Automata Theory (AAT) for the purpose of analysing RNNs. However, AAT is underdeveloped in many ways, with limitations on its current ability to describe certain fine-grained expressivity aspects, which clearly transfer to MAT. A key limitation is that AAT does not focus on the complexity of the functions that connect the stateful components in a cascade, and specificically it provides no results on how the complexity of such functions influences the expressive capacity of a model. Now that our MAT makes AAT relevant for the study of RNNs, there is a new motivation in futher developing AAT .

Dynamics-dependent state space. Requiring continuity throughout means that the main work of assigning meaning to states is done in selecting the state space X . Further, the dynamics f need to have codomain X , which can make verifying constructions complicated. In the context of learning parameters for f , as the parameters vary, the state space must change accordingly, making it less straighforward to derive results regarding learning. Nonetheless, MAT already allows for indirect analysis of learning stability, via the notion of robustness, as discussed in Section 3.2.

Focus on unbounded-length expressivity. Most of our work studies the ability of RNNs to recognise languages where the length of strings is unbounded. Additional results could be proved regarding the ability of RNNs to recognise languages where strings have bounded length. Some of our notions and results-e.g., robustness or GCSs-can still be applied in this context, but otherwise MAT may require to be extended significantly.

## 7 Related Work

Our dynamical systems approach follows the framework by Knorozova and Ronca [2024a,b]. This set of results focuses on RNC + , which are cascades of 1-dimensional Elman-RNN neurons, with dynamics function f ( x, u ) = tanh( w · x + u ) having w ≥ 0 . Expressivity of RNC + in terms of regular languages is shown to be exactly the star-free languages. Their setup is not directly relatable to ours under η -finiteness, but it implicitly assumes that the state-space is compact , and uses similar convergence arguments as Sarrof et al. [2024] and Grazzi et al. [2025], combined with AAT. The authors hope to further develop the theoretical foundations of expressivity theory, and to incorporate further theories, such as the work in [Knorozova and Ronca, 2024a,b], into Metric Automata Theory.

Related expressivity results for SSMs are given in [Sarrof et al., 2024, Grazzi et al., 2025, Merrill et al., 2024], and for ReLU-activated Elman-RNNs and LSTMs in [Weiss et al., 2018]. We defer the discussion of such results to Appendix H.

## 8 Future Work

The framework we set up fills in the gaps in the existing literature in terms of general theoretical methodology, as well as understanding of empirical phenomena. At the same time, it opens up new avenues for future research in connection to automata theory, model design, and learning. We especially see robustness as being of practical interest and as a subject of future research. Next we discuss a few concrete points that are on our research agenda. First, we plan to devise additional experiments to fully understand the impact of our results on learning models-e.g., measuring robustness trade-offs between xLSTM and SSM length-generalisation on star-free tasks. Second, we plan to study how tokenization affects the models ability to perform state-tracking and realise automata transitions. For example, Grazzi et al. [2025] (paragraph under Theorem 3, page 7) note that allowing more input symbols per transition (e.g., ' 3 + 2 + 4 = 4 ') allows simpler gates to implement automata. Third, we would like to explore the potential of robustness in driving design decisions behind model architectures and training algorithms. For example, the inherent non-robustness of linear RNNs suggests that the solutions that may be learned for the model's parametrisation are very sensitive, especially when it comes to length-generalisation abilities. Fourth, we plan to employ the GCS theory for investigating the ability of RNNs to recognise languages beyond regular. A notable family of languages to consider is the one of counter languages, already mentioned in Construction 1 and in the analysis of the performance of Mamba in Section 5. Finally, we would like to use the GCS theory to further clarify length-generalisation phenomena.

## 9 Conclusions

We have presented Metric Automata Theory, an elegant and principled theory that generalises classic Automata Theory to dynamical systems, and to RNNs in particular. The fundamental notions and key properties of the theory we have described, as well as the deep understanding of several widelyadopted RNNs that we were able to provide using the theory, justify the ambition of the theory to be a unifying theory for the study of RNNs, and also dynamical systems in general. The introduced notions, e.g. of robustness, leave many exciting avenues for deeper study.

## References

Jeffrey L. Elman. Finding structure in time. Cogn. Sci. , 14(2):179-211, 1990.

- Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation , 9(8): 1735-1780, 1997.
- Kyunghyun Cho, Bart van Merrienboer, Çaglar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 1724-1734, 2014.
- Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xLSTM: Extended long short-term memory. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems (NeurIPS) , 2024.
- Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In Proceedings of the Tenth International Conference on Learning Representations (ICLR) , 2022.
- Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. CoRR , abs/2312.00752, 2023.
- Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. HiPPO: recurrent memory with optimal polynomial projections. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems (NeurIPS) , 2020.
- Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems (NeurIPS) , 2024.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems (NeurIPS) , 2017.
- Nadezda Alexandrovna Knorozova and Alessandro Ronca. On the expressivity of recurrent neural cascades. In Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI) , pages 10589-10596, 2024a.
- Nadezda Alexandrovna Knorozova and Alessandro Ronca. On the expressivity of recurrent neural cascades with identity. In Proceedings of the 21st International Conference on Principles of Knowledge Representation and Reasoning (KR) , 2024b.
- Gail Weiss, Yoav Goldberg, and Eran Yahav. On the practical computational power of finite precision RNNs for language recognition. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL) , pages 740-745, 2018.
- William Merrill, Gail Weiss, Yoav Goldberg, Roy Schwartz, Noah A. Smith, and Eran Yahav. A formal hierarchy of RNN architectures. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL) , 2020.
- Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. What formal languages can transformers express? A survey. Trans. Assoc. Comput. Linguistics , 12:543-561, 2024.
- William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought. In The Twelfth International Conference on Learning Representations , 2024.
- Michael Hahn. Theoretical limitations of self-attention in neural sequence models. Trans. Assoc. Comput. Linguist. , 8, 2020.
- Satwik Bhattamishra, Michael Hahn, Phil Blunsom, and Varun Kanade. Separations in the representational capabilities of transformers and recurrent architectures. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.

- Yash Sarrof, Yana Veitsman, and Michael Hahn. The expressive capacity of State Space Models: A formal language perspective. In Proceedings of the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS) , 2024.
- Riccardo Grazzi, Julien Siems, Arber Zela, Jörg K. H. Franke, Frank Hutter, and Massimiliano Pontil. Unlocking state-tracking in linear RNNs through negative eigenvalues. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR) , 2025.
- Juris Hartmanis and R. E. Stearns. Algebraic structure theory of sequential machines . Prentice-Hall international series in applied mathematics. Prentice-Hall, Englewood Cliffs, N.J, 1966.
- Abraham Ginzburg. Algebraic Theory of Automata . Academic Press, 1968.
- Michael A. Arbib. Theories of abstract automata . Prentice-Hall series in automatic computation. Prentice-Hall, Englewood Cliffs, N.J, 1969.
- Pál Dömösi and Chrystopher L Nehaniv. Algebraic theory of automata networks: An introduction . SIAM, 2005.
- Kenneth Krohn and John Rhodes. Algebraic theory of machines. I. Prime decomposition theorem for finite semigroups and machines. Trans. Am. Math. Soc. , 116, 1965.
- Liwen Zhang, Gregory Naitzat, and Lek-Heng Lim. Tropical geometry of deep neural networks. In Proceedings of the 35th International Conference on Machine Learning (ICML) , volume 80 of Proceedings of Machine Learning Research , pages 5819-5827. PMLR, 2018.
- Satwik Bhattamishra, Arkil Patel, Varun Kanade, and Phil Blunsom. Simplicity bias in transformers and their ability to learn sparse boolean functions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), Volume 1: Long Papers , pages 5767-5791, 2023.
- Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Exposing attention glitches with flip-flop language modeling. In Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS) , 2023.
- Gregoire Deletang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Chris Cundy, Marcus Hutter, Shane Legg, Joel Veness, and Pedro A Ortega. Neural networks and the chomsky hierarchy. In The Eleventh International Conference on Learning Representations (ICLR) , 2023.
- William Merrill, Jackson Petty, and Ashish Sabharwal. The illusion of state in State-Space Models. In Proceedings of the Forty-first International Conference on Machine Learning (ICML) , 2024.
- Stephen. Willard. General Topology. Dover Books on Mathematics. Dover Publications, Newburyport, 1st edition, 2012.
- Marcel Paul Schützenberger. On finite monoids having only trivial subgroups. Information and Control , 8(2):190-194, 1965.
- Nadezda Alexandrovna Knorozova and Alessandro Ronca. On the expressivity of recurrent neural cascades. CoRR , abs/2312.09048, 2023.
- George Cybenko. Approximation by superpositions of a sigmoidal function. Math. Control. Signals Syst. , 5(4):455, 1992.
- Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural Networks , 2(5):359-366, 1989.
- Stephen P. Boyd and Lieven. Vandenberghe. Convex optimization . Cambridge University Press, Cambridge, 2006 - 2004.
- Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR , abs/1412.3555, 2014.

- Hava T. Siegelmann and Eduardo D. Sontag. On the computational power of neural nets. J. Comput. Syst. Sci. , 50(1), 1995.
- Joe Kilian and Hava T. Siegelmann. The dynamic universality of sigmoidal neural networks. Inf. Comput. , 128(1), 1996.
- J. Nicholas Hobbs and Hava T. Siegelmann. Implementation of universal computation via small recurrent finite precision neural networks. In 2015 International Joint Conference on Neural Networks (IJCNN) , 2015.
- Stephen Chung and Hava T. Siegelmann. Turing completeness of bounded-precision recurrent neural networks. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems (NeurIPS) , 2021.

## Appendices

The appendices provide proofs of the theorems stated in the main body, as well as more detailed exposition of preliminary notions, and illustrative figures. It is structured as a suppleental body of work which can be read from top to bottom, and which gives a detailed presentation of Metric Automata Theory and its main results. While the main body gives a big picture overview of the key notions and results, the appendices aim to serve as a foundational text, showcasing how Metric Automata Theory can be used to develop new theories and draw novel insights about RNN architectures-in addition to providing full proofs of all results stated in the main body.

Appendix A provides standard preliminary notions , required for later sections and in particular for proving our results.

Appendix B presents the foundations of Metric Automata Theory (MAT) , which build on several different fields-metric spaces, dynamical systems, algebraic and classic automata theory. Also establishing novel and fundamental connections across such fields. We prove Theorem 1 in this appendix.

Appendix C introduces the novel notion of ϵ -robust dynamics, which allows us to argue about real-world floating point implementations of models. It also describes numerical and parametrisation stability properties of systems, thus going beyond the phenomena which can be described by discrete systems. We provide proofs of Theorem 2 and Theorem 5.

Appendix D employs Metric Automata Theory and its connection to Algebraic Automata Theory to show a collection of expressivity results in the η -finite setting, including Theorems 3, 4, 6 and 7.

Appendix E explores the setting of Geometrically-Constrained Systems (GCS) , in connection to the empirical length-generalisation capabilities of Mamba, which go beyond its finite-precision expressivity. We give proofs of Theorem 8 and Theorem 9.

Appendix F gives further details on the visualisation experiments we conducted to showcase the state-space collapse suffered by Mamba SSMs.

Appendix G contains technical proofs and constructions deferred from other sections, which are not necessary to fully comprehend the overall argument they are used in.

Appendix H continues the discussion of related work from Section 7, notably contrasting the frameworks of Sarrof et al. [2024] and Grazzi et al. [2025].

## A Additional Preliminaries

In this Appendix, we introduce the preliminary notions for the remainder of this work.

Section A.1 covers basic mathematical notions and notation used throughout.

Section A.2 introduces the necessary background in Metric Spaces and Topology, notably properties of compactness and path-connectedness .

Section A.3 defines the language of Dynamical Systems, which we use to describe RNNs and to build our theory.

Section A.4 shows the key Algebraic Automata Theory results and notions which we use in our work.

Finally, Section A.5 and Section A.6 cover MLPs and introduce relevant RNN architectures.

## A.1 Basic Concepts and Notation

We introduce basic mathematical concepts and notation required in later sections.

## A.1.1 Numeric Domains

We write B = { 0 , 1 } for the Boolean domain, we write N = { 0 , 1 , . . . } for the natural numbers, we write N &gt; 0 = { 1 , 2 , . . . } for the natural numbers excluding zero, we write R for the real numbers, we write R + for the positive real numbers including zero, we write R &gt; 0 for the positive real numbers

excluding zero, i.e., R &gt; 0 = R + \ { 0 } , and we write C = {⟨ a, b ⟩ | a, b ∈ R } for the complex numbers-where every pair ⟨ a, b ⟩ is to be seen as the complex number a + ib .

For i, j ∈ N with m ≤ n , we define the notation [ i..j ] := { i, i +1 , . . . , j } .

In the rest of the section, let Z be a set.

## A.1.2 Powersets

We write P ( Z ) for the powerset of Z , and we define P + ( Z ) := P ( Z ) \ {∅} .

## A.1.3 Tuples and Matrices

For n ∈ N , the set of Z -valued n -vectors , or n -tuples over Z , is Z n := {⟨ z 1 , . . . , z n ⟩ | z i ∈ Z } . We typically write an element of Z n as z = ⟨ z 1 , . . . , z n ⟩ . For m,n ∈ N , the set of Z -valued ( m × n ) -vectors , or m × n matrices over Z , is Z m × n := {⟨ z 1 , . . . , z m ⟩ | z i ∈ Z n } . We typically write an element of Z m × n as Z = ⟨ z 1 , . . . , z n ⟩ .

We use the compact notation Z [ i..j ] to specify the set Z i ×··· × Z j resulting from the Cartesian product of the sets Z i , . . . , Z j , meaning that they are contextually introduced by the notation.

## A.1.4 Sequences

A sequence over Z with indices I ⊆ N is a function s : I → Z ′ ⊆ Z ,which we commonly present as ( z i ) i ∈ I where z i = s ( i ) for every i ∈ I . A sequence is finite if so is its index set, and it is infinite otherwise. When s is an infinite sequence with index set of the form I = { m, m +1 , . . . } , we adopt a simplified notation and write the sequence as ( z i ) i ≥ m , instead of ( z i ) i ∈ I . When s is a finite sequence, the cardinality of its index set is called the length of s . The empty sequence , denoted by ε , is the sequence having length zero, i.e., the sequence with indices I = ∅ . Any finite sequence s with indices I = [ i..j ] can be presented as the list z i , . . . , z j by letting z k = s ( k ) for every k ∈ [ i..j ] ; in this case, the sequence can also be written in compact form as z [ i..j ] . We write Z ω for the set of all infinite sequences on Z , we write Z ∗ for the set of all finite sequences on Z , we write Z + for the set of all non-empty finite sequences on Z , and we write Z ℓ for the set of all sequence of a given length ℓ ∈ N -noting that this definition of Z ℓ clearly corresponds to the definition given above of Z ℓ as the set of all ℓ -tuples over Z .

We often say that a property holds eventually for a sequence ( z i ) i ≥ m if there exists m ′ ≥ m such that it holds for the sequence ( z i ) i ≥ m ′ . That is, the property holds for some tail of the sequence.

## A.1.5 Strings

A string over a finite set Σ is a concatenation (juxtaposition) of elements of Σ . Namely, a string is an expression σ 1 σ 2 · · · σ n with σ i ∈ Σ , for every i ∈ [1 ..n ] . In this context, we call Σ an alphabet , and we call each element σ i a letter or symbol of the string s . We can equivalently see a string σ 1 σ 2 · · · σ n as the finite sequence σ [1 ..n ] , following the definition of finite sequence given above, and hence apply all notions already introduced for finite sequences. In particular, we have that the length of a string σ 1 σ 2 · · · σ n is n , that ε is the empty string, that Σ ℓ is the set of all strings of given length ℓ ∈ N over alphabet Σ , that Σ ∗ is the set of all strings over alphabet Σ , and that Σ + is the set of all non-empty strings over alphabet Σ .

## A.1.6 Functions and Transformations

The image of a function f : X → Y is Im f := { f ( x ) | x ∈ X } ⊆ Y . We say that f is an identity if f ( x ) = x for every x ∈ X , and we say that f is a permutation if it is a bijection. A transformation of X is a function f : X → X where the codomain coincides with the domain. Note that every identity transformation is also a permutation, and hence it is sometimes important to distinguish permutations that are not identities by referring to them as non-identity permutations .

## A.1.7 Equivalence

For ∼ an equivalence relation on Z , the equivalence class of z w.r.t. ∼ is the set [ z ] ∼ := { z ′ ∈ Z | z ′ ∼ z } . We denote by Z/ ∼ the set of equivalence classes of Z w.r.t. ∼ .

## A.2 Metric Spaces and Topology

We follow [Willard, 2012] as a general reference for this section, revisiting the notation. Let X be a set fixed for the rest of this section.

## A.2.1 Metrics

A metric , or distance function , is a function d : X × X → R &gt; 0 that satisfies all the following properties for every x, y, z ∈ X :

- a) d ( x, y ) = 0 ⇐⇒ x = y

$$b) d ( x, y ) ≥ 0 (positivity)$$

- c) d ( x, y ) = d ( y, x ) (symmetry)

- d) d ( x, y ) + d ( y, z ) ≥ d ( x, z ) (triangle inequality)

Notable metrics, relevant to us, are the following ones.

- The Euclidean distance , or L 2 -norm distance , is defined as

<!-- formula-not-decoded -->

- The discrete metric is defined as

̸

<!-- formula-not-decoded -->

We will omit X from a metric when it is clear from the context. For instance, we will write L 2 and D for L 2 X and D X , respectively.

## A.2.2 Metric spaces

A metric space is a tuple S = ⟨ X,d ⟩ where d : X × X → R is a metric. Given metric spaces X = ⟨ X,d X ⟩ and Y = ⟨ Y, d Y ⟩ , an isometry between X and Y (or distance-preserving function ) is a bijective function f : X → Y such that, for every 1 , x 2 ∈ X , we have d X ( x 1 , x 2 ) = d Y ( f ( x 1 ) , f ( x 2 )) . When an isometry exists, the spaces X and Y are said to be isometric . Intuitively, two isometric spaces are essentially the same metric space. Notable metric spaces, relevant to us, are the following ones, for n ∈ N &gt; 0 .

- The Euclidean n -space ⟨ R n , L 2 ⟩ .
- The complex n -space ⟨ C n , L 2 ⟩ , seen as isometric to ⟨ R 2 n , L 2 ⟩ , by the following isometry:

<!-- formula-not-decoded -->

In particular, by the isometry above, all our results for Euclidean n -spaces transfer to complex n -spaces seamlessly.

We omit the metric when referring to metric spaces, since in the following sections we only consider Euclidean n -spaces ⟨ R n , L 2 ⟩ and complex n -spaces ⟨ C n , L 2 ⟩ , that are always equipped with the L 2 as described above. Thus we simply refer to them as R n and C n , respectively.

A subspace ⟨ Y, d Y ⟩ of ⟨ X,d X ⟩ is a metric space with Y ⊆ X and d Y given by restriction of d X to Y × Y .

We define the open ball B X ( x, r ) and closed ball B X ( x, r ) at x ∈ X of radius r ≥ 0 in ⟨ X,d ⟩ as the set of points in X with distance δ &lt; r and δ ≤ r from x , respectively:

<!-- formula-not-decoded -->

A subspace ( Y, d Y ) of ( X,d X ) is a metric space with Y ⊆ X and d Y given by restriction of d X to Y × Y . We say that a subspace S ⊆ X is bounded , if there is some x ∈ X and ∞ &gt; M ≥ 0 s.t. S ⊆ B X ( x, M ) . We call a subspace S ⊆ X is open in X if for all s ∈ S there is some ϵ s &gt; 0 s.t. B X ( s, ϵ s ) ⊆ S . S is closed in X if X \ S is open in X .

Example 1 . The open intervals ( a, b ) and ( a, ∞ ) are open in R (with the usual metric). The closed interval [ a, b ] is closed in R . The subspace { 0 , 2 -n : n ∈ N } is closed in R , while { 2 -n : n ∈ N } is neither closed nor open in R . ■

## A.2.3 Topology

The notion of open subspaces in terms of open balls defines a topology on any metric space, which determines what functions are continuous . Formally, a topological space is a tuple ( S, T ) , with S being the underlying set, and T ⊆ P ( S ) being the collection of open sets, such that S and ∅ , the union of any collection of open sets is open, and the intersection of any finite collection of open sets is open. The open sets definition in terms of open balls for a metric space satisfies these properties. Many aspects of Metric Automata Theory could be easily restated in the language of Topology Theory, but we choose a more concrete setting, to make it more accessible.

Intuitively, the closed subspaces of X are precisely the ones which contain all their limit points, i.e. if ( x n ) n ≥ 1 ⊆ S converges to some limit l ∈ X , then l ∈ S .

Fact A.2.1. For a metric space X , a subset S ⊆ X is closed iff for all sequences ( x n ) n ≥ 1 ⊆ S converging to l ∈ X we have that l ∈ S . (see §10, Cor. 10.5 of Willard [2012], as every metric space is first-countable)

Note that the notion of opennes/closeness is not inherent to the subspace S : it also depends on the superspace X , since the definition involves balls in X . In fact, any subspace S ⊆ X is by definition both open and closed as a subspace of itself, regardless of whether is open or closed in X . Any time we use opennes or open balls, we need to excercise caution and be clear which space the openness is referring to.

Example 2 . Consider M = R 2 and X = R ×{ 0 } = { ( x, 0) ∈ R 2 : x ∈ R } . ( -1 , 1) ×{ 0 } ⊆ X is an open ball at (0 , 0) of radius 2 in X , and thus an open set. However, it is not even an open set in M ! For any ϵ &gt; 0 we have || (0 , 0) -(0 , ϵ ) || = ϵ , but (0 , ϵ ) / ∈ S , and so no open X -ball centred at (0 , 0) is wholly contained in S . ■

In fact, any subspace S ⊆ X is by definition both open and closed as a subspace of itself, regardless of whether is open or closed in X .

A continuous function f : ( M,d ) → ( M ′ , d ′ ) is the a set function f : M → M ′ such that for all sequences ( x n ) n ≥ 1 ⊆ M converging to some x ∈ M , the mapped sequence ( f ( x n ) ) ⊆ M ′ converges to f ( x ) ∈ M ′ . The ϵ -δ definition of continuity, as well as the topological definition of continuity ( Y ⊆ M ′ open = ⇒ f -1 ( Y ) ⊆ M open) are equivalent in the metric space setting.

Example 3 . Let S be a subspace of X . Then the inclusion map ι : S → X , given by set-theoretical inclusion S ⊆ X , is continuous. ■

The topological definition of continuity makes clear the following:

Fact A.2.2. All functions f : ( M,d ) → ( M ′ , d ′ ) are continuous for a discrete metric space ( M,d ) .

Next, we introduce two elementary notions in Topology and Metric Space Theory: compactness and path-connectedness .

## A.2.4 Compactness

Definition 9. A space X is called compact if all coverings of X by open subsets of X admit a finite subcover. For metric spaces, equivalently X is (sequentially) compact, if all sequences in X have a subsequence converging to a limit in X (see 17G.3 of Willard [2012]). ■

The following is a characterization of compact subspaces of R d .

Fact A.2.3. (Heine-Borel) X ⊆ Ω is a compact subspace iff. X is a bounded, closed subset of R d (see 17.9 of Willard [2012] ).

Example 4 . Subspaces [ a, b ] , { a } , { 0 , 2 -n : n ∈ N } are compact in R . ( a, b ) , { 2 -n : n ∈ N } are not closed, and so they are not compact. R is not bounded, and so it is not compact. ■

Turns out that compactness, unlike openness, is inherent to the subspace, as demonstrated by the following theorem:

Fact A.2.4. A continuous image of a compact space is compact (see 17.7 of Willard [2012] )

Finally, Tychonoff Theorem tells us that compactness is a property which is preserved by cartesian products.

Fact A.2.5. (Tychonoff) The cartesian product of two compact spaces is compact (see 17.8 of Willard [2012] )

## A.2.5 Path-connectedness

Definition 10. A path in X from a to b is a continuous function γ : [0 , 1] → X such that γ (0) = a and γ (1) = b . A space X is called path-connected if for all a, b ∈ X there is a path from a to b . ■

Path-connectedness partitions the space into components, which we will later think of as atomic parts of the state-space for a dynamical system. - any continuous decoder assigning discrete symbols to the state-space must be constant on a path-connected component, see Lemma 22.

See Section 27D of Willard [2012] for the following:

Fact A.2.6. The relation ∼ on X given by a ∼ b ⇐⇒ there is a path from a to b in X is an equivalence. The equivalence classes of ∼ are the maximal path-connected subspaces of X .

Example 5 . Any convex subspace of R d is path-connected, in particular open and closed R d -balls are path-connected. ( -1 , 0) ∪ (0 , 1) has 2 path-connected components: ( -1 , 0) and (0 , 1) . ■

Just like compactness, path-connectedness is an inherent property of the subspace, and is preserved by Cartesian products (see 27B of Willard [2012]):

Fact A.2.7. A continuous image of a path-connected space is path-connected.

Fact A.2.8. The cartesian product of two path-connected spaces is path-connected.

## A.3 Dynamical Systems

Following Knorozova and Ronca [2024a], we adopt dynamical systems as an general formalism to describe all systems that operate by maintaining a state recurrently. This allows for treating such systems in a uniform way despite their differences. In this work specifically, we will use dynamical systems to formalise Finite Automata and several RNN architectures in Section A.6.

Definition 11. A (dynamical) system is a tuple S = ⟨ X,U,f,x 0 , Y, h ⟩ , where X is the state space , U is the input space , f : X × U → X is the dynamics function , x 0 ∈ X is the initial state , Y is the output space and h : X × U → Y is the output function. We have that X,U,Y are metric spaces, and f, h are continuous . In our analysis it will be useful to refer to the tuple D = ⟨ X,U,f ⟩ as the dynamics of S , allowing us to focus on just the state transitions.

Given x 0 ∈ X , D defines a map from sequences of inputs ( u n ) n ≥ 1 ⊆ U to sequences of states ( x n ) n ≥ 0 ⊆ X , given by

<!-- formula-not-decoded -->

With this, we can define the state-sequence function D : X × U ∗ → X as

<!-- formula-not-decoded -->

S defines a map from sequences of inputs ( u n ) n ≥ 1 ⊆ U to sequcences of states ( x n ) n ≥ 1 ⊆ X and sequences of outputs ( y n ) n ≥ 1 ⊆ Y , given by

<!-- formula-not-decoded -->

Hence we say that S defines the function U + → Y , with S ( u [1 ..n ] ) = y n . In the special case that h is independent of U , we may define S ( ϵ ) = h ( x 0 ) , extending the definition to S : U ∗ → Y . ■ Lemma 10 (State continuity) . Let S = ⟨ X,U,f ⟩ be a dynamics, and for input sequence ( u n ) N n ≥ 1 ⊆ U and x 0 ∈ X let ( x n ) N ⊆ X be the sequence of states

<!-- formula-not-decoded -->

Then x n is a continuous function of x 0 , u 1 , . . . , u n for all n ∈ 1 ..N . Consequently y n = h ( x n , u n ) is also a continuous function of x 0 , u 1 , . . . , u n , for any continuous h .

Proof. By induction. Writing x n ( u 1 , . . . , u n ) we have that

<!-- formula-not-decoded -->

is also a continuous function of x 0 , u 1 , . . . , u n +1 .

n ≥ 1

The formalism of cascades provides a flexible way to describe dynamical systems consisting of subsystems forming an acyclic network. Their flexibility will allows us, e.g., to consider not only feed-forward layers of SSMs as in Grazzi et al. [2025], Sarrof et al. [2024], but also more complex architectures with, e.g., blocks in parallel, and mixes of different types of neurons.

Definition 12. A feed-forward cascade C is a form of dynamics ⟨ X,U,f ⟩ with X = X 1 ×···× X n , and dynamics function of the form

<!-- formula-not-decoded -->

We may see C as consisting of dynamics D 1 , . . . , D n where

<!-- formula-not-decoded -->

and write C = D 1 ⇝ · · · ⇝ D n .

■

Thus, the cascade is evaluated in a feedforward fashion: on input u , first the state of D 1 is updated, then for all subsequent components D i , the state of D i is updated based on u and the updated states of D 1 , . . . , D i -1 . This differs from some recurrent neural network literature, where D i is updated based on u and the initial states of D 1 , . . . , D i -1 , i.e. the update happens at the same time for all components. We refer to such cascades as serial cascades .

Definition 13. A serial cascade C is a form of dynamics ⟨ X,U,f ⟩ where states are of the form X = X 1 ×··· × X n , and the dynamics function is of the form

<!-- formula-not-decoded -->

We may see C as consisting of dynamics D 1 , . . . , D n where

<!-- formula-not-decoded -->

and write C = D 1 ⋉ · · · ⋉ D n .

■

Serial cascading can be achieved with feed-forward cascades, and the distinction between the two is irrelevant for our purposes. For details, see Appendix G.2.

In further sections, it will be useful to allow connection functions in a cascade, transforming the inputs between components. It will not alter the expressivity results, but it allows us to e.g. define one canonical FLIP-FLOP dynamics, rather than a family of FLIP-FLOP-like dynamics for every possible input and output set.

Definition 14. For dynamics D 1 , D 2 with D i = ⟨ X i , U i , f i ⟩ for all i ∈ [1 .. 2] , and for continuous i : U → U 1 and g : U × X 1 → U 2 , we define the feed-forward cascade with input i and connection g , written i ⇝ D 1 g ⇝ D 2 , and the serial cascade with input i and connection g , written i ⋉ D 1 g ⋉ D 2 as the dynamics ⟨ X 1 × X 2 , U, f ⟩ , ⟨ X 1 × X 2 , U, f ′ ⟩ respectively, where f and f ′ are given by

<!-- formula-not-decoded -->

and f ′ ( ⟨ x 1 , x 2 ⟩ , u ) = 〈 f 1 ( x 1 , i ( u )) , f 2 ( x 2 , g ( u, x 1 ) 〉 . Note that for U 2 = U 1 × X 2 and g = id , we recover the usual notion of feed-forward cascade and serial cascade/. ■

For dynamics D = ⟨ X,U,f ⟩ and continuous function g : Z → U , we define the dynamics with input function D g = 〈 X,Z, ( x, z ) ↦→ f ( x, g ( z ) )〉 . With the notation from the previous definition, note that D 1 ,i ⇝ D 2 ,g ≡ i ⇝ D 1 g ⇝ D 2 , and D 1 ,i ⋉ D 2 ,g ≡ i ⋉ D 1 g ⋉ D 2 . In our expressivity results we will not care about how the dynamics of a neuron interpret the input function, only about the induced transformations of the state-space. Thus, in further sections in proofs we will only consider feed-forward cascading without connection functions, without loss of generality, in order to simplify notation. Further discussion about serial cascades and connecting functions is deferred to Appendix B.5. The next lemma shows the intuitive fact, that it does not matter in which order we "connect" the components of the cascade. In the following propositions, it will be useful to view a cascade D 1 ⇝ · · · ⇝ D n as ( D 1 ⇝ · · · ⇝ D n -1 ) ⇝ D n for inductive proofs.

Definition 15. For dynamics D 1 , D 2 , where D i = ⟨ X i , U i , f i ⟩ for all i ∈ [1 .. 2] , write D 1 ≡ D 2 if X 1 = X 2 , U 1 = U 2 and f 1 = f 2 .

Lemma 11. The cascading operation is associative , i.e. we have

<!-- formula-not-decoded -->

where ' ≡ ' is as introduced in Definition 15

Proof. Say we have D i = ⟨ X i , U × X [1 ,i ] , f i ⟩ for i ∈ 1 .. 3 . Both the LHS and RHS dynamics have state space X 1 × X 2 × X 3 and input space U . Consider a state ⟨ x 1 , x 2 , x 3 ⟩ ∈ X 1 × X 2 × X 3 and input u ∈ U .

Write x ′ 1 = f 1 ( x 1 , u ) , x ′ 2 = f 2 ( x 2 , ⟨ u, x ′ 1 ⟩ ) , x ′ 3 = f 3 ( x 3 , ⟨ u, x ′ 1 , x ′ 2 ⟩ ) . Also write f 23 for the dynamics function of D 2 ⇝ D 3 and f 12 for the dynamics function of D 1 ⇝ D 2 . Then the state update of the LHS system is as follows:

<!-- formula-not-decoded -->

where the second line follows from the definition of cascade dynamics for D 2 ⇝ D 3 , and the third line follows from associativity of the cartesian product. Analogously,

<!-- formula-not-decoded -->

Now, we have x ′ 12 = f 12 ( ⟨ x 1 , x 2 ⟩ , u ) = 〈 x ′ 1 , f 2 ( x 2 , ⟨ u, x ′ 1 ⟩ ) 〉 = ⟨ x ′ 1 , x ′ 2 ⟩ , and so

<!-- formula-not-decoded -->

Thus both ways of composing the dynamics D 1 , D 2 , D 3 results in the same dynamics function.

## A.4 Algebraic Automata Theory (AAT)

We present an extended version of the background on Algebraic Automata Theory given in the preliminaries of the main body.

Algebraic Automata Theory (AAT) allows for studying finite automata through the lens of algebraic notions such as semigroups and groups, c.f. [Hartmanis and Stearns, 1966, Ginzburg, 1968, Arbib, 1969, Dömösi and Nehaniv, 2005]. Its fundamental theorem is the seminal Prime Decomposition Theorem by Krohn and Rhodes [1965], that shows how every semiautomaton can be decomposed into a cascade of elementary prime semiautomata. One prime semiautomaton is the flip-flop , that describes the elementary system with the ability to store and manipulate one bit of information.

Definition 16. The flip-flop is the two-state semiautomaton defined as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

AAT often focuses on state transformations rather than on the transition function δ of an automaton. State transformations are the functions δ σ ( q ) := δ ( q, σ ) obtained by fixing an input σ . They allow us to characterise semiautomata in terms of semigroups and groups. In particular, the transitive closure of the state transformations of an automaton forms a semigroup, and a monoid or group in special

cases. From this algebraic point of view, the flip-flop is characterised by the flip-flop semigroup, which is in fact given by the set of state transformations of FLIP-FLOP. All the other primes are characterised by finite simple groups, and for this reason they are called group-like . Specifically, their state transformations form a finite simple group.

Automata whose semiautomaton can be decomposed purely into flip-flops are called group-free , and they play a central role in our theory and in general, due to the following theorem whose proof also involves the celebrated theorem by Schützenberger [1965]) on aperiodic semiautomata, cf. [Ginzburg, 1968].

Theorem 12. The star-free languages is the class of languages recognised by groupfree automata.

All other automata, that do not admit the above decomposition, are called non-group-free , since their prime decompositions always include group-like semiautomata. They admit the following characterisation in terms of state transformations, relevant to our results.

Theorem 13. (Lemma 9 of [Knorozova and Ronca, 2024a] 1 ) If a semiautomaton ⟨ Q, Σ , δ ⟩ is not group-free, then there exist Q ′ ⊆ Q and σ ∈ Σ such that the state transformation δ σ : Q → Q is a non-identity permutation on Q ′ .

Our theory will extend the applicability of AAT to the study of general dynamical systems. And in particular to analyse the structure of such systems using algebraic means like group theory. A notion from AAT that is key to our results is the notion of realisation for Mealy machines (cf. Definitions 1.14 and 1.15 of [Hartmanis and Stearns, 1966]).

Realisation describes how a machine can imitate another machine after a renaming of inputs and outputs-noting that actual names of inputs and outputs are not important in order to characterise what functionalities a machine is fundamentally able to implement.

We recall that a Mealy machine is a tuple ⟨ Q, Σ , δ, Γ , θ ⟩ where ⟨ Q, Σ , δ ⟩ is a semiautomaton, Γ is an output alphabet, and θ : Q × Σ → Γ is an output function.

A Mealy machine defines the mapping Q × Σ + → Γ given by

<!-- formula-not-decoded -->

where D M is the semiautomaton of M .

Given a (finite) automaton A = ⟨ Q, Σ , δ, q 0 , Γ , θ ⟩ , the associate Mealy machine M A = ⟨ Q, Σ , δ, Γ , θ ⟩ is obtained by dropping the initial state from automaton A .

Given a semiautomaton D A = ⟨ Q, Σ , δ ⟩ we define its canonical Mealy machine as

<!-- formula-not-decoded -->

Definition 17 (Definitions 1.14 and 1.15 of [Hartmanis and Stearns, 1966]) . If M = ⟨ Q, Σ , δ, Γ , θ ⟩ and M ′ = ⟨ Q ′ , Σ ′ , δ ′ , Γ ′ , θ ′ ⟩ are Mealy machines, then the triple ( α, ι, ζ ) is called an assignment of M into M ′ when the functions

<!-- formula-not-decoded -->

satisfy the two conditions below for every q ∈ Q , every q ′ ∈ α ( q ) , and every σ ∈ Σ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

′ ′ ■

If an assignment of M into M exists, then M is said to be a realisation of M .

The following results tells us how a machine M ′ that is a realisation of another machine M actually implements its behaviour. Any trajectory through M factors through M ′ , with ι and ζ acting as the encoder and decoder, respectively, and with α providing an initial state to start from.

Theorem 14. (Theorem 1.5 in §1.3 of [Hartmanis and Stearns, 1966]) If M ′ = ⟨ Q ′ , Σ ′ , δ ′ , Γ ′ , θ ′ ⟩ is a realisation of M = ⟨ Q, Σ , δ, Γ , θ ⟩ through an assignment ( α, ι, ζ ) , then for all x 0 ∈ Q , w ∈ Σ + , and x ′ 0 ∈ α ( x 0 )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

1 Lemma 9 of [Knorozova and Ronca, 2024a] can be found in the appendix of its extended version [Knorozova and Ronca, 2023].

We will use the following version of the Krohn-Rhodes decomposition theorem, presented in [Hartmanis and Stearns, 1966], which uses the notion of realisability.

Theorem 15. (Theorem 7 . 8 , § 8 , Hartmanis and Stearns [1966]) Let M be a Mealy machine, with group-free semiautomaton. Then M can be realised by a machine with serial cascade dynamics, consisting of FLIP-FLOP components.

## A.5 Multilayer Perceptrons

A Multilayer Perceptron (MLP) is a tuple

<!-- formula-not-decoded -->

where d ∈ N &gt; 0 is called the depth or number of layers , n = ⟨ n, n 2 , n 3 , . . . , n d , m ⟩ is called architecture , U ⊆ R n is the input domain, Y ⊆ R m is the output domain (or codomain), α : R → R is called activation function , β : R → R is called activation function of the last layer , W = ⟨ W 1 , . . . , W d ⟩ with W i ∈ R n i × n i +1 called weight matrices , and b = ⟨ b 1 , . . . , b d ⟩ with b i ∈ R m i called bias vectors . Then, N defines the function f : U ⊆ R n → U ⊆ R m given by the composition f 1 ◦ · · · ◦ f d of the functions f i : R n i → R n i +1 defined as

<!-- formula-not-decoded -->

We often identify N with the function f , and hence see the network as a function N : U → Y . The functions f i are called layers , with the first layer f 1 called the input layer , the last layer f d called the output layer , and the other layers called hidden layers . The (maximum) width of N is max { n 2 , . . . , n d } . Typical choices for the activation function α are sigmoid( x ) := 1 1+exp( -x ) and the Rectified Linear Unit ReLU( x ) := max { 0 , x } . The same choices are valid for the last-layer activation function β ; however, as it computes the output of the network, it is often specialised by choosing β to be: the identity function (e.g., for regression tasks), sigmoid (e.g., for binary classification), softmax (e.g., for modelling distributions).

MLPs are universal approximators as long as their activation function α is non-polynomial , as established by several well-known Universal Approximation Theorems for feedforward neural networks, cf. [Cybenko, 1992, Hornik et al., 1989].

Theorem 16 (Universal Approximation) . Let α be any non-polynomial activation function. Additionally, let X ⊆ R n be compact, and let f : X ⊆ R n → Y ⊆ R m be continuous. For every ϵ &gt; 0 , there exists a 2-layer MLP N with activation function α , and identity as its last-layer activation function, such that the following inequality holds:

<!-- formula-not-decoded -->

Note that ReLU and sigmoid are non-polynomial activation functions.

In light of the above theorem, in the rest we will focus on MLPs having non-polynomial activation function α , as well as identity as their last-layer activation function β . This will be relevant in all expressivity results for RNNs whose architecture includes MLPs-as also discussed in Section A.6.

## A.6 Recurrent Neural Network Architectures

We present the Recurrent Neural Network (RNN) architectures studied in the following sections.

Classical RNNs are networks of neurons with hidden state h ∈ R d state and update rule of the form

<!-- formula-not-decoded -->

where ϕ is commonly a linear transformation composed with a non-linearity, like sigmoid or tanh . We model such neurons as dynamical systems, with hidden state taking values in X , and inputs taking values in U . The hidden state of the neuron at step t may be available to other neurons in the network as part of their input at time t +1 .

In modern Machine Learning applications, notably NLP, the networks are in the form of feed-forward connections, with learnable transformations between the neurons. Also some neurons may appear in

parallel, and some neurons might additionally include residual connections. Most generally, we can model such RNNs as acyclic networks, and for nodes N 1 , . . . , N L consider the connection functions ψ i,j , describing the transformation which is applied to the value going from neuron N i to neuron N j . The network input also may be given to N i , after going through some transformation ι i . As the network is acyclic, we may assume that there are no connection functions ψ i,j for i &gt; j . Finally, the inputs to N i are accumulated by some α i . Now, we may express the network as a feed-forward cascade D 1 ⇝ · · · ⇝ D L , with D i = ⟨ X i , U × X [1 ..n ] , f i ⟩ , where X i is the state-space of neuron N i , U is the input space of the network, and f i is given by

<!-- formula-not-decoded -->

This is how our framework allows to pull the details about the state-less transformations of the input or state-space into the dynamics function.

Classical (Vanilla) RNNs. Vanilla RNNs are networks where the state is updated through a linear combination of the previous state and current input, followed by the application of a non-linear activation function. A prominent example of a vanilla RNN architecture is the Elman RNN , which is given by dynamics D = ⟨ X,U,f ⟩ with state space X ⊆ R state , input space U ⊆ R input , and dynamics function

<!-- formula-not-decoded -->

where A X ∈ R state × state is a matrix defining a linear transformation of the state, A U ∈ R state × input is a matrix defining a linear transformation of the input, and b ∈ R state is the bias vector.

State Space Models. State Space Models (SSMs) are a family of models based on linear recurrence with particular parametrisation. Notable ones are Mamba [Gu and Dao, 2023] and S4 [Gu et al., 2020].

To model linear recurrence in general, we introduce Linear Recurrent Dynamics , defined as dynamics D = ⟨ X,U,f ⟩ , with state space X ⊆ K d state , input space U = K d input , where K = R or K = C , and with dynamics function

<!-- formula-not-decoded -->

where A ( u ) ∈ K d state × d state is the state-transition gate and B ( u ) ∈ K d state is the input gate .

SSM architectures often combine linear recurrence blocks with linear projections, non-linearities, residual connections and convolutions. Our theory can easily model such setups with cascade compositions-introduced in Section 2. Consider the Mamba block:

<!-- formula-not-decoded -->

where the input sequence u [1 ..n ] ∈ U + and output sequence o [1 ..n ] ∈ Y + are processed sequentially, each linear i is a linear projection, σ is a non-linearity, SSM is an SSM block, Conv is a causal convolution, and × is element-wise multiplication. Only Conv and SSM are stateful transformations here. In Figure 6, we present it in the form of a system with cascade dynamics.

We introduce a general class of dynamics as an abstraction for convolution blocks.

Definition 18. Finite Context Dynamics (FCDs) with context length ℓ are dynamics D = ⟨ X,U,f ⟩ such that their state depends only on the most recent ℓ inputs. That is, in view of Lemma 10, there is a continuous function C : U ℓ → X such that

<!-- formula-not-decoded -->

for all x ∈ X and w ∈ U ∗ with | w | ≥ ℓ , where w -i is the i -th-to-last element of w .

xLSTM. The recently introduced model xLSTM [Beck et al., 2024] is a successor of the LSTM architecture [Hochreiter and Schmidhuber, 1997], and it achieves performance competitive with transformer architectures. It makes use of both non-linear and linear recurrences. xLSTM introduces two types of blocks: sLSTM and mLSTM. In this work we will focus on the sLSTM block.

Figure 6: The feedforward cascade structure of a Mamba block. Only Conv and SSM are stateful, so the cascade has 2 components. Structure on the left as it is presented in [Gu and Dao, 2023].

<!-- image -->

The state space of a sLSTM is R 3 , and the input space is R d for some d ≥ 1 . The dynamics function of the form ( ⟨ c, n, h ⟩ , u ) ↦→ 〈 f c ( ⟨ c, n, h ⟩ , u ) , f n ( ⟨ c, n, h ⟩ , u ) , f h ( ⟨ c, n, h ⟩ , u ) 〉 , where

<!-- formula-not-decoded -->

where each l s : s ∈ o, i, z, f is a function of the form w t s · u + r s · h + b s , for w s ∈ R d , r s , b s ∈ R , ψ is either exp or σ , and φ is tanh .

## B Foundations of Metric Automata Theory

In this appendix, we develop the key notions of Metric Automata Theory within the η -finiteness framework.

In Sections B.1 and B.2 we introduce the basic properties of η -finite spaces and dynamics.

In Section B.3 we develop the correspondence between η -finite systems and finite automata, which is crucial to unlocking the powerful theorems of AAT. We provide the proof for Theorem 1.

In Sections B.4 and B.5 we import the notion realizability to continuous systems, via the correspondence with automata, and use it to translate algebraic decomposition theorems into the setting of η -finiteness.

## B.1 The Notion of η -Finiteness

We begin by introducing η -finiteness , which is a central notion of Metric Automata Theory and our novel finite-precision framework.

Definition 19. Let X ⊆ Ω for some d ≥ 1 . Call X η -finite if it is a finite union of compact, path-connected sets.

Immediately from the definition we have that an η -finite space is necessarily compact -in the case of metric spaces, finite union of bounded, closed sets is bounded and closed. The next result resolves the technicality, that the defining sets in the union of a η -finite X need not be disjoint.

Lemma17. Let X be η -finite. Then X has finitely many path connected components, say X 1 , . . . , X n , and each of X i is compact. We shall refer to them as the η -components of X .

Proof. By def, X = ⋃ N i =1 Y i for some compact and path-connected subsets. By induction on N : If N = 1 , then the claim is immediate. Now, consider the inductive hypothesis for N ≥ 2 , that X ′ = ⋃ N -1 i =1 Y i has finitely many path connected components X 1 , . . . , X n , each compact. The path connected components of X are then unions of elements from { X 1 , . . . , X n , Y N } . Each of these sets is compact, and so each such finite union is compact: clearly it is still bounded, and a finite union of closed sets is still closed.

Example 6 . Any finite alphabet is η -finite , with each symbol in a separate η -component. The subspace [ -2 , 1] ∪{ 2 } ⊆ R is η -finite. The subspace ( -2 , 1) ∪{ 2 } is not η -finite, since it is not compact. The subspace { 0 , 2 -n : n ∈ N } is compact but not η -finite, since it is not a finite union of path-connected sets. ■

Both compactness and path-connectedness are preserved by continuous mappings and (finite) Cartesian products, see Facts A.2.4, A.2.5, A.2.7, and A.2.8. This gives us the corresponding results for η -finite spaces.

Lemma 18. Continuous image of an η -finite space is η -finite.

Proof. Write X = ⋃ N i =1 X i for path-connected, compact sets X i . Let f : X → Y be continuous. We have:

<!-- formula-not-decoded -->

By Facts A.2.4 and A.2.7, each f ( X i ) is compact and path-connected. Thus by definition f ( X ) is η -finite.

Lemma 19. The Cartesian product X × Y space of η -finite spaces is η -finite. The η -components of X × Y are the products of η -components of X and η -components of Y .

Proof. Let X 1 , . . . , X n and Y 1 , . . . , Y , be the C-components of X,Y respectively. We have X = ⋃ n i =1 X i , Y = ⋃ m j =1 Y j and so

<!-- formula-not-decoded -->

By Facts A.2.8 and A.2.5 each X i × Y j is path-connected. Therefore by def. X × Y is η -finite. Moreover, the η -components of X × Y are unions of the products X i × Y j . Now, fix i ∈ [1 ..n ] , j ∈ [1 ..j ] . Let Z be the η -componentof X × Y containing X i × Y j . consider the projection map π X : X × Y → X . As the projection is continuous, the image, π X ( Z ) is path-connected in X by Fact A.2.7. Moreover, X i ∈ π X ( Z ) . Thus, as X i is a maximal path-connected subspace of X , we have X i = π X ( Z ) . Similarly, considering the projection π Y : X × Y → X , we have Y j = π X ( Z ) . Since X i × Y j ⊆ Z , we therefore must have X i × Y j = Z . Therefore X × Y has finitely many η -components, and they are the products of η -components of X and η -components of Y .

Lemma 20. Let X be η -finite, with η -component X 1 , . . . , X n . For some δ &gt; 0 we have

̸

<!-- formula-not-decoded -->

Proof. It is sufficient to show this in the case that X has two η -components, say X 1 , X 2 . Define f : X 1 × X 2 → R ≥ 0 by f ( x 1 , x 2 ) = ∥ x 1 -x 2 ∥ . This is continuous, and so Im f is compact, as X 1 × X 2 is compact. Since X 1 , X 2 are disjoint, 0 / ∈ Im f . Thus 0 is not a limit point of Im f , and so for some δ &gt; 0 we have that [0 , δ ) ⊈ Im f .

Corollary 21. Let X ⊆ Ω be η -finite and ( x n ) n ≥ 1 ⊆ X converge in Ω . Then ( x n ) n ≥ 1 is eventually contained in a single η -component of X .

Lemma 22. Let X be an η -finite space and Σ a finite alphabet. Then a function f : X → Σ is continuous if and only if it is constant on the η -components of X

Proof. ( ⇐ ) Suppose f : X → Σ is constant on η -components of X . Let ( x n ) n ≥ 1 ⊆ X converge to x ∈ X . Then by Lemma 20, ( x n ) n ≥ 1 is eventually contained in the same η -component as x . Thus f ( x n ) = f ( x ) eventually, in particular f ( x n ) → f ( x ) as n →∞ . Hence f is continuous.

( ⇒ ) If f is continuous, then it maps η -component of X to path-connected subspaces of Σ . Therefore f must be constant on η -components.

## B.2 Dynamical Systems and η -Finiteness

Definition 20. We say that dynamics ⟨ X,U,f ⟩ are η -finite if both X and U are η -finite. A system S is η -finite if its dynamics are η -finite.

Example 7 . Take X = [ -1 , -1 / 2] ∪ [1 / 2 , 1] and U = {-1 , 0 , 1 } . The both X and U are η -finite. Define f : X × U → X by:

<!-- formula-not-decoded -->

Thus under input u = 0 the dynamics function performs the identity transformation on X , and under inputs u = 1 , -1 , X is mapped to 1 , -1 respectively. The dynamics D = ⟨ X,U,f ⟩ is η -finite. ■

Note, that by Lemma 19, a cascade of η -finite components is itself η -finite.

Lemma 23. Let D = ⟨ X,U,f ⟩ be a η -finite dynamics, and h : X × U → Y be continuous. Then the image of h , Im h ⊆ Y , is η -finite.

Proof. Immediately follows from Lemma 18.

Lemma 24 (Path-connected ⇒ same state) . Let D = ⟨ X,U,f ⟩ be a dynamics, and consider x 0 , x ′ 0 ∈ X , and input sequences ( u n ) n ≥ 1 , ( u ′ n ) n ≥ 1 ⊆ U , and the corresponding state sequences ( x n ) n ≥ 1 , ( x ′ n ) n ≥ 1 ⊆ X . Suppose that for all n ≥ 1 , u n ∼ U u ′ n , and x 0 ∼ X x ′ 0 . Then for all n ≥ 1 we have that x n ∼ X x ′ n , i.e.,

<!-- formula-not-decoded -->

Proof. Let n ≥ 1 . By 10, we have that there is for each n a continuous function x n ( x 0 , u 1 , ..u n ) determining the n -th state. Now, since each pair u i , u ′ i for i ∈ 1 ..n is path-connected in U , we have

that ⟨ u 1 ..n ⟩ and ⟨ u ′ 1 ..n ⟩ are path-connected in U n - the path connecting them applies the corresponding 1-d paths pointwise. Thus by continuity of x n ,

<!-- formula-not-decoded -->

are path-connected in X .

Corollary 25. Let S = ⟨ X,U,f,x 0 , Y, h ⟩ be a η -finite system, and let us consider input sequences ( u n ) n ≥ 1 , ( u ′ n ) n ≥ 1 ⊆ U such that for all n u n and u ′ n are in the same path-connected component. Then the corresponding state sequences ( x n ) n ≥ 1 , ( x ′ n ) n ≥ 1 ⊆ X , and the corresponding output sequences ( y n ) n ≥ 1 , ( y ′ n ) n ≥ 1 ⊆ Y are such that for all n x n and x ′ n are in the same path-connected component of X and y n and y ′ n are in the same path-connected component of Im h

In light of the above results, we introduce the notion of equivalent sequences, for convenience in later proofs.

Definition 21. Let X be a η -finite space. Call sequences ( x n ) n ≥ 1 , ( x ′ n ) n ≥ 1 ⊆ X equivalent , if for each n we have that x n and x ′ n are in the same component of X . Call these sequences eventually equivalent , if they have equivalent tail sequences.

Overall, the notions of η -finiteness and η -component have very favourable theoretical properties . Any continuous mapping f : X → Y , with X and Y η -finite, is guaranteed to map every element of an η -component of X into a single η -component of Y .

In the case of η -finite systems, this means that the dynamics function acts on the η -components of the state-space (referred to as η -states) in the same way for each input within an η -component of the input-space (referred to as η -input). Moreover, every point within an η -component of the output function image (which is always η -finite), must be decoded as the same alphabet symbol. We formalize these properties in the following section.

## B.3 Representing η -Finite Systems as Automata and Proof of Theorem 1

For set A and equivalence ∼ on A , write A ⧸ ∼ for the set of its equivalence classes. For a ∈ A write [ a ] A for the ∼ -equivalence class containing a .

For η -finite spaces A , we will write A for the set A ⧸ ∼ A , with ∼ A being the path-connectedness equivalence. For X,Y being η -finite spaces, we have by Lemma 19 that X × Y = X × Y .

Definition 22. Any η -finite dynamical system S = ⟨ X,U,f,x 0 , Y, h ⟩ defines its canonical automaton

<!-- formula-not-decoded -->

Similarly, any η -finite dynamics D = ⟨ X,U,f ⟩ defines its canonical semiautomaton D A = ⟨ X,U,f ⟩ . ■

Note that by Lemma 23, Im h is indeed η -finite. f : ( X ) × ( U ) → ( X ) is defined as [ x ] ∼ X , [ u ] ∼ U ↦→ [ f ( x, u )] ∼ X . h : X × U → Im h is defined as [ x ] ∼ X , [ u ] ∼ U ↦→ [ h ( x, u )] ∼ Im h . This is well defined by Lemma 25.

For a η -finite dynamical system S = ⟨ X,U,f,x 0 , Y, h ⟩ , define the canonical regular function F S : ( U ) + → Im h to be the function defined by the FSA A S . The following lemma shows that the dynamics of the canonical automaton determine-up to path-connectedness-the dynamics of the system.

Lemma 26. Let D = ⟨ X,U,f ⟩ be a η -finite dynamics, and D A be its canonical semiautomaton. Then

<!-- formula-not-decoded -->

where [ w ] ∼ U ∈ U ∗ denotes the word with each letter of w replaced by its equivalence class.

Proof. By induction on the length of w . For the base case w = ε , we have D A ( [ x 0 ] ∼ X , [ ε ] ∼ U ) = D A ( [ x 0 ] ∼ X , ε ) = [ x 0 ] ∼ X and by definition D ( x 0 , ε ) = x 0 , so that [ D ( x 0 , ε )] ∼ X = [ x 0 ] ∼ X .

Now, suppose for w ∈ U ∗ we have D A ( [ x 0 ] ∼ X , [ w ] ∼ U ) = [ D ( x 0 , w )] ∼ X , and let [ u ] ∼ U ∈ U . Write w [ u ] ∼ U for the word obtained by appending [ u ] ∼ U at the end of w , we have

<!-- formula-not-decoded -->

Thus by induction the statement holds for all w ∈ U ∗ .

Lemma 27. Let S be a η -finite system and F S be its canonical regular function. Then, F S is implemented by S with encoder enc : U → U given by [ u ] ∼ U ↦→ u ′ with u ′ ∈ [ u ] ∼ U chosen arbitrarily, and with decoder dec : Im h → Im h , given by y ↦→ [ y ] ∼ Im h .

Proof. enc is continuous, since U is a finite alphabet. dec is continuous by Lemma 22. Let D A be the dynamics of A S , and let D S be the dynamics of S . Then we have

<!-- formula-not-decoded -->

where w -1 denotes the last symbol in word w . Now consider w ∈ ( U ⧸ ∼ U ) + and write [ u ] ∼ U for w -1 . By Lemma 26, we have D A ( [ x 0 ] ∼ X , w ) = [ D S ( x 0 , enc( w ) )] ∼ X , so that

<!-- formula-not-decoded -->

This concludes the proof.

Lemma 28. Let η -finite system S = ⟨ X,U,f,x 0 , Y, h ⟩ implement function F : Σ + → Γ with encoder enc : Σ → U and decoder dec : Im h → Γ . Then there are (continuous) functions enc ′ : Σ → U and dec ′ : Im h → Γ such that

<!-- formula-not-decoded -->

where F S : ( U ) + → ( Im h ) is the canonical function for S .

Proof. Define enc ′ as σ ↦→ [ enc( σ ) ] ∼ U for all σ ∈ Σ .

As for dec ′ , define it as [ y ] ∼ Im h ↦→ dec( y ) . This is well-defined: Consider y 1 , y 2 ∈ Im h such that y 1 , y 2 ∈ [ y ] ∼ Im h . Since y 1 , y 2 are path-connected in Im h , by continuity of dec : Im h → Γ we have that h ( y 1 ) , h ( y 2 ) are path-connected in Γ . Therefore necessarily h ( y 1 ) = h ( y 2 ) .

Let A S be the canonical FSA of S . Denote the dynamics of S as D S and the dynamics of A S as D A . By Lemma 26, we have

<!-- formula-not-decoded -->

Thus we have for all w ∈ Σ +

<!-- formula-not-decoded -->

Finally, enc ′ and dec ′ are continuous, since their domains are finite alphabets.

Theorem 1. An η -finite system S can-implement the same functions as its canonical automaton, which are necessarily regular.

Proof. Suppose S = ⟨ X,U,f,x 0 , Y, h ⟩ implements a function F : Σ → Γ , with encoder enc : Σ → U and decoder dec : Y → Γ . By Lemma 28, we have that the canonical FSA of S , say A S = ⟨ X,U,f, [ x 0 ] ∼ X , Im h, h ⟩ , implements F with encoder enc ′ and decoder dec ′ .

Moreover, consider the FSA A ′ = ⟨ X, Σ , δ, [ x 0 ] ∼ X , Γ , θ ⟩ , where δ : X × Σ → X is given by

<!-- formula-not-decoded -->

and θ : X × Σ → Γ is given by

<!-- formula-not-decoded -->

Then we have that F ( w ) = A ′ ( w ) for all w ∈ Σ + . Thus F is necessarily regular.

Now, suppose that A S implements a function F : Σ → Γ , with encoder enc : Σ → U and decoder dec : Im h → Γ . By Lemma 27, S implements F S with encoder enc and decoder dec . Thus we have the following: for all w ∈ Σ +

<!-- formula-not-decoded -->

so that S implements F with encoder enc ◦ enc and decoder dec ◦ dec .

## B.4 Algebraic Theory of η -Finite Systems

The connection between η -finite systems and canonical automata is extremely useful. It gives us a way to employ the powerful characterisations and results of AAT to any η -finite system dynamics. Namely, we can extend the notion of realisability to continuous η -finite systems, via the canonical automaton.

Definition 23. We say that η -finite dynamics D ′ are a realisation of η -finite dynamics D when M ( C ( D ′ )) is a realisation of M ( C ( D )) of D .

We that automaton A ′ is a realisation of system A , if the associated machine M A ′ is a realisation of of the associated machine M A via an assignment ( α, ι, ζ ) , and the respective initial states x ′ 0 , x 0 are such that x ′ 0 ∈ α ( x 0 ) .

Say that η -finite system S ′ is a realisation of system S , if A S ′ is a realisation of A S , where A S , A S ′ are the canonical automata. ■

The notion of realisation for machines is transitive. See § 1 . 3 of Hartmanis and Stearns [1966].

Fact B.4.1. If M is a realisation of M ′ and M ′ is a realisation of M ′′ , then M realies M ′′ .

It is easy to see that the notion of realisation for dynamics and systems is also transitive.

Lemma 29. Suppose that semiautomaton D ′ is a realisation of semiautomaton D . Then

1) for any machine M with dynamics D , the canonical machine M ( D ′ ) of D ′ is a realisation of M ,

2) for any automaton A with dynamics D , an initial state can be picked for M ( D ′ ) such that the resulting automaton is a realisation of A .

Proof. Say D = ⟨ Q, Σ , δ ⟩ and D ′ = ⟨ Q ′ , Σ ′ , δ ′ ⟩ . Suppose we have an assignment ( α, ι, ζ ) from D to D ′ . That is, α : Q →P + ( Q ′ ) , ι : Σ → Σ ′ , ζ : Q ′ × Σ ′ → Q × Σ

Let M = ⟨ Q, Σ , δ, Γ , θ ⟩ be a Mealy machine with semiautomaton D . The canonical machine for D ′ is

<!-- formula-not-decoded -->

Define ζ ′ : ( Q ′ × Σ ′ ) → Γ by ζ ′ = θ ◦ ζ . Want to show: ( α, ι, ζ ′ ) give an assignment of M into M ( D ′ ) . We already have that the condition I) is satisfied.

Now, for any q ∈ Q,σ ∈ Σ and q ′ ∈ α ( Q ) we have that ζ ◦ θ ′ ( q ′ , ι ( σ )) = id( q, σ ) = ( q, σ ) , since ( α, ι, ζ ) give an assignment of D into D ′ . Thus

<!-- formula-not-decoded -->

So ( α, ι, ζ ′ ) also satisfy condition II). Thus the 1) part of the statement holds.

Now for the part 2): Let A = ⟨ Q, Σ , δ, q 0 , Γ , θ ⟩ be a system with dynamics D . By part 1), the associated machine M A = ⟨ Q, Σ , δ, q 0 , Γ , θ ⟩ has some assignment ( α, ι, ζ ) into M D ′ . α ( x 0 ) is a nonempty set, and so we may arbitrarily pick x ′ 0 ∈ α ( x 0 ) . Then the automaton A ′ = ⟨ Q ′ , Σ ′ , δ ′ , q ′ 0 , Q ′ × Σ ′ , id ⟩ obtained from setting initial state x ′ 0 for machine M D ′ , by definition is a realisation of A .

We have the following proposition to connect our notion of dynamical systems with Algebraic Automata Theory.

Before proceeding, we remark that Definition 1 must be made fully precise by saying that a decoder is a function dec : Im h → Γ where h is the output function of system S , (rather than a function dec : Y → Γ ).

Theorem 30. Let S and S ′ be η -finite systems, and A S , A S ′ their respective canonical automata. If A S ′ is a realisation of A S , then S ′ can implement all the functions that S can implement.

Proof. Say we have A S = ⟨ X,U,f,x 0 , Im h, h ⟩ and A S ′ = ⟨ X ′ , U ′ , f ′ , x ′ 0 , Im h ′ , h ′ ⟩ .

Say that an assignment of A S into A S ′ is given by α : X →P + ( X ′ ) , ι : U → U ′ and ζ : Im h ′ → Im h . Let F S : ( U ) + → Im h be the canonical regular function for S . By Lemma 28, it suffices to show that A S ′ can implement F S .

Define the encoder enc : U → U ′ as enc = ι and decoder dec : Im h ′ → Im h as dec = ζ . Let D,D ′ be the dynamics of A S , A S ′ resp. By Theorem 1.4 in § 1 . 3 of [Hartmanis and Stearns, 1966], we have for all x ′ ∈ α ( x 0 ) and all w ∈ ( U ) + , that

<!-- formula-not-decoded -->

Thus, for all w ∈ ( Σ ) + we have

<!-- formula-not-decoded -->

This concludes the proof.

Example 8 . The reverse implication to Theorem 30 does not hold in general. Consider Σ = Σ ′ = { σ } , Q = { a, b } , Q ′ = { a } and unary dynamics functions δ : Q × Σ → Q defined as δ ( q, σ ) = q for every q ∈ Q , and depicted next.

σ

<!-- image -->

And similarly δ ′ : Q ′ × Σ → Q ′ defined as δ ′ ( q, σ ) = q for every q ∈ Q ′ , and depicted next.

<!-- image -->

Define system S = ⟨ Q, Σ , δ, x 0 = a, Γ = Q,θ ⟩ with θ : ( q, σ ) ↦→ q , and system S ′ = ⟨ Q ′ , Σ , δ ′ , q ′ 0 = a, Γ ′ = Q ′ , θ ′ ⟩ with θ ′ : ( q, σ ) ↦→ q .

The only possible state trajectories for either systems are the constant trajectories x n = x 0 = a and x ′ n = x ′ 0 = a . Thus, a function Σ + → Γ can be represented by either system if and only if it is constant. So we have that both systems implement the same functions.

However, there is no assignment ( α, ι, ζ ) from S to S ′ . This is because Γ ′ is a singleton, and so any potential ζ : Γ ′ → Γ must be constant. At the same time, it must hold that α ( a ) , α ( b ) are non-empty and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This is a contradiction, as ζ must be constant.

■

Theorem 31. Let D,D ′ be η -finite dynamics. Suppose that D ′ is a realisation of D . Then any function implemented by a system with dynamics D can be implemented by some system with dynamics D ′ .

Proof. Let D A = ⟨ X,U,f ⟩ and D A ′ = ⟨ X ′ , U ′ , f ′ ⟩ be the canonical semiautomata of D and D ′ , respectively. Then D A ′ realises D A .

Let S be a system with dynamics D implementing function F . Its canonical automaton A S has dynamics D A , and so by Lemma 29 there is an automaton A ′ = ⟨ X ′ , U ′ , f ′ , x ′ 0 , Γ ′ , θ ′ ⟩ with dynamics D A ′ which realises A S .

Consider the system S ′ = ⟨ X ′ , U ′ , f ′ , x ′ 0 , X ′ × U ′ , id ⟩ , where x ′ 0 ∈ X ′ is s.t. [ x ′ 0 ] ∼ X ′ = x ′ 0 . Its canonical automaton is A S ′ = ⟨ X ′ , U ′ , f ′ , x ′ 0 , X ′ × U ′ , id ⟩ . A S ′ realises A ′ with the assignment α : X ′ → P + ( X ′ ) g.b. x ′ ↦→ { x ′ } , ι : U ′ → U ′ g.b. u ′ ↦→ u ′ and finally ζ : X ′ × U ′ → Γ ′ g.b ( x ′ , u ′ ) ↦→ θ ′ ( x ′ , u ′ ) . Thus by Theorem 30, S ′ can implement all functions that S can implement.

## B.5 Cascade Decomposition and η -Finite Systems

In this section we bridge the gap between the AAT decomposition results, which apply to serial cascading, and our η -finite framework, which focuses on feed-forward connections. We begin by showing how taking the canonical semiautomaton 'commutes' with feed-forward cascading.

Lemma 32. Let D 1 ⇝ · · · ⇝ D n be η -finite feed-forward cascade dynamics. Then we have

<!-- formula-not-decoded -->

where ' ≡ ' is as per Definition 15.

Proof. By induction, it suffices to show the statement for n = 2 .

We have D 1 = ⟨ X 1 , U 1 , f 1 ⟩ and D 2 = ⟨ X 2 , U 1 × X 1 , f 2 ⟩ . Now, C ( D 1 ) = ⟨ X 1 , U 1 , f 1 ⟩ and C ( D 2 ) = ⟨ X 2 , U 1 × X 1 , f 2 ⟩ . Note, that here we use that, by Lemma 19, ( U 1 × X 1 ) = U 1 × X 1 .

Thus, we may write the cascade

<!-- formula-not-decoded -->

where f is the dynamics function of the feed-forward cascade C ( D 1 ) ⇝ C ( D 2 ) .

At the same time, writing D 1 × D 2 = ⟨ X 1 × X 2 , U 1 , f ′ ⟩ , we have

<!-- formula-not-decoded -->

where again we use Lemma 19 to get ( X 1 × X 2 ) = X 1 × X 2 . It remains to show that f = f ′ . For [ x 1 ] ∼ X 1 ∈ X 1 , [ x 2 ] ∼ X 2 , [ u ] ∼ U 1 we have

<!-- formula-not-decoded -->

This concludes the proof.

Note: we treat objects such as X 1 × X 2 and ( X 1 × X 2 ) as identical, even though one is a product of equivalence classes, and the other is an equivalence class of a product. However, from Lemma 19, we can identify the two in a natural way, that is in a way that is consistent with applying functions component-wise.

Next, we show that cascading interacts well with realisability, up to introducing a connection function.

Lemma 33. Suppose D i = ⟨ X i , U i , f i ⟩ , D ′ i = ⟨ X ′ i , U ′ i , f ′ i ⟩ are such that D ′ i is a realisation for D i , for each i ∈ [1 .. 2] . Then, for any feed-forward cascade i ⇝ D 1 h ⇝ D 2 with input i and connection h , there is a continuous function g : U ′ 1 × X ′ 1 → U ′ 2 such that D ′ 1 g ⇝ D ′ 2 realises i ⇝ D 1 h ⇝ D 2 .

Proof. Let ( α i , ι i , ζ i ) be the assignment of M ( C ( D i )) = ⟨ X i , U i , f i , X i × U i , id ⟩ into M ( C ( D ′ i )) = ⟨ X ′ i , U ′ i , f ′ i , X ′ i × U ′ i , id ⟩ , for each i ∈ [1 .. 2] . We assume w.l.o.g. that h = id and i = id , i.e., we can consider the usual feed-forward cascade D 1 ⇝ D 2 , by replacing D 1 with D 1 ,i and D 2 with D 2 ,h . In that case, we have U 2 = U 1 × X 1 .

Define g : U ′ 1 × X ′ 1 → U ′ 2 given by g ( u ′ , x ′ ) = ι 2 ( u, x ) ∈ U ′ 2 where ( x, u ) = ζ 1 ( x ′ , u ′ ) ∈ X 1 × U 1 = U 2 .

Define

<!-- formula-not-decoded -->

Let ( x 1 , x 2 ) ∈ X 1 × X 2 , u 1 ∈ U 1 and ( x ′ 1 , x ′ 2 ) ∈ α ( ( x 1 , x 2 ) ) . Let f and f ′ g be the dynamics functions of C ( D 1 ) ⇝ C ( D 2 ) and C ( D ′ 1 ) ⇝ C ( D ′ 2 ) g respectively. We have that f ′ g ( ⟨ x ′ 1 , x ′ 2 〉 , ι ( u 1 ) ) = 〈 x ′ 1 , new , x ′ 2 , new 〉 , where

<!-- formula-not-decoded -->

by Property I) of assignment, and

<!-- formula-not-decoded -->

Now, by Property II) of assignment we have ζ 1 ( x ′ 1 , new , ι ( u )) = ( f 1 ( x 1 , u 1 ) , u 1 ) , since x 1 , new ∈ α 1 ( f 1 ( x 1 , u 1 ) ) . Thus

<!-- formula-not-decoded -->

So, altogether 〈 x ′ 1 , new , x ′ 2 , new 〉 ∈ α ( f ( ⟨ x 1 , x 2 ⟩ , u 1 ) ) , so Property I) of assignment is satisfied. Now

<!-- formula-not-decoded -->

where ( b, c, a ) = ζ 2 ( x ′ 2 , new , g 〈 ι ( u 1 ) , x ′ 1 , new 〉 ) = ζ 2 ( x ′ 2 , new , ι 2 ( u 1 , f 1 ( x 1 , u 1 ) ) ) . Thus

<!-- formula-not-decoded -->

and so Property II) is satisfied. We may now choose a continuous g ′ : U ′ 1 × X ′ 1 → U ′ 2 such that g ′ = g by Lemma 22. Then we have that C ( D ′ 2 ,g ′ ) = C ( D ′ 2 ) g . Overall, the cascade D ′ 1 ⇝ D ′ 2 ,g ′ realises D 1 ⇝ D 2 .

The decomposition theorems of AAT are stated for serial cascades, while RNNs in practice usually work with feed-forward cascades. In Appendix G.2, we show how D 1 ⋉ D 2 can be realised by D 1 g 1 ⇝ R X g 2 ⇝ D 2 for some continuous functions g 1 , g 2 , and the repeat dynamics R X over state-space X of D 1 .

Definition 24. The repeat dynamics on state space X are the dynamics R X = ⟨ X 2 , X, r X ⟩ , where r X ( ⟨ x old , x new ⟩ , x ) = ⟨ x new , x ⟩ . ■

Thus we have that with initial state ⟨ a, b ⟩ ∈ X 2 and input sequence ( u n ) n ≥ 1 ∈ X ω , the state sequence is ( s n = ⟨ x n -1 , x n ⟩ ) n ≥ 0 ∈ ( X 2 ) ω with x -1 = a, x 0 = b . Note that a repeat dynamics is a Finite Context Dynamics.

For η -finite spaces, R X can be decomposed in terms of 2-state repead dynamics.

Theorem 34. Let X be an η -finite space. Then the repeat dynamics on X , R X , are realised by a feed-forward cascade of the repeat dynamics R 2 on { 0 , 1 } .

Proof. Let X 1 , . . . , X n be the η -components of X . We can think of the canonical automaton as the repeat dynamics on X , R X = { X 2 , X, r X ⟩ .

Consider C n = f 1 ⇝ D 1 f 2 ⇝ D 2 . . . f n ⇝ D n = ⟨{ 0 , 1 } 2 × n , X, f C ⟩ , with D i ≡ R 2 for all i ∈ [1 ..n ] , and with f i : X ×{ 0 , 1 } 2 × i -1 →{ 0 , 1 } given by

̸

<!-- formula-not-decoded -->

Thus, each D i works in parallel, treating inputs x i as 1 , and others as 0 . Then we can retrieve the state of R X by checking which D i has 1 at the old position, and which D j has 1 at new position. This corresponds to state ⟨ x i , x j ⟩ .

The assignment this corresponds to is the following: define α : X 2 → P + ( { 0 , 1 } 2 n ) by α ( ⟨ x i , x j ⟩ ) = { E i,j } , where E i,j ∈ { 0 , 1 } 2 × n is s.t. [ E i,j ] 1 ,i = 1 , [ E i,j ] 2 ,j = 1 and remaining entries are all 0 . We also define ι : X → X as the identity, and ζ : { 0 , 1 } 2 × n × X → X 2 × X as mapping ( E i,j , x ) ↦→ ( ⟨ x i , x j ⟩ , x ) , with other inputs mapped arbitrarily.

Altogether, we have a recipe for proving positive results. It is sufficient to show that an architecture can realise FLIP-FLOP, to show that it can implement all group-free functions with serial cascades. If it further can realize R 2 , then it can implement all group-free functions with feed-forward cascades.

Theorem 35. Suppose that η -finite dynamics D is a realisation of FLIP-FLOP , and η -finite dynamics E a realization of R 2 . Then feed-forward cascades of D and E components can implement all group free functions.

Proof. Let F be a group-free function. By Theorem 12, F is implemented by a serial cascade of FLIP-FLOP's, say C . By the construction in Appendix G.2, we have that C is realised by a feed-forward cascade of FLIP-FLOP's and repeat semiautomata, say C ′ . By Lemma 34, each repeat semiautomaton is a feed-forward cascade of R 2 components. Therefore C ′ is realised by a feed-forward cascade of FLIP-FLOP's components and R 2 components, say C ′′ . By Lemma 33, a feed-forward cascade of D and E components realises C ′′ , say C ′′′ . Thus, by transitivity of realisability, C ′′′ realises C , and thus by Theorem 31, C ′′ can implement F .

Figure 7: The image of an η -component under an ϵ -robust transition lies inside the target η -component, within ϵ -distance of its boundary.

<!-- image -->

## C Robust Systems

In this appendix we introduce a central notion of robustness that allows us to extend Metric Automata Theory to the study of concrete finite-precision implementations.

Arithmetic operations with floating point numbers are difficult to analyse, since addition and multiplication are not exactly commutative, associative and distributive. Thus, for example, the recurrent form and the convolutional form of the SSM update are not exactly equivalent (also noted by Merrill et al. [2024]-see footnote 3 in Definition 2.1). A theoretical framework which specifies an explicit datatype either is hard to analyse, or introduces additional simplifying assumptions.

The central notion that allows us to extend Metric Automata Theory to the study of finite-precision implementations is the notion of ϵ -robustness . Intuitively, it describes stability of the dynamics under transition perturbations.

In Section C.1 we prove Theorem 2, thus showing that robustness provides a way to connect η -finite systems to their floating-point implementations on real-world computer architectures, without requiring us to commit to any particular standard of floating-point operations.

In Section C.2 we show that robustness provides stability under perturbing the parametrs of a model which describes the dynamics. We will later present a strongly robust dynamics based on the sLSTM model, which uses a particular choice of parameters. Our results show, that in such cases the parameters may be perturbed by some amount and the robust system will retain its behaviour.

Lastly, in Section C.3 we prove Theorem 5 and further describe what kind of connecting functions are required for strongly robust η -finite cascades, by showing that 2-layer MLPs suffice.

Robustness marks the departure of Metric Automata Theory from Classical Automata and Formal Languages Theory, allowing us to study phenomena that do not occur with discrete state-spaces.

For completeness, we restate Definition 2 paying closer attention to the role of inputs in the notion of strong ϵ -robustness.

Definition 2. For ϵ &gt; 0 and X ⊆ Ω X , U ⊆ Ω U , dynamics D = ⟨ X,U,f ⟩ are ϵ -robust (in Ω X ) if, for every x ∈ X and every u ∈ U , it holds that B Ω X ( f ( x, u ) , ϵ ) ⊆ X -i.e., y ∈ X for all y ∈ Ω X s.t. ∥ f ( x, u ) -y ∥ ≤ ϵ . Furthermore, we say that dynamics D are strongly ϵ -robust (in Ω X and Ω U ) if they are ϵ -robust (in Ω X ), each η -component of X contains an Ω X -ball of radius at least ϵ and each η -component of U contains an Ω U -ball of radius at least ϵ .

Note that the property of robustness is with respect to the ambient space Ω X , which contains the state space X . Thus, it is possible that a dynamics is ϵ -robust w.r.t. some ambient space (e.g., R ), and not ϵ -robust w.r.t. another ambient space (e.g., C ). This captures the property, that for a η -finite dynamics, a function approximating f within ϵ , and taking values in Ω , will implement the same transitions.

Lemma 36. Let C = D 1 ⇝ · · · ⇝ D n be a cascade, with D i = ⟨ X i , U × X [1 ,i -1] , f i ⟩ and X i ⊆ Ω i , U ⊆ Ω U . Then C is (strongly) ϵ -robust w.r.t. Ω 1 ×··· × Ω n if D i is (strongly) ϵ -robust w.r.t. Ω i for all i ∈ 1 ..n .

Proof. By induction, it suffices to show the statement for n = 2 . First, suppose that D i is ϵ -robust for i ∈ 1 , 2 . Let ⟨ x 1 , x 2 ⟩ ∈ X 1 × X 2 , u ∈ U and take ⟨ y 1 , y 2 ⟩ ∈ Ω 1 × Ω 2 s.t. || f ( ⟨ x 1 , x 2 ⟩ , u ) -⟨ y 1 , y 2 ⟩|| 2 ≤ ϵ . We have, by def of cascading

<!-- formula-not-decoded -->

By definition of the L 2 norm, since ||⟨ x ′ 1 , x ′ 2 ⟩ - ⟨ y 1 , y 2 ⟩|| ≤ ϵ , we also have

<!-- formula-not-decoded -->

Thus, by ϵ -robustness, we have that y i ∈ X i for i ∈ 1 , 2 , and hence ⟨ y 1 , y 2 ⟩ ∈ X 1 × X 2 . All together, C is ϵ -robust w.r.t. Ω 1 × Ω 2 .

Suppose further that D 1 , D 2 are strongly ϵ -robust. Let Z be a η -component of X 1 × X 2 . Then Z is of the form Z 1 × Z 2 for Z i η -component of X i , see proof of Lemma 19. We have by stronglyrobustness that B Ω i ( z i , ϵ ) ⊆ Z i for some z i ∈ Z i . By triangle inequality: B Ω 1 × Ω 2 ( ( z 1 , z 2 ) , ϵ ) ⊆ B Ω 1 ( z 1 , ϵ ) × B Ω 2 ( z 2 , ϵ ) ⊆ Z 1 × Z 2 . Finally, the input space of D 1 ⇝ D 2 is the same as the input space of D 1 , so by strongly-robustness we have that each η -component of U contains a closed Ω U -ball with radius ϵ .

## C.1 Finite Datatypes and Proof of Theorem 2

We now consider approximations of dynamical systems using a finite datatype D ⊆ Ω . D can for example represent the Python float type. We simply consider D as a discrete subset of Ω , abstracting away the details regarding arithmetic properties of such a datatype.

Definition 25. A finite datatype is a set D ⊆ Ω = R d having finite cardinality. A finite-datatype implementation of a system S is then a system whose input, state, and output spaces are finite datatypes, and whose dynamics and output functions are implemented using floating-point operations.

Definition 26. Call a set S an ϵ -covering of X ⊆ Ω , if for all x ∈ X there is a s ∈ S s.t. || x -s || ≤ ϵ .

Definition 27. Define int + p = { 0 , . . . , 2 p -1 -1 } to be the p -bit unsigned integers. Define int p = { 2 p -1 , . . . , 0 , . . . , 2 p -1 -1 } to be the p -bit signed integers. Define D p to be floating point numbers with 2 p -bit significand and p -bit exponent:

<!-- formula-not-decoded -->

Similarly, define D ′ p to be floating point numbers with p bits of integer precision and p bits of fractional precision:

<!-- formula-not-decoded -->

Lemma 37. Let X ⊆ Ω = R d be compact. Then, for p sufficiently large, i.e. with sufficient precision, D ′ d p is an ϵ -covering of X .

Proof. X is a compact subspace of Ω , and therefore bounded. So, there is some integer k ≥ 1 s.t. X ⊆ [ -2 k , 2 k -1] d . There is also some integer l ≥ 1 s.t. ϵ/ √ d ≥ 2 -l . Take p ≥ max( k, l ) . The set D ′ p is an 2 -p -cover of [2 -p , 2 p -1] . Now for any x ∈ X ⊆ [ -2 p , 2 p -1] d , we have that for each i ∈ 1 . . . d there is y i ∈ D ′ p s.t. | [ x ] i -y i | ≤ 2 -p . Therefore, writing y ∈ [ -2 p , 2 p -1] d for ( y 1 , . . . , y d )

<!-- formula-not-decoded -->

Therefore ( D ′ p ) d an ϵ -cover of X .

Lemma 38. Let X ⊆ Ω = R d be compact. Then, for p sufficiently large, i.e. with sufficient precision, D d p is an ϵ -covering of X .

Proof. By the previous Lemma, for some p we have that D ′ p is an ϵ -cover of X . We have for each a ∈ int p , b ∈ int + p :

<!-- formula-not-decoded -->

Now, 2 p a + b ≥ 2 p ( -2 p ) -2 p &gt; -2 2 p +1 and 2 p a + b ≤ 2 p · 2 p +2 p &lt; 2 2 p +1 , so that 2 p a + b ∈ int 2 p +2 . Since p +1 &lt; 2 p +1 , we have p +1 ∈ int p +1 . So, D ′ p ⊆ D p +1 , and therefore D p +1 is also an ϵ -cover for X , for sufficiently large p .

Definition 28. Let X , U be η -finite spaces having components X [1 ..r ] , U [1 ..s ] and subspaces X ′ ⊆ X , U ′ ⊆ U , respectively. Let us consider dynamics

<!-- formula-not-decoded -->

We say that dynamics D are simulated by dynamics ˆ D , with error at most ϵ , if we have that the disjointness condition (C1) holds for every i ∈ [1 ..r ] , the disjointness condition (C2) holds for every j ∈ [1 ..s ] , and the approximation condition (C3) holds.

̸

<!-- formula-not-decoded -->

̸

Lemma 39. Suppose η -finite dynamics D = ⟨ X,U,f ⟩ are ϵ -robust, and are simulated by η -finite dynamics ˆ D = ⟨ X ′ , U ′ , ˆ f ⟩ with error ϵ . Then ˆ D is a realisation of D .

Proof. Consider the canonical semiautomata D A = ⟨ X,U,f ⟩ and ˆ D A = ⟨ X ′ , U ′ , f ′ ⟩

Define α : X →P + ( X ′ ) as

<!-- formula-not-decoded -->

which is indeed non-empty by definition of simulation, and well-defined as X ′ ⊆ X , and so if x ′ 1 ∼ X ′ x ′ 2 then also x ′ 1 ∼ X x ′ 2 . Also define ι : U → U ′ by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ζ is indeed well-defined: suppose x ′ 1 , x ′ 2 ∈ [ x ′ ] ∼ X ′ and u ′ 1 , u ′ 2 ∈ [ u ′ ] ∼ U ′ . Then since X ′ ⊆ X and U ′ ⊆ U we also have x ′ 1 , x ′ 2 ∈ [ x ′ ] ∼ X , since x ′ 1 ∼ X x ′ 2 and u ′ 1 , u ′ 2 ∈ [ u ′ ] ∼ U , since u ′ 1 ∼ U u ′ 2 .

Now, ( α, ι, ζ ) is an assignment of M ( D A ) into M ( ˆ D A ) : for all [ x ] ∼ X ∈ X and [ u ] ∼ U ∈ U , and for all [ x ′ ] ∼ X ′ ∈ α ([ x ] ∼ X ) we have

<!-- formula-not-decoded -->

On the other hand, we have

<!-- formula-not-decoded -->

We have that x ′ ∈ [ x ] ∼ X , since [ x ′ ] ∈ α ([ x ] ∼ X ) . We have by simulation with error at most ϵ

<!-- formula-not-decoded -->

and so f ′ ( x ′ , u ′ ) ∈ [ f ( x, u )] ∼ X , since D is ϵ -robust. Hence f ′ ( [ x ′ ] ∼ X ′ , ι ([ u ] ∼ U ) ) ∈ α ( f ([ x ] ∼ X , [ u ] ∼ U ) ) . Thus Part I) of definition of assignment is satisfied.

Moreover, we have

<!-- formula-not-decoded -->

so that Part II) of the definition is satisfied.

Figure 8: Given sufficient precision, the transitions of strongly ϵ -robust dynamics can be realized with approximate dynamics on a finite datatype, which gives a ϵ -covering for the state-space.

<!-- image -->

Lemma 40. Consider η -finite dynamics D = ⟨ X,U,f ⟩ , s.t. each component of X and U contains a closed ball of radius ϵ (in Ω X , Ω U resp.)

Then given datatypes D X ⊆ X, D U ⊆ U with sufficient precision, there is a function ˆ f : D X × D U → D X s.t. ⟨ D X , D U , ˆ f ⟩ simulates ⟨ X,U,f ⟩ with error ϵ .

Proof. Suppose D X is an ϵ -covering of X , and D is an ϵ -covering of U . Let X 1 , .., X r , r ≥ 1 be the connected components of X . Let i ∈ 1 ..r , we have by assumption, that for some x i ∈ X i

<!-- formula-not-decoded -->

Since D X is an ϵ -covering of X , there is some d i ∈ D X s.t. ∥ x i -d i ∥ ≤ ϵ , and therefore d i ∈ B ( x i , ϵ ) ⊆ X i .

Similarly, there is an element of D U in each component of U . Now, we may construct ˆ f as follows: for x ∈ D X and u ∈ D U

<!-- formula-not-decoded -->

with ties broken arbitrarily. Then, as D X is an ϵ -covering of X , ∥ ˆ f ( x, u ) -f ( x, u ) ∥ ≤ ϵ as desired.

We now have the setup, and necessary results for Theorem 2.

Theorem 2. Every η -finite system with strongly robust dynamics can be implemented with floatingpoint operations, given sufficient precision.

Proof. Apply Lemma 39 and Lemma 40 to obtain a realisation of S using a finite datatype, e.g. using D p or D ′ p for sufficiently large p .

## C.2 Parametrised Systems

The stability of robust dynamics can also be a desirable property in the context of learning. Consider a parametrised model describing the trained model. If the system described by the model is ϵ -robust and it is sufficiently smooth with respect to its parameters, then perturbing the model parameters will not change the behaviour of the system. Thus a robust system is intuitively more likely to be attained by a learning algorithm.

Definition 29. Let f : Θ × Ω X × Ω U → Ω be continuous. Write f θ for the function f ( θ, -, -) . A dynamics parametrised by Θ is of the form D θ = ⟨ X,U,f θ ⟩ .

Theorem 41. (Corollary 36.20 of [Willard, 2012]) A continuous functions on a compact metric space X is uniformly continuous , that is for all ϵ &gt; 0 there exists δ &gt; 0 such that for all x, y ∈ X || x -y || ≤ δ = ⇒ || f ( x ) -f ( y ) || ≤ ϵ .

Theorem 42. Let η -finite dynamics D θ = ⟨ X,U,f θ ⟩ be parametrised by Θ , and let Θ be compact. Suppose D θ is ϵ -robust (w.r.t Ω X ). Then for some δ &gt; 0 , we have that for ρ ∈ Θ s.t. || θ -ρ || ≤ δ the dynamics D ρ = ⟨ X,U,f ρ ⟩ is well-defined. Moreover, for any system S θ with dynamics D θ , the system S ρ obtained by switching out D θ for D ρ has the same canonical automaton.

Proof. Since D θ is η -finite, we have that X and U are compact. Thus the Cartesian product Θ × X × U is compact. Thus, by Theorem 41 for all ϵ &gt; 0 we have some δ &gt; 0 such that for all ( θ, x, u ) , ( θ, x, u ) ∈ Θ × X × U

<!-- formula-not-decoded -->

Now, take ρ ∈ B Θ ( θ, δ ) . We have for all x ∈ X and u ∈ U that

<!-- formula-not-decoded -->

Thus f ( ρ, x, u ) ∈ B ( f ( θ, x, u ) , ϵ ) ⊆ X , since D θ is ϵ -robust. Moreover, letting X 1 , .., X r be the components of X and U 1 , .., U s be the components of U , we have that X ∩ X i = ∅ for i ∈ 1 ..r and U ∩ U i for i ∈ 1 ..s . Thus D ρ simulates D θ with error ϵ .

̸

Now, the canonical semiautomaton for D θ is ⟨ X,U,f θ ⟩ and the canonical semiautomaton for D ρ is ⟨ X,U,f ρ ⟩ . By Lemma 39, we have that f θ and f ρ give the same transitions. Therefore the two semiautomata are the exact same. Taking S θ , S ρ as in the statement, we see that they indeed must have the same canonical automaton.

## C.3 Robust Cascade Decomposition and Proof of Theorem 5

Coming back to connecting functions discussed in Appendix B.5, we have the following refinement of the result.

Theorem 43. Let D be a strongly robust η -finite dynamics, which are a realisation of FLIP-FLOP . Then all group-free functions can be implemented by some strongly robust serial cascade of D components. Moreover, the connection functions in such cascade can be given by depth-2 MLPs.

Proof. Say D = ⟨ X,U,f ⟩ is strongly ϵ -robust. By Theorem 35, for any group-free function F , there is a serial cascade C of D -components which can implement it. By Lemma 36, C is also strongly robust. Say, C = g 1 ⇝ D 1 · · · g L ⇝ D L = ⟨ X L , U ′ , f C ⟩ , with U ′ an η -finite space, D i ≡ D and g i : U ′ × X i -1 → U .

Let U 1 , . . . , U n be the η -components of U . By strong robustness, for each i ∈ [1 ..n ] , there is u i ∈ U i s.t. B Ω U ( u i , ϵ ) ⊆ U i By Lemma 22, we can w.l.o.g. assume that g i has its image in { u 1 , . . . , u n } , while still inducing the same mapping U ′ × X i -1 → U .

By Theorem 16, there is a MLP N i : Ω U ′ × Ω i -1 X → Ω U which ϵ -approximates g i , since U ′ × X i -1 is compact and g i continuous. For ⟨ u ′ , x 1 , . . . , x i -1 ⟩ ∈ U ′ × X i -1 we have f i ( ⟨ u ′ , x 1 , . . . , x i -1 ⟩ ) = u j for some j ∈ [1 ..n ] , so

<!-- formula-not-decoded -->

Thus N i sends elements of U ′ × X i -1 to the same η -components of U as g i . Moreover, N i is continuous.

Overall, the canonical automaton for g 1 ⇝ D 1 g 2 ⇝ · · · g L ⇝ D L is the same as the canonical automaton for N 1 ⇝ D 1 N 2 ⇝ · · · N L ⇝ D L . Thus the strongly robust cascade with D components and MLP connections can implement F .

Appendix G.3 shows constructions for strongly robust η -finite xLSTM FLIP-FLOP and R 2 dynamics. All together, we obtain Theorem 5:

Theorem 5 (xLSTM does start-free robustly) . All star-free languages can be recognised by xLSTM cascades, as well as by floating-point implementations of xLSTM cascades given sufficient precision.

Proof. We have that there are strongly robust xLSTM dynamics that realise FLIP-FLOP and R 2 . Thus by Theorem 43, every group-free function can be implemented by a cascade of strongly robust xLSTM dynamics. Any such cascade is itself strongly robust, by Lemma 36, and thus can be realized by floating-point operations, given sufficient precision, by Theorem 2

Moreover, by Theorem 43 we know that for these cascades, it suffices to use MLP connecting functions. By Theorem 42 we also have that the parametrizations of sLSTM blocks which yields FLIP-FLOP and R 2 can also be changed, within some δ , retaining the behaviour of the dynamics.

## D Expressivity Results for State Space Models

In this Appendix we reap rewards of establishing the preliminary framework of Metric Automata Theory for η -finite dynamics. We can now prove expressivity results by establishing structural properties of dynamics, which are preserved by feed-forward cascades, and which are generally applicable.

In Section D.1 we introduce the notion of contracting dynamics, which describes dynamics that are not able to keep track of a state over unbounded input lengths. We use this notion to prove Theorems 3 and 4.

In Section D.2 we introduce another structural property, called aperiodicity . It is the η -finiteness corresponding notion to group-freeness in Finite Automata. We use aperiodicity to prove Theorem 6.

Finally, in Section D.3 we focus on the SSM parametrisation of Mamba, and prove Theorem 7.

## D.1 Contracting Dynamics and Proofs of Theorems 3 and 4

Definition 30. Call η -finite dynamics ⟨ X,U,f ⟩ a contracting dynamics , if for any initial points x 0 , x ′ 0 ∈ X and eventually equivalent input sequences ( u n ) n ≥ 1 , ( u ′ n ) n ≥ 1 ⊆ U , we have that the corresponding state sequences ( x n ) n ≥ 1 , ( x ′ n ) n ≥ 1 ⊆ U are eventually equivalent.

Thus, a for a contracting dynamics, it does not matter what state the evaluation of the inputs starts from-eventually all initial states lead to the same behaviour under a fixed input sequence. The intuition behind the name is the following-eventually all possible states that the dynamics could be in under the input sequence collapse to a single η -component.

Example 9 . Clearly, all Finite Context Dynamics (Definition 18) are contracting.

■

Lemma 44. Let C = D 1 ⇝ · · · ⇝ D n be a cascade of η -finite contracting dynamics. Then C is a contracting dynamics.

Proof. By induction, it is sufficient to show the statement for n = 2 .

Let us consider C = D 1 ⇝ D 2 with D 1 = ⟨ X,U,f 1 ⟩ and D 2 = ⟨ Z, U × X,f 2 ⟩ . The dynamics function of the cascade is:

<!-- formula-not-decoded -->

Consider arbitrary ⟨ x 0 , z 0 ⟩ , ⟨ x ′ 0 , z ′ 0 ⟩ ∈ X × Z and ( u t ) t ≥ 1 , ( u ′ t ) t ≥ 1 ∈ U ω , eventually equivalent in U . Take

<!-- formula-not-decoded -->

By inductive hypothesis, D 1 is contracting, and so since we have

<!-- formula-not-decoded -->

we have that ( x n ) n ≥ 1 , ( x ′ n ) n ≥ 1 ∈ X ω are eventually equivalent. Thus also ( ⟨ u n , x n +1 ⟩ ) n ≥ 1 , ( ⟨ u ′ n , n ′ n +1 ⟩ ) n ≥ 1 ∈ ( U × X ) ω are eventually equivalent.

Note that we have z n +1 = f 2 ( z n , ⟨ u n , x n +1 ⟩ ) and z ′ n +1 = f 2 ( z ′ n , ⟨ u ′ n , x ′ n +1 ⟩ ) . Since D 2 is by assumption contracting, and the two input sequence are eventually equivalent by continuity of f n , we get that ( z n ) n ≥ 1 , ( z ′ n ) n ≥ 1 ∈ Z ω are eventually equivalent.

<!-- formula-not-decoded -->

Lemma 45. Suppose a η -finite Linear Recurrent Dynamics D is ϵ -robust. Then D is contracting.

Proof. Suppose that D = ⟨ X,U,f ⟩ is ϵ -robust.

Let x 0 , x ′ 0 ∈ X and ( u n ) n ≥ 1 , ( u ′ n ) n ≥ 1 ⊆ U which are eventually equivalent-say for n ≥ N . For each component of U , say U 1 , . . . , U k , define a representative element r 1 , . . . , r k . Define (˜ u n ) n ≥ 1 ⊆ U to be such that ˜ u n = r c where U c is the component containing u n + N . Thus (˜ u n ) n ≥ 1 is equivalent to ( u n + N ) n ≥ 1 and ( u ′ n + N ) n ≥ 1 .

Write A n = A (˜ u n ) and B n = B (˜ u n ) and f n ( x ) = f ( x, ˜ u n ) . For S ⊆ Ω , define

<!-- formula-not-decoded -->

For β ∈ R ≥ 0 , write β · S = { β · s : s ∈ S } . Take M = sup x,y ∈ X || x -y || . We have that M is finite, since X is compact, and hence bounded. Also, denote X (0) = X , X ( n +1) = { f ( x, ˜ u n ) : x ∈ X ( n ) } = { D ( x } .

̸

We have, by induction that ∆( X ( n ) ) ⊆ ( M M +2 nϵ ) · ∆( X ) : for n = 0 this is immediate. For n ≥ 1 , by inductive hypothesis we have ∆( X ( n -1) ) ⊆ ( M M +2( n -1) ϵ ) · ∆( X ) . Consider u = 0 , u ∈ ∆( X ( n ) ) . Take v = u || u || . We have that

<!-- formula-not-decoded -->

for some x, y ∈ X ( n -1) and β ∈ [0 , 1] . We have that x -y ∈ ∆( X ( n -1) ) ⊆ ( M M +2( n -1) ϵ ) · ∆( X ) , so for some x ′ , y ′ ∈ X we have

<!-- formula-not-decoded -->

Now:

<!-- formula-not-decoded -->

so by robustness, f n ( x ′ ) + ϵ · v ∈ X and f n ( y ′ ) -ϵ · v . Thus

<!-- formula-not-decoded -->

So, we have u = γ · l for some l ∈ ∆ X and

<!-- formula-not-decoded -->

So u ∈ ( M +2 nϵ M ) · ∆( X ) , and thus indeed ∆( X ( n ) ) ⊆ ( M M +2 nϵ ) · ∆( X ) . Therefore sup x,x ′ ∈ X ( n ) || x -x ′ ||→ 0 as n →∞ .

Now, consider the state-sequences ( D ( x 0 , u [1 ..n ] ) ) n ≥ 1 , ( D ( x ′ 0 , u ′ [1 ..n ] ) ) n ≥ 1 . We have by Lemma 24

<!-- formula-not-decoded -->

and similarly D ( x ′ 0 , u ′ [(1+ N ) .. ( n + N )] ) ∼ X D ( x ′ N , ˜ u [1 ..n ] ) . Now, D ( x ′ N , ˜ u [1 ..n ] ) , D ( x N , ˜ u [1 ..n ] ) ∈ X ( n ) . Thus we have

<!-- formula-not-decoded -->

Therefore, eventually D ( x ′ N , ˜ u [1 ..n ] ) and D ( x N , ˜ u [1 ..n ] ) ∈ X ( n ) are in the same η -component of X .

Figure 9: State sequence of aperiodic dynamics under iterated input always η -converges.

<!-- image -->

Theorem 3 (Non-robustness of LRDs) . Suppose an η -finite LRD D is such that its canonical semiautomaton D A has at least two states, and an input inducing an identity transformation. Then D cannot be ϵ -robust for any ϵ &gt; 0 .

Proof. Let D = ⟨ X,U,f ⟩ be an η -finite LRD, such that its canonical semiautomaton D = ⟨ X,U,f ⟩ has at least two distinct η -states, say x, x ′ , and an input u inducing identity transformation on X .

For contradiction suppose that D is robust. Then by Lemma 45, D is contracting. Thus for x 0 ∈ x, x ′ 0 ∈ x ′ and u ∈ u we have that the sequences ( x n = D ( x 0 , u n ) ) n ≥ 1 , ( x ′ n = D ( x ′ 0 , u n ) ) n ≥ 1 ∈ X ω are eventually equivalent. Since [ u ] ∼ U = u induces the identity transformation of X , we have that the corresponding sequences ([ x n ] ∼ X ) n ≥ 1 , ([ x ′ n ] ∼ X ) n ≥ 1 ∈ X ω are constant, equal x, x ′ respectively. Thus necessarily x = x ′ . This is a contradiction.

Lemma 46. Contracting dynamics cannot implement the state-sequence function of FLIP-FLOP .

Proof. Consider a system S with some encoder enc : { set , reset , id } → U and decoder dec : Y → { low , high } . Suppose that the dynamics D = ⟨ X,U,f ⟩ of S are contracting. Consider x 0 ∈ X and input sequences ( u n ) n ≥ 1 , ( u ′ n ) n ≥ 1 ⊆ U , given by

<!-- formula-not-decoded -->

They are eventually equivalent, and so the corresponding state sequences x n = D ( x 0 , ⟨ u 1 ..n ⟩ ) and x ′ n = D ( x 0 , ⟨ u ′ 1 ..n ⟩ ) are also eventually equivalent. Thus

<!-- formula-not-decoded -->

for large enough n , since { high , low } is a discrete space.

However, the two sequences of inputs correspond to different flip flop states - thus D cannot be a dynamics for a system that implements a flip flop.

Theorem 4 (LRDs cannot do FLIP-FLOP robustly) . FLIP-FLOP cannot be implemented by a cascade of η -finite ϵ -robust LRDs for any ϵ &gt; 0 .

Proof. A cascade of such LRDs is contracting by Lemmas 45 and 44. Thus, by Lemma 46, it cannot implement FLIP-FLOP.

## D.2 Aperiodic Dynamics and Proof of Theorem 6

Definition 31. For a η -finite space X , we say a sequence ( x n ) n ≥ 1 ⊆ X η -converges in X , if eventually all its terms lie in the same η -component of X .

If the sequence of states of a system η -converges, it means that the behaviour of that system is eventually the same.

Definition 32. Call a η -finite dynamics D = ⟨ X,U,f ⟩ aperiodic , if for all x 0 ∈ X and input sequences ( u n ) n ≥ 1 ⊆ U η -convergent in U , we have that the corresponding state sequence ( x n ) n ≥ 1 ⊆ X is η -convergent in X .

An example of a aperiodic dynamics is given by the FLIP-FLOP dynamics. An input sequence that η -converges must eventually be constantly set or reset . In that case, the state is eventually high , low respectively.

Lemma 47. Let D be a η -finite Linear Recurrent Dynamics, with A ( u ) having all its eigenvalues being non-negative. Then D is aperiodic.

Proof. This is a similar argument as for Theorem 1 in Grazzi et al. [2025], with some simplifications stemming from the fact that we can use associativity of linear operations freely.

Let D = ⟨ X,U,f ⟩ be an η -finite Linear Recurrent Dynamics, with X ⊆ R d , s.t. A ( u ) has all its eigenvalues being real, for all u ∈ U . Say f ( x, u ) = A ( u ) · x + B ( u ) .

Consider a sequence ( u n ) n ≥ 1 ∈ U ω , η -convergent in U , and x 0 ∈ X . Let ( x n = D ( x 0 , u 1 ...n ) ) n ≥ 1 ∈ X ω be the corresponding state sequence. We have some N s.t. for n ≥ N all u n are contained in the same component of U , we may pick a representative r ∈ U of that component.

Write A = A ( r ) , B = B ( r ) . By Lemma 25, we have for n ≥ N that

<!-- formula-not-decoded -->

We consider the state sequence in the diagonalized space of A . Write A = P -1 JP for the Jordan normal form of A . Here J is block diagonal, with say blocks J 1 , ..., J s , J b ∈ R m b × m b being a Jordan Block with λ b on the diagonal being an eigenvalue of A , and 1 on the right off-diagonal. Also P ∈ R d × d , since all eigenvalues of A are real.

Take ¯ x n = Px ′ n , then we have

<!-- formula-not-decoded -->

We will consider the difference z n = ¯ x n +1 -¯ x n . Unrolling the recurrence we get

<!-- formula-not-decoded -->

The i -th entry of this difference, where i is in say the b -th block of J , is

<!-- formula-not-decoded -->

This is of the form considered in Lemma 64. Thus, [ z n ] i ∈ R is eventually monotone, and so it either converges in R or is unbounded as n →∞ .

Now, if z n → 0 , that is [ z n ] i for all i ∈ [1 ..d ] , then we have that also, by continuity of linear maps, x ′ n +1 -x ′ n = P -1 z n → 0 , so that x ′ n must eventually be in the same component of X by Lemma 20. Therefore also ( x n ) n ≥ 1 is η -convergent in X .

Otherwise, one of the entries of z n either is unbounded, or converges to a non-zero limit. In both cases, the corresponding entry of x n is unbounded as n →∞ , and so this is impossible in a η -finite space X .

Overall, this shows that D must be aperiodic.

Lemma 48. Let D = ⟨ X,U,f ⟩ be a η -finite Finite Context Dynamics. Then D is aperiodic.

Proof. Let l be the context length of D . Let x 0 ∈ X and ( u n ) n ≥ 1 ∈ U ω be η -convergent in U . Let ¯ u ∈ U lie in the component of U which contains the tail of ( u n ) n ≥ 1 , say for n ≥ N . For n ≥ N + l we have that u n -l +1 , ..., u n ∼ ¯ u , and so

<!-- formula-not-decoded -->

Thus x n is in the component of X containing C (¯ u l ) .

Lemma 49. Let C = D 1 ⇝ · · · ⇝ D k be a cascade of η -finite aperiodic dynamics D 1 , . . . , D k . Then C is aperiodic.

Proof. By induction, is is sufficient to show the statement for n = 2 .

Let us consider C = D 1 ⇝ D 2 with D 1 = ⟨ X,U,f 1 ⟩ and D 2 = ⟨ Z, U × X,f 2 ⟩ . The dynamics function of the cascade is:

<!-- formula-not-decoded -->

Consider a sequence ( u t ) t ≥ 1 ∈ U ω η -convergent in U , and ⟨ x ′ 0 , x 0 ⟩ ∈ X ′ × X .

As D 1 is aperiodic, the corresponding sequence ( x n ) n ≥ 1 ⊆ X ω is η -convergent in X . Equivalently, ( x t +1 ) t ≥ 1 is η -convergent in X . Moreover, then the sequence ( u ′ n = ⟨ u n , x n +1 ⟩ ) n ≥ 1 is η -convergent in U × X . Since D 2 is aperiodic, the sequence ( z n ) n ≥ 1 ∈ Z is therefore η -convergent in Z .

All together, ( ⟨ x n , z n ⟩ ) n ≥ 1 is η -convergent in X × Z . ♢

Theorem 50. η -finite dynamics are aperiodic if and only if their canonical semiautomaton is groupfree

Proof. Let D = ⟨ X,U,f ⟩ have canonical semiautomaton D A = ⟨ X,U,f ⟩

̸

( ⇒ ) First, suppose that D A is not group-free. By Theorem 13, there exist some S ⊆ X and u ∈ U s.t. f ( -, u ) induces a non-trivial permutation on S . That is, since S is a finite set, we have s ∈ X s.t. D A ( s, u n ) = D A ( s, u n +1 ) for all n ≥ 1 . Here u n denotes the word of length n consisting of repeated symbol u .

Take u ∈ U s.t. [ u ] ∼ U = u and x ∈ X s.t. [ x ] ∼ X = s . Then, we have that for all n ≥ 1 that

̸

<!-- formula-not-decoded -->

The input sequence ( u n ) n ≥ 1 is η -convergent in U , but the corresponding state sequence ( D ( s, u n )) n ≥ 1 is not. Thus, D is not aperiodic.

( ⇐ ) Now, suppose that D A is group free. By Theorem 12, D A can be realized by a serial cascade of FLIP-FLOPs, say C . We also have, that C can be realized by a feed-forward cascade C ′ of FLIP-FLOPs and repeat semiautomata, all of which are aperiodic (as repeat semiautomata are FCDs). Thus by Lemma 49, C ′ is aperiodic. It remains to show that dynamics realised by aperiodic dynamics are also aperiodic.

Let ( α, ι, ζ ) be an assignment of D A into C ′ . Consider an η -convergent input sequence ( u n ) n ≥ 1 ⊆ U and x 0 ∈ X , with the corresponding state sequence ( x n = D A ( x 0 , u [1 ..n ] ) ) n ≥ 0 ⊆ X . Since ( u n ) ⊆ U is η -convergent, it is in fact eventually constant, since U is a discrete space.

Since C ′ realizes D A , by Theorem 14, we have, for x ′ 0 ∈ α ( x 0 )

<!-- formula-not-decoded -->

where M ( D A ) , M ( C ) are the canonical machines for D A , C , respectively. Now, ( u n ) is eventually constant and so also ( ι ( u n ) ) is eventually constant. C is aperiodic, and so the sequence C ( x ′ 0 , ι ( u [1 ..n ] ) ) is η -convergent (and thus eventually constant, as C is a semiautomaton). All together

<!-- formula-not-decoded -->

by def. of canonical machines, and therefore this sequence is also eventually constant.

Thus the state sequence D A ( x 0 , u [1 ..n ] ) itself is eventually constant.

Equivalently, by Lemma 26, for any s η -convergent sequence ( u n ) ⊆ U and x 0 ∈ X the state sequence ( D ( x 0 , u [1 ..n ] ) ) ⊆ X is η -convergent, and so D is indeed aperiodic.

## D.3 Parametrisation of Mamba and Proof of Theorem 7

Sarrof et al. [2024] show that any star-free language can be recognized by an SSM like Mamba (Gu et al. [2022]), using the Krohn and Rhodes Theorem from Algebraic Automata Theory. However, in their construction, they assume that gates of the form A ( u ) = 0 can be used, which is not the case for architectures utilizing strictly positive parametrization, like Mamba.

We show in Construction 3 a modified η -finite system construction, which only requires gates with diagonal entries in the range [ ϵ, 1] , for a suitable ϵ &gt; 0 . As it turns out, further restricting diagonal entries to lie in ( -1 , 1) makes it impossible to implement a flip flop.

Mamba ([Gu et al., 2022]) parametrization is of the form

<!-- formula-not-decoded -->

and ⊙ is the element-wise product R d × R d → R d . This gives -[ ∆ u ⊙ exp( z u ) ] i &lt; 0 for i ∈ 1 . . . d , and thus A ( u ) i ∈ (0 , 1) for i ∈ 1 . . . d . We will show in this section that an SSM using Mamba blocks cannot implement a flip flop for unbounded .

However, experimental results in [Sarrof et al., 2024] show that this architecture does well in experimental evaluations and demonstrates length generalization for star-free modelling tasks. For tasks involving periodic modelling, the model fails to length generalize. This motivates us to investigate the geometric complexity of the state space when evaluated on sequences of bounded length in Appendix E.

Construction 3. There is a η -finite system with Linear Recurrent Dynamics with diagonal entries in [ ϵ, 1] , for some ϵ &gt; 0 , which realize FLIP-FLOP dynamics.

Take ϵ = 1 / 4 . Consider X = X l ∪ X h ⊆ R , where

<!-- formula-not-decoded -->

Then X q 0 , X l , X h are the components of X , and X is η -finite. Take U , e : { s, r, i } → U and f : X × U → X to be such that

<!-- formula-not-decoded -->

We have X ⊆ ¯ B (0 , 2 + ϵ ) , and so ( ϵ/ 4 · -)( X ) ⊆ ¯ B (0 , ϵ/ 4 · (2 + ϵ )) ⊆ ¯ B (0 , ϵ ) . Thus we see that f maps X to X l under input r and to X h under input s . Under input i , f acts as identity. Thus these dynamics indeed realize FLIP-FLOP, through assignment that identifies with α mapping high ↦→ X h , low ↦→ X l , ι mapping set ↦→ s, reset ↦→ r, id ↦→ i and ζ mapping X l to low and X h to high .

Lemma 51. Let D = ⟨ X,U,f ⟩ be an η -finite Linear Recurrent dynamics with A ( u ) diagonal, with entries in ( -1 , 1) for all u ∈ U . Then D is contracting.

Proof. Let x 0 , x ′ 0 ∈ X and ( u n ) n ≥ 1 ⊆ U . For each component of U , say U 1 , . . . , U k , define a representative element r 1 , . . . , r k . Define ( u ′ n ) n ≥ 1 ⊆ U to be such that u ′ n = r c where U c is the component containing u n . Thus ( u ′ n ) n ≥ 1 is equivalent to ( u n ) n ≥ 1 , and ( u ′ n ) n ≥ 1 takes finitely many values r 1 , . . . , r k .

Now, consider A 1 , . . . , A k , where A c = A ( r c ) . For each c ∈ [1 ..r ] , let λ c be the largest size eigenvalue of A c . Then we have | λ c | &lt; 1 , and

<!-- formula-not-decoded -->

Let λ ∈ arg max c ∈ 1 ..r | λ c | , then we have | λ | &lt; 1 and

<!-- formula-not-decoded -->

Now, we have that for the state sequences ( x n ) n ≥ 1 , ( x ′ n ) n ≥ 1 corresponding to initial states x 0 , x ′ 0 resp., and the input sequence ( u ′ n ) n ≥ 1 , the following holds:

<!-- formula-not-decoded -->

Thus eventually x n and x ′ n must be in the same component of X .

Altogether, we arrive at the following result (for η -finite dynamics), restated here more precisely than in the main body.

Theorem 7. SSMs with Mamba parametrisation cannot recognise FLIP-FLOP as η -finite systems.

Proof. Mamba blocks are feed-forward cascades of LRDs of the type considered in Lemma 51 and convolution blocks (FCDs)-see Figure 6. Thus η -finite feed-forward cascades of Mamba blocks are contracting, and so by Lemma 46, cannot implement FLIP-FLOP.

## E Geometrically Constrained Systems

In this appendix, we depart the setting of η -finiteness, and explore geometrically-constrained systems (GCSs) . This setting allows for systems implementing functions beyond regular, but shares many properties with the η -finite setting. We develop the theory of GCS to explain empirical capabilities of Mamba, and to showcase the flexibility and generalizability of Metric Automata Theory.

In Section E.1 we develop a notion analogous to aperiodicity from Section D.2. We then prove Theorem 9.

In Section E.2 we introduce a generalisation of η -finiteness, called weak η -finiteness . We use it to argue that the cascade decomposition results for η -finite dynamics still apply to dynamics with convex-covering state-spaces.

In Section E.3 we show that η -finite dynamics are a special case of convex-constrained dynamics. Finally, we show a construction of a FLIP-FLOP using a Mamba convex-constrained SSM, and argue using weakly η -finiteness that Theorem 8 holds.

Definition 33. For Ω = R d or C d , we call C ⊆ Ω a convex-covering if C is a finite union of open, convex sets in Ω . We say that X ⊆ Ω is convex-covered by C if X ⊆ C .

We say X is convex-separated by C if (i) it is convex-covered by C and (ii) each path-connected component of C contains at most one path-connected component of X . ■

Note: any convex set in Ω = R d or C d is path-connected. Thus any convex-covering C has finitely many path-connected components.

Definition 34. Let Ω = R d or Ω = C d , and let C ⊆ Ω . We say that dynamics D = ⟨ X,U,f ⟩ are convex-covered by C if X is convex-covered by C . We define a system geometrically-constrained by C as a tuple S C = ⟨ X,U,f,C,x 0 , Y, h ⟩ , where its dynamics ⟨ X,U,f ⟩ is a dynamics convex-covered by C , x 0 ∈ X is the initial state, and h : C × U → Y is the continuous output function. ■

The difference between a shortcut system and a system is that the dynamics function is defined only on X , while the output function is define on the convex-covering C .

Weextend the definition of implementing a function to shortcut systems: S C implements F : Σ + → Γ with encoder enc : Σ → U and decoder dec : Im h → Γ if enc , dec are continuous and F ( w ) = dec ◦ S ( enc( w ) ) .

Construction 4. Consider Linear Recurrent Dynamics with state-space X = Z , input space U = { a, b } and dynamics function f ( n, a ) = n +1; f ( n, b ) = n -1 . The space C = ( -∞ , -0 . 5) ∪ ( -0 . 5 , 0 . 5) ∪ (0 . 5 , ∞ ) is a convex-covering for this dynamics. We may define the output function h : C →{ 0 , 1 } to map points in ( -∞ , -0 . 5) ∪ (0 . 5 , ∞ ) to 0 and points in ( -0 . 5 , 0 . 5) to 1 . Picking initial state x 0 = 0 , we have that this GCS outputs 0 precisely when the input has the same number of a s and b s. This recognizes the language

```
{ w ∈ { a, b } + : w has as many a s as b s. } ,
```

whose dynamics can be interpreted as a counter, with a corresponding to +1 and b corresponding to -1 .

Lemma 52. For a cascade D = D 1 ⇝ · · · ⇝ D n with D i convex-covered/convex-separated by C i we have that C is convex-covered/convex-separated by C = C 1 ×··· × C n

Proof. Suppose D i is convex-covered by C i for i ∈ [1 ..n ] . First, C 1 ×··· × C n is indeed a convexcovering. A product of convex sets is convex, and so a product of finite unions of convex sets is also a finite union of convex sets (by commutativity of set product and union, see proof of Lemma 19). Thus, X 1 ×··· × X n ⊆ C and D is convex-covered by C .

Now, suppose further that D i is convex-separated by C i for i ∈ [1 ..n ] . The path-connected components of C are of the form ∏ n i =1 G i , where G i is a path-connected component of C i . Similarly, path-connected components of X = X 1 ×··· × X n are of the form ∏ n i =1 Z i where Z i is a pathconnected component of X i .

We have that ∏ n i =1 Z i intersects ∏ n i =1 G i precisely when Z i intersects G i for each i ∈ [1 ..n ] . Hence, there is exactly one component of C intersecting ∏ n i =1 Z i , i.e., C convex-separates D .

We begin by defining a restricted type of cascade. This model corresponds more to the idea of joining the cascade components by their respective output function. Thus, we require that the connection between sequential blocks respects convex-coverings.

Definition 35. A constrained cascade D 1 C 1 ⇝ · · · C n -1 ⇝ D n w.r.t. covering C 1 ×···× C n is a dynamics D 1 ⇝ · · · ⇝ D n , where D i = ⟨ X i , U × C [1 .. ( i -1)] , f i ⟩ and D i is convex-covered by C i .

We can think of a constrained cascade as a feed-forward cascade with connections D 1 g 1 ⇝ · · · g n -1 ⇝ D n where each g i is continuous on U × C [1 ..i -1] .

## E.1 Aperiodic Convex-covered Dynamics and Proof of Theorem 9

We define an analogous notion of aperiodicity for convex-covered dynamics. First we extend the notion of η -convergence to convex-coverings.

Definition 36. For a space X , we say a sequence ( x n ) n ≥ 1 ∈ X ω PC-converges in X , if eventually all its terms lie in the same path-connected component of X . ■

This is an identical notion to η -convergence, but we give it a different name, since it applies to nonη -finite spaces.

Definition 37. Call dynamics D = ⟨ X,U,f ⟩ aperiodic w.r.t. convex-covering C , if D is convexcovered by C and if for every sequence ( u n ) n ≥ 1 ∈ U ω PC-convergent in U and x 0 ∈ X , the state sequence ( D ( x 0 , u 1 ...n ) ) n ≥ 1 ∈ X ω ⊆ C ω is PC-convergent in C .

Note the difference in definition: we require that the state sequence is eventually in the same component of C , instead of the same component of X !

Lemma 53. Let D = D 1 ⇝ · · · ⇝ D n be a cascade s.t. D i is aperiodic w.r.t. convex-covering C i for i ∈ [1 ..n ] . Then D is aperiodic w.r.t. convex-covering C = C 1 ×··· × C n .

Proof. Analogous to proof of Lemma 49, applied to the cascade D ′ 1 ⇝ · · · ⇝ D ′ n , where D ′ i = ⟨ C i , U × C [1 ,...i -1] , f i ⟩ .

Definition 38. We call a function F : Σ + → Γ alternating if, for some σ ∈ Σ , the sequence ( F ( σ n ) ) n ≥ 1 ∈ Γ ω changes value infinitely many times. ■

Theorem 54. Let D be a dynamics aperiodic w.r.t. convex-covering C . Let S C be a shortcut system constrained by C with dynamics D . Then S C can not implement any alternating function.

Proof. Say D = ⟨ X,U,f ⟩ and S C = ⟨ X,U,f,x 0 , C, Y, h ⟩ . Suppose for contradiction that S C with encoder enc : Σ → U and decoder dec : Im h → Γ implements an alternating function F : Σ + → Γ .

Let σ ∈ Σ be a symbol such that ( F ( σ n ) ) n ≥ 1 changes value infinitely many times. Since D is aperiodic w.r.t. C we have that ( D ( x 0 , enc( σ ) n ) ) n ≥ 1 ⊆ X ⊆ C is eventually in the same path-connected component of C . As dec ◦ h : C × U → Γ is continuous we thus have that

<!-- formula-not-decoded -->

is eventually in the same path-connected component of Γ , i.e. eventually constant. This is a contradiction.

We now introduce an elementary theorem about convex sets in R d (or C d ).

Theorem 55 (Minkowski's Hyperplane Separation Theorem) . Let A,B ⊆ R d be two disjoint, nonempty convex sets. If both are open, then there exists a non-zero vector v ⊆ R d and constant c ∈ R s.t.

<!-- formula-not-decoded -->

with ⟨· , ·⟩ being the dot product.

Proof. By Section 2.5.1 of [Boyd and Vandenberghe, 2006 - 2004], we have that there exists a non-zero vector v ⊆ R d and constant c ∈ R s.t.

<!-- formula-not-decoded -->

Now, these inequalities in fact must be strict. For contradiction suppose that ⟨ a, v ⟩ = c for some a ∈ A . Since A is open, we have that for some ϵ &gt; 0 B R d ( a, ϵ ) ⊆ A . Thus a + ϵ · v || v || 2 2 ∈ A ( || v || 2 = 0 as v is a non-zero vector). But then 〈 a + ϵ · v || v || 2 2 , v 〉 = a + ϵ &gt; a by linearity of the dot product. Similarly for B .

̸

Theorem 9. Let D be an η -finite Linear Recurrent Dynamics, with its state-transition gates having all non-negative eigenvalues. Let C be a covex-regular covering of D . Then D is aperiodic w.r.t. C .

Proof. Let D = ⟨ X,U,f ⟩ be a Linear Recurrent Dynamics, with X ⊆ R d , convex-covered by C , s.t. A ( u ) has all its eigenvalues being real, for all u ∈ U . Say f ( x, u ) = A ( u ) · x + B ( u ) .

Consider a sequence ( u n ) n ≥ 1 ∈ U , state-convergent in U , and x 0 ∈ X . Let ( x n = D ( x 0 , u 1 ...n ) ) ⊆ X be the corresponding state sequence. We have some N s.t. for n ≥ N all u n are contained in the same component of U , we may pick a representative r ∈ U of that component.

Write A = A ( r ) , B = B ( r ) . By Lemma 25, we have for n ≥ N that

<!-- formula-not-decoded -->

Like in proof of Theorem 47, we consider the state sequence in the diagonalized space of A . Write A = P -1 JP for the Jordan normal form of A . Here J is block diagonal, with say blocks J 1 , ..., J s , J b ∈ R m b × m b being a Jordan Block with λ b -eigenvalue of A -on the diagonal, and 1 on the right off-diagonal.

Define y n = x n +1 -x n and y ′ n = P ( x ′ n +1 -x ′ n ) , then

<!-- formula-not-decoded -->

Thus, unrolling the recurrence we get

<!-- formula-not-decoded -->

The i -th component of y ′ n , where i is in say the b -th block of J , is

<!-- formula-not-decoded -->

The binomial coefficients are polynomial in n . Thus we may write [ y ′ n ] i = ∑ v j · n b j · a n j , where b j ∈ Z ≥ 0 and a j = λ b ≥ 0 , which is of the form in Lemma 64. Since y n = Py ′ n , we have

<!-- formula-not-decoded -->

which again is of the form in Lemma 64.

Now, for contradiction suppose that x ′ n is not state-convergent in C . Then, since C has finitely many components, there are two distinct components of C , say C 1 , C 2 such that x ′ n is in both C 1 and in C 2 infinitely often. Furthermore, since C 1 , C 2 are finite unions of open convex sets, there are convex, open sets S 1 , S 2 which are disjoint, non-empty, and x ′ n is in both S 1 and S 2 infinitely often (*).

By Theorem 55, there is a non-zero vector v ∈ R d and constant c ∈ R s.t. ⟨ s 1 , v ⟩ &gt; c ∀ s 1 ∈ S 1 and ⟨ s 2 , v ⟩ &gt; c ∀ s 2 ∈ S 2 .

Thus, ⟨ x ′ n , v ⟩ &gt; c infinitely often, and ⟨ x ′ n , v ⟩ &lt; c infinitely often.

We have

<!-- formula-not-decoded -->

is again in the form from Lemma 64. Thus it is eventually monotone. Therefore eventually ⟨ y n , v ⟩ ≤ 0 , in or ⟨ y n , v ⟩ ≥ 0 . By linearity of the inner product

<!-- formula-not-decoded -->

Thus, eventually also ⟨ x n , v ⟩ is monotone-contradiction with (*).

## E.2 Weakly η -finite Dynamics

In this section we introduce the topological notion of connectedness , as well as the necessary results to establish the finite state properties of GCSs where the state-space coincides with the convex-covering.

Definition 39. A topological space X is called disconnected , if there are disjoint non-empty sets H,K in X such that X = H ∪ K . Then X is called connected if it is not disconnected.

Connectedness is, as it turns out, a generalization of path-connectedness.

Fact E.2.1. (Theorem 27.2, [Willard, 2012]) Every path-connected space is connected.

Similarly to compactness and path-connectedness, connectedness is preserved by continuous mappings and products.

Fact E.2.2. (Theorem 26.2, [Willard, 2012]) The continuous image of a connected space is connected.

Fact E.2.3. (Theorem 26.10, [Willard, 2012]) A nonempty product space is connected iff each factor space is connected.

Similarly to path-connectedness, connectedness induces an equivalence on the space.

Definition 40. For x ∈ X , define C x as the union of connected subspaces of X containing x . We call it the C-component at x . We write x ≈ X y when y ∈ C x .

Note, that in [Willard, 2012] C-components are simply referred to as components .

Fact E.2.4. ≈ X is an equivalence relation, partitioning X into maximal (with respect to inclusion) connected subspaces of X . C x is the equivalence class of ≈ X containing x . See Theorem 26.7 and Definition 26.11 of [Willard, 2012] for details.

Fact E.2.5. (Theorem 26.12, [Willard, 2012]) The C-components of X are closed in X .

Thus, we think of C-components as a partition of the space that is a coarsening of the path-connected components. For an example of a space that has one C-connected component and 2 path-connected components, see the topologist's sine curve (Example 27.3, [Willard, 2012]).

Definition 41. We call a space X weakly η -finite , if it has finitely many C-components.

Example 10 . Any finite alphabet is weakly η -finite, with each symbol being in a separate Ccomponent. ■

Our goal now is to show that weakly η -finiteness enjoys the same favourable theoretical properties as η -finiteness.

Lemma 56. A continuous image of a weakly η -finite space is weakly η -finite.

Proof. Let C 1 , . . . , C n be the C-components of X , and let f : X → Y be continuous. Each f ( C i ) is connected, and so Im f is a union of finiely many connected spaces f ( C 1 ) , . . . , f ( C n ) . Thus, the equivalence classes of ≈ Im f must be unions of these images. Thus ≈ Im f must have finitely many equivalence classes.

Lemma 57. The Cartesian product X × Y space of weakly η -finite spaces is weakly η -finite. The C-components of X × Y are the products of C-components of X and C-components of Y .

Proof. Let C 1 , . . . , C n and E 1 , . . . , E , be the C-components of X,Y respectively. We have X = ⋃ n i =1 C i , Y = ⋃ m j =1 and so

<!-- formula-not-decoded -->

By Fact E.2.2 each C i × E j is connected. Thus, the C-components of X × Y are unions of the products C i × E j . Now, fix i ∈ [1 ..n ] , j ∈ [1 ..j ] . Let Z be the C-component of X × Y containing C i × E j . consider the projection map π X : X × Y → X . As the projection is continuous, the image, π X ( Z ) is connected in X . Moreover, C i ∈ π X ( Z ) . Thus, as C i is a maximal connected subspace of X , we have C i = π X ( Z ) . Similarly, considering the projection π Y : X × Y → X , we have E j = π X ( Z ) . Since C i × E j ⊆ Z , we therefore must have C i × E j = Z . Therefore X × Y has finitely many C-components, and they are the products of C-components of X and C-components of Y .

Lemma 58. Let X be weakly η -finite and Σ be a finite alphabet. Then f : X → Σ is continuous if and only if it is constant on the C-components of X .

Proof. ( ⇒ ) Let f : X → Σ be continuous. Let C be a C-component of X . By Fact E.2.2, f ( C ) ⊆ Σ is connected, and so f ( C ) = { σ } for some σ ∈ Σ . I.e., f is constant on the C-components of X .

( ⇐ ) Let f : X → Σ be constant on the C-components. Let Y ⊆ Σ be closed. Then f -1 ( Y ) ⊆ X must be a union of finitely many C-components, since X is weakly η -finite. By Fact E.2.5, we have that each C-component is closed, and therefore also f -1 ( Y ) is closed, as a finite union of closed sets. Thus f is continuous.

Now, we have all the properties needed to carry out the arguments in Appendix B.3.

Definition 42. We call dynamics D = ⟨ X,U,f ⟩ weakly η -finite if X and U are weakly η -finite. We call a system S weakly η -finite if its dynamics are weakly η -finite.

By Lemma 57, we immediatly have that cascades of weakly η -finite dynamics are weakly η -finite.

Example 11 . η -finite dynamics are weakly η -finite.

■

Theorem 59. A convex-covering C is weakly η -finite, with its C-components coinciding with its path-connected components.

Proof. Let C 1 , . . . , C n be path-connected components of C . Each C i is a union of finitely many open (in R d ) convex sets, and so is also open. Let Z be a C-component of C . Then Z is a union of the path-connected components, and so Z is also open. An open, connected subspace of R d is pathconnected, see Corollary 27.6 of [Willard, 2012]. Thus Z must actually be one of the path-connected components.

Lemma 60. Let D = ⟨ X,U,f ⟩ be a geometrically-contrained system, convex-covered by C , with X = C . Then D is weakly η -finite, and the C-components of X are the path-connected components.

Proof. C has finitely many path-connected components, and so it is weakly η -finite, since pathconnectedness implies connectedness. Now, each C-component of C is a union of the path-connected components, all of which are open in Ω = R d . Hence each C-component of C is open in Ω . By Corollary 27.6 of [Willard, 2012], C-components of C are therefore path-connected. Thus the path-connected components and C-connected components of C coincide.

Since a C-component has to be mapped by a continuous function into a single C-component, we have that a version of Lemma 24 also holds for weakly η -finite dynamics. For a weakly η -finite system S = ⟨ X,U,f,x 0 , Y, h ⟩ and weakly η -finite dynamics D = ⟨ X,U,f ⟩ , we can thus define the analogous canonical automata

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, replacing path-equivalence ∼ with C-component-equivalence ≈ in Lemmas 26, 27, 28 and Theorem 1, we get that the canonical automata of weakly η -finite systems have the same capability in terms of implementing functions.

Likewise, the realization results of Appendix B.4 and Appendix B.5 carry over to the setting of weakly η -finiteness. Thus we may apply the structural theorems of Algebraic Automata Theory in the case of weakly η -finite dynamics. We defer exploring the properties of weakly η -finite dynamics in detail to future work.

## E.3 η -finite Systems as GCSs and Proof of Theorem 8

We start by showing that η -finite dynamics that are convex-separated by C can implement exactly the same functions in a η -finite system as in a GCS constrained by C .

Lemma 61. Suppose η -finite dynamics D are convex-separated by C . The following are equivalent:

- There is a system with dynamics D that can implement F : Σ + → Γ .
- There is a shortcut system S C constrained by C with dynamics D that can implement F : Σ + → Γ .

Proof. ( ⇒ ) Let S = ⟨ X,U,f,x 0 , Y, h ⟩ be a system with dynamics D that implements F with some encoder enc : Σ → U and decoder dec : Y → Γ .

̸

Let C 1 , ..., C s be the path-connected components of C . Fix γ ∈ Γ and define h ′ : C × U → Γ as follows: for i ∈ 1 . . . s , if C i ∩ X = ∅ , take h ′ ( c, u ) = γ , where γ ∈ Γ . If C i ∩ X = ∅ , take h ′ ( c, u ) = dec ◦ h ( x, u ) for ( c, u ) ∈ C i × U , where x ∈ X i . This is well-defined: For all x, x ′ ∈ C i ∩ X , since C is a convex-separator of X , we have that x and x ′ are in the same path-connected component of X . Therefore necessarily dec ◦ h ( x, u ) = dec ◦ h ( x ′ , u ) .

Want to show : h ′ is continuous. Let ( ( c n , u n ) ) n ≥ 1 ⊆ C × U be a sequence converging to ( c, u ) ∈ C × U . Then ( c n ) n ≥ 1 converges to c in C and ( u n ) n ≥ 1 converges to u in U .

Let C i be the component that contains c . Since C i is open, there is some ϵ &gt; 0 s.t. B Ω ( c, ϵ ) ⊆ C i . Since c n → c , we must have that eventually ( c n ) lies in B Ω ( c, ϵ ) ⊆ C i .Similarly, let U j be the η -component of U that contains u . Then, by Lemma 20, as u n → u , we must have that eventually ( u n ) lies in U j . Thus eventually ( ⟨ c n , u n ⟩ ) n ≥ 1 lies in C i × U j . By definition of h ′ , it is constant on C i × U j . Thus ( h ′ ( u n , c n ) ) n ≥ 1 is eventually equal h ′ ( c, u ) .

Now, define S c = ⟨ X,U,f,x 0 , C, Y, h ′ ⟩ . As h ′ : C × U → Y is continuous, this is a well-def. shortcut system constrained by C . Moreover, since h ′ constrained to X × U is equal to dec ◦ h , we have that S C with encoder enc and decoder id : Γ → Γ implement F .

( ⇐ ) Let S c = ⟨ X,U,f,x 0 , C, Y, h ⟩ be a shortcut constrained by C . Suppose that S C implements F with some encoder enc : Σ → U and dec : Y → Γ . Then taking h : X × U → Y to be the restriction of h , we get that the system S = ⟨ X,U,f,x 0 , Y, h ′ ⟩ with encoder enc and decoder dec implements F .

Lemma 62. Let X be a η -finite space. Then X is convex-separated by some convex-covering C .

Proof. Let X 1 , . . . , X k be the components of X . Take

<!-- formula-not-decoded -->

Then we have δ &gt; 0 by Lemma 20. Define

<!-- formula-not-decoded -->

Then C δ i is an open cover of X i . Since X i is compact, by definition of compactness there is a finite subcover ¯ C δ i ⊆ C δ i which also covers X i . Moreover, by definition of δ , this subcover does not intersect other components of X . Taking C i = ⋃ ¯ C δ i we have that C = C 1 ∪ · · · ∪ C k is a convex-covering that convex-separates X .

Construction 5. FLIP-FLOP dynamics can be implemented by a Linear Recurrent Dynamics with entries in [ δ, 1 -δ ] , for some δ &gt; 0 .

Let ϵ &lt; 1 . Take D = ⟨ X,U,f ⟩ with X = X l ∪ X h , where X l = ( -1 , 0) , X h = (0 , 1) and U, f such that:

<!-- formula-not-decoded -->

With output function X l ↦→ low and X h ↦→ high , this implements FLIP-FLOP. The set C = X is a convex-covering of this dynamics.

Hence, Mamba can implement FLIP-FLOP as a constrained system, and so constrained cascades of Mamba blocks can implement any star-free language.

Corollary 63. η -finite dynamics are in particular convex-separated dynamics, and implement the same functions in η -finite systems and in GCSs.

Theorem 8. SSMs with Mamba parametrisation can recognise all star-free languages as GCSs.

Proof. By Construction 5, there is a Mamba block dynamics D , with a convex-covering state space, and η -finite input space, that realise FLIP-FLOP as weakly η -finite dynamics. A Mamba block can also have a convolution, and so there is a Mamba block dynamics E , with a convex-covering state space, and η -finite input space, that realise R 2 as weakly η -finite dynamics (details omitted. Also a sLSTM-like η -finite construction is possible, see Appendix G.3). Thus, by weakly η -finite analogue of Theorem 14, all group-free functions can be realized by feed-forward cascades of D and E components. Such cascades are actually constrained cascades of Mamba block GCSs, since the convex-coverings of D and E coincide with their state-spaces.

Figure 10: FLIP-FLOP task [Liu et al., 2023]. PCA of a trained 1-layer Mamba states for each channel: red and blue are state sequences under i0 inputs, starting from w1 and w0 respectively. After ≈ 1000 inputs, both state sequences give the same predictions on the read instruction r , incorrectly. The 'doubled' state trajectories are due to each transition consisting of 2 input tokens.

<!-- image -->

## F Details of The Experiments

We have created visualizations based on the [Liu et al., 2023] FLIP-FLOP task. The dataset is available at https://huggingface.co/datasets/synthseq/flipflop/ . The objective of the task is to predictively model a sequence of instructions of the form sx , where s ∈ w,r,i , x ∈ 0,1 . w indicates that the next symbol is to be stored, r indicates that the next symbol should be the retrieved value and i indicates no action. The specific task we trained on corresponds to the "clean" prediction mode, where only prediction following an r instruction need to be predicted. We note that the aim of our experiments was to obtain empirical evidence of Mamba having contracting dynamics, and a comprehensive experimental study is beyond the scope of our paper.

We trained 1-layer Mamba on sequence lengths 32, 64, and 512, observing similar state-collapse phenomena, as predicted by our results. Additionally [Sarrof et al., 2024] note that in their experiments Mamba needed more training steps to converge than reported by Liu et al. [2023] for an LSTM. This is another evidence towards the influence of robustness on stability of training.

The code used to perform the experiments is based on the repository shared in Grazzi et al. [2025], with some environment modifications to make it work on the 2025-04-09 Google Colab release. The forked repository is available at https://github.com/adankow/unlocking\_state\_tracking , with a Google Colab notebook file containing the set-up, simple training loop, and hidden state visualisation code.

## G Additional Proofs and Constructions

## G.1 Monotone Sequence Lemma

Lemma 64. Let d ≥ 1 , a 1 , ..., a d ≥ 0 , b 1 , .., b d ∈ Z ≥ 0 and v 1 , ..., v d ∈ R . The sequence

<!-- formula-not-decoded -->

is eventually monotone.

̸

Proof. If all v i = 0 , then x n = 0 for all n , in particular the sequence is monotone. Otherwise, we may assume that v i = 0 for all i , and that

<!-- formula-not-decoded -->

If a 1 = 0 , then again x n = 0 , and it is monotone. Otherwise, we can take d 1 : 1 ≤ d 1 ≤ d such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where P ( n ) is the polynomial ∑ d 1 i =1 v i · n b i .

̸

Case 1: a 1 = 1 . We have a 1 &gt; 0 and

<!-- formula-not-decoded -->

̸

We have that ( a i /a 1 ) → 0 as n →∞ , since a 1 &gt; a i for d 1 +1 ≤ i ≤ d . On the other hand, P ( n ) is a non-zero polynomial, since its leading term is v 1 · n b 1 and v 1 = 0 , and so P ( n ) →±∞ as n →∞ . Thus, x n = 0 for sufficiently large n . Moreover,

̸

<!-- formula-not-decoded -->

We have P ( n ) /n b 1 → v 1 as n →∞ , since v 1 · n b 1 is the leading term of P ( n ) . Also n b i -b 1 grows at most polynomially, while ( a i /a 1 ) n goes to 0 exponentially, since a i &lt; a 1 for d 1 +1 ≤ i ≤ d . Therefore ∑ d i = d 1 +1 v i · n b i -b 1 ( a i /a 1 ) n -→ 0 as n →∞ . Lastly we have ( n +1) b 1 n b 1 → 1 as n →∞ . All together

<!-- formula-not-decoded -->

In particular, eventually x n is positive, or eventually it is negative. There are 4 cases:

- If a 1 ∈ (0 , 1) and x n is positive eventually, then x n is decreasing eventually.
- If a 1 ∈ (1 , ∞ ) and x n is positive eventually, then x n is increasing eventually.
- If a 1 ∈ (0 , 1) and x n is negative eventually, then x n is increasing eventually.
- If a 1 ∈ (1 , ∞ ) and x n is negative eventually, then x n is decreasing eventually.

Case 2: a 1 = 1 . We proceed by induction on b 1 . If b 1 = 0 , then necessarily d 1 = 1 , and P ( n ) = v 1 . Then we have by Case 1 that x n -P ( n ) = x n -v 1 is eventually monotone, and so also x n is eventually monotone.

We may write

For the inductive step, consider

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can again write ∑ d i = d 1 +1 v i · a n i · ( a i ( n +1) b i -n b i ) as ∑ d ′ i =1 v ′ i · n b ′ i · ( a ′ i ) n , with a ′ i &lt; a 1 = 1 . On the other hand Q ( n ) = P ( n +1) -P ( n ) is a polynomial with leading coefficient of degree &lt; b 1 . Thus we may apply inductive hypothesis to

<!-- formula-not-decoded -->

to conclude that y n is eventually monotone. Thus, either x n +1 -x n = y n ≤ 0 eventually, or x n +1 -x n = y n ≥ 0 eventually. Hence x n is eventually monotone.

## G.2 Sequential Cascade Construction

The serial cascade can be realised in terms of the feedforward cascade ⇝ . Consider i ∈ 1 , 2 and D i = ⟨ X i , U i , f i ⟩ . Define the repeat dynamics on X 1 to be the system R X 1 = ⟨ X 2 1 , U × X 1 , r ⟩ , with r given by

<!-- formula-not-decoded -->

Thus R X 1 can delay the propagation of the state of D 1 by one time step. Also, define the modified dynamics D ′ 2 = ⟨ X 2 , U × X 3 1 , f ′ 2 ⟩ , with f ′ 2 given by

<!-- formula-not-decoded -->

Note that R X 1 is equivalent to the usual repeat dynamics over X 1 , ⟨ X 2 1 , X 1 , r X ⟩ , but with input function ( u, x ) ↦→ x .

Now, the feed-forward cascade D 1 ⇝ R X 1 ⇝ D ′ 2 is well-defined, and has the following transitions:

<!-- formula-not-decoded -->

Now, suppose we have system S = ⟨ X 1 × X 2 , U, f, ( x 1 , 0 , x 2 , 0 ) , Y, h ⟩ with dynamics D 1 ⋉ D 2 . Then there is a system S ′ with dynamics D 1 ⇝ R X 1 ⇝ D 2 which realises S : take S ′ = ⟨ X 3 1 × X 2 , U, f ′ , x ′ 0 , Y, h ′ ⟩ with x ′ 0 = ( x 1 , 0 , x 1 , 0 , x 1 , 0 , x 2 , 0 ) , h ′ ( ⟨ x 1 , 1 , x 1 , 2 , x 1 , 3 , x 2 ⟩ , u ) = h ( ⟨ x 1 , 1 , x 2 ⟩ , u ) and take

<!-- formula-not-decoded -->

Take ι : U → U and ζ : Y → Y to be the identities. We then have for all ( x 1 , x 2 ) ∈ X 1 × X 2 , u ∈ U and x ′ ∈ α (( x 1 , x 2 )) :

<!-- formula-not-decoded -->

where x ′ 1 = f 1 ( x 1 , u ) and x ′ 2 = f 2 ( x 2 ⟨ u, x 1 ⟩ ) , so that ( x ′ 1 , x ′ 2 ) = f ( ( x 1 , x 2 ) , u ) . Moreover x ′ 0 ∈ α ( x 0 ) .

Finally, we have

<!-- formula-not-decoded -->

so that indeed S ′ is a realisation of S . Note, that we did not need to introduce any new transitions on X 1 or X 2 in order to carry out this construction. In particular, if D 1 and D 2 are linear recurrent dynamics, then D 1 , D ′ 2 are linear recurrent dynamics. Also R X 1 is a Finite Context Dynamics.

## G.3 Robust Flip-Flop realisations

Recall the sLSTM parametrisation: the state space of a sLSTM is R 3 , and the input space is R d for some d ≥ 1 . The dynamics function of the form ( ⟨ c, n, h ⟩ , u ) ↦→ 〈 f c ( ⟨ c, n, h ⟩ , u ) , f n ( ⟨ c, n, h ⟩ , u ) , f h ( ⟨ c, n, h ⟩ , u ) 〉 , where

<!-- formula-not-decoded -->

where each l s : s ∈ o, i, z, f is a function of the form w t s · u + r s · h + b s , for w s ∈ R d , r s , b s ∈ R , ψ is either exp or σ , and φ is tanh .

## G.3.1 Strongly robust sLTSM FLIP-FLOP realization

We present a construction for a one layer sLSTM FLIP-FLOP, which is strongly robust. The key idea is to only use the h state to implement the dynamics. Then, we can use Theorem 42, and similar arguments involving uniform continuity, to extend the construction to be strongly robust in the states h, c, n and the input space u . We shall present the arguments in more detail here, to demonstrate how robustness can be used to prove properties of systems, in particular how to extend robustness to strong robustness.

Let ψ = σ . Set w s = 0 and r s = 0 for s = f, i, z . Set b f = -3 , b z = 2 , b i = 0 . Then we have l ≡ -3 , l ≡ 0 , l ≡ 2 . Thus the updates simplify as

<!-- formula-not-decoded -->

Finally, take d = 1 and l o ( h, u ) = u +10 h -5 .

For now, let us fix c as c ∗ = tanh (2) 1 -σ ( -3) ≈ 1 . 01202 and n as n ∗ = 1 1 -σ ( -3) ≈ 1 . 049787 , i.e. the fix points of the linear recurrences given by f c and f n . Then we have that

<!-- formula-not-decoded -->

Moreover, f c ( ⟨ c ∗ ,n ∗ ,h ⟩ ,x ) f n ( ⟨ c ∗ ,n ∗ ,h ⟩ ,x ) = c ∗ n ∗ = tanh2 , so that the update for h simplifies as

<!-- formula-not-decoded -->

We can set U = { u set , u reset , u id } , with u set = 8 , u reset = -8 and u id = 0 , and H low = [ -0 . 05 , 0 . 2] , H high = [0 . 8 , 1 . 05]

Now, for h ∈ [0 , 1] we have

<!-- formula-not-decoded -->

Therefore f ( ⟨ c, n, h ⟩ , u set ) ∈ [0 . 85 , 1] . Similarly

<!-- formula-not-decoded -->

Therefore f ( ⟨ c, n, h ⟩ , u reset ) ∈ [0 , 0 . 05] . Now, for h ≤ 0 . 2

<!-- formula-not-decoded -->

and so f ( ⟨ c, n, h ⟩ , u id ) ∈ [0 , 0 . 05] . Also for h ≥ 0 . 8

<!-- formula-not-decoded -->

and so f ( ⟨ c, n, h ⟩ , u id ) ∈ [0 . 8 , 1] . Thus we see that the dynamics

<!-- formula-not-decoded -->

realise the FLIP-FLOP dynamics, and is η -finite and ϵ -robust, for ϵ = 0 . 05 . Furthermore, we can modify the input space U , to make it strongly ϵ -robust.

Consider U ′ = [0 , 10] . H × U ′ is compact, and f is continuous on H × U ′ , so by Theorem 41 it is uniformly continuous on H × U ′ . In particular, for ϵ ′ = ϵ/ 2 , there exists δ &gt; 0 such that

<!-- formula-not-decoded -->

for all ( x, u ) , ( x ′ , u ′ ) ∈ X ′ × U ′ . Thus, we may take δ ′ = min( δ, 1) and U ′′ = [ u set ± δ ′ ] ∪ [ u reset ± δ ′ ] ∪ [ u id ± δ ′ ] . Now, consider h ∈ H , u ∈ U ′′ and h ′ ∈ R such that || h ′ -f ( h, u ) || ≤ ϵ ′ . We have || u -u ′ || ≤ δ ′ for some u ′ ∈ { u set , u reset , u id } , and so

<!-- formula-not-decoded -->

All together

<!-- formula-not-decoded -->

Since ( h, u ′ ) ∈ H × U and ⟨ H,U,f ⟩ is ϵ -robust, we get that h ′ ∈ H . Hence f also gives a well defined dynamics function H × U ′′ → H , which moreover is ϵ ′ -robust. Thus, we have ⟨ H,U ′′ , f ⟩ is η -finite and strongly min( ϵ ′ , δ ′ ) -robust. It also realizes FLIP-FLOP, since the input components induce the same η -transitions as { u set , u reset , u id } by path-connectedness.

Finally, we extend the dynamics to c and n . We can see f as parametrized by θ ∈ [ c ∗ ± 0 . 5] , ρ ∈ [ n ∗ ± 0 . 5] , given by

<!-- formula-not-decoded -->

So, f = f c ∗ ,n ∗ . We see that f θ,ρ is continuous in θ and ρ , and [ c ∗ ± 0 . 5] × [ n ∗ ± 0 . 5] is compact. Thus by Theorem 42, there is some γ &gt; 0 such that f θ,ρ induces the same function H × U ′′ → H as f c ∗ ,n ∗ . Also, similarly to how we extended U to U ′′ , we can choose γ such that the resulting dynamics are always ϵ/ 4 -robust

Lets take X = H × C × N where C = [ c ∗ ± γ ] and N = [ n ∗ ± γ ] . We have that the sLSTM dynamics gives a well-defined, robust dynamics function X × U → X : we already have that the restriction of the dynamics to the h component is robust. For the c and n components, since σ ( -3) &lt; 1 , the state updates given by f c and f n (which are independent of u ) are contractions towards c ∗ and n ∗ respectively, with rate σ ( -3) . Thus f c sends C = [ c ∗ ± γ ] to [ c ∗ ± γ · σ ( -3)] and f n sends N = [ n ∗ ± γ ] to [ n ∗ ± γ · σ ( -3)] . All together, the sLSTM dynamics are strongly min( ϵ/ 4 , δ ′ , γ (1 -σ (3))) -robust, and realize FLIP-FLOP.

## G.3.2 Strongly robust sLSTM repeat dynamics

To realize any repeat semiautomata, as defined in Appendix G.2, it is sufficient to realize the two state repeat semiautomaton R 2 = ⟨{ 0 , 1 } 2 , { 0 , 1 } , r ⟩ , with r ( ⟨ x old , x new ⟩ , x ) = ⟨ x new , x ⟩ .

Here, the construction is extremely similar to the FLIP-FLOP one. We first show a robust dynamics on just the h cell, using f ( h, u ) = σ ( u +10 h -5) · tanh(2) which realize R 2 . Then we can use the same argument as before to extend it to strongly robust dynamics on all 3 cells.

We can use the h cell to represent x new , by simply reusing the previous strongly robust construction for setting the high and low state, with dynamics function f ( h, u ) = f h ( ⟨ c ∗ , n ∗ , h ⟩ , u ) , state space H and input space [ u set ± δ ′ ] ∪ [ u ] . We then have that for some γ &gt; 0 for all c ∈ [ c ∗ ± γ ] and n ∈ [ n ∗ ± γ ] the dynamics function f h ( ⟨ c, n, h ⟩ , u ) still performs

Define X 00 = [ -0 . 01 , 0 . 015] , X 01 = [0 . 02 , 0 . 05] , X 10 = [0 . 95 , 0 . 98] , X 11 = [0 . 985 , 1 , 01] and u 0 = -8 . 1 , u 1 = 8 . 1 . Note that X = X 00 ∪ X 01 ∪ X 10 ∪ X 11 has 4 η -components. Also, define

X 0 = X 01 ∪ X 10 and X 1 = X 10 ∪ X 11 . In our construction X ab will correspond to the state of R 2 after the last two inputs were ab , a, b ∈ { 0 , 1 } .

We have

<!-- formula-not-decoded -->

As σ is increasing, we therefore have f ( X 1 , u 1 ) ⊆ [0 . 99999 , 1] ⊂ X 11 . Similarly, we have

<!-- formula-not-decoded -->

Therefore f ( X 0 , u 1 ) ⊆ [0 . 952 , 0 . 974] ⊂ X 10 . Similarly for u 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore f ( X 1 , u 0 ) ⊆ [0 . 025 , 0 . 0475] ⊂ X 01 . Similarly

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore f ( X 0 , u 0 ) ⊆ [0 , 0 . 000004] ⊂ X 00 . Thus ⟨ X, { u 0 , u 1 } , f ⟩ are well-defined dynamics and the 4 η -components correspond to 4 possible values for the last 2 inputs. Hence clearly they can realize R 2 . Moreover, the dynamics are strongly robust. The remainder of the argument is the same as for the FLIP-FLOP construction.

## G.3.3 Strongly robust Elman-RNN FLIP-FLOP construction

The following is a modification of a construction in [Knorozova and Ronca, 2024a]. Consider the dynamics function

<!-- formula-not-decoded -->

for x, u ∈ R . Wehave that for all x, u , f ( x, u ) ∈ [ -1 , 1] . Define X low = [ -1 . 1 , tanh( -1)] , X high = [tanh(1) , 1 . 1] . Note that tanh(1) ≈ 0 . 76159 , tanh( -1) ≈ -0 . 76159

We have

<!-- formula-not-decoded -->

As tanh is increasing, we have f ([ -1 . 1 , 1 . 1] , 4) ⊆ [0 . 9467 , 0 . 999993] ⊂ X high . Similarly, f ([ -1 . 1 , 1 . 1] , -4) ⊆ [ -0 . 999993 , -0 . 9467] ⊂ X low . Moreover

<!-- formula-not-decoded -->

Thus, f ( X high , 0) ⊆ [0 . 908 , 0 . 9757] ⊂ X high . Similarly f ( X low , 0) ⊆ [ -0 . 9757 , -0 . 909] ⊂ X low . Thus we see that, taking X = X low ∪ X high , u set = 4 , u reset = -4 , u id = 0 , the η -finite dynamics ⟨ X, { u set , u reset , u id } , f ⟩ are well-defined, and realize FLIP-FLOP. Also clearly they are robust. Now, by the same argument as for the sLTSM FLIP-FLOP realisation, we can extend the input space, using Theorem 16 and Theorem 42, to obtain a strongly robust construction.

## H Further Discussion on Related Work

Sarrof et al. [2024] show that, in the finite-precision setting, regular languages that can be modelled by diagonal linear-recurrences with non-negative entries-like Mamba-are precisely the star-free languages. The setting differs from ours, in that it allows finite fractional precision, but unbounded number of integer bits. With that, a number of positive expressivity results for counter languages is given. The empirical experiments show that SSMs indeed can model such languages on in-distribution lengths, but with limited length-generalisation. The finite-precision arguments in this work are not fully formal, essentially ignoring the error of the linear dynamics carried out in finite precision. Weiss et al. [2018] use the same finite-precision setup with unbounded integer bits to show that ReLU-activated Elman-RNN and LSTM can implement counting behaviour, while Elman-RNNs with squashing activations and GNU [Cho et al., 2014, Chung et al., 2014] cannot.

Grazzi et al. [2025] extend Mamba and DeltaNet parametrisations to allow for gates with negative eigenvalues. The work proves that linear recurrences, with gates having non-negative eigenvalues, are restricted to modelling star-free recurrent languages in the finite-precision setting. Their framework differs from [Sarrof et al., 2024] and ours, assuming that the linear recurrence is computed in convolutional form in some finite datatype D , with some operations carried out in infinite precision before casting back into D . This setting is more explicit in its assumptions than [Sarrof et al., 2024], but not generalisable to other types of recurrence.

̸

Merrill et al. [2024] use the parallelisability aspect of SSMs to obtain an expressivity classification in terms of Circuit Complexity in the log-precision setting , i.e., precision logarithmic in the input length. Assuming a particular datatype is used to carry out the operations, it shows that SSMs, including Mamba, can be simulated in the TC-0 circuit class. Thus Mamba is unable to solve the S 5 word problem, even with log-precision, under the widely-accepted conjecture that TC-0 = NC-1. The log-precision framework offers a unique perspective on the drawbacks of parallelism of SSMs.

The Turing-completeness capabilities of Elman-RNNs as offline models of computation are studied in [Siegelmann and Sontag, 1995, Kilian and Siegelmann, 1996, Hobbs and Siegelmann, 2015, Chung and Siegelmann, 2021]; differently, we study RNNs as online models, reading input elements as they arrive. A form of asymptotic expressivity of RNNs is studied in [Merrill et al., 2020], when weights tend to infinity; differently, we consider actual weights. A rich literature surveyed in [Strobl et al., 2024] focuses on the expressivity of Transformers, that constitute an alternative to RNNs as they also operate on sequences.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification:

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification:

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

Justification:

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

Justification:

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification:

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

Justification:

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

Justification:

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

Justification:

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

Justification:

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification:

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

Justification:

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification:

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

Justification:

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification:

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

Justification:

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

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [No]

Justification: Not used

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.