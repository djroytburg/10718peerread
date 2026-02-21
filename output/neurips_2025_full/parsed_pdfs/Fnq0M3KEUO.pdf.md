## A Counterfactual Semantics for Hybrid Dynamical Systems

Andy Zane 1 , 2

andy@basis.ai

Dmitry Batenkov

1

dima@basis.ai

Jeremy Zucker 3 jeremy.zucker@pnnl.gov

Rafal Urbaniak 1

rafal@basis.ai

Sam Witty 4 ˚

sam@sorbus.ai

1 Basis Research Institute, New York, NY 2 University of Massachusetts Amherst, Amherst, MA 3 Pacific Northwest National Laboratory, Richland, WA 4 Sorbus AI

## Abstract

Models of hybrid dynamical systems are widely used to answer questions about the causes and effects of dynamic events in time. Unfortunately, existing causal reasoning formalisms lack support for queries involving the dynamically triggered, discontinuous interventions that characterize hybrid dynamical systems. This mismatch can lead to ad-hoc and error-prone causal analysis workflows in practice. To bridge the gap between the needs of hybrid systems users and current causal inference capabilities, we develop a rigorous counterfactual semantics by formalizing interventions as transformations to the constraints of hybrid systems. Unlike interventions in a typical structural causal model, however, interventions in hybrid systems can easily render the model ill-posed. Thus, we identify mild conditions under which our interventions maintain solution existence, uniqueness, and measurability by making explicit connections to established hybrid systems theory. To illustrate the utility of our framework, we formalize a number of canonical causal estimands and explore a case study on the probabilities of causation with applications to fishery management. Our work simultaneously expands the modeling possibilities available to causal inference practitioners and begins to unlock decades of causality research for users of hybrid systems.

## 1 Introduction

Models of continuous-time dynamical systems are powerful tools for describing real-world mechanisms. From contrastive queries about system behavior under different control policies (Kirk, 2004), to sensitivity analyses designed to aid in understanding which parameters drive system variation (Cacuci, 2003), scientists, policy makers, and engineers often use such models to answer their 'what-if' and causal questions. Unfortunately, causal reasoning with continuous-time systems can be ad-hoc, manual, and error-prone in daily practice.

In parallel, researchers in causal inference have built rigorous tools for answering an expansive taxonomy of causal queries. For example, causal questions about effect estimation (Pearl, 2009; Rubin, 1974; Imbens &amp; Rubin, 2015), counterfactual reasoning (Pearl, 2009, Ch. 7), mediation

˚ Research conducted at Basis.

analysis (Pearl, 2001), responsibility, blame (Chockler &amp; Halpern, 2004), attribution, and explanation (Halpern &amp; Pearl, 2005a,b; Beckers, 2022) can all be succinctly expressed as estimands constructed from parallel worlds (Balke &amp; Pearl, 1994; Avin et al., 2005; Shpitser &amp; Pearl, 2008) or potential outcomes (Rubin, 1974). The toolkit also affords a formal means of determining when those estimands can be reduced to computationally tractable, probabilistic estimation problems (Pearl, 1995; Shpitser &amp;Pearl, 2006; Hernán &amp; Robins, 2023). These insights have made it possible to build general-purpose technology for causal reasoning, such as the causal probabilistic programming language ChiRho (Bingham et al., 2021; Witty, 2023; Basis-Research, 2025). 2

Despite significant progress over the last decade (Mooij et al., 2013; Hansen &amp; Sokol, 2014; Blom et al., 2019; Forré &amp; Mooij, 2020; Peters et al., 2020; Blom et al., 2021; Bongers, 2022; Blom &amp; Mooij, 2023; Boeken &amp; Mooij, 2024; Peters &amp; Halpern, 2025), however, gaps remain in the technical capacity of modern causal reasoning machinery to operate on the full breadth of interventions that can be encoded in continuous-time dynamical systems. In particular, a counterfactual semantics for dynamically triggered, instantaneous intervention has not yet been established. With such an intervention semantics in hand, causal reasoning can be more fully mechanized for causal questions about dynamic temporal events, dramatically expanding the rigor and variety of queries available to users of continuous-time dynamical systems.

Such interventions underpin many closed-loop control problems: for example, HVAC systems activate when temperature thresholds are reached; lockdown and masking measures can be implemented according to levels of Sars-CoV-2 in wastewater (Kappus-Kron et al., 2024); commercial fishing pressure can be reduced once annual harvest limits are reached (Anon, 2007b; Warlick et al., 2018); central banks adjust interest rates depending on economic indicators like inflation and unemployment; reservoir managers release water depending on storage thresholds and agricultural needs (Ray, 2003); and power grids activate 'peaker plants' (or stored energy) when demand exceeds certain thresholds (Zhuk et al., 2016). Despite limited attention from the causality community, these systems have garnered significant interest from control theorists in the form of continuous-time, hybrid dynamical systems (Schaft et al., 2000; Goebel et al., 2012; Sanfelice, 2021) that encode both continuous and instantaneous dynamics in a set of differential and difference constraining equations.

To construct a counterfactual semantics for state-dependent, instantaneous intervention, we formalize a class of transformations on hybrid system constraints that induce the desired counterfactual behavior. An intervention creates a twin, parallel world with transformed constraints, but in a way that ensures both the twin and original worlds share randomly sampled values for initial conditions and parameters. This induces a familiar joint distribution over counterfactual outcomes (Rubin, 1974; Balke &amp; Pearl, 1994; Shpitser &amp; Pearl, 2006, 2008) that can, in turn, be used as input to established causal estimands, such as an expected treatment effect or the probabilities of necessary and sufficient causation.

Our contributions are:

1. A formal, counterfactual semantics for dynamically triggered, instantaneous interventions in continuous-time dynamical systems.
2. Under minimal requirements on interventional specifications, proof that sufficient conditions for solution existence, uniqueness, and finite-time measurability are preserved in the intervened system. Our framework also explicitly connects to established well-posedness conditions on hybrid dynamical systems.
3. A case study on the probabilities of necessary and sufficient causation applied to fishery management, demonstrating extensibility to non-trivial causal estimands rarely applied to dynamical systems.

## 2 Related Work

In Causality. Many researchers have contributed to the systematization of causality for dynamical systems. Hansen &amp; Sokol (2014), for example, show that dynamical systems can be unrolled into directed, structural causal models (SCMs). 3 In the context of ordinary differential equations (ODEs),

2 https://github.com/BasisResearch/chirho

3 Structural causal models (SCMs) are systems of deterministic functions on endogenous variables that additionally incorporate exogenous random noise. SCMs come equipped with a widely studied interventional semantics. We refer the reader to highly influential work of Pearl (2009) for further background.

if f is the right-hand side of the continuous-time differential equation x 1 ' f p x q , we can write structural equations x t ' x t ´ ∆ t ` f p x t ´ ∆ t , u q ∆ t , where t ě 0 , x t P R is the value of the state variable x at time t , u P R is a fixed realization of exogenous noise, and x 0 is fixed. Taking ∆ t Ñ 0 , we can recover the system's dynamics arbitrarily well. This limit results in SCMs with infinitely many variables - a modality that has been recently studied as 'Generalized Structural Equation Models' (GSEMs) (Peters &amp; Halpern, 2021; Halpern &amp; Peters, 2022; Peters &amp; Halpern, 2025). With ∆ t ą 0 , this becomes the familiar discrete time approximation, which has been widely researched in causal inference (Spirtes, 2013; Pearl, 2009; Murphy, 2002; Wang et al., 2018; Assaad et al., 2022; Runge et al., 2023; Zan et al., 2024).

This forward-Euler representation, however, is not the preferred tool of hybrid systems theorists, making it ill-suited for identifying conditions under which intervention preserves established wellposedness conditions. Additionally, under the forward-Euler representation, interventional transformations that induce state-dependent jumps require 'soft' intervention (Correa &amp; Bareinboim, 2020) on all endogenous nodes that might jump. Indeed, the state-dependent jump conditions must be 'checked' at all points in time. We discuss this more precisely in appendix J.

Somewhat sidestepping the temporal representation issue, most causal research on continuous-time dynamical systems has employed foundational ideas in cyclic graphical models (Iwasaki &amp; Simon, 1994; Spirtes, 2013; Lacerda et al., 2008; Hyttinen et al., 2012) to develop causal abstractions of a system's equilibrium behavior (Dash, 2003; Mooij et al., 2013; Hansen &amp; Sokol, 2014; Blom et al., 2019; Forré &amp; Mooij, 2020; Bongers, 2022; Blom &amp; Mooij, 2023). Equilibrium-focused frameworks, however, can fail to expose complex causal relationships in transient dynamics (Peters et al., 2020).

Extensions such as the 'time-splitting' operation (Boeken &amp; Mooij, 2024), or the application of GSEMs to hybrid automata by Peters &amp; Halpern (2025), enhance the expressiveness of graphical approaches by supporting static-time discontinuities. In contrast, our work targets dynamically triggered interventions, which cannot be straightforwardly analyzed using methods like time-splitting. Indeed, the order - and, therefore, the induced time-split graph structure - of dynamically triggered interventions depends on state evolution, and therefore on exogenous noise. Our approach avoids these issues by directly defining counterfactual interventions on hybrid system constraints. While the non-graphical framing means that standard graphical identifiability criteria are not immediately available, building our semantics on established hybrid systems theory opens pathways to leveraging longstanding methods and conditions for system identification of dynamical systems (Walter &amp; Pronzato, 1997; Ljung, 2012; Raue et al., 2009; Stuart, 2010), such as the 'persistence of excitation', which has been studied directly in the context of hybrid systems (Johnson, 2023; Saoud et al., 2024).

Our approach follows the spirit of recent developments in constraint-based causal modeling. For example, Beckers et al. (2023) extend SCMs in order to handle logical constraints (such as unit conversions), while Blom et al. (2019) interpret equilibrium equations of dynamical systems, along with their corresponding algebraic invariants, as a collection of constraints. In both cases, a model is characterized by a collection of constraints, and interventions are defined as transformations of those constraints (e.g., by changing, disabling, or enabling them). At a high level, we take a similar approach. Hybrid systems, however, are characterized by a unique class of constraints requiring special considerations around Zeno behavior, set-valued theory, non-uniqueness even in 'well-posed' cases, set-valued stable points, etc. In short, analyzing the post-intervention properties of hybrid systems is made easier via direct use of existing hybrid systems frameworks, rather than existing causal frameworks. Naturally, each school of thought is best suited to different tasks, and we look forward to future work that deftly exercises the comparative advantages of each.

In Control Theory. Control theory and causality share overlapping goals, yet historically operate separately. This paper integrates causal reasoning directly into established, hybrid dynamical systems frameworks (Goebel et al., 2012; Sanfelice, 2021). In particular, our formalization of dynamically triggered intervention as constraint transformation mirrors controller-plant compositions from hybrid control theory, which are also shown to preserve established conditions for system well-posedness (Sanfelice, 2021). Hybrid system theory presents challenges, however, due to potential non-uniqueness of solutions under general conditions (Goebel et al., 2012), complicating counterfactual reasoning. To address this practically, we follow common simulation practices (e.g., preferring flowing solutions when multiple are possible) and explicitly formalize these assumptions (Sanfelice et al., 2023a). Our contributions thus link causal semantics to established hybrid systems theory and practice, enabling rigorous and computationally feasible causal analysis.

Figure 1: Three parallel worlds constructed by starting with a dose-decay model (fig. 1a) and then transforming that model to reflect dosage at a fixed, static time (fig. 1b), and dosage when the concentration hits a threshold (fig. 1c and example 1). This paper develops the first explicitly counterfactual semantics for the dynamically triggered, state-dependent case (fig. 1c). Three sample trajectories are shown for each world, with initial condition and dose-decay rate held fixed across worlds for each sample trajectory. Notice that the state-dependent interventions occur at different times for different trajectories induced by different initial conditions and/or parameters.

<!-- image -->

## 3 Parameterized Hybrid Systems

As a first approximation, the present work focuses on continuous, ordinary differential equations models with random initial conditions and parameters. Many intuitive interventions, however, can be conveniently defined as instantaneous (discontinuous) changes to the dynamically evolving state. Thus, we focus on hybrid systems that afford both continuous 'flow' and event-based 'jumps' in state. Jumps can arise as a product of interventions and/or discontinuous dynamics in the unintervened system. With state space S Ď R n and following the framework laid out by Goebel et al. (2012), many hybrid systems can be characterized as comprising four elements: a flow set C Ă S ; a differential inclusion F : S Ñ R n ; a jump set D Ă S ; and a set-valued jump map G : S Ñ S . In general, the system evolves according to its differential inclusion F when its state is in the flow set C and according to the jump map G when in the jump set D . Readers who are unfamiliar with inclusions and set-valued maps should refer to appendix A.1. Hybrid systems often alternate between continuous flow and discontinuous jumps, though consecutive jumps remain well-defined. Many hybrid systems, then, can be characterized with the tuple p C, F, D, G q . We ground this out in the following example.

Example 1 (Dosage Model) . Consider modeling the exponential decay of drug concentration x at rate β , where medical providers intervene to administer additional dosage when x reaches a threshold γ . To model these dynamics, we can seek state evolutions obeying

<!-- formula-not-decoded -->

where 9 x denotes the time derivative of the state, and x ` the state immediately following a jump. The solution map of a hybrid system typically takes as 'input' an initial condition ξ P S , but can also be parameterized to additionally incorporate a vector θ -in example 1, θ ' r β, γ s .

Definition 1 (Parameterized Hybrid System) . Let S Ď R n , Θ Ď R m . A parametrized hybrid system P is a tuple P ' p H , S , Θ q where for each θ P Θ , H p θ q ' p C p θ q , F θ , D p θ q , G θ q is a standard hybrid system (Goebel et al., 2012, Def. 2.2), i.e.

- C : Θ Ñ S is a set-valued mapping returning the flow set,
- F θ p x q ' F p x , θ q @ x P S and @ θ P Θ , where F : S ˆ Θ Ñ R n is a differential inclusion, with C p θ q Ă dom F θ Ď S for all θ P Θ ,
- D : Θ Ñ S is a set-valued mapping returning the jump set, and
- G θ p x q ' G p x , θ q @ x P S and @ θ P Θ , where G : S ˆ Θ Ñ S is an ordered (i.e. returning an ordered collection of sets to keep track of interventions, cf. definition 6) set-valued jump map, with D p θ q Ă dom G θ Ď S for all θ P Θ .

Without explicit parametrization, we write H ' p C, F, D, G q , and also often expand H in P , writing equivalently P ' p H , S , Θ q ' p C, F, D, G, S , Θ q .

Canonically, solutions to hybrid systems are functions of both continuous time t P R ě 0 and discrete event indices j P N . Following (Goebel et al., 2012, Sects. 2.2-2.3), we define, for each possible parameterization θ P Θ and initial condition ξ P S , a 'solution' to H p θ q to be a 'hybrid arc', which is formally a set-valued map ϕ p¨ ; ξ , θ q : R ě 0 ˆ N Ñ R n . We review Goebel et al.'s (2012) rigorous characterization of hybrid arcs as solutions to hybrid systems in appendix A.3. For ease of exposition in the main body of this paper, however, we use the concept of a time-parameterized solution map φ , which we describe informally, below, in definition 2. An expanded, formal treatment of time-parameterized solution maps can be found in appendix A.4.

Definition 2 ((Informal) Time-Parameterized Solution Map) . Let φ p¨ ; ξ , θ q : r 0 , t ` q Ñ R n be called the time-parameterized solution map of P ' p H , S , Θ q , where t ` ' min ξ , θ sup t dom ϕ p¨ ; ξ , θ q and where the hybrid arc ϕ p¨ ; ξ , θ q uniquely satisfies H p θ q from initial state ξ , @ ξ , θ P S ˆ Θ .

The reader will note that r 0 , t ` q Ă R . In this paper, we focus strictly on finite time horizons, leaving the analysis of hybrid equilibria to future work - indeed, only the simplest hybrid systems equilibrate to a point, so equilibrium states are most productively defined as belonging to a set. Analyzing the causally relevant behavior of such sets requires machinery beyond our current scope, but our direct connection to established hybrid systems theory, in conjunction with the rich history of causal research on equilibrium models, provides a firm foundation to explore this in the future. Additionally, because hybrid arcs can dynamically evolve in event indices, Zeno and non-flowing solutions are possible, which can make t ` ' 0 (if it only jumps) or arbitrarily small (if allowable initial conditions are close to Zeno accumulation points). We do not provide universal criteria in this paper under which t ` is arbitrarily large.

While we take the hybrid system P to accurately describe causally relevant mechanisms in the world, we impose assumptions on P indirectly . In particular, we assume that some auxiliary 'upstream' system P Ò can be 'lowered' to produce P ' lower p P Ò q , and that the upstream P Ò satisfies standard hybrid well-posedness conditions from the literature (the so-called hybrid basic conditions , detailed in assumption 4 of the appendix, and folded into assumption 1 below). While these conditions support our theoretical results and facilitate future extensions (e.g., to stability analyses), they inherently admit solution non-uniqueness, particularly at state-space boundaries where solutions could either jump or flow. Non-uniqueness, however, complicates both measurability arguments and downstream causal analysis. In this work, then, we formalize a practical approach that is standard in simulating hybrid systems by specifying that the solutions should be 'flow preferring' - if a solution could both flow and jump, we choose the solution that flows (Sanfelice et al., 2023a; Sanfelice &amp; Teel, 2010). 4 Note also that a flow-preferring specification is consistent with computational implementations that trigger jumps when the jump-set boundary is crossed . 5

A key component of the hybrid basic conditions is the outer semi-continuity of the jump set G in the upstream system. Maintaining this property through intervention requires some bookkeeping on the boundaries between interventional jump sets, but must be handled such that 'lowering' favors more recently applied model transformations. We achieve this bookkeeping through the use of an ordered set-valued map G ' x ÞÑ Ť K k ' 1 G k p x q , where last p G q ' G K . We fully formalize the ordered set-valued map in the appendix (c.f. definition 6).

Definition 3. Let P ' p C, F, D, G, S , Θ q ' p H , S , Θ q . We set preferflow p D,C,F q ' θ ÞÑ D p θ qz t ξ P S : there is a flowing solution to H p θ q from ξ u ;

<!-- formula-not-decoded -->

The existence of a flowing solution from ξ is meant in the sense established in the hybrid systems literature. See appendix A.5 (definitions 13 and 14) for details. We can now state our collected assumptions on a hybrid system, and prove the sufficiency of those assumptions for the existence, uniqueness, and measurability of the system's solution. See appendix A.6 (assumptions 3 to 5) for details, and appendix F for proof of lemma 1.

Assumption 1. The parameterized hybrid system can be written as P ' lower p P Ò q , where P Ò ' p C, F, D, G, S , Θ q ' p H , S , Θ q , and the following hold for all ξ P S , θ P Θ :

4 Other approaches include preferring solutions that jump, or by resolving ambiguities randomly (Teel &amp; Hespanha, 2015).

5 We should say, of a thick jump set, similar to what we have described in definition 17. Thickening jump sets is also common in practical computational environments (Sanfelice et al., 2023b).

1. there exists a unique, nontrivial solution to the differential inclusion F (i.e. the continuous part of H p θ q ) that is Borel measurable in ξ , θ at any fixed t P r 0 , 8q ; 6
2. C is outer semi-continuous (osc) and C p θ q closed; F is osc, locally bounded, and F p x , θ q is convex @ x P S ;
3. D p θ q is closed and G p D q is Borel; G is osc, locally bounded; last p G q is single-valued and Borel measurable in ξ , θ at any fixed t P r 0 , 8q .

Lemma 1. Let P satisfy assumption 1. Then P has a unique time-parameterized solution φ p¨ ; ξ , θ q : r 0 , t ` q Ñ R n that is Borel-measurable in initial conditions ξ and parameters θ at any fixed t P r 0 , t ` q .

## 4 Instantaneous Interventions as Constraint Transformations

We now formally define a general class of instantaneous interventions. We show that under certain natural assumptions, the class of systems meeting assumption 1 is 'closed' under intervention - i.e., intervened systems will meet assumption 1 if the original system does.

An instantaneous intervention can be implemented via modifications to the jump map and the jump set functions, respectively G and D in definition 1. To support parameterized interventions and/or stateful jumps, one can simply augment the state space S and the parameter space Θ , essentially preserving all properties of interest (appendix B). 7

Definition 4 (Instantaneous Intervention) . Consider set-valued mappings ˜ D : Θ Ñ S and ˜ G : S ˆ Θ Ñ S and parameterized hybrid system P ' p C, F, D, G, S , Θ q . Now, let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In words, ˜ D defines when (or where in the state space) the intervention will occur, while ˜ G defines the state transition induced by the intervention. We make two important set-subtractions in this definition to preserve some useful and simplifying properties. First, we define G 1 in eq. (3) as preferring the new (i.e., the interventional) jump map ˜ G wherever the original and new jump sets overlap. Second, the new flow set in eq. (1) has the interior of the new jump set removed. This preserves non-overlap between flow and jump sets, except possibly on the boundary, which we discuss below.

D 1 p θ q is defined (eq. (2)) as the union over jump sets, except that a flow-preferring subtraction (definition 3) is made first on ˜ D p θ q . Because C 1 p θ q has the interior of ˜ D p θ q already removed, this subtraction operates only on the boundary of ˜ D p θ q . In other words, D 1 p θ q will always contain the interior of ˜ D p θ q , and will have the parts of its boundary removed where F θ flows tangentially to or away from the interventional jump set.

## 4.1 Intervention Preserves Existence, Uniqueness, and Measurability

Interventional transformations should preserve key model properties. For causal models with explicit forward simulations, this is largely trivial. Hybrid systems, however, only implicitly characterize forward simulations (i.e., solutions) by specifying a set of constraints. Transformations to these constraints can easily fail to maintain key properties in the intervened world, such as whether a

6 This assumption is insufficient, on its own, to guarantee a unique, measurable solution for the full hybrid system. Indeed, this insufficiency constitutes a key challenge addressed by hybrid systems researchers.

7 This covers interventions with random parameters, in addition to those that require some 'memory' of past system events. For example, if a jump event should only occur k P N times, that event could increment a counter i , and include only i ă k in its interventional jump set.

Figure 2: Depiction of our 'lowering' proof strategy for theorem 1. Proving theorem 1 requires a simple inductive generalization from lemma 4, which asserts that a single interventional transformation preserves key system properties, and is what we visualize here. Solid arrows indicate constraint transformations, while dotted arrows indicate that properties of one system imply properties of another. Assume that the parameterized hybrid system P accurately describes a domain of interest, and that it can be constructed by applying the lower transformation (definition 3) to a system P Ò that fulfills assumption 1. Applying lower to such a system preserves existence and induces uniqueness and measurability (lemma 1). To simulate the effects of an intervention, we transform P into the model P 1 that describes the intervened world. P 1 can also be constructed, however, by applying a slightly modified intervention ( instint Ò , definition 20) to P Ò and then 'lowering'. Intervention on P Ò maintains key properties in P 1 Ò , which can, as before, be lowered to a system P 1 that must have a unique, measurable solution (lemma 1).

<!-- image -->

solution is unique and measurable, or exists at all. The key theoretical contribution of this work, then, is to identify assumptions sufficient to ensure our interventional semantics preserves these properties through model transformation. Formal proof of theorem 1 is provided in appendix D, but we also include fig. 2 as a visual aid and proof sketch.

Assumption 2 (Assumptions on Interventional Specifications) . Consider mappings ˜ D : Θ Ñ S and ˜ G : S ˆ Θ Ñ S and parameterized hybrid system P ' p C, F, D, G, S , Θ q . For all θ P Θ , assume

- (I1) ˜ D p θ q is closed and well-behaved relative to P (assumption 6). 8 Additionally, the interior graph G p int ˜ D q is open 9 and the graph G p ˜ D q is Borel;
- (I2) ˜ G θ : S Ñ S is outer semi-continuous and locally bounded relative to ˜ D p θ q , and ˜ D p θ q Ă dom ˜ G θ . Additionally, ˜ G is single-valued and Borel-measurable.

Theorem 1 (Compositions of Instantaneous Interventions Preserve Key Properties) . Consider parameterized hybrid system P that meets assumption 1, and any finite sequence of K setvalued mappings p ˜ D k q and p ˜ G k q , where each ˜ D k and ˜ G k fulfill assumption 2 relative to P . Let instint k ' instint p¨ , ˜ D k , ˜ G k q (definition 4) and

<!-- formula-not-decoded -->

P 1 then meets assumption 1, and by lemma 1 P 1 has a unique time-parameterized solution φ p¨ ; ξ , θ q : r 0 , t ` q Ñ R n , Borel-measurable in initial conditions ξ and parameters θ at any fixed t P r 0 , t ` q .

## 5 Causal Estimands as Functionals of Twin Distributions

In this section, we exercise our framework to define three basic causal estimands. Importantly, many of the more complex causal analyses build on these basic inference capabilities. Most targets of causal inference take the form of (conditional) expectations, and so we must now use our measurability results to define those expectations with respect to random solution maps. 10 First, we will generalize

8 An assumption that the interventional jump set is well-behaved reduces, essentially, to asserting that a flowing solution cannot oscillate across the boundary of ˜ D infinitely often. This is satisfied by many systems of interest under reasonable regularity assumptions - for instance, if the flow map is analytic and B ˜ D is Lipschitz.

9 This ensures its interior does not suddenly appear/disappear as θ varies.

10 In this paper, we do not address the random dynamics that characterize stochastic differential constraints (Øksendal, 2003; Cassandras &amp; Lygeros, 2010; Hansen &amp; Sokol, 2014; Boeken &amp; Mooij, 2024), or independent, per-jump randomness (Teel, 2013; Teel et al., 2014; Teel &amp; Hespanha, 2015).

the parameterized hybrid system to include random initial conditions and parameters. Then, we will establish some notation and define the expected treatment effect, data-conditional treatment effect, and the basic counterfactual query using our machinery.

Definition 5 (Hybrid System with Random Inputs) . A parameterized hybrid system with random inputs is characterized by the tuples

<!-- formula-not-decoded -->

We take the probability space p Ω , F , P q as implied by R , where ξ : Ω Ñ S and θ : Ω Ñ Θ are measurable with respect to F and the Borel σ -algebras on S Ď R n and Θ Ď R m .

When clear from context, for some ω P Ω , we often write ξ and θ in place of ξ p ω q and θ p ω q , respectively. We distinguish random variables ξ and θ from possible values ξ P S and θ P Θ by the upright font. From here, we can directly consider evaluations of the solution as a measurable random variable. A direct consequence of lemma 1, which states that lower induces measurability when applied to a system that fulfills assumption 1, is the following.

Corollary 1 (Random Time-Parameterized Solution is Measurable) . Consider parameterized hybrid system with random inputs R ' p P , ξ , θ q , where P satisfies assumption 1. Then, by lemma 1, P has a unique, time-parameterized solution map φ , and the composition ω ÞÑ φ p t ; ξ p ω q , θ p ω qq defines an F -measurable random variable at any fixed t P r 0 , t ` q .

Having established conditions under which intervention preserves measurability, we can begin constructing estimands from the parallel worlds created through intervention. In estimands, we use symbolic subscripts to delineate parallel worlds. Consider an original system R 0 ' p P 0 , ξ , θ q . We might then apply an intervention characterized by ˜ D s and ˜ G s to yield P s ' instint p P 0 , ˜ D s , ˜ G s q . By convention, we use R s ' p P s , ξ , θ q in reference to the full specification for the intervened world, and t ÞÑ φ s p t ; ξ , θ q for its random, time-parameterized solution (corollary 1). We often write φ t s in place of φ s p t ; ξ , θ q for brevity. Lastly, supposing we wish to focus on a particular element of the state vector at time t , we sometimes define a random function that appropriately indexes into the solution vector. For example, we might have that h s p t ; ξ , θ q ' φ (i) s p t ; ξ , θ q always, where h represents the solution map for the p i q 'th state element. We similarly sometimes use h t s ' h s p t ; ξ , θ q . We can exercise this notation with the following examples.

Example 2 (Expected Treatment Effect) . Consider R 0 ' p P 0 , ξ , θ q and interventional jump set ˜ D and map ˜ G . Assume these components fulfill assumptions 1 and 2. Let P 1 ' instint p P 0 , ˜ D, ˜ G q and φ 0 and φ 1 be the time-parameterized solution maps of the original and intervened worlds. Let y 0 and y 1 be the solution maps for the p i q 'th element of the state vector. The expected treatment effect at some time τ P r 0 , min ' t ` 0 , t ` 1 ‰ q ' r 0 , t ` q can be written equivalently as

<!-- formula-not-decoded -->

Example 3 (Data-Conditional Treatment Effect) . Building immediately off example 2, we can specify a data-conditional treatment effect that takes factual observations into account. 11 Let w 0 be the solution map for some element of the state vector. For some finite set of observation times t t k u K k ' 1 Ă r 0 , t ` 0 q , the data-conditional treatment effect can then be written as 12

<!-- formula-not-decoded -->

Example 4 (Counterfactual Outcome) . Also building off example 2, consider factual outcome event that y 0 p τ ; ξ , θ q ' ¯ y τ 0 P R . The counterfactual outcome, then, can be derived by conditioning on that factual event.

<!-- formula-not-decoded -->

11 While identification results for specific causal estimands are beyond the scope of this paper, system identification has already been studied for hybrid systems under the condition of 'persistence of excitation' (Johnson, 2023; Saoud et al., 2024). Under such conditions, a posterior density p p ξ , θ | v 0 ' D q , for example, where D is a realization of v 0 , is sufficiently well-behaved to estimate targets defined in this paper.

12 Without loss of generality, we write that the data are subject to Gaussian observation noise. Many practical settings call for observation noise, but we also note that the deterministic relationship between inputs ( ξ , θ ) and state trajectories means that inference behaves poorly without observation noise.

Table 1: Identities for the probabilities of causation in the fishery management example. Under TAC quota q i , the biomass of the fished species at time τ is given by b τ q i . The outcome Y is achieved when that biomass meets or exceeds γ . We rely on the standard exogeneity conditions Y x K K X and Y x 1 K K X , 15 and the fact that, conditioned on X ' 1 ( X 2 ), Y reduces to the outcome only in the world with allowable catch set to q 1 ( q 2 ).

| query                   | outcome                                                                             | probability                                                                                                                                                                                                                 |
|-------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| nec. suf. nec. and suf. | Y ' I r b τ q 2 ď γ s Y ' I r b τ q 2 ą γ s Y ' I r b τ q 2 ď γ s I r b τ q 1 ě γ s | Pr p Y x 1 ' 0 &#124; X ' 1 ,Y ' 1 q ' Pr `' b τ q 2 ď γ ‰ &#124; b τ q 1 ą γ ˘ Pr p Y x 1 ' 1 &#124; X ' 0 ,Y ' 0 q ' Pr `' b τ q 2 ą γ ‰ &#124; b τ q 1 ď γ ˘ Pr p Y x ' 1 ,Y x 1 ' 0 q ' Pr ` b τ q 1 ą γ, b τ q 2 ď γ ˘ |

Many of the more complex causal inference tasks - such as mediation analysis, the estimation of population-level conditional average treatment effects, or even actual cause assessments - are constructed from the counterfactual building blocks we propose here. Indeed, once a counterfactual semantics is established, and a twin-world or potential-outcomes syntax (e.g., differentiating y 0 from y 1 ) is enumerated, many estimands are straightforward and familiar to develop. In the next section, we explore just such a class of estimand: the probabilities of causation.

## 6 Case Study: Necessary and Sufficient Causation

To illustrate a more sophisticated application of our interventional semantics, we map the standard definitions for the probabilities of necessary and sufficient causation (originally formalized by Pearl (1999)) onto dynamically triggered, discontinuous interventions in hybrid systems. In particular, we work in the fishery management domain where regulators employ Total Allowable Catch (TAC) policies to dynamically end the commercial fishing season after caught biomass reaches certain quotas. If interested, the reader may wish to review appendix G.1, in which we provide motivating historical context for this domain. Additionally, we review Pearl's original formulation of the probabilities of causation (PoC) in appendix G.2. Throughout appendix G, we provide full simulation analyses of the case study. Code is available here, 13 and relies on the dynamical systems package from the causal probabilistic programming language ChiRho (Basis-Research, 2025). 14

We focus on a hypothetical fishery involving three trophic levels - apex predators, intermediate predators (the fished species), and forage fish - with dynamics captured by the differential equations presented by Zhou &amp; Smith (2017). Throughout a single season, fishing pressure is modeled at a constant rate applied to the intermediate predator, plus some bycatch on the apex trophic level. Regulators intervene by ending the fishing season (setting the catch rate to zero) when the integrated catch reaches a predefined TAC quota. The goal of these policies is to ensure that the biomass of the target fishery species recovers to sustainable level γ by the beginning of the next season.

In this context, stakeholders may debate the necessity and/or sufficiency of certain regulatory policies in maintaining joint ecological and economic goals for the fishery. The probabilities of causation are formal tools supporting the assessment of causal attribution between causes and their (supposed) effects. Pearl (1999) first formalized the PoC for binary treatments and outcomes - here, however, both the TAC quota and the biomass are scalar valued. We therefore follow Kawakami et al.'s (2024) generalization of the PoCs to support contrastive queries between scalar-valued treatments and their thresholded outcomes (see their Def. 3.1). Consider two TAC quotas q 1 and q 2 , and the following natural language queries. In table 1, we provide the formalized estimands written in our notation.

- necessity: in worlds where the end-of-year biomass levels exceed the target level γ (success) under quota q 1 , what is the probability of failure had regulators used quota q 2 instead?
- sufficiency: in worlds where the end-of-year biomass levels remain below the target level γ (failure) under q 1 , what is the probability of success had regulators used q 2 instead?
- necessity and sufficiency: what is the probability that both (1) q 1 results in success and (2) q 2 results in failure?

13 https://basisresearch.github.io/counterfactuals-for-hybrid-systems

14 https://github.com/BasisResearch/chirho

15 If some parameter or initial condition were influenced by a confounder that also influenced X , this would not be the case, and conditioning on X would be required in the identities listed in table 1.

For readers less familiar with the applications of the PoC to decision and policy making, we provide an expanded narrative scaffolding for this example in appendix G.4. In appendix G.5, we provide an additional example designed to highlight how certain natural language ambiguities in causal attribution queries - particularly those involving multi-faceted, real world events and policies can be formally clarified.

The PoC queries above rely on the construction of twin, contrastive worlds - one with TAC quota q 1 , and the other with q 2 . To model these worlds, we start with a system P characterizing year-round fishing pressure (i.e., no regulatory intervention), and then transform its constraints to add a dynamic, season-ending intervention. Notationally, let h i represent the harvest rate, and b i the biomass, at trophic level i , and let z be the total catch (integral of 9 z ' h 2 b 2 ) at the intermediate trophic level. The system state can be conceptualized as r z, h 1:3 , b 1:3 s ' x P S ' R 7 ě .

The regulatory, season-ending intervention can be modeled by dynamically setting harvest rates to zero when the catch exceeds a threshold q i (with i P t 1 , 2 u ). By using our interventional semantics, we can construct parallel worlds with the same random initial conditions and parameters. See appendix G.3 for a generalization of this model to multi-season time scales.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 7 Limitations and Future Work

Most research developing causal inference tools starts by casting a problem in the format of structural causal models (SCMs) (Pearl, 2009). Our work differs in that we construct our counterfactual semantics directly in the parlance of hybrid systems. These two tacks are compatible, however. For example, with our measurability results in hand, the time-parameterized solution map φ can be treated as a structural equation with initial conditions ξ and parameters θ viewed as parent variables in a larger SCM. Our interventional semantics, then, exposes the causal dynamics of the hybrid system for manipulation. When φ is interpreted as a structural equation, our semantics could be viewed as characterizing a family of 'soft-interventions' (Correa &amp; Bareinboim, 2020) on the solution map. Importantly, the adjoint method (Chen et al., 2018) can be used in tandem with auto-differentiation machinery to learn 'event function' (i.e. jump set) parameters, 16 thereby supporting end-to-end differentiation of composite SCM and hybrid system models. Relatedly, equivalent forward-Euler representations may prove useful in actual cause analysis of hybrid systems (Halpern &amp; Peters, 2022).

This leaves a few limitations to review. First, we do not present non-parametric, estimand-specific identification results - indeed, there may exist sufficient conditions for estimand identification that are weaker than those established for full system identification. Second, as discussed following definition 1, we focus only on finite time regimes, leaving analysis of hybrid equilibria to future work. Furthermore, we do not provide conditions under which intervention preserves non-Zeno behavior.

## 8 Conclusion

This paper has strengthened the connection between the modeling capabilities offered by hybrid systems theory and the causal reasoning capabilities developed by the causal inference research community.

We characterize and demonstrate a counterfactual semantics for a class of dynamically triggered, instantaneous interventions that underpin many closed-loop control problems. Bypassing an explicit re-casting of hybrid systems as structural causal models, we use hybrid systems as the primary modeling substrate. This allows clear connections to the extensive body of work on hybrid systems theory, in which we can derive and characterize mild conditions under which solution existence, uniqueness and measurability are preserved in the intervened system.

Finally, we illustrate the flexibility and power of the resulting framework by first formalizing common causal estimands for hybrid systems, and then by developing a case study using the three probabilities of causation in the context of fishery management.

16 See https://github.com/rtqichen/torchdiffeq/blob/master/examples/bouncing\_ball.py for an example.

## Acknowledgments and Disclosure of Funding

AZ, DB, RU, JZ, and SW were supported on DARPA Automating Scientific Knowledge Extraction and Modeling (ASKEM) program Grant HR00112220036. We thank Anirban Chaudhuri, Sabina Altus, Joseph Cottam, and Neeraj Kumar for their insights and contributions throughout the ASKEM program; our colleagues at Basis for helpful comments and discussions; Eli Bingham for guiding our thinking and software design choices around these ideas; Paul Wintz for answering many questions about hybrid systems and pointing us to relevant literature; and David Jensen for helpful comments and discussion. Pacific Northwest National Laboratory (PNNL) is a multiprogram national laboratory operated by Battelle for the DOE under Contract DEAC05-76RLO 1830. The views, opinions and/or findings expressed are those of the author and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government. Distribution Statement 'A' (Approved for Public Release, Distribution Unlimited).

## References

- Anon. Magnuson-Stevens Fishery Conservation and Management Act, 1976. URL https://www. fisheries.noaa.gov/s3//dam-migration/msa-amended-2007.pdf .
- Anon. Sustainable Fisheries Act, 1996.
- Anon. Magnuson-Stevens Act Provisions; Foreign Fishing; Fisheries off West Coast States and in the Western Pacific; Pacific Coast Groundfish Fishery; Annual Specifications and Management Measures, January 2000. URL https://www.federalregister.gov/ documents/2000/01/04/99-33966/magnuson-stevens-act-provisions-foreignfishing-fisheries-off-west-coast-states-and-in-the-western . Volume: 65 Docket Number: 991223347-9347-01.
- Anon. Magnuson-Stevens Fishery Conservation and Management Reauthorization Act, 2007a. URL https://www.fisheries.noaa.gov/s3//dam-migration/msa-amended-2007.pdf .
- Anon. Atlantic Highly Migratory Species (HMS); U.S. Atlantic Swordfish Fishery Management Measures, June 2007b. URL https://www.federalregister.gov/documents/2007/06/07/E710727/atlantic-highly-migratory-species-hms-us-atlantic-swordfishfishery-management-measures . Volume: 72 Docket Number: 061121306-7105-02.
- Anon. Fisheries Off West Coast States; Pacific Coast Groundfish Fishery Management Plan; Amendments 20 and 21; Trawl Rationalization Program, December 2010a. URL https://www.federalregister.gov/documents/2010/12/15/2010-30527/fisheriesoff-west-coast-states-pacific-coast-groundfish-fishery-management-planamendments-20-and . Volume: 75 Docket Number: 100212086-0532-05.
- Anon. Report of the 2009 Atlantic Swordfish Stock Assessment Session. Collect. Vol. Sci. Pap. ICCAT , 65(1):1-123, 2010b. URL https://www.iccat.int/Documents/CVSP/CV065\_2010/ n\_1/CV065010001.pdf .
- Anon. Modernizing Recreational Fisheries Management Act, 2018.
- Charles K. Assaad, Emilie Devijver, and Eric Gaussier. Survey and Evaluation of Causal Discovery Methods for Time Series. Journal of Artificial Intelligence Research , 73:767-819, February 2022. ISSN 1076-9757. doi: 10.1613/jair.1.13428. URL https://www.jair.org/index.php/jair/ article/view/13428 .
- Chen Avin, Ilya Shpitser, and Judea Pearl. Identifiability of path-specific effects. In Proceedings of the 19th international joint conference on artificial intelligence , IJCAI'05, pp. 357-363, Edinburgh, Scotland, 2005. Morgan Kaufmann Publishers Inc.
- Alexander Balke and Judea Pearl. Probabilistic evaluation of counterfactual queries. In Proceedings of the twelfth AAAI national conference on artificial intelligence , AAAI'94, pp. 230-237, Seattle, Washington, 1994. AAAI Press.
- Basis-Research. Chirho, 2025. URL https://github.com/BasisResearch/chirho .

- Sander Beckers. Causal Explanations and XAI. In Bernhard Schölkopf, Caroline Uhler, and Kun Zhang (eds.), Proceedings of the First Conference on Causal Learning and Reasoning , volume 177 of Proceedings of Machine Learning Research , pp. 90-109. PMLR, April 2022. URL https://proceedings.mlr.press/v177/beckers22a.html .
- Sander Beckers and Joseph Y. Halpern. Abstracting Causal Models, July 2019. URL http://arxiv. org/abs/1812.03789 . arXiv:1812.03789 [cs].
- Sander Beckers, Frederick Eberhardt, and Joseph Y. Halpern. Approximate Causal Abstraction, June 2019. URL http://arxiv.org/abs/1906.11583 . arXiv:1906.11583 [cs].
- Sander Beckers, Joseph Halpern, and Christopher Hitchcock. Causal Models with Constraints. In Mihaela van der Schaar, Cheng Zhang, and Dominik Janzing (eds.), Proceedings of the second conference on causal learning and reasoning , volume 213 of Proceedings of machine learning research , pp. 866-879. PMLR, April 2023. URL https://proceedings.mlr.press/v213/ beckers23a.html .
- Eli Bingham, James Koppel, Alexander Lew, Robert Ness, Zenna Tavares, Sam Witty, and Jeremy Zucker. Causal Probabilistic Programming Without Tears. In Proceedings of the third conference on probabilistic programming , 2021.
- Tineke Blom and Joris M. Mooij. Causality and Independence in Perfectly Adapted Dynamical Systems, February 2023. URL http://arxiv.org/abs/2101.11885 . arXiv:2101.11885 [cs].
- Tineke Blom, Stephan Bongers, and Joris M. Mooij. Beyond Structural Causal Models: Causal Constraints Models, August 2019. URL http://arxiv.org/abs/1805.06539 . arXiv:1805.06539 [cs].
- Tineke Blom, Mirthe M. van Diepen, and Joris M. Mooij. Conditional independences and causal relations implied by sets of equations, January 2021. URL http://arxiv.org/abs/2007. 07183 . arXiv:2007.07183 [cs].
- Philip Boeken and Joris M. Mooij. Dynamic Structural Causal Models, July 2024. URL http: //arxiv.org/abs/2406.01161 . arXiv:2406.01161 [math].
- Stephan Bongers. Causal Modeling &amp; Dynamical Systems: A New Perspective On Feedback . PhD thesis, Universiteit van Amsterdam, 2022. URL https://hdl.handle.net/11245.1/ 652541c6-8959-498c-8958-fd28f198bfdf .
- Dan G. Cacuci. Sensitivity &amp; Uncertainty Analysis, Volume 1 . Chapman and Hall/CRC, 0 edition, May 2003. ISBN 978-1-135-44298-9. doi: 10.1201/9780203498798. URL https://www. taylorfrancis.com/books/9781135442989 .
- Christos G. Cassandras and John Lygeros. Stochastic Hybrid Systems . Automation and Control Engineering. Taylor and Francis, Hoboken, 2010. ISBN 978-0-8493-9083-8.
- Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural Ordinary Differential Equations. In Advances in neural information processing systems , volume 31, 2018. URL https://proceedings.neurips.cc/paper\_files/paper/2018/ file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf .
- H. Chockler and J. Y. Halpern. Responsibility and Blame: A Structural-Model Approach. Journal of Artificial Intelligence Research , 22:93-115, October 2004. ISSN 1076-9757. doi: 10.1613/jair.1391. URL https://jair.org/index.php/jair/article/view/10386 .
- J. Correa and E. Bareinboim. General Transportability of Soft Interventions: Completeness Results. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in neural information processing systems , volume 33, pp. 10902-10912, Vancouver, Canada, June 2020. Curran Associates, Inc. / Causal Artificial Intelligence Lab, Columbia University. Number: R-68.
- Lori A. Cramer, Courtney Flathers, Deanna Caracciolo, Suzanne M. Russell, and Flaxen Conway. Graying of the Fleet: Perceived Impacts on Coastal Resilience and Local Policy. Marine Policy , 96:27-35, October 2018. ISSN 0308597X. doi: 10.1016/j.marpol.2018.07.012. URL https: //linkinghub.elsevier.com/retrieve/pii/S0308597X17308631 .

- Denver Dash. Caveats for Causal Reasoning with Equilibrium Models. May 2003. URL https: //d-scholarship.pitt.edu/7811/ .
- Patrick Forré and Joris M. Mooij. Causal Calculus in the Presence of Cycles, Latent Confounders and Selection Bias. In Ryan P. Adams and Vibhav Gogate (eds.), Proceedings of the 35th uncertainty in artificial intelligence conference , volume 115 of Proceedings of machine learning research , pp. 71-80. PMLR, July 2020. URL https://proceedings.mlr.press/v115/forre20a.html .
- Rafal Goebel, Ricardo G. Sanfelice, and Andrew R. Teel. Hybrid dynamical systems: modeling, stability, and robustness . Princeton university press, Princeton (N.J.), 2012. ISBN 978-0-69115389-6.
- Joseph Y. Halpern and Judea Pearl. Causes and Explanations: a Structural-Model Approach. Part I: Causes. British Journal for the Philosophy of Science , 56(4):843-887, 2005a. doi: 10.1093/bjps/ axi147. Publisher: Oxford University Press.
- Joseph Y. Halpern and Judea Pearl. Causes and Explanations: A Structural-Model Approach. Part II: Explanations. The British Journal for the Philosophy of Science , 56(4):889-911, December 2005b. ISSN 0007-0882, 1464-3537. doi: 10.1093/bjps/axi148. URL https://www.journals. uchicago.edu/doi/10.1093/bjps/axi148 .
- Joseph Y. Halpern and Spencer Peters. Reasoning about Causal Models with Infinitely Many Variables. Proceedings of the AAAI Conference on Artificial Intelligence , 36(5):5668-5675, June 2022. ISSN 2374-3468, 2159-5399. doi: 10.1609/aaai.v36i5.20508. URL https://ojs.aaai.org/index. php/AAAI/article/view/20508 .
- Niels Hansen and Alexander Sokol. Causal Interpretation of Stochastic Differential Equations. Electronic Journal of Probability , 19(none), January 2014. ISSN 1083-6489. doi: 10.1214/EJP.v19-2891. URL https://projecteuclid.org/journals/electronicjournal-of-probability/volume-19/issue-none/Causal-interpretation-ofstochastic-differential-equations/10.1214/EJP.v19-2891.full .
- Miguel Hernán and James M. Robins. Causal Inference: What If . Taylor and Francis, Boca Raton, first edition edition, 2023. ISBN 978-1-4200-7616-5 978-0-367-71133-7.
- Antti Hyttinen, Frederick Eberhardt, and Patrik O. Hoyer. Learning Linear Cyclic Causal Models with Latent Variables. Journal of Machine Learning Research , 13(109):3387-3439, 2012. URL http://jmlr.org/papers/v13/hyttinen12a.html .
- Guido W. Imbens and Donald B. Rubin. Causal Inference for Statistics, Social, and Biomedical Sciences . Cambridge books. Cambridge University Press, September 2015. URL https://ideas. repec.org/b/cup/cbooks/9780521885881.html . Number: 9780521885881.
- Yumi Iwasaki and Herbert A. Simon. Causality and Model Abstraction. Artificial Intelligence , 67 (1):143-194, May 1994. ISSN 00043702. doi: 10.1016/0004-3702(94)90014-0. URL https: //linkinghub.elsevier.com/retrieve/pii/0004370294900140 .
- Ryan Johnson. Parameter Estimation for Hybrid Dynamical Systems . PhD thesis, University of California Santa Cruz, 2023.
- Haley Kappus-Kron, Dana Ahmad Chatila, Ainsley Mabel MacLachlan, Nicole Pulido, Nan Yang, and David A. Larsen. Precision public health in schools enabled by wastewater surveillance: A case study of COVID-19 in an Upstate New York middle-high school campus during the 2021-2022 academic year. PLOS Global Public Health , 4(1):e0001803, January 2024. ISSN 2767-3375. doi: 10.1371/journal.pgph.0001803. URL https://dx.plos.org/10.1371/journal.pgph. 0001803 .
- Yuta Kawakami, Manabu Kuroki, and Jin Tian. Probabilities of Causation for Continuous and Vector Variables. In Negar Kiyavash and Joris M. Mooij (eds.), Proceedings of the fortieth conference on uncertainty in artificial intelligence , volume 244 of Proceedings of machine learning research , pp. 1901-1921. PMLR, July 2024. URL https://proceedings.mlr.press/v244/kawakami24a. html .

- Donald E. Kirk. Optimal Control Theory: an Introduction . Dover Publications, Mineola, N.Y, 2004. ISBN 978-0-486-43484-1.
- Gustavo Lacerda, Peter Spirtes, Joseph Ramsey, and Patrik O. Hoyer. Discovering Cyclic Causal Models by Independent Components Analysis. In Proceedings of the twenty-fourth conference on uncertainty in artificial intelligence , UAI'08, pp. 366-374, Helsinki, Finland, 2008. AUAI Press. ISBN 0-9749039-4-9. Number of pages: 9 tex.address: Arlington, Virginia, USA.
- Donna J. Lee, Sherry Larkin, and Charles M. Adams. A Bioeconomic Analysis of Management Alternatives for the U.S. North Atlantic Swordfish Fishery. Marine Resource Economics , 15:77 96, 2000. URL https://api.semanticscholar.org/CorpusID:150518179 .
- Ang Li and Judea Pearl. Probabilities of Causation with Nonbinary Treatment and Effect. Proceedings of the AAAI Conference on Artificial Intelligence , 38(18):20465-20472, March 2024. ISSN 23743468, 2159-5399. doi: 10.1609/aaai.v38i18.30030. URL https://ojs.aaai.org/index.php/ AAAI/article/view/30030 .
- Lennart Ljung. System Identification: Theory for the User . Prentice-Hall information and system sciences series. Prentice Hall, Upper Saddle River, NJ, 2. ed., 14. printing edition, 2012. ISBN 978-0-13-656695-3.
- Joris M. Mooij, Dominik Janzing, and Bernhard Schölkopf. From Ordinary Differential Equations to Structural Causal Models: the deterministic case, April 2013. URL http://arxiv.org/abs/ 1304.7920 . arXiv:1304.7920 [stat].
- Kevin P. Murphy. Dynamic bayesian networks: Representation, inference and learning . phd, EECS Department, University of California, Berkeley, 2002.
- John D. Neilson, Freddy Arocha, Shannon L. Cass-Calay, Jaime Mejuto, Mauricio Ortiz, Gerald P. Scott, Craig Smith, Paulo Travassos, George Tserpes, and Irene V. Andrushchenko. The recovery of atlantic swordfish: The comparative roles of the regional fisheries management organization and species biology. Reviews in Fisheries Science , 21:59 - 97, 2013. URL https://api. semanticscholar.org/CorpusID:55132285 .
- Mauricio Ortiz, Shannon L Cass-Calay, and Gerald P Scott. A Potential Framework for Evaluating the Efficacy of Biomass Limit Reference Point in the Presence of Natural Variability and Parameter Uncertainty: An Application to Northern Albacore Tuna (Thunnus Alalunga). Collect. Vol. Sci. Pap. ICCAT , 65(4):1254-1267, 2010.
- Judea Pearl. Causal Diagrams for Empirical Research. Biometrika , 82(4):669-688, 1995. ISSN 0006-3444, 1464-3510. doi: 10.1093/biomet/82.4.669. URL https://academic.oup.com/ biomet/article-lookup/doi/10.1093/biomet/82.4.669 .
- Judea Pearl. Probabilities Of Causation: Three Counterfactual Interpretations And Their Identification. Synthese , 121(1/2):93-149, 1999. ISSN 00397857. doi: 10.1023/A:1005233831499. URL http://link.springer.com/10.1023/A:1005233831499 .
- Judea Pearl. Direct and Indirect Effects. In Proceedings of the seventeenth conference on uncertainty in artificial intelligence , UAI'01, pp. 411-420, San Francisco, CA, USA, 2001. Morgan Kaufmann Publishers Inc. ISBN 1-55860-800-1. Number of pages: 10 Place: Seattle, Washington.
- Judea Pearl. Causality: Models, Reasoning and Inference . Cambridge University Press, USA, 2 edition, 2009. ISBN 0-521-89560-X.
- Jonas Peters, Stefan Bauer, and Niklas Pfister. Causal Models for Dynamical Systems, January 2020. URL http://arxiv.org/abs/2001.06208 . arXiv:2001.06208 [stat].
- Spencer Peters and Joseph Halpern. A Unifying Framework for Causal Modeling With Infinitely Many Variables. Journal of Artificial Intelligence Research , 83, August 2025. ISSN 1076-9757. doi: 10.1613/jair.1.15612. URL https://www.jair.org/index.php/jair/article/view/ 15612 .
- Spencer Peters and Joseph Y. Halpern. Causal Modeling With Infinitely Many Variables, December 2021. URL http://arxiv.org/abs/2112.09171 . arXiv:2112.09171 [cs].

- A. Raue, C. Kreutz, T. Maiwald, J. Bachmann, M. Schilling, U. Klingmüller, and J. Timmer. Structural and Practical Identifiability Analysis of Partially Observed Dynamical Models by Exploiting the Profile Likelihood. Bioinformatics , 25(15):1923-1929, August 2009. ISSN 1367-4811, 1367-4803. doi: 10.1093/bioinformatics/btp358. URL https://academic.oup.com/bioinformatics/ article/25/15/1923/213246 .
2. Andrea J. Ray. Reservoir Management in the Interior West. In Henry F. Diaz and Barbara J. Morehouse (eds.), Climate and water: Transboundary challenges in the americas , pp. 193-217. Springer Netherlands, Dordrecht, 2003. ISBN 978-94-015-1250-3. doi: 10.1007/978-94-0151250-3\_9. URL https://doi.org/10.1007/978-94-015-1250-3\_9 .
3. Victor R. Restrepol, Joseph E. Powers, Stephen C. Turner, and John M. Hoenig. Using Simulation to Quantify Uncertainty in Sequential Population Analysis (SPA) and Derived Statistics, with Application to the North Atlantic Swordfish Fishery. In International Council for the Exploration of the Sea , 2011. URL https://api.semanticscholar.org/CorpusID:251047550 .
4. Donald B. Rubin. Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. Journal of Educational Psychology , 66(5):688-701, October 1974. ISSN 1939-2176, 0022-0663. doi: 10.1037/h0037350. URL https://doi.apa.org/doi/10.1037/h0037350 .
5. Jakob Runge, Andreas Gerhardus, Gherardo Varando, Veronika Eyring, and Gustau Camps-Valls. Causal inference for time series. Nature Reviews Earth &amp; Environment , 4:487-505, 2023. doi: 10.1038/s43017-023-00431-y. URL https://doi.org/10.1038/s43017-023-00431-y .
6. Ricardo Sanfelice, Paul Wintz, David Copp, and Pablo Nanez. Behavior in the Intersection of C and D, 2023a. URL https://hyeq.github.io/simulink/intersection-of-C-and-D .
7. Ricardo Sanfelice, Paul Wintz, David Copp, and Pablo Nanez. HyEQ Toolbox, 2023b. URL https://github.com/pnanez/HyEQ\_Toolbox .
8. Ricardo G. Sanfelice. Hybrid Feedback Control . Princeton University Press, Princeton, New Jersey, 2021. ISBN 978-0-691-18022-9 978-0-691-18953-6.
9. Ricardo G. Sanfelice and Andrew R. Teel. Dynamical Properties of Hybrid Systems Simulators. Automatica , 46(2):239-248, February 2010. ISSN 00051098. doi: 10.1016/j.automatica.2009.09. 026. URL https://linkinghub.elsevier.com/retrieve/pii/S0005109809004361 .
10. Adnane Saoud, Mohamed Maghenem, Antonio Loría, and Ricardo G. Sanfelice. Hybrid Persistency of Excitation in Adaptive Estimation for Hybrid Systems. IEEE Transactions on Automatic Control , 69(12):8828-8835, December 2024. ISSN 0018-9286, 1558-2523, 2334-3303. doi: 10.1109/TAC.2024.3422248. URL https://ieeexplore.ieee.org/document/10582521/ .
11. Abraham Jan van der Schaft, Johannes M. Schumacher, and Arjan van der Schaft. An Introduction to Hybrid Dynamical Systems . Number 251 in Lecture notes in control and information sciences. Springer, London, 2000. ISBN 978-1-85233-233-4.
12. Ilya Shpitser and Judea Pearl. Identification of Joint Interventional Distributions in Recursive SemiMarkovian Causal Models. In Proceedings of the 21st national conference on artificial intelligence - volume 2 , AAAI'06, pp. 1219-1226. AAAI Press, 2006. ISBN 978-1-57735-281-5. Place: Boston, Massachusetts Number of pages: 8.
13. Ilya Shpitser and Judea Pearl. Complete Identification Methods for the Causal Hierarchy. Journal of Machine Learning Research , 9(64):1941-1979, 2008. URL http://jmlr.org/papers/v9/ shpitser08a.html .
14. Peter L. Spirtes. Directed Cyclic Graphical Representations of Feedback Models, February 2013. URL http://arxiv.org/abs/1302.4982 . arXiv:1302.4982 [cs].
- A. M. Stuart. Inverse problems: A Bayesian perspective. Acta Numerica , 19:451-559, May 2010. ISSN 0962-4929, 1474-0508. doi: 10.1017/S0962492910000061. URL https://www.cambridge.org/core/product/identifier/S0962492910000061/type/ journal\_article .

- Nathan Taylor, Bruno Mourato, and Denham Parker. Preliminary Closed-Loop Simulation of Management Procedure Performance for Southern Swordfish. Collect. Vol. Sci. Pap. ICCAT , 79(2):705-714, 2022. URL https://www.iccat.int/Documents/CVSP/CV079\_2022/n\_2/ CV079020705.pdf .
- Andrew R. Teel. Lyapunov Conditions Certifying Stability and Recurrence for a Class of Stochastic Hybrid Systems. Annual Reviews in Control , 37(1):1-24, April 2013. ISSN 13675788. doi: 10.1016/j.arcontrol.2013.02.001. URL https://linkinghub.elsevier.com/retrieve/pii/ S1367578813000023 .
- Andrew R. Teel and Joao P. Hespanha. Stochastic Hybrid Systems: A Modeling and Stability Theory Tutorial. In 2015 54th IEEE Conference on Decision and Control (CDC) , pp. 3116-3136, Osaka, December 2015. IEEE. ISBN 978-1-4799-7886-1. doi: 10.1109/CDC.2015.7402688. URL http://ieeexplore.ieee.org/document/7402688/ .
- Andrew R. Teel, Anantharaman Subbaraman, and Antonino Sferlazza. Stability Analysis for Stochastic Hybrid Systems: A Survey. Automatica , 50(10):2435-2456, October 2014. ISSN 00051098. doi: 10.1016/j.automatica.2014.08.006. URL https://linkinghub.elsevier.com/retrieve/ pii/S0005109814003070 .
- E. Walter and Luc Pronzato. Identification of Parametric Models from Experimental Data . Communications and control engineering. Springer ; Masson, Berlin ; New York : Paris, 1997. ISBN 978-3-540-76119-8.
- Zhe Wang, Yuan Liang, David C. Zhu, and Tongtong Li. The Relationship of Discrete DCM and Directed Information in fMRI-Based Causality Analysis. IEEE Transactions on Molecular, Biological and Multi-Scale Communications , 4(1):3-13, March 2018. ISSN 2372-2061, 2332-7804. doi: 10. 1109/TMBMC.2018.2887210. URL https://ieeexplore.ieee.org/document/8579229/ .
- Amanda Warlick, Erin Steiner, and Marie Guldin. History of the West Coast groundfish trawl fishery: Tracking socioeconomic characteristics across different management policies in a multispecies fishery. Marine Policy , 93:9-21, July 2018. ISSN 0308597X. doi: 10.1016/j.marpol.2018.03.014. URL https://linkinghub.elsevier.com/retrieve/pii/S0308597X17307911 .
- Sam A Witty. Bayesian Structural Causal Inference with Probabilistic Programming . PhD thesis, University of Massachusetts Amherst, 2023. URL https://scholarworks.umass.edu/ dissertations\_2/2922 .
- Lei Zan, Charles K. Assaad, Emilie Devijver, Eric Gaussier, and Ali Aït-Bachir. On the Fly Detection of Root Causes from Observed Data with Application to IT Systems. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management , pp. 5062-5069, Boise ID USA, October 2024. ACM. ISBN 979-8-4007-0436-9. doi: 10.1145/3627673.3680010. URL https://dl.acm.org/doi/10.1145/3627673.3680010 .
- S Zhou and Adm Smith. Effect of Fishing Intensity and Selectivity on Trophic Structure and Fishery Production. Marine Ecology Progress Series , 585:185-198, December 2017. ISSN 0171-8630, 1616-1599. doi: 10.3354/meps12402. URL http://www.int-res.com/abstracts/meps/ v585/p185-198/ .
- A. Zhuk, Yu. Zeigarnik, E. Buzoverov, and A. Sheindlin. Managing Peak Loads in Energy Grids: Comparative Economic Analysis. Energy Policy , 88:39-44, 2016. ISSN 0301-4215. doi: https: //doi.org/10.1016/j.enpol.2015.10.006. URL https://www.sciencedirect.com/science/ article/pii/S0301421515301348 .
- Bernt Øksendal. Stochastic Differential Equations . Universitext. Springer Berlin Heidelberg, Berlin, Heidelberg, 2003. ISBN 978-3-540-04758-2 978-3-642-14394-6. doi: 10.1007/978-3-642-143946. URL http://link.springer.com/10.1007/978-3-642-14394-6 .

## A Supplementary Definitions and Standard Assumptions

## A.1 Differential Inclusions and Set-Valued Maps

We follow Goebel et al. (2012) in generalizing to hybrid systems with inclusion constraints. A differential inclusion F : S Ñ R n , for example, specifies the constraint that the time derivative 9 x of the state must be included in the set F p x q Ď R n . Note that the equality constraint 9 x ' f p x q for some f : S Ñ R n is a special case of the broader notion of differential inclusion. To clarify, the stacked double arrows in, for example, S Ñ R n indicate a set-valued mapping from S to a subset of R n . Goebel et al. (2012) define the domain of a set-valued mapping V : X Ñ Y as dom V ' t x P X : V p x q ‰ Hu . The graph of V is then

<!-- formula-not-decoded -->

## A.2 Ordered Set-Valued Maps

Ordered set-valued maps are special cases of set-valued maps, which we use in this paper to keep track of interventions.

Definition 6 (Ordered Set-Valued Map) . Let G ' p G 1 , . . . , G K q be a finite sequence of set-valued maps. We call G an ordered set-valued map , which means it is equipped with the following operation:

<!-- formula-not-decoded -->

Therefore, dom G : ' Ť K k ' 1 dom G k . Given two sequences G ' p G 1 , . . . , G K q and H ' p H 1 , . . . , H L q , we denote G \ H ' p G 1 , . . . , G K , H 1 , . . . , H L q . By slight abuse of notation, we sometimes identify a map G with the corresponding one-element sequence p G q , and also use G in place of G : when the context requires a 'vanilla' set-valued map.

## A.3 Solution Concept

The following definitions and propositions are given almost exactly as stated by Goebel et al. (2012), except that we adapt them slightly for explicitly parameterized hybrid systems (definition 1).

The nature of hybrid systems implies that their solutions should be functions of both continuous time t P R ě 0 and discrete time j P N . Let t j denote the time of the j -th discrete event, with t j ď t j ` 1 for all j P N and t 0 ' 0 . Following (Goebel et al., 2012, Sects. 2.2-2.3), we define, for each possible parameterization θ P Θ and initial condition ξ P S , a 'solution' to H p θ q to be a 'hybrid arc', which is formally a set-valued map ϕ p¨ ; ξ , θ q : R ě 0 ˆ N Ñ R n . We can formalize this time-event space (of which dom ϕ is an example) as follows:

Definition 7 (Hybrid Time Domain from Goebel et al. (2012) (Def. 2.3)) . E Ă R ě 0 ˆ N is a compact hybrid time domain if it is a finite union of sequence of closed intervals E ' Ť J ´ 1 j ' 0 ` r t j , t j ` 1 s ˆ t j u ˘ , where 0 ' t 0 ď t 1 ď ¨¨ ¨ t J , and E is a hybrid time domain if for each p T, J q P E , the set E X ` r 0 , T s ˆ t 0 , 1 , . . . , J u ˘ is a compact hybrid time domain.

Generally, dom ϕ is unknown until after a particular solution ϕ is found, as it depends on the exact sequence of state-dependent jump times; therefore, it is natural to consider ϕ as a-priori set-valued.

Definition 8 (Solution Concept adapted from Goebel et al. (2012) (Def 2.6)) . Consider parameterized hybrid system P ' p H , S , Θ q , with H ' p C, F, D, G q . For θ P Θ , any solution ϕ p¨ ; ξ , θ q to H p θ q must satisfy ϕ p 0 , 0; ξ , θ q ' ξ P C p θ q Y D p θ q and the constraints implied by H p θ q , i.e.:

1. for all j P N such that I j : ' t t : p t, j q P dom ϕ u has nonempty interior, we have ϕ p t, j q P C p θ q , @ t P int I j , and 9 ϕ p t, j q P F θ p ϕ p t, j qq , for almost all t P I j ; [continuous flow regime]
2. for all p t, j q P dom ϕ s.t. p t, j ` 1 q P dom ϕ , we have ϕ p t, j ; ξ , θ q P D p θ q and ϕ p t, j ` 1; ξ , θ q P G θ p ϕ p t, j ; ξ , θ qq [discrete jump regime] .

It is convenient to work with solutions that cannot be extended, as formalized by the following concept.

Definition 9 (Maximal Solutions adapted from Goebel et al. (2012) (Def 2.7)) . A solution ϕ p¨ ; ξ , θ q to H p θ q (as in definition 8) is maximal if there does not exist another solution ϕ 1 p¨ ; ξ , θ q to H p θ q such that dom ϕ p¨ ; ξ , θ q is a proper subset of dom ϕ 1 p¨ ; ξ , θ q and ϕ p t, j ; ξ , θ q ' ϕ 1 p t, j ; ξ , θ q for all p t, j q P dom ϕ p¨ ; ξ , θ q .

Unless specified otherwise, we always consider maximal solutions in this paper. With the solution concept established, we can now state conditions for the existence and uniqueness of solutions. We again borrow from Goebel et al. (2012), and adapt accordingly to support parameterized systems (definition 1).

Proposition 1 (Basic Existence adapted from Goebel et al. (2012) (Proposition 2.10)) . Consider parameterized hybrid system P ' p H , S , Θ q ' p C, F, D, G, S , Θ q , and a standard hybrid system H p θ q ' p C p θ q , F θ , D p θ q , G θ q for some θ P Θ . Let ξ P C p θ q Y D p θ q . If ξ P D p θ q or

- (VC) there exists ϵ ą 0 and an absolutely continuous function z : r 0 , ϵ s Ñ R n such that z p 0 q ' ξ , 9 z p t q P F θ p z p t qq for almost all t P r 0 , ϵ s and z p t q P C p θ q for all t P p 0 , ϵ s ,

then there exists a non-trivial solution ϕ p¨ ; ξ , θ q to H with ϕ p 0 , 0; ξ , θ q ' ξ . 17 If (VC) holds for every ξ P C p θ q Y D p θ q , then there exists a nontrivial solution to H p θ q from every point of C p θ q Y D p θ q . If the foregoing further holds for H p θ q at every θ P Θ , we say P fulfills the conditions for basic existence.

Proposition 2 (Basic Uniqueness adapted from Goebel et al. (2012) (Proposition 2.11)) . Consider parameterized hybrid system P ' p H , S , Θ q ' p C, F, D, G, S , Θ q , and a standard hybrid system H p θ q ' p C p θ q , F θ , D p θ q , G θ q for some θ P Θ . For every ξ P C p θ q Y D p θ q there exists a unique maximal solution ϕ p¨ ; ξ , θ q with ϕ p 0 , 0; ξ , θ q ' ξ provided that the following conditions hold.

- (a) For every ξ P C p θ qz D p θ q , T ą 0 , if two absolutely continuous z 1 , z 2 : r 0 , T s Ñ S are such that 9 z i p t q P F θ p z i p t qq for almost all t P r 0 , T s , z i p t q P C p θ q for all t P p 0 , T s , and z i p 0 q ' ξ , i ' 1 , 2 , then z 1 p t q ' z 2 p t q for all t P r 0 , T s ;
- (b) for every ξ P C p θ qX D p θ q , there does not exist ϵ ą 0 and an absolutely continuous function z : r 0 , ϵ s Ñ S such that z p 0 q ' ξ , 9 z p t q P F θ p z p t qq for almost all t P r 0 , ϵ s and z p t q P C p θ q for all t P p 0 , ϵ s ;
- (c) for every ξ P D p θ q , G θ p ξ q consists of one point.

If the foregoing further holds at every θ P Θ , we say that P fulfills the conditions for basic uniqueness.

## A.4 Finite-Time Measurability of Solution in Initial Conditions and Parameters

Measurability is key to coherently defining causal estimands as (conditional) expectations. In particular, we use the measurability of a time-parameterized 'solution map' jointly in the initial state and parameters. By 'solution map', we refer either to functions ϕ or φ that, when provided some ξ P S and θ P Θ , yield hybrid arc p t, j q ÞÑ ϕ p t, j ; ξ , θ q and time-parameterized function t ÞÑ φ p t ; ξ , θ q respectively.

As stated following definition 1, in this paper, we focus strictly on finite time horizons. Definition 10, below, makes this finite-time limitation precise, and then employs that definition to formalize the time-parameterized solution map and its measurability.

Definition 10 ( t ` Uniquely Evaluable) . Consider parameterized hybrid system P ' p C, F, D, G, S , Θ q that fulfills conditions for basic existence and uniqueness (propositions 1 and 2). Define t ` ' min ξ , θ sup t dom ϕ p¨ ; ξ , θ q , meaning that for every ξ P S , θ P Θ yielding unique solution t, j ÞÑ ϕ p t, j ; ξ , θ q , @ t P r 0 , t ` q there exists j P N such that p t, j q P dom ϕ p¨ , ¨ ; ξ , θ q . We then say that P is t ` uniquely evaluable . 18

17 A non-trivial solution is one with more than a single point in its domain (Goebel et al., 2012, Def 2.5)

18 Unless the space of initial conditions and parameters are limited to exclude reachable states arbitrarily close to Zeno points, Zeno systems will always have arbitrarily small t ` . Additionally, this definition implicitly excludes evaluation times at the end (in continuous time) of eventually discrete solutions. If such solutions are not complete, one might wish to include the final time in the evaluable interval.

Definition 11 (Time Parameterized Solution Map) . Consider t ` uniquely evaluable parameterized hybrid system P ' p C, F, D, G, S , Θ q and its solution map ϕ . Define for all t P r 0 , t ` q , ξ P S , θ P Θ its time-parameterized solution

<!-- formula-not-decoded -->

where j ` t p ξ , θ q is the index of the last discrete jump at time t

<!-- formula-not-decoded -->

Definition 12 ( t ` Measurable) . Consider t ` uniquely evaluable parameterized hybrid system P and its time-parameterized solution map φ . If, for every fixed t P r 0 , t ` q , ξ , θ ÞÑ φ p t ; ξ , θ q is a Borel-measurable function, we say that P has a t ` measurable time-parameterized solution map φ .

## A.5 Flow Preferring Subtraction and Lowering

Definition 13 (Flow-Preferring Subtraction) . Consider parameterized hybrid system P ' p C, F, D, G, S , Θ q that meets the hybrid basic conditions (assumption 4). We borrow the following viability condition from proposition 1 on a point ξ P S , for some θ P Θ .

- (VC) there exists ϵ ą 0 and an absolutely continuous function z : r 0 , ϵ s Ñ R n such that z p 0 q ' ξ , 9 z p t q P F p z p t q , θ q for almost all t P r 0 , ϵ s and z p t q P C p θ q for all t P p 0 , ϵ s ,

We can then transform D to be flow preferring by writing

<!-- formula-not-decoded -->

Recall the definition of ordered set-values maps (definition 6) affording the last p G q operation on G , the jump map.

Definition 14 (Lowering) . Consider parameterized hybrid system P ' p C, F, D, G, S , Θ q that meets the hybrid basic conditions (assumption 4). We write that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.6 Collected Assumptions on the Hybrid System

Assumption 3 (Unique, Complete, and Borel Solution Exists for Differential Inclusion for all S ) . Consider parameterized hybrid system P ' p C, F, D, G, S , Θ q . Assume that

- (F1) for every ξ P S , θ P Θ , T ą 0 , if two absolutely continuous z 1 , z 2 : r 0 , T s Ñ S are such that 9 z i p t q P F θ p z i p t qq for almost all t P r 0 , T s , z i p t q P S for all t P p 0 , T s , and z i p 0 q ' ξ , i ' 1 , 2 , then z 1 p t q ' z 2 p t q for all t P r 0 , T s ;
- (F2) for all ξ P S and θ P Θ , such a z 1 exists for every T P p 0 , 8q ;
- (F3) with z p t ; ξ , θ q ' z 1 p t q for all ξ P S , θ P Θ , and t P r 0 , 8q , ξ , θ ÞÑ z p t ; ξ , θ q is a Borel-measurable function for every t P r 0 , 8q .

Importantly, note that assumption 3 only relates to the differential inclusion, and does not preclude P from jumping, or from pathologies associated with jumps. Additionally, observe that z p t ; ξ , θ q ' ϕ p t, 0; ξ , θ q -that is, statements on z trivially apply to the solution mapping up to and including the time of the first jump.

Assumption 4 (Hybrid Basic Conditions adapted from Goebel et al. (2012) (Assump. 6.5)) . Consider parameterized hybrid system P ' p C, F, D, G, S , Θ q , and assume for all θ P Θ that the following hold.

- (A1) C p θ q and D p θ q are closed subsets of S ;
- (A2) F θ : S Ñ R n is outer semi-continuous and locally bounded relative to C p θ q , C p θ q Ă dom F θ , and F p x , θ q is convex for every x P C p θ q ;
- (A3) G θ : S Ñ S is outer semi-continuous and locally bounded relative to D p θ q , and D p θ q Ă dom G θ .

In particular, (A1) implies that D p θ q and C p θ q must overlap on any shared boundary - solutions that start at or graze this boundary can, non-uniquely, either jump or flow. Additionally, the outer semi-continuity of G θ (A3) requires that, at the boundaries of the pieces in a piecewise G θ , G θ must return values from multiple pieces. Solutions hitting those boundaries can jump to multiple states.

Assumption 5 (Collected Assumptions on the Original System) . The parameterized hybrid system P can be constructed as P ' lower p P Ò q , where P Ò ' p C, F, D, G, S , Θ q , such that:

- (P1) P Ò satisfies assumption 4;
- (P2) P Ò fulfills the conditions for basic existence (proposition 1);
- (P3) P Ò has a unique solution to its differential inclusion F from everywhere in S and Θ (assumption 3);
- (P4) C p θ q is outer semi-continuous at every θ P Θ ;
- (P5) the graph G p D q of the jump set mapping D is Borel;
- (P6) last p G q is single-valued on its domain, with last p G qp x , θ q ' t g p x , θ qu , and g Borelmeasurable for all x , θ P dom last p G q .

## A.7 Well-Behaved Jump Set

Definition 15 (Well-Behaved Set) . Consider Θ Ď R m , S Ď R n , arbitrary set-valued mapping A : Θ Ñ S , and differential inclusion F : S ˆ Θ Ñ R n . Suppose that for every θ P Θ and ξ P S where

(VC S ) there exists ϵ ą 0 and an absolutely continuous function z : r 0 , ϵ s Ñ R n such that z p 0 q ' ξ , 9 z p t q P F p z p t q , θ q for almost all t P r 0 , ϵ s and z p t q P S for all t P p 0 , ϵ s , there also exists some ϵ 1 P p 0 , ϵ s such that

<!-- formula-not-decoded -->

In such a case, we say that A is well-behaved relative to S for Θ and F . For a parameterized hybrid system P ' p C, F, D, G, S , Θ q , we sometimes say that A is well-behaved relative to P .

Assumption 6 (Well-Behaved Interventional Subset) . Consider set-valued mapping ˜ D : Θ Ñ S and parameterized hybrid system P . Assume ˜ D is well-behaved relative to P (definition 15).

Observation 1 (Flow into Subdivisions of C by ˜ D ) . Consider set-valued mapping ˜ D : Θ Ñ S that meets assumption 6 relative to some parameterized hybrid system P ' p C, F, D, G, S , Θ q . It is then the case that, for every θ P Θ and ξ P S where

(VC) there exists ϵ ą 0 and an absolutely continuous function z : r 0 , ϵ s Ñ R n such that z p 0 q ' ξ , 9 z p t q P F p z p t q , θ q for almost all t P r 0 , ϵ s and z p t q P C p θ q for all t P p 0 , ϵ s , there also exists some ϵ 1 P p 0 , ϵ s such that

<!-- formula-not-decoded -->

Proof. Suppose the proposed antecedent and note that a trajectory z pp 0 , ϵ sq Ď C p θ q Ď S fulfills the antecedent of the assumed well-behaved property of ˜ D relative to S (assumption 6). This implies that there exists ϵ 1 P p 0 , ϵ s such that either z pp 0 , ϵ 1 sq Ď int ˜ D p θ q or z pp 0 , ϵ 1 sq Ď S z int ˜ D p θ q . z pp 0 , ϵ 1 sq Ď int ˜ D p θ q is precisely the first case of our desired consequent. Thus, we need only show that z pp 0 , ϵ 1 sq Ď S z int ˜ D p θ q and z pp 0 , ϵ sq Ď C p θ q imply z pp 0 , ϵ 1 sq Ď C p θ qz int ˜ D p θ q . We have z pp 0 , ϵ 1 sq Ď z pp 0 , ϵ sq Ď C p θ q , and can thus take the intersection to see this implies the second case of the desired consequent:

<!-- formula-not-decoded -->

Remark 1 (Universality of Assumption 6) . If and only if assumption 6 holds relative to some parameterized hybrid system P , then it also holds relative to instint p P q , instint Ò p P q , and lower p P q .

Proof. Assumption 6 holding relative to P ' p C, F, D, G, S , Θ q pertains only to F, S , Θ , which are unaffected by instint , instint Ò , and lower .

## B Space Augmentation

It is often useful to parameterize interventions, and a fully expressive interventional semantics benefits from stateful jump maps/sets. Thus, it will be useful to establish a primitive transformation that simply augments the parameter and state spaces, without changing the component functions of the system. Subsequent transformations can then operate on this augmented system. Note that, in eq. (22), we write the transformed jump map in its expanded form as an ordered set-valued map (definition 6).

Definition 16. (Space Augmentation) Consider ˜ S Ď R ˜ n , and ˜ Θ Ď R ˜ m . For any parameterized hybrid system P ' p C, F, D, G, S , Θ q with G ' p G 1 , . . . , G L q , let, for all x P S , ˜ x P ˜ S , θ P Θ , ˜ θ P ˜ Θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then

Observation 2 (Compositions of Space Augmentation Preserves Key Properties) . Consider parameterized hybrid system P that meets assumption 5, and any finite sequence p ˜ S k , ˜ Θ k q of length K such that ˜ S k Ď R ˜ n k and ˜ Θ k Ď R ˜ m k . Let spaug k ' spaug p¨ , ˜ S k , ˜ Θ k q (definition 16) and

<!-- formula-not-decoded -->

P 1 then meets assumption 5 and has a unique solution (propositions 1 and 2) with a t ` measurable (definition 12) time-parameterized solution map φ (definition 11).

Proof. The proposition follows from induction, K ă 8 , and the fact that the space augmentation operation fulfills the same pattern described in fig. 2 for instint . That is, spaug commutes with lower , and it preserves (P1-6) (assumption 5) on an upstream system P 1 Ò ' spaug p P Ò , . . . q . For commutativity, recall that lowering makes a flow-preferring subtraction from the jump set (definition 13), and chooses the last map in the ordered jump map. A flow-preferring subtraction on D 1 p θ q ' D p θ q ˆ ˜ S is dictated entirely by the behavior of F p θ q on C p θ q -i.e. preferflow p D 1 , C 1 , F 1 q ' preferflow p D,C,F q ˆ ˜ S , which implies commutativity on D 1 . Commutativity of the jump map is more straightforward, as, by construction, last p G 1 q ' G L p x , θ q ˆ t ˜ x u ' last p G qp x , θ q ˆ t x u . Assumptions (P1-6) (collected in assumption 5) straightforwardly follow after noting, as we have used in the proof of observation 3, that since every topological space is both open and closed in itself (i.e., clopen), any product with such a space as a factor inherits the open (or closed) property from the other factor relative to the product topology. From here, along similar lines argued in the proof of observation 3, properties like graph closure, outer semi-continuity, Borelness, etc. are preserved obviously by construction.

## C Static-Time and Do Interventions as Special Cases

As a special case of instint , we can also define an intervention that occurs at a fixed, predefined time. 19

19 Note that, while this definition lets us analyze static-time interventions in this theoretical framework, we do find computational implementations in line with the time-splitting operation (Boeken &amp; Mooij, 2024) more practical.

Definition 17 (Static-Time Intervention) . Consider a parameterized hybrid system P defined as the tuple ` C, F, D, G, R 2 ě 0 ˆ S , Θ ˘ . Let time be tracked in the first dimension of the state space, and, in the second dimension, a variable recording whether the intervention has occurred, such that p t, k, x q P R 2 ě 0 ˆ S . Assume k ' 0 at t ' 0 by convention and that F is such that dk { dt ' 0 always. Let ˜ D p θ q ' r λ, λ ` ϵ s ˆ r 0 , . 1 s ˆ S for all θ P Θ , a fixed λ ě 0 , and any ϵ ą 0 . 20 For some ˜ G : S ˆ Θ Ñ S and all p t, k, x , θ q P R 2 ě 0 ˆ S ˆ Θ . We then define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The definition above, it should be noted, is a special case of a more general 'repeated' static-time intervention rstatint (definition 19), which is shown to satisfy the same existence, uniqueness, and measurability theory that we establish below for instint .

Driving one level more granular, we arrive at a transformation representing something akin to the canonical 'do' intervention - again as a special case of instint . This notion has been defined for dynamical systems both via a time-splitting mechanism (Boeken &amp; Mooij, 2024) and by casting a continuous time system as its infinitely precise Euler approximation interpreted as an SCM (Hansen &amp;Sokol, 2014).

Definition 18 (Do-Intervention) . Building directly off definition 17, if ˜ G p x , θ q ' t v u for some fixed v P S and all x , θ P S ˆ Θ , then for some fixed λ ě 0 we write

<!-- formula-not-decoded -->

Alternatively, one might wish to fix an index i P t 1 , . . . , n u and a value v P R 1 . With ˜ G p x , θ q ' r x p 1: i ´ 1 q , v, x p i ` 1: n q s @ x P S and @ θ P Θ , we write instead do ` P , x p i q t ' v ˘ .

These interventional classes form a sort of hierarchy. The jump map of a static-time intervention can be considered the 'pre-treatment' model for a do intervention, and the trigger mechanism encoded in the jump set can be considered a pre-treatment model for when a static intervention occurs. At the highest level, a state-dependent intervention - especially those that can be triggered many times can be thought of as a soft intervention on system dynamics. By couching these interventions directly in the language of established hybrid systems theory, we can more easily borrow theoretical results from that vast body of literature.

## C.1 Repeated Static-Time Intervention

Definition 19 (Repeated Static-Time Intervention) . Consider a parameterized hybrid system P defined as the tuple ` C, F, D, G, R 2 ě 0 ˆ S , Θ ˘ . Without loss of generality with respect to positioning in the state vector, let time be tracked in the first dimension of the state space, and, in the second dimension, a variable recording whether a specified static intervention has recently occurred, such that p t, k, x q P R 2 ě 0 ˆ S . Assume k ' 0 at t ' 0 by convention and that F is such that dk { dt ' 0 always. Also, assume that, for some countable set of unique intervention times Λ Ă R 1 ě 0 , there exists an ϵ such that 0 ă ϵ ă inf ␣ | λ 1 ´ λ 2 | : λ 1 , λ 2 P Λ 2 , λ 1 ‰ λ 2 ( Yt . 1 u . For all p t, k, x , θ q P R 2 ě 0 ˆ S ˆ Θ

20 The ϵ construction ensures the jump set is 'thick'. With a measure-zero jump set, there can exist a solution that reaches the jump set and immediately flows through it, never jumping. A flowing solution is viable from any closed boundary of a jump set where a vector field implied by F θ points into the flow set. Incidentally, (uniformly) 'thick' static-time jump sets also ensure that jumps cannot occur infinitely often. We do not include ϵ as an argument here, as intervention's behavior is identical regardless of the choice of ϵ ą 0 . The counter, k , is required to avoid repeated jumps from the thick jump following λ in time.

and some ˜ G : S ˆ Θ Ñ S , let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With instint i p¨q ' instint p¨ , ˜ D i , ˜ G i q for i P t 1 , 2 u , we can define

<!-- formula-not-decoded -->

Observation 3 ( rstatint Preserves Collected Assumptions) . Continuing from definition 19, if ˜ G meets assumption 2, then P 1 , ˜ D i , and ˜ G i meet assumption 2, and theorem 1 would thus apply to rstatint .

Proof. Assumption 2 comprises sub-conditions (I1) and (I2). (I1) first needs that ˜ D i p θ q is closed @ θ P Θ , which follows here from ˜ D i p θ q being a product of closed sets with the topological space S . Since every topological space is both open and closed in itself (i.e., clopen), any product with such a space as a factor inherits the open (or closed) property from the other factor relative to the product topology. Note that the intervals in the unions over intervals constructed from λ P Λ are guaranteed to be disjoint and uniformly separated by selecting ϵ to be positive and smaller than the closest two intervention times, which means the corresponding countable union must be closed. Similarly, by uniform separation, we have the (I1)-required well-behavedness (definition 15) of ˜ D i p θ q relative to P . (I1) also requires that the graph G p int ˜ D q is open. Note that these jump sets are constant in Θ , and therefore their interior graph is the product Θ ˆ A , where A Ă R n is open. Θ is a topological space, so the product with A inherits the openness of A -thus G p int D q is open. By a similar argument, ˜ D i is closed, which implies that G p ˜ D i q is closed, thereby ensuring G p ˜ D i q is Borel as required by (I1). (I2) asserts straightforward requirements on ˜ G i , none of which are affected by taking a cartesian product with the single-valued, continuous (and therefore both inner and outer semi-continuous) set valued mappings t, k ÞÑ tp t, k ˘ 1 qu . Theorem 1, then, applies here because rstatint is a composition of instint operations with specifications that meet assumption 2.

## D Proof of Theorem 1

The following proof refers to assumption 5, which is an expanded version of assumption 1 that is referenced by theorem 1 in the main text.

Proof. By induction and K ă 8 , we have via lemma 2 that P 1 will meet assumption 5. Note that, by remark 1, if assumption 2 holds relative to P , it will hold relative to any intermediate system in the chain of transformations from P to P 1 . Then, by lemma 5, existence, uniqueness, and measurability follow from P 1 fulfilling assumption 5.

## E Proof that Instantaneous Intervention Preserves Key Properties

Lemma 2 (Instantaneous Intervention Preserves Key Properties) . Consider a parameterized hybrid system P that meets assumption 5. Now, consider set-valued mappings ˜ D : Θ Ñ S and ˜ G : S ˆ Θ Ñ S that fulfill assumption 2 relative to P . The intervened system P 1 ' instint p P q (definition 4) will then also meet assumption 5, and therefore will have a unique and t ` measurable solution for each θ P Θ , ξ P C p θ q Y D p θ q according to lemma 5.

Proof. The proof closely follows fig. 2. In assumption 5, we have that P can be constructed by 'lowering' (definition 14) from a system P Ò that fulfills certain conditions. In lemma 3, we prove that assumptions 2 and 5 imply that the system P 1 is equivalent to a system reached by performing a slightly modified intervention on P Ò (definition 20), and then applying lower . The intervention on P Ò is proven in lemma 4 to preserve properties on the higher system sufficient to say that the lowered system P 1 meets assumption 5. Intermediate statements and proofs for lemmas 3 and 4 and definition 20 can be found in appendix E.1.

## E.1 Intermediate Results for Lemma 2

Lemma 2 argues that an intervened system P 1 ' instint p P q can also be constructed by applying a slightly different interventional transformation to a different system P Ò , and then 'lowering' (definition 14). Additionally, if P meets assumption 5 by way of P Ò , then P 1 must also meet assumption 5. This can be established by showing that the intervention on P Ò preserves properties that allow it to be properly lowered. First, we will define this alternative intervention, then prove commutativity between intervention and lowering, and finally prove that the alternative intervention preserves the properties listed in assumption 5. In the following definition, we use the fact that G is an ordered set-valued map (definition 6), which supports appending ˜ G to the sequence of maps that compose G .

Definition 20 (Instantaneous Intervention for Higher System) . Consider set-valued mappings ˜ D : Θ Ñ S and ˜ G : S ˆ Θ Ñ S and parameterized hybrid system P Ò ' p C, F, D, G, S , Θ q . Now, let

<!-- formula-not-decoded -->

then

<!-- formula-not-decoded -->

Since G 1 is an ordered set-valued map (definition 6), we can derive the following identity, which helps establish some useful intuitions.

<!-- formula-not-decoded -->

Below, we additionally use the fact that last p G 1 q ' G ˜ D .

Lemma 3 (Commutativity of instint and lower ) . Consider parameterized hybrid system P Ò ' p C, F, D, G, S , Θ q that meets assumption 5, and set-valued mappings ˜ D : Θ Ñ S and ˜ G : S ˆ Θ Ñ S that meet assumption 2 relative to P Ò . The following equality then holds, with instint acting as in definition 4 and instint Ò as in definition 20:

<!-- formula-not-decoded -->

Proof. First, we adopt the subscript convention, where we use i as a symbol (not a variable) mapping to the intervention operation, and l as a symbol mapping to the lowering operation. The subscript il ,

for example, indicates a system that has been intervened upon and then lowered. With this convention, we have

<!-- formula-not-decoded -->

We now want to show that every element of the tuple P li equals to the corresponding element in the tuple P il . We begin with tuple elements that are unaffected by both lower and instint . These include the parameter space, the state space, and the flow map, meaning we trivially have that

<!-- formula-not-decoded -->

For the flow set, note that lower leaves it unmodified and that both the higher and lower overloads of instint list the exact same transformation on the flow set. Thus C li ' C il ' C i ' θ ÞÑ C p θ qz int ˜ D p θ q .

We now show equivalence in the jump map - a largely straightforward effort despite its verbosity. Consider the system instint Ò p P Ò q ' p C i , F, D i , G i , S , Θ q . We have that G i is an ordered setvalued map that, when lowered, yields its last component:

<!-- formula-not-decoded -->

Now we consider the path wherein lowering occurs first. We have that G l p x , θ q ' last p G q p x , θ q . By plugging G l into the definition of instint for a lowered system (eq. (3)), equivalence between G li and G il becomes clear.

Finally, we show equivalence in the jump set. In what follows, let C 1 p θ q ' C p θ qz int ˜ D p θ q for all θ P Θ and let ˜ D Y D refer to θ ÞÑ ˜ D p θ q Y D p θ q -wedrop explicit dependence generally on θ for brevity. Also, let the set V A ' t ξ P S : (VC) holds for ξ relative to flow set A p θ q and F θ u . We can write the 'intervention first' path as

<!-- formula-not-decoded -->

Following the 'lower first' path and looking to instint as applied to lowered systems (definition 4), we have

<!-- formula-not-decoded -->

Now, note that because C 1 p θ q Ď C p θ q , we have V C 1 Ď V C , and therefore:

<!-- formula-not-decoded -->

Additionally, by assumption 6 and observation 1, we have that V C Ď V int ˜ D Y V C 1 Ď ˜ D Y V C 1 , where the second subset relation follows from the closure of ˜ D -nothing can 'flow into' the interior of ˜ D without being in the closure of that interior. This leads to

<!-- formula-not-decoded -->

With both D li Ě D il and D il Ě D li , it must be that D li ' D il .

This concludes the proof of equivalence between every element of P li and P li , meaning P li ' P il .

Lemma 4 (Intervention on Higher System Preserves Key Properties) . Consider parameterized hybrid system P ' lower p P Ò q that meets assumption 5, and set-valued mappings ˜ D : Θ Ñ S and ˜ G : S ˆ Θ Ñ S that meet assumption 2 relative to P . Now, consider the following systems:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then P 1 satisfies assumption 5.

Proof. We break the proof into six parts, one for each of the preserved assumptions listed in assumption 5. Recall the explicit form of P 1 Ò given by eq. (34).

Basic Hybrid Conditions (P1). To show that instint Ò (definition 20) preserves assumption 4, we proceed through the three sub-conditions (A1), (A2), and (A3).

For (A1), we must demonstrate closure of the intervened jump and flow sets. For the flow set, note that by definition 20, C 1 p θ q ' C p θ qz int ˜ D p θ q , and that by (A1) holding for P Ò , C p θ q is closed. C 1 p θ q is thus the result of subtracting an open set from a closed set, and is therefore closed. We have similarly required that ˜ D p θ q is closed. Definition 20 specifies that D 1 p θ q ' D p θ q Y ˜ D p θ q , which is the union of closed sets and therefore closed.

For (A2), note that since the flow map F θ is unchanged, it trivially remains outer semi-continuous. We then have that C 1 p θ q Ď C p θ q , from which we can conclude that local boundedness relative to C p θ q and convexity of F p x , θ q for every x P C p θ q implies those properties relative to C 1 p θ q . Additionally, we have that C 1 p θ q Ď C p θ q Ă dom F θ .

Finally, for (A3), we require the outer semi-continuinity of G 1 θ , its local boundedness relative to D 1 p θ q , and that D 1 p θ q Ă dom G 1 θ . The following arguments closely mimick the developments in Definition 2.11 and Lemma 2.21 from Sanfelice (2021) - they show that the composition of a hybrid 'plant' and hybrid 'controller' into a closed loop hybrid system will meet the basic conditions if the plant and controller meet those conditions. In what follows, we work with the identity of G 1 derived in eq. (35).

Outer semi-continunity of G 1 θ means that for every convergent sequence p x i q P D 1 p θ q to x and every convergent sequence p x ` i q P S to x ` , where x ` i P G 1 p x i , θ q for each i , we have that x ` P G 1 p x , θ q .

Note that this is equivalent to graph closure. Now, by closure of D p θ q , D 1 p θ q , G p G θ q , and G p ˜ G θ q , the only potentially problematic limiting points of sequences lying in D p θ qz ˜ D p θ q , or ˜ D p θ qz D p θ q , must lie on the intersection D p θ q X ˜ D p θ q . The intersecting piece, however, returns G p x , θ q Y ˜ G p x , θ q , which will necessarily contain those limiting points.

Local boundedness of G 1 θ relative to D 1 p θ q , then, follows from the local boundedness of G θ relative to D p θ q and of ˜ G θ relative to ˜ D p θ q , and the fact that G θ and ˜ G θ are queried by G 1 θ only from the sets on which they are locally bounded.

Finally, we need that D 1 p θ q Ă dom G 1 θ . Recall the piecewise construction of G 1 θ in definition 20 and that D p θ q Ă dom G θ , ˜ D p θ q Ă dom ˜ G θ . We can then write the following, where we again drop dependence on θ for brevity and write G Y ˜ G in place of x ÞÑ G p x , θ q Y ˜ G p x , θ q .

<!-- formula-not-decoded -->

Thus, we have that (A1), (A2) and (A3) are all preserved in P 1 Ò ' instint Ò p P Ò q , meaning it meets assumption 4.

Basic Existence (P2). To show that conditions for proposition 1 are preserved, we recall from the proof of lemma 6 that it is sufficient to show that (VC) is met (with respect to P 1 Ò ) for all ξ P C 1 p θ qz D 1 p θ q for any θ P Θ .

Ignoring whether the flow appropriately remains in the transformed flow set C 1 p θ q , we know by assumption 3 that there must be some ϵ ą 0 amount of time from which some continuous function can flow from every ξ P S while respecting the differential inclusion. To confirm that (VC) holds at ξ for P 1 Ò , we can check whether some ϵ 1 P p 0 , ϵ s exists where z p t q P C 1 p θ q for all t P p 0 , ϵ 1 s . First, we can decompose the region where (VC) must hold into a union over two cases.

<!-- formula-not-decoded -->

If ξ P int C 1 p θ qz D 1 p θ q Ď int C 1 p θ q , there must be some such ϵ 1 by the openness of int C 1 p θ q in C 1 p θ q .

We can then decompose the boundary region B C 1 p θ qz D 1 p θ q as follows, where we've dropped the dependence on θ for brevity.

<!-- formula-not-decoded -->

The first equality in the final line follows from the assumed closure of ˜ D p θ q implying that B ˜ D p θ qz ˜ D p θ q ' H .

By analogous decomposition to eq. (40), we have that ξ P B C p θ qz D p θ q must meet (VC) with respect to P Ò . By assumption 6 and observation 1, we then know that the solution must 'flow into' either int ˜ D or into C p θ qz int ˜ D p θ q . Because ˜ D is closed, flow into its interior requires that ξ P ˜ D p θ q Ď D 1 p θ q , which we need not consider. This leaves only flows into C p θ qz int ˜ D p θ q , which by construction (definition 20) is equivalent to C 1 p θ q , and therefore satisfies (VC) with respect to P 1 Ò with ϵ 1 as described in observation 1. Thus, instint Ò (definition 20) preserves the conditions for existence as outlined in proposition 1.

Unique Flowing Solution Everywhere (P3). Assumption 3 is preserved trivially, since instint (definition 20) does not alter F , Θ , or S , which are the only system elements involved in assumption 3.

In the remaining results, we use the following observation, leaving its verification from definitions to the reader.

Observation 4. Let A,B : Θ Ñ S be two set-valued maps. Then the graph and set operations commute:

<!-- formula-not-decoded -->

Outer Semi-Continuity of the Flow Set (P4). We need to show the outer semi-continuity of C 1 at every θ P Θ . Recall from definition 20 that C 1 p θ q ' C p θ qz int ˜ D p θ q . By observation 4, G p C 1 q ' G p C p θ qqz G p int ˜ D p θ qq . By assumption 5, we have the outer semi-continuity of C , which directly implies the closure of its graph. By assumption 2, we have that G p int ˜ D p θ qq is open. Thus, G p C 1 q is closed, and therefore by (Goebel et al., 2012, Lemma 5.10) C 1 is outer semi-continuous.

Borel Jump Set Graph (P5). Recall from definition 20 that D 1 p θ q ' ˜ D p θ q Y D p θ q . The Borel σ -algebra is closed under unions, and thus by observation 4 the graph G p D 1 q must also be Borel.

Borel Measurable, Single-Valued Jump Map (P6). We want to show both that last p G 1 qp x , θ q ' t g 1 p x , θ qu for some g 1 and all x , θ P dom last p G 1 q -i.e. that last p G 1 q is single valued - and that g 1 is a Borel-measurable function of initial conditions and parameters on the domain of the intervened jump map. By definition 20, we have that

<!-- formula-not-decoded -->

Now, we have by (I2) (assumption 2) that ˜ G is single-valued, and by (P6) (assumption 5) that G is single-valued. Thus, there must be some g 1 such that last p G 1 qp x , θ q ' t g 1 p x , θ qu on the domain of last p G 1 q .

We now want to show that g 1 is Borel-measurable for every x , θ P dom last p G 1 q . Note that we can equivalently write the following, where we use the lower-case ˜ g and g in reference to the functions that yield the singletons arising from evaluations of ˜ G last p G q .

<!-- formula-not-decoded -->

Note now that the indicator functions involving the jump sets can be written as piecewise functions over a partition defined by the graph of the jump sets. We have assumed that the graphs of ˜ D and D are Borel. Further, by observation 4, both indicator functions can be written as piecewise over Borel partitions, meaning they must be Borel measurable. Again, by (I2) (assumption 2) we have that g and ˜ g are Borel-measurable. Therefore, g 1 must also be Borel-measurable.

Having shown the preservation of each sub-condition listed in assumption 5, this concludes the proof. Indeed, if P Ò meets those sub-conditions, then P 1 Ò will as well. In other proofs, this result can be trivially applied to conclude that lower p P 1 Ò q fulfills assumption 5.

## F Proof that Lowering Induces Existence, Uniqueness, and Measurability

Lemma 1 follows immediately from the following, more precise statement.

Lemma 5 (Existence, Uniqueness, and Measurability of P ) . Consider a parameterized hybrid system P that meets assumption 5. P , then, fulfills the conditions for basic existence (proposition 1), basic uniqueness (proposition 2), and t ` measurability (definition 12).

Proof. This result follows directly from combining lemma 6 and corollary 2, both stated in the following sections.

## F.1 Lowering Preserves Existence and Induces Uniqueness

Lemma 6 (Lowering Preserves Existence and Induces Uniqueness) . Consider parameterized hybrid system P Ò that can be lowered (definition 14) to construct a system P meeting assumption 5. P then fulfills conditions for basic existence and uniqueness (propositions 1 and 2).

We split this proof into two components, one for the preservation of existence, and another for the induction of uniqueness.

## Lowering Preserves Existence

Proof. Recall from definition 14 ( lowering ) that P 1 ' p C, F, D 1 , G 1 , S , Θ q , with

<!-- formula-not-decoded -->

Now, pick some θ P Θ and note that C p θ q ' C p θ q , which follows from the basic conditions on P asserting that C p θ q is closed. By P fulfilling proposition 1, we have that every ξ P ' C p θ q Y D p θ q ‰ z D p θ q ' C p θ qz D p θ q must meet (VC). It will be sufficient, analogously, to show that every ξ P C p θ qz D 1 p θ q also meets (VC), which is precisely the same condition because lower affects neither C nor F . Note that

<!-- formula-not-decoded -->

This yields two cases under which we must check that (VC) holds. For the first case, we know already that (VC) holds for all ξ P r C p θ qz D p θ qs . For the second, we have by construction that (VC) holds. By the subset relation, these cases subsume the desired set.

We thus have our sufficient condition, that (VC) is met for any ξ P C p θ qz D 1 p θ q . As we have placed no constraints on θ , this holds for the entirety of Θ .

## Lowering Induces Uniqueness

Proof. Uniqueness for P 1 involves three conditions on the hybrid system as stated in proposition 2, which we will review in reverse order of complexity. Before proceeding, recall from definition 14 ( lowering ) that P 1 ' p C, F, D 1 , G 1 , S , Θ q , where for all θ P Θ and x P S

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall, also, the convention that G 1 θ p x q ' G 1 p x , θ q for all x , θ P S ˆ Θ . Now, pick some θ P Θ .

Condition (a) requires the uniqueness of solutions to the differential inclusion on the flow set. This condition is precisely what we have presupposed in assumption 3, except that we make the stronger claim that uniqueness holds for any flow in S Ě C p θ q .

Condition (c) requires that G 1 θ is single-valued on the jump set. We have assumed in (P6) that last p G q is no more than single-valued on dom G , which implies that it is single valued on dom G θ for every fixed θ P Θ . Additionally, by construction of D 1 and the basic conditions on P , we have that

<!-- formula-not-decoded -->

Thus, G 1 θ is exactly single valued on D 1 .

Condition (b) requires that the solution cannot flow from the overlap of the jump and flow sets. Precisely, for every ξ P C p θ q X D 1 p θ q , (VC) as used in definition 13 does not hold. By assumption 4 on P , we have that C p θ q ' C p θ q , and recalling definition 13, it is sufficient to show that (VC) does not hold for any ξ in the following set:

<!-- formula-not-decoded -->

This concludes the proof.

## F.2 Lowering Induces Measurability

We first state sufficient conditions for measurability, and then prove that sufficiency. Ultimately, this yields a corollary stating that lowering induces measurability. We make use of the intermediate results and definitions established in appendix F.3.

Assumption 7 (Collected Conditions for Measurability) . Consider parameterized hybrid system P ' p C, F, D, G, S , Θ q . Assume that P

- (M1) is t ` uniquely evaluable (definition 10);
- (M2) has a unique solution to its differential inclusion everywhere (assumption 3);
- (M3) has an outer semi-continuous and closed flow set C p θ q at every θ P Θ ;
- (M4) G is single-valued on dom G , with G p x , θ q ' t g p x , θ qu , and g Borel-measurable for all x , θ P dom G .

Theorem 2 (Measurability of Solution) . Consider parameterized hybrid system P ' p C, F, D, G, S , Θ q and its time-parameterized solution map φ (definition 11). If P meets assumption 7, then φ is t ` measurable (definition 12).

Proof. Under assumption 7, finite jump times and values are Borel measurable in ξ , θ (lemma 7). Additionally, under assumption 3, the solution is Borel-measurable in ξ , θ up to the first jump (F2-3). We are thus able to write the time-parameterized solution as follows, where t 0 p ξ , θ q ' 0 always. For all t P r 0 , t ` q , ξ P S , and θ P Θ :

<!-- formula-not-decoded -->

which comprises a countable sum over Borel-measurable functions of ξ , θ , and is therefore itself Borel measurable. Note that, while we have only shown Borel-measurability for ξ j p ξ , θ q ' ϕ p t j ´ 1 p ξ , θ q , j ; ξ , θ q when t j ´ 1 p ξ , θ q ă t ` , the joint requirement that t j ´ 1 p ξ , θ q ď t ă t ` avoids those unmeasurable cases.

Corollary 2 (Lowering Induces Measurability) . Consider parameterized hybrid system P Ò that can be lowered (definition 14) to construct a system P meeting assumption 5. P then fulfills conditions for the t ` measurability of its time-parameterized solution map φ (definition 11).

Proof. Assumption 5, when combined with the fact that 'lowering' (definition 14) induces uniqueness and preserves existence (lemma 6), subsumes or implies conditions sufficient for the result (assumption 7 and theorem 2). In particular, (M2) maps to (P3), (M3) maps to (A1) and (P4), and (M4) maps to (P6). For (M1), note that t ` measurability requires only t ` ě 0 in addition to existence and uniqueness, which come from lemma 6.

Proof of lemma 5. This result follows directly from combining lemma 6 in appendix F.1 and corollary 2 above.

## F.3 Measurability of Jump Times and Values

Definition 21 (Flowable Region) . Consider parameterized hybrid system P ' p C, F, D, G, S , Θ q . For all θ P Θ , let C F p θ q denote the set of states from which there exist a flowing solution (respecting F θ ) that remains in C p θ q after its start. Precisely, this means that there exist ϵ ą 0 and an absolutely continuous function z : r 0 , ϵ s Ñ S such that z p 0 q P C F p θ q and 9 z p t q P F p z p t qq for almost all t P r 0 , ϵ s and z p t q P C p θ q for all t P p 0 , ϵ s .

Observation 5. Consider parameterized hybrid system P ' p C, F, D, G, S , Θ q that has a unique solution to its differential inclusion everywhere (assumption 3), and where C p θ q is closed for every θ P Θ . In this case, the closure of the flowable region is C F p θ q ' C p θ q .

Proof. From assumption 3, for every θ P Θ , we have that an absolutely continuous function z exists from every z p 0 q ' ξ P S Ě C F p θ q that satisfies F θ . Every interior point ξ P int C p θ q , then, must be in C F p θ q , as some flow must be possible while remaining in C p θ q . With the closure of the flow set, we thus have C p θ q ' int C p θ q Ď C F p θ q . Now, for points ξ P S z int C p θ q , note that flow into C p θ q is only possible from B C p θ q Ď C p θ q . This ensures that C F p θ q cannot contain points outside of C p θ q , further implying that C F p θ q Ď C p θ q ' C p θ q . Thus, by a two-sided inclusion, we have C F p θ q ' C p θ q for every θ P Θ .

Lemma 7 (Measurability of Jump Times and Values) . Consider parameterized hybrid system P ' p C, F, D, G, S , Θ q and its solution map ϕ . If P meets assumption 7, then the time of the j ą 0 'th jump,

<!-- formula-not-decoded -->

is a Borel measurable function of ξ , θ .

Additionally, the solution values at these jump times

<!-- formula-not-decoded -->

are also Borel measurable functions of ξ , θ if t j p ξ , θ q ă t ` .

Proof. Let t 1 p ξ , θ q ' sup t t | p t, 0 q P dom ϕ p¨ ; ξ , θ qu be the first jump time. Note that if the set tp ξ , θ q : t 1 p ξ , θ q ě α u is Borel for all α P R , then t 1 must be Borel measurable. Indeed, we can write that set as a countable intersection of Borel sets, which implies Borelness. Below, we use the closure of the 'flowable region' C F p θ q (definition 21) to rewrite the pre-image on S ˆ Θ of the first jump occurring at or after time α . In particular, we use its closure in order to include the time at which the jump occurs (by including states that flow can reach but not flow from). Note also that, by observation 5, we have that C F p θ q ' C p θ q under conditions already provided in assumption 7. We have

<!-- formula-not-decoded -->

Assumption 7 requires that C p θ q is outer semi-continuous at all θ P Θ . This holds if and only if its graph G p C q is closed (Sanfelice, 2021, pg. 49). Closed sets are Borel, so G p C q must be Borel.

The Borelness of tp ξ , θ q : p θ , ϕ p τ, 0; ξ , θ qq P G p C qu follows from the Borelness of G p C q and ϕ being continuous in ξ , θ on r 0 , t s , and therefore Borel measurable.

The ability to write the set as a countable intersection over rationals is justified by the standard argument. For any fixed τ P r 0 , α s , choose a sequence p τ n q Ă Q Xr 0 , α s such that τ n Ñ τ . This is always possible due to the density of Q Xr 0 , α s in r 0 , α s . If for each n P N we have

<!-- formula-not-decoded -->

then, because ϕ is continuous in time and G p C q is closed, the limit is also included

<!-- formula-not-decoded -->

Countable intersections of Borel sets are Borel, and thus t 1 must be Borel measurable.

We must now expand from the measurability of the first jump to the measurability of all jumps. We can write the second jump time as follows, with g being a function that, when evaluated, returns the single value of G (M4).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Wehave that g is Borel-measurable on the domain of G (M4) and that, by definition of a parameterized hybrid system (definition 1), D p θ q Ă dom G θ , and therefore know that g will only be evaluated where it is assumed to be measurable (M4). Additionally, we have that ϕ is Borel-measurable for j ' 0 up to and including t 1 p ξ , θ q (F2-3). Thus, ξ 2 is the composition of Borel-measurable functions and is therefore itself Borel-measurable. t 2 , subsequently, is the sum of a Borel-measurable function and the composition of Borel-measurable functions. The measurability of t j ` 1 for j ą 0 then follows from its recursive form. We use h p n q p x q to represent the n -fold composition of h with itself. By standard inductive arguments we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As t j ` 1 comprises only sums of compositions of Borel-measurable functions, it must also be Borelmeasurable. Additionally, note that ξ j ` 1 p ξ , θ q ' ϕ p t j p ξ , θ q , j ; ξ , θ q can also be written as the composition of Borel-measurable functions, thereby proving the measurability of jump values.

## G Probabilities of Causation and Fishery Management

## G.1 Historical Context for the Fishery Management Problem

Notions of causal necessity and sufficiency are often productively employed in policy discourse, especially where competing interests require human-understandable justifications as to whether a particular policy is sufficient and/or necessary to achieve desired outcomes. Recall the control theoretic settings involving state-dependent, instantaneous interventions that we have enumerated in the introduction: health-related lockdown measures, interest rate adjustments, and many engineering problems involve cost benefit tradeoffs, where policies are designed to be sufficient for the benefits, but only as costly as necessary . In modern resource management, for example, tragedies of the commons frequently demand a challenging balance between ecological objectives and short and long-term economic outcomes. Additionally, such cases often involve models that our interventional semantics is designed to operate on.

Fishery management offers a particularly rich set of problems where the probabilities of causation can help streamline policy discourse. Over the last few decades, numerous fishery management crises have followed a similar arc: first, growing markets and new technologies result in overfishing to unsustainable biomass levels; then, regulators impose strict catch quotas, gear restrictions, data collection requirements, area closures, and other measures designed to allow stocks to rebuild; after rebuilding stocks, fishing resumes, ideally at more sustainable levels. In 2000, for example, the NMFS and NOAA 21 announced emergency regulatory measures in response to the failure of the Pacific coast groundfish fishery (Anon, 2000). This was followed by an economically tumultuous rebuilding period of around 10 years (Warlick et al., 2018), after which fishing restrictions changed and loosened (Anon, 2010a). Similarly, the 1990s saw significant declines in the Atlantic swordfish fishery (Neilson et al., 2013). In 2000, the ICCAT 22 established an ultimately successful 10-year plan to rebuild the stock (Neilson et al., 2013; Anon, 2010b).

Naturally, these measures were not without significant economic consequences and backlash, both short and long term (Anon, 2000; Cramer et al., 2018; Anon, 2007b). Indeed, in the United States,

21 That is, National Marine Fisheries Service and National Oceanic and Atmospheric Administration.

22 That is, International Commission for the Conservation of Atlantic Tunas.

the Magnuson-Stevens Act (MSA) mandates the multi-objective of avoiding unnecessary economic sacrifice while pursuing long-term economic and ecological outcomes (Anon, 1976, 1996, 2007a, 2018). Myriad ecological and bio-economic dynamical systems approaches were developed during and after these crises to better balance competing objectives (Lee et al., 2000; Ortiz et al., 2010; Restrepol et al., 2011; Taylor et al., 2022). On some occasions, post-mortems were employed to, for example, determine the degree to which rebuilding success was due to management actions or to natural factors such as species biology Neilson et al. (2013). In essence, the goal of such efforts, as stated in the MSA, is to identify and implement sufficient rebuilding measures that would induce no more economic hardship than necessary.

## G.2 Formal Probabilities of Causation

The formal definitions of the probabilities of causation were originally provided by Pearl (1999). These queries are traditionally defined for binary treatment X and outcome Y variables - we enumerate those binary definitions here, and then develop some intuition. In our fishery management example (section 6), we expand to the non-binary setting in keeping with definitions provided by Kawakami et al. (2024) for scalar treatment and outcome variables.

Definition 22 (Probabilities of Causation) . Let X,Y be binary variables within a structural causal model M , and let x , x 1 , y , y 1 , denote the propositions X ' 1 , X ' 0 , Y ' 1 , Y ' 0 respectively. Denote by Y x and Y x 1 the counterfactual outcomes obtained by performing the do-interventions do p X ' 1 q and do p X ' 0 q . 23 The probabilities of causation Pearl (1999), then, are defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

PN p x, y q quantifies the probability that x was necessary to produce outcome y ; PS p x, y q quantifies the probability that x alone would suffice to produce y ; and PNS p x, y q jointly quantifies the event that x is both necessary and sufficient for outcome y .

To compute the probability of necessity, we consider only (condition on) worlds where the events x and y occurred, and then evaluate the probability of Y being false if we intervene to make X false. Similarly, the capacity to produce an outcome - the probability of sufficiency - is computed by conditioning on X and Y being false, and evaluating the probability of Y being true if we now intervene to make X true. A notion balancing the dimensions of necessity and sufficiency is the probability of necessity and sufficiency, which is not a function of the separate probabilities. To evaluate PNS , we do not condition either way, 24 but rather evaluate the probability that both intervening to make X true results in Y x ' 1 and intervening to make X false results in Y x 1 ' 1 .

## G.3 Multi-Year Horizon

In our analysis of the fishery management problem, we analyze only the year-long time scale, but we can define a multi-season model with an arbitrarily long time horizon. Note that, here, we will need to prepend time to the state vector, which becomes r t, z, h 1:3 , b 1:3 s ' x P S ' R 8 ě . See table 2 for a full labeling of model parameters and states.

First, model the season's starting condition via a jump set that triggers at the beginning of each year. Jumps at the season's start obey a map that (1) resets the integrated catch z to zero and (2) sets fishing harvest rates to their noisy, non-null values. Let θ h 2 ∼ N p . 7 , . 07 q and θ h 3 ∼ N p . 07 , . 007 q be elements of θ , season-start times be Z ě 0 (i.e. the beginning of each year), and rstatint (definition 19) be a generalization of statint (definition 17) that applies the jump map at countably many times. 25 Let P 0 ' pp C, F, H , ¨q , S , Θ q describe the fishery in its unfished, natural state, where

23 The canonical do intervention do p X i ' x i q fixes the structural equation for X i to a constant, i.e. s i ' x i .

24

Any conditioning here would bias the outcome. Suppose X and Y are causally disconnected. If you condition on x, y , you make Y x true by default, and if you condition on x 1 , y 1 , you make Y x 1 false, effectively making one of the components satisfied for free, which is undesirable.

25 Definition 19 also requires a binary auxiliary variable to be part of the state-space - without loss of generality, however, we omit that here for smoother exposition. Additionally, this construction assumes the same

Figure 3: Examples of the biomass trajectories of apex and intermediate predators, as simulated from the model proposed by Zhou &amp; Smith (2017). The panels comprise simulations with, (left) no fishing pressure, (center) fishing pressure kept up throughout the year, (right) fishing regulators ending the season when reported catch meets the total allowable catch (TAC) quota of 50 biomass units.

<!-- image -->

Table 2: Parameters and notation for the fishery example.

| name                                                                                                                                                                                    | notation                          | in season                                                       | after season   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|-----------------------------------------------------------------|----------------|
| time total catch fishing pressure forage fishing pressure intermediate fishing pressure apex biomass forage biomass intermediate biomass apex desired outcome lower threshold TAC quota | t z h 1 h 2 h 3 b 1 b 2 b 3 γ q i | 0 h 2 ' Normal p . 7 , . 07 q h 3 ' Normal p . 07 , . 007 q 130 | 0 0 0 130      |

S ' C ' R 8 ě 0 , and ¨ simply indicates the irrelevance of the jump map in the natural state of the fishery.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The season's end can be described by setting the harvest rates to zero when the catch exceeds a threshold q i (with i P t 1 , 2 u ). From these, we can construct parallel worlds with the same random initial conditions and parameters.

fishing pressure year over year. This can be generalized to independently sampled pressures at each year, though some additional theoretical machinery would be required for an infinite time horizon. See Teel &amp; Hespanha (2015) for one possibility.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## G.4 Narrative Fishery Management Example

In the main body of the paper, we emphasized the construction of the probabilities of causation, rather than their application. Still, some readers may appreciate a more narrative structure around these concepts. We provide that here.

Example 5 (Probabilities of Causation for Total Allowable Catch (TAC) Quotas) . Now, suppose a new commercial fishery is being opened up and that, in the first year, fishery managers allow commercial fishing year-round. R s models this world (or equivalently, R q 1 when the TAC quota q 1 is sufficiently large so as to have zero probability of being reached). This results in a failure to preserve the intermediate level biomass above the desired level γ . Suppose γ ' 130 units. Facing ecological scrutiny, fishery managers ask: given that we allowed year-round fishing and failed to achieve our outcome, what TAC quota would have a high probability of being sufficient for success? This is a probability of sufficiency query. They introduce a strict TAC quota of 30 units with a relatively high probability of sufficiency (fig. 4). In the next season, they succeed in meeting biomass targets. Subsequently, however, economic interests and local representatives insist that such a low, strict TAC was not necessary to achieve this outcome. They point out that, in comparison to a more lenient TAC of 50 units, there is a low probability that the strict TAC of 30 was necessary (fig. 5). Fishery managers, in turn, worry that the probability of success with a TAC of 50 might be too low. Before the start of the next season, stakeholders resolve the disagreement by identifying a TAC that yields a high probability of sufficiency and necessity when contrasted with year-round fishing ( R s ), all while avoiding stricter catch limitations that are not justified by gains in the probability of necessity and sufficiency (fig. 6).

The simulated results presented here ran on a consumer grade laptop in the order of one hour.

## G.5 Event-Time Attribution

In the main body of the paper (and in example 5), we analyzed the probabilities of causation as they relate to contrastive policies. In other words, we asked causal attribution questions at the policy level. Queries about the probabilities of causation, however, such as 'was x necessary to achieve y ', are ambiguous when a real world event x is multi-faceted and potential alternative actions are plentiful. In example 5, we mapped the events x and x 1 onto particular TAC quotas. In this next example, however, we will define our event of interest as involving the time at which an intervention occurs. We can make this precise by constructing twin worlds using the tools provided in this paper -particularly by additionally employing the static intervention statint (definition 17).

Example 6 (Probability of Necessity of State-Dependent Intervention Timing) . Consider worlds where the season ends before some time λ , and biomass goal γ is achieved at a later time τ (for instance, at the end of the year). Fishery managers wonder whether they might fail to meet their biomass goals if, contrary to fact in those cases, the season had ended at or after time λ . The relevant question is: was ending the season before time λ necessary for achieving the biomass goal? We can answer this question by asking a probability of necessity query. Unlike in example 5, however, the causal attribution question relates to the time at which the intervention occurs. Consider the following binary predicates, where T p φ q 1 q extracts the time at which the season ends due to the TAC quota q 1 : 26

<!-- formula-not-decoded -->

26 See appendix I for information and general notation for extracting event times from solutions to hybrid systems. And note that, in lemma 7, we prove that jump times are Borel-measurable in ξ and θ .

Probabilities of causation (no pressure vs. quota=30)

<!-- image -->

Figure 4: Step 1 in the example narrative (example 5). Within the first year of commercial fishing, the fishery has no quota (here it is enough to set it to q ' 120 , which is never met), and falls below sustainable biomass. Conditioning on this failure, the regulators seek an intervention with a high probability of changing this outcome next time along the counterfactual dimension. They implement a strict TAC of 30 , evaluating the probability of sufficiency (the probability of achieving sustainable biomass above the desired threshold of 130) to be 0.87. Probabilities of causation (quota=30 vs. quota=50)

Figure 5: Step 2 in the example narrative (example 5). The season ran with a TAC of 30 units, and the intermediate biomass target reference limit ( γ ' 130 ) was met. Conditioning on this, parties interested in increasing the fishing quota ask whether such a low TAC was necessary . They seek an alternative quota along the counterfactual dimension that, when contrasted with the factual TAC of 30 , reveals the factual TAC as probably unnecessary. As a counterexample, they choose a TAC of 50 , which yields a relatively low probability of necessity ( . 18 ). Probabilities of causation (no pressure vs. quota=35)

<!-- image -->

Figure 6: Step 3 in the example narrative (example 5). This time, before the season starts and prior to seeing what the outcome will be, both sides aim to find a quota with a large probability of both necessity and sufficiency. They contrast proposed TAC quotas with a baseline, status quo TAC of 120 units (never met). They notice that the probability surface flattens out above . 60 , meaning further improvement in the probability of necessity and sufficiency would require excessive limitations in quota. Ultimately, they agree on a quota that results in a value above 0 . 6 , i.e., a TAC quota of 0 . 35 .

<!-- image -->

Figure 7: Samples from the Bayesian dynamics based on the fishery model presented by Zhou &amp; Smith (2017), but with the season ending at different times. We show three end dates and their effect on the biomass at the intermediate trophic level.

<!-- image -->

Recall that the probability of necessity is P p Y do p X ' 0 q ' 0 | X ' 1 , Y ' 1 q (with shorthand P p y 1 x 1 | x, y q , see table 1). To coherently characterize this in our example, we must define what it means to perform the intervention do p X ' 0 q . Given our definition of the predicate X above, do p X ' 0 q suggests an intervention that results in a world where the season ends at or after λ , with all else (such as the noise or the resulting fishing pressure of 0) remaining equal. Importantly, there are many such worlds, which means the probability of necessity must adopt an existential flavor: 'under exogenous noise where X ' 1 and Y ' 1 , what is the probability that all worlds consistent with intervention do p X ' 0 q fail to meet the outcome?' 27 To precisely define this set of interventional worlds, we build off notation from example 5, and consider a twin world under a static intervention occurring at some time λ 1 ě λ , but with the same interventional jump map utilized in the world P q 1 .

<!-- formula-not-decoded -->

As described following definition 18, the trigger dynamics of the state-dependent intervention can also be considered a sort of 'treatment mechanism' determining the time at which a static intervention occurs. By constructing a world where direct control over the intervention timing is possible, we are able to disentangle these mechanisms. Importantly, note that while P λ 1 is constructed via a transformation on P s , it is equivalent to a single-season world constructed from an intervention on P q 1 that directly controls the season-ending time independently of causally upstream events in the system's simulation.

<!-- formula-not-decoded -->

Note that if b τ λ 1 monotonically decreases as λ 1 Ñ8 , then we can equivalently write the event y 1 x 1 as I r b τ λ 1 ă γ s . Indeed, under our model and distributions on ξ and θ , this is the case, and so we can finally precisely express the probability that ending the season before time λ is causally necessary to achieve the biomass outcome:

<!-- formula-not-decoded -->

Unlike in example 5, conditioning on the factual interventional event is required, because knowing that the season ended before λ carries information about the model parameters: the earlier the TAC quota is met (at times prior to λ ), the faster the catch rate. Faster catch rates stem from some combination of higher fishing pressure ( h 1:3 ), higher initial biomass ( b 1:3 at t ' 0 ), higher growth

27 This can equivalently be stated: '...probability that there does not exist a world under do p X ' 1 q where the outcome is met.' See recent work by Li &amp; Pearl (2024) and Kawakami et al. (2024) for more information on how non-binary variables lead to these kinds of existential statements. This also relates to abstract interventions, which has been studied by Beckers &amp; Halpern (2019); Beckers et al. (2019).

Figure 8: For each λ i from a grid of intervention times we (1) condition on the season ending before λ i and on the successful outcome, and (2) we intervene so that the end of the season occurs at λ i . The top panel shows the probability that ending the season before λ i was necessary to achieve biomass targets, while the bottom panel shows the counterfactual biomass distribution under interventions ending the season at various λ i . In the violin plot, we differentiate between counterfactual uncertainty for all worlds (orange and gray), and counterfactual uncertainty after selecting only worlds where the TAC quota was reached before λ i and biomass goals were met at τ . In other words, the gray violins show biomass probabilities if regulators had ended the season at λ i in cases where they ended before λ i and met biomass goals. The probability of necessity, then, is the proportion of the gray distribution falling below the target level of γ .

<!-- image -->

rates, etc., all of which influence whether the biomass target will be achieved under alternative season-ending times, even after conditioning on success in the factual world. 28

Returning to the example, consider a range over fixed threshold λ , approximated by a finite sequence p λ i q . For each λ i , we (1) condition on the season ending before λ i and on the intermediate biomass at τ being above γ , and (2) intervene so that the end of the season occurs at λ i (and not earlier). The relevant probability of necessity query is whether intervening before λ i was necessary for the success. That is, for each λ i we inspect the posterior predictive distribution of the intermediate biomass at τ under the intervention, and inspect the probability that this outcome is below γ . The results of an estimation are available in fig. 8.

The simulated results presented here ran on a consumer grade laptop in the order of one hour.

## H Holling-Tanner Fishery Model

The fishery management model presented by Zhou &amp; Smith (2017) describes the population dynamics for a given trophic level according to the Holling-Tanner model:

<!-- formula-not-decoded -->

where B is the biomass of the species, r is the intrinsic growth rate, K is the carrying capacity, M is the mortality rate due to predation, and F is the fishing mortality rate. Elsewhere in the paper, we have avoided using Zhou &amp; Smith's notation, so-as to avoid overloads with the hybrid system literature. In our paper, we use h instead of F , and the lowercase b for biomass, with subscript i indicating trophic level. In this appendix section, however, we will use Zhou &amp; Smith's notation.

28 In other words, we do not necessarily have exogeneity here: Y x ✚ ✚ K K X ; Y x 1 ✚ ✚ K K X . This further aligns with understanding the state-dependent trigger dynamics as a sort of 'treatment mechanism.' Exogeneity, here, is violated because ξ and θ can be considered 'parents' of both X and Y . In violating exogeneity, non-parametric identification results for this example may be out of reach (Li &amp; Pearl, 2024; Kawakami et al., 2024). As this paper does not address non-parametrics, however, a parametric identification of initial conditions and parameters would be sufficient for the identifiability of PNS/PN/PS. Parametric identification in hybrid systems has been studied by Johnson (2023).

The mortality rate due to predation is modeled as:

<!-- formula-not-decoded -->

where p is the maximum predation rate, B pred is the biomass of the predator, and D is the biomass at which predation reaches half its maximum.

The carrying capacity for a predator species is given by:

<!-- formula-not-decoded -->

where e is the efficiency of converting prey biomass into predator biomass.

The bottom trophic level dynamics follows:

<!-- formula-not-decoded -->

where M 12 is the mortality rate due to predation from intermediate predators.

Species in the intermediate level act as both predator and prey:

<!-- formula-not-decoded -->

The top trophic level follows:

<!-- formula-not-decoded -->

The catch rate for the intermediate trophic level is given by the following - note that, in the main body of our paper, we use z for the integrated catch, meaning Catch below corresponds to 9 z .

<!-- formula-not-decoded -->

Fishing efforts for each trophic level are assumed to remain constant over time unless intervened on.

<!-- formula-not-decoded -->

## I Practical Utilities for Tracking Intervention Times and Values

In many counterfactual estimands, we must translate an event's characteristics from one world to another. To do so, we require the ability to extract certain event properties from a hybrid system's solution. By recording event specifications in auxiliary state variables, these can be straightforwardly read off of solution evaluations at any particular time. First, consider an original system P with time recorded faithfully in the first dimension, and compatible interventional jump set ˜ D and jump map ˜ G . Assume the intervention preserves the faithful recording of time and that ˜ G is single valued everywhere.

To start, we augment the state space with an intervention jump counter j , an intervention time t k , and an intervention value v k . Our goal is to record the time and jump value corresponding to the k 'th occurrence of the intervention. Let ˜ S ' S ˆ R 2 ě 0 ˆ R and ˜ Θ ' H , and augment the state space accordingly.

<!-- formula-not-decoded -->

Now, augment the original interventional specification to appropriately track event details in these auxiliary state variables. For all admissible inputs, and fixed integer k , let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The intervention time and value can then be read directly off of a solution satisfying the constraints of instint ` P 1 , ˜ D 1 , ˜ G 1 ˘ . To track whether an event occurred k times, we can initially set, for example, t k ' ´ 1 . Apositive t k would indicate that the event had occurred. This can be switched to 8 instead for more natural time inequalities if boundedness of the states is not a concern.

## I.1 Notation for Extracting Times and Values

We now describe some general notation for extracting values of t k and v k from a solution. Consider a hybrid arc ϕ i satisfying a system arising from the i 'th instint transformation of that system, where the interventional jump set and map had been augmented to record its k 'th jump. Given the solution's time parameterization t ÞÑ φ p t ; ξ , θ q , we use T p k q i p φ m ě i p¨ ; ξ , θ qq for the time at which the k 'th jump occurred, and V p k q i p φ m ě i p¨ ; ξ , θ qq to extract the state's value immediately following the jump. When clear from context, the function V extracts only one element of the state. The caveat that m ě i simply specifies that properties of the k 'th jump due to transformation i can be read off of a solution to any further transformed system. As shorthand, we sometimes write φ m ě i ' φ m ě i p¨ ; ξ , θ q , taking the random inputs as implicit. Additionally, in settings involving interventions that occur only once in the relevant time window, or where the order of interventional transformation is clear and denoted using a symbolic subscript like s , we use, for example, T s p φ s q to extract the event's time.

## J State-Dependent Intervention in the Forward Euler Representation

Consider a forward-Euler approximation of a system of ODEs, where, for simplicity, we will assume that ∆ t ą 0 . If f is the right-hand side of the continuous-time differential equation x 1 ' f p x q , we can write structural equations x t ' x t ´ ∆ t ` f p x t ´ ∆ t , u q ∆ t , where t ě 0 , x t P R is the value of the state variable x at time t , u P R is a fixed realization of exogenous noise (representing unknown parameters, for example), and x 0 is fixed to some constant initial condition. Suppose now that we wish to intervene such that the system jumps according to a function g ' x ÞÑ x ` 1 when x falls to some threshold τ . To implement this, we must modify the structural equation for x t for all t under question. That is, we replace the original structural equation with the following piecewise construction.

Let ¯ x t ' x t ´ ∆ t ` f p x t ´ ∆ t , u q ∆ t denote the value that would be obtained under the original (nonintervened) Euler update, and let D ' t x P R | x ď τ u denote the domain in which the jump is triggered. Then the intervened structural equation can be approximated as follows:

<!-- formula-not-decoded -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our claims are described in the abstract and clearly enumerated at the end of the introduction. The first manifests in the definition of an instantaneous intervention (definition 4), the second in theorem 1 and its proof (appendix D), and the third in section 6.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: A limitations section is provided and includes references to other locations in the paper where we discuss limitations reviewed therein.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Proof sketches are provided in the main body and references are provided to detailed formal proofs in the appendix.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: We do not include experiments in the main body, but do provide some simulated estimation results in the appendix. We provide ample model details therein for reproducibility and provide links to our simulation code.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: We provide links to our simulation code, which runs in self-contained Jupyter notebooks.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The main paper does not include experiments, but the simulation analyses offered in the supplementary material do have all model parameters and variable distributions clearly enumerated.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: We do not include experiments in the main paper, but our simulation analyses in the appendix do include credible intervals representing Bayesian prior predictive marginals.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The main body does not include experiments, but for our supplementary simulated analyses, we do state our very small computational requirements.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research does not raise any ethical concerns and is unrelated to areas covered by the Code of Ethics.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no direct path for societal impact from our work, beyond, of course, pipelines that blindly use potentially incorrect models for high-stakes decision making.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: See above.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use such assets.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: This paper does not release any new assets.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper involves neither crowdsourcing nor research with human subjects.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.