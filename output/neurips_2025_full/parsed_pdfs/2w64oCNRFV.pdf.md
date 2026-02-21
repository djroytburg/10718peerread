## From Black-box to Causal-box: Towards Building More Interpretable Models

## Inwoo Hwang Yushu Pan Elias Bareinboim

Causal Artificial Intelligence Lab Columbia University inwoo.hwang@columbia.edu {yushupan, eb}@cs.columbia.edu

## Abstract

Understanding the predictions made by deep learning models remains a central challenge, especially in high-stakes applications. A promising approach is to equip models with the ability to answer counterfactual questions - hypothetical 'what if?' scenarios that go beyond the observed data and provide insight into a model reasoning. In this work, we introduce the notion of causal interpretability, which formalizes when counterfactual queries can be evaluated from a specific class of models and observational data. We analyze two common model classes - blackbox and concept-based predictors - and show that neither is causally interpretable in general. To address this gap, we develop a framework for building models that are causally interpretable by design. Specifically, we derive a complete graphical criterion that determines whether a given model architecture supports a given counterfactual query. This leads to a fundamental tradeoff between causal interpretability and predictive accuracy, which we characterize by identifying the unique maximal set of features that yields an interpretable model with maximal predictive expressiveness. Experiments corroborate the theoretical findings.

## 1 Introduction

Despite the remarkable success of deep learning models across a wide range of tasks - including image recognition [7, 16], natural language processing [3, 35], and reinforcement learning [32, 34] these models remain fundamentally opaque. Although they are highly effective at predicting labels based on statistical correlations in the data, they lack the capacity to explain the reasoning behind their predictions, earning them the colloquial label of 'black boxes.' In other words, current models are difficult to interpret: they lack the ability to justify why a particular decision was made, identify which input factors were most influential, or reason about how outcomes might differ under alternative, counterfactual conditions. This interpretability gap raises concerns in high-stakes domains such as healthcare, law, and scientific discovery, where understanding how and why a model makes a decision is as important as the decision itself.

Arich body of research on explainable AI (XAI) has been developed to better understand the behavior of learned models. For instance, post-hoc explanation methods such as LIME [30], SHAP [20], and Grad-CAM [31] generate local or visual attributions in terms of pixels or extracted features to help interpret predictions. Other approaches aim to build intrinsically interpretable models, such as those that impose sparsity constraints [22], restrict final layers [38], or leverage decision tree structures [37], often trading off model complexity for greater transparency. While these techniques offer useful insights, they fail to bridge the gap between low-level features and high-level, human-understandable features that might explain the behavior of a model.

One promising avenue for bridging this gap is counterfactual reasoning. Answering what if questions - such as 'Would the diagnosis have changed if a different treatment had been administered?' or

Figure 1: (a) Illustration of different model classes: counterfactually consistent models (blue) and blackbox/concept-based models (yellow). (b) Original input image and corresponding predictions from each model. (c) Counterfactual predictions: models in the top row predict consistently across instantiations within the class, while those in the bottom row produce inconsistent predictions.

<!-- image -->

'Would the person have been classified differently if their income were higher?' - plays a central role in human reasoning and forms the basis of many explanatory and decision-making processes [1, 26, 27]. Enabling AI systems to reason counterfactually opens the door to more interpretable models - ones that can not only predict outcomes accurately but also explain their decisions in a meaningful, human-aligned way.

Recently, concept-based prediction models [15, 23] have been proposed to improve interpretability by enabling reasoning over human-understandable features. These models aim to answer counterfactual queries of the form: 'Given an input x , how would the model's prediction change if a feature W were modified from w to w ′ ?' Such queries allow users to explore the influence of high-level features like the presence of a smile or the existence of a tumor - on a model's prediction, providing a possible route to assess whether the model reasoning aligns with human expectations.

Despite their appeal, existing concept-based approaches are oblivious to the causal relationships between features. As a result, they may not reflect the real-world mechanisms or incorporate commonsense knowledge faithfully. While some recent methods attempt to introduce causal structure into concept-based models [4], they frequently lack guarantees of counterfactual consistency - that is, the property that models within the exact class yield consistent answers to the same counterfactual query.

To illustrate this limitation, consider a task of predicting facial attractiveness. Suppose two models, C and D, from the same concept-based class, represented by the yellow circle in Fig. 1-(a), are trained on the same dataset. They first will have the identical attribute prediction, for example, both will predict a lower attractiveness score for the given image (Fig. 1-(c), yellow). However, when they evaluate the counterfactual question 'What would the attractiveness be had the person smiled?', model C will maintain the low attractiveness score while model D will raise the attractiveness score (Fig. 1-(c), yellow). This discrepancy reveals a deeper issue: the model class is not counterfactually interpretable, as it does not constrain the space of counterfactual responses. In such cases, users have no principled way to determine which answer to trust, rendering the query effectively unanswerable. In contrast, the model class in blue is desirable since any pair of models - such as model A and B will give the exact same answer for both attribute and counterfactual predictions. In this case, one can assert that the attractiveness would be raised had the person smiled, which indicates the model made the decision based on the feature 'Smile' and this is aligned with human understanding [8].

In this work, we introduce the notion of causal interpretability, which concerns whether a prediction model can be interpreted consistently across counterfactual scenarios - drawing a connection between XAI and causal inference [1, 26]. Intuitively, a model class is said to be causally interpretable if all models within the class yield consistent predictions under counterfactual interventions, as illustrated in blue in Fig. 1. We then show that a blackbox model, which maps inputs directly to labels, is never causally interpretable. That is, such models fundamentally lack the structure needed to answer counterfactual questions. We also demonstrate theoretically that concept-based models [15], which rely on all observed features for prediction, are also not guaranteed to be causally interpretable. Interestingly, we show that causal interpretability can be recovered by constraining the model to use only a certain subset of features.

Against this background, we develop a general approach for building causally interpretable models that can answer counterfactual queries consistently by design. Specifically, we propose a complete graphical criterion for determining whether a model that uses a given set of features for prediction is causally interpretable with respect to a counterfactual query. This enables the understanding of (i) which counterfactual questions a given model can answer, and (ii) which models can answer a given counterfactual question. Our framework also reveals a fundamental tradeoff between causal interpretability and predictive accuracy. We characterize the unique maximal set of features that preserves causal interpretability, thereby providing a principled method for building models with maximal expressive power under interpretability constraints. A notable practical implication is that our approach does not require full specification of the causal graph or modeling of unobserved confounders; it only involves the descendants of the target features in the counterfactual query. Experimental results corroborate the proposed theory. More specifically, our contributions are as follows:

- (Sec. 2) We introduce the notion of causal interpretability (Def. 2), which states whether we can evaluate the prediction of the model under counterfactual conditions from observational data. Based on this formulation, we show that a blackbox model is never interpretable (Prop. 1), whereas a concept-based model is also not interpretable in general, in contrast to prior belief.
- (Sec. 3) We develop a graphical criterion that determines whether the model is causally interpretable with respect to the query (Thm. 1). We characterize the unique maximal set of features yielding interpretable architecture (Thm. 2) and provide a practical way of evaluating such queries from the data (Thm. 3). Finally, these results reveal a fundamental tradeoff between the causal interpretability and predictive accuracy (Thm. 4).

Preliminary. Here, we introduce notations and terminologies used in the paper. We use bold letters to denote a set of random variables or their assignments. We use capital letters to denote a random variable or a random vector (e.g., X ) and lower case letters to denote their assignments (e.g., x ). x ∪ Z denotes the subset of x corresponding to variables in Z and x \ Z denotes the value of X \ Z consistent with x .

We employ a structural causal model [1, 26] as our semantical framework. A structural causal model (SCM) M is a 4-tuple ⟨ U , V , F , P ( U ) ⟩ , where U is a set of exogenous variables, V = { V 1 , · · · , V n } is a set of endogenous variables, F = { f V 1 , · · · f V n } is a set of functions determining V as V j ← f V j ( Pa V j , U V j ) , where Pa V j ⊆ V \ { V j } and U V j ⊆ U for all V j ∈ V , and P ( U ) is a distribution over U . An SCM M induces a causal diagram G and a distribution over the endogenous P ( V ) . We use graphical kinship to represent the relationships between the variables. ND ( W ) denotes non-descendants of a variable W , and ND ( W ) := ∩ W i ∈ W ND ( W i ) denotes non-descendants of a set of variables W . We now define an SCM that describes a generative process that includes images X and labels prediction ̂ Y [24].

Definition 1 (Augmented SCM) . An augmented SCM (ASCM) over a generative level SCM M 0 = ⟨ U 0 , V 0 , F 0 , P 0 ( U 0 ) ⟩ is a tuple M = ⟨ U , { V , X , ̂ Y } , F , P ( U ) ⟩ such that (1) exogenous variables U = { U 0 , U X } ;

(2) V = V 0 are labeled observed endogenous variables, X is an m -dimensional mixture variable, and ̂ Y is a (predicted) label;

(3) F = {F 0 , f X , f ̂ Y } , where f X maps from (the respective domains of) V ∪ U X to X and a classifier f ̂ Y maps from (the respective domains of) the subset of { V , X } to ̂ Y ; and (4) P ( U 0 ) = P 0 ( U 0 ) .

<!-- image -->

̂

(a) Blackbox prediction (BP).

̂

(b) Concept-based prediction (CP).

̂

(c) Generalized CP (GCP).

Figure 2: Causal diagrams for different types of predictive models.

An ASCM M represents a sequential generative procedure of latent generative factors (i.e., concepts) V , the image X , and the label prediction ̂ Y . First, the latent features V are generated by the underlying M 0 . The induced causal diagram G V is called a latent causal graph (LCG). The highdimensional mixture X (e.g., image) is then generated from V (and U X ), and subsequently, ̂ Y is generated from the subset of { V , X } , where f ̂ Y is a classifier that predicts the label. We let Ω := {M : ASCM over M 0 } be the space of ASCMs. Omitted proofs are provided in Appendix A.2.

## 2 Causal Intepretability - Foundations

In this section, we formalize the notion of causal interpretability and examine whether existing approaches could elicit counterfactual questions consistently in a valid manner.

We start by analyzing two important classes of predictive models: blackbox and concept-based models. As illustrated in Fig. 2a, blackbox prediction (BP) models make a prediction on the label from the image pixels X (i.e., f ̂ Y : D ( X ) →D ( ̂ Y ) ). In contrast, concept-based prediction (CP) models predict the label based on the generative factors of the image (i.e., f ̂ Y : D ( V ) →D ( ̂ Y ) ), as illustrated in Fig. 2b. In other words, the classifier of a concept-based model uses the features to make the predictions, instead of the image itself. Formally, a class of BP models and a class of CP models are respectively denoted as Ω BP and Ω CP, where Ω BP := {M ∈ Ω | f ̂ Y : D ( X ) →D ( ̂ Y ) } and Ω CP := {M ∈ Ω | f ̂ Y : D ( V ) → D ( ̂ Y ) } . The following examples illustrate the generative process of BP and CP models.

Example 1 (Blackbox Model) . Consider a task of estimating the attractiveness of a human face represented in an image X . Augmented generative process (ASCM) of the prediction by a BP model is given as M BP = ⟨ U = { U F , U S , U C 1 , U C 2 , U X } , {{ F, S, C } , X , ̂ Y } , F BP , P BP ( U ) ⟩ , where

<!-- formula-not-decoded -->

̂ Y is the label (attractiveness) prediction, the exogenous variables U F , U S , U C 1 , U C 2 are independent binary variables, and P BP ( U F = 1) = 0 . 4 , P BP ( U S = 1) = 0 . 6 , P BP ( U C 1 = 1) = 0 . 3 , P BP ( U C 2 = 1) = 0 . 6 . The exogenous variable U X (representing other generative factors) can include (or be correlated to) { U F , U S , U C 1 , U C 2 } . The causal diagram induced by M BP is shown in Fig. 2a.

In terms of prediction, the process of obtaining ̂ Y has three steps. First, latent generative features F (gender), S (smiling), and C (high cheekbones) are generated. Then, f X maps the observed generative features { F, S, C } and unobserved generative factors U X to the images X in the pixel levels. Finally, the predictor f ̂ Y takes these pixels as input to estimate ̂ Y in the corresponding model. The functions f X and f ̂ Y can be aggregated as ̂ Y ← f ̂ Y ◦ f X ( F, S, C, U X ) . This illustrates that the prediction of ̂ Y by a BP model is made based on all observed features { F, S, C } and unobserved features U X . ■

Ω′

<!-- image -->

Ω

Ω′

Ω

Figure 3: ( Left ) Ω ′ is causally interpretable if a query can be uniquely computed from the observational data. ( Right ) A query cannot be uniquely computed from the observational data if Ω ′ is not causally interpretable.

Example 2 (Concept-based Model) . The main difference between the class of CP models Ω CP and the class of BP models Ω BP is the form of the classifier f ̂ Y . Consider the same generative process of observed features V 0 = { F, S, C } 1 and the image X in Ex. 1. Let us consider a CP model M CP = ⟨ U = { U F , U S , U C 1 , U C 2 , U X } , {{ F, S, C } , X , ̂ Y } , F CP , P CP ( U ) ⟩ , where the generative process of F, S, C, X is the same as Eq. (1) , ̂ Y is generated as

<!-- formula-not-decoded -->

and P CP ( U ) is equal to P BP ( U ) in Ex. 1. In words, this means that instead of predicting ̂ Y based on pixels (i.e., image X ), the classifier f ̂ Y directly predicts ̂ Y based on observed features F, S, C . The causal diagram induced by M CP is shown in Fig. 2b. ■

Examples 1 and 2 illustrate two different types of predictive models, where the classifier predicts the label directly from the image X (i.e., Ω BP) or from the generative features V (i.e., Ω CP). While both types have showcased their capability to achieve reasonably high predictive accuracy in many domains [5, 7, 10, 11, 14-16, 23, 33, 40], it is unclear at this moment whether we can interpret how they would predict under counterfactual scenarios, such as ' how attractive the person would be had the one been smiling? '. The following notion of causal interpretability formally states whether the counterfactual questions can be answered from the model.

Definition 2 (Causal Interpretability) . Consider a specific model class Ω ′ ⊂ Ω , where Ω is the space of ASCMs. We say the class Ω ′ is causally interpretable w.r.t. a query Q if Q M 1 = Q M 2 for ∀M 1 , M 2 ∈ Ω ′ s.t. P M 1 ( V , X , ̂ Y ) = P M 2 ( V , X , ̂ Y ) .

In words, Ω ′ denotes a certain design choice of the models for predicting the label, that is, it is a space of prediction model candidates (i.e., model class). Ω ′ , for instance, can be Ω BP, when we want to predict the label directly from the image (Fig. 2a), or Ω CP, when the classifier uses all observed features (Fig. 2b). For a query Q , we are concerned with the counterfactual questions such as ' What if the person had smiled? ', which is written in counterfactual notion as P ( ̂ Y S =1 | X = x ) , and more generally as Q ( W ) := P ( ̂ Y W | X ) . 2

In other words, the notion of causal interpretability states whether one can understand the behavior of the model under different counterfactual conditions. If the model is causally interpretable, the counterfactuals can be evaluated from the observational data (Fig. 3, left). Otherwise, the model fundamentally cannot answer the counterfactual question from observational data, and thus, we cannot interpret their behavior under counterfactual scenarios (Fig. 3, right). We now analyze two types of

1 In practice, the annotations of the features are provided in many real-world datasets across various domains, e.g., human face [19], medical images [21], and animal species [36]. Otherwise, the common practice is to extract their annotations with vision-language models [29], which is shown to be effective [23, 39].

2 Note that the definition is general in terms of the query Q , which could vary across different domains, e.g., natural direct effect in fairness analysis [28].

predictive models discussed above (i.e., BP model in Ex. 1 and CP model in Ex. 2) and examine their causal interpretability, i.e., whether they can evaluate counterfactuals from observational data.

Example 3 (Continued from Ex. 1) . Consider the BP model M BP in Ex. 1. Let U X includes another independent variable U S , namely, U X = { U S , U -x } , where U S ⊥ U \ U S ; let the observational quantity P ( F = 0 , S = 1 , C = 1 | X = x ) = 1 , which means that the face is of a male ( F = 0 ), who is smiling ( S = 1 ), and with the cheekbones high ( C = 1 ), given in an image X = x . The generative process of ̂ Y is as ̂ Y ← f ̂ Y ◦ f X ( F, S, C, U X ) = 1 [ S &gt; 0 . 5] .

Consider another BP model M ′ BP with the same generative process of M BP, but for in M ′ BP , the classifier f ′ ̂ Y is given by: ̂ Y ← f ̂ Y ◦ f X ( F, S, C, U X ) = 1 [ U S &gt; 0 . 5] . Since S = U S , the two BP models M BP and M ′ BP agrees with the observational data, i.e., P M BP ( V , X , ̂ Y ) = P M ′ BP ( V , X , ̂ Y ) , which will lead to the same predictions (and corresponding accuracy).

Now, consider the counterfactual quantity "Given the image X = x , would the prediction still be attractive ( ̂ Y = 1 ) had the person not smiled ( S = 0 )?", namely, Q ( S ) = P ( ̂ Y S =0 = 1 | X = x ) . Intuitively, a smaller value of P ( ̂ Y S =0 = 1 | X = x ) implies the model is more reliable since changing a face to non-smiling reduces the attractiveness in general based on common sense knowledge [8]. For the first BP model M BP, Q ( S ) evaluates as P M BP ( ̂ Y S =0 = 1 | X = x ) = 1 [ S = 0 &gt; 0 . 5] = 0 . However, for the second BP model M ′ BP , Q ( S ) evaluates as P M ′ BP ( ̂ Y S =0 = 1 | X = x ) = 1 [ U S = 1 &gt; 0 . 5] = 1 . Details for these derivations are provided in Appendix A.

Note that each BP model evaluates the counterfactual query in a completely different way, and the two models are somewhat inconsistent. In practice, if one chooses the class of BP models Ω BP for this prediction task, the above counterfactual question cannot be answered correctly, since two BP models can give an exact opposite answer even if the two models agree perfectly with the observational distribution and their predictions. In other words, the blackbox model class cannot answer counterfactual Q ( S ) consistently from observational data, and its behavior cannot be interpreted under corresponding counterfactual conditions. ■

One may surmise that Ex. 3 is a pathological case, which for some reason does not allow the evaluation of counterfactual queries in a consistent manner. The next result shows that this is not the case for an arbitrary query Q ( W ) and a latent causal graph G V .

Proposition 1 (Non-interpretability of BP) . For any latent causal graph G V , Ω BP is not causally interpretable w.r.t. Q ( W ) for any W ⊆ V .

Given this impossibility results for the class of blackbox models, one may be tempted to believe that a CP architecture is causally interpretable, as it predicts the label directly from the features where the unobserved factors U X are filtered out. However, the following illustrates that this is not the case.

Example 4 (Continued from Ex. 2) . Consider the CP model M CP in Ex. 2. Similar to Ex. 3, consider an observational quantity P ( F = 0 , S = 1 , C = 1 | X = x ) = 1 . ̂ Y is generated as follows:

<!-- formula-not-decoded -->

Now consider another CP model M ′ CP that is the same as M CP, except for C ← f ′ C ( S, U C 1 ) = ( S ∨ U C 1 ) ∧ U C 2 and P ( U C 1 = 1) = 0 . 5 . We have P M CP ( V , X , ̂ Y ) = P M ′ CP ( V , X , ̂ Y ) and M ′ CP is compatible with the graphical constraints in Fig. 2b. Now consider the same counterfactual quantity P ( ̂ Y S =0 = 1 | X = x ) in Ex. 3. For M CP, we have P M CP ( ̂ Y S =0 = 1 | X = x ) = P M CP ( C S =0 = 1 | F = 0 , S = 1 , C = 1) = 0 . 3 . However, for the second CP model, P M ′ CP ( ̂ Y S =0 = 1 | X = x ) = P M ′ CP ( C S =0 = 1 | F = 0 , S = 1 , C = 1) = 0 . 5 . This implies that the two CP models are also inconsistent w.r.t Q ( S ) . In other words, even prediction using features V , not pixels X , counterfactual queries induced by the CP models can still differ from each other. ■

## 3 A Causal Approach Towards More Interpretable Models

In this section, we establish a principled way of understanding causal interpretability from a graphical point of view and propose a generalized framework for building causally interpretable models.

## 3.1 Generalized Concept-based Models

We first define generalized concept-based prediction (GCP) models, a broader class that predicts the label from an arbitrary set of observed features.

Definition 3 (Generalized Concept-based Prediction) . Let T ⊆ V be a set of features that is used as a predictor of the label. That is, a classifier f ̂ Y makes a prediction on a label based on T . We say such predictive models as generalized concept-based models. A class of GCP models that employ the features T for prediction is denoted as Ω GCP ( T ) := {M∈ Ω | f ̂ Y : D ( T ) →D ( ̂ Y ) } .

Compared to CP models, GCP models employ a selected set of features T ⊆ V as a predictor of the label, which relaxes the requirement of CP where all features are considered.

The selection of the features T in a GCP model should be specified during the model building stage, and our goal is to understand the implications of different choices of T and which ones could lead to causally interpretable models (i.e., satisfying Def. 2). To answer this question systematically, we introduce a graphical criterion for determining whether a model satisfies causal interpretability.

Theorem 1 (Graphical Criterion) . Consider GCP models that employ a set of features T as a predictor of the label. Ω GCP ( T ) is causally interpretable w.r.t. a query Q ( W ) if and only if T ⊆ W ∪ ND ( W ) .

In words, this result says that a query Q ( W ) can be evaluated if the model uses the features among W or non-descendants of W to make a prediction on the label. In other words, the models that use any descendant of W cannot answer counterfactual question and no guarantee can be provided on how they would make predictions under the corresponding counterfactual scenarios. 3

Thm. 1 enables one to identify the architectures (associated with T ) that are causally interpretable with respect to given counterfactual queries. Interestingly, the models that are potentially causally interpretable are not unique. The following formalizes the notion of admissible architectures.

Definition 4 (T-Admissible Set) . We say T is T-admissible w.r.t. W ⋆ = { W 1 , W 2 , · · · } if Ω GCP ( T ) is interpretable w.r.t. Q ( W i ) for all W i ∈ W ⋆ . A set of T-admissible sets w.r.t. W ⋆ is denoted as T-Ad ( W ⋆ ) .

To illustrate, T-admissible set represents model architectures that can answer (potentially multiple) counterfactual queries Q ( W 1 ) , Q ( W 2 ) , · · · . For example, in Fig. 2, eligible models that one can evaluate Q ( S ) is GCP models whose classifier employs { S } , { F } , or { S, F } as a predictor of the label, i.e., T-admissible set corresponds to the query Q ( { S } ) is T-Ad ( { S } ) = {{ S } , { F } , { S, F }} .

Given the multiplicity of admissible models, our goal is to find the models that use as many features as possible to predict the label ̂ Y , i.e., maximal T , as it would be beneficial in terms of predictive accuracy. We denote it as a maximal T-admissible set , which is formally defined below.

Definition 5 (Maximal T-Admissible Set) . Suppose S ∈ T-Ad ( W ⋆ ) and S ′ ̸∈ T-Ad ( W ⋆ ) for any S ′ ⊋ S . We denote such S as Max-T-Ad ( W ⋆ ) .

In other words, a maximal T-admissible set is a T-admissible set that would cease to be T-admissible if any additional variable were added to it. Note that once a set is not T-admissible, adding more variables never makes it T-admissible again by Thm. 1. Identifying a maximal T-admissible set would lead to a model with maximal predictive power while retaining causal interpretability. One might suspect that multiple maximal T-admissible sets could exist, making it unclear which to select to maximize the predictive expressiveness. However, the next result says that this is not the case, since we can establish the uniqueness of the maximal T-admissible set.

Theorem 2 (Uniqueness of Maximal T-Admissible Set) . For the queries Q ( W ⋆ ) , a maximal Tadmissible set is unique and can be written as:

<!-- formula-not-decoded -->

Also, T ∈ T-Ad ( W ⋆ ) if and only if T ⊆ Max-T-Ad ( W ⋆ ) .

3 Note that for the case of X = T , Ω BP is not interpretable w.r.t. any Q ( W ) since X is a descendant of W for any W ⊆ V , generalizing Prop. 1. Similarly, Ω GCP ( T ) is also never interpretable if X ∈ T , i.e., hybrid models that make predictions based on the combination of the image and features.

Figure 4: ( Left ) As we want a model to answer more counterfactual queries ( W 1 ⋆ ⊆ W 2 ⋆ ), the predictive power would decrease (Max-T-Ad ( W 2 ⋆ ) ⊆ Max-T-Ad ( W 1 ⋆ ) ). ( Right ) As the predictive power increases ( T 1 ⊆ T 2 ), interpretable counterfactuals would decrease (W-Ad ( T 2 ) ⊆ W-Ad ( T 1 ) ).

<!-- image -->

To illustrate, for the group of queries Q ( W 1 ) , Q ( W 2 ) , · · · , the maximal T-admissible set is unique and it is the intersection of non-descendants of W i plus W i . Interestingly, identifying a maximal T-admissible set only requires the descendants of W and does not rely on the full specification of the causal graph. For example, given the features {cheekbone, smiling, gender} and the query 'What if the person had smiled?', it only requires the knowledge of descendants of "smiling", which is 'cheekbone'. This does not rely on the full latent causal graph, which is often challenging to obtain.

An important practical implication of Thms 1 and 2 is that, given a query Q ( W ) , one could incorporate additional features as long as they are non-descendants of W , which would help improve accuracy while retaining the causal interpretability w.r.t. P ( ̂ Y W | X ) . For example, given the T-admissible set {smiling, gender} and the query 'Would the person be attractive had they smiled?', one can incorporate additional features, e.g., age or hair color, that are non-descendants of smiling.

So far, we have described how to find causally interpretable models that can answer counterfactual queries. We now describe a practical way of evaluating such queries from the data.

Theorem 3 (Closed Form) . If Ω GCP ( T ) is causally interpretable w.r.t. Q ( W ) , the following holds:

<!-- formula-not-decoded -->

This implies that the counterfactual quantity can be elicited from a two-step prediction - (1) a classifier P ( ̂ Y | T ) and (2) a feature extractor P ( T | X ) . For example, Q ( S ) introduced in Ex. 3 can be computed using observational data and the maximal T-admissible set {S, F} as: P ( ̂ Y S =0 | X ) = ∑ s,f P ( ̂ Y | S = 0 , f ) P ( s, f | X ) . Specifically, { S, F } are extracted from P ( S, F | X ) and the prediction is made by classifying P ( ̂ Y | S = 0 , F ) , conditioning S = 0 . Note that Eq. (5) only holds when the model is causally interpretable, and it does not hold for non-interpretable ones.

## 3.2 Fundamental Trade-Off between Causal Interpretability and Accuracy

So far, we have developed the machinery for building causally interpretable models that can answer counterfactual queries. Now, we discuss which queries can be read from the given predictive model architecture. The following formalizes such notions of admissible queries.

Definition 6 (W-Admissible Set) . We say W is W-admissible w.r.t. T if Ω GCP ( T ) is causally interpretable w.r.t. Q ( W ) . A set of W-admissible sets w.r.t. T is denoted as W-Ad ( T ) .

For example, in Fig. 2b, CP model that uses the features { F, S, C } as the predictor of the label can answer counterfactual queries Q ( { F } ) , Q ( { C } ) , Q ( { F, S } ) , Q ( { F, C } ) and Q ( { F, S, C } ) , i.e., W-Ad ( { F, S, C } ) = {{ F } , { C } , { F, S } , { F, C } , { F, S, C }} by applying Thm. 1. Similarly, in Fig. 2c, we have W-Ad ( { S, C } ) = {{ F } , { S } , { C } , { F, S } , { F, C } , { S, C } , { F, S, C }} . Here, one might notice that the model using a larger set of features can answer a smaller number of counterfactual questions. Our next result establishes a trade-off between accuracy and causal interpretability.

Theorem 4 (Causal Interpretability-Accuracy Trade-Off) . The following holds:

(i) If T 1 ⊆ T 2 , then W-Ad ( T 2 ) ⊆ W-Ad ( T 1 ) . (ii) If W 1 ⋆ ⊆ W 2 ⋆ , then Max-T-Ad ( W 2 ⋆ ) ⊆ Max-T-Ad ( W 1 ⋆ ) .

Figure 5: (a) Example images of BarMNIST dataset. (b) Causal diagram of GCP models. Red arrows represent the possible usage for predicting the label. (c) Causal interpretability-accuracy trade-off.

<!-- image -->

Figure 6: Estimation of counterfactual queries. Blue dots and orange marks denote estimation of counterfactual queries and ground truth value, respectively.

<!-- image -->

In other words, Thm. 4-(i) states that the counterfactuals that can be evaluated from the model decrease (W-Ad ( T 2 ) ⊆ W-Ad ( T 1 ) ) as the predictors increase ( T 1 ⊆ T 2 ). Similarly, Thm. 4-(ii) states that the predictive power would decrease (Max-T-Ad ( W 2 ⋆ ) ⊆ Max-T-Ad ( W 1 ⋆ ) ) as we want the models to answer more counterfactual queries ( W 1 ⋆ ⊆ W 2 ⋆ ). This reveals a fundamental trade-off between causal interpretability and accuracy, where better predictive power would compromise the interpretability, and vice versa, as illustrated in Fig. 4.

## 4 Experiments

In this section, we evaluate our framework for estimating counterfactuals and compare it with prior approaches. Experimental details and additional experimental results are provided in Appendix B.

## 4.1 Synthetic datasets

We design the BarMNIST dataset [17, 24] where the digits are colored and a bar appears at the top of the image, as shown in Fig. 5a. Specifically, we consider the features 'bar' ( B ), 'digit' ( D ), and 'color' ( C ), where D,C are correlated and D has a direct causal effect on B , as illustrated in Fig. 5b. The true label is generated from all of the features and unobserved factors.

The dataset allows us to compare the estimation of counterfactuals from each model with the groundtruth. We trained 4 different models, each using T = { B,D,C } , { B,D } , { D,C } , and { D } as the predictor of the label. As shown in Fig. 5c, the model using T = { B,D,C } achieves the best accuracy, followed by T = { B,D } and T = { D,C } , and the model using T = { D } shows the lowest accuracy. On the other hand, the best model ( T = { B,D,C } ) in terms of accuracy shows a high estimation error on the counterfactual query of changing the digit. Thm. 1 suggests that any estimation using observed data cannot capture the true counterfactual prediction of this model, since it uses B , which is the descendant of D . For the same reason, T = { B,D } is not causally interpretable, in contrast to T = { D,C } and T = { D } . Our theory (Thm. 2) also suggests that there exists a unique maximal set of features that maintains causal interpretability, in this case, T = { D,C } .

In Fig. 6, we take a closer look at how these models estimate counterfactuals. As shown in Fig. 6a, T = { D,C } and T = { D } are admissible models for the counterfactual query of changing the digit. On the other hand, for changing color (Fig. 6b), all models are admissible and output a correct estimate of the counterfactual query, since C is not a descendant of any other features.

Figure 7: Visualization of interpreting counterfactual predictions on CelebA examples.

<!-- image -->

## 4.2 Real-world datasets

CelebA dataset [19] contains human face images with the annotations on facial expressions and attributes, such as 'smiling', 'age', 'gender', etc. We consider a model predicting the label 'attractiveness' and examine how a model makes a prediction under counterfactual conditions 'Would the person look attractive had they smiled?'. In the real world, it is impossible to observe a counterfactual outcome, but our theory allows us to interpret the behavior of (causally interpretable) models under counterfactual conditions. Based on Thm. 1, we choose the features that are not the descendants of smiling. Fig. 7 illustrates the counterfactual prediction of the model using non-descendant features (i.e., 'smiling' and 'gender'). We can interpret its behavior under the counterfactual condition that it predicts a higher attractiveness had the one smiled, which is aligned with human common sense.

## 5 Conclusion

In this work, we introduced the notion of causal interpretability, which states whether counterfactual queries can be evaluated from a model and observational data. By examining commonly used model classes - blackbox and concept-based models - we demonstrated that neither is causally interpretable. To this end, we developed a graphical criterion that determines whether the model is causally interpretable with respect to the query (Thm. 1). We characterize the unique maximal set of features yielding interpretable architecture (Thm. 2) and provide a practical way of evaluating such queries from the data (Thm. 3). Our results reveal a fundamental tradeoff between the causal interpretability and predictive accuracy (Thm. 4). Theoretical findings are corroborated by the experimental results. Additional discussions and limitations are provided in Appendix C.

## Acknowledgments and Disclosure of Funding

We thank anonymous reviewers for their constructive comments. This research is supported in part by the NSF, ONR, AFOSR, DoE, Amazon, JP Morgan, and The Alfred P. Sloan Foundation.

## References

- [1] Elias Bareinboim, Juan D. Correa, Duligur Ibeling, and Thomas Icard. On pearl's hierarchy and the foundations of causal inference. In Probabilistic and Causal Inference: The Works of Judea Pearl , page 507-556. Association for Computing Machinery, New York, NY, USA, 2022.
- [2] Juan Correa, Sanghack Lee, and Elias Bareinboim. Nested counterfactual identification from arbitrary surrogate experiments. Advances in Neural Information Processing Systems , 34: 6856-6867, 2021.
- [3] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 , 2018.
- [4] Gabriele Dominici, Pietro Barbiero, Mateo Espinosa Zarlenga, Alberto Termine, Martin Gjoreski, Giuseppe Marra, and Marc Langheinrich. Causal concept graph models: Beyond causal opacity in deep learning. arXiv preprint arXiv:2405.16507 , 2024.

- [5] Gabriele Dominici, Pietro Barbiero, Francesco Giannini, Martin Gjoreski, Giuseppe Marra, and Marc Langheinrich. Counterfactual concept bottleneck models. In The Thirteenth International Conference on Learning Representations , 2025.
- [6] Mateo Espinosa Zarlenga, Pietro Barbiero, Gabriele Ciravegna, Giuseppe Marra, Francesco Giannini, Michelangelo Diligenti, Zohreh Shams, Frederic Precioso, Stefano Melacci, Adrian Weller, et al. Concept embedding models: Beyond the accuracy-explainability trade-off. Advances in Neural Information Processing Systems , 35:21400-21413, 2022.
- [7] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [8] Simone Horn, Natalia Matuszewska, Nikolaos Gkantidis, Carlalberta Verna, and Georgios Kanavakis. Smile dimensions affect self-perceived smile attractiveness. Scientific reports , 11 (1):2779, 2021.
- [9] Inwoo Hwang, Yesong Choe, Yeahoon Kwon, and Sanghack Lee. On positivity condition for causal inference. In International Conference on Machine Learning , pages 20818-20841. PMLR, 2024.
- [10] Aya Abdelsalam Ismail, Tuomas Oikarinen, Amy Wang, Julius Adebayo, Samuel Don Stanton, Hector Corrada Bravo, Kyunghyun Cho, and Nathan C. Frey. Concept bottleneck language models for protein design. In The Thirteenth International Conference on Learning Representations , 2025.
- [11] Sujin Jeon, Hyundo Lee, Eungseo Kim, Sanghack Lee, Byoung-Tak Zhang, and Inwoo Hwang. Locality-aware concept bottleneck model. arXiv preprint arXiv:2508.14562 , 2025.
- [12] Yonghan Jung, Jin Tian, and Elias Bareinboim. Estimating identifiable causal effects through double machine learning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 12113-12122, 2021.
- [13] Yonghan Jung, Iván Díaz, Jin Tian, and Elias Bareinboim. Estimating causal effects identifiable from a combination of observations and experiments. Advances in Neural Information Processing Systems , 36:46446-46490, 2023.
- [14] Eunji Kim, Dahuin Jung, Sangha Park, Siwon Kim, and Sungroh Yoon. Probabilistic concept bottleneck models. In International Conference on Machine Learning , pages 16521-16540. PMLR, 2023.
- [15] Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In International conference on machine learning , pages 5338-5348. PMLR, 2020.
- [16] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems , 25, 2012.
- [17] Yann LeCun. The mnist database of handwritten digits. http://yann. lecun. com/exdb/mnist/ , 1998.
- [18] Adam Li, Yushu Pan, and Elias Bareinboim. Disentangled representation learning in nonmarkovian causal systems. Advances in Neural Information Processing Systems , 37:104843104903, 2024.
- [19] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Large-scale celebfaces attributes (celeba) dataset. Retrieved August , 15(2018):11, 2018.
- [20] Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Advances in neural information processing systems , 30, 2017.
- [21] Michael Nevitt, David Felson, and Gayle Lester. The osteoarthritis initiative. Protocol for the cohort study , 1:2, 2006.

- [22] Andrew Ng et al. Sparse autoencoder. CS294A Lecture notes , 72(2011):1-19, 2011.
- [23] Tuomas Oikarinen, Subhro Das, Lam M. Nguyen, and Tsui-Wei Weng. Label-free concept bottleneck models. In The Eleventh International Conference on Learning Representations , 2023.
- [24] Yushu Pan and Elias Bareinboim. Counterfactual image editing. In International Conference on Machine Learning , pages 39087-39101. PMLR, 2024.
- [25] Judea Pearl. Causal diagrams for empirical research. Biometrika , 82(4):669-688, 1995.
- [26] Judea Pearl. Causality . Cambridge university press, 2009.
- [27] Judea Pearl and Dana Mackenzie. The book of why: the new science of cause and effect . Basic books, 2018.
- [28] Drago Plecko and Elias Bareinboim. Causal fairness analysis. arXiv preprint arXiv:2207.11385 , 2022.
- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pages 8748-8763. PmLR, 2021.
- [30] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. " why should i trust you?" explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining , pages 1135-1144, 2016.
- [31] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision , pages 618-626, 2017.
- [32] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. Nature , 529(7587):484-489, 2016.
- [33] Chung-En Sun, Tuomas Oikarinen, Berk Ustun, and Tsui-Wei Weng. Concept bottleneck large language models. In The Thirteenth International Conference on Learning Representations , 2025.
- [34] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction , volume 1. MIT press Cambridge, 1998.
- [35] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems , volume 30, pages 5998-6008, 2017.
- [36] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.
- [37] Alvin Wan, Lisa Dunlap, Daniel Ho, Jihan Yin, Scott Lee, Henry Jin, Suzanne Petryk, Sarah Adel Bargal, and Joseph E Gonzalez. Nbdt: Neural-backed decision trees. arXiv preprint arXiv:2004.00221 , 2020.
- [38] Eric Wong, Shibani Santurkar, and Aleksander Madry. Leveraging sparse linear layers for debuggable deep networks. In International Conference on Machine Learning , pages 1120511216. PMLR, 2021.
- [39] Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, and Mark Yatskar. Language in a bottle: Language model guided concept bottlenecks for interpretable image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 19187-19197, 2023.
- [40] Mert Yuksekgonul, Maggie Wang, and James Zou. Post-hoc concept bottleneck models. In The Eleventh International Conference on Learning Representations , 2023.

## Appendix

| A Proofs and Additional Examples                       | A Proofs and Additional Examples                       |   13 |
|--------------------------------------------------------|--------------------------------------------------------|------|
| A.1                                                    | Derivations in Examples . . . .                        |   13 |
| A.2                                                    | Omitted Proofs . . . . . . . . .                       |   14 |
| A.3                                                    | Additional Examples . . . . . .                        |   17 |
| Experiments                                            | Experiments                                            |   18 |
| B.1                                                    | Dataset . . . . . . . . . . . . .                      |   18 |
| B.2                                                    | Experimental Details . . . . . .                       |   19 |
| B.3                                                    | Additional Experimental Results                        |   19 |
| C Additional Discussions, Limitations, and Future Work | C Additional Discussions, Limitations, and Future Work |   21 |

## A Proofs and Additional Examples

## A.1 Derivations in Examples

## A.1.1 Derivation in Ex. 3

In Ex. 3, for the first BP model M BP, we evaluate Q ( S ) from M BP as follows:

<!-- formula-not-decoded -->

However, for the second BP model, we evaluate Q ( S ) from M ′ BP as:

<!-- formula-not-decoded -->

## A.1.2 Derivation in Ex. 4

In Ex. 4, for M CP,

<!-- formula-not-decoded -->

## A.2 Omitted Proofs

In this section, we present the proofs of our theoretical results in Sec. 2 and 3. We first formally introduce the causal diagram induced by an SCM.

Definition 7 (Causal Diagram [1, Def. 13]) . Consider an SCM M = ⟨ U , V , F , P ( U ) ⟩ . We construct a graph G using M as follows:

- (1) add a vertex for every variable in V ,
- (2) add a directed edge ( V j → V i ) for every V i , V j ∈ V if V j appears as an argument of f V i ∈ F ,
- (3) add a bidirected edge ( V j ←→ V i ) for every V i , V j ∈ V if the corresponding U V i , U V j ⊆ U are not independent or if f V i and f V j share some U ∈ U as an argument.

We refer to G as the causal diagram induced by M (or 'causal diagram of M ' for short).

■

We then formally introduce the identifiability of a counterfactual query given an observational distribution and a causal diagram G .

Definition 8 (Counterfactual Identification) . A counterfactual query P ( y 1[ x 1 ] , y 2[ x 2 ] , ... ) is said to be identifiable from P ( V ) and G , if P ( y 1[ x 1 ] , y 2[ x 2 ] , ... ) is uniquely computable from the distributions P ( V ) in any SCM that induces G .

Then we start from two lemmas as a tool for the proof of Thm. 1.

Lemma 1. Consider an SCM M over V . Suppose that there exists a path made entirely of bi-directed edges between V i , V j ∈ V in G . Consider two sets A , B ⊆ V and A ∩ B = ∅ . Let the intervened values are not consistent with the factual values, namely, b ̸⊆ v . Then the query P ( a b | v ) is identifiable from P ( V ) and G if and only if A ⊆ ND ( B ) , where ND ( B ) = ∩ B i ∈ B ND ( B i ) .

Proof. ( ⇒ ) Suppose A ⊆ ND ( B ) . We have P ( a b | v ) = P ( a | v ) = 1 [ a = v ] which implies that P ( a b | v ) is uniquely computable.

( ⇐ ) Suppose there exists A ∈ A such that A ∈ Des ( B ) . By Correa et al. [2, Thm. 3], P ( a b | v ) is an inconsistent factor since B ⊆ V and b ⊆ v , and thus, it is not identifiable from P ( V ) .

Lemma 2 (Correa et al. [2, Lemma. 1]) . Consider an SCM over V induce observational distribution P ( V ) and diagram G . Suppose A 2 takes input as A 1 . Then ∑ a 1 P ( A 1[ b 1 ] , A 2[ b 2 ] , ... ) is identifiable if and only if P ( A 1[ b 1 ] , A 2[ b 2 ] , ... ) is identifiable.

Now, we are ready to proceed to the proof of Thm. 1.

Theorem 1 (Graphical Criterion) . Consider GCP models that employ a set of features T as a predictor of the label. Ω GCP ( T ) is causally interpretable w.r.t. a query Q ( W ) if and only if T ⊆ W ∪ ND ( W ) .

<!-- image -->

̂

Figure 9: Diagrams used in the proof of Prop. 1.

Proof. According to Defs 2, 3 and 8, this is equivalent to prove that query P ( ̂ y w ′ | x ) is identifiable iff T ⊆ ND ( W ) ∪ W given the observational distribution P ( V , X , ̂ Y ) and the diagram G Aug over { V , X , ̂ Y } (shown in Fig. 8). To illustrate, the diagram G over V is an arbitrary given DAG; for any V i ∈ V , V i point to X and bi-directed connected to X ; only a subset T ⊆ V point to ̂ Y . Denote Z = T \ W .

<!-- formula-not-decoded -->

Eq. (6) holds since ̂ Y w are independent with X and V since all parents of ̂ Y w ′ (which is T w ′ ) are conditioned on. Eq. (7) holds since the T ∩ W should be consistent with the intervened value in w (and the remaining variables Z in T taking z ′′ . Eq. (8) holds due to ̂ Y ⊥ W | T in G W , where G W is the graph removing outgoing edge of W . Using do-calculus, we have:

<!-- formula-not-decoded -->

Wewill prove that Eq. (8) is identifiable if and only if T ⊆ ND ( W ) ∪ W , which is equivalent to prove Eq. (8) is identifiable iff Z ⊆ ND ( W ) since Z = T \ W . According to Eq. (8), the only undermined term is P ( z ′′ w ′ | v , x ) . Since V and X are bi-directly connected, Lemma 1 suggests P ( z ′′ w ′ | v , x ) is identifiable iff Z ⊆ ND ( W ) . Then, P ( ̂ y | z ′′ , w ′ ) P ( z ′′ w ′ | v , x ) P ( v | x ) is identifiable iff Z ⊆ ND ( W ) . According to Lemma 2, Eq. (8) is identifiable iff T ⊆ ND ( W ) ∪ W .

Proposition 1 (Non-interpretability of BP) . For any latent causal graph G V , Ω BP is not causally interpretable w.r.t. Q ( W ) for any W ⊆ V .

Proof. Since the observational P ( X ) is identifiable, we will prove that P ( ̂ y w ′ , x ) is not identifiable given a blackbox model structure and observational distribution P ( X , ̂ Y , V ) .

<!-- formula-not-decoded -->

P ( x ′ w ′ , x , w ) is not identifiable according to Lemma 1. Then Lemma 2 suggests that P ( ̂ y w ′ , x ) is not identifiable.

Theorem 2 (Uniqueness of Maximal T-Admissible Set) . For the queries Q ( W ⋆ ) , a maximal Tadmissible set is unique and can be written as:

<!-- formula-not-decoded -->

Also, T ∈ T-Ad ( W ⋆ ) if and only if T ⊆ Max-T-Ad ( W ⋆ ) .

Proof. (i) First, we will show that S := ∩ W i ∈ W ⋆ ( W i ∪ ND ( W i ) ) is a T-admissible set w.r.t Q ( W ⋆ ) . For each W i ∈ W ⋆ , we have

<!-- formula-not-decoded -->

Therefore, by Thm. 1, ∩ W i ∈ W ⋆ ( W i ∪ ND ( W i ) ) is a T-admissible set w.r.t Q ( W i ) for all W i ∈ W ⋆ . Thus, we have S ∈ T-Ad ( W ⋆ ) .

(ii) Now, we will show that S is a maximal T-admissible set w.r.t W ⋆ . Suppose there exists S ′ such that S ′ ∈ T-Ad ( W ⋆ ) and S ′ ⊋ S . Since S ′ ∈ T-Ad ( W ⋆ ) , S ′ ∈ T-Ad ( W i ) for all W i ∈ W ⋆ . Hence,

<!-- formula-not-decoded -->

Therefore, S ′ ⊆ ∩ W i ∈ W ⋆ ( W i ∪ ND ( W i ) ) = S , which contradicts S ′ ⊋ S . Therefore, S is a maximal T-admissible set w.r.t W ⋆ .

(iii) Now, we will show that S is a unique maximal T-admissible set. Suppose there exists another maximal T-admissible set S ′ . Since S ′ ∈ T-Ad ( W ⋆ ) , we have S ′ ⊆ S by the same reason in (ii). If S ′ ⊊ S , then it contradicts that S ′ is a maximal T-admissible set, since S is a T-admissible set. Therefore, we have S = S ′ . In other words, a maximal T-admissible set is unique and can be written as Max-T-Ad ( W ⋆ ) = ∩ W i ∈ W ⋆ ( W i ∪ ND ( W i ) ) .

(iv) Now, we will show that T ∈ T-Ad ( W ⋆ ) if and only if T ⊆ Max-T-Ad ( W ⋆ ) . Suppose T ∈ T-Ad ( W ⋆ ) . Then, by (ii), we have T ⊆ ∩ W i ∈ W ⋆ ( W i ∪ ND ( W i ) ) . Also, we showed that Max-T-Ad ( W ⋆ ) = ∩ W i ∈ W ⋆ ( W i ∪ ND ( W i ) ) . Therefore, we have T ⊆ Max-T-Ad ( W ⋆ ) . Now, suppose that T ⊆ Max-T-Ad ( W ⋆ ) . We have T ⊆ ∩ W i ∈ W ⋆ ( W i ∪ ND ( W i ) ) , and thus, T ⊆ W i ∪ ND ( W i ) for all W i ∈ W ⋆ . Therefore, T ∈ T-Ad ( W i ) for all W i ∈ W ⋆ , and thus, T ∈ T-Ad ( W ⋆ ) .

Theorem 3 (Closed Form) . If Ω GCP ( T ) is causally interpretable w.r.t. Q ( W ) , the following holds:

<!-- formula-not-decoded -->

Proof. From Eq. (8), we have

<!-- formula-not-decoded -->

Note that this equation is identifiable if only if Z ⊆ W ∪ ND ( W ) . Then

<!-- formula-not-decoded -->

This conclude P ( ̂ Y w | x ) = ∑ t P ( ̂ Y | w ′ ∩ T , t \ W ) P ( t | x ) since Eq. 12 holds for any t , w .

Theorem 4 (Causal Interpretability-Accuracy Trade-Off) . The following holds: (i) If T 1 ⊆ T 2 , then W-Ad ( T 2 ) ⊆ W-Ad ( T 1 ) . (ii) If W 1 ⋆ ⊆ W 2 ⋆ , then Max-T-Ad ( W 2 ⋆ ) ⊆ Max-T-Ad ( W 1 ⋆ ) .

Proof. (i) Let T 1 ⊆ T 2 . Suppose W ∈ W-Ad ( T 2 ) . By Def. 6 and Thm. 1, we have

<!-- formula-not-decoded -->

Since T 1 ⊆ T 2 , it follows that T 1 ⊆ W ∪ ND ( W ) . Therefore, by Def. 6 and Thm. 1, W ∈ W-Ad ( T 1 ) . Thus, for all W ∈ W-Ad ( T 2 ) , we have W ∈ W-Ad ( T 1 ) . Hence, we have

<!-- formula-not-decoded -->

(ii) Let W 1 ⋆ ⊆ W 2 ⋆ . Then, we have

<!-- formula-not-decoded -->

Therefore, we have Max-T-Ad ( W 2 ⋆ ) ⊆ Max-T-Ad ( W 1 ⋆ ) by Thm. 2.

## A.3 Additional Examples

The following example illustrates how GCP and CP models compare.

Example 5 (GCP) . Consider the generative process of observed concepts V 0 = { F, S, C } and the image X , as in Ex. 1 (BP model) and Ex. 2 (CP model). Consider a GCP model M GCP = ⟨ U = { U F , U S , U C 1 , U C 2 , U X } , {{ F, S, C } , X , ̂ Y } , F GCP , P GCP ( U ) ⟩ , where

<!-- formula-not-decoded -->

and P GCP ( U ) is equal to P CP ( U ) in Ex. 2. The causal diagram induced by GCP model M GCP is shown in Fig. 2c. To illustrate, instead of predicting the label based on pixels in images X (BP models) or all observed features { F, S, C } (CP models), GCP model makes a prediction using a selected subset of features T = { S, F } (i.e., smiling and gender) in this case. ■

The following example illustrates the case where the GCP model is causal interpretable.

Example 6 (Continued from Ex. 5) . Consider Ω CP in Ex. 4. Thm. 1 suggests Ω CP is not interpretable w.r.t. to query Q ( S ) P ( Y S =0 | X ) . This is because C ∈ De ( S ) , where

|   F |   S |   C |   P ( F,S,C ) = 1 |
|-----|-----|-----|-------------------|
|   0 |   0 |   0 |             0.168 |
|   0 |   0 |   1 |             0.072 |
|   0 |   1 |   0 |             0.096 |
|   0 |   1 |   1 |             0.144 |
|   1 |   0 |   0 |             0.112 |
|   1 |   0 |   1 |             0.048 |
|   1 |   1 |   0 |             0.144 |
|   1 |   1 |   1 |             0.216 |

W = { S } , i.e., the prediction of ̂ Y is made based on C , a descendant of S . In contrast, Ω GCP ( { S,F } ) in Ex. 5 is said to be causally interpretable w.r.t. to query P ( Y S =0 | X ) since f GCP ̂ Y only takes T = { S, F } ⊆ S ∪ ND ( S ) as input. To illustrate, let us consider the GCP model M GCP in Ex. 5. Similar to Examples 3 and 4, let the observational quantity P ( F = 0 , S = 1 , C = 1 | X = x ) = 1 and let f ̂ Y be:

<!-- formula-not-decoded -->

Now, consider another GCP model

M ′ GCP = ⟨ U ′ = { U ′ F , U ′ S 1 , U ′ S 1 , U ′ C 1 , U ′ C 2 , U ′ X } , {{ F, S, C } , X , ̂ Y } , F GCP ′ , P GCP ′ ( U ) ⟩ , (15) where

<!-- formula-not-decoded -->

and P ( U ′ F = 1) = 0 . 52 , P ( U ′ S 1 = 1) = 0 . 5 , P ( U ′ S 2 = 1) = 9 / 13 , P ( U ′ C 1 = 1) = 0 . 5 , P ( U ′ C 2 = 1) = 0 . 6 . It is verifiable that P M GCP ( V ) = P M ′ GCP ( V ) as shown in Table 1. Since f ̂ Y is the same in both M GCP and M ′ GCP , P M GCP ( V , ̂ Y ) = P M ′ GCP ( V , ̂ Y ) . Let the distribution of U X satisifies that P M GCP ( V , X , ̂ Y ) = P M ′ GCP ( V , X , ̂ Y ) . M ′ CP is compatible the graphical constraints induced by the model in Fig. 2b. Notice that f ′ F , f ′ S , f ′ C in M ′ GCP are totally different to f F , f S , f C in M GCP. For the first GCP model M GCP,

<!-- formula-not-decoded -->

Similarly, for the second GCP model M ′ GCP ,

<!-- formula-not-decoded -->

This shows that the two GCP models are consistent with the query. In other words, if one uses the features { S, F } to predict ̂ Y , the model architecture in Fig. 2c is guaranteed to provide a unique answer for the counterfactual question "What would the attractiveness prediction be had the person not smiled?" (i.e., P ( Y S =0 | X ) ). Then one can trust the counterfactual quantities induced by any model with this architecture. ■

## B Experiments

In this section, we describe the details for the experiments and provide additional experimental results.

## B.1 Dataset

## B.1.1 BarMNIST

For BarMNIST experiment discussed in Sec. 4.1, the data generating process is as follows:

<!-- formula-not-decoded -->

<!-- image -->

̂

(a) Causal diagram.

(b) Estimation of counterfactuals.

Figure 10: (a) Causal diagram of GCP models. Red arrows represent the possible usage for predicting the label. (b) Estimation of counterfactual queries. Blue dots and orange marks denote estimation of counterfactual queries and ground truth value, respectively.

the exogenous variables U D , U C , U B 1 , U B 2 , U B 3 , U Y are independent binary variables, and P ( U D = 1) = 0 . 5 , P ( U C = 1) = 0 . 4 , P ( U B 1 = 1) = 0 . 9 , P ( U B 2 = 1) = 1 / 18 , P ( U B 3 = 1) = 0 . 5 , P ( U Y = 1) = 0 . 1 .

Following this process, we generated 60,000 images and corresponding labels, where each image is annotated with 3 binary features, i.e., bar ( B ), color ( C ), and digit ( D ). Here, D = 0 represents the digits from 0 to 4 and D = 1 represents the digits from 5 to 9.

## B.1.2 CelebA

CelebA dataset [19] contains 202,599 celebrity facial images, where each image is annotated with 40 different attributes. In our experiments, we used the attribute 'attractiveness' as the label, where the label and all other features are binary.

## B.2 Experimental Details

In BarMNIST, we used ResNet18 for the feature extractor. For the classifier, we used a three-layer MLP with the hidden dimension of 32 and leakyrelu activation. We set the batch size to 1024 and trained the models for 100 epoch. We used Adam optimizer with a learning rate of 0.0003.

In CelebA, we used ResNet34 for the feature extractor and used linear classifier. We set the batch size to 512 and trained the models for 100 epochs. We used SGD optimizer with the learning rate of 0.001. We resized the image with center crop into 64 × 64 for training.

For the training of our model and baselines, we used binary classification loss for both the feature extractor and the classifier, where they are trained simultaneously in an end-to-end manner. All experimental results are averaged over 5 independent runs. We report a standard error as the error bar in Figs. 6, 10 and 11. All experiments are conducted on a single NVIDIA A100 GPU. For the implementation, we utilized publicly available code from Espinosa Zarlenga et al. [6]. We used GPT-4o to generate the counterfactual images shown in Figs. 7 and 11 to provide an intuitive understanding of the counterfactual questions.

## B.3 Additional Experimental Results

## B.3.1 BarMNIST

To validate our theory with a different graph structure, we consider a causal diagram in Fig. 10a where the goal is to predict the digit D from the image. The data generating process is as follows:

<!-- formula-not-decoded -->

where the exogenous variables U B , U C 1 , U C 2 , U D are independent binary variables, where P ( U B = 1) = 0 . 6 , P ( U C 1 = 1) = 0 . 5 , P ( U C 2 = 1) = 0 . 1 , P ( U D = 1) = 0 . 1 .

̂

Figure 11: (a) We examine the prediction of the models under counterfactual condition. We use causal prior knowledge that smiling has causal effects on the features 'cheekbones' and 'opened mouth'. (b) Average difference between the estimated counterfactual prediction and the prediction on the observed (factual) image. (c) Qualitative examples for our model and baselines.

<!-- image -->

The baseline model uses the features B and C for predicting the label, and our model uses B for making a prediction. Our theory (Thm. 1) suggests that our model is causally interpretable, but not the baseline which uses C , a descendant of B . We compare our model and baselines for estimating the counterfactual prediction of the model, where the query is to change the bar, i.e., P ( ̂ D B =0 | x ) .

Fig. 10b illustrates the estimation of counterfactual queries (blue dots) and ground truth values (orange marks). This shows that our model correctly estimates counterfactual queries. In contrast, the estimation of the baseline significantly differs from the ground truth. This corroborates our theory that our estimation can properly interpret the counterfactual behavior of the causally interpretable models, but it is not possible for non-interpretable ones.

## B.3.2 CelebA

Here, we provide a detailed analysis of CelebA experiments in Sec. 4.2. Fig. 11-(a) illustrates the counterfactual question and causal prior we utilized to construct our model. Specifically, we leverage the common-sense knowledge that smiling has direct causal influence to the features 'cheekbones' and 'opened mouth'. To construct our model, we choose features that are non-descendants of smiling, specifically 'smile' and 'gender' as feature set V . Baselines include descendant features. In Fig. 11, baseline 1 uses the features 'smiling', 'gender', and 'cheekbones' and baseline 2 uses the features 'smiling', 'gender', 'cheekbones', and 'opened mouth'.

Fig. 11-(b) shows the average difference between the estimated counterfactual prediction and the prediction on the observed image. Fig. 11-(c) shows qualitative examples comparing our method and baselines. The first column in Fig. 11-(c) shows the input image, and the second column illustrates the counterfactual image, as a reference to provide a better understanding of the counterfactual query.

The theory suggests that a causally interpretable model can properly estimate its prediction under counterfactual conditions. As shown in Fig. 11-(b) and (c), our model, which is causally interpretable, consistently increases the attractiveness across the instances, which is also aligned with human reasoning. In contrast, as illustrated in Fig. 11-(c), the estimation of the baselines (which use similar feature set as ours) shows that smiling often does not increase attractiveness (red boxes). In fact, our theory suggests that it is not possible to interpret the counterfactual behavior of non-interpretable models using only observational data, and any attempts to estimate it would lead to inconsistent results.

## C Additional Discussions, Limitations, and Future Work

Estimation of the concepts. In the closed-form formula in Eq. (5), the concepts W and T are ground-truth concepts. Since the labels of ground-truth concepts are available, one can estimate P ( T | X ) over ground truth concept T . For clarity, let us denote this estimated distribution as ̂ P ( T | X ) . In the prediction stage, the true concepts W and T of an image instance X are not given directly. Instead, the predicted concepts W and T are sampled through the estimated ̂ P ( T | X ) . When ̂ P ( T | X ) is accurate, the sampled (predicted) concepts are expected to align closely with the ground-truth concepts. However, if the estimation has an error, the predicted concepts may deviate from the true ones, and this error will naturally propagate into the counterfactual evaluation via Eq. (5).

Our goal with this formulation is to formally characterize how these counterfactual quantities can be computed from the observational distribution under ideal conditions (i.e., accurate estimation). The challenge of robustly estimating P ( T | X ) from finite data is indeed fundamental and highly relevant to practice, but falls outside the scope of this work. Nevertheless, it would be a valuable direction for future investigation, particularly in light of ongoing research in counterfactual estimation within the causal inference literature [12, 13] and the importance of creating more interpretable methods in practice.

Causal graph. Our work reveals that understanding and harnessing causal relationships among the generative features are crucial for building interpretable models that can properly evaluate counterfactual questions. It is important to note that our framework only requires the causal prior on the descendants, and this is a much relaxed assumption compared to the conventional assumption in causal inference, where the full specification of the causal graph is needed [9, 18].

Real-world datasets. In real-world datasets, it is infeasible to evaluate the actual value of the counterfactual query because the underlying ground-truth data-generating process for real-world datasets is not given, specifically, the mechanisms of V are not known. For example, it is unknown how nature decides the generation process of human facial features. Due to this inevitable restriction, we thoroughly validated our theory in BarMNIST datasets (where we have the ground-truth SCM), including causal interpretability-accuracy tradeoff.

Still, our theory allows us to understand the interplay between causal interpretability and accuracy in real-world datasets. For example, given T-admissible set smiling, gender and the query 'Would the person be attractive had they smiled?', if one wants to incorporate additional query 'Would the person be attractive had they be a men?', we know the model using this T maintains causal interpretability w.r.t. both queries, and thus, would not compromise accuracy.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state our main claims. Theoretical results are presented in Sec. 2 and 3 and experimental results are presented in Sec. 4.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitation section is provided in Appendix C.

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

Justification: Proofs are provided in Appendix A.

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

Justification: Experimental details are provided in Appendix B.

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

Justification: We provide the experimental details in Appendix B.

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

Justification: Experimental details are provided in Appendix B.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Fig. 6 includes error bars.

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

Justification: Experimental details are provided in Appendix B.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We reviewed the code of ethics and followed it.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss societal impacts in Appendix C.

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

Justification: Our work poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly cited the original paper of the dataset or code.

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

Justification: We do not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our work does not involve human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our work does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development does not involve LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.