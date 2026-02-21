## Is the acquisition worth the cost? Surrogate losses for Consistent Two-stage Classifiers

## Florence Regol

Block, Toronto, Canada and McGill University, ILLS, Mila florence.robert-regol@mail.mcgill.ca

Theodore Glavas

McGill University, ILLS, Mila

Mark Coates

McGill University, ILLS, Mila

Joseph Cotnareanu

McGill University, ILLS, Mila

## Abstract

Recent years have witnessed the emergence of a spectrum of foundation models, covering a broad range of capabilities and costs. Often, we effectively use foundation models as feature generators and train classifiers that use the outputs of these models to make decisions. In this paper, we consider an increasingly relevant setting where we have two classifier stages. The first stage has access to features x and has the option to make a classification decision or defer, while incurring a cost, to a second classifier that has access to features x and z . This is similar to the 'learning to defer' setting, with the important difference that we train both classifiers jointly, and the second classifier has access to more information. The natural loss for this setting is an ℓ 01 c loss, where a penalty is paid for incorrect classification, as in ℓ 01 , but an additional penalty c is paid for consulting the second classifier. The ℓ 01 c loss is unwieldy for training. Our primary contribution in this paper is the derivation of a hinge-based surrogate loss ℓ c hinge that is much more amenable to training but also satisfies the property that ℓ c hinge -consistency implies ℓ 01 c -consistency.

## 1 Introduction

With the emergence of a spectrum of foundation models, covering a broad range of capabilities and costs, we are increasingly faced with a decision as to which model to use. For example, can we make a decision locally, on an edge device, or should we incur the additional communication and computational cost of sending the query to a more powerful remote model? In many cases, we use the pre-trained foundation model essentially as a feature generator, and strive to train a classifier that uses the output of the foundation model as its input. In this setting, we then face a task of training two classifiers, while simultaneously learning when to defer to the more powerful model.

One approach to solve this problem is to 1) train the more powerful classifier first; and 2) train the decision module with the smaller classifier afterwards (either jointly or separately). This strategy has proven successful and benefits from strong theoretical foundations [Keswani et al., 2021, Wilder et al., 2021, Verma et al., 2022, Mao et al., 2023, 2024b], but intuitively, this appears inefficient. Indeed, with this approach, both classifiers expend effort exploring regions of the input space where their predictions will ultimately not be used. Because of this, it is important to consider and formalize the problem where the classifiers and the module deciding which model to use are trained jointly.

While there has been a significant body of work establishing consistent losses for the related problem of classification with learning to defer to experts or with reject [Verma et al., 2022, Herbei and Wegkamp, 2006], these losses do not cover the case where we jointly train multiple inference

classifiers along with the decision module. In these well-studied settings, the task is to learn one classifier and to either defer to an oracle (learning to reject) [Chow, 1970], to an expert (learning to defer) [Madras et al., 2018] or to multiple experts (learning to defer to multiple experts) [Verma et al., 2022]. However, these frameworks assume that the experts are external to the problem setting. They do not address how to train the experts alongside the base classifier.

In this work, we address the problem of jointly training two classifiers with a decision module. The two classifiers incur different costs, with the implication that the more expensive classifier offers better performance. We model this problem by introducing an additional information variable, Z , which represents the extra information available to the more powerful classifier. The decision module and the base classifier both have access to the same base input variable X . We refer to this setup as the two-stage classification problem.

We provide the optimal solution to this problem, as well as a surrogate loss function that is more suitable for training. The surrogate loss, which is based on the hinge loss, aligns with the standard cost-aware 0 -1 loss formulation commonly used in classification tasks. We validate our theoretical findings on synthetic datasets and demonstrate the practical relevance of the problem by presenting results on a standard large language model (LLM) task, where two LLMs of varying sizes are used to answer multi-question math problems. Additionally, we provide a proof that the cross-entropy loss, which is sometimes used heuristically in existing literature, is not Bayes consistent with the natural 0 -1 c loss, further justifying the need to explore this problem at a theoretical level.

Our main contributions are as follows:

1. We formulate a problem setting for learning a model that integrates two classifiers, where one has access to additional information but comes with a cost c . The goal is to train the models and simultaneously learn the decision function to determine whether to consult the more powerful classifier for a given sample.
2. We present a surrogate loss function based on the hinge loss, which is suitable for training with cost-aware classification tasks. We show that it is consistent with respect to the 0 -1 c loss that is natural for the considered problem.
3. We validate the theoretical findings, which are the primary contribution of the work, on synthetic datasets and provide practical insights through experiments on a standard LLM task.

## 2 Related Work

Loss consistency is an important topic that has been widely explored, as it serves as the fundamental link between the loss we optimize in practice and the actual loss we aim to minimize. Foundational results have been established for classical risks [Steinwart, 2007, Tewari and Bartlett, 2007, Bartlett et al., 2006], and the emergence of new target losses has prompted the development of new consistency results. Learning to defer (L2D) is a wide category of settings in which the task is to learn a classifier and a deferral rule, either to reject (learning to abstain) [Chow, 1957, 1970, Herbei and Wegkamp, 2006, Cao et al., 2022, Wiener and El-Yaniv, 2011, Geifman and El-Yaniv, 2019] or to defer to one or more experts of varying costs [Madras et al., 2018, Keswani et al., 2021, Wilder et al., 2021].

Mozannar and Sontag [2020] were the first to provide Bayes-consistency results for their proposed generalized cross entropy loss for learning to defer, followed by Verma et al. [2022], who used a one-vs-all loss. Awasthi et al. [2022] explored stronger guarantees than Bayes-consistency by introducing H -consistency bounds [Long and Servedio, 2013]. Mozannar et al. [2023] prove that earlier approaches, such as [Mozannar and Sontag, 2020, Verma et al., 2022], fall short of realizable H -consistency, and propose a new algorithm without a Bayes-consistency proof. This shortcoming is addressed by Mao et al. [2024a], who recently published a unifying work, introducing a new family of surrogate losses for the learning to defer problem with a single expert, and providing Bayesconsistency, realizable H -consistency, and H -consistency bounds. Verma et al. [2022] extended the work of Mozannar and Sontag [2020], Verma et al. [2022] to the multi-expert setting with Bayes-consistency guarantees. Mao et al. [2024b] introduced general cost functions and surrogate losses, extending the results of Mozannar and Sontag [2020] with H -consistency bounds for joint training, and offering stronger guarantees than Bayes-consistency. Mao et al. [2023] also provided H -consistency bounds for a slightly different setting where training the classifier and the deferral rule is done separately.

It appears that consistency results have been thoroughly studied in the case where only a single classifier is trained. However, the setting where multiple classifiers are trained jointly with a decision module has been largely neglected. This type of architecture is used in adaptive computation or dynamic networks [Han et al., 2022a], a branch of research focused on developing architectures that adaptively allocate computation. Since the main objective is to improve average inference efficiency, such dynamic network architectures have attracted significant interest in the development of scalable LLM inference [Liu et al., 2023, Elbayad et al., 2020, Zeng et al., 2024, Xia et al., 2024, Leviathan et al., 2023, Chen et al., 2024].

Although there is growing interest in these types of networks, the losses used to train such models are mostly heuristic and lack strong theoretical foundations. In practice, these models typically train the classifiers separately and rely on threshold-based decisions [Han et al., 2022a, Schuster et al., 2022]. Some theoretical research has been conducted on this separated training approach: Jitkrittum et al. [2023] explored the connection between threshold decisions and risk under a principled 0-1 loss, identifying conditions under which the two coincide. However, this separate training is not guaranteed to be the best approach. In fact, the importance of jointly learning the classifiers and the decision module has been empirically demonstrated [Han et al., 2022b, Yu et al., 2022, Regol et al., 2024, Krzepkowski et al., 2024], motivating the development of joint learning approaches [Regol et al., 2024] and classifier-deferral-aware training methods [Han et al., 2022b, Yu et al., 2022]. These works lack a connection to surrogate losses and a well-defined risk framework, which is the gap we aim to address in this work.

## 3 The two-stage classification problem

We consider a setting of two-stage classification where there are two classifiers: f 1 and f 2 . The second classifier, f 2 , has access to additional information z , but it also incurs an additional cost, denoted as c . We therefore have the choice between using the prediction of the first classifier, or to pay the additional cost and then use the more informed second classifier.

In practice, z can be explicitly modeled as an additional input signal or feature, which may come with higher access costs. For instance, in recommendation systems, different types of user data queries can vary significantly in terms of latency and infrastructure expense. A common approach is to first run a lightweight model for initial inference, and then selectively identify instances that would benefit from a more complex model with access to richer features. This tiered architecture is notably used by Youtube's recommendation system [Covington et al., 2016], for instance.

Alternatively, z can conceptually represent the augmented modeling capability of a larger model that has more parameters and/or was trained on a larger data set.

Denote by X the feature space, Z the additional information space, and Y = { 1 , . . . , K } the label space. We are given instance-label-information triples { ( x i , z i , y i ) } n i =1 independently and identically drawn from an underlying distribution D with probability function p ( X,Z,Y ) . We additionally introduce the decision module r : X → 0 , 1 , which indicates whether we are using, for the final decision, the first classifier f 1 if r ( x ) = 0 or to defer to the second classifier f 2 if r ( x ) = 1 .

The goal of two-stage classification is to train a two-stage classifier f : X ×Z → Y that encompasses both the classifiers f 1 : X → Y , f 2 : X × Z → Y , and the decision module r . The set H of two-stage classifiers is therefore defined as follows:

<!-- formula-not-decoded -->

̸

The loss associated with such a setting is the zero-one-exit loss ℓ 01 c , which can be expressed as a variant of the traditional zero-one loss ℓ 01 ( f ( · ) , y ) = [ f ( · ) = y ] :

̸

✶ where ✶ [ · ] is the indicator function. The cost c can be an instance-specific function, i.e., c ( x ) , provided it is known and deterministic. Since the additional information z is only accessible at a cost c , the first classifier and the decision function do not have access to it; the classifiers f 1 ( x ) and r ( x ) take only x as input.

<!-- formula-not-decoded -->

̸

Our task is to train a two-stage classifier f ∈ H , as defined by (1), that can minimize the expectation of ℓ 01 c over the data distribution. The risk is:

<!-- formula-not-decoded -->

and its optimal value R ∗ 01 c = R 01 c ( f ∗ ) is obtained by the Bayes-optimal classifier:

<!-- formula-not-decoded -->

The 01 c loss is discrete, and thus difficult to work with. We would like to be able to identify a surrogate loss ℓ ϕ such that ℓ ϕ -consistency implies ℓ 01 c -consistency. This is our main contribution in this work. We specify a surrogate loss function that satisfies this property, and show that other heuristic surrogate losses that are used in the literature for joint training [Regol et al., 2024, Ding et al., 2024] do not. Taking a step beyond this, we specify how to construct and train a two-stage classifier using the posited surrogate loss and present empirical results to validate our result.

## 3.1 The solution

We start by providing the solution to the optimization problem specified by (4). We first define a compact notation for the posteriors:

<!-- formula-not-decoded -->

Lemma 3.1. The optimal solution f ∗ = arg min f ∈H R 01 c ( f ) is the following:

<!-- formula-not-decoded -->

See Appendix A.4 for the proof of the lemma.

Using our previous definition of a two-stage classifier, this would correspond to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The optimal solution is interesting. It hints towards a model that is slightly different from most existing methods. Yes, the optimal decision should depend on p max = max η ( x ) , but the threshold for p max should be set based on the expected future gain : τ &lt; E p ( Z | x ) [max ζ ( x, Z )] -c . In that setting, r ( x ) should identify the set on which max η ( x ) ≥ E p ( z | x ) max ζ ( x, z ) -c , and, for these elements only, it should select the class with the largest probability according to the posterior η ( x ) . This result is similar to the solution for the decision rule given fixed classifiers first provided by Jitkrittum et al. [2023], which would read as max y η y ( x ) ≥ max y ζ y ( x, Z ) -c . However, our explicit modeling of the two-tiered information available to f 1 , r and f 2 provides a more practical and detailed solution, as it allows us to integrate the constraint that r cannot fully access the information available to f 2 . This modeling choice leads to a decision based on the expected future gain.

Unsurprisingly, the first and second classifiers simply predict the class with the highest probability according to their respective posteriors, but only for the samples assigned to them.

## 4 The proposed hinge-based surrogate loss

A common strategy to develop a consistent loss for more complex risk functions is to propose a surrogate loss and verify its consistency. This strategy was employed by early work for the learning to defer problem [Mozannar and Sontag, 2020, Verma and Nalisnick, 2022].

Our proposed surrogate loss is built on a multiclass version of the hinge loss [Tarigan and van de Geer, 2008]. We chose this version because it is Bayes-consistent, unlike other multiclass hinge

losses. We use a hinge loss rather than the more popular cross entropy is because of its linear scaling, which allows to account for the cost in an additive way as in Eqn. 2. Following the definition of the multiclass hinge loss from [Tarigan and van de Geer, 2008], the classifiers are based on K -dimensional real valued outputs t ( x ) , v ( x, z ) ∈ R K , with the constraints that ∑ K i =1 t i ( x ) = 0 , and ∑ K i =1 v i ( x, z ) = 0 . The label prediction is obtained by returning the max element of the vector. The decision function ˜ r ( x ) returns a real value bounded between 0 and 1. For brevity, we omit the dependence on the inputs and only write t , v . We can therefore introduce the link function φ that connects the real valued output and a soft decision function ˜ r ( x ) to a two-stage classifier function :

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

The loss is composed of a sum of two terms: the first trains the first classifier, and the second trains the second classifier. The balance or weight assigned to each term on a per-sample basis is intuitively controlled by the learned soft decision ˜ r . If ˜ r ( x ) indicates that a sample should be inferred by f 1 , then f 1 will receive more weight during training at that point. In the second term, corresponding to f 2 , we include an additional fixed term Kc K -1 that encodes the penalty of using the second classifier. This encourages ˜ r ( · ) to favor the first term unless the benefit of using f 2 outweighs the cost. We can then define the associated risk as:

<!-- formula-not-decoded -->

and consider the triplet of minimizers t ∗ ( x ) , v ∗ ( x, z ) , ˜ r ∗ ( x ) of such a risk:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the following theorem, we establish the consistency of our proposed surrogate loss w.r.t. ℓ 01 c , meaning that if a learned two-stage classifier f converges to the optimal surrogate risk R ∗ hinge , it also converges to the optimal target risk R ∗ 01 c .

Theorem 4.1. There exists a link function φ s.t. for any distribution p ( x, z, y ) , we have that:

<!-- formula-not-decoded -->

i.e., the surrogate loss ℓ c hinge ( v , t , r, x, z, y ) 11 is consistent with respect to the loss of interest ℓ 01 c ( φ ( t , v , r ) , x, z, y ) .

The proof is provided in Appendix A.5. The proof is built by showing that 1) the minimizers of both risks are unique and coincide:

<!-- formula-not-decoded -->

(Lemma A.1, with proof provided in Appendix A.5.1); and 2) that for some increasing function Ψ with Ψ(0) = 0 , the following holds:

<!-- formula-not-decoded -->

(Lemma A.2, with proof included in Appendix A.5.2). Taken together, these results guarantee consistency. We can actually establish that

<!-- formula-not-decoded -->

The bound on the risk gap provided by (18) allows us to further quantify the relationship between the two optimization problems, showing that the consistency is not merely asymptotic. This upper bound is tight and attainable for some cases of η and ζ . The K -1 K term comes from the scaling of the multi-hinge loss, while the factor of 2 accounts for corner cases where the routing decision is uncertain ( ˜ r ( x ) = 0 . 5 ) and the model perfectly estimates the posteriors η and ζ .

## 4.1 Cross entropy version

One might be tempted to build a similar formulation using the widely used cross-entropy loss -log( p y ) . Some heuristics in the literature for training two-stage or early exit models are built around a similar version of this loss [Regol et al., 2024]. Interestingly, we can prove that such a loss is in fact not Bayes consistent with the 0 -1 c loss that we presented. To build a cross entropy version of the proposed loss, we now need to assume that the model outputs predicted class probabilities p 1 ∈ ∆ K for f 1 and p 2 ∈ ∆ K for f 2 , where ∆ K is the K -dimensional simplex and φ is the same link function that was previously defined. The cross entropy version of the loss that we consider adds an arbitrary function of the cost g ( c ) and is given by:

<!-- formula-not-decoded -->

We again consider the associated risk:

<!-- formula-not-decoded -->

and the minimizing function:

<!-- formula-not-decoded -->

The following lemma shows that this coss-entropy surrogate loss cannot be Bayes-consistent. Lemma 4.2. There is no function g ( · ) for which the solution f ∗ ce to the associated problem in Eqn. 21 is equal to the Bayes-classifier f ∗ defined in Eqn. 6 for all distributions p ( X,Z,Y ) .

The proof is included in Appendix A.6.

## 5 Experiments

## 5.1 Synthetic Experiments

To validate our findings, we present a synthetic experiment in which the ground-truth posteriors are known. We design a simple K -class classification task with one-dimensional inputs X and Z to enable visualization of the learned functions. Our primary interest lies in visualizing the decision boundary ˜ r ( x ) of a model f trained with the proposed surrogate loss. This boundary should closely approximate the optimal decision rule r ∗ ( x ) , as defined in Eqn. 9. For completeness, we additionally include an experiment using the related learning-to-defer baselines, which we adapt to this particular setting in Appendix A.3.

Task Description The inputs X and Z are drawn uniformly from the interval [ -1 , 1) . The label Y is sampled from a categorical distribution with parameter θ = [ θ 1 , θ 2 , . . . , θ K ] T ∈ [0 , 1] K , where ∑ K i =1 θ i = 1 and p ( Y = i ) = θ i . The function θ ( x, z ) is defined piecewise by partitioning the domain of x, z into K -1 slanted regions. Full details of the construction of the synthetic dataset are provided in Appendix A.1.1. The random variables are distributed as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The constructed task can be visualized in Figure 1, where we show the class distribution in terms of most likely class and the samples x i , y i , z i ∼ p ( X,Z,Y ) for K = 5 . For this example, we can see that at x = 0 , the value of z provides essentially no additional information to estimate the correct posterior, which should translate into no deferral to f 2 ( r ∗ ( x = 0) = 0 ). At x = 0 . 25 , the variable z becomes informative. Therefore, the optimal decision function r ∗ ( x ) will alternate as vertical strips along the x-axis, with width of size that varies based on the cost parameter c .

Given this construction, the exact posterior probabilities can be computed in closed form, allowing us to derive the optimal decision rule r ∗ ( x ) . To approximate the expectations E p ( Z | x ) , we use Monte Carlo estimation by sampling from p ( Z | x ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 1: Visualization of multi class synthetic dataset with k = 5 . Left) Max probability labels arg max y p ( Y = y | X,Z ) . The black shaded region indicates where the optimal decision rule is to defer to f 2 ( r ∗ ( x ) = 1 ), given a cost of c = 0 . 03 . Right) Samples x, z, y ∼ p ( X,Z,Y ) of the synthetic experiment.

<!-- image -->

In our experiments, we use M = 1000 samples to approximate the expectations.

Training details We build simple 3-layer neural networks (NN) for t , v , and ˜ r . Following the requirements for ˜ r ( x ) , the corresponding network takes x as input and ends with a sigmoid activation. For t and v , the NNs take x and ( x, z ) as inputs, respectively, and output a real-valued vector of dimension K -1 . Appendix A.1.2 provides parameter size and layer details. We use the Adam optimizer with learning rate lr = 0 . 001 , a batch size of 512 and train for 50 epochs using our surrogate loss defined in Eqn. 11. The training set size is N tr = 10 , 000 and the test set size is N te = 1 , 000 .

Result and discussion Figure 2 illustrates the ground truth and predicted decision boundaries for cost values c = 0 . 03 , 0 . 07 and number of classes K = 3 , 5 . We observe that the model trained with the proposed surrogate loss successfully learns the correct decision boundary across different cost values and numbers of classes K . The learned decision function ˜ r ( x ) perfectly tracks with the ground truth r ∗ ( x ) . Additionally, although the trained model can output any value in the range ˜ r ( x ) ∈ [0 , 1] due to the sigmoid activation, it learns to produce sharp values near 0 or 1, which is the optimal behavior.

Now if we turn to a model trained with the additive version of the cross-entropy-based surrogate loss introduced in Eqn 19, using the identity function g ( c ) = c with K = 5 , we observe in Figure 3 that the behavior of the learned decision function ˜ r ( x ) differs significantly. First, we note that since consistency cannot be established for this surrogate loss, it is not possible to precisely target a desired cost level in the ℓ 01 c loss, unlike with the hinge-based surrogate. Looking at the results, the learned decision boundaries are generally unstable and uneven. The correct pattern of deferral for K = 5 can be observed in the top-right plot of Figure 2, where we see that four regions should be evenly spaced out and deferred to f 2 (regardless of c ). This pattern is not adequately learned in Figure 3. For instance, we see that the right-most region of x that should be deferred is slowly erased as the cost increases.

Lastly, we visualize the behavior of the learned model f hinge during training in Figure 4. We track the empirical target risk estimated from sampling ˆ R 01 c ( f ) = 1 N ∑ N i =1 ℓ 01 c ( f ( x i , z i ) , y i ) and observe that it converges to the (empirical) optimal risk ˆ R ∗ 01 c as expected.

## 5.2 Large Language Model Experiment

To illustrate a practical setting of the problem we consider, we present an experiment based on large language models (LLMs). In this experiment, we use two LLMs of different sizes, which correspond to different inference costs. The additional inference cost used by the larger model corresponds to c in our setup. The task involves solving multi-answer math questions. The intuition behind this setting is that some test questions should be more difficult than others. Therefore, it would be desirable to efficiently dispatch simpler questions to the smaller LLM and more challenging ones to the larger LLM. This allows us to achieve strong performance at a reasonable inference cost.

Figure 2: Top) Ground truth decision boundary ˆ r ∗ ( x ) with 2 costs values c = 0 . 03 , 0 . 07 and number of classes K = 3 , 5 . Bottom) Learned ˜ r ( x ) of the model that was trained with our surrogate loss. In all cases, the two decision boundaries are perfectly aligned, confirming our result that the model trained with the proposed surrogate loss successfully learns the optimal decision function. As the cost increases, the black region which represents points deferred to f 2 shrinks.

<!-- image -->

Figure 3: Top) Ground truth decision boundary ˆ r ∗ ( x ) with 4 costs values and number of classes K = 5 . Bottom) Learned ˜ r ( x ) of a model trained with the additive cross-entropy surrogate loss with g ( c ) = c , for varying c values. Unlike the model trained with the hinge-based surrogate, the learned decision patterns are generally wrong and not consistent.

<!-- image -->

Task description We use the Instruction-Tuned Pre-trained models LLaMA 3 8B and LLaMA 3 70B [Grattafiori et al., 2024] to solve multi-answer math questions from the AQUA dataset [Zhong et al., 2024]. The AQUA dataset is composed of multiple-choice math reasoning questions, each with 5 choices. We frame the task as a 5-class classification problem, where the model must select the correct option from a fixed set. The inputs x and z are formed by extracting the hidden-states from the final tokens of the 8B LLM and the 70B LLM, respectively. We use the first 1000 AQUA [Zhong et al., 2024] datapoints from the test split as our dataset, and use a 80 / 10 / 10 train/val/test split.

Training details We use a similar architecture to the one previously presented. The model is trained for 1000 epochs using a learning rate of 0.001, a batch size of 32, and early stopping with a patience of 20 epochs. Additional details are provided in Appendix A.1.3.

Figure 4: Empirical 0 -1 c risks of the learned function trained with the surrogate loss and of the optimal solution for K = 5 . We can see that for varying cost values, function trained with the surrogate loss converges to the optimal solution.

<!-- image -->

Results and discussion Although we do not have access to the ground truth decision function in this setting, we can examine the accuracy evaluated on the selected samples vs. all samples of the model trained with the surrogate loss. The selected samples of f 1 or f 2 are the samples routed to these functions by ˜ r ( x ) . Ideally, the two-stage classifier model f should learn to route 'hard' examples to f 2 , and 'easier' examples to f 1 . In practice, the surrogate loss can have two effects: 1) f 1 and f 2 are additionally trained on their respective selected samples; and 2) f 1 and f 2 may receive smaller gradient updates depending on the average deferral rates.

These two effects can be observed in Figure 5. In the left figure tracking f 1 , we see that the average accuracy slightly increases as the cost increases (and consequently the deferral rate decreases), and the inverse behavior can be seen for f 2 in the right figure. f 1 should, in principle, be given an easier task, so we can expect its selected accuracy to be higher than the average accuracy, which we observe in the left panel of Figure 5. The two values are closest when the selected samples comprise almost all the data (i.e., a deferral rate of 90% ). For f 2 , the selected accuracy is closer to the average accuracy. This could suggest that training only on the samples deferred to f 2 does not result in better performance-possibly because these consist of 'harder' instances.

Figure 5: Deferral rate and average accuracy on all samples and on selected samples by left) LLaMA 3 8B f 1 and by right) LLaMA 3 70B f 2 . The confidence intervals are computed on 10 trials.

<!-- image -->

In addition to aggregate performance, we can also inspect which types of queries are routed to each model. Figure 6 shows examples of math questions that were consistently routed to the smaller model ( f 1 ) and the larger model ( f 2 ) across various cost settings. From the presented examples, it appears that the 'easy' questions that were consistently routed to the small LLM ( f 1 ) generally involve basic arithmetic or proportions. In contrast, the labeled 'hard' questions that were consistently routed to the large LLM ( f 2 ) seem to require more comprehensive knowledge (such as motion or number theory). This suggests that the routing function aligns with our perceived notion of complexity and the type of reasoning required. See Appendix A.2 for the complete list of questions that were consistently routed to f 1 and f 2 .

## Example questions routed to the small LLM ( f 1 )

Question 1: The cost of 10 kg of mangos is equal to the cost of 24 kg of rice. The cost of 6 kg of flour equals the cost of 2 kg of rice. The cost of each kg of flour is $22. Find the total cost of 4 kg of mangos, 3 kg of rice and 5 kg of flour?

Question 2: A man buys an article and sells it at a profit of 20%. If he had bought it at 20% less and sold it for Rs.75 less, he could have gained 25%. What is the cost price?

## Example questions routed to the larger LLM ( f 2 )

Question 1: Two trains 140 m and 160 m long run at the speed of 60 km/hr and 40 km/hr respectively in opposite directions on parallel tracks. The time which they take to cross each other is?

Question 2: If the product of two numbers is 17820 and their H.C.F. is 12, find their L.C.M.

Figure 6: Sampled questions that are consistently being routed to f 1 or f 2 across different costs.

## 6 Conclusion and Limitations

In conclusion, this work aims to solidify the theoretical foundation behind the design and use of loss functions for the increasingly relevant problem of training multiple models with different costs, while also learning which model to use. We formalized this problem using a principled 0 -1 cost-based loss formulation and proposed a surrogate loss based on the hinge loss, showing its consistency.

Limitations A clear limitation of our work is that we only consider two models in our setup, whereas dynamic networks often require more than two. Extending our approach to the multi-stage setting would be a valuable direction for future research. Appendix A.7 provides a sketch of how our method can be generalized to the multi-stage setting with L classifiers. Moreover, although the theoretical results guarantee loss consistency, the hinge loss is less commonly used in practice. While we have presented a simple proposal of a cross-entropy surrogate loss and shown that it is insufficient for this setting, exploring alternative, more stable losses would be an important next step to ensure the development of practical and principled methods.

## 7 Social Impact

Although we believe that this theoretical paper poses minimal direct societal impact, the broader problem of cost-sensitive deferral systems may raise concerns related to fairness and access. In such systems, the model determines whether a query is 'simple' and can be handled by a smaller model, or 'difficult' and should be deferred to a more powerful model, which may involve higher computational cost or latency. This can introduce bias in how different users' queries are treated. For instance, if a particular user or group systematically submits queries that the system deems 'hard', they may consistently experience greater latency, potentially leading to unfair treatment or limited access. Additionally, this introduces new potential pathways for bias to enter the system, as the deferral rule itself can be biased. This could further exacerbate disparities in user experience and overall system fairness.

## Acknowledgements

We thank Prof. Rui Pires da Silva Castro for his valuable insights and suggestions, which greatly contributed to the development of our solution. This research was partially funded by the Natural Sciences and Engineering Research Council of Canada (NSERC), [reference number 260250]. Cette recherche a été partiellement financée par le Conseil de recherches en sciences naturelles et en génie du Canada (CRSNG), [numéro de référence 260250]. Ce projet de recherche n o 324302 est rendu possible grâce au financement du Fonds de recherche du Québec.

## References

Pranjal Awasthi, Anqi Mao, Mehryar Mohri, and Yutao Zhong. H-consistency bounds for surrogate loss minimizers. In Proc. Int. Conf. on Machine Learning (ICML) , 2022.

- Peter L Bartlett, Michael I Jordan, and Jon D McAuliffe. Convexity, classification, and risk bounds. Journal of the American Statistical Association , 101(473):138-156, 2006.
- Yuzhou Cao, Tianchi Cai, Lei Feng, Lihong Gu, Jinjie GU, Bo An, Gang Niu, and Masashi Sugiyama. Generalizing consistent multi-class classification with rejection to be compatible with arbitrary losses. In Proc. Adv. Neural Info. Process. Syst. (NeurIPS) , 2022.
- Yanxi Chen, Xuchen Pan, Yaliang Li, Bolin Ding, and Jingren Zhou. Ee-llm: Large-scale training and inference of early-exit large language models with 3d parallelism. In Proc. Int. Conf. Mach. Learn. (ICML) , 2024.
- C. Chow. On optimum recognition error and reject tradeoff. IEEE Transactions on Information Theory , 16(1):41-46, 1970.
- C. K. Chow. An optimum character recognition system using decision functions. IRE Transactions on Electronic Computers , EC-6(4):247-254, 1957.
- Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In Proc. ACM Conf. on Recommender Systems , page 191-198, 2016.
- Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Rühle, Laks V. S. Lakshmanan, and Ahmed Hassan Awadallah. Hybrid LLM: Cost-efficient and quality-aware query routing. In Proc. Int. Conf. Learning Representations (ICLR) , 2024.
- Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli. Depth-adaptive transformer. In Proc. Int. Conf. Learn. Representations (ICLR) , 2020.
- Yonatan Geifman and Ran El-Yaniv. SelectiveNet: A deep neural network with an integrated reject option. In Proc. Int. Conf. on Machine Learning (ICML) , 2019.
- Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, ..., and Zhiyu Ma. The Llama 3 herd of models, 2024.
- Y. Han, G. Huang, S. Song, L. Yang, H. Wang, and Y. Wang. Dynamic neural networks: A survey. IEEE Trans. on Pattern Analysis; Mach. Intell. , 44(11):7436-7456, 2022a.
- Yizeng Han, Yifan Pu, Zihang Lai, Chaofei Wang, Shiji Song, Junfen Cao, Wenhui Huang, Chao Deng, and Gao Huang. Learning to weight samples for dynamic early-exiting networks. In Proc. European Conf. on Computer Vision (ECCV) , 2022b.
- Radu Herbei and Marten H. Wegkamp. Classification with reject option. The Canadian Journal of Statistics / La Revue Canadienne de Statistique , 34(4):709-721, 2006.
- Wittawat Jitkrittum, Neha Gupta, Aditya K Menon, Harikrishna Narasimhan, Ankit Rawat, and Sanjiv Kumar. When does confidence-based cascade deferral suffice? In Proc. Adv. Neural Info. Process. Syst. (NeurIPS) , 2023.
- Vijay Keswani, Matthew Lease, and Krishnaram Kenthapadi. Towards unbiased and accurate deferral to multiple experts. In Proc. AAAI/ACM Conf. on AI, Ethics, and Society , 2021.
- Bartłomiej Krzepkowski, Monika Michaluk, Franciszek Szarwacki, Piotr Kubaty, Jary Pomponi, Tomasz Trzci´ nski, Bartosz Wójcik, and Kamil Adamczewski. Joint or disjoint: Mixing training regimes for early-exit models, 2024.
- Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In Proc. Int. Conf. Mach. Learn. (ICML) , 2023.
- Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Ré, and Beidi Chen. Deja vu: Contextual sparsity for efficient llms at inference time. In Proc. Int. Conf. Mach. Learn. (ICML) , 2023.

- Phil Long and Rocco Servedio. Consistency versus realizable h-consistency for multiclass classification. In Proc. Int. Conf. on Machine Learning (ICML) , 2013.
- David Madras, Elliot Creager, Toniann Pitassi, and Richard S. Zemel. Learning adversarially fair and transferable representations. arXiv preprint: arXiv 1802.06309, 2018.
- Anqi Mao, Christopher Mohri, Mehryar Mohri, and Yutao Zhong. Two-stage learning to defer with multiple experts. In Proc. Adv. Neural Info. Process. Syst. (NeurIPS) , 2023.
- Anqi Mao, Mehryar Mohri, and Yutao Zhong. Realizable h -consistent and bayes-consistent loss functions for learning to defer. In Proc. Adv. Neural Info. Process. Syst. (NeurIPS) , 2024a.
- Anqi Mao, Mehryar Mohri, and Yutao Zhong. Principled approaches for learning to defer with multiple experts. In Artificial Intelligence and Image Analysis , 2024b.
- Hussein Mozannar and David Sontag. Consistent estimators for learning to defer to an expert. In Proc. Int. Conf. on Machine Learning (ICML) , 2020.
- Hussein Mozannar, Hunter Lang, Dennis Wei, Prasanna Sattigeri, Subhro Das, and David Sontag. Who should predict? exact algorithms for learning to defer to humans. In Proc. Int. Conf. on Artificial Intelligence and Statistics (AISTAT) , 2023.
- Florence Regol, Joud Chataoui, and Mark Coates. Jointly-learned exit and inference for a dynamic neural network. In Proc. Int. Conf. Learn. Representations (ICLR) , 2024.
- Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q. Tran, Yi Tay, and Donald Metzler. Confident adaptive language modeling. In Proc. Adv. in Neural Inf. Proces. Syst. (NeurIPS) , 2022.
- Ingo Steinwart. How to compare different loss functions and their risks. Constructive Approximation , 26(2):225-287, Aug 2007.
- Bernadetta Tarigan and Sara A. van de Geer. A moment bound for multi-hinge classifiers. Journal of Machine Learning Research , 9(71):2171-2185, 2008.
- Ambuj Tewari and Peter L. Bartlett. On the consistency of multiclass classification methods. Journal of Machine Learning Research , 8(36):1007-1025, 2007.
- Rajeev Verma and Eric T. Nalisnick. Calibrated learning to defer with one-vs-all classifiers. In Proc. Int. Conf. on Machine Learning (ICML) , 2022.
- Rajeev Verma, Daniel Barrej'on, and Eric Nalisnick. Learning to defer to multiple experts: Consistent surrogate losses, confidence calibration, and conformal ensembles. In Proc. Int. Conf. on Artificial Intelligence and Statistics (AISTAT) , 2022.
- Yair Wiener and Ran El-Yaniv. Agnostic selective classification. In Proc. Adv. Neural Info. Process. Syst. (NeurIPS) , 2011.
- Bryan Wilder, Eric Horvitz, and Ece Kamar. Learning to complement humans. In Proc. Int. Joint Conf. on Artificial Intelligence , 2021.
- Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. arXiv preprint: arXiv 2401.07851, 2024.
- Haichao Yu, Haoxiang Li, Gang Hua, Gao Huang, and Humphrey Shi. Boosted dynamic neural networks. In Proc. AAAI Conf. on Artif. Intell. , 2022.
- Ziqian Zeng, Yihuai Hong, Hongliang Dai, Huiping Zhuang, and Cen Chen. ConsistentEE: A consistent and hardness-guided early exiting method for accelerating language models inference. In Proc. AAAI Conf. Artif. Intell. , pages 19506-19514, Mar. 2024.
- Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. AGIEval: A human-centric benchmark for evaluating foundation models. In Findings of the Association for Computational Linguistics , June 2024.

## A Appendix

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The contributions are stated in the introduction are supported by theoritical results and experiments.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We include a discussion on the limitation of our work in the conclusion.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Proofs are included in the Appendix.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Implementation details are included in the main text and in the Appendix.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: n/a

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The details are included in the main text and in the Appendix.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report statistical significance on our real data experiment. res or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The required computational resources were minimal.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This research is inline with NeurIPS Code of Ethics.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification:

answerNA

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: [NA]

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: [NA]

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: [NA]

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: [NA]

## A.1 Additional experimental details

## A.1.1 Synthetic task

In this section, we provide additional details of the synthetic task. We model Y by a categorical distribution with parameter θ = [ θ 1 , θ 2 , . . . , θ k ] T ∈ [0 , 1] k satisfying ∑ k i =1 θ i = 1 and p ( Y = i ) = θ i . θ is modeled as a piecewise function by partitioning the range of X into k -1 equally sized bins { B 1 , B 2 , . . . , B k -1 } . Assuming the range of X is [ a, b ) , a, b ∈ R , we define B i = [ a + ( i -1)( b -a ) k -1 , a + ( i )( b -a ) k -1 ) . Within every bin B i , only θ i and θ i +1 take on non-zero values, following a scaled and shifted sigmoid:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(35)

Using this model, we arrive at the closed form for the posteriors:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we can derive the optimal decision function r ∗ ( x ) :

<!-- formula-not-decoded -->

## A.1.2 Synthetic Model details

We describe the architecture of the model used in the synthetic experiment. The hidden size of all networks is 64 . Each neural network is defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.1.3 LLMModel details

We describe the architecture of the model used in the LLM experiment. The hidden size of all networks is 128 . We performed a grid search for the hidden size across the values { 32 , 64 , 128 , 256 } and for the learning rate across the values { 0 . 01 , 0 . 001 , 0 . 0001 } .

Each neural network is defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.2 Complete list of deferred questions (LLM experiments)

In this section, we provide a comprehensive list of the questions that were consistently deferred to f 2 or sent to f 1 in the LLM experiment.

## Example questions sent to the small LLM ( f 1 ):

1. A man buys an article and sells it at a profit of 20%. If he had bought it at 20% less and sold it for Rs. 75 less, he could have gained 25%. What is the cost price?
2. The cost of 10 kg of mangos is equal to the cost of 24 kg of rice. The cost of 6 kg of flour equals the cost of 2 kg of rice. The cost of each kg of flour is $22. Find the total cost of 4 kg of mangos, 3 kg of rice and 5 kg of flour?
3. The speed of a boat in upstream is 100 kmph and the speed of the boat downstream is 180 kmph. Find the speed of the boat in still water and the speed of the stream.
4. A and B working together could mow a field in 28 days and with the help of C they could have mowed it in 21 days. How long would C take by himself?
5. Evaluate the expression:

<!-- formula-not-decoded -->

6. In an examination, 60% failed in Math and 40% failed in French. If 15% failed in both subjects, what percentage of students passed in both?
7. One train crosses a bridge of length 340 m in 42 seconds, and the same train crosses another bridge of length 500 m in 50 seconds. What is the approximate speed of the train in km/hr?
8. Eshan and Mary each wrote two or three poems every day over a period of time. Eshan wrote 43 poems while Mary wrote 61. What is the number of days in this period?
9. Find the value of x in the sequence of numbers 5, 1, 6, 0, 4, 8, x , 2 if the sum of the first 7 numbers is 30 and the average is 4.
10. Roja and Pooja start moving in opposite directions from a pole. They are moving at speeds of 7 km/hr and 3 km/hr respectively. After 4 hours, what will be the distance between them?

## Example questions deferred to the larger LLM ( f 2 ):

1. If the product of two numbers is 17820 and their H.C.F. is 12, find their L.C.M.
2. Two passenger trains start at the same hour in the day from two different stations and move towards each other at the rate of 14 kmph and 21 kmph respectively. When they meet, it is found that one train has traveled 60 km more than the other one. What is the distance between the two stations?
3. Which is the odd one: 10, 25, 45, 54, 60, 75, 80?
4. Two trains 140 m and 160 m long run at the speed of 60 km/hr and 40 km/hr respectively in opposite directions on parallel tracks. The time which they take to cross each other is?
5. A ladder 100 feet long is leaning against a vertical wall. Its lower end is 60 feet from the bottom of the wall. The side of the largest cubical box that can be placed between the wall and the ladder without disturbing the ladder is (to the nearest foot)?
6. On dividing a certain number by 5, 7 and 8 successively, the remainders obtained are 2, 3 and 4 respectively. When the order of division is reversed and the number is successively divided by 8, 7 and 5, what will be the respective remainders?
7. A tour group of 25 people paid a total of $670 for entrance to a museum. If this price included a 5% sales tax, and all the tickets cost the same amount, what was the face value of each ticket price without the sales tax?

8. Arectangular floor is covered by a rug except for a strip 4 meters wide along each of the four edges. If the floor is 25 meters by 20 meters, what is the area of the rug in square meters?
9. In each of the following questions a number series is given with one term missing. Choose the correct alternative that will continue the same pattern and fill in the blank space.

<!-- formula-not-decoded -->

10. In a game of 500 points there are three participants A, B, and C. A gives to B 80 points and to C 101 points. Then how many points can B give to C?
11. When magnified 1,000 times by an electron microscope, the image of a certain circular piece of tissue has a diameter of 2 centimeters. The actual diameter of the tissue, in centimeters, is:
12. From the given equation, find the value of x :

<!-- formula-not-decoded -->

13. The sum of the non-prime numbers between 50 and 60, non-inclusive, is:
14. Solve the system of equations to find the values of c and d :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

15. How many minutes does Aditya take to cover a distance of 400 meters, if he runs at a speed of 20 km/hr?
16. An engineer designed a ball so that when it was dropped, it rose with each bounce exactly one-half as high as it had fallen. The engineer dropped the ball from an 18-meter platform and caught it after it had traveled 53.4 meters. How many times did the ball bounce?

## A.3 Additional baselines

We include additional baselines that assume different settings from ours, such as the learning-to-deferto-expert setting, which trains a classifier alongside a deferral function that defers to a fixed expert, as in [Mozannar and Sontag, 2020]. We also consider simple thresholding methods that only produce results for deferral, assuming fixed classifiers, including the popular confidence-based thresholding rule investigated by Jitkrittum et al. [2023]. These methods generally rely on training cross-entropy models for f 1 , and obtain the associated predicted probabilities p 1 . To adapt these baselines to our setting, we train the second classifier f 2 separately using a standard cross-entropy loss, and then follow the scheme of the baselines to train the first classifier f 1 and obtain the decision function ˜ r ( x ) . Let ˆ acc f denote the empirical accuracy of a model f evaluated on a validation set.

We include the following rules:

CT-c: We pretrain f 1 using cross-entropy. For a given cost c , we set the threshold τ using the empirical accuracy of the second classifier minus the cost τ = ˆ acc f 2 -c : and define the deferral rule as:

<!-- formula-not-decoded -->

Soft deferral: We pretrain f 1 using cross-entropy, and sample the deferral decision from a Bernouilli distribution with 1 -max y ∈Y p 1 y as a parameter:

<!-- formula-not-decoded -->

CT: We pretrain f 1 using cross-entropy, and search for the optimal threshold τ that yields the smallest empirical risk on a validation set.

<!-- formula-not-decoded -->

L2D: We use our the pretrained f 2 as the expert for the method of Mozannar and Sontag [2020]. We set the confidence parameter in our expert, α , to be the average accuracy of f 2 : α = ˆ acc f .

Figure 7 shows the empirical risk during training for a setting with a cost of c = 0 . 3 and number of classes k = 5 . We observe that our proposed approach converges the fastest and achieves the lowest empirical risk. In Table 1, we report the empirical risk ˆ R 01 c along with the standard deviation computed across 10 trials for varying costs. Our proposed surrogate loss consistently attaining the lowest empirical risk across all cost settings.

Figure 7: Empirical risk ˆ R 01 c of the learned function trained with our proposed surrogate loss and other baselines for K = 5 and c = 0 . 3 .

<!-- image -->

Table 1: Empirical risk ˆ R 01 c for different baselines and deferral costs c with K = 5 . The mean and standard deviation are computed across 10 trials. The bolded entry denotes the lowest value.

| Baseline      | c = 0 . 03      | c = 0 . 05      | c = 0 . 07      |
|---------------|-----------------|-----------------|-----------------|
| CT-c          | 0.3701 ± 0.0013 | 0.3842 ± 0.0007 | 0.4031 ± 0.0014 |
| CT            | 0.3700 ± 0.0008 | 0.3784 ± 0.0016 | 0.3853 ± 0.0036 |
| Soft deferral | 0.3700 ± 0.0012 | 0.3789 ± 0.0008 | 0.3855 ± 0.0014 |
| L2D           | 0.3712 ± 0.0015 | 0.3826 ± 0.0022 | 0.3906 ± 0.0025 |
| Hinge (ours)  | 0.3695 ± 0.0004 | 0.3777 ± 0.0004 | 0.3842 ± 0.0005 |

## A.4 Proof of the solution f ∗

In this section, we provide the proof of Lemma 3.1, which states that the optimal solution

<!-- formula-not-decoded -->

is the following:

<!-- formula-not-decoded -->

Or alternatively:

where

R

≜

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We start by evaluating the risk w.r.t to the function:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

and prove the result by showing that any function f o = f s results in a higher risk, therefore showing that f s is the optimal solution.

The risk is given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can partition X in two regions based on the decision function of f s , i.e. : A = { x ; max η ( x ) ≥ E p ( z | x ) max ζ ( x, z ) -c } and B = { x ; max η ( x ) ≤ E p ( z | x ) max ζ ( x, z ) -c } and split the expectation in two terms:

<!-- formula-not-decoded -->

S

∫

∫

∑

y

x

∈

S

z

Looking at both terms separately, starting with R A where f s does not use z (or corresponds to f 1 ):

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can obtain more straightforwardly R B :

<!-- formula-not-decoded -->

Hence, the risk of f s is given by;

<!-- formula-not-decoded -->

s

[

ℓ

01

c

(

f

(

x, z

)

, y

)]

p

(

y

|

x, z

)

p

(

z

|

x

)

p

(

x

)

dxdz.

(62)

In the following, we prove that which would imply that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Showing that δ A d ≤ 0 and δ B d ≤ 0 We first consider δ A d :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can (once again) further partition the space based on f o . We divide A d in two based on the decision function of f o :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Now, we consider a different two-stage classifier f o ∈ H and f s = f o . We show that any f o ∈ H will lead to a higher risk R 01 c ( f s ) ≤ R 01 c ( f o ) , therefore proving that f s = f ∗ .

̸

We further partition the space X by splitting A and B where f s = f o and f s = f o

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

We then use those new partitions to further decompose the risks of f s and f o in 4 terms:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can then write the difference between the risks as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Continuing our development of δ A d :

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

(96)

At this stage we can focus on one term at the time, starting with the part of A d where f o is not using z which is the integral over A dx :

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Going to the second term [**], when f 0 uses z . We start by restating the definition of the set A : A = { x ; max η ( x ) ≥ E p ( z | x ) [max y ζ y ( x, z )] -c } . (101)

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining both results together, we have that

<!-- formula-not-decoded -->

Following similar steps, we also have that

̸

<!-- formula-not-decoded -->

Final step Since we have both that δ B d ≤ 0 and δ A d ≤ 0 , we can conclude that

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes the proof.

## A.5 Proof of consistency for the multi class surrogate hinge loss

We start by restating our surrogate loss:

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

where we have t ∈ R K and v ∈ R K as the real-vectored outputs for f 1 and f 2 respectively with the constraints that || v || 1 = 0 and || t || 1 = 0 , and ˜ r ∈ [0 , 1] as a soft decision output.

Since our surrogate optimization provides us with the triplet t , v , ˜ r , we map these to a two-stage classifier f ∈ H using the following a link function φ : R K × R K × [0 , 1] →Y :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can then define our risk as usual:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and consider the triplet of minimizers t ∗ ( x ) , v ∗ ( x, z ) , ˜ r ∗ ( x ) of such a risk, which correspond to a two-stage solution:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We prove that our surrogate loss is Bayes-consistent w.r.t to the ℓ 01 c loss by showing that 1)

<!-- formula-not-decoded -->

and 2)

<!-- formula-not-decoded -->

Taken together, those results guarantee that for any distribution p ( x, z, y ) , we have that:

<!-- formula-not-decoded -->

which defines Bayes-consistency [Steinwart, 2007].

## A.5.1 The solutions f ∗ of R 01 c ( f ) and f ∗ hinge coincide

Restating the definitions of the solution of the target and surrogate problems;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In this section, we show that the two solutions f ∗ , f ∗ hinge coincide.

Lemma A.1. For any distribution p ( X,Z,Y ) ;

<!-- formula-not-decoded -->

Proof. We have previously shown in Appendix A.4 that the solution of our targeted problem

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we show that the two-stage classifier obtained from the triplet minimizer of our surrogate loss f ∗ hinge = φ ( t ∗ , v ∗ , ˜ r ∗ ) corresponds to this solution.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

We can push the optimization problem inside the expectation w.r.t p ( x ) as t ( x ) , v ( x, z ) , ˜ r ( x ) are all functions of x (and the inner expectation term is guaranteed to be bounded):

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Since the loss is a linear combination of two terms that respectively depend on t and v , we can see that for any ˜ r , the minimizers for t and v will always be equal to the minimizer of the individual terms:

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

is given by:

For the multi class hinge loss that we are considering, it is known that the minimizing functions are given by the following [Tarigan and van de Geer, 2008]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which gets converted into f ∗ 1 and f ∗ 2 by the link function φ (see Eqn 10):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we turn to the decision function ˜ r ( x ) . Using ˜ r as shorthands for ˜ r ( x ) :

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Since it is a linear combination of 1 -˜ r ( x ) and ˜ r ( x ) , it is clear that the minimizer ˜ r ∗ ( x ) will either be at 0 or 1. We can therefore rewrite the optimization problem as the following:

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We consider the two cases for A ( x, ˜ r ) .

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the second case, using similar steps:

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

This allows us to write the solution as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We have therefore shown that as

This concludes the proof.

<!-- formula-not-decoded -->

Since we now have that ˜ r ∗ ( x ) is restricted to the binary values ˜ r ∗ ( x ) = { 0 , 1 } , we can rewrite the optimal classifiers that we previously obtained:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we can see that the optimal t ∗ and v ∗ leads to the same solution for the internal classifiers of f ∗ . We have therefore shown that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.5.2 Gap of the hinge loss

Next, we aim to show that for some increasing function Ψ with Ψ(0) = 0 , we can upper bound the risk gap of our loss of interest R 01 c ( φ ( t , v , ˜ r )) -R ∗ 01 c with the risk gap of our surrogate hinge loss R hinge ( t , v , ˜ r ) -R ∗ hinge .

Lemma A.2. For any distribution p ( X,Z,Y ) :

<!-- formula-not-decoded -->

Proof. We start by developing the hinge risk R hinge ( t , v , ˜ r ) :

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Next we develop the term associated to the optimal hinge risk R ∗ hinge :

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Bringing both R ∗ hinge and R hinge ( f ) to evaluate the gap G ≜ R hinge ( f ) -R ∗ hinge , we can decompose the gap by a sum of 4 terms that are driven by the ground truth decision cases, i.e. r ∗ = 0 or ∗ = 1 and the decision of the model ˜ r ( x ) :

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

We can define the corresponding gap to each case as follows:

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

We can obtain a similar decomposition for the risk gap of our target risk R 01 c ( f ) . We recall the definition:

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

̸

R 01 c ( f ) = E p ( x ) [ ✶ [ r ( x ) = 0](1 -η f 1 ( x ) ) + ✶ [ r ( x ) = 1] E p ( z | x ) [1 -ζ f 2 ( x,z ) + c ] ] (201) and decompose the gap risk with terms based on similar cases:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This makes sense: if the optimal decision is defer and we don't, the risk is diminished by the saved computation c (the second case). If the optimal decision is not to defer and do, the risk is increased by the computation cost c (the third case).

<!-- formula-not-decoded -->

and rewrite the total gap as:

To prove the result, we show that :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

G 1 inequality Starting with G 1 and F 1 :

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We need to find a mapping Ψ such that the following holds:

<!-- formula-not-decoded -->

Using the simple scaling function Ψ( x ) = 2( K -1) K x , we can see that the previous inequality holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

G 2 inequality For the next inequality with G 2 and F 2 , again using the same function Ψ( x ) = 2( K -1) K x , the inequality holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(231)

Following similar steps, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Putting all results together, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with Ψ(0) = 0 and is increasing. This concludes the proof.

## A.6 Proof of the failure of the cross entropy version

In this section, we prove that the cross entropy version of the surrogate loss we presented cannot be Bayes-consistent.

We recall the entropy version, with p 1 ∈ ∆ K for f 1 and p 2 ∈ ∆ K for f 2 , with the same link function φ . The cross entropy version of the loss that we consider is given by:

<!-- formula-not-decoded -->

where g ( c ) is an arbitrary function. We again consider its associated risk:

<!-- formula-not-decoded -->

and minimizing function:

<!-- formula-not-decoded -->

Lemma 4.2. Cross-entropy surrogate loss is not Bayes Consistent.

There is no g ( c ) for which:

<!-- formula-not-decoded -->

Proof. Following a similar reasoning as the proof of Lemma A.1 , we can readily find that the optimal predicted probability vectors p 1 and p 2 in f ∗ ce should match the posteriors:

<!-- formula-not-decoded -->

Now, to find the optimal decision function of the cross entropy risk ˜ r ∗ ce , we can again obtain the solution as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by following the same steps that were taking to obtain Eqn. 153. We again consider the two cases:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The solution for the cross-entropy decision function is hence given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we recall the solution decision of our target problem;

<!-- formula-not-decoded -->

we are searching for a function g ( c ) for which

If we define the decision sets of x :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we can rewrite the condition Eqn. 248 as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies that the function g ( c ) should satisfy:

<!-- formula-not-decoded -->

However, our condition for r ∗ ( x ) is on the absolute scale, not on the log scale r ∗ ( x ) = η ∗ ( x ) &lt; E p ( z | x ) [ ζ ∗ ( x, z )] + c . We will therefore have that for some η, ζ

<!-- formula-not-decoded -->

This implies that there is no g ( c ) that can satisfy

This concludes the proof.

<!-- formula-not-decoded -->

## A.7 Extension to the multi-classifier setting

In this section, we provide a high-level view of how we could extend our results to the more general case of L + 1 classifiers. In this setting, there are still two stages, but there are L classifiers to choose from at the second stage (with associated costs c 1 , . . . c L ). We would need to introduce additional random variables Z 1 , . . . , Z L , and the loss in Eqn 2 would need to be generalized to multiple classifiers and costs:

̸

<!-- formula-not-decoded -->

̸

̸

The corresponding solution for the decision boundaries (Eqn. 9) would become more complex. Instead of comparing the maximum posterior probability η ( x ) to a single expected future gain minus cost, the comparison would now need to be made against the best potential expected future gain for each classifier:

<!-- formula-not-decoded -->

The term max l ∈ 1 ,L ( E p ( Z l | x ) [max y p ( y | x, Z l )] -c l ) returns the index of the best model that can be used at the second stage.

Then, we could propose a multi-classifier surrogate loss (replacing Eqn. 11) by using a soft decision function that is now a multiclass probability vector ˜ r ( x ) ∈ [0 , 1] L +1 with ∑ L l =0 ˜ r l ( x ) = 1 . We would also need to introduce L new learnable vectors t (0) , . . . , t ( L ) , and an index-dependent cost term K K -1 g ( l, c 1 , . . . , c L , K ) , where g ( l, c 1 , . . . , c L , K ) is some linear function of the costs that would need to be derived and obtained from the proof. The hinge surrogate loss could (potentially) have the following form:

̸

<!-- formula-not-decoded -->

It remains to be seen whether we can verify the consistency of a loss of this form.