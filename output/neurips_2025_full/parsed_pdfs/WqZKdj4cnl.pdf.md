## Learning-Augmented Algorithms for k -median via Online Learning

## Anish Hebbar

Duke University anishshripad.hebbar@duke.edu

Amit Kumar IIT Delhi

amitk@cse.iitd.ac.in

## Abstract

The field of learning-augmented algorithms seeks to use ML techniques on past instances of a problem to inform an algorithm designed for a future instance. In this paper, we introduce a novel model for learning-augmented algorithms inspired by online learning. In this model, we are given a sequence of instances of a problem and the goal of the learning-augmented algorithm is to use prior instances to propose a solution to a future instance of the problem. The performance of the algorithm is measured by its average performance across all the instances, where the performance on a single instance is the ratio between the cost of the algorithm's solution and that of an optimal solution for that instance. We apply this framework to the classic k -median clustering problem, and give an efficient learning algorithm that can approximately match the average performance of the best fixed k -median solution in hindsight across all the instances. We also experimentally evaluate our algorithm and show that its empirical performance is close to optimal, and also that it automatically adapts the solution to a dynamically changing sequence.

## 1 Introduction

The field of learning-augmented algorithms has seen rapid growth in recent years, with the aim of harnessing the ever-improving capabilities of machine learning (ML) to solve problems in combinatorial optimization. Qualitatively, the goal in this area can be stated in simple terms: apply ML techniques on prior problem instances to inform algorithmic choices for a future instance . This broad objective has been interpreted in a variety of ways - online algorithms with better competitive ratios using ML predictions, faster (offline) algorithms based on ML advice, better approximation factors for (offline) NP-hard problems using ML advice, etc. This breadth of interpretation has led to a wealth of interesting research at the intersection of algorithm design and machine learning.

In this paper, we take inspiration from the field of online learning to propose a new model for learning-augmented algorithms. The stated goal of learning-augmented algorithms of using prior instances to inform the solution of a future instance bears striking similarity with online learning. In its simplest form, online learning comprises a sequence of instances in which the learner has to select from a set of actions to minimize unknown loss functions. The choice of the learner is based on the performance of these actions on prior instances defined by their respective loss functions. The goal is to optimize the performance of the learner in the long run - it aims to match the average performance of the best fixed action in hindsight. Viewing learning-augmented algorithms through the lens of online learning, we ask:

Can a learning-augmented algorithm compete in average performance over a sequence of problem instances with the best fixed solution in hindsight?

## Rong Ge

Duke University rongge@cs.duke.edu

## Debmalya Panigrahi

Duke University debmalya@cs.duke.edu

To instantiate this question, we consider the classic k -median problem in clustering. In this problem, we are given a set of n points in a metric space, and the goal is to find k centers such that the sum of distances of the points to their closest centers is minimized. We note that k -median is among the most well-studied problems in unsupervised learning. In the learning-augmented framework, consider an online sequence of k -median instances V 1 , V 2 , . . . , V t , . . . , V T on some metric space M . Now, suppose at time t , the learning-augmented algorithm A has access to the prior instances V 1 , V 2 , . . . , V t -1 and has to devise a solution comprising k centers for the next instance V t (that is unknown to it). Once the algorithm A reveals its solution Y t , the instance V t is revealed and the loss of the algorithm is the quality of the solution Y t for V t , namely the ratio of the cost of Y t and that of an optimal solution for V t .

We define the loss to be the ratio instead of the absolute cost because the instances that arrive can have very different scales. In particular, if one instance is very large and the rest are small, it suffices to perform well on just the large instance when looking at absolute cost. Thus, taking cost ratios ensures that the algorithm is incentivized to do well on every instance, regardless of size. Secondly, it is impossible to obtain regret bounds independent of the number of points n when considering absolute costs. To see this, suppose we get a very large instance with essentially all n points that is far away from all the previous (much smaller) instances which are cumulatively on o ( n ) points. Any online learning algorithm would then incur a cost of n ∆ for this single instance, where ∆ is the aspect ratio of the metric space. 1 On the other hand, the benchmark solution can simply optimize for this large instance and incur small cost for this instance. It suffers a large cost on the previous instances, but these instances being cumulatively small (only o ( n ) points), the overall cost of the benchmark remains small in terms of absolute cost.

Similar to online learning, A cannot hope to match optimal performance for a single instance, but across time, can it match the performance of the best fixed solution in hindsight? In other words, we would like to learn a competitive solution from prior instances when such a solution exists. In this paper, we show that this is indeed possible - for k -median clustering, we design an online learning algorithm that approximately matches the performance of the best fixed solution across any sequence of instances on a metric space.

A natural approach to the above goal would be to employ the tools of online learning directly by encoding each solution as an 'expert' and running an experts' learning algorithm (see e.g., [Hazan, 2016]) on the problem instances. This does not work for two reasons, and these also reveal the main conceptual challenges in this problem. The first challenge is computational . The space of solutions to a discrete optimization problem such as k -median clustering is exponentially large and non-convex . Therefore, we cannot run experts' algorithms that maintain explicit weights over the entire set of solutions, neither can we use a black-box online convex optimization (OCO) algorithm (see e.g., [Hazan, 2016]). The second challenge is information-theoretic . Since the sequence of instances is revealed online, the learning algorithm does not know future instances when it proposes a solution. Therefore, the metric space itself, and the corresponding space of solutions, is dynamically growing over time. Indeed, the best solution in hindsight may not even be an available option to the algorithm at some intermediate time t . The main technical contribution of this paper is in overcoming this set of fundamental challenges to obtain a learning-augmented algorithm for k -median that matches the average performance of the best solution in hindsight.

Our algorithms have potential applications in clustering evolving data streams, which has been identified as an important open challenge in data stream mining research [Krempl et al., 2014]. Some examples include aggregating similar news stories for recommendations downstream [Gong et al., 2017] and maintaining accurate clustering of sensor data that fluctuates due to environmental conditions [Wang et al., 2024]. Another application lies in clustering network traffic where patterns of network user connections change slowly over time in normal circumstances, but can also undergo sudden shifts if a malicious attack occurs [Gong et al., 2017, Wang et al., 2024, Feng, 2006].

## 1.1 Our Contributions

Problem Model and Definition. Our first contribution is the new learning-augmented model that we propose in this paper. We define it in the context of the k -median problem, and call it LEARN-

1 The aspect ratio of a metric space is the ratio of the maximum to minimum pairwise distance between distinct points in the metric space.

MEDIAN. The problem setting comprises an online sequence of instances of k -median defined on a metric space M = ( V, d ) with a set of n points V and distance function d : V × V → R ≥ 0 . The algorithm does not assume knowledge of the metric space ( V, d ) , its size n , or the length of the time horizon T upfront. The instance at time t comprises a set of points V t ⊆ V . Using knowledge of prior instances V 0 , V 1 , . . . , V t -1 , the learning-augmented algorithm has to produce a solution Y t comprising k points in V &lt;t := V 0 ∪ V 1 ∪ . . . ∪ V t -1 . 2 This solution is then applied to the instance V t and incurs cost

<!-- formula-not-decoded -->

Correspondingly, the (approximation) loss of the algorithm is given by

<!-- formula-not-decoded -->

̸

Since the ratio defining ρ ( Y t , I t ) is invariant under scaling, we may assume that min x,y ∈ V ; x = y d ( x, y ) = 1 , and denote the aspect ratio max x,y ∈ V d ( x, y ) by ∆ . Also, to avoid degenerate cases where OPT t = 0 , causing the ratio to be infinity, we assume each instance must have at least k +1 distinct points.

The goal is to match the average performance over time of the best fixed solution Y ∗ (in hindsight), where Y ∗ := argmin Y ⊆ V, | Y | = k ∑ T t =1 ρ ( Y, V t ) .

Our Results. We say that a learning-augmented algorithm for LEARN-MEDIAN obtains an ( α, β ) -approximation if

<!-- formula-not-decoded -->

We call α the competitive ratio and β the regret of the algorithm. The goal is to obtain sublinear regret, i.e., β = o ( T ) , while minimizing the competitive ratio α .

Our first result is a deterministic algorithm with α = O ( k ) , β = o ( T ) . Precisely, we get

<!-- formula-not-decoded -->

We also give a matching lower bound showing that the competitive ratio of O ( k ) is necessary, i.e., there is no deterministic algorithm with competitive ratio α = o ( k ) and sub-linear regret β = o ( T ) .

Next, we use randomization to improve the competitive ratio to O (1) . We give a randomized algorithm with α = O (1) , β = o ( T ) :

<!-- formula-not-decoded -->

As in the deterministic case, we show that the O (1) factor in the competitive ratio is necessary, i.e., there is no algorithm (randomized or deterministic) with competitive ratio α = 1 + ε for arbitrarily small ε &gt; 0 and sub-linear regret β = o ( T ) .

Finally, note that the regret bounds in both results depend on k, ∆ . We show that this dependence is necessary, i.e., there is no algorithm (randomized or deterministic) with competitive ratio α = O (1) and regret β = o ( k ∆) .

Our Techniques. We give a sketch of the main techniques in our algorithms. Both the deterministic and randomized algorithm share all the steps, except the last one.

2 The set V 0 initializes the problem. The algorithm does not generate a solution for V 0 .

- -In the first step, we convert the LEARN-MEDIAN instance to a bounded version of this problem where each instance comprises exactly k points in the metric space. We call this latter problem LEARN-BOUNDED-MEDIAN. The conversion is done online at every time t .
- -Next, we relax the k -median problem by allowing fractional solutions, i.e., the k centers are allowed to be distributed fractionally across the points of the metric space. Our main technical work is in producing a fractional solution to the k -median instances of LEARNBOUNDED-MEDIAN. Note that the feasible fractional solution define a (convex) simplex, but the number of dimensions of the simplex grows over time as more points appear in the k -median instances. The main algorithmic ideas that we develop for this problem are outlined in the paragraph below.
- -Our final step is in rounding the fractional solutions to integer k -median solutions. Here, we give two different algorithms, one deterministic and the other randomized, both of which adapt ideas from LP-based rounding of k -median.

We now outline the main ideas in obtaining a fractional solution to the k -median instances. The main challenge is that the convex space of solutions available to the algorithm expands over time, yet the algorithm must compete with a solution that is only guaranteed to lie in the final convex set. This means that in applying standard online convex optimization (OCO) tools such as online mirror descent (OMD), we only have partial knowledge of the gradient at any intermediate step corresponding to the dimensions/points that have appeared previously. Unfortunately, this limitation makes standard regularizers ineffective: the squared L 2 -norm regularizer fails to achieve sub-linear regret since the dimension of the gradient depends on the time horizon, while the negative entropy regularizer is unable to handle new dimensions since the gradient is undefined at zero. Instead, we give a non-standard changing regularizer parametrized by the number of unique points seen so far. Correspondingly, we depart from standard OCO analysis that compares against a fixed solution in hindsight. Instead, we divide the timeline into phases and compare the performance of our algorithm against a dynamic set of locally optimal solutions. The main advantage is that within a phase, we only compare our algorithm against a solution that exists in the current convex set. The phases are chosen carefully so that we can use metric properties to show that these locally optimal solutions have total cost comparable to the hindsight optimum. Morally, we ensure that if the hindsight optimum is far from the locally optimal solutions, then it performs poorly on the instances in these phases, and therefore, cannot be the best solution in hindsight across all the instances. Putting all these components together in an online learning algorithm is the main technical contribution of this paper.

Experimental Evaluation. We perform experiments to validate our theoretical results empirically. We first run our algorithms on synthetic data generated from simple distributions such as a uniform distribution on a square or on a set of distinct clusters. In these cases, we confirm that the algorithms quickly converge to the optimal solution, and the average performance is nearly optimal in long run, thereby significantly outperforming theoretical bounds. In our second set of experiments, we use a sequence of dynamically changing distributions with the goal of understanding the responsiveness of our algorithm to changing data. We observe that as the algorithm discovers new data sets corresponding to new regions of the metric space, it reacts quickly to the changing environment and changes the location of k centers, in some cases even outperforms the static best solution in hindsight.

## 1.2 Related Work

Our work fits in the burgeoning field of learning-augmented algorithms, where the goal is to leverage ML predictions to improve the performance of algorithms for optimization problems. This paradigm was initially proposed for the online caching problem by [Lykouris and Vassilvitskii, 2021], and has since been applied to many problem domains (see [Mitzenmacher and Vassilvitskii, 2022] for a survey). In particular, various clustering problems such as k -median and k -means clustering [Ergun et al., 2022, Nguyen et al., 2023, Gamlath et al., 2022] and facility location [Jiang et al., 2022, Fotakis et al., 2025] have been studied in this framework, both in the offline and online settings. Our work sharply deviates from these existing lines of research in that we consider a model comprising a sequence of problem instances inspired by online learning, while prior work focused on solving a single instance using ML predictions. A notable exception is Khodak et al. [2022], where the authors consider an online learning based model to learn predictions from a sequence of problem instances, and apply it to a variety of matching, scheduling, and rent-or-buy problems. In some sense, our

model is a more direct counterpart of this idea, in that we apply the online learning paradigm to the performance of the algorithm directly instead of using a surrogate learned parameter.

From a technical standpoint, our work is related to the field of combinatorial online learning, where the goal is to develop low-regret algorithms for problems with exponentially large discrete action spaces. Traditional online learning tools like multiplicative weights update (MWU) are typically inapplicable due to computational barriers stemming from the sheer size of the solution space. To address this, several works have adapted online convex optimization and gradient-based techniques for combinatorial problems. Applications include online facility location [Christou et al., 2023, Pasteris et al., 2021], online k -means clustering [Cohen-Addad et al., 2021], online routing [Awerbuch and Kleinberg, 2008], online learning of disjunctions [Helmbold et al., 2002], coalition formation [Cohen and Agmon, 2024], online submodular minimization [Hazan and Kale, 2012], online matrix prediction [Hazan et al., 2012], and online ranking [Fotakis et al., 2020]. One distinction in our model is that we consider the average approximation ratio across the instances, while these works consider average cost. More importantly, the novelty in our work lies in designing a new online learning algorithm and analysis for the fractional problem, while most papers focus on reducing the combinatorial problem to an OCO formulation and apply standard OCO tools to the resulting problem.

Our work also bears some similarity with prior work on facility location problems in evolving metric spaces [Eisenstat et al., 2014, An et al., 2017]. But, in technical terms, this line of research is quite different in that it focuses on metric distances changing over time, while our focus is on new locations being added. Another related line of work is that of universal algorithms for optimization problems. These algorithms aim to compute a single solution that performs well across all possible realizations of the input, thus providing robust approximation guarantees. Examples include universal algorithms for Steiner tree and TSP [Jia et al., 2005, Bhalgat et al., 2011, Gupta et al., 2006], set cover [Grandoni et al., 2008, Jia et al., 2005], clustering [Ganesh et al., 2023]. Unlike our work which is in the online setting, universal algorithms are offline and thereby use a very different set of tools.

Finally, our work is also related to data-driven algorithms (see [Balcan, 2021] for a survey), where the typical goal is to pick the best algorithm out of a palette of algorithms whose behavior is dictated by a numerical parameter. In this line of work, the typical assumption is that the instances are drawn from an unknown distribution, and the goal is to obtain sample complexity bounds for a given target average performance in comparison to the best parameter choice. In contrast, in our setting, the algorithm is allowed to output any valid solution in each round and the final results are about average performance across an adversarial set of instances. [Balcan et al., 2018] considers the parameterized framework but in an online learning setting, where they pick a parameter for each instance that arrives online. The benchmark in this case is the algorithm corresponding to the best fixed parameter in hindsight. Their techniques heavily utilize the parametrization of the algorithms, because of which they are not applicable to our setting.

Outline of the paper. In Section 2, we give the reduction from LEARN-MEDIAN to LEARNBOUNDED-MEDIAN, for which we then give a fractional algorithm in Section 3, and integer algorithms via online rounding in Section 4. Details of these technical sections appear in the appendix. We present our experiments and empirical results in Section 5; again, details are in the appendix. Finally, the lower bounds that complement our algorithmic results are also given in the appendix.

## 2 Reduction from LEARN-MEDIAN to LEARN-BOUNDED-MEDIAN

One difficulty in designing an algorithm for LEARN-MEDIAN is that the k -median instances V t can have arbitrary size, which would mean that the metric space on which we design the algorithm expands in an arbitrary manner in each online step.

To alleviate this problem, in this section, we give a reduction from LEARN-MEDIAN to a simpler problem we call LEARN-BOUNDED-MEDIAN, which has at most k weighted points per instance. In particular, this reduction eliminates the dependence on the size of instances n in our regret bounds. Otherwise, We will see in the next section that the subgradient that we compute at each step of the fractional algorithm can have a large L ∞ norm that depends on n , which would then affect the regret term. On the other hand, if we apply our mirror descent algorithm (given in the next section) to only k points in each round, then the total number of points under consideration is at most kT . Thus, the regret term is independent of n . Asecond benefit of the reduction is that it reduces the running time of the learning algorithm, and makes it independent of the size of the instances n . (The reduction uses a

k -median algorithm that runs on instances of size n , but this is unavoidable in any case.) We typically expect that n will be much larger than all other parameters in the problem. Using the reduction, we only need to run our learning algorithm on at most kT points, which is independent of n .

It is important to choose exactly k points in the reduction because if we use fewer or more than k points, then the weights for the reduced instance are inconsistent with the k -median optimum. For example, if we use &lt; k points and there are k closely knit clusters, then the resulting cost of the solution (which determines weights) is arbitrarily large in comparison. Similarly, if we use &gt; k points and there are k +1 closely knit clusters, then the resulting cost of the solution is arbitrarily small.

Bounded Instances of LEARN-MEDIAN . Similar to LEARN-MEDIAN, a LEARN-BOUNDEDMEDIAN instance is specified by a metric space M = ( V, d ) and we are also given a sequence of k -median instances. At time t , instance R t is specified by a set of at most k weighted points. Each point x ∈ R t has an associated weight w t x (for simplicity, we often omit the superscript t when it is clear from context). The instances further satisfy ∑ x ∈ R t w x ≤ ( k +1) . All other definitions are the same as before, except that the cost is now defined with respect to the weights:

<!-- formula-not-decoded -->

We often just use Cost t ( Y ) when the instance is clear from context.

Reduction to LEARN-BOUNDED-MEDIAN . We show the following reduction:

Theorem 2.1. Given an instance for the LEARN-MEDIAN problem, there exists an algorithm A that maps each sub-instance V t to a sub-instance R t = A ( V t ) , resulting in a LEARN-BOUNDEDMEDIAN instance. If solutions Y 1 , ..., Y T for the new instance of LEARN-BOUNDED-MEDIAN satisfy

<!-- formula-not-decoded -->

where α ≥ 1 and R = R 0 ∪ R 1 . . . ∪ R T , then we have

<!-- formula-not-decoded -->

We now describe the reduction - the detailed analysis is deferred to the appendix. The main idea is to replace each set V t by k centers approximating the optimal k -median cost of V t - these weighted centers act as proxies for the points in V t . More formally, consider an offline instance of the k -median problem comprising the points in V t . We use a constant factor approximation algorithm A for min C ⊆ V t , | C | = k ∑ x ∈ V t D ( C, x ) to obtain a set of k centers C t = { c 1 , . . . , c k } . Let V ( i ) t denote the points in V t that are assigned to c i in the approximate solution. In the weighted instance R t , we choose R t = C t , and the weight of a point c i ∈ R t is set to | V ( i ) t | ∑ x ∈ Vt D ( C t ,x ) . This completes the reduction to an instance of LEARN-BOUNDED-MEDIAN.

## 3 Fractional Algorithm via Online Mirror Descent with Hyperbolic Entropy Regularizer

To design an online algorithm for LEARN-BOUNDED-MEDIAN, perhaps the simplest idea is follow the leader (FTL), where the algorithm outputs the best fixed solution for the prior instances. However, finding the exact FTL solution is computationally intractable; moreover, we show (in the appendix) that using an approximate FTL solution fails to give a competitive algorithm even for k = 1 .

A common approach to address intractability is to use a convex relaxation. Indeed, [Fotakis et al., 2021] introduced a method that could solve LEARN-BOUNDED-MEDIAN instances if the entire metric space ( V, d ) were known upfront. Unfortunately, not knowing the metric space changes the problem significantly. For example, if the metric space is known, then it is possible to achieve pure (sub-linear) regret using a fractional solution, i.e., a competitive ratio of 1 . In contrast, we give a simple example in the appendix that rules this out even for k = 1 in our setting.

Our algorithm maintains a fractional solution for points in R &lt;t , updated by online mirror descent (OMD) using a hyperbolic entropy regularizer (Definition 3.1). New points introduced have fractional value initialized to 0. To analyze the performance of the algorithm, we construct a sequence of solutions that only uses points in R &lt;t and can only change k log T times. This allows us to show that the fractional solution is constant competitive with sublinear regret. We give further details of these steps in the rest of this section.

Fractional solutions for LEARN-BOUNDED-MEDIAN . At time t , only points in R &lt;t are revealed. Let d t -1 = | R &lt;t | be the number of points available at time t , and ∆ k d t denote { z ∈ R d t ≥ 0 : || z || 1 = k } . A fractional solution y t is in the set K t -1 which is the intersection 3 of ∆ k d t -1 and [0 , 1] d t -1 .

Given a vector y t ∈ K t -1 , for each point x the fractional assignment cost is defined by assigning the point x fractionally to the centers in y t : D ( y t , x ) := min 0 ≤ α i ≤ ( y t ) i , ∑ i α i =1 ∑ i ∈ [ d t -1 ] α i d ( v i , x ) . The cost for the instance is still the weighted sum as before: Cost t ( y t ) := ∑ x ∈ R t w x D ( y t , x ) .

Our algorithm. At each time t , a weighted subset of points R t arrives. At time t = 0 , we initialize y 0 (which is a k -dimensional vector) to any k arbitrary points in R 0 . At step t , we maintain a fractional solution y t on the points R &lt;t . To go from y t to y t +1 , we perform one step of mirror descent based on the β -hyperbolic entropy regularizer defined below:

Definition 3.1 ( β -hyperbolic entropy [Ghai et al., 2020]) . For any x ∈ R d and β &gt; 0 , define the β -hyperbolic entropy of x , denoted ϕ β ( x ) as:

<!-- formula-not-decoded -->

The associated Bregman divergence is given by B ϕ β ( x || y ) = ϕ β ( x ) -ϕ β ( y ) -⟨∇ ϕ β ( y ) , x -y ⟩ =

<!-- formula-not-decoded -->

Note that we considered other regularizers such as 1 / 2 || x || 2 2 (which recovers online gradient descent) and ∑ i x i log x i (which recovers exponentiated gradient descent). The problem with the former is that the regret term in our analysis is not sublinear in T , while the latter is ill-defined for our setting as the gradient is not defined at 0 , which we need for the new points. Before taking one mirror descent step, for points in R t but not in R &lt;t , we set their fractional values to 0 . This gives the fractional solution y t +1 for the next iteration. The detailed algorithm is given in Algorithm 1 .

Analysis. To prove Algorithm 1 is constant competitive, we compare its cost to a changing sequence of optimal solutions. To construct such a sequence of optimal solutions, we first partition the sequence of k -median instances into phases. Let y ∗ ∈ K T be an integral solution defined by a set C of k centers. Let the centers in C be c (1) , . . . , c ( k ) . The set C partitions V into k subsets - let V ( i ) be the subset of V for which the closest center is c ( i ) (we break ties arbitrarily).

Considering each center in this optimal solution separately, we split the solution into different phases at times j i , which are the smallest time t such that the total weight of the points in V ( i ) ∩ R ≤ t (i.e., points in V ( i ) arriving by time t ) exceeds k · 2 j . Define the set P := { j i : i ∈ [ k ] , j ∈ [log( w ( V ( i ) ))] } ⊆ [1 , T ] be the set of times when a new phase starts. It's easy to check that | P | ≤ k log T as the total weight of arriving points is at most k +1 in each step.

For each phase, we bound the cost of the algorithm using standard tools from the OCO literature:

Lemma 3.2. For any time interval I := [ t a , t b ] ⊆ [1 , T ] , let z be an integral vector in K t a . Then,

<!-- formula-not-decoded -->

This lemma shows that up to additive factors, in each phase we can be competitive against any solution. Therefore it remains to show that there exists a series of solutions that only use available points whose cost is competitive against the offline optimal solution y ∗ .

3 In fact, we take K t -1 = ∆ k d t -1 in the experiments for simplicity as the analysis is the same.

```
Algorithm 1: Algorithm for an instance of LEARN-BOUNDED-MEDIAN 1.1 Initialize y 0 to any k points in R 0 , i.e., y 0 ∈ R k with ( y 0 ) i = 1 ∀ i ∈ [ k ] . 1.2 for t = 1 , 2 , . . . , T do 1.3 Let the points in R ≤ t -1 be v 1 , . . . , v d t -1 . 1.4 for each x ∈ R t do 1.5 let α x i be the fractional assignment of x to v i , i.e., D ( y t , x ) = ∑ i α x i d ( v i , x ) . 1.6 Define M ( x ) := max i : α x i > 0 d ( v i , x ) . 1.7 Sub-gradient Step: Define ∇ t ∈ R d t as follows: for each i ∈ [ d t ] , ( ∇ t ) i := -∑ x ∈ R t w x ( M ( x ) -min( M ( x ) , d ( x, v i )) . 1.8 Learning Rate: Set η t := 1 G t √ t where G t = max t ′ ≤ t ||∇ t ′ || ∞ . 1.9 Update Step: Define a vector x t +1 ∈ R d t as follows: for each i ∈ [ d t ] , ( x t +1 ) i := sinh(arcsinh( d t ( y t ) i -η t ( ∇ t ) i ) d t , where we use ( y t ) i = 0 in the above equation for each i ∈ [ d t ] \ [ d t -1 ] . 1.10 Projection Step: Define y t +1 := argmin y ∈ K t B ϕ 1 /d t ( y || x t +1 )
```

Lemma 3.3. For each p ∈ P , let the start and end time of phase p be denoted s p and e p respectively. Then, there exist solutions z ( p ) such that

<!-- formula-not-decoded -->

where y ∗ is an arbitrary integral vector in K T .

## 4 Integral Solutions via Online Rounding

In this section, we give online rounding algorithms for the LEARN-BOUNDED-MEDIAN problem. Let y t be the fractional solution computed by Algorithm 1 before the arrival of the t th instance R t . Our rounding algorithms will take the fractional solution y t as input and return an integer solution Y t with exactly k centers. The rounding loss is defined as the ratio of the (expected) cost of the integer solution Y t to that of the fractional solution y t .

We give two rounding algorithms. The first algorithm is deterministic and has a rounding loss of O ( k ) . The second algorithm is randomized and has only O (1) rounding loss. These are adaptations of existing rounding algorithms for k -median [Charikar et al., 1999, Charikar and Li, 2012]; the main difference is that we need to bound the rounding loss for points we haven't seen as well.

Deterministic Rounding Algorithm. The deterministic rounding algorithm is based on a natural greedy strategy: at time t , maintain a set Y t that is initially empty. Consider the points in R &lt;t = R 0 ∪ . . . R t -1 in non-decreasing order of their connection costs D ( y t , i ) . When considering a point i , add it to Y t if its connection cost to Y t exceeds its fractional connection cost in y t by an Ω( k ) factor. This process ensures that | Y t | ≤ k - indeed, the intuition is that each time we add a point to Y t , we can draw a ball around it which has almost 1 unit of fractional mass (according to y t ). The criteria for adding a point to Y t ensures that the total connection cost to Y t remains within O ( k ) factor of that to y t . Thus, we get a rounding algorithm that rounds y t to an integral solution Y t and loses an O ( k )

factor in approximation - we give the details in the appendix. Combining this fact with Theorem 2.1 and Lemma 3.3, we get the desired result (1).

Randomized Rounding Algorithm. The randomized rounding algorithm proceeds in two phases: the first phase is similar to the deterministic rounding algorithm described above. However, instead of keeping a threshold of Ω( k ) for adding a center to Y t , it uses a constant threshold. This results in | Y t | being larger than k . Subsequently a randomized strategy is used to prune this set to size k . We show in the appendix that this algorithm has constant approximation gap. Combining this fact with Theorem 2.1 and Lemma 3.3, we get the desired result (2).

In conclusion, we obtain that the time complexity of our overall algorithm (besides the reduction step) is O ( k 2 T 3 ) , as the bottleneck is computing the O ( kT × kT ) distance matrix in each round.

## 5 Experimental Results

In this section, we empirically evaluate the performance of our algorithms. In all our experiments, we use a heuristic to improve the performance of the rounding algorithms, where we do a binary search over the best threshold on the distance of centers to open (this value was (2 k +2) D ( y t , i ) for deterministic rounding and 4 D ( y t , i ) for randomized rounding). We set this threshold to the smallest possible that still gives the desired number of centers k . We report the main experimental findings in this section, and give additional results and experiments in the appendix.

Our first two experiments are for simple i.i.d. instances. In Uniform Square , each round has a set of points sampled uniformly from the unit square [0 , 1] × [0 , 1] . In Multiple Clusters , the underlying metric space consists of a set of clusters, and the points are sampled uniformly from these clusters. In both cases, the distances are given by the underlying Euclidean metric. As seen in Fig. 1, both the deterministic and randomized algorithms converge to expected solutions - for Uniform Square , the centers are distributed (roughly) uniformly over the square and for Multiple Clusters , the centers spread themselves on the different clusters. In the adjoining plots, we show the approximation ratio of the algorithms, which approaches 1 once the influence of additive regret declines.

<!-- image -->

Time

Time

Figure 1: For i.i.d. instances (Uniform Square and Multiple Clusters ), the optimal (black plus), deterministic (blue cross), and randomized (red diamond) solutions (top figure) and approximation ratios - avg. and std. dev. over 10 random instances (bottom figure).

In the third experiment Oscillating Instances , we have two distinct clusters, and instances comprise alternating batches of geometrically increasing length that draws points from one of the clusters. The optimal solution alternates between the two clusters, switching every time the current batch becomes long enough to dominate all previous instances. In Fig. 2a, we show two important properties of our algorithm in this setup. First, for k = 1 , we plot the ratio over time t between ∑ t τ =1 ρ ( Y τ , V τ ) and

<!-- image -->

(a) Approximation ratio for Oscillating Instances

<!-- image -->

(c) Fractional mass of algorithm over time at the center for Uniform Hypersphere (avg. and std. dev. over 10 runs)

<!-- image -->

(b) Fractional mass of algorithm and OPT over time in one of the clusters for Oscillating Instances

<!-- image -->

- (d) Fractional mass of algorithm in each cluster in Scale Changing

Figure 2: Experimental Results for Dynamic Instances

∑ t τ =1 ρ ( OPT t , V τ ) , where OPT t is the best fixed solution for instances V 1 , ..., V t . Interestingly, our algorithm outperforms the optimal solution by virtue of being adaptive while the optimal solution is fixed. Next, in Fig. 2b we trace the migration of fractional mass between the two clusters for k = 2 . We observe the algorithm quickly moves one unit of mass to the current cluster, and then slows down since moving more fractional mass does not significantly reduce cost.

In the next set of experiments, we explore settings where the metric space changes significantly over time. In Uniform Hypersphere , we choose k = 1 and every round consists of points chosen at random from the surface of the unit sphere. Around the 100 th iteration, we also include the origin in the instance. This changes the space of solutions drastically because the origin is now a much better solution than any of the surface points. As seen in Fig. 2c, the algorithm moves fractional mass to the origin once it becomes available, eventually converging entirely to this location.

Our final experiment is called Scale Changing . Here, we have 5 clusters with 10 points each. The clusters grow geometrically far apart in distance. The sequence of instances iterates over these clusters in batches of geometrically increasing length. Fig. 2d shows the fractional mass of the algorithm's solution in the all clusters. We see that our algorithm adapts to the changing environment by increasing the fractional mass in the cluster being currently presented, and as earlier, this process slows after unit fractional mass has been moved to the new cluster.

## 6 Closing Remarks

In this paper, we studied the problem of solving a sequence of k -median clustering instances to match the best fixed solution in hindsight. One limitation of our techniques is that they are tailored to metric problems. While these techniques might extend to problems similar to k -median such as k -means clustering and facility location, they do not apply to other problem domains such as covering for which learning-augmented algorithms have been extensively studied. Nevertheless, our problem formulation bridging online learning and discrete optimization applies to the entire range of problems in combinatorial optimization. We leave the design of algorithms in our framework for other problem domains as interesting future work.

## Acknowledgments and Disclosure of Funding

Debmalya Panigrahi and Anish Hebbar were supported in part by NSF awards CCF-195570 and CCF-2329230. Rong Ge was supported in part by NSF Award DMS-2031849.

## References

- Hyung-Chan An, Ashkan Norouzi-Fard, and Ola Svensson. Dynamic facility location via exponential clocks. ACM Transactions on Algorithms (TALG) , 13(2):1-20, 2017.
- Baruch Awerbuch and Robert Kleinberg. Online linear optimization and adaptive routing. Journal of Computer and System Sciences , 74(1):97-114, 2008.
- Maria-Florina Balcan. Data-driven algorithm design. In Tim Roughgarden, editor, Beyond the Worst-Case Analysis of Algorithms , page 626-645. Cambridge University Press, 2021.
- Maria-Florina Balcan, Travis Dick, and Ellen Vitercik. Dispersion for data-driven algorithm design, online learning, and private optimization. In 2018 IEEE 59th Annual Symposium on Foundations of Computer Science (FOCS) , pages 603-614. IEEE, 2018.
- Anand Bhalgat, Deeparnab Chakrabarty, and Sanjeev Khanna. Optimal lower bounds for universal and differentially private steiner trees and tsps. In International Workshop on Approximation Algorithms for Combinatorial Optimization , pages 75-86. Springer, 2011.
- Moses Charikar and Shi Li. A dependent lp-rounding approach for the k-median problem. In International Colloquium on Automata, Languages, and Programming , pages 194-205. Springer, 2012.
- Moses Charikar, Sudipto Guha, Éva Tardos, and David B Shmoys. A constant-factor approximation algorithm for the k-median problem. In Proceedings of the thirty-first annual ACM symposium on Theory of computing , pages 1-10, 1999.
- Dimitrios Christou, Stratis Skoulakis, and Volkan Cevher. Efficient online clustering with moving costs. Advances in Neural Information Processing Systems , 36:25325-25364, 2023.
- Saar Cohen and Noa Agmon. Online learning of partitions in additively separable hedonic games. In Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI) , pages 2722-2730, 2024.
- Vincent Cohen-Addad, Benjamin Guedj, Varun Kanade, and Guy Rom. Online k-means clustering. In International Conference on Artificial Intelligence and Statistics , pages 1126-1134. PMLR, 2021.
- David Eisenstat, Claire Mathieu, and Nicolas Schabanel. Facility location in evolving metrics. In International Colloquium on Automata, Languages, and Programming , pages 459-470. Springer, 2014.
- Jon C Ergun, Zhili Feng, Sandeep Silwal, David P Woodruff, and Samson Zhou. Learning-augmented k -means clustering. In Proceedings of the Tenth International Conference on Learning Representations (ICLR) , 2022.
- Cao Feng. Density-based clustering over an evolving data stream with noise. SDM 2006 , 2006.
- Dimitris Fotakis, Thanasis Lianeas, Georgios Piliouras, and Stratis Skoulakis. Efficient online learning of optimal rankings: Dimensionality reduction via gradient descent. Advances in Neural Information Processing Systems , 33:7816-7827, 2020.
- Dimitris Fotakis, Georgios Piliouras, and Stratis Skoulakis. Efficient online learning for dynamic k-clustering. In International Conference on Machine Learning , pages 3396-3406. PMLR, 2021.
- Dimitris Fotakis, Evangelia Gergatsouli, Themistoklis Gouleakis, Nikolas Patris, and Thanos Tolias. Improved bounds for online facility location with predictions. In Toby Walsh, Julie Shah, and Zico Kolter, editors, AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA , pages 26973-26981. AAAI Press, 2025.

- Buddhima Gamlath, Silvio Lattanzi, Ashkan Norouzi-Fard, and Ola Svensson. Approximate cluster recovery from noisy labels. In Conference on Learning Theory , pages 1463-1509. PMLR, 2022.
- Arun Ganesh, Bruce M Maggs, and Debmalya Panigrahi. Universal algorithms for clustering problems. ACM Transactions on Algorithms , 19(2):1-46, 2023.
- Udaya Ghai, Elad Hazan, and Yoram Singer. Exponentiated gradient meets gradient descent. In Algorithmic learning theory , pages 386-407. PMLR, 2020.
- Shufeng Gong, Yanfeng Zhang, and Ge Yu. Clustering stream data by exploring the evolution of density mountain. Proceedings of the VLDB Endowment , 11(4):393-405, 2017.
- Fabrizio Grandoni, Anupam Gupta, Stefano Leonardi, Pauli Miettinen, Piotr Sankowski, and Mohit Singh. Set covering with our eyes closed. In 2008 49th Annual IEEE Symposium on Foundations of Computer Science , pages 347-356. IEEE, 2008.
- Anupam Gupta, Mohammad T Hajiaghayi, and Harald Räcke. Oblivious network design. In Proceedings of the seventeenth annual ACM-SIAM symposium on Discrete algorithm , pages 970-979, 2006.
- Elad Hazan. Introduction to online convex optimization. Found. Trends Optim. , 2(3-4):157-325, 2016. doi: 10.1561/2400000013. URL https://doi.org/10.1561/2400000013 .
- Elad Hazan and Satyen Kale. Online submodular minimization. The Journal of Machine Learning Research , 13(1):2903-2922, 2012.
- Elad Hazan, Satyen Kale, and Shai Shalev-Shwartz. Near-optimal algorithms for online matrix prediction. In Conference on Learning Theory , pages 38-1. JMLR Workshop and Conference Proceedings, 2012.
- David P Helmbold, Sandra Panizza, and Manfred K Warmuth. Direct and indirect algorithms for on-line learning of disjunctions. Theoretical Computer Science , 284(1):109-142, 2002.
- L Jia, G Lin, G Noubir, R Rajaraman, and R Sundaram. Universal algorithms for tsp, steiner tree, and set cover. In Proceedings of the 36th Annual ACM Symposium on Theory of Computing , 2005.
- Shaofeng H-C Jiang, Erzhi Liu, You Lyu, Zhihao Gavin Tang, and Yubo Zhang. Online facility location with predictions. In Proceedings of the Tenth International Conference on Learning Representations (ICLR) , 2022.
- Misha Khodak, Maria-Florina Balcan, Ameet Talwalkar, and Sergei Vassilvitskii. Learning predictions for algorithms with predictions. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , 2022.
- Georg Krempl, Indre Žliobaite, Dariusz Brzezi´ nski, Eyke Hüllermeier, Mark Last, Vincent Lemaire, Tino Noack, Ammar Shaker, Sonja Sievi, Myra Spiliopoulou, et al. Open challenges for data stream mining research. ACM SIGKDD explorations newsletter , 16(1):1-10, 2014.
- Thodoris Lykouris and Sergei Vassilvitskii. Competitive caching with machine learned advice. Journal of the ACM (JACM) , 68(4):1-25, 2021.
- Michael Mitzenmacher and Sergei Vassilvitskii. Algorithms with predictions. Commun. ACM , 65(7): 33-35, June 2022. ISSN 0001-0782. doi: 10.1145/3528087. URL https://doi.org/10.1145/ 3528087 .
- Thy Nguyen, Anamay Chaturvedi, and Huy Lê Nguyen. Improved learning-augmented algorithms for k -means and k -medians clustering. In Proceedings of the Eleventh International Conference on Learning Representations (ICLR) , 2023.
- Stephen Pasteris, Ting He, Fabio Vitale, Shiqiang Wang, and Mark Herbster. Online learning of facility locations. In Algorithmic Learning Theory , pages 1002-1050. PMLR, 2021.
- Zhengru Wang, Xin Wang, and Shuhao Zhang. Mostream: A modular and self-optimizing data stream clustering algorithm. In 2024 IEEE International Conference on Data Mining (ICDM) , pages 500-509. IEEE, 2024.

## Technical Appendices and Supplementary Material

## A Reduction from LEARN-MEDIAN to LEARN-BOUNDED-MEDIAN

Recall that in Section 2 we first reduce LEARN-MEDIAN to a simpler problem we call LEARNBOUNDED-MEDIAN, which has at most k weighted points per instance. We do this because the original k -median instances V t can have arbitrary size, which would mean that the metric space on which we design the algorithm expands in an arbitrary manner in each online step.

We give the details of this reduction step in this section.

## A.1 Bounded Instances of LEARN-MEDIAN

A LEARN-BOUNDED-MEDIAN instance is specified by a metric space M = ( V, d ) and we are also given a sequence of subinstances. At time t , instance R t is specified by a set of at most k weighted points R t . Each point x ∈ R t has an associated weight w t x (for simplicity, we often omit the superscript t when it is clear from context). The instances further satisfy ∑ x ∈ R t w x ≤ ( k +1) . The cost for a round is now given by the weighted k -median cost:

<!-- formula-not-decoded -->

We often just use Cost t ( Y ) when the instance is clear from context.

## A.2 Reduction

We now show that any instance of LEARN-MEDIAN can be reduced (in an online manner) to an instance of LEARN-BOUNDED-MEDIAN while preserving the competitive ratio up to a constant factor. More formally, we show

Theorem 2.1. Given an instance for the LEARN-MEDIAN problem, there exists an algorithm A that maps each sub-instance V t to a sub-instance R t = A ( V t ) , resulting in a LEARN-BOUNDEDMEDIAN instance. If solutions Y 1 , ..., Y T for the new instance of LEARN-BOUNDED-MEDIAN satisfy

<!-- formula-not-decoded -->

where α ≥ 1 and R = R 0 ∪ R 1 . . . ∪ R T , then we have

<!-- formula-not-decoded -->

We will first describe the reduction. When the sub-instance V t arrives at time t , we shall create a corresponding sub-instance R t as follows: consider an offline instance of the k -median problem where the only points available are V t . We use a constant factor approximation algorithm A for the k -median problem to obtain a set of k centers. More formally, we find a constant factor approximation to the following problem:

<!-- formula-not-decoded -->

Let C t be the subset of k centers returned by the above approximation algorithm. Let the points in C t be { c 1 , . . . , c k } and V ( i ) t denote the points in V t that are assigned to c i . Thus,

<!-- formula-not-decoded -->

In the sub-instance R t , the set of points is just R t = C t , and the weight of a point c i ∈ R t is equal to

<!-- formula-not-decoded -->

Note that the sum of weights of the above points is

<!-- formula-not-decoded -->

because | V t | ≥ k +1 , and the minimum distance between any two distinct points in the metric space is 1 due to our scaling. This completes the description of the instance of LEARN-BOUNDED-MEDIAN. We now show that a competitive algorithm for the new instance implies the same for the original LEARN-MEDIAN instance, while losing only a constant factor in the competitive ratio.

To prove this, we will need the following well-known fact about k -median.

Claim A.1. Consider a weighted instance of the k -median problem where the set of clients is given by a subset I of points in the underlying metric space. Let C ⋆ be an optimal solution (i.e., optimal set of k centers) to the instance and ̂ C be an optimal solution where the set of centers is required to be a subset of I . Then,

<!-- formula-not-decoded -->

where w x is the weight of the client x .

Proof. Suppose y was the optimal k -median solution for I , with centres y (1) , . . . , y ( k ) , partitioning I into the corresponding clusters S 1 , . . . S k . Note that y ( i ) is the optimal 1 -median of the cluster S i . For any set of clients X , let w ( X ) = ∑ x ∈ X w x denote the total weight of the set.

<!-- formula-not-decoded -->

We then divide both sides by w ( S i ) to obtain

<!-- formula-not-decoded -->

Since the right hand side is a weighted combination of ∑ x ∈ S i w x d ( x ′ , x ) for different x ′ ∈ S i , this implies that y ′ ( i ) = argmin x ′ ∈ S i ∑ x ∈ S i w x d ( x ′ , x ) satisfies

<!-- formula-not-decoded -->

If we denote y ′ as the collection of the k centres y ′ ( i ) , by adding up the above inequality for each i ∈ [ k ] we obtain the desired result.

<!-- formula-not-decoded -->

We are now ready to prove the correctness of the reduction.

Proof of Theorem 2.1. For a time t , the subset C t is a constant factor approximation to the k -median instance specified by the set of points V t . In fact, we have the additional restriction that C t is a subset of V t . Now, Claim A.1 implies that if Y ∗ t is an optimal solution to this k -median instance, then

<!-- formula-not-decoded -->

Let Y t = { y (1) t , . . . , y ( k ) t } and C t = { c (1) t , . . . , c ( k ) t } . Let V ( j ) t be the subset of V t that is assigned to c ( j ) t , i.e., the points in V t for which the closest center in C t is c ( j ) t . Now we upper bound the competitive ratio of the solution Y t in the instance V t :

<!-- formula-not-decoded -->

Let Y ∗ be the optimal subset of k points that minimizes ∑ T t =1 ρ ( Y ∗ , V t ) . We now lower bound ∑ t ρ ( Y ∗ , V t ) . We again use the partition of V t into V ( j ) t , j ∈ [ k ] and triangle inequality to get:

<!-- formula-not-decoded -->

Consequently, combining (4) and (5) we obtain

<!-- formula-not-decoded -->

where in the last step we used that α ≥ 1 and ρ ( Y ∗ , V t ) ≥ 1 . This proves the desired result.

## B Detailed Proofs for Fractional Algorithm

In this section, we give a detailed description of the online algorithm described in Section 3 for maintaining a fractional solution to an instance of the LEARN-BOUNDED-MEDIAN problem. We also give the missing proofs in the analysis of this algorithm.

Problem Setup: Consider an instance M = ( V, d ) of the LEARN-BOUNDED-MEDIAN problem. There are T +1 rounds in total, and in each round t , a weighted sub-instance R t arrives. This subinstance is specified by a set of k points R t in a metric space M . Note that the points in the metric

space M are revealed over time, and hence, till time t , only the points in R ≤ t := ∪ t ′ ≤ t R t in the metric space have been revealed. Let d t denote | R ≤ t | , and ∆ k d t denote { z ∈ R d t ≥ 0 : || z || 1 = k } where each coordinate i refers to a corresponding point in R ≤ t . By ensuring that a point in R t corresponds to the same coordinate in ∆ d ′ t for all t ′ ≥ t , we have ∆ d t ⊆ ∆ d t ′ for all t, t ′ where t ′ ≥ t . Finally, let K t denote the intersection of ∆ k d t and [0 , 1] d t . In an integral solution, we would require that each time t , the algorithm outputs a subset Y t of k points in R ≤ t -1 ; however, in a fractional solution we relax this requirement as follows - the online algorithm outputs a vector y t ∈ K t -1 for each time t . Given a vector y t ∈ K t -1 , the fractional assignment cost at time t is given by:

<!-- formula-not-decoded -->

where D ( y t , x ) is obtained by fractionally assigning x to an extent of 1 to the closest points in y t . In other words, let v i denote the point corresponding to coordinate i in K t -1 . Then, D ( y t , x ) := min α ∈ [0 , 1] d t -1 ∑ i ∈ [ d t -1 ] α i d ( v i , x ) , where ∑ i α i = 1 and 0 ≤ α i ≤ ( y t ) i .

## B.1 Basic facts about the 1-median problem

In this section, we state some basic facts about the 1 -median problem. Given a weighted set X of points in a metric space M , let opt 1 ( X ) be the optimal weighted 1 -median cost of X , i.e.,

<!-- formula-not-decoded -->

where w x is the weight of the point x . The minimizer y above shall be referred as the optimal 1 -median center of X . The following result shows that opt 1 ( X ) is closely approximated by a weighted sum of the pair-wise distances between the points in X .

Fact B.1. Let X be a weighted set of points in a metric space. Then,

<!-- formula-not-decoded -->

Proof. Let y ∗ be an optimal 1 -median center of X . Then,

<!-- formula-not-decoded -->

Dividing both sides by 2 w ( X ) now gives the desired bound.

The following result states that the weighted distance of X from a point v can be well-approximated by the weighted distance between v and an optimal 1 -median of X .

Fact B.2. Let X be a weighted set of points in a metric space and y ∗ be an optimal 1 -median of X . For any point v in the metric space,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We have:

<!-- formula-not-decoded -->

Computing the optimal 1 -median center y ∗ can be hard as we don't know the entire metric spaceinstead the following result shows that replacing y ∗ by the optimal center amongst X retains the property stated above.

Corollary B.3. Let X be a set of points in a metric space M and v be an arbitrary point in M . Let x ⋆ ∈ X be a point that minimizes ∑ x ∈ X w x d ( x ∗ , x ) . Then,

<!-- formula-not-decoded -->

Proof. Let y ∗ be an optimal 1 -center of X . Using Fact B.1, we have:

<!-- formula-not-decoded -->

Proceeding as in the proof of Fact B.2, we get

<!-- formula-not-decoded -->

The result now follows from Claim A.1,

<!-- formula-not-decoded -->

## B.2 Regularizer and its properties

In this section, we define the regularizer that our online mirror descent algorithm shall use. We also give useful properties of the regularizer.

Definition B.4 ( β -hyperbolic entropy) . [Ghai et al., 2020] For any x ∈ R d and β &gt; 0 , define the β -hyperbolic entropy of x , denoted ϕ β ( x ) as:

<!-- formula-not-decoded -->

Note that ϕ β ( x ) is convex and twice differentiable, and its gradient is given by

<!-- formula-not-decoded -->

Consequently its Hessian is the diagonal matrix,

<!-- formula-not-decoded -->

We now establish strong convexity properties of the function ϕ β .

Definition B.5 (Strongly convex function) . A twice differentiable function f : K → R on a convex set K ⊆ R d is said to be α -strongly convex with respect to a norm || · || on K if for all x, y ∈ K

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma B.6. The function ϕ β is 1 k + βd -strongly convex over ∆ k d with respect to the ℓ 1 norm.

Proof. We use the second characterization in Definition B.5. Consider vectors x ∈ ∆ k d and y ∈ R d such that || y || 1 = 1 . Then

<!-- formula-not-decoded -->

where the first inequality follows from Cauchy-Schwarz.

or equivalently,

Corollary B.7. The function ϕ 1 d is 1 k +1 -strongly convex over ∆ k d with respect to the ℓ 1 norm.

We now recall the notion of Bregman divergence and state some well-known properties [Hazan, 2016].

Definition B.8 (Bregman Divergence) . Let g : K → R be a convex function defined on a convex set K ⊆ R d . Given two points x, y ∈ K , The Bregman divergence B g ( x || y ) with respect the function g is defined as

<!-- formula-not-decoded -->

Note that convexity of g implies that B g ( x || y ) ≥ 0 .

Fact B.9 (Law of Cosines for Bregman Divergence) . Let g : K → R be a convex differentiable function. Then

<!-- formula-not-decoded -->

Fact B.10 (Generalized Pythagorean Theorem for Bregman Divergences) . Let g : K → R be a convex, differentiable function and K ′ ⊆ K be a convex subset. Given x ∈ K,z ∈ K ′ , let y ∈ K ′ be a minimizer of B g ( y || z ) . Then

<!-- formula-not-decoded -->

A direct application of the definition of Bregman divergence and (6) yields:

Claim B.11. Given x, y ∈ R d ,

<!-- formula-not-decoded -->

We now give an upper bound on B ϕ β ( x || y ) .

Lemma B.12. Let x, y ∈ ∆ k d , assume β &lt; 1 and || x || ∞ ≤ 1 . Then

<!-- formula-not-decoded -->

Proof. Consider vectors x and y as in the statement of the Lemma. Using Claim B.11, we get

<!-- formula-not-decoded -->

where we have used the facts that x i ≤ 1 and β &lt; 1 .

## B.3 Algorithm

Let us now recall the fractional online algorithm that we described in Section 3. At each time t , a weighted subset of points R t arrives. At time t = 0 , we initialize y 0 (which is a k -dimensional vector) to any k arbitrary points in R 0 . At step t , we maintain a fractional solution y t on the points R &lt;t . To go from y t to y t +1 , we perform one step of mirror descent based on the β -hyperbolic entropy regularizer defined earlier. Before taking the mirror descent step, for points in R t but not in R &lt;t , we set their fractional values to 0 . This gives the fractional solution y t +1 for the next iteration.

## Algorithm 1: Algorithm for an instance of LEARN-BOUNDED-MEDIAN

<!-- formula-not-decoded -->

## B.4 Analysis

In this section, we analyze the algorithm. Recall that the vector y t lies in R d t -1 . But we shall often consider it to lie in R d t as well by setting ( y t ) i to 0 for all i ∈ [ d t ] \ [ d t -1 ] .

Claim B.13. For each t ∈ [ T ] ,

<!-- formula-not-decoded -->

Proof. Using (6) and the definition of x t +1 , we have that for any i ∈ [ d t ] ,

<!-- formula-not-decoded -->

Using the above claim, we bound the difference between B ϕ 1 /d t ( y t || x t +1 ) and B ϕ 1 /d t ( y t +1 || x t +1 ) .

Claim B.14. For any time t ,

<!-- formula-not-decoded -->

Proof. Using the using the definition of Bregman Divergence, we see that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The desired result now follows from the fact that the maximum value of the function f ( z ) := -1 2( k +1) z 2 + η t ||∇ t || ∞ z is at most kη 2 t ||∇ t || 2 ∞ .

We now analyze the performance of our algorithm with respect to an arbitrary integral solution.

Claim B.15. Let y ∈ K t be an integral vector. Then,

<!-- formula-not-decoded -->

Proof. Using the definition of ∇ t , it suffices to show that for each x ∈ R t :

<!-- formula-not-decoded -->

where M ( x ) and v i are as defined in Algorithm 1 . For ease of notation, relabel the points in R ≤ t arranged in increasing order of distance from x , i.e., d ( x, v 1 ) ≤ d ( x, v 2 ) ≤ . . . ≤ d ( x, v d t ) . Let i ∗ be the smallest index such that ∑ i ≤ i ∗ ( y t ) i ≥ 1 . Observe that M ( x ) = d ( x, v i ∗ ) and α x i = ( y t ) i for all i &lt; i ∗ and is 0 for all i &gt; i ∗ . Since y is an integral vector D ( y, x ) = d ( v i 0 , x ) for some i 0 ∈ [ d t ] . Thus, (7) is equivalent to showing:

<!-- formula-not-decoded -->

Two cases arise: (i) i 0 ≥ i ∗ , or (ii) i 0 &lt; i ∗ . In the first case, y i = 0 for all i &lt; i ∗ and hence, (8) follows easily because d ( x, v i 0 ) ≥ d ( x, v i ∗ ) . In the second case, the r.h.s. of (8) can be expressed as

<!-- formula-not-decoded -->

This proves the desired result.

Phases. We shall divide the timeline into phases, and shall bound the cost incurred by the algorithm in a phase with respect to an off-line solution that only uses points revealed till the beginning of the phase. The following result bounds the cost incurred during a time interval:

Lemma 3.2. Consider a time interval I := [ t a , t b ] ⊆ [1 , T ] , and let z be an integral vector in K t a . Then, √

<!-- formula-not-decoded -->

Proof. Consider a time t ∈ I . Using Claim B.15, we get:

<!-- formula-not-decoded -->

Summing this over all t ∈ I , and using Claim B.13, we see that ∑ t ∈ I ( Cost t ( y t ) -Cost t ( z )) is at most

<!-- formula-not-decoded -->

Using Lemma B.12 and Claim B.14, the above expression is at most (note that d t is a non-decreasing function of t )

<!-- formula-not-decoded -->

Observe that ||∇ t || ∞ ≤ ( k +1)∆ , and thus G t ≤ ( k +1)∆ . Indeed, by definition of ∇ t ,

<!-- formula-not-decoded -->

because the total weight of the points in R t is at most ( k +1) . Using this observation and the fact that η t = 1 G t √ t , (9) is at most

<!-- formula-not-decoded -->

The desired result now follows from the fact that d T ≤ Tk .

We now define the phases. Let y ∗ ∈ K T be an integral solution defined by a set C of k centers. Let V := R 1 ∪ R 2 . . . ∪ R T denote the set of points that arrive over the last T timesteps. Let the centers in C be c (1) , . . . , c ( k ) . The set C partitions V into k subsets - let V ( i ) be the subset of V for which the closest center is c ( i ) (we break ties arbitrarily). For a non-negative integer j , let j i be the smallest time t such that the total weight of the points in V ( i ) ∩ ( R 1 ∪ . . . R t ) (i.e., points in V ( i ) arriving by time t ) exceeds k · 2 j . Let w ( V ( i ) ) denote the total weight of the points in V ( i ) . Define the set P := { j i : i ∈ [ k ] , j ∈ [log( w ( V ( i ) ))] } ⊆ [1 , T ] . Observe that

<!-- formula-not-decoded -->

because at each time t , the total weight of the arriving points is at most ( k +1) . We shall treat the indices in P as subset of [1 , T ] , and hence, these partition the timeline into | P | +1 phases. We shall apply Lemma 3.2 for each of these phases.

In order to apply this lemma, we need to specify a candidate solution z p for each of these phases p . We need some more definitions to describe this integral solution. For each i ∈ [ k ] and index j , define V ( i ) j := V ( i ) ∩ ( R ≤ j i \ R ≤ j i -1 ) , i.e., the points in V ( i ) whose corresponding arrival time lies in the range ( j i -1 , j i ] . We shall often refer to the subset V ( i ) j as a bucket of V ( i ) .

For each of the set of points V ( i ) j , let x ( i ) j denote the optimal 1 -median solution where the center is restricted to lie in V ( i ) j only, i.e.,

<!-- formula-not-decoded -->

For a phase p and index i ∈ [ k ] , the time slots in p are contained in a single bucket V ( i ) j of V ( i ) . Let ℓ ( i ) p denote the index corresponding to x ( i ) j -1 in K T , i.e., the optimal 1 -median solution of the last bucket in V ( i ) that does not intersect this phase. We are now ready to define the candidate integral solution z ( p ) for a phase p .

Fix a phase p and let s p denote the start time of this phase. Define an integral vector z ( p ) ∈ K s p as follows: for each i ∈ [ k ] , we set z ( p ) j = 1 where j = ℓ ( i ) p ; all other coordinates of z ( p ) are 0 - in case there is no bucket in V ( i ) ending before time s p , we set z ( p ) to be an arbitrary point. In other words, this solution selects k points, namely, the 1 -median center of the last bucket in V ( i ) that ends before s p . This completes the description of the vector z ( p ) corresponding to a phase p . Lemma 3.2 shows that the total cost incurred by the algorithm during a phase p is close to that incurred by the fixed solution z ( p ) . Thus, it remains to show that the total cost incurred by z ( p ) during a phase is close to that incurred by y ∗ .

Lemma B.16. Consider any i ∈ [ k ] and index j ≥ 1 . Then,

<!-- formula-not-decoded -->

Proof. Consider a point x ∈ V ( i ) j . Using Corollary B.3, we obtain

<!-- formula-not-decoded -->

where the last inequality follows from the fact that w ( V ( i ) j -1 ) ≥ k · 2 j -1 . Thus,

<!-- formula-not-decoded -->

Since the weight of the points arriving at any particular time is at most k +1 , we have the bound w ( V ( i ) j -1 ∪ V ( i ) j ) ≤ k · 2 j -1 + k · 2 j +2 k +2 ≤ 7 k · 2 j -1 . Consequently, the r.h.s. above is at most

<!-- formula-not-decoded -->

which is at most 42 · opt 1 ( V ( i ) j -1 ∪ V ( i ) j ) (using Fact B.1).

We are now ready to bound the total cost incurred by the solution z ( p ) during phase p .

Lemma 3.3. For each p ∈ P , let the start and end time of phase p be denoted s p and e p respectively. Then, there exist solutions z ( p ) such that

<!-- formula-not-decoded -->

where y ∗ is an arbitrary integral vector in K T .

Proof. For a point x ∈ V ( i ) , let p x be the phase containing the arrival time of x . Then,

<!-- formula-not-decoded -->

For x ∈ V ( i ) 0 , d ( x ( i ) ℓ ( i ) p , x ) ≤ ∆ . For j ≥ 1 and x ∈ V ( i ) j , d ( x ( i ) ℓ ( i ) p , x ) = d ( x ( i ) j -1 , x ) . Thus, the r.h.s. of (12) is upper bounded by

<!-- formula-not-decoded -->

Combining Lemma 3.2 and Lemma 3.3, since there are at most O ( k log T ) phases, we get: Theorem B.17. Let y ∗ be an arbitrary integral vector in K T . Then,

<!-- formula-not-decoded -->

## C Details about the Rounding Algorithms

In this section, we give a detailed description of the online rounding algorithms described in Section 4 for the LEARN-BOUNDED-MEDIAN problem. Let y t be the fractional solution computed by Algorithm 1 before the arrival of the t th instance R t . Our rounding algorithms will take the fractional solution y t as input and return an integer solution Y t with exactly k centers. The rounding loss is defined as the ratio of the (expected) cost of the integer solution Y t to that of the fractional solution y t .

We give two rounding algorithms. The first algorithm is deterministic and has a rounding loss of O ( k ) . The second algorithm is randomized and has only O (1) rounding loss. These are adaptations of existing rounding algorithms for k -median [Charikar et al., 1999, Charikar and Li, 2012]; the main difference is that we need to bound the rounding loss for points we haven't seen as well.

## C.1 Deterministic Rounding Algorithm

Initially, there are no centers in Y t . The algorithm considers each point in R &lt;t = R 0 ∪ . . . R t -1 in non-decreasing order of their connection costs D ( y t , i ) , and decides whether to open a center at the current point. A new center is opened at the current point (call it i ) if the connection cost of i to the previously opened centers in Y t exceeds (2 k +2) times its connection cost in the fractional solution y t . I.e., Y t ← Y t ∪ { i } if D ( Y t , i ) &gt; (2 k +2) · D ( y t , i ) .

We show in the next lemma that the algorithm produces a feasible solution, i.e., at most k centers are opened in Y t .

Lemma C.1. The deterministic algorithm produces a feasible solution, i.e., opens at most k centers.

Proof. For j ∈ Y t , let B j = { x ∈ R &lt;t : d ( j, x ) ≤ ( k +2) · D ( y t , j ) } be the ball around j of radius k +2 times the fractional cost of j . We will show that

- (a) The fractional mass on points in B j in y t is strictly greater than 1 -1 k +1 .
- (b) The balls B j are disjoint for different j .

Given these properties, if | Y t | ≥ k +1 , then the total fractional mass y t in the balls B j for j ∈ Y t is strictly greater than k , which is a contradiction.

We show the two properties. For property (a), if the fractional mass y t in B j is at most 1 -1 k +1 , then j must be served at ar least 1 k +1 fraction by centers that are outside B j . It follows that D ( y t , j ) ≥ 1 k +1 · ( k + 2) · D ( y t , j ) , which is a contradiction. For property (b), suppose B j , B j ′ overlap where j ′ &gt; j , i.e., D ( y t , j ′ ) ≥ D ( y t , j ) . Then, d ( j, j ′ ) ≤ 2( k +2) · D ( y t , j ′ ) , which means that the algorithm will not open a center at j ′ , a contradiction.

For the rounding loss, note that the algorithm explicitly ensures that for any i ∈ R &lt;t , we have D ( Y t , i ) ≤ (2 k +2) · D ( y t , i ) . The next claim bounds the rounding loss for points j ∈ R t .

Lemma C.2. For any point j ∈ R t , we have D ( Y t , j ) ≤ (4 k +3) · D ( y t , j ) .

Proof. Let i be the point in R &lt;t that is closest to j . Clearly,

<!-- formula-not-decoded -->

since the entire fractional mass of y t is on points in R &lt;t . Moreover,

<!-- formula-not-decoded -->

since the RHS is at most the cost of serving i using j 's fractional centers. Adding the two inequalities, we get D ( y t , j ) ≥ D ( y t ,i ) 2 . Note that there is a center in Y t that is within distance (2 k +2) · D ( y t , i ) of i since i ∈ R 1 ∪ R 2 ∪ . . . ∪ R t -1 . The distance from this center to j is at most

<!-- formula-not-decoded -->

This implies that the rounding loss of the deterministic rounding algorithm is 4 k +3 .

## C.2 A Randomized Rounding Algorithm

The rounding algorithm has two phases. The first phase selects a set of centers ¯ Y t that might be larger than k . This phase is deterministic. The second phase subselects at most k centers from ¯ Y t to form the final solution Y t . This phase is randomized. The guarantee that at most k centers are selected in Y t holds deterministically, but the cost of the solution will be bounded in expectation. We desribe these two phases below.

Phase 1: This phase is similar to the deterministic algorithm in the previous section but with different parameters. Initially, there are no centers in ¯ Y t . The algorithm considers each point in R &lt;t in non-decreasing order of their connection costs D ( y t , i ) , and decides whether to open a center at the current point. A new center is opened at the current point (call it i ) if the connection cost of i to the previously opened centers in ¯ Y t exceeds 4 times its connection cost in the fractional solution y t . I.e., ¯ Y t ← ¯ Y t ∪ { i } if D ( Y t , j ) &gt; 4 · D ( y t , i ) .

Phase 2: For any center i ∈ ¯ Y t , let r i be the distance to its closest center in ¯ Y t . We define the weight of i , denoted w i , as the the total fractional mass in y t in the open ball of radius r i / 2 centered at i . I.e.,

<!-- formula-not-decoded -->

We start by creating a matching on the centers in ¯ Y t . The first matched pair is the closest pair of centers in ¯ Y t . The matching continues iteratively by pairing the two closest unmatched centers in ¯ Y t

in each step. Eventually, either all centers in ¯ Y t are in matched pairs or there is a solitary center that goes unmatched. For any center i ∈ ¯ Y t that is in a matched pair, we denote the center it got matched to by i ′ .

Next, we order the matched pairs arbitrarily to create an induced ordering on the points that satisfies the property that i, i ′ are adjacent. If there is an unmatched point, it is added at the end of this order. Let us call the ordering i 1 , i 2 , . . . , i | ¯ Y t | . We now associate an interval with each point as follows:

for point i s , its associated interval is I s := [ ∑ s -1 j =1 w i j , ∑ s j =1 w i j ) , i.e., the points occupy adjacent intervals on the real line in the given order. Next, we generate a value θ uniformly at random from [0 , 1) , and add to Y t every point i s ∈ ¯ Y t such that a + θ ∈ I s for some non-negative integer a .

First, we establish that the algorithm is correct, i.e., | Y t | ≤ k .

Lemma C.3. The output Y t of the randomized rounding algorithm has at most k centers.

Proof. It is sufficient to show that the sum of weights of the centers in ¯ Y t is at most k . In turn, this follows if we show that the open balls B i := { j : d ( i, j ) &lt; r i / 2 } are disjoint for different i ∈ ¯ Y t . Suppose not; let d ( i 1 , i 2 ) &lt; r i 1 + r i 2 2 for i 1 , i 2 ∈ ¯ Y t . Then, r i 1 ≤ d ( i 1 , i 2 ) and r i 2 ≤ d ( i 1 , i 2 ) by definition of r i . This is a contradiction.

Next, we bound the rounding loss of the algorithm. Note that for any point j ∈ R 1 ∪ R 2 ∪ . . . ∪ R t -1 , we have the following explicit property from the construction of ¯ Y t :

<!-- formula-not-decoded -->

Another important property sets a lower bound on the weight of any point:

Claim C.4. For any point i ∈ ¯ Y t , we have

<!-- formula-not-decoded -->

Therefore, for any matched pair i, i ′ ∈ ¯ Y t , at least one of i, i ′ is in Y t .

Proof. For the first inequality, note that D ( y t , i ) ≥ 0 · w i +( r i / 2) · (1 -w i ) since at least 1 -w i fraction of the service for i in the fractional solution y t comes from outside the open ball of radius r i / 2 centered at i . Re-arranging this inequality gives the first inequality.

For the second inequality, let j be the closest center to i in ¯ Y t at the time that i was considered in phase 1 of the rounding algorithm. Since i was added to ¯ Y t , it must be that d ( i, j ) &gt; 4 · D ( y t , i ) . Moreover, any center ℓ added to ¯ Y t later in the algorithm must satisfy d ( i, ℓ ) &gt; 4 · D ( y t , ℓ ) ≥ 4 · D ( y t , i ) . Therefore, r i &gt; 4 · D ( y t , i ) , which implies the second inequality.

Therefore, for any matched pair i, i ′ , their cumulative weight w i + w i ′ &gt; 1 . This implies that at least one of i, i ′ will be added to Y t in phase 2.

We use these property to bound the rounding loss for Y t :

Lemma C.5. The following properties hold:

1. For any point i ∈ ¯ Y t , we have E [ D ( Y t , i )] ≤ 4 · D ( y t , i ) .
2. For any point j ∈ R 0 ∪ . . . ∪ R t -1 , we have E [ D ( Y t , j )] ≤ 8 · D ( y t , j ) .
3. For any j ∈ R t , we have E [ D ( Y t , j )] ≤ 17 · D ( y t , j ) .

Proof. We prove each of these properties separately. Note that Pr[ i ∈ Y t ] ≥ min( w i , 1) in phase 2 of the algorithm. Using Claim C.4, we have

<!-- formula-not-decoded -->

Moreover, with probability 1 , we have D ( Y t , i ) ≤ 2 r i . This follows from two cases: (a) If i is matched to its closest point in ¯ Y t , then by Claim C.4, at least one of i, i ′ is in Y t . (b) Otherwise, suppose i is unmatched or it is matched to i ′ which is not its closest point in ¯ Y t . Let j be closest point

to i in ¯ Y t . Then, the fact that j was matched to j ′ and not to i implies that d ( j, j ′ ) ≤ d ( i, j ) = r i . By triangle inequality, d ( i, j ′ ) ≤ 2 r i , and at least one of j, j ′ is in Y t by Claim C.4.

Combining the two observations above, we get

<!-- formula-not-decoded -->

Next, we show the second property. Consider a point j ∈ R &lt;t \ ¯ Y t . since j was not added to ¯ Y t , there exists i ∈ ¯ Y t such that D ( ¯ Y t , i ) ≤ D ( ¯ Y t , j ) and d ( i, j ) ≤ 4 · D ( y t , j ) . Therefore,

<!-- formula-not-decoded -->

where we used the first property in the second to last inequality.

Finally, we show the third property. Consider a point j ∈ R t . Let i be the point in R &lt;t that is closest to j . Clearly,

<!-- formula-not-decoded -->

since the entire fractional mass of y t is on points in R &lt;t . Moreover,

<!-- formula-not-decoded -->

since the RHS is at most the cost of serving i using j 's fractional centers. Adding the two inequalities, we get D ( y t , i ) ≤ 2 · D ( y t , j ) . Therefore,

<!-- formula-not-decoded -->

where we used the second property in the second to last inequality.

This implies that the rounding loss of the randomized rounding algorithm is 17 .

## D Final Bounds

We now put together our reduction, fractional algorithm, and rounding procedure to obtain deterministic and randomized algorithms for LEARN-MEDIAN. By combining Theorem 2.1, Theorem B.17 and Lemma C.2, we have the following theorem.

Theorem D.1. We give a deterministic algorithm for LEARN-MEDIAN with the following performance bound:

<!-- formula-not-decoded -->

where Y ∗ is the best fixed solution in hindsight.

Similarly, by combining Theorem 2.1, Theorem B.17 and Lemma C.5, we have the following theorem.

Theorem D.2. We give a randomized algorithm for LEARN-MEDIAN with the following expected performance bound:

<!-- formula-not-decoded -->

where Y ∗ is the best fixed solution in hindsight.

## E Detailed Experiments

In this section, we give a detailed description of the experiments in Section 5. We use a heuristic to improve the performance of the rounding algorithms in all our experiments, where we do a binary search over the best threshold on the distance to the already opened centers. For the deterministic rounding algorithm for LEARN-BOUNDED-MEDIAN, we open a new center at i if its distance to the already opened centers exceeds (2 k +2) D ( y t , i ) . We can always slacken this factor of (2 k +2) to

something lower as long as the total number of centers opened does not exceed k , without worsening the theoretical approximation ratio. Similarly in the randomized rounding algorithm for LEARNBOUNDED-MEDIAN, we can slacken the threshold of 4 D ( y t , i ) in the initial filtration step, as long as the fractional mass in the balls we later consider are at least 1 / 2 .

We compute optimal-in-hindsight solutions for the instances using Gurobi's ILP solver, accessed under an academic license. Unless mentioned otherwise, experiments were conducted using Google Colab (code available at https://github.com/neurips2025-colab/neurips2025 ), using four Intel ® Xeon ® CPUs (2.20 GHz) with 13 GB of RAM each, running in parallel over about 24 compute hours. In all our experiments, the underlying metric is the Euclidean metric. Unless mentioned otherwise, we plot the ratio between ∑ t τ =1 ρ ( Y τ , V τ ) and ∑ t τ =1 ρ ( Y ∗ , V τ ) in our approximation ratio plots, where Y ∗ is the optimal-in-hindsight solution for the entire input sequence.

Uniform Square: In this example, the underlying metric space consists of 400 uniformly random points in the unit square [0 , 1] × [0 , 1] . 10 points arrive each round, chosen uniformly at random, and we run the experiment with a time horizon of T = 1000 . We compare the cost of the optimal solution with the deterministic and randomized algorithms, as well as the intermediate fractional solution that we maintain. We generate 10 random instances in total, and report the standard deviation and mean of the approximation ratio over time. For the randomized algorithm, we first average the approximation ratio for each instance over 5 random runs of the rounding algorithm. We run this experiment with k = 2 , 3 , 6 .

As we see in Fig. 3, both the randomized and deterministic algorithms converge to natural solutions, distributed roughly uniformly over the unit square. The approximation ratio also approaches 1 , especially after the influence of the initial additive regret declines.

<!-- image -->

Time

Figure 3: (Uniform Square): The optimal (black plus), deterministic (blue cross), and randomized (red diamond) solutions for one of the random instances (left) and approximation ratios - avg. and std. dev. over 10 random instances (right) for k = 2 , 3 , 6 .

Uniform Rectangle: In this example, the underlying metric space consists of 400 uniformly random points in the rectangle [0 , 1] × [0 , 10] . 10 points arrive each round, chosen uniformly at random, and we run the experiment with a time horizon of T = 1000 . We compare the cost of the optimal solution with the deterministic and randomized algorithms, as well as the intermediate fractional solution that we maintain. We generate 10 random instances in total, and report the standard deviation and mean of the approximation ratio over time. For the randomized algorithm, we first average the approximation ratio for each instance over 5 random runs of the rounding algorithm. We run this experiment with k = 2 , 3 , 6 .

As we see in Fig. 4, both the randomized and deterministic algorithms converge to natural solutions, distributed roughly uniformly over the rectangle. The approximation ratio also approaches 1 , especially after the influence of the initial additive regret declines.

<!-- image -->

Time

Figure 4: (Uniform Rectangle): The optimal (black plus), deterministic (blue cross), and randomized (red diamond) solutions for one of the random instances (left) and approximation ratios - avg. and std. dev. over 10 random instances (right) for k = 2 , 3 , 6 .

Multiple Clusters: In this example, the underlying metric space consists of k clusters with center in the unit square [0 , 1] × [0 , 1] . The cluster centers are chosen uniformly at random from the unit square, and we then generate points in a radius of 0 . 05 around each cluster center, for a total of about 400 points. 20 points arrive each round, chosen uniformly at random, and we run the experiment with a time horizon of T = 1000 . We compare the cost of the optimal solution with the deterministic and randomized algorithms, as well as the intermediate fractional solution that we maintain. We generate 10 random instances in total, and report the standard deviation and mean of the approximation ratio over time. For the randomized algorithm, we first average the approximation ratio for each instance over 5 random runs of the rounding algorithm. We run this experiment with k = 4 , 8 , 12 , 16 .

Figure 5: (Multiple Clusters): the optimal (black plus), deterministic (blue cross), and randomized (red diamond) solutions for one of the random instances (left) and approximation ratios - avg. and std. dev. over 10 random instances (right) for k = 4 , 8 , 12 , 16 .

<!-- image -->

Once again, as we see in Fig. 5, both the randomized and deterministic algorithms converge to natural solutions, with centers distributed over the different clusters. The approximation ratio also approaches 1 , especially after the influence of the initial additive regret declines.

Uniform Hypersphere: In this example, k = 1 and the underlying metric space consists of 400 points: one point is the origin and the rest are chosen uniformly at random from the unit hypersphere. We run the experiment with a time horizon of T = 2000 . For the first 100 rounds, each instance consists of 10 points chosen at random from the boundary of the hypersphere. However, in the next round we also include the origin in the instance, thus revealing the origin to our algorithm. We then

Figure 6: (Uniform Hypersphere): Fractional mass of algorithm over time at the center (avg. and std. dev. over 10 runs), d = 2 , 8

<!-- image -->

continue generating instances as before. We do this for for 2 different dimensions of the hypersphere ( d = 2 , 8 ). We generate 10 random instances in total, and report the standard deviation and mean of the total mass that the fractional algorithm places on the origin.

As we see in Fig. 6, the fractional mass is initially fully on the boundary, but once we introduce the origin the mass slowly shifts to being just on the origin. This happens because the origin is a better solution for the instance sequence compared to any solution solely on the boundary. Moreover, this transition is more rapid for higher dimensions, as typical distance between 2 random points on the hypersphere increases with dimension. On the other hand, the distance from the origin to the hypersphere boundary is always 1 .

Oscillating Instances: In this example, the underlying metric space consists of 2 clusters with 10 points each. The 2 clusters are generated by picking points uniformly at random from squares of side length 0 . 4 centered at (0 , 0) and (1 , 0) respectively, and we run the experiment with a time horizon of T = 3 5 . The instance sequence alternates between the first and second cluster, where we give the first cluster for the first 3 rounds, then the second cluster up to the 9 th round, then the first cluster up to the 27 th round, and so on. One difference in this example is that we compare our performance against the dynamically optimal fractional solution as opposed to a fixed optimal-in-hindsight integer solution: that is, for each time step we compare against the best fixed solution for the instances so far. That is, we plot the ratio between ∑ t τ =1 ρ ( y τ , V τ ) and ∑ t τ =1 ρ ( OPT t , V τ ) , where OPT t is the best fixed fractional solution for instances V 1 , ..., V t and y τ is the intermediate fractional solution we maintain. We also study how the fractional mass of the (dynamic) optimal solution and the algorithm in the first cluster changes over time in order to obtain a qualitative understanding of how the algorithm shifts its centers towards new points that arrive. We run this experiment with k = 1 , 2 , 3 , 4 .

As we see in Fig. 7, k = 1 , the algorithm starts shifting fractional mass before the optimal solution. Consequently, our algorithm ends up outperforming the optimal solution in the time horizon that we consider, as seen from the approximation ratio plots. For k = 2 , 3 , 4 , we observe that the algorithm quickly moves a unit of mass to the cluster that is currently arriving, and then slows down as the gain in increasing the fractional mass on the cluster is minimal.

Scale Changes: In this example, the underlying metric space consists of 5 clusters with 10 points each. The clusters are generated by picking points uniformly at random from squares of side length 0 . 4 centered at ( ⌊ 10 i -1 ⌋ , 0) for i = 0 , 1 , 2 , 3 , 4 , and we run the experiment with a time horizon of T = 3 5 . In this example, we have 5 clusters with 10 points each, with a time horizon of T = 3 5 . The instance sequence iterates through the clusters, where we give the first cluster for the first 3 rounds, then the second cluster up to the 9 th round, and so on. We study how the fractional mass of the algorithm in the cluster that is currently arriving changes over time in order to obtain a qualitative understanding of how the algorithm shifts its centers toward new far away points that arrive. The approximation ratios plotted are between ∑ t τ =1 ρ ( y τ , V τ ) and ∑ t τ =1 ρ ( OPT t , V τ ) , where OPT t is the best fixed fractional solution for instances V 1 , ..., V t and y τ is the intermediate fractional solution we maintain. We run this experiment with k = 1 , 2 , 3 .

As we see in Fig. 8, we once again see that for k = 1 , the fractional algorithm ends up outperforming the optimal solution. For k = 2 , 3 , we observe that the algorithm quickly moves a unit of mass to the

121

<!-- image -->

Time

Time

Figure 7: (Oscillating Instances): Fractional mass of algorithm and OPT over time in one of the clusters (left) and corresponding dynamic approximation ratios (right), for k = 1 , 2 , 3 , 4 .

cluster that is currently arriving, and then slows down rapidly as the gain in increasing the fractional mass on the cluster is very minimal. The change is even more rapid in this case compared to the previous experiment due to the large scale changes which lead to a larger gradient norm, and hence a more conservative learning rate.

One question that arises from the approximation ratio plots for k = 2 , 3 is the large approximation factor, but this can be explained by the additive term that is quite large in this case. For k &gt; 1 , the dynamic optimal solution immediately shifts a center from the previous clusters to the new one as it is a much better solution for the sequence so far, while the algorithm accrues a large cost until it shifts a mass of one into the new cluster. To illustrate this, we run the experiment again for k = 2 , 3 with 4 clusters and a time horizon of T = 50000 (we only use 4 clusters this time due to computational reasons, the time required for the additive term to be insignificant grows with ∆ ), where we continue giving the 4 th cluster even after the first 81 rounds. We see that the approximation ratio eventually approaches 1 , see Fig. 9.

<!-- image -->

Time

Time

Figure 8: (Scale Changes): Fractional mass of algorithm over time in one of the clusters (left) and corresponding dynamic approximation ratios (right), for k = 1 , 2 , 3 .

<!-- image -->

Time

Time

Figure 9: (Scale Changes): Dynamic approximation ratio plots with T = 50000 , for k = 2 , 3 .

Small Drifts: In this example, the metric space consists of 2500 points, and we run the experiment with a time horizon of T = 250 . The underlying instances are created by randomly drawing 10 points (out of which 5 random points arrive in a round) in a disc centered around the origin, but then shifting the center of the disc to the right by 0 . 02 for each instance. We compare the cost of the optimal solution with the deterministic and randomized algorithms, as well as the intermediate fractional solution that we maintain. We generate 10 random instances in total, and report the standard deviation and mean of the approximation ratio over time. For the randomized algorithm, we first average the approximation ratio for each instance over 5 random runs of the rounding algorithm. We run this experiment with k = 1 , 2 , 3 .

As we see in Fig. 10, the final average approximation ratios were less than 2 in all cases. (In contrast, if the algorithm were to use the initial solutions throughout, i.e., does not automatically adapt to the

drift, then the cost ratios in each round are around 5 -15 .) The approximation ratio is small initially as we compare against the optimal-in-hindsight solution for the entire input sequence.

<!-- image -->

Time

Figure 10: (Small Drifts): The optimal (black plus), deterministic (blue cross), and randomized (red diamond) solutions for one of the random instances (left) and approximation ratios - avg. and std. dev. over 10 random instances (right) for k = 1 , 2 , 3 .

Finally, note that the time complexity of our learning algorithm (besides the reduction step) is O ( k 2 T 3 ) , independent of the number of points n . To empirically confirm scalability, we ran experiments with n = 1000 , 2000 , . . . , 6000 while keeping k, T fixed ( k = 5 , T = 100 ) and observed that the running time of the learning algorithm (besides the reduction step) was always 2 -3 seconds on Google Colab ( 2 . 20 GHz Intel® Xeon® CPU, 51 GB RAM).

## F Lower Bounds

In Theorem D.1 we gave an efficient deterministic algorithm for LEARN-MEDIAN with a multiplicative O ( k ) approximation factor and sublinear regret. We now give an information theoretic lower bound that states that any algorithm for LEARN-MEDIAN with sublinear regret must incur an Ω( k ) multiplicative factor loss.

Theorem F.1. Any deterministic algorithm for the LEARN-MEDIAN problem cannot achieve a multiplicative o ( k ) approximation guarantee with an additive regret term that is sublinear in T . That is, any bound as below is impossible

<!-- formula-not-decoded -->

where Y is any fixed solution for LEARN-MEDIAN . Moreover, this holds even when the metric space is known upfront.

Proof. The lower bound result is inspired by the competitive ratio lower bound for deterministic online paging. Suppose for the sake of contradiction that there is an online algorithm A that outputs a solution Y t at time t and satisfies the condition (14). We now describe our hard instance M = ( V, d ) . The underlying metric space M consists of n = 2( k +1) points. It contains k +1 different clusters of 2 points each, such that the distance between any 2 points belonging to different clusters is ∆ ≫ 1 . The distance between any 2 points in the same cluster is 1 .

We construct our sequence of sub-instances V 0 , . . . , V T in an adversarial fashion: consider a round t , and let Y t be the set of k centers outputted by A in round t . Since there are k +1 clusters in the metric space, it follows that there is always at least 1 cluster which does not overlap with Y t . Thus, given Y t , define V t to consist of this cluster and any other k -1 clusters. For the round zero, we just take V 0 to be the entire metric space.

Clearly, A must suffer a cost of at least 2∆ k for each round after round zero. This is because the optimal solution for V t places exactly 1 point in each of the k clusters with a total connection cost of k , while Y t must have a connection cost of at least 2∆ as it doesn't place any center in one of the clusters present in V t .

Let us now examine the optimal fixed solution in hindsight for this adversarial input sequence. The optimal solution places 1 center each in the k clusters that appear the most frequently. Thus, it suffers a total cost at most 2∆ T ( k +1) k . Putting it together and using (14), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which gives a contradiction as T →∞ as k, n, ∆ are fixed parameters of the underlying metric space.

We now extend the above result to the setting where the online algorithm is randomized: again, we show that the result in Theorem D.2 is essentially tight, i.e., any randomized algorithm with sublinear additive regret must incur a constant factor multiplicative loss.

Theorem F.2. Any (randomized) algorithm for the LEARN-MEDIAN problem cannot achieve a multiplicative (1 + ε ) approximation guarantee with an additive regret term that is sublinear in T . That is, any bound as below is impossible

<!-- formula-not-decoded -->

where 0 ≤ ε &lt; 1 is an arbitrary constant and Y is any fixed solution for LEARN-MEDIAN .

or equivalently,

Proof. Suppose for the sake of contradiction that there is an algorithm A that satisfies (15) for some constant 0 ≤ ε &lt; 1 . We first describe our hard instance M = ( V, d ) . Our metric space M consists of n = mk points, m ≫ k . It contains k different clusters of m points each, such that the distance between any 2 points belonging to 2 different clusters is ∆ ≫ k . The clusters themselves are star graphs with m -1 leaves and a center point, the distance between any 2 leaves being 2 and the center-leaf distance being 1 .

We construct our sequence of sub-instances V 0 , . . . , V T in an adversarial fashion. In round zero, we just take V 0 to be all the leaves. For t = 1 , . . . , T -1 , V t consists of 2 randomly chosen leaves from each cluster. For the last round, we take V T to be all the cluster centers and one other leaf.

We claim that the algorithm A must suffer an expected loss of at least 4(1 -3 /m ) in each round from 1 to T -1 . Let Y t be the subset of size k selected by A in round t . If Y t does not contain a point from each of the clusters it has a connection cost of at least ∆ for round t ; otherwise Y t contains exactly one point from each cluster. For a given cluster, the probability that the point in Y t doesn't belong to the two points of this cluster in V t is at least (1 -1 /m )(1 -2 /m ) ≥ 1 -3 /m . Thus linearity of expectation implies that the expected connection cost of Y t is at least min { ∆ , 4 k (1 -3 /m ) } ≥ 4 k (1 -3 /m ) . On the other hand, the optimal connection cost for V t , 1 ≤ t ≤ T -1 is k , corresponding to picking 1 of the chosen leaves from each of the clusters.

Let us now examine the optimal fixed solution in hindsight for this adversarial input sequence. We claim that it suffers a total loss of at most 2 T . Indeed, consider the fixed solution Y that picks each of the cluster centers, it suffers a loss of 1 1 = 1 in the last round as it is the optimal solution for V T , and suffers a loss of at most 2 k k = 2 for each round from 1 to T -1 . This is because its connection cost is at most 2 k , and the optimal connection cost for V t , 1 ≤ t ≤ T -1 is k , corresponding to picking 1 of the chosen leaves from each of the clusters. Putting it together, we obtain

<!-- formula-not-decoded -->

or equivalently,

<!-- formula-not-decoded -->

which gives a contradiction by taking m large enough that 2 -6 /m &gt; 1 + ϵ and then taking T →∞ .

Finally, we show that an additive loss of Ω( k ∆) is unavoidable if the multiplicative loss is a constant.

Theorem F.3. Any (randomized) algorithm for the LEARN-MEDIAN problem must suffer Ω( k ∆) additive loss even when allowing for O (1) multiplicative error. That is, any bound as below is impossible

<!-- formula-not-decoded -->

where Y is any other fixed solution for LEARN-MEDIAN .

Proof. Suppose for the sake of contradiction that there is an algorithm A satisfying (16). We first describe our hard instance M = ( V, d ) . The metric space M consists of n = mk points, m ≫ k . It contains k different clusters of m points each, such that the distance between any 2 points belonging to different clusters is ∆ ≫ 1 . The distance between any 2 points in the same cluster is 1 .

Take T = k -1 and V t to be the ( t +1) th cluster for 0 ≤ t ≤ k -1 . The algorithm A must suffer a loss of at least m ∆ m -k ≥ ∆ / 2 for each round t after round zero. This is because it has not seen the ( t +1) th cluster at time t and hence cannot place any center there, thus suffering a connection cost of at least m ∆ , and the optimal solution for V t consists of k points from the same cluster, with a total connection cost of m -k . Thus, the total loss suffered by the algorithm is at least ( k -1)∆ 2 .

On the other hand, the optimal fixed solution in hindsight places 1 center in each of the k clusters, suffering a loss of m -1 m -k ≤ 2 each round after round zero. Thus, the total loss suffered by the optimal solution is at most 2( k -1) . Putting it together, we obtain

<!-- formula-not-decoded -->

which gives a contradiction as ∆ →∞ , as desired.

Approximate Follow-the-Leader So far we have given information theoretic lower bounds on the multiplicative loss and the additive regret term for any online algorithm. Follow The Leader (FTL) is another commonly used algorithm - at each time t , output Y t that minimizes the total cost ∑ t ′ &lt;t cost t ′ ( Y t ) . In our setting, computing the optimal cost (i.e., cost of an optimal k -median instance) is NP-hard. Thus, a natural extension of this approach would be Approximate FTL: at each time t , output Y t that is a constant factor approximation to the objective function ∑ t t ′ =1 ρ ( Y, V t ′ ) . We show strong lower bounds for this algorithm even when k = 1 :

Theorem F.4. Approximate FTL cannot achieve a o (log ∆ / log log ∆) multiplicative approximation guarantee with sublinear regret even when k = 1 , i.e., any bound as below is impossible:

<!-- formula-not-decoded -->

where Y t is the set of centers outputted by Approximate FTL at time t and Y is an arbitrary set of k centers.

Proof. For ease of notation, assume that Approximate FTL uses a 4 -approximation algorithm for the (off-line) k -median problem - the argument remains similar for any constant approximation algorithm. We shall also fix k = 1 .

We now describe the metric space M . M consists of a star graph with 2 λ leaves, where λ is a parameter that we shall fix later. At each of the leaves, we have two vertices that are at distance 2 from each other (thus, M is given by the shortest path metric on a two level tree, where the root has 2 λ children and every node at the first level has two children). Let the children of the root r be labeled v 1 , . . . , v 2 λ . The two children of v i are labeled v 1 i and v 2 i . The edge ( r, v i ) has length ∆ i = ∆ λ i , where ∆ = λ 2 λ +1 (assume λ ≫ 1 ).

At time t = 0 , we give the entire metric space as the instance. The subsequent input sequence is divided into 2 λ phases. For each time t in phase h ∈ [2 λ ] , the input V t consists of the two points v 1 h and v 2 h . Phase h lasts for T h := λ h T 0 timesteps, where T 0 ≫ 1 . Note that T h ∆ h = T 0 ∆ for all h ∈ [2 λ ] .

The off-line solution Y places a center at the root of the star. Thus, for any time t in phase h , ρ ( Y, V t ) = 2(∆ h +1) 2 = ∆ h +1 . Thus, we see that

<!-- formula-not-decoded -->

because T = ∑ h λ h T 0 ≤ 2 λ 2 λ T 0 ≤ ∆ T 0 and the number of phases is 2 λ .

̸

Now, we estimate the corresponding quantity for Approximate FTL. Consider a phase h . The 1 -median instance at a time t in this phase consists of T h ′ points at each of v 1 h ′ and v 2 h ′ for h ′ &lt; h and a certain number of points at the children of v h . Since T h ∆ h = T 0 ∆ for all h , we claim that this 1-median instance has optimal cost at least ( h -2) T 0 ∆ . Indeed, consider a 1 -median solution that places a center at the root or at v j or its children. Then all points at v 1 h ′ and v 2 h ′ , where h ′ ≤ h -1 , h ′ = h , incur a cost at least 2 T h ′ ∆ h ′ 2 = T 0 ∆ . Now, consider the solution that places a center at v h -1 (at any time during phase h ). Again it is easy to see that its total cost is at most 4( h -2) T 0 ∆+2( T h -1 )+4 λT h ∆ h 2 ≤ T 0 ∆(2 h -3+2 λ ) , which is at most 4(( h -2) T 0 ∆) when h ≥ λ +3 , and hence is a 4 -approximation for h ∈ [ λ +3 , 2 λ ]

Thus, we can assume that for any phase h ≥ λ +3 , A outputs v h -1 as the 2-approximate 1-median center. Now consider such a phase h : at each time t during this phase, optimal cost for V t is 2 , but

the algorithm incurs at least 2∆ h -1 = 2 λ ∆ h . Thus, ρ ( Y t , V t ) ≥ λ ∆ h . Summing over all times t in this phase, we see that

<!-- formula-not-decoded -->

Summing over all the phases h ≥ λ +3 (note that there are λ -2 such phases), we get

<!-- formula-not-decoded -->

The result now follows from the above and inequality (18) (note that λ = θ (log ∆ / log log ∆) and T 0 is a parameter independent of n, k, ∆ ).

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All our claims have corresponding theorems and formal proofs in the main text or appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss that our techniques are tailored for metric problems, and leave extensions to other domains as interesting future work (see Section 6).

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

Justification: All our theorems have formal proofs in the main text or the appendix.

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

Justification: The code used for the experiments is available at https://github.com/ neurips2025-colab/neurips2025 , along with instructions.

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

Justification: The code used for the experiments is available at https://github.com/ neurips2025-colab/neurips2025 , along with instructions.

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

Justification: All the details of the experiments are described in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report the standard deviation of the mean across multiple runs when appropriate in our experiments, in particular when running our algorithms on randomly generated instances.

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

Justification: Although our experiments do not require much compute, we describe the resources used to run them in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: All the authors have read and reviewed the Neurips Code of Ethics and have conformed to it.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is mostly a theoretical work.

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

Justification: Our work is unlikely to have significant negative downstream impacts, considering its theoretical nature.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We use Gurobi with an academic license to compute optimal integer solutions.

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

Justification: The code used for the experiments is available at https://github.com/ neurips2025-colab/neurips2025 , along with instructions.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing or research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The authors only use LLMs for minor editing of LaTeX and coding purposes.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.