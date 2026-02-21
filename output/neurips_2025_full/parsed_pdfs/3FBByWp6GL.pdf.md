## Learning to Specialize: Joint Gating-Expert Training for Adaptive MoEs in Decentralized Settings

Yehya Farhat ∗†

Hamza ElMokhtar Shili ∗†

Fangshuo Liao ∗†

Chen Dun ∗†

Mirian Del Carmen Hipolito Garcia ∗‡

Guoqing Zheng §

Ahmed Hassan Awadallah ‡

Robert Sim ‡

Dimitrios Dimitriadis §

Anastasios Kyrillidis †

Department of Computer Science, Rice University Microsoft

## Abstract

Mixture-of-Experts (MoEs) achieve scalability by dynamically activating subsets of their components. Yet, understanding how expertise emerges through joint training of gating mechanisms and experts remains incomplete, especially in scenarios without clear task partitions. Motivated by inference costs and data heterogeneity, we study how joint training of gating functions and experts can dynamically allocate domain-specific expertise across multiple underlying data distributions. As an outcome of our framework, we develop an instance tailored specifically to decentralized training scenarios, introducing Dynamically Decentralized Orchestration of MoEs or DDOME . DDOME leverages heterogeneity emerging from distributional shifts across decentralized data sources to specialize experts dynamically. By integrating a pretrained common expert to inform a gating function, DDOME achieves personalized expert subset selection on-the-fly, facilitating just-in-time personalization. We empirically validate DDOME within a Federated Learning (FL) context: DDOME attains from 4% up to a 24% accuracy improvement over state-of-the-art FL baselines in image and text classification tasks, while maintaining competitive zero-shot generalization capabilities. Furthermore, we provide theoretical insights confirming that the joint gating-experts training is critical for achieving meaningful expert specialization.

## 1 Introduction

Due to the success of large-scale deep learning [28, 2, 3, 15, 5, 18], it is now widely accepted as a design philosophy that ' the larger (model/dataset), the better ". Yet, the computational and economic costs associated with training and deploying large monolithic models raise concerns: Are we wisely allocating resources by training a single, monolithic model rather than employing specialized submodels that operate efficiently and adaptively?

Mixture of experts (or MoEs) [10, 12] represent a well-known model architecture that embodies this principle. Unlike traditional ensemble methods [7], MoEs train expert submodels jointly, using

∗ Equal contribution.

† {yehya.farhat, hamza.shili, fangshuo.liao, cd46, anastasios}@rice.edu

‡ {mirianh, ahmed.awadallah, rsim}@microsoft.com

§ Work done while at Microsoft.{gzheng}@percepta.ai, {dbdim}@amazon.com

a gating function to selectively activate subsets of experts based on input data. Recent successful instances of MoEs demonstrate impressive scalability benefits by leveraging sparse activation of network layers [30, 5, 18, 27, 25, 32].

Despite these successes, achieving semantic specialization -where experts learn distinct functionsremains challenging in single-domain tasks, where data does not naturally partition into subtasks or distinct distributions. Recent findings even suggest that sophisticated gating strategies might offer limited benefit compared to random gating functions in such single-domain contexts [36], indicating that explicit specialization does not emerge without additional mechanisms or constraints. Existing methods [19, 6] addressing meaningful specialization primarily focus on distinct task boundaries and separate datasets per expert to facilitate explicit expert specialization. This observation leads to a fundamental question: How can we effectively foster meaningful specialization within MoEs in single-domain scenarios, where clear task boundaries are not naturally available?

Understanding this question has broad implications for real-world settings where data exhibits subtle variations rather than clear domain boundaries. For instance, consider pretraining a model to predict the type and stage of cancer from CT scans. In this case, cancer prediction is the shared task, while the clients-different hospitals or clinics-each possess private datasets reflecting their own data distributions (e.g., one may specialize in prostate cancer, another in breast cancer). Similar situations arise in personalized content recommendation, language modeling across dialects, and client-specific adaptation in decentralized environments. Studying this phenomenon can lead to strategies for dynamically allocating computational resources, thereby improving both model efficiency and overall performance.

Main hypothesis and our contributions. We explore how joint training of gating mechanisms and experts within MoE frameworks can enable adaptive specialization under single-domain settings. We hypothesize that MoEs can leverage implicit data heterogeneity to achieve expert specialization, guided by a dynamic gating function that learns concurrently with experts.

As an instance, we introduce a decentralized MoE framework designed to leverage data characteristics across different nodes for personalized expert specialization. We dub this system Dynamically Decentralized Orchestration of MoEs or DDOME , a distributed MoE system tailored for decentralized learning scenarios (Figure 1). DDOME maintains a collection of independent expert models (we consider image and text classification tasks in this work), adaptively selected by a gating function influenced by shared representations from a pretrained common expert. This design allows DDOME to dynamically specialize subsets of experts across heterogeneous data distributions without explicit task annotations. Some of our findings include:

- We theoretically show decoupling the training of the gating and expert modules leads to suboptimal specialization, highlighting the necessity of joint training for effective expert allocation.
- DDOME effectively leverages implicit client-specific data characteristics to dynamically specialize experts during joint training, without the need for explicit task definitions.
- DDOME can dynamically select experts and achieve just-in-time personalization on unseen clients during testing. DDOME accurately classify unseen data with small adaptations.
- We achieve these reducing the overall communication cost by not sending the whole MoE module to all clients, compared to state-of-the-art methods [29].
- Some highlights of DDOME in practice: Within a Federated Learning (FL) setup, DDOME achieves ∼ 95% accuracy on FL CIFAR10, ∼ 78% accuracy on FL CIFAR100, and ∼ 75% on FL Yahoo! Answers text classification as a just-in-time personalization method on unseen clients, where the second best SOTA method achieves ∼ 71% , ∼ 74% , and ∼ 69% respectively.

## 2 Background

Notation. Vectors and matrices are represented with bold font (e.g., x ), while scalars by plain font (e.g., x or S ). Capital letters distinguish matrices from vectors (e.g., W vs w ). Calligraphic uppercase letters denote sets (e.g., D ); the cardinality of D is represented as |D| . [ N ] is [ N ] = { 1 . . . N } .

Problem formulation. Let S be the total number of training clients. Each client s has its own local data, denoted as D s . We will assume that D s = { x i , y i } |D s | i =1 , where x i is the i -th input sample and y i its corresponding label in a supervised setting. Abstractly, let W denote the collection of trainable model parameters. The goal is to find values for W that achieve good accuracy on all data D = ∪ s D s , by minimizing the following optimization objective:

<!-- formula-not-decoded -->

where ℓ ( W , D s ) = 1 |D s | ∑ { x i ,y i }∈D s ℓ ( W , { x i , y i } ) . Here, with a slight abuse of notation, ℓ ( W , D s ) denotes the local loss function for user s , associated with a local model W s (not indicated above), that gets aggregated with the models of other users. W s could be a full copy of the global model at the current training round or a selected submodel out of the global one, randomly chosen or based on the client's characteristics.

It is desired that the trained global model ̂ W ≈ W ⋆ is applied to unseen test clients that come with different non-i.i.d local data. Previous approaches handling a similar scenario [33, 31] assume we have access to part of the new client's labeled local data and fine-tune ̂ W . We consider this a limitation since new users are likely unwilling/unable to provide accurate labeled data and might not have sufficient resources to contribute to a fine-tuning phase of the whole model.

## 3 The Necessity of Joint Router-Expert Training

To motivate our system, we first examine whether joint training of a gating function and experts is necessary for such specialization to emerge. This question is nontrivial, as it remains unclear whether decoupled training can support effective expert allocation in the absence of explicit task boundaries. We provide a theoretical justification for this necessity by studying a simplified scenario with top-1 routing and linear experts on Gaussian input data.

Formally, consider a ground-truth orthonormal list v ⋆ 1 , . . . , v ⋆ m ∈ R d . This partitions R d into m subsets, denoted by:

<!-- formula-not-decoded -->

We consider learning a function f ⋆ that maps the input space R d to labels in R that depends on which region C j an input data x is sampled from. In particular, we consider there exists another orthonormal list w ⋆ 1 , . . . , w ⋆ m that connects the input data x with the output data y ∈ R by:

<!-- formula-not-decoded -->

Our goal is to learn an MoE model parameterized by θ = { ( v j , w j ) } m j =1 where { v j } m j =1 are the parameters of the gating function, and w j is the parameter of the j th expert:

<!-- formula-not-decoded -->

In particular, given an input vector x , we define the output of the j th expert as w ⊤ j x , and the vector Vx = [ v ⊤ 1 x , . . . , v ⊤ m x ] the gating output. Therefore, the indicator function I { v ⊤ j x ≥ max ℓ ∈ [ m ] v ⊤ ℓ x } can be seen as the top1 routing based on the gating function's output. We formulate the training of the MoE model as minimizing the following MSE loss:

<!-- formula-not-decoded -->

Note there exists θ ⋆ such that L ( θ ⋆ ) = 0 , by defining θ ⋆ = {( v ⋆ j , w ⋆ j )} m j =1 . To this end, the theorem below characterizes the limitation of the disjoint training of experts and gating parameters. √

Theorem 1. Assume that m ≤ d . Assume the gating parameters are initialized according to v j, 0 ∼ N ( 0 , γ 2 I d ) independently for all j ∈ [ m ] . Let ˆ θ be the parameter obtained by training the experts first, and then the gating function. Formally, let ˆ θ = { ( ˆ w j , ˆ v j ) } m j =1 be defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remarks: Here, (1) describes the process of training the experts' parameters { w j } m j =1 , while fixing the gating's weights at initialization { v j, 0 } j m =1 . After the experts parameters are learned, (2) trains the gating's parameters, while fixing the experts' weights at { ˆ w j } j m =1 learned in (1). What Theorem 1 demonstrates is that training experts first followed by the training of gating function with the trained expert weights frozen incurs a test loss that is lower bounded by a constant factor. Note that w ⋆ j has unit norm; then, by the standard concentration property of Gaussian random vectors, almost all labels y should have constant magnitude. Therefore, even a trivial choice of the parameters by choosing all v j and w j to be 0 will incur a constant loss. This means that the joint training of the expert and the gating's parameters will not improve upon the trivial choice of the parameters by more than a constant factor, motivating the need to perform joint training of the two parts. The proof of Theorem 1 -with all the details of the assumptions made- is provided in Appendix L.

Guided by this result, we now turn to the design of our proposed system, DDOME . DDOME operationalizes this principle in a decentralized learning setup and leverages data heterogeneity across different nodes to achieve expert specialization, even within a single domain.

## 4 Overview of DDOME

System components. Our system is depicted in Figure 1, with components grouped across the server (using purple boxes) and client (using cyan boxes).

Server-side. Parts (a), (b), (f), (h) in Figure 1. The server maintains two key modules: i ) a pool of M experts (MoE module), each initialized with the same architecture (e.g., TinyBert); and ii ) a gating function that ranks experts based on data characteristics. The experts can be randomly initialized or be pretrained; their weights are denoted as W i , for i ∈ [ M ] . The gating function is a small MLP with parameters W r that outputs a relevance score for each expert based on input representations.

Client-side. Parts (c), (d), (e), (g) in Figure 1. Each client has access to the same frozen, pretrained common expert that serves as a local feature extractor. This common expert transforms raw inputs into embeddings, which are further fed into server's gating function. Note that the common expert is not retrained during our procedure but used as an embedding mechanism. The result of the gating function is a sparse-enforced selection of experts. The selected experts, say experts i and j , are dispatched to the client to be locally trained. Each client jointly updates the assigned experts' parameters -denoted as W i ∈ e s , with a slight abuse of notation- along with the gating function W r . Finally, the updated parameters W i ∈ e s and W r are sent back to the server to be aggregated with other updates coming from other participating clients during the training round.

The gating function. The gating function consists of two components: i ) a pretrained common model that acts as a feature extractor, converting each client's local data samples into embeddings. By design, our gating function should be model agnostic to the pretrained common expert. This model is treated as a fixed, black-box encoder and does not require further training. ii ) an expert-ranking network , which takes the extracted embeddings as input and outputs a relevance score for each expert. This network is updated locally by each participating client in that training round using its own data. The expert-ranking network aggregates scores across local data samples and selects the most relevant experts to be sent to each active client. This sparse-enforced, per-round selection ensures that each client only interacts with a targeted subset of experts, promoting both efficiency and specialization.

Expert-Client relationship. To encourage early and stable specialization, we introduce a client activation strategy, called anchor clients . This strategy serves two key purposes: i ) it guides experts toward meaningful initial specialization, and ii ) it helps the gating function better characterize each expert's behavior. Specifically, given M experts in the system, we pre-select M clients (out of a much larger S ≫ M ) to act as anchor clients. Each anchor client is persistently aligned with a specific expert in a one-to-one fashion and is activated more frequently than regular clients. During training, these clients are optimized with an independent loss. This design encourages consistent, distinctive expert behaviors; this strategy improves convergence stability and expert diversity (Section 5).

from active clients from active clients

Figure 1: For each client: i ) The server uses the gating function to select a subset of experts based on the local data distribution (parts (a) , (b) , (d) , (e) ); ii ) The client updates expert and gating function's weights (part (g) ) and sends these back to the server. iii ) the server aggregates and update the new weights (part (h) ). The above are repeated for all FL rounds.

<!-- image -->

## Module details

Pretraining. Each client utilizes a pretrained common expert with parameters W c . E.g., such an expert could be a pretrained model on ImageNet for image classification purposes. We restrict our methodology such that: i ) we ensure our algorithm is agnostic to both the common expert's architecture and performance; ii ) we assume only access to the common expert's embedding capabilities; and iii ) we do not modify/retrain the common expert. The common expert is sent to/downloaded by all clients only once, before training. For client s , we perform one-time inference on all local data using the common expert and store the corresponding output features, noted as ˆ x s i for each x i ∈ D s .

The set of expert models. Our methodology involves M experts, each being an independent model of the same architecture. 5 For the i -th expert, i ∈ [ M ] , we denote its parameters as W i and the corresponding model function as f ( · , W i ) ). See also Figure 1 (a) . We consider two cases in our experiments for completeness: The M experts are randomly initialized in our image classification experiments to provide full plasticity during training, while for text classification the M experts are initialized using weights transferred from a pretrained TinyBERT model. In each round, different subsets of experts are selected to be communicated to and updated by active clients based on their local data (see Figure 1 (f) ). Per round, the updated experts are sent back to the server to be aggregated before the next round starts; see Figure 1 (h) .

The gating function mechanics. We randomly initialize an expert-ranking network with parameters W r . This is a small-scale, two-layer MLP that, for each active client, takes embeddings from the common expert and predicts a relevance score over all M experts. Specifically, for client s , we denote the score as g (ˆ x s i , W r ) ∈ R M , for the i -th data sample, based on:

<!-- formula-not-decoded -->

The final decision on the topK experts is made via the rule:

<!-- formula-not-decoded -->

over all embedded local data samples ˆ x s i , i ∈ [ |D s | ] where the TOPK ( · ) function selects the dominating experts, based on the current state of W r and { ˆ x s i } n i =1 .

The 'anchor clients' mechanism. Given M experts, we designate a subset of M clients, with roughly distinct local data distributions, as anchor clients . 6 All remaining clients are referred to as normal clients . Each anchor client is pre-assigned to a unique expert in a one-to-one relationship,

5 This choice is made for simplicity. Diverse architectures per expert are left for future work.

6 The anchor client selection process is detailed in Appendix A.

## Algorithm 1 DDOME

Parameters : T rounds, S training clients, U testing clients, M experts, ℓ 1 local iterations, experts' function and parameters f ( · , W i ) , gating function's function and parameters g ( · , W r ) , common expert's parameters W c .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we denote the index of the expert assigned to the q -th anchor client as I q . At the beginning of each communication round, a subset of N clients are activated and are selected to participate, where N ≪ S and S is the total number of clients in the system. We divide the N active clients into two groups: N a anchor clients and N c normal clients, such that N = N a + N c . Anchor clients N a are sampled from the set of M anchor clients, and normal clients N c from the remaining S -M clients. 7

The idea is that, since M ≪ ( S -M ) , we more frequently sample the anchor clients. This frequent activation, coupled with fixed expert assignments, encourages each expert to specialize to the data distribution of its associated anchor client. In other words, experts are consistently trained on similar data distributions, fostering more stable and distinct specialization over time.

The training process. For both normal and anchor clients, the server sends the current copy of parameters of the gating function, W r . The gating function selects a subset of experts; the output e s abstractly contains the set of chosen experts W i for client s , where i ∈ e s . The server receives e s and sends the parameters W i , for i ∈ e s , to the corresponding client s ; this routine reduces the communication cost -as compared to existing methods [29]- and encourages expert specialization. We provide a theoretical analysis of the communication cost in Appendix K

Per training round, each normal client, using the standard cross-entropy loss, will locally update both W r and W i 's. Formally, this amounts to (see also Figure 1, part (g) ):

<!-- formula-not-decoded -->

On the other hand, for an anchor client q , we only send the I q expert to encourage expert specialization. Such an expert is trained regularly on the anchor's local distribution. Accordingly, we encourage the expert ranker network to recognize such rough specialization of the selected expert by using a simple independent loss. The two loss functions for anchor clients are as follows:

<!-- formula-not-decoded -->

where 1 I q is the one-hot encoding indicating I q .

7 The anchor-to-normal sampling ratio is discussed in Section 5.

After all clients finish the local training round, the server applies a simple aggregation step to average the updated copies of W r and W i 's. Adapting the MoE loss to this setting is non-trivial . After expert selection, each client only observes and trains a small subset of experts ( K ≪ M ), posing a major challenge for the gating function: if it selects 'incorrect' experts, it may not only harm performance but also interfere with the specialization process by misaligning the experts. Despite this, we find that the gating function is able to learn effective expert assignments.

Test-time generalization to unseen clients. We are given new clients with unseen local data distributions during testing. We only send K experts to each test client, and we cannot get access to local test data labels to perform fine-tuning. We first send W r to the test client and select the topK experts, according to aggregated expert ranking score. Then, for each test sample , instead of using the weighted average of the output of all selected experts, we use the output of the expert with the highest expert ranking score to fully utilize the specialization of the expert. I.e., both experts might be utilized for different data samples instead of averaging their performance. See Algorithm 1.

## 5 Experiments

The learning scenario we consider. We focus our experiments on supervised image and text classification learning tasks within a Federated Learning (FL) setup [23, 21, 13], as FL offers a practical and representative environment for evaluating our system. We use the CIFAR data suite [17, 8] and EMNIST [4] for image tasks, and the Yahoo! Answers dataset [35] for text. Following common FL practice, we partition data by class to transform the full datasets into non-i.i.d. subsets. We assume the FL server-client protocol, where clients participate in training using local data; we assume there are 100 clients while we can only activate 10% clients per round. However, our system deviates from traditional FL implementations [23, 21, 13]; in those, one assumes a sole global model that is being shared with active clients, and updates to this model are being aggregated by the server per synchronization round. E.g., in image classification scenarios a large, ResNet-type network -like ResNet34, ResNet101, or ResNet200- could be used [9]. For our system, the 'global' model consists of multiple independent models (experts), all sharing the same architecture. The pretrained common expert, architecturally identical to the task-specific experts, supports on-the-fly expert selection. In our experiments, we assume between 5 and 10 experts per deployment.

Task and model description. For the image classification task, we use ResNet-34 as the expert model architecture [9] for the federated CIFAR experiments, and a two-layer MLP for the federated EMNIST experiments. For text classification, we use TinyBERT [11]. In all cases, the gating function is implemented as a two-layer MLP followed by a softmax layer, which outputs relevance scores over all experts. For the image classification, clients are trained using the SGD optimizer with momentum (SGDM), with a learning rate of 0.01, momentum of 0.9, batch size of 256, and one local epoch per round. For text classification, all client models are trained using SGDM with identical hyperparameters. All clients use a batch size of 16, and one local epoch per round. The gating function is trained using an SGD optimizer; for image classification, this uses a fixed learning rate of 0.001, while for text classification the same initial learning rate is used but it incorporates a step decay learning rate schedule. The model aggregation on the server side is performed with FedAvg [24].

System. Experiments were conducted on different hardware setups. For image classification tasks, we used an NVIDIA RTX A6000 GPU with 46GB of VRAM. Training the default configuration with 10 experts required approximately 6 hours. For text classification tasks, we used an NVIDIA A40 GPU with 48GB of VRAM, and training the default configuration with 5 experts required over 100 hours (using 2 workers) due to the increased computational cost. Training was performed in a distributed fashion using between 2 and 10 workers.

Dataset. We conduct experiments on CIFAR10, CIFAR100 [17], EMNIST [4], and Yahoo! Answers [35]. For EMNIST, we use the 'ByClass' split, which contains 814,255 character images across 62 unbalanced classes. For Yahoo! Answers, we randomly sample 50,000 training and 10,000 test examples to create a dataset of the same size as CIFAR, with an equal number of samples per class. 8 To increase task difficulty, we exclude the answer text and train models solely on the question title and content. For all cases, the training dataset is randomly partitioned across 100 clients. We followed the same procedure for the anchor clients but avoided replacement, aiming to preserve the label diversity in each subset. We establish a one-to-one mapping between these clients and the experts, corresponding to each group of labels. This path allows to i ) have one expert available for

8 Full dataset training is computationally expensive, given the hardware configuration we had.

each group; and ii ) retain the flexibility to activate the anchor clients during the training rounds. The complete client distribution for all datasets is detailed in Appendix A.

Zero-Shot Personalization. Let us first describe the baselines to compare against: 9

- FedMix [29] trains an ensemble of models adapted to the data space's sub-regions. By definition, FedMix sends all the experts to each client to specialize them, heavily increasing communication costs. For this implementation, we initialized the common expert from the initial pretrained model checkpoint, and we used it to embed local data in the gate function and help the routing.
- FedAvg [24] is the de facto approach for FL and allows a fair comparison regarding fixed communication cost. Here, we initialize the global model with the initial common expert checkpoint and aggregate the updates from all sampled clients per iteration.
- FedProx [22] tackles heterogeneity by introducing a regularization term that limits the distance between local/global models at the cost of additional computation overhead per round. We follow the same strategy for initialization with FedAvg .
- Scaffold [14] handles non-iidness by applying control variates for the server and clients at the expense of doubling communication cost per round compared to FedAvg . This method tends to become unstable during training, as previous studies have shown [20]. We follow the same strategy for initialization with FedAvg .
- The Average Ensembles [16] train two models (initialized from the common expert) as in FedAvg , but with different random seeds. It then combines them by averaging output probabilities. While it provides flexibility with respect to resources, it has higher inference costs.

|                    |           | CIFAR 10   | CIFAR 10   | CIFAR 10   | CIFAR 10   | CIFAR 10   | CIFAR 100   | CIFAR 100   | CIFAR 100   | CIFAR 100   | CIFAR 100   |
|--------------------|-----------|------------|------------|------------|------------|------------|-------------|-------------|-------------|-------------|-------------|
| Method             | # Clients | Rounds     | M          | K          | Acc.       | Acc.       | Rounds      | M           | K           | Acc.        | Acc.        |
| Common Expert      | -         | -          | -          | -          | 73%        | 93%        | -           | -           | -           | 67%         | 73%         |
| FedMix [29]        | 100       | 1250       | 2          | 2          | 31.3%      | 42.9%      | 2000        | 2           | 2           | 49.7%       | 48.3%       |
| FedAvg [24]        | 100       | 1250       | -          | -          | 31.2%      | 58.4%      | 2000        | -           | -           | 72.9%       | 74.0%       |
| FedProx [22]       | 100       | 1250       | -          | -          | 72.7%      | 71.4%      | 2000        | -           | -           | 72.8%       | 74.0%       |
| Avg Ensembles [16] | 100       | -          | -          | -          | 23.9%      | 53.7%      | -           | -           | -           | 72.8%       | 74.1%       |
| DDOME              | 100       | 1250       | 5          | 2          | 91.8 %     | 95.7%      | 2000        | 10          | 2           | 75.7 %      | 78.6 %      |

Table 1: Average zero-shot personalization score for unseen test clients on CIFAR10/100. See subfigures (d) and (b) of Figure 5 in Appendix C for statistical significance of the respective results. We use two different pretrained common experts as feature extractor for each dataset: a) The lower bound model at which the gating function can outperform the initial common expert accuracy, illustrated in Figure 8 in Appendix; b) The average model that represents a good accuracy that is relatively easy to achieve using ResNet-34 architecture. Sampling is performed under the scheme of N a = 5 anchor and N c = 5 normal clients per training round; here, N = N a + N c = 10 . See Appendix E for DDOME gating function's effectiveness on individual samples.

Tables 1-3 summarize our findings on this setup. Whereas FedMix requires all experts to be transmitted to each client, i.e., M = K , DDOME allows the selection of K experts, here K = 2 , without the need to send them all. This reduces communication costs and ensures the client receives the most pertinent information from the relevant experts.

In terms of baselines, we observe that for the CIFAR datasets both behave differently. We attribute this gap to the number of classes each client holds. In the CIFAR10 scenario, each client has fewer classes, which can amplify the model drift problem in all baselines. Furthermore, FedAvg 's performance deteriorates sharply when we test it on the new CIFAR10 clients that were not used for training due to the heterogeneous data distribution during training and then in the testing phase. Similarly, Average Ensembles faces a performance ceiling, as the ensembles inherit the limitations of the FedAvg aggregation method. On the other hand, FedProx can surpass the initial performance of the common expert for the CIFAR100 scenario but degrades quickly when using few labels per client, as in the CIFAR10 setup. To the best of our ability, we attempted multiple hyperparameter settings for Scaffold , yet we were unable to produce a useful model under this distribution; it became unstable during training (achieving only 10% for CIFAR10 / &lt;5% for CIFAR100). Further comparison against domain adaptation methods, as in FedADG [34] and FedSR [26], is shown in Appendix I; for the cases we consider, we observe that current implementations are bound to having a small number of clients to perform competitively. These trends are not limited to vision tasks.

9 Note that we are aware that there are dozens more generic FL algorithms to compare against; yet, our aim is to provide a proof-of-concept for our methodology on training and selecting the right experts in such a setting.

On the Yahoo! Answers dataset (Table 2), we observe a similar pattern. All baselines struggle to consistently outperform the pretrained common expert, with FedAvg again showing limited generalization. However, unlike CIFAR10, the severity of model drift is reduced-likely due to clients having access to more diverse text samples per class. De-

|               |           | Yahoo! Answers   | Yahoo! Answers   | Yahoo! Answers   | Yahoo! Answers   | Yahoo! Answers   |
|---------------|-----------|------------------|------------------|------------------|------------------|------------------|
| Method        | # Clients | Rounds           | M                | K                | Acc.             | Acc.             |
| Common Expert | -         | -                | -                | -                | 61%              | 69%              |
| FedMix        | 100       | 2000             | 2                | 2                | 58.08%           | 59.46%           |
| FedAvg        | 100       | 2000             | -                | -                | 60.75%           | 60.18%           |
| FedProx       | 100       | 2000             | -                | -                | 60.15%           | 61.18%           |
| Avg Ensembles | 100       | -                | -                | -                | 62.42%           | 69.36%           |
| DDOME         | 100       | 2000             | 5                | 2                | 64.55 %          | 74.90%           |

Table 2: Average zero-shot personalization score for unseen test clients on Yahoo! Answers. The structure of the table follows that of Table 1.

spite this, the limitations FedAvg and Average Ensembles persist. As in CIFAR100, FedProx slightly outperforms the common expert in one configuration, but fails to do so consistently.

For data diversity, we report results on the EMNIST dataset; please refer to Table 3 for more information. We note that we have also considered lower than 73% accuracy for the common expert (e.g., 67%). Yet, such an initial performance was too low to improve further using any of the methods in comparison. This led to the inclusion of the 73% and 80% cases. This highlights the importance of the common expert in our framework, underlying that our methodology does not 'magically' work for all cases. Still, proper preparation is needed to obtain favorable performance.

The global accuracy reported at the end of training demonstrates the effectiveness and consistency of DDOME in all datasets, with significantly better performance than other algorithms. Please refer to Appendix B for a detailed end-to-end performance of the methods in Table 1 under different clients' distribution.

Additional ablation studies and experiments. Appendix F contains thorough ablation studies

Table 3: Average zero-shot personalization score for unseen test clients on EMNIST. The structure of the table follows that of Table 1. See Figure 6 in Appendix D for statistical significance of the respective results.

|               |           | EMNIST   | EMNIST   | EMNIST   | EMNIST   | EMNIST   |
|---------------|-----------|----------|----------|----------|----------|----------|
| Method        | # Clients | Rounds   | M        | K        | Acc.     | Acc.     |
| Common Expert | -         | -        | -        | -        | 73%      | 80%      |
| FedMix        | 100       | 2000     | 2        | 2        | 9.4%     | 15.9%    |
| FedAvg        | 100       | 2000     | -        | -        | 72.1%    | 72.1%    |
| FedProx       | 100       | 2000     | -        | -        | 72.0%    | 72.0%    |
| Avg Ensembles | 100       | -        | -        | -        | 74.5%    | 74.4%    |
| DDOME         | 100       | 2000     | 10       | 2        | 79.9 %   | 80.5%    |

on the initial conditions of the common expert and how it boosts the performance, and the value of anchor clients and their ratio with normal clients. Appendix E contains assessment of the DDOME gating function's effectiveness on individual samples. Appendix G considers the incremental learning scenario, where either the pool of clients dynamically increases over time or changes over time. Appendix H considers the case where M = K and compares FedMix versus DDOME . Appendix I compares DDOME against domain generalization methods. Finally, Appendix J discusses how DDOME differ from other clustering-based methods applied on similar scenarios.

## 6 Broader impacts and limitations

Broader impacts. Our work advances efficient and adaptive specialization of Mixture-of-Experts models, enabling personalized ML in decentralized environments. Positive impacts include privacypreserving FL and resource-efficient personalized applications, reducing computational and communication costs. However, we acknowledge potential risks, such as exacerbating fairness issues if expert specialization amplifies biases inherent in heterogeneous data. Future deployment should carefully monitor and mitigate such risks.

Limitations. Despite the practical and theoretical promise, our study exhibits several limitations that could inform future research directions:

- Dependence on pretrained common experts: The success of our approach relies heavily on the availability of an effective pretrained common expert (see Appendix F).
- Communication overhead: While DDOME reduces communication costs relative to sending all experts to clients, it still incurs higher communication overhead compared to traditional singlemodel methods.

- Expert initialization and diversity: Our experiments indicate that expert diversity at initialization significantly impacts specialization effectiveness. We primarily evaluated homogeneous expert architectures; further study is necessary to understand the impacts of architectural heterogeneity.
- Limited scalability testing: Current experimental setup tests up to a moderate number of clients.
- Complexity of gating mechanism tuning: Finding optimal gating configurations might become computationally expensive as scale and diversity increase.

## 7 Conclusions

In this work, we investigated how joint training of gating mechanisms and experts can enable adaptive specialization in Mixture-of-Experts (MoE) frameworks under single-domain, heterogeneous data settings. We introduced DDOME ( Dynamically Decentralized Orchestration of MoEs ), a distributed MoE architecture specifically tailored for decentralized training scenarios.

Our empirical evaluations across various datasets demonstrate that DDOME achieves state-of-the-art performance, surpassing existing FL methods by leveraging implicit data heterogeneity. Complementing empirical findings, we provided a rigorous theoretical justification for the necessity of joint training between gating and experts. Our analysis highlights that disjoint or sequential training of these components significantly limits achievable specialization, reinforcing the importance of coordinated parameter updates. Future work will explore architectural diversity among experts, scalability enhancements, and extensions to multi-domain settings.

## References

- [1] Ruqi Bai, Saurabh Bagchi, and David I. Inouye. Benchmarking algorithms for federated domain generalization, 2023.
- [2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems , 33:1877-1901, 2020.
- [3] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311 , 2022.
- [4] Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre Van Schaik. EMNIST: Extending MNIST to handwritten letters. In 2017 international joint conference on neural networks (IJCNN) , pages 2921-2926. IEEE, 2017.
- [5] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. The Journal of Machine Learning Research , 23(1):5232-5270, 2022.
- [6] Suchin Gururangan, Mike Lewis, Ari Holtzman, Noah A Smith, and Luke Zettlemoyer. DEMix layers: Disentangling domains for modular language modeling. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 5557-5576, 2022.
- [7] Lars Kai Hansen and Peter Salamon. Neural network ensembles. IEEE transactions on pattern analysis and machine intelligence , 12(10):993-1001, 1990.
- [8] Chaoyang He, Alay Dilipbhai Shah, Zhenheng Tang, Di Fan1Adarshan Naiynar Sivashunmugam, Keerti Bhogaraju, Mita Shimpi, Li Shen, Xiaowen Chu, Mahdi Soltanolkotabi, and Salman Avestimehr. FedCV: a federated learning framework for diverse computer vision tasks. arXiv preprint arXiv:2111.11066 , 2021.
- [9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [10] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation , 3(1):79-87, 1991.
- [11] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. Tinybert: Distilling bert for natural language understanding. arXiv preprint arXiv:1909.10351 , 2019.
- [12] Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the EM algorithm. Neural computation , 6(2):181-214, 1994.
- [13] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning, 2019. URL https://arxiv.org/abs/1910.06378 .
- [14] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning, 2021.
- [15] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. arXiv preprint arXiv:2205.05198 , 2022.
- [16] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems , volume 25. Curran Associates, Inc., 2012. URL https://proceedings.neurips.cc/paper\_files/paper/2012/file/ c399862d3b9d6b76c8436e924a68c45b-Paper.pdf .

- [17] Alex Krizhevsky et al. Learning multiple layers of features from tiny images. 2009.
- [18] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668 , 2020.
- [19] Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models. arXiv preprint arXiv:2208.03306 , 2022.
- [20] Qinbin Li, Yiqun Diao, Quan Chen, and Bingsheng He. Federated learning on non-iid data silos: An experimental study, 2021.
- [21] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks, 2018. URL https://arxiv.org/ abs/1812.06127 .
- [22] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks, 2020.
- [23] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics , pages 1273-1282. PMLR, 2017.
- [24] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. Communication-efficient learning of deep networks from decentralized data, 2023.
- [25] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. Multimodal contrastive learning with LIMoE: the language-image mixture of experts. arXiv preprint arXiv:2206.02770 , 2022.
- [26] A. Tuan Nguyen, Philip Torr, and Ser Nam Lim. Fedsr: A simple and effective domain generalization method for federated learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 38831-38843. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ fd946a6c99541fddc3d64a3ea39a1bc2-Paper-Conference.pdf .
- [27] Joan Puigcerver, Carlos Riquelme, Basil Mustafa, Cedric Renggli, André Susano Pinto, Sylvain Gelly, Daniel Keysers, and Neil Houlsby. Scalable transfer learning with expert models. arXiv preprint arXiv:2009.13239 , 2020.
- [28] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis &amp; insights from training Gopher. arXiv preprint arXiv:2112.11446 , 2021.
- [29] Matthias Reisser, Christos Louizos, Efstratios Gavves, and Max Welling. Federated mixture of experts, 2021.
- [30] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 , 2017.
- [31] Kangkang Wang, Rajiv Mathews, Chloé Kiddon, Hubert Eichner, Françoise Beaufays, and Daniel Ramage. Federated evaluation of on-device personalization. arXiv preprint arXiv:1910.10252 , 2019.
- [32] Zhao You, Shulin Feng, Dan Su, and Dong Yu. SpeechMoE: Scaling to large acoustic models with dynamic routing mixture of experts. arXiv preprint arXiv:2105.03036 , 2021.
- [33] Tao Yu, Eugene Bagdasaryan, and Vitaly Shmatikov. Salvaging federated learning by local adaptation. arXiv preprint arXiv:2002.04758 , 2020.

- [34] Liling Zhang, Xinyu Lei, Yichun Shi, Hongyu Huang, and Chao Chen. Federated learning with domain generalization, 2023.
- [35] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. Advances in neural information processing systems , 28, 2015.
- [36] Simiao Zuo, Xiaodong Liu, Jian Jiao, Young Jin Kim, Hany Hassan, Ruofei Zhang, Tuo Zhao, and Jianfeng Gao. Taming sparsely activated transformer with stochastic experts. arXiv preprint arXiv:2110.04260 , 2021.

## A Clients distribution

We created a federated version of the datasets by introducing two partitioning strategies to split the samples across 100 clients:

- Quantity-based label imbalance : Each client holds data samples of K labels. We first randomly assign K different labels to each client. Then, per label, we randomly assign samples to clients along with labels (with replacement). This way, the number of different labels for each client is fixed. For the CIFAR100 dataset, we use K = 10 . For the CIFAR10 and Yahoo! datasets, we use K = 4 .

Anchor clients: We followed the same method to create the anchor clients, except we prevented replacement when randomly selecting the labels. This way, we created a) 5 anchor clients with K = 2 on CIFAR10 and Yahoo! and b) 10 anchor clients with K = 10 on the CIFAR100 dataset.

- Distribution-based on label imbalance : We simulated the label imbalance of each client by allocating a portion of the samples (with replacement) of each label according to the Dirichlet distribution ( α = 0 . 1 ). As illustrated in Figure 2, the test clients are randomly unseen combinations of K labels that never appear during training.

Anchor clients: We use the same Dirichlet distribution ( α = 0 . 1 ) to randomly create a) 5 anchor clients on CIFAR10 and b) 10 anchor clients on the CIFAR100 dataset.

Figure 2: Example of distribution-based label imbalance partition on CIFAR10 dataset ( α = 0 . 1 )

<!-- image -->

We conducted a simulation of a federated version of 100 clients on the EMNIST dataset, using the "ByClass" split. This split presents a greater challenge than the CIFAR10/100 datasets, as some classes have a much larger number of samples, resulting in 62 unbalanced classes. The clients were created in accordance with the Quantity-based label imbalance approach, with K = 5 .

Note that for test users, we do not repeat any distribution from the training clients; this way, we create an example where the distribution of the images over all users is different.

## B DDOME end-to-end performance

## B.1 Quantity based strategy

We begin to evaluate the performance of our method and baselines by measuring the zero-shot personalized model accuracy on several unseen test clients with a Quantity-based label imbalance distribution strategy, as explained in Appendix A. The results are illustrated in Figure 3.

In Figure 3, we can observe that FedAvg cannot keep improving once it's initialized from the pretrained checkpoint. This surprising result stems from three major issues: i ) the clients' learning rate parameters are inconsistent with previous training, ii ) the heterogeneous data distribution on the training clients introduces a high degree of model variability, and iii ) the pretrained expert struggles to improve or adapt to the federated distribution. Moreover, implementing FedProx required careful fine-tuning of the µ parameter to achieve good accuracy and fast convergence. On the other hand, despite trying multiple hyperparameter settings, we could not produce a useful model using the Scaffold method; it became unstable during training and often collapsed or got stuck in a poor model. This suggests that our method is more robust than these baselines in the current setup.

Figure 3: DDOME on CIFAR10 (left) and CIFAR100 (right) datasets, against FedMix , FedAvg and Average Ensembles based on Table 1, using an initial common expert of 73% accuracy.

<!-- image -->

## B.2 Distribution based strategy

Using the distribution-based strategy -detailed in Appendix A- we implement two additional challenging scenarios, where further heterogeneity and complexity are inserted via labels distribution: i ) we use the Dirichlet probability rule to generate skewed and imbalanced label distributions, mimicking real-world applications; ii ) we relax the assumption of disjoint labels for the anchor clients and allow label overlap, creating a more complex scenario, given that experts are initialized from scratch.

Table 4: Best global test accuracies from the last ten evaluation rounds reported on different non-iid algorithms under Dirichlet distribution ( α = 0 . 1 ).

|               | CIFAR 10   | CIFAR 100   |
|---------------|------------|-------------|
| Common Expert | 73.39%     | 73.73%      |
| FedAvg        | 51.3%      | 73.6%       |
| FedProx       | 52.8%      | 73.6%       |
| Scaffold      | 10.0%      | 01.0%       |
| FedMix        | 29.8%      | 65.3%       |
| DDOME         | 80.8%      | 77.8%       |

Table 4 indicates DDOME leverages the common expert's original 73% accuracy to reach up to 80% accuracy, even on highly skewed scenarios. While heterogeneity should decrease the overall performance, DDOME outperforms the methods under comparison, where experts learn to better generalize to unseen data.

Further, in Figure 4, we show the test accuracy envelope curves for all the algorithms under consideration. It is clear that DDOME show superior performance throughout execution, often surpassing the text accuracy of the common expert, which is already sufficiently trained, and there is limited space for improvement. This figure also shows the behavior of the models during training: even DDOME shows the variability of test accuracy over iterations, indicating that keeping the model at the very end of the execution might not always be the best practice.

Figure 4: Evaluation of different non-iid algorithms under Dirichlet distribution ( α = 0 . 1 ) on CIFAR10 dataset.

<!-- image -->

## C Performance under different sampling ratios

There is an initial degree of randomness in the gating function. During the first couple of iterations, it sends random topK experts to each client while the experts learn to specialize in the different regions of the label space. However, we found a way to keep consistency during these initial rounds: through the anchor clients . Figure 9 shows that by introducing at least 30% anchor clients during each round, we can ensure a balance against the wrong selection of the gating function by letting them act as regularizers. Additionally, Figure 5 shows the impact on the performance when we remove the anchor clients rule from sampling and allow only random selection from the pool of available clients. It is clear from Figure 5 that a 'warming-up' phase is necessary for DDOME : using a sufficient number of anchor clients, one achieves stability and better final accuracy, by warm-starting the system using more specialized clients.

Figure 5: Global testing accuracy for CIFAR100 (a-b) and CIFAR10 (c-d) datasets on two different sampling strategies: a) + c): 10 random clients without replacement per iteration. b) + d): 5 random anchor clients + 5 normal clients without replacement per iteration along different methods.

<!-- image -->

## D EMNIST Byclass statistical significance results

We evaluate a two-layer MLP on the federated EMNIST dataset, running each algorithm for 1000 communication rounds across three different initialization seeds. Results show that DDOMe consistently surpasses the performance of the initial common expert, demonstrating its robustness and effectiveness across varying accuracy levels.

Figure 6: Statistical significance results for EMNIST dataset using a common expert with 73% accuracy (left subfigure) and 80% accuracy (right subfigure).

<!-- image -->

## E Gating function Per-Sample Performance

After training, we thoroughly evaluate our gating function, using the checkpoints trained with the 73% common expert on CIFAR10 and CIFAR100 datasets on the DDOME algorithm. Our fine-grained evaluation demonstrates that our gating function can analyze the characteristics of each unseen test client's local sample and adaptively select a subset of experts that match those characteristics. This is crucial in ensuring that our gating function can generalize well to new data. After selecting the topK experts, the gating function chooses the highest score/confidence expert to predict each test

data sample. Our results, reported in Table 5, show that our gating function can achieve high accuracy on the selection.

Table 5: Evaluation per-sample level on CIFAR10 and CIFAR100 datasets.

| CIFAR100           | CIFAR100           | CIFAR100           | CIFAR100   | CIFAR10            | CIFAR10            | CIFAR10            | CIFAR10    |
|--------------------|--------------------|--------------------|------------|--------------------|--------------------|--------------------|------------|
| Client             | Incorrect          | Correct            | Error Rate | Client             | Incorrect          | Correct            | Error Rate |
| 0                  | 278                | 722                | 27.8%      | 0                  | 227                | 3773               | 5.7%       |
| 1                  | 281                | 719                | 28.1%      | 1                  | 122                | 3878               | 3.1%       |
| 2                  | 263                | 737                | 26.3%      | 2                  | 563                | 3437               | 14.1%      |
| 3                  | 251                | 749                | 25.1%      | 3                  | 103                | 3897               | 2.6%       |
| 4                  | 261                | 739                | 26.1%      | 4                  | 78                 | 3922               | 2.0%       |
| 5                  | 309                | 691                | 30.9%      |                    |                    |                    |            |
| 6                  | 260                | 740                | 26.0%      |                    |                    |                    |            |
| 7                  | 285                | 715                | 28.5%      |                    |                    |                    |            |
| 8                  | 255                | 745                | 25.5%      |                    |                    |                    |            |
| 9                  | 267                | 733                | 26.7%      |                    |                    |                    |            |
| Average Error Rate | Average Error Rate | Average Error Rate | 27.1%      | Average Error Rate | Average Error Rate | Average Error Rate | 5.5%       |

## F Ablation studies

Ablation study: Initial common expert impact. We study performance tradeoffs when utilizing different common experts for the gating function decisions. Our findings indicate that the amount of training allocated in the initial common expert has a critical effect on the overall performance of DDOME . For example, suppose the gating function uses a poor common expert for training. In that case, it can lead to poor performance (collapses to selecting a single expert) and, therefore, unable to improve beyond the baseline.

Figures 7-8 show that the breakpoint of the gating function for the CIFAR100 dataset is approximately 66% accuracy by the common expert . In Figure 8, it becomes clear that a significant cause of this breakpoint is that the experts cannot surpass the common expert's initial accuracy. This is attributed to the lack of an adequate selection of experts, which is essential for the gradient updates of each expert to be aligned with the same part of the task. Figure 7 also reveals the following: the 67% case, given a few more iterations, can match the performance of the 73% case. This suggests a 'phase-transition' might exist, where more effort (i.e., communication) is needed to improve beyond the common expert's performance. This implies that the performance of DDOME depends on the quality of the experts.

Figure 7: DDOME 's performance on CIFAR100 dataset, using different initial accuracy for common expert (legends of the plot); the setup in Table 1 is used.

<!-- image -->

Figure 8: Zero-shot personalization accuracy per expert during training on CIFAR100.

<!-- image -->

Ablation study: Common expert boosts experts' performance. To test this, we initialized each expert from the common expert and continued training for 2000 rounds. In Table 6, we observe the final score of each method. Surprisingly, for DDOME , it takes a few more rounds to overcome the baseline than when the experts are initialized from scratch. This is because the pretrained model is optimal for the entire dataset. To successfully specialize experts, retraining the model on the specific subset of labels is necessary.

Table 6: Average zero-shot accuracy for CIFAR100 after 2000 rounds.

| Common Expert     | 73.73%   |
|-------------------|----------|
| FedMix            | 73.78%   |
| FedAvg            | 73.99%   |
| Average Ensembles | 74.10%   |
| DDOME             | 83.27%   |

We also plot in Figure 8 the performance of each expert (denoted as exp X ) over the communication rounds for different initial accuracies of the common expert. It is evident that, for our setting, using a common expert with an accuracy below 67% We can outperform the other methods once the gating function can utilize a slightly better common expert.

Ablation study: The 'anchor/normal' client ratio. The sampling scheme must be carefully studied to ensure the best performance of DDOME . Each expert has a distinct distribution; i.e., their local objectives only align with a particular subset of labels. Ensuring consistency in the experts' updates is essential to prevent them from drifting away from their own 'task'. As mentioned, we assume we have some control over the activation of the clients during training.

<!-- image -->

Communication Rounds

Figure 9: DDOME \_ Na \_ X \_ Nc \_ Y means that X Y = N a N c , and N = N a + N c . 30% anchor/normal client ratio is enough to match baseline accuracy. However, the model becomes more inconsistent by converging more slowly.

Our solution is the use of anchor clients, whose primary purpose is to act as regularizers, ensuring consistency in the expert updates during training. To find the optimal ratio of anchor/normal clients N a N c , we conduct experiments varying this ratio; see Figure 9. Sampling half of the clients per round as anchor quickly surpasses the baseline of the common expert and maintains high consistency in subsequent iterations. Using a lower ratio of 30% anchor clients per round also achieved similar performance, allowing some flexibility in the sampling. Contrarily, when we sampled clients randomly from the available pool (i.e., no 'anchor clients'), DDOME shows difficulty improving performance,

as experts' updates become inconsistent. Appendix C shows the end-to-end performance difference across different methods using these sampling ratios for both datasets.

## G Incremental Learning

Incremental learning is a paradigm that aims to update and refine existing knowledge from new data rather than discarding or retraining from scratch. This can benefit scenarios where data is dynamic, scarce, or costly to acquire and where learning models must adapt to changing environments or tasks. We performed a comprehensive comparison using the same benchmarking methods in Table 1 to contrast each algorithm's learning process.

## G.1 Dynamically increase the client's pool

For this setup, we split the CIFAR100 dataset into five groups with non-overlapping labels. Each group held 20 different clients with random samples within the label range. Then, we allowed only one group of labels to be trained for 200 iterations. Afterward, we increased the pool of clients with a new group each 200 iterations, monitoring the global accuracy of the models over time. In Figure 10, we can observe that DDOME is not affected if the entire set of clients is not present from the outset; its gating function develops adaptively, without compromising its ability to capture the old distributions. In contrast, Fed-Mix drops its performance by approximately 4% compared to the original results in Table 1.

Figure 10: Incremental Learning scenario on CIFAR100, dynamically increasing the total pool of clients.

<!-- image -->

## G.2 Dynamically switch the client's pool

We employ a cyclical learning approach based on the first setup for the second scenario. Instead of simply increasing the total pool of clients, we only allow one of the five groups of clients to contribute to the training process at a time. This means that every 600 iterations, we switch the pool of available clients, allowing us to see new labels and ensuring that the labels seen during the initial iterations will never be seen again during the training process. This cyclical approach allows us to benefit from the data's diversity while ensuring that the model is constantly being exposed to new information.

Figure 11 illustrates that even when DDOME is approximately 2% below FedAvg at the end of the training, the former continues to improve. At the same time, the other methods begin to decline over the iterations. This is likely due to the anchor clients acting as regularizers to adjust the gradient directions during optimization, as the client pool presents a more complex setup. The anchor clients can provide a more stable optimization process.

Figure 11: Incremental learning scenario on CIFAR100, dynamically switching the total pool of clients

<!-- image -->

## H Performance under matching number of experts M = K

We present additional experiments to compare FedMix vs. DDOME , using the same number of total experts. I.e., M = K to disentangle our method's behavior under different experts. The results are shown in Table 7, where it is evident that even if we send the complete set of experts per worker, our approach performs better. Yet, the resulting model might be less accurate than the common expert, implying that malicious 'interference' exists.

Table 7: Best global test accuracy reported during training on the CIFAR10 dataset under Dirichlet distribution ( α = 0 . 1 ) with a fixed number of models communicated to each client. Both methods were initialized from the same common expert with an initial accuracy of 73.39%.

|               | M = K = 2   | M = K = 5   |
|---------------|-------------|-------------|
| Common Expert | 73.39%      | 73.39%      |
| FedMix        | 42.76%      | 43.86%      |
| DDOME         | 60.16%      | 75.77%      |

## I Comparison against Domain Generalization Methods

Our scenario can also be framed as a Domain Generalization problem. Thus, we evaluate DDOME against state-of-art methods, such as FedSR [26] and FedADG [34], that handle robustness to distribution shifts on test-time. Results in Table 8 demonstrate that the ability of FedADG and FedSR to evaluate unseen domains is tightly bound to a small number of clients. Once we increase the underlying distribution (e.g., 100 different clients), these methods cannot exploit the cross-relationship among domains [1].

Table 8: Best global test accuracy reported during training on the CIFAR10 dataset using quantity-based label imbalance. We sample 10 (of 100 available) random clients during 900 iterations with replacement. All methods were initialized from the same common expert reported in the Table.

| Common Expert   | 93.05%   |
|-----------------|----------|
| FedSR [26]      | 28.24%   |
| FedADG [34]     | 41.83%   |
| DDOME           | 87.86%   |

## J Clustering analysis

To provide a more extensive comparison of our expert models, it is essential to highlight that the core idea is not to summarize clients into several models, as many clustering-related works do. Clustering methods are limited to scenarios where clients are inherently grouped; all clients in the same group will have similar local data distributions, while clients across groups will share few data. Instead, we target a more realistic scenario, where each client has a non-iid and mixed-data distribution, making client clustering based on local distributions less meaningful. To illustrate this, we have performed an example of client clustering using K-means on local class distributions as shown in Figure 12,

where each dot represents one client and the annotated numbers are this client's two main data classes. The color represents the K-means clustering result. Clearly, clustering does not create meaningful groups of clients, and training individual experts in each group does not provide any specialization of experts.

Figure 12: Clients clustering with label frequency.

<!-- image -->

## K Theoretical communication cost analysis of DDOME and FedMix

## Variable definitions

M : number of expert models.

R : number of communication rounds.

S : number of clients.

N

: active clients.

k : topk experts ( k ≤ M ).

P r : parameters of router.

P e : parameters of a single expert.

P c : parameters of common expert.

ϵ : size to communicate an expert index.

## K.1 Analysis of DDOME

## Communication Cost Derivation

Total cost = initial setup + cumulative round cost.

Initial Cost ( C init ) : This one-time, server-to-client cost involves sending the common expert model to all clients S .

<!-- formula-not-decoded -->

Per-Round Cost ( C round ) : For each of the R rounds, the cost is the sum of downlink (Server-toClient) and uplink communication (Client-to-Server).

Downlink Cost ( C down ) : Router ( P r ) and k experts to N active clients.

<!-- formula-not-decoded -->

Uplink Cost ( C up ) : topk indices + updates for router + k experts to N active clients.

<!-- formula-not-decoded -->

Note: The communication here does not represent computational order.

The total cost for a single round is therefore:

<!-- formula-not-decoded -->

Total Communication Cost ( C total ): The total cost over R rounds is the sum of the initial cost and the cumulative round costs.

<!-- formula-not-decoded -->

Since the parameter sizes are much larger than index sizes ( P r , P e ≫ ϵ ), we can approximate the total cost as:

<!-- formula-not-decoded -->

## K.2 Analysis of FedMix

The FedMix algorithm communicates all experts in every round.

Communication Cost Derivation

Per-Round Cost ( C round, FedMix ) In each round, every active client downloads and uploads all the models.

Downlink Cost ( C down, FedMix ) : The server sends all M experts ( M · P e ) to each of the N active clients.

<!-- formula-not-decoded -->

Uplink Cost ( C up, FedMix ) : Each of the N active clients sends back the updated parameters of the M experts.

<!-- formula-not-decoded -->

Total Communication Cost ( C total, FedMix ) The total cost is the per-round cost multiplied by the number of rounds, R .

<!-- formula-not-decoded -->

## K.3 Limit Analysis

This analysis examines the asymptotic behavior of the communication costs under the following assumptions:

- The number of active clients is a fixed fraction of the total: N = S/c for some constant c &gt; 0 .
- The parameter size of the initial common expert is equal to that of a single expert: P c = P e . Note: This is true in our experiments.

## K.3.1 Analysis as the number of experts M →∞

## · DDOME :

<!-- formula-not-decoded -->

The cost is independent of M , implying that the communication cost is bounded and does not increase as the number of experts grow.

## · FedMix :

<!-- formula-not-decoded -->

The cost is a linear function of M . As the number of experts grows, the cost grows without bound. This means the communication cost is unbounded, making the algorithm very unscalable compared to DDOME.

## K.3.2 Analysis as Number of Rounds R →∞

We analyze the cost as the training process becomes infinitely long.

- DDOME The cost is of course unbounded but grows linearly with R . The rate of growth is given by the partial derivative with respect to R :

<!-- formula-not-decoded -->

- FedMix The rate of growth with respect to R is:

<!-- formula-not-decoded -->

Comparison Both costs are unbounded as R →∞ . However, comparing their growth rates reveals that for k ≪ M , DDOME grows significantly slower, making it more efficient for longer training durations.

<!-- formula-not-decoded -->

## K.3.3 Analysis as Number of Clients S →∞

In this case we will add the ϵ term back in, as it directly depends on S .

- DDOME The rate of growth with respect to S is:

<!-- formula-not-decoded -->

- FedMix Algorithm The rate of growth with respect to S is:

<!-- formula-not-decoded -->

Comparison . The goal is to prove that FedMix rate of change is greater. We will show that this is indeed the case under certain (realistic) conditions.

We seek to demonstrate that: 2 R c MP e &gt; P e + R c (2 P r +2 kP e + kϵ )

We proceed by rearranging this inequality to find the condition under which it holds true.

<!-- formula-not-decoded -->

2( M -k ) -2 P r P e -kϵ P e , is a positive number slightly less than 2( M -k ) and almost certainly greater than 1.

Thus, the inequality FedMix &gt; DDOME holds true if:

<!-- formula-not-decoded -->

This condition is generally satisfied in practical scenario. For example we can plug in our own experimental values from cifar10 and Yahoo!

<!-- formula-not-decoded -->

In this case it is easy to see that this inequality is indeed satisfied, showing us that the rate of change of the communication cost of FedMix, as a function of the number of clients will remain bounded below by DDOME.

Conclusion . Across all scaling dimensions of interest-experts (M), rounds (R), and clients (S)-DDOME communication is fundamentally more efficient and scalable.

## L Missing Proof from Section 3

Here, we first state the assumption of the theorem in Section 3.

Assumption 1. Let Σ ∈ R d × d be a positive definite matrix. There exists a constant c sc &gt; 0 such that

<!-- formula-not-decoded -->

For the simplicity of the analysis, we define

<!-- formula-not-decoded -->

## L.1 Analysis of the Initialized Router Weights

Let v 1 , . . . , v m ∼ N ( 0 , 1 d I d ) . Write V = [ v 1 , . . . , v m ] ∈ R d × m . By standard concentration of covariance matrix, when m ≪ d , we have that with probability at least 1 -exp( -Θ( d ))

<!-- formula-not-decoded -->

This gives that σ m ( V ) ≥ 1 √ 2 . For the simplicity of the analysis, we assume that this event happens. Let the singular value decomposition of V be given by V = U V Σ V R ⊤ V .

Lemma 1. Let u ⋆ 1 , . . . , u ⋆ m ′ be a fixed orthonormal list, and let U ⋆ = [ u ⋆ 1 , . . . , u ⋆ m ] with some m ′ ≤ d . Then with probability at least 1 -exp ( -Θ ( m 2 m ′ 2 )) we have that

<!-- formula-not-decoded -->

Proof. Since v j ∈ N ( 0 , 1 d I d ) for all j ∈ [ m ] , we have that U ⋆ ⊤ v j =: a j ∼ N ( 0 , 1 d I m ′ ) . Therefore

Therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the standard concentration inequality of Gaussian random vector, we have that

<!-- formula-not-decoded -->

Thus, with probability at least 1 -exp ( -Θ ( m 2 m ′ 2 )) we have that

<!-- formula-not-decoded -->

Recall that V = U V Σ V R ⊤ V . Therefore

<!-- formula-not-decoded -->

Thus, we have that

<!-- formula-not-decoded -->

Recall that the ground-truth router parameters v ⋆ 1 , . . . , v ⋆ m is an orthonormal list, and the concatenation is denoted by V ⋆ , and that U V denotes the left singular vectors of V . By Lemma 1, we have that

<!-- formula-not-decoded -->

when m ≤ √ d 2 . Therefore, dim ( span ( U V ) + span ( V ⋆ )) = 2 m . Otherwise, if dim ( span ( U V ) + span ( V ⋆ )) &lt; 2 m , there exists x ∈ span ( U V ) ∩ span ( V ⋆ ) , which implies that there exists s , s ′ ∈ R m such that U V s = V ⋆ s ′ . This gives

<!-- formula-not-decoded -->

which is a contradiction. Therefore, we can let ˆ U V ∈ R d × 2 m to be the matrix representing an orthonormal basis of span ( U V ) + span ( V ⋆ ) where the top m columns are U V . Let ˜ U V ∈ R d × ( d -2 m ) be the orthogonal complement of ˆ U V . Then we can re-write x ∼ N ( 0 , 1 d I d ) as

<!-- formula-not-decoded -->

We make the following assumption:

Assumption 2. Let Σ ∈ R d × d be a positive definite matrix. There exists a constant c sc &gt; 0 such that

<!-- formula-not-decoded -->

Let c 2 m denote the lower bound in Assumption 2 with Gaussian measure 1 2 m .

Lemma 2. Let ˆ U V , ˜ U V , s 1 , s 2 be defined as above. If Assumption 2 hold, then we have that

<!-- formula-not-decoded -->

Proof. Write s 1 = [ q ⊤ 1 , q ⊤ 2 ] ⊤ where q 1 , q 2 ∈ R m . Then we have that q 1 , q 2 ∼ N ( 0 , 1 d I m ) . Moreover, by the construction of ˆ U V , we have that q 2 is independent of v 1 , . . . , v m and q 1 = U ⊤ V x . Therefore

<!-- formula-not-decoded -->

where P 1 = [ I m 0 ] and P 2 = [ 0 I m ] . Recall that q 1 = U ⊤ V x . Then we have that x ⊤ v j = q ⊤ 1 U ⊤ V v j . Since V has the SVD

<!-- formula-not-decoded -->

we have that U ⊤ V v j = Σ V r j where r j is the j th row of R V . Thus

<!-- formula-not-decoded -->

Let ˆ q 1 = R V Σ V q 1 . Then we have that q 1 = Σ -1 V R ⊤ V ˆ q 1 , and

<!-- formula-not-decoded -->

Moreover, the indicator function can thus be written as

<!-- formula-not-decoded -->

Thus, the original objective is re-written as

<!-- formula-not-decoded -->

By Assumption 2, we have that

<!-- formula-not-decoded -->

Therefore, we have that

<!-- formula-not-decoded -->

Proof of Theorem 1. Notice that the objective in ( ?? ) is a quadratic form with Hessian E x [ I j ( x ) xx ⊤ ] . Recall that x = ˆ U V s 1 + ˜ U V s 2 . Thus, we can write

<!-- formula-not-decoded -->

since s 2 is independent of I j ( x ) . Based on Lemma 2, we have that the objective in ( ?? ) is a strongly convex quadratic since the two expectation terms are strictly positive. Therefore, ˆ w j has the form

<!-- formula-not-decoded -->

Notice that for y we have

Therefore, ˆ w j can be written as

<!-- formula-not-decoded -->

Let w ⋆ r be given for any r ∈ [ m ] . Then we have that

<!-- formula-not-decoded -->

Therefore, the difference between ˆ w j and w ⋆ r is given by

<!-- formula-not-decoded -->

Due to the orthogonality between ˆ U V and ˜ U V , the magnitude of ˆ w j -w ⋆ r must be lower bounded by the magnitude of its projection onto ˜ U V . Thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, we can obtain that

<!-- formula-not-decoded -->

We further notice that

<!-- formula-not-decoded -->

Therefore, we have that

<!-- formula-not-decoded -->

By Lemma 1, we have that with probability at least 1 -exp ( -Θ ( m 4 )) , we have that ∥ ∥ ∥ ˆ U ⊤ V W ⋆ ∥ ∥ ∥ 2 ≤ 3 m √ d . Moreover, diving deeper into ∥ ζ ∥ j , we have

̸

<!-- formula-not-decoded -->

̸

Since E s 1 [ I j ( x )] -1 E s 1 [ I j ( x ) I ⋆ ℓ ( x )] ≤ 1 , and by Lemma 2, we have

<!-- formula-not-decoded -->

Therefore, we can obtain that

<!-- formula-not-decoded -->

for some constant c . This gives that

̸

<!-- formula-not-decoded -->

where

This gives

Therefore,

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Using a similar approach, we can conclude that

<!-- formula-not-decoded -->

Therefore, for the test loss, we can write

<!-- formula-not-decoded -->

This gives that

<!-- formula-not-decoded -->

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly and accurately reflect the contributions and scope of the paper, explicitly highlighting empirical improvements and theoretical insights. The stated claims match the provided experimental and theoretical results.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: A dedicated section clearly outlines limitations, including dependency on pretrained models, communication overhead, expert initialization concerns, scalability tests, and gating complexity.

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

Justification: The paper clearly states assumptions and provides the complete and detailed proof of the main theoretical result (Theorem 1) in the Appendix, including references to relevant lemmas and explicit assumptions.

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

Justification: The paper clearly discloses architectures, hyperparameters, optimizers, dataset splits, and methodological details necessary for reproducing the experiments.

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

Justification: Although detailed instructions for reproducibility are provided, the paper does not explicitly mention releasing data or code publicly.

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

Justification: Training details including datasets, data splits, hyperparameters, optimizers, and procedures are clearly specified in the experimental section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The paper does not explicitly include error bars, confidence intervals, or other statistical significance measures, due to computational constraints.

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

Justification: The paper clearly describes the hardware, GPU types, memory, execution times, and the distributed training setup for both image and text classification tasks.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: There is no indication of ethical concerns, and the research aligns fully with the NeurIPS Code of Ethics guidelines.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper includes a dedicated 'Broader Impact' section that explicitly discusses positive impacts, such as improved privacy-preserving federated learning, enhanced resource efficiency, and personalized learning in decentralized environments. It also acknowledges potential negative impacts, highlighting the risk of exacerbating fairness issues

if biases present in heterogeneous data are amplified through expert specialization. The authors clearly recommend monitoring and mitigation strategies during deployment.

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

Justification: The paper does not involve releasing datasets or pretrained models that pose high risks of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [No]

Justification: Although datasets and model architectures are cited, explicit mentions of licenses or terms of use for these assets are missing.

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

Justification: The paper does not introduce any new datasets, models, or code assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The research does not involve crowdsourcing or human subject experiments.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The research does not involve human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The research does not use LLMs as any original or core component.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.