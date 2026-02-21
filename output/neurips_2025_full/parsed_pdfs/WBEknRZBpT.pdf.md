## Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling

Hongtao Xu 1 , 2 , 3 Wenting Shen 3 Yuanxin Wei 4 Ang Wang 3 Guo Runfan 2 Tianxing Wang 3 Yong Li 3 Mingzhen Li 2 † Weile Jia 2 †

1 School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences 2 State Key Lab of Processors, Institute of Computing Technology, CAS

3 Alibaba Group 4 Sun Yat-sen University

† Corresponding authors

## Abstract

Long-context supervised fine-tuning (Long-SFT) plays a vital role in enhancing the performance of large language models (LLMs) on long-context tasks. To smoothly adapt LLMs to long-context scenarios, this process typically entails training on mixed datasets containing both long and short sequences. However, this heterogeneous sequence length distribution poses significant challenges for existing training systems, as they fail to simultaneously achieve high training efficiency for both long and short sequences, resulting in sub-optimal end-to-end system performance in Long-SFT. In this paper, we present a novel perspective on data scheduling to address the challenges posed by the heterogeneous data distributions in Long-SFT. We propose Skrull, a dynamic data scheduler specifically designed for efficient long-SFT. Through dynamic data scheduling, Skrull balances the computation requirements of long and short sequences, improving overall training efficiency. Furthermore, we formulate the scheduling process as a joint optimization problem and thoroughly analyze the trade-offs involved. Based on those analysis, Skrull employs a lightweight scheduling algorithm to achieve near-zero cost online scheduling in Long-SFT. Finally, we implement Skrull upon DeepSpeed, a stateof-the-art distributed training system for LLMs. Experimental results demonstrate that Skrull outperforms DeepSpeed by 3.76x on average (up to 7.54x) in real-world long-SFT scenarios.

## 1 Introduction

Long-context capabilities are important for large language models (LLMs) to handle various tasks such as long document summarization, question answering, multi-turn dialogue and code generation. Mainstream LLMs such as Llama [21, 10], Qwen [19] and GPT-4 [18] can support the context window of up to 128K tokens. Google's Gemini [9] can even achieve up to 1M tokens per context window. Typically, additional training phases like long-context supervised fine-tuning (Long-SFT) as well as long-context continue pre-training (Long-CPT) are employed to extend the context length. For example, Llama3 [10] is fine-tuned with 99.89% short sequence (averaging under 1K tokens) and 0.11% long sequence (averaging around 37K tokens). Qwen2.5-Turbo [19] gradually extends context length by training on 40% long sequences and 60% short sequences. Training on those meticulous gathered datasets enables smoothly adaptation of LLMs to longer context while still maintaining the performance on short context tasks.

However, this heterogeneous data distribution in Long-SFT poses significant challenges for existing distributed LLM training frameworks [14, 20, 15], exhibiting sub-optimal efficiency. For instance, the heterogeneous data distribution poses a dilemma for parallelism and memory-reduction strategies. Specifically, long sequences necessitate context parallelism and other memory-reduction approaches

due to their tremendous memory requirements. However, those approaches compromise the training efficiency for short ones due to the overheads like unnecessary communication and GPU underutilization. Moreover, the wide sequence length distribution in long-SFT worsen the mismatch of computation characteristics in Attention module, which exhibit quadratic computational complexity and linear memory consumption [7, 6], leading to another dilemma for load balance problem.

To tackle the above challenges, we propose Skrull, a dynamic data scheduler dedicated for Long-SFT scenarios. Skrull efficiently handle the unique data distributions in Long-SFT scenario through two main components: Distributed-Aware Context Parallelism (DACP) and Global Data Scheduling (GDS). DACP selectively shards sequences and schedules them across different workers to minimize the performance degradation while maintains the ability of handling long sequence. GDS enlarge the scope of scheduling and improve the GPU utilization during training. The two components collaborate with each other at different scheduling granularities. Furthermore, to achieve the optimal performance, we formulate the scheduling process as a joint optimization problem and design a lightweight heuristic algorithm to solve it at runtime. Experimental results demonstrate that Skrull improves the end-to-end training performance by 3.76x on average (up to 7.54x) compared to DeepSpeed, a state-of-the-art distributed LLM training framework.

Our key contributions are summarized as follows:

- We provide a new perspective of data scheduling to address the heterogeneous sequence length distribution.
- We propose a new context parallelism called DACP based on fine-grained data scheduling, which maintaining both the processing capabilities for long sequences and efficiency for short sequences, enabling efficient training on heterogeneous data distribution in long-SFT scenario.
- We implement coarse-grained global data scheduling (GDS) and further formulate GDS and DACP as a joint optimization problem through performance modeling.
- We design a lightweight heuristic algorithm and achieve performance gains by 3.76x on average (with a peak improvement of 7.54×) in real-world datasets.

## 2 Preliminaries

Data Parallelism (DP). Data parallelism [15, 24, 20] partitions the training samples to multiple workers and each worker maintains a complete model weight replica. In each iteration, all workers process a subset of global batch independently and then synchronize the gradients across all DP ranks. However, due to the inherent synchronization semantic in DP, the load balance becomes a noticeable problem, especially in long context scenarios.

Context Parallelism (CP). Context parallelism partitions the input tensor along the sequence length dimension and distributes it to multiple workers [12, 16, 10]. CP is emerging as an inevitable parallel strategy when handling long context. In the Transformer architecture, the primary challenge of CP stems from the parallelization of Attention module because each tokens needs to attend to other tokens in the sequence. Consequently, the communication in CP is inevitable. Notably, DACP, proposed in this paper, leverages data scheduling to minimize the overheads caused by CP and is orthogonal to specific CP implementations.

## 3 Observation

## 3.1 Heterogeneous Sequence Length Distribution

As shown in Figure 1a, we observe pronounced variance in the sequence length distribution across real-world Long-SFT datasets, including Wikipedia [2], LMsysChat1M [25] and ChatQA2-Long-SFT [1]. Among them, the sequence length distribution of ChatQA2-long-SFT exhibits a bimodal pattern, where the proportions of long and short sequences are nearly equal. Specifically, approximately 40% of sequences are shorter than 8K tokens, while the remaining 60% exceed this threshold. As comparison, long-tail distributions represent another typical pattern in Long-SFT datasets. In Llama3's internally collected Long-SFT datasets [10], we find that 99.89% of sequences are under 1K

tokens on average, while the remaining 0.11% are approximately 37K tokens, showcasing extremely skewed long-tail distribution. Due to data accessibility constraints, we plot the sequence length distribution of Wikipedia and LMsysChat1M in Figure 1a, which have the identical feature with Llama3's Long-SFT dataset. Table 1 lists the portions under different lengths thresholds for these three datasets, highlighting the heterogeneous sequence length distribution in Long-SFT.

<!-- image -->

- (a) Sequence length distribution.
- (b) Performance (FLOPS) with different CP workers.

Figure 1: Sequence length distribution on different datasets, and corresponding performance impact.

Table 1: Percentage of sequence length in real-world datasets.

| Dataset          | <1K    | <4K    | <8k    | <32K   | <128K   | Longest   |
|------------------|--------|--------|--------|--------|---------|-----------|
| Wikipedia        | 87.88% | 99.34% | 99.92% | 99.99% | 100.0%  | 78K       |
| LMsysChat1M      | 87.12% | 99.35% | 99.87% | 99.98% | 99.99%  | 1643K     |
| ChatQA2-Long-SFT | 21.92% | 31.48% | 40.43% | 99.86% | 100.0%  | 99K       |

## 3.2 Performance Degradations for Short Sequences

In this section, we discuss our observation on the performance degradations and GPU under-utilization for short sequences in Long-SFT training. During the training process, the context parallelism degree and other memory reduction strategies such as gradient accumulation are set to accommodate the longest sequence in datasets to avoid out-of-memory errors (OOMs). However, these training settings degrade their performance for the shorter sequences, which make up the majority in Long-SFT datasets. As shown in Figure 1b, we test the performance of Attention module [6] under different CP degrees. Results demonstrate, especially for the short sequences, higher CP degree exacerbates kernel execution efficiency. Additionally, context parallelism also brings unnecessary communication overhead to short sequences. Also, the memory reduction strategies tailored to long sequence lead to low GPU memory utilization for the most time.

## 4 Skrull

We introduce design of Skrull and the efficient implementation for online Long-SFT training in this section. Figure 2 illustrates the workflow of Skrull. From the perspective of data scheduling, Skrull consists of two parts: (i) Global data scheduling (GDS): For every iteration, Skrull takes the global batch as input and employs coarse-grained scheduling to generate the optimal micro-batches for each DP ranks. (ii) Distributed-aware Context Parallelism (DACP): Taking the micro-batch produced in GDS, Skrull further employs finer-grained scheduling to selectively distribute the sequences and assign them to different CP workers. For the convenience of formulation, we sequentially introduce DACP in Section 4.1, GDS in Section 4.2 and the efficient implementations in Section 4.3.

Figure 2: Workflow of Skrull. (a) Offline profiling: Given model and training settings, it provides performance estimation for data scheduling. (b) GDS: produce optimal batching strategies for DACP. (c) DACP: dynamically scheduling data to specific hardware with balanced workload and minimum overheads. (d) Performance gains of DACP: it shows how the reduced communication volumn and overlapping improve the performance.

<!-- image -->

## 4.1 Distributed-aware Context Parallelism

To simultaneously achieve high efficiency for all the sequences, we propose d istributeda ware c ontext p arallelism (DACP). As shown in Figure 2(c), DACP dynamically determines whether to distribute the sequences to avoid unnecessary overheads or not. On the one hand, DACP preserves the original context parallel settings to maintain the ability of handling long sequences. On the other hand, DACP selectively schedules short sequences entirely within a single device to minimize the degradation. Therefore, based on distinct computational characteristics, DACP classifies sequences into two categories: (i) distributed sequences requiring context parallelism, and (ii) local sequences needing efficient processing and intended to reside entirely within a single device. Notably, these sequences are still processed within a shared CP group without increasing the number of GPUs used for training. Furthermore, as illustrated in Figure 2(d), DACP brings an additional opportunity to overlap the communication of distributed sequences and the computation of local sequences in Attention module due to the inherent independence between distributed and local sequences.

However, the scheduling process presents significant challenges. First, inappropriate sequence classification may lead to out-of-memory errors (OOMs). Second, the local sequences are varying in length and pose load imbalance issue across CP ranks. To fully explore the relationship between scheduling plans and performance gains, we first analyze the computation and memory features in Appendix C. Through offline profiling, we model the computation (see in Appendix C.2) by FLOPs function and latency estimation function T comp . Additionally, we map the sequence length to the memory consumption and derive the BucketSize C which indicates the capacity of total sequence length per ranks. The BucketSize C plays a vital role in measuring the memory constrain during Skrull's scheduling. More details are listed in Appendix C.1. Similarly, we model communication volume function V olume and latency function T comm , as detailed in Appendix C.3. Finally, we formulate the scheduling process as an optimization problem as follows. The frequently used notions are listed in Table 2.

DACP Formulation. We first define the sequence classification array D ∈ { 0 , 1 } K ( 0 for local sequence and 1 for distributed sequence) and local sequence assignment matrix P ∈ { 0 , 1 } K × N ( 1 for assignment and 0 for not). For example, D k = 1 indicates that the k -th sequence with length of S k is scheduled to be computed in a distributed manner. Similarly, P kj = 1 indicates that the k -th sequence is assigned to device j , implying D k = 0 . Given a micro-batch comprising K sequences with lengths S k ( k = (0 , . . . , K -1) ), BucketSize C and CP degree N , the scheduling process of

Table 2: Symbols used in this paper.

| Symbol                          | Description                                                                                                                                                                                                              | Symbol                 | Description                                                                                                                 |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| S k D k N T comp B k ij V olume | Length of the k -th sequence in a batch. Distribute k -th sequence or not. CP degrees. Computation cost estimation. Assign k -th sequence to i - th DP rank and j - th micro-batch. Communication volume count function. | C P kj FLOPs T comm ws | BucketSize per rank. Assign k -th sequence to CP rank j FLOPs estimation function. Communication cost estimation. DP degree |

## DACP can be formulated as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, our optimization goal is to find the optimal D and P to minimize TDACP , which represents the total duration in one micro-batch. As shown in Equation 1, TDACP is determined by the maximum execution time Time j across all CP ranks j . Specifically, as described in Equation 2, Time j consists of two components: (1) the overlapping term, defined as the maximum of the communication time T comm ( V ) and the computation time T comp ( Local j ) for local sequences, and (2) the computation time T comp ( Dist ) for distributed sequences. Here, T comm depends on the communication volume V, as modeled in Equation 5. Similarly, T comp utilizes the results from Equations 3 and 4, which compute the FLOPs for local sequences on CP rank j and distributed sequences, respectively. Finally, Equation 6 ensures the completeness of data scheduling, while Equation 7 enforces the memory constraint.

## 4.2 Global Data Scheduling

Section 4.1 discusses the data scheduling in the scope of micro-batch. However, only relying on scheduling in DACP is insufficient. The reasons are as follows.

First, the heterogeneous sequence length distribution also leads the to severe load imbalance across different micro-batches, resulting in the sub-optimal training efficiency in Long-SFT scenarios. Second, to achieve maximum performance gains in DACP, meticulous micro-batching strategy is essential. For example, pairing long and short sequences with appropriate memory pressure can expand the valid scheduling space for DACP. Specifically, micro-batches with large total sequence lengths increase the risk of OOMs and limit the optimizations in DACP, such as selective sharding. In contrast, micro-batches with small total sequence lengths introduce GPU under-utilization, degrading the end-to-end performance. Therefore, as shown in Figure 2(b), Skrull employs Global Data Scheduling (GDS), which derives the optimal micro-batching strategy from the global batch. We limit the scheduling scope to the global batch because it represents the maximum scope that maintains mathematical equivalence for mainstream optimizers such as Adam [13] and AdamW [17].

Joint Formulation We re-formulate the scheduling process as a joint optimization problem that integrates both DACP and GDS. We first define the batching matrix B kij ∈ { 0 , 1 } K × N , which

indicates whether the k -th sequence is scheduled into the j -th micro-batch of DP rank i . Given a global batch B consisting of K sequences with lengths S k , we re-formulate the scheduling process as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, TDACP represents the function in Equation 1. Equation 9 ensures all sequences are assigned exactly once. The memory constraint in Equation 10 prevents the OOMs while the Time ij shown in Equation 11 provides cost estimations for each micro-batch using DACP formulation in Equation 1. As shown in Equation 8, the total execution duration per iteration is determined by the DP rank with the longest cumulative execution time across its micro-batches.

Overall, the optimization target is to the minimize the total execution time per iteration by deducing the optimal scheduling plan, which is represented by a combination of B kij , D k and P kj .

## 4.3 Efficient Online Scheduling

Although some solvers like [4] can derive the optimal scheduling plan, its long solving time makes it impractical for scheduling during runtime. To achieve online scheduling during Long-SFT, we resort to design lightweight heuristic scheduling algorithm. Notably, our scheduling algorithm is integrated into the DataLoader and introduces near-zero overhead to the training process.

## 4.3.1 Memory vs. Computation: Trade-off Analysis

Memory and Computation are the key factors related to the performance, as shown in the formulations of scheduling. We should achieve optimal performance while not violating memory constraint, presenting a trade-off. Therefore, we first analyze the trade-off between Computation and Memory when deducing the scheduling strategies, highlighting the considerations when designing the scheduling algorithms.

Sequence classification: deduce the array D . We analyze the sequence classification (array D in Section 4.1). From the perspective of Computation , D impacts the communication volume and computation of sharded sequences (Equation 5 and 4). More sharded sequences will incur more performance degradation, which comes from both communication overhead and kernel execution (refer to Section 3.2). However, from the perspective of Memory , more distributed sequences will bring more balanced memory consumption (Equation 7), which can lower the risk of OOMs, as the remaining local sequences with varying lengths are hard to be assigned evenly. Besides, although the overlapping in DACP can alleviate the performance degradation problem to some extent (Equation 2), it is still non-trivial to decide the optimal classification array D .

Local sequence assignment: deduce the matrix P . Then, we analyze local sequence assignment, which is represented by P . From the perspective of Computation , P impacts the Equation 3, which implies the computation workload in each CP ranks, thus affects the load balance. The ideal situation is to balance the local sequences for computation balance among CP ranks.However, from the perspective of Memory , the scheduling which balances the computation leads to the unbalance of memory consumption, which increases the risk of OOMs.

Unfortunately, we cannot balance the computation and memory at the same time. The reason is that, after applying FlashAttention [7, 6], the correlation between computation complexity and and sequence length ( n ) is O ( n 2 ) , however, the correlation between memory is O ( n ) . Moreover, with the sequence length increasing, the portion of Attention module gradually dominates the computation load, making it more difficult to balance the computation and memory. Worse still, the model

configuration (e.g., KV heads, hidden size) also impacts. Due to the limited page, we list the details in Appendix C.

Therefore, we need to carefully deal with the memory footprint balance and the computation complexity balance and we design the following heuristics.

## 4.3.2 Heuristics

Scheduling Algorithm of DACP. We first summarize three principles of algorithm design in DACP. (i) Avoid sharding : We strive to avoid sequence sharding and assume that all sequences will be handled locally first. (ii) Prioritize computation : We prioritize balancing computation over memory to achieve better performance. (iii) Roll-back mechanism : We continuously monitor the estimated memory consumption and revert decisions when necessary. The roll-back mechanism guarantees the memory constrains outlined in Equation 7 and Equation 10, while enabling more aggressive scheduling attempts based on (i) and (ii). Our heuristic for DACP is listed in Algorithm 1. Given a micro-batch containing K sequences with lengths S [ K ] and a predefined BucketSize C , the algorithm outputs the sequence classification and assignment results in the form of an array ret . In this array, a value of -1 at the i-th position indicates that the i-th sequence is to be sharded, while a value v = (0 , . . . , ws -1) indicates that the i-th sequence is assigned to CP rank v entirely. To better balance computation while ensuring memory constrains, we maintain two arrays during DACP scheduling: RemainBucket RB and Loads L , which represent the current memory budget and computation load, respectively. We first sort the sequences in ascending order. For the each sequence, we sequentially assign it to the bucket (as well as CP rank) with minimum L to avoid sharding and prioritize balancing computation (line 6-8). If the bucket cannot accommodate the sequence, we attempt to assign it to the bucket with the maximum RB to avoid sharding (line 10-12). If both attempts fail, we classify the sequence as a distributed sequence and attempt to shard it (line 14-16). However, if the bucket with minimum of RB cannot handle the sub-sequence after sharding, this indicates that the earlier process incorrectly classified inappropriate sequences as local sequences within this bucket. To address this, we employ a roll-back mechanism (line 18 and Appendix B.1). This mechanism identifies a local sequence in the bucket, shards it to reduce memory pressure, and resumes the assignment process. If the roll-back fails due to the absence of local sequences in the bucket, we return a DACP scheduling error. In such cases, GDS will also revert the batching plan (see the Section 4.3.2). Notably, every assignment updates RB and L through the predefined functions UpdateLocal and UpdateAll . The details of these functions including RollBack are further elaborated in Appendix B.1.

Scheduling Algorithm of GDS. Algorithm 2 demonstrates the heuristic scheduling algorithm of GDS. Given a global batch containing K sequences with lengths S [ K ] , DP world size ws and DP rank dp \_ rank , the algorithm returns the scheduling result mbs , which consists of multiple micro-batches as inputs for Algorithm 1. We summarize three principles in our algorithm design. (i) Prioritize computation : We prioritize balancing computation across DP workers. To achieve this, we estimate the FLOPs (Appendix C.2) and employ a bin-packing algorithm to balance computational workloads at a coarse granularity (line 1). (ii) Pair long and short sequences : We sort the sequences within each DP rank and batch them in an interleaved manner (line 7). This approach ensures that long sequences are assigned more evenly across micro-batches. Additionally, each micro-batch contains several short sequences, enhancing both task overlapping and load balancing. (iii) Improve memory utilization : We estimate the total memory requirements and try to improve the concurrency with less number of micro-batches. Thanks to the roll-back mechanism (line 8), this method maximizes memory utilization while not increase the risk of OOMs. As shown in line 5, we gradually increase the number of micro-batches if the scheduling fails and requires a roll-back.

## 5 Evaluation

Experimental Setup. We conduct experiments using a testbed consisting of 4 nodes interconnected via a high-performance InfiniBand network, with each node equipped with 8 Nvidia H100 GPUs connected via 900GB/s NVLink. Then, We implement Skrull on top of DeepSpeed, a state-of-the-art distributed LLM training system and enable Zero-2 optimization as our baseline. Additionally, we implement sorted batching method in LongAlign [3] for more comparison, which sort the dataset by sequence length and select random consecutive groups for each batch to improve the long-SFT

## Algorithm 1 Heuristic scheduling algorithm of DACP

```
Require: SeqNum K , SeqLens S [ K ] , BucketSize C , CP degree N Ensure: Scheduling Result ret [ K ] 1: Sort(SeqLens, ascending=True) 2: for i = 0 to N -1 do 3: RB [ i ] ← C, L [ i ] ← 0 ▷ Initialization 4: end for 5: for i = 0 to K -1 do 6: t ← argmin ( L ) ▷ Find rank t with minimum workload 7: if RB [ t ] ≥ S [ i ] then 8: ret [ i ] ← t , UpdateLocal ( i, t ) 9: else 10: t ← argmax ( RB ) 11: if RB [ t ] ≥ S [ i ] then 12: ret [ i ] ← t , UpdateLocal ( i, t ) 13: else 14: t ← argmin ( RB ) 15: if RB [ t ] ≥ S [ i ] /N then 16: ret [ i ] ←-1 , UpdateAll ( i ) ▷ Distribute the sequence 17: else 18: Assert RollBack ( t, RB, L ) 19: i ← i -1 ▷ Roll-back to avoid OOMs 20: continue 21: end if 22: end if 23: end if 24: end for 25: return ret
```

## Algorithm 2 Heuristic Scheduling Algorithm of GDS

```
Require: SeqNum K , SeqLens S [ K ] , BucketSize C , CP degree N , DP WorldSize ws , DP_Rank dp _ rank Ensure: Micro-batches mbs 1: Bin [ ws ] ← Binpack ( ws, FLOPs ( S [ K ])) ▷ Coarse-fined balance 2: Subset ← Bin [ dp _ rank ] , init ←⌈ Sum ( Subset ) /C × N ⌉ -1 3: Sort ( Subset , ascending=True) 4: while init ≤ K +1 do 5: init ← init +1 , mbs ← [] 6: for j ← 0 to init do 7: mbs. append ( Subset [ j :: init ]) ▷ Pair long and short sequences 8: if Sum ( mbs [ -1]) ≥ C × N or not scheduling_in_DACP ( mbs [ -1]) then 9: Continue ▷ Rollback if overload or DACP sheduling fails 10: end if 11: end for 12: end while 13: return mbs
```

training efficiency. We evaluation our optimizations on Qwen2.5-0.5B and Qwen2.5-7B using the three real-world datasets described in Section 3.1. Although Wikipedia and LMsysChat1M are not specifically gathered for Long-SFT, we still choose them as our evaluation datasets due to their long-tail distribution, which is exactly identical to Meta's in-house Long-SFT dataset [10]. In contrast, ChatQA2-long-SFT dataset [1] is specifically gathered for Long-SFT and exhibits bimodal distribution of data length, which is also similar to the dataset mentioned in [19]. Through offline profiling, we configure the BucketSize to 26K and 13K for Qwen2.5-0.5B and Qwen2.5-7B, respectively. Further details regarding BucketSize configuration can be found in Appendix C.1. All the experiments share the same training settings with &lt;DP=4, CP=8, BatchSize=64&gt;, zero-2 enabled and selective recomputation strategy except for training Qwen-2.5-7B with ChatQA2-long-SFT dataset. Due to the increased memory requirements, we adjust its parallel settings with &lt;DP=2, CP=16, BatchSize=40&gt;. The global batch size is equal to DP size multiplied by BatchSize. Due to the limited page, we list precision validation in Appendix A.

Figure 3: Overall performance and step-by-step evaluation. The settings represent the DP degree, CP degree and batch size, respectively.

<!-- image -->

Figure 4: Left (a) shows the performance impact under different BatchSize and BucketSize. Right (b) shows Skrull's effectiveness and compatibility with Lora in larger models.

<!-- image -->

Overall Performance. Figure 3 illustrates the speedup achieved by Skrull, with the performance measured in terms of average iteration time. Skrull significantly outperforms the sorted batching strategy Experimental results demonstrate that Skrull outperforms DeepSpeed and sorted batching method by an average of 3.76x and achieve peak improvement of 7.54×. The average speedups for Qwen-0.5B and Qwen-7B are 5.50x and 2.03x, respectively. We attribute this difference to the variation in BucketSize, which directly influences the valid data scheduling space. Also, Skrull outperforms sorted batching method by an averge of 3.45x with the peak improvement of 6.85x. Additionally, from the perspective of datasets, the performance on Wikipedia and LMsysChat1M are similar due to the similar data distribution, which both exhibit long-tail feature. In this distribution, the short sequences dominate the datasets thus showcasing more optimization potential. In contrast, the long sequences also account for the majority in ChatQA2-Long-SFT dataset, which exhibits bimodal distribution, leading to relatively small optimization space. Specifically, when training Qwen-7B with this datasets, the major sequence length exceeds the BucketSize thus leading to limited speedup. We can further extend the BucketSize by combining more optimization techniques like parameter-efficient fine-tuning (PEFT) [11, 5].

Step-by-step Evaluation. Additionally, we conduct step-by-step evaluation with the same training settings mentioned above. As shown in Figure 3, we successively enable DACP and GDS to test the effectiveness of each part in Skrull. Experimental results show that both components are effective and can cooperate well to further improve the end-to-end system performance in Long-SFT.

Performance Impact of BatchSize and BucketSize. To investigate the performance impact of BatchSize and BucketSize, we conduct experiments on ChatQA2-long-SFT using Qwen2.5-0.5B with the default setting of &lt;DP=1, CP=8, BatchSize=64, BucketSize=26K&gt;. As shown in Figure 4(a), we adjust the BatchSize from 8 to 54 and the end-to-end speedup also improves. We attribute this

performance gain to the expanded scheduling scope afforded by larger batch sizes. However, as the BatchSize increases further, the sampled batches gradually converge to the sequence length distribution of the dataset, causing the performance gains stabilized within a reasonable range. Additionally, we also evaluate the effect of BucketSize. Figure 4(a) shows that increasing the BucketSize from 8K to 32K (values in parentheses) improves the speedup until an out-of-memory (OOM) error occurs. This indicates that while a larger BucketSize enhances performance, it also raises the risk of OOM errors. Therefore, it is important to set appropriate BucketSize, highlighting the importance of performance modeling module in Skrull.

| Table 3: Scheduling Strategies Comparison   | Table 3: Scheduling Strategies Comparison   | Table 4: Latency Comparison per Iteration   | Table 4: Latency Comparison per Iteration   | Table 4: Latency Comparison per Iteration   | Table 4: Latency Comparison per Iteration   |
|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|
| Method                                      | Speedup                                     | Iteration                                   | Baseline                                    | RR                                          | Skrull                                      |
| RR w/ roll-back                             | 1.17 ×                                      | 1                                           | (36, 36)                                    | (10, 49)                                    | (21, 51)                                    |
| RR w/o roll-back                            | OOM                                         | 2                                           | (40, 40)                                    | (29, 47)                                    | (37, 50)                                    |
| Skrull w/ roll-back                         | 1.40 ×                                      | 3                                           | (35, 35)                                    | (14, 45)                                    | (32, 49)                                    |
| Skrull w/o roll-back                        | OOM                                         | 4                                           | (46, 46)                                    | (46, 47)                                    | (45, 49)                                    |

Case Study In this section, we present a quantitative analysis of the training process of Skrull. We conduct experiments using the Qwen2.5-0.5B model with the ChatQA2-Long-SFT dataset under the configuration &lt;DP=1, CP=8, BatchSize=64&gt;. To evaluate the effectiveness of our heuristics implementation, we compare it against a round-robin (RR) scheduling strategy, which assigns sequences in a simple round-robin manner (details in Appendix B.2). Additionally, we test both scheduling algorithms with and without roll-back mechanism to further assess the trade-off design in Skrull. As shown in Table 3, Skrull significantly outperforms the RR scheduling. We analyze this result by examining quantitative data in the first four iterations. Table 4 (presented in tuple format) reports the minimum and maximum peak memory usage (in gigabytes) across all GPUs during each iteration. Compared to the baseline, both Skrull and RR scheduling allocate more sequences locally at the cost of increased memory imbalance. While such imbalance is acceptable as long as it dose not exceed memory capacity, it raises the risk of OOMs. Therefore, as shown in Table 3, without the roll-back mechanism, both scheduling strategies result in OOMs, underscoring the importance of this safeguard. In contrast to RR scheduling, Skrull achieves a better computational balance (indicated by the speedup) while adhering to memory constraints, demonstrating the effectiveness of its trade-off design in Skrull.

## 6 Related Works

From the perspective of data engineering, those works [23, 3, 19, 10] involve meticulously gathering training datasets for long context fine-tuning. From the perspective of training system, LongAlign [3] adopts a sorted batching strategy to optimize system efficiency in long context fine-tuning phase. Chunkflow [22] organize the training data into fixed size chunks, enabling controllable peak memory consumption and reduced pipeline bubbles. Additionally, some works employ dynamic parallelism settings [8] to handle varying length sequences, which is similar to long-SFT. In contrast, Skrull adopts fixed parallelism settings and is orthogonal to those methods. Another type of works are parameter efficient finetuning (PEFT) [11, 5] and Skrull is also effective for this methods.

## 7 Conclusion

In this paper, we provide a new perspective of data scheduling to enhance the training efficiency in Long-SFT scenarios. The heterogeneous data distribution in Long-SFT poses dilemmas for existing training systems on configuring parallelism strategies and ensuring the load balance. To tackle those challenges, we propose Skrull, a dynamic data scheduler dedicated for Long-SFT. Through dynamic data scheduling, Skrull achieves efficient training on both long sequences and short sequences. Additionally, we formulate the scheduling process as a joint optimization and adopt a lightwight scheduling algorithm. Experimental results demonstrate that Skrull outperforms DeepSpeed by 3.76x on average (up to 7.54x) in real-world long-SFT. Furthermore, we believe that Skrull can serve as an effective solution in other scenarios especially when dealing with mixture of long and short training data, such as reinforcement learning from human feedback (RLHF).

## Acknowledgments

This work is supported by the following funding: Beijing Natural Science Foundation (4254087), National Science Foundation of China (62502501, 92270206, 62372435), China National Postdoctoral Program for Innovative Talents (BX20240383), Strategic Priority Research Program of Chinese Academy of Sciences (XDB0500102), and Alibaba Research Intern Program. The model training were performed on the robotic AI-Scientist platform of Chinese Academy of Science and Alibaba Cloud Platform for AI (PAI).

## References

- [1] nvidia/ChatQA2-long-SFT-data · datasets at hugging face, . URL https://huggingface. co/datasets/nvidia/ChatQA2-Long-SFT-data .
- [2] pleisto/wikipedia-cn-20230720-filtered · datasets at hugging face, . URL https:// huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered .
- [3] Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. LongAlign: A recipe for long context alignment of large language models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024 , pages 1376-1395, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.74. URL https://aclanthology.org/2024.findings-emnlp.74/ .
- [4] Suresh Bolusani, Mathieu Besançon, Ksenia Bestuzheva, Antonia Chmiela, João Dionísio, Tim Donkiewicz, Jasper van Doornmalen, Leon Eifler, Mohammed Ghannam, Ambros Gleixner, Christoph Graczyk, Katrin Halbig, Ivo Hedtke, Alexander Hoen, Christopher Hojny, Rolf van der Hulst, Dominik Kamp, Thorsten Koch, Kevin Kofler, Jurgen Lentz, Julian Manns, Gioni Mexi, Erik Mühmer, Marc E. Pfetsch, Franziska Schlösser, Felipe Serrano, Yuji Shinano, Mark Turner, Stefan Vigerske, Dieter Weninger, and Liding Xu. The scip optimization suite 9.0, 2024. URL https://arxiv.org/abs/2402.17702 .
- [5] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. LongLoRA: Efficient fine-tuning of long-context large language models. URL https:// arxiv.org/abs/2309.12307 . \_eprint: 2309.12307.
- [6] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR) , 2024.
- [7] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS) , 2022.
- [8] Hao Ge, Fangcheng Fu, Haoyang Li, Xuanyu Wang, Sheng Lin, Yujie Wang, Xiaonan Nie, Hailin Zhang, Xupeng Miao, and Bin Cui. Enabling parallelism hot switching for efficient training of large language models. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles , pages 178-194. ACM. ISBN 979-8-4007-1251-7. doi: 10.1145/3694715.3695969. URL https://dl.acm.org/doi/10.1145/3694715.3695969 .
- [9] Google Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. URL https://arxiv.org/abs/2403.05530 .
- [10] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab

AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt,

Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models. URL http://arxiv.org/abs/2407.21783 .

- [11] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR , 2022. URL http://dblp.uni-trier.de/db/conf/iclr/iclr2022.html# HuSWALWWC22 .
- [12] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Reza Yazdani Aminadabi, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. System optimizations for enabling training of extreme long sequence transformer models. In Proceedings of the 43rd ACM Symposium on Principles of Distributed Computing , pages 121-130. ACM. ISBN 979-84007-0668-4. doi: 10.1145/3662158.3662806. URL https://dl.acm.org/doi/10.1145/ 3662158.3662806 .
- [13] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. URL https://arxiv.org/abs/1412.6980 .
- [14] Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. In MLSys . URL https://proceedings.mlsys.org/paper\_files/paper/ 2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html .
- [15] Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, and Soumith Chintala. Pytorch distributed: experiences on accelerating data parallel training. Proc. VLDB Endow. , 13(12):3005-3018, August 2020. ISSN 2150-8097. doi: 10.14778/3415478.3415530. URL https://doi.org/ 10.14778/3415478.3415530 .
- [16] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ringattention with blockwise transformers for near-infinite context. In The Twelfth International Conference on Learning Representations , 2024. URL https://openreview.net/forum?id=WsRHpHH4s0 .
- [17] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations , 2019. URL https://openreview.net/forum? id=Bkg6RiCqY7 .

- [18] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O'Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774 .
- [19] Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. URL http://arxiv.org/abs/2412.15115 .
- [20] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: memory optimizations toward training trillion parameter models. In Proceedings of the International Conference

for High Performance Computing, Networking, Storage and Analysis , SC '20. IEEE Press, 2020. ISBN 9781728199986.

- [21] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. URL https://arxiv.org/abs/2307.09288 . \_eprint: 2307.09288.
- [22] Xiulong Yuan, Hongtao Xu, Wenting Shen, Ang Wang, Xiafei Qiu, Jie Zhang, Yuqiong Liu, Bowen Yu, Junyang Lin, Mingzhen Li, Weile Jia, Yong Li, and Wei Lin. Efficient long context fine-tuning with chunk flow. In Forty-second International Conference on Machine Learning , 2025. URL https://openreview.net/forum?id=rzn2OgflOK .
- [23] Liang Zhao, Tianwen Wei, Liang Zeng, Cheng Cheng, Liu Yang, Peng Cheng, Lijie Wang, Chenxia Li, Xuejie Wu, Bo Zhu, Yimeng Gan, Rui Hu, Shuicheng Yan, Han Fang, and Yahui Zhou. LongSkywork: A training recipe for efficiently extending context length in large language models. URL https://arxiv.org/abs/2406.00605 . \_eprint: 2406.00605.
- [24] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023. URL https://arxiv.org/abs/ 2304.11277 .
- [25] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric Xing, Joseph E. Gonzalez, Ion Stoica, and Hao Zhang. LMSYS-chat-1m: A large-scale real-world LLM conversation dataset. In The Twelfth International Conference on Learning Representations , 2024. URL https://openreview. net/forum?id=BOfDKxfwt0 .

## A Precision Validation

To evaluate the loss equivalence, we compare the loss carve between Skrull and standard training method when training Qwen2.5-0.5B on LMsysChat1M. The data scheduling in Skrull alter the accumulation order and we can observe slightly numerical differences due to the non-associativity of floating-point operations. However, Skrull do not alter any contents and orders in each global batch, the optimization trajectory remains equivalent. Therefore, as shown in Figure 5, Skrull does not influence the convergence.

<!-- image -->

Steps

Figure 5: Loss comparison between Skrull and standard training method.

## B Heuristic Scheduling Algorithm

## B.1 Function Definition of Heuristic Algorithm

```
Require: SeqNum K , SeqLens S [ K ] , Buckets C , CP degree N , Loads L [ N ] , RemainBucket RB [ N ] , DACP scheduling result ret 1: function UPDATELOCAL( idx , rank ) 2: RB [ rank ] ← RB [ rank ] -S [ idx ] ▷ Update remaining bucket capacity 3: L [ rank ] ← L [ rank ] + FLOPs ( S [ idx ]) ▷ Update current load 4: end function 5: function UPDATEALL( idx ) 6: for i = 0 to N -1 do 7: RB [ i ] ← RB [ i ] -S [ idx ] /N ▷ Distribute across all buckets 8: L [ i ] ← L [ i ] + FLOPs ( S [ idx ] , N ) ▷ Update all loads 9: end for 10: end function 11: function ROLLBACK( rank , RB , L ) 12: for i = 0 to K -1 do 13: if ret [ i ] == rank then 14: ret [ i ] ←-1 ▷ Distribute the sequence 15: RB [ rank ] ← RB [ rank ] -S [ i ] + S [ i ] /N 16: if L is not None then 17: L [ rank ] ← L [ rank ] -FLOPs ( S [ i ]) + FLOPs ( S [ i ] , N ) 18: end if 19: return True ▷ Success Roll-back 20: end if 21: end for 22: return False ▷ Roll-back Failed 23: end function
```

Algorithm 3 Function Definations in scheduling algorithm for DACP

## Algorithm 4 Round-Robin Scheduling Algorithm

```
1: Input: SeqNum N , SeqLens S , Buckets C , WorldSize ws 2: Output: PartitionIds P 3: for i = 1 to m -1 do 4: t ← FindMaxBucketsIds() 5: if C [ t ] ≥ S [ i ] then 6: P [ i ] ← t ◁ fit the max bucket 7: else 8: j ← FindMinBucketsIds() 9: if C [ j ] ≥ S [ i ] /ws then 10: P [ i ] ←-1 ◁ partition the sequence 11: else 12: Assert RollBack ( j, C ) ◁ with roll-back 13: i ← i -1 14: continue 15: end if 16: end if 17: end for
```

## B.2 Round-robin Scheduling Algorithm

## C Performance Modeling

## C.1 Memory Estimation

Due to limited pages, we discuss the memory estimation methodology of Skrull in this section. The key point of this section is the determination of BucketSize C , which maps memory capacity to sequence token length.

We first analyze the memory consumption during LLMs training. The memory consumption can be roughly categorized into two components: the static memory and the dynamic memory. The static memory, which typically includes model parameters and optimizer states, remains roughly constant throughout the training process given specific model configurations and parallelism strategies. In contrast, the dynamic memory or activation memory, varies with the input workload. In transformer architectures, activation memory is proportional to the sequence length. For instance, the Linear module, LayerNorm and Attention module (using FlashAttention [7, 6]) exhibit a linear relationship with sequence length. Therefore, we can estimate activation memory for a given sequence length S using the following equation:

<!-- formula-not-decoded -->

Here, the coefficient α and constant β is determined at offline profiling. Notably, some memory reduction strategies, such as gradient checkpoints, only affect the α and β . We can still apply offline profiling method to estimate activation memory. In our implementation, we found that β is usually negligible. Additionally, we employ sequence packing to eliminate padding and enhance performance, allowing us to directly use the total sequence length for memory estimation. Consequently, through offline profiling, we can deduce the BucketSize C under various settings.

## C.2 Computation Estimation

In this section, we describe the methodology used to estimate the computational cost T comp .

Accurately modeling the computational cost as a function of sequence length S is non-trivial. Simply assuming a linear or quadratic relationship with sequence length is insufficient because the computational FLOPs of TransformerLayer are dominated by the Linear and Attention modules, exhibiting a hybrid of linear and quadratic dependencies on S . The relative contributions of these components vary depending on the specific model configuration. Therefore, we formulate a function of FLOPs to provide roughly computational cost estimation given a specific model configuration and sequence length S .

Given the model configuration of hidden dimension h , key/value hidden dimension h kv and training batchsize b (usually be 1 when employ sequence packing), the FLOPs is estimated as the Equation 13.

<!-- formula-not-decoded -->

For each sequence, the T comp can be estimated as:

<!-- formula-not-decoded -->

where all the α and β is determined when offline profiling.

Furthermore, as shown in Figure 6, we plot the relationship between FLOPs and sequence length for Qwen-2.5-0.5B and Qwen-2.5-7B. The results highlight the distinct characteristics of long and short sequences. For short sequences, both computational workload and activation memory consumption scale roughly linearly with sequence length. However, for the long sequences, the computational workload grows rapidly due to the dominance of the quadratic term, while memory consumption still remains linear, leading to the problem of trade-off between balancing computation and memory, which is discussed in detail in Section 4.3 where we present insights into our heuristic algorithm design.

Additionally, the transition point at which the quadratic term dominates varies depending on the model configuration. As demonstrated in Figure 6, Qwen-2.5-7B, which has a larger hidden dimension h , exhibits a more rapid increase in FLOPs compared to Qwen-2.5-0.5B. Although Qwen-2.5-0.5B has slower FLOPs increase, we take it as example to further discuss the distinct characteristics between long and short sequences. In Qwen-2.5-0.5B, the quadratic term begins to dominate only when the sequence length S exceeds approximately 4K, exhibits roughly linear relationship in short sequences. However, when S = 32 K , the total computational workload is 30 times greater than when S = 4 K , while the memory consumption increases only 4-fold. These estimations further elucidate the distinct characteristics of long and short sequences.

<!-- image -->

Sequence Length

Figure 6: FLOPs VS Sequence Length on Qwen-2.5 0.5B and 7B

## C.3 Communication Estimation

For the T comm , we can simply profile in offline ways. Concretely, when the communication volume is smaller then a threshold, the fixed overhead of communication dominates the latency. However, with size increased, the fixed overhead become negligible and the latency is approximately proportional to communication volumes. We can deduce the thresholds, fixed overhead and the estimation function through a simple profiling. As shown in Table 5, we plot the communication performance profiling results. Therefore, we can fit the Equation 16 according to communication volume V in different hardware environments. Then, we can derive the communication volume according to sequence length S under different model configurations as shown in Equation 15, where hidden kv and b means hidden dimension of Key/Value and batch size.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Table 5: Collective Communication Latency Profiling.

|   Size (MB)/Latency(us) |   All_gather |   All_to_All |   Reduce_scatter |   All_reduce |
|-------------------------|--------------|--------------|------------------|--------------|
|                       2 |        53.29 |        80.62 |            59.48 |        84.65 |
|                       4 |        72.52 |        78.63 |            79.26 |       113.3  |
|                       8 |        97.86 |       110.9  |           104.7  |       168.4  |
|                      16 |       199.3  |       163.2  |           177.4  |       312.2  |
|                      32 |       286.2  |       277.5  |           269.5  |       479.2  |
|                      64 |       488.6  |       502.4  |           458.8  |       859.7  |
|                     128 |       910.6  |       939.2  |           864.3  |      1642.9  |
|                     256 |      1758.4  |      1803.9  |          1663.9  |      3197.9  |
|                     512 |      3416.4  |      3411.2  |          3239.5  |      6181.2  |
|                    1024 |      6467.9  |      6629.6  |          6294.3  |     12126    |

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We list our contributions in abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the BatchSize will limit the optimization in our evaluation section.

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

Justification: We list the details about how we analysis the memory and computation in Appendix in this pdf.

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

Justification: We provide pseudocode and evaluation settings in our paper

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

## Answer: [NA]

Justification: We use public data in our experiments. Due to some approval process, we will make our code public as soon as possible.

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

Justification: We list in our evaluation sections.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Our experimental results are averaged by multiple runnings.

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

Justification: We list in evaluation testbed.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Yes.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is about system optimization.

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

Justification: Our work is about system optimization.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Yes, we cite the related paper.

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

Justification: We do not release new assets

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The paper do not involve usage of LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.