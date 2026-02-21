## Turbocharging Gaussian Process Inference with Approximate Sketch-and-Project

## Pratik Rathore

Stanford University pratikr@stanford.edu

Shaghayegh Fazliani Stanford University fazliani@stanford.edu

## Zachary Frangella

Stanford University zfran@stanford.edu

Michał Derezi´ nski

University of Michigan derezin@umich.edu

## Abstract

Gaussian processes (GPs) play an essential role in biostatistics, scientific machine learning, and Bayesian optimization for their ability to provide probabilistic predictions and model uncertainty. However, GP inference struggles to scale to large datasets (which are common in modern applications), since it requires the solution of a linear system whose size scales quadratically with the number of samples in the dataset. We propose an approximate, distributed, accelerated sketch-and-project algorithm ( ADASAP ) for solving these linear systems, which improves scalability. We use the theory of determinantal point processes to show that the posterior mean induced by sketch-and-project rapidly converges to the true posterior mean. In particular, this yields the first efficient, condition number-free algorithm for estimating the posterior mean along the top spectral basis functions, showing that our approach is principled for GP inference. ADASAP outperforms state-of-the-art solvers based on conjugate gradient and coordinate descent across several benchmark datasets and a large-scale Bayesian optimization task. Moreover, ADASAP scales to a dataset with &gt; 3 · 10 8 samples, a feat which has not been accomplished in the literature.

## 1 Introduction

Gaussian processes (GPs) are a mainstay of modern machine learning and scientific computing, due to their ability to provide probabilistic predictions and handle uncertainty quantification. Indeed, GPs arise in applications spanning Bayesian optimization [Hernández-Lobato et al., 2017], genetics [McDowell et al., 2018], health care analytics [Cheng et al., 2020], materials science [Frazier and Wang, 2016], and partial differential equations [Chen et al., 2025].

GP inference for a dataset with n samples requires the solution of a dense n × n linear system, which is challenging to solve at large scale. Direct methods like Cholesky decomposition require O ( n 3 ) computation, limiting them to n ∼ 10 4 . Consequently, two main approaches have arisen for large-scale inference: (i) exact inference based on iterative methods for solving linear systems [Wang et al., 2019, Lin et al., 2023, 2024] and (ii) approximate inference based on inducing points and variational methods [Titsias, 2009, Hensman et al., 2013]. The state-of-the-art approaches for largescale inference are preconditioned conjugate gradient (PCG) [Wang et al., 2019] and stochastic dual descent (SDD) [Lin et al., 2024]. PCG offers strong convergence guarantees and good performance on ill-conditioned problems, but is slow when n ∼ 10 6 . In contrast, SDD scales to n ≫ 10 6 , but lacks strong theoretical guarantees, can slow down in the face of ill-conditioning, and introduces a stepsize parameter which can be challenging to tune. Inducing points and variational methods [Titsias, 2009, Hensman et al., 2013] scale to large datasets, but are often outperformed by exact

## Sachin Garg

University of Michigan sachg@umich.edu

Madeleine Udell Stanford University udell@stanford.edu

Figure 1: ADASAP attains lower root mean square error (RMSE) and mean negative log likelihood (NLL) than start-of-the-art methods SDD [Lin et al., 2024] and PCG on the houseelec dataset. SDD-1 and SDD-10 correspond to two particular stepsize selections for the SDD method. The solid lines indicate the mean performance of each method, while the shaded regions indicate the range between worst and best performance of each method over five random splits of the data.

<!-- image -->

inference [Wang et al., 2019, Lin et al., 2023, 2024]. Altogether, the current state of algorithms for large-scale GP inference is unsatisfactory, as practitioners have to trade-off between quality of inference, robustness to ill-conditioning, ease of setting hyperparameters, and scalability.

To address this gap in the literature, we introduce the A pproximate D istributed A ccelerated S ketcha ndP roject ( ADASAP ) algorithm. ADASAP is rooted in the sketch-and-project framework of Gower and Richtárik [2015], and obtains the robustness to ill-conditioning of PCG and the scalability of SDD, while having reliable, default hyperparameters that work well in practice. For example, ADASAP dramatically outperforms tuned SDD and PCG on a dataset with n &gt; 10 6 samples (Fig. 1).

Our contributions are as follows:

1. We develop ADASAP for large-scale GP inference. ADASAP uses approximate preconditioning and acceleration to address ill-conditioning, and uses distributed computing to improve the speed of bottleneck operations. ADASAP also comes with effective default hyperparameters that work out of the box.
2. Using the theory of determinantal point processes, we show that sketch-and-project-style methods converge faster than stochastic first-order methods like SDD in the presence of illconditioning. In particular, we give a first-of-its-kind condition number-free time complexity bound for estimating the GP posterior mean to moderate accuracy, explaining the excellent test error performance of ADASAP .
3. We empirically verify that ADASAP , with its default hyperparameters, outperforms state-ofthe-art competitors on benchmark large-scale GP inference tasks, and is capable of scaling to a dataset with n &gt; 3 · 10 8 samples.
4. We show ADASAP yields the best performance on a large-scale Bayesian optimization task from Lin et al. [2023].

## 2 GP Regression and Inference

Let X be a set. A random function f : X → R is called a Gaussian process if for any finite subset { x 1 , x 2 , . . . , x n } ⊂ X of points (where n is any positive integer), the random vector ( f ( x 1 ) , f ( x 2 ) , . . . , f ( x n )) follows a multivariate Gaussian distribution [Rasmussen and Williams, 2005]. A Gaussian process is typically denoted as f ∼ GP ( m,k ) , where m : X → R is the mean function and k : X ×X → R is the covariance function (or kernel). Throughout this paper, the kernel function k follows broadcasting conventions, where operations between individual points and sets of points produce vectors or matrices of kernel evaluations as appropriate.

Suppose we have a Gaussian process prior f ∼ GP ( m,k ) . Given observations ( X,y ) ∈ R n × d × R n and likelihood variance λ &gt; 0 , we would like to perform inference using the GP posterior, i.e., (i) sampling from the posterior and (ii) computing the posterior mean.

For conciseness, define K := k ( X,X ) ∈ R n × n . Then a sample from the GP posterior is a random function f n ∼ GP ( m n , k n ) [Rasmussen and Williams, 2005], where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The standard approach for drawing posterior samples at locations X ∗ is to perform a linear transformation of standard Gaussian random variables ζ ∼ N (0 , I ) [Wilson et al., 2020, 2021]:

<!-- formula-not-decoded -->

Unfortunately, the approach in (3) does not scale when | X ∗ | is large, since the matrix square root k n ( X ∗ , X ∗ ) 1 / 2 requires O ( | X ∗ | 3 ) computation.

## 2.1 Pathwise conditioning

To address the scaling challenges in (3), Wilson et al. [2020, 2021] develop pathwise conditioning. Pathwise conditioning rewrites (3) as

<!-- formula-not-decoded -->

Equation (4) enables scalable sampling from the posterior as f can be approximated using random features [Rahimi and Recht, 2007, Rudi and Rosasco, 2017] for k , which costs O ( q | X ∗ | ) , where q is the number of random features. This yields a significant reduction in cost from the standard posterior sampling scheme in (3): by leveraging pathwise conditioning, posterior sampling can be performed by generating f via random features [Wilson et al., 2020, 2021] and solving ( K + λI ) -1 ( y -f ( X n ) -ζ ) using an iterative method like PCG or SDD. Our experiments (Section 5) use pathwise conditioning to sample from the posterior.

## 2.2 Drawbacks of state-of-the-art exact inference methods

The current state-of-the-art for large-scale posterior sampling is SDD [Lin et al., 2024], a stochastic first-order algorithm combining coordinate descent [Richtárik and Takáˇ c, 2014, Qu and Richtárik, 2016] with momentum and geometric averaging. Stochastic methods like SDD can better handle single (or lower) precision than PCG [Wu et al., 2024, Rathore et al., 2025], which is essential to maximizing GPU performance and reducing storage costs [Abdelfattah et al., 2021]. While SDD overcomes the scalability challenges associated with PCG, it faces challenges due to the illconditioning of K + λI , which determines its worst-case convergence rate [Nemirovski and Yudin, 1983]. Therefore, reliable large-scale inference requires a method that has cheaper per-iteration costs than PCG and can handle the ill-conditioning of K + λI . However, to the best of our knowledge, no existing method effectively tackles both issues: SGD and SDD offer speed and scalability but suffer from ill-conditioning, while PCG offers robustness to ill-conditioning but struggles to scale beyond n &gt; 10 6 . In Section 3 we propose SAP , which is robust to ill-conditioning; in Section 4, we develop ADASAP , an extension of SAP that scales to large datasets.

## 3 Sketch-and-project for GP Inference

Section 2 shows the main challenge of large-scale GP inference is the need to solve a large linear system with the ill-conditioned matrix K + λI , which cannot be done satisfactorily with PCG or SDD. We introduce the SAP algorithm, which effectively addresses the conditioning challenges of GP inference. However, SAP alone does not address the scalability challenges of GP inference, motivating the development of ADASAP in Section 4.

## 3.1 Sketch-and-project

We formally present SAP in Algorithm 1. SAP resembles randomized block coordinate descent, except that it employs subspace preconditioning . That is, it preconditions the subspace gradient

## Algorithm 1 SAP for K λ W = Y

Require: blocksize b , distribution of row indices D over [ n ] b , number of iterations T max , initialization W 0 ∈ R n × n rhs , averaging boolean tail\_average

```
for t = 0 , 1 , . . . , T max -1 do Sample row indices B of size b from [ n ] according to D D t ← I T B ( K BB + λI ) -1 ( K B n W t + λI B W t -Y B ) ▷ Search direction; costs O ( bnn rhs + b 3 ) W t +1 ← W t -D t ▷ Update parameters; costs O ( bn rhs ) if tail_average then ¯ W t +1 ← 2 / ( t +1) ∑ t j =( t +1) / 2 W j end if end for return ¯ W T if tail_average else W T ▷ Approximate solution to K λ W = Y
```

K B n W t + λI B W t -Y B by the inverse of the subspace Hessian K BB + λI . Thus, unlike SDD, SAP includes second-order information in its search direction. In addition, we have also augmented vanilla SAP from Gower and Richtárik [2015] with tail averaging [Jain et al., 2018, Epperly et al., 2024a], which helps address the inherent noise in the iterates, and plays a crucial role in our analysis.

SAP (assuming the number of right-hand sides n rhs = 1 for simplicity) only costs O ( bn + b 3 ) periteration, which is significantly smaller than the O ( n 2 ) iteration cost of PCG, while being comparable with SDD, which costs O ( bn ) per-iteration. Furthermore, PCG typically requires memory linear in n to store the preconditioner, while SAP only incurs and additional O ( b 2 ) storage cost. Thus, SAP strikes a balance between PCG and methods like SDD. It incorporates second-order information, like PCG, but does so at a modest increase in cost relative to SDD and SGD. By leveraging the theory of determinantal point processes [Kulesza and Taskar, 2012], we show that tail-averaged SAP improves dependence upon the condition number in the early phase of the convergence, which establishes it as a potential solution to the conditioning challenges discussed in Section 2, particularly in the presence noisy data and generalization error.

## 3.2 Fast convergence along topℓ subspace for GP inference

Prior works [Dicker et al., 2017, Lin et al., 2020, 2023] have shown that the components of the solution most relevant for learning lie along the dominant eigenvectors of K , which correspond to the dominant spectral basis functions of the reproducing kernel Hilbert space (RKHS) H induced by kernel k with respect to X . Lin et al. [2023] shows that SGD enjoys an improved early convergence along these directions, but attains a slow sublinear rate asymptotically. Motivated by these observations, we study the convergence rate of tail-averaged SAP along the dominant spectral basis functions, showing that it obtains a two-phase convergence rate: an initial condition number-free sublinear rate, followed by an asymptotic linear rate where the condition number dependence is mitigated by the blocksize b . The proofs of the results in this section appear in Appendix B.3.

Following Lin et al. [2023], we define the spectral basis functions of H as

<!-- formula-not-decoded -->

The spectral basis functions u (1) , u (2) , ... are defined in such a way that each of them takes maximal values at the observation points while being orthogonal to the previous ones. Hence, to accurately estimate the posterior mean in the data-dense regions, it is most important to minimize the error along the dominant basis functions.

We focus on posterior mean estimation, which corresponds to solving the linear system K λ W = y using tail-averaged SAP . Recall that, given a sequence of weight vectors W 1 , ..., W t produced by sketch-and-project (which are vectors, not matrices, as n rhs = 1 ), tail averaging obtains a weight estimate by averaging the second half of this sequence. We use these weights to define an estimate of the GP posterior mean m n , which is given by ˆ m t = k ( · , X ) ¯ W t .

Theorem 3.2 is our main theoretical result. It provides a convergence guarantee for the SAP posterior mean estimate ˆ m t along the topℓ spectral basis functions, where ℓ is a parameter that can be chosen

freely. Theorem 3.2 shows that while the estimate converges to the true posterior mean over the entire RKHS H , its initial convergence rate along the dominant basis functions is even faster. Our theoretical results rely on the smoothed condition number , which we define below.

Definition 3.1 (Smoothed condition number) . Let A ∈ R n × n be a positive-semidefinite matrix with eigenvalues λ 1 ≥ . . . ≥ λ n . We define the smoothed condition number ϕ ( b, p ) of A as

<!-- formula-not-decoded -->

The smoothed condition number controls both the SGD-style sublinear convergence rate and the PCG-style linear convergence rate in Theorem 3.2. Note that, for notational convenience, we use 2 b to denote the SAP blocksize in the below statement.

Theorem 3.2. Suppose we have a kernel matrix K ∈ R n × n , observations y ∈ R n from a GP prior with likelihood variance λ ≥ 0 , and let ϕ ( · , · ) denote the smoothed condition number of K + λI . Given any ℓ ∈ [ n ] , let proj ℓ denote orthogonal projection onto the span of u (1) , ..., u ( ℓ ) . For any 1 ≤ b ≤ n/ 2 , in ˜ O ( nb 2 ) time we can construct a distribution over row index subsets B of size 2 b such that SAP (Algorithm 1) initialized at zero after t iterations satisfies

<!-- formula-not-decoded -->

This result provides a composite convergence rate for SAP : the first rate, 8 ϕ ( b,ℓ ) t , is a sublinear rate, similar to SGD [Lin et al., 2023], while the second rate, ( 1 -1 2 ϕ ( b,n ) ) t/ 2 , is a linear convergence rate. The linear rate is independent of the subspace norm, and thus, in particular, also applies for ℓ = n , i.e., when comparing posterior means within the entire space. However, for ℓ ≪ n , the sublinear rate wins out in the early phase of the convergence, since ϕ ( b, ℓ ) ≪ ϕ ( b, n ) .

For ill-conditioned kernels with small prior variance λ , the linear rate 1 -1 / (2 ϕ ( b, n )) is close to 1 (as b increases, the preconditioning effect becomes stronger, which improves the rate). On the other hand, the constant in the sublinear rate ϕ ( b, ℓ ) is much smaller, since it takes advantage of the topℓ subspace norm, which makes it effectively independent of the conditioning of K .

If the SAP blocksize is proportional to ℓ and K + λI exhibits any polynomial spectral decay, then ϕ ( b, ℓ ) = O (1) , making the sublinear rate condition number-free. Most popular kernels exhibit polynomial spectral decay [Ma and Belkin, 2017, Kanagawa et al., 2018]. Furthermore, polynomial spectral decay is a standard assumption in the generalization analysis of kernel methods in the fixed and random design settings [Caponnetto and DeVito, 2007, Bach, 2013, Rudi et al., 2015]. To the best of our knowledge, our result is the first of its kind, as previous comparable guarantees for stochastic methods [Lin et al., 2023, Proposition 1] still depend on the condition number. We note that our condition number-free rate is attained only when seeking moderately accurate estimates of the posterior mean (which is often the case due to the presence of noise in the data). When seeking highly accurate estimates, the asymptotic (condition number dependent) linear rate will eventually take over. We illustrate these claims with the following corollary:

Corollary 3.3. Suppose that the matrix K exhibits polynomial spectral decay, i.e., λ i ( K ) = Θ( i -β ) for some β &gt; 1 . Then for any ℓ ∈ { 1 , . . . , n } and ϵ ∈ (0 , 1) , choosing block size b = 4 ℓ we can find ˆ m that with probability at least 0 . 99 satisfies ∥ proj ℓ ( ˆ m ) -proj ℓ ( m n ) ∥ 2 H ≤ ϵ · ∥ proj ℓ ( m n ) ∥ 2 H in

<!-- formula-not-decoded -->

Remark 3.4 . The ˜ O (( n 2 + nℓ 2 ) /ϵ ) runtime (phase one, attained for moderate accuracy ϵ ) is entirely independent of the condition number of K and the likelihood variance λ . In Appendix B, we show that when λ is sufficiently small (the highly ill-conditioned case), then the phase one complexity of sketch-and-project can be further improved by increasing the block size b to attain ˜ O ( nb 2 +( n 2 + nb 2 )( ℓ/b ) β -1 /ϵ ) runtime. In particular, when ℓ ≪ b ≪ √ n , we obtain meaningful convergence in o ( n 2 ) time, i.e., before reading all of the data. Thus, the topℓ subspace convergence of SAP can actually benefit from ill-conditioned kernels as the leading spectral basis functions become more dominant.

Although our theoretical results require tail averaging, we do not believe it is needed for good practical performance of sketch-and-project algorithms. Indeed, we run sketch-and-project with and without tail averaging in Appendix C.6, and we find that (i) tail averaging does not improve subspace convergence by a substantial amount and (ii) tail averaging performs worse than not using tail averaging at larger blocksizes.

## 4 ADASAP : Approximate, Distributed, Accelerated SAP

We introduce ADASAP (Algorithm 2), a scalable extension of SAP for GP inference. We first introduce the modifications made to SAP for scalability (Section 4.1), before presenting ADASAP in full (Section 4.2). For a detailed discussion of how ADASAP relates to prior work, please see Appendix A.

## 4.1 The key ingredients: approximation, distribution, and acceleration

We begin by discussing the essential elements of the ADASAP algorithm: (i) approximate subspace preconditioning, (ii) distribution, (iii) acceleration, and how they enhance performance.

## 4.1.1 Approximate subspace preconditioning

SAP enjoys significant improvements over PCG, as it only requires O ( bn + b 3 ) computation periteration. Unfortunately, SAP is limited in how large a blocksize b it may use, as factoring K BB + λI to perform subspace preconditioning costs O ( b 3 ) . This is problematic, as Theorem 3.2 shows the convergence rate of SAP improves as b increases.

To address this challenge, ADASAP draws inspiration from previous works [Erdogdu and Montanari, 2015, Frangella et al., 2024b, Rathore et al., 2025], which replace exact linear system solves in iterative algorithms with inexact solves based on low-rank approximations. ADASAP replaces K BB in SAP with a rankr randomized Nyström approximation ˆ K BB [Williams and Seeger, 2000, Gittens and Mahoney, 2016, Tropp et al., 2017]. This strategy is natural, as kernels exhibit approximate low-rank structure [Bach, 2013, Rudi et al., 2015, Belkin, 2018]. Indeed, Rathore et al. [2025] develops an approximate SAP solver for kernel ridge regression with one right-hand side, based on approximating K BB by ˆ K BB , and observes strong empirical performance.

ADASAP computes ˆ K BB following the numerically stable procedure from Tropp et al. [2017], which is presented in Appendix C. The key benefit is that ( ˆ K BB + ρI ) -1 can be applied to vectors in O ( br ) time (Appendix C), where ρ &gt; 0 is a damping parameter to ensure invertibility. This reduces the cost compared to the O ( b 3 ) exact SAP update and allows ADASAP to use larger blocksizes: on the taxi dataset ( n = 3 . 31 · 10 8 ), ADASAP uses blocksize b = 1 . 65 · 10 5 .

## 4.1.2 Distributed matrix-matrix products

SAP with approximate subspace preconditioning allows large blocksize b , but two bottlenecks remain: computing K B n W t and constructing the sketch K BB Ω . The former costs O ( bn ) , while the latter costs O ( b 2 r ) . As matrix-matrix multiplication is embarrassingly parallel, distributed multi-GPU acceleration can address these bottlenecks. By partitioning K B n across N workers GPUs, ColDistMatMat significantly reduces the time to compute the product K B n Ω . Similarly, RowDistMatMat for K BB Ω reduces the time to compute the sketch. On our largest dataset, taxi, using 4 GPUs achieves a 3 . 4 × speedup (Fig. 4), which corresponds to near-perfect parallelism.

## 4.1.3 Nesterov acceleration

Prior work has shown Nesterov acceleration [Nesterov, 1983] improves the convergence rate of SAP and approximate SAP [Tu et al., 2017, Gower et al., 2018, Derezi´ nski et al., 2024, Rathore et al., 2025]. Hence we use Nesterov acceleration in ADASAP to improve convergence.

## 4.2 ADASAP algorithm

The pseudocode for ADASAP is presented in Algorithm 2. ADASAP uses tail averaging, approximate subspace preconditioning, distributed matrix-matrix products, and Nesterov acceleration. Assuming

## Algorithm 2 ADASAP for K λ W = Y

Require: distribution of row indices D over [ n ] b , distribution over Nyström sketching matrices D Nyström over R b × r , number of iterations T max , initialization W 0 ∈ R n × n rhs , workers {W 1 , . . . , W N workers } , averaging boolean tail\_average, acceleration parameters µ , ν

## # Initialize acceleration parameters

```
β ← 1 -√ µ/ν, γ ← 1 / √ µν, α ← 1 / (1 + γν ) V 0 ← W 0 , Z 0 ← W 0 for t = 0 , 1 , . . . , T max -1 do # Step I: Compute matrix-matrix product K B n W t ▷ Costs O ( bnn rhs / N workers ) Sample row indices B of size b from [ n ] according to D K B n W t ← ColDistMatMat ( W t , B , {W 1 , . . . , W N workers } ) ▷ Algorithm 3 # Step II: Compute Nyström sketch K BB Ω ▷ Costs O ( rb 2 / N workers ) Sample Ω ∈ R b × n from D Nyström K BB Ω ← RowDistMatMat (Ω , B , {W 1 , . . . , W N workers } ) ▷ Algorithm 4 # Step III: Compute Nyström approximation and get stepsize ▷ Costs O ( b 2 + br 2 ) U, S ← RandNysAppx ( K BB Ω , Ω , r ) ▷ Algorithm 5 P B ← USU T +( S rr + λ ) I ▷ Get preconditioner. Never explicitly formed! η B ← GetStepsize ( P B , K BB + λI ) ▷ Algorithm 6 # Step IV: Compute updates using acceleration ▷ Costs O ( brn rhs + nn rhs ) D t ← I T B P -1 B ( K B n W t + λI B W t -Y B ) W t +1 , V t +1 , Z t +1 ← NestAcc ( W t , V t , Z t , D t , η B , β, γ, α ) ▷ Algorithm 7 if tail_average then ¯ W t +1 ← 2 / ( t +1) ∑ t j =( t +1) / 2 W j end if end for return ¯ W T if tail_average else W T ▷ Approximate solution to K λ W = Y
```

perfect parallelism, ADASAP has per iteration runtime of O ( nn rhs b/ N workers + b 2 r/ N workers ) , a significant improvment over the O ( nn rhs b + b 3 ) iteration time of SAP . The per-iteration runtime of ADASAP is comparable to distributed SDD-in other words, ADASAP effectively preconditions the problem, reducing the total iterations required, without significantly slowing down each iteration. Moreover, unlike SDD, ADASAP automatically sets the stepsize at each iteration, removing the need for expensive tuning. By default, we set the acceleration parameters µ and ν to λ and n/b , respectively (as done in Rathore et al. [2025]), and find they yield excellent performance in Section 5.

## 4.2.1 Theory vs. Practice

Our theory in Section 3 does not cover the approximate preconditioning, acceleration, or uniform sampling that is used in ADASAP . Despite the theory-practice gap between SAP (Algorithm 1) and ADASAP (Algorithm 2), we believe the theory developed for SAP can be extended to ADASAP . This belief is rooted in Derezi´ nski and Yang [2024], Derezi´ nski et al. [2024], Rathore et al. [2025], which show that SAP methods still converge when using approximate preconditioning, acceleration, and uniform sampling. We leave this extension as a direction for future research.

For simplicity, our experiments run ADASAP without tail averaging. Tail averaging is needed to establish Theorem 3.2, but we do not expect it to yield significant practical improvements (Appendix C.6). This is in line with the SGD literature, where averaging leads to better theoretical convergence rates, but the last iterate delivers similar performance in practice [Shamir and Zhang, 2013, Johnson and Zhang, 2013].

Table 1: RMSE and mean negative log-likelihood (NLL) obtained by ADASAP and competitors on the test set. The results are averaged over five 90%-10% train-test splits of each dataset. We bold a result if it gets to within 0 . 01 of the best found RMSE or mean NLL.

|           | Dataset n d k                            | yolanda 3 . 60 · 10 5 100 RBF          | song 4 . 64 · 10 5 90 Matérn-3/2       | benzene 5 . 65 · 10 5 66 Matérn-5/2    | malonaldehyde 8 . 94 · 10 5 36 Matérn-5/2   | acsincome 1 . 50 · 10 6 9 RBF          | houseelec 1 . 84 · 10 6 9 Matérn-3/2   |
|-----------|------------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|---------------------------------------------|----------------------------------------|----------------------------------------|
| Test RMSE | ADASAP ADASAP-I SDD-1 SDD-10 SDD-100 PCG | 0.795 0.808 0.833 0.801 Diverged 0.795 | 0.752 0.782 0.808 0.767 Diverged 0.752 | 0.012 0.168 0.265 0.112 Diverged 0.141 | 0.015 0.231 0.270 Diverged Diverged 0.273   | 0.789 0.795 0.801 0.792 Diverged 0.875 | 0.027 0.066 0.268 0.119 Diverged 1.278 |
|           |                                          | 1.179                                  |                                        | -2.673 -0.217                          | -2.259                                      | 1.229                                  |                                        |
| NLL       | ADASAP                                   |                                        | 1.121                                  | 0.531 -0.762                           | 0.466 0.903                                 | 1.235 1.242                            | -2.346                                 |
| Mean      | ADASAP-I SDD-1                           | 1.196 1.225 1.187                      | 1.170 1.203 1.149                      | Diverged                               |                                             | 1.232                                  | -2.185 -0.281                          |
| Test      | SDD-10                                   |                                        |                                        | -0.124                                 | Diverged                                    |                                        | -1.804                                 |
|           | SDD-100                                  | Diverged                               | Diverged                               |                                        | Diverged 0.925                              | Diverged                               | Diverged                               |
|           | PCG                                      | 1.179                                  | 1.121                                  |                                        |                                             | 1.316                                  | 2.674                                  |

## 5 Experiments

We present three sets of experiments showing ADASAP outperforms state-of-the-art methods for GP inference: (i) GP inference on large benchmark datasets (Section 5.1), (ii) GP inference on huge-scale transportation data analysis with n &gt; 3 · 10 8 samples (Section 5.2), and (iii) the Bayesian optimization task from Lin et al. [2023] (Section 5.3). We evaluate ADASAP against the following competitors:

- ADASAP-I : A variant of ADASAP where the subspace preconditioner P B is set to the identity matrix. ADASAP-I is the same as accelerated randomized block coordinate descent.
- SDD: The coordinate descent method introduced by Lin et al. [2024]. We tune SDD using three different stepsizes, and denote these variants by SDD-1, SDD-10, and SDD-100.
- PCG: A combination of block CG [O'Leary, 1980] with Nyström preconditioning [Frangella et al., 2023].

Our experiments are run in single precision on 48 GB NVIDIA RTX A6000 GPUs using Python 3.10, PyTorch 2.6.0 [Paszke et al., 2019], and CUDA 12.5. We use 2, 3, and 1 GPU(s) per experiment in Sections 5.1 to 5.3, respectively. Code for reproducing our experiments is available at https://github.com/pratikrathore8/scalable\_gp\_inference.

Additional details are in Appendix D.

## 5.1 GP inference on large-scale datasets

We benchmark on six large-scale regression datasets from the UCI repository, OpenML, and sGDML [Chmiela et al., 2017]. The results are reported in Table 1. ADASAP achieves the lowest RMSE and mean negative log-likelihood (NLL) on each dataset; the NLL is computed using 64 posterior samples (via pathwise conditioning). SDD-10 is competitive with ADASAP on yolanda and acsincome, but performs much worse on the other datasets. SDD is also sensitive to the stepsize: SDD-1 attains a larger RMSE and NLL than ADASAP on all datasets, while SDD-100 diverges on all datasets. PCG obtains the same RMSE and NLL as ADASAP on yolanda and song, but its performance degrades for larger datasets. Additionally, Fig. 2 shows ADASAP achieves the lowest RMSE and NLL throughout the optimization process, demonstrating its efficiency with respect to wall-clock time.

## 5.2 Showcase: Transportation data analysis with &gt; 3 · 10 8 samples

To demonstrate the power of ADASAP on huge-scale problems, we perform GP inference on a subset of the taxi dataset (https://github.com/toddwschneider/nyc-taxi-data) with n = 3 . 31 · 10 8 samples

Figure 2: Performance of ADASAP and competitors on RMSE and mean NLL, as a function of time, for benzene, malonaldehyde, and houseelec. The solid curve indicates mean performance over random splits of the data; the shaded regions indicate the range between the worst and best performance over random splits of the data. ADASAP outperforms the competition.

<!-- image -->

Figure 3: Comparison between ADASAP and competitors on transportation data analysis. ADASAP attains the lowest RMSE and it obtains a 1 . 8 × speed up over the second-best method, SDD-10. SDD-100 diverges and PCG runs out of memory, so they do not appear in the figure.

<!-- image -->

and dimension d = 9 : the task is to predict taxi ride durations in New York City. To the best of our knowledge, this is the first time that full GP inference has been scaled to a dataset of this size. Due to memory constraints, we are unable to compute 64 posterior samples as done in Section 5.1 for computing NLL, so we only report RMSE.

The results are shown in Fig. 3. ADASAP obtains the lowest RMSE out of all the methods. Once again, SDD is sensitive to the stepsize: SDD-1 obtains an RMSE of 0.60, as opposed to ADASAP , which obtains an RMSE of 0.50, while SDD-100 diverges. SDD-10 obtains an RMSE of 0.52, which is similar to that of ADASAP . However, ADASAP reaches the RMSE attained by SDD-10 in 45% less time than SDD-10, which translates to a difference of 14 hours of runtime! Furthermore, PCG runs out of memory, as the memory required for storing the sketch in single precision is 3 . 31 · 10 8 · 100 · 4 ≈ 130 GB, which exceeds the 48 GB of memory in the A6000 GPUs used in our experiments.

## 5.3 Large-scale Bayesian optimization

We run ADASAP and competitors on a variant of the synthetic large-scale Bayesian optimization tasks from Lin et al. [2023]. These Bayesian optimization tasks consist of finding the maximum of black

box functions f : [0 , 1] 8 → R sampled from a Matérn-3/2 Gaussian process. We use two different lengthscales (2.0 and 3.0) and 5 random functions per lengthscale. To avoid misspecification, we set the kernel of each model to match that of the black box function.

The results are shown in Table 2. ADASAP makes the biggest improvement over the random search baseline. As we have seen in the other experiments, SDD is sensitive to the stepsize: SDD-10 and SDD-100 make no progress. PCG also makes less progress than ADASAP across both lengthscales.

Table 2: Percentage improvement over random search for Bayesian optimization, averaged over five seeds. SDD-10 and SDD-100 provide no improvement over random search because they are unstable.

|          |   Lengthscale = 2.0 |   Lengthscale = 3.0 |
|----------|---------------------|---------------------|
| ADASAP   |               10.42 |               13.86 |
| ADASAP-I |                7.04 |               11.27 |
| SDD-1    |                6.5  |               11.17 |
| SDD-10   |                0    |                0    |
| SDD-100  |                0    |                0    |
| PCG      |                0.13 |                5.54 |

## 6 Conclusion

We introduce ADASAP , an approximate, distributed, accelerated sketch-and-project method for GP inference. We demonstrate that ADASAP outperforms state-of-the-art GP inference methods like PCG and SDD on large-scale benchmark datasets, a huge-scale dataset with &gt; 3 · 10 8 samples, and large-scale Bayesian optimization. Moreover, we show that SAP -style methods are theoretically principled for GP inference-we prove that SAP is the first efficient, condition number-free algorithm for estimating the posterior mean along the top spectral basis functions. Future work should extend the theoretical results for SAP to ADASAP and investigate ADASAP in lower precision (e.g., float16).

## Acknowledgments and Disclosure of Funding

We would like to thank Jihao Andreas Lin for helpful discussions regarding this work. PR, ZF, SF, and MUgratefully acknowledge support from the National Science Foundation (NSF) Award IIS-2233762, the Office of Naval Research (ONR) Awards N000142212825, N000142412306, and N000142312203, the Alfred P. Sloan Foundation, and from IBM Research as a founding member of Stanford Institute for Human-centered Artificial Intelligence (HAI). MD and SG gratefully acknowledge support from NSF Award CCF-2338655.

## References

- Ahmad Abdelfattah, Hartwig Anzt, Erik G Boman, Erin Carson, Terry Cojean, Jack Dongarra, Alyson Fox, Mark Gates, Nicholas J Higham, Xiaoye S Li, Jennifer Loe, Piotr Luszczek, Srikara Pranesh, Siva Rajamanickam, Tobias Ribizel, Barry F Smith, Kasia Swirydowicz, Stephen Thomas, Stanimire Tomov, Yaohung M Tsai, and Ulrike Meier Yang. A survey of numerical linear algebra methods utilizing mixed-precision arithmetic. The International Journal of High Performance Computing Applications , 35(4):344-369, 2021.
- Sivaram Ambikasaran, Daniel Foreman-Mackey, Leslie Greengard, David W Hogg, and Michael O'Neil. Fast direct methods for Gaussian processes. IEEE Transactions on Pattern Analysis and Machine Intelligence , 38(2):252-265, 2015.
- Nima Anari and Michał Derezi´ nski. Isotropy and log-concave polynomials: Accelerated sampling and high-precision counting of matroid bases. In 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS) , 2020.
- Nima Anari, Yang P Liu, and Thuy-Duong Vuong. Optimal sublinear sampling of spanning trees and determinantal point processes via average-case entropic independence. SIAM Journal on Computing , pages FOCS22-93, 2024.

- Haim Avron, Kenneth L Clarkson, and David P Woodruff. Faster kernel ridge regression using sketching and preconditioning. SIAM Journal on Matrix Analysis and Applications , 38(4):11161138, 2017.
- Francis Bach. Sharp analysis of low-rank kernel matrix approximations. In Conference on Learning Theory , 2013.
- Mikhail Belkin. Approximation beats concentration? An approximation view on inference with smooth radial kernels. In Conference On Learning Theory , 2018.
- Daniele Calandriello, Michał Derezi´ nski, and Michal Valko. Sampling from a k-DPP without looking at all items. In Advances in Neural Information Processing Systems , 2020.
- Andrea Caponnetto and Ernesto DeVito. Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics , 7:331-368, 2007.
- Benjamin Charlier, Jean Feydy, Joan Alexis Glaunes, François-David Collin, and Ghislain Durif. Kernel operations on the GPU, with autodiff, without memory overflows. Journal of Machine Learning Research , 22(74):1-6, 2021.
- Yifan Chen, Houman Owhadi, and Florian Schäfer. Sparse Cholesky factorization for solving nonlinear PDEs via Gaussian processes. Mathematics of Computation , 94(353):1235-1280, 2025.
- Li-Fang Cheng, Bianca Dumitrascu, Gregory Darnell, Corey Chivers, Michael Draugelis, Kai Li, and Barbara E Engelhardt. Sparse multi-output gaussian processes for online medical time series prediction. BMC medical informatics and decision making , 20:1-23, 2020.
- Stefan Chmiela, Alexandre Tkatchenko, Huziel E. Sauceda, Igor Poltavsky, Kristof T. Schütt, and Klaus-Robert Müller. Machine learning of accurate energy-conserving molecular force fields. Science Advances , 3(5):e1603015, 2017.
- Kurt Cutajar, Michael Osborne, John Cunningham, and Maurizio Filippone. Preconditioning kernel matrices. In Proceedings of the 33rd International Conference on Machine Learning , 2016.
- Michał Derezi´ nski and Michael W Mahoney. Determinantal point processes in randomized numerical linear algebra. Notices of the American Mathematical Society , 68(1):34-45, 2021.
- Michał Derezi´ nski and Elizaveta Rebrova. Sharp analysis of sketch-and-project methods via a connection to randomized singular value decomposition. SIAM Journal on Mathematics of Data Science , 6(1):127-153, 2024.
- Michał Derezi´ nski and Jiaming Yang. Solving dense linear systems faster than via preconditioning. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing , 2024.
- Michał Derezi´ nski, Daniel LeJeune, Deanna Needell, and Elizaveta Rebrova. Fine-grained analysis and faster algorithms for iteratively solving linear systems. arXiv preprint arXiv:2405.05818 , 2024.
- Michał Derezi´ nski, Deanna Needell, Elizaveta Rebrova, and Jiaming Yang. Randomized Kaczmarz methods with beyond-Krylov convergence. arXiv preprint arXiv:2501.11673 , 2025.
- Mateo Díaz, Ethan N Epperly, Zachary Frangella, Joel A Tropp, and Robert J Webber. Robust, randomized preconditioning for kernel ridge regression. arXiv preprint arXiv:2304.12465 , 2023.
- Lee H Dicker, Dean P Foster, and Daniel Hsu. Kernel ridge vs. principal component regression: Minimax bounds and the qualification of regularization operators. Electronic Journal of Statistics , 11:1022-1047, 2017.
- Ethan N Epperly, Gil Goldshlager, and Robert J Webber. Randomized Kaczmarz with tail averaging. arXiv preprint arXiv:2411.19877 , 2024a.
- Ethan N. Epperly, Joel A. Tropp, and Robert J. Webber. Embrace rejection: Kernel matrix approximation by accelerated randomly pivoted Cholesky. arXiv preprint arXiv:2410.03969 , 2024b.

- Murat A Erdogdu and Andrea Montanari. Convergence rates of sub-sampled Newton methods. In Advances in Neural Information Processing Systems , 2015.
- Zachary Frangella, Joel A. Tropp, and Madeleine Udell. Randomized Nyström preconditioning. SIAM Journal on Matrix Analysis and Applications , 44(2):718-752, 2023.
- Zachary Frangella, Pratik Rathore, Shipu Zhao, and Madeleine Udell. Promise: Preconditioned stochastic optimization methods by incorporating scalable curvature estimates. Journal of Machine Learning Research , 25(346):1-57, 2024a.
- Zachary Frangella, Pratik Rathore, Shipu Zhao, and Madeleine Udell. Sketchysgd: reliable stochastic optimization via randomized curvature estimates. SIAM Journal on Mathematics of Data Science , 6(4):1173-1204, 2024b.
- Peter I Frazier and Jialei Wang. Bayesian optimization for materials design. Information science for materials discovery and design , pages 45-75, 2016.
- Jacob Gardner, Geoff Pleiss, Kilian Q Weinberger, David Bindel, and Andrew G Wilson. GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration. In Advances in Neural Information Processing Systems , 2018.
- Alex Gittens and Michael W Mahoney. Revisiting the Nyström method for improved large-scale machine learning. The Journal of Machine Learning Research , 17(1):3977-4041, 2016.
- Robert Gower, Filip Hanzely, Peter Richtarik, and Sebastian U Stich. Accelerated stochastic matrix inversion: General theory and speeding up BFGS rules for faster second-order optimization. In Advances in Neural Information Processing Systems , 2018.
- Robert Gower, Dmitry Kovalev, Felix Lieder, and Peter Richtárik. RSN: randomized subspace Newton. In Advances in Neural Information Processing Systems , 2019.
- Robert M Gower and Peter Richtárik. Randomized iterative methods for linear systems. SIAM Journal on Matrix Analysis and Applications , 36(4):1660-1690, 2015.
- Philip Greengard, Manas Rachh, and Alex H. Barnett. Equispaced Fourier representations for efficient Gaussian process regression from a billion data points. SIAM/ASA Journal on Uncertainty Quantification , 13(1):63-89, 2025.
- Filip Hanzely, Nikita Doikov, Yurii Nesterov, and Peter Richtarik. Stochastic subspace cubic Newton method. In Proceedings of the 37th International Conference on Machine Learning , 2020.
- James Hensman, Nicolò Fusi, and Neil D Lawrence. Gaussian processes for big data. In Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence , 2013.
- James Hensman, Alexander Matthews, and Zoubin Ghahramani. Scalable variational Gaussian process classification. In Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics , 2015.
- James Hensman, Nicolas Durrande, and Arno Solin. Variational Fourier features for Gaussian processes. Journal of Machine Learning Research , 18(151):1-52, 2018.
- José Miguel Hernández-Lobato, James Requeima, Edward O. Pyzer-Knapp, and Alán Aspuru-Guzik. Parallel and distributed Thompson sampling for large-scale accelerated exploration of chemical space. In Proceedings of the 34th International Conference on Machine Learning , 2017.
- Nicholas J Higham. Accuracy and stability of numerical algorithms . SIAM, 2002.
- Prateek Jain, Sham M Kakade, Rahul Kidambi, Praneeth Netrapalli, and Aaron Sidford. Parallelizing stochastic gradient descent for least squares regression: mini-batching, averaging, and model misspecification. Journal of Machine Learning Research , 18(223):1-42, 2018.
- Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In Advances in Neural Information Processing Systems , 2013.

- Motonobu Kanagawa, Philipp Hennig, Dino Sejdinovic, and Bharath K Sriperumbudur. Gaussian processes and kernel methods: A review on connections and equivalences. arXiv preprint arXiv:1807.02582 , 2018.
- Jacek Kuczy´ nski and Henryk Wo´ zniakowski. Estimating the largest eigenvalue by the power and lanczos algorithms with a random start. SIAM Journal on Matrix Analysis and Applications , 13(4): 1094-1122, 1992.
- Alex Kulesza and Ben Taskar. Determinantal point processes for machine learning. Foundations and Trends® in Machine Learning , 5(2-3):123-286, 2012.
- D. Leventhal and A. S. Lewis. Randomized methods for linear constraints: Convergence rates and conditioning. Mathematics of Operations Research , 35(3):641-654, 2010.
- Jihao Andreas Lin, Javier Antoran, Shreyas Padhy, David Janz, José Miguel Hernández-Lobato, and Alexander Terenin. Sampling from Gaussian process posteriors using stochastic gradient descent. In Advances in Neural Information Processing Systems , 2023.
- Jihao Andreas Lin, Shreyas Padhy, Javier Antoran, Austin Tripp, Alexander Terenin, Csaba Szepesvari, José Miguel Hernández-Lobato, and David Janz. Stochastic gradient descent for Gaussian processes done right. In International Conference on Learning Representations , 2024.
- Junhong Lin, Alessandro Rudi, Lorenzo Rosasco, and Volkan Cevher. Optimal rates for spectral algorithms with least-squares regression over Hilbert spaces. Applied and Computational Harmonic Analysis , 48(3):868-890, 2020.
- Siyuan Ma and Mikhail Belkin. Diving into the shallows: A computational perspective on large-scale shallow learning. In Advances in Neural Information Processing Systems , 2017.
- Per-Gunnar Martinsson and Joel A Tropp. Randomized numerical linear algebra: Foundations and algorithms. Acta Numerica , 29:403-572, 2020.
- Ian C McDowell, Dinesh Manandhar, Christopher M Vockley, Amy K Schmid, Timothy E Reddy, and Barbara E Engelhardt. Clustering gene expression time series data using an infinite gaussian process mixture model. PLoS computational biology , 14(1):e1005896, 2018.
- Victor Minden, Anil Damle, Kenneth L Ho, and Lexing Ying. Fast spatial Gaussian process maximum likelihood estimation via skeletonization factorizations. Multiscale Modeling &amp; Simulation , 15(4): 1584-1611, 2017.
- Mojmir Mutny, Michal Derezinski, and Andreas Krause. Convergence analysis of block coordinate algorithms with determinantal sampling. In International Conference on Artificial Intelligence and Statistics , 2020.
- Arkadi S Nemirovski and David B Yudin. Problem complexity and method efficiency in optimization . Wiley-Interscience, 1983.
- Y Nesterov. A method of solving a convex programming problem with convergence rate o (1/k** 2). Doklady Akademii Nauk SSSR , 269(3), 1983.
- Dianne P O'Leary. The block conjugate gradient algorithm and related methods. Linear Algebra and its Applications , 29:293-322, 1980.
- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems , 2019.
- Zheng Qu and Peter Richtárik. Coordinate descent with arbitrary sampling i: Algorithms and complexity. Optimization Methods and Software , 31(5):829-857, 2016.
- Zheng Qu, Peter Richtarik, Martin Takac, and Olivier Fercoq. SDNA: Stochastic dual Newton ascent for empirical risk minimization. In Proceedings of the 33rd International Conference on Machine Learning , 2016.

- Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems , 2007.
- Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian processes for machine learning . The MIT Press, 11 2005.
- Pratik Rathore, Zachary Frangella, Jiaming Yang, Michał Derezi´ nski, and Madeleine Udell. Have ASkotch: A neat solution for large-scale kernel ridge regression. arXiv preprint 2407.10070 , 2025.
- Peter Richtárik and Martin Takáˇ c. Iteration complexity of randomized block-coordinate descent methods for minimizing a composite function. Mathematical Programming , 144(1):1-38, 2014.
- Anton Rodomanov and Dmitry Kropotov. A randomized coordinate descent method with volume sampling. SIAM Journal on Optimization , 30(3):1878-1904, 2020.
- Alessandro Rudi and Lorenzo Rosasco. Generalization properties of learning with random features. In Advances in Neural Information Processing Systems , 2017.
- Alessandro Rudi, Raffaello Camoriano, and Lorenzo Rosasco. Less is more: Nyström computational regularization. Advances in neural information processing systems , 28, 2015.
- Ohad Shamir and Tong Zhang. Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In Proceedings of the 30th International Conference on Machine Learning , 2013.
- Stefan Steinerberger. Randomized Kaczmarz converges along small singular vectors. SIAM Journal on Matrix Analysis and Applications , 42(2):608-615, 2021.
- Thomas Strohmer and Roman Vershynin. A randomized Kaczmarz algorithm with exponential convergence. Journal of Fourier Analysis and Applications , 15(2):262-278, 2009.
- Michalis Titsias. Variational learning of inducing variables in sparse Gaussian processes. In Artificial intelligence and statistics , 2009.
- Lloyd N Trefethen and David Bau. Numerical linear algebra . SIAM, 1997.
- Joel A Tropp, Alp Yurtsever, Madeleine Udell, and Volkan Cevher. Fixed-rank approximation of a positive-semidefinite matrix from streaming data. In Advances in Neural Information Processing Systems , 2017.
- Stephen Tu, Shivaram Venkataraman, Ashia C. Wilson, Alex Gittens, Michael I. Jordan, and Benjamin Recht. Breaking locality accelerates block Gauss-Seidel. In Proceedings of the 34th International Conference on Machine Learning , 2017.
- Ke Wang, Geoff Pleiss, Jacob Gardner, Stephen Tyree, Kilian Q Weinberger, and Andrew Gordon Wilson. Exact Gaussian processes on a million data points. In Advances in Neural Information Processing Systems , 2019.
- Christopher Williams and Matthias Seeger. Using the Nyström method to speed up kernel machines. In Advances in Neural Information Processing Systems , 2000.
- Andrew Wilson and Hannes Nickisch. Kernel interpolation for scalable structured gaussian processes (kiss-gp). In Proceedings of the 32nd International Conference on Machine Learning , 2015.
- James Wilson, Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, and Marc Deisenroth. Efficiently sampling functions from Gaussian process posteriors. In Proceedings of the 37th International Conference on Machine Learning , 2020.
- James T. Wilson, Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, and Marc Peter Deisenroth. Pathwise conditioning of Gaussian processes. Journal of Machine Learning Research , 22(105):1-47, 2021.
- Kaiwen Wu, Jonathan Wenger, Haydn T Jones, Geoff Pleiss, and Jacob Gardner. Large-scale Gaussian processes via alternating projection. In International Conference on Artificial Intelligence and Statistics , 2024.

## A Related Work

We review the literature on Gaussian process (GP) inference and sketch-and-project, which are the two key areas that serve as foundations for ADASAP .

## A.1 GP inference

Naive GP inference based on solving the linear systems in (1) exactly and generating posterior samples via (3) has cubic cost in number of training and test points, making it prohibitively expensive for datasets whose size exceeds n ∼ 10 4 . Given its prominence in scientific computing and machine learning, much work has been to done to scale GP inference to the big-data setting. Prior work in this area can be (roughly) divided into four approaches: (i) PCG-based inference, (ii) SGD-based inference, (iii) low-dimensional inference, and (iv) variational inference.

## A.1.1 PCG-based inference

A natural approach to scaling GP inference is to replace the exact linear system solves by a direct method with approximate linear system solves by an iterative method. Gardner et al. [2018] showed that by utilizing batched PCG and GPU acceleration, GP inference could be scaled to datasets of size n ∼ 10 5 . Wang et al. [2019] further showed that by distributing the kernel matrix across multiple GPUs, that inference could be applied to datasets of size n ∼ 10 6 . Unfortunately, these approaches still have shortcomings. While Gardner et al. [2018], Wang et al. [2019] show that exact GPs can be scaled to datasets with n ∼ 10 6 by leveraging PCG and GPUs, they do not address the cubic complexity of generating posterior samples in (3). Moreover, when n ∼ 10 6 , PCG can become prohibitively expensive from both a memory and computational standpoint, as most preconditioners require O ( nr ) storage [Cutajar et al., 2016, Avron et al., 2017, Díaz et al., 2023] and each PCG iteration costs O ( n 2 ) .

## A.1.2 SGD-based inference

To address the limitations of PCG-based inference, Lin et al. [2023] proposed to replace PCG with SGD. This reduces iteration cost from O ( n 2 ) to O ( bn ) , where b is the gradient batchsize. In addition, they address the challenge of the cubic cost of generating posterior samples by adopting the pathwise conditioning strategy [Wilson et al., 2020, 2021]. Lin et al. [2023] show SGD outperforms PCG and Stochastic Variational Gaussian Processes (SVGP) [Hensman et al., 2013] on various posterior sampling tasks. Lin et al. [2024] further improved upon SGD by introducing Stochastic Dual Descent (SDD). SDD is a coordinate descent algorithm that leverages the dual formulation of the KRR problem to reduce the dependence on the conditioning of K from κ ( K ( K + λI )) to κ ( K + λI ) . In addition, SDD incorporates momentum and geometric averaging to further enhance performance. Compared to SGD, PCG, and SVGP, Lin et al. [2024] shows that SDD yields the best empirical performance.

While Lin et al. [2023, 2024] address the scaling limitations of PCG and the issue of efficiently sampling from the posterior, there is still room for improvement. In terms of theory, Lin et al. [2023] only establishes that SGD is guaranteed to converge at a reasonable rate along the top eigenvectors of K . Convergence along the small eigenvectors is extremely slow, as the rate in Lin et al. [2023] along the ℓ th eigenvector depends upon 1 / √ λ i . For SDD, the convergence rate is unknown, as Lin et al. [2024] provides no convergence analysis. Aside from unsatisfactory convergence analyses, it is known from optimization theory, that in the worst case, the convergence rate of SGD and SDD is controlled by the condition number [Nemirovski and Yudin, 1983]. As ill-conditioning is a defining property of kernel matrices, this means SDD and SGD will converge slowly. Another price incurred by SDD and SGD for their fast performance is the presence of additional hyperparameters. In particular, the stepsize for these methods must be set appropriately to obtain satisfactory convergence. Tuning the stepsize, while possible, can become expensive for large-scale inference on datasets like taxi. This is in contrast to PCG, which is stepsize-free.

ADASAP enjoys the scalability benefits of SDD and SGD, but without their limitations. It only operates on a batch of data at each iteration, so it avoids the O ( n 2 ) iteration cost of PCG. By incorporating subspace preconditioning, ADASAP addresses the ill-conditioning of the kernel matrix to achieve fast

convergence. ADASAP also comes with reliable default hyperparameters, which obviates the need for costly hyperparameter tuning.

## A.1.3 Low-dimensional inference

When the dimension d of the data matrix X is low ( d ≤ 3 ), GP inference can be made far more efficient by exploiting the structure of the kernel matrix and the low-dimensionality of the data. Common approaches in this vein include rank-structured matrices [Ambikasaran et al., 2015, Minden et al., 2017], sparse Cholesky factorization [Chen et al., 2025], and kernel interpolation [Wilson and Nickisch, 2015, Greengard et al., 2025]. In most cases these methods enable O ( n ) training time and inference, a massive improvement over standard GP inference methods. This linear time complexity has allowed GP training to scale to datasets with n as large as 10 9 [Greengard et al., 2025]. Unfortunately, the complexity of these algorithms grows exponentially with the dimension, making them impractical when d &gt; 3 .

## A.1.4 Variational inference

To address the challenge of large-scale GP inference, many prior works have focused on developing scalable approximate GP methods. The most popular methods are based on variational inference [Titsias, 2009, Hensman et al., 2013, 2015, 2018]. The most well-known approach is SVGP [Hensman et al., 2013], which selects a set of m inducing points by maximizing an evidence lower bound (ELBO). The use of inducing points leads to only need to work with an n × m kernel matrix instead of the full n × n kernel matrix. This leads reduces the cost of training to O ( nm 2 + m 3 ) , a significant improvement over exact inference methods when m can be taken to be much smaller than n . While SVGP and related algorithms significantly reduce the cost of GPs, they generally exhibit worse performance on inference than methods which make use of the full data [Wang et al., 2019, Lin et al., 2023, 2024]. Thus, it is advisable to use exact inference whenever possible.

## A.2 Sketch-and-project

ADASAP builds off the sketch-and-project framework for solving consistent linear systems. Sketchand-project was first formulated in Gower and Richtárik [2015], who showed that randomized Kaczmarz [Strohmer and Vershynin, 2009], randomized coordinate descent [Leventhal and Lewis, 2010], and randomized Newton [Qu et al., 2016], are all special cases of sketch-and-project. They also established a linear convergence rate for sketch-and-project, which is controlled by the smallest eigenvalue of an expected projection matrix. However, Gower and Richtárik [2015] were unable to provide a fine-grained analysis of sketch-and-project in terms of the spectral properties of the linear system, except for a few special cases.

Follow-up work has combined sketch-and-project with techniques such as Nesterov acceleration [Tu et al., 2017] and extended sketch-and-project to quasi-Newton and Newton type methods Gower et al. [2018, 2019], Hanzely et al. [2020]. Despite these extensions, a sharp convergence analysis of sketch-and-project remained elusive, even in the original setting of consistent linear systems.

Recent work has finally given a sharp analysis of sketch-and-project. Using powerful tools from random matrix theory, Derezi´ nski and Rebrova [2024] provided the first sharp analysis of sketch-andproject when the sketching matrix is sub-Gaussian or sparse. In particular, they were the first to prove that if the matrix exhibits a favorable spectral decay profile, then sketch-and-project exhibits condition number-free convergence. Derezi´ nski and Yang [2024] improve upon these results, showing that by employing a pre-processing step, one can avoid using expensive sketching matrices at each step, and simply sample rows uniformly. They accomplish this by leveraging tools from determinantal point process theory [Mutny et al., 2020, Rodomanov and Kropotov, 2020, Anari et al., 2024]. Derezi´ nski et al. [2024] further refines these results by incorporating Nesterov acceleration and demonstrating an improved computational complexity relative to a fine-grained analysis of PCG-that is, an analysis of PCG that goes beyond the worst-case ˜ O ( √ κ ) iteration complexity [Trefethen and Bau, 1997].

In addition to improvements in the analysis of exact sketch-and-project for linear systems, significant strides have been made when approximate subspace preconditioning is employed. Derezi´ nski and Yang [2024], Derezi´ nski et al. [2024] both use PCG to approximately solve the linear system defined by the subspace Hessian in each iteration of SAP , while Derezi´ nski et al. [2025] augment that system with Tikhonov regularization to further improve its conditioning. Rathore et al. [2025] consider an

approximate sketch-and-project method for solving kernel ridge regression with one right-hand side. They replace K BB + λI in SAP with ˆ K BB + ρI , where ˆ K BB is a randomized Nyström approximation of K BB . They establish linear convergence for the method, with a rate that is comparable to exact SAP when the kernel matrix exhibits an appropriate rate of spectral decay.

Among the above-mentioned works, ADASAP is closest to Rathore et al. [2025], but with significant algorithmic differences such as tail averaging, handling multiple right-hand sides, and distributing bottleneck operations over multiple devices. On the theoretical side, our analysis focuses on the convergence along the top ℓ -subspace using exact SAP with tail-averaging. In contrast, Rathore et al. [2025] focus on convergence of approximate sketch-and-project (with and without acceleration) over the entire space , and as a result, they are only able to show a fast convergence guarantee under certain strong conditions on the spectral decay (effectively limiting the condition number of the problem). Consequently, the global convergence analysis of Rathore et al. [2025] does not explain the rapid initial progress that SAP-style algorithms make on test error. We believe that our two-phase analysis consisting of (i) fast condition number-free sublinear convergence, followed by (ii) a slower global convergence rate dependent upon the subspace condition number, better captures this phenomenon. In particular, our result is the first step in developing a systematic analysis of the generalization properties of SAP-style algorithms.

To the best of our knowledge, our work is the first to systematically investigate the convergence rate of SAP with tail averaging along the top eigenspace. Epperly et al. [2024a] combines tail averaging with randomized Kaczmarz for solving inconsistent linear systems, but they do not investigate the convergence rate along the top spectral subspaces. Steinerberger [2021], Derezi´ nski and Rebrova [2024] look at the convergence rate of the expected iterates along a particular eigenvector. However, as noted in Section 3.2, this does not lead to a meaningful convergence rate for the iterates produced by the algorithm, and this also fails to capture the behavior over an entire subspace. Thus, our analysis provides the first concrete characterization of the convergence rate of tail-averaged SAP along the top eigenvectors.

## B Proofs of the Main Results

In this section, we give the detailed proofs for the main theoretical results given in Section 3.2, namely Theorem 3.2 and Corollary 3.3. We start with an informal overview of the analysis in Section B.1, followed by a formal convergence analysis of SAP for solving positive-definite linear systems in Section B.2, and finally adapting the results to GP posterior mean estimation in Section B.3.

Throughout, we adopt the shorthand A λ := A + λI , where A is a square matrix, λ ∈ R , and I denotes the identity matrix of the same size as A .

## B.1 Overview of the analysis

Let w ⋆ := K -1 λ y ∈ R n be the solution of the linear system defined by K λ and y , an let us use w t ∈ R n to denote the iterates produced by sketch-and-project when solving that system. To convert from the posterior mean error to the Euclidean norm, we first write the spectral basis functions as u ( i ) = 1 √ k ( , X ) v i , where ( λ i , v i ) is the i th -largest eigenpair of K . Thus, for any w R n and

λ i · ∈ m = k ( · , X ) w ∈ H , we have ∥ proj ℓ ( m -m n ) ∥ H = ∥ Q ℓ K 1 / 2 ( w -w ⋆ ) ∥ ≤ ∥ Q ℓ K 1 / 2 λ ( w -w ⋆ ) ∥ , where Q ℓ = ∑ ℓ i =1 v i v T i is the projection onto the topℓ subspace of K .

Having converted to the Euclidean norm, we express the sketch-and-project update as a recursive formula for its (scaled) residual vector as follows:

<!-- formula-not-decoded -->

where Π B = K 1 / 2 λ I T B ( K B , B + λI ) † I B K 1 / 2 λ is a random projection defined by B . To characterize the expected convergence of sketch-and-project, we must therefore control the average-case properties of Π B . We achieve this by relying on a row sampling technique known as determinantal point processes (DPPs) [Kulesza and Taskar, 2012, Derezi´ nski and Mahoney, 2021]. A b -DPP( K λ ) is defined as a distribution over size b index sets B ⊆ [ n ] such that Pr[ B ] ∝ det( K BB + λI ) . DPPs can be sampled from efficiently [Calandriello et al., 2020, Anari et al., 2024]: after an initial preprocessing cost of ˜ O ( nb 2 ) , we can produce each DPP sample of size b in ˜ O ( b 3 ) time.

Prior work [Derezi´ nski and Yang, 2024] has shown that when B is drawn according to 2 b -DPP( K λ ) , then the expectation of Π B has the same eigenbasis as K . Concretely, we can show that ¯ Π := E Π B = V Λ V T with Λ = diag( ¯ λ 1 , ..., ¯ λ n ) , where ¯ λ i ≥ 1 1+ ϕ ( b,i ) and V consists of the eigenvectors of K . This leads to a convergence guarantee for the expected residual vector:

<!-- formula-not-decoded -->

where we used that ¯ Π , K λ , and Q ℓ commute with each other. This would suggest that we should obtain a fast linear rate (1 -1 1+ ϕ ( b,ℓ ) ) t for the sketch-and-project iterates in the topℓ subspace norm. However, it turns out that the actual residual does not attain the same convergence guarantee as its expectation because, unlike ¯ Π , the random projection Π B does not commute with K λ or Q ℓ . Indeed,

<!-- formula-not-decoded -->

and even though the matrix ( I -Π B ) Q ℓ ( I -Π B ) is low-rank, its expectation may not be, since Π B does not commute with Q ℓ . This means that we have no hope of showing a linear convergence rate along the topℓ subspace. Nevertheless, we are still able to obtain the following bound:

<!-- formula-not-decoded -->

This results in a bias-variance decomposition: the first term is the bias, which exhibits fast convergence in topℓ subspace, while the second term is the variance, which accounts for the noise coming from the orthogonal complement subspace. We then use tail averaging to decay this noise, which allows us to benefit from the fast convergence of expected iterates through a sublinear rate ϕ ( b,ℓ ) t .

Finally, to attain the asymptotically faster linear rate for the residual vectors, we observe that every topℓ subspace norm is upper-bounded by the full norm: ∥ Q ℓ ∆ t ∥ ≤ ∥ ∆ t ∥ . Thus, we can effectively repeat the above analysis with ℓ = n (since Q n = I ), in which case the variance term becomes zero, and we recover a linear rate of the form ( 1 -1 1+ ϕ ( b,n ) ) t .

## B.2 Convergence along topℓ dimensional subspace for positive-definite matrices

In this section, we consider the general problem of solving a linear system Aw = y for an n × n positive-definite matrix A . We provide theoretical results providing the fast convergence guarantees for the iterate sequence generated by SAP along the topℓ dimensional subspace of A , with Theorem B.3 being the main result of this section. We use Theorem B.3 in the next section to derive the subspace convergence result for posterior mean in GP inference, proving Theorem 3.2.

Preliminaries and notation We start by introducing the notation and some results from existing literature for SAP and DPPs. Let A be an n × n symmetric positive-definite matrix and let A = V DV T be its eigendecomposition, where D is diagonal with entries λ 1 ≥ ... ≥ λ n . For other matrices we use λ i ( M ) to denote the eigenvalues of M . For any vector v ∈ R n , define ∥ v ∥ A := √ v T Av . Let V ℓ ∈ R n × ℓ consist of the topℓ orthonormal eigenvectors of A and Q ℓ = V ℓ V T ℓ be the orthogonal projector corresponding to the topℓ eigenvectors of A . Werely on subsampling from fixedsize DPPs [Kulesza and Taskar, 2012, Derezi´ nski and Mahoney, 2021] for our theoretical results. Here, for convenience, we consider the block size to be 2 b , in contrast to b as considered in Algorithm 1. For a fixed b and a positive-definite matrix A , we identify 2 b -DPP( A ) as a distribution over subsets of { 1 , . . . , n } of size 2 b , where any subset B has probability proportional to the corresponding principal submatrix of A , i.e., det( A B , B ) . The subsampling matrix S ∈ R 2 b × n corresponding to the set B is defined as a matrix whose rows are the standard basis vectors associated with the indices in B . For notational convenience, we will sometimes say that S is drawn from 2 b -DPP( A ), or S ∼ 2 b -DPP( A ), keeping the index set B implicit.

We use the following result from the literature that provides an exact characterization of the eigenvectors of the expected projection matrix that arises in the analysis of sketch-and-project methods, when using subsampling from fixed-size DPPs.

Lemma B.1 (Expected projection under 2 b -DPP( A ), adapted from Lemma 4.1 [Derezi´ nski and Yang, 2024]) . Let A ∈ R n × n be a symmetric positive-definite matrix with eigenvalues λ 1 ≥ λ 2 ≥ ... ≥ λ n , and let 1 ≤ b &lt; n/ 2 be fixed. Let S ∼ 2 b -DPP( A ). Then, we have

<!-- formula-not-decoded -->

where D ′ is a diagonal matrix with j th diagonal entry is lower bounded by λ j λ j + 1 b ∑ i&gt;b λ i .

As exact sampling from DPPs is often expensive and requires performing operations as costly as performing the eigendecomposition of A [Kulesza and Taskar, 2012], numerous works have looked at approximate sampling from these distributions [Calandriello et al., 2020, Anari et al., 2024]. Since we exploit the exact characterization of the eigenvectors of the expected projection matrix, our theory requires a very accurate DPP sampling algorithm. Fortunately, the Markov Chain Monte Carlo tools developed by Anari and Derezi´ nski [2020], Anari et al. [2024] allow near-exact sampling from fixed-size DPPs (in the sense that the samples are with high probability indistinguishable from the exact distribution), while offering significant computational gains over exact sampling.

Lemma B.2 (Sampling from b -DPP( A ), adapted from Anari et al. [2024]) . Given a positive-definite matrix A , there exists an algorithm that draws t ≥ 1 approximate samples from 2 b -DPP( A ) in time O ( nb 2 log 4 n + tb 3 log 3 n ) . Furthermore, each drawn sample is indistinguishable from an exact sample from 2 b -DPP( A ) with probability at least 1 -n -O (1) .

We now provide the main result of this section, incorporating the above two results in the analysis of sketch-and-project (SAP) along the topℓ subspace of A . Since throughout this section we focus on a linear system with a single right-hand side, we will use w t instead of W t to denote the SAP iterates, as these are always vectors. The update rule for SAP is now:

<!-- formula-not-decoded -->

where Π = A 1 / 2 S T ( SAS T ) † SA 1 / 2 and w ⋆ = A -1 y . Let ¯ Π denote E Π .

In the rest of the section we prove the following result, which is a slight generalization of Theorem 3.2.

Theorem B.3 (Fast convergence along topℓ subspace) . Let b &lt; n/ 2 be fixed and S ∈ R 2 b × n be a random subsampling matrix sampled from 2 b -DPP( A ). Furthermore, let 1 ≤ ℓ &lt; b be also fixed and λ ℓ ( ¯ Π) ≥ 2 λ n ( ¯ Π) . For any t &gt; 1 define ¯ w t = 2 t ∑ t -1 i = t/ 2 w i where w i are generated using (5) . Then,

<!-- formula-not-decoded -->

where ϕ ( b, p ) = 1 b ∑ i&gt;b λ i λ p .

The assumption λ ℓ ( ¯ Π) ≥ 2 λ n ( ¯ Π) is not restrictive buts lets us provide a cleaner analysis by avoiding the edge case of λ ℓ ( ¯ Π) ≈ λ n ( ¯ Π) . In fact, in this corner case, we can simply rely on the existing SAP analysis from previous works and recover our main result, Theorem 3.2. For completeness, we derive Theorem B.9 for posterior GP mean inference in Section B.3 without the assumption λ ℓ ( ¯ Π) ≥ 2 λ n ( ¯ Π) . The proof of Theorem B.3 appears after the proof of Theorem B.8. We build towards the proof starting with the following result for SAP [Derezi´ nski and Yang, 2024]:

Lemma B.4 (Linear convergence with SAP) . Let S ∈ R 2 b × n be a random subsampling matrix sampled from 2 b -DPP( A ). Then,

<!-- formula-not-decoded -->

However, in the following result, we show that along the topℓ eigenspace of A , the expected iterates can converge at a much faster rate than 1 -λ n ( ¯ Π) .

Lemma B.5 (Convergence of expected iterates along topℓ subspace) . Let S ∈ R 2 b × n be a random subsampling matrix sampled from 2 b -DPP( A ). Then

<!-- formula-not-decoded -->

where E t denotes conditional expectation given w t .

Proof. We have

<!-- formula-not-decoded -->

Taking expectation on both sides and squaring we get,

<!-- formula-not-decoded -->

As A = V DV T we have Π = V D 1 / 2 V T S T ( SV DV T S T ) † SV D 1 / 2 V T . Due to Lemma B.1 we know that ¯ Π and A share the same eigenvectors, therefore, ¯ Π and Q ℓ commute. Consequently,

<!-- formula-not-decoded -->

We now analyze convergence along the topℓ dimensional subspace in L2-norm by considering E ∥ A 1 / 2 ( w t +1 -w ⋆ ) ∥ 2 Q ℓ . We have,

<!-- formula-not-decoded -->

In particular, we need to upper bound E [( I -Π) Q ℓ ( I -Π)] . We prove the following lemma:

Lemma B.6 (L2-norm error along topℓ subspace) . Let S ∈ R 2 b × n be a random subsampling matrix sampled from 2 b -DPP( A ). Then

<!-- formula-not-decoded -->

Proof. First, we rewrite E [( I -Π) Q ℓ ( I -Π)] as follows:

Crucially, the first two terms terms in (6) live in the topℓ subspace, but the third term does not. Nevertheless, we are still able to bound it as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used Jensen's inequality in the last relation. Substituting in (6) we get,

<!-- formula-not-decoded -->

Using (7) we now upper bound E ∥ A 1 / 2 ( w t +1 -w ⋆ ) ∥ 2 Q ℓ as

<!-- formula-not-decoded -->

which concludes the proof.

Unrolling this recursive bound, and combining it with the convergence in the full norm, we obtain the following convergence guarantee for SAP iterates without tail averaging.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Define γ = 1 -λ n ( ¯ Π) . Invoking Lemma B.6 we deduce,

<!-- formula-not-decoded -->

Now, recursively applying the tower rule and Lemma B.4 yields

<!-- formula-not-decoded -->

Unfolding the preceding display yields,

<!-- formula-not-decoded -->

which concludes the proof.

<!-- formula-not-decoded -->

We now use the tail averaging idea similar to Epperly et al. [2024a], obtaining fast convergence along the topℓ subspace. The proof of Theorem B.3 is then derived from the following result, after combining it with fixed-size DPP sampling guarantees of Lemma B.1.

Theorem B.8 (Fast convergence along subspace for tail-averaged iterate) . Let S ∈ R 2 b × n be a random subsampling matrix sampled from 2 b -DPP( A ). For any t &gt; 2 define ˆ w t = 2 t ∑ t -1 i ≥ t/ 2 w i where w i are generated using update rule (5) . Then

<!-- formula-not-decoded -->

where α = 1 -λ ℓ ( ¯ Π) 1 -λ n ( ¯ Π) .

Proof. Let ˆ w t = 2 t ∑ t -1 i ≥ t/ 2 w i . Consider E ∥ A 1 / 2 ( ˆ w t -w ⋆ ) ∥ 2 Q ℓ , set t a = t/ 2 , and let E w r denote the expectation conditioned on w r . Applying the tower rule yields

<!-- formula-not-decoded -->

For r &lt; s , we have

<!-- formula-not-decoded -->

Recursing on the above relation yields

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

( ¯

where γ = 1 -λ n ( ¯ Π) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting t a = t/ 2 , the result immediately follows.

Completing the proof of Theorem B.3 It remains to use the guarantees for the eigenvalues of the matrix ¯ Π from Lemma B.1.

Proof of Theorem B.3. Recalling the definition of α from Theorem B.8 and our assumption that λ ℓ ( ¯ Π) ≥ 2 λ n ( ¯ Π) , we obtain for γ = 1 -λ n ( ¯ Π) that (1 -α ) -1 &lt; 2 γ/λ ℓ ( ¯ Π) . Consequently,

<!-- formula-not-decoded -->

where (1) applies Lemma B.1 and (2) defines ϕ ( b, ℓ ) = 1 b ∑ i&gt;b λ i λ ℓ . Observing the elementary inequalities:

<!-- formula-not-decoded -->

we immediately deduce from Theorem B.8 that

<!-- formula-not-decoded -->

## B.3 Posterior mean inference along topℓ subspace for GPs

We begin by providing some background on GP inference in the Hilbert space setting. This allows for graceful transition from Hilbert space norm over the posterior mean to vector norms over R n . We recall that f is a Gaussian process and { ( x i , y i ) } n i =1 represents the training data. The posterior Gaussian process is characterized by N ( m n ( · ) , k n ( · , · )) , where

Let H be the reproducing kernel Hilbert space (RKHS) associated with the kernel k ( · , · ) . Assuming m ( · ) = 0 , the mean function m n ( · ) can be identified as an element of the subspace H n defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, m n = ∑ n i =1 ( w ⋆ ) i k ( · , x i ) , where w ⋆ = ( K + λI ) -1 y . Furthermore, note that for any element m ′ ∈ H n , we have ∥ m ′ ∥ 2 H = w T Kw = ∥ w ∥ 2 K . The operator C n :=

1 n ∑ n i =1 k ( · , x i ) ⊗ k ( · , x i ) is known as the empirical covariance operator. Let v j denote the j th unit eigenvector of K with eigenvalue λ j . It is straightforward to show that u j := 1 √ λ j ∑ n i =1 v ji k ( · , x i ) is a unit eigenvector of the unnormalized empirical covariance operator ∑ n i =1 k ( · , x i ) ⊗ k ( · , x i ) with eigenvalue λ j . Let V ℓ ∈ R n × ℓ consists of topℓ orthogonal eigenvectors of K as columns and Q ℓ = V ℓ V T ℓ be a projection matrix onto the subspace spanned by v 1 , . . . , v ℓ . Consider the ℓ -dimensional subspace H ℓ defined as

<!-- formula-not-decoded -->

We claim that H ℓ is the subspace formed by topℓ eigenvectors of the empirical covariance operator. This can be seen clearly by choosing w = v j for 1 ≤ j ≤ ℓ , we get u j ∈ H ℓ . We have the following conclusions.

- For any m ′ = ∑ n i =1 w i k ( · , x i ) , we have ∥ m ′ -m n ∥ 2 H = ∥ w -w ⋆ ∥ 2 K .
- The element m ′ Q ℓ := ∑ n i =1 ( Q ℓ w ) i k ( · , x i ) is an orthogonal projection of m ′ onto H ℓ . This can be seen as

<!-- formula-not-decoded -->

and finally noting that ⟨ m ′ Q ℓ , m ′ Q c ℓ ⟩ H = w T ( I -Q ℓ ) KQ ℓ w = 0 .

- For any m ′ = ∑ n i =1 w i k ( · , x i ) , we have ∥ proj ℓ ( m ′ ) -proj ℓ ( m n ) ∥ 2 H = ∥ Q ℓ ( w -w ⋆ ) ∥ 2 K , where proj ℓ ( m ′ ) denotes orthogonal projection of m ′ onto H ℓ .

We now derive the main result of this section by using Theorem B.3. We replace A by K λ in the statement of Theorem B.3 and use λ ′ i to denote the i th eigenvalue of K λ , i.e., λ i + λ where λ i is the i th eigenvalue of K . Furthermore, Theorem 3.2 can be derived from the following result by noticing that sampling from 2 b -DPP( K λ ) costs time O ( nb 2 log 4 n ) for preprocessing and an additional O ( b 3 log 3 n ) for actual sampling at every iteration (see Lemma B.2). Here is the main result of the section:

Theorem B.9 (Subspace convergence for GP inference) . Let b &lt; n/ 2 and S ∈ R 2 b × n be a random subsampling matrix sampled from 2 b -DPP( K λ ). For any t &gt; 2 define ˆ w t = 2 t ∑ t -1 i = t/ 2 w i where w i are generated using the update rule (5) . Then, sketch-and-project initialized at 0 satisfies

<!-- formula-not-decoded -->

where ˆ m t = ∑ n i =1 ˆ w ti k ( · , w i ) and ϕ ( b, p ) = 1 b ∑ i&gt;b λ i + λ λ p + λ .

Proof. If λ ℓ ( ¯ Π) ≥ 2 λ n ( ¯ Π) , then we rely on Theorem B.3 and replace A by K λ . We have the following observation: For any t &gt; 2 , we have

<!-- formula-not-decoded -->

Therefore for all t &gt; 2 , we have,

<!-- formula-not-decoded -->

Let ˆ m t = ∑ n i =1 ˆ w ti k ( · , x i ) where ˆ w ti denote i th coordinate of ˆ w t and m n = ∑ n i =1 w ∗ i k ( · , x i ) where w ⋆ = K -1 λ y . Then we have,

<!-- formula-not-decoded -->

On the other hand if λ ℓ ( ¯ Π) &lt; 2 λ n ( ¯ Π) , then we simply use the SAP analysis and get

<!-- formula-not-decoded -->

where the last inequality can be obtained using a recursive argument similar to Theorem B.8 by plugging in linear convergence guarantees for SAP (using Lemma B.4 and Lemma B.5 with ℓ = n ). We get,

<!-- formula-not-decoded -->

Combining the results for both scenarios: λ ℓ ( ¯ Π) ≥ 2 λ n ( ¯ Π) or λ ℓ ( ¯ Π) &lt; 2 λ n ( ¯ Π) , we conclude the proof.

Time complexity analysis We now use the above guarantee to provide the time complexity analysis for estimating the GP posterior mean. The following result immediately implies Corollary 3.3.

Corollary B.10. Suppose that the matrix K exhibits polynomial spectral decay, i.e., λ i ( K ) = Θ( i -β ) for some β &gt; 1 . Then for any ℓ ∈ { 1 , . . . , n } , λ = O (1) and ϵ ∈ (0 , 1) , choosing b = 2 ℓ we can find ˆ m that with probability at least 0 . 99 satisfies ∥ proj ℓ ( ˆ m ) -proj ℓ ( m n ) ∥ 2 H ≤ ϵ ∥ proj ℓ ( m n ) ∥ 2 H in

<!-- formula-not-decoded -->

Proof. As y = f ( X ) + g , where g ∼ N (0 , λI ) and f ( X ) ∼ N (0 , K ) , we have y ∼ N (0 , K λ ) . This implies E [ yy T ] = K λ . Therefore, E ∥ K 1 / 2 λ w ⋆ ∥ 2 = n and E ∥ Q ℓ K 1 / 2 λ w ⋆ ∥ 2 = ℓ . So, using standard Gaussian concentration of measure, it follows that with probability 0 . 999 , ∥ K 1 / 2 λ w ⋆ ∥ 2 ≤ O (1) n ℓ ∥ K 1 / 2 λ w ⋆ ∥ 2 Q ℓ . Furthermore,

<!-- formula-not-decoded -->

We get the following relation:

<!-- formula-not-decoded -->

Now let b &gt; ℓ + ∑ i&gt;b λ i / ( λ ℓ + λ ) . We have ϕ ( b, ℓ ) ≤ 1 b ∑ i&gt;b λ i λ ℓ + λ + nλ b ( λ ℓ + λ ) ≤ 1 + nλ b ( λ ℓ + λ ) . We consider following two cases:

Case 1: nλ b ( λ ℓ + λ ) ≤ 1 . In this case we get ϕ ( b, ℓ ) &lt; 2 . After t = O ( n ℓϵ ) iterations we get E ∥ proj ℓ ( ˆ m t ) -proj ℓ ( m n ) ∥ 2 H ≤ ϵ ∥ proj ℓ ( m n ) ∥ 2 H .

Case 2: nλ b ( λ ℓ + λ ) &gt; 1 . We get λ &gt; bλ ℓ n . In this case we have ϕ ( b, n ) = 1 b ∑ i&gt;b λ i + λ λ n + λ &lt; n b + 1 b ∑ i&gt;b λ i λ &lt; 2 n b . Therefore, after t = O ( n log ( n (1+ λ/λ ℓ ) ℓϵ ) b ) iterations, we get E ∥ proj ℓ ( ˆ m t ) -proj ℓ ( m n ) ∥ 2 H ≤ ϵ ∥ proj ℓ ( m n ) ∥ 2 H .

On the other hand, in either case

<!-- formula-not-decoded -->

as we assumed b &gt; ∑ i&gt;b λ i λ ℓ + λ . Furthermore, for given spectral decay for any β &gt; 1 , we have ℓ + ∑ i&gt;b λ i λ ℓ &lt; 2 ℓ . This implies for b = 2 ℓ we obtain ϵ accuracy result in t = O ( n ℓ min { log( n/ℓ ) ϵ , ( 1 + ℓ n λ ℓ + λ λ n + λ ) log( n/ℓϵ ) }) , where we used λ = O (1) to get log ( n ℓ (1 + λ/λ ℓ ) ) = O ( log( n ℓ ) ) . Then, after applying Markov's inequality, ˆ m = ˆ m t obtains the desired ϵ accuracy with probability 0.999. The total time complexity follows by combining the number of iterations with cost of approximate sampling from 2 b -DPP( A ) (Lemma B.2) and additional per iteration cost of O ( nℓ + ℓ 3 log 3 n ) . Note that while the sampling algorithm is not exact, taking the union bound with respect to the high probability guarantees in Lemma B.2 ensures that all of the samples are indistinguishable from true DPP with probability 0.999. Finally, union bounding over the three 0.999 probability events we have invoked concludes the proof.

Improved guarantee for very ill-conditioned problems Here, we show that when λ is very small (i.e., K λ is highly ill-conditioned), then we can obtain an even better time complexity in the first phase of the convergence, addressing the claim in Remark 3.4.

Corollary B.11. Suppose that the matrix K exhibits polynomial spectral decay, i.e., λ i ( K ) = Θ( i -β ) for some β &gt; 1 and λ &lt; 1 nb β -1 , then we can find ˆ m that with probability at least 0 . 99 satisfies ∥ proj ℓ ( ˆ m ) -proj ℓ ( m n ) ∥ 2 H ≤ ϵ ∥ proj ℓ ( m n ) ∥ 2 H in

<!-- formula-not-decoded -->

Proof. We reconsider case 1 from the proof of Corollary B.10. Using the given spectral decay profile for K and that λ &lt; 1 nb β -1 , we get ϕ ( b, ℓ ) = O ( b -β ℓ -β ) . Therefore after t = n ℓϵ b -β ℓ -β we obtain the ϵ approximation guarantee. The total runtime complexity becomes:

<!-- formula-not-decoded -->

## C Additional Algorithmic Details for ADASAP

We provide additional implementation details for ADASAP . Appendix C.1 describes how we distribute matrix-matrix products across rows and columns in the algorithms ColDistMatMat and RowDistMatMat . Appendix C.2 describes the practical implementation of the randomized Nyström approximation and provides pseudocode for RandNysAppx . Appendix C.3 describes how we compute

preconditioned smoothness constants and provides pseuodocode for GetStepsize . Appendix C.4 provides pseudocode for Nesterov acceleration in NestAcc . Appendix C.5 provides a scaling plot illustrating the speedups achieved by using multiple GPUs in ADASAP . Appendix C.6 investigates the impact of tail averaging on the performance of ADASAP .

All operations involving kernel matrices are performed using pykeops [Charlier et al., 2021], which allows us to avoid instantiating kernel matrices explicitly in memory. To see the full details of our implementation, we recommend the reader to view our codebase.

## C.1 Distributed matrix-matrix products

Here, we provide details for how we implement the distributed matrix-matrix products in ADASAP . ColDistMatMat (Algorithm 3) shows how we distribute the matrix-matrix product K B n W t in ADASAP and RowDistMatMat (Algorithm 4) shows how we distribute the calculation of the Nyström sketch in ADASAP . Our implementation of these algorithms uses torch.multiprocessing to spawn a CUDA context on each device (i.e., a worker) and uses pykeops to generate the column and row block oracles.

## Algorithm 3 ColDistMatMat

```
Require: Right-hand side matrix W ∈ R n × n rhs , row indices B , workers {W 1 , . . . , W N workers } Partition rows of W as { W 1 , . . . , W N workers } Send W i to W i Generate column block oracle K col W i using B ( K B n W ) i ←K col W i [ W i ] ▷ Compute column block products in parallel Aggregate K B n W ← ∑ N workers i =1 ( K B n W ) i return K B n W
```

## Algorithm 4 RowDistMatMat Require: Right-hand side matrix Ω ∈ R n × r , row indices B , workers {W 1 , . . . , W N workers } Send Ω to each W i Generate row block oracle K row W i using B ( K BB Ω) i ←K row W i [Ω] ▷ Compute row block products in parallel Aggregate K BB Ω ← [ ( K BB Ω) T 1 . . . ( K BB Ω) T N workers ] T return K BB Ω

## C.2 Randomized Nyström approximation

Here, we present a practical implementation of the Nyström approximation used in ADASAP (Algorithm 2) in RandNysAppx (Algorithm 5). The Randomized Nyström approximation of K BB with test matrix Ω ∈ R b × r [Tropp et al., 2017] is given by:

<!-- formula-not-decoded -->

The preceding formula is numerically unstable, so ADASAP uses RandNysAppx , which is based on Tropp et al. [2017, Algorithm 3]. eps ( x ) is defined as the positive distance between x and the next largest floating point number of the same precision as x . The resulting Nyström approximation ˆ M is given by USU T , where U ∈ R p × r is an orthogonal matrix that contains the approximate topr eigenvectors of M , and S ∈ R r contains the topr eigenvalues of M . The Nyström approximation is positive-semidefinite but may have eigenvalues that are equal to 0 . In our algorithms, this approximation is always used in conjunction with a regularizer to ensure positive definiteness.

The dominant cost in Algorithm 5 is computing the SVD of B at a cost of O ( pr 2 ) . This is the source of the O ( pr 2 ) cost in Phase III of ADASAP .

## Algorithm 5 RandNysAppx

```
Require: sketch M Ω ∈ R p × r , sketching matrix Ω ∈ R p × r , approximation rank r ≤ p ∆ ← eps(Ω T M Ω . dtype) · Tr(Ω T M Ω) ▷ Compute shift for stability C ← chol (Ω T M Ω+∆Ω T Ω) ▷ Cholesky decomposition: C T C = Ω T M Ω+∆Ω T Ω B ← Y C -1 ▷ Triangular solve [ U, Σ , ∼ ] ← svd ( B, 0) ▷ Thin SVD S ← max { 0 , diag( S 2 -∆ I ) } ▷ Compute eigs, and remove shift with element-wise max return U, S
```

## C.2.1 Applying the Nyström approximation to a vector

In our algorithms, we often perform computations of the form ( ˆ M + ρI ) -1 g = ( USU T + ρI ) -1 g . This computation can be performed in O ( rp ) time using the Woodbury formula [Higham, 2002]:

<!-- formula-not-decoded -->

We also use the randomized Nyström approximation to compute preconditioned smoothness constants in GetStepsize (Algorithm 6). This computation requires the calculation ( P + ρI ) -1 / 2 v for some v ∈ R p , which can also be performed in O ( pr ) time using the Woodbury formula:

<!-- formula-not-decoded -->

In single precision, (8) is unreliable for computing ( P + ρI ) -1 g . This instability arises due to roundoff error: the derivation of (8) assumes that ˆ U T ˆ U = I , but we have empirically observed that orthogonality does not hold in single precision. To improve stability, we compute a Cholesky decomposition LL T of ρS -1 + U T U , which takes O ( pr 2 ) time since we form U T U . Using the Woodbury formula and Cholesky factors,

<!-- formula-not-decoded -->

This computation can be performed in O ( pr ) time, since the O ( r 2 ) cost of triangular solves with L T and L is negligible compared to the O ( pr ) cost of multiplication with U T and U .

Unlike Eq. (8), we find using (9) in GetStepsize yields excellent performance, i.e., we do not need to perform any additional stabilization.

## C.3 Computing the stepsize

Here, we provide the details of the GetStepsize procedure in ADASAP (Algorithm 2) for automatically computing the stepsize. Our procedure is inspired by Rathore et al. [2025], who show that approximate SAP with the Nyström approximation converges when

<!-- formula-not-decoded -->

That is, the correct stepsize to use is the reciprocal of the 'preconditioned subspace smoothness constant'.

To compute λ 1 ( ( P B + ρI ) -1 / 2 ( K BB + λI )( P B + ρI ) -1 / 2 ) , GetStepsize uses randomized powering [Kuczy´ nski and Wo´ zniakowski, 1992]. This technique has been used in several previous works on preconditioned optimization to great effect [Frangella et al., 2024a,b, Rathore et al., 2025]. Given a symmetric matrix H , preconditioner P , and damping ρ , randomized powering computes

<!-- formula-not-decoded -->

using only matrix-vector products with the matrices H and ( P + ρI ) -1 / 2 . When P is calculated using RandNysAppx , GetStepsize can efficiently compute a matrix-vector product with ( P + ρI ) -1 / 2

Figure 4: Multi-GPU scaling of ADASAP on the taxi dataset. ADASAP obtains near-linear scaling with the number of GPUs.

<!-- image -->

using (9). In practice, we find that 10 iterations of randomized powering are sufficient for estimating the preconditioned smoothness constant. The pseudocode for randomized powering, based on the presentation in Martinsson and Tropp [2020], is shown in Algorithm 6. Since ADASAP only runs GetStepsize for 10 iterations, the total cost of the procedure is O ( b 2 ) , which makes it the source of the O ( b 2 ) cost of Phase III in Algorithm 2.

```
Algorithm 6 GetStepsize Require: symmetric matrix H , preconditioner P , damping ρ , maximum iterations N ← 10 v 0 ← randn( P. shape[0]) v 0 ← v 0 / ∥ v 0 ∥ 2 ▷ Normalize for i = 0 , 1 , . . . , N -1 do v i +1 ← ( P + ρI ) -1 / 2 v i v i +1 ← Hv i +1 v i +1 ← ( P + ρI ) -1 / 2 v i +1 v i +1 ← v i +1 / ∥ v i +1 ∥ 2 ▷ Normalize end for λ ← ( v N -1 ) T v N return 1 /λ
```

## C.4 Nesterov acceleration

We present pseudocode for Nesterov acceleration in NestAcc (Algorithm 7).

## Algorithm 7 NestAcc Require: α

```
iterates W t , V t , Z t , search direction D t , stepsize η B , acceleration parameters β , γ , W t +1 ← Z t -η B D t V t +1 ← βV t +(1 -β ) Z t -γη B D t Z t +1 ← αV t +(1 -α ) W t +1 return W t +1 , V t +1 , Z t +1
```

## C.5 Parallel scaling of ADASAP

Here we present Fig. 4, which shows the parallel scaling of ADASAP on the taxi dataset.

Figure 5: Performance of ADASAP with and without tail averaging. One 'data pass' corresponds to one pass through the kernel matrix. Tail averaging does not improve convergence by a substantial margin.

<!-- image -->

## C.6 Impact of tail averaging on performance

The theoretical results in Section 3 require tail averaging. Here we assess whether tail averaging leads to practical improvements in the convergence of sketch-and-project algorithms for kernel ridge regression. To do so, we run a synthetic experiment using an RBF kernel with n = 1000 samples.

Fig. 5 displays the relative errors over the topℓ subspace, which are computed using the expression

<!-- formula-not-decoded -->

When ℓ = 10 and the blocksize b is small, tail averaging slightly improves the convergence rate. However, when ℓ ∈ { 100 , 1000 } , tail averaging does not improve the convergence rate. In fact, as the blocksize increases, tail averaging results in slower convergence!

## D Additional Details for Experiments

Here we provide additional details for the experiments that are not provided in the main paper.

## D.1 Determining hyperparameters for regression

We use a zero-mean prior for all datasets. We train the kernel variance, likelihood variance, and lengthscale (we use a separate lengthscale for each dimension of X ) using the procedure of Lin et al. [2023], which we restate for completeness:

1. Select a centroid point from the training data X uniformly at random.
2. Select the 10,000 points in the training data that are closest to the centroid in Euclidean norm.
3. Find hyperparameters by maximizing the exact GP likelihood over this subset of training points.
4. Repeat the previous three steps for 10 centroids and average the resulting hyperparameters.

## D.2 Optimizer hyperparameters

We present the hyperparameters for ADASAP , ADASAP-I , PCG, and SDD that were not described in the main paper.

For GP inference on large-scale datasets, we use blocksize b = n/ 100 in ADASAP , ADASAP-I , and SDD; blocksize b = n/ 2 , 000 for transporation data analysis, and b = n/ 5 for Bayesian optimization,

We set the rank r = 100 for both ADASAP and PCG.

Similar to Lin et al. [2024], we set the stepsize in SDD to be one of { 1 /n, 10 /n, 100 /n } (this grid corresponds to SDD-1, SDD-10, and SDD-100), the momentum to 0 . 9 , and the averaging parameter to 100 /T max .

## D.3 Additional details for GP inference experiments

song and houseelec are from the UCI repository, yolanda and acsincome are from OpenML, and benzene and malonaledehyde are from sGDML [Chmiela et al., 2017]. We select the kernel function k for each dataset based on previous work [Lin et al., 2023, Epperly et al., 2024b, Rathore et al., 2025].

We standardize both the features and targets for each dataset. For fairness, we run all methods for an equal amount of passes through each dataset: we use 50 passes for yolanda, song, benzene, and malonaldehyde and 20 passes for acsincome and houseelec.

We use pathwise conditioning with 2,048 random features to (approximately) sample from the GP posterior.

## D.4 Additional details for transporation data analysis

We standardize both the features and targets and use a RBF kernel. Due to computational constraints, we use a single 99%-1% train-test split, and run each method for a single pass through the dataset.

## D.5 Additional details for Bayesian optimization

Our implementation of Bayesian optimization largely mirrors that of Lin et al. [2023]. We only present the high-level details here, and refer the reader to Lin et al. [2023] for the fine details of the implementation.

We draw the target functions f : [0 , 1] 8 → R from a zero-mean GP prior with Matérn-3/2 kernel using 5,000 random features. At each iteration, we choose the acquisition points using parallel Thompson sampling [Hernández-Lobato et al., 2017]. As part of this process, we use the multi-start gradient optimization maximization strategy given in Lin et al. [2023]. At each step, we acquire 1,000 new points, which we use to evaluate the objective function. Concretely, if x new is an acquired point, we compute y new = f ( x new ) + ϵ , where ϵ ∼ N (0 , 10 -6 ) . We then add ( x new , y new ) to the training data for the next step of optimization. In our experiments, we initialize all methods with a dataset consisting of 250,000 observations sampled uniformly at random from [0 , 1] 8 .

## D.6 Additional timing plots for Section 5.1

Here we present timing plots for the datasets used in Section 5.1 that were not shown in the main paper. Fig. 6 shows that all the methods (except PCG) perform similarly on both test RMSE and test mean NLL. However, Fig. 7 shows that ADASAP and PCG attain a much lower train RMSE than the competitors. This suggests that the similar performance of the methods on test RMSE and test

Figure 6: Performance of ADASAP and competitors on RMSE and mean NLL, as a function of time, for benzene, malonaldehyde, and houseelec. The solid curve indicates mean performance over random splits of the data; the shaded regions indicate the range between the worst and best performance over random splits of the data. ADASAP performs similar to the competition.

<!-- image -->

Figure 7: Performance of ADASAP and competitors on train RMSE, as a function of time, for yolanda and song. The solid curve indicates mean performance over random splits of the data; the shaded regions indicate the range between the worst and best performance over random splits of the data.

<!-- image -->

mean NLL is not due to differences in optimization, but rather, it is because the datasets are not well-modeled by Gaussian processes.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The theoretical and empirical claims made in the abstract and introduction are justified in the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations of our work. In particular, we describe the key limitation of our work, which is that we do not show a strong theoretical guarantee for ADASAP due to its differences with SAP .

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

Justification: We clearly state all assumptions in our theorem statements, and a proof of each claim is provided within the appendix.

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

Justification: Yes, we include all the code to reproduce our experiments, and provide the details for the experiments within the paper itself.

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

Justification: We have provided all the code for the experiments on GitHub, and linked to the code in the main paper. We also provide instructions in our GitHub repository on how to download the data.

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

Justification: We provide all of the essential details, such as data splits, hyperparameters, and optimizers within the main paper and appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Our experiments for GP inference and Bayesian optimization use five random splits to ensure that the improvements attained by ADASAP are not just due to randomness. We also show the worst and best performance of each method in the timing plots, which demonstrates that ADASAP is more robust to the random seed compared to the competitors.

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

Justification: We clearly state the hardware and number of iterations that we use to run each algorithm.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: After reviewing the NeurIPS Code of Ethics, we believe that our research conforms with this code.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The work performed in this paper does not have a clear societal impact.

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

Justification: We do not believe the datasets used in this paper require safeguards for use.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Yes, we properly credit the authors of data and code that we use in our experiments.

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

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper involves neither crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We do not use LLMs for any of the core methods in this research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.