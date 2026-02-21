16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

## Multi-Marginal Schr¨ odinger Bridge Matching

## Anonymous Author(s)

Affiliation Address email

## Abstract

Understanding the continuous evolution of populations from discrete temporal snapshots is a critical research challenge, particularly in fields like developmental biology and systems medicine where longitudinal tracking of individual entities is often impossible. Such trajectory inference is vital for unraveling the mechanisms of dynamic processes. While Schr¨ odinger Bridge (SB) offer a potent framework, their traditional application to pairwise time points can be insufficient for systems defined by multiple intermediate snapshots. This paper introduces Multi-Marginal Schr¨ odinger Bridge Matching (MSBM), a novel algorithm specifically designed for the multi-marginal SB problem. MSBM extends iterative Markovian fitting (IMF) to effectively handle multiple marginal constraints. This technique ensures robust enforcement of all intermediate marginals while preserving the continuity of the learned global dynamics across the entire trajectory. Empirical validations on synthetic data and real-world single-cell RNA sequencing datasets demonstrate the competitive or superior performance of MSBM in capturing complex trajectories and respecting intermediate distributions, all with notable computational efficiency.

## 1 Introduction

Understanding the continuous evolution of populations from discrete temporal snapshots represents a significant challenge in various scientific disciplines, particularly in fields like developmental biology [7, 42] and systems medicine [29] where tracking individual entities longitudinally is often unfeasible. The ability to infer trajectories from such snapshot data is crucial for elucidating the underlying mechanisms of dynamic processes. The Schr¨ odinger Bridge (SB) problem, originally rooted in statistical mechanics [43], has garnered substantial interest in machine learning as an entropy-regularized, continuous-time formulation of optimal transport [20, 30]. It seeks to identify the most probable evolutionary path between prescribed initial and terminal distributions, and has been successfully employed in generative modeling [3, 4, 9, 26, 27, 37, 38, 45, 49].

However, many real-world scenarios present observations or constraints at multiple time points, not just at the beginning and end of a process. For instance, in single-cell RNA sequencing (scRNA-seq) experiments, which are pivotal for studying complex biological processes like cell differentiation, cells are typically destroyed upon measurement [6, 17, 28]. This destructive nature makes it impossible to track individual cells over time, thus necessitating the inference of developmental trajectories from population-level snapshots collected at several intermediate stages. Similarly, meteorological systems may have partial observations across various times [11, 32]. Such situations necessitate a multi-marginal generalization of the SB problem (mSBP), where the path measure must align with prescribed marginal distributions at multiple intermediate time points. While the traditional SB framework offers a powerful approach, its standard application to pairwise time points can prove insufficient for systems characterized by multiple intermediate snapshots. Although more specialized methods for mSBP have recently been developed [8, 18, 44], the direct application of some multi-marginal approaches can lead to error accumulation if not carefully managed, particularly

when learned controls are even slightly inaccurate. These challenges highlight the need for robust 39 and scalable solutions for the mSBP that can effectively integrate information across all observed 40 time points. 41

This paper introduces Multi-Marginal Schr¨ odinger Bridge Matching (MSBM), a novel algorithm 42 specifically developed to address the multi-marginal SB problem by building upon and extending the 43 Iterative Markovian Fitting (IMF) algoritmhs [36, 45]. MSBM is designed to effectively manage mul44 tiple marginal constraints by constructing local SBs on each interval and seamlessly integrating them. 45 This local construction strategy, underpinned by a shared global parametrization of control functions, 46 ensures the robust enforcement of all intermediate marginal distributions while crucially preserving 47 the continuity of the learned global dynamics across the entire trajectory. Empirical validations 48 conducted on synthetic datasets as well as real-world single-cell RNA sequencing data demonstrate 49 that MSBM achieves competitive or superior performance in capturing complex trajectories and 50 accurately respecting intermediate distributions, all while exhibiting notable computational efficiency. 51 Our work aims to provide a robust and scalable computational method for these multi-marginal 52 settings, addressing the critical need for consistent and tractable dynamic inference when data is 53 available as snapshots at multiple time points. 54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

We summarize our contributions as follows:

- We extend the theoretical and algorithmic foundations of SBs, including the IMF iteration and optimal control perspectives, to the challenging multi-marginal setting.
- We introduce an efficient modeling approach for trajectory inference, that constructs and smoothly integrates local SBs across sub-intervals, inherently allows for parallelized training, leading to significant speed-ups.
- Through comprehensive experiments on both synthetic and real-world single-cell RNA sequencing data, we demonstrate that MSBM accurately models complex population dynamics and outperforms state-of-the-art methods in both trajectory fidelity and computational speed.

Notation. Let P [0 ,T ] denote the space of continuous functions taking values in R d on the interval [0 , T ] . We use an uppercase letter P ∈ P [0 ,T ] to represent a path measure. For a path measure P ∈ P [0 ,T ] , the marginal distribution at discrete time points T = { t 0 , . . . , t k } , where 0 = t 0 &lt; t 1 &lt; · · · &lt; t k = T is denoted by P T ∈ P T , where we define P T as the set of measures P over R d ×|T | . Additionally, the conditional distribution of P , given T , is denoted by P |T ∈ P [0 ,T ] . Moreover, a path measure P can be defined as mixture. For any Borel measurable set A ∈ B (Ω) , P can be defined by P ( A ) = ∫ R d ×|T | P |T ( A | x T ) d P T ( x T ) , where P ∈ P 0 ,T and P ∈ P T , and we use the shorthand x T := ( x 1 , · · · , x k ) and [0 : k ] := { 0 , 1 , · · · , k } . The Kullback-Leibler (KL) divergence between two probability measures µ and ν on space X is defined as D KL ( µ | ν ) = ∫ X log dµ dν ( X ) dµ ( X ) when µ is absolutely continuous with respect to ν ( µ ≪ ν ), and D KL ( µ | ν ) = + ∞ otherwise. We will often refer to probability measures on R d and their Lebesgue densities interchangeably, under the standard assumption of absolute continuity. Finally, for a function V : [0 , T ] × R d → R , we define the gradient and laplcaian operators with respect to x ∈ R d as ∇V and ∆ V , respectively, and its partial derivative with respect to time t ∈ [0 , T ] as ∂ t V .

## 2 Preliminaries

## 2.1 Schr¨ odinger Bridge Matching (SBM)

The Schr¨ odinger Bridge problem (SBP) [16, 43] is a stochastic optimal transport problem [30] that seeks the optimal transport plan for endpoint marginals ρ 0 and ρ T . In this paper, we focus on the dynamical representation, where a reference distribution Q ∈ P [0 ,T ] is induced by the SDEs:

<!-- formula-not-decoded -->

where f t : R d → R d is a drift, σ ∈ R is a diffusion, and W t ∈ R d is a standard Wiener process. 83 With the base reference path measure Q , the dynamic representation of the SB [20, 35, 39] is: 84

<!-- formula-not-decoded -->

Recent advancements in dynamical optimal transport [37, 45] have introduced a novel numerical 85 methodology for solving SBP . This approach reframes SBP by decomposing its dynamical constraints 86 into the time-evolving marginal distributions P t for all t ∈ [0 , T ] and the joint coupling P 0 ,T . This 87 optimization relies on IMF [45], a technique that iteratively refines the path measure P ∈ P [0 ,T ] . 88 IMF alternates between two projection called Markovian and Reciprocal projections to preserve the 89 correct endpoint marginals ( ρ 0 , ρ T ) throughout the optimization. 90

91

92

93

94

95

96

97

98

99

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

Reciprocal Projection R . For a given reference measure Q from (1), and a path measure P with marginals specified at end points T = { 0 , T } the reciprocal projection is defined as:

<!-- formula-not-decoded -->

This projection constructs a new path measure by taking the endpoint coupling P 0 ,T from P and forming a mixture of bridge process using Q conditioned on these end points. Sampling from Π := R ( P , T ) involves drawing end points samples ( X 0 , X T ) ∼ P 0 ,T and then generating a path X T t between them using conditional reference measure Q | 0 ,T which induced by following SDEs, for any ( x 0 , x T ) :

<!-- formula-not-decoded -->

If Q | 0 ,T has tractable bridge formulation, for example, when Q is chosen as a Brownian motion i . e ., d X t = σd W t , sampling the path at time t given the endpoints can be performed as:

<!-- formula-not-decoded -->

Markov Projection M . Although the reciprocal projection R in (2) preserves end point marginals ( ρ 0 , ρ T ) , its sampling process in (4) requires both ( X 0 , X T ) , making it non-Markovian and thus ill-suited for generative modeling aimed at sampling from ρ T without knowing X T . The Markov projection M resolves this by projecting Π := R ( P , T ) into a family of Markov process while ensuring P ⋆ = Π t for all t ∈ [0 , T ] . Again, when Q is chosen as a Brownian motion i . e ., d X t = σd W t , the Markov projection of Π , P ⋆ = M (Π , T ) , is induced by following SDEs:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Intuitively, the term E Q T | t [ X T | X t = x ] can be understood as a prediction of the target state X ⋆ t . Flow matching [23] of Bridge matching [37] tackles the approximation X ⋆ T ≈ E Q T | t [ X T | X t = x ] by learning a drift function. This learned drift guides the evolution of X ⋆ t such that its terminal state aligns with the target, often by regressing the drift agains a target drift derived from samples of ( X 0 , X T ) under the reference conditional bridge measure Q | 0 ,T .

Building upon the projections R and M , Schr¨ odinger Bridge Matching (SBM) methods [37, 45] refines the path measure through an alternating iteraive procedure:

<!-- formula-not-decoded -->

Initialized with P (0) = P (0) T Q | 0 ,T , utilizing P (0) T is independent coupling of ρ 0 and ρ T along with the reference conditional bridge measure Q |T . Please refer to [37, 45] for more details.

## 3 Multi-Marginal Iterative Markovian Fitting

Dynamic SB methods, as discussed in Section 2, have traditionally focused on problems defined 116 by two endpoint marginal distributions, ( ρ 0 , ρ T ) . However, in real-world applications, particularly 117 in fields like developmental biology (e.g., scRNA-seq studies of cellular differentiation), systems 118 are often observed through snapshots at multiple intermediate time points, not just at the beginning 119 and end of a process. This prevalence of multi-stage data highlights a critical limitation of standard 120 SB approaches. While the theoretical extension of SB methods to handle multiple marginals has 121 been explored [1, 31], the development of robust and scalable computational methods for these 122 multi-marginal settings has lagged. Recently, methods with IPF-type objectives have been derived 123 for multi-marginal cases [8, 44]. However, challenges persist in ensuring global dynamic consistency 124 across all intervals, maintaining computational tractability as the number of marginals increases. 125

126

127

128

129

130

131

132

133

134

135

136

137

138

139

140

141

142

143

144

145

146

147

148

149

150

151

152

153

154

155

156

157

158

159

160

161

In this section, we extends the SBM framework -conventionally applied to problems with two endpoint marginals ( ρ 0 , ρ T ) and foundational to IMF methods -to handle cases involving k + 1 multiple snapshots ( ρ 0 , ρ t 1 , · · · , ρ T ) on discrete time stamps T = { t 0 , t 1 , · · · , t k } where 0 = t 0 &lt; t 1 &lt; · · · &lt; t k = T 1 . Similar to SBP , the dynamic multi-marginal Schr¨ odinger Bridge problem can be formally defined as [10] the entropy minimization problem:

<!-- formula-not-decoded -->

To find a most probable path P mSBP , the solution of mSBP under multiple constraints, we will generalize the principles of SBM in Section 2.1 to the multi-marginal cases in Section 3.1. The extension of dynamic SB optimality [20, 35] and the associated stochastic optimal control problem [39] to multimarginal settings is presented in Appendix A.

## 3.1 Multi-Marginal Projection operators

To develop multi-marginal extension of SBM, we investigate how the IMF framework can be adapted to scenarios with multiple snapshots ( i . e ., where the set of time points T has cardinality |T | &gt; 2 ). This adaptation necessitates extending the fundamental building blocks of SBM-specifically, the reciprocal projection R and the Markov projection M -to handle multiple marginal constraints.

Multi-Marginal Reciprocal Projection R mm . First, we state and prove a proposition that characterizes the reciprocal structure of conditional path measures. In particular, we focus on a mixture of bridges Π = Π T Q |T ∈ P [0 ,T ] constrained by the marginals at multiple timestamps in T .

Proposition 1 (Reciprocal Property) . For any x T := ( x 0 , x t 1 , · · · , x T ) ∈ R d × ( k +1) and t ∈ [ t i -1 , t i ) , the marginal distribution of Q |T ( ·| x T ) at t satisfies:

<!-- formula-not-decoded -->

Therefore, for any P ∈ P [0 ,T ] the reciprocal projection R mm ( P , T ) admits the following factorization:

<!-- formula-not-decoded -->

A key implication of the reciprocal property, detailed in Proposition 1, is that a mixture of diffusion bridges constrained on T factorizes into independent segments over successive time intervals. This factorization simplifies the analysis and simulation of the overall path measure. Since each segment can then be treated as a standard conditional bridge process as in (3), closed-form sampling, such as in (4), can be applied independently in parallel to each subinterval { t i -1 , t i } i ∈ [1: k ] . This tractability is essential for developing an efficient multi-marginal SBM algorithm.

Multi-Marginal Markov Projection M mm . With the reciprocal property and factorization in (9), we show that the Markov projection on multi-marginal case can be constructed by similar fashion.

Proposition 2 (Multi-Marginal Markovian Projection) . Let Π ∈ P [0 ,T ] admit factorzation in (9) . The multi-marginal Markov projection of Π , P ⋆ := M mm (Π , T ) ∈ P [0 ,T ] , is associated with the SDE:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, v ⋆ satisfies the Fokker-Planck equation (FPE) [40]:

<!-- formula-not-decoded -->

where p t is marginal density of Π t . In other words, P ⋆ t = Π t for all t ∈ [0 , T ] . d

As established in Proposition 2, constructing a global diffusion process via (10) with the optimal control v ⋆ (11)) yields a multi-marginal Markov projection X ⋆ [0 ,T ] that is continuous over the entire time interval [0 , T ] . The continuity arises because the local Markov projections, X ⋆ [ t i -1 ,t i ] , on each sub-interval are derived from factorized conditional bridge Q | t i -1 ,t i in (9). These bridges are

1 Our framework accommodates arbitrary time intervals between successive time stamps.

162

163

164

165

166

167

168

169

170

171

172

173

174

175

176

177

178

179

180

181

182

183

184

anchored by identical marginal distributions at there shared boundaries; for instance, both X ⋆ [ t i -1 ,t i ] and X ⋆ [ t i ,t i +1 ] is guaranteed to match the marginal distribution ρ t i at time t i . Consequently, these local diffusion processes connect seamlessly at adjacent timestamps, resulting in a smooth and well-defined path for X ⋆ [0 ,T ] . The well-defined nature of the global path, in conjunction with the projections R mm and M mm , is fundamental to successfully applying the SBM framework to the mSBP . Finally, the uniquness condition for standard SB [45, Proposition 5] can also be extended to multi-marginal case.

Proposition 3 (Uniqueness) . Let P ⋆ be a Markov measure which is reciprocal class of Q satisfying P ⋆ t = ρ t for all t ∈ T . Then, P ⋆ is unique solution P mSBP of the mSBP .

Building on the projection operators R mm , M mm with the uniquness result of Proposition 3, we can apply the iterative algorithm used in SBM algorithm [45, Algorithm 1] to the multi-marginal setting:

<!-- formula-not-decoded -->

The convergence guarantees proved for the iteration apply equally well to the multi-marginal case.

Proposition 4 (Convergence) . P ( n ) = P mSBP of mSBP as n ↑ ∞ with iterative procedure in (13) .

## 3.2 Practical Implementation.

In practice, at each iteration n of (13) we approximate the optimal control v ⋆ from (11) by a neural network v θ . By Girsanov theorem, θ are chosen to minimize the following training objective function:

<!-- formula-not-decoded -->

where β T ( t ) = min u { u &gt; t | t ∈ T } ∈ [0 , T ] is the most recent time point in T after time t . With this notation, the SBM can be generalized to the case of multi-marginal constraints. For example, when T = { 0 , T } then (14) reduces to the objective function described in [45].

The learned Markov control v θ ⋆ ( t, x t ) then ensures P θ ⋆ t = Π t for all t ∈ [0 , T ] . Moreover, prior SBM algorithms interleave forward and backward-time Markov projections to re-anchor the terminal distribution and prevent bias between P ( n ) T and Π T accumulate for each n ∈ N . In the multi-marginal setting, we again build the backward-time Markov projection as in Proposition 2 by gluing the local bridge reversals, so that P ⋆ is governed by both SDEs (10) and the corresponding backward dynamics:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the backward optimal control u ⋆ in (16) can be approximated with neural network u ϕ where ϕ 185 is chosen to minimize the following training objective function with γ T ( t ) = max u { u &lt; t | t ∈ T } : 186

<!-- formula-not-decoded -->

187

188

189

190

191

192

193

194

195

196

197

198

199

## 4 Multi-Marginal Schr¨ odinger Bridge Matching

A na¨ ıve extension of the standard SBM using, multi-marginal projections R mm and M mm in Sec 3, encounters significant limitations not present in the traditional two-endpoint setting. In such an extension, each iteration typically enforces marginal constraints only at the global endpoints ( ρ , ρ ) .

The multi-marginal coupling Π T at each iteration n of (13) is then derived by propagating the projected dynamics in (10) or (15) solely from these end points ρ or ρ , respectively.

0 T ( n ) 0 T

This approach leads to critical issues specific to the multi-marginal context. Firstly, if the learned controls, such as v ⋆ (forward) or u ⋆ (backward), are even slightly inaccurate, significant biases can arise between the inferred intermediate marginals (Π ( n ) t 1 , · · · Π ( n ) t k -1 ) and the target marginals ( ρ t 1 , · · · , ρ t k -1 ) . Secondly, these discrepancies tend to accumulate iteratively. This accumulation is exacerbated because, beyond an initialization Π (0) = P (0) T Q |T with P (0) T , independent joint coupling of { ρ t } t ∈T , where the joint distribution might be informed by all prescribed data distributions, the subsequent self-refinement process for the dynamics often does not directly incorporate the

200

201

202

203

204

205

206

207

208

209

210

211

212

213

214

215

216

217

218

219

220

221

222

223

224

225

226

227

## Algorithm 1 Training of MSBM

| 1:     | Input: Snapshots { ρ t } t ∈T , bridge Q &#124;T , N ∈ N                                                       |
|--------|----------------------------------------------------------------------------------------------------------------|
| 2:     | Let { P (0) T i } i ∈ [1: k ] joint coupling of { ρ t ∈T i } i ∈ [1: k ] .                                     |
| 3:     | for n ∈ { 0 ,...,N - 1 } do                                                                                    |
| 4:     | for i ∈ { 1 , . . . ,k - 1 } do in parallel                                                                    |
| 5:     | Let Π (2 n ) T i = P (2 n ) T i                                                                                |
| 6:     | Estimate L ( ϕ, T i , Π (2 n ) T i , Q &#124;T i )                                                             |
| 7:     | Estimate ˜ L ( ϕ ) = ∑ k i =1 L ( ϕ, T i , Π (2 n ) T i , Q &#124;T i )                                        |
| 8:     | u ϕ ⋆ = argmin ϕ ∑ k i =1 ˜ L ( ϕ )                                                                            |
| 9: 10: | Simulate local backward SBs { P i, (2 n +1) } i ∈ [1: k ] for i ∈ { 1 , . . . ,k - 1 } do in parallel (2 n +1) |
| 11:    | Let Π (2 n +1) T i = P T i                                                                                     |
| 12:    | Estimate L ( θ, T i , Π (2 n +1) T i , Q &#124;T i )                                                           |
| 13:    | Estimate ˜ L ( θ ) = ∑ k i =1 L ( θ, T i , Π (2 n +1) T i , Q &#124;T i )                                      |
| 14:    | v θ ⋆ = argmin θ ∑ k i =1 L ( θ, T i , Π (2 n +1) T i )                                                        |
| 15:    | Simulate local forward SBs { P i, (2 n +2) }                                                                   |
| 16:    | [ t i - 1 ,t i ] end for                                                                                       |
| 17:    | Output: v ⋆ θ ,u ⋆ ϕ                                                                                           |

## Algorithm 2 Simulation of MSBM (forward)

Input:

Initial ρ 0 , learned control v θ ⋆

Sample X 0 ∼ ρ 0

Simulate forward SDE over

[0

, T

]

d

X

⋆

t

= [ f t + σv θ ⋆ ( t, X ⋆ t )] dt + σd W t ,

Output:

Trajectory X ⋆ [0 ,T ]

Figure 1: (Left) The na¨ ıve extension fails to model intermediate states due to the accumulation of errors. (Right) In contrast, MSBM successfully models the ground truth data.

<!-- image -->

intermediate data distributions ( ρ t 1 , · · · , ρ t k -1 ) into its training objective except ρ 0 and ρ T . Without explicit targets for the intermediate marginals guiding each iteration, the inferred paths between ρ 0 and ρ T can 'collapse' or drift away from the desired states. Consequently, precisely satisfying all intermediate constraints becomes increasingly challenging as iterations proceed.

To address this issue of error accumulation and ensure all marginal constraints { ρ t } t ∈T are satisfied, we propose a method that involves constructing local SBs on each interval [ t i -1 , t i ] and then seamlessly gluing them together. Instead of propagating dynamics from the global endpoints ρ 0 and ρ T alone, our approach first establishes local SBs for each segment. The resulting local couplings are then systematically integrated to satisfy all specified marginal distributions { ρ t } t ∈T across the entire time interval [0 , T ] . This local construction strategy helps prevent the compounding of errors at intermediate time points while still aiming to achieve the overall multi-marginal SB solution, P mSBP . The theoretical basis is provided by the following result.

Corollary 5 (Multi-Marginal Schr¨ odinger Bridge) . Assume a sequence of controls { v i , u i } i ∈ [1: k ] , where each v i , u i induced local SBs P i of SBP over local interval [ t i -1 , t i ] with distributions ( ρ t i -1 , ρ t i ) in a forward and backward direction, respectively. If lim t ↑ t i v i ( t, x ) = v i +1 ( t, x ) and lim t ↓ t i -1 u i ( t, x ) = u i -1 ( t, x ) for all i ∈ [1 : k ] , then P mSBP of mSBP induced by following SDEs:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Building upon Corollary 5, we introduce our Multi-Marginal Schr¨ odinger Bridge Matching (MSBM) method to solve the mSBP . A cornerstone of MSBM is divide the global mSBP into local SBP s while maintaining the continuity of the composite drift functions v ⋆ and u ⋆ in (18c) across adjacent intervals, which guarantees a globally continuous diffusion process inducing P mSBP . Furthermore, by explicitly constraining each local SBs, P i , on its corresponding marginals ( ρ t i -1 , ρ t i ) , MSBM is designed to mitigate the accumulation of bias at intermediate marginals, as shown in Figure 1.

̸

A key challenge of the MSBM is rigorously satisfying the continuity conditions at the boundaries of local controls: lim t ↑ t i v i ( t, x ) = v i +1 ( t, x ) and lim t ↓ t i -1 u i ( t, x ) = u i -1 ( t, x ) for all i ∈ [1 : k ] . If these conditions are not met, discontinuities or 'kinks' can arise at the intermediate time steps. Such kinks would imply that the overall path measure P ⋆ = M mm ( P ⋆ , T ) . This would, in turn, hinder the optimlaity for mSBP , because, following Proposition 3, the desired continuous Markov process is a fixed point of both R mm and Markov projections M mm under multiple time points T :

<!-- formula-not-decoded -->

To construct local SBs such that the continuity requirements for forming a valid global solution are 228 met, thereby preventing the aforementioned kinks and ensuring (19), our MSBM introduces a shared 229 global parametrization v θ , u ϕ for its respective local controls { v i , u i } i ∈ [1: k ] for each sub-interval, 230 where each local controls are parallel updated with following aggregate objective function: 231

<!-- formula-not-decoded -->

where T i = { t i -1 , t i } define sub-intervals with local coupling Π T i for end-points marginals in interval [ t i -1 , t i ] and L is defined in (14) and (17) for forward and backward direction, respectively.

The MSBM training procedure, summarized in Algorithm 1, adapts the standard IMF algorithm presented in [45, Algorithm 1]. A key distinction in our MSBM approach is the parallel application of the IMF procedure to each local time interval, utilizing globally shared forward v θ and backward u ϕ across all local intervals. This parallel processing across sub-intervals contributes to a significant reduction in overall training time.

## 5 Related Work

The solution of SBP often utilize Iterative Proportional Fitting (IPF) [19], with modern adaptations learning SDE drifts for two-marginal settings [4, 9, 13, 49]. A distinct iterative approach, IMF, as featured in [37, 45], offers improved stability by alternating projections onto different classes of path measures. Moreover, emerging research also explores non-iterative algorithm [12, 38]. These methodologies primarily concentrate on the SB problem itself, iteratively refining path measures or directly computing the bridge measure. Moreover, the SB algorithm is studied under the assumption that the optimal coupling is given [27, 46]. While recent studies have extended foundational SB ideas to the multi-marginal setting of mSBP , research in this area remains relatively limited.

In multi-marginal setting, [8] extends the problem to phase space to encourage smoother trajectories and introduces a novel training methodology inspired by the Bregman iteration [5] to handle multiple marginal constraints. Relatedly, [44] presented an approach that, similar to our work, segments the problem across intervals; they learn piecewise SBs and use likelihood-based training to iteratively refine a global reference dynamic. While these methods are often IPF-based or focus on specific reference refinement strategies, our MSBM extends the previous IMF-type algorithm into multimarginal setting and effectively handles multiple constraints. We demonstrate that our MSBM framework offers substantial gains in training efficiency. This enhanced efficiency is primarily attributed to its direct multi-marginal formulation that adeptly manages multiple constraints, thereby circumventing the computationally intensive iterative refinements common in IPF-based methods

Paralleling these SB-centric developments, other significant lines of work model dynamic trajectories by directly learning potential functions or velocity fields, often drawing from optimal transport or continuous normalizing flows. For instance, [18, 24-26] extend SBs to incorporate potentials or mean-field interactions, connecting to stochastic optimal control and earlier mean-field game frameworks [22, 41]. The broader field of trajectory inference from snapshot data, crucial for applications like scRNA-seq, has seen methods like [48] using CNFs with dynamic OT, and [15] employing Neural ODEs on learned data manifolds. More recently, [33, 34] offer variational objectives to learn dynamics from marginal samples.

## 6 Experiments

232

233

234

235

236

237

238

239

240

241

242

243

244

245

246

247

248

249

250

251

252

253

254

255

256

257

258

259

260

261

262

263

264

265

266

In this section, we empirically demonstrate the effectiveness of our MSBM. Specifically, our goal 267 is to infer a dynamic model from datasets composed of samples from marginal distributions ρ t 268 observed at discrete time points. We evaluate MSBM on both synthetic datasets and real-world single269 cell RNA sequencing datasets, including human embryonic stem cells (hESC) [11] and embryoid 270 body (EB) [32]. To ensure consistency and fair comparison, our experiments follow the respective 271 experimental setups established by baseline methods. In particular, for the petal dataset, we adopt 272 the experimental setup from DMSB [8], and for the hESC dataset, we follow SBIRR [44]. For 273 the EB dataset, we perform evaluations on both 5-dim and 100-dim PCA-reduced data; here, we 274 follow the 100-dim experimental setup of DMSB and the 5-dim setup from NLSB [18]. Accordingly, 275

276

277

278

279

280

281

282

283

284

285

286

287

288

289

290

291

292

293

294

295

296

297

298

299

300

301

302

303

304

305

306

307

308

309

310

311

312

313

314

315

Figure 3: Comparison of generated population dynamics using MIOFlow, DMSB and MSBM on a 2-dim petal dataset. All trajectories are generated by simulating the dynamics from ρ t 0 .

<!-- image -->

we utilize evaluation metrics consistent with previous studies, including the Sliced-Wasserstein Distance (SWD)[2], Maximum Mean Discrepancy (MMD)[14], as well as the 1-Wasserstein ( W 1 ) and 2-Wasserstein ( W 2 ) distances. All experimental results reported are averaged mean value over three independent runs with different random seeds. We highlight the best-performing results in bold and the second-best results in blue. Further experimental details are provided in Appendix C.

## 6.1 Synthetic Data

Petal The petal dataset [15] serves as a simple yet complex challenge because it mimics the natural dynamics seen in processes such as cellular differentiation, which include phenomena like bifurcations and merges. We compare our MSBM with MIOFlow [15] and DMSB [8] in Figure 2. As shown in Figure 3, we observe that MSBM exhibits the most accurate and clearly defined trajectory, closely resembling the

Figure 2: Evaluation results of W 2 and MMD.

<!-- image -->

ground truth. Furthermore, Figure 2 demonstrates the evaluation results for the trajectories through W 2 and MMD distances, highlighting that MSBM consistently outperforms MIOFlow and DMSB.

## 6.2 Single-cell Sequencing Data

We evaluated our MSBM on real-world single-cell RNA sequencing data from two sources: 1 ) human embryonic stem cells (hESCs) [11] undergoing differentiation into definitive endoderm over a 4-day period, measured at 6 distinct time points ( t 0 :0 hours, t 1 :12 hours, t 2 :24 hours, t 3 :36 hours, t 4 :72 hours, and t 5 :96 hours); 2 ) embryoid body (EB) cells [32] differentiating into mesoderm, endoderm, neuroectoderm, and neural crest over 27 days, with samples collected at 5 time windows ( t 0 :0-3 days, t 1 :6-9 days, t 2 :12-15 days, t 3 :18-21 days, and t 4 :24-27 days). Following the experimental setup of baselines, we preprocessed these datasets using the pipeline outlined in [48], and the collected cells were projected into a lower-dimensional space using principal component analysis (PCA).

hESC To follow the experimental setup from SBIRR [44], we reduced the data to the first five principal components and excluded the final time point t 6 from our dataset, resulting in three training time points T = { t 0 , t 2 , t 4 } and two intermediate test points T test = { t 1 , t 3 } . Our objective was to train the dynamics based on the available marginals at the training points in T and interpolate the intermediate test marginals at T test , which were not observed during training. Table 1 demonstrates that our proposed MSBM method performs competitively, achieving lower W 2 distances.

Embryoid Body We validate our MSBM on both 5-dim and 100-dim PCA spaces. First, for the 5-dim experiment, we adopt the experimental setup from NLSB. Given 5 observation time points

Table 1: Performance on the 5dim PCA of hESC dataset. W 2 is compute between test ρ t i and generated ˆ ρ t i by simulating the dynamics from test ρ t 0 .

|                                | W 2 ↓          | W 2 ↓          | Runtime                   |
|--------------------------------|----------------|----------------|---------------------------|
| Methods                        | t 1            | t 3            | hours                     |
| TrajectoryNet † DMSB † SBIRR † | 1.30 1.10 1.08 | 1.93 1.51 1.33 | 10.19 15.54 0.36 (0.38) ∗ |
| MSBM (Ours)                    | 1.09           | 1.30           | 0.09                      |

T = { t 0 , t 1 , t 2 , t 3 , t 4 } , we divide the data using train/test splits ρ tr T /ρ te T , with the goal of predicting population-level dynamics from ρ tr t 0 . Similar to NLSB, we train the dynamics based on ρ tr T and

316

317

318

319

320

321

322

323

324

325

326

327

328

329

330

331

332

333

334

335

336

337

338

339

340

341

342

343

344

345

346

347

348

349

350

351

352

353

354

355

356

357

Table 3: Performance on the 100-dim PCA of EB dataset. MMD and SWD are computed between test ρ te t i and generated ˆ ρ t i by simulating the dynamics from test ρ te t 0 . Figure 4: Comparison of generated population dynamics using DMSB and MSBM on a 100-dim PCA of EB dataset. The plot displays the first two principal components as the x and y axes, respectively.

<!-- image -->

† result from [8].

evaluate the W 1 distance between ρ te t i and the generated ˆ ρ t i from previous test snapshot ρ te t i -1 . In Table 2, we find that MSBM outperforms several SB methods.

For the 100-dim experiment, we borrow the experimental setup from DMSB, where the goal is predict population dynamics given that observations are available for all time points T (denoted as Full in Table 3), or when one of the snapshot is left out (denoted as t i in Table 3, where snapshot ρ tr t i at t i is excluded during training). The high performance in this task represent the robustness of the model to accurately predict population dynamics. In Table 3, MSBM consistently yields performance improvements. Moreover, as shown in Figure 4, the trajectories and generated marginal distributions ˆ ρ T in PCA space further justifies the numerical result and highlights the variety and quality of the samples produced by MSBM.

Table 2: Performance on the 5-dim PCA of EB dataset. W 1 is computed between test ρ te t i and generated ˆ ρ t i by simulating the dynamics from previous test ρ te t i -1 .

Computational Efficiency For an fair comparison of training efficiency against recent multi-marginal SB al-

|                      | W 1 ↓   | W 1 ↓   | W 1 ↓   | W 1 ↓   | W 1 ↓   |
|----------------------|---------|---------|---------|---------|---------|
| Methods              | t 1     | t 2     | t 3     | t 4     | Mean    |
| Neural SDE † [21]    | 0 . 69  | 0 . 91  | 0 . 85  | 0 . 81  | 0 . 82  |
| TrajectoryNet † [48] | 0 . 73  | 1 . 06  | 0 . 90  | 1 . 01  | 0 . 93  |
| IPF (GP) † [49]      | 0 . 70  | 1 . 04  | 0 . 94  | 0 . 98  | 0 . 92  |
| IPF (NN) † [4]       | 0 . 73  | 0 . 89  | 0 . 84  | 0 . 83  | 0 . 82  |
| SB-FBSDE † [9]       | 0 . 56  | 0 . 80  | 1 . 00  | 1 . 00  | 0 . 84  |
| NLSB † [18]          | 0 . 68  | 0 . 84  | 0 . 81  | 0 . 79  | 0 . 78  |
| OT-CFM † [47]        | 0 . 78  | 0 . 76  | 0 . 77  | 0 . 75  | 0 . 77  |
| WLF-SB ‡ [34]        | 0 . 63  | 0 . 79  | 0 . 77  | 0 . 75  | 0 . 73  |
| MSBM (Ours)          | 0 . 64  | 0 . 73  | 0 . 72  | 0 . 73  | 0 . 71  |

† result from [18], ‡ result from [34]. gorithms, we benchmarked DMSB and SBIRR on the identical hardware configuration employed for MSBM (denoted by ∗ in Table 1). On the hESC dataset, MSBM achieved a runtime over 4× faster than SBIRR. Furthermore, on the petal and 100-dim PCA of EB dataset, MSBM significantly outperformed DSMB in training speed, with detailed results presented in Figure 5.

This enhanced computational efficiency primarily originates from core algorithmic differences. SBIRR, for example, utilizes maximum likelihood training, which requires extensive gradient computations and the storage of all intermediate paths. DMSB employs an IPF-type objective with Bregman Iteration [5]. In contrast, MSBM directly optimizes controls using an IMF-type objective, which not only eliminates the need to store intermediate states but also facilitates parallel computation across sub-intervals. This approach substantially promotes faster convergence of the algorithm.

## 7 Conclusion and Limitation

This paper revisits previously established frameworks for the SBP , extending them to the mSBP . Specifically, we introduce a computationally efficient framework for mSBP , termed MSBM, which builds upon existing SBM methods [37, 45]. MSBM is tailored for various trajectory inference problems where snapshots of data are available at multi-marginal time steps. Through the successful adaptation of the IMF algorithm to this multi-marginal setting, our approach significantly accelerates training processes while ensuring accurate dynamic modeling when compared to existing methods.

Despite these advantages, the performance degradation of MSBM is more pronounced than that of DMSB when a time point is omitted in Table 3. This may occur because the including velocity term could better accommodate unknown trajectory. Furthermore, the current MSBM framework is restricted to the case involving snapshot data samples, highlighting a need for enhancements to address problems with continuous potentials, such mean-field games [18, 24-26].

<!-- image -->

358

359

360

361

362

363

364

365

366

367

368

369

370

371

372

373

374

375

376

377

378

379

380

381

382

383

384

385

386

387

388

389

390

391

392

393

394

395

396

397

398

399

400

401

## References

- [1] Aymeric Baradat and Christian L´ eonard. Minimizing relative entropy of path measures under marginal constraints. arXiv preprint arXiv:2001.10920 , 2020.
- [2] Nicolas Bonneel, Julien Rabin, Gabriel Peyr´ e, and Hanspeter Pfister. Sliced and Radon Wasserstein barycenters of measures. Journal of Mathematical Imaging and Vision , 51:22-45, 2015.
- [3] Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, and Arnaud Doucet. Schrodinger bridge flow for unpaired data translation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [4] Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion Schr¨ odinger bridge with applications to score-based generative modeling. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems , 2021.
- [5] L.M. Bregman. The relaxation method of finding the common point of convex sets and its application to the solution of problems in convex programming. USSR Computational Mathematics and Mathematical Physics , 1967.
- [6] Jason D Buenrostro, Beijing Wu, Ulrike M Litzenburger, Dave Ruff, Michael L Gonzales, Michael P Snyder, Howard Y Chang, and William J Greenleaf. Single-cell chromatin accessibility reveals principles of regulatory variation. Nature , 523(7561):486-490, 2015.
- [7] Charlotte Bunne, Stefan G Stark, Gabriele Gut, Jacobo Sarabia Del Castillo, Mitch Levesque, Kjong-Van Lehmann, Lucas Pelkmans, Andreas Krause, and Gunnar R¨ atsch. Learning singlecell perturbation responses using neural optimal transport. Nature methods , 20(11):1759-1768, 2023.
- [8] Tianrong Chen, Guan-Horng Liu, Molei Tao, and Evangelos Theodorou. Deep momentum multi-marginal schr¨ odinger bridge. Advances in Neural Information Processing Systems , 36:57058-57086, 2023.
- [9] Tianrong Chen, Guan-Horng Liu, and Evangelos Theodorou. Likelihood training of schr¨ odinger bridge using forward-backward SDEs theory. In International Conference on Learning Representations , 2022.
- [10] Yongxin Chen, Giovanni Conforti, Tryphon T Georgiou, and Luigia Ripani. Multi-marginal schr¨ odinger bridges. In International Conference on Geometric Science of Information , pages 725-732. Springer, 2019.
- [11] Li-Fang Chu, Ning Leng, Jue Zhang, Zhonggang Hou, Daniel Mamott, David T Vereide, Jeea Choi, Christina Kendziorski, Ron Stewart, and James A Thomson. Single-cell rna-seq reveals novel regulators of human embryonic stem cell differentiation to definitive endoderm. Genome biology , 17:1-20, 2016.
- [12] Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, and Arnaud Doucet. Schrodinger bridge flow for unpaired data translation. Advances in Neural Information Processing Systems , 37:103384-103441, 2024.
- [13] Wei Deng, Weijian Luo, Yixin Tan, Marin Biloˇ s, Yu Chen, Yuriy Nevmyvaka, and Ricky T. Q. Chen. Variational schr¨ odinger diffusion models. In Forty-first International Conference on Machine Learning , 2024.
- [14] Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Sch¨ olkopf, and Alexander Smola. A kernel two-sample test. The Journal of Machine Learning Research , 13(1):723-773, 2012.
- [15] Guillaume Huguet, Daniel Sumner Magruder, Alexander Tong, Oluwadamilola Fasina, Manik 402 Kuchroo, Guy Wolf, and Smita Krishnaswamy. Manifold interpolating optimal-transport flows 403 for trajectory inference. Advances in neural information processing systems , 35:29705-29718, 404 2022. 405

- [16] Benton Jamison. The Markov processes of Schr¨ odinger. Zeitschrift f¨ ur Wahrscheinlichkeitsthe406 orie und verwandte Gebiete , 32(4):323-331, 1975. 407
- [17] Allon M Klein, Linas Mazutis, Ilke Akartuna, Naren Tallapragada, Adrian Veres, Victor Li, 408 Leonid Peshkin, David A Weitz, and Marc W Kirschner. Droplet barcoding for single-cell 409 transcriptomics applied to embryonic stem cells. Cell , 161(5):1187-1201, 2015. 410

411

412

- [18] Takeshi Koshizuka and Issei Sato. Neural Lagrangian Schr¨ odinger bridge: Diffusion modeling for population dynamics. arXiv preprint arXiv:2204.04853 , 2022.

413

414

415

416

417

418

419

420

421

422

423

424

425

426

427

428

429

430

431

432

433

434

435

436

437

438

439

440

441

442

443

444

445

446

447

448

449

450

451

452

453

- [19] Solomon Kullback. Probability densities with given marginals. The Annals of Mathematical Statistics , 39(4):1236-1243, 1968.
- [20] Christian L´ eonard. A survey of the Schr¨ odinger problem and some of its connections with optimal transport. arXiv preprint arXiv:1308.0215 , 2013.
- [21] Xuechen Li, Ting-Kam Leonard Wong, Ricky TQ Chen, and David Duvenaud. Scalable gradients for stochastic differential equations. In International Conference on Artificial Intelligence and Statistics , pages 3870-3882. PMLR, 2020.
- [22] Alex Tong Lin, Samy Wu Fung, Wuchen Li, Levon Nurbekyan, and Stanley J. Osher. Alternating the population and control neural networks to solve high-dimensional stochastic mean-field games. Proceedings of the National Academy of Sciences , 2021.
- [23] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations , 2023.
- [24] Guan-Horng Liu, Tianrong Chen, Oswin So, and Evangelos Theodorou. Deep generalized schr¨ odinger bridge. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems , 2022.
- [25] Guan-Horng Liu, Tianrong Chen, and Evangelos A Theodorou. Deep generalized schr \ ' odinger bridges: From image generation to solving mean-field games. arXiv preprint arXiv:2412.20279 , 2024.
- [26] Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos Theodorou, and Ricky T. Q. Chen. Generalized schr¨ odinger bridge matching. In The Twelfth International Conference on Learning Representations , 2024.
- [27] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. I 2 SB: Image-to-image Schr¨ odinger bridge. arXiv preprint arXiv:2302.05872 , 2023.
- [28] Evan Z Macosko, Anindita Basu, Rahul Satija, James Nemesh, Karthik Shekhar, Melissa Goldman, Itay Tirosh, Allison R Bialas, Nolan Kamitaki, Emily M Martersteck, et al. Highly parallel genome-wide expression profiling of individual cells using nanoliter droplets. Cell , 161(5):1202-1214, 2015.
- [29] Kenneth G Manton, XiLiang Gu, and Gene R Lowrimore. Cohort changes in active life expectancy in the us elderly population: Experience from the 1982-2004 national long-term care survey. The Journals of Gerontology Series B: Psychological Sciences and Social Sciences , 63(5):S269-S281, 2008.
- [30] Toshio Mikami. Stochastic optimal transportation: stochastic control with fixed marginals . Springer Nature, 2021.
- [31] Abdulwahab Mohamed, Alberto Chiarini, and Oliver Tse. Schr¨ odinger bridges with multimarginal constraints. 2021.
- [32] Kevin R Moon, David Van Dijk, Zheng Wang, Scott Gigante, Daniel B Burkhardt, William S Chen, Kristina Yim, Antonia van den Elzen, Matthew J Hirn, Ronald R Coifman, et al. Visualizing structure and transitions in high-dimensional biological data. Nature biotechnology , 37(12):1482-1492, 2019.

454

455

456

457

458

459

460

461

462

463

464

465

466

467

468

469

470

471

472

473

474

475

476

477

478

479

480

481

482

483

484

485

486

487

488

489

490

491

492

493

494

- [33] Kirill Neklyudov, Rob Brekelmans, Daniel Severo, and Alireza Makhzani. Action matching: Learning stochastic dynamics from samples. In Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research . PMLR, 23-29 Jul 2023.
- [34] Kirill Neklyudov, Rob Brekelmans, Alexander Tong, Lazar Atanackovic, Qiang Liu, and Alireza Makhzani. A computational framework for solving Wasserstein Lagrangian flows. arXiv preprint arXiv:2310.10649 , 2023.
- [35] Michele Pavon and Anton Wakolbinger. On free energy, stochastic control, and Schr¨ odinger processes. In Modeling, Estimation and Control of Systems with Uncertainty: Proceedings of a Conference held in Sopron, Hungary, September 1990 , pages 334-348. Springer, 1991.
- [36] Stefano Peluchetti. Non-denoising forward-time diffusions, 2022.
- [37] Stefano Peluchetti. Diffusion bridge mixture transports, schr¨ odinger bridge problems and generative modeling. Journal of Machine Learning Research , 24(374):1-51, 2023.
- [38] Stefano Peluchetti. BM$ˆ2$: Coupled schr¨ odinger bridge matching. Transactions on Machine Learning Research , 2025.
- [39] Paolo Dai Pra. A stochastic control approach to reciprocal diffusion processes. Applied Mathematics and Optimization , 23:313-329, 1991.
- [40] Hannes Risken and Hannes Risken. Fokker-planck equation . Springer, 1996.
- [41] Lars Ruthotto, Stanley J. Osher, Wuchen Li, Levon Nurbekyan, and Samy Wu Fung. A machine learning framework for solving high-dimensional mean field game and mean field control problems. Proceedings of the National Academy of Sciences , 2020.
- [42] Geoffrey Schiebinger, Jian Shu, Marcin Tabaka, Brian Cleary, Vidya Subramanian, Aryeh Solomon, Joshua Gould, Siyan Liu, Stacie Lin, Peter Berube, et al. Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. Cell , 176(4):928-943, 2019.
- [43] Erwin Schr¨ odinger. ¨ Uber die umkehrung der naturgesetze . Verlag der Akademie der Wissenschaften in Kommission bei Walter De Gruyter u . . . , 1931.
- [44] Yunyi Shen, Renato Berlinghieri, and Tamara Broderick. Multi-marginal Schr¨ odinger bridges with iterative reference refinement. arXiv preprint arXiv:2408.06277 , 2024.
- [45] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schr¨ odinger bridge matching. Advances in Neural Information Processing Systems , 36, 2024.
- [46] Vignesh Ram Somnath, Matteo Pariset, Ya-Ping Hsieh, Maria Rodriguez Martinez, Andreas Krause, and Charlotte Bunne. Aligned diffusion schr \ ' odinger bridges. arXiv preprint arXiv:2302.11419 , 2023.
- [47] Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research , 2024. Expert Certification.
- [48] Alexander Tong, Jessie Huang, Guy Wolf, David Van Dijk, and Smita Krishnaswamy. Trajectorynet: A dynamic optimal transport network for modeling cellular dynamics. In International conference on machine learning , pages 9526-9536. PMLR, 2020.
- [49] Francisco Vargas, Pierre Thodoroff, Austen Lamacraft, and Neil Lawrence. Solving Schr¨ odinger 495 bridges via maximum likelihood. Entropy , 23(9):1134, 2021. 496

497

498

499

500

501

502

503

504

505

506

507

508

509

510

511

512

513

514

515

516

517

518

519

520

521

522

523

524

525

526

527

528

529

530

531

532

533

534

535

536

537

538

539

540

541

542

543

544

545

546

547

548

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The key claims stated in the abstract and introduction correspond appropriately to the scope of the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The conclusion section provides a discussion on the limitations.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate 'Limitations' section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Yes, we are confident that our proof and assumptions are both valid and adequate.

## Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Yes, all the necessary data to reproduce the results can be found in the Appendix C.

## Guidelines:

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

Justification: We provided our code.

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

Justification: We have included the details of the experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Yes, we ran our code three times and reported the mean and standard deviations in the appendix. Due to space limitations, only the mean values are presented in the main text. The complete results can be found in Appendix C.

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

Answer: [Yes]

Justification: Yes, the necessary resources are included in the experimental details section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We support the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper presents work aimed at advancing the field of machine learning. Our research may have various societal consequences. However, we do not believe any of these require specific emphasis here.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: We believe our paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Yes, the license and terms of use are noted.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809

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

Justification: We do not involve crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: We do not involve crowdsourcing or research with human subjects

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

810

811

812

813

814

815

816

817

818

819

820

821

822

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We do not use LLM for core methodology, scientific rigorousness, or originality of the research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.