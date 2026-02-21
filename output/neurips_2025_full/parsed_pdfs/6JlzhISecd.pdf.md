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

## A Stochastic Approximation Approach for Efficient Decentralized Optimization on Random Networks

## Anonymous Author(s)

Affiliation Address email

## Abstract

A challenging problem in decentralized optimization is to develop algorithms with fast convergence on random and time varying topologies under unreliable and bandwidth-constrained communication network. This paper studies a stochastic approximation approach with a Fully Stochastic Primal Dual Algorithm ( FSPDA ) framework. Our framework relies on a novel observation that randomness in time varying topology can be incorporated in a stochastic augmented Lagrangian formulation, whose expected value admits saddle points that coincide with stationary solutions of the decentralized optimization problem. With the FSPDA framework, we develop two new algorithms supporting efficient sparsified communication on random time varying topologies FSPDA-SA allows agents to execute multiple local gradient steps depending on the time varying topology to accelerate convergence, and FSPDA-STORM further incorporates a variance reduction step to improve sample complexity. For problems with smooth (possibly non-convex) objective function, within T iterations, we show that FSPDA-SA (resp. FSPDA-STORM ) finds an O (1 / √ T ) -stationary (resp. O (1 /T 2 / 3 ) ) solution. Numerical experiments show the benefits of the FSPDA algorithms.

## 1 Introduction

Consider n agents that communicate on an undirected and connected graph/network G = ( V , E ) with V = [ n ] := { 1 , . . . , n } , E ⊆ V × V . Each agent i ∈ [ n ] has access to a continuously differentiable (possibly non-convex) local objective function f i : R d → R and maintains a local decision variable x i ∈ R d . Denote x = [ x ⊤ 1 , ..., x ⊤ n ] ⊤ ∈ R nd . Our aim is to tackle:

<!-- formula-not-decoded -->

In other words, (1) seeks a x ⋆ ∈ R d that minimizes F ( x ) := (1 /n ) ∑ n i =1 f i ( x ) . We are interested in the stochastic optimization setting where each f i ( x i ) is given by (with slight abuse of notation)

<!-- formula-not-decoded -->

where P i represents the i -th data distribution. Problem (1) is relevant to the distributed learning problem especially in the decentralized case where a central server is absent. Prior works [Nedic and Ozdaglar, 2009, Lian et al., 2017, Nedic et al., 2017, Qu and Li, 2017] demonstrated that decentralized algorithms can tackle (1) efficiently through repeated message exchanges among the neighbors and local stochastic gradient updates.

Towards an efficient decentralized algorithm for (1), an important direction is to consider a time 29 varying graph topology setting where the active edge set in G changes over time. This is a generic 30 setting covering cases when the communication links are unreliable, or the agents choose not to 31 communicate in a certain round (a.k.a. local updates) [Koloskova et al., 2019a, Nadiradze et al., 2021]. 32

| Prior Works                                                                                                                                                                                                                                                                                                     | SG                  | TV                          | w/o BH              | Rate                                                                                                                                       |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|-----------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Prox-GPDA [Hong et al., 2017] NEXT [Lorenzo and Scutari, 2016] DSGD [Koloskova et al., 2020] Swarm-SGD [Nadiradze et al., 2021] CHOCO-SGD [Koloskova et al., 2019a] Decen-Scaffnew [Mishchenko et al., 2022] Local-GT [Liu et al., 2024] LED [Alghunaim, 2024] FSPDA-SA ( This Work ) FSPDA-STORM ( This Work ) | ✗ ✗ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ | ✗ ✓ ✓ ✓ ✗ ‡ ✗ † ✗ † ✗ † ✓ ✓ | ✓ ✓ ✗ ✗ ✗ ✓ ✓ ✓ ✓ ✓ | Asympt. Asympt. O ( σ/ √ nT ) O ( σ 2 / √ T ) O ( σ/ √ nT ) O ( σ/ √ nT ) O ( σ/ √ nT ) O ( σ/ √ nT ) O ( σ/ √ nT ) O ( σ 2 / 3 /T 2 / 3 ) |

Table 1: Comparison of decentralized algorithms for non-convex optimization. In the table, ' SG ' is 'Stochastic Gradient', ' TV ' is 'Time Varying Graph', ' w/o BH ' is 'Without Bounded Heterogeneity', and ' Rate ' is the expected squared gradient norm E [ ∥∇ F (¯ x ) ∥ 2 ] after T iterations. Note that σ 2 is the variance of stochastic gradient. ‡ CHOCO-SGD incorporates broadcast gossip as a special case of compression. † ProxSkip, Local-GT, LED consider local updates with periodic communication.

By assuming that a random topology is drawn at each iteration, the convergence of decentralized 33 stochastic gradient (DSGD) has been studied in [Lobel and Ozdaglar, 2010, Nadiradze et al., 2021] 34 and is later on unified by [Koloskova et al., 2020] with tighter bounds for local updates, periodic 35 sampling, etc. An alternative [Ram et al., 2010] is to analyze DSGD for the B -connectivity setting 36 which requires the union of every B consecutive time varying topologies to yield a connected graph. 37 Nevertheless, these works focused on vanilla DSGD that may have slow convergence (in transient 38 stage) and is limited to bounded data heterogeneity. The prior restrictions can be relaxed using 39 advanced algorithms such as gradient tracking [Qu and Li, 2017], EXTRA [Shi et al., 2015] and 40 primal-dual framework [Hong et al., 2017, Hajinezhad and Hong, 2019, Yi et al., 2021]. 41

As noted by [Koloskova et al., 2021], analyzing the convergence of sophisticated algorithms with time 42 varying topology, such as gradient tracking [Qu and Li, 2017] is challenging due to the non-symmetric 43 product of two (or more) mixing matrices. Existing works considered various restrictions on the 44 time varying topology G ( t ) = ( V , E ( t ) ) and/or the problem (1): [Koloskova et al., 2021, Liu et al., 45 2024] studied gradient tracking with local updates that essentially takes E ( t ) = E periodically and 46 E ( t ) = ∅ otherwise, also see [Mishchenko et al., 2022, Guo et al., 2023, Alghunaim, 2024] for a 47 similar result and note that such algorithms require extra synchronization overhead; [Kovalev et al., 48 2021, 2024] considered a setting where G ( t ) is connected for any t ; [Nedic et al., 2017, Li and Lin, 49 2024] focused on (accelerated) gradient tracking with deterministic gradient when F ( x ) is (strongly) 50 convex; [Lorenzo and Scutari, 2016] also considered deterministic gradient with possibly non-convex 51 F ( x ) but only provides asymptotic convergence guarantees; [Lei et al., 2018, Yau and Wai, 2023] 52 considered asymptotic convergence guarantees in the case of strictly (or strongly) convex F ( x ) . We 53 provide a non-exhaustive list summarizing the convergence of existing works in Table 1. 54

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

The above discussion highlights a gap in the existing literature -

Is there any algorithm that achieves fast convergence on time varying (random) topology?

This paper gives an affirmative answer through developing the Fully Stochastic Primal Dual Algorithm ( FSPDA ) framework that leads to efficient decentralized algorithms tackling (1) in its general form. The framework features the design of a new stochastic augmented Lagrangian function.

As pointed out by [Chang et al., 2020], many decentralized algorithms (including gradient tracking)

can be interpreted as primal-dual algorithms finding a saddle point of the augmented Lagrangian func- tion. However, its extension to time varying topology is not straightforward due to the inconsistency

in dual variables updates. To overcome this challenge, we propose a stochastic equality constrained reformulation of (1) to model randomness in topology. Then, the latter yields a stochastic augmented

Lagrangian function. Applying stochastic approximation (SA) to solve the latter leads to the

framework. Our contributions are 66

FSPDA

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

83

84

85

86

87

88

89

90

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

- We propose two new algorithms: (i) FSPDA-SA is derived by vanilla SA that applies primal-dual stochastic gradient descent-ascent on the stochastic augmented Lagrangian, (ii) FSPDA-STORM uses an additional control variate / momentum term to reduce the drift term's variance in a recursive manner. Both algorithms are fully stochastic as the random time varying topology is treated as a part of randomness. Additionally, our framework supports sparsified communication, i.e., the agents can choose to communicate a subset of primal coordinates at each iteration.
- We show that after T iterations, FSPDA-SA (resp. FSPDA-STORM ) finds in expectation a solution whose squared gradient norm is O (1 / √ T ) (resp. O (1 /T 2 / 3 ) ). The convergence analysis is derived from a new Lyapunov function design that involves an unsigned inner product term and incorporates a variance condition on the random time varying topologies. Interestingly, we show empirically that using momentum in dual updates benefits the consensus error convergence.
- We also demonstrate that both FSPDA-SA and FSPDA-STORM can be implemented in a fully asynchronous manner, i.e., the agents can communicate and compute at different time slots, and supports local update as the algorithms allow for arbitrary time varying topology. That said, we remark that the convergence rates with local updates of FSPDA-SA and FSPDA-STORM are only suboptimal.

We provide numerical experiments to show that FSPDA-SA and FSPDA-STORM outperform existing algorithms in terms of iteration and communication complexity.

Notations. Let W ∈ R d × d be a symmetric (not necessarily positive semidefinite) matrix, the W -weighted (semi) inner product of vectors a , b ∈ R d is denoted as ⟨ a | b ⟩ W := a ⊤ Wb . Similarly, the W -weighted (semi) norm is denoted by ∥ a ∥ 2 W := ⟨ a | a ⟩ W . The subscript notation is omitted for I -weighted inner products. For any square matrix X , ( X ) † denotes its pseudo inverse.

## 2 The Fully Stochastic Primal Dual Algorithm ( FSPDA ) Framework

This section develops the FSPDA framework for tackling (1) and describes two variants of the framework leading to decentralized stochastic optimization of (1). Let ˜ A ∈ {-1 , 0 , 1 } |E|× n be an incidence matrix of G . By defining A = ˜ A ⊗ I d ∈ {-1 , 0 , 1 } |E| d × nd , we observe that the consensus constraint in (1) is equivalent to Ax = 0 .

Our first step is to model the randomness in the time varying topology using the random variable (r.v.) ξ a ∼ P a . For each realization ξ a , we define the random incidence matrix A ( ξ a ) := I ( ξ a ) A ∈ {-1 , 0 , 1 } |E| d × nd where I ( ξ a ) ∈ { 0 , 1 } |E| d ×|E| d is a binary diagonal matrix. In addition to selecting each edge of G randomly, I ( ξ a ) selects a random subset of d coordinates. As we will see later, this allows our approach to simultaneously achieve random sparsification for communication compression.

Assume that E ξ a ∼ P a [ I ( ξ a )] is a positive diagonal matrix, (1) is equivalent to:

<!-- formula-not-decoded -->

Denote ξ = ( ξ 1 , . . . , ξ n , ξ a ) , FSPDA hinges on the following augmented Lagrangian function of (3):

<!-- formula-not-decoded -->

where ˜ η &gt; 0 , ˜ γ &gt; 0 are penalty parameters. It can be verified that the saddle points of L ( x , λ ) correspond to the KKT points of (1) [Bertsekas, 2016]. For brevity, in the rest of this paper, we may drop the subscript in ξ whenever the notation is clear from the context.

FSPDA is developed from applying stochastic approximation (SA) to seek a saddle point of (4). By recognizing A ( ξ ) ⊤ A ( ξ ) = A ⊤ A ( ξ ) , we consider the stochastic gradients:

<!-- formula-not-decoded -->

where ∇ f ( x ; ξ ) = [ ∇ f 1 ( x 1 ; ξ 1 ); . . . ; ∇ f n ( x n ; ξ n )] ∈ R nd . Notice that to facilitate algorithm development, we have taken a deterministic A for the term in ∇ x L related to λ . Now observe the i th d -dimensional block of A ⊤ A ( ξ ) x which can be aggregated within N i ( ξ ) the neighborhood of the i th agent as:

<!-- formula-not-decoded -->

where C ij ( ξ ) ∈ { 0 , 1 } d × d is diagonal and depends on the selected coordinates for the edge ( i, j ) under randomness ξ . Eq. (6) only relies on x j from neighbor j that is connected on the time varying

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

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

topology G ( ξ ) . For illustration, an example of the above random graph model is given by Figure 3 in Appendix A. Importantly, (5) shows that with the stochastic augmented Lagrangian function, the time varying topology can be treated implicitly as a part of the randomness in the stochastic primal-dual gradients. The framework is thus described as being fully stochastic as in [Bianchi et al., 2021], and departs from [Liu et al., 2024, Alghunaim, 2024] that treat the topology as fixed during the derivation of primal-dual algorithm(s). From (5), (6), we derive two variants of FSPDA .

FSPDA-SA Algorithm. The first variant of FSPDA is derived from a direct application of stochastic gradient descent-ascent (SGDA) updates. Take α &gt; 0 , β &gt; 0 as the step sizes, we have

<!-- formula-not-decoded -->

Taking the variable substitution ̂ λ := A ⊤ λ yields the following recursion:

<!-- formula-not-decoded -->

Note that x 0 , ̂ λ 0 can be initialized arbitrarily.

FSPDA-STORM Algorithm. The second variant of FSPDA reduces the variance of the stochastic gradient term in (5) using the recursive momentum variance reduction technique [Cutkosky and Orabona, 2019]. Herein, the key idea is to utilize a control variate in estimating the (primal-dual) gradients of L ( x , λ ) . Take α, β &gt; 0 and a x , a λ ∈ [0 , 1] as the momentum parameters, we have x t +1 = x t -α m t x , λ t +1 = λ t + β m t λ as the primal-dual updates, and

<!-- formula-not-decoded -->

The aim of m t +1 x is to estimate ∇ x L ( x t +1 , λ t +1 ) . Now, instead of the straightforward estimator ∇ x L ( x t +1 , λ t +1 ; ξ t +1 ) , we include an extra zero-mean term m t x - ∇ x L ( x t , λ t ; ξ t +1 ) to reduce the variance of the stochastic gradient estimation. The latter is a control variate that is computed recursively. Particularly, it has been shown in [Cutkosky and Orabona, 2019] that it can effectively reduce variance with a carefully designed parameter a x , provided that the stochastic gradient map satisfies a mean-square Lipschitz condition. We summarize the algorithm as follows.

<!-- formula-not-decoded -->

Note that to achieve the theoretical performance (see later in Sec. 3), x 0 , ̂ λ 0 , m 0 x , m 0 λ shall be initialized as x 0 i = ¯ x 0 , ̂ λ 0 i = ( α/η ) n -1 ( ∇ F (¯ x 0 ) - ∇ f i (¯ x 0 )) , m 0 x,i = ∇ F (¯ x 0 ) , m 0 λ,i = 0 according to (23). We remark that a simple initialization choice ̂ λ 0 = m 0 x,i = m 0 λ,i = 0 works well in practice.

Both FSPDA-SA and FSPDA-STORM are decentralized algorithms that can be implemented on random time varying topology, and support randomized sparisification for further communication compression. The key is to observe that in (8), (10), the only information required for agent i is to obtain ∑ j ∈N i ( ξ t a ) C ij ( ξ t a )( x t j -x t i ) , and in addition ∑ j ∈N i ( ξ t a ) C ij ( ξ t a )( x t -1 j -x t -1 i ) for FSPDA-STORM , at iteration t .

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

## 2.1 Implementation Details and Connection to Existing Works

We discuss several features of the FSPDA algorithms and their connections to existing works.

Local &amp; Asynchronous Updates. The local update scheme where each agent i is allowed to update its own local variables x i , λ i for multiple iterations without a communication step is a common practice in decentralized optimization [Liu et al., 2024, Li and Lin, 2024, Alghunaim, 2024, Mishchenko et al., 2022]. As discussed before, such scheme can be seen as a special case of the FSPDA framework where the time varying topology E ( t ) is chosen such that the latter alternates between E ( t ) = E and E ( t ) = ∅ .

Furthermore, FSPDA-SA allows for the general case of asynchronous updates. This is done so by taking the stochastic gradient as ∇ f i ( x t i ; ξ t ) = b i ( ξ t ) b i ∇ f i ( x t i ; ξ t ) such that b i ( ξ t ) ∈ { 0 , 1 } with E [ b i ( ξ t )] = 1 /b i for some constant b i &gt; 0 . Detailed discussions for a fully asynchronous implementation of FSPDA-SA can be found in Appendix A.

Connection to Existing Works. Evaluating x t +2 -x t +1 from the FSPDA-SA sequence and observe that the combination of (8a) and (8b) is equivalent to the second order recursion:

<!-- formula-not-decoded -->

This reduces the FSPDA-SA recursion into a primal-only sequence by eliminating the dual sequence λ t . In the deterministic optimization setting when A ( ξ ) ≡ A and ∇ f ( x ; ξ ) ≡ ∇ f ( x ) , (11) is equivalent to the EXTRA algorithm [Shi et al., 2015] using the mixing matrix W = I -γ Diag( ˜ W1 ) + γ ˜ W where ˜ W is the 0-1 adjacency matrix of G . Here, with an appropriate choice of γ , W will be doubly stochastic and satisfies the convergence requirement in [Shi et al., 2015]. Similar observations have been made in [Nedic et al., 2017] for the gradient tracking and DIGing algorithms.

On the other hand, for stochastic optimization on random networks, (11) suggests each agent to keep the current and previous iterates received from neighbors in the corresponding time varying topology. In this case, (11) yields an extension of the EXTRA/GT algorithms to time varying topology.

## 3 Convergence Analysis of FSPDA

This section presents the convergence rate analysis of FSPDA for (1). Unless otherwise specified, we focus on the case with smooth but possibly non-convex objective function. Specifically, we consider:

Assumption 3.1. Each f i is L -smooth, i.e., for i = 1 , . . . , n ,

<!-- formula-not-decoded -->

There exists f ⋆ &gt; -∞ such that f i ( x ) ≥ f ⋆ for any x ∈ R d .

Note this implies that the global objective function F ( · ) is L -smooth but possibly non-convex.

We further assume that the random network G ( ξ a ) is connected in expectation, yet each realization G ( ξ a ) may not be connected. Let R = E [ I ( ξ a )] , this leads to the following property concerning the expected graph Laplacian matrix A ⊤ RA = E [ A ( ξ a ) ⊤ A ] . Defining the matrix K := ( I n -11 ⊤ /n ) ⊗ I d , we have

Assumption 3.2. There exists ρ max ≥ ρ min &gt; 0 and ¯ ρ max ≥ ¯ ρ min &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It holds that A ⊤ RAK = A ⊤ RA = KA ⊤ RA . The above assumption can be satisfied if G is connected [Yi et al., 2021], [Yi et al., 2018, Lemma 2] and diag( R ) &gt; 0 such that each edge is selected with a positive probability. As an important consequence, if γ ≤ ρ min /ρ 2 max , we have

<!-- formula-not-decoded -->

We thus observe that the operator ( I -γ A ⊤ RA ) serves a similar purpose as the mixing matrix 180 in a average consensus algorithms and ρ min can be interpreted as the spectral radius of G similar 181

to [Koloskova et al., 2020, Eq. (12)]. Moreover, if we define Q := ( A ⊤ RA ) † such that it holds 182 QA ⊤ RA = A ⊤ RAQ = K , Assumption 3.2 implies that ρ -1 max K ⪯ Q ⪯ ρ -1 min K . 183

184

Next we consider several assumptions on the noise variance of the random quantities in FSPDA :

185

186

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

Assumption 3.3. For any fixed x i ∈ R d , i ∈ [ n ] , there exists σ i ≥ 0 such that

<!-- formula-not-decoded -->

To simplify notations, we define ¯ σ 2 := (1 /n ) ∑ n i =1 σ 2 i .

Assumption 3.4. For any fixed x ∈ R nd , there exists σ A ≥ 0 such that

<!-- formula-not-decoded -->

Assumption 3.3 is standard. Meanwhile for Assumption 3.4, the variance term σ 2 A measures the quality of the random topology G ( ξ a ) in approximating the expected graph Laplacian A ⊤ RA . The latter is important as it contributes to the variance in the drift term of FSPDA . Observe that σ 2 A decreases with the proportion of edges selected in each random subgraph G ( ξ a ) .

To facilitate our discussions, we define the following quanitites:

<!-- formula-not-decoded -->

Convergence of FSPDA-SA . We summarize the convergence rate for FSPDA-SA as follows. The proof can be found in Appendix C:

Theorem 3.5. Under Assumptions 3.1, 3.2, 3.3, 3.4. Suppose that the step sizes satisfy the conditions defined in (46) . Then, for any T ≥ 1 with the random stopping iteration T ∼ Unif { 0 , ..., T -1 } , the iterates generated by FSPDA-SA satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any a &gt; 0 , where F 0 , C σ are defined in (44) , (50) .

Setting a = O ( n/ √ T ¯ σ 2 ) , α = √ n/ ( T ¯ σ 2 ) (and assuming ¯ σ &gt; 0 ), we have

<!-- formula-not-decoded -->

which is the same asymptotic convergence rate as a centralized SGD algorithm that takes n stochastic gradient samples uniformly from each agent, i.e., linear speedup [Lian et al., 2017]. Also, using a = 1 , the consensus error converges as a rate of E [∑ n i =1 ∥ x T i -¯ x T ∥ 2 ] = O ( n 2 σ 2 A ρ max / ( Tρ 2 min )) under the same step size choice used in (19). Notice that for T ≫ 1 , the effect of random topology only degrades the convergence of consensus error, keeping the transient rate in (19) unaffected. If the gradients are deterministic ( ¯ σ = 0 ), setting a = ( L 2 η ∞ ρ min ) 1 / 3 , α = α ∞ will yield a better convergence rate as E [ ∥∇ F (¯ x T ) ∥ 2 ] = O ( σ 4 A √ n/T ) . Without a transient phase, the error due to random graph and coordinate sparsification is persistent through σ 4 A in the above convergence rate.

We further show that the convergence of FSPDA-SA can be accelerated if the objective function of (1) satisfies the Polyak-Lojasiewicz (PL) condition:

Assumption 3.6. There exists a constant µ &gt; 0 such that 2 µ ( F ( x ) -f ⋆ ) ≤ ∥∇ F ( x ) ∥ 2 , ∀ x ∈ R d .

Assumption 3.6 includes strongly convex functions as a special case, but also includes other nonconvex functions; see [Karimi et al., 2016]. We observe:

Corollary 3.7. Suppose the assumptions and step size conditions in Theorem 3.5 hold. Furthermore, with Assumption 3.6, there exists δ ∈ (0 , 1) such that for any t ≥ 0 ,

<!-- formula-not-decoded -->

for F t , C σ defined in (44) , (70) , and δ = min { αµ/ 4 , γρ min / 16 , ηβ/ (3 ρ min ) , η/ 12 } .

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

228

229

230

231

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

The proof can be found in Appendix C.6. By setting α = c ln( T ) / ( n 2 T ) in (20), with a carefully chosen c and a sufficiently large T such that α ≤ α ∞ , we can ensure that

<!-- formula-not-decoded -->

In the case of deterministic gradient, i.e., ¯ σ 2 = 0 , by setting α = α ∞ , (20) ensures a linear convergence rate of E [ F (¯ x T ) -f ⋆ + ∥ x T ∥ 2 K ] = O ((1 -δ ) T ) , which shows that the performance of FSPDA-SA is on par with [Nedic et al., 2017, Xu et al., 2017], despite it only requires one round of (sparsified) transmission per iteration.

Convergence of FSPDA-STORM . To exploit the benefits of control variates, we need an additional assumption on the stochastic gradient map:

<!-- formula-not-decoded -->

,

<!-- formula-not-decoded -->

The above assumption is also known as the mean-square smoothness condition, see [Cutkosky and Orabona, 2019], which is strictly stronger than Assumption 3.1. We observe the following convergence guarantee for FSPDA-STORM , whose proof can be found in Appendix D.

Theorem 3.9. Under Assumptions 3.1, 3.2, 3.3, 3.4, 3.8. Suppose that the step sizes satisfy the conditions in (184) -(214) . Then, for any T ≥ 1 with the random stopping iteration T ∼ Unif { 0 , ..., T -1 } , the iterates generated by FSPDA-STORM satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the constants F 0 , a , e , f are defined in (110) .

Setting α = O (¯ σ -2 / 3 T -1 / 3 ) , η = O ( n ) , γ = O ( T -1 / 3 ) , β = O ( n -1 T -2 / 3 ) , a x = O (¯ σ -4 / 3 T -2 / 3 ) , a λ = O ( T -1 / 3 ) , f = O ( n -1 T 1 / 3 ) (see (111) -(117)), and initializing the algorithm such that ∥ v 0 ∥ 2 K = O ( T -2 / 3 ) , ∥ m 0 x -(1 /n ) 1 ⊤ ⊗ ∇ f ( x 0 ) ∥ 2 = O ( T -1 / 3 ) and ∥ m 0 x -∇ x L ( x 0 , λ 0 ) ∥ 2 = O ( T -1 / 3 ) , we have

<!-- formula-not-decoded -->

In regard to the order of ¯ σ and T , provided that n is small, the convergence rate of FSPDA-STORM matches the lower bound [Arjevani et al., 2023] for non-convex functions under the same smoothness assumption. Moreover, by the same choice of step sizes, the consensus error converges at the rate of E [∑ n i =1 ∥ x T i -¯ x T ∥ 2 ] = O (¯ σ 2 / 3 nρ -1 min T -2 / 3 ) . We remark that in (25), the rate remains constant as n increases such that FSPDA-STORM does not offer the same linear speedup observed in Theorem 3.5 for FSPDA-SA . Nevertheless, as T ≫ 1 , the rate of FSPDA-STORM will surpass that of FSPDA-SA and other decentralized algorithms on time varying topologies.

Lastly, we provide detailed discussions on the convergence rates above, e.g., transient time, effects of random topology, etc., in Appendix B.

## 3.1 Insight from Analysis: Fixed Point Iteration of FSPDA-SA

<!-- formula-not-decoded -->

This shows that the evolution of { ¯ x t } t ≥ 0 is similar to that of 'centralized' SGD applied on (1) except that the local gradients are evaluated on the local iterates. However, it is still not straightforward to analyze the convergence of FSPDA-SA as the update of x t involves the dual variable λ t which lacks an intuitive interpretation for constructing the right Lyapunov function.

To this end, we study the fixed point(s) of (8) to gain insights. Suppose that for some t ⋆ , the fixed point conditions E [ λ t ⋆ +1 | ξ : t ⋆ ] = λ t ⋆ , E [ x t ⋆ +1 | ξ : t ⋆ ] = x t ⋆ hold. Since R is a diagonal matrix with positive diagonal elements, we observe

<!-- formula-not-decoded -->

On the other hand, the primal update yields 246

<!-- formula-not-decoded -->

Since x t ⋆ 1 = x t ⋆ 2 = · · · = x t ⋆ n at the fixed point (due to (27)), by the consensus condition across two time steps, it implies

<!-- formula-not-decoded -->

From (29), we see that ̂ λ t shall converge to the difference between global and local gradient. Inspired by the above, to facilitate the analysis later, we define

<!-- formula-not-decoded -->

for any t ≥ 0 . In particular, we see that ∥ v t ∥ 2 K measures the violation of (29) in tracking the average deterministic gradient using the dual variables. The latter will be instrumental in analyzing the consensus error bound, as revealed in Lemma C.2.

## 4 Numerical Experiments

This section reports the numerical experiments on practical performance of FSPDA . For the time varying topology, we take an extreme setting where for each realization G ( ξ a ) , only one edge will be selected uniformly at random from G . We evaluate the performance with the worst-agent metric, i.e., we present the training loss as max i ∈ [ n ] F ( x t i ) , and the stationarity/gradient-norm measure as max i ∈ [ n ] ∥∇ F ( x t i ) ∥ 2 . This captures the worst-case of the solutions produced by the algorithms. Unless otherwise specified, all algorithms are initialized with x 0 i = ¯ x 0 , and for FSPDA we initialize ̂ λ 0 = m 0 x,i = m 0 λ,i = 0 , and the stochastic gradients are estimated with a batch size of 256. In the interest of space, omitted details and hyperparameters of the experiments can be found in Appendix F.

MNIST Experiments. The first set of experiments considers a moderate-scale setting of training a one hidden layer feed-forward neural network with 100 hidden neurons (total number of parameters d = 79,510) on the MNIST dataset with m = 60 , 000 samples of 784 -dimensional features.

In the first experiment, we consider the static topology G as an Erdos-Renyi graph with connectivity of p = 0 . 5 and n = 10 agents. We compare the proposed FSPDA-SA , FSPDA-STORM with six benchmark algorithms utilizing different types of time-varying topology. Among them, DSGD [Koloskova et al., 2020] and Swarm-SGD [Nadiradze et al., 2021] use the general time varying topology setting as FSPDA where each edge of G ( ξ a ) is active uniformly at random, in addition to random sparsification used FSPDA-SA and adaptive quantized used in Swarm-SGD ; CHOCO-SGD [Koloskova et al., 2019b] takes G ( ξ a ) as an broadcasting subgraph where one agent selects all his/her neighbors; Decen-Scaffnew [Mishchenko et al., 2022], LED [Alghunaim, 2024], and K-GT [Liu et al., 2024] utilize local updates where G ( ξ a ) is either taken as an empty topology, or as the static topology G . We configure these algorithms such that they have the same communication cost (in terms of bits transmitted over network) on average . For instance, the local update algorithms ( Decen-Scaffnew , LED , K-GT ) only communicate once using G every O ( |E| d k ) iterations to match the communication cost of k -coordinate sparse one-edge random graph used in FSPDA .

The local objective function held by each agent is the cross-entropy classification loss on a local dataset with m i = 6000 samples, plus a regularization loss λ 2 ∥ x i ∥ 2 with λ = 10 -4 , where x i are the weight parameters of the feed-forward neural network classifier. We split the training set into n = 10 disjoint sets such that each set contains only one class label and assign each set to one agent as its local dataset. Note that as we do not shuffle the data samples across local datasets, the local objective function held by different agents will become highly heterogeneous.

Fig. 1 compares the squared gradient norm, training loss, consensus error of the benchmarked algorithms. We first note that both FSPDA algorithms have significantly outperformed DSGD , Swarm-SGD on the general time varying topology as well as CHOCO-SGD . Meanwhile, the performance of FSPDA is comparable to the local update algorithms Decen-Scaffnew , LED , K-GT . Notice that the latter

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

267

268

269

270

271

272

273

274

275

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

Figure 1: Feed-forward neural network classification training on MNIST using 10 6 iterations.

<!-- image -->

Figure 2: Resnet-50 classification training on Imagenet.

<!-- image -->

require additional synchronization steps which may not be suitable for random networks. Lastly, we notice that as T ≫ 1 , FSPDA-STORM can slightly outperform FSPDA-SA due to its O (1 /T 2 / 3 ) rate as shown in our analysis. We further expand the experiments by a series of ablation studies over data heterogeneity, sparsity levels, graph topologies, gradient noise and dual momentum in Appendix E.

Imagenet Experiments. The second set of experiments consider a large-scale setting for training a Resnet-50 network (total number of parameters d = 25,557,032) on the Imagenet dataset (training dataset of 1,281,168 images from 100 classes, re-scaled and cropped to 256 × 256 image dimensions). We consider cross-entropy classification loss plus the same L2 norm regularization loss as in the previous setup. We split the dataset across a network of n = 8 nodes where the static graph G is taken as the fully connected topology. The performance metrics are measured at the network average iterate ¯ x t . Inspired by [Loshchilov and Hutter, 2016, Eq. (5)] we adopt a cosine learning rate scheduling with 5 epochs of linear warm up for every algorithm. In particular, the step sizes α, η of FSPDA-SA are scheduled simultaneously such that α t /η t remains constant, as illustrated in Appendix F. We draw a batch of 128 samples to estimate the stochastic gradient.

We focus on the communication efficiency and only compare FSPDA-SA , CHOCO-SGD , Swarm-SGD in this experiment due to limited resources. The results are reported in Figure 2 that compare the test accuracy and training loss against iteration number and bits transmitted. When compared with CHOCO-SGD , FSPDA-SA achieves almost the same accuracy using one-edge random graphs with at least 100x reduction in communication cost on 100 epoch training. Also notice that further compressing the communication to 0.1% sparse coordinates in FSPDA-SA requires more training epochs to recover the same level of accuracy.

Conclusions. This paper proposed a fully stochastic primal dual gradient algorithm ( FSPDA ) framework for decentralized optimization over arbitrarily time varying random networks. We utilize a new stochastic augmented Lagrangian function and apply SA to search for its saddle point. We develop two algorithms, one is by plain SA ( FSPDA-SA ), and one uses control variates for variance reduction ( FSPDA-STORM ). We prove that both algorithms achieve state-of-the-art convergence rates, while relaxing assumptions on both bounded heterogeneity and the type of time varying topologies.

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

358

359

360

361

362

## References

- Sulaiman A Alghunaim. Local exact-diffusion for decentralized optimization and learning. IEEE Transactions on Automatic Control , 2024.
- Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth. Lower bounds for non-convex stochastic optimization. Mathematical Programming , 199(1): 165-214, 2023.
- Dimitri Bertsekas. Nonlinear Programming , volume 4. Athena Scientific, 2016.
- Pascal Bianchi, Walid Hachem, and Adil Salim. A fully stochastic primal-dual algorithm. Optimization Letters , 15(2):701-710, 2021.
- Tsung-Hui Chang, Mingyi Hong, Hoi-To Wai, Xinwei Zhang, and Songtao Lu. Distributed learning in the nonconvex world: From batch data to streaming and beyond. IEEE Signal Processing Magazine , 37(3):26-38, 2020.
- Ashok Cutkosky and Francesco Orabona. Momentum-based variance reduction in non-convex sgd. Advances in neural information processing systems , 32, 2019.
- Luyao Guo, Sulaiman A Alghunaim, Kun Yuan, Laurent Condat, and Jinde Cao. Revisiting decentralized proxskip: Achieving linear speedup. arXiv preprint arXiv:2310.07983 , 2023.
- Davood Hajinezhad and Mingyi Hong. Perturbed proximal primal-dual algorithm for nonconvex nonsmooth optimization. Mathematical Programming , 176(1):207-245, 2019.
- Mingyi Hong, Davood Hajinezhad, and Ming-Min Zhao. Prox-pda: The proximal primal-dual algorithm for fast distributed nonconvex optimization and learning over networks. In International Conference on Machine Learning , pages 1529-1538. PMLR, 2017.
- Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. Foundations and trends® in machine learning , 14(1-2):1-210, 2021.
- Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximalgradient methods under the polyak-łojasiewicz condition. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2016, Riva del Garda, Italy, September 19-23, 2016, Proceedings, Part I 16 , pages 795-811. Springer, 2016.
- Anastasia Koloskova, Tao Lin, Sebastian U Stich, and Martin Jaggi. Decentralized deep learning with arbitrary communication compression. In International Conference on Learning Representations , 2019a.
- Anastasia Koloskova, Sebastian Stich, and Martin Jaggi. Decentralized stochastic optimization and gossip algorithms with compressed communication. In International Conference on Machine Learning , pages 3478-3487. PMLR, 2019b.
- Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian Stich. A unified theory of decentralized sgd with changing topology and local updates. In International Conference on Machine Learning , pages 5381-5393. PMLR, 2020.
- Anastasiia Koloskova, Tao Lin, and Sebastian U Stich. An improved analysis of gradient tracking for decentralized machine learning. Advances in Neural Information Processing Systems , 34: 11422-11435, 2021.
- Dmitry Kovalev, Elnur Gasanov, Alexander Gasnikov, and Peter Richtarik. Lower bounds and optimal algorithms for smooth and strongly convex decentralized optimization over time-varying networks. Advances in Neural Information Processing Systems , 34:22325-22335, 2021.
- Dmitry Kovalev, Ekaterina Borodich, Alexander Gasnikov, and Dmitrii Feoktistov. Lower bounds and optimal algorithms for non-smooth convex decentralized optimization over time-varying networks. arXiv preprint arXiv:2405.18031 , 2024.

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

402

403

404

405

406

407

- Jinlong Lei, Han-Fu Chen, and Hai-Tao Fang. Asymptotic properties of primal-dual algorithm for distributed stochastic optimization over random networks with imperfect communications. SIAM Journal on Control and Optimization , 56(3):2159-2188, 2018.
- Huan Li and Zhouchen Lin. Accelerated gradient tracking over time-varying graphs for decentralized optimization. Journal of Machine Learning Research , 25(274):1-52, 2024.
- Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang, and Ji Liu. Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent. Advances in neural information processing systems , 30, 2017.
- Yue Liu, Tao Lin, Anastasia Koloskova, and Sebastian U Stich. Decentralized gradient tracking with local steps. Optimization Methods and Software , pages 1-28, 2024.
- Ilan Lobel and Asuman Ozdaglar. Distributed subgradient methods for convex optimization over random networks. IEEE Transactions on Automatic Control , 56(6):1291-1306, 2010.
- Paolo Di Lorenzo and Gesualdo Scutari. Next: In-network nonconvex optimization. IEEE Transactions on Signal and Information Processing over Networks , 2(2):120-136, 2016.
- Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 , 2016.
- Songtao Lu, Xinwei Zhang, Haoran Sun, and Mingyi Hong. Gnsd: A gradient-tracking based nonconvex stochastic algorithm for decentralized optimization. In 2019 IEEE Data Science Workshop (DSW) , pages 315-321. IEEE, 2019.
- Konstantin Mishchenko, Grigory Malinovsky, Sebastian Stich, and Peter Richtárik. Proxskip: Yes! local gradient steps provably lead to communication acceleration! finally! In International Conference on Machine Learning , pages 15750-15769. PMLR, 2022.
- Giorgi Nadiradze, Amirmojtaba Sabour, Peter Davies, Shigang Li, and Dan Alistarh. Asynchronous decentralized sgd with quantized and local updates. Advances in Neural Information Processing Systems , 34:6829-6842, 2021.
- Angelia Nedic and Asuman Ozdaglar. Distributed subgradient methods for multi-agent optimization. IEEE Transactions on Automatic Control , 54(1):48-61, 2009.
- Angelia Nedic, Alex Olshevsky, and Wei Shi. Achieving geometric convergence for distributed optimization over time-varying graphs. SIAM Journal on Optimization , 27(4):2597-2633, 2017.
- Shi Pu, Alex Olshevsky, and Ioannis Ch Paschalidis. A sharp estimate on the transient time of distributed stochastic gradient descent. IEEE Transactions on Automatic Control , 67(11):59005915, 2021.
- Tiancheng Qin, S Rasoul Etesami, and César A Uribe. Communication-efficient decentralized local sgd over undirected networks. In 2021 60th IEEE Conference on Decision and Control (CDC) , pages 3361-3366. IEEE, 2021.
- Guannan Qu and Na Li. Harnessing smoothness to accelerate distributed optimization. IEEE Transactions on Control of Network Systems , 5(3):1245-1260, 2017.
- S Sundhar Ram, Angelia Nedi´ c, and Venugopal V Veeravalli. Distributed stochastic subgradient projection algorithms for convex optimization. Journal of optimization theory and applications , 147:516-545, 2010.
- Wei Shi, Qing Ling, Gang Wu, and Wotao Yin. Extra: An exact first-order algorithm for decentralized consensus optimization. SIAM Journal on Optimization , 25(2):944-966, 2015.
- Jinming Xu, Shanying Zhu, Yeng Chai Soh, and Lihua Xie. Convergence of asynchronous distributed gradient methods over stochastic networks. IEEE Transactions on Automatic Control , 63(2): 434-448, 2017.

408

409

410

411

412

413

414

415

416

- Chung-Yiu Yau and Hoi-To Wai. Fully stochastic distributed convex optimization on time-varying graph with compression. In 2023 62nd IEEE Conference on Decision and Control (CDC) , pages 145-150. IEEE, 2023.
- Xinlei Yi, Lisha Yao, Tao Yang, Jemin George, and Karl H Johansson. Distributed optimization for second-order multi-agent systems with dynamic event-triggered communication. In 2018 IEEE Conference on Decision and Control (CDC) , pages 3397-3402. IEEE, 2018.
- Xinlei Yi, Shengjun Zhang, Tao Yang, Tianyou Chai, and Karl H Johansson. Linear convergence of first-and zeroth-order primal-dual algorithms for distributed nonconvex optimization. IEEE Transactions on Automatic Control , 67(8):4194-4201, 2021.

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

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: [NA]

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: [NA]

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

495

496

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

Justification: [NA]

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

Justification: [NA]

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

522

Justification: [NA]

523

Guidelines:

524

525

- The answer NA means that paper does not include experiments requiring code.

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

549

550

551

552

553

554

555

556

557

558

559

560

561

562

563

564

565

566

567

568

569

570

571

572

573

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

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Due to limited computing resources and time constraints, we are unable to perform multiple runs of our algorithms and report the error bars. We will produce the error bar statistics if time permits.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).

574

575

576

577

578

579

580

581

582

583

584

585

586

587

588

589

590

591

592

593

594

595

596

597

598

599

600

601

602

603

604

605

606

607

608

609

610

611

612

613

614

615

616

617

618

619

620

621

622

- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: [NA]

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: [NA]

## Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

675

676

677

678

679

680

681

682

683

684

685

686

687

688

689

690

691

692

693

694

695

696

697

698

699

700

701

702

703

704

705

706

707

708

709

710

711

712

713

714

715

716

717

718

719

720

721

722

723

724

725

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

726

727

728

729

730

731

732

733

734

735

736

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.