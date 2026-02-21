12

13

14

15

16

17

18

19

20

## Accelerating First-Order Methods for Bilevel Optimization under General Smoothness

## Anonymous Author(s)

Affiliation Address email

## Abstract

Bilevel optimization is pivotal in machine learning applications such as hyperparameter tuning and adversarial training. While existing methods for nonconvexstrongly-convex bilevel optimization can find an /epsilon1 -stationary point under Lipschitz continuity assumptions, two critical gaps persist: improving algorithmic complexity and generalizing smoothness conditions. This paper addresses these challenges by introducing an accelerated framework under Hölder continuity-a broader class of smoothness that subsumes Lipschitz continuity. We propose a restarted accelerated gradient method that leverages inexact hypergradient estimators and establishes theoretical oracle complexity for finding /epsilon1 -stationary points. Empirically, experiments on data hypercleaning and hyperparameter optimization demonstrate superior convergence rates compared to state-of-the-art baselines.

## 1 Introduction

Bilevel optimization is a powerful paradigm with applications in various machine learning tasks, such as hyperparameter tuning [Franceschi et al., 2018, MacKay et al., 2019, Chen et al., 2024], adversarial training [Lin et al., 2020a,b, Wang et al., 2021, 2022], and reinforcement learning [Kunapuli et al., 2008, Yang et al., 2019, Hong et al., 2023]. It involves two levels of optimization, where the objective at the upper level depends on the solution to a lower-level optimization problem. The general bilevel problem can be expressed as:

<!-- formula-not-decoded -->

In this formulation, f ( x, y ) denotes the upper-level objective, while g ( x, y ) denotes the lower-level objective.

This study examines the nonconvex-strongly-convex framework, wherein the lower-level function 21 g ( x, y ) exhibits strong convexity with respect to y , while the upper-level function f ( x ) is possibly 22 nonconvex. In this case, the lower-level objective admits a unique solution Y ∗ ( x ) = { y ∗ ( x ) } . Then 23 Problem (1) is equivalent to minimizing the hyper-objective function 24

<!-- formula-not-decoded -->

As shown in Grazzi et al. [2020], Pedregosa [2016], the hyper-gradient ∇ ϕ ( x ) is given by: 25

The goal of this paper is to find the point x such that ϕ ( x ) is an /epsilon1 -stationary point, i.e., ‖∇ ϕ ( x ) ‖ ≤ /epsilon1 . 26 For nonconvex-strongly-convex bilevel optimization, previous work [Chen et al., 2023, Kwon et al., 27

<!-- formula-not-decoded -->

2023, Yang et al., 2023] primarily focuses on assuming Lipschitz continuity of ∇ f , ∇ g , ∇ 2 g , and 28 ∇ 3 g , and either approximates the hyper-gradient ∇ ϕ ( x ) or minimizes a penalty function. Approx29 imating the hyper-gradient ∇ ϕ ( x ) requires first-order oracle access to f and second-order oracle 30 access to g , whereas minimizing the penalty function only requires first-order oracle access to both 31 f and g . 32

33

34

35

36

Two key open questions remain: (i) For first-order methods, it remains open whether the existing algorithmic complexities for finding approximate first-order stationary points in nonconvex-stronglyconvex bilevel optimization can be further improved under high order smoothness, and (ii) whether the Lipschitz continuity assumptions can be generalized to the Hölder continuity.

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

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

## 1.1 Related Work

Nonconvex optimization: For unconstrained nonconvex objectives with Lipschtiz continuous gradient, the classical gradient descent (GD) is known to find an /epsilon1 -stationary point within O ( /epsilon1 -2 ) gradient computations [Nesterov, 2013]. This rate is optimal among the first-order methods [Cartis et al., 2010, Carmon et al., 2020]. Under the additional assumption of Lipschitz continuous Hessians, accelerated gradient descent (AGD) [Carmon et al., 2017, 2018, Jin et al., 2018] finds an /epsilon1 -stationary point in ˜ O ( /epsilon1 -7 / 4 ) evaluations. Li and Lin [2023] and Marumo and Takeda [2024a] further show that AGD with restarts achieves O ( /epsilon1 -7 / 4 ) complexity for finding /epsilon1 -stationary points, without additional log factors. Under the more general assumption of Hölder continuity of the Hessian, Marumo and Takeda [2024b] proposed a universal, parameter-free heavy-ball method equipped with two restart mechanisms, achieving a complexity bound of O ( H 1 / (2+2 ν ) ν /epsilon1 -(4+3 ν ) / (2+2 ν ) ) in terms of function and gradient evaluations, where ν ∈ [0 , 1] and H ν denote the Hölder exponent and constant, respectively.

Bilevel Optimization Methods: To approximate the hyper-gradient, gradient-based methods contain approximate implicit differentiation (AID) [Domke, 2012, Grazzi et al., 2020, Ji et al., 2021, Huang et al., 2025, Grazzi et al., 2020] and iterative differentiation (ITD) [Domke, 2012, Grazzi et al., 2020, Ji et al., 2021, Grazzi et al., 2020, Shaban et al., 2019]. Using the hyper-gradient (2), one can find an /epsilon1 -stationary point of ϕ ( x ) within ˜ O ( /epsilon1 -2 ) first-order oracle calls from f and ˜ O ( /epsilon1 -2 ) second-order oracle calls from g [Ghadimi and Wang, 2018, Ji et al., 2021]. In practical implementations, these methods typically rely on access to Jacobian or Hessian-vector product oracles. Kwon et al. [2023] proposed a fully first-order method that does not require Jacobian or Hessian-vector product oracles, and finds an /epsilon1 -stationary point using only first-order gradients of f and g . Inspired by Kwon et al. [2023]'s work, Chen et al. [2023] proposed a method that achieves a near-optimal convergence rate of ˜ O ( /epsilon1 -2 ) , which is comparable to second-order methods.

Table 1: Complexity bounds for finding /epsilon1 -stationary points under Lipschitz continuity assumptions.

O

O

| Algorithm                                                                                                                               | Gc( f , /epsilon1 )                                                                                                                                                  | Gc( g , /epsilon1 )                                                                                                                                                | JV( g , /epsilon1 )                                                                      | HV( g , /epsilon1 )                                                                        |
|-----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| AID-BiO (Ji et al. [2021]) ITD-BiO (Ji et al. [2021]) RAHGD (Yang et al. [2023]) F 2 BA(Chen et al. [2023]) Proposed method (this work) | O ( κ 3 /epsilon1 - 2 ) O ( κ 3 /epsilon1 - 2 ) ˜ O ( κ 11 / 4 /epsilon1 - 7 / 4 ) ˜ O ( /lscriptκ 4 /epsilon1 - 2 ) ˜ ( /lscript 3 / 4 κ 13 / 4 /epsilon1 - 7 / 4 ) | O ( κ 3 /epsilon1 - 2 ) O ( κ 4 /epsilon1 - 2 ) ˜ O ( κ 13 / 4 /epsilon1 - 7 / 4 ) ˜ O ( /lscriptκ 4 /epsilon1 - 2 ) ˜ ( /lscript 3 / 4 κ 13 / 4 /epsilon1 - 7 / 4 | O ( κ 3 /epsilon1 - 2 ) ˜ O ( κ 4 /epsilon1 - 2 ) ˜ O ( κ 11 / 4 /epsilon1 - 7 / 4 ) \ \ | ˜ O ( κ 3 /epsilon1 - 2 ) ˜ O ( κ 4 /epsilon1 - 2 ) ˜ O ( κ 13 / 4 /epsilon1 - 7 / 4 ) \ \ |

## 1.2 Our Contribution

In this paper, we propose an accelerated first-order algorithm for solving nonconvex-strongly convex bilevel optimization problems. Our main contributions are summarized as follows:

1. We introduce an accelerated first-order method framework-originally developed for nonconvex optimization-into the setting of nonconvex-strongly convex bilevel optimization, and consider more general Hölder continuity assumptions on f and g .
2. We prove that, with a carefully designed restart condition, the iterates generated by our proposed method remain uniformly bounded within each epoch. Based on this, we demonstrate that the algorithm is convergent with accelerated performance.

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

3. Even under the standard Lipschitz continuity setting, our method improves the first-order oracle complexity for finding an /epsilon1 -stationary point of ϕ ( x ) to ˜ O ( /lscript 3 / 4 κ 13 / 4 /epsilon1 -7 / 4 ) , without requiring access to second-order oracles, where /lscript and κ denote the problem's largest smoothness and condition number. This bound improves upon previously known results, as summarized in Table 1.
4. Our experimental results further support the theoretical convergence guarantees.

Organization. The rest of this work is organized as follows. Section 2 delineates the assumptions and specific algorithmic subroutines. Section 3 formally presents our proposed algorithm along with some basic lemmas. Section 4 provides a complexity bound for finding approximate first-order stationary points. In Section 5, we provide some numerical experiments to show the outstanding performance of our proposed method. Section 6 concludes the paper and discusses future directions. Technical analyses are deferred to the appendix.

Notation. Let a, b ∈ R d be vectors, where 〈 a, b 〉 represents their inner product and ‖ a ‖ denotes the Euclidean norm. For a matrix A ∈ R m × n , ‖ A ‖ is used to denote the operator norm, which is equivalent to the largest singular value of the matrix. Let Gc ( f, /epsilon1 ) and Gc ( g, /epsilon1 ) denote the number of gradient evaluations with respect to f and g , respectively. Let JV ( g, /epsilon1 ) denote the number of Jacobian-vector products ∇ 2 xy g ( x, y ) v , and HV ( g, /epsilon1 ) denote the number of Hessian-vector products ∇ 2 yy g ( x, y ) v . The diameter R of a compact set C is defined as R := max x 1 ,x 2 ∈ C ‖ x 1 -x 2 ‖ .

## 2 Preliminaries

In this section, we present the key definitions and assumptions used throughout the paper.

Definition 1 (Restricted Hölder Continuity) . Let h be a twice differentiable function. We say that ∇ 2 h is restrictively ( ν, H ν ) -Hölder continuous with diameter R &gt; 0 if

<!-- formula-not-decoded -->

- When R = + ∞ , we call ∇ 2 h is ( ν, H ν ) -Hölder continuous if ν ∈ [0 , 1] and H ν &lt; + ∞ . 92

93

We make the following assumptions on the upper-level function f and lower-level function g :

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

Assumption 1. We make the following assumptions:

- i. The function ϕ ( x ) is lower bounded.
- ii. The function g ( x, y ) is µ -strongly convex in y , and has L g -Lipschitz continuous gradients.
- iii. The function g ( x, y ) has ρ g -Lipschitz continuous Hessians and is ( ν g , M g ) -Hölder continuous in its third-order derivatives.
- iv. The function f ( x, y ) is C f -Lipschitz continuous in y and has L f -Lipschitz continuous gradients.
- v. The Hessian ∇ 2 xx f ( x, y ) is ( ν f , H f ) -Hölder continuous.
- vi. The mixed and second-order partial derivatives ∇ 2 xy f ( x, y ) , ∇ 2 yx f ( x, y ) , and ∇ 2 yy f ( x, y ) are ρ f -Lipschitz continuous.

The assumptions employed in this study are consistent with those commonly adopted in prior literature [Chen et al., 2023, Huang et al., 2025, Kwon et al., 2023, Yang et al., 2023]. To introduce Hölder continuity, we extend the Lipschitz continuity assumptions about the Hessian of f , and the third-order derivative of g to our assumptions (iii), (v), (vi).

Definition 2. Under Assumption 1, we define the largest smoothness constant as

<!-- formula-not-decoded -->

and the condition number as κ := /lscript/µ .

Observe that problem (1) can be reformulated as: 109

<!-- formula-not-decoded -->

where g ∗ ( x ) = g ( x, y ∗ ( x )) is the value function. A nature penalty problem associated with problem (3) is where λ &gt; 0 is a penalty parameter. This problem is equivalent to minimizing the following auxiliary function:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It has been proven in [Chen et al., 2023] that L ∗ λ ( x ) and ∇ L ∗ λ ( x ) asymptotically approximate ϕ ( x ) and ∇ ϕ ( x ) , respectively, as λ is sufficiently large. Moreover, ∇ L ∗ λ ( x ) is Lipschitz continuous and its Lipschitz constant does not involve λ . We restate their result below for completeness.

Lemma 1 (Chen et al. [2023, Lemma 4.1]) . Under Assumption 1, for λ ≥ 2 L f /µ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

iii. ∇ L /star λ ( x ) is O ( /lscriptκ 3 ) -Lipschitz continuous.

In the remainder of the article, we denote the Lipshitz continuous constant of ∇ L ∗ λ ( x ) in Lemma 1 by L = O ( /lscriptκ 3 ) for convenience. Then we introduce a lemma showing that ∇ 2 L ∗ λ ( x ) is restrictively ( ν f , H ν ) -Hölder continuous with diameter R , where the detailed expression of H ν , depending on λ and D , can be found in (16) of Appendix B.1.

<!-- formula-not-decoded -->

Lemma 2. Under Assumption 1, for λ ≥ 2 L f /µ , ∇ 2 L /star λ ( x ) is restrictly ( ν f , H ν ( λ, R )) -Hölder continuous with diameter R &gt; 0 , where

## 3 Restarted Accelerated gradient descent under General Smoothness

In this section, we present our algorithm in Algorithm 1 and discuss several of its key properties. The algorithm has a nested loop structure. The outer loop uses the accelerated gradient descent (AGD) method with a restart schemes, inspired from the recently works in Li and Lin [2023], Marumo and Takeda [2024a]. The iteration counter k is reset to 0 when AGD restarts, whereas the total iteration counter K is not. We refer to the period between a reset of k and the next reset as an epoch. We introduce a subscript t to denote the number of restarts. It is important to note that the subscript t in Algorithm 1 is primarily included to facilitate a simpler convergence analysis. Provided that no ambiguity occurs, we omit the subscript t , which means that the iterates are within the same epoch.

In Lines 4 and 5, we invoke AGD, which is summarized in Algorithm 2, to find estimators of y ∗ ( w t,k ) and y ∗ λ ( w t,k ) , respectively. AGD achieves linear convergence when applied to the minimization of smooth and strongly convex functions g ( x, · ) and f ( x, · ) + λg ( x, · ) . We note that the iteration number of inner AGD steps plays an important role in the complexity analysis. We will provide the parameters setting for AGD subroutines in Section 4. In the following, we describe some operations involved in the algorithm.

110

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

143

144

Restart Condition. Here, we focus on the iterates within a single epoch and omit the subscript t , which indexes different epochs. Then we define S k = ∑ k i =1 ‖ x i -x i -1 ‖ 2 , and the restart condition

<!-- formula-not-decoded -->

where the constant H ν will be defined in (6) below. If (5) holds, the epoch terminates; otherwise, 145 it continues. We say that an epoch ends at iteration k , if S k triggers the restart condition (5). It 146 is worth noting that unlike the restart condition in Li and Lin [2023], Yang et al. [2023], our restart 147 condition is independent of /epsilon1 . 148

149

150

151

152

153

154

## Algorithm 1 Restarted Accelerated gradient descent under General Smoothness (RAGD-GS)

- 1: Input: initial point x 0 , 0 ; gradient Lipschitz constant L &gt; 0 ; Hessian Hölder constant H ν &gt; 0 and ν f ∈ [0 , 1] ; momentum parameter θ k ∈ (0 , 1) ; parameters α, α ′ &gt; 0 , β, β ′ ∈ (0 , 1) , { T t,k } , { T ′ t,k } of AGD

<!-- formula-not-decoded -->

- 3: repeat
- 4: z t,k ← AGD( g ( w t,k , · ) , z t,k -1 , T t,k , α, β )
- 7: x t,k +1 ← w t,k -1 L u t,k

<!-- formula-not-decoded -->

- 8: w t,k +1 ← x t,k +1 + θ k +1 ( x t,k +1 -x t,k )
- 10: if ( k +1) 4+ ν f H 2 ν S ν f k &gt; L 2 then
- 9: k ← k +1 , K ← K +1
- 11: x t +1 , 0 ← x t,k

y

0

,

z

+1

- 14: end if
- 12: t +1 , -1 ← t 13: k ← 0 , t ← t +1
- 15: until ‖∇ L λ ( ¯ w t,k ) ‖ ≤ /epsilon1
- 16: Output: averaged solution ¯ w t,k defined by (7)

Hölder Constant H ν . From Lemma 2, ∇ 2 L /star λ ( x ) is restrictively ( ν f , H ν ( λ, R )) -Hölder continuous with diameter R &gt; 0 . Here we choose a specific R and the corresponding H ν ( λ, R ) , denoted by D and H ν , satisfying

The derivation of H ν and D is provided in (18) of Appendix C. Then ∇ 2 L ∗ λ ( x ) is restrictively ( ν f , H ν ) -Hölder continuous with diameter D . In the case of Lipschitz continuity, i.e., ν f = ν g = 1 , (6) implies H ν = O ( /lscriptκ 5 ) and D = O ( κ -2 ) .

Averaged Solution. Inspired by Marumo and Takeda [2024a], we set θ k = k k +1 and define 155

<!-- formula-not-decoded -->

where p k,i = 2( i +1) k ( k +1) . We can update ¯ w k in the following manner: ¯ w k = k -1 k +1 ¯ w k -1 + 2 k +1 w k -1 .

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

Remark 1. It is noteworthy that Condition 1 holds in Algorithm 1 as long as the inner loop iteration number T t,k and T ′ t,k are large enough. This will be formally addressed in our convergence analysis later, in Theorem 2.

Under Condition 1, the bias of ∇ L ∗ λ ( w t,k ) and its estimator ˆ ∇ L ∗ λ ( w t,k ) can be bounded as shown 167 below: 168

The following lemma shows that { x i } k -1 i =0 and { w i } k -1 i =0 are bounded within any epoch ending at iteration k .

Lemma 3. Let Assumption 1 holds, H ν and D = R be given in (6) , and ¯ w k be defined in (7) . For any epoch ending at iteration k , the following holds:

<!-- formula-not-decoded -->

Condition 1 (Inexact gradients) . Under Assumption 1 and given σ &gt; 0 , we assume that the estimators y t,i and z t,i satisfy the conditions

<!-- formula-not-decoded -->

for any t -th epoch ending at iteration k , where i = 0 , . . . , k -1 .

<!-- formula-not-decoded -->

,

-

1

←

,

0

w

t

+1

,

0

←

x

t

+1

,

0

<!-- formula-not-decoded -->

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

185

186

for any t -th epoch ending at iteration k , where i = 0 , . . . , k -1 .

## 4 Complexity Analysis

In this section, we analyze the performance of Algorithm 1. We begin in Section 4.1 by presenting several useful lemmas that rely on the boundedness of the iterates generated within a single epoch. These results serve as key tools for our subsequent analysis. We then establish the descent property of the objective function and derive an upper bound for ‖∇ L ∗ λ ( w k ) ‖ for all k ≥ 2 . Finally, in Section 4.2, we present the main complexity results for Algorithm 1.

## 4.1 Tools for Analysis

We use the following two Hessian-free inequalities to analyze the complexity of Algorithm 1.

Lemma 5. Under Assumption 1 and with λ ≥ 2 L f /µ , the following holds for any x 1 , . . . , x n satisfying max 1 ≤ i ≤ j ≤ n ‖ x i -x j ‖ ≤ D and q 1 , . . . , q n ≥ 0 such that n q =1 q i = 1 :

where H ν and D are defined in (6).

<!-- formula-not-decoded -->

Lemma6. Under Assumption 1 and with λ ≥ 2 L f /µ , the following holds for any x and x ′ satisfying ‖ x -x ′ ‖ ≤ D :

<!-- formula-not-decoded -->

where H ν and D are defined in (6).

To analyze the behavior of L ∗ λ ( · ) in one epoch, we define the potential function Φ k as follows, following Marumo and Takeda [2024a]:

<!-- formula-not-decoded -->

The following lemma shows that Φ k is a decreasing sequence if ‖ x k -x k -1 ‖ and σ are sufficiently 187 small. 188

Lemma 7. Suppose that Assumption 1, Condition 1, and λ ≥ 2 L f /µ hold. Then we have 189

<!-- formula-not-decoded -->

Lemma 8. Suppose that Assumption 1, Condition 1, and λ ≥ 2 L f /µ hold. Then the decrease value 190 of L ∗ λ ( · ) in one epoch satisfies: 191

<!-- formula-not-decoded -->

Lemma 8 shows that, if we use exact gradient ∇ L ∗ λ ( x ) , the objective function value L ∗ λ ( x ) always 192 decreases as long as S k &gt; 0 . The following lemma provide an upper bound on the gradient norm. 193

Lemma 9. Suppose that Assumption 1, Condition 1, and λ ≥ 2 L f /µ hold. The following is true 194 when k ≥ 2 : 195

196

where c = 2 √ 6 + 27 .

<!-- formula-not-decoded -->

197

198

199

200

<!-- formula-not-decoded -->

## 4.2 Main results

In the following proposition, we show that the iteration complexity of the outer loop is bounded.

Proposition 1. Suppose that Assumption 1, Condition 1, and λ ≥ 2 L f /µ hold. Let c = 2 √ 6 + 27 as defined in Lemma 9, and define ∆ λ = L ∗ λ ( x 0 , 0 ) -min x ∈ R dx L ∗ λ ( x ) . Let

Algorithm 1 terminates within 201

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

total iterations, outputting ¯ w t,k satisfying ‖∇ L ∗ λ ( ¯ w t,k ) ‖ ≤ /epsilon1 . Moreover, Algorithm 1 terminates within

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

228

epochs.

We present the complexity analysis of our algorithm, aiming to establish its guarantee for finding an O ( /epsilon1 ) -stationary point of Problem (1).

Theorem 1. Suppose that both Assumption 1 and Condition 1 hold. Define ∆ = ϕ ( x 0 , 0 ) -min x ∈ R dx ϕ ( x ) . Let λ = max( O ( κ ) , O ( /lscriptκ 3 ) //epsilon1, O ( /lscriptκ 2 ) / ∆) and set the other parameters as specified in (12) , Algorithm 1 terminates within

<!-- formula-not-decoded -->

iterates, outputting ¯ w t,k satisfying ‖∇ ϕ ( ¯ w k ) ‖ ≤ 2 /epsilon1 . Moreover, Algorithm 1 terminates within epochs.

When ν f = ν g = 1 , Theorem 1 shows that within O ( ∆ /lscript 3 / 4 κ 11 / 4 /epsilon1 -7 / 4 ) outer iterations and O (∆ /lscript 1 / 2 κ 5 / 2 /epsilon1 -3 / 2 ) epochs, the algorithm will find an O ( /epsilon1 ) -stationary point. It is better than the corresponding result in Yang et al. [2023], Chen et al. [2023], as shown in Table 1.

Remark 2. Throughout the proof, we only use the restricted Hölder and Lipschitz properties, where restricted Lipschitz continuity can be defined analogously to Definition 1. Therefore, the assumption on global Lipschitz and Hölder smoothness in Assumption 1 can be relaxed to restricted smoothness.

To make Condition 1 hold, it suffices to run AGD for a sufficiently large number of iterations, which only introduces a logarithmic factor to the total complexity. This gives the following result.

Theorem 2. Suppose that Assumption 1 holds. In the t -th epoch, we set the inner-loop iteration numbers T t,k and T ′ t,k according to (44) , (45) , (46) , and (47) in Appendix D. We then run Algorithm 1 with the parameters specified in Theorem 1. Under these settings, all y t,k and z t,k satisfy Condition 1. Moreover, the total first-order oracle complexity is

<!-- formula-not-decoded -->

We defer the proof to Appendix D. Under the Hölder continuity assumption, to the best of our knowledge, we are the first to propose a method that finds an /epsilon1 -stationary point. Furthermore, under the Lipschitz continuity assumption, our approach outperforms all existing methods in the literature, as the proposed method RAGD-GS relies solely on first-order oracle information.

and when ν f = ν g = 1 , the first-order oracle complexity is ˜ O ( ∆ /lscript 3 / 4 κ 13 / 4 /epsilon1 -7 / 4 ) .

<!-- formula-not-decoded -->

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

246

247

248

249

250

251

252

253

## 5 Numerical Experiment

This section compares the performance of the proposed method with several existing methods, including RAHGD Yang et al. [2023], BA (Ghadimi and Wang [2018]), AID (Ji et al. [2021]), ITD (Ji et al. [2021]) and F 2 BA Chen et al. [2023]. For the bilevel approximation (BA) method introduced in Ghadimi and Wang [2018], we implement a conjugate gradient approach to compute Hessian-vector products since the original work doesn't specify this computational detail. We refer to this modified version as BA-CG to distinguish it from other algorithm. Our experiments were conducted on a PC with Intel Core i7-13650HX CPU (2.60GHz, 20 cores), 24GB RAM, and the platform is 64-bit Windows 11 Home Edition (version 26100).

## 5.1 Data Hypercleaning

Data hypercleaning (Franceschi et al. [2017]; Shaban et al. [2019]) is a bilevel optimization problem aimed at cleaning noisy labels in datasets. The cleaned data forms the validation set, while the rest serves as the training set. The problem is formulated as:

<!-- formula-not-decoded -->

where D tr and D val are the training and validation sets, respectively, W is the weight matrix of the classifier, σ ( · ) is the sigmoid function, and C r is a regularization parameter. In our experiments, we follow Franceschi et al. [2017] and set C r = 0 . 001 .

For MNIST LeCun et al. [1998], we used |D tr | = 20 , 000 training samples (partially noisy) and |D val | = 5 , 000 clean validation samples, with corruption rate p indicating the ratio of noisy labels in the training set. In Figures 1 and 2, inner and outer learning rates are searched over {0.001, 0.01, 0.1, 1, 10, 100}. For all methods except BA, inner GD/AGD steps are from {50, 100, 200, 500}; for BA, we choose GD steps from { ⌈ c ( k +1) 1 / 4 ⌉ : c ∈ { 0 . 5 , 1 , 2 , 4 }} as in Ghadimi and Wang [2018]. For F 2 BA and our method, λ is selected from {100, 300, 500, 700}. The results, shown in Figures 1 and 2, demonstrate that our proposed method achieves acceleration effects comparable to those in Yang et al. [2023], and outperforms all other methods.

Figure 1: Corruption rate p = 0 . 2

<!-- image -->

## 5.2 Hyperparameter Optimization

Hyperparameter optimization is a bilevel optimization task aimed at minimizing the validation 254 loss. We compare our proposed algorithms with baseline algorithms on the 20 Newsgroups 255 dataset [Grazzi et al., 2020], which consists of 18,846 news articles divided into 20 topics, with 256 130,170 sparse tf-idf features. The dataset is split into training, validation, and test sets with sizes 257 |D tr | = 5 , 657 , |D val | = 5 , 657 , and |D test | = 7 , 532 , respectively. The optimization problem is 258

17.5

15.0

12.5

10.0

7.5

Train Loss

5.0

2.5

RAGD-GS

AID-BiO

RAHGD

BA-CG

F2BA

ITD-BiO

60

0

20

40

80

100

120

140

running time  (s)

Figure 2: Corruption rate p = 0 . 4

formulated as: 259

<!-- formula-not-decoded -->

For the evaluation in Figure 3, inner and outer learning rates are selected from {0.001, 0.01, 260 0.1, 1, 10, 100}, and GD/AGD steps from {5, 10, 30, 50}. For BA, we choose GD steps from 261 { ⌈ c ( k +1) 1 / 4 ⌉ : c ∈ { 0 . 5 , 1 , 2 , 4 }} as in Ghadimi and Wang [2018]. For F 2 BA and our method, λ 262 is chosen from {100, 300, 500, 700}. As shown in Figure 3, our proposed method exhibits perfor263 mance comparable to that of Yang et al. [2023], while significantly outperforming other competing 264 algorithms by converging faster and reaching a lower test loss. 265

Figure 3: Results of test loss and test accuracy evaluated on the test set.

<!-- image -->

## 6 Conclusion

266

- This work introduces an accelerated first-order method framework for solving nonconvex-strongly 267 convex bilevel optimization problems, extending techniques from nonconvex optimization to a 268
- broader setting under generalized Hölder continuity assumptions on both the upper-level and lower269
- level objectives. We show that, with a carefully designed restart condition, the iterates remain uni270
- formly bounded within each epoch, ensuring both stability and convergence. In addition, we provide 271
- first-order oracle complexity bounds along with rigorous error analysis and convergence guarantees. 272
- Our theoretical results are further supported by empirical evidence, demonstrating the effectiveness 273
- and robustness of the proposed algorithm. An important open question is whether a fully first-order 274
- method can find an /epsilon1 -approximate second-order stationary point without using /epsilon1 -dependent parame275
- ters, which we leave for future work. 276

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

316

317

318

319

320

321

322

## References

- Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. 'convex until proven guilty': Dimension-free acceleration of gradient descent on non-convex functions. In International conference on machine learning , pages 654-663. PMLR, 2017.
- Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. Accelerated methods for nonconvex optimization. SIAM Journal on Optimization , 28(2):1751-1772, 2018.
- Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points i. Mathematical Programming , 184(1):71-120, 2020.
- Coralia Cartis, Nicholas IM Gould, and Ph L Toint. On the complexity of steepest descent, newton's and regularized newton's methods for nonconvex unconstrained optimization problems. Siam journal on optimization , 20(6):2833-2852, 2010.
- He Chen, Haochen Xu, Rujun Jiang, and Anthony Man-Cho So. Lower-level duality based reformulation and majorization minimization algorithm for hyperparameter optimization. In International Conference on Artificial Intelligence and Statistics , pages 784-792. PMLR, 2024.
- Lesi Chen, Yaohua Ma, and Jingzhao Zhang. Near-optimal nonconvex-strongly-convex bilevel optimization with fully first-order oracles. arXiv preprint arXiv:2306.14853 , 2023.
- Justin Domke. Generic methods for optimization-based modeling. In Artificial Intelligence and Statistics , pages 318-326. PMLR, 2012.
- Luca Franceschi, Michele Donini, Paolo Frasconi, and Massimiliano Pontil. Forward and reverse gradient-based hyperparameter optimization. In International conference on machine learning , pages 1165-1173. PMLR, 2017.
- Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International conference on machine learning , pages 1568-1577. PMLR, 2018.
- Saeed Ghadimi and Mengdi Wang. Approximation methods for bilevel programming. arXiv preprint arXiv:1802.02246 , 2018.
- Riccardo Grazzi, Luca Franceschi, Massimiliano Pontil, and Saverio Salzo. On the iteration complexity of hypergradient computation. In International Conference on Machine Learning , pages 3748-3758. PMLR, 2020.
- Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale stochastic algorithm framework for bilevel optimization: Complexity analysis and application to actor-critic. SIAM Journal on Optimization , 33(1):147-180, 2023.
- Minhui Huang, Xuxing Chen, Kaiyi Ji, Shiqian Ma, and Lifeng Lai. Efficiently escaping saddle points in bilevel optimization. Journal of Machine Learning Research , 26(1):1-61, 2025.
- Kaiyi Ji, Junjie Yang, and Yingbin Liang. Bilevel optimization: Convergence analysis and enhanced design. In International conference on machine learning , pages 4882-4892. PMLR, 2021.
- Chi Jin, Praneeth Netrapalli, and Michael I Jordan. Accelerated gradient descent escapes saddle points faster than gradient descent. In Conference On Learning Theory , pages 1042-1085. PMLR, 2018.
- Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. SIAM review , 51(3): 455-500, 2009.
- Gautam Kunapuli, Kristin P Bennett, Jing Hu, and Jong-Shi Pang. Classification model selection via bilevel programming. Optimization Methods &amp; Software , 23(4):475-489, 2008.
- Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert D Nowak. A fully first-order method for stochastic bilevel optimization. In International Conference on Machine Learning , pages 18083-18113. PMLR, 2023.

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

- Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE , 86(11):2278-2324, 1998.
- Huan Li and Zhouchen Lin. Restarted nonconvex accelerated gradient descent: No more polylogarithmic factor in the in the o (epsilonˆ(-7/4)) complexity. Journal of Machine Learning Research , 24(157):1-37, 2023.
- Tianyi Lin, Chi Jin, and Michael Jordan. On gradient descent ascent for nonconvex-concave minimax problems. In International conference on machine learning , pages 6083-6093. PMLR, 2020a.
- Tianyi Lin, Chi Jin, and Michael I Jordan. Near-optimal algorithms for minimax optimization. In Conference on learning theory , pages 2738-2779. PMLR, 2020b.
- Matthew MacKay, Paul Vicol, Jon Lorraine, David Duvenaud, and Roger Grosse. Self-tuning networks: Bilevel optimization of hyperparameters using structured best-response functions. arXiv preprint arXiv:1903.03088 , 2019.
- Naoki Marumo and Akiko Takeda. Parameter-free accelerated gradient descent for nonconvex minimization. SIAM Journal on Optimization , 34(2):2093-2120, 2024a.
- Naoki Marumo and Akiko Takeda. Universal heavy-ball method for nonconvex optimization under hölder continuous hessians. Mathematical Programming , pages 1-29, 2024b.
- Yurii Nesterov. Introductory lectures on convex optimization: A basic course , volume 87. Springer Science &amp; Business Media, 2013.
- Fabian Pedregosa. Hyperparameter optimization with approximate gradient. In International conference on machine learning , pages 737-746. PMLR, 2016.
- Amirreza Shaban, Ching-An Cheng, Nathan Hatch, and Byron Boots. Truncated back-propagation for bilevel optimization. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 1723-1732. PMLR, 2019.
- Jiali Wang, He Chen, Rujun Jiang, Xudong Li, and Zihao Li. Fast algorithms for stackelberg prediction game with least squares loss. In International Conference on Machine Learning , pages 10708-10716. PMLR, 2021.
- Jiali Wang, Wen Huang, Rujun Jiang, Xudong Li, and Alex L Wang. Solving stackelberg prediction game with least squares loss via spherically constrained least squares reformulation. In International conference on machine learning , pages 22665-22679. PMLR, 2022.
- Haikuo Yang, Luo Luo, Chris Junchi Li, and Michael I Jordan. Accelerating inexact hypergradient descent for bilevel optimization. arXiv preprint arXiv:2307.00126 , 2023.
- Zhuoran Yang, Yongxin Chen, Mingyi Hong, and Zhaoran Wang. Provably global convergence of actor-critic: A case for linear quadratic regulator with ergodic cost. Advances in neural information processing systems , 32, 2019.

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

## A Notations for Tensors

We adopt the tensor notation from Kolda and Bader [2009]. For a three-way tensor X ∈ R d 1 × d 2 × d 3 , the entry at ( i 1 , i 2 , i 3 ) is denoted by [ X ] i 1 ,i 2 ,i 3 . The inner product between X and Y is defined as

The operator norm is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where [ x 1 ◦ x 2 ◦ x 3 ] i 1 ,i 2 ,i 3 := [ x 1 ] i 1 [ x 2 ] i 2 [ x 3 ] i 3 . This definition generalizes the matrix spectral norm and the Euclidean norm for vectors to three-way tensors. Let X ∈ R d 1 × d 2 × d 3 be a three-way tensor, and let A ∈ R d ′ 1 × d 1 be a matrix. The mode-1 product of X and A , denoted by X × 1 A ∈ R d ′ 1 × d 2 × d 3 , is defined component-wise as

<!-- formula-not-decoded -->

Mode-2 and mode-3 products, denoted by X × 2 B and X × 3 C , are defined analogously for matrices B ∈ R d ′ 2 × d 2 and C ∈ R d ′ 3 × d 3 , respectively. Moreover, the operator norm satisfies the submultiplicative property under modei multiplication:

<!-- formula-not-decoded -->

## B Proof of lemmas in Section 2

Lemma B.1 (Lemma B.2 by Chen et al. [2023]) . Under Assumption 1, for λ ≥ 2 L f /µ , it holds that ‖ y /star λ ( x ) -y /star ( x ) ‖ ≤ C f λµ .

<!-- formula-not-decoded -->

Lemma B.2 (Lemma B.5 by Chen et al. [2023]) . Under Assumption 1, for λ ≥ 2 L f /µ , it holds that ‖∇ y /star ( x ) -∇ y /star λ ( x ) ‖ ≤ D 2 /λ , where

Lemma B.3 (Lemma B.6 by Chen et al. [2023]) . Under Assumption 1, for λ ≥ 2 L f /µ , it holds that ‖∇ y ∗ ( x ) ‖ ≤ L g /µ , ‖∇ y /star λ ( x ) ‖ ≤ 4 L g /µ .

This implies that y ( x ) is ( L /µ ) -Lipschitz continuous, y ( x ) is (4 L /µ ) -Lipschitz continuous.

377

where 378

<!-- formula-not-decoded -->

Proof. We begin by differentiating the identity 379

<!-- formula-not-decoded -->

with respect to x . This yields 380

<!-- formula-not-decoded -->

≥ f

∗ g ∗ λ g Lemma B.4. Under Assumption 1, for λ 2 L /µ , we have

<!-- formula-not-decoded -->

Rearranging terms to isolate ∇ 2 y ∗ ( x ) , we obtain 381

<!-- formula-not-decoded -->

Analogously, we have 382

<!-- formula-not-decoded -->

Next, we estimate the difference between the corresponding third-order derivatives in the original 383 and penalized problems. To begin with, we observe that 384

<!-- formula-not-decoded -->

Similarly, for the mixed partial derivative and its contraction with ∇ y ∗ ( x ) , we have 385

<!-- formula-not-decoded -->

Furthermore, we control the error in the third-order term involving two contractions: 386

<!-- formula-not-decoded -->

Combining the above inequalities, we are now ready to bound the difference between the second 387 derivatives: 388

<!-- formula-not-decoded -->

389

390

391

/unionsq /intersectionsq

LemmaB.5. Under Assumption 1, for λ ≥ 2 L f /µ , the mappings ∇ y ∗ ( x ) and ∇ y ∗ λ ( x ) are Lipschitz continuous with constants ( 1 + L g µ ) 2 ρ g µ and ( 1 + 4 L g µ ) 2 ( 2 ρ g µ + ρ f L f ) , respectively.

392

Proof. Recall that and 393

<!-- formula-not-decoded -->

By (13) and (14), we can obtain the Lipschitz constants of ∇ y ∗ ( x ) and ∇ y ∗ λ ( x ) by directly bounding 394 ‖∇ 2 y ∗ ( x ) ‖ and ‖∇ 2 y ∗ λ ( x ) ‖ . Specifically, we have 395

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here we use Lemma B.3, λ ≥ 2 L f /µ , ‖∇ 3 xxy g ( x, y ) ‖ ≤ ρ g , ‖∇ 3 xyy g ( x, y ) ‖ ≤ ρ g , 396 ‖∇ 3 yyy g ( x, y ) ‖ ≤ ρ g , ‖∇ 2 yy g ( x, y ) ‖ ≥ µ , ‖∇ 2 yy L λ ( x, y ) ‖ ≥ 1 2 λµ , ‖∇ 3 xxy f ( x, y ) ‖ ≤ ρ f , 397 ‖∇ 3 xyy f ( x, y ) ‖ ≤ ρ f and ‖∇ 3 yyy f ( x, y ) ‖ ≤ ρ f . 398

399

400

## B.1 Proof of Lemma 2

Proof. We decompose ∇ 2 L ∗ λ ( x ) into two components: 401

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/unionsq /intersectionsq where 402

and 403

To analyze the variation of A ( x ) , we observe: 404

<!-- formula-not-decoded -->

The first step applies the triangle inequality. The second step relies on the ( ν f , H f ) -Hölder continu405 ity of ∇ 2 xx f , the bound ∇ 2 yx f ( · , · ) /precedesequal L f , and Lemma B.2. Here, C 1 = O ( /lscriptκ ν f ) , C 2 = O ( /lscriptκ 3 ) . 406

Next, we evaluate ∇ B ( x ) by differentiating: 407

<!-- formula-not-decoded -->

To bound the Lipschitz constant of B ( x ) , we control ‖∇ B ( x ) ‖ as follows: 408

<!-- formula-not-decoded -->

Using the smoothness and Hölder continuity assumptions on g , as well as bounds from Lemma B.1, 409 Lemma B.2, and Lemma B.4, we arrive at: 410

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote the entire right-hand side as C 3 = O ( λ 1 -ν g /lscriptκ 4+ ν g ) . Finally, we estimate the restricted 411 Hölder constant of ∇ 2 L ∗ λ ( x ) : 412

Define 413

<!-- formula-not-decoded -->

Thus, ∇ 2 L /star λ ( x ) is restrictively ( ν f , H ν ( λ, R )) -Hölder continuous with diameter R . In the case 414 ν f = 1 and ν g = 1 , this implies ∇ 2 L /star λ ( x ) is O ( /lscriptκ 5 ) -Lipschitz continuous. /unionsq /intersectionsq 415

## C Proof of lemmas in Section 3 416

## C.1 AGD subroutines 417

## Algorithm 2 AGD( h, z 0 , T, α, β )

- 1: Input: objective function h ( · ) ; start point z 0 ; iteration number T ≥ 1 ; step-size α &gt; 0 ; momentum parameter β ∈ (0 , 1)
- 3: for t = 0 , . . . , T -1 do
- 2: ˜ z 0 ← z 0
- 4: z t +1 ← ˜ z t -α ∇ h (˜ z t )
- 6: end for
- 5: ˜ z t +1 ← z t +1 + β ( z t +1 -z t )
- 7: Output: z T

This method boasts an optimal convergence rate as shown below: 418

- Lemma C.1 (Nesterov [2013], Section 2) . Running Algorithm 2 on an /lscript h -smooth and µ h -strongly 419 convex objective function h ( · ) with α = 1 //lscript h and β = ( √ κ h -1 ) / ( √ κ h +1 ) produces an output 420 z T satisfying 421

422

where z ∗ = arg min z h ( z ) and κ h = /lscript h /µ h denotes the condition number of the objective h .

## C.2 Proof of Lemma 3 423

Proof. Consider an epoch ending at iteration k ≥ 2 . By applying the Cauchy-Schwarz inequality 424 to the restart condition (5), we obtain 425

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies that the diameter of conv ( { x i } k -1 i =0 ) is less than ( L H ν ) 1 ν f . By solving a system of equa426 tions: 427

where H ν ( λ, R ) is defined in (16). We have 428

429

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

430

431

432

433

434

435

436

437

## C.3 Proof of Lemma 4

Proof. Consider the exact gradient of L ∗ λ ( · ) :

and the inexact gradient estimator used by Algorithm 1:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the triangle inequality, the Lipschitz continuity assumptions in Condition 1, and the condition 438 L f ≤ 1 2 λµ ≤ λL g , we obtain: 439

<!-- formula-not-decoded -->

## D Proof of lemmas in Section 4 441

Lemma D.1. Under Assumption 1 and with λ ≥ 2 L f /µ , the following holds for any x and x ′ : 442

<!-- formula-not-decoded -->

## D.1 Proof of Lemma 5 443

<!-- formula-not-decoded -->

Computing the weighted average sum, we have 445

<!-- formula-not-decoded -->

Denote this specific R by D . The boundedness of { x i } k -1 i =1 has been ensured by (17). From line 8 in Algorithm 1, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last inequality holds due to θ k ∈ (0 , 1) . So where ¯ w k is defined in (7). The first inequality holds because ¯ w k ∈ conv( { w i } k -1 i =0 ) , and the maximum diameter of the convex hull is attained by a pair of its vertices.

/unionsq

/intersectionsq

<!-- formula-not-decoded -->

## D.2 Proof of Lemma 6 450

The second inequality holds due to ‖ x i -¯ x ‖ ≤ max 1 ≤ i ≤ j ≤ n ‖ x i -x j ‖ ≤ D , Lemma 2 and 447 equation (6). The last inequality uses Hölder inequality. The last equality holds due to ∑ n i =1 q i = 1 448 and ∑ n i =1 q i ‖ x i -¯ x ‖ 2 = ∑ 1 ≤ i&lt;j ≤ n q i q j ‖ x i -x j ‖ 2 . /unionsq /intersectionsq 449

Proof.

<!-- formula-not-decoded -->

The last inequality follows from Lemma 5 by setting n = 2 , ( x 1 , x 2 ) = ( x, x ′ ) , and ( q 1 , q 2 ) = 451 ( t, 1 -t ) . 452 /unionsq /intersectionsq 453

## D.3 Proof of Lemma 7 454

Proof. Let 455

From Lemma D.1, we have 456

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Lemma 6 and Lemma 3, it follows that ‖ w k -x k ‖ ≤ ‖ x k -x k -1 ‖ ≤ D and 457

<!-- formula-not-decoded -->

By summing inequalities (20) and (21), we evaluate the expression as follows 458

<!-- formula-not-decoded -->

To evaluate the first term on the right-hand side, we decompose it into four terms: 459

<!-- formula-not-decoded -->

Let n = 2 , q 1 = 1 / (1 + θ k ) , q 2 = θ k / (1 + θ k ) in Lemma 5, we have 460

<!-- formula-not-decoded -->

Now, we proceed to evaluate (A), (B), (C) and (D) respectively. 461

<!-- formula-not-decoded -->

Here we use equality 2 〈 a, b 〉 = 1 L ‖ a ‖ 2 + L ‖ b ‖ 2 -L ∥ ∥ b -1 L a ∥ ∥ 2 , x k +1 = w k -1 L ˆ ∇ L ∗ λ ( w k ) , w k = 462 x k + θ k ( x k -x k -1 ) and (23). Plugging the evaluations into (22), we have 463

<!-- formula-not-decoded -->

Next, to bound the last term on the right-hand side of (24), by triangle inequality and (23), we have 464

<!-- formula-not-decoded -->

Squaring both sides yields 465

<!-- formula-not-decoded -->

and 466

<!-- formula-not-decoded -->

Here we use the inequalities ( a + b ) 2 ≤ (1 + 1 θ k ) a 2 + (1 + θ k ) b 2 and ( a + b ) 2 ≤ 2( a 2 + b 2 ) . 467 Rearranging the terms yields 468

<!-- formula-not-decoded -->

By plugging this bound into (24): we obtain 469

<!-- formula-not-decoded -->

Considering (9), (25) and θ k ≤ 1 , we have 470

Φ k +1 -Φ k ≤ L ∗ λ ( x k +1 ) -L ∗ λ ( x k ) + θ 2 k +1 2 ( P k +1 + 1 2 L ‖∇ L ∗ λ ( x k ) ‖ 2 + L ‖ x k +1 -x k ‖ 2 ) -θ 2 k 2 ( P k + 1 2 L ‖∇ L ∗ λ ( x k -1 ) ‖ 2 + L ‖ x k -x k -1 ‖ 2 ) ≤‖ x k -x k -1 ‖ 2+ ν f ( 2 H ν (1 + ν f )(2 + ν f )(3 + ν f ) θ 2+ ν f k + H ν 1 + ν f θ 3+ ν f 2 k ) + ‖ x k -x k -1 ‖ 2+2 ν f 2 H 2 ν (1 + ν f ) 2 θ 2+ ν f k L + θ 2 k +1 -θ k 2 P k +1 + θ 2 k +1 -θ k (1 + θ k ) 4 L ‖∇ L ∗ λ ( x k ) ‖ 2 + σ 2 2 L + σ ‖ x k +1 -x k ‖ . From Young's inequalities and θ 2 k +1 -θ k ≤ 0 , we have 471 -P k +1 = -〈∇ L ∗ λ ( x k ) , x k +1 -x k 〉 ≤ 1 2 L ‖∇ L ∗ λ ( x k ) ‖ 2 + L 2 ‖ x k +1 -x k ‖ 2 . Finally, we derive the inequality below: 472 Φ k +1 -Φ k ≤‖ x k -x k -1 ‖ 2+ ν f ( 2 H ν (1 + ν f )(2 + ν f )(3 + ν f ) θ 2+ ν f k + H ν 1 + ν f θ 3+ ν f 2 k ) + ‖ x k -x k -1 ‖ 2+2 ν f 2 H 2 ν (1 + ν f ) 2 θ 2+ ν f k L + θ 2 k +1 + θ k -2 4 L ‖ x k +1 -x k ‖ 2 -θ 2 k 4 L ‖∇ L ∗ λ ( x k ) ‖ 2 + σ 2 2 L + σ ‖ x k +1 -x k ‖ . /unionsq /intersectionsq 473

## D.4 Proof of Lemma 8 474

Proof. Summing Lemma 7 from i = 0 , . . . , k -1 and telescoping yields 475

The second inequality holds due to { θ k } is non-decreasing and non-negative. Moreover, by the 476 definition of Φ k in (9) , we have 477

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Power-Mean Inequality, we have 478

<!-- formula-not-decoded -->

Substituting (27), (28), and (29) into (26), we obtain 479

<!-- formula-not-decoded -->

Applying the restart condition (5) and noting that S k -1 ≤ S k , we further obtain 480

<!-- formula-not-decoded -->

Since 0 ≤ ν f ≤ 1 , and 481

we obtain 482

<!-- formula-not-decoded -->

483

## D.5 Proof of Lemma 9 484

Proof. Define 485

486

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/unionsq /intersectionsq

From ¯ w k ∈ conv ( { w i } k -1 i =0 ) , Lemma 3 and Lemma 5, we have 487

<!-- formula-not-decoded -->

Here we use inequality p k,i ≤ p k,k -1 = 1 /Z k = 2 / ( k +1) for all 0 ≤ i &lt; k . Regarding the last 488 term in (30), we have 489

<!-- formula-not-decoded -->

The above inequalities hold by the triangle inequality, 0 ≤ θ k ≤ 1 and Cauchy-Schwarz inequality, 490 respectively. Then 491

<!-- formula-not-decoded -->

Plugging (31) into (30), we have 492

<!-- formula-not-decoded -->

Then for k ≥ 2 , combing with (32), we have 493

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that Z k = k +1 2 and k 3 12 ≤ ∑ Z 2 i ≤ k 3 6 , we have 494

where c is a constant, c = 2 √ 6+27 . The last inequality holds due to ∑ k -1 i =1 i 5 2 + ν f 2 ≤ 1 2 k 7 2 + ν f 2 . /unionsq /intersectionsq 495

## D.6 Proof of Proposition 1

496

Proof. Consider an epoch ends at iteration k and ignore the subscript t . If ¯ w k is not an /epsilon1 -first-order 497 stationary point and k ≥ 2 , from Lemma 9, we have: 498

/epsilon1 ≤ σ + cL √ S k -1 /k 3 ≤ σ + cL √ S k /k 3 . If k = 1 , σ + cL √ S k /k 3 = σ + cL ‖ x 1 -x 0 ‖ = σ + c ‖ ˆ ∇ L ∗ λ ( x 0 ) ‖ ≥ /epsilon1 . Here we set σ = 1 64 c +1 /epsilon1 , 499 the above inequality is 500

From (33), We have 501

<!-- formula-not-decoded -->

502

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From restart condition (5), we have 503

<!-- formula-not-decoded -->

Then we can bound S k as: 504

<!-- formula-not-decoded -->

From Lemma 8, (34) and (35), in this epoch, decrease of L ∗ λ ( x ) is 505

<!-- formula-not-decoded -->

Sum above inequality over all epochs and denote the number of total iterates as K , we have 506

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As a result, we can denote the expression in the right side of (37) as K max . Substitute H ν = 507 λ ν f (1 -ν g ) O ( /lscriptκ 3+(1+ ν g ) ν f ) and L = O ( /lscriptκ 3 ) for (37), we have 508

We can also bound S k as: 509

<!-- formula-not-decoded -->

From Lemma 8, (34), (35), in this epoch, decrease of L ∗ λ ( x ) is 510

<!-- formula-not-decoded -->

Sum above inequalities over all epochs, we have 511

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substitute H ν = λ ν f (1 -ν g ) O ( /lscriptκ 3+(1+ ν g ) ν f ) and L = O ( /lscriptκ 3 ) for (40), we have 512

513

## D.7 Proof of Theorem 1 514

Proof. From Lemma 1, we have ‖∇ L ∗ λ ( x ) - ∇ ϕ ( x ) ‖ ≤ O ( /lscriptκ 3 ) /λ . From Lemma 1, we have 515 | L ∗ λ ( x ) -ϕ ( x ) | ≤ O ( κ 2 ) /λ . Denote the number of total iterates as K , from Proposition 1, the 516 following holds: 517

<!-- formula-not-decoded -->

Substitute (38) and (41) with λ = max( O ( κ ) , O ( /lscriptκ 3 ) //epsilon1, O ( /lscriptκ 2 ) / ∆) , the theorem is proved. /unionsq /intersectionsq 518

/unionsq

/intersectionsq

## D.8 Proof of Theorem 2 519

Lemma D.2. Consider the t -epoch generated by Algorithm 1 and ending at iteration k , we claim 520 that for any t and its corresponding k , we can find some constant C to satisfy: 521

<!-- formula-not-decoded -->

Proof. For the t -epoch except the last epoch, ¯ w t,k is not an /epsilon1 -first-order stationary point. Since 522 L ∗ λ ( x ) has L -Lipschitz continuous gradient, we have 523

<!-- formula-not-decoded -->

where we use x k +1 = w k -1 L ˆ ∇ L ∗ λ ( w k ) . We also have 524

<!-- formula-not-decoded -->

Combining the above inequalities leads to 525

<!-- formula-not-decoded -->

where we use ‖ x k -w k ‖ = θ k ‖ x k -x k -1 ‖ ≤ ‖ x k -x k -1 ‖ in (a) ≤ , the triangle inequality in (b) ≤ 526 and Lemma 4 in (c) ≤ . 527

Summing over the above inequality, and using x 0 = x -1 , we have 528

<!-- formula-not-decoded -->

where we use the Cauchy-Schwarz inequality in (d) ≤ , non-negativity of norm in (e) ≤ , the restart condi529 tion (5) in (f) ≤ and (35) in (g) ≤ . For the last term in (42), we have 530

<!-- formula-not-decoded -->

where we use the restart condition (5) in ( a ) ≤ , x k = w k -1 -1 L ˆ ∇ L ∗ λ ( w k -1 ) in ( b ) ≤ , Lemma 3 in ( c ) ≤ and 531 Lemma 4 in ( d ) ≤ . Combined with (42), we obtain 532

<!-- formula-not-decoded -->

We claim that for any t -th epoch ending at iteration k , we can find some constant C to satisfy: 533

<!-- formula-not-decoded -->

Otherwise, (43) shows that L ∗ λ ( w t,k ) can go to -∞ , which contradicts to min x ∈ R dx ϕ ( x ) &gt; -∞ 534 in Assumption 1 and | L ∗ λ ( x ) -ϕ ( x ) | ≤ O ( /lscriptκ 2 /λ ) in Lemma 1. /unionsq /intersectionsq 535

With the help of Lemma D.2, we provide the proof of Theorem 2. 536

Proof. We firstly show the boundedness of ‖ y ∗ ( w t, 0 ) ‖ . Suppose that the t -epoch ends at iteration k , 537 we have 538

<!-- formula-not-decoded -->

The first inequality holds due to triangular inequality, the second inequality holds due to y ∗ ( x ) is 539 L g /µ -Lipschitz continuous and the last inequality holds due to Lemma 4 and Lemma D.2. Then we 540 have 541

<!-- formula-not-decoded -->

where T is the total number of epochs. We can set { T t,i , T ′ t,i } as follows: let 542

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for i ≥ 1 , and 543

<!-- formula-not-decoded -->

for i = 0 , where T is the total number of epochs. From Theorem 1, we know that 544

<!-- formula-not-decoded -->

Then we prove (8) holds for z t,i by induction. For i = 0 , by the definition of T t, 0 in (46), we have 545

<!-- formula-not-decoded -->

From Lemma C.1, if i ≥ 1 , we have 546

<!-- formula-not-decoded -->

547

548

549

550

551

552

where the inequality (a) ≤ follows from the triangle inequality, (b) ≤ uses the inductive hypothesis and the fact that y ∗ ( x ) is L g /µ -Lipschitz continuous, (c) ≤ holds by the definition w t,i = x t,i + θ i ( x t,i -x t,i -1 ) , (d) ≤ applies Lemma 3 and Lemma D.2, and (e) ≤ follows from (44). Therefore, by mathematical induction, we conclude that (8) holds for all z t,i with { T t,i } defined in (44),(46). Similarly, we can prove that (8) holds for y t,i with T ′ t,i defined in (45), (47). So all y t,i and z t,i satisfy Condition 1. The total first-order oracle complexity is ∑ t,i T t,i , i.e.,

When ν f = ν g = 1 , the first-order oracle complexity is ˜ O ( ∆ /lscript 3 / 4 κ 13 / 4 /epsilon1 -7 / 4 ) . 553 /unionsq /intersectionsq 554

<!-- formula-not-decoded -->

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

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the paper's main contributions, including the development of provably convergent algorithms for nonconvex-strongly convex bilevel problems under general smoothness assumptions. These claims are supported by the theoretical results in Section 4 and the experimental validations in Section 5, aligning well with the scope of the paper.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations and future directions of our work, please refer to Section 6.

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

Justification: We provide the full set of assumptions and complete, rigorous proofs for all lemmas, propositions, and theorems. The formal statements are presented in Section 3 and Section 4, with detailed proofs included in the Appendix.

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

Justification: We provide all necessary details to reproduce our main experimental results, including dataset descriptions, evaluation metrics and algorithmic settings in Section 5.

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

Justification: We provide the full implementation of our proposed method along with detailed instructions to reproduce the main experimental results in the supplementary materials. This includes code, environment setup, data generation procedures, and run commands.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We specify all the training and test details in Section 5.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Although the paper does not report error bars or statistical significance tests, we have verified that the results are stable across different random seeds.

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

763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We provide sufficient information on the computer resources used for all experiments, including compute workers, memory and time of execution. Please refer to Section 5 for full information.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research complies with the NeurIPS Code of Ethics in all respects. All ethical guidelines and considerations were carefully followed throughout the study. The experiments are conducted using publicly available datasets and standard computing resources.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: The paper discuss both potential positive societal impacts and negative societal impacts of the work performed. This work is theoretical and focuses on algorithmic developments in bilevel optimization. However, we acknowledge that future applications of this line of work could have societal consequences, which should be carefully considered in those contexts.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out

815

816

817

818

819

820

821

822

823

824

825

826

827

828

829

830

831

832

833

834

835

836

837

838

839

840

841

842

843

844

845

846

847

848

849

850

851

852

853

854

855

856

857

858

859

860

861

862

863

864

865

866

that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: We do not release any data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets).

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All external assets used in this paper, such as datasets and code packages, are properly cited with appropriate references.

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

Justification: The paper does not introduce or release any new datasets, codebases, or pretrained models.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: We do not introduce any human subject.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: We do not introduce any human subject.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

920 921 922 923 924 925 926 927

## Answer: [NA]

Justification: No large language models (LLMs) were used in the core methods or any key components of this research, so no specific declaration regarding LLM use is required.

## Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.