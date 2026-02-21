12

13

14

15

16

17

18

19

20

## Lower-level Duality Based Penalty Methods for Bilevel Hyperparameter Optimization

## Anonymous Author(s)

Affiliation Address email

## Abstract

Hyperparameter optimization (HO) is a critical task in machine learning and can be formulated as a bilevel optimization problem. However, many existing algorithms for addressing nonsmooth lower-level problems involve solving sequential subproblems, which are computationally expensive. To address this challenge, we propose penalty methods for solving HO, leveraging strong duality between the lower-level problem and its dual. We show that the penalized problem closely approximates the optimal solutions of the original HO under certain conditions. Moreover, we develop first-order single-loop algorithms to solve the penalized problems efficiently. Theoretically, we establish the convergence of the proposed algorithms. Numerical experiments demonstrate the efficiency and superiority of our method.

## 1 Introduction

Hyperparameter optimization (HO) arises in many diverse fields, neural architecture search [16, 29, 57], feature learning [35], ensemble models [25], semi-supervised learning [42] and sample-weighting schemes [34, 77, 74, 82]. The hyperparameters control model complexity, training stability and convergence. Unlike model parameters, they need to be chosen externally. A poor choice can cripple performance, whereas good hyperparameters greatly enhance accuracy, robustness and generalization.

Regularization is a common way to guide hyperparameter tuning, especially in regression and classification [32]. By adding a penalty term to the empirical risk, one trades off data fitting against model complexity to curb overfitting. The general framework can be formulated as

<!-- formula-not-decoded -->

where l ( x ) represents the loss function and λ = ( λ 1 , λ 2 , ..., λ M +1 ) encompasses hyperparameters. 21 Meanwhile, R i ( x ) , i = 1 , 2 , ..., M +1 denotes the regularizers related to norms, which can be 22 categorized as follows: 23

<!-- formula-not-decoded -->

For each i , ∥ · ∥ ( i ) represents a specific norm, such as the ℓ 1 , ℓ 2 , ℓ ∞ , ℓ 1 , 2 norm for vectors, the 24 spectre or nuclear norm for matrices, or other commonly used norms. Note that these two types of 25 regularizers may appear simultaneously or individually. 26

Based on the formulation (1), training/validation approach is involved as a sophisticated method. This 27 method optimizes parameters in the form (1) on the training set and observes the corresponding error 28 on the validation set. The approach can be summarized as bilevel optimization framework [57, 9] and 29 Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

has demonstrated outstanding performance in practical applications [66, 31, 35, 15]. In essence, the 30 process can be outlined in the following bilevel optimization (BLO) [72, 28]: 31

<!-- formula-not-decoded -->

where L, l, R i : R n → R ∪ { + ∞} are proper, closed functions, x is the parameter to learn, and 32 λ is hyperparameter. In BLO (3), the lower-level (LL) problem serves as a base learner, aiming 33 to determine the optimal hypothesis on the training set for a given hyperparameter configuration. 34 In contrast, the upper-level (UL) problem aims to identify the hyperparameter and corresponding 35 hypothesis that minimizes the given criteria on the validation set. We explain the mathematical 36 forms of the component functions in problem (3) using several illustrative examples listed in Table 37 1, including elastic net [100], sparse group Lasso [83], logistic regression [68, 46], low-rank matrix 38 completion [20] and smoothed support vector machine [78, 65]. 39

Table 1: Examples of bilevel hyperparameter optimization [48, 31, 46] in the form (3).

| Machine learning algorithm                                                                                    | Upper Criteria                                                                                                                                                                                                                     | Base Learner                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|---------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Elastic net Sparse group Lasso Smoothed support vector machine Low-rank matrix completion Logistic regression | 1 2 ∑ i ∈ I val &#124; b i - x T a i &#124; 2 1 2 ∑ i ∈ I val &#124; b i - x T a i &#124; 2 ∑ i ∈ I val l h ( b i w T a i ) ∑ ( i,j ) ∈ Ω val 1 2 &#124; M ij - x i θ - z j β - Γ ij &#124; 2 ∑ j ∈ I val log(1+ e - b j x T a j ) | 1 2 ∑ i ∈ I tr &#124; b i - x T a i &#124; 2 + λ 1 ∥ x ∥ 1 + λ 2 2 ∥ x ∥ 2 2 1 2 ∑ i ∈ I tr &#124; b i - x T a i &#124; 2 + ∑ M m =1 λ m ∥ x ( m ) ∥ 2 + λ M +1 ∥ x ∥ 1 ∑ i ∈ I tr l h ( b i w T a i )+ λ 2 ∥ w ∥ 2 (with constraint - ¯ w ≤ w ≤ ¯ w .) ∑ ( i,j ) ∈ Ω tr 1 2 &#124; M ij - x i θ - z j β - Γ ij &#124; 2 + λ 0 ∥ Γ ∥ ∗ + ∑ G g =1 λ g ∥ θ ( g ) ∥ 2 + ∑ G g =1 λ g + G ∥ β ( g ) ∥ 2 ∑ j ∈ I tr log(1+ e - b j x T a j )+ λ 2 ∥ x ∥ 2 |

l h denotes the smoothed hinge loss given by l h ( x ) = 1 2 -x if x ≤ 0 , 1 2 (1 -x ) 2 if 0 ≤ x ≤ 1 and 0 else.

## 1.1 Related Work

Hyperparameter Optimization. A variety of approaches have been developed for hyperparameter optimization (HO) [44]. The simplest model-free techniques include grid search [45] and random search [12]. More advanced methods such as Bayesian optimization [11, 84] iteratively select evaluation points based on prior observations. However, these approaches often struggle with scalability when faced with high-dimensional parameter spaces.

Bilevel Optimization. Bilevel optimization (BLO) underpins many machine learning tasks, including meta-learning [33], adversarial learning [19, 86, 87], reinforcement learning [80, 85, 93, 89], model selection [47, 39], generative adversarial networks [38, 40], and game theory [55]. Early methods primarily relied on gradient-based algorithms, which can be broadly classified into two categories: Iterative Differentiation (ITD) and Approximate Implicit Differentiation (AID). ITD methods unroll the lower-level problem and compute hypergradients via backpropagation [34, 35, 41, 61, 5, 77], while AID methods derive gradients from the lower-level optimality conditions [72, 73, 63, 92, 91].

Recent advances include fully first-order methods that avoid Hessian and implicit gradient computations [23, 54, 24]. To address the challenge of multiple lower-level minima, [59] introduce a value-function-based reformulation, leading to penalization-based algorithms [60]. This line of work has grown into a prominent direction, with various penalty-based single-level reformulations proposed in [79, 64, 50, 49, 56]. Another promising direction leverages the Moreau envelope to smooth the bilevel structure, yielding single-loop, Hessian-free algorithms capable of converging to well-defined KKT points [37, 95, 94].

For BLO with nonsmooth lower-level problems, [14] propose an implicit differentiation framework based on block coordinate descent, which is later extended to general nonsmooth settings [15]. Other approaches include DC methods [96, 97] and penalized DC formulations [36], both requiring the computation of the lower-level value function. Smoothing-based strategies have also been explored to handle nonsmoothness [3, 2, 71]. Additionally, [23] present a gradient-free method with inexact subproblem solutions, while [22] reformulate BLO via duality, avoiding the value function entirely and solving the problem through cone programming. [62] further extend the Moreau envelope approach to nonsmooth lower-level problems, offering efficient single-loop algorithms.

## 1.2 Motivations and Contributions

In this work, we focus on solving the bilevel optimization (3). We extend the reformulation initially proposed by [22] and incorporate penalty strategy. We demonstrate that our framework is applicable to commonly used hyperparameter optimization problems schemed in (3). Moreover, we propose

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

111

the L ower-level D uality Based P enalty M ethods (LDPM), which are first-order algorithms specifically designed for the penalized problem. The algorithms efficiently handle the nonsmooth norm components with epigraphic projections. Notably, our algorithms are single-loop and Hessian-free, relying solely on the first-order information of the functions in (3). Theoretically, we establish the convergence results of the algorithms under mild conditions. We summarize our contributions as follows.

- We propose a penalty method based on lower-level duality for hyperparameter optimization (3), which is in the form of BLO with nonsmooth LL problem.
- We introduce two first-order single-loop algorithms to solve the penalized problem and provide theoretical proof of the convergence.
- We evaluate the efficiency of our algorithms with numerical experiments on synthetic and real-world data. Experimental results validate superiority of our algorithm in practical scenarios.

## 2 Penalty-based Approach

In this section, we propose our penalization framework for the original problem (3). Prior to this, we observe that the loss functions of base learners in Table 1 share a unified structure of the form φ ( A x -b ) , where A x -b abstracts the data-sample relationship. Accordingly, we denote that

<!-- formula-not-decoded -->

where l ( x ) corresponds to the loss on validation and training sets as described in (3). We now provide a detailed discussion of the mathematical forms of the function φ for problems in Table 1, along with the expressions A t , b t :

Least squares loss : φ ( t ) = 1 2 t 2 , with A t x -b t = A tr x -b tr .

Smoothed hinge loss : φ ( t ) = l h ( t ) with A t w -b t = ( b tr A tr ) w .

Logistic loss : φ ( t ) = log(1 + e -t ) , with A t x -b t = ( b tr A tr ) x .

Building on the inner structure of φ and R i , our approach is grounded in a reformulation based on the duality of LL problem. We embrace the idea initially proposed by [22] and summarize the following lemma, which is a modification and extension of [22, Theorem 2.1].

̸

Lemma 2.1. Given the convex lower semi-continuous functions l and R i , if ri(dom l ∩ ( ∩ M +1 i =1 dom R i )) = ∅ 1 , then problem (3) has the following equivalent form:

<!-- formula-not-decoded -->

where ρ = ( ρ 1 , ..., ρ M +1 ) and A t , b t , φ are consistent with those in (4) and φ ∗ and R ∗ i are the conjugate functions of φ and R i for i = 1 , 2 , ..., M +1 , respectively. 2

Remark 2.2 . Slater's condition is broadly satisfied by all examples in Table 1, ensuring strong duality for the LL problem in (3) without requiring strong convexity. For instance, the least squares loss is not strongly convex, yet strong duality still holds under this condition.

We present a detailed proof of Lemma 2.1 in the Appendix A.1. Notably, each problem listed in Table 1 can be reformulated into the structure of (5). For clarity, we calculate the closed-form expressions of the conjugate functions in Appendix A.3. We remark that our reformulation utilizes the structure of l ( x ) in (4), which is different from the one in [22, Theorem 2.1]. Notably, each problem listed in Table 1 can be reformulated into the structure of (5).

To elaborate, we discuss the terms R ∗ i in (5) as follows. For i = 1 , 2 , ..., M , R i denotes a norm, i.e., R i ( x ) = ∥ x ∥ ( i ) . In this case, we know that R ∗ i ( y ) is the indicator function of the set {∥ y ∥ ∗ ( i ) ≤ 1 }

1 This condition is commonly known as Slater's condition. ri( · ) denotes the relative interior of the set.

2 We define the conjugate h ∗ ( y ) = sup x { y T x -h ( x ) } for a function h .

where ∥ · ∥ ∗ ( i ) denoted the dual norm of ∥ · ∥ ( i ) [18, Example 3.26]. The term R M +1 denotes the 112 squared ℓ 2 -norm, i.e., R M +1 ( x ) = 1 2 ∥ x ∥ 2 2 . In this case, we can compute that λ M +1 R ∗ M +1 ( ρ M +1 λ M +1 ) = 113 ∥ ρ M +1 ∥ 2 2 2 λ M +1 [18, Example 3.27]. To refine the intricate constraints of (5), we introduce auxiliary 114 variables r i and s satisfying R i ( x ) ≤ r i and ∥ ρ M +1 ∥ 2 2 2 λ M +1 ≤ s . This results in a further reformulation 115 based on Lemma 2.1. 116

Proposition 2.3. The original problem (3) can be reformulated as 117

<!-- formula-not-decoded -->

For simplicity, we rewrite the left-hand of the first inequality constraint in (6) as: 118

<!-- formula-not-decoded -->

Now we consider the penalization of problem (6) as follows, 119

<!-- formula-not-decoded -->

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

where F k ( z ) := L ( x ) + β k p ( x , λ , r , ξ , s ) + β k 2 ∥ A t ξ + M +1 ∑ i =1 ρ i ∥ 2 with z := ( x , λ , ρ , r , ξ , s ) for convenience, and β k serves as the penalty parameter. This penalty strategy is commonly employed in bilevel optimization [79, 62, 95, 94, 59]. Inspired by [70, Theorem 17.1], the following theorem reveals the relationship between the optimal solutions of penalization and reformulation (6).

Theorem 2.4. Assume L, l and R i are lower semi-continuous, with the loss function l and the regularization term R i in LL objective being convex. Suppose the penalty parameter satisfying β k →∞ . If z k +1 is the minimizer of penalized problem (8) with β k , then every limit point z ∗ of the sequence { z k } is a solution to the reformulation (6) .

The proof of Theorem 2.4 is provided in Appendix A.2. From the equivalence between (5) and (6), it follows that if z is the solution of (6), then ( x , λ , ρ , ξ ) is the corresponding solution of (5). Thus, Theorem 2.4 also reveals the connection between (5) and (8).

## 3 Epigraphical Projection-based First-order Algorithms

We develop our algorithms based on the penalized formulation (8), beginning with general assumptions on the original problem (3) to support analysis and algorithm design.

Assumption 3.1. The UL objective L is α L -smooth with respect to LL variable x . Additionally, as a loss function, L is non-negative, i.e., L ( x ) ≥ 0 for all x .

Assumption 3.2. The function φ is convex. Moreover, the function φ and its conjugate φ ∗ is α p -and α d -smooth, respectively.

Remark 3.3 . Assumptions 3.1 and 3.2 are satisfied by commonly used loss functions. Specifically, the problems listed in Table 1 adhere to Assumptions 3.1 and 3.2. We remark the UL objective L can be nonconvex , which remains compatible with our framework.

Remark 3.4 . The smoothness properties of l are naturally inherited by φ . Therefore, Assumption 3.2 implies that l is convex and Lipschitz smooth. Combined with the definition of R i in (2), Assumption 3.2 ensures that the LL problem in (3) is convex. Importantly, our framework relies only on the first-order differentiability and does not require the LL objective in (3) to exhibit strong convexity.

However, the primary challenges of solving (8) stem from the nonsmooth nature of the constraints, 145 particularly when different norms are involved. We define corresponding sets for the constraints in 146

problem (8) in the form of cones as follows: 147

<!-- formula-not-decoded -->

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

Furthermore, each set in (9) is projection-friendly, which facilitates efficient epigraphic projection of corresponding norms. The details of the projection operations are discussed in Appendix B.

Given these insights, a natural approach to manage the constraints in (8) is through projections onto K i and K d i . To address problems with different regularizers, we discuss the proposed algorithms in various scenarios. Section 3.1 focuses on problem (3) with single-round global regularization applied to the entire vector x . Section 3.2 extends this to problems with multiple interacting regularizers. This division provides a structured approach to handling varying constraints and regularization terms.

## 3.1 Separable Regularizers

In this subsection, we explore the algorithm for (3) when the LL problem incorporates separate regularizers, structured as a single group of component-wise terms. Specifically, the LL problem in (3) can be expressed as

<!-- formula-not-decoded -->

where x ( i ) represents the i -th subvector of x with x = ( x (1) , ..., x ( M ) ) and ∥ · ∥ ( t ) represents a prescribed norm applied to each group.

When M = 1 , the LL problem of (3) involves a single regularizer R 1 ( x ) , corresponding to simpler models such as toy Lasso or logistic regression. In this case, the constraints of (8) simplify as follows. If R 1 ( x ) = ∥ x ∥ ( t ) and ∥ · ∥ ( t ) is a norm, the constraints of (8) reduce to:

<!-- formula-not-decoded -->

If R 1 ( x ) = 1 2 ∥ x ∥ 2 2 , the constraints of (8) simplify to:

<!-- formula-not-decoded -->

The constraints (10)-(11) are consistent with the structure in (9) and can be compactly expressed as

<!-- formula-not-decoded -->

When M &gt; 1 , the LL problem of (3) incorporates group regularization, where group-wise ℓ 2 -regularization is the most common choice. This setting is widely adopted in practice, as illustrated by examples such as group Lasso in Table 1. Although the problem may appear to involve multiple regularization terms and hyperparameters, it essentially amounts to applying a single-round regularization process over the entire variable x . Under this structure, the constrains of (8) simplifies to:

<!-- formula-not-decoded -->

where ρ ( i ) is the i -th subvector of ρ with ρ = ( ρ (1) , ..., ρ ( M ) ) . Since constraints of (13) are independent for each i , they can be equivalently expressed as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Importantly, we observe that K in (12) and (14) remains projection-friendly, facilitating efficient implementation. Accordingly, we adopt a gradient projection method to solve the penalized problem (8), as outlined in Algorithm 1. In each iteration, we update z as

<!-- formula-not-decoded -->

where e k &gt; 0 is the step size, and proj K ( x ) is the projection of x onto K . 177

which implies that

178

179

180

181

182

183

184

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

## Algorithm 1 First-order Projection-based Method

- 1: Input λ 0 , ξ 0 , sequences { β k } , { e k } . Initialize x 0 , r 0 , ρ 0 , s 0 .
- 2: for k = 0 , 1 , 2 , ... do
- 3: Update z k +1 with projection gradient descent as (15).
- 4: end for

In Algorithm 1, we choose the penalty parameter as β k = β (1 + k ) p with a constant β &gt; 0 and 0 &lt; p &lt; 1 / 2 , which corresponds to Theorem 2.4. Such a selection strategy is common in penalty method and augmented Lagrangian methods [70, 69, 27, 62, 95, 94]. The initialization of Algorithm 1 is detailed in Appendix C.1. We remark that Algorithm 1 is a single loop algorithm that does not require solving any subproblem.

Next, we proceed to the convergence analysis of Algorithm 1, specifically investigating the nonasymptotic convergence properties of the sequence { z k } generated by Algorithm 1. By leveraging the reformulation in Lemma 2.1 and the definition of p , it follows that p ( x , λ , r , ξ , s ) ≥ 0 and no interior point exists for the feasible set [98, 59, 22]. In this case, the classical KKT condition for nonsmooth constrained optimization [76] are unsuitable for our analysis. Instead, we adopt the approximation KKT conditions introduced in [4]. We denote merit functions below,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The residual function ϕ k res ( z ) quantifies the stationarity for (8), because ϕ k res ( z ) = 0 if and only if z is a stationary point of (8). Meanwhile, the function ϕ fea ( z ) is interpreted as a feasibility measure for the penalized constraints of problem (6) [67]. Indeed, the merit functions in (16) and (17) are associated with the reformulation (5). Combined with the structure of BLO, ϕ fea ( z ) regulates optimality conditions of LL problem of (3). We clarify corresponding conclusions in Proposition C.1.

Theorem 3.5. Suppose Assumptions 3.1 and 3.2 hold. If the step size { e k } in Algorithm 1 satisfies 0 &lt; e k ≤ min { 1 α L + β k ∥ A t ∥ 2 2 α p , 1 β k , 1 β k ( α d + ∥ A t ∥ 2 2 ) } , the sequence { z k } generated by Algorithm 1 satisfies

<!-- formula-not-decoded -->

Furthermore, if the sequence { F k ( z k ) } is bounded, then it holds that

<!-- formula-not-decoded -->

We remark that boundedness assumptions on { F k ( z k ) } are widely adopted in relevant literature [95, 94, 62]. We provide explanations, proofs and more details in Appendix C.3.

## 3.2 Nonseparable Regularizers

In this subsection, we focus on developing a first-order algorithm for solving (3) in scenarios involving multiple interacting regularizers. These cases arise when the LL problem of (3) incorporates multiple regularization terms applied to the entire vector x , such as elastic net or sparse group Lasso. Our discussion centers on addressing the penalized formulation in this setting, leveraging the reformulation (8). Using the definitions of K i and K d i from (9), the constraints of (8) can be written as

<!-- formula-not-decoded -->

which can be further expressed as

<!-- formula-not-decoded -->

We denote K d ∗ := K d 1 ×··· × K d M +1 . (18) can be equivalently expressed as

<!-- formula-not-decoded -->

Since each K d i is projection-friendly, the product set K d ∗ inherits this property. In contrast, the intersection ∩ M +1 i =1 K i defined over the shared variable ( x , r ) may not be projection-friendly. Although

projection onto such intersections has been studied [6, 58], the required iterations are often complex. 211 To address this, we reformulate the constraint to avoid direct projection onto the intersection: 212

<!-- formula-not-decoded -->

For each i , since both K d ∗ and K i are projection-friendly, the product set K i ×K d ∗ is also projection213 friendly. Consequently, we introduce auxiliary variables u i for constraints (19), leading to the 214 following reformulation of (8): 215

<!-- formula-not-decoded -->

where u = ( u 1 , ..., u M +1 ) . We define the indicator function as g i ( z ) = I K i ×K d ∗ ( z ) , i = 1 , 2 , ..., M + 216 1 . The augmented Lagrangian function of problem (20) is given by: 217

<!-- formula-not-decoded -->

218

219

220

221

222

223

224

225

where µ := ( µ 1 , ..., µ M +1 ) denotes the Lagrangian multiplier associated with constraint z = u i . Based on L k γ ( z , u , µ ) , we adopt an alternative approach to solve (20) inspired by the core idea of the Alternating Direction Method of Multipliers (ADMM). This method alternates between updating primal variables z and u in separate subproblems, followed by a dual ascent step to update µ . At the k -th iteration, we update z by performing a gradient step with given z k :

<!-- formula-not-decoded -->

where the update direction d k z corresponds to the gradient of L k γ with respect to z evaluated at ( z k , u k , µ k ) and e k is the step size of k -th iteration. This is equivalent to minimize the proximal subproblem of L k γ :

<!-- formula-not-decoded -->

Next, for the u -subproblem, we update u i by minimizing L k γ with respect to u i as 226

<!-- formula-not-decoded -->

which is equivalent to performing the direct projection onto K i ×K d ∗ , yielding:

<!-- formula-not-decoded -->

Finally, for the dual multipliers µ i , we update them as 228

<!-- formula-not-decoded -->

## Algorithm 2 Alternating approaches for (20)

- 1: Input λ 0 , ξ 0 , sequences { β k } , { e k } , a constant γ . Initialize x 0 , r 0 , ρ 0 i , s 0 , set u 0 i = z 0 .
- 2: for k = 0 , 1 , 2 , ... do
- 3: Update z k +1 with (21).
- 4: Update u k +1 with (23).
- 5: Update µ k +1 with (24).
- 6: end for

The penalty parameter is updated as β k = β (1 + k ) p , where β &gt; 0 is a constant and 0 &lt; p &lt; 1 2 . The 229 initialization of Algorithm 2 is also detailed in Appendix C.1. We remark that Algorithm 2 differs 230 from standard ADMM or DRS in two key aspects: (i) the augmented Lagrangian L k γ varies with 231 the iteration-dependent parameter β k . (ii) instead of exactly minimizing L k γ in the z -subproblem, 232 we adopt its first-order approximation at z k . The strategy is commonly employed in gradient-based 233 alternating minimization approaches [1, 17]. 234

227

In the following, we discuss the convergence property of Algorithm 2. Similar to the analysis for 235 Algorithm 1, we utilize the stationarity and feasibility measure commonly used in penalty methods 236 [67, 95, 94]. We define the following merit functions in the same arguments as (16) and (17): 237

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where K := ( K 1 ∩ · · · ∩ K M +1 ) ×K d 1 ×···K d M +1 . Based on the above functions, we establish the convergence results for Algorithm 2 in Theorem 3.7. In pursuit of this, we make the following assumption, which is popularly employed in ADMM approaches [90, 8, 81, 26].

<!-- formula-not-decoded -->

Theorem 3.7. Suppose Assumptions 3.1, 3.2 and 3.6 hold. If the step sizes in Algorithm 2 satisfy 0 &lt; e ≤ e k &lt; min { β k α L + β k ∥ A t ∥ 2 2 α p , 1 α d + ∥ A t ∥ 2 2 , 1 } , the sequence { z k } generated by Algorithm 2 satisfies lim k →∞ ϕ k res ( z k +1 ) = 0 . Furthermore, if the sequence { F k ( z k ) } is bounded, then it holds that lim k →∞ ϕ fea ( z k ) = 0 .

Note that the lower bound e for step sizes e k is commonly utilized in single-loop Hessian-free algorithms for BLO [94, 95, 62]. We provide the detailed proof for Theorem 3.7 in Appendix C.4.

## 4 Numerical Experiments

In this section, we evaluate the numerical performance of our proposed LDPM through experiments on both synthetic and real datasets. Specifically, we compare LDPM with several existing hyperparameter optimization algorithms under the BLO framework (3), including search methods, TPE [13], IGJO [31], IFDM [14, 15], VF-iDCA [36], LDMMA [22], BiC-GAFFA [94], as detailed in Appedix D.1.

We consider all hyperparameter optimization problems listed in Table 1. Performance is evaluated using validation and test errors based on the obtained LL minimizers, as well as the total running time. These metrics are standard in the evaluation of bilevel hyperparameter optimization algorithms [36, 31]. For each problem, we perform experiments across various data settings or datasets with 10 repetitions, and report the aggregated statistical results. Depending on the regularization structure of each problem, we apply either Algorithm 1 or 2, as detailed in Section 3.

## 4.1 Experiments on synthetic data

We focus on two prototypical tasks built from simple synthetic data: least squares regression with various Lasso-type regularizers and low-rank matrix completion, as listed in Table 1. The synthetic data consists of observation matrices sampled from specific distributions and response vectors generated with controlled noise. The detailed data generation process is provided in Appendix D.2.

Lasso-type Regression. We consider three regularizers: elastic net [100], group Lasso [99], and sparse group Lasso [83]. These formulations all promote sparsity while balancing model complexity and predictive accuracy. Table 2 presents the statistical results for the sparse group Lasso problem, including validation error, test error, and running time. Results for the elastic net and group Lasso problems are reported in Tables 3 and 4, respectively. Detailed experimental settings for each method are provided in the corresponding subsections of Appendix D.2. Overall, LDPM demonstrates superior performance on synthetic data, consistently achieving the lowest test errors while requiring the least computational time compared to baseline methods.

Low-rank matrix completion. For this problem, we conduct the numerical experiments on 60 × 60 matrices [36, 31]. The data generation process, detailed statistical results, and corresponding analysis are presented in Appendix D.2.4.

Sensitivity of parameters. We conduct sensitivity experiments on both Algorithm 1 and Algorithm 2. The results summarized in Table 6 show that both algorithms exhibit stable convergence across various parameter settings.

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

Table 2: Sparse group Lasso problems on synthetic data, where p represents the number of features.

| Settings   | p = 600          | p = 600            | p = 600            | p = 1200         | p = 1200           | p = 1200           |
|------------|------------------|--------------------|--------------------|------------------|--------------------|--------------------|
| Settings   | Time(s)          | Val. Err.          | Test Err.          | Time(s)          | Val. Err.          | Test Err.          |
| Grid       | 6 . 36 ± 1 . 88  | 84 . 73 ± 5 . 29   | 87 . 34 ± 15 . 91  | 13 . 68 ± 2 . 49 | 84 . 68 ± 4 . 31   | 86 . 00 ± 18 . 43  |
| Random     | 6 . 02 ± 2 . 01  | 135 . 17 ± 5 . 95  | 147 . 43 ± 25 . 54 | 12 . 64 ± 2 . 84 | 137 . 87 ± 14 . 21 | 146 . 25 ± 15 . 52 |
| IGJO       | 1 . 58 ± 0 . 28  | 101 . 93 ± 4 . 07  | 96 . 36 ± 13 . 72  | 7 . 35 ± 1 . 46  | 130 . 56 ± 14 . 02 | 106 . 70 ± 4 . 01  |
| VF-iDCA    | 0 . 56 ± 0 . 15  | 56 . 96 ± 5 . 58   | 76 . 84 ± 11 . 33  | 8 . 63 ± 2 . 91  | 86 . 38 ± 6 . 40   | 87 . 58 ± 8 . 90   |
| LDMMA      | 0 . 57 ± 0 . 13  | 82 . 70 ± 5 . 03   | 72 . 44 ± 14 . 72  | 4 . 72 ± 2 . 15  | 83 . 93 ± 7 . 32   | 84 . 03 ± 9 . 08   |
| BiC-GAFFA  | 0 . 39 ± 0 . 02  | 67 . 42 ± 6 . 28   | 71 . 45 ± 10 . 74  | 2 . 52 ± 0 . 29  | 82 . 21 ± 5 . 03   | 79 . 81 ± 7 . 66   |
| LDPM       | 0 . 35 ± 0 . 03  | 65 . 11 ± 6 . 62   | 69 . 48 ± 9 . 40   | 2 . 15 ± 0 . 14  | 81 . 39 ± 6 . 51   | 78 . 11 ± 6 . 35   |
| Settings   | p = 2400         | p = 2400           | p = 2400           | p = 4800         | p = 4800           | p = 4800           |
|            | Time(s)          | Val. Err.          | Test Err.          | Time(s)          | Val. Err.          | Test Err.          |
| Grid       | 24 . 23 ± 4 . 05 | 95 . 63 ± 14 . 13  | 84 . 86 ± 15 . 09  | 47 . 09 ± 6 . 34 | 128 . 94 ± 24 . 11 | 115 . 41 ± 17 . 62 |
| Random     | 22 . 17 ± 6 . 85 | 120 . 04 ± 15 . 36 | 146 . 77 ± 16 . 70 | 46 . 3 ± 5 . 57  | 99 . 41 ± 16 . 55  | 122 . 49 ± 19 . 46 |
| IGJO       | 11 . 14 ± 7 . 44 | 91 . 59 ± 14 . 97  | 115 . 98 ± 14 . 94 | 29 . 76 ± 9 . 44 | 99 . 75 ± 15 . 14  | 106 . 49 ± 7 . 48  |
| VF-iDCA    | 14 . 31 ± 1 . 45 | 63 . 21 ± 5 . 36   | 81 . 92 ± 10 . 54  | 45 . 12 ± 3 . 10 | 73 . 66 ± 10 . 53  | 96 . 09 ± 9 . 14   |
| LDMMA      | 7 . 50 ± 0 . 21  | 66 . 23 ± 7 . 47   | 79 . 09 ± 13 . 75  | 36 . 14 ± 3 . 65 | 78 . 61 ± 12 . 32  | 95 . 81 ± 9 . 43   |
| BiC-GAFFA  | 5 . 11 ± 0 . 10  | 86 . 83 ± 13 . 53  | 76 . 38 ± 8 . 60   | 5 . 03 ± 0 . 63  | 94 . 34 ± 8 . 19   | 92 . 05 ± 7 . 13   |
| LDPM       | 4 . 87 ± 0 . 05  | 92 . 32 ± 6 . 62   | 74 . 14 ± 2 . 79   | 4 . 58 ± 0 . 17  | 91 . 35 ± 6 . 04   | 90 . 21 ± 5 . 74   |

## 4.2 Experiments on real-world data

To assess the robustness of our algorithm in practical settings, we conduct experiments on real-world datasets that are larger and exhibit more complex sampling distributions. Specifically, we consider experiments on elastic net, smoothing support vector machine and sparse logistic regression, as listed in Table 1. All datasets are drawn from the LIBSVM repository 3 [21]. For each repetition, we randomly shuffle and split the data into training, validation and test sets.

Elastic Net. In this part, we conduct experiments on datasets gisette [43] and sensit [30]. We summarize the comparative experimental results in Table 7 and show the validation and test error curves over time for each algorithm in Figure 1. Even in these high-dimensional settings, LDPM delivers competitive accuracy while maintaining fast convergence. Additional experimental details are provided in Appendix D.4.1.

Figure 1: Comparison of the algorithms on Elastic Net problem for real-world datasets.

<!-- image -->

Smoothed Support Vector Machine. In this part, we perform 6 -fold cross-validation using medical statistics datasets, including diabetes, sonar, a1a [7]. Details of the datasets and experimental setup are given in Appendix D.4.2. We plots the validation and test errors of each algorithm over time in Figure 2, which clearly shows that LDPM converges more rapidly and achieves lower error levels than the competing methods.

Sparse Logistic Regression. In this part, we conduct experiments on three large-scale document classification datasets, news20.binary, rcv1.binary and real-sim. Dataset characteristics and experimental details are provided in Appendix D.4.3. In this experiment, we compare LDPM with search methods, IFDM and BiC-GAFFA. We plot the validation and test error curves over time in Figure 3 and report the corresponding final validation and test accuracies in Table 9 for comparison. LDPM consistently converges faster and achieves the lowest validation and test errors.

## 5 Conclusion

In this paper, we introduce a penalty framework based on lower-level duality for bilevel hyperparameter optimization. Notably, we solve the penalized problem using single-loop first-order algorithms. Theoretically, we establish convergence guarantees for the proposed algorithms. Empirically, through numerical experiments on both synthetic and real-world datasets, our methods exhibit superior performance compared to existing approaches, particularly among the illustrated HO examples.

3 https://www.csie.ntu.edu.tw/ cjlin/libsvmtools/datasets/

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

## References

- [1] Vahid Abolghasemi, Saideh Ferdowsi, and Saeid Sanei. A gradient-based alternating minimization approach for optimization of the measurement matrix in compressive sensing. Signal Processing , 92(4):999-1009, 2012.
- [2] Jan Harold Alcantara, Chieu Thanh Nguyen, Takayuki Okuno, Akiko Takeda, and Jein-Shan Chen. Unified smoothing approach for best hyperparameter selection problem using a bilevel optimization strategy. Mathematical Programming , pages 1-40, 2024.
- [3] Jan Harold Alcantara and Akiko Takeda. Theoretical smoothing frameworks for general nonsmooth bilevel problems. arXiv preprint arXiv:2401.17852 , 2024.
- [4] Roberto Andreani, José Mario Martínez, and Benar Fux Svaiter. A new sequential optimality condition for constrained optimization and algorithmic consequences. SIAM Journal on Optimization , 20(6):3533-3554, 2010.
- [5] Antreas Antoniou, Harrison Edwards, and Amos Storkey. How to train your maml. In International conference on learning representations , 2018.
- [6] Aleksandr Y Aravkin, James V Burke, Dmitry Drusvyatskiy, Michael P Friedlander, and Scott Roy. Level-set methods for convex optimization. Mathematical Programming , 174:359-390, 2019.
- [7] Arthur Asuncion and David Newman. Uci machine learning repository, 2007.
- [8] Xiaodi Bai, Jie Sun, and Xiaojin Zheng. An augmented lagrangian decomposition method for chance-constrained optimization problems. INFORMS Journal on Computing , 33(3):10561069, 2021.
- [9] Fan Bao, Guoqiang Wu, Chongxuan Li, Jun Zhu, and Bo Zhang. Stability and generalization of bilevel programming in hyperparameter optimization. Advances in neural information processing systems , 34:4529-4541, 2021.
- [10] Amir Beck. First-order methods in optimization . SIAM, 2017.
- [11] James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. Algorithms for hyperparameter optimization. Advances in neural information processing systems , 24, 2011.
- [12] James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of machine learning research , 13(2), 2012.
- [13] James Bergstra, Daniel Yamins, and David Cox. Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures. In International conference on machine learning , pages 115-123. PMLR, 2013.
- [14] Quentin Bertrand, Quentin Klopfenstein, Mathieu Blondel, Samuel Vaiter, Alexandre Gramfort, and Joseph Salmon. Implicit differentiation of lasso-type models for hyperparameter optimization. In International Conference on Machine Learning , pages 810-821. PMLR, 2020.
- [15] Quentin Bertrand, Quentin Klopfenstein, Mathurin Massias, Mathieu Blondel, Samuel Vaiter, Alexandre Gramfort, and Joseph Salmon. Implicit differentiation for fast hyperparameter selection in non-smooth convex learning. Journal of Machine Learning Research , 23(149):143, 2022.
- [16] Christopher M Bishop. Neural networks for pattern recognition . Oxford university press, 1995.
- [17] Nicholas Boyd, Geoffrey Schiebinger, and Benjamin Recht. The alternating descent conditional gradient method for sparse inverse problems. SIAM Journal on Optimization , 27(2):616-639, 2017.
- [18] Stephen P Boyd and Lieven Vandenberghe. Convex optimization . Cambridge university press, 2004.

354

355

356

357

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

- [19] Michael Brückner and Tobias Scheffer. Stackelberg games for adversarial prediction problems. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining , pages 547-555, 2011.
- [20] Emmanuel Candes and Benjamin Recht. Exact matrix completion via convex optimization. Communications of the ACM , 55(6):111-119, 2012.
- [21] Chih-Chung Chang and Chih-Jen Lin. Libsvm: a library for support vector machines. ACM transactions on intelligent systems and technology (TIST) , 2(3):1-27, 2011.
- [22] He Chen, Haochen Xu, Rujun Jiang, and Anthony Man-Cho So. Lower-level duality based reformulation and majorization minimization algorithm for hyperparameter optimization. arXiv preprint arXiv:2403.00314 , 2024.
- [23] Lesi Chen, Yaohua Ma, and Jingzhao Zhang. Near-optimal fully first-order algorithms for finding stationary points in bilevel optimization. arXiv preprint arXiv:2306.14853 , 2023.
- [24] Lesi Chen, Jing Xu, and Jingzhao Zhang. Bilevel optimization without lower-level strong convexity from the hyper-objective perspective. arXiv preprint arXiv:2301.00712 , 2023.
- [25] Marc Claesen, Frank De Smet, Johan Suykens, and Bart De Moor. Ensemblesvm: A library for ensemble learning using support vector machines. arXiv preprint arXiv:1403.0745 , 2014.
- [26] Xiangyu Cui, Rujun Jiang, Yun Shi, Rufeng Xiao, and Yifan Yan. Decision making under cumulative prospect theory: An alternating direction method of multipliers. INFORMS Journal on Computing , 2024.
- [27] Hari Dahal, Wei Liu, and Yangyang Xu. Damped proximal augmented lagrangian method for weakly-convex problems with convex constraints. arXiv preprint arXiv:2311.09065 , 2023.
- [28] Stephan Dempe and Alain Zemkoho. Bilevel optimization. In Springer optimization and its applications , volume 161. Springer, 2020.
- [29] Gonzalo I Diaz, Achille Fokoue-Nkoutche, Giacomo Nannicini, and Horst Samulowitz. An effective algorithm for hyperparameter optimization of neural networks. IBM Journal of Research and Development , 61(4/5):9-1, 2017.
- [30] Marco F Duarte and Yu Hen Hu. Vehicle classification in distributed sensor networks. Journal of Parallel and Distributed Computing , 64(7):826-838, 2004.
- [31] Jean Feng and Noah Simon. Gradient-based regularization parameter selection for problems with nonsmooth penalty functions. Journal of Computational and Graphical Statistics , 27(2):426-435, 2018.
- [32] Matthias Feurer and Frank Hutter. Hyperparameter optimization. Automated machine learning: Methods, systems, challenges , pages 3-33, 2019.
- [33] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning , pages 11261135. PMLR, 2017.
- [34] Luca Franceschi, Michele Donini, Paolo Frasconi, and Massimiliano Pontil. Forward and reverse gradient-based hyperparameter optimization. In International Conference on Machine Learning , pages 1165-1173. PMLR, 2017.
- [35] Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International conference on machine learning , pages 1568-1577. PMLR, 2018.
- [36] Lucy L Gao, Jane Ye, Haian Yin, Shangzhi Zeng, and Jin Zhang. Value function based difference-of-convex algorithm for bilevel hyperparameter selection problems. In International Conference on Machine Learning , pages 7164-7182. PMLR, 2022.

399

400

401

402

403

404

405

406

407

408

409

410

411

412

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

- [37] Lucy L Gao, Jane J Ye, Haian Yin, Shangzhi Zeng, and Jin Zhang. Moreau envelope based difference-of-weakly-convex reformulation and algorithm for bilevel programs. arXiv preprint arXiv:2306.16761 , 2023.
- [38] Gauthier Gidel, Hugo Berard, Gaëtan Vignoud, Pascal Vincent, and Simon Lacoste-Julien. A variational inequality perspective on generative adversarial networks. arXiv preprint arXiv:1802.10551 , 2018.
- [39] Tommaso Giovannelli, Griffin Dean Kent, and Luis Nunes Vicente. Inexact bilevel stochastic gradient methods for constrained and unconstrained lower-level problems. arXiv preprint arXiv:2110.00604 , 2021.
- [40] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM , 63(11):139-144, 2020.
- [41] Riccardo Grazzi, Luca Franceschi, Massimiliano Pontil, and Saverio Salzo. On the iteration complexity of hypergradient computation. In International Conference on Machine Learning , pages 3748-3758. PMLR, 2020.
- [42] Lan-Zhe Guo, Zhen-Yu Zhang, Yuan Jiang, Yu-Feng Li, and Zhi-Hua Zhou. Safe deep semisupervised learning for unseen-class unlabeled data. In International conference on machine learning , pages 3897-3906. PMLR, 2020.
- [43] Isabelle Guyon, Steve Gunn, Asa Ben-Hur, and Gideon Dror. Result analysis of the nips 2003 feature selection challenge. Advances in neural information processing systems , 17, 2004.
- [44] Frank Hutter, Jörg Lücke, and Lars Schmidt-Thieme. Beyond manual tuning of hyperparameters. KI-Künstliche Intelligenz , 29:329-337, 2015.
- [45] MohammadNoor Injadat, Abdallah Moubayed, Ali Bou Nassif, and Abdallah Shami. Systematic ensemble model selection approach for educational data mining. Knowledge-Based Systems , 200:105992, 2020.
- [46] Kwangmoo Koh, Seung-Jean Kim, and Stephen Boyd. An interior-point method for large-scale l1-regularized logistic regression. Journal of Machine learning research , 8(Jul):1519-1555, 2007.
- [47] Gautam Kunapuli, K Bennett, Jing Hu, and Jong-Shi Pang. Bilevel model selection for support vector machines. In CRM proceedings and lecture notes , volume 45, pages 129-158, 2008.
- [48] Gautam Kunapuli, Kristin P Bennett, Jing Hu, and Jong-Shi Pang. Classification model selection via bilevel programming. Optimization Methods &amp; Software , 23(4):475-489, 2008.
- [49] Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert Nowak. On penalty methods for nonconvex bilevel optimization and first-order stochastic approximation. arXiv preprint arXiv:2309.01753 , 2023.
- [50] Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert D Nowak. A fully first-order method for stochastic bilevel optimization. In International Conference on Machine Learning , pages 18083-18113. PMLR, 2023.
- [51] Chong Li and Kung Fu Ng. On constraint qualification for an infinite system of convex inequalities in a banach space. SIAM Journal on Optimization , 15(2):488-512, 2005.
- [52] Chong Li, Kung Fu Ng, and Ting Kei Pong. Constraint qualifications for convex inequality systems with applications in constrained optimization. SIAM Journal on Optimization , 19(1):163-187, 2008.
- [53] Jiajin Li, Caihua Chen, and Anthony Man-Cho So. Fast epigraphical projection-based incremental algorithms for wasserstein distributionally robust support vector machine. Advances in Neural Information Processing Systems , 33:4029-4039, 2020.

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

- [54] Junyi Li, Bin Gu, and Heng Huang. A fully single loop algorithm for bilevel optimization without hessian inverse. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 7426-7434, 2022.
- [55] Tao Li and Suresh P Sethi. A review of dynamic stackelberg game models. Discrete &amp; Continuous Dynamical Systems-B , 22(1):125, 2017.
- [56] Bo Liu, Mao Ye, Stephen Wright, Peter Stone, and Qiang Liu. Bome! bilevel optimization made easy: A simple first-order approach. Advances in neural information processing systems , 35:17248-17262, 2022.
- [57] Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055 , 2018.
- [58] Meijiao Liu and Yong-Jin Liu. Fast algorithm for singly linearly constrained quadratic programs with box-like constraints. Computational Optimization and Applications , 66:309326, 2017.
- [59] Risheng Liu, Xuan Liu, Xiaoming Yuan, Shangzhi Zeng, and Jin Zhang. A value-functionbased interior-point method for non-convex bi-level optimization. In International conference on machine learning , pages 6882-6892. PMLR, 2021.
- [60] Risheng Liu, Xuan Liu, Shangzhi Zeng, Jin Zhang, and Yixuan Zhang. Value-function-based sequential minimization for bi-level optimization. IEEE Transactions on Pattern Analysis and Machine Intelligence , 2023.
- [61] Risheng Liu, Yaohua Liu, Shangzhi Zeng, and Jin Zhang. Towards gradient-based bilevel optimization with non-convex followers and beyond. Advances in Neural Information Processing Systems , 34:8662-8675, 2021.
- [62] Risheng Liu, Zhu Liu, Wei Yao, Shangzhi Zeng, and Jin Zhang. Moreau envelope for nonconvex bi-level optimization: A single-loop and hessian-free solution strategy. arXiv preprint arXiv:2405.09927 , 2024.
- [63] Jonathan Lorraine, Paul Vicol, and David Duvenaud. Optimizing millions of hyperparameters by implicit differentiation. In International conference on artificial intelligence and statistics , pages 1540-1552. PMLR, 2020.
- [64] Zhaosong Lu and Sanyou Mei. First-order penalty methods for bilevel optimization. SIAM Journal on Optimization , 34(2):1937-1969, 2024.
- [65] JunRu Luo, Hong Qiao, and Bo Zhang. Learning with smooth hinge losses. Neurocomputing , 463:379-387, 2021.
- [66] Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In International conference on machine learning , pages 2113-2122. PMLR, 2015.
- [67] Ashkan Mohammadi. Penalty methods to compute stationary solutions in constrained optimization problems. arXiv preprint arXiv:2206.04020 , 2022.
- [68] Todd G Nick and Kathleen M Campbell. Logistic regression. Topics in biostatistics , pages 273-301, 2007.
- [69] Jorge Nocedal and Stephen Wright. Numerical optimization . Springer Science &amp; Business Media, 2006.
- [70] Jorge Nocedal and Stephen J Wright. Numerical optimization . Springer, 1999.
- [71] Takayuki Okuno, Akiko Takeda, Akihiro Kawana, and Motokazu Watanabe. On lphyperparameter learning via bilevel nonsmooth optimization. Journal of Machine Learning Research , 22(245):1-47, 2021.
- [72] Fabian Pedregosa. Hyperparameter optimization with approximate gradient. In International 490 conference on machine learning , pages 737-746. PMLR, 2016. 491

- [73] Aravind Rajeswaran, Chelsea Finn, Sham M Kakade, and Sergey Levine. Meta-learning with 492 implicit gradients. Advances in neural information processing systems , 32, 2019. 493

494

495

496

- [74] Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. Learning to reweight examples for robust deep learning. In International conference on machine learning , pages 4334-4343. PMLR, 2018.

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

- [75] R Tyrrell Rockafellar. Convex analysis , volume 18. Princeton university press, 1970.
- [76] R Tyrrell Rockafellar and Roger J-B Wets. Variational analysis , volume 317. Springer Science &amp;Business Media, 2009.
- [77] Amirreza Shaban, Ching-An Cheng, Nathan Hatch, and Byron Boots. Truncated backpropagation for bilevel optimization. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 1723-1732. PMLR, 2019.
- [78] Shai Shalev-Shwartz and Tong Zhang. Stochastic dual coordinate ascent methods for regularized loss minimization. Journal of Machine Learning Research , 14(1), 2013.
- [79] Han Shen and Tianyi Chen. On penalty-based bilevel gradient descent method. In International Conference on Machine Learning , pages 30992-31015. PMLR, 2023.
- [80] Han Shen, Zhuoran Yang, and Tianyi Chen. Principled penalty-based methods for bilevel reinforcement learning and rlhf. arXiv preprint arXiv:2402.06886 , 2024.
- [81] Yuan Shen, Zaiwen Wen, and Yin Zhang. Augmented lagrangian alternating direction method for matrix separation based on low-rank factorization. Optimization Methods and Software , 29(2):239-263, 2014.
- [82] Jun Shu, Qi Xie, Lixuan Yi, Qian Zhao, Sanping Zhou, Zongben Xu, and Deyu Meng. Meta-weight-net: Learning an explicit mapping for sample weighting. Advances in neural information processing systems , 32, 2019.
- [83] Noah Simon, Jerome Friedman, Trevor Hastie, and Robert Tibshirani. A sparse-group lasso. Journal of computational and graphical statistics , 22(2):231-245, 2013.
- [84] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical bayesian optimization of machine learning algorithms. Advances in neural information processing systems , 25, 2012.
- [85] Bradly Stadie, Lunjun Zhang, and Jimmy Ba. Learning intrinsic rewards as a bi-level optimization problem. In Conference on Uncertainty in Artificial Intelligence , pages 111-120. PMLR, 2020.
- [86] Jiali Wang, He Chen, Rujun Jiang, Xudong Li, and Zihao Li. Fast algorithms for stackelberg prediction game with least squares loss. In International Conference on Machine Learning , pages 10708-10716. PMLR, 2021.
- [87] Jiali Wang, Wen Huang, Rujun Jiang, Xudong Li, and Alex L Wang. Solving stackelberg prediction game with least squares loss via spherically constrained least squares reformulation. In International Conference on Machine Learning , pages 22665-22679. PMLR, 2022.
- [88] Po-Wei Wang, Matt Wytock, and Zico Kolter. Epigraph projections for fast general convex programming. In International Conference on Machine Learning , pages 2868-2877. PMLR, 2016.
- [89] Yue Frank Wu, Weitong Zhang, Pan Xu, and Quanquan Gu. A finite-time analysis of two timescale actor-critic methods. Advances in Neural Information Processing Systems , 33:1761717628, 2020.
- [90] Yangyang Xu, Wotao Yin, Zaiwen Wen, and Yin Zhang. An alternating direction algorithm for matrix completion with nonnegative factors. Frontiers of Mathematics in China , 7:365-384, 2012.

- [91] Haikuo Yang, Luo Luo, Chris Junchi Li, Michael Jordan, and Maryam Fazel. Accelerating 537 inexact hypergradient descent for bilevel optimization. In OPT2023: Optimization for Machine 538 Learning , 2023. 539

540

541

- [92] Junjie Yang, Kaiyi Ji, and Yingbin Liang. Provably faster algorithms for bilevel optimization. Advances in Neural Information Processing Systems , 34:13670-13682, 2021.

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

- [93] Zhuoran Yang, Yongxin Chen, Mingyi Hong, and Zhaoran Wang. Provably global convergence of actor-critic: A case for linear quadratic regulator with ergodic cost. Advances in neural information processing systems , 32, 2019.
- [94] Wei Yao, Haian Yin, Shangzhi Zeng, and Jin Zhang. Overcoming lower-level constraints in bilevel optimization: A novel approach with regularized gap functions. arXiv preprint arXiv:2406.01992 , 2024.
- [95] Wei Yao, Chengming Yu, Shangzhi Zeng, and Jin Zhang. Constrained bi-level optimization: Proximal lagrangian value function approach and hessian-free algorithm. arXiv preprint arXiv:2401.16164 , 2024.
- [96] Jane J Ye, Xiaoming Yuan, Shangzhi Zeng, and Jin Zhang. Difference of convex algorithms for bilevel programs with applications in hyperparameter selection. arXiv preprint arXiv:2102.09006 , 2021.
- [97] Jane J Ye, Xiaoming Yuan, Shangzhi Zeng, and Jin Zhang. Difference of convex algorithms for bilevel programs with applications in hyperparameter selection. Mathematical Programming , 198(2):1583-1616, 2023.
- [98] Jane J Ye and DL Zhu. Optimality conditions for bilevel programming problems. Optimization , 33(1):9-27, 1995.
- [99] Ming Yuan and Yi Lin. Model selection and estimation in regression with grouped variables. Journal of the Royal Statistical Society Series B: Statistical Methodology , 68(1):49-67, 2006.
- [100] Hui Zou and Trevor Hastie. Regression shrinkage and selection via the elastic net, with 561 applications to microarrays. JR Stat Soc Ser B , 67:301-20, 2003. 562

## A Proofs for Section 2 563

In this subsection, we provide the proofs of the results concerning the penalty framework in Section 564 2. 565

## A.1 Proof of Lemma 2.1 566

The following proof follows [22]. 567

Proof. We prove the conclusion based on the formulation (3). First we introduce augmented variables 568 z and z i , i = 1 , 2 , ..., M +1 and deduce the equivalent form of LL problem of (3), 569

<!-- formula-not-decoded -->

̸

Since l, R i are convex and the constraints are affine, strong duality holds under Slater's condition. If 570 ri ( dom l ∩ ( ∩ M +1 i =1 dom R i )) = ∅ , then (27) is equivalent to its Lagrangian dual problem: 571

<!-- formula-not-decoded -->

where ξ is Lagrangian multiplier of constraint A t x -b t = z , while ρ i are those associated with 572 constraints x = z i . By adding the negative signs, we obtain 573

<!-- formula-not-decoded -->

The above problem can be further simplified as, 574

<!-- formula-not-decoded -->

Meanwhile, leveraging the value function of the lower-level problem, the constraint of (3) is equivalent 575 to 576

<!-- formula-not-decoded -->

From the equivalence of (27) and (28), (29) is further equivalent to 577

<!-- formula-not-decoded -->

Because the inequality in (30) holds if and only if there exists a feasible pair ( ξ , ρ ) satisfying (30), 578 dropping the max operator, we obtain that the constraint in (3) is equivalent to 579

<!-- formula-not-decoded -->

We complete the proof. 580

## A.2 Proof of Theorem 2.4 581

582

583

584

585

Proof. We adopt the convention A ( z ) = 1 2 ∥ A t ξ + M +1 ∑ i =1 ρ i ∥ 2 . It is straightforward that A ( z ) ≥ 0 . Let ¯ z be any limit point of the sequence { z k } and { z j k } ⊂ { z k } be the subsequence such that z j k → ¯ z . Assume that z ∗ is a solution of the reformulation (6). Then it holds that L ( x ∗ ) ≤ L ( x ) for all z = ( x , λ , ρ , r , ξ , s ) feasible to (6). Note that any point z feasible to (6) is also feasible to (8).

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

Since z k +1 is the minimizer of the problem (8) with β k , it follows that

<!-- formula-not-decoded -->

where ( a ) follows from the feasibility of z k +1 and z ∗ for the penalized problem (8) and the optimality of z k +1 , ( b ) holds because z ∗ is feasible to (6). From (31), we deduce

<!-- formula-not-decoded -->

Since the functions L, p and A are lower semi-continuous in z , letting k = k j and taking the limit j →∞ for the above inequality, we have p (¯ z ) + A (¯ z ) ≤ 0 with β k →∞ .

Since the assumptions of Theorem 2.4 are consistent with those of Lemma 2.1, we obtain the following relation from the formulation of p in (7)

<!-- formula-not-decoded -->

which directly implies that p ( z ) ≥ 0 . Combined with A ( z ) ≥ 0 for all z , we further deduce that p (¯ z ) = 0 and A (¯ z ) = 0 . Therefore, ¯ z is feasible for (6). Since z ∗ is optimal for (6), it holds that L ( x ∗ ) ≤ L (¯ x ) .

Letting k = k j and taking the limit j →∞ for (31), we have L (¯ x ) ≤ L ( x ∗ ) . Hence, we deduce that L (¯ x ) = L ( x ∗ ) and ¯ z is also an optimal solution of (6). This completes the proof.

## A.3 Conjugate functions for problems listed in Table 1

we calculate the closed-form expression of the conjugate functions of φ in problems as follows:

For least squares loss , φ ∗ ( v ) = 1 2 v 2 .

For smoothed hinge loss , φ ∗ ( v ) = 1 2 v 2 + v if -1 &lt; v &lt; 0 and φ ∗ ( v ) = ∞ otherwise.

For logistic loss , φ ∗ ( v ) = -v log( v ) -(1 -v ) log(1 -v ) if 0 &lt; v &lt; 1 and φ ∗ ( v ) = ∞ otherwise.

## B Epigraphical Projections

In this section, we discuss the projection onto the cones in Algorithms 1 and 2. According to different cases detailed in Section 3.1 and 3.2, we discuss the projections when involving different norm regularizers.

## B.1 Projections Involving Vector Norms

The most commonly used norms in hyperparameter optimization include the ℓ 1 -, ℓ 2 - and ℓ ∞ -norm, each serving distinct purposes depending on the specific application. When R i represents a single norm, the explicit forms of K i and K d i defined in (9) are expressed as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, we investigate the projection onto the epigraph { ( x , t ) | ∥ x ∥ q ≤ t } of the ℓ q -norm ( q = 614 1 , 2 , ∞ ). The projection for the ℓ 2 -norm epigraph has a well-known closed-form solution, as detailed 615 below: 616

617

618

Proposition B.1. [10, Example 6.37] Let L n 2 = { ( x , t ) | ∥ x ∥ 2 ≤ t } , for any ( x , t ) ∈ R n × R , we have

619

620

621

622

<!-- formula-not-decoded -->

Next, we discuss the ℓ 1 norm epigraphic projection. We first provide the following theorem on the projection onto epigraphs of convex functions.

Theorem B.2. [10, Theorem 6.36] Let C = epi( g ) = { ( x , t ) | g ( x ) ≤ t } where g is convex. Then for any ( x , t ) ∈ R n × R , it holds that

<!-- formula-not-decoded -->

where λ ∗ is any positive root of the function 623

<!-- formula-not-decoded -->

624

625

626

635

636

637

638

639

640

641

In addition, ψ is nonincreasing.

Proposition B.3. [10, Example 6.38] Let L n 1 = { ( x , t ) | ∥ x ∥ 1 ≤ t } , for any ( x , t ) ∈ R n × R , we have

<!-- formula-not-decoded -->

where T λ = prox λ ∥·∥ 1 denotes the proximal of ℓ 1 -norm, defined as 627

<!-- formula-not-decoded -->

Here, λ ∗ is any positive root of the nonincreasing function ψ ( λ ) = ∥T λ ( x ) ∥ 1 -λ -s . In practice, 628 the ℓ 1 norm epigraphical projection can be computed in linear time using the quick-select algorithm 629 proposed by [88]. 630

Finally, the projection for the ℓ ∞ norm epigraph can be computed directly via the Moreau decompo631 sition. Let L n ∞ = { ( x , t ) | ∥ x ∥ ∞ ≤ t } , then the projection is given by 632

<!-- formula-not-decoded -->

When R i represents the squared ℓ 2 norm, the corresponding rotated second-order cones are defined 633 as K M +1 and K d M +1 in (9). According to Theorem B.2, for any ( x , t ) ∈ R n × R , we have 634

<!-- formula-not-decoded -->

where λ ∗ is any positive root of the nonincreasing function ψ ( λ ) = ( 1 2 λ + t )(1+2 λ 2 ) -∥ x ∥ 2 2 . Similar to ℓ 1 -norm epigraphic projection, it can also be effectively solved in linear time with quick-select algorithm proposed by [88].

For the rotated second-order cone K d M +1 = { ( ρ , λ, s ) | ∥ ρ ∥ 2 2 ≤ 2 λs } where ρ ∈ R n , an equivalent representation is given by { ( ρ , λ, s ) | ∥ ( ρ , λ, s ) ∥ 2 ≤ λ + s } . We introduce auxiliary variables w = ( ρ , λ, s ) ∈ R n +2 and t = λ + s ∈ R . In this way, the projection onto K d M +1 for given (¯ ρ , ¯ λ, ¯ s ) is equivalent to the following optimization problem with ( ¯ w , ¯ t ) :

<!-- formula-not-decoded -->

642

643

644

645

646

647

648

649

650

651

652

653

654

655

656

657

658

659

660

661

662

663

664

665

666

667

668

669

670

671

672

673

674

675

676

677

where c 0 = (0 , ..., 0 , 1 , 1) ∈ R n +2 . The problem can be solved directly using the analytic solution provided in [53, Proposition 6.4].

When the regularization involves a group component-wise regularizers, i.e., R i ( x ) = ∥ x ( i ) ∥ ( t ) , where represents the i -th subvector of x with x = ( x (1) , ..., x ( M ) ) , as described in Section 3.1. In this case, we observe that projection onto the set K i and K d i corresponds to the ℓ 1 , ℓ 2 or ℓ ∞ -norm. The same projection applies to the vector ρ = ( ρ (1) , ..., ρ ( M ) ) .

## B.2 Projections Involving Matrix Norms

Now we study the projection onto the epigraphs of nuclear norm ∥ · ∥ ∗ and spectral norm ∥ · ∥ op . Since our reformulation relies on conjugate functions and the conjugate of a norm is its dual norm, we need to take both into consideration.

For a matrix X ∈ R m × n , the nuclear norm is defined as ∥ X ∥ ∗ = min { m,n } ∑ i =1 σ i ( X ) and the spectral norm is defined as ∥ X ∥ op = max i σ i ( X ) , where σ i ( X ) is singular values for X .

Given a matrix A ∈ R m × n and a scalar t , the projection onto the epigraph of the nuclear norm { X ∈ R m × n , τ ≥ 0 | ∥ X ∥ ∗ ≤ τ } involves solving the following optimization problem

<!-- formula-not-decoded -->

where ∥ · ∥ F denotes Frobenius norm of a matrix.

- If ∥ A ∥ ∗ ≤ t , the point ( A,t ) already lies in the epigraph and the projection is simply ( X,τ ) = ( A,t ) .
- If ∥ A ∥ ∗ &gt; t , we first compute the singular value decomposition of A as A = U Σ V , where Σ = diag { σ 1 , σ 2 , ..., σ r } is the single value matrix of A and U ∈ R m × r , V ∈ R n × r . According to [1,Theorem 6.36], the projected matrix is obtained by soft-thresholding the singular values:

<!-- formula-not-decoded -->

r

where λ is determined by the equation ∑ i =1 max( σ i -λ, 0) = t + λ . This equation is typically solved efficiently via a bisection search. Subsequently, we obtain the solution τ ∗ = t + λ and reconstruct the projected matrix as X ∗ = U ¯ Σ V T where ¯ Σ = diag { ¯ σ 1 , ¯ σ 2 , ..., ¯ σ r } . The projected pair ( X ∗ , τ ∗ ) is the closest point to ( A,t ) in the epigraph of the nuclear norm.

Given a matrix A ∈ R m × n and a scalar t , now we consider projection onto the epigraph of the nuclear norm { X ∈ R m × n , τ ≥ 0 | ∥ X ∥ op ≤ τ }

- If ∥ A ∥ op ≤ t , the point ( A,t ) already lies in the epigraph and the projection is simply ( X,τ ) = ( A,t ) .
- If ∥ A ∥ op &gt; t , we first compute the singular value decomposition of A as A = U Σ V , where Σ = diag { σ 1 , σ 2 , ..., σ r } is the single value matrix of A and U ∈ R m × r , V ∈ R n × r . Since the epigraph of the spectral norm is defined by the constraint ∥ X ∥ op = max i σ i ( X ) ≤ τ , we need to adjust the singular values so that the largest does not exceed the new scalar τ ∗ as

<!-- formula-not-decoded -->

To determine τ ∗ , we solve the one-dimensional optimization problem

<!-- formula-not-decoded -->

In practice, the optimal τ ∗ can be efficiently computed using a bisection search.

Subsequently, we reconstruct the projected matrix as X ∗ = U ˜ Σ V T where ˜ Σ = 678 diag { ˜ σ 1 , ˜ σ 2 , . . . , ˜ σ r } . 679

The projected pair ( X ∗ , τ ∗ ) is the closest point to ( A,t ) in the epigraph of the spectral 680 norm. 681

From the above discussions, it is evident that the projections can be computed efficiently. 682

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

## C Explanations and proofs for Section 3

In this section, we provide additional explanations and the proofs for the convergence results of our proposed algorithms in Section 3.

## C.1 Initialization of Algorithm 1 and 2

We initialize the starting point by following the algorithms for BLO proposed in [36, 22, 95]. For Algorithm 1, given the input λ 0 , ξ 0 , we initialize x 0 by solving the LL problem of (3). The remaining initial variables are set as r 0 i = R i ( x ) , ρ 0 = -∇ l ( x 0 ) and s 0 = ∥ ρ 0 ∥ 2 / 2 λ 0 1 . For Algorithm 2, given the input λ 0 , ξ 0 , we also initialize x 0 with solving the LL problem of (3). The other initial variables are set as r 0 i = R i ( x 0 ) , ρ 0 i = -1 M +1 A t ξ 0 and s 0 = ∥ ρ 0 M +1 ∥ 2 / 2 λ 0 M +1 .

This initialization strategy ensures a feasible starting point for the corresponding reformulation of original BLO, thereby facilitating convergence and enhancing the overall efficiency of the optimization process.

## C.2 Explanations for Merit Functions

To initiate the proof of the convergence results, we establish the rationale for selecting ϕ k res and ϕ fea as the merit measures. Note that ϕ k res and ϕ fea in Section 3.1 and 3.2 are both defined based on the penalized formulation (8) within a unified framework as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where K = ( K 1 ∩···∩K M +1 ) ×K d 1 ×···K d M +1 . For the case of single-round global regularization discussed in Section 3.1, the set K reduces to K = K 1 ×K d 1 and ( ρ 1 , ..., ρ M +1 ) is replaced by a single ρ .

From Lemma 2.1, we know that (5) is a direct reformulation of (3). For convenience, we simplify the left hand of the first constraint as:

<!-- formula-not-decoded -->

Similar to (8), we construct the penalized formulation for (5) as follows, 705

<!-- formula-not-decoded -->

where β k serves as the penalty parameter.

Proposition C.1. If ϕ fea ( z ) = 0 , then ( x , λ , ρ , ξ ) is a feasible point to (5) . Moreover, if ϕ fea ( z ) = 0 and ϕ res ( z ) = 0 both hold, then ( x , λ , ρ , ξ ) is a stationary point of (34) .

Proof. (a) When ϕ fea = 0 holds:

706

707

708

709

710

711

From the non-negativity of the function p and ∥ · ∥ 2 , if ϕ fea ( z ) = 0 , it holds that p ( x , λ , r , ξ , s and A t ξ + M +1 ∑ i =1 ρ i = 0

According to the constraints of (8), we know that 712

<!-- formula-not-decoded -->

) = 0

Additionally, we restore λ M +1 R ∗ M +1 ( ρ M +1 λ M +1 ) with the inequality ∥ ρ M +1 ∥ 2 2 2 λ M +1 ≤ s . Consequently, we 713 observe that 714

<!-- formula-not-decoded -->

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

726

727

<!-- formula-not-decoded -->

- For ( ρ i , λ i ) , i = 1 , ..., M , we have

<!-- formula-not-decoded -->

where K d i = { ( ρ i , λ i ) | ∥ ρ i ∥ ∗ ( i ) ≤ λ i } . From (30) and the definition of p , we know that 728 F ( x , λ , ρ , ξ ) ≥ 0 for all ( x , λ , ρ , ξ ) . If ϕ fea ( z ) = 0 , the following chain of inequalities 729 holds: 730

<!-- formula-not-decoded -->

which naturally reduces to equalities. Consequently, we have F ( x , λ , ρ , ξ ) = 731 p ( x , λ , r , ξ , s ) , implying that R i ( x ) = r i , i = 1 , ..., M . Therefore, we obtain that 732

<!-- formula-not-decoded -->

which implies that ( x , λ , ρ , ξ ) is feasible to (5).

(b) When ϕ k res ( z ) = 0 and ϕ fea ( z ) = 0 both hold:

In this part, we use Moreau-Rockafellar theorem [75, Theorem 23.8] to calculate the sum rule of subdifferentials. If f 1 and f 2 are convex and lower continuous at x and f 2 is differentiable at x ∈ int(dom( f 1 )) ∩ int(dom( f 2 )) , then it holds that

<!-- formula-not-decoded -->

We analyze ϕ k res ( z ) = 0 for each component of z .

- For x and r , we have

<!-- formula-not-decoded -->

where K i = { ( x , r ) | R i ( x ) ≤ r i } . Let ∂R i denote the limiting subdifferential of the function R i [76]. According to the definition of the normal cone of inequality constraints [51, 52] and the definition of K i in (9), we know that

<!-- formula-not-decoded -->

where cone denotes the conic hull of a set. Combining with (35), we obtain

<!-- formula-not-decoded -->

- For ξ , we have

Meanwhile, we note that for i = 1 , ..., M , R ∗ i is the indicator function of the set {∥ y ∥ ∗ ( i ) ≤ 733 1 } . Combining with the fact that the normal cone is equivalent to the subdifferential of 734 indicator function, for the variables ρ i and λ i , the above formulation implies that 735

<!-- formula-not-decoded -->

736

737

738

739

<!-- formula-not-decoded -->

where ( a ) follows the fact ∥ ρ ∥ ∗ ( i ) ≤ λ i and ( ∗ ) holds from the direct calculation of the subdifferential.

- For ( ρ M +1 , λ M +1 , s ) , we have

<!-- formula-not-decoded -->

where K d M +1 = { ( ρ M +1 , λ M +1 , s ) | ∥ ρ M +1 ∥ 2 2 ≤ 2 λ M +1 s } . Similar to the deduction for 740 ( ρ i , λ i ) in (38) and (39), we can obtain 741

<!-- formula-not-decoded -->

In summary, we find that the equations (36), (37), (38), (39) and (40) coincide with the 742 stationary conditions of (34). Therefore, we conclude that ( x , λ , ξ , ρ ) is a stationary point 743 of (34). 744

<!-- formula-not-decoded -->

From deduction (29) and (30), we conclude that ϕ fea ( z ) = 0 implies 746

<!-- formula-not-decoded -->

Following the reasoning in Theorem 2.4, we conclude that as β k → ∞ , any limit point of the 747 sequence of optimal solutions to (34) with β k is an optimal solution of (5). According to (36), we 748 further obtain that 749

<!-- formula-not-decoded -->

as β k →∞ . Thess results demonstrate that ϕ k res and ϕ fea can effectively character the optimality 750 condition of the LL problem in (3). In summary, the selection of ϕ k res and ϕ fea is reasonable. 751

752

We provide the proofs for the convergence results of Algorithm 1 and 2 in the subsequent sections.

753

754

755

## C.3 Proof of Theorem 3.5

We first recall the update for the variables of z in Algorithm 1 as follows. We calculate the update directions of z as d k z = β k ( d k x , d k λ , d k ρ , d k r , d k ξ , d k s ) , where

<!-- formula-not-decoded -->

With these directions, the gradient descent step is performed as 756

<!-- formula-not-decoded -->

For ¯ z k +1 = (¯ x k +1 , ¯ λ k +1 , ¯ ρ k +1 , ¯ r k +1 , ¯ ξ k +1 , ¯ s k +1 ) , we subsequently apply the projection

<!-- formula-not-decoded -->

Note that the variable ξ is not involved in the projection step and thus it is evolved directly as 758 ξ k +1 = ¯ ξ k +1 . 759

Next, we discuss the sufficient decrease property for Algorithm 1. 760

Lemma C.2. Suppose Assumption 3.2 hold. For k ∈ N , let { z k } be generated from Algorithm 1. 761 Define V k = 1 β k F k ( z k ) , then the following inequality holds: 762

<!-- formula-not-decoded -->

Furthermore, if the step sizes satisfy 0 &lt; e k ≤ min { 2 α L + β k ∥ A t ∥ 2 2 α p , 2 β k , 2 β k ( α d + ∥ A t ∥ 2 2 ) } , it holds that 763 V k +1 ≤ V k . 764

Proof. Given Assumption 3.2 that φ is α p -smooth, we know that l is ∥ A t ∥ 2 2 α p -smooth. By applying 765 the sufficient decrease lemma [10, Lemma 5.7], we obtain that 766

<!-- formula-not-decoded -->

Based on the convexity of the cones and the second projection theorem [10, Theorem 6.41], we have 767 ⟨ (¯ x k +1 , ¯ r k +1 ) -( x k +1 , r k +1 ) , ( x k , r k ) -( x k +1 , r k +1 ) ⟩ ≤ 0

768

which implies

<!-- formula-not-decoded -->

Given β k = β (1+ k ) p , we have 1 β k +1 ≤ 1 β k . Combining the above inequalities and the non-negativity 769 of L , we derive 770

<!-- formula-not-decoded -->

The same derivation process applies to ρ , λ i , r i , leading to the following results: 771

<!-- formula-not-decoded -->

For the variable s , we deduce that ¯ s k +1 = s k -e k and ⟨ ¯ s k +1 -s k +1 , s k -s k +1 ⟩ ≤ 0 , which implies 772 that 773

<!-- formula-not-decoded -->

Next, we define H k ( ξ ) = φ ∗ ( ξ ) + ξ T b t + 1 2 ∥ A t ξ + ρ k ∥ 2 , noting that H k is ( α d + ∥ A t ∥ 2 2 ) -smooth. 774 Then the update of ξ in Algorithm 1 can be expressed as 775

<!-- formula-not-decoded -->

757

Applying the sufficient decrease lemma [10, Lemma 5.7], we obtain 776

<!-- formula-not-decoded -->

which simplifies to 777

<!-- formula-not-decoded -->

Summing up the estimates (44)-(47), we arrive at the inequality (43). Furthermore, when the step 778 size satisfies 0 &lt; e k ≤ min { 2 α L + β k ∥ A t ∥ 2 2 α p , 2 β k , 2 β k ( α d + ∥ A t ∥ 2 2 ) } , the right-hand side of (43) becomes 779 negative, ensuring that V K +1 ≤ V k . 780

Now we provide the proof for Theorem 3.5. 781

Proof. We compress (43) from k = 0 to K -1 and obtain that 782

<!-- formula-not-decoded -->

From the non-negativity of L and p , we know that V K ≥ 0 and V 0 -V K ≤ V 0 . Subsequently, 783 according to the update rule of variables ( x , λ , ρ , r , ξ , s ) in Algorithm 1, we have that 784

<!-- formula-not-decoded -->

Therefore, it holds that 785

<!-- formula-not-decoded -->

Furthermore, we have similar conclusions for λ , r , ρ , s as follows, 786

<!-- formula-not-decoded -->

Now we define 787

<!-- formula-not-decoded -->

where ( ∗ ) holds from d k z = ∇ z F k ( z k ) . Using the directions specified in (41) and the relationship 788 given in (49) and (50), we obtain 789

<!-- formula-not-decoded -->

Based on the definition of the residual function ϕ k res in (16) and the relationship (51), we know that 790

<!-- formula-not-decoded -->

Subsequently, we estimate the value ∥ M k z ∥ with respect to z . By using Assumptions 3.1 and 3.2, we 791 find that ∥∇ z F k ( z k +1 ) -∇ z F k ( z k ) ∥ ≤ β k L z ∥ z k +1 -z k ∥ where L z = max { α L + β k ∥ A t ∥ 2 2 α p β k , α d + 792 ∥ A t ∥ 2 2 , 1 } . Then we have 793

<!-- formula-not-decoded -->

By combining (52) and the inequality (53), we deduce that 794

<!-- formula-not-decoded -->

When the step sizes are set as 0 &lt; e k ≤ 1 M k ≤ min { 1 α L + β k ∥ A t ∥ 2 2 α p , 1 β k , 1 β k ( α d + ∥ A t ∥ 2 2 ) } , we know 795 that 0 &lt; e k ≤ 1 β k , which implies that β k ≤ 1 e k . Then we conclude from (54) that there exists a 796 constant C res &gt; 0 such that 797

<!-- formula-not-decoded -->

From (48), we deduce that 798

<!-- formula-not-decoded -->

By compressing (55) from k = 0 to ∞ and combining with the inequality (56), we obtain that 799

<!-- formula-not-decoded -->

Given β k = β (1 + k ) p and 0 &lt; p &lt; 1 2 , we conclude that 800

<!-- formula-not-decoded -->

From the definition of ϕ fea in (17), we know that

<!-- formula-not-decoded -->

If the sequence { F k ( z k ) } is bounded, we know that there exists M &gt; 0 such that F k ( z k ) ≤ M for each k . Meanwhile, L ( x k ) ≥ 0 holds from Assumption 3.1. Then we have

<!-- formula-not-decoded -->

801

802

803

804

805

806

807

which implies that ϕ fea ( z k ) = O ( 1 K p ) .

## C.4 Proof of Theorem 3.7

Proof. From the update rule for u in (23), we have

<!-- formula-not-decoded -->

Additionally, the update rule for µ in (24) implies

<!-- formula-not-decoded -->

According to Assumptions 3.1 and 3.2, we know that L k γ ( z , u , µ ) is M k -smooth with respect to z , 808 where M k = max { α L + β k ∥ A t ∥ 2 2 α p β k , α d + ∥ A t ∥ 2 2 , 1 } . According to [10, Lemma 5.7], we have 809

<!-- formula-not-decoded -->

Given the update rule z k +1 = z k -e k ∇ z L k γ ( z k , u k , µ k ) , the inequality becomes 810

<!-- formula-not-decoded -->

Combining (57), (58) and (59) and dividing both sides by β k , we conclude 811

<!-- formula-not-decoded -->

According to β k = β (1 + k ) p , we obtain that 1 β k +1 ≤ 1 β k . With the non-negativity of L , it holds that 812 L k +1 γ ( z k +1 , u k +1 , µ k +1 ) ≤ L k γ ( z k +1 , u k +1 , µ k +1 ) , which implies that 813

<!-- formula-not-decoded -->

Now we define U k = L k γ ( z k , u k , µ k ) . Given that 0 &lt; e k ≤ min { β k α L + β k ∥ A t ∥ 2 2 α p , 1 α d + ∥ A t ∥ 2 2 , 1 } ≤ 814 1 M k , we can deduce from (60) that 815

<!-- formula-not-decoded -->

From the expression for L k γ , we can deduce the following, 816

<!-- formula-not-decoded -->

According to Assumption 3.6, we know that there exists some M µ such that ∥ µ k ∥ 2 ≤ M µ for all 817 k ∈ N . Additionally, the functions L and p are non-negative. This implies that 818

<!-- formula-not-decoded -->

indicating that U k is lower bounded. By telescoping the inequality (61) for k = 0 to ∞ , we get 819

<!-- formula-not-decoded -->

The sufficient decrease property (60) ensures that the U 0 - L b ≥ U 0 -U k ≥ 0 for any k ∈ N . 820 Combining with the fact that 0 &lt; 1 e k ≤ 1 e are bounded, and both e k and γ is positive, we obtain from 821 (63) that 822

<!-- formula-not-decoded -->

Additionally, the step size e k satisfies 0 &lt; e &lt; e k ≤ min { β k α L + β k ∥ A t ∥ 2 2 α p , 1 α d + ∥ A t ∥ 2 2 , 1 } ≤ 1 M k . This 823 implies that max {∥ A t ∥ 2 2 α p , α d + ∥ A t ∥ 2 2 , 1 } ≤ lim k →∞ 1 e k ≤ 1 e . Therefore, (64) ensures that 824

<!-- formula-not-decoded -->

From the update of µ i , we further derive that 825

<!-- formula-not-decoded -->

Meanwhile, from the form (22) for updating u i , we derive 826

<!-- formula-not-decoded -->

where ( a ) utilizes the fact that the normal cone is equivalent to the subdifferential of indicator 827 functions and ( b ) follows from the update of µ k +1 i . In (67), we use Moreau-Rockafellar theorem [75, 828 Theorem 23.8] to calculate the sum rule of subdifferentials. (67) implies that 829

<!-- formula-not-decoded -->

Combining the outer semi-continuity of the normal cone and (66), we can obtain that 830

<!-- formula-not-decoded -->

Furthermore, according to the definition K = ( K 1 ∩ · · · ∩ K M +1 ) × K d ∗ , we know that K = 831 ( K 1 ×K d ∗ ) ∩ · · · ∩ ( K M +1 ×K d ∗ ) . It implies that 832

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining with the definition of F k in (8), the above equality can be further expressed as 835

<!-- formula-not-decoded -->

Now we define 836

From (69), we know that 837

From (68), we know 833

From the update of z , we have 834

848

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we evaluate ∥ M k z ∥ as follows. According to (70), we know that 838

<!-- formula-not-decoded -->

With the notation M k , we know that F k ( z ) is ( β k M k ) -smooth wit respect to z . Then we have 839

<!-- formula-not-decoded -->

where ( a ) use the fact that e k ≤ 1 M k . Combining the definition of ϕ k res in (25), we obtain 840

<!-- formula-not-decoded -->

(63) and (65) imply that ∥ z k +1 -z k ∥ ≤ O (1 / √ k ) , ∥ µ k +1 -µ k ∥ ≤ O (1 / √ k ) and ∥ u k i -z k ∥ ≤ 841 O (1 / √ k ) . Combining with the fact that 0 &lt; 1 e k ≤ 1 e and 0 &lt; p &lt; 1 / 2 , we take the limit as k →∞ 842 in the above inequality and obtain that 843

<!-- formula-not-decoded -->

If the sequence { F k ( z k ) } is bounded, we know that there exists a constant M such that F k ( z k ) ≤ M 844 for all k . From the formulation ϕ fea in (26), we observe that 845

<!-- formula-not-decoded -->

where ( a ) holds from the non-negativity of L from Assumption 3.1. With the non-negativity of ϕ fea , 846 we take the limit k →∞ in the above inequality and obtain that 847

<!-- formula-not-decoded -->

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

867

868

869

870

871

872

873

874

875

876

877

878

879

880

881

882

## D Experiments

All experiments are implemented using Python 3.9 on a computer equipped with an Apple M2 chip (8-core architecture: 4 performance cores and 4 efficiency cores), running the macOS operating system with 8 GB memory. The competing methods are implemented using the code provided by [36, 22, 95].

## D.1 Introduction for Competitors

We now introduce the competing methods evaluated in our experiments:

- Grid Search : We perform a 10 × 10 uniformly-spaced grid search over the hyperparameter space.
- Random Search : We uniformly sample 100 configurations for each hyperparameter direction.
- Implicit Differentiation : This category includes IGJO [31] and IFDM [14, 15], both of which rely on implicit differentiation techniques.
- TPE : We adopt the Tree-structured Parzen Estimator approach [13], a widely used Bayesian optimization method.
- VF-iDCA : [36] formulates the lower-level problem as a value function and approximately solves the bilevel problem via DC programming.
- LDMMA : Based on lower-level duality, [22] reformulates the original problem (3) into a more tractable form.
- BiC-GAFFA : [94] solves the bilevel optimization problem using a gap function-based framework.

Weapply IFDM only to the elastic net and logistic regression problems, as its available implementation supports only these two among our tested tasks. LDMMA is used exclusively for Lasso-type regression and the smoothed support vector machine, as its reformulation is not compatible with logistic regression. Furthermore, [36] does not provide experimental results for logistic regression, and therefore we do not include it in the comparison for that task.

## D.2 Experimental on Synthetic Data

For experiments on synthetic data, we consider hyperparameter optimization for elastic net, group Lasso, and sparse group Lasso. These models are equipped with a least squares loss and different regularization terms. We outline the specific mathematical form of (3) for each problem below.

Elastic net [100] is a linear combination of the Lasso and ridge penalties. Its formulation in (3) is given by:

<!-- formula-not-decoded -->

Group Lasso [99] is an extension of the Lasso with penalty to predefined groups of coefficients. This problem is captured in (3) as:

<!-- formula-not-decoded -->

where x ( i ) is a sub-vector of x and x = ( x (1) , ..., x ( M ) ) . 883

Sparse group Lasso [83] combines the group Lasso and Lasso penalties, which are designed to 884 encourage sparsity and grouping of predictors [31]. Its formulation in (3) is represented as: 885

<!-- formula-not-decoded -->

886

887

888

889

890

891

892

893

894

895

896

897

898

899

900

901

902

903

904

905

906

907

908

909

910

911

912

913

914

915

916

917

918

919

920

921

922

923

924

where x ( i ) is a sub-vector of x and x = ( x (1) , ..., x ( M ) ) .

Based on the different cases discussed in Section 3.1 and Section 3.2, we naturally employ Algorithm 1 to solve (72), and Algorithm 2 to address (71) and (73). To evaluate the performance of each method, we calculate validation and test error with obtained LL minimizers in each experiment. We provide detailed experimental settings and report the results for elastic net and group lasso below.

## D.2.1 Elastic Net

The synthetic data is generated following the methodology described by [31], as outlined below. Feature vectors a i ∈ R p are sampled from a multivariate normal distribution with a mean of 0 and covariance structure cor( a ij , a ik ) = 0 . 5 | j -k | . The response vector b is computed as b i = β ⊤ a i + σϵ i , where β i ∈ R p is generated such that each element takes a value of either 0 or 1, with exactly 15 nonzero elements. The noise ϵ is sampled from a standard normal distribution, and the value of σ is determined to ensure that the signal-to-noise ratio satisfies SNR ∆ = ∥ A β ∥ / ∥ b -A β ∥ = 2 . Since [95] does not provide experiments or code for the elastic net problem, we compare only with search-based methods, IGJO, IFDM, VF-iDCA and LDMMA in this experiment. We implement the algorithms we compared with the same settings according to the description in [36, 22]. For LDPM with Algorithm 2, we set β k = (1 + k ) 0 . 3 , e k = 0 . 1 and γ = 10 . For elastic net problem, the stopping criterion is set as ∥ z k +1 -z k ∥ / ∥ z k +1 ∥ ≤ 0 . 1 .

We conduct repeated experiments with 10 randomly generated synthetic data, and calculate the mean and variance. The numerical results on elastic net are reported in Table 3. Overall, LDPM achieves the lowest test error while maintaining a significantly reduced time cost, especially for large-scale datasets. In contrast, the search methods incur a high computational cost and exhibit poor performance on the test dataset. The gradient-based method IGJO demonstrates slightly better accuracy and efficiency but converges very slowly.

As discussed in [36, 22], both VF-iDCA and LDMMA achieve consistently low validation errors across various experiments, indicating strong learning performance on training and validation sets. However, they tend to suffer from overfitting, as reflected in increasing test errors over iterations and poor generalization to unseen data. This phenomenon occurs across experiments with several machine learning models.

Table 3: Elastic net problems on synthetic data, where | I tr | , | I val | , | I te | and p represent the number of training observations, validation observations, predictors and features, respectively.

| Settings                                                                            | Methods                                  | Time(s)                                                                                                         | Val. Err.                                                                                                       | Test Err.                                                                                                       | Settings                                                                             | Time(s)                                                                                                              | Val. Err.                                                                                                       | Test Err.                                                                                                       |
|-------------------------------------------------------------------------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| &#124; I tr &#124; = 100 &#124; I val &#124; = 20 &#124; I te &#124; = 250 p = 250  | Grid Random IGJO IFDM VF-iDCA LDMMA LDPM | 5 . 76 ± 0 . 33 5 . 74 ± 0 . 26 1 . 54 ± 0 . 84 1 . 20 ± 0 . 50 3 . 16 ± 0 . 63 1 . 64 ± 0 . 07 0 . 60 ± 0 . 02 | 7 . 05 ± 2 . 02 7 . 01 ± 2 . 01 4 . 99 ± 1 . 69 4 . 19 ± 0 . 91 2 . 72 ± 1 . 57 0 . 00 ± 0 . 00 2 . 56 ± 0 . 80 | 6 . 98 ± 1 . 14 7 . 01 ± 1 . 11 5 . 42 ± 1 . 21 4 . 81 ± 1 . 39 5 . 18 ± 1 . 40 6 . 97 ± 0 . 79 4 . 92 ± 0 . 51 | &#124; I tr &#124; = 100 &#124; I val &#124; = 100 &#124; I te &#124; = 250 p = 450  | 11 . 72 ± 1 . 32 12 . 85 ± 2 . 11 3 . 37 ± 1 . 85 1 . 44 ± 2 . 85 6 . 08 ± 2 . 24 3 . 95 ± 0 . 22 1 . 02 ± 0 . 03    | 6 . 05 ± 1 . 47 6 . 04 ± 1 . 45 5 . 22 ± 1 . 50 4 . 89 ± 0 . 12 3 . 13 ± 0 . 78 0 . 00 ± 0 . 00 3 . 42 ± 0 . 39 | 6 . 49 ± 0 . 82 6 . 49 ± 0 . 83 5 . 72 ± 0 . 91 4 . 98 ± 0 . 17 5 . 39 ± 0 . 92 6 . 56 ± 0 . 70 4 . 23 ± 0 . 37 |
| &#124; I tr &#124; = 100 &#124; I val &#124; = 100 &#124; I te &#124; = 250 p = 250 | Grid Random IGJO IFDM VF-iDCA LDMMA LDPM | 6 . 09 ± 0 . 60 6 . 44 ± 1 . 28 3 . 86 ± 2 . 09 1 . 17 ± 0 . 38 4 . 74 ± 1 . 77 0 . 98 ± 0 . 09 0 . 73 ± 0 . 08 | 6 . 39 ± 1 . 09 4 . 39 ± 1 . 10 4 . 41 ± 0 . 98 4 . 54 ± 1 . 06 2 . 35 ± 1 . 56 0 . 00 ± 0 . 00 3 . 41 ± 0 . 48 | 6 . 27 ± 1 . 02 6 . 27 ± 1 . 05 4 . 31 ± 0 . 95 4 . 38 ± 1 . 06 4 . 47 ± 1 . 11 5 . 61 ± 0 . 77 3 . 51 ± 0 . 40 | &#124; I tr &#124; = 100 &#124; I val &#124; = 100 &#124; I te &#124; = 100 p = 2500 | 32 . 99 ± 3 . 81 33 . 82 ± 2 . 66 31 . 30 ± 6 . 41 3 . 94 ± 2 . 28 23 . 21 ± 4 . 96 16 . 26 ± 1 . 44 4 . 83 ± 0 . 08 | 7 . 81 ± 1 . 53 6 . 44 ± 1 . 53 7 . 78 ± 1 . 12 7 . 57 ± 0 . 79 0 . 00 ± 0 . 00 0 . 00 ± 0 . 00 1 . 65 ± 0 . 14 | 8 . 82 ± 0 . 92 8 . 67 ± 0 . 94 8 . 61 ± 0 . 82 8 . 10 ± 1 . 45 4 . 61 ± 0 . 77 5 . 67 ± 1 . 21 4 . 37 ± 0 . 65 |

In our experiments, we report the numerical results of VF-iDCA and LDMMA based on the final iteration output when the algorithm terminates. In contrast, [36, 22] reports the best results observed across all iterations. As a result, the test errors reported for VF-iDCA and LDMMA in Table 3 appear slightly worse in our study. Additionally, our test error is slightly worse than that reported in [22] only under the first data setting in Table 3. [22] implements LDMMA with employing off-the-shelf solver MOSEK in MATLAB to solve the subproblems. Therefore, LDMMA yields highly favorable results for small-scale problems, while its efficiency deteriorates significantly as the data size increases, making it less effective for large-scale problem instances.

We observe that the running time performance of IFDM is highly competitive and significantly fast in large scale. This is because the IFDM algorithm leverages the sparsity of the Jacobian of the hyper-objective in bilevel optimization, which is also stated in [15].

925

926

927

928

929

930

931

932

933

934

935

936

937

938

939

940

941

942

943

944

945

946

947

948

949

950

951

952

953

954

955

956

957

958

959

960

## D.2.2 Sparse Group Lasso

We generate the synthetic data with the method in [31], including 100 training, validation and test samples, respectively. The feature vector a i ∈ R p is drawn from a standard normal distribution. The response vector b is computed as b i = β ⊤ a i + σϵ i , where β = [ β (1) , β (2) , β (3) ] , β ( i ) = (1 , 2 , 3 , 4 , 5 , 0 , . . . , 0) , for i = 1 , 2 , 3 . The noise vector ϵ follows a standard normal distribution, and σ is set such that the signal-to-noise ratio (SNR) is 2. For different dimensions in Table 2, we set the group size to 30 for p = 600 and p = 1200 , and to 300 for p = 2400 and p = 4800 . Notably, compared to [36, 22], our feature vector dimensions are larger, while the number of samples is evidently smaller.

We compare our method with search methods, IGJO, VF-iDCA, LDMMA and BiC-GAFFA in this experiment. For the compared method BiC-GAFFA, we follow the recommended procedure outlined in [95]. For the other comparison methods, we adopt the exact settings from [36, 22]. For LDPM with Algorithm 2, we set β k = (1 + k ) 0 . 3 , γ = 10 and the step size e k = 0 . 001 . For sparse group Lasso problem, the stopping criterion is set as ∥ z k +1 -z k ∥ / ∥ z k +1 ∥ ≤ 0 . 2 .

From Table 2, we observe that LDPM achieves lowest test error and outperforms other algorithms in terms of time cost. As the scale of data increases, LDPM consistently finds the best hyperparameters and model solutions. In comparison, search methods become extremely unstable when facing dozens of hyperparameters. IGJO converges slowly and requires huge amount of computation. Similar to the experiments on the elastic net problem, LDMMA and VF-iDCA still exhibit a certain degree of overfitting. Both LDPM and BiC-GAFFA belong to the class of single-loop Hessian-free algorithms. Since LDPM employs projection to handle nonsmooth constraints, it achieves slightly better performance and efficiency compared to BiC-GAFFA.

## D.2.3 Group Lasso

Compared to the sparse group Lasso problem, this experiment removes the ℓ 1 -norm regularization term, leading to a reduction in the complexity of the LL problem. However, this omission also results in weaker control over the sparsity of x , potentially affecting the structure and interpretability of the solution. While the lower computational complexity may improve efficiency, the trade-off is a less strictly enforced sparsity constraint, which could affect the ability to capture key features in high-dimensional settings.

The synthetic data is generated following the same procedure as described in Appendix D.2.2. For this experiment, we adopt the same settings for other compared algorithms as those used in the experiment for the sparse group Lasso problem in Appendix D.2.2. For LDPM, we conduct Algorithm 1 with β k = (1 + k ) 0 . 3 and e k = 0 . 01 .

We conduct experiments with different data scales and report numerical results over 10 repetitions in Table 4. The overall comparison results in Table 4 are similar to those in Table 2. In this case, LDPM only requires projected gradient descent, leading to a significant improvement in efficiency.

Table 4: Group Lasso problems on the synthetic data, where p represents the number of features.

| Settings         | p = 600                           | p = 600                               | p = 600                               | p = 1200                          | p = 1200                              | p = 1200                             |
|------------------|-----------------------------------|---------------------------------------|---------------------------------------|-----------------------------------|---------------------------------------|--------------------------------------|
|                  | Time(s)                           | Val. Err.                             | Test Err.                             | Time(s)                           | Val. Err.                             | Test Err.                            |
| Grid Random IGJO | 5 . 72 ± 1 . 69                   | 93 . 20 ± 5 . 82                      | 96 . 07 ± 17 . 50                     | 12 . 31 ± 2 . 24                  | 93 . 15 ± 4 . 74                      | 94 . 60 ± 20 . 27                    |
|                  | 5 . 42 ± 1 . 81                   | 148 . 69 ± 6 . 55                     | 162 . 17 ± 28 . 09                    | 11 . 38 ± 2 . 56                  | 151 . 66 ± 15 . 63                    | 160 . 88 ± 17 . 07                   |
|                  | 1 . 42 ± 0 . 25                   | 112 . 12 ± 4 . 48                     | 105 . 99 ± 15 . 09                    | 6 . 62 ± 1 . 31                   | 143 . 62 ± 15 . 42                    | 117 . 37 ± 4 . 41                    |
| VF-iDCA          | 0 . 50 ± 0 . 14                   | 62 . 66 ± 6 . 14                      | 84 . 52 ± 12 . 46                     | 7 . 77 ± 2 . 62                   | 95 . 02 ± 7 . 04                      | 96 . 34 ± 9 . 79                     |
| LDMMA            | 0 . 51 ± 0 . 12                   | 90 . 97 ± 5 . 53                      | 79 . 68 ± 16 . 19                     | 4 . 25 ± 1 . 94                   | 92 . 32 ± 8 . 05                      | 92 . 43 ± 9 . 99                     |
| BiC-GAFFA        | 0 . 35 ± 0 . 02                   | 74 . 16 ± 6 . 91                      | 78 . 60 ± 11 . 81                     | 2 . 27 ± 0 . 26                   | 90 . 43 ± 5 . 53                      | 87 . 79 ± 8 . 43                     |
| LDPM             | 0 . 32 ± 0 . 03                   | 71 . 62 ± 7 . 28                      | 76 . 43 ± 10 . 34                     | 1 . 94 ± 0 . 13                   | 89 . 53 ± 7 . 16                      | 85 . 92 ± 6 . 99                     |
|                  | p = 2400                          | p = 2400                              | p = 2400                              | p = 4800                          | p = 4800                              | p = 4800                             |
| Settings         | Time(s)                           | Val. Err.                             | Test Err.                             | Time(s)                           | Val. Err.                             | Test Err.                            |
| Grid Random IGJO | 21 . 81 ± 3 . 65                  | 105 . 19 ± 15 . 54                    | 93 . 35 ± 16 . 60                     | 42 . 38 ± 5 . 71                  | 141 . 83 ± 26 . 52                    | 126 . 95 ± 19 . 38                   |
|                  | 19 . 95 ± 6 . 17 10 . 03 ± 6 . 69 | 132 . 04 ± 16 . 90 100 . 75 ± 16 . 47 | 161 . 45 ± 18 . 37 127 . 58 ± 16 . 43 | 41 . 67 ± 5 . 01 26 . 78 ± 8 . 50 | 109 . 35 ± 18 . 21 109 . 73 ± 16 . 66 | 134 . 74 ± 21 . 41 117 . 14 ± 8 . 23 |
| VF-iDCA          | 12 . 88 ± 1 . 31                  | 69 . 53 ± 5 . 90                      | 90 . 11 ± 11 . 59                     | 40 . 61 ± 2 . 79                  | 81 . 03 ± 11 . 58                     | 105 . 70 ± 10 . 05                   |
| LDMMA            | 6 . 75 ± 0 . 19                   | 72 . 85 ± 8 . 22                      | 87 . 00 ± 15 . 13                     | 32 . 53 ± 3 . 29                  | 86 . 47 ± 13 . 55                     | 105 . 39 ± 10 . 37                   |
| BiC-GAFFA        | 4 . 60 ± 0 . 09                   | 95 . 51 ± 14 . 88                     | 84 . 02 ± 9 . 46                      | 4 . 53 ± 0 . 57                   | 103 . 77 ± 9 . 01                     | 101 . 26 ± 7 . 84                    |
| LDPM             | 4 . 38 ± 0 . 05                   | 101 . 55 ± 7 . 28                     | 81 . 55 ± 3 . 07                      | 4 . 12 ± 0 . 15                   | 100 . 49 ± 6 . 64                     | 99 . 23 ± 6 . 31                     |

961

## D.2.4 Low-rank Matrix Completion

We consider low-rank matrix completion problem on synthetic data. The formulation in (3) of the 962 low-rank matrix completion is given as: 963

<!-- formula-not-decoded -->

964

965

966

967

968

969

970

971

972

973

974

975

976

977

978

979

980

981

982

983

984

985

986

987

988

989

990

991

992

993

994

995

The data generation procedure follows the approach in [31, 36]. Specifically, two entries per row and column are selected as the training set Ω tr , and one entry per row and column is selected as the validation set Ω val . The remaining entries form the test set Ω test. The row and column features are each grouped into 12 groups, with 3 covariates per group, resulting in p = 36 and G = 12 .

The true coefficients are set as α ( g ) = g 1 3 for g = 1 , . . . , 4 and β ( g ) = g 1 3 for g = 1 , 2 , with all other group coefficients set to zero. The low-rank effect matrix Γ is generated as a rank-one matrix Γ = uv ⊤ , where u and v are sampled from the standard normal distribution.

The row features X and column features Z are also sampled from a standard normal distribution and then scaled so that the Frobenius norm of X α 1 ⊤ +( Z β 1 ⊤ ) ⊤ matches that of Γ . Finally, the matrix observations are generated as

<!-- formula-not-decoded -->

where ϵ ij is standard Gaussian noise, and the noise level σ is chosen such that the signal-to-noise ratio (SNR) equals 2.

In this experiment, we compare LDPM with grid serach, random search, TPE, IGJO, VF-iDCA. For grid search, we explore two hyperparameters µ 1 and µ 2 with the regularization parameters defined as λ 0 = 10 µ 1 and λ g = 10 µ 2 for each g = 1 , . . . , 2 G . A 10 × 10 grid uniformly spaced over the range [ -3 . 5 , -1] × [ -3 . 5 , -1] is employed, consistent with the approach of [31]. For both the random search and TPE methods, the optimization is conducted over transformed variables u g = log 10 ( λ m ) for m = 0 , 1 , 2 , . . . , 2 G , where each u g is drawn from a uniform distribution on the interval [ -3 . 5 , -1] . For IGJO, the initial values for the regularization vector λ are set to [0 . 005 , 0 . 005 , . . . , 0 . 005] . For VF-iDCA, the initial guess for the auxiliary parameter r is chosen as [1 , 0 . 1 , 0 . 1 , . . . , 0 . 1] . The algorithm is terminated when the stopping criterion ( ∥ z k +1 -z k ∥ ) / ∥ z k ∥ ≤ 0 . 1 is satisfied. For LDPM with Algorithm 2, we set β k = (1 + k ) 0 . 3 , γ = 10 and the step size e k = 0 . 025 .

Throughout all experiments, feature grouping is performed sequentially as follows, every three consecutive features are assigned to the same group, starting from the first feature onward.

We present the statistical results in repeated experiments in Table 5. Both VF-iDCA and LDPM incur longer runtimes than search methods because they perform more intensive iterative updates-VFiDCA leverages inexact DC-programming steps to more faithfully enforce the low-rank and groupsparsity penalties. This additional computational effort yields tighter approximation of the underlying low-rank factors, resulting in substantially lower validation and test errors. LDPM repeatedly perform costly matrix projections as discussed in Appendix B.2 to enforce the rank constraints accurately. These intensive projection steps allow them to recover the underlying low-rank structure more precisely, which translates into substantially lower validation and test errors.

Table 5: Low-rank matrix completion problems on synthetic data

| Methods   | Time(s)            | Val. Acc.       | Test Acc.       |
|-----------|--------------------|-----------------|-----------------|
| Grid      | 21 . 02 ± 0 . 95   | 0 . 71 ± 0 . 21 | 0 . 76 ± 0 . 20 |
| Random    | 33 . 12 ± 2 . 10   | 0 . 72 ± 0 . 22 | 0 . 79 ± 0 . 19 |
| TPE       | 36 . 80 ± 9 . 45   | 0 . 69 ± 0 . 20 | 0 . 75 ± 0 . 18 |
| IGJO      | 1205 . 0 ± 312 . 5 | 0 . 67 ± 0 . 20 | 0 . 71 ± 0 . 17 |
| VF-iDCA   | 55 . 20 ± 12 . 05  | 0 . 65 ± 0 . 18 | 0 . 69 ± 0 . 15 |
| LDPM      | 62 . 10 ± 15 . 31  | 0 . 58 ± 0 . 14 | 0 . 66 ± 0 . 13 |

996

997

998

999

1000

1001

1002

1003

1004

1005

1006

1007

1008

1009

1010

1011

1012

1013

1014

1015

1016

1017

1018

1019

1020

1021

1022

1023

1024

1025

## D.3 Sensitivity of Parameters

In this part, we conduct experiments to analyze the sensitivity of our methods to different parameter combinations. We evaluate both Algorithm 1 and Algorithm 2. To investigate the parameter sensitivity of Algorithm 1, we carry out supplementary experiments on the group Lasso problem with a problem dimension of 1200. In each trial, we vary one parameter while keeping the others fixed. The corresponding convergence times and projected gradient descent (PGD) iteration counts are summarized in Table 6a. A similar analysis is also performed for Algorithm 2 on the sparse group Lasso instance, also with a dimension of 1200. The convergence performance, including time and steps, is likewise reported in Table 6b.

| Strategy      | e k   | β              | p    | Steps   | Time(s)          | Strategy   | e k        | β   | p       | γ     | Steps   | Time(s)   |
|---------------|-------|----------------|------|---------|------------------|------------|------------|-----|---------|-------|---------|-----------|
| Original      | 0.01  | 1              | 0.3  | 29      | 2.04             | Original   | 0.01       | 1   | 0.3     | 10    | 36      | 2.30      |
| e k           | 0.005 | 1              | 0.3  | 42      | 3.75             | e k        | 0.005 0.05 | 1 1 | 0.3 0.3 | 10    | 49      | 4.97      |
| e k           | 0.05  | 1              | 0.3  | 18      | 1.67             | e k        | 0.08       | 1   | 0.3     | 10 10 | 21 17   | 1.89 1.54 |
| e k           | 0.08  | 1              | 0.3  | 14      | 1.42             | e k        | 0.01       | 2   | 0.3     | 10    | 48      | 4.16      |
| β             | 0.01  | 2              | 0.3  | 40      | 3.60             | β          | 0.01       | 10  | 0.3     | 10    | 56      | 4.35      |
| β             | 0.01  | 10             | 0.3  | 44      | 3.89             | β          | 0.01       | 40  | 0.3     | 10    | 52      | 5.15      |
| β             | 0.01  | 40             | 0.3  | 38      | 3.95             | β          | 0.01       | 1   | 0.05    | 10    | 129     | 16.57     |
|               | 0.01  | 1              | 0.05 | 95      | 11.72            | p          | 0.01       | 10  | 0.15    | 10    | 58      | 6.12      |
| p             | 0.01  | 10             | 0.15 | 56      | 4.85             | β          | 0.01       | 40  | 0.5     | 10    | 72      | 8.83      |
| p             |       |                |      |         |                  | β          | 0.01       | 1   | 0.3     | 5     | 62      | 5.12      |
| (a) Parameter | 0.01  | 40 Sensitivity | 0.5  | 31 for  | 2.93 Algorithm 1 | γ          | 0.01       | 1   | 0.3     | 20    | 39      | 2.48      |

(b) Parameter Sensitivity for Algorithm 2

Table 6: Parameter Sensitivity Analysis for LDPM

In Algorithm 2, larger γ enforces the constraint more aggressively, so the primal residual in z -subproblem drops quickly. Smaller γ makes z -update more flexible, but the residual decays more slowly, so it end up needing more iterations and longer overall runtime. As presented in Table 6, the algorithm consistently achieves convergence and exhibits strong robustness across a broad spectrum of parameter configurations, highlighting its stability and reliability under varying conditions.

## D.4 Experimental on Real-world Datasets

This section of the experiments aims to demonstrate the numerical performance of our method on real-world datasets.

## D.4.1 Elastic Net

We consider elastic net problem on high dimendional datasets gisette and sensit. The mathmatical formulation follows (71). The datasets have a large number of features, which are suitable for evaluating the performance of regularization techniques like the elastic net. Following the approach in [36], we partition the datasets as follows: 50 and 25 examples are extracted as the training set, respectively; 50 and 25 examples are used as the validation set, respectively; and the remaining data was reserved for testing. For the same reasons as in Appendix D.2.1, we also compare LDPM with search method, IGJO, IFDM, VF-iDCA and LDMMA in this experiment. We conduct compared algorithms with the same settings as [36, 22]. For LDPM with Algorithm 2, we set β k = (1 + k ) 0 . 3 , e k = 0 . 01 and γ = 5 . The stopping criterion in this experiment is also set as ∥ z k +1 -z k ∥ / ∥ z k +1 ∥ ≤ 0 . 1 . We report the experimental results in Figure 1 and summarize them in Table 7 as auxiliary experimental results. These demonstrate that LDPM consistently achieves competitive performance while maintaining fast computational speeds on real-world datasets for elastic net problems.

As described in [36, 22], the implementation of VF-iDCA and LDMMA relies heavily on optimization 1026 solvers. In particular, the subproblems of LDMMA are entirely dependent on the commercial solver 1027 MOSEK, while the subproblems of VF-iDCA also rely on the CVXPY package, utilizing ECOS or 1028 CSC as solvers. For large-scale datasets, frequent solver calls can become a major computational 1029 bottleneck, limiting the scalability of these methods in high-dimensional or complex problem settings. 1030 Furthermore, the conic programming reformulation proposed in [22] introduces second-order cone 1031

1032

1033

1034

1035

1036

1037

1038

1039

1040

1041

1042

1043

1044

1045

1046

1047

1048

1049

1050

1051

1052

1053

Table 7: Elastic net problem on datasets gisette and sensit, where | I tr | , | I val | , | I te | and p represent the number of training samples, validation samples, test samples and features, respectively.

| Dataset Methods                                  | Time(s)                                                                                                                            | Val. Err.                                                                                                                     | Test Err.                                                                           | Dataset   | Time(s)                                                                                                                                     | Val. Err.                                                                           | Test Err.                                                                                                       |
|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| gisette Grid Random IGJO IFDM VF-iDCA LDMMA LDPM | 37 . 21 ± 4 . 80 0 56 . 67 ± 9 . 55 0 18 . 24 ± 3 . 17 0 35 . 40 ± 0 . 74 0 10 . 75 ± 2 . 72 0 9 . 45 ± 2 . 98 0 4 . 85 ± 0 . 23 0 | . 24 ± 0 . 02 0 . . 22 ± 0 . 05 0 . . 24 ± 0 . 02 0 . . 22 ± 0 . 02 0 . . 01 ± 0 . 00 0 . . 01 ± 0 . 00 0 . . 09 ± 0 . 05 0 . | 24 ± 0 . 02 26 ± 0 . 02 23 ± 0 . 03 23 ± 0 . 03 22 ± 0 . 01 21 ± 0 . 01 14 ± 0 . 03 | sensit    | 1 . 62 ± 0 . 19 1 . 1 . 46 ± 0 . 12 1 . 0 . 57 ± 0 . 14 0 . 6 . 35 ± 0 . 04 0 . 0 . 47 ± 0 . 06 0 . 0 . 41 ± 0 . 05 0 . 0 . 28 ± 0 . 02 0 . | 41 ± 0 . 75 52 ± 0 . 58 52 ± 0 . 18 37 ± 0 . 10 27 ± 0 . 03 25 ± 0 . 04 08 ± 0 . 01 | 1 . 33 ± 0 . 47 1 . 48 ± 0 . 43 0 . 61 ± 0 . 14 0 . 41 ± 0 . 23 0 . 52 ± 0 . 06 0 . 50 ± 0 . 04 0 . 34 ± 0 . 05 |

constraints, making LDMMA inherently a second-order algorithm. Consequently, its efficiency deteriorates significantly when applied to large-scale problems.

In this experiment, we omit the validation/test error-vs-time curves in Figure 1 for both the grid/random search methods and IFDM because their numerical instability leads to highly erratic traces. As discussed in [31, 14], implicit differentiation methods can suffer from numerical instability when applied to problems with sparse regularization like elastic net. In such cases, the inner optimization problems often have poor conditioning, causing oscillatory behavior during convergence.

## D.4.2 Smoothed Support Vector Machine

The smoothed support vector machine incorporates smoothed hinge loss function and squared ℓ 2 -norm regularization. The formulation in (3) of the smoothed support vector machine is given as:

<!-- formula-not-decoded -->

where l h denotes the smoothed hinge loss function detailed in Table 1. Since there is only one regularization term in (75), we conduct LDPM using Algorithm 1 according to the discussion in Section 3.

We use the LIBSVM toolbox 4 to load the datasets and extract the corresponding observation matrix and label vector for each dataset. Each dataset is divided into two separate parts: a cross-validation training set Ω consisting of 3 ⌊ N/ 6 ⌋ samples, and a test set Ω test containing the remaining samples. Within this division, the training set is further partitioned into multiple equal parts, and we iteratively use one part as the validation set while utilizing the remaining parts as the training set to solve the SVM problem. For the experiments, we conducted 6 -fold cross-validation on the training and validation sets across all three datasets to optimize the hyperparameters.

During the process of solving the smoothed support vector machine problem with K -fold crossvalidation, the loss function on the validation set is defined as follows:

<!-- formula-not-decoded -->

Following the approach used for support vector machine [48], we reformulate the primal problem 1054 into the following bilevel optimization model for the smoothed support vector machine: 1055

<!-- formula-not-decoded -->

where w 1 , w 2 , . . . , w K are K parallel copies of c and w . ¯ w ub and ¯ w lb are the upper and lower 1056 bounds of ¯ w . Similarly, we define the loss function on the training set in a manner analogous to (76): 1057

<!-- formula-not-decoded -->

4 https://www.csie.ntu.edu.tw/ cjlin/libsvmtools/datasets/

We also implement other competitive methods following the effective practice in [36, 22]. For LDPM 1058 with Algorithm 1, the penalty parameter is configured as β k = (1 + k ) 0 . 3 and the step size in each 1059 iteration is fixed at e k = 0 . 1 . We plot the convergence curves of each algorithm for validation and 1060 test error in Figure 2. 1061

Figure 2: Comparison of the algorithms for SSVM problem on real-world datasets.

<!-- image -->

## D.4.3 Sparse Logistic Regression 1062

The sparse logistic regression [46] is equipped with logistic loss function and ℓ 1 -norm regularization. 1063 Its formulation in (3) is 1064

<!-- formula-not-decoded -->

Similar to Appendix D.4.2, we also apply LDPM with Algorithm 1 in this experiment. Following the 1065 experimental setup in [15], we conduct our evaluations on large-scale real-world datasets. Specifically, 1066 we use the same datasets as [15], namely news20, rcv1 and real-sim, all of which can be downloaded 1067 from LIBSVM website 5 . Table 8 provides a brief introduction to the basic characteristics of these 1068 three datasets.

Table 8: Dataset Overview

| Datasets                           | Samples   | Features      | Sparsity   | Ratio     |
|------------------------------------|-----------|---------------|------------|-----------|
| news20.binary rcv1.binary real-sim | 19 , 996  | 1 , 355 , 191 | 0 . 034%   | 0 . 5236  |
|                                    | 20 , 242  | 47 , 236      | 0 . 155%   | 0 . 46948 |
|                                    | 72 , 309  | 20 , 958      | 0 . 245%   | 0 . 33113 |

This experiment is initially conducted in [15]. Since VF-iDCA and LDMMA are not suitable for solving large-scale problems, and the reformulation of LDMMA is not applicable to the logistic loss function, we do not compare these algorithms in this experiment. We compare our method with search methods, IFDM, and BiC-GAFFA. Random search uniformly samples 50 hyperparameter values in the interval [ λ max -4 log(10) , λ max ] . The algorithm settings for IFDM follow the configurations in [15] for each real dataset without modification. For BiC-GAFFA, we use γ 1 = 10 , γ 2 = 0 . 01 , η k = 0 . 01 , r = 5 , α k = 0 . 01 , ρ = 0 . 3 , with a maximum iteration limit of 1000 . For LDPM with Algorithm 1, we set β k = (1 + k ) 0 . 3 , e k = 0 . 05 .

In this experiment, we implement the code provided in [15]. Each experiment is repeated 10 times to compute the average and variance of runtime, validation error, validation accuracy, test error, and test accuracy. The convergence curves of each algorithm with respect to validation and test error are illustrated in Figure 3. Additionally, we calculate the corresponding accuracy and report them in Table 9.

5 https://www.csie.ntu.edu.tw/ cjlin/libsvmtools/datasets/

1069

1070

1071

1072

1073

1074

1075

1076

1077

1078

1079

1080

1081

1082

Figure 3: Comparison of the algorithms for sparse logistic regression on real-world datasets.

<!-- image -->

Table 9: Accuracy of sparse logistic regression problem on real-world datasets.

| Dataset       | Methods                    | Time(s)                                                               | Val. Acc.                                                           | Test Acc.                                                           |
|---------------|----------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------|
| news20.binary | Random IFDM BiC-GAFFA LDPM | 654 . 63 ± 33 . 26 41 . 16 ± 6 . 81 32 . 64 ± 4 . 48 30 . 85 ± 3 . 29 | 81 . 49 ± 1 . 10 86 . 87 ± 1 . 14 90 . 98 ± 1 . 03 90 . 59 ± 1 . 15 | 80 . 89 ± 1 . 24 84 . 07 ± 1 . 09 90 . 17 ± 0 . 81 92 . 94 ± 0 . 73 |
| rcv1.binary   | Random IFDM BiC-GAFFA LDPM | 214 . 46 ± 67 . 15 21 . 08 ± 5 . 47 15 . 92 ± 0 . 94 14 . 13 ± 1 . 43 | 96 . 51 ± 1 . 19 97 . 95 ± 0 . 26 98 . 72 ± 0 . 25 98 . 70 ± 0 . 33 | 94 . 24 ± 2 . 39 96 . 12 ± 1 . 29 96 . 50 ± 1 . 21 97 . 92 ± 1 . 29 |
| real-sim      | Random IFDM BiC-GAFFA LDPM | 624 . 45 ± 38 . 03 25 . 86 ± 1 . 57 18 . 08 ± 0 . 71 17 . 93 ± 0 . 68 | 68 . 30 ± 1 . 10 91 . 23 ± 2 . 18 93 . 28 ± 1 . 48 95 . 10 ± 1 . 13 | 67 . 65 ± 1 . 23 91 . 10 ± 1 . 31 91 . 68 ± 2 . 42 94 . 19 ± 1 . 57 |

Overall, we observe from Figure 3 and Table 9 that LDPM achieves the lowest time cost and test 1083 error in the experiment on sparse logistic regression. 1084

1085

1086

1087

1088

The comprehensive experimental results provide strong evidence of the efficiency and practicality of our algorithm in addressing bilevel hyperparameter optimization. These results highlight its effectiveness in real-world applications, demonstrating its ability to achieve superior performance while maintaining computational efficiency.

1089

1090

1091

1092

1093

1094

1095

1096

1097

1098

1099

1100

1101

1102

1103

1104

1105

## E Further Discussions

LDPM effectively solves bilevel optimization problems of the form (3), as demonstrated by strong empirical results. However, the core of LDPM relies on a projected gradient descent, which currently cannot handle nonsmooth loss functions without dedicated solvers, such as the hinge loss in SVMs. In contrast, [36, 22] circumvent this issue by leveraging existing solvers to deal with such nonsmooth components.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.

1106

1107

1108

1109

1110

1111

1112

1113

1114

1115

1116

1117

1118

1119

1120

1121

1122

1123

1124

1125

1126

1127

1128

1129

1130

1131

1132

1133

1134

1135

1136

1137

1138

1139

1140

1141

1142

1143

1144

1145

1146

1147

1148

1149

1150

1151

1152

1153

1154

1155

1156

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

Justification: The abstract and introduction clearly summarize the paper's key contributions and accurately reflect the scope and content presented in the main body.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper clearly acknowledges the limitations of the proposed approach, particularly regarding its assumptions and potential generalizability.

## Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

1157

1158

1159

1160

1161

1162

1163

1164

1165

1166

1167

1168

1169

1170

1171

1172

1173

1174

1175

1176

1177

1178

1179

1180

1181

1182

1183

1184

1185

1186

1187

1188

1189

1190

1191

1192

1193

1194

1195

1196

1197

1198

1199

1200

1201

1202

1203

1204

1205

1206

1207

1208

1209

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The paper clearly states all necessary assumptions and provides complete and rigorous proofs for each theoretical result.

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

Justification: The paper clearly describes the experimental settings, datasets, evaluation metrics, and implementation details, enabling reproduction of the main results and supporting the paper's key claims.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed

1210

1211

1212

1213

1214

1215

1216

1217

1218

1219

1220

1221

1222

1223

1224

1225

1226

1227

1228

1229

1230

1231

1232

1233

1234

1235

1236

1237

1238

1239

1240

1241

1242

1243

1244

1245

1246

1247

1248

1249

1250

1251

1252

1253

1254

1255

1256

1257

1258

1259

1260

1261

1262

1263

1264

instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The paper clearly describes the experimental setup, which allows for the reproducibility of the main experimental results.

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

Justification: The paper provides a clear and comprehensive description of all the necessary training and testing details, including data splits, hyperparameters, their selection process, and the type of optimizer used, which ensures that the results can be understood and reproduced.

1265

1266

1267

1268

1269

1270

1271

1272

1273

1274

1275

1276

1277

1278

1279

1280

1281

1282

1283

1284

1285

1286

1287

1288

1289

1290

1291

1292

1293

1294

1295

1296

1297

1298

1299

1300

1301

1302

1303

1304

1305

1306

1307

1308

1309

1310

1311

1312

1313

1314

1315

1316

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The paper clearly reports the statistical results, as all experiments were repeated and the corresponding statistical significance and error bars were appropriately provided, ensuring the reliability of the reported findings.

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

Justification: The paper clearly specifies the computational resources used for the experiments, including the type of compute workers, memory, and execution time, ensuring that readers can understand and reproduce the experimental setup.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

1317

1318

1319

1320

1321

1322

1323

1324

1325

1326

1327

1328

1329

1330

1331

1332

1333

1334

1335

1336

1337

1338

1339

1340

1341

1342

1343

1344

1345

1346

1347

1348

1349

1350

1351

1352

1353

1354

1355

1356

1357

1358

1359

1360

1361

1362

1363

1364

1365

1366

1367

1368

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper fully adheres to the NeurIPS Code of Ethics, ensuring that all ethical guidelines and considerations were followed during the study.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: The paper does not discuss the potential positive or negative societal impacts of the work performed, as the primary focus is on the technical and theoretical aspects of the research.

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

Justification: Since our work does not involve the release of data or models that could pose significant risks for misuse, such as pretrained language models, image generators, or scraped datasets, there are no specific safeguards required.

Guidelines:

- The answer NA means that the paper poses no such risks.

1369

1370

1371

1372

1373

1374

1375

1376

1377

1378

1379

1380

1381

1382

1383

1384

1385

1386

1387

1388

1389

1390

1391

1392

1393

1394

1395

1396

1397

1398

1399

1400

1401

1402

1403

1404

1405

1406

1407

1408

1409

1410

1411

1412

1413

1414

1415

1416

1417

1418

1419

1420

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: In our comparison experiments, we ran methods from other researchers, utilizing the code and datasets provided in their papers. We ensured proper crediting and respect for the licensing and terms of use associated with both the code and the datasets.

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

Answer: [No]

Justification: The authors have only reported on existing methods and datasets, without introducing new assets that require additional documentation.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

1421

1422

1423

1424

1425

1426

1427

1428

1429

1430

1431

1432

1433

1434

1435

1436

1437

1438

1439

1440

1441

1442

1443

1444

1445

1446

1447

1448

1449

1450

1451

1452

1453

1454

1455

1456

1457

1458

1459

1460

1461

1462

1463

1464

1465

Answer: [NA]

Justification: The paper does not involve human subjects or crowdsourcing experiments, so this question is not applicable.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Since the research does not involve human subjects or crowdsourcing experiments, there are no associated risks.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: No LLMs were used in the core methods or any important components of the research, so no declaration is required.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.