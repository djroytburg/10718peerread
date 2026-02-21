## On the Complexity of Verifying Quantized GNNs with Readout

## Anonymous Author(s)

Affiliation Address email

## Abstract

|   1 | In this paper, we introduce a logical language for reasoning about quantized graph   |
|-----|--------------------------------------------------------------------------------------|
|   2 | neural networks (GNNs) with Global Readout. We then prove that verifying quan-       |
|   3 | tized GNNs with Global Readout is NEXPTIME-complete. We also experimentally          |
|   4 | show the relevance of quantization in the context of ACR-GNNs.                       |

## 1 Introduction 5

- Graph neural networks (GNNs) are models used for classification and regression tasks on graphs or 6
- graph-node pairs, aka pointed graphs. GNNs are applied for recommendation in social network [30], 7
- knowledge graphs [40], chemistry [29], drug discovery [39], etc. 8
- Quantization designates the fact that numbers are represented by a small amount of bits, opposed 9
- to e.g., integers or real numbers whose number of bits can be arbitrary long. Standard IEEE 754 10
- 64-bit floats, INT8, or FP8 [22] enter in our setting. Essentially, our setting reflects GNNs as they are 11
- practically implemented (e.g., in PyTorch), rather than idealized GNNs that assume integer or perfect 12
- mathematical real number weights, as studied in previous research comparing GNNs and logic [4], 13
- [24] or [8]. 14
- GNNs, as several other machine learning models are difficult to interpret, understand and verify. This 15 is a major issue for their adoption, morally and legally, with the enforcement of regulatory policies 16 like the EU AI Act [13]. In the literature, verifying quantized GNNs has already been addressed [32]. 17 The methodology is to design a logical language to represent both the properties to check and the 18 computation of a GNN. However, global readout has not been considered whereas it is an essential 19 element of GNNs, especially for graph classification. 20

21

22

- In this paper, we focus on verifying Aggregate-Combine Graph Neural Networks with global Readout (ACR-GNNs) and we design a logical framework called q L .

23

24

25

26

27

28

- Example 1. Assume a class of knowledge graphs (KGs) representing communities of people and animals, where each node corresponds to an individual. Each individual can be Animal, Human, Leg,
- Fur, White, Black, etc. These concepts can be encoded with features x 0 , x 1 , . . . , x 5 , . . . respectively,
- taking values 0 or 1 . Edges in a KG represent a generic 'has' relationship: a human can have an
- animal (pet); an animal can have a human (owner), a leg, a fur; a fur can have a color; etc. Suppose
- that A is a GNN processing those KGs and is trained to supposedly recognize dogs. We can verify
- that the nodes recognized by A are animals-arguably a critical property of the domain-by checking 29
- the validity (i.e., the non-satisfiability of the negation) of φ A → x 0 = 1 where φ A is a q L -formula 30
- corresponding to A 's computation, true in exactly the pointed graphs accepted by A . Ideally, A 31
- should not overfit the concept of dog as a perfect prototypical animal. For instance, three-legged 32
- dogs do exist. We can verify that A lets it be a possibility by checking the satisfiability of the formula 33
- φ A ∧ ♢ ≤ 3 ( x 2 = 1) . 34

More complex q L formulas can be written to express graph properties to be evaluated against an 35 ACR-GNN, that will be formalized later in Example 2: 1. Has a human owner, whose pets are all 36 two-legged. 2. A human in a community that has more than twice as many animals as humans, and 37 more than five animals without an owner 1 . 3. An animal in a community where some animals have 38 white and black fur. 39

40

41

42

43

Contribution. In Section 3, we define logic q L extending the one from [32] for capturing global readout. It is expressive enough to capture quantized ACR-GNNs with arbitrary activation functions. Moreover, q L can serve as a flexible graph property specification language reminiscent of modal logics [9], for expressing e.g. properties 1-3 in Example 1.

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

Section 4 shows that the satisfiability problem of q L is in NEXPTIME, i.e. it can be decided by a non-deterministic algorithm in exponential time. To do that, we reuse the concept of mathematical logic called Hintikka sets [] which are complete sets of subformulas that can be true at a given vertex of a graph. We then introduce a quantized variant of Quantifier-Free Boolean algebra Presburger Arithmetic (QFBAPA) logic, denoted by QFBAPA /u1D542 , and prove that it is in NP as the original QFBAPA on integers. We then reduce the satisfiability problem of q L to the one of QFBAPA /u1D542 .

In Section 5, we then prove that q L is NEXPTIME-complete, while it is PSPACE-complete without global readout [32]. In a similar way, we also add global counting to the logic K ♯ previously introduced by [24]. We show that it corresponds to AC-GNNs over ℤ with global readout and trReLU activation functions. We prove that the satisfiability problem is NEXPTIME-complete, partially addressing a problem left open in the literature-that is, for the case of integer values and trReLU activation functions [7, 8]. Details are in the appendix for keep the main text concise.

As NEXPTIME is highly intractable, in Section 6, we relax the satisfiability problem of q L and ACR-GNNs, searching graph counterexamples whose number of vertices is bounded. This problem is NP-complete. We provide an implementation in this line.

We experimentally show in Section 7 that quantization of GNNs provide minimal accuracy degradation. Our results confirm that the quantized models retain strong predictive performance while achieving substantial reductions in model size and inference cost. These findings demonstrate the practical viability of quantized ACR-GNNs for deployment in resource-constrained environments.

Related work. [4] showed that ACR-GNNs are capable of capturing the expressive power of FOC 2 , that is, two-variable first-order logic with counting. Recent work has explored the logical expressiveness of GNN variants in more detail. Notably, [24] and [7] introduced logics to exactly characterize the capabilities of different forms of GNNs. Similarly, [11] analyzed Max-Sum-GNNs through the lens of Datalog. [32] considered the expressivity of GNN with quantized parameters but without global readout.

On the verification side, [17] studied the complexity of verification of quantized feedforward neural networks (FNNs), while [31, 34] investigated reachability and reasoning problems for general FNNs and GNNs. Approaches to verification are proposed via integer linear programming (ILP) by [18] and [41], and via model checking by [33].

From a logical perspective, reasoning over structures involving arithmetic constraints is closely tied to several well-studied logics. Relevant work includes Kuncak and Rinard's decision procedures for QFBAPA ([20]), as well as developments by [12], [2], [6], and [14]. These logics form the basis for the characterizations established in [24, 7].

Quantization techniques have studied in neural networks, with surveys such as [15, 23] providing com- prehensive overviews focused on maintaining model accuracy. Although most practical advancements

target convolutional neural networks (CNNs), many of the underlying principles extend to GNNs as well ([42]). NVIDIA has demonstrated hardware-ready quantization strategies ([38]), and frameworks

like PyTorch ([1]) support both post-training quantization and quantization-aware training (QAT), the latter simulating quantization effects during training to improve low-precision performance. QAT has

been particularly effective in closing the gap between quantized and full-precision models, especially for highly compressed or edge-deployed systems ([19]).

In the context of GNNs, [35] proposed

Degree-Quant, incorporating node degree information to mitigate quantization-related issues. Based

1 Interestingly, q L goes beyond graded modal logic and even first-order logic. The property of item 2 in Example 1 cannot be expressed in FOL.

Figure 1: DAG data structure for the formula agg ( x 1 + x 2 ) + ( x 1 + x 2 ) ≥ 3 .

<!-- image -->

on this, [43] introduced A 2 Q , a mixed-precision framework that adapts bitwidths on graph topology 86 to achieve high compression with minimal performance loss. 87

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

112

113

114

115

116

117

118

119

## 2 Background

Let /u1D542 be a set of quantized numbers, and let n denote the bitwidth of /u1D542 , that is, the number of bits required to represent a number in /u1D542 . The bitwidth n is written in unary; this is motivated by the fact that n is small and that we would in any case need to allocate n -bit consecutive memory for storing a number. Formally, we consider a sequence /u1D542 1 , /u1D542 2 , . . . corresponding to bitwidths 1, 2, etc., but we retain the notation /u1D542 for simplicity. We suppose that /u1D542 saturates: e.g., if x ≥ 0 , y ≥ 0 , x + y ≥ 0 (i.e., no modulo behavior like in int in C for instance). We suppose that 1 ∈ /u1D542 .

We consider Aggregate-Combine Graph Neural Networks with global Readout (ACR-GNNs), a standard class of message-passing GNNs [4, 16]. An ACR-GNN layer is defined by a triple ( comb , agg , agg g ) , where comb : /u1D542 3 m → /u1D542 n is a combination function, and agg , agg g are local and global aggregation functions that map multisets of vectors in /u1D542 m to a single vector in /u1D542 m .

An ACR-GNN is composed of a sequence of such layers ( L (1) , . . . , L ( L ) ) followed by a final classification function cls : /u1D542 m →{ 0 , 1 } . Given a graph G = ( V, E ) and an initial node labelling x 0 : V →{ 0 , 1 } k , the state of a node u in layer i is recursively defined as:

```
x i ( u ) = comb ( x i -1 ( u ) , agg ( {{ x i -1 ( v ) | uv ∈ E }} ) , agg g ( {{ x i -1 ( v ) | v ∈ V }} ))
```

The final output of the GNN for a pointed graph ( G,u ) is A ( G,u ) = cls ( x L ( u )) . A more detailed definition is provided in Appendix C.2.

Our study focuses on a specific subclass where both agg and agg g perform summation over vectors, and where comb ( x, y, z ) = ⃗ σ ( xC + yA 1 + zA 2 + b ) , using matrices C, A 1 , A 2 with entries from /u1D542 , and a bias b ∈ /u1D542 . The classification function is a linear threshold: cls ( x ) = ∑ i a i x i ≥ 1 with weights a i ∈ /u1D542 . Moreover, we assume that all arithmetic operations are executed according to the arithmetic related to /u1D542 . It is assumed that the context makes clear the /u1D542 and arithmetic being used. We note [[ A ]] the set of pointed graphs ( G,u ) such that A ( G,u ) = 1 . An ACR-GNN A is satisfiable if [[ A ]] is non-empty. The satisfiability problem for ACR-GNNs is: Given a ACR-GNN A , decide whether A is satisfiable.

## 3 Logic q L for Representing GNN Computations and Properties on Graphs

We set up a logical framework called q L extending the logic in [32] with global aggregation: it is a lingua franca to represent GNN computation and properties on graphs.

Syntax. Let F be a finite set of features and /u1D542 be some finite-width arithmetic. We consider a set of expressions defined by the following grammar in Backus-Naur form:

<!-- formula-not-decoded -->

where c is a number in /u1D542 , x i is a feature in F , α is a symbol for denoting the activation function, and agg and agg ∀ denote the aggregation function for local and global readout respectively. A formula is a construction of the formula ϑ ≥ k where ϑ is an expression and k is an element of /u1D542 . If -1 ∈ /u1D542 , and -ϑ is not, we can write -ϑ instead of ( -1) × ϑ . Other standard abbreviations can be used.

Formulas are represented as direct acyclic graphs, aka circuits, meaning that we do not repeat the same 120 expressions several times. For instance, the formula agg ( x 1 + x 2 )+( x 1 + x 2 ) ≥ 3 can be represented 121 as the DAG given in Figure 1. Formulas can also be represented by a sequence of assignments via 122 new fresh intermediate variables. For instance: y := x 1 + x 2 , z := agg ( y ) + y, res := z ≥ 3 . 123

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

Semantics. Consider a graph G = ( V, E ) , where vertices in V are labeled via a labeling function ℓ : V → /u1D542 n with feature values. The value of an expression ϑ in a vertex u ∈ V is denoted by [[ ϑ ]] G,u and is defined by induction on ϑ :

```
[[ c ]] G,u = c, [[ x i ]] G,u = ℓ ( u ) i , [[ ϑ + ϑ ′ ]] G,u = [[ ϑ ]] G,u + /u1D542 [[ ϑ ′ ]] G,u , [[ c × ϑ ]] G,u = c × /u1D542 [[ ϑ ]] G,u , [[ α ( ϑ )]] G,u = [[ α ]]([[ ϑ ]] G,u ) , [[ agg ( ϑ )]] G,u = Σ v | uEv [[ ϑ ]] G,v , [[ agg ∀ ( ϑ )]] G,u = Σ v ∈ V [[ ϑ ]] G,v ,
```

We define [[ ϑ ≥ k ]] = { G,u | [[ ϑ ]] G,u ≥ /u1D542 [[ k ]] G,u } (we write ≥ for the symbol in the syntax and ≥ /u1D542 for the comparison in /u1D542 ). A formula φ is satisfiable if [[ φ ]] is non-empty. The satisfiability problem for q L is: Given a q L -formula φ , decide whether φ is satisfiable.

̸

ACR-GNN verification tasks. We are interested in the following decision problems. Given a GNN A , and a q L formula φ : (VT1, sufficiency) Do we have [[ φ ]] ⊆ [[ A ]] ? (VT2, necessity) Do we have [[ A ]] ⊆ [[ φ ]] ? (VT3, consistency) Do we have [[ φ ]] ∩ [[ A ]] = ∅ ?

Representing a GNN computation. To reason formally about ACR-GNNs, we represent their computations using q L . Logic q L facilitates the modeling of the acceptance condition of ACR-GNNs.

We explain this via example. Consider a two-layer ACR-GNN A with input and output dimension 2, using summation for aggregation, activation via α ( x ) := max(0 , min(1 , x )) -the truncated ReLUand a classification function 2 x 1 -x 2 ≥ 1 . The combination functions are:

```
comb 1 (( x 1 , x 2 ) , ( y 1 , y 2 ) , ( z 1 , z 2 )) := ( σ (2 x 1 + x 2 +5 y 1 -3 y 2 +1) σ ( -x 1 +4 x 2 +2 y 1 +6 y 2 -2) ) , comb 2 (( x 1 , x 2 ) , ( y 1 , y 2 ) , ( z 1 , z 2 )) := ( σ (3 x 1 -y 1 +2 z 2 ) σ ( -2 x 1 +5 y 2 +4 z 1 ) ) .
```

Note that this assumes that A operates over /u1D542 with at least three bits. Then, the corresponding q L formula φ A is given by: ψ 1 = α (2 x 1 + x 2 + 5 agg ( x 1 ) -3 agg ( x 1 ) + 1) , ψ 2 := α ( -x 1 + 4 x 2 +2 agg ( x 1 ) + 6 agg ( x 2 ) -2) , χ 1 := α (3 ψ 1 -agg ( ψ 1 ) + 2( agg ∀ ( psi 2 ))) , χ 2 := α ( -2 ψ 1 + 5( agg ( ψ 2 )) + 4 agg ∀ ( psi 1)) , φ A := 2( χ 1 ) -χ 2 ≥ 1 . To sum up, given a GNN A , we compute q L -formula in poly-time in the size of A with [[ A ]] = [[ φ A ]] (as done in [32]).

Simulating a modal logic in the logic q L . In this section, we show that extending q L with modal operators [9] does not increase the expressivity. We can even compute an equivalent q L without Boolean connectives and without modal operators in poly-time. It means that formulas like φ A 1 → x 0 = 1 or φ A 1 ∧ ♢ ≤ 3 ( x 2 = 1) have equivalent formulas in q L .

Assume that α is ReLU. Let Atm 0 be the set of atomic formulas of q L of the form ϑ ≥ 0 . We suppose that ϑ takes integer values. In general, ϑ ≥ k is an atomic formula equivalent to ϑ -k ≥ 0 . Without loss of generality, we thus assume that formulas of q L are over Atm 0 . Let modal q L be the propositional logic on Atm 0 extended with modalities and a restricted variant of graded modalities where number k in /u1D542 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

```
} g }
```

and modalities ♢ ≤ k φ and ♢ ≤ k g φ defined the same way but with ≤ /u1D542 . We can turn back to the graph properties mentioned in Example 1.

Example 2. We first define a few simple formulas to characterize the concepts of the domain. Let φ A := x 0 = 1 (Animal), φ H := x 1 = 1 (Human), φ L := x 2 = 1 (Leg), φ F := x 3 = 1 (Fur), φ W := x 4 = 1 (White), and φ B := x 5 = 1 (Black).

1. Has a human owner, whose all pets are two-legged: ♢ ( φ H ∧ □ ( φ A → ♢ =2 φ L )) .
2. A human in a community that has more that twice as many animals as humans, and more than five animals without an owner: φ H ∧ ( agg ∀ ( x 0 ) -2 × agg ∀ ( x 1 ) ≥ 0) ∧ ♢ ≥ 5 g (( φ A ∧ □ ( ¬ φ H )) .

162

163

164

165

3. An animal in a community where some animals have white and black fur:

```
.
```

```
φ A ∧ ♢ g ( ♢ ( φ F ∧ ♢ φ W ) ∧ ♢ ( φ F ∧ ♢ φ B ))
```

We can see the boolean operator ¬ , and the various modalities as functions from Atm 0 into Atm 0 , and the boolean operator ∨ as a function from Atm 0 × Atm 0 to Atm 0 .

```
f ¬ ( ϑ ≥ 0) := -ϑ -1 ≥ 0 f ∨ ( ϑ 1 ≥ 0 , ϑ 2 ≥ 0) := ϑ 1 + ReLU ( ϑ 2 -ϑ 1 ) ≥ 0 f □ ( ϑ ≥ 0) := agg ( -ReLU ( -ϑ )) ≥ 0 f ♢ ≥ k ( ϑ ≥ 0) := agg ( ReLU ( ϑ +1) -ReLU ( ϑ )) -k ≥ 0 f ♢ ≤ k ( ϑ ≥ 0) := k -agg ( ReLU ( ϑ +1) -ReLU ( ϑ )) ≥ 0
```

For the corresponding global modalities ( f □ g ( ϑ ≥ 0) , f ♢ ≥ k ( ϑ ≥ 0) , and f ♢ ≤ k ( ϑ ≥ 0) ), it suffices to 166 use agg ∀ in place of agg . The previous transformations can be generalized to arbitrary formulas of 167 modal q L as follows. 168

```
mod 2 expr ( ϑ ≥ 0) := ϑ ≥ 0 mod 2 expr ( ¬ φ ) := f ¬ ( mod 2 expr ( φ )) mod 2 expr ( φ 1 ∨ φ 2 ) := f ∨ ( mod 2 expr ( φ 1 ) , mod 2 expr ( φ 2 )) mod 2 expr ( ⊞ φ ) := f ⊞ ( mod 2 expr ( φ )) , ⊞ ∈ { □ , □ g , ♢ ≥ k , ♢ ≥ k g , ♢ ≤ k , ♢ ≤ k g }
```

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

We can show that formulas of modal q L can be captured by a unique expression ϑ ≥ 0 . This is a consequence of the following lemma 2 .

Lemma 3. Let φ be a formula of modal q L . The formulas φ and mod 2 expr ( φ ) are equivalent.

Now, ACR-GNN verification tasks can be solved by reduction to the satisfiability problem of q L . VT1 by checking that φ ∧ ¬ φ A is not satisfiable; VT2 by checking that ¬ φ ∧ φ A is not satisfiable; VT3 by checking that φ ∧ φ A is satisfiable.

## 4 NEXPTIME Membership of the Satisfiability Problem

In this section, we prove the NEXPTIME membership of reasoning in modal quantized logic, and also of solving of ACR-GNN verification tasks (by reduction to the former). Remember that the activation function α can be arbitrary in our setting. Our result holds with the loose restriction that [[ α ]] is computable in exponential-time in the bit-width n of /u1D542 .

Theorem 4. The satisfiability problem of q L is decidable and in NEXPTIME, and so is VT3 . VT1 and VT2 are in coNEXPTIME.

In order to prove Theorem 4, we adapt the NEXPTIME membership of the description logic ALCSCC ++ from [2] to logic q L . The difference resides in the definition of Hintikka sets and the treatment of quantization. The idea is to encode the constraints of a q L -formula φ in a formula of exponential length of a quantized version of QFBAPA, that we prove to be in NP.

## 4.1 Hintikka Sets

Consider q L -formula φ . Let E ( φ ) be the set of subexpressions in φ . For instance, if φ is 3 × agg ( α ( x 2 + agg ∀ ( x 1 ))) ≥ 5 then E ( φ ) := { agg ( α ( x 2 + agg ∀ ( x 1 )) , α ( x 2 + agg ∀ ( x 1 ) , x 2 , agg ∀ ( x 1 ) , x 1 } . From now on, we consider equality subformulas that are of the form ϑ = k where ϑ is a subexpression of φ and k ∈ /u1D542 .

Definition 5. A Hintikka set H for φ is a subset of subformulas of φ such that:

1. For all ϑ ∈ E ( φ ) , there is a unique value k ∈ /u1D542 such that ϑ = k ∈ H
2. ϑ = k , ϑ = k ∈ H then ϑ + ϑ = k + k ∈ H
3. 1 1 2 2 1 2 1 2 3. If ϑ ≥ k ∈ H then c × ϑ = k ′ ∈ H where k ′ = c × /u1D542 k 4. ϑ = k ∈ H and α ( ϑ )= k ′ implies k ′ = [[ α ]]( k )

2 For simplicity, we do not present how to handle ϑ ≥ 0 when ϑ is not an integer. We could introduce several activation functions α in q L , one of them could be interpreted as the Heavyside step function. In the sequel Definition 5, Point 4 is just repeated for each α .

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

Informally, a Hintikka set is a set of equality subformulas obtained from a choice of a value for each subexpression of φ (point 1), provided that the set is consistent at the current vertex (point 2-4). Note that the notion of Hintikka set does not take any constraints about agg and agg ∀ into consideration since checking consistency of aggregation would require information about the neighbor or the whole graph.

Example 6. If φ is 3 × agg ( α ( x 2 + agg ∀ ( x 1 ))) ≥ 5 then the following set is an example of Hintikka set: { agg ( α ( x 2 + agg ∀ ( x 1 )) = 8 , α ( x 2 + agg ∀ ( x 1 )) = 9 , x 2 + agg ∀ ( x 1 ) = 9 , x 2 = 7 , agg ∀ ( x 1 ) = 2 , x 1 = 5 } .

Proposition 7. The number of Hintikka sets is bounded by 2 n | φ | where | φ | is the size of φ , and n is the bitwidth of /u1D542 .

## 4.2 Quantized Version of QFBABA (Quantifier-free Boolean Algebra and Presburger Arithmetics)

AQFBAPA formula is propositional formula where each atom is either an inclusion of sets or equality of sets or linear constraints [20]. Sets are denoted by Boolean algebra expression e.g., ( S ∪ S ′ ) \ S ′′ , or U where U denotes the set of all points in some domain. Here S , S ′ , etc. are set variables. Linear constraints are over | S | denoting the cardinality of the set denoted by the set expression S . For instance, the QFBAPA-formula ( pianist ⊆ happy ) ∧ ( | happy | + |U \ pianist | ≥ 6) ∧ ( | happy | &lt; 2) is read as 'all pianists are happy and the number of happy persons + the number of persons that are not pianists is greater than 6 and the number of happy persons is smaller than 2'.

We now introduce a quantized version QFBAPA /u1D542 of QFBAPA. It has the same syntax as QFBAPA except that hard-coded numbers in expressions are in /u1D542 . Concerning the semantics, every numerical expression is interpreted in /u1D542 . For each set expression S , the interpretation of | S | is not the cardinality c of the interpretation of S , but the result of the computation 1 + 1 + . . . +1 in /u1D542 with c occurrences of 1 in the sum.

We consider that /u1D542 that saturates, meaning that if x + y exceed the upper bound limit of /u1D542 , there is a special value denoted by + ∞ such that x + y = + ∞ .

Proposition 8. If bitwidth n is in unary, and if /u1D542 saturates, then satisfiability in QFBAPA /u1D542 is in NP .

## 4.3 Reduction to QFBAPA /u1D542

Let φ be a formula of q L . For each Hintikka set H , we introduce the set variable X H that intuitively represents the H -vertices, i.e., the vertices in which subformulas of H hold. The following QFBAPA /u1D542 -formulas say that the interpretation of X H form a partition of the universe. For each subformula ϑ ′ = k , we introduce the set variable X ϑ ′ = k that intuitively represents the vertices in which ϑ ′ = k holds. Formula (1) expresses that { X H } H form a partition of the universe. Formula (2) makes the bridge between variables X ϑ ′ = k and X H .

̸

<!-- formula-not-decoded -->

We introduce also a variable S H that denotes the set of all successors of some H -vertex. If there is no H -vertex then the variable S H is just irrelevant.

The following QFBAPA /u1D542 -formula encodes the semantics of agg ( ϑ ) . More precisely, it says that for all subexpressions agg ( ϑ ) , for all values k , for all Hintikka sets H containing subformula agg ( ϑ )= k , for all H containing agg ( ϑ )= k , it says that, if there is some H -vertex (i.e., vertices in S H ), then the aggregation obtained by summing over the successors of some H -vertex is k .

̸

<!-- formula-not-decoded -->

In the previous sum, we partition S H into subsets S H ∩ X ϑ = k ′ for all possible values k ′ . Each contribution for a successor in S H ∩ X ϑ = k ′ is k ′ . Werely here on the fact 3 that (1+1+ . . . +1) × k ′ =

3 This is true for some fixed-point arithmetics but not for floating-point arthmetics. See Appendix B.

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

Figure 2: Encoding a torus of exponential size with (modal) q L formulas. ( x, y ) are the vertices of the graph that correspond to locations in the torus while φ N and φ E denote intermediate vertices indicating the direction (resp., north and east).

<!-- image -->

k ′ + k ′ + . . . + k ′ . We also fix a specific order over values k ′ in the summation (it means that agg ( ϑ ) is computed as follows: first order the successors according to the taken values of ϑ in that specific order, then perform the summation). Finally, the semantics of agg ∀ is captured by the formula:

̸

<!-- formula-not-decoded -->

Note that intuitively Formula (4) implies that for X agg ∀ ( ϑ )= k is interpreted as the universe, for the value k which equals the semantics of ∑ k ′ ∈ /u1D542 | X ϑ = k ′ | × k ′ .

̸

Given φ = ϑ ≥ k , we define tr ( φ ) := ψ ∧ ∨ k ′ ≥ k X ϑ = k ′ = ∅ where ψ the conjunction of Formulas 1-4. The function tr requires to compute all the Hintikka sets. So we need in particular to check Point 4 of Definition 5 and we get the following when [[ α ]] is computable in exponential time in n .

Proposition 9. tr ( φ ) is computable in exponential-time in | φ | and n .

Proposition 10. Let φ be a formula of q L . φ is satisfiable iff tr ( φ ) is QFBAPA /u1D542 satisfiable.

Finally, in order to check whether a q L -formula φ is satisfiable, we construct a QFBAPA /u1D542 -formula tr ( φ ) in exponential time. As the satisfiability problem of QFBAPA /u1D542 is in NP, we obtain that the satisfiability problem of q L is in NEXPTIME. We proved Theorem 4,

Remark 11. Our methodology can be generalized to reason in subclasses of graphs. For instance, we may tackle the problem of satisfiability in a graph where vertices are of bounded degree bounded by d . To do so, we add the constraint ∧ H | S H | ≤ d .

## 5 Complexity Lower Bound

The NEXPTIME upper-bound is tight. Having defined modalities in q L and stated Lemma 3, Theorem 12 is proven by adapting the proof of NEXPTIME-hardness of deciding the consistency of ALCQ -T C Boxes presented in [36]. So we already have the hardness result for ReLU.

NEXPTIME-hardness is proven via a reduction from the tiling problem by Wang tiles of a torus of size 2 n × 2 n . A Wang tile is a square with colors, e.g., , , etc. That problem takes as input a number n in unary, and Wang tile types, and an initial condition - let say the bottom row is already given. The objective is to decide whether the torus of 2 n × 2 n can be tiled while colors of adjacent Wang tiles match. A slight difficulty resides in adequately capturing a two-dimensional grid structure-as in Figure 2-with only a single relation. To do that, we introduce special formulas φ E and φ N to indicate the direction (east or north). In the formula computed by the reduction, we also need to bound the number of vertices corresponding to tile locations by 2 n × 2 n . Thus /u1D542 needs to encode 2 n × 2 n . We need a bit-width of at least 2 n .

Theorem 12. The satisfiability problem in q L is NEXPTIME-hard, and so is VT3 . VT1 and VT2 are coNEXPTIME-hard.

Remark 13. It turns out that the verification task only needs the fragment of q L where agg is applied directly on an expression α ( .. ) . Indeed, this is the case when we represent a GNN in q L or when we translate logical formulas in q L (Lemma 3). Reasoning about q L when /u1D542 = ℤ and the activation function is truncated ReLU is also NEXPTIME-complete (see Appendix E).

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

## 6 Bounding the Number of Vertices

̸

The satisfiability problem is NEXPTIME-complete, thus far from tractable. The complexity comes essentially because counterexamples can be arbitrary large graphs. However, usually we are search for small counterexamples. Let G ≤ N be the set of pointed graphs with at most N vertices. We consider the q L and ACR-GNN satisfiability problems with a bound on the number of vertices : given a number N given in unary, 1. given a q L -formula φ , is it the case that [[ φ ]] ∩ G ≤ N = ∅ , 2. given an ACR-GNN A , is it the case that [[ A ]] ∩ G ≤ N = ∅ .

Theorem 14. The satisfiability problems with bounded number of vertices are NP-complete.

We then can extend the methodology of [33] but for verifying GNNs. Our implementation proposal is a Python program that takes a learnt quantized GNN A as an input, a precondition, a postcondition and a bound N . It then produces a C program that mimics the execution of A on an arbitrary graph with at most N vertices, and embeds the pre/postcondition. We then apply ESBMC (efficient SMT-based context-bounded model checker) [21] on the C program.

## 7 Quantization Effects on Accuracy, Performance and Model Size

To confirm that the GNN models considered in this paper are promising, we now investigate the application of Dynamic Post-Training Quantization (PTQ) to Aggregate-Combined Readout Graph Neural Networks (ACR-GNNs). Our experimental design builds on the framework introduced in [4], using their publicly available implementation [5] as the baseline. ACR-GNNs with specific structural configurations are used as the primary model class for evaluation. Dynamic PTQ, implemented in PyTorch [1, 26], converts a pre-trained floating-point model into a quantized version without retraining. This approach quantizes weights to INT8 statically, while activations remain in floating point until dynamically quantized at compute time. This enables efficient INT8-based computation, reducing memory usage and improving inference speed. PyTorch's implementation employs pertensor quantization for weights and stores activations in floating-point format between operations. The evaluation focuses on accuracy, model size, and latency. Experiments are conducted on both synthetic and real-world datasets, with the synthetic benchmark-based on dense Erd ¨ os-R ´ enyi graph structures and logical labeling schemes-serving as the primary focus.

The synthetic graphs were generated using the dense Erd ¨ os-R ´ enyi model, a classical approach for constructing random graphs. Each graph includes five initial node colours, encoded as one-hot feature vectors. Following [4], labels were assigned using formulas from the logic fragment FOC 2 . Specifically, a hierarchy of classifiers α i ( x ) was defined as:

<!-- formula-not-decoded -->

where ∃ [ N,M ] denotes the quantifier 'there exist between N and M nodes" satisfying a given condition. Each classifier α i ( x ) can be expressed within FOC 2 , as the bounded quantifier can be rewritten using ∃ ≥ N and ¬∃ ≥ M +1 . Each property p i corresponds to a classifier α i with i ∈ 1 , 2 , 3 . Summary statistics for the dataset are provided in Appendix G, Table 3.

Table 1: Accuracy difference (%) and model size (MB) of the ACR-GNN model before and after dynamic post-training quantization (PTQ) across FO-properties p 1 , p 2 , and p 3 . Values are reported for three model depths (1, 2, and 3 layers) and three dataset splits (Train, Test 1, Test 2). Accuracy values represent the change after quantization (QINT8 - FP32). p 1 , p 2 , p 3 are FO-properties described in Appendix G.

|    | p 1     | p 1     | p 1     | p 2     | p 2     | p 2     | p 3     | p 3     | p 3     |           |
|----|---------|---------|---------|---------|---------|---------|---------|---------|---------|-----------|
| #  | Train   | Test 1  | Test 2  | Train   | Test 1  | Test 2  | Train   | Test 1  | Test 2  | Size (MB) |
| 1  | -0.452% | -0.760% | +0.522% | -0.127% | -0.183% | +8.891% | -0.299% | -0.648% | -0.693% | 0.034     |
| 2  | -0.001% | 0.000%  | -0.043% | +0.083% | -0.125% | +0.144% | -0.178% | -0.226% | +0.018% | 0.068     |
| 3  | -0.036% | +0.062% | -0.494% | -0.161% | -0.143% | -0.342% | -0.015% | +0.280% | -0.346% | 0.103     |

Table 1 presents the difference in accuracy and model size between the quantized (QINT8 4 ) and original (FP32) versions of the ACR-GNN model across three configurations (1, 2, and 3 layers). The

4 The difference between INT8 and QINT8 lies in their implementation and is detailed in Appendix G

̸

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

354

evaluation is conducted on three FO-properties ( p 1 , p 2 , p 3 ) over three data splits: Train, Test1, and Test2. The table highlights how quantization affects accuracy at various depths. In most cases, the impact of quantization on accuracy is minor and bounded, with some configurations even showing positive differences. For instance, in the 2-layer configuration-the overall best performer-the accuracy loss remains within ± 0 . 1 across all properties and splits, while yielding a model size reduction of 0.068 MB. The 1-layer model shows greater fluctuation: while p 2 on Test2 experiences a significant positive spike (+8.891), p 3 on Test2 drops by -0.693. This suggests sensitivity to quantization in shallow models, likely due to limited representational capacity. The results confirm that dynamic post-training quantization (PTQ) enables significant compression-up to 60% reduction in size-while maintaining acceptable levels of accuracy. Additional breakdowns, including baseline results and extended configurations, are provided in Appendix G.

Table 2: PPI benchmark. Accuracy (%) and size (MB) of the ACR-GNN with ReLU activation function before and after dynamic PTQ across different layer configurations.

|    | Original (FP32)   | Original (FP32)   | Original (FP32)   | Original (FP32)   | Quantized (QINT8)   | Quantized (QINT8)   | Quantized (QINT8)   | Quantized (QINT8)   | Difference   | Difference   | Difference   | Difference   |
|----|-------------------|-------------------|-------------------|-------------------|---------------------|---------------------|---------------------|---------------------|--------------|--------------|--------------|--------------|
| #  | Train             | Val               | Test              | Size (MB)         | Train               | Val                 | Test                | Size (MB)           | Train        | Val          | Test         | Size (MB)    |
| 1  | 54.7%             | 43.1%             | 39.5%             | 0.922             | 55.0%               | 50.8%               | 50.2%               | 0.242               | +0.3%        | +7.7%        | +10.7%       | 0.680        |
| 2  | 52.5%             | 44.6%             | 45.7%             | 1.718             | 52.3%               | 47.8%               | 47.2%               | 0.451               | -0.2%        | +3.2%        | +1.5%        | 1.267        |
| 3  | 52.3%             | 42.6%             | 44.0%             | 2.515             | 51.9%               | 45.7%               | 42.8%               | 0.660               | -0.4%        | +3.1%        | -1.2%        | 1.855        |

Table 2 shows the results of evaluating the ACR-GNN model on the Protein-Protein Interaction (PPI) benchmark before and after applying dynamic post-training quantization (PTQ). The evaluation covers three model configurations (1 to 3 layers) and reports performance in terms of accuracy (Train, Validation, and Test) and model size (in MB). Quantization results in substantial compression across all configurations. The model size decreases from 0.922 MB to 0.242 MB (a 73% reduction) for the 1-layer network, while the 2- and 3-layer models achieve reductions of 1.267 MB and 1.855 MB, respectively. Accuracy-wise, quantization leads to improvements in the Validation and Test sets for shallower networks. The 1-layer model gains +0.077 on validation and +0.107 on test accuracy, indicating potential for enhanced generalization. The 2-layer model shows minor improvements across all splits, with negligible loss in training accuracy. However, the 3-layer configuration reveals a slight drop in test accuracy (-0.012), suggesting increased sensitivity to quantization at greater depth. See Appendix G, Tables 16,17, and 18 for additional quantitative breakdowns.

## 8 Conclusion and Future Work

The central result is the NEXPTIME-complete of the logic q L in which both the computations of GNNs and modal properties can be expressed. It helps to understand the inherent complexity of verifying quantized GNNs. We also provide a prototype for verifying GNNs over a set of graphs with a bounded number of vertices. Finally some experiments confirmed that the quantization of ACR-GNNs is promising.

There are many directions to go. First, characterizing the modal flavor of q L for other activation functions than ReLU. New extensions of q L could be proposed to tackle other classes GNNs. Verification of neural networks is challenging and is currently tackled by the verification community [10]. So it will be for GNNs as well. Our verification tool with a bound on the number of vertices is still preliminary. One obvious path would be to improve the tool, to compare different approaches (bounded model checking vs. linear programming as in [18]) and apply it to real GNN verification scenarios. Designing a practical verification procedure in the general case (without any bound on the number of vertices) and overcoming the high computational complexity is an exciting challenge for future research towards the verification of GNNs.

Limitations. Section 4 and 5 reflect theoretical results. Some practical implementations of GNNs may not fully align with them. In particular, the order in the (non-associative) summation over values in /u1D542 is fixed in formulas (3) and (4). It means that we suppose that the aggregation agg ( ϑ ) is computed in that order too (we sort the successors of a vertex according the values of ϑ and then perform the summation). The verification tool discussed in Section 6 remains a prototype, thus its application warrants careful consideration.

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

399

400

401

402

403

404

## References

- [1] Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, CK Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Michael Suo, Phil Tillet, Eikan Wang, Xiaodong Wang, William Wen, Shunting Zhang, Xu Zhao, Keren Zhou, Richard Zou, Ajit Mathews, Gregory Chanan, Peng Wu, and Soumith Chintala. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24) . ACM, April 2024.
- [2] Franz Baader, Bartosz Bednarczyk, and Sebastian Rudolph. Satisfiability and query answering in description logics with global and local cardinality constraints. In Giuseppe De Giacomo, Alejandro Catalá, Bistra Dilkina, Michela Milano, Senén Barro, Alberto Bugarín, and Jérôme Lang, editors, ECAI 2020 - 24th European Conference on Artificial Intelligence, 29 August-8 September 2020, Santiago de Compostela, Spain, August 29 - September 8, 2020 - Including 10th Conference on Prestigious Applications of Artificial Intelligence (PAIS 2020) , volume 325 of Frontiers in Artificial Intelligence and Applications , pages 616-623. IOS Press, 2020.
- [3] Franz Baader, Ian Horrocks, Carsten Lutz, and Uli Sattler. Introduction to Description Logic . Cambridge University Press, 2017.
- [4] Pablo Barceló, Egor V. Kostylev, Mikaël Monet, Jorge Pérez, Juan L. Reutter, and Juan Pablo Silva. The logical expressiveness of graph neural networks. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net, 2020.
- [5] Pablo Barceló, Egor V. Kostylev, Mikaël Monet, Jorge Pérez, Juan L. Reutter, and Juan Pablo Silva. Gnn-logic. https://github.com/juanpablos/GNN-logic.git , 2021.
- [6] Bartosz Bednarczyk, Maja Orlowska, Anna Pacanowska, and Tony Tan. On classical decidable logics extended with percentage quantifiers and arithmetics. In Mikolaj Bojanczyk and Chandra Chekuri, editors, 41st IARCS Annual Conference on Foundations of Software Technology and Theoretical Computer Science, FSTTCS 2021, December 15-17, 2021, Virtual Conference , volume 213 of LIPIcs , pages 36:1-36:15. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2021.
- [7] Michael Benedikt, Chia-Hsuan Lu, Boris Motik, and Tony Tan. Decidability of graph neural networks via logical characterizations. In Karl Bringmann, Martin Grohe, Gabriele Puppis, and Ola Svensson, editors, 51st International Colloquium on Automata, Languages, and Programming, ICALP 2024, July 8-12, 2024, Tallinn, Estonia , volume 297 of LIPIcs , pages 127:1-127:20. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2024.
- [8] Michael Benedikt, Chia-Hsuan Lu, and Tony Tan. Decidability of graph neural networks via logical characterizations. CoRR , abs/2404.18151v4, 2025.
- [9] Patrick Blackburn, Maarten de Rijke, and Yde Venema. Modal Logic , volume 53 of Cambridge Tracts in Theoretical Computer Science . Cambridge University Press, 2001.
- [10] Lucas C. Cordeiro, Matthew L. Daggitt, Julien Girard-Satabin, Omri Isac, Taylor T. Johnson, Guy Katz, Ekaterina Komendantskaya, Augustin Lemesle, Edoardo Manino, Artjoms Sinkarovs, and Haoze Wu. Neural network verification is a programming language challenge. CoRR , abs/2501.05867, 2025.
- [11] David J. Tena Cucala and Bernardo Cuenca Grau. Bridging max graph neural networks and Datalog with negation. In Pierre Marquis, Magdalena Ortiz, and Maurice Pagnucco, editors, Proceedings of the 21st International Conference on Principles of Knowledge Representation and Reasoning, KR 2024, Hanoi, Vietnam. November 2-8, 2024 , 2024.

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

- [12] Stéphane Demri and Denis Lugiez. Complexity of modal logics with presburger constraints. J. Appl. Log. , 8(3):233-252, 2010.
- [13] European Parliament. Artificial Intelligence Act, 2024.
- [14] Pietro Galliani, Oliver Kutz, and Nicolas Troquard. Succinctness and complexity of ALC with counting perceptrons. In Pierre Marquis, Tran Cao Son, and Gabriele Kern-Isberner, editors, Proceedings of the 20th International Conference on Principles of Knowledge Representation and Reasoning, KR 2023, Rhodes, Greece, September 2-8, 2023 , pages 291-300, 2023.
- [15] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. Asurvey of quantization methods for efficient neural network inference. In Low-power computer vision , pages 291-326. Chapman and Hall/CRC, 2022.
- [16] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017 , volume 70 of Proceedings of Machine Learning Research , pages 1263-1272. PMLR, 2017.
- [17] Thomas A. Henzinger, Mathias Lechner, and Dorde Zikelic. Scalable verification of quantized neural networks. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, ThirtyThird Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021 , pages 3787-3795. AAAI Press, 2021.
- [18] Pei Huang, Haoze Wu, Yuting Yang, Ieva Daukantas, Min Wu, Yedi Zhang, and Clark W. Barrett. Towards efficient verification of quantized neural networks. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada , pages 21152-21160. AAAI Press, 2024.
- [19] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 2704-2713, 2018.
- [20] Viktor Kuncak and Martin Rinard. Towards efficient satisfiability checking for boolean algebra with presburger arithmetic. In Frank Pfenning, editor, Automated Deduction - CADE-21 , pages 215-230, Berlin, Heidelberg, 2007. Springer Berlin Heidelberg.
- [21] Rafael Menezes, Mohannad Aldughaim, Bruno Farias, Xianzhiyu Li, Edoardo Manino, Fedor Shmarov, Kunjian Song, Franz Brauße, Mikhail R. Gadelha, Norbert Tihanyi, Konstantin Korovin, and Lucas C. Cordeiro. ESBMC 7.4: Harnessing the Power of Intervals. In 30 th International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS'24) , volume 14572 of Lecture Notes in Computer Science , page 376-380. Springer, 2024.
- [22] Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart F. Oberman, Mohammad Shoeybi, Michael Y. Siu, and Hao Wu. FP8 formats for deep learning. CoRR , abs/2209.05433, 2022.
- [23] Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, and Tijmen Blankevoort. A white paper on neural network quantization. ArXiv , abs/2106.08295, 2021.
- [24] Pierre Nunn, Marco Sälzer, François Schwarzentruber, and Nicolas Troquard. A logic for reasoning about aggregate-combine graph neural networks. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024 , pages 3532-3540. ijcai.org, 2024.

- [25] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, 455 P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, 456 M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine 457 Learning Research , 12:2825-2830, 2011. 458

459

460

- [26] PyTorch Team. Quantization - PyTorch 2.x Documentation. https://pytorch.org/docs/ stable/quantization.html , 2024. Accessed: 2025-05-16.

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

495

496

497

498

499

500

- [27] PyTorch Team. torch.quantize\_per\_tensor - pytorch 2.x documentation. https: //pytorch.org/docs/stable/generated/torch.quantize\_per\_tensor.html# torch-quantize-per-tensor , 2024. Accessed: 2025-05-16.
- [28] PyTorch Team. torch.tensor - pytorch 2.x documentation. https://pytorch.org/docs/ stable/tensors.html#torch.Tensor , 2024. Accessed: 2025-05-16.
- [29] Patrick Reiser, Marlen Neubert, André Eberhard, Luca Torresi, Chen Zhou, Chen Shao, Houssam Metni, Clint van Hoesel, Henrik Schopmans, Timo Sommer, and Pascal Friederich. Graph neural networks for materials science and chemistry. Communications Materials , 3(93), 2022.
- [30] Amirreza Salamat, Xiao Luo, and Ali Jafari. Heterographrec: A heterogeneous graph-based neural networks for social recommendations. Knowl. Based Syst. , 217:106817, 2021.
- [31] Marco Sälzer and Martin Lange. Reachability is NP-complete even for the simplest neural networks. In Paul C. Bell, Patrick Totzke, and Igor Potapov, editors, Reachability Problems 15th International Conference, RP 2021, Liverpool, UK, October 25-27, 2021, Proceedings , volume 13035 of Lecture Notes in Computer Science , pages 149-164. Springer, 2021.
- [32] Marco Sälzer, François Schwarzentruber, and Nicolas Troquard. Verifying quantized graph neural networks is pspace-complete. CoRR , abs/2502.16244, 2025.
- [33] Luiz H. Sena, Xidan Song, Erickson H. da S. Alves, Iury Bessa, Edoardo Manino, and Lucas C. Cordeiro. Verifying Quantized Neural Networks using SMT-Based Model Checking. CoRR , abs/2106.05997, 2021.
- [34] Marco Sälzer and Martin Lange. Fundamental limits in formal verification of message-passing neural networks. In ICLR , 2023.
- [35] Shyam Anil Tailor, Javier Fernandez-Marques, and Nicholas Donald Lane. Degree-quant: Quantization-aware training for graph neural networks. In International Conference on Learning Representations , 2021.
- [36] Stephan Tobies. The complexity of reasoning with cardinality restrictions and nominals in expressive description logics. J. Artif. Intell. Res. , 12:199-217, 2000.
- [37] G. S. Tseitin. On the Complexity of Derivation in Propositional Calculus , pages 466-483. Springer Berlin Heidelberg, Berlin, Heidelberg, 1983.
- [38] Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev, and Paulius Micikevicius. Integer quantization for deep learning inference: Principles and empirical evaluation. CoRR , abs/2004.09602, 2020.
- [39] Jiacheng Xiong, Zhaoping Xiong, Kaixian Chen, Hualiang Jiang, and Mingyue Zheng. Graph neural networks for automated de novo drug design. Drug Discovery Today , 26(6):1382-1393, 2021.
- [40] Zi Ye, Yogan Jaya Kumar, Goh Ong Sing, Fengyan Song, and Junsong Wang. A comprehensive survey of graph neural networks for knowledge graphs. IEEE Access , 10:75729-75741, 2022.
- [41] Yedi Zhang, Zhe Zhao, Guangke Chen, Fu Song, Min Zhang, Taolue Chen, and Jun Sun. Qvip: An ILP-based formal verification approach for quantized neural networks. In Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering , ASE '22, New York, NY, USA, 2023. Association for Computing Machinery.

501

502

503

504

505

506

- [42] Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. Graph neural networks: A review of methods and applications. AI open , 1:57-81, 2020.
- [43] Zeyu Zhu, Fanrong Li, Zitao Mo, Qinghao Hu, Gang Li, Zejian Liu, Xiaoyao Liang, and Jian Cheng. A 2 Q : Aggregation-aware quantization for graph neural networks. In The Eleventh International Conference on Learning Representations , 2023.

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

## A Proofs of statements in the main text

Lemma 3. Let φ be a formula of modal q L . The formulas φ and mod 2 expr ( φ ) are equivalent.

Proof. We have to prove that for all G,u , we have G,u | = φ iff G,u | = mod 2 expr ( φ ) . We proceed by induction on φ .

- The base case is obvious: G,u | = φ iff G,u | = mod 2 expr ( φ ) is G,u | = φ iff G,u | = mod 2 expr ( φ ) .
- G,u | = ¬ φ iff G,u ̸| =

iff (by induction) G,u ̸| = mod 2 expr ( φ

- φ )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

iff G,u | = ϑ ≤ -1 (because we suppose that ϑ takes its value in the integers

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- G,u | = ( φ 1 ∨ φ 2 )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

( ⇐ ) Conversely, by contrapositive, if G,u | = ( ϑ 2 &lt; 0) and G,u | = ( ϑ 1 &lt; 0) , then G,u | = ϑ 1 + ReLU ( ϑ 2 -ϑ 1 ) = ϑ 1 + ϑ 2 -ϑ 1 = ϑ 2 &lt; 0 or G,u | = ϑ 1 + ReLU ( ϑ 2 -ϑ 1 ) = ϑ 1 +0 = ϑ 1 &lt; 0 . In the two cases, G,u | = ϑ 1 + ReLU ( ϑ 2 -ϑ 1 ) &lt; 0 .

- G,u | = ♢ ≥ k φ iff the number of vertices v that are successors of u and with G,v | = φ is greater than k

iff the number of vertices v that are successors of u and with G,v | = mod 2 expr ( φ ) is greater than k iff (written ϑ ≥ 0 ) iff the number of vertices v that are successors of u and with G,v | = ϑ ≥ 0 is greater than k

iff the number of vertices v that are successors of u and with G,v | = ReLU ( ϑ + 1) -ReLU ( ϑ ) = 1 is greater than k (since we know by defining of modal q L that ϑ takes its value in integers)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Other cases are similar.

Proposition 7. The number of Hintikka sets is bounded by 2 n | φ | where | φ | is the size of φ , and n is the bitwidth of /u1D542 .

Proof. For each expression ϑ , we choose a number in /u1D542 . There is 2 n different numbers. There are | φ | number of expressions. So we get (2 n ) | φ | = 2 n | φ | possible choices for a Hintikka set.

Proposition 8. If bitwidth n is in unary, and if /u1D542 saturates, then satisfiability in QFBAPA /u1D542 is in NP .

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

Proof. Here is a non-deterministic algorithm for the satisfiability problem in QFBAPA /u1D542 .

1. Let χ be a QFBAPA /u1D542 formula.
2. For each set expression B appearing in some | B | , guess a non-negative integer number k B in /u1D542 .
3. Let χ ′ be a (grounded) formula in which we replaced | B | by k B .
4. Check that χ ′ is true (can be done in poly-time since χ ′ is a grounded formula, it is a Boolean formula on variable-free equations and inequations in /u1D542 ).
5. If not we reject.
6. We now build a standard QFBAPA formula δ = ∧ B constraint ( B ) where:

<!-- formula-not-decoded -->

where limit is the maximum number that is considered as infinity in /u1D542 .

7. Run a non-deterministic poly-time algorithm for the QFBAPA satisfiability on δ . Accepts if it accepts. Otherwise reject.

The algorithm runs in poly-time. Guessing a number n B is in poly-time since it consists in guessing n bits ( n in unary). Step 4 is just doing the computations in /u1D542 . In Step 6, δ can be computed in poly-time.

If χ is QFBAPA /u1D542 satisfiable, then there is a solution σ such that σ | = χ . At step 2, we guess n B = | σ ( B ) | /u1D542 . The algorithm accepts the input.

Conversely, if the algorithm accepts its input, χ ′ is true for the chosen values n B . δ is satisfiable. So there is a solution σ such that σ | = δ . By the definition of constraint , σ | = χ .

Remark 15. If the number n of bits to represent /u1D542 is given in unary and if /u1D542 is "modulo", then the satisfiability problem in QFBAPA /u1D542 is also in NP. The proof is similar except than now constraint ( B ) = ( | B | = k B + Ld B ) where d B is a new variable.

Proposition 9. tr ( φ ) is computable in exponential-time in | φ | and n .

̸

Proof. In order to create tr ( φ ) , we write an algorithm where each big conjunction, big disjunction, big union and big sum is replaced by a loop. For instance, ∧ H = H ′ is replaced by two inner loops over Hintikka sets. Note that we create check whether a candidate H is a Hintikka set in exponential time in n since Point 4 can be checked in exponential time in n (thanks to our loose assumption on the computability of [[ α ]] in exponential time in n . There are 2 n | φ | many of them. In the same way, ∧ k ∈ /u1D542 is a loop over 2 n values. There is a constant number of nested loops, each of them iterating over an exponential number (in n and | φ | of elements. QED.

Proposition 10. Let φ be a formula of q L . φ is satisfiable iff tr ( φ ) is QFBAPA /u1D542 satisfiable.

Proof. ⇒ Let G,u such that G,u | = φ . We set σ ( X ϑ ′ = k ) := { v | [[ ϑ ′ ]] G,v = k } and σ ( X H ) = { v | G,v | = H } where G,u | = H means that for all ϑ ′ = k ∈ H , we have [[ ϑ ′ ]] G,v = k . For all Hintikka sets H such that there is v such that G,v | = H , we set: σ ( S H ) := { w | vEw } .

We check that σ | = tr ( φ ) . First, σ satisfies Formulas 1 and 2 by definition of σ . Now, σ also satisfies Formula 3. Indeed, if agg ( ϑ ′ ) = k ∈ H , then if there is no H -vertex in G then the implication is true. Otherwise, consider the H -vertex v . But, then by definition of X agg ( ϑ ′ )= k , [[ agg ( ϑ ′ )]] G,v = k . But then the semantics of agg exactly corresponds to ∑ k ′ ∈ /u1D542 | S H ∩ X ϑ = k ′ | × k ′ = k . Indeed, each S H ∩ X ϑ = k ′ -successor contributes with k ′ . Thus, the contribution of successors where ϑ is k ′ is | S H ∩ X ϑ = k ′ | × k ′ .

Formula 4 is also satisfied by σ . Actually, let k such that σ | = X agg ∀ ( ϑ )= k = U . This means that the value of agg ∀ ( ϑ ) (which does not depend on a specific vertex u but only on G ) is k . The sum ∑ k ′ ∈ /u1D542 | X ϑ = k ′ | × k ′ = k is the semantics of agg ∀ ( ϑ ) = k .

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

Finally, as G,u | = φ , and φ is of the form ϑ ≥ k , there is k ′ ≥ k such that [[ ϑ ]] G,u = k ′ . So 590 X ϑ = k ′ = ∅ . 591

̸

⇐ Conversely, consider a solution σ of tr ( φ ) . We construct a graph G = ( V, E ) as follows. 592

```
V := σ ( U ) E := { ( u, v ) | for some H , u ∈ σ ( X H ) and v ∈ σ ( S H ) } ℓ ( v ) i := k where v ∈ X x i = k
```

i.e. the set of vertices is the universe, and we add an edge between any H -vertex u and a vertex v ∈ σ ( S H ) , and the labeling for features is directly given X x i = k . Note that the labeling is welldefined because of formulas 1 and 2.

As σ | = | X φ | ≥ 1 , there exists u ∈ σ ( X φ ) . Let us prove that G,u | = φ . By induction on ϑ ′ , we prove that u ∈ X ϑ ′ = k implies [[ ϑ ′ ]] G,u = k . The base case is obtained via the definition of ℓ . Cases for +, × and α are obtained because each vertices is in some σ ( X H ) for some H . As the definition of Hintikka set takes care of the semantics of +, × and α , we have [[ ϑ 1 + ϑ 2 ]] G,u = [[ ϑ 1 ]] G,u +[[ ϑ 2 ]] G,u , etc.

[[ agg ( ϑ )]] G,u = Σ v | uEv [[ ϑ ]] G,v and [[ agg ∀ ( ϑ )]] G,u = Σ v ∈ V [[ ϑ ]] G,v hold because of σ satisfies respectively formula 3 and 4.

Theorem 12. The satisfiability problem in q L is NEXPTIME-hard, and so is VT3 . VT1 and VT2 are coNEXPTIME-hard.

Proof. We reduce the NEXPTIME-hard problem of deciding whether a domino system D = ( D,V,H ) , given an initial condition w 0 . . . w n -1 ∈ D n , can tile an exponential torus [36]. In the domino system, D is the set of tile types, and V and H respectively are the respectively vertical and horizontal color compatibility relations. We are going to write a set of modal q L formulas that characterize the torus ℤ 2 n +1 × ℤ 2 n +1 and the domino system. We use 2 n +2 features. We use x 0 , . . . x n -1 , and x ′ 0 , . . . , x ′ n -1 , to hold the (binary-encoded) coordinates of vertices in the torus. We use the feature x N to denote a vertex 'on the way north' (when x N = 1 ) and x E to denote a vertex 'on the way east' (when x E = 1 ), with abbreviations φ N := x N = 1 , and φ E := x E = 1 . See Figure 2.

For every n ∈ ℕ , we define the following set of formulas. T n = 614

```
{ □ g ( x N = 1 ∨ x N = 0) , □ g ( x E = 1 ∨ x E = 0) , □ g ( ∧ n -1 k =0 ( x i = 1 ∨ x i = 0)) , □ g ( ∧ n -1 k =0 ( x ′ i = 1 ∨ x ′ i = 0)) , □ g ( ¬ ( x N = 1 ∧ x E = 1)) , □ g ( ¬ ( φ N ∨ φ E ) → agg (1) = 2) , □ g ( ¬ ( φ N ∨ φ E ) → ( agg ( x N ) = 1)) , □ g ( ¬ ( φ N ∨ φ E ) → ( agg ( x E ) = 1)) , □ g ( φ N → agg (1) = 1) , □ g ( φ E = 1 → agg (1) = 1) , ♢ =1 g φ (0 , 0) , ♢ =1 g φ (2 n -1 , 2 n -1) , □ g ( ¬ ( φ N ∨ φ E ) → φ east ) , □ g ( ¬ ( φ N ∨ φ E ) → φ north ) , ♢ ≤ 2 n × 2 n g ¬ ( φ N ∨ φ E ) , ♢ ≤ 2 n × 2 n g φ N , ♢ ≤ 2 n × 2 n g φ E }
```

where φ (0 , 0) := ∧ n -1 k =0 x i = 0 ∧ ∧ n -1 k =0 x ′ i = 0 , and φ (2 n -1 , 2 n -1) := ∧ n -1 k =0 x i = 1 ∧ ∧ n -1 k =0 x ′ i = 1 615 represent two nodes, namely those at coordinates (0 , 0) and (2 n -1 , 2 n -1) . The formulas φ north and 616 φ east enforce constraints on the coordinates of states, such that going north increases the coordinate 617 encoding using the x i features by one, leaving the x ′ i features unchanged, and going east increases 618 coordinate encoding using the x ′ i features by one, leaving the x i features unchanged. For every 619

formula φ , ∀ east.φ stands for □ ( φ E → □ φ ) and ∀ north.φ stands for □ ( φ N → □ φ ) . 620

<!-- formula-not-decoded -->

The problem of deciding whether a domino system D = ( D,V,H ) , given an initial condition 621 w 0 . . . w n -1 ∈ D n , can tile a torus of exponential size can be reduced to the problem satisfiability in 622 q L , checking the satisfiability of the set of formulas T ( n, D , w ) = T n ∪ T D ∪ T w , where T n is as 623 above, T D encodes the domino system, and T w encodes the initial condition as follows. We define 624

<!-- formula-not-decoded -->

625

626

627

628

629

630

631

632

633

634

635

636

637

638

639

640

641

642

643

Theorem 14. The satisfiability problems with bounded number of vertices are NP-complete. 644

where for every d ∈ D , there is a feature x d and φ d := x d = 1 . Finally, we define

<!-- formula-not-decoded -->

The size of T ( n, D , w ) is polynomial in the size of the tiling problem instance, that is in | D | + | H | + | V | + n . The rest of the proof is analogous to the proof of [36, Corollary 3.9]. The NEXPTIMEhardness of q L follows from Lemma 3 and [36, Corollary 3.3] stating the NEXPTIME-hardness of deciding whether a domino system with initial condition can tile a torus of exponential size.

For the complexity of ACR-GNN verification tasks, we observe the following.

̸

1. We reduce the satisfiability problem in (modal) q L (restricted to graded modal logic + graded universal modality, because it is sufficient to encode the tiling problem) to VT3 in poly-time as follows. Let φ be a q L . We build in poly-time an ACR-GNN A that recognizes all pointed graphs. We have φ is satisfiable iff [[ φ ]] ∩ [[ A ]] = ∅ So VT3 is NEXPTIME-hard.
2. The validity problem of q L (dual problem of the satisfiability problem, i.e., given a formula φ , is φ true in all pointed graphs G,u ?) is coNEXPTIME-hard. We reduce the validity problem of q L to VT2. Let φ be a q L formula. We construct an ACR-GNN A that accepts all pointed graphs. We have φ is valid iff [[ A ]] ⊆ [[ φ ]] . So VT2 is coNEXPTIME-hard.
3. We reduce the validity problem of q L to VT1. Let ψ be a q L formula. (again in graded modal logic + graded global modalities). So by [4], We construct in poly-time an ACRGNN A that is equivalent to ψ (by [4]). We have ψ is valid iff [[ ⊤ ]] ⊆ [[ A ]] . So VT1 is coNEXPTIME-hard.

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

678

Proof. NP upper bound is obtained by guessing a graph with at most N vertices and then check that φ holds. The obtained algorithm is non-deterministic, runs in poly-time and decides the satisfiability problem with bounded number of vertices. NP-hardness already holds for agg -free formulas by reduction from SAT for propositional logic (the reduction is mod 2 expr , see Lemma 3).

## B Checking distributivity

We provide C source code for checking distributivity. The reader may run the model checker ESBMC on it to see whether distributivity holds or not.

## C Extension of logic K ♯ and ACR-GNNs over ℤ

A (labeled directed) graph G is a tuple ( V, E, ℓ ) such that V is a finite set of vertices, E ⊆ V × V a set of directed edges and ℓ is a mapping from V to a valuation over a set of atomic propositions. We write ℓ ( u )( p ) = 1 when atomic proposition p is true in u , and ℓ ( u )( p ) = 0 otherwise. Given a graph G and vertex u ∈ V , we call ( G,u ) a pointed graph .

## C.1 Logic

Consider a countable set Ap of propositions. We define the language of logic K ♯,♯ g as the set of formulas generated by the following BNF:

<!-- formula-not-decoded -->

where p ranges over Ap , and c ranges over ℤ . We assume that all formulas φ are represented as directed acyclic graph (DAG) and refer by the size of φ to the size of its DAG representation.

Atomic formulas are propositions p , inequalities and equalities of linear expressions. We consider linear expressions over /u1D7D9 φ and ♯φ and ♯ g φ . The number /u1D7D9 φ is equal to 1 if φ holds in the current world and equal 0 otherwise. The number ♯φ is the number of successors in which φ hold. The number ♯ g φ is the number of worlds in the model in which φ hold. The language seems strict but we write ξ 1 ≤ ξ 2 for ξ 2 -ξ 1 ≥ 0 , ξ = 0 for ( ξ ≥ 0) ∧ ( -ξ ≥ 0) , etc.

As in modal logic, a formula φ is evaluated in a pointed graph ( G,u ) (also known as pointed Kripke model). We define the truth conditions ( G,u ) | = φ ( φ is true in u ) by

<!-- formula-not-decoded -->

and the semantics [[ ξ ]] G,u (the value of ξ in u ) of an expression ξ by mutual induction on φ and ξ as follows.

<!-- formula-not-decoded -->

A local modality □ φ can be defined as □ φ := ( -1) × ♯ ( ¬ φ ) ≥ 0 . That is, to say that φ holds in all successors, we say that the number of successors in which ¬ φ holds is zero. Similarly, a global/universal modality can be defined as □ g φ := ( -1) × ♯ g ( ¬ φ ) ≥ 0 .

## C.2 Aggregate-Combine Graph Neural Networks

In this section, we consider a detailed definition of quantized (global) Aggregate-Combine GNNs (ACR-GNN) [4], also called message passing neural networks [16]. We stick to the former term.

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

A (global) ACR-GNN layer L = ( comb , agg , agg g ) is a tuple where comb : ℝ 2 m → ℝ n is a so-called combination function , agg is a so-called local aggregation function , mapping multisets of vectors from ℝ m to a single vector from ℝ n , agg g is a so-called global aggregation function , also mapping multisets of vectors from ℝ m to a single vector from ℝ n . We call m the input dimension of layer L and n the output dimension of layer L . Then, a (global) ACR-GNN is a tuple ( L (1) , . . . , L ( L ) , cls ) where L (1) , . . . , L ( L ) are L ACR-GNN layers and cls : ℝ m →{ 0 , 1 } is a classification function . We assume that all GNNs are well-formed in the sense that output dimension of layer L ( i ) matches input dimension of layer L ( i +1) as well as output dimension of L ( L ) matches input dimension of cls .

Let G = ( V, E ) be a graph with atomic propositions p 1 , . . . , p k and A = ( L (1) , . . . , L ( L ) , cls ) an ACR-GNN. We define x 0 : V → { 0 , 1 } k , called the initial state of G , as x 0 ( u ) := ( ℓ ( u )( p 1 ) , . . . , ℓ ( u )( p k )) for all u ∈ V . Then, the i -th layer of A computes an updated state of G by

<!-- formula-not-decoded -->

where agg , agg g , and comb are respectively the local aggregation, global aggregation and combination function of the i -th layer. Let ( G,u ) be a pointed graph. We write A ( G,u ) to denote the application of A to ( G,u ) , which is formally defined as A ( G,u ) = cls ( x L ( u )) where x L is the state of G computed by A after layer L . Informally, this corresponds to a binary classification of node u .

In this work, we exclusively consider the following form of ACR-GNN A : all local and global aggregation functions are given by the sum of all vectors in the input multiset, all combination functions are given by comb ( x, y, z ) = ⃗ σ ( xC + yA 1 + zA 2 + b ) where ⃗ σ ( x ) is the componentwise application of the truncated ReLU σ ( x ) = max (0 , min (1 , x )) , with matrices C , A 1 and A 2 and vector b of /u1D542 parameters, and where the classification function is cls ( x ) = ∑ i a i x i ≥ 1 , where a i are from /u1D542 as well.

We note [[ A ]] the set of pointed graphs ( G,u ) such that A ( G,u ) = 1 . An ACR-GNN A is satisfiable if [[ A ]] is non-empty. The satisfiability problem for ACR-GNNs is: Given a ACR-GNN A , decide whether A is satisfiable.

## D Capturing GNNs with K ♯,♯ g

In this section, we demonstrate that the expressive power of (global) ACR-GNNs, as defined in Section C.2 and K ♯,♯ g , is equivalent. Informally, this means that for every formula φ of K ♯,♯ g , there exists an ACR-GNNs A that expresses the same query, and vice-versa. To achieve this, we define a translation of one into the other and substantiate that this translation is efficient. This enables ways to employ K ♯,♯ g for reasoning about ACR-GNN.

We begin by showing that global ACR-GNNs are at least as expressive as K ♯,♯ g . We remark that the arguments are similar to the proof of Theorem 1 in [24].

Theorem 16. Let φ ∈ K ♯,♯ g be a formula. There is A φ such that for all pointed graphs ( G,u ) we have ( G,u ) | = φ if and only if A φ ( G,u ) = 1 . Furthermore, A φ can be built in polynomial time regarding the size of φ .

Proof sketch. We construct a GNN A φ that evaluates the semantics of a given K ♯,♯ g formula φ for some given pointed graph ( G,v ) . The network consists of n layers, one for each of the n subformulas φ i of φ , ordered so that the subformulas are evaluated based on subformula inclusion. The first layer evaluates atomic propositions, and each subsequent messages passing layer l i uses a fixed combination and fixed aggregation function to evaluate the semantics of φ i .

The correctness follows by induction on the layers: the i -th layer correctly evaluates φ i at each vertex of G , assuming all its subformulas are correctly evaluated in previous layers. Finally, the classifying function cls checks whether the n -th dimension of the vector after layer l n , corresponding to the semantics of φ n for the respective vertex v , indicates that φ n = φ is satisfied by ( G,v ) . The network size is polynomial in the size of φ due to the fact that the total number of layers and their width is polynomially bounded by the number of subformulas of φ . A full formal proof is given in Appendix F.

723

724

725

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

737

738

739

740

741

742

743

744

745

746

747

748

749

750

751

752

753

754

755

756

757

758

759

760

761

762

763

764

Theorem 17. Let A be a GNN. We can compute in polynomial time wrt. |A| a K ♯,♯ g -formula φ A , represented as a DAG, such that [[ A ]] = [[ φ A ]] .

Proof sketch. We construct a K ♯,♯ g -formula φ A that simulates the computation of a given GNN A . For each layer l i of the GNN, we define a set of formulas φ i,j , one per output dimension, that encode the corresponding node features using linear threshold expressions over the formulas from the previous layer. At the base, the input features are the atomic propositions p 1 , . . . , p m 1 .

Each formula φ i,j mirrors the computation of the GNN layer, including combination, local aggregation, and global aggregation. The final classification formula φ A encodes the output of the linear classifier on the top layer features. Correctness follows from the fact that all intermediate node features remain Boolean under message passing layers with integer parameters and truncated ReLU activations. This allows expressing each output as a Boolean formula over the input propositions. The construction is efficient: by reusing shared subformulas via a DAG representation, the total size remains polynomial in the size of A .

## E Complexity of the satisfiability of K ♯,♯ g and its implications for ACR-GNN verification

In this section, we establish the complexity of reasoning with K ♯,♯ g .

Instrumentally, we first show that every K ♯,♯ g formula can be translated into a K ♯,♯ g formula that is equi-satisfiable, and has a tree representation of size at most polynomial in the size of the original formula. An analogous result was obtained in [24] for K ♯ . It can be shown using a technique reminiscent of [37] and consisting of factorizing subformulas that are reused in the DAG by introducing a fresh proposition that is made equivalent. Instead of reusing a 'possibly large' subformula, a formula then reuses the equivalent 'small' atomic proposition.

Lemma 18. The satisfiability problem of K ♯,♯ g reduces to the satisfiability of K ♯,♯ g with tree formulas in polynomial time.

Proof. Let φ be a K ♯,♯ g formula represented as a DAG. For every subformula ψ (i.e., for every node in the DAG representation of φ ), we introduce a fresh atomic proposition p ψ . We can capture the meaning of these new atomic propositions with the formula Φ := ∧ ψ node in the DAG sem ( ψ ) where:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, define φ t := p φ ∧ □ g Φ , where □ g Φ := ( -1) × ♯ g ( ¬ Φ) ≥ 0 , enforcing the truth of Φ in every vertex. The size of its tree representation is polynomial in the size of φ . Moreover, φ t is satisfiable iff φ is satisfiable.

Theorem 19. K ♯,♯ g tree -satisfiability problem is NEXPTIME-complete.

Proof. For membership, we translate the problem into the NEXPTIME-complete problem of concept description satisfiability in the Description Logics with Global and Local Cardinality Constraints [2], noted ALCSCC ++ . The Description Logic ALCSCC ++ uses the Boolean Algebra with Presburger Arithmetic [20], noted QFBAPA, to formalize cardinality constraints. See Section H for a presentation of ALCSCC ++ and QFBAPA.

Let φ 0 be a K ♯,♯ g formula.

For every proposition p occurring in φ 0 , let A p be an ALCSCC ++ concept name. Let R be an ALCSCC ++ role name. For every occurrence of /u1D7D9 φ in φ 0 , let ZOO φ be an ALCSCC ++ role name. ZOO -roles stand for 'zero or one'. The rationale for introducing ZOO -roles is to be able to capture

the value of /u1D7D9 φ in ALCSCC ++ making it equal to the number of successors of the role ZOO φ which 765 can then be used in QFBAPA constraints. A similar trick was used, in another context, in [14]. Here, 766 we enforce this with the QFBAPA constraint 767

<!-- formula-not-decoded -->

which states that ZOO φ has zero or one successor, and has one successor exactly when (the translation 768 of) φ is true. The concept descriptions τ ( φ ) and arithmetic expressions τ ( ξ ) are defined inductively 769 as follows: 770

<!-- formula-not-decoded -->

Finally, we define the ALCSCC ++ concept description C φ 0 = τ ( φ 0 ) ⊓ sat ( χ 0 ) .

Claim 20. The concept description C φ 0 is ALCSCC ++ -satisfiable iff the formula φ 0 is K ♯,♯ g -satisfiable. Moreover, the concept description C φ 0 has size polynomial in the size of φ 0 .

Proof. From right to left, suppose that φ 0 is K ♯,♯ g -satisfiable. It means that there is a pointed graph ( G,u ) where G = ( V, E ) and u ∈ V , such that ( G,u ) | = φ 0 . Let I 0 = (∆ I 0 , · I 0 ) be the ALCSCC ++ interpretation over N C and N R , such that N C = { A p | p a proposition in φ 0 } , N R = { R } ∪ { ZOO φ | /u1D7D9 φ ∈ φ 0 } , ∆ I 0 = V , A I 0 p = { v | v ∈ V, ( G,v ) | = p } for every p in φ 0 , R I 0 = E , ZOO I 0 φ = { ( v, v ) | v ∈ V, ( G,v ) | = φ } for every /u1D7D9 φ in φ 0 . We can show that u ∈ C I 0 φ 0 . Basically I 0 is like G with the addition of adequately looping ZOO -roles. An individual in ∆ I 0 has exactly one ZOO φ -successor (itself), exactly when φ is true, and no successor otherwise; A p is true exactly where p is true, and the role R corresponds exactly to E .

From left to right, suppose that C φ 0 is ALCSCC ++ -satisfiable. It means that there is an ALCSCC ++ finite interpretation I 0 = (∆ I 0 , · I 0 ) and an individual d ∈ ∆ I 0 such that d ∈ C I 0 φ 0 . Let G = ( V, E ) be a graph such that V = ∆ I 0 , E = R I 0 , and ℓ ( d )( p ) = 1 iff d ∈ A I 0 p . We can show that ( G,d ) | = φ 0 .

Since there are at most | φ 0 | subformulas in φ 0 , the representation of ZOO φ for every subformula φ of φ 0 can be done in size log 2 ( | φ 0 | ) . For every formula φ , the size of the concept description τ ( φ ) is polynomial (at most O ( n log( n )) ). The overall size of τ ( φ 0 ) is polynomial in the size of φ 0 , and so is the size of sat ( ξ 0 ) (at most O ( n 2 (log( n )) 2 ).

The NEXPTIME-membership follows from Claim 20 and the fact that the concept satisfiability problem in ALCSCC ++ is in NEXPTIME (Theorem 25).

For the hardness, we reduce the problem of consistency of ALCQ -T C Boxes which is NEXPTIMEhard [36, Corollary 3.9]. See Section I and Theorem 27 that slightly adapts Tobies' proof to show that the problem is hard even with only one role.

We define the translation τ from the set of ALCQ concept expressions and ALCQ cardinality constraints, with only one role R .

```
τ ( A ) = p A τ ( ¬ C ) = ¬ τ ( C ) τ ( C 1 ⊔ C 2 ) = τ ( C 1 ) ∨ τ ( C 2 ) τ ( ≥ n R.C ) = ♯τ ( C ) + ( -1) × n ≥ 0 τ ( ≥ n C ) = ♯ g τ ( C ) + ( -1) × n ≥ 0 τ ( ≤ n C ) = ( -1) × ♯ g τ ( C ) + n ≥ 0
```

771

772

773

774

775

776

777

778

779

780

781

782

783

784

785

786

787

788

789

790

791

792

793

794

795

It is routine to check the following claim. 796

797

Claim 21. Let TC be an ALCQ -T C Box. TC is consistent iff ∧ χ ∈ TC τ ( χ ) is K ♯,♯ g -satisfiable.

798

799

800

801

802

803

804

805

806

807

808

809

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

Moreover, the reduction is linear. Hardness thus follows from the NEXPTIME-hardness of consistency of ALCQ -T C Boxes.

Lemma 18 and Theorem 19 yield the following corollary.

Corollary 22. K ♯,♯ g -satisfiability problem is NEXPTIME-complete.

Furthermore, from Theorem 16 and Corollary 22, we obtain the complexity of reasoning with ACR-GNNs with truncated ReLU and integer weights.

Corollary 23. Satisfiability of ACR-GNN with global readout and truncated ReLU is NEXPTIMEcomplete.

The decidability of the problem is left open in [7] and in the recent long version [8] when the weights are rational numbers. The theorem answers it positively in the case of integer weights and pinpoints the computational complexity.

## F Formal proofs

Proof of Theorem 16. Let φ be a K ♯,♯ g formula over the set of atomic propositions p 1 , . . . , p m . Let φ 1 , . . . , φ n denote an enumeration of the subformulas of φ such that φ i = p i for i ≤ m , φ n = φ , and whenever φ i is a subformula of φ j , it holds that i ≤ j . Without loss of generality, we assume that all subformulas of the form ξ ≥ 0 are written as

<!-- formula-not-decoded -->

for some index sets J, J ′ , J ′′ ⊆ { 1 , . . . , n } .

We construct the GNN A φ in a layered manner. Note that A φ is fully specified by defining the combination function comb i , including its local and global aggregation, for each layer l i with i ∈ { 1 , . . . , n } and the final classification function cls . Each comb i produces output vectors of dimension n . The first layer comb 1 has input dimension 2 m and is defined by comb 1 ( x, y, z ) = ( x, 0 , . . . , 0) , ensuring that the first m dimensions correspond to the truth values of the atomic propositions p 1 , . . . , p m , while the remaining entries are initialized to zero. Note that comb 1 is easily realized by an FNN with ReLU activations. For i &gt; 1 , the combination function comb i is defined as

<!-- formula-not-decoded -->

where C , A 1 , A 2 are n × n matrices corresponding to self, local (neighbor), and global aggregation respectively, and b ∈ ℝ n is a bias vector. The parameters are defined sparsely as follows:

- C ii = 1 for all i ≤ m (preserving the atomic propositions),

<!-- formula-not-decoded -->

- If φ i = φ j ∨ φ l , then C ji = C li = 1 , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that each comb i has the same functional form, differing only in the non-zero entries of its parameters. The classification function is defined by cls ( x ) = x n ≥ 1 .

Let l i denote the i th layer of A φ , and fix a vertex v in some input graph. We show, by induction on i , that the following invariant holds: for all j ≤ i , ( x i ( v )) j = 1 if and only if v | = φ j , and ( x i ( v )) j = 0 otherwise. Assume that i = 1 . By construction, x 1 ( v ) contains the truth values of the atomic propositions p 1 , . . . , p m in its first m coordinates. Thus, the statement holds at layer 1 .

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

867

Next, assume the statement holds for layer x i -1 . Let j &lt; i . By assumption, the semantics of φ j are already correctly encoded in x j -1 and preserved by comb i due to the fixed structure of C , A 1 , A 2 , and b . Now consider j = i . The semantics of all subformulas of φ i are captured in x i -1 , either at the current vertex or its neighbors. By the design of comb i , which depends only on the values of relevant subformulas, we conclude that φ i is correctly evaluated. This holds regardless of whether φ i is a negation, disjunction, or numeric threshold formula. Thus, the statement holds for all i , and in particular for x n ( v ) and φ n = φ . Finally, the classifier cls evaluates whether x n ( v ) n ≥ 1 , which is equivalent to G,v | = φ . The size claim is obvious given that n depends polynomial on the size of φ . We note that this assumes that the enumeration of subformulas of φ does not contain duplicates.

Proof of Theorem 17. Let A be a GNN composed of layers l 1 , . . . , l k , where each comb i has input dimension 2 m i , output dimension n i , and parameters C i , A i, 1 , A i, 2 , and b i . The final classification is defined via a linear threshold function cls ( x ) = a 1 x 1 + · · · + a n k x n k ≥ 1 . We assume that the dimensionalities match across layers, i.e. m i = n i -1 for all i ≥ 2 , so that the GNN is well-formed.

Weconstruct a formula φ A over the input propositions p 1 , . . . , p m 1 inductively, mirroring the structure of the GNN computation.

We begin with the first layer l 1 . For each j ∈ { 1 , . . . , n 1 } , we define:

<!-- formula-not-decoded -->

Now suppose that we have already constructed formulas φ i -1 , 1 , . . . , φ i -1 ,n i -1 for some layer i ≥ 2 . Then, for each output index j ∈ { 1 , . . . , n i } , we define:

<!-- formula-not-decoded -->

Once all layers have been encoded in this way, we define the final classification formula as

<!-- formula-not-decoded -->

Let G,v be a pointed graph. The correctness of our translation follows directly from the following observations: all weights and biases in A are integers, and the input vectors x 0 ( u ) assigned to nodes u in G are Boolean. Moreover, each layer applies a linear transformation followed by a pointwise truncated ReLU, which preserves the Boolean nature of the node features. It follows that the intermediate representations x i ( v ) remain in { 0 , 1 } n i for all i . Consequently, each such feature vector can be expressed via a set of Boolean K ♯,♯ g -formulas as constructed above. Taken together, this ensures that the overall formula φ A faithfully simulates the GNN's computation.

It remains to argue that this construction can be carried out efficiently. Throughout, we represent the (sub)formulas using a shared DAG structure, avoiding duplication of equivalent subterms. This ensures that subformulas φ i -1 ,k can be reused without recomputation. For each layer, constructing all φ i,j requires at most n i · m i steps, plus the same order of additional operations to account for global aggregation terms. Since the number of layers, dimensions, and parameters are bounded by |A| , and each operation can be performed in constant or linear time, the total construction is polynomial in the size of A .

## G Experimental data and further analyses

This study investigates the application of dynamic Post-Training Quantization (PTQ) to Aggregate868 Combined Readout Graph Neural Networks (ACR-GNNs). Implemented in PyTorch [1, 26], dynamic 869 PTQ transforms a pre-trained floating-point model into a quantized version without requiring retrain870 ing. In this approach, model weights are statically quantized to INT8, while activations remain in 871 floating-point format until they are dynamically quantized at compute time. This hybrid representation 872 enables efficient low-precision computation using INT8-based matrix operations, thereby reducing 873 memory footprint and improving inference speed. PyTorch's implementation applies per-tensor 874 quantization to weights and stores activations as floating-point values between operations to balance 875 precision and performance. 876

We adopt INT8 and QINT8 representations as the primary quantization format. According to theory, 877 INT8 refers to 8-bit signed integers that can encode values in the range [ -128 , 127] . In contrast, 878 QINT8, as defined in the PyTorch documentation [1, 27, 28], is a quantized tensor format that wraps 879 INT8 values together with quantization metadata: a scale (defining the float value represented by one 880 integer step) and a zero-point (the INT8 value corresponding to a floating-point zero). This additional 881 information allows QINT8 tensors to approximate floating-point representations efficiently while 882 enabling high-throughput inference. 883

884

885

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

To evaluate the practical impact of quantization, we conducted experiments on both synthetic and real datasets. The synthetic data setup was based on the benchmark introduced by [4]. Graphs were generated using the dense Erd ¨ os-R ´ enyi model, a classical method for constructing random graphs, and each graph was initialized with five node colours encoded as one-hot feature vectors. The dataset is structured as follows, as shown in Table 3. The training set consists of 5000 graphs, each with 40 to 50 nodes and between 560 and 700 edges. The test set is divided into two subsets. The first subset comprises 500 graphs with the same structure as the training set, featuring 40 to 50 nodes and 560 to 700 edges. The second subset contains 500 larger graphs, with 51 to 69 nodes and between 714 and 960 edges. This design allows us to evaluate the model's generalization capability to unseen graph sizes.

Table 3: Dataset statistics summary.

|            |         | Node   | Node   | Node   | Edge   | Edge   | Edge   |
|------------|---------|--------|--------|--------|--------|--------|--------|
| Classifier | Dataset | Min    | Max    | Avg    | Min    | Max    | Avg    |
| p 1        | Train   | 40     | 50     | 45     | 560    | 700    | 630    |
| p 1        | Test1   | 40     | 50     | 45     | 560    | 700    | 633    |
| p 1        | Test2   | 51     | 60     | 55     | 714    | 960    | 832    |
| p 2        | Train   | 40     | 50     | 45     | 560    | 700    | 630    |
| p 2        | Test1   | 40     | 50     | 44     | 560    | 700    | 628    |
| p 2        | Test2   | 51     | 60     | 55     | 714    | 960    | 832    |
| p 2        | Train   | 40     | 50     | 44     | 560    | 700    | 629    |
| p 2        | Test1   | 40     | 50     | 45     | 560    | 700    | 630    |
| p 2        | Test2   | 51     | 60     | 55     | 714    | 960    | 831    |

For this experiment, we used simple ACR-GNN models with the following specifications. We applied the sum function for both the aggregation and readout operations. The combination function was defined as: comb ( x, y, z ) = ⃗ σ ( xC + yA + zR + b ) , where ⃗ σ denotes the activation function. Following the original work, we set the hidden dimension to 64, used a batch size of 128, and trained the model for 20 epochs using the Adam optimizer with default PyTorch parameters. We used two activation functions for the experimental part, ReLU and truncated ReLU. For implementation, we used PyTorch [1]: nn.ReLU and nn.Hardtanh(0, 1) in accordance.

We trained ACR-GNN on complex formulas FOC 2 for labeling. They are presented as a classifier α i ( x ) that constructed as:

<!-- formula-not-decoded -->

where ∃ [ N,M ] stands for 'there exist between N and M nodes'. satisfying a given property.

Observe that each

α

i

(

x

)

is in FOC

2

, as

∃

[

N,M

]

can be expressed by combining

∃

≥

N

and

¬∃

≥

M

+1

.

The data set has the following specifications: Erd ¨ os-R ´ enyigraphs and is labeled according to α 1 ( x ) , α 2 ( x ) , and α 3 ( x ) :

- α 0 ( x ) := Blue ( x )
- p 1 : α 1 ( x ) := ∃ [8 , 10] y ( α 0 ( y ) ∧ ¬ E ( x, y ))
- p 2 : α 2 ( x ) := ∃ [10 , 30] y ( α 1 ( y ) ∧ ¬ E ( x, y ))
- p 3 : α 3 ( x ) := ∃ [10 , 30] y ( α 2 ( y ) ∧ ¬ E ( x, y ))

In this section, we present experiments for two activation functions: ReLU and truncated ReLU 911 (implemented via nn.Hardtanh(0,1) ) to study the influence of the activation function on the model. 912

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

925

926

927

928

929

930

Experiments for the ACR-GNN were conducted with different numbers of hidden layers, ranging from 1 to 10. To measure the precision of the results, we use the strategy as [4]: accuracy is calculated as the total number of correctly classified nodes among all nodes in all graphs in the dataset.

Table 4: Accuracy of the ACR-GNN with ReLU according to the number of layers.

|       |        |        |        | p 2   | p 2    | p 2    | p 3   | p 3    | p 3    |
|-------|--------|--------|--------|-------|--------|--------|-------|--------|--------|
| Layer | Train  | Test 1 | Test 2 | Train | Test 1 | Test 2 | Train | Test 1 | Test 2 |
| 1     | 96.9%  | 96.4%  | 74.8%  | 69.8% | 71.0%  | 56.7%  | 69.1% | 68.8%  | 75.4%  |
| 2     | 100.0% | 100.0% | 99.5%  | 83.7% | 84.5%  | 75.3%  | 76.6% | 76.8%  | 77.0%  |
| 3     | 97.6%  | 97.3%  | 87.2%  | 83.6% | 84.2%  | 75.1%  | 76.7% | 76.4%  | 66.9%  |
| 4     | 68.6%  | 68.4%  | 67.3%  | 83.5% | 84.0%  | 76.1%  | 77.7% | 76.3%  | 46.6%  |
| 5     | 68.5%  | 68.3%  | 67.0%  | 83.5% | 83.9%  | 77.6%  | 78.2% | 76.8%  | 34.1%  |
| 6     | 68.5%  | 68.4%  | 66.1%  | 83.6% | 84.1%  | 79.6%  | 77.6% | 75.8%  | 34.8%  |
| 7     | 68.5%  | 68.5%  | 67.3%  | 83.5% | 83.8%  | 80.5%  | 77.1% | 77.7%  | 49.4%  |
| 8     | 68.5%  | 68.4%  | 65.8%  | 83.4% | 83.8%  | 73.2%  | 76.7% | 75.7%  | 75.1%  |
| 9     | 68.5%  | 68.3%  | 66.7%  | 83.0% | 83.4%  | 79.1%  | 77.3% | 76.9%  | 48.0%  |
| 10    | 68.6%  | 68.3%  | 65.5%  | 83.1% | 83.7%  | 77.3%  | 76.4% | 75.6%  | 37.4%  |

Table 4 presents the accuracy of the ACR-GNN model with ReLU activation across three FOproperties ( p 1 , p 2 , and p 3 ), evaluated on Train, Test1, and Test2 splits. For p 1 , the model achieves high accuracy in the first three layers, peaking at 99.5% on Test2 at layer 2. From layer 4 and beyond, the accuracy on Test2 declines and stabilizes around 66-67%, suggesting a decreased performance in deeper models for this property. For p 2 , initial accuracy is modest (e.g., 69.8% on Train and 56.7% on Test2 at layer 1), but improves rapidly with depth, surpassing 83% from layer 2 onward on Train and Test1. In particular, the accuracy of Test2 continues to improve with depth, reaching a peak at 80.5% in layer 7, indicating that p 2 benefits from deeper architectures. In contrast, p 3 exhibits less consistent behavior. Accuracy improves early, reaching 77.0% on Test2 at layer 2, but then drops sharply: Test2 accuracy drops to 46.6% at layer 4 and reaches a minimum of 34.1% at layer 5. Some recovery is observed at layers 7 and 8, yet performance remains unstable, with Test2 accuracy at 37.4% by layer 10. Overall, the results demonstrate that model depth significantly affects performance depending on the target property. While p 2 benefits from deeper configurations, both p 1 and p 3 achieve higher generalization performance in shallower networks, with deeper layers leading to overfitting or reduced representation quality on unseen data.

Table 5: Accuracy of the ACR-GNN with ReLU after dynamic PTQ according to the number of layers.

|       | p 1    | p 1    | p 1    | p 2   | p 2    | p 2    | p 3   | p 3    | p 3    |
|-------|--------|--------|--------|-------|--------|--------|-------|--------|--------|
| Layer | Train  | Test 1 | Test 2 | Train | Test 1 | Test 2 | Train | Test 1 | Test 2 |
| 1     | 96.5%  | 95.7%  | 75.3%  | 69.7% | 70.8%  | 65.6%  | 68.8% | 68.2%  | 74.7%  |
| 2     | 100.0% | 100.0% | 99.4%  | 83.8% | 84.4%  | 75.5%  | 76.4% | 76.6%  | 77.0%  |
| 3     | 97.6%  | 97.4%  | 86.7%  | 83.5% | 84.1%  | 74.7%  | 76.7% | 76.7%  | 66.5%  |
| 4     | 68.6%  | 68.5%  | 66.9%  | 83.3% | 84.2%  | 76.2%  | 77.6% | 76.1%  | 44.6%  |
| 5     | 68.5%  | 68.2%  | 67.2%  | 83.4% | 84.0%  | 77.8%  | 78.3% | 76.6%  | 33.4%  |
| 6     | 68.6%  | 68.4%  | 66.2%  | 83.5% | 83.9%  | 80.3%  | 77.4% | 75.6%  | 35.8%  |
| 7     | 68.5%  | 68.4%  | 67.1%  | 83.3% | 83.6%  | 80.6%  | 77.1% | 77.6%  | 48.7%  |
| 8     | 68.5%  | 68.3%  | 65.8%  | 83.3% | 83.7%  | 73.2%  | 76.7% | 75.5%  | 74.6%  |
| 9     | 68.5%  | 68.3%  | 66.6%  | 83.0% | 83.6%  | 78.9%  | 77.1% | 76.2%  | 44.3%  |
| 10    | 68.5%  | 68.2%  | 58.1%  | 83.0% | 83.7%  | 77.5%  | 76.3% | 75.4%  | 36.6%  |

Table 5 presents the node-level accuracy of the ACR-GNN model with ReLU activation after applying 931 dynamic post-training quantization (PTQ). Results are reported for three FO-properties ( p 1 , p 2 , 932

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

961

962

963

964

965

966

967

968

969

970

971

972

and p 3 ), evaluated across the Train, Test1, and Test2 splits. For p 1 , the quantized model achieves near-perfect accuracy at layer 2 (Train: 100.0%, Test1: 100.0%, Test2: 99.4%), indicating optimal performance at this depth. Beyond layer 3, accuracy gradually degrades, with Test2 accuracy falling to 58.1% by layer 10. This suggests that deeper networks may amplify quantization-related degradation, especially in generalization.For p 2 , the quantized model demonstrates stable and robust accuracy across most depths. Starting from moderate performance in layer 1 (Train: 69.7%, Test2: 65.6%), accuracy increases quickly and exceeds 83.0% from layer 2 onward in Train and Test1 splits. In particular, the accuracy of Test2 continues to improve up to layer 7 (80.6%), showing resilience to quantization effects even in deeper architectures.In contrast, p 3 exhibits more irregular behavior. Accuracy improves slightly in the early layers (Test2 peaks at 77.0% at layer 2), but then drops substantially, reaching a low of 33.4% at layer 5. Despite stable Train and Test1 accuracy ( 76-78%), the significant reduction in Test2 suggests overfitting and reduced generalization performance in deeper networks due to quantization. Dynamic PTQ preserves performance well for p 2 in depths, but negatively impacts p 1 and especially p 3 in deeper configurations. This underscores the need for depth-sensitive or property-sensitive quantization strategies when deploying GNNs under resource constraints.

Table 6: Difference in the percentages of the accuracy of ACR-GNN with ReLU before and after dynamic PTQ, rounded to two decimal places.

|       | p 1    | p 1    | p 1    | p 2    | p 2    | p 2    | p 3    | p 3    | p 3    |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Layer | Train  | Test 1 | Test 2 | Train  | Test 1 | Test 2 | Train  | Test 1 | Test 2 |
| 1     | -0.45% | -0.76% | 0.52%  | -0.13% | -0.18% | 8.89%  | -0.30% | -0.65% | -0.69% |
| 2     | 0.00%  | 0.00%  | -0.04% | 0.08%  | -0.13% | 0.14%  | -0.18% | -0.23% | 0.02%  |
| 3     | -0.04% | 0.06%  | -0.49% | -0.16% | -0.14% | -0.34% | -0.02% | 0.28%  | -0.35% |
| 4     | 0.01%  | 0.02%  | -0.40% | -0.19% | 0.19%  | 0.06%  | -0.05% | -0.20% | -1.99% |
| 5     | -0.06% | -0.13% | 0.19%  | -0.11% | 0.06%  | 0.26%  | 0.03%  | -0.22% | -0.73% |
| 6     | 0.02%  | 0.01%  | 0.06%  | -0.03% | -0.18% | 0.70%  | -0.23% | -0.25% | 0.95%  |
| 7     | 0.00%  | -0.11% | -0.16% | -0.19% | -0.26% | 0.12%  | -0.00% | -0.17% | -0.75% |
| 8     | -0.03% | -0.09% | -0.01% | -0.12% | -0.12% | -0.02% | -0.05% | -0.28% | -0.49% |
| 9     | -0.03% | -0.01% | -0.04% | 0.01%  | 0.21%  | -0.13% | -0.26% | -0.72% | -3.74% |
| 10    | -0.00% | -0.10% | -7.38% | -0.14% | 0.05%  | 0.20%  | -0.08% | -0.14% | -0.78% |

Table 6 reports the accuracy differences in percentage points between the original ACR-GNN model with ReLU activation and its dynamically quantized counterpart, using Post-Training quantization (PTQ). The results cover three FO properties ( p 1 , p 2 , p 3 ), three dataset splits (Train, Test1, Test2). Positive values indicate better accuracy after quantization, while negative values indicate degradation. For p 1 , quantization generally causes negligible or negative changes in accuracy. For example, at layer 2, the differences are minimal (Train: 0.00%, Test1: 0.00%, Test2: -0.04%), showing nearidentical behavior between the models. However, deeper networks experience more substantial performance drops, especially at layer 10 in Test2 (-7.38%), indicating increased instability due to depth quantization. These patterns highlight a general sensitivity to depth, particularly when generalizing to larger test graphs. In contrast, p 2 exhibits greater resilience to quantization, with occasional performance gains. A notable improvement appears in layer 1 on Test2 (+8.89%), along with smaller gains in layers 5 (+0.26%), 6 (+0.70%) and 10 (+0.20%). However, inconsistencies are still present, for example, a Test2 drop at layer 3 (-0.34%) - which implies that while p 2 benefits more than p 1 , gains are not uniform across the board. p 3 , on the other hand, exhibits the most erratic behavior and is generally more susceptible to quantization. Although a modest gain appears in layer 6 in Test2 (+0.95%), severe degradation is observed in layer 4 (-1.99%) and layer 9 (-3.74%). Across layers and divisions, accuracy losses dominate, suggesting that p 3 is particularly sensitive to quantization, especially in deeper models. In summary, dynamic PTQ results in non-uniform effects across properties, dataset splits, and depths. Although p 2 shows the most consistent tolerance and even improvement in certain cases, p 1 and p 3 are more susceptible to degradation, especially in the Test2 split in deeper configurations. These results emphasize the importance of property-specific and depth-aware quantization strategies to maintain performance in FO-property learning with GNN.

Table 7 presents the accuracy of the ACR-GNN model with truncated ReLU activation on three FO properties ( p 1 , p 2 , and p 3 ), evaluated on the Train, Test1, and Test2 datasets as the number

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

Table 7: Accuracy of the ACR-GNN with truncated ReLU according to the number of layers.

|       | p 1    | p 1    | p 1    | p 2   | p 2    | p 2    | p 3   | p 3    | p 3    |
|-------|--------|--------|--------|-------|--------|--------|-------|--------|--------|
| Layer | Train  | Test 1 | Test 2 | Train | Test 1 | Test 2 | Train | Test 1 | Test 2 |
| 1     | 98.7%  | 98.4%  | 87.0%  | 77.2% | 78.3%  | 51.1%  | 69.9% | 69.8%  | 71.5%  |
| 2     | 100.0% | 100.0% | 98.3%  | 69.8% | 70.0%  | 63.7%  | 75.2% | 76.5%  | 75.3%  |
| 3     | 63.1%  | 61.7%  | 57.9%  | 67.8% | 67.6%  | 62.9%  | 66.3% | 65.7%  | 70.6%  |
| 4     | 58.4%  | 58.0%  | 48.6%  | 66.4% | 66.3%  | 61.3%  | 61.2% | 59.2%  | 50.3%  |
| 5     | 55.7%  | 54.3%  | 50.4%  | 63.0% | 64.3%  | 39.6%  | 64.4% | 65.1%  | 66.5%  |
| 6     | 55.5%  | 54.6%  | 50.1%  | 63.0% | 64.3%  | 39.5%  | 58.2% | 57.3%  | 34.6%  |
| 7     | 53.8%  | 54.2%  | 51.4%  | 63.4% | 64.9%  | 41.7%  | 57.1% | 56.0%  | 23.3%  |
| 8     | 52.7%  | 53.6%  | 50.8%  | 63.1% | 64.0%  | 40.0%  | 61.4% | 61.5%  | 55.3%  |
| 9     | 52.5%  | 52.5%  | 51.1%  | 65.0% | 65.0%  | 49.2%  | 57.2% | 56.0%  | 24.7%  |
| 10    | 54.7%  | 54.8%  | 51.1%  | 63.0% | 64.3%  | 39.6%  | 57.2% | 55.6%  | 23.4%  |

of GNN layers increases from 1 to 10. For p 1 , the model exhibits strong performance in shallow configurations, peaking at layer 2 with 100.0% (Train), 100.0% (Test1), and 98.3% (Test2) accuracy. However, performance deteriorates significantly beyond this point: by layer 3, Test2 accuracy drops to 57.9%, and continues to decline in deeper layers, stabilizing around 51.1% by layer 10. This trend suggests overfitting, as training accuracy remains high while generalization performance on Test2 degrades with depth. The accuracy profile of p 2 is more stable. While initial performance is moderate (Test2: 51.1% at layer 1), the model maintains consistent accuracy from layer 3 onward, with minor fluctuations. The narrower gap between training and testing accuracy indicates that p 2 is less sensitive to overfitting and more robust to increasing depth. For p 3 , the model initially performs well, reaching 75.3% on Test2 at layer 2. However, deeper architectures result in a steep decline in generalization performance: Test2 accuracy falls to 50.3% at layer 4, 34.6% at layer 6, and just 23.3% by layer 7. Despite relatively stable scores on Train and Test1, the Test2 drop-evidenced by a gap of over 38 percentage points at layer 7-reflects significant overfitting. In summary, ACR-GNN model with truncated ReLU benefits most from shallow architectures for p 1 and p 3 , whereas p 2 exhibits more resilient behavior across network depths. These results highlight the need for depth-aware design when targeting different FO properties under quantization constraints.

Table 8: Accuracy of the ACR-GNN with truncated ReLU after dynamic PTQ according to the number of layers.

|       | p 1    | p 1    | p 1    | p 2   | p 2    | p 2    | p 3   | p 3    | p 3    |
|-------|--------|--------|--------|-------|--------|--------|-------|--------|--------|
| Layer | Train  | Test 1 | Test 2 | Train | Test 1 | Test 2 | Train | Test 1 | Test 2 |
| 1     | 98.8%  | 98.8%  | 86.4%  | 76.2% | 77.8%  | 59.5%  | 69.4% | 69.3%  | 74.8%  |
| 2     | 100.0% | 100.0% | 94.4%  | 69.6% | 69.7%  | 42.4%  | 74.8% | 76.3%  | 59.6%  |
| 3     | 61.5%  | 59.1%  | 54.9%  | 67.8% | 68.0%  | 63.6%  | 66.1% | 65.3%  | 70.7%  |
| 4     | 58.3%  | 57.7%  | 47.9%  | 66.2% | 66.7%  | 43.1%  | 61.0% | 57.5%  | 46.0%  |
| 5     | 55.4%  | 54.0%  | 50.5%  | 63.0% | 64.3%  | 39.6%  | 63.9% | 57.4%  | 65.5%  |
| 6     | 55.5%  | 55.8%  | 50.0%  | 63.0% | 64.3%  | 39.8%  | 57.5% | 56.8%  | 32.5%  |
| 7     | 53.4%  | 53.1%  | 50.9%  | 62.4% | 62.5%  | 44.8%  | 56.8% | 56.2%  | 24.5%  |
| 8     | 52.5%  | 53.6%  | 51.0%  | 61.4% | 63.0%  | 40.0%  | 61.4% | 62.7%  | 50.0%  |
| 9     | 52.6%  | 52.4%  | 51.2%  | 65.0% | 65.7%  | 53.7%  | 57.2% | 55.6%  | 23.7%  |
| 10    | 54.8%  | 53.9%  | 51.3%  | 63.1% | 64.3%  | 39.6%  | 56.9% | 55.1%  | 23.6%  |

Table 8 reports the accuracy of the ACR-GNN model after applying dynamic PTQ across three logical 989 query patterns ( p 1 , p 2 , p 3 ) and a range of GNN layers ( l from 1 to 10). A general observation is that 990 dynamic PTQ causes more pronounced performance degradation as the number of layers increases, 991 particularly for p 1 and p 3 . While accuracy remains high for shallow configurations, especially at 992 l = 1 and l = 2 (e.g., p 1 reaches 98.8% on Test1 at l = 1 and 100.0% on Train and Test1 at l = 2 )-a 993 sharp decline follows beyond l = 2 . For instance, p 1 training accuracy drops from 100.0% at l = 2 994

995

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

1026

1027

1028

1029

1030

1031

1032

1033

1034

to 61.5% at l = 3 , with continued degradation in deeper layers.In contrast, p 2 starts with slightly lower accuracy but exhibits relatively stable behavior across layers. Its accuracy remains in the 60-78% range across all datasets, showing less sensitivity to depth. However, a gradual decline in the precision of Test2 is noticeable, ranging from 59.5% at l = 1 to 39.6% at l = 10 , suggesting that generalization to more complex test graphs is still affected by quantization. The pattern p 3 is the most affected. Although some recovery is observed at intermediate layers (e.g., 70.7% Test2 accuracy at l = 3 ), performance deteriorates with increasing depth, reaching only 23.6% on Test2 at l = 10 . In summary, dynamic PTQ enables significant model compression for ACR-GNNs, but at the cost of accuracy, particularly in deeper architectures and complex FO-query patterns such as p 1 and p 3 . Shallow configurations (e.g., l ≤ 2 ) maintain good performance after quantization, indicating that careful depth-aware quantization strategies are essential for preserving generalization.

Table 9: Difference in the percentages of the accuracy of ACR-GNN with truncated ReLU before and after dynamic PTQ.

|       | p 1   | p 1    | p 1    | p 2   | p 2    | p 2    | p 3   | p 3    | p 3    |
|-------|-------|--------|--------|-------|--------|--------|-------|--------|--------|
| Layer | Train | Test 1 | Test 2 | Train | Test 1 | Test 2 | Train | Test 1 | Test 2 |
| 1     | 0.1%  | 0.3%   | -0.6%  | -1.0% | -0.5%  | 8.4%   | -0.5% | -0.5%  | 3.4%   |
| 2     | 0.0%  | 0.0%   | -3.9%  | -0.2% | -0.3%  | -21.3% | -0.5% | -0.2%  | -15.7% |
| 3     | -1.6% | -2.7%  | -3.0%  | 0.0%  | 0.4%   | 0.7%   | -0.2% | -0.4%  | 0.1%   |
| 4     | -0.2% | -0.3%  | -0.8%  | -0.2% | 0.5%   | -18.2% | -0.2% | -1.7%  | -4.3%  |
| 5     | -0.3% | -0.3%  | 0.2%   | 0.0%  | 0.0%   | 0.0%   | -0.6% | -7.7%  | -1.0%  |
| 6     | -0.0% | 1.2%   | -0.1%  | -0.0% | 0.0%   | 0.3%   | -0.6% | -0.5%  | -2.2%  |
| 7     | -0.4% | -1.2%  | -0.5%  | -1.0% | -2.3%  | 3.1%   | -0.4% | 0.2%   | 1.2%   |
| 8     | -0.2% | 0.0%   | 0.2%   | -1.7% | -1.0%  | -0.0%  | 0.0%  | 1.3%   | -5.3%  |
| 9     | 0.2%  | -0.1%  | 0.1%   | 0.0%  | 0.7%   | 4.5%   | 0.1%  | -0.5%  | -1.0%  |
| 10    | 0.1%  | -0.9%  | 0.3%   | 0.0%  | 0.0%   | 0.0%   | -0.3% | -0.5%  | 0.2%   |

Table 9 presents the percentage changes in accuracy of the ACR-GNN model with truncated ReLU after applying Dynamic Post-Training quantization (PTQ), across three query patterns ( p 1 , p 2 , p 3 ) and for different numbers of GNN layers ( l = 1 to l = 10 ). The difference is calculated as the quantized accuracy minus the original, scaled to a percentage. In the case of this table, we can see changes layer by layer. Here, where l = 1 , we observe small improvements in accuracy. If we examine this more precisely, for p 1 , the precision improves across all datasets, with the highest gain in Test2 (+11.1%). p 2 shows a mixed pattern with small increases in Train / Test1, but a decrease in Test2 (-6.1%). p 3 remains stable, showing minimal change ( ≤ 1 . 2% ). When l = 2 , the results show early degradation, as p 2 suffers significant drops, especially on Test2 (-33.0%), while p 3 sees a drop in Test2 of -17.4%, p 1 remains unchanged on Train / Test1 and slightly lower (-5.0%) on Test2. A major drop occurs when l = 3 for p 1 , with -36.1% on Train and -38.3% on Test1. p 2 also shows a negative trend, but Test2 is impacted less than in Layer 2. Interestingly, p 3 has a positive change in Test2 (+4.2%), indicating some robustness in this setting. The continuous trend for layers from 4 to 9. For l = 10 , p 1 appears to recover slightly in Test2 (-6.8%, compared to - 15% previously). However, p 2 and p 3 still show substantial losses (-37.9% and -13.1% respectively), suggesting that deeper architectures struggle consistently after dynamic quantization. In summary, Table 9 highlights the accuracy losses due to dynamic PTQ. This correlates with the literature [15], where the authors noted some loss in accuracy, but the quantized model should provide better results in comparing the size. Although some early layers benefit slightly, deeper layers consistently show reduced accuracy, especially in Test2, the data set with larger, more complex graphs. The pattern confirms that dynamic PTQ, though efficient, can harm generalization, particularly in deeper and more expressive GNN configurations.

After presenting the accuracy results before and after applying dynamic Post-Training Quantization (PTQ), we proceed to analyze the influence of the activation function on the performance of the model. This comparison is provided both graphically and in tabular form. For the graphical representation, we utilized box plots, a statistical tool designed to visualize the distribution of a variable in terms of its quartiles. In these plots, the box itself spans from the first quartile (Q1) to the third quartile (Q3), with the median value (Q2) marked by a line within the box. The whiskers of the box plot extend to the minimum and maximum values that do not qualify as outliers, providing insight into the spread

and concentration of the data. In addition to these visualizations, a detailed table complements the 1035 analysis by presenting summary statistics. The table includes the mean, standard deviation, minimum, 1036 and maximum values for each configuration. It also presents the three quartiles: Q1, which represents 1037 the 25th percentile, Q2, or the median, which is the 50th percentile, and Q3, the 75th percentile. 1038 These quartiles divide the data into four equal parts, helping to identify the central tendency and 1039 variability. Furthermore, we calculate the interquartile range (IQR), defined as the difference between 1040 the third quartile (Q3) and the first quartile (Q1), which serves as a measure of statistical dispersion. 1041 Based on the IQR, we also determine the lower and upper bounds using the standard rule, which 1042 involves subtracting 1.5 times the IQR from Q1 and adding it to Q3, respectively. These bounds 1043 enable the identification of potential outliers and provide a more comprehensive understanding of 1044 how the activation function and quantization impact the distribution of model accuracy. All metrics 1045 were applied to all datasets: Train, Test1, and Test2. For the visualization part, we used the Python 1046 library Plotly. 1047

<!-- image -->

Config

Figure 3: Detailed summary statistics across configurations for p 1 formula.

Table 10: Detailed summary statistics across configurations for p 1 formula.

| Statistic    |   ReLU |   ReLU + PTQ |   Truncated ReLU |   Truncated ReLU + PTQ |
|--------------|--------|--------------|------------------|------------------------|
| Mean         |  0.758 |        0.755 |            0.628 |                  0.623 |
| Std          |  0.132 |        0.134 |            0.178 |                  0.177 |
| Min          |  0.655 |        0.581 |            0.486 |                  0.479 |
| 25% (Q1)     |  0.683 |        0.682 |            0.525 |                  0.524 |
| 50% (Median) |  0.685 |        0.685 |            0.547 |                  0.544 |
| 75% (Q3)     |  0.841 |        0.839 |            0.609 |                  0.589 |
| Max          |  1     |        1     |            1     |                  1     |
| IQR          |  0.158 |        0.157 |            0.084 |                  0.065 |
| Lower Bound  |  0.446 |        0.447 |            0.399 |                  0.427 |
| Upper Bound  |  1.078 |        1.073 |            0.734 |                  0.686 |

Table 10 and Figure 3 present summary statistics for the accuracy results obtained from four config1048 urations of the ACR-GNN model: ReLU, ReLU with dynamic Post-Training Quantization (PTQ), 1049 Truncated ReLU, and Truncated ReLU with PTQ. The results show that the highest mean accuracy 1050 is achieved with the ReLU configuration (0.758), closely followed by ReLU + PTQ (0.755). This 1051 indicates that applying dynamic quantization to the ReLU model does not significantly reduce the av1052 erage accuracy. In contrast, both Truncated ReLU (0.628) and Truncated ReLU + PTQ (0.623) result 1053 in noticeably lower mean values, suggesting that this activation function may degrade performance 1054 on the p 1 query pattern. The median values align with the mean, further confirming this trend. In 1055 terms of variability, the standard deviation is lower for the ReLU-based models ( 0.13), whereas the 1056 truncated ReLU configurations show higher variability ( 0.18). This pattern is also reflected in the 1057 interquartile range (IQR): ReLU configurations exhibit wider IQRs (0.158 and 0.157), while truncated 1058 versions have narrower ranges (0.084 and 0.065). Despite the narrower spread, the performance is 1059 consistently lower with truncated ReLU. All configurations include samples that achieve a maximum 1060

accuracy of 1.0, indicating that optimal predictions are possible in all cases. However, minimum 1061 accuracy drops more sharply in truncated ReLU models (0.486 and 0.479) compared to ReLU (0.655 1062 and 0.581), indicating a higher risk of underperformance. The lower and upper bounds provide 1063 insight into potential outliers. The lower bounds are lower in the truncated models, while the upper 1064 bounds are higher in ReLU configurations (exceeding 1.0 due to statistical calculation), indicating a 1065 wider spread and potentially higher ceiling for performance.

<!-- image -->

Config

Figure 4: Detailed summary statistics across configurations for p 2 formula.

Table 11: Detailed summary statistics across configurations for p 2 formula.

| Statistic    |   ReLU |   ReLU + PTQ |   Truncated ReLU |   Truncated ReLU + PTQ |
|--------------|--------|--------------|------------------|------------------------|
| Mean         | 0.7992 |       0.802  |           0.6064 |                 0.5967 |
| Std          | 0.0615 |       0.0511 |           0.1085 |                 0.1122 |
| Min          | 0.567  |       0.656  |           0.395  |                 0.396  |
| 25% (Q1)     | 0.7738 |       0.7758 |           0.617  |                 0.5515 |
| 50% (Median) | 0.834  |       0.833  |           0.6385 |                 0.6305 |
| 75% (Q3)     | 0.837  |       0.8368 |           0.6598 |                 0.6608 |
| Max          | 0.845  |       0.844  |           0.783  |                 0.778  |
| IQR          | 0.0632 |       0.061  |           0.0428 |                 0.1093 |
| Lower Bound  | 0.6789 |       0.6843 |           0.5529 |                 0.3876 |
| Upper Bound  | 0.9319 |       0.9282 |           0.7239 |                 0.8246 |

Table 11 and Figure 4 present a comprehensive overview of the accuracy results in four model 1067 configurations: ReLU, ReLU with dynamic post-training quantization (PTQ), Truncated ReLU, and 1068 Truncated ReLU with PTQ - for the query formula p 2 . From the mean accuracy values, ReLU and 1069 ReLU + PTQ clearly outperform the other configurations, achieving 0.7992 and 0.8020, respectively. 1070 This indicates that both setups yield strong overall performance, with dynamic quantization having a 1071 slightly positive effect on average accuracy in this case. In contrast, Truncated ReLU (0.6064) and 1072 Truncated ReLU + PTQ (0.5967) show substantially lower mean values, highlighting a notable drop 1073 in predictive performance when using truncated activation. Looking at the variability, the standard 1074 deviation is lower for the ReLU configurations (0.0615 and 0.0511), suggesting a more consistent 1075 accuracy. The truncated versions, especially the quantized one (0.1122), are more dispersed, reflecting 1076 greater instability. This is further emphasized by the IQR values: 0.0632 and 0.0610 for ReLU and 1077 ReLU + PTQ versus 0.0428 for Truncated ReLU and a larger 0.1093 for Truncated ReLU + PTQ. 1078 The larger IQR for Truncated ReLU + PTQ implies a larger fluctuation in the middle 50% of the data, 1079 despite its lower central values. The median values confirm this trend: both ReLU configurations 1080 cluster around 0.833-0.834, while truncated versions fall between 0.6305 and 0.6385. The lower 1081 bounds, derived from Q1 - 1.5 × IQR, are also lower in the Truncated ReLU + PTQ case (0.3876), 1082 indicating a greater potential for underperformance and a higher risk of poor accuracy. The maximum 1083 and minimum values highlight the performance extremes. ReLU configurations reach up to 0.845 1084 and 0.844, significantly higher than the 0.783 and 0.778 of truncated variants. The lower minimum 1085 accuracy (0.395-0.396) in truncated settings further reinforces concerns about their reliability. 1086

1066

Accuracy Distribution Across Activation Functions and Quantization Settings for pformula

Figure 5: Detailed summary statistics across configurations for p 3 formula.

<!-- image -->

Table 12: Detailed summary statistics across configurations for p 3 formula.

| Statistic    |   ReLU |   ReLU + PTQ |   Truncated ReLU |   Truncated ReLU + PTQ |
|--------------|--------|--------------|------------------|------------------------|
| Mean         | 0.6883 |       0.6844 |           0.5821 |                 0.5694 |
| Std          | 0.1434 |       0.1466 |           0.1441 |                 0.1427 |
| Min          | 0.341  |       0.334  |           0.233  |                 0.236  |
| 25% (Q1)     | 0.6888 |       0.6835 |           0.56   |                 0.5575 |
| 50% (Median) | 0.7635 |       0.7615 |           0.602  |                 0.575  |
| 75% (Q3)     | 0.7688 |       0.767  |           0.6645 |                 0.6545 |
| Max          | 0.782  |       0.783  |           0.765  |                 0.763  |
| IQR          | 0.08   |       0.0835 |           0.1045 |                 0.097  |
| Lower Bound  | 0.5687 |       0.5582 |           0.4032 |                 0.412  |
| Upper Bound  | 0.8888 |       0.8922 |           0.8213 |                 0.8    |

Table 12 and Figure 5 provide descriptive statistics for the accuracy of the ACR-GNN model under 1087 four configurations-ReLU, ReLU with dynamic Post-Training Quantization (PTQ), Truncated 1088 ReLU, and Truncated ReLU with PTQ-for the p 3 query formula. Starting with the mean accuracy, 1089 ReLU (0.6883) and ReLU + PTQ (0.6844) again outperform the Truncated ReLU configurations, 1090 which register noticeably lower means of 0.5821 and 0.5694, respectively. This indicates that models 1091 that use ReLU activations are generally more effective for p 3 . The standard deviation values are 1092 relatively similar across all configurations (approximately 0.14), suggesting that while the truncated 1093 configurations perform worse on average, they do not fluctuate more widely than the ReLU-based 1094 ones. The minimum values further emphasize the performance gap: ReLU models maintain minimum 1095 accuracies above 0.33, while truncated variants drop to as low as 0.233. This shows that truncated 1096 configurations are more prone to poor performance in the worst-case scenarios. In terms of quartiles, 1097 ReLU and ReLU + PTQ have Q1 and Q3 clustered around 0.68-0.77, indicating that the middle 50% 1098 of their results are concentrated within a tight and relatively high accuracy range. Truncated ReLU 1099 variants have their Q1 around 0.56 and Q3 near 0.65, which not only shows lower performance but 1100 also a wider IQR (0.1045 for Truncated ReLU and 0.0970 for Truncated ReLU + PTQ). This reflects 1101 more variability across the central portion of the data in the truncated setups. The median accuracy is 1102 again higher in ReLU configurations (around 0.76), compared to 0.60 and 0.575 for truncated ones, 1103 reinforcing the conclusion that ReLU configurations are more reliable. Examining the bounds, the 1104 ReLU models show a lower bound above 0.55 and upper bounds above 0.88, suggesting strong and 1105 consistent performance. Truncated models exhibit lower bounds near 0.40 and upper bounds around 1106 0.80, indicating both a lower floor and a lower ceiling in performance. 1107

1108

1109

1110

1111

1112

Across all query patterns ( p 1 , p 2 , and p 3 ), ReLU and ReLU + PTQ consistently demonstrate higher average accuracy and more stable performance, making them the most reliable configurations. In contrast, Truncated ReLU and its quantized variant result in lower accuracy and greater variability, especially in worst-case scenarios. Dynamic PTQ tends to maintain or slightly enhance performance in ReLU models, but its effect on truncated activations is less favorable, often introducing further

inconsistency. Overall, ReLU-based configurations-quantized or not-are better suited for the 1113 ACR-GNN model across the evaluated formulas. 1114

Other parameters of interest to us are the time and size of the models. In the event of changes in 1115 size, it is easy to compare the data using the bar plots presented in Figure 6. The size changes in 1116 percentages we calculated according to the formula: 1117

<!-- formula-not-decoded -->

1118

1119

In other words, this formula shows how much the dynamic PTQ value deviates from the original value as a percentage of the original value.

In this section, we compare parameters for different activation functions. We observe that the results 1120 of size changes in the following models remain unchanged when we modify the training dataset. We 1121 present the results not only graphically but also in a tabular format. In the plots, it is possible to see 1122 the trends and, in the tabular format, the numerical changes. 1123

Table 13: Detailed information about the size of the model. The size values are in megabytes and refer to the file sizes of the GNNs.

|   Layer |   Original Size (MB) |   Quantized Size (MB) |   Difference (MB) | Reduction (%)   |
|---------|----------------------|-----------------------|-------------------|-----------------|
|       1 |                0.057 |                 0.023 |             0.034 | 59.604%         |
|       2 |                0.112 |                 0.044 |             0.068 | 60.993%         |
|       3 |                0.167 |                 0.064 |             0.103 | 61.559%         |
|       4 |                0.221 |                 0.085 |             0.137 | 61.804%         |
|       5 |                0.276 |                 0.105 |             0.171 | 61.975%         |
|       6 |                0.331 |                 0.126 |             0.206 | 62.068%         |
|       7 |                0.386 |                 0.146 |             0.24  | 62.148%         |
|       8 |                0.441 |                 0.167 |             0.274 | 62.194%         |
|       9 |                0.496 |                 0.187 |             0.309 | 62.230%         |
|      10 |                0.551 |                 0.208 |             0.343 | 62.251%         |

Table 13 provides a detailed comparison of the model sizes before and after applying dynamic

1124 post-training quantization (PTQ). As the number of layers increases, both the original and quantized 1125 model sizes grow; however, the percentage reduction remains remarkably consistent, ranging from 1126 approximately 60.993% at 2 layers to 62.251% at 10 layers. This stable percentage reduction, 1127 approximately 60-62%-indicates that PTQ effectively compresses the model regardless of its depth, 1128 significantly reducing the memory footprint without altering the underlying architecture of the GNN. 1129 Such a reduction is particularly crucial for deployments in resource-constrained environments. 1130

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

Furthermore, after presenting the tabular data, our graphs (Figure 6) reveal a clear trend: While the absolute sizes of the original and quantized models increase with the number of layers, the relative reduction achieved through dynamic PTQ remains consistent. The size of the original model increases approximately linearly from 0.057 MB for l = 1 to 0.551 MB at l = 10 , while the quantized model grows from 0.023 MB to 0.208 MB, preserving the growth structure, but on a reduced scale. The absolute size difference increases from 0.034 MB in l = 1 to 0.343 MB in l = 10 , demonstrating that quantization becomes more beneficial for deeper models. Overall, the consistent percentage reduction across all tested configurations confirms that PTQ scales effectively, delivering stable compression rates and making it an attractive option for deeper GNN deployments in real-world edge or mobile environments.

Moreover, we observed that the query property had no noticeable impact on the model size. This can be clearly seen in the bar plots in Figure 6a, Figure 6c, and Figure 6e.

We also measured the change over time. Specifically, we considered three distinct time metrics: 1143 Elapsed time (the time taken during training), Time Original (the time required for inference 1144 on the test datasets using the original trained model), and Time quantized (the inference time on 1145 the test datasets using the quantized model). These results are presented in Figure 7. 1146

The data in Figure 7 reflect the impact of dynamic PTQ on the ACR-GNN model in three query 1147 patterns ( p 1 , p 2 , and p 3 ) and for GNN depths ranging from 1 to 10 layers. Across all patterns, 1148

<!-- image -->

(a) Size changes in MB for the first formula

<!-- image -->

(c) Size changes in MB for the second formula

<!-- image -->

- (e) Size changes in MB for the third formula

<!-- image -->

(b) Size changes in MB for the first formula. Difference present in percentage.

(d) Size changes in MB for the second formula. Difference present in percentage.

<!-- image -->

(f) Size changes in MB for the third formula Difference present in percentage.

<!-- image -->

Figure 6: Impact of dynamic Post-Training quantization on model size (MB). Changes of size in percentages

quantized models consistently require more inference time than their original counterparts. This 1149 increased time is expected as a result of the real-time quantization of weights and activations during 1150 inference. Additionally, both the original and quantized models exhibit a consistent, near-linear 1151 increase in inference time with model depth, suggesting that computational complexity grows linearly 1152 as layers are added. 1153

Despite this overhead, which ranges between 0.1 and 0.9 s depending on the number of layers, the 1154 significant reduction in model size (as demonstrated in Table 13 and the corresponding graphs) makes 1155 quantized models especially attractive for resource-constrained environments where minimizing the 1156 memory footprint is more critical than achieving the lowest possible latency. 1157

To test the technique not only on synthetic data, we chose the Protein-Protein Interactions (PPI) 1158 benchmark. The PPI dataset consists of graph-level mini-batches, with separate splits for Training, 1159 Validation, and Testing. 1160

In Table 14, we present a summary of the PPI dataset, which consists of 20 training graphs, 2 1161 validation graphs, and 2 test graphs. Each graph contains nodes with 50-dimensional features and 1162 supports multi-label classification with 121 possible labels. On average, each node is associated with 1163

<!-- image -->

(a) Time changes in seconds for the first formula

<!-- image -->

(b) Time changes in seconds for the second formula

<!-- image -->

- (c) Time changes in seconds for the third formula

Figure 7: Impact of dynamic Post-Training quantization on Latency (sec)

Table 14: Dataset summary.

| Dataset    |   Num Graphs |   Node Feature Dim |   Label Dim |   Avg Active Labels/Node |   Avg Degree |
|------------|--------------|--------------------|-------------|--------------------------|--------------|
| Train      |           20 |                 50 |         121 |                    37.2  |        54.62 |
| Validation |            2 |                 50 |         121 |                    35.64 |        61.07 |
| Test       |            2 |                 50 |         121 |                    36.22 |        58.64 |

approximately 36 labels, indicating a densely labelled dataset. The average node degree is also high, 1164 ranging from 54.6 in the training set to 61.1 in the validation set, reflecting the dense connectivity of 1165 the protein-protein interaction graphs. The dataset presents a complex multi-label classification task 1166 with consistently rich structure across all splits. 1167

Table 15: Dataset statistics summary.

|            | Node   | Node   | Node    | Edge   | Edge   | Edge     |
|------------|--------|--------|---------|--------|--------|----------|
| Dataset    | Min    | Max    | Avg     | Min    | Max    | Avg      |
| Train      | 591    | 3480   | 2245.30 | 7708   | 106754 | 61318.40 |
| Validation | 3230   | 3284   | 3257.00 | 97446  | 101474 | 99460.00 |
| Test       | 2300   | 3224   | 2762.00 | 61328  | 100648 | 80988.00 |

The statistics of the dataset presented in Table 15 contain large graphs with varying sizes between 1168 the train, the validation, and the test splits. Training graphs range from 591 to 3,480 nodes, with an 1169 average of 2,245 nodes per graph, and between 7,708 and 106,754 edges (average 61,318 edges). 1170 Validation graphs are more consistent in size, with 3,230 to 3,284 nodes and 97,446 to 101,474 edges, 1171 averaging 3,257 nodes and 99,460 edges. The test graphs have 2,300 to 3,224 nodes, averaging 1172 2,762 nodes, and 61,328 to 100,648 edges, averaging 80,988. These statistics confirm that the dataset 1173

contains large and densely connected graphs and demonstrate a distributional shift in graph size and 1174 edge count between training and test data. This information is helpful in evaluating the model's 1175 ability to generalize to unseen and variable graph structures. 1176

One key difference between the synthetic data and the PPI dataset is that the latter involves a 1177 multi-label classification task, rather than a binary classification task, because the PPI dataset is 1178 a common benchmark where each node (representing proteins) can have multiple labels, such as 1179 protein functions or interactions. Also, it is important to mention the key differences between the 1180 synthetic data and the real one. Here, the authors used the code function EarlyStopping : Utility 1181 for stopping training early if no further improvement is observed. The second difference is that the 1182 code is structured to run multiple experiments to collect statistics (mean and standard deviation) of 1183 the model performance, ensuring that the results are robust across different random initializations. In 1184 this case, we performed the experiments 10 times for each model, with a combination layer equal to 1 1185 and a number of layers ranging from 1 to 10. The number of hidden dimensions is equal to 256. 1186

1187

1188

1189

1190

For these experiments, we used two activation functions to compare the results with synthetic data. The presentation of the results follows the same approach as for synthetic data. Moreover, in the case of real data [4] used the F1 Score as an evaluation metric. This metric is commonly used to evaluate classification tasks.

1191

1192

1193

1194

1195

1196

1197

According to the Scikit-learn library [25], the F1 score is defined in the following way. The F1 score can be interpreted as a harmonic mean of precision and recall, where an F1 score reaches its best value at 1 and its worst score at 0. The relative contribution of precision and recall to the F1 score is equal. The formula for the F1 score is as follows:

<!-- formula-not-decoded -->

where, TP - is the number of true positives, FN - is the number of false negatives, FP - is the number of false positives. F1 is calculated by default as 0.0 when there are no true positives, false negatives, or false positives.

The reference code's results [5] are structured as follows: a table showing the loss and accuracy for 1198 each dataset (train, validation, and test). Here, we present only the accuracy of the model according 1199 to the number of layers, as we do for the synthetic data. For better representation, we formed the 1200 model's output in a tabular representation. 1201

Table 16: Accuracy for the original and quantized (dynamic PTQ) models. PPI Benchmark.

(a) Accuracy of the ACR-GNN with ReLU according to the number of layers.

(b) Accuracy of the ACR-GNN with ReLU after dynamic PTQ according to the number of layers.

|   Layer | Train   | Validation   | Test   |   Layer | Train   | Validation   | Test   |
|---------|---------|--------------|--------|---------|---------|--------------|--------|
|       1 | 54.7%   | 43.1%        | 39.5%  |       1 | 55.0%   | 50.8%        | 50.2%  |
|       2 | 52.5%   | 44.6%        | 45.7%  |       2 | 52.3%   | 47.8%        | 47.2%  |
|       3 | 52.3%   | 42.6%        | 44.0%  |       3 | 51.9%   | 45.7%        | 42.8%  |
|       4 | 52.3%   | 39.2%        | 40.6%  |       4 | 51.9%   | 37.4%        | 34.1%  |
|       5 | 49.6%   | 39.7%        | 39.1%  |       5 | 48.9%   | 39.1%        | 40.8%  |
|       6 | 49.3%   | 43.5%        | 43.3%  |       6 | 48.9%   | 42.9%        | 43.8%  |
|       7 | 51.7%   | 39.9%        | 38.5%  |       7 | 51.4%   | 43.0%        | 40.6%  |
|       8 | 50.8%   | 36.3%        | 35.8%  |       8 | 50.5%   | 35.9%        | 36.8%  |
|       9 | 48.0%   | 43.8%        | 33.2%  |       9 | 47.7%   | 40.8%        | 40.9%  |
|      10 | 47.1%   | 36.9%        | 36.8%  |      10 | 46.5%   | 36.2%        | 38.7%  |

Table 16 reports the precision of the ACR-GNN model with ReLU activation in varying numbers 1202 of layers, both in its original form and after applying dynamic post-training quantization (dPTQ). 1203 The results are presented for the training, validation, and test sets of the PPI benchmark. For both 1204 versions of the model, the performance does not increase consistently with the number of layers. 1205 Instead, accuracy typically peaks within the first few layers and tends to degrade or fluctuate as the 1206 network's depth increases. In particular, the highest accuracies for the training, validation, and test 1207

sets are achieved with 1 or 2 layers, indicating that shallower architectures are better suited for this 1208 task. Specifically, the original model achieves its best test accuracy (45.7%) at 2 layers, while the 1209 quantized model achieves an even higher test accuracy (50.2%) at just 1 layer. Dynamic quantization 1210 slightly improves generalization performance in the early layers. At layer 1, the quantized model 1211 surpasses the original in both validation (50.8% vs. 43.1%) and test accuracy (50.2% vs. 39.5%), 1212 suggesting that quantization can have a regularizing effect in low-depth configurations. However, as 1213 the number of layers increases beyond 4, the performance of both models tends to decline, likely due 1214 to over-smoothing or optimization difficulties common in deep GNNs. 1215

Table 17: Difference in accuracy of ACR-GNN with ReLU before and after dynamic PTQ. PPI Benchmark.

|   Layer | Train   | Validation   | Test   |
|---------|---------|--------------|--------|
|       1 | 0.3%    | 7.7%         | 10.7%  |
|       2 | -0.2%   | 3.2%         | 1.5%   |
|       3 | -0.4%   | 3.1%         | -1.2%  |
|       4 | -0.4%   | -1.8%        | -6.5%  |
|       5 | -0.7%   | -0.6%        | 1.7%   |
|       6 | -0.4%   | -0.6%        | 0.5%   |
|       7 | -0.3%   | 3.1%         | 2.1%   |
|       8 | -0.3%   | -0.4%        | 1.0%   |
|       9 | -0.3%   | -3.0%        | 7.7%   |
|      10 | -0.6%   | -0.7%        | 1.9%   |

Table 17 reports the absolute difference in precision between the quantized and original ACR-GNN 1216 model with ReLU on the PPI benchmark, between training, validation and test sets for varying 1217 numbers of layers. Positive values indicate better performance after quantization, while negative 1218 values reflect performance degradation. At layer 1, the quantized model shows the largest gains, with 1219 improvements of 7.7% on validation and 10.7% on the test set, suggesting a clear generalization 1220 advantage in shallow architectures. Smaller, but consistent improvements are also observed at layers 1221 2 and 7, particularly in the validation and test sets. In contrast, certain layers exhibit minor drops 1222 in accuracy. For example, layer 4 shows the largest decrease in the test set (6.5%). Overall, the 1223 results indicate that dynamic quantization can lead to modest accuracy improvements, particularly in 1224 shallow to mid-depth GNNs, with negligible or slightly negative effects in deeper configurations. This 1225 highlights the potential of quantization for lightweight deployment with minimal accuracy trade-offs. 1226

Table 18: Detailed information about the model size before and after quantization. PPI Benchmark. Sizes are in megabytes.

|   Layer |   Original Model (MB) |   Quantized Model (MB) |   Difference (MB) | Reduction (%)   |
|---------|-----------------------|------------------------|-------------------|-----------------|
|       1 |                 0.922 |                  0.242 |             0.68  | -73.749%        |
|       2 |                 1.718 |                  0.451 |             1.267 | -73.765%        |
|       3 |                 2.515 |                  0.66  |             1.855 | -73.772%        |
|       4 |                 3.311 |                  0.868 |             2.443 | -73.776%        |
|       5 |                 4.108 |                  1.077 |             3.031 | -73.778%        |
|       6 |                 4.904 |                  1.286 |             3.618 | -73.779%        |
|       7 |                 5.701 |                  1.495 |             4.206 | -73.780%        |
|       8 |                 6.497 |                  1.704 |             4.794 | -73.780%        |
|       9 |                 7.294 |                  1.912 |             5.382 | -73.781%        |
|      10 |                 8.09  |                  2.121 |             5.969 | -73.781%        |

Table 18 presents the memory footprint of the ACR-GNN model at different layer depths, comparing 1227 the original model (complete precision) with its dynamically quantized counterpart. The table 1228 also includes both absolute and percentage differences in size, highlighting the compression effect 1229 introduced by dynamic post-training quantization. Across all layers, the quantized model consistently 1230 exhibits a size reduction of approximately 73.78% compared to the original model. For example, at 10 1231

layers, the model size decreases from 8.09MB to 2.12MB, yielding an absolute reduction of 5.97MB. 1232 This trend is consistent and proportional across all depths, indicating that the memory savings scale 1233 linearly with the model's complexity (i.e., the number of layers). These results demonstrate the 1234 effectiveness of dynamic quantization in significantly reducing model size without the need for 1235 retraining. 1236

Table 19: Elapsed times (in seconds) for the original and quantized (dynamic PTQ) models. PPI Benchmark.

(a) Elapsed times for the original model.

(b) Elapsed times for the quantized model.

|   Layer |   Train |   Validation |   Test |   Layer |   Train |   Validation |   Test |
|---------|---------|--------------|--------|---------|---------|--------------|--------|
|       1 |   0.913 |        0.115 |  0.113 |       1 |   0.921 |        0.134 |  0.112 |
|       2 |   1.4   |        0.158 |  0.182 |       2 |   1.469 |        0.178 |  0.129 |
|       3 |   1.447 |        0.188 |  0.172 |       3 |   1.41  |        0.211 |  0.173 |
|       4 |   1.982 |        0.257 |  0.224 |       4 |   1.694 |        0.252 |  0.181 |
|       5 |   2.225 |        0.295 |  0.247 |       5 |   2.538 |        0.322 |  0.304 |
|       6 |   2.846 |        0.318 |  0.236 |       6 |   2.878 |        0.307 |  0.313 |
|       7 |   3.42  |        0.442 |  0.328 |       7 |   3.538 |        0.328 |  0.299 |
|       8 |   3.12  |        0.437 |  0.343 |       8 |   3.236 |        0.36  |  0.342 |
|       9 |   3.626 |        0.433 |  0.39  |       9 |   3.936 |        0.605 |  0.481 |
|      10 |   4.011 |        0.41  |  0.376 |      10 |   3.783 |        0.464 |  0.375 |

Table 21 reports the inference times of the original and dynamically post-training quantized ACR1237 GNN models across training, validation, and test datasets, measured at various layer depths. The 1238 results reveal that quantization does not significantly reduce inference time in most configurations 1239 and, in some cases, results in slightly higher latency. For the training set, the execution time of the 1240 quantized model closely follows that of the original, with negligible differences across all layers. In 1241 the validation and test sets, while some improvements are observed at shallow depths (e.g., the layer 1242 2 test time reduces from 0.182 to 0.129 s), the overall pattern indicates no consistent speedup from 1243 quantization. In fact, certain configurations, such as layers 9 and 10 in the validation set, exhibit 1244 increased latency in the quantized version compared to the original. 1245

Table 20: Difference in elapsed time (in seconds) and corresponding percentage difference of ACRGNN with ReLU before and after dynamic PTQ on the PPI Benchmark.

| Layer   | Train    | Train    | Validation   | Validation   | Test     | Test     |
|---------|----------|----------|--------------|--------------|----------|----------|
|         | Diff (s) | %Diff    | Diff (s)     | %Diff        | Diff (s) | %Diff    |
| 1       | -0.008   | 0.915%   | -0.019       | 16.307%      | 0.001    | -1.085%  |
| 2       | -0.069   | 4.931%   | -0.020       | 12.308%      | 0.053    | -29.114% |
| 3       | 0.037    | -2.525%  | -0.023       | 12.238%      | -0.001   | 0.309%   |
| 4       | 0.288    | -14.531% | 0.005        | -1.990%      | 0.043    | -19.096% |
| 5       | -0.313   | 14.091%  | -0.027       | 9.291%       | -0.057   | 23.218%  |
| 6       | -0.032   | 1.131%   | 0.011        | -3.463%      | -0.077   | 32.455%  |
| 7       | -0.118   | 3.465%   | 0.114        | -25.741%     | 0.029    | -8.918%  |
| 8       | -0.116   | 3.709%   | 0.077        | -17.556%     | 0.001    | -0.276%  |
| 9       | -0.310   | 8.555%   | -0.172       | 39.611%      | -0.091   | 23.218%  |
| 10      | 0.228    | -5.678%  | -0.054       | 13.105%      | 0.001    | -0.192%  |

Table 20 presents the difference in inference time between the original and dynamically quantized 1246 (dPTQ) ACR-GNN models, reported in absolute (seconds) and relative (%) terms, across various 1247 layer depths. The results show that quantization has an inconsistent effect on inference time, with 1248 no clear trend of improvement. In some configurations, dynamic quantization slightly reduces 1249 inference time; for example, layer 2 shows a 0.053s reduction on the test set, corresponding to a 1250

1251

1252

1253

1254

1255

1256

1257

29.11% improvement. Similarly, layer 5 achieves an improvement in test time of 23.22%, and layer 6 shows the largest test time speedup of 32.46%. However, in other cases, such as layer 4 in the training set (+0.288s, -14.53%) and layer 10 (+0.228s, -5.68%), quantization increases execution time. The relative differences on the validation set also vary widely, with notable slowdowns at layers 7 (-25.74%) and 9 (-39.61%). These inconsistencies highlight that run-time performance does not always benefit from dynamic quantization, and the effectiveness likely depends on the specific computation pattern and how well the underlying hardware supports quantized operations.

Table 21: Elapsed time (in seconds) for ACR-GNN with and without dynamic post-training quantization (dPTQ). PPI Benchmark

| Layer   | Train    | Train   | Validation   | Validation   | Test     | Test   |
|---------|----------|---------|--------------|--------------|----------|--------|
| Layer   | Original | dPTQ    | Original     | dPTQ         | Original | dPTQ   |
| 1       | 0.780    | 0.858   | 0.102        | 0.112        | 0.077    | 0.094  |
| 2       | 0.986    | 0.966   | 0.130        | 0.131        | 0.109    | 0.107  |
| 3       | 1.138    | 1.161   | 0.157        | 0.159        | 0.149    | 0.140  |
| 4       | 1.371    | 1.366   | 0.159        | 0.204        | 0.156    | 0.160  |
| 5       | 1.645    | 1.682   | 0.201        | 0.211        | 0.173    | 0.199  |
| 6       | 1.833    | 1.766   | 0.242        | 0.256        | 0.188    | 0.205  |
| 7       | 2.166    | 2.156   | 0.282        | 0.261        | 0.239    | 0.242  |
| 8       | 2.355    | 2.534   | 0.317        | 0.300        | 0.241    | 0.283  |
| 9       | 2.539    | 2.652   | 0.337        | 0.349        | 0.302    | 0.292  |
| 10      | 2.842    | 3.122   | 0.386        | 0.461        | 0.326    | 0.348  |

Table 21 reports the elapsed time (in seconds) required to perform inference on the training, validation, 1258 and test sets using the ACR-GNN model with ReLU activation, both in its original form and after 1259 applying dynamic post-training quantization (dPTQ). The measurements reflect the running time of 1260 the trained models only; the time required for model training is not included in these results. The 1261 values indicate that inference time generally increases with the number of layers, as expected, and 1262 the impact of quantization on runtime varies across depths. In some cases, dPTQ slightly reduces 1263 inference time (e.g., Layer 6, Train), while in others it introduces moderate overhead, particularly for 1264 deeper models. 1265

1266

1267

1268

The experiments were run on a Samsung Galaxy Book4 laptop with an Intel Core i7-150U processor, 16 GB RAM, and 1 TB SSD storage. Additional experiments were conducted using Kaggle's cloud platform with an NVIDIA Tesla P100 GPU (16 GB RAM).

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

## H Description logics with global and local cardinality constraints

The Description Logic ALCSCC ++ [2] extends the basic Description Logic ALC [3] with concepts that capture cardinality and set constraints expressed in the quantifier-free fragment of Boolean Algebra with Presburger Arithmetic (QFBAPA) [20].

We assume that we have a set of set variables and a set of integer constants .

AQFBAPA formula is a Boolean combination ( ∧ , ∨ , ¬ ) of set constraints and cardinality constraints .

A set term is a Boolean combination ( ∪ , ∩ , · ) of set variables , and set constants U , and ∅ . If S is a set term, then its cardinality | S | is an arithmetic expressions . Integer constants are also arithmetic expressions. If T 1 and T 2 are arithmetic expressions, so is T 1 + T 2 . If T is an arithmetic expression and c is an integer constant, then c · T is an arithmetic expression.

Given two set terms B 1 and B 2 , the expressions B 1 ⊆ B 2 and B 1 = B 2 are set constraints . Given two arithmetic expressions T 1 and T 2 , the expressions T 1 &lt; T 2 and T 1 = T 2 are cardinality constraints . Given an integer constant c and an arithmetic expression T , the expression c dvd T is a cardinality constraint .

A substitution σ assigns ∅ to the set constant ∅ , a finite set σ ( U ) to the set constant U , and a subset 1283 of σ ( U ) to every set variable. A substitution is first extended to set terms by applying the standard 1284

set-theoretic semantics of the Boolean operations. It is further extended to map arithmetic expressions 1285 to integers, in such that way that every integer constant c is mapped to c , for every set term B , the 1286 arithmetic expression | B | is mapped to the cardinality of the set σ ( B ) , and the standard semantics for 1287 addition and multiplication is applied. 1288

1289

1290

1291

1292

The substitution σ (QFBAPA) satisfies the set constraint B 1 ⊆ B 2 if σ ( B 1 ) ⊆ σ ( B 2 ) , the set constraint B 1 = B 2 if σ ( B 1 ) = σ ( B 2 ) , the cardinality constraint T 1 &lt; T 2 if σ ( T 1 ) &lt; σ ( T 2 ) , the cardinality constraint T 1 = T 2 if σ ( T 1 ) = σ ( T 2 ) , and the cardinality constraint c dvd T if c divides σ ( T ) .

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

We can now define the syntax of ALCSCC ++ concept descriptions and their semantics. Let N C be a set of concept names, and N R be a set of role names, such that N C ∩ N R = ∅ . Every A ∈ N C is a concept description of ALCSCC ++ . Moreover, if C , C 1 , C 2 , . . . are concept descriptions of ALCSCC ++ , then so are: C 1 ⊓ C 2 , C 1 ⊔ C 2 , ¬ C , and sat ( χ ) , where χ is a set or cardinality QFBAPA constraint, with elements of N R and concept descriptions C 1 , C 2 , . . . used in place of set variables.

A finite interpretation is a pair I = (∆ I , · I ) , where ∆ I is a finite non-empty set of individuals, and · I is a function such that: every A ∈ N C is mapped to A I ⊆ ∆ I , and every R ∈ N R is mapped to R I ⊆ ∆ I × ∆ I . Given an element of d ∈ ∆ I , we define R I ( d ) = { d ′ | ( d, d ′ ) ∈ R I } .

The semantics of the language of ALCSCC ++ makes use QFBAPA substitutions to interpret QFBAPA constraints in terms of ALCSCC ++ finite interpretations. Given an element d ∈ ∆ I , we can define the substitution σ I d in such a way that: σ I d ( U ) = ∆ I , σ I d ( ∅ ) = ∅ , and A ∈ N C and R ∈ N R are considered QFBAPA set variables and substituted as σ I d ( A ) = A I , and σ I d ( R ) = R I ( d ) .

The finite interpretation I and the QFBAPA substitutions σ I d are mutually extended to complex expressions such that: σ I d ( C 1 ⊓ C 2 ) = ( C 1 ⊓ C 2 ) I = C I 1 ∩ C I 2 ; σ I d ( C 1 ⊔ C 2 ) = ( C 1 ⊔ C 2 ) I = C I 1 ∪ C I 2 ; σ I d ( ¬ C ) = ( ¬ C ) I = ∆ I \ C I ; and σ I d ( sat ( χ )) = ( sat ( χ )) I = { d ′ ∈ ∆ I | σ I d ′ (QFBAPA) satisfies χ } .

̸

Definition 24. The ALCSCC ++ concept description C is satisfiable if there is a finite interpretation I such that C I = ∅ .

Theorem 25 ([2]) . The problem of deciding whether an ALCSCC ++ concept description is satisfiable is NEXPTIME-complete.

## I ALCQ and T C Boxes consistency

ALCQ is the Description Logic adding qualified number restrictions to the standard Description Logic ALC , analogously to how Graded Modal Logic extends standard Modal Logic with graded modalities.

Let N C and N R be two non-intersecting sets of concept names, and role names respecively. A concept name A ∈ N C is an ALCQ concept expressions of ALCQ . If C is an ALCQ concept expression, so is ¬ C . If C 1 and C 2 are ALCQ concept expressions, then so is C 1 ⊓ C 2 . If C is an ALCQ concept expression, R ∈ N R , and n ∈ ℕ , then ≥ n R.C is an ALCQ concept expression.

A cardinality restriction of ALCQ is is an expression of the form ( ≥ n C ) or ( ≤ n C ) , where C an ALCQ concept expression and n ∈ ℕ .

An ALCQ -T C Box is a finite set of cardinality restrictions.

An interpretation is a pair I = (∆ I , · I ) , where ∆ I is a non-empty set of individuals, and · I is a function such that: every A ∈ N C is mapped to A I ⊆ ∆ I , and every R ∈ N R is mapped to R I ⊆ ∆ I × ∆ I . Given an element of d ∈ ∆ I , we define R I ( d ) = { d ′ | ( d, d ′ ) ∈ R I } . An interpretation I is extended to complex concept descriptions as follows: ( ¬ C ) I = ∆ I \ C I ; ( C 1 ⊓ C 2 ) I = C I 1 ∩ C I 2 ; and ( ≥ n R.C ) I = { d | | R I ( d ) ∩ C I | ≥ n } .

An interpretation I satisfies the cardinality restriction ( ≥ n C ) iff | C I | ≥ n and it satisfies the cardinality restriction ( ≤ n C ) iff | C I | ≤ n . A T C Box TC is consistent if there exists an interpretation that satisfies all the cardinality restrictions in TC .

Theorem 26 ([36]) . Deciding the consistency of ALCQ -T C Boxes is NEXPTIME-hard.

The proof can be slightly adapted to show that the result holds even when there is only one role.

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

Figure 8: Encoding a torus of exponential size with an ALCQ -T C Box with one role.

<!-- image -->

Some abbreviations are useful. For every pair of concepts C and D , C → D stands for ¬ C ⊔ D . For every concept C , role R , and non-negative integer n , we define: ( ≤ n R.C ) := ¬ ( ≥ ( n +1) R.C ) , ( ∀ R.C ) := ( ≤ 0 R. ¬ C ) , ( ∀ C ) := ( ≤ 0 ¬ C ) , (= n R.C ) := ( ≥ n R.C ) ⊓ ( ≤ n R.C ) , and (= n C ) := ( ≥ n C ) ⊓ ( ≤ n C ) .

Theorem 27. Deciding the consistency of ALCQ -T C Boxes is NEXPTIME-hard even if | N R | = 1 .

Proof. Let next be the unique role in N R . We use the atomic concepts N to denote an individual 'on the way north' and E to denote an individual 'on the way east'. See Figure 8.

For every n ∈ ℕ , we define the following ALCQ -T C Box.

<!-- formula-not-decoded -->

such that the concepts C (0 , 0) , C (2 n -1 , 2 n -1) are defined like in [36, Figure 3], and so are the concepts D north and D east , except that for every concept C , ∀ east.C now stands for ∀ next. ( E →∀ next.C ) and ∀ north.C now stands for ∀ next. ( N →∀ next.C ) .

The problem of deciding whether a domino system D = ( D,V,H ) , given an initial condition w 0 . . . w n -1 , can tile a torus of exponential size can be reduced to the problem of consistency of ALCQ -T C Boxes, checking the consistency of T ( n, D , w ) = T n ∪ T D ∪ T w , where T n is as above, T D encodes the domino system, and T w encodes the initial condition as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The rest of the proof remains unchanged. 1349

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

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We introduce a logical language for reasoning about quantized graph neural networks (GNNs) with Global Readout in Section 3. We then prove that verifying quantized GNNs with Global Readout is NEXPTIME-complete in Section 4 and Section 5. We also experimentally show the relevance of quantization in the context of ACR-GNNs in Section 7.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Limitations are addressed in Section 8.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: All the theorems, formulas, and proofs in the paper are numbered and crossreferenced. The assumptions are stated and the full proofs are present in the appendix, with sketches of proofs in the main text.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The authors provide the replication package with code and description of the files.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provided clear instructions on how to access the data and reproduce the experimental results in the supplemental materials, including required scripts and environment setup.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The experimental setting is described in sufficient detail in the main body of the paper, including datasets, tools, parameters, and evaluation metrics, to support understanding and reproducibility of the results.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The authors provided a code in the supplementary materials that generates the detailed summary statistics across configurations for FOC 2 . The method for computing these plots is included in the code.

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

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The experiments were run on a Samsung Galaxy Book4 laptop with an Intel Core i7-150U processor, 16 GB RAM, and 1 TB SSD storage. Additional experiments were conducted using Kaggle's cloud platform with an NVIDIA Tesla P100 GPU (16 GB RAM). The runtime for the synthetic dataset experiments is reported in Table 21, and full instructions for reproducing the results are provided in the supplementary materials.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Broader impacts are addressed in the introduction, explaining that the black-box nature of NN is a major issue for their adoption, morally and legally, with the enforcement of regulatory policies like the EU AI Act. NN that can be formally verified solve this. We do not think that this work may have negative societal impacts.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: For the reference ACR-GNN, we used the original paper [4] and the official implementation available at [5]. The code is distributed under the MIT License, and we have properly credited the authors and complied with the license terms.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We are releasing new code introduced in this work under the MIT License. The repository includes a README with setup instructions, usage examples, and description of each module, enabling other researchers to reproduce our results.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as

well as details about compensation (if any)?

Answer: [NA]

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

Justification: The paper does not involve crowdsourcing nor research with human subjects.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

1464

Justification: The core method development in this research does not involve LLMs as any 1465 important, original, or non-standard components. 1466