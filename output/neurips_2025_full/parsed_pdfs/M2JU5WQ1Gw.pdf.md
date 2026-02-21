## Uncovering Critical Sets of Deep Neural Networks via Sample-Independent Critical Lifting

## Anonymous Author(s)

Affiliation Address email

## Abstract

This paper investigates the sample dependence of critical points for neural networks. We introduce a sample-independent critical lifting operator that associates a parameter of one network with a set of parameters of another, thus defining sampledependent and sample-independent lifted critical points. We then show by example that previously studied critical embeddings do not capture all sample-independent lifted critical points. Finally, we demonstrate the existence of sample-dependent lifted critical points for sufficiently large sample sizes and prove that saddles appear among them.

## 1 Introduction 9

Neural networks have achieved remarkable success in a wide range of applications, but the under10 standing of their performance is still elusive. Theoretical studies are thus made to uncover such 11 mysteries (Sun et al., 2020). One major focus is the analysis of the loss landscape. This line of 12 study is challenging due to the complicated, various kinds of network structure and loss function, and 13 importantly, its dependence on data samples. 14

15

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

Recent research has increasingly focused on how critical points in the loss landscape depend on the training data. A notable direction in this line of work involves the Embedding Principle (Zhang et al., 2022, 2021; Bai et al., 2024), which is motivated by the following question: given the critical points of a neural network, what can be inferred about the critical points of another network, without knowing the specific training samples? Critical embedding operators between neural networks of different widths, such as splitting embeddings, null embeddings, and more general compatible embeddings, have been proposed and studied in Zhang et al. (2022, 2021). Critical lifting operators in depth between networks of varying depths have been proposed and studied in Bai et al. (2024). However, the full extent to which these operators explain sample (in)dependence remains unclear. Parallel to this, many studies have investigated the behavior of critical points when specific information about the samples is known. For instance, Cooper (2021) relates the dimensionality of the global minima manifold to the number of samples in a generic setting, while ref. Zhang et al. (2023) explores a teacher-student setup and reveals a hierarchical, branch-wise structure of the loss landscape near global minima that varies with sample size.

29

30

31

32

33

34

In this paper, we advance the understanding of sample dependence of critical points by focusing on neural networks of different widths that represent the same output function. Our main contributions are as follows:

- (a) We introduce a sample-independent critical lifting operator, which maps parameters from a narrower network to a set of parameters in a wider network, preserving both the output function and criticality regardless of the training samples.

35

36

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

- (b) We demonstrate that not all sample-independent lifted critical points arise from previously studied embedding operators, thus highlighting a broader structure beyond existing frameworks Zhang et al. (2022, 2021).
- (c) We identify a class of output-preserving critical sets that, for sufficiently large sample sizes, generally contain sample-dependent critical points. These sets consist entirely of saddle points for one-hidden-layer networks and contains sample-dependent saddles for multi-layer networks.

## 2 Related Works

Embedding Principle. The Embedding Principle (EP) was first observed for two neural networks of different widths, stating that 'the loss landscape of any network 'contains' all critical points of all narrower networks' (Zhang et al., 2021). In refs. Zhang et al. (2021, 2022), specific critical embedding operators have been proposed and studied. These are linear operators mapping parameters of a narrower network to a wider one which preserve output function and criticality - the image of a critical point is always a critical point. Earlier works also observe the similar phenomenon for one hidden layer neural networks (Fukumizu and ichi Amari, 2000; Fukumizu et al., 2019). More recently, EP for two neural networks of different depths was observed (Bai et al., 2024). The paper introduces critical lifting operators associating a parameter of a shallower network to a set of parameters of a deeper one, where output function and criticality are preserved. In our work, we use the same idea to define sample-independent critical lifting operators, but we focus on two neural networks of different widths and show that not all sample-independent lifted critical points arise from known embedding operators.

Sample dependence of critical points. Attempts have been made to explain how the choice of samples affects the geometry of loss landscape. Many works focus on global minima. In Cooper (2021), it is shown that for generic samples, the global minima is a manifold whose codimension equals the sample size. Ref. Simsek et al. (2021) observes that under the teacher-student setting, part of the global minima of neural networks persist as samples change. In Zhang et al. (2023) this is further emphasized, and it studies how the other (sample-dependent) global minima varies 'gradually vanish' as sample size increases, as well as how it affects the behavior of gradient dynamics nearby. Other works, such as Simsek et al. (2023), study critical points assuming samples have specific distributions. Our work applies to both global and non-global critical points, and we emphasize sample-dependent lifted critical points for sufficiently large sample size, thus complementing the previous studies.

Analysis of saddles. It has been shown that gradient dynamics almost always avoid saddles (Lee et al., 2017). Thus, it is essential to discover saddles in loss landscape of neural networks. Refs. Fukumizu and ichi Amari (2000); Fukumizu et al. (2019); Simsek et al. (2021); Zhang et al. (2022, 2021) showed that embedding local minima of a narrower network to a wider one tends to produce saddles. Additionally, research by Venturi et al. and Li et al. revealed that, when the network is heavily overparameterized, saddles not only exist but in fact there are no spurious valleys. Similar patterns have been observed in deep linear networks (Nguyen and Hein, 2017; Nguyen, 2019; Kawaguchi, 2016). In this paper, we show under mild assumptions on the training set-up that for one hidden layer networks, all sample-independent lifted critical points are saddles, and sample-dependent lifted saddles exist for multi-layer networks.

## 3 Preliminaries

Let N := { 1 , 2 , 3 , ... } . Given N ∈ N , denote by R N the (real) Euclidean space of dimension N . Given Lebesgue measurable subsets E 2 ⊆ E 1 ⊆ R N , the measure of E 2 in E 1 refers to the induced Lebesgue measure on E 1 . For example, we would say R × { (0 , 0) } ⊆ R 3 has zero measure in R 2 ×{ 0 } ⊆ R 3 . Then we define our notations and assumptions for neural networks and loss functions as follows.

## 3.1 Fully Connected Neural Networks

For simplicity, we only discuss fully-connected neural networks without bias terms . We refer to this network architecture whenever we mention a neural network. An L hidden layer neural network with

parameter size N , input dimension d and output dimension D is denoted by H : R N × R d → R D . It 86 is defined iteratively as follows. First, we define the zero-th layer (input layer) as the identity function, 87 with a redundant parameter θ (0) : 88

<!-- formula-not-decoded -->

Second, we choose an activation σ : R → R . Then, for every l ∈ { 1 , ..., L } , let m l denote the number of neurons at the l -th layer. Define the l -th layer neurons by

<!-- formula-not-decoded -->

̸

where m l is the width of H ( l ) , H ( l ) k l is the k l -th component of H ( l ) , and θ ( l ) := ( ( w ( l ) k l ) m l k l =1 , θ ( l -1) ) , each w ( l ) k l being a vector in R m l -1 . Note that with our notation, each H ( l ) k l is independent of w ( l ) k for all k = k l . Finally, define H ( θ, x ) = [ a j · H ( L ) ( θ ( L ) , x )] D j =1 as the whole neural network, where θ := ( ( a j ) D j =1 , θ ( L ) ) .

Assumption 3.1. Assume that the activation σ : R → R is a non-polynomial analytic function.

This assumption takes into consideration the commonly used activations such as tanh ( 1 -e -x 1+ e -x ), sigmoid ( 1 1+ e -x ), swish ( x 1+ e -x ), Gaussian ( e -ax 2 ), etc. Moreover, it is easy to see that when σ is analytic, the neurons { H ( l ) } L l =1 are all analytic and thus so is the whole network H .

Definition 3.1 (wider/narrower neural network) . Given two L hidden layer neural networks H 1 , H 2 both with input dimension d , output dimension D , and the hidden layer widths { m l } L l =1 , { m ′ l } L l =1 , respectively. We say H 2 is a wider network than H 1 , or H 1 a narrower network than H 2 , if m l ≤ m ′ l for all 1 ≤ l ≤ L .

## 3.2 Loss Function

Denote the set of samples as { ( x i , y i ) n i =1 } , where ( x i ) n i =1 ∈ R nd are sample inputs and ( y i ) n i =1 ∈ R nD are sample outputs. Given ℓ : R D × R D → [0 , ∞ ) , we define the loss function (for neural networks with input dimension d and output dimension D ) as

<!-- formula-not-decoded -->

In this paper, we will often deal with neural networks of different widths. As a slight abuse of notation, we shall use R for the loss function (corresponding to fixed samples ( x i , y i ) n i =1 ) for all neural networks with the same input and output dimensions. Also note that we shall write R S when emphasizing the samples S = { ( x i , y i ) n i =1 } of R .

Assumption 3.2. We consider analytic ℓ . For each 1 ≤ j ≤ D , let ∂ j ℓ denote the j -th partial derivative for its first entry. We assume that ℓ ( p, q ) = 0 if and only if p = q , and ∂ p ℓ ( p, q ) = 0 if and only if p = q . Here ∂ p ℓ ( p, q ) = [ ∂ j ℓ ( p, q )] D j =1 is the gradient of ℓ with respect to its first entry.

Remark 3.1. A common example is ℓ ( p, q ) = | p -q | 2 . In this case, the loss function is the one used in regression: R ( θ ) = ∑ n i =1 | H ( θ, x i ) -y i | 2 .

## 4 Sample Independent and Dependent Lifted Critical Points

Definition 4.1 (sample-independent critical lifting) . Given two fully-connected neural networks H 1 , H 2 . Denote their parameter spaces by Θ 1 , Θ 2 , respectively. For each θ 1 ∈ Θ 1 let S ( θ 1 ) be the collection of samples for which θ 1 is a critical point:

<!-- formula-not-decoded -->

Denote by C θ 1 ,S the set of output and criticality preserving parameters of H 2 :

<!-- formula-not-decoded -->

Define a sample-independent critical lifting operator as a map τ from Θ 1 to the power set of Θ 2 by

<!-- formula-not-decoded -->

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

Definition 4.2 (sample-dependent/independent lifted critical points) . Given two fully-connected neural networks H 1 , H 2 . Given θ 1 and S ∈ S ( θ 1 ) as in Definition 4.1. We say a parameter θ 2 ∈ C θ 1 ,S is a sample-independent lifted critical point (from θ 1 ) if θ 2 ∈ τ ( θ 1 ) = ⋂ S ∈S ( θ 1 ) C θ 1 ,S . Otherwise, we say θ 2 is a sample-dependent lifted critical point.

Remark 4.1. To make the sample-independent critical lifting operator non-trivial we should require that H 1 , H 2 have the same input and output dimensions - otherwise τ ( θ 1 ) = ∅ for all θ 1 ∈ Θ 1 . In this work, we further consider the case in which H 1 , H 2 have the same activation, same depth, but one is wider/narrower than the other.

## 4.1 Sample Independent Lifted Critical Points

Recall that a critical embedding is an affine linear map from the parameter space of a narrower neural network to that of a wider one, which preserves output, representation and criticality (Zhang et al., 2022). In particular, for any samples given, the image of a critical point is always a critical point. So by definition we have the following result summarized from (Zhang et al., 2022, 2021).

Proposition 4.1.1 (critical embeddings produce sample-independent lifted critical points) . The parameters produced by critical embedding operators are sample-independent lifted critical points.

In refs. Zhang et al. (2022, 2021) some specific critical embedding operators are proposed and studied - the splitting embedding, null-embedding and general compatible embedding. Unfortunately, these embedding operators are not enough to produce all sample-independent lifted critical points for deep neural networks. This follows from the following example:

Example. Consider a three hidden layer neural network with d ( d is arbitrary) dimensional input, one dimensional output and hidden layer widths { m 1 , m 2 , m 3 } :

<!-- formula-not-decoded -->

Given two such networks H 1 , H 2 with hidden layer widths { m 1 , m 2 , m 3 } and { m 1 , m 2 , m 3 +1 } , respectively. Define

<!-- formula-not-decoded -->

as subsets in the parameter spaces of H 1 , H 2 , respectively. Then the image of E narr under the splitting embedding, null-embedding and general compatible embedding (altogether) is a proper subset of E wide . Intuitively, this is because these operators 'assign' a relationship between the weights on the added second layer neuron to the parameter in E narr. On the other hand, it is easy to see that all parameters in E narr and E wide yield the same, constant zero output function, and are critical points, for arbitrary samples ( x i , y i ) n i =1 , n ∈ N . Therefore, the previously studied embedding operators do not produce all sample-independent lifted critical points when mapping E narr to E wide. In particular, whatever sample we choose, we cannot avoid the sample-independent lifted critical points which are not produced by these embedding operators. See Proposition A.2.1 for details of a proof of the example.

Remark 4.2. The example can be generalized to L ≥ 3 hidden layer neural networks.

## 4.2 Sample Dependent Lifted Critical Points

We now turn our focus to sample-dependent lifted critical points. Starting with the one-hidden-layer, one dimensional output case, we show that under mild assumptions on activation and loss function, sample-dependent lifted critical points are saddles. These results extend to deeper architectures, where we identify a set of output-preserving parameters containing sample-dependent critical point and sample-dependent saddles. For both results, we highlight the requirement on sample size for these critical points to exist.

We start with the one hidden layer, one dimensional output case. For an m -neuron-wide one hidden layer neural network, we write it as H ( θ, x ) = ∑ m k =1 a k σ ( w k · x ) for simplicity, where θ = ( a k , w k ) m k =1 .

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

Proposition 4.2.1 (saddles, one hidden layer) . Given samples ( x i , y i ) n i =1 such that x i = 0 for all i and x i ± x j = 0 for 1 ≤ i &lt; j ≤ n . Given integers m,m ′ such that m&lt;m ′ . For any critical point θ narr = ( a k , w k ) m k =1 of the loss function corresponding to the samples such that R ( θ narr ) = 0 , the set of ( w ′ k ) m ′ k = m +1 ∈ R ( m ′ -m ) d of weights making the parameter

̸

<!-- formula-not-decoded -->

a critical point for the loss function has zero measure in R ( m ′ -m ) d . Furthermore, any such critical point is a saddle.

Remark 4.3. Due to symmetry of the network structure, the results hold under permutation of the entries of θ wide .

Proof. We show that for a.e. w ′ m ′ ∈ R d , the partial derivative ∂R ∂a ′ m ′ is non-zero, thus proving the first part of the result. The key to showing such a critical point must be a saddle is that any θ wide of the form (2) preserves output function, namely, we have H ( θ narr , x ) = H ( θ wide , x ) for all x . See Proposition A.2.2 for more details.

Then we show that there are sample-dependent lifted critical points when the sample size is larger than the parameter size of the narrower network.

Theorem 4.2.1 (sample-dependent lifted critical points, one hidden layer) . Assume that ℓ : R × R → R satisfies: the range of ∂ p ℓ ( p, · ) contains an open interval around 0 . Given integers m,m ′ ∈ N such that m&lt;m ′ . Fix θ narr = ( a k , w k ) m k =1 . When sample size n &gt; 1 + ( d +1) m , there are sampledependent lifted critical points θ wide from θ narr of the form (2). Furthermore, when n &gt; 2 + ( d +1) m there are sample-dependent lifted saddles of the form (2).

Remark 4.4. It is clear that for any even integer s , ℓ ( x, y ) = ( p -q ) s satisfies the hypothesis on ℓ . In fact, by Lemma A.1.4, this holds for all ℓ such that ℓ ( p, q ) = ℓ ( p -q, 0) . We also show in Lemma A.1.5 that the binary cross-entropy loss of distribution p relative to distribution q , given by ℓ ( p, q ) = q log p +(1 -q ) log(1 -p ) , satisfies this hypothesis.

̸

Proof. Specifically, we prove that for any ( x i ) n i =1 ∈ R nd with x i = 0 for all i and x i ± x j = 0 for 1 ≤ i &lt; j ≤ n , and for a.e. w ′ ∈ R d , there are sample outputs ( y i ) n i =1 , ( y ′ i ) n i =1 such that

```
θ wide = ( a 1 , w 1 , ..., a m , w m , 0 , w ′ , ..., 0 , w ′ )
```

is a critical point for the loss function corresponding to ( x i , y ′ i ) n i =1 , but not so to ( x i , y i ) n i =1 . For N ≥ 2 + ( d +1) m , we can choose ( y ′ i ) n i =1 so that not all ℓ ( H ( θ wide , x i ) , y i ) 's vanish.

̸

Remark 4.5. Note that for one hidden layer neural networks every sample-dependent lifted critical point either achieves zero loss, or is a saddle. For simplicity, assume that the activation function is an even or odd function. Given a critical point θ narr = ( a k , w k ) m k =1 with R ( θ narr ) = 0 . Consider any critical point θ wide = ( a ′ k , w ′ k ) m ′ k =1 representing the same output function as θ narr . By linear independence of neurons (see Lemma A.1.1), a ′ ¯ k = 0 whenever w ′ ¯ k / ∈ { w k , -w k } m k =1 . On the other hand, if w ′ ¯ k ∈ { w k , -w k } m k =1 then θ wide is a sample-independent lifted critical point. Therefore, up to permutation of the entries, a sample-independent lifted critical point from θ narr takes the form (2), thus by Proposition 4.2.1 it must be a saddle. Similar argument works for activations with no parity.

Now we generalize the results to multi-layer neural networks whose output dimensions are arbitrary.

̸

Proposition 4.2.2 (saddles, general case) . Given samples ( x i , y i ) n i =1 with x i = 0 for all i and x i ± x j = 0 for 1 ≤ i &lt; j ≤ n . Given integers { m l } L l =1 , { m ′ l } L l =1 such that m l &lt; m ′ l for every 1 ≤ l ≤ L . Consider two L hidden layer neural networks with input dimension d , hidden layer widths { m l } L l =1 , { m ′ l } L l =1 , and output dimension D . Denote their parameters by θ narr , θ wide, respectively. Let θ narr be a critical point of the loss function corresponding to the samples ( x i , y i ) n i =1 , such that R ( θ narr ) = 0 . Denote the following sets:

̸

```
E = { θ wide = (( a ′ j ) D j =1 , θ ( L ) wide ) : H ( θ wide , · ) = H ( θ narr , · ) , a ′ j = ( a j 1 , ..., a jm L , 0 , ..., 0) } ; E ∗ = { θ wide ∈ E : ∇ R ( θ wide ) = 0 } .
```

̸

Namely, E is a set of parameters preserving output function, E ∗ is the set of parameters in E also 209 preserving criticality. Then E ∗ = E . Furthermore, E ∗ contains saddles. 210

̸

̸

̸

̸

Remark 4.6. When D = L = 1 , we recover the one hidden layer, one dimensional output case. 211

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

Proof. The extra neurons at each layer of the wider network allows us to freely choose the corresponding parameters so that we have some output-preserving θ wide with H ( L -1) ( θ wide , x i ) = 0 for all i and H ( L -1) ( θ ( L -1) wide , x i ) ± H ( L -1) ( θ ( L -1) wide , x j ) = 0 for 1 ≤ i &lt; j ≤ n . Since

̸

<!-- formula-not-decoded -->

This reduces to the proof of Proposition 4.2.1. See Proposition A.2.4 for more details.

Similarly, sample-dependent lifted critical points exist for multi-layer neural networks. The proof of the theorem below follows the same idea as that of Theorem 4.2.1.

Theorem 4.2.2 (sample-dependent lifted critical points, general case) . Assume that ℓ : R D × R D → R satisfies: the range of ∂ p ℓ ( p, · ) contains a neighborhood around 0 ∈ R D . Consider two L hidden layer neural networks with the same assumptions as in Proposition 4.2.2. Denote their parameters by θ narr , θ wide, respectively. Denote the parameter size of the narrower network by N . Fix θ narr. Then there are sample-dependent lifted critical points when sample size n ≥ 1+ N D . Furthermore, there are

1+

D

+

∑

L

l

=2

m

l

(

m

D

sample-dependent lifted saddles when

n

≥

.

Remark 4.7. When D = L = 1 , we recover the one hidden layer, one dimensional output case. Also note that commonly seen losses such as ℓ ( p, q ) = ( p -q ) s , p, q ∈ R D for any even number s satisfy the hypothesis on ℓ .

## 5 Illustration

In this section we illustrate our results in Section 4 through a toy example. In the example, a specific critical point of a one neuron tanh network H (( a, w ) , x ) = a tanh( wx ) is lifted to a set of parameters of a two neuron tanh network H (( a 1 , w 1 , a 2 , w 2 ) , x ) = a 1 tanh( w 1 x ) + a 2 tanh( w 2 x ) , where a, w, a k , w k , x are real numbers. Specifically, we fix θ 1 = (1 , ¯ w ) with ¯ w = 1 . 0258 , sample size n = 4 , sample inputs ( x 1 , x 2 , x 3 , x 4 ) = (1 / 4 , 1 , 4 , 16) and vary y i 's. We use ℓ : R × R → R , ℓ ( p, q ) = ( p -q ) 2 . So

<!-- formula-not-decoded -->

To make θ 1 a critical point, ( y i ) 4 i =1 should solve the linear system 234

<!-- formula-not-decoded -->

235

236

237

Let ε i := tanh( ¯ wx i ) -y i for 1 ≤ i ≤ 4 . Clearly, the solution set for ( ε i ) 4 i =1 is a two dimensional subspace in R 4 , and varying ( y i ) 4 i =1 is equivalent to varying ( ε i ) 4 i =1 . Numerically, an approximate solution curve for ( ε i ) 4 i =1 = ( ε i ( t )) 4 i =1 is given by

<!-- formula-not-decoded -->

First, we show that the image of θ 1 under splitting embeddings remains critical, and is independent 238 of the samples. Note that the set of points produced by splitting embeddings is the line E := 239 { ( δ, ¯ w, 1 -δ, ¯ w ) : δ ∈ R } and the partial derivatives of the loss function satisfy 240

<!-- formula-not-decoded -->

′

l

-

1

-

m

-

l

1

)+

N

̸

Since w 1 = w 2 = ¯ w is fixed over E , we illustrate the vector field 241

<!-- formula-not-decoded -->

as ( a 1 , a 2 ) varies, for the samples we randomly choose. This is indicated in Figure 1 below. As we can see, the vector field vanishes (approximately) along the line { a 1 + a 2 = 1 } , which implies that E is critical under these samples.

Second, we consider critical points in the set E ′ := { (1 , ¯ w, 0 , w ) : w ∈ R } . According to Proposition 4.2.1, the points in E ′ are saddles. In the experiment, we fix the samples by setting ( ε i ) 4 i =1 = (1 , -0 . 5835 , 0 . 3 , -0 . 1) } and check the loss values for different ( a 2 , w 2 ) , meanwhile keeping ( a 1 , w 1 ) = (1 , ¯ w ) fixed. For these samples, there are three critical points in E ′ . As illustrated in Figure 2, the loss function takes values greater and less than R ( θ 1 ) ≈ 1 . 4405 near each of them, thus showing that they are all saddles.

<!-- image -->

al a1

a1

Figure 1: Plot of the vector field ( a 1 , a 2 ) ↦→ ( ∂R ∂a 1 ( a 1 , ¯ w,a 2 , ¯ w ) , 3 a 1 ∂R ∂w 1 ( a 1 , ¯ w,a 2 , ¯ w ) ) for ( a 1 , a 2 ) ∈ (0 . 1 , 0 . 9) 2 with respect to ( ε i ( -4)) 4 i =1 (left), ( ε i (0)) 4 i =1 (middle) and ( ε i (3)) 4 i =1 . In all three figures, the vector field vanishes approximately along the line { a 1 + a 2 = 1 } , indicating that the parameters produced by splitting embeddings are sample-independent saddles.

<!-- image -->

w,2

w,2

(0t.4 pegx) [°e

Figure 2: Contour plot of the loss function along the ( w 2 , a 2 ) -plane with respect to ( ε i (0)) 4 i =1 . The points, marked in red, are approximately (0 , 0) (left), (0 . 1236 , 0) (middle) and (1 . 0258 , 0) (right). They correspond to the critical points (1 , ¯ w, 0 , 0) , (1 , ¯ w, 0 , 0 . 1236) , (1 , ¯ w, 0 , 1 . 0258) in E ′ , respectively. From the level curves we can see that these three points are all saddles. Note that in the rightmost figure w 2 -axis is scaled by 10 for illustration purpose.

Finally, we show the existence of sample-dependent critical points in E ′ . We illustrate this by plotting 251 the zero set of the function 252

<!-- formula-not-decoded -->

As shown in the proof of Proposition A.2.2, a parameter of the form (1 , ¯ w, 0 , w ) is a critical point 253 for the loss corresponding to ( ε i ( t )) 4 i =1 if and only if φ ( t, w ) = 0 . In Figure 3 we can see that for 254 ( t, w ) ∈ ( -0 . 5 , 0 . 5) × ( -0 . 8 , 0 . 8) , the zero set of φ has two curves; the value of w on the blue curve 255

242

243

244

245

246

247

248

249

250

varies as t varies, which implies that sample-dependent lifted critical points of the form (1 , ¯ w, 0 , w ) 256 exist. 257

<!-- image -->

t

Figure 3: The zero set of φ ( t ) = ∑ 4 i =1 ε i ( t )tanh( wx i ) for ( t, w ) ∈ ( -0 . 5 , 0 . 5) × ( -0 . 8 , 0 . 8) . The blue curve minus the origin, which arises when t ranges approximately from -0 . 05 to 0 . 3 , is locally the graph of a non-constant function in t . This indicates that there is a sample-dependent lifted critical point for each such t . Also note that the grey curve { (0 , t ) } indicates a sample-independent lifted critical point (1 , ¯ w, 0 , 0) . It arises due to the fact that tanh(0) = 0 .

## 6 Conclusion and Discussion

In this paper, we propose the sample-independent critical lifting operator (Definition 4.1) and study the sample-independent/dependent lifted critical points. We first show by example that the previously studied critical embeddings may not produce all sample-independent lifted critical points. We then focused on sample-dependent lifted critical points, identifying a specific family of such points and proving that they are necessarily saddles when the loss is non-zero. The sample-independent critical lifting operator provides a way to study the structural aspects of loss landscape dictated purely by the network architecture. Our study of sample-independent critical points reveals the limitation of previously studied embedding operators, suggesting a more delicate relationship between neural networks of different widths. Our study of sample-dependent critical points provides insights into how samples affect the loss landscape.

The paper raises as many questions as the information it provides. First, for sample-independent critical points, we are unclear if all of them are produced by critical embedding operators (not limited to those previously studied ones). We conjecture that they fully characterize all sample-independent lifted critical points for one hidden layer neural networks. Meanwhile, it is interesting to investigate how the completeness of the characterization depends on the network architecture, e.g., choice of activation function, depth/width of network, etc.

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

Second, we do not have a clear picture about sample-dependent lifted critical points for multi-layer neural networks. Recall that we have shown that all sample-dependent critical points must be of the form (2), but a general form of these points is unclear for multi-layer networks. We expect the existence of additional sample-dependent critical points beyond what we discovered in the paper. Meanwhile, we are interested in the gradient dynamics near the sample-dependent saddles we discovered. Since they are necessarily degenerate and may not have a negative eigenvalue, previous results, e.g., those in Lee et al. (2017) cannot apply immediately.

Third, a better understanding of the sample-independent lifting operator is needed. For example, 282 our construction of sample-dependent lifted critical point requires a specific sample size threshold, 283 which naturally leads to the question whether sample-dependent lifted critical points exist when 284 we keep the sample size fixed while varying samples. More generally, one can study 'constrained 285

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

sample-independent lifting operator' concerning samples with fixed property. This would help us better understand how different aspects of data affect the loss landscape.

## References

- R. Sun, D. Li, S. Liang, T. Ding, The global landscape of neural networks, Nonconvex Optimization for Signal Processing and Machine Learning 37 (2020) 95-108.
- Y. Zhang, Y. Li, Z. Zhang, T. Luo, Z.-Q. J. Xu, Embedding principle: a hierarchical structure of loss landscape of deep neural networks, Journal of Machine Learning 1 (2022) 60-113.
- Y. Zhang, Z. Zhang, T. Luo, Z.-Q. J. Xu, Embedding principle of loss landscape of deep neural networks, NeurIPS 34 (2021) 14848-14859.
- Z. Bai, T. Luo, Z.-Q. J. Xu, Y. Zhang, Embedding principle in depth for the loss landscape analysis of deep neural networks, CSIAM Transactions on Applied Mathematics 5 (2024) 350-389.
- Y. Cooper, Global minima of overparameterized neural networks, SIAM Journal on Mathematics of Data Science 3 (2021) 676-691.
- L. Zhang, Y. Zhang, T. Luo, Structure and gradient dynamics near global minima of two-layer neural networks, arXiv:2309.00508 (2023).
- K. Fukumizu, S. ichi Amari, Local minima and plateaus in hierarchical structures of multilayer perceptrons, Neural Networks 13 (2000) 317-327.
- K. Fukumizu, S. Yamaguchi, Y. ichi Mototake, M. Tanaka, Semi-flat minima and saddle points by embedding neural networks to overparameterization, NeurIPS 32 (2019).
- B. Simsek, F. Ged, A. Jacot, F. Spadaro, C. Hongler, W. Gerstner, J. Brea, Geometry of the loss landscape in overparametrized neural networks: Symmetry and invariances, Proceedings of Machine Learning Research 139 (2021).
- B. Simsek, A. Bendjeddou, W. Gerstner, J. Brea, Should under-parameterized student networks copy or average teacher weights?, NeurIPS (2023).
- J. D. Lee, I. Panageas, G. Piliouras, M. Simchowitz, M. I. Jordan, B. Recht, First-order methods almost always avoid saddle points, arxiv:1710.07406 (2017).
- L. Venturi, A. S. Bandeira, J. Bruna, Spurious valleys in one-hidden-layer neural network optimization landscapes, Journal of Machine Learning Research 20 (2019) 1-34.
- D. Li, T. Ding, R. Sun, On the benefit of width for neural networks: Disappearance of basins, SIAM Journal on Optimization 32 (2022) 1728-1758.
- Q. Nguyen, M. Hein, The loss surface of deep and wide neural networks, ICML 70 (2017) 2603-2612.
- Q. Nguyen, On connected sublevel sets in deep learning, ICML (2019) 4790-4799.
- K. Kawaguchi, Deep learning without poor local minima, NeurIPS (2016).
- S. G. Krantz, H. R. Parks, A Primer of Real Analytic Functions, Birkhäuser Advanced Texts Basler Lehrbücher, 2nd ed., Birkhäuser Boston, MA, 2002.
- B. Mityagin, The zero set of a real analytic function, arxiv:1512.07276 (2015).

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

## A Appendix

## A.1 Preparing Lemmas

Lemma A.1.1. Let σ : R → R be a non-polynomial analytic function. Then for any d, n ∈ N and any x 1 , ..., x n ∈ R d \ { 0 } with x i ± x j = 0 for 1 ≤ i &lt; j ≤ m , the functions { w ↦→ σ ( w · x i ) } n i =1 are linearly independent.

̸

Proof. We will actually prove a slightly stronger result shown below:

Let σ : R → R be an analytic non-polynomial activation function. Then the following results hold for any d, m ∈ N and any x 1 , ..., x n ∈ R d \ { 0 }

̸

- (a-1) When σ is the sum of a non=zero polynomial and an even/odd analytic non-polynomial, { σ ( w · x i ) } n i =1 are linearly independent if x i ± x j = 0 .
- (a-2) When σ does not have parity and does not satisfy (a-1), then { σ ( w · x i ) } n i =1 are linearly independent if and only if x i 's are distinct.

̸

- (b) When σ is an even or odd function, { σ ( w · x i ) } n i =1 are linearly independent if and only if x i ± x j = 0 for 1 ≤ i &lt; j ≤ n .

The proof below deals with these cases. For (a-1) we have

- σ is the sum of a polynomial and an even, non-polynomial analytic function. Then σ ( s ) , the s -th derivative of σ , is an even function for sufficiently large s . Since x i ± x j = 0 for 1 ≤ i &lt; j ≤ n , there is some v ∈ R d such that | x i · v | are distinct and non-zero. It follows from (b) that the (single-variable, even or odd) functions { z ↦→ ( v · x i ) s σ ( s ) (( v · x i ) z ) } n i =1 are linearly independent. Thus, { z ↦→ σ (( v · x i ) z ) } n i =1 and thus { σ ( w · x i ) } n i =1 are linearly independent.

̸

- σ is the sum of a polynomial and an odd, non-polynomial analytic function. Then σ ( s ) is an odd function for sufficiently large s . Argue in the same way as in (a-1) we show the desired result.

̸

For (a-2), note that there are infinitely many even and odd numbers s even , s odd ∈ N , such that σ ( s even ) (0) , σ ( s odd ) (0) = 0 . Then the result follows from Lemma B.5 in Simsek et al. (2021). One can also refer to other works, such as Zhang et al. (2023).

̸

Then we prove (b). First assume that σ is an even function. Then there are even, non-zero numbers { s j } ∞ j =1 such that σ ( s j ) (0) , the s j -th derivative of σ at 0 , is non-zero, for all j ∈ N . Given x 1 , ..., x n ∈ R d \ { 0 } such that x i ± x j = 0 for 1 ≤ i &lt; j ≤ n . Assume α 1 , ..., α n ∈ R makes the linear combination of these neurons, ∑ n i =1 α i σ ( w · x i ) , a constant function. Since x i ± x j = 0 for 1 ≤ i &lt; j ≤ n , there is some v ∈ R d such that | x i · v | are distinct and non-zero. Therefore,

<!-- formula-not-decoded -->

Rewriting this in power series expansion near the origin, we obtain 354

<!-- formula-not-decoded -->

̸

The power series holds for all z in a sufficiently small open interval around 0 . Thus, we must have 355 σ ( s j ) (0) ∑ n i =1 α i ( v · x i ) s j = 0 for all j ∈ N . Let i 1 ∈ { 1 , ..., n } be (the unique number) such that 356 | v · x i 1 | = max 1 ≤ i ≤ n | v · x i | . If α i 1 = 0 we would have 357

<!-- formula-not-decoded -->

̸

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

as j → ∞ . Thus, α i 1 = 0 and we need only consider the rest n -1 neurons. Therefore, by an induction on n we can see that α 1 = ... = α n = 0 . This proves the case for even activation.

Then assume that σ is an odd function. Again, let v ∈ R d be such that | v · x i | 's are distinct and non-zero. Let α 1 , ..., α n ∈ R be such that ∑ n i =1 α i σ (( v · x i ) z ) is a constant function in z . Its directional derivative along v is given by

<!-- formula-not-decoded -->

must also be constant zero. Since σ ′ is an even, analytic, non-polynomial function, our proof above shows that α i ( v · x i ) = 0 for all 1 ≤ i ≤ n , which then implies α i = 0 for all 1 ≤ i ≤ n . Therefore, the neurons are linearly independent.

Conversely, if x i -x j = 0 for some distinct i, j , then we obtain two identical neurons. If x i + x j = 0 then σ ( w · x i ) = σ ( w · x j ) for even function σ and σ ( w · x i ) + σ ( w · x j ) = 0 for odd activation σ . In either case we obtain two linearly dependent neurons. This completes the proof.

̸

Lemma A.1.2. Let N ∈ N and g : R N → R a smooth function. Let x ∗ ∈ R N be a critical point of g such that for any neighborhood U of x ∗ , there is some x ∈ U with ∇ g ( x ) = 0 and g ( x ) = g ( x ∗ ) . Then x ∗ is a saddle.

̸

<!-- formula-not-decoded -->

Proof. We will show that any neighborhood U of x ∗ contains points y 1 , y 2 with g ( y 1 ) &lt; g ( x ∗ ) &lt; g ( y 2 ) . So fix U . Choose an x ∈ U with ∇ g ( x ) = 0 and g ( x ) = g ( x ∗ ) . Since ∇ g ( x ) = 0 , the gradient flow γ : [0 , ∞ ) →∞ starting at x is not static; moreover, for some small δ &gt; 0 we have γ [0 , δ ) ⊆ U . Since the value of g is (strictly) decreasing along γ , we may choose y 1 := γ ( δ 2 ) , because

∗

Similarly, we can find some y 2 ∈ U with g ( y 2 ) &gt; g ( x ) .

Definition A.1 ((real) analytic function, rephrase of Defn. 2.2.1 in Krantz and Parks (2002)) . Let N,M ∈ N and Ω ⊆ R N be open. A function f : Ω → R is (real) analytic if for each x ∈ Ω , f can be represented by a convergent multi-variable power series in some neighborhood of x . Similarly, a function f : Ω → R M is (real) analytic if each of its components is real analytic.

Remark A.1. Let Ω and U be open, and f, g : Ω → R , h : U → Ω be analytic functions. By Proposition 2.2.2 and Proposition 2.2.8 in Krantz and Parks (2002), αf + βg, fg, f ◦ h are analytic functions, i.e., analyticity is preserved by linear combination, multiplication and composition among analytic functions. Moreover, by Proposition 2.2.3 in Krantz and Parks (2002), the partial derivatives of an analytic function are also analytic. In particular, this means when σ and ℓ are analytic, the neural network, the loss function, and the partial derivatives of the loss function are analytic.

The following lemma is of great importance for the proofs in Section A.2.

Lemma A.1.3 (Mityagin (2015)) . Let N ∈ N , Ω ⊆ R N be open and f : Ω → R be analytic. Then either f is constant zero on Ω , or f -1 (0) has zero measure in Ω .

Lemma A.1.4. Let ℓ : R 2 → R be a function satisfying Assumption 3.2. Further assume that ℓ ( p, q ) = ℓ ( p -q, 0) for all ( p, q ) ∈ R 2 . Then the range of ∂ p ℓ ( p, · ) contains an open interval around 0 for every p ∈ R .

Proof. Note that we can write ℓ ( p, q ) = u ( p -q ) for an analytic function u : R → [0 , ∞ ) , such that u is not constant zero and u ( z ) = 0 if and only if z = 0 . Since u achieves its minimum at z = 0 , there is an interval I containing 0 ∈ R such that d u d z ( z ) ≥ 0 for z ∈ (0 , ∞ ) ∩ I and d u d z ( z ) ≤ 0 for z ∈ ( -∞ , 0) ∩ I . Moreover, z = 0 is a zero of d u d z . Since u is analytic and not constant zero, the zeroes of d u d z is discrete, so by shrinking I if necessary, we would have d u d z ( z ) &gt; 0 for z ∈ (0 , ∞ ) ∩ I and d u d z ( z ) &lt; 0 for z ∈ ( -∞ ) ∩ I . This shows that the range of d u d z contains an open interval around 0 .

Now ∂ p ℓ ( p, q ) = d u d z ( p -q ) . Thus,

<!-- formula-not-decoded -->

̸

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

It follows that the range of ∂ p ℓ ( p, · ) contains an open interval around 0 .

Lemma A.1.5. Let ℓ ( p, q ) = q log p +(1 -q ) log(1 -p ) for p, q ∈ (0 , 1) . Then the range of ∂ p ℓ ( p, · ) contains an open interval around 0 for every p ∈ R .

Proof. This follows from a straightforward computation. Note that ∂ p ℓ ( p, q ) = q p -1 -q 1 -p and for each p , the derivative of q ↦→ ∂ p ℓ ( p, q ) is a strictly positive constant 1 p + 1 1 -p . Since ∂ p ℓ ( p, p ) = 0 , this implies that for q in a neighborhood I around p , ∂ p ℓ ( p, I ) contains an open interval around 0 .

## A.2 Proof of Results

Proposition A.2.1 (Example in Section 4.1) . Assume that σ (0) = 0 . For two three hidden layer neural networks, neither the splitting embedding, nor the null embedding operator, nor general compatible embedding operator produce all sample-independent lifted critical points.

Proof. Let H be a three hidden layer neural network with d ( d ∈ N is arbitrary) dimensional input, one dimensional output, and hidden width { m 1 , m 2 , m 3 } . Thus, H can be written as

<!-- formula-not-decoded -->

Fix arbitrary samples ( x i , y i ) n i =1 . Consider parameters for H of the form 414

<!-- formula-not-decoded -->

Namely, all the w (2) k 2 and w (1) k 1 's are zero vectors. Then, using σ (0) = 0 we can inductively see that 415 H (1) ( θ (1) , x ) = 0 ∈ R m 1 , H (2) ( θ (2) , x ) = 0 ∈ R m 2 and H (3) ( θ (3) , x ) = 0 ∈ R m 3 for all x . The 416 partial derivatives for R are as follows. Here ∂ p ℓ denotes the partial derivative of ℓ with respect to its 417 first entry (note that ℓ : R × R → R ). 418

<!-- formula-not-decoded -->

In other words, we show that any parameter satisfying (3) is a critical point of the loss function, 419

regardless of samples. 420

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

Define 443

444

445

446

447

448

449

450

451

452

Now consider two three hidden layer networks H,H ′ both with input dimension d , output dimension D , and hidden layer widths { m l } L l =1 , { m ′ l } L l =1 , respectively. Assume that m ′ 1 = m 1 , m ′ 2 = m 2 , m 2 &gt; 1 and m ′ 3 = m 3 +1 . In this case, H ′ is just one neuron wider than H and the embedding of parameters from that of H to H ′ by general compatible embedding is just splitting embedding or null-embedding. For splitting embedding, note that for any θ satisfying (3), up to permutation of entries a parameter θ ′ given by EP and satisfying (3) takes the form

<!-- formula-not-decoded -->

for some δ ∈ R . In particular, δw (3) m 3 , (1 -δ ) w (3) m 3 are parallel vectors in R m 2 . However, because m 2 &gt; 1 , not every θ ′ satisfying (3) has two parallel w (3) k 3 's. For null embedding, the weight it assigns to the extra neuron is fixed to 0. Thus, these two embedding operators (altogether) do not produce all sample-independent lifted critical points.

Remark A.2. Using the same proof idea, we can show that for two arbitrary L ≥ 3 hidden layer neural networks, not all sample-independent lifted critical points are produced by these embedding operators.

̸

Proposition A.2.2 (Proposition 4.2.1 in Section 4.2) . Given samples ( x i , y i ) n i =1 such that x i = 0 for all i and x i ± x j = 0 for 1 ≤ i &lt; j ≤ n . Given integers m,m ′ such that m&lt;m ′ . For any critical point θ narr = ( a k , w k ) m k =1 of the loss function corresponding to the samples such that R ( θ narr ) = 0 , the set of ( w ′ k ) m ′ k = m +1 ∈ R ( m ′ -m ) d of weights making the parameter

<!-- formula-not-decoded -->

a critical point for the loss function has zero measure in R ( m ′ -m ) d . Furthermore, any such critical point is a saddle.

Proof. Denote θ wide := ( a ′ k , w ′ k ) m k =1 , so by hypothesis we have a ′ k = 0 for all m &lt; k ≤ m ′ . Note that for any ( w ′ k ) m ′ k = m +1 , θ wide preserves output function, i.e., H ( θ wide , x ) = H ( θ narr , x ) for all x . Thus, for any w ′ m ′ ∈ R d , the partial derivative for a ′ m ′ is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

so that ∂R ∂a ′ m ′ ( θ wide ) = 0 if and only if φ ( w ′ m ′ ) = 0 . Since i) σ is a non-polynomial analytic function, ii) x i = 0 for all i , and iii) x i ± x j = 0 for all 1 ≤ i &lt; j ≤ n , by Lemma A.1.1 we have that { w ↦→ σ ( w · x i ) } n i =1 are linearly independent. Meanwhile, since R ( θ narr ) = 0 , there must be some i ∈ { 1 , ..., n } with ℓ ( H ( θ narr , x i ) , y i ) = 0 . But then by Assumption 3.2 on ℓ , we have H ( θ narr , x i ) = y i and thus ∂ p ℓ ( H ( θ narr , x j ) , y j ) = 0 for some j ∈ { 1 , ..., n } . Therefore, φ is a non-trivial linear combination of analytic, linearly independent functions, so it is analytic and not constant zero. But this implies that the set of φ -1 (0) has zero measure in R d . It follows that the set of ( w ′ k ) m ′ k = m +1 of weights making θ wide a critical point for the loss function has zero measure in R ( m ′ -m ) d .

453

454

455

456

457

̸

̸

Let θ wide be a critical point of the loss function. We now show that it is saddle. Let U be a neighborhood of θ wide. Since φ -1 (0) has zero measure, U contains a point

<!-- formula-not-decoded -->

̸

where w ′′ m ′ / ∈ φ -1 (0) , and thus ∇ R ( θ ′′ wide ) = 0 . On the other hand, as we mentioned above, H ( θ ′′ wide , x i ) = H ( θ narr , x i ) = H ( θ wide , x i ) for all i , whence R ( θ ′′ wide ) = R ( θ wide ) . Then Lemma A.1.2 shows that θ wide is a saddle.

̸

̸

̸

̸

̸

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

Proposition A.2.3 (Theorem 4.2.1 in Section 4.2) . Assume that ℓ : R 2 → R satisfies: the range of ∂ p ℓ ( p, · ) contains an open interval around 0 ∈ R . Given integers m,m ′ , n ∈ N such that m&lt;m ′ and n ≥ 1 + ( d +1) m , given θ narr = ( a k , w k ) m k =1 . For any fixed ( x i ) n i =1 ∈ R nd with x i ± x j = 0 and for a.e. w ′ ∈ R d , there are sample outputs ( y i ) n i =1 , ( y ′ i ) n i =1 such that

<!-- formula-not-decoded -->

is a critical point for the loss function corresponding to ( x i , y ′ i ) n i =1 , but not so to ( x i , y i ) n i =1 . Furthermore, when n ≥ 2 + ( d +1) m we can choose ( y ′ i ) n i =1 so that θ wide is a saddle.

Proof. We use the notations in the proof of Proposition A.2.2. Recall that for θ wide of the form (2) to be a critical point, we must have w ′ m ′ ∈ φ -1 (0) , where

<!-- formula-not-decoded -->

Define

<!-- formula-not-decoded -->

̸

Since n ≥ 1+( d +1) m , the kernel of M is non-trivial. Fix v ∈ ker M \{ 0 } . By linear independence of the neurons { w ↦→ σ ( w · x i ) } n i =1 , the function ∑ n i =1 v i σ ( w · x i ) is not constant zero (in w ), so its zero set has zero measure in R d (Lemma A.1.3) and for a.e. w ′ we have ∑ n i =1 v i σ ( w ′ · x i ) = 0 .

Then define and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that for any k &gt; m , any k 0 ∈ { 1 , ..., d } , and for any samples S = { ( x i , y i ) n i =1 } , we have (using a k = 0 )

<!-- formula-not-decoded -->

Therefore, ∇ R S ( θ wide ) = 0 if and only if [ ∂ p ℓ ( H ( θ narr , x i ) , y i )] n i =1 ∈ ker M ′ . By our construction above, v ∈ ker M \ ker M ′ . Let v ′ ∈ ker M ′ . The hypothesis on ℓ implies that the range of the map ( q i ) n i =1 ↦→ [ ∂ p ℓ ( H ( θ narr , x i ) , q i )] n i =1

contains a product neighborhood of 0 ∈ R n . This implies the existence of ( y i ) n i =1 and ( y ′ i ) n i =1 such that [ ∂ p ℓ ( H ( θ narr , x i ) , y i )] n i =1 is a non-zero multiple of v and [ ∂ p ℓ ( H ( θ narr , x i ) , y ′ i )] n i =1 is a non-zero multiple of v ′ . Then

̸

<!-- formula-not-decoded -->

̸

In particular, φ ( w ′ , ( y i ) n i =1 ) = 0 . Therefore, θ wide is a critical point for the loss corresponding to ( x i , y ′ i ) n i =1 , but not a critical point for the loss corresponding to ( x i , y i ) n i =1 .

Nowassume that n ≥ 2+( d +1) m . In this case ker M ′ is non-trivial, so we can find v ′ ∈ ker M ′ \{ 0 } , and then ( y ′ i ) n i =1 such that [ ∂ p ℓ ( H ( θ narr , x i ) , y ′ i )] n i =1 is a non-zero multiple of v ′ . Then θ wide is a critical point at which the loss function is non-zero. Thus, by Lemma A.1.2 it is a saddle.

̸

Proposition A.2.4 (Proposition 4.2.2 in Section 4.2) . Given samples ( x i , y i ) n i =1 with x i = 0 for all i and x i ± x j = 0 for 1 ≤ i &lt; j ≤ n . Given integers { m l } L l =1 , { m ′ l } L l =1 such that m l &lt; m ′ l for every 1 ≤ l ≤ L . Consider two L hidden layer neural networks with input dimension d , hidden layer widths { m l } L l =1 , { m ′ l } L l =1 , and output dimension D . Denote their parameters by θ narr , θ wide , respectively. Let θ narr be a critical point of the loss function corresponding to the samples ( x i , y i ) n i =1 , such that R ( θ narr ) = 0 . Denote the following sets:

̸

<!-- formula-not-decoded -->

̸

Namely, E is a set of parameters preserving output function, E ∗ is the set of parameters in E also 490 preserving criticality. Then E ∗ = E . Furthermore, E ∗ contains saddles. 491

̸

̸

Proof. We first show by induction that there is a parameter θ ( L -1) wide such that 492

̸

<!-- formula-not-decoded -->

̸

According to our notation for neural networks (Section 3.1), we denote the entries of θ narr as 493

<!-- formula-not-decoded -->

Start with l = 1 . The linear independence of neurons (Lemma A.1.1) guarantees the existence of some w ′ (1) m 1 +1 , ..., w ′ (1) m ′ 1 such that for every m 1 &lt; k 1 ≤ m ′ 1 , we have σ ( w ′ (1) k 1 · x i ) ± σ ( w ′ (1) k 1 · x j ) = 0 for 1 ≤ i &lt; j ≤ n . Define

̸

<!-- formula-not-decoded -->

̸

Then the first layer neuron H (1) ( θ (1) wide , x ) = [ σ ( w k 1 · x )] m ′ 1 k 1 =1 satisfies (a) H (1) k 1 ( θ (1) wide , · ) = H (1) k 1 ( θ (1) narr , · ) for 1 ≤ k 1 ≤ m 1 , (b) H (1) ( θ (1) wide , x i ) = 0 for all 1 ≤ i ≤ n and (c) H (1) ( θ (1) wide , x i ) ± H (1) ( θ (1) wide , x i ) = 0 for 1 ≤ i &lt; j ≤ n . Assume that for some l ∈ { 1 , ..., L -1 } we have found θ ( l ) wide such that the following holds:

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

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

<!-- formula-not-decoded -->

Then, for the construction of θ ( l +1) wide we do the following: 504

505

506

507

508

509

<!-- formula-not-decoded -->

- For each m l +1 &lt; k l +1 ≤ m ′ l +1 , find w ′ ( l +1) k l +1 ∈ R m ′ l such that σ ( w ( l +1) k l +1 H ( l ) ( θ ( l ) wide , x i ) ) = 0 for all i and σ ( w ( l +1) k l +1 H ( l ) ( θ ( l ) wide , x i ) ) ± σ ( w ( l +1) k l +1 H ( l ) ( θ ( l ) wide , x j ) ) = 0 for 1 ≤ i &lt; j ≤ n . The existence of w ( l +1) k ′ l +1 is due to the linear independence of the neurons { w ↦→ σ ( wH ( l ) ( θ ( l ) wide , x i ) )} n i =1 from our induction hypothesis (b).

Set θ ( l +1) wide = ( ( w ′ ( l +1) k l +1 ) m ′ l +1 k l +1 =1 , θ ( l ) wide ) . We have 510

̸

<!-- formula-not-decoded -->

Namely, (a), (b) and (c) are satisfied for H ( l +1) ( θ ( l +1) wide , x ) , thus proving the induction step. 511 Recall that the (wider) neural network takes the form 512

<!-- formula-not-decoded -->

̸

̸

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

For any θ ( L -1) wide such that H ( L -1) k L -1 ( θ ( L -1) wide , x ) ) = H ( L -1) k L -1 ( θ ( L -1) narr , x ) for all 1 ≤ k L -1 ≤ m L -1 , define E ( θ ( L -1) wide ) as the set of parameters θ wide = (( a ′ j ) D j =1 , ( w ′ ( L ) k L ) m ′ L k L =1 , θ ( L -1) wide ) with the following properties:

- For each 1 ≤ j ≤ D , a ′ j = ( a j 1 , ..., a jm L , 0 , ..., 0) .
- For each 1 ≤ k L ≤ m L , w ′ ( L ) k L = ( w ( L ) k L , 0) .
- For each m L &lt; k L ≤ m ′ L , w ′ ( L ) k L ∈ R m ′ L -1 is arbitrary.

Then define

<!-- formula-not-decoded -->

Clearly, E ( θ ( L -1) wide ) is a connected subset of E of dimension ≥ 1 and E ∗ ( θ ( L -1) wide ) is a subset of E ∗ . We would like to show that for some θ ( L -1) wide , ∇ R is not constant zero on E ( θ ( L -1) wide ) . This means the restriction of ∇ R to E ( θ ( L -1) wide ) is not constant zero, whence has zero measure in E ( θ ( L -1) wide ) . Let θ ( L -1) wide be constructed as above. Fix θ wide ∈ E ( θ ( L -1) wide ) . For each ¯ j consider the partial derivative of the loss function against a ¯ jm ′ L :

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The second equality holds because by definition the parameters in E preserve output function. Similar 526 to the proof for Proposition A.2.2, we define an analytic function 527

<!-- formula-not-decoded -->

̸

̸

Note that ∂R ∂a ¯ jm ′ L ( θ wide ) = 0 if and only if w ′ ( L ) m ′ L ∈ φ -1 (0) . Since R ( θ narr ) = 0 , there must be some i with e i ¯ j = 0 . Since H ( L -1) ( θ ( L -1) wide , x i ) = 0 for all i and H ( L -1) ( θ ( L -1) wide , x i ) ± H ( L -1) ( θ ( L -1) wide , x j ) = 0 for 1 ≤ i &lt; j ≤ n , the functions

<!-- formula-not-decoded -->

are linearly independent. Therefore, φ is a non-trivial linear combination of analytic, linearly independent functions, so it is analytic and not constant zero. This means φ -1 (0) has zero measure in R d . In particular, ∂R ∂a ¯ jm L is not constant zero on E ( θ ( L -1) wide ) , so neither is the restriction of ∇ R to E ( θ ( L -1) wide ) , proving our claim.

̸

Our proof above shows that for any θ wide ∈ E ∗ ( θ ( L -1) wide ) and any neighborhood U of θ wide we have U ∩ ( E ( θ ( L -1) wide ) \ E ∗ ( θ ( L -1) wide ) ) = ∅ . Meanwhile, the loss function is constant on E ( θ ( L -1) wide ) . Thus, by Lemma A.1.2 we conclude that θ wide is a saddle.

LemmaA.2.1. Given θ narr. Let θ ( L -1) wide be constructed as in Proposition A.2.4. Let θ wide ∈ E ( θ ( L -1) wide ) . Then for any j ∈ { 1 , ..., D } and k L ∈ { 1 , ..., m L } we have ∂H ∂a ′ jk L ( θ wide , · ) = ∂H ∂a jk L ( θ narr , · ) . Moreover, for any l ∈ { 1 , ..., L } the following holds:

<!-- formula-not-decoded -->

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

̸

̸

543

<!-- formula-not-decoded -->

Proof. The proof is basically straightforward computations. By definition we have 544

<!-- formula-not-decoded -->

Recall that in our construction, w ′ ( L ) k L = ( w ( L ) k L , 0) and H ( L -1) k L -1 ( θ ( L -1) wide , x ) = H ( L -1) k L -1 ( θ ( L -1) narr , x ) for 545 all 1 ≤ k L -1 ≤ m L -1 , whence 546

<!-- formula-not-decoded -->

This proves the first part of the lemma. 547

To prove the result for ∂H ∂w ′ ( l ) k l k l -1 ( θ wide , · ) we observe that 548

<!-- formula-not-decoded -->

where A ′ , A are the matrices whose rows are a ′ j , a j 's: 549

<!-- formula-not-decoded -->

and for each 1 ≤ ¯ l ≤ L we define 550

<!-- formula-not-decoded -->

Again, recall that w ′ ( l +1) k l +1 = ( w ( l +1) k l +1 , 0) . In particular, when k l &gt; m l we have w ′ ( l +1) k l +1 k l = 0 . Thus, 551

<!-- formula-not-decoded -->

which shows ∂H ∂w ′ ( l ) k l k l -1 ( θ wide , x ) = 0 when k l &gt; m l . Now let k l ≤ m l and k l -1 ∈ { 1 , ..., m l -1 } . For 552 each l &lt; ¯ l ≤ L define 553

<!-- formula-not-decoded -->

anbd similarly, define 554

<!-- formula-not-decoded -->

We shall first prove that the first m ¯ l entries of v ′ ( ¯ l ) and the first m ¯ l entries of v ( ¯ l ) coincide for each 555 l ≤ ¯ l ≤ L . The key is that by our construction of θ ( L -1) wide , for any 1 ≤ ¯ l ≤ L and any k ¯ l ≤ m ¯ l we 556 have 557

<!-- formula-not-decoded -->

Since we also have H ( l -1) k l -1 ( θ ( l -1) wide , x ) = H ( l -1) k l -1 ( θ ( l -1) narr , x ) and w ′ ( l ) k l +1 k l = w ( l ) k l +1 k l for 1 ≤ k l +1 ≤ 558 m l +1 , our claim clearly holds for v ′ ( l ) and v ( l ) . Suppose the result holds for some ¯ l &lt; L . Then we 559

can write v ′ ( ¯ l ) as v ′ ( ¯ l ) = ( v ( ¯ l ) , u ) T for some vector u . Then 560

<!-- formula-not-decoded -->

This completes the induction step. Finally, 561

<!-- formula-not-decoded -->

562

563

564

565

566

567

568

569

570

completing the proof.

Proposition A.2.5 (Theorem 4.2.2 in Section 4.2) . Assume that ℓ : R 2 → R satisfies: the range of ∂ p ℓ ( p, · ) contains a neighborhood around 0 ∈ R D . Given θ narr. Let θ ( L -1) wide be constructed as in Proposition A.2.4. Let N denote the parameter size of the narrower network.

̸

- (a) Consider sample size n ≥ 1+ N D . For any fixed ( x i ) n i =1 ∈ R nd with x i ± x j = 0 and for a.e. θ wide ∈ E ( θ ( L -1) wide ) , there are sample outputs ( y i ) n i =1 , ( y ′ i ) n i =1 such that θ wide is a critical point for the loss function corresponding to ( x i , y ′ i ) n i =1 but not so to ( x i , y i ) n i =1 .
- (b) Consider sample size n ≥ 1+ D + ∑ L l =2 m l ( m ′ l -1 -m l -1 )+ N D . Then we can choose ( y ′ i ) n i =1 so that E ( θ ( L -1) wide ) contains saddles.

Proof. The proof is almost identical to that of Proposition A.2.2. 571

572

- (a) Define M as an N -rows, Dn -columns block matrix

<!-- formula-not-decoded -->

For any samples S =: ( x i , y i ) n i =1 we have ∇ R S ( θ narr ) = 0 if and only if 573

<!-- formula-not-decoded -->

̸

where ∂ p ℓ denotes the gradient of ℓ with respect to its first entry. Since n ≥ 1+ N D , M 574 has more columns than rows and ker M is non-trivial. Fix any v ∈ ker M \ { 0 } and find 575 ( y i ) n i =1 such that the (vectorized) vector of partial derivatives [ ∂ p ℓ ( H ( θ wide , x i ) , y i )] n i =1 is 576 a non-zero multiple of v . Thus, ∂ j ℓ ( H ( θ narr , x i ) , y i ) = 0 for some i, j . Recall that our 577

<!-- formula-not-decoded -->

580

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

̸

construction of θ ( L -1) wide implies H ( L -1) ( θ ( L -1) wide , x i ) ± H ( L -1) ( θ ( L -1) wide , x j ) = 0 . By Lemma 578 A.1.1, the analytic function 579

<!-- formula-not-decoded -->

̸

is not constant zero. Thus, for a.e. w ′ ∈ R m ′ L we have φ ( w ′ ) = 0 . In particular, the set

<!-- formula-not-decoded -->

has full-measure in E ( θ ( L ) wide ) . Note that any θ wide in this set is not a critical point of the loss 581 function corresponding to ( x i , y i ) n i =1 , because the partial derivative for a ′ jm ′ L is non-zero 582

(see also (4) for the formula of ∂H ∂a ′ jk L ). 583

584

Fix θ wide in this set. Define

<!-- formula-not-decoded -->

By Lemma A.2.1, part of each submatrix D θ H ( θ wide , x i ) of M ′ is D θ H ( θ narr , x i ) . In 585 particular, by rearranging the rows if necessary M ′ can be written as the following block 586 matrix 587

<!-- formula-not-decoded -->

Let v ′ ∈ ker M ′ and find some ( y ′ i ) n i =1 such that [ ∂ p ℓ ( H ( θ wide , x i ) , y i )] n i =1 is a non-zero 588 multiple of v ′ . Then 589

<!-- formula-not-decoded -->

which implies that θ wide is a critical point of the loss corresponding to ( x i , y ′ i ) n i =1 .

(b) By Lemma A.2.1, the entries of U consists of the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first part gives ∑ L l =2 m l ( m ′ l -1 -m l -1 ) number of rows of U , while the second part gives D ( m ′ l -1 -m l ) number of rows of U . However, for any θ wide ∈ E ( θ ( L -1) wide ) such that w ′ ( L ) m L +1 = ... = w ′ ( L ) m ′ L , this reduces to only D different rows (see also (4) for the formula of ∂H ∂a ′ jk L ). In other words, for such θ wide we have a D + ∑ L l =2 m l ( m ′ l -1 -m l -1 ) + N row ∑ L ′

matrix M ′′ with ker M ′′ = ker M ′ . Since n ≥ 1+ D + l =2 m l ( m l -1 -m l -1 )+ N D , M ′ and M ′′ have more rows than columns, so there is some v ′ ∈ ker M ′′ \ { 0 } . Find ( y ′ i ) n i =1 such that [ ∂ p ℓ ( H ( θ wide , x i ) , y i )] n i =1 is a non-zero multiple of v ′ . Then

<!-- formula-not-decoded -->

̸

which implies that θ wide is a critical point of the loss corresponding to ( x i , y ′ i ) n i =1 . Mean601 while, since [ ∂ p ℓ ( H ( θ wide , x i ) , y i )] n i =1 = 0 , by Assumption 3.2 the loss function is non-zero 602 at θ wide (and thus non-zero at θ narr).It follows from Lemma A.1.2 that θ wide is a saddle. 603

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

623

624

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

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims in abstract and introduciton are mostly a summary of Section 4

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See Section 6

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

Justification: The assumptions are made in Section 3 and in the statements of each result. The detailed proofs can be found in Appendix.

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

Justification: The experiment is described in detail in Section 5.

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

## Answer: [Yes]

Justification: The experiment is described in detail in Section 5. Note that the experiment is only for illustration of results in Section 4.

## Guidelines:

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

Answer: [NA]

Justification: The paper is completely theoretical.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper is completely theoretical.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).

763

764

765

766

767

768

769

770

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

796

797

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

- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The paper is completely theoretical.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper follows NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper is completely theoretical.

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

Justification: The paper does not release data or models.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assests.

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

Justification: The paper does not introduce new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing experiments nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

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

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The authors only use LLMs (specifically, ChatGPT) for editing the paper and formatting figures.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.