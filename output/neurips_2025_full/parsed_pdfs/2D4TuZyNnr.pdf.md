## REASONING COMPILER : LLM-Guided Optimizations for Efficient Model Serving

Annabelle Sujun Tang University of California San Diego sujun@ucsd.edu

## Christopher Priebe ∗

University of California San Diego cpriebe@ucsd.edu

Lianhui Qin University of California San Diego

Rohan Mahapatra ∗ University of California San Diego rohan@ucsd.edu

Hadi Esmaeilzadeh University of California San Diego lianhui@ucsd.edu

hadi@ucsd.edu

## Abstract

While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining , can leverage the context-aware decision space of compiler optimizations to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed REASONING COMPILER ) that formulates optimization as a sequential, context-aware decision process guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-informed transformations that reflect the current program state and accumulated performance feedback. MCTS incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating a structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization. 1

## 1 Introduction

The rise of model serving for LLMs, diffusion models, and other neural models has enabled a new class of intelligent systems, driving transformative applications in healthcare, education, and scientific discovery. These models incur significant computational demands during inference, which proportionally translate into substantial monetary costs. Driving down the cost of model serving is critical, not merely to broaden access and democratize inference, but to catalyze faster cycles of innovation in model design and deployment. Achieving this goal demands reducing inference runtime on computational infrastructure, resources that are not only expensive but also increasingly limited in availability. Compiler optimizations are a critical enabler, not only for cost-efficient inferencing across diverse applications but also for empowering rapid research iteration.

1 Code is available at https://github.com/he-actlab/REASONING\_COMPILER

* Equal contribution

Existing compilers struggle with neural models due to the exponentially large space of valid program transformations (e.g., tiling, fusion, and layout changes). Each decision, such as selecting a tiling factor or a parallelization strategy, introduces dependencies and constraints that influence the feasibility and performance benefits of subsequent transformations. Rule-based optimizations also often rely on hand-tuned heuristics that can overfit to a specific workload or hardware target. The seminal work in superoptimization [1-3] aimed to tackle these shortcomings through enumerative or symbolic search, but the search space proved combinatorial and rugged. STOKE [4] showed that high-quality programs often lie in regions separated by low-probability paths, and therefore adopted Markov chain Monte Carlo (MCMC)-based randomized search. Neural compilers followed suit, using evolutionary search or simulated annealing to navigate similarly irregular landscapes [5-8]. While these methods have shown promise in discovering performant configurations, they are fundamentally sample-inefficient. They overlook synergistic transformations that emerge only when decisions are made with contextual awareness. These techniques also often explore redundant subspaces or invalid configurations.

In contrast, we set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining , can leverage the context-aware decision space of compiler optimizations to significantly improve sample efficiency. To that end, we introduce a novel compilation framework that couples LLM reasoning with Monte Carlo tree search (MCTS) [9] to guide compiler optimization. Hence, in our approach, compiler optimization is cast as a sequential decision-making process , in which each transformation, such as tiling, fusion, or vectorization, is selected with awareness of the current program state, while also assimilating downstream information and propagating its implications upstream to guide future decisions. Our approach avoids the prohibitive cost of fine-tuning LLMs as compilation policies, nor does it require additional training or task-specific adaptation. In this formulation, the LLM evaluates partial transformation sequences and proposes contextually appropriate next steps, drawing upon hardware-informed cost models and the historical trajectory of optimization decisions to inform its proposal. The LLM serves as a context-aware proposal engine: given the current schedule and its observed performance, it generates candidate transformations that are likely to be effective in the context of the traversed trajectory. These LLM-guided reasoning choices are integrated into an MCTS framework that provides a structured mechanism for balancing exploration and exploitation by evaluating LLM-suggested transformations, expanding promising branches, and leveraging rollout feedback to adaptively steer the search toward high-performing regions of the exponentially large optimization space.

This integration of LLM-based chain-of-thought (CoT) guidance with tree search combines contextual reasoning and adaptability with principled, structured decision-making , enabling the compiler to navigate the complexity of the search space with significantly improved sample efficiency. We evaluate the REASONING COMPILER and compare its improvements and sample efficiency with TVM, which employs evolutionary search. Results show that the REASONING COMPILER consistently achieves significantly higher speedups than what TVM achieves using significantly fewer samples. On five representative benchmarks ( Llama-3-8B Attention Layer , DeepSeek-R1 MoE Layer , FLUX Attention Layer , FLUX Convolution Layer , and Llama-4-Scout MLP Layer ) and across five hardware platforms (Amazon Graviton2, AMD EPYC 7R13, Apple M2 Pro, Intel Core i9, and Intel Xeon E3), the REASONING COMPILER achieves 5.0 × average speedup using 5.8 × fewer samples, resulting in an average of 10.8 × improvement over TVM in sample efficiency. For the end-to-end Llama-3-8B benchmark across five hardware platforms, the REASONING COMPILER uses 3.9 × fewer samples to achieve a 4.0 × speedup, yielding a 5.6 × sample efficiency improvement. These results underscore the promise of LLM-guided reasoning in neural compilation for efficient and scalable model serving.

## 2 Problem Formalization

<!-- formula-not-decoded -->

We consider the problem of optimizing an input program p 0 ∈ P representing a layer from a neural network for some objective function f : P ↦→ R ≥ 0 . This objective function represents an evaluation of the program on the target platform for some figure of merit (e.g., latency, power, utilization). Any program p ∈ P can be transformed through the application of some transformation/optimization (used interchangeably from here on out) o ∈ O , where each optimization is a

function o : P ↦→ P that performs a targeted transformation to the program, thus introducing a new variant of the program that is semantically equivalent to the original program but may perform better or worse on a target hardware platform. In this way, successive application of transformations to a program can yield significant performance differences from the original. Therefore, given some maximum transformation sequence length T , the goal is to find a sequence of transformations S opt. = ⟨ o 1 , o 2 , . . . , o n ⟩ such that n ≤ T and f ( p opt. ) = max S ′ ⊆ O ∗ , | S ′ |≤ T f (( o ′ k ◦ · · · ◦ o ′ 1 )( p 0 )) where p opt. = ( o n ◦ o n -1 ◦ · · · ◦ o 1 )( p 0 ) and O ∗ is the Kleene star 2 of O . These constraints collectively define the optimization objective given in Equation (1).

To facilitate an efficient search over the space of valid program transformation sequences, we cast the optimization problem as a finite-horizon Markov decision process (MDP) defined by the tuple M = ⟨ S , A , P , R ⟩ . This formulation provides a structured approach for sequential decisionmaking in the transformation space, allowing the search process to account for how individual transformations compound over time to affect final program performance. Compared to unstructured methods such as exhaustive or purely stochastic search, which often require a large number of expensive program evaluations, casting the problem as an MDP enables more deliberate exploration, offering the potential for improved sample efficiency. Each state s t ∈ S corresponds to a program p t ∈ P obtained by applying a sequence of transformations to the original program p 0 , i.e., s t = p t = ( o t ◦ · · · ◦ o 1 )( p 0 ) . An action a t ∈ A corresponds to selecting a transformation o ∈ O to apply at step t , transitioning the current program to a new variant. Since the application of a transformation is deterministic, the transition function P ( s t +1 | s t , a t ) is 1 if s t +1 = a t ( s t ) and 0 otherwise. The reward function is defined as the objective value caused by the optimization sequence, i.e., R ( s t , a t ) = s · f ( a t ( p t )) where s ∈ { +1 , -1 } is chosen so that larger rewards are always better.

By formulating the problem as an MDP, we enable the use of planning algorithms such as Monte Carlo tree search (MCTS) [9] to explore program transformation sequences. Under standard assumptions, such as finite branching, bounded rewards, and a tree policy (e.g., UCT) that guarantees persistent exploration, MCTS is consistent on finite-horizon problems: as the number of simulations tends to infinity, it converges (with probability 1) to the optimal root action/sequence S opt. that maximizes the objective. With any finite simulation budget, it returns a high-quality but approximate solution. Consequently, our framework (see §3) yields a sequence S ′ opt. that approximately maximizes the objective in practice while enjoying asymptotic optimality in theory.

## 3 REASONING COMPILER : Integrating LLM-Guided Contextual Reasoning with Monte Carlo Tree Search

We present the REASONING COMPILER , a novel compilation framework that unifies the structured exploration capabilities of Monte Carlo tree search (MCTS) [9] with the contextual, history-aware reasoning of large language models (LLMs). While MCTS provides a principled approach to exploring sequences of program transformations, compiler optimization introduces a unique challenge: the successive application of transformations can exhibit complex, non-local interactions that are difficult to capture through purely stochastic or myopic policies. To address this, we employ an LLM to model program transformation context, tracking which transformations have been applied, how they impact performance, and what directions remain promising. This contextualization is essential to enabling effective and sample-efficient search in compiler optimization.

Optimization interactions are complex, making efficient search challenging. Unlike tasks where actions are relatively independent, program transformations compose in subtle and complex ways. For example, the profitability of applying loop tiling may depend on the prior application of loop fusion or unrolling. Additionally, transformations can introduce new, unforeseen opportunities/constraints for future transformations. These dependencies make the space of valid and useful transformation sequences both combinatorial and deeply contextual. While black-box methods such as evolutionary search and some implementations of reinforcement learning have achieved notable success in compiler autotuning [6, 8, 10, 11], they often do not explicitly model the nuanced structural and temporal dependencies between transformations. This can limit their ability to generalize across contexts, as optimization efficacy depends on transformation histories. Even when guided by local

2 The Kleene star operator, denoted with an asterisk ( ∗ ), represents the set of all finite-length sequences, including the empty sequence, formed from elements of a given set.

&lt;latexi sh

1\_b

64="v0EVI2u

p

Wnz

ASQD

Z93

w

&gt;

B7H

c

N

8

J

Ur q/

L

PR

K

T

m

G

gY

y5

+

M

f

d

o

j

k

XF

O

C

&lt;latexi sh

1\_b

64="rBd/

R0J

TP

Zk

HVWA

5j

w

&gt;

7n

c

DLSgM

F

2p

f

u

I

G

C

Uomz

+

X

9

Y

Q

y

3

N

q

K

v

O

E

8

&lt;latexi sh

1\_b

64="YUp

NXH

KmVRcL

r2

Z8

W3

Go

&gt;A

B7n

D

S

F

f

du vgQk

/

j

T

5

wM

E

I

C

9

OJ

z

y

0

q+

P

&lt;latexi sh

1\_b

64="

Py7

Vv

LpfY

I

qM5

z8

o

&gt;A

B

n

c

D

S

N

F

2

r

Zdu gQk

0

U

m

3

T

+

XCj

9

J

K

O

RW

E

G

w

/

H

&lt;latexi sh

1\_b

64="K92k

Egv

Dw rojq

B

fU

Y

&gt;A

7n

c

V

LS

M

F

3

ZX

W

J

p

Ny

0Q8m

T

I5

H+

G

u/R

O

P

z

d

C

&lt;latexi sh

1\_b

64="gjFIG

m

v8n c+

5

S7

PB

QA

&gt;

VDL

M

2pr

f

Zdu

k

w

W

C

9o

J

O

q

X

T

y0

z

3

R

/

H

NE

K

Y

U

&lt;latexi sh

1\_b

64="J3Zn

LP

or

QUDu7FX

K

c

A

&gt;

B

V

SgN

EOz

G

M

H

Y

8

d

Rj

C

w

m

2

k

/

I

T

q

W

p

5

9

y

f

0

+

v

&lt;latexi sh

1\_b

64="O

NWM

8g o5nPu

+0

zUj

I

&gt;A

B7X

c

VDLS

E

y

r

f

Y9

F

K

2

Q

HJ

mZ

p

/

wd

v

3

k

G

q

T

C

R

&lt;latexi sh

1\_b

64="zPI

kwjA9

7Gd

L

v

C

uc

Q

&gt;

B8X

VD

SgN

EOy

r

fUY

Fo

K

2

m

5

n

R

Z

W

0M

q

+

H

J

3

/

T

p

&lt;latexi sh

1\_b

64="CwY0K

SO

AIU

EoZ

9

3cR

5

&gt;

B7n

VDL

gM

F

2pr

f

du

k

X8

W

J

m

j

G

Q

q

v

+T

y

z

N

/

P

H

&lt;latexi sh

1\_b

64="rE

n

5ACd

Z

0Y

FV

w

R

&gt;

B7

cj

DJSgN

K

XGL

vT

Q

O

9

I

k

Pp

U

8

/

W

q

o

f

2u

m

+z

H

3

y

M

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="zPI

kwjA9

7Gd

L

v

C

uc

Q

&gt;

B8X

VD

SgN

EOy

r

fUY

Fo

K

2

m

5

n

R

Z

W

0M

q

+

H

J

3

/

T

p

&lt;latexi sh

1\_b

64="DL+y

M

X

Om

B7

u2G

c5p

w

&gt;A

n

V

JSgN

EK

W

CjP

d

zo

0

T9

k

R

8

/

UH

q

F

f

v

3

Z

r

I

Q

Y

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="XrgK

q2oH

L/W30

j

kA

&gt;

B7

c

V

NS8

EJ

U

9

n

PR

p

y

Z

I

+Q

C

z

uO

D

d

M

m

v

f

F

Y

T

w

5

G

&lt;latexi sh

1\_b

64="3S8QU0jw

LToKBz

X

n

NDM

&gt;A

7

c

V

g

F

2pr

f

Zdu

k

I

v

y

m

Y

JR

P

C

+

/

O

9

G

W

q

5

E

H

&lt;latexi sh

1\_b

64="XrgK

q2oH

L/W30

j

kA

&gt;

B7

c

V

NS8

EJ

U

9

n

PR

p

y

Z

I

+Q

C

z

uO

D

d

M

m

v

f

F

Y

T

w

5

G

&lt;latexi sh

1\_b

64="

SNJKv

CTEn

DZy8

g

HR

Q

&gt;A

B7

cjV

2

W

OX

MU

z

o

0

5

d

kp

f

I/w r3

P

G

9

q

Y

L

+

u

F

m

&lt;latexi sh

1\_b

64="rE

n

5ACd

Z

0Y

FV

w

R

&gt;

B7

cj

DJSgN

K

XGL

vT

Q

O

9

I

k

Pp

U

8

/

W

q

o

f

2u

m

+z

H

3

y

M

&lt;latexi sh

1\_b

64="

OUSmNRr

G8Q

XDE

n

A

2j

&gt;

CJ

c

V

L

gM

F

3

vqk

w

K

f

0

5

P

y

B

z

I

u

d

W

9

o

H

T

7

Y

/

Z

p

&lt;latexi sh

1\_b

64="z

AjLWqVB

8JX

ZMG

g

m3

E

&gt;

CIH

c

D

S

N

F

U

2v

0

w

RT

uK9o

p

Q

y

+

n

Y

7

r

d

f

k

P5

O

/

&lt;latexi sh

1\_b

64="E2

mMR

W

I

Yj fr+cSLD

OzB8

&gt;A

C

H

V

N

FJ3

PU

d

gu

K

0

v

TQ

y

pk

G

/

5

o

q

9

7

n

Z

w

X

&lt;latexi sh

1\_b

64="XrgK

q2oH

L/W30

j

kA

&gt;

B7

c

V

NS8

EJ

U

9

n

PR

p

y

Z

I

+Q

C

z

uO

D

d

M

m

v

f

F

Y

T

w

5

G

&lt;latexi sh

1\_b

64="L

9

3Hp

EB

I

v

r+AmT75

ZfU

&gt;

X

cjVD

S

N

F

2

dugkWoC0

Q

MJ

y

RK

G

/

n

P

O

Y

8

z

w

q

&lt;latexi sh

1\_b

64="

ORNwgTZL0S

coXmU

V23E

&gt;A

B+

j

D

F

pr

f

du kW

C

v

Y

y

Q5

MJ

K

GI

/

n

9

P

H

8

7

q

z

&lt;latexi sh

1\_b

64="mdyXf

/R3

DZU

KWJ5G

YwBqc

&gt;A

9

jV

LSgN

EOz

Mr

H

2P

8

I

n

Q

p

F

+

v

k

T

o

C

u

0

7

&lt;latexi sh

1\_b

64="9uLA

0oD5X7

z

NdK

y

32W

w

&gt;

B

H

c

V

Sg

EO

r

fUY

P

Q

TM

I

j

R

Zm

k

C

8

n

v

q

+

G

/

F

J

p

&lt;latexi sh

1\_b

64="YEc

ZBDQrLV+/XH5

8v

T

fw

&gt;A

7n

SgN

Oy

U

9

oM

CGF

0

j

2

km

z

WPIR

d

K

q

p

3

u

J

&lt;latexi sh

1\_b

64="YEc

ZBDQrLV+/XH5

8v

T

fw

&gt;A

7n

SgN

Oy

U

9

oM

CGF

0

j

2

km

z

WPIR

d

K

q

p

3

u

J

&lt;latexi sh

1\_b

64="3S8QU0jw

LToKBz

X

n

NDM

&gt;A

7

c

V

g

F

2pr

f

Zdu

k

I

v

y

m

Y

JR

P

C

+

/

O

9

G

W

q

5

E

H

&lt;latexi sh

1\_b

64="+/9

WK

dNQ

U

p

Ro

E

w

&gt;A

B

X

c

VDLS

F

2

r

f

Z

u

v

Y

y

0I

mk3

J

8

Cj

5

P

gM

7

n

O

T

H

q

G

z

&lt;latexi sh

1\_b

64="3S8QU0jw

LToKBz

X

n

NDM

&gt;A

7

c

V

g

F

2pr

f

Zdu

k

I

v

y

m

Y

JR

P

C

+

/

O

9

G

W

q

5

E

H

&lt;latexi sh

1\_b

64="Cfw

G

d3OJ

UR8MZ

L

cKk

&gt;A

B7n

V

NS

E

r

q/

9

0

F

oWy2m

p

I

P

j

zX

Y

H

+

Q

T

D

g

u

5

v

&lt;latexi sh

1\_b

64="XrgK

q2oH

L/W30

j

kA

&gt;

B7

c

V

NS8

EJ

U

9

n

PR

p

y

Z

I

+Q

C

z

uO

D

d

M

m

v

f

F

Y

T

w

5

G

&lt;latexi sh

1\_b

64="Uqg cr

BEwJHn

5Z

K7

3j p9mA

&gt;

VDLS

M

F

2

f

du

k

Q

I

v

Y

X

/

WT

G

R

8

o

P

z

N

0

C

y

+

O

Figure 1: Overview of the optimization workflow. The algorithm explores the tree to select a candidate node. At this node, the LLM is prompted with contextual information to generate a sequence of transformations, which are then applied to produce optimized code variants.

<!-- image -->

reward signals, they may struggle to capture the interplay between past decisions and future opportunities, limiting their effectiveness in deeply contextual optimization landscapes. Our insight is that efficient search in this space benefits from an agent that reasons over transformation history, structural code changes, and observed performance dynamics to choose the next step.

## 3.1 LLM-Guided Contextual Reasoning for Program Transformation Proposal

Contextual reasoning via LLMs. To address these challenges, the REASONING COMPILER leverages a large language model (LLM) as a contextual reasoning engine. The LLM is tasked with synthesizing program transformations that are not only syntactically valid but also informed by the program's full history and structure. By prompting the LLM with a rich, structured representation of the current optimization state, we enable it to reason over the cumulative effects of prior transformations, analyze performance trends, and identify differential improvements over prior programs.

Figure 1 illustrates the optimization workflow. From the root program, the REASONING COMPILER traverses the tree by computing the UCT score [12], selecting a promising leaf node (i.e., program) p i for expansion by balancing exploitation of high-reward paths and exploration of under-sampled branches based on visit statistics and empirical mean rewards (see §3.2).

Prompt construction. At each expansion step in the search, the LLM receives a prompt that includes the source code and predicted performance for the current program p i , its parent p i -1 , and its grandparent p i -2 . It also includes the ordered sequences of transformations that were applied to reach each of these program variants, denoted S i , S i -1 , and S i -2 . Finally, the full set of available transformation operations O is included. Given this context, the LLM is explicitly instructed to: (1) analyze the differences between program variants and their associated performance scores, identifying which transformations contributed to observed performance changes; (2) reason about potential interactions between previously applied and candidate future transformations, including both synergistic and antagonistic effects; (3) synthesize a new sequence of transformations that is justified in the context of the current program structure and transformation history; and (4) provide a rationale for the proposed sequence, referencing specific code features and transformation interactions. This structured prompt is designed to elicit chain-of-thought (CoT) reasoning [13], encouraging the LLM to perform deep, multi-step analysis and move beyond surface-level edits, instead generating proposals that are both semantically meaningful and tailored to the evolving optimization trajectory.

Transformation proposal and validation. The LLM proposes a candidate transformation o i +1 ∈ O in the form of a string. Given the generative nature of the LLM, the output may include an invalid or unrecognized transformation even though it is guided by a predefined set of valid transformations. To ensure correctness, the output string is first parsed and filtered to retain only a transformation that matches known valid names and transformation parameters. If no valid transformation is found, the REASONING COMPILER samples a random transformation from the valid set. The successfully validated and applied transformation yields a new program variant p i +1 , with its transformation history updated as S i +1 = S i ⊕⟨ o i +1 ⟩ , where ⊕ denotes sequence concatenation. This new program variant is scored using a hardware cost model and used to update the MCTS tree (see §3.2).

It is important to emphasize that the LLM is not the centerpiece of our contribution, but a necessary enabler of effective search in this domain. Compiler optimization poses a uniquely challenging setting due to the non-local, compositional nature of transformation interactions. Traditional blackbox search or heuristic-guided methods struggle to navigate such spaces efficiently. The REASONING

&lt;latexi sh

1\_b

64="v0EVI2u

p

Wnz

ASQD

Z93

w

&gt;

B7H

c

N

8

J

Ur q/

L

PR

K

T

m

G

gY

y5

+

M

f

d

o

j

k

XF

O

C

&lt;latexi sh

1\_b

64="rBd/

R0J

TP

Zk

HVWA

5j

w

&gt;

7n

c

DLSgM

F

2p

f

u

I

G

C

Uomz

+

X

9

Y

Q

y

3

N

q

K

v

O

E

8

&lt;latexi sh

1\_b

64="YUp

NXH

KmVRcL

r2

Z8

W3

Go

&gt;A

B7n

D

S

F

f

du vgQk

/

j

T

5

wM

E

I

C

9

OJ

z

y

0

q+

P

&lt;latexi sh

1\_b

64="

Py7

Vv

LpfY

I

qM5

z8

o

&gt;A

B

n

c

D

S

N

F

2

r

Zdu gQk

0

U

m

3

T

+

XCj

9

J

K

O

RW

E

G

w

/

H

&lt;latexi sh

1\_b

64="K92k

Egv

Dw rojq

B

fU

Y

&gt;A

7n

c

V

LS

M

F

3

ZX

W

J

p

Ny

0Q8m

T

I5

H+

G

u/R

O

P

z

d

C

&lt;latexi sh

1\_b

64="gjFIG

m

v8n c+

5

S7

PB

QA

&gt;

VDL

M

2pr

f

Zdu

k

w

W

C

9o

J

O

q

X

T

y0

z

3

R

/

H

NE

K

Y

U

&lt;latexi sh

1\_b

64="J3Zn

LP

or

QUDu7FX

K

c

A

&gt;

B

V

SgN

EOz

G

M

H

Y

8

d

Rj

C

w

m

2

k

/

I

T

q

W

p

5

9

y

f

0

+

v

&lt;latexi sh

1\_b

64="O

NWM

8g o5nPu

+0

zUj

I

&gt;A

B7X

c

VDLS

E

y

r

f

Y9

F

K

2

Q

HJ

mZ

p

/

wd

v

3

k

G

q

T

C

R

&lt;latexi sh

1\_b

64="CwY0K

SO

AIU

EoZ

9

3cR

5

&gt;

B7n

VDL

gM

F

2pr

f

du

k

X8

W

J

m

j

G

Q

q

v

+T

y

z

N

/

P

H

&lt;latexi sh

1\_b

64="zPI

kwjA9

7Gd

L

v

C

uc

Q

&gt;

B8X

VD

SgN

EOy

r

fUY

Fo

K

2

m

5

n

R

Z

W

0M

q

+

H

J

3

/

T

p

&lt;latexi sh

1\_b

64="rE

n

5ACd

Z

0Y

FV

w

R

&gt;

B7

cj

DJSgN

K

XGL

vT

Q

O

9

I

k

Pp

U

8

/

W

q

o

f

2u

m

+z

H

3

y

M

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi

<!-- image -->

&lt;latexi sh

1\_b

64="zPI

kwjA9

7Gd

L

v

C

uc

Q

&gt;

B8X

VD

SgN

EOy

r

fUY

Fo

K

2

m

5

n

R

Z

W

0M

q

+

H

J

3

/

T

p

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="zPI

kwjA9

7Gd

L

v

C

uc

Q

&gt;

B8X

VD

SgN

EOy

r

fUY

Fo

K

2

m

5

n

R

Z

W

0M

q

+

H

J

3

/

T

p

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="

p

zXSg98LKH2Yn

G

y

5

c

I

&gt;A

B

VD

T

MwE

q

R

JU

r

d

PZ

C

O

Q

/

F

3+

N

u

W

m

j

o

v

f

k

7

sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

Figure 2: Structured tree search where nodes are (a) selected and expanded with the LLM suggested transformations, (b) scored by a learned hardware cost model, and (c) updated with performance estimates to guide future search.

COMPILER uses structured search (via MCTS) with learned contextual reasoning (via LLM + CoT) to overcome these challenges. The result is a sample-efficient optimization algorithm capable of discovering performant transformation sequences in high-dimensional, high-interaction spaces.

## 3.2 Structured Optimization via Monte Carlo Tree Search

MCTSas a sample-efficient planner. As described in §2, we cast program optimization as a finitehorizon decision process over the space of transformation sequences. Framing the problem as an MDP allows the REASONING COMPILER to consider long-term optimization effects and leverage planning algorithms such as Monte Carlo tree search (MCTS) to explore this space efficiently.

MCTS operates over a tree T = ⟨ V, E ⟩ where V = P and E = O such that each node p ∈ P is a program from the state space S and each edge o ∈ O corresponds to a transformation from the action space A . This tree structure naturally supports the reuse of common transformation prefixes and allows the planner to backpropagate value estimates from downstream program variants to upstream decisions. Such reuse is critical in compiler optimization, where transformation sequences exhibit both compounding effects and long-range interactions.

Selection via UCT. During the selection phase, MCTS traverses T from the root, recursively selecting child programs p i to maximize the UCT (Upper Confidence bounds applied to Trees) criterion:

<!-- formula-not-decoded -->

where W ( p i ) is the cumulative reward of p i , N ( p i ) is the visit count of node (i.e., program) p i , and c governs the exploration-exploitation tradeoff.

LLM-guided expansion. As shown in Figure 2(a), once a promising leaf node p i is selected, an LLM is queried to propose a transformation conditioned on p i and its ancestors (see §3.1). The model generates a candidate transformation o i +1 ∈ O , which is applied to p i to produce a new program p i +1 = o i +1 ( p i ) . This results in a new node p i +1 added to T corresponding to the updated program and extended transformation path. To ensure T remains acyclic, if p i +1 already exists in the tree, it is not added. By leveraging the LLM's contextual reasoning, the system proposes globally informed transformations that extend beyond myopic heuristics.

Rollout for local reward estimation. As shown in Figure 2(b), once a new node p i +1 is added to the tree, the REASONING COMPILER performs a lightweight MCTS rollout to estimate the longterm impact of the transformation sequence that produced it. This is done by sampling a randomized sequence of legal transformations o 1 , . . . , o q and applying them to obtain a terminal program p sim = ( o q ◦· · · ◦ o 1 )( p i +1 ) . Directly measuring hardware-level performance requires compiling and running on real hardware, which is too expensive for the inner loop of a planning algorithm. Following standard practice in compiler autotuning, the REASONING COMPILER uses a learned, hardwareinformed surrogate ˆ f for f that is cheap to evaluate and accelerates search while preserving final quality [6, 7, 10, 14-16]. We convert this prediction into a rollout reward W ( p i +1 ) = s · ˆ f ( p sim ) , where s ∈ { +1 , -1 } is chosen so that larger values indicate better performance (e.g., s = -1 for latency). This noisy but informative proxy lets MCTS trade off immediate and downstream effects without real-hardware runs.

&lt;latexi sh

1\_b

64="DL+y

M

X

Om

B7

u2G

c5p

w

&gt;A

n

V

JSgN

EK

W

CjP

d

zo

0

T9

k

R

8

/

UH

q

F

f

v

3

Z

r

I

Q

Y

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="XrgK

q2oH

L/W30

j

kA

&gt;

B7

c

V

NS8

EJ

U

9

n

PR

p

y

Z

I

+Q

C

z

uO

D

d

M

m

v

f

F

Y

T

w

5

G

&lt;latexi sh

1\_b

64="3S8QU0jw

LToKBz

X

n

NDM

&gt;A

7

c

V

g

F

2pr

f

Zdu

k

I

v

y

m

Y

JR

P

C

+

/

O

9

G

W

q

5

E

H

&lt;latexi sh

1\_b

64="v0EVI2u

p

Wnz

ASQD

Z93

w

&gt;

B7H

c

N

8

J

Ur q/

L

PR

K

T

m

G

gY

y5

+

M

f

d

o

j

k

XF

O

C

&lt;latexi sh

1\_b

64="rBd/

R0J

TP

Zk

HVWA

5j

w

&gt;

7n

c

DLSgM

F

2p

f

u

I

G

C

Uomz

+

X

9

Y

Q

y

3

N

q

K

v

O

E

8

&lt;latexi sh

1\_b

64="YUp

NXH

KmVRcL

r2

Z8

W3

Go

&gt;A

B7n

D

S

F

f

du vgQk

/

j

T

5

wM

E

I

C

9

OJ

z

y

0

q+

P

&lt;latexi sh

1\_b

64="

Py7

Vv

LpfY

I

qM5

z8

o

&gt;A

B

n

c

D

S

N

F

2

r

Zdu gQk

0

U

m

3

T

+

XCj

9

J

K

O

RW

E

G

w

/

H

&lt;latexi sh

1\_b

64="K92k

Egv

Dw rojq

B

fU

Y

&gt;A

7n

c

V

LS

M

F

3

ZX

W

J

p

Ny

0Q8m

T

I5

H+

G

u/R

O

P

z

d

C

&lt;latexi sh

1\_b

64="gjFIG

m

v8n c+

5

S7

PB

QA

&gt;

VDL

M

2pr

f

Zdu

k

w

W

C

9o

J

O

q

X

T

y0

z

3

R

/

H

NE

K

Y

U

&lt;latexi sh

1\_b

64="J3Zn

LP

or

QUDu7FX

K

c

A

&gt;

B

V

SgN

EOz

G

M

H

Y

8

d

Rj

C

w

m

2

k

/

I

T

q

W

p

5

9

y

f

0

+

v

&lt;latexi sh

1\_b

64="O

NWM

8g o5nPu

+0

zUj

I

&gt;A

B7X

c

VDLS

E

y

r

f

Y9

F

K

2

Q

HJ

mZ

p

/

wd

v

3

k

G

q

T

C

R

&lt;latexi sh

1\_b

64="CwY0K

SO

AIU

EoZ

9

3cR

5

&gt;

B7n

VDL

gM

F

2pr

f

du

k

X8

W

J

m

j

G

Q

q

v

+T

y

z

N

/

P

H

&lt;latexi sh

1\_b

64="zPI

kwjA9

7Gd

L

v

C

uc

Q

&gt;

B8X

VD

SgN

EOy

r

fUY

Fo

K

2

m

5

n

R

Z

W

0M

q

+

H

J

3

/

T

p

&lt;latexi sh

1\_b

64="rE

n

5ACd

Z

0Y

FV

w

R

&gt;

B7

cj

DJSgN

K

XGL

vT

Q

O

9

I

k

Pp

U

8

/

W

q

o

f

2u

m

+z

H

3

y

M

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="DL+y

M

X

Om

B7

u2G

c5p

w

&gt;A

n

V

JSgN

EK

W

CjP

d

zo

0

T9

k

R

8

/

UH

q

F

f

v

3

Z

r

I

Q

Y

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="XrgK

q2oH

L/W30

j

kA

&gt;

B7

c

V

NS8

EJ

U

9

n

PR

p

y

Z

I

+Q

C

z

uO

D

d

M

m

v

f

F

Y

T

w

5

G

&lt;latexi sh

1\_b

64="J

uU30XoCHEj

y

OfY

5G

I

&gt;A

B7n

c

VDLSgM

F

2pr

Zd

k

Q

v

N

8m

T

w

R

P

K

+

/

W

q

9

z

&lt;latexi sh

1\_b

64="3S8QU0jw

LToKBz

X

n

NDM

&gt;A

7

c

V

g

F

2pr

f

Zdu

k

I

v

y

m

Y

JR

P

C

+

/

O

9

G

W

q

5

E

H

&lt;latexi sh

1\_b

64="J3Zn

LP

or

QUDu7FX

K

c

A

&gt;

B

V

SgN

EOz

G

M

H

Y

8

d

Rj

C

w

m

2

k

/

I

T

q

W

p

5

9

y

f

0

+

v

&lt;latexi sh

1\_b

64="V5

r

Cu3

Qc

0

O

J

g

U

&gt;A

GH

DLS

NBE

z

X

FP

oZ

v

y

8

T

7

j

k9

M

qW/

w

p

f

I

d

n

m

2

K

R

Y

+

&lt;latexi sh

1\_b

64="2IZECo

S

uq

7

/

j

X

5B

+U

&gt;A

c

VDL

gM

F

3

rP

JN

QpkR

8

G

d

wT

m

y9

f

0

nY

H

O

W

v

K

z

&lt;latexi sh

1\_b

64="vjFG

or

PLUzA

0

Hnw

&gt;

B7

c

V

NS8

EJ3

q/

9

2

R

Kp

Wy

m

Z

fu

Q

+

D

I

T

Y

MO

dk

X

5

g

C

&lt;latexi sh

1\_b

64="vr/5g0oH

d

uLRJnmcW

p8w

M

&gt;A

B7

VD

S

N

EOy fUY9

P

K

2

Q

I

k

zCZ

T

3

j

F

X

G+

q

&lt;latexi sh

1\_b

64="EMo2Ak3zQ

f9r

D

Y0+

L

U

R

&gt;

C

H

c

V

S

N

FJ

vq

X

u

K

BT

dOpm

G

8

Z7j

5y

W

g

n

I

w

P

/

&lt;latexi sh

1\_b

64="Cfw

G

d3OJ

UR8MZ

L

cKk

&gt;A

B7n

V

NS

E

r

q/

9

0

F

oWy2m

p

I

P

j

zX

Y

H

+

Q

T

D

g

u

5

v

&lt;latexi sh

1\_b

64="3S8QU0jw

LToKBz

X

n

NDM

&gt;A

7

c

V

g

F

2pr

f

Zdu

k

I

v

y

m

Y

JR

P

C

+

/

O

9

G

W

q

5

E

H

&lt;latexi sh

1\_b

64="QDgNO

9w

V

GEcI

+7

J

/

&gt;A

C

3

Z

LS

B

v

m

Y

R

5

FXTjMo nk

0q zf

P

W

y

u

8

r

2

H

K

d

U

&lt;latexi sh

1\_b

64="M

ZdDouw

Y

Ag

O

VrI

8

&gt;

C

X

c

LS

F

3U

2v

W

G

f

JB

zj k0

m

K

T/

9

R

+7

H

n

5

NE

Q

pq

P

y

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="XrgK

q2oH

L/W30

j

kA

&gt;

B7

c

V

NS8

EJ

U

9

n

PR

p

y

Z

I

+Q

C

z

uO

D

d

M

m

v

f

F

Y

T

w

5

G

&lt;latexi sh

1\_b

64="rE

n

5ACd

Z

0Y

FV

w

R

&gt;

B7

cj

DJSgN

K

XGL

vT

Q

O

9

I

k

Pp

U

8

/

W

q

o

f

2u

m

+z

H

3

y

M

&lt;latexi sh

1\_b

64="FQS7

r9EN

uyPd

C

ck

&gt;A

B8X

VDL

g

O

fUY

o

K

2

I5

HJ

mZn

p

/

w

v

3

G

jq

T

W

0

M

z

R

+

&lt;latexi sh

1\_b

64="O

NWM

8g o5nPu

+0

zUj

I

&gt;A

B7X

c

VDLS

E

y

r

f

Y9

F

K

2

Q

HJ

mZ

p

/

wd

v

3

k

G

q

T

C

R

&lt;latexi sh

1\_b

64="

Py7

Vv

LpfY

I

qM5

z8

o

&gt;A

B

n

c

D

S

N

F

2

r

Zdu gQk

0

U

m

3

T

+

XCj

9

J

K

O

RW

E

G

w

/

H

&lt;latexi sh

1\_b

64="K92k

Egv

Dw rojq

B

fU

Y

&gt;A

7n

c

V

LS

M

F

3

ZX

W

J

p

Ny

0Q8m

T

I5

H+

G

u/R

O

P

z

d

C

&lt;latexi sh

1\_b

64="gjFIG

m

v8n c+

5

S7

PB

QA

&gt;

VDL

M

2pr

f

Zdu

k

w

W

C

9o

J

O

q

X

T

y0

z

3

R

/

H

NE

K

Y

U

&lt;latexi sh

1\_b

64="J3Zn

LP

or

QUDu7FX

K

c

A

&gt;

B

V

SgN

EOz

G

M

H

Y

8

d

Rj

C

w

m

2

k

/

I

T

q

W

p

5

9

y

f

0

+

v

&lt;latexi sh

1\_b

64="v0EVI2u

p

Wnz

ASQD

Z93

w

&gt;

B7H

c

N

8

J

Ur q/

L

PR

K

T

m

G

gY

y5

+

M

f

d

o

j

k

XF

O

C

Backpropagation. As shown in Figure 2(c), the estimated reward W ( p i +1 ) is then backpropagated to all ancestors along the path to the root according to the update step W ( p A ) ← W ( p A )+ W ( p i +1 ) where p A is some ancestor program. The visit counts are also updated according to the update step N ( p A ) ← N ( p A ) + 1 . These updates refine the empirical estimates that guide future selections.

## 4 Evaluation

We implement the REASONING COMPILER as an extension to MetaSchedule [8]. The framework introduces three modular components: (1) a prompt generator that serializes the current scheduling state, including the IRModule, transformation trace (i.e., the applied schedule history), and hardware cost model outputs, into structured prompts that capture the textual difference from the base IRModule and reflect the current schedule's performance; (2) an LLM interface that queries an external API (e.g., OpenAI) and parses the LLM's output into candidate transformation sequences; and (3) a tree manager that performs MCTS with selection based on UCT score, expansion using LLM suggested transformations, simulation with a hardware-informed cost model, and backpropagation for tree statistics updates.

## 4.1 Methodology

We evaluate the REASONING COMPILER on five representative computational kernels drawn from production-scale models: (1) a self-attention layer from Llama-3-8B [17], (2) a mixture-of-experts (MoE) layer from DeepSeek-R1 [18], (3) a self-attention layer from FLUX (stable diffusion) [19], (4) a convolution layer from FLUX [19], and (5) an MLP layer from Llama-4-Scout [20]. In addition, we perform an end-to-end evaluation of Llama-3-8B . Compiler optimization is framed as a sequential decision process and guided by MCTS [9] using the Upper Confidence bounds applied to Trees (UCT) criterion [12] with exploration parameter c = √ 2 and branching factor B = 2 following prior work [21, 22]. During search, the LLM (OpenAI GPT-4o mini [23]) is queried using hierarchical context-specifically, the parent and grandparent schedules and their transformations-to enable informed proposal generation. We compare three optimization strategies: (1) TVM MetaSchedule [8], which uses Evolutionary Search ; (2) MCTS without LLM guidance ( MCTS ); and (3) the REASONING COMPILER that uses prompt-based proposal generation ( LLM-Guided MCTS ). All experiments are conducted using Apache TVM v0.20.0 [10, 24]. Our experimental environment is a dedicated Intel Core i9 workstation under a fixed software and hardware stack to isolate scheduling effects. This environment covers all five kernels above and is the ablation environment. To show portability and scalability across consumer and datacenter processors, we evaluate each of the five kernels on five hardware platforms: Amazon Graviton2, AMD EPYC 7R13, Apple M2 Pro, Intel Core i9, and Intel Xeon E3. Each experiment is repeated 20 times, and we report the mean performance to ensure statistical stability. Additionally, we leverage OpenAI and HuggingFace model serving APIs to access the respective models. The implementation is open-sourced.

Figure 3: Relative speedup over pre-optimized code as a function of evaluated transformation proposals. The REASONING COMPILER achieves superior sample efficiency, discovering highquality code with fewer samples across all operators in low-budget regimes.

<!-- image -->

## 4.2 Experimental Results

We assess the sample efficiency of the REASONING COMPILER by analyzing how code quality evolves with increasing search budget, quantified in terms of evaluated transformation proposals. Figure 3 presents results across five representative workloads, encompassing both transformer-style attention layers and convolution-heavy architectures. Across all benchmarks, our method achieves

competitive or superior code performance with significantly fewer samples than state-of-the-art black-box autotuners such as TVM with Evolutionary Search . These results directly support the central hypothesis of our work: leveraging LLM-driven, context-aware reasoning enables more efficient and effective exploration of the compiler optimization space .

Rapid convergence in low-sample regimes. A consistent trend across all benchmarks is the rapid ascent of code quality in the initial stages of search. This early-stage performance is critical in practice, as real-world compiler pipelines often operate under strict tuning time budgets. Figure 3 shows Relative Speedup over Pre-Optimized Code on the y-axis, with the number of evaluated transformation proposals on the x-axis. Speedup is defined as the ratio of the execution time of the unoptimized code to that of the optimized code after tuning. Higher values indicate more efficient and optimized code. For instance, on the Llama-3-8B self-attention layer, the REASONING COMPILER achieves a 7.08 × speedup over the untuned baseline with just 36 samples, whereas TVM with Evolutionary Search requires 72 samples, which is twice the budget to achieve comparable gains. On the Llama4-Scout MLP layer, the gap is even more pronounced: the REASONING COMPILER achieves 12.7 × speedup at 20 samples, while TVM falls short of this mark even after 3000 samples.

Quantitative sample efficiency. To formally quantify sample efficiency, we compare the number of samples required by each method to reach target speedups. On the FLUX self-attention layer, the REASONING COMPILER attains a 2 × speedup using only 36 samples, while TVM with Evolutionary Search requires more than 600 samples, a 16 × reduction in tuning cost. On the FLUX convolution layer, the REASONING COMPILER consistently outperforms TVM across nearly all budget levels and reaches TVM's final performance after evaluating just 400 samples.

Speedup relative to baselines. The REASONING COMPILER not only produces better code, but does so more aggressively and earlier in the search process. For example, on the DeepSeekR1 MoE Layer , the REASONING COMPILER achieves a 3.3 × speedup over TVM with Evolutionary Search at 36 samples; on the Llama-4-Scout MLPlayer, the REASONING COMPILER achieves a 9.3 × speedup over TVM at 20 samples. This trend, which shows strong initial gains followed by conver-

Table 1: Sample efficiency comparison between the REASONING COMPILER and TVM with Evolutionary Search on layer-wise benchmarks across various hardware platforms.

| Hardware Platform   |                            | TVM       | TVM       | REASONING COMPILER   | REASONING COMPILER   | Improvement      | Improvement            |
|---------------------|----------------------------|-----------|-----------|----------------------|----------------------|------------------|------------------------|
| Hardware Platform   | Benchmark                  | # Samples | Speedup # | Samples              | Speedup              | Sample Reduction | Sample Efficiency Gain |
| Amazon Graviton2    | Llama-3-8B Layer           | 510       | 3.9 ×     | 60                   | 5.1 ×                | 8.5 ×            | 11.1 ×                 |
| Amazon Graviton2    | DeepSeek-R1 MoE Layer      | 980       | 2.7 ×     | 150                  | 5.9 ×                | 6.5 ×            | 14.4 ×                 |
| Amazon Graviton2    | FLUX Attention Layer       | 320       | 1.6 ×     | 130                  | 3.0 ×                | 2.5 ×            | 4.6 ×                  |
| Amazon Graviton2    | FLUX Convolution Layer     | 160       | 1.8 ×     | 20                   | 4.1 ×                | 8.0 ×            | 18.2 ×                 |
| Amazon Graviton2    | Llama-4-Scout MLP Layer    | 1,630     | 1.7 ×     | 500                  | 4.0 ×                | 3.3 ×            | 7.7 ×                  |
| AMD EPYC 7R13       | Llama-3-8B Layer           | 1,400     | 2.1 ×     | 200                  | 12.1 ×               | 7.0 ×            | 40.3 ×                 |
| AMD EPYC 7R13       | DeepSeek-R1 MoE Layer      | 2,290     | 1.7 ×     | 330                  | 2.3 ×                | 6.9 ×            | 9.4 ×                  |
| AMD EPYC 7R13       | FLUX Attention Layer       | 2,460     | 1.5 ×     | 230                  | 3.1 ×                | 10.7 ×           | 22.1 ×                 |
| AMD EPYC 7R13       | FLUX Convolution Layer     | 2,520     | 1.3 ×     | 470                  | 4.8 ×                | 5.4 ×            | 19.6 ×                 |
| AMD EPYC 7R13       | Llama-4-Scout MLP Layer    | 510       | 6.4 ×     | 100                  | 10.2 ×               | 5.1 ×            | 8.1 ×                  |
| Apple M2 Pro        | Llama-3-8B Attention Layer | 1,010     | 3.3 ×     | 190                  | 9.7 ×                | 5.3 ×            | 15.6 ×                 |
| Apple M2 Pro        | DeepSeek-R1 MoE Layer      | 1,040     | 2.8 ×     | 230                  | 4.8 ×                | 4.5 ×            | 7.8 ×                  |
| Apple M2 Pro        | FLUX Attention Layer       | 270       | 2.1 ×     | 50                   | 3.7 ×                | 5.4 ×            | 9.5 ×                  |
| Apple M2 Pro        | FLUX Convolution Layer     | 2,260     | 1.5 ×     | 510                  | 5.5 ×                | 4.4 ×            | 16.2 ×                 |
| Apple M2 Pro        | Llama-4-Scout MLP Layer    | 2,460     | 2.2 ×     | 440                  | 3.4 ×                | 5.6 ×            | 8.6 ×                  |
| Intel Core i9       | Llama-3-8B Attention Layer | 920       | 10.5 ×    | 130                  | 11.0 ×               | 7.1 ×            | 7.4 ×                  |
| Intel Core i9       | DeepSeek-R1 MoE Layer      | 1,632     | 9.1 ×     | 192                  | 9.1 ×                | 8.5 ×            | 8.5 ×                  |
| Intel Core i9       | FLUX Attention Layer       | 1,000     | 5.1 ×     | 150                  | 5.4 ×                | 6.7 ×            | 7.0 ×                  |
| Intel Core i9       | FLUX Convolution Layer     | 400       | 2.3 ×     | 72                   | 2.3 ×                | 5.6 ×            | 5.6 ×                  |
| Intel Core i9       | Llama-4-Scout MLP Layer    | 230       | 5.6 ×     | 20                   | 12.7 ×               | 11.5 ×           | 26.1 ×                 |
| Intel Xeon E3       | Llama-3-8B Attention Layer | 2,760     | 3.9 ×     | 320                  | 5.8 ×                | 8.6 ×            | 12.8 ×                 |
| Intel Xeon E3       | DeepSeek-R1 MoE Layer      | 1,000     | 3.7 ×     | 180                  | 4.4 ×                | 5.6 ×            | 6.6 ×                  |
| Intel Xeon E3       | FLUX Attention Layer       | 1,340     | 1.4 ×     | 450                  | 3.4 ×                | 3.0 ×            | 7.1 ×                  |
| Intel Xeon E3       | FLUX Convolution Layer     | 220       | 1.9 ×     | 40                   | 2.2 ×                | 5.5 ×            | 6.4 ×                  |
| Intel Xeon E3       | Llama-4-Scout MLP Layer    | 1,200     | 2.0 ×     | 300                  | 6.1 ×                | 4.0 ×            | 12.2 ×                 |
| Geomean             | -                          | -         | 2.7 ×     | -                    | 5.0 ×                | 5.8 ×            | 10.8 ×                 |

gence, demonstrates that the REASONING COMPILER quickly identifies high-performing regions of the search space, while TVM's uninformed search requires substantial exploration to reach similar quality.

Operator-specific trends. We observe that certain operator types, such as matrix multiplication operations extracted from attention layers and MLP layers, exhibit sharper performance improvements. This is likely due to recurring structural patterns such as loop fusion, tiling, and vectorization, which pretrained LLMs can more readily recognize and exploit. Convolutional operators, by contrast, expose a broader and less regular transformation space. Nonetheless, the REASONING COMPILER consistently matches or exceeds baseline performance with fewer samples, underscoring its effectiveness across diverse operator characteristics.

Sample efficiency across hardware platforms. As shown in Table 1, the REASONING COMPILER demonstrates superior sample efficiency compared to TVM with Evolutionary Search across five hardware platforms on five benchmarks. We define sample efficiency as the speedup achieved per sample ( Speedup # of Samples ). On average, across all 25 platform-operator pairs, the REASONING COMPILER achieves a 5.0 × speedup using 5.8 × fewer samples, resulting in a 10.8 × improvement in sample efficiency. The performance gains are particularly significant for compute-intensive workloads. For instance, for the Llama-3-8B self-attention layer on AMD EPYC 7R13, the REASONING COMPILER achieved a 12.1 × speedup in just 200 samples, while TVM required 1,400 samples to reach a 2.1 × speedup. This represents a 7.0 × sample reduction and a 40.3 × sample efficiency gain. On Intel Core i9, the REASONING COMPILER often matches or exceeds TVM's peak with fewer trials: on the Llama-4-Scout MLP layer, the REASONING COMPILER used 11.5 × fewer samples for a 26.1 × efficiency gain.

Table 2: Sample efficiency comparison between the REASONING COMPILER and TVM with Evolutionary Search on end-to-end Llama-3-8B across various hardware platforms.

|                   | TVM       | TVM     | REASONING COMPILER   | REASONING COMPILER   | Improvement      | Improvement            |
|-------------------|-----------|---------|----------------------|----------------------|------------------|------------------------|
| Hardware Platform | # Samples | Speedup | # Samples            | Speedup              | Sample Reduction | Sample Efficiency Gain |
| Amazon Graviton2  | 4,560     | 3.7 ×   | 1,440                | 5.1 ×                | 3.2 ×            | 4.4 ×                  |
| AMDEPYC7R13       | 410       | 2.0 ×   | 140                  | 2.2 ×                | 2.9 ×            | 3.2 ×                  |
| Apple M2 Pro      | 4,820     | 2.2 ×   | 1,770                | 3.9 ×                | 2.7 ×            | 4.8 ×                  |
| Intel Core i9     | 3,800     | 2.2 ×   | 720                  | 4.9 ×                | 5.3 ×            | 11.8 ×                 |
| Intel Xeon E3     | 4,640     | 5.0 ×   | 670                  | 5.0 ×                | 6.9 ×            | 6.9 ×                  |
| Geomean           | -         | 2.8 ×   | -                    | 4.0 ×                | 3.9 ×            | 5.6 ×                  |

End-to-end sample efficiency. For end-to-end Llama-3-8B across the five hardware platforms in Table 2, the REASONING COMPILER 's sample efficiency improvement over TVM ranges from 3.2 × on AMD EPYC to 11.8 × on Intel Core i9. End-to-end speedups range from 2.2 × on AMD EPYC to 5.1 × on Amazon Graviton2. The REASONING COMPILER consistently achieves significantly higher speedups: using 3.9 × fewer samples, it achieves a 4.0 × speedup and yields a 5.6 × geometric-mean sample efficiency improvement over TVM with Evolutionary Search .

Implications. These findings reinforce our core thesis: compiler optimization should be cast as a structured decision process, enriched by prior knowledge and contextual reasoning. Our integration of LLMs into Monte Carlo tree search results in a strategically guided and sample-efficient search, particularly valuable in scenarios with constrained tuning budgets. By generating performant code with orders-of-magnitude fewer samples, our framework offers both practical deployment advantages and a compelling alternative to conventional, sample-inefficient compilation pipelines.

## 4.3 Ablation Studies

## 4.3.1 Impact of LLM Choice and Reasoning Strategy

To better understand the contributions of different components in our approach, we conduct an ablation study focused on the effects of LLM selection and reasoning modality. Figure 4(a) shows the relative speedup over unoptimized code as a function of the number of schedule samples evaluated by the REASONING COMPILER on the Llama-3-8B self-attention layer using a range of LLM models

Figure 4: Ablation studies for the Llama-3-8B self-attention layer. (a) Comparing different LLMs as proposal engines shows stronger LLMs lead to faster convergence. (b) Increasing the prompt's historical trace depth improves sample efficiency.

<!-- image -->

<!-- image -->

for API calls. The x-axis indicates the cumulative number of schedules explored, while the y-axis shows the best speedup achieved so far. This setup enables us to directly compare how effectively various LLMs leverage contextual information to guide the search. The general trend of the results supports our central claim: compiler optimization benefits from goal-directed, context-aware reasoning in terms of sample efficiency. Below, we discuss the specific behaviors that exemplify different reasoning strategies.

Large instruction-tuned Llama3.3 (70B) achieves exceptional sample efficiency . The instruction-tuned Llama3.3-70B model rapidly attains near-optimal performance, reaching a 9.69 × speedup after only 36 samples, roughly 86% of the GPT-4o mini's maximum speedup but with less than 6% of its sampling budget. This corresponds to an approximately 15 × improvement in sample efficiency. Instruction tuning also significantly improves the ability of LLMs to generate domain-specific, context-aware transformation proposals. The consistent performance advantage of instruction-tuned models over untuned counterparts of comparable size confirms that semantic task alignment, combined with sufficient model capacity, synergistically enhances the effectiveness of sequential context reasoning in guiding compiler optimizations.

DeepSeek-R1-Distill-Qwen (32B) excels in long-horizon optimization. The DeepSeek-R1Distill-Qwen-32B model, employing a mixture-of-experts (MoE) architecture, exhibits a more gradual improvement, starting with a 7.07 × speedup at 18 samples and reaching 9.98 × after 579 samples. The sparse expert routing inherent in MoE architectures likely facilitates exploration of complex transformation sequences over extended horizons, complementing context-aware reasoning by enabling specialized and conditional decision-making.

Lower-parameter models also achieve high sample efficiency. Despite their reduced scale, smaller models still produce notable speedups relative to the untuned baseline. For example, Llama3.1-Instruct (8B) reaches a 5.87 × speedup, and DeepSeek-R1-Distill-Qwen (7B) achieves a 4.86 × speedup at just 36 samples. When compared to the widely used Evolutionary Search strategy, which requires around 72 samples to achieve a 7.0 × speedup and fails to reach comparable performance for tuning the DeepSeek-R1 MoE layer. Even after 3000 samples, these smaller models consistently outperform. The REASONING COMPILER with lower-parameter models achieves at least twice the sample efficiency of TVM with Evolutionary Search , making them well-suited for efficient compiler optimization in local or edge deployments.

Open-source models match proprietary models in performance. Our results demonstrate that open-source LLMs, when adequately scaled and instruction-tuned, match or exceed the performance of proprietary baselines such as GPT-4o mini. This underscores the broad applicability of our approach and its independence from proprietary data or architectures, enabling widespread adoption of context-aware, LLM-guided compiler optimization.

## 4.3.2 Impact of Historical Trace Depth on Optimization Efficiency

Figure 4(b) presents the relative speedup over unoptimized code as a function of the number of schedule samples evaluated by the REASONING COMPILER on the Llama-3-8B self-attention layer. Using a deeper historical trace (see Figure 1) in the prompt (parent + grandparent + greatgrandparent) leads to faster convergence compared to the shallower trace (parent + grandparent). For example, at 36 samples, the deeper trace achieves a speedup of approximately 7.13 × , slightly surpassing the 7.08 × of the shallower trace. However, by 72 samples, the deeper trace saturates at 11.36 × speedup, while the shallower trace reaches only 8.38 × , requiring many more samples

(around 579) to approach 11.3 × performance. This demonstrates that including longer historical context enables the LLM to better capture dependencies and synergies in transformation sequences, resulting in more sample-efficient and goal-directed exploration, validating the advantage of context-aware reasoning.

## 5 Related Work

Superoptimization. While our high-level goal of discovering highly efficient program variants shares motivation with the superoptimization literature, our formulation and tractability differ substantially. Superoptimization aims to find the globally optimal instruction-level program, typically via enumerative [1, 3], symbolic [2], or stochastic [4] search over low-level assembly variants; hybrid [25] and neural [26] approaches have also been explored. STOKE [4] showed that highquality programs often reside in low-probability regions and made the leap to use randomized search (MCMC). Neural compilers followed suit and relied on evolutionary search or simulated annealing algorithms [5-8]. In contrast, the REASONING COMPILER treats optimization as a planning problem that leverages MCTS to reason contextually about dependencies among transformations over structured intermediate representations.

ML-Based Autotuning. Autotuning frameworks optimize performance-critical parameters (e.g., loop tile sizes, phase orderings, memory layouts) using a variety of ML-based techniques, including linear models [27, 28], tree-based methods [29, 30], Bayesian networks [31, 32], evolutionary algorithms [29, 33, 34], clustering [28, 34], and reinforcement learning [11, 33-35]. The REASONING COMPILER shares the same goal of performance-driven parameter selection, but distinguishes itself by combining LLM-based contextual reasoning with structured search (via MCTS) to explore transformation sequences in a history- and structure-aware manner.

Techniques for Neural Compilation. A large body of work targets the optimization of neural network inference pipelines, spanning graph-level transformations and scheduling [36-43] and lowlevel code generation [5, 10, 44-49]. Many modern systems incorporate learned components (e.g., cost models) and search strategies to navigate large configuration spaces; for example, TVM/Ansor [6, 7, 10] and FlexTensor [50] use learned performance models and evolutionary strategies for tuning. While highly effective for tensor-program tuning, these approaches often emphasize local parameter optimization or rely on domain-specific heuristics. The REASONING COMPILER moves beyond these works by introducing LLM-based contextual reasoning over transformation history, structural changes, and performance trends, enabling history- and structure-aware exploration not addressed in prior neural compilation work.

LLMs for Code Reasoning and Optimization. LLMs have demonstrated capabilities in code generation [51-56], fuzzing [57], bug repair [58], and even high-level optimization [59]. Recent work has explored the use of LLMs to generate phase orderings or perform disassembly [60, 61]. The REASONING COMPILER advances these approaches by embedding an LLM in a structured decision loop, leveraging it for context-aware reasoning within a grounded search process.

## 6 Conclusion

Compiling neural workloads remains a bottleneck for scalable model serving: traditional compilers struggle with combinatorial transformation spaces, and the state-of-the-art neural compilers rely on stochastic search, lacking sample efficiency and contextual awareness. This paper introduced the REASONING COMPILER , a novel framework that formulates compiler optimization as a sequential, context-aware decision process, pairing LLM-generated proposals with MCTS and performance feedback to reason and navigate through the optimization space efficiently. By enabling LLM reasoning in the compiler optimization process, we achieve a leap from randomized search to informed and guided compilation. Our results show that the REASONING COMPILER consistently yields faster runtimes with markedly fewer evaluations without any retraining. These gains directly translate to reduced operational cost of LLM services, lower energy usage per query, improved system responsiveness, more agile model deployment, faster model training, and accelerated innovation cycles, among other benefits. Looking ahead, the same LLM that guides compilation can accelerate its own inferencing, creating a virtuous, self-optimizing cycle in which sped-up LLMs enable more efficient transformations and progressively better models and services.

## Acknowledgments

We thank the anonymous reviewers for their insightful feedback. This work was in part supported by the National Science Foundation (NSF) award CCF #2107598. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes not withstanding any copyright notation thereon. The views contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied by the U.S. Government.

## References

- [1] Henry Massalin. Superoptimizer: a look at the smallest program. In ASPLOS , 1987.
- [2] Rajeev Joshi, Greg Nelson, and Keith Randall. Denali: a goal-directed superoptimizer. In PLDI , 2002.
- [3] Sorav Bansal and Alex Aiken. Automatic generation of peephole superoptimizers. In ASPLOS , 2006.
- [4] Eric Schkufza, Rahul Sharma, and Alex Aiken. Stochastic superoptimization. In ASPLOS , 2013.
- [5] Nicolas Vasilache, Oleksandr Zinenko, Theodoros Theodoridis, Priya Goyal, Zachary DeVito, William S. Moses, Sven Verdoolaege, Andrew Adams, and Albert Cohen. Tensor Comprehensions: Framework-agnostic high-performance machine learning abstractions. arXiv , 2018.
- [6] Tianqi Chen, Lianmin Zheng, Eddie Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. Learning to optimize tensor programs. In NeurIPS , 2018.
- [7] Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, and Ion Stoica. Ansor: Generating high-performance tensor programs for deep learning. In OSDI , 2020.
- [8] Junru Shao, Xiyou Zhou, Siyuan Feng, Bohan Hou, Ruihang Lai, Hongyi Jin, Wuwei Lin, Masahiro Masuda, Cody Hao Yu, and Tianqi Chen. Tensor program optimization with probabilistic programs. In NeurIPS , 2022.
- [9] Cameron B Browne, Edward Powley, Daniel Whitehouse, Simon M Lucas, Peter I Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. A survey of Monte Carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in games , 4(1):1-43, 2012.
- [10] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. TVM: An automated end-to-end optimizing compiler for deep learning. In OSDI , 2018.
- [11] Byung Hoon Ahn, Prannoy Pilligundla, and Hadi Esmaeilzadeh. Chameleon: Adaptive code optimization for expedited deep neural network compilation. In ICLR , 2020.
- [12] Levente Kocsis and Csaba Szepesvári. Bandit based Monte-Carlo planning. In ECML , 2006.
- [13] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS , 2022.
- [14] Minjia Zhang, Menghao Li, Chi Wang, and Mingqin Li. DynaTune: Dynamic tensor program optimization in deep neural network compilation. In ICLR , 2021.
- [15] Byung Hoon Ahn, Sean Kinzer, and Hadi Esmaeilzadeh. Glimpse: Mathematical embedding of hardware specification for neural compilation. In DAC , 2022.
- [16] Perry Gibson and José Cano. Transfer-Tuning: Reusing auto-schedules for efficient tensor program code generation. In PACT , 2023.
- [17] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, et al. The Llama 3 herd of models. arXiv , 2024.
- [18] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv , 2025.
- [19] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [20] Meta. The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation. https://ai.meta.com/blog/llama-4-multimodal-intelligence/, 2025.

- [21] Rémi Coulom. Efficient selectivity and backup operators in Monte-Carlo tree search. In CG , 2007.
- [22] Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning , 47(2-3):235-256, 2002.
- [23] OpenAI. GPT-4o mini API. https://platform.openai.com/docs/models/gpt-4o-mini.
- [24] Apache TVM Community. Apache TVM v0.20.0. https://github.com/apache/tvm/releases/tag/ v0.20.0, 2025.
- [25] Phitchaya Mangpo Phothilimthana, Aditya Thakur, Rastislav Bodik, and Dinakar Dhurjati. Scaling up superoptimization. In ASPLOS , 2016.
- [26] Rudy Bunel, Alban Desmaison, M. Pawan Kumar, Philip H. S. Torr, and Pushmeet Kohli. Learning to superoptimize programs. In ICLR , 2017.
- [27] Mark Stephenson and Saman Amarasinghe. Predicting unroll factors using supervised classification. In CGO , 2005.
- [28] Amir H. Ashouri, Andrea Bignoli, Gianluca Palermo, Cristina Silvano, Sameer Kulkarni, and John Cavazos. MiCOMP: Mitigating the compiler phase-ordering problem using optimization sub-sequences and machine learning. ACM Trans. Archit. Code Optim. , 14(3), 2017.
- [29] Douglas Simon, John Cavazos, Christian Wimmer, and Sameer Kulkarni. Automatic construction of inlining heuristics using machine learning. In CGO , 2013.
- [30] Ameer Haj-Ali, Hasan Genc, Qijing Huang, William Moses, John Wawrzynek, Krste Asanovi´ c, and Ion Stoica. ProTuner: Tuning programs with Monte Carlo tree search. arXiv , 2020.
- [31] Amir Hossein Ashouri, Giovanni Mariani, Gianluca Palermo, Eunjung Park, John Cavazos, and Cristina Silvano. COBAYN: Compiler autotuning framework using Bayesian networks. ACM Trans. Archit. Code Optim. , 13(2), 2016.
- [32] Erik Orm Hellsten, Artur Souza, Johannes Lenfers, Rubens Lacouture, Olivia Hsu, Adel Ejjeh, Fredrik Kjolstad, Michel Steuwer, Kunle Olukotun, and Luigi Nardi. BaCO: A fast and portable Bayesian compiler optimization framework. In ASPLOS , 2023.
- [33] Mircea Trofin, Yundi Qian, Eugene Brevdo, Zinan Lin, Krzysztof Choromanski, and David Li. MLGO: a machine learning guided compiler optimizations framework. arXiv , 2021.
- [34] Haolin Pan, Yuanyu Wei, Mingjie Xing, Yanjun Wu, and Chen Zhao. Towards efficient compiler auto-tuning: Leveraging synergistic search spaces. In CGO , 2025.
- [35] Ameer Haj-Ali, Qijing Jenny Huang, John Xiang, William Moses, Krste Asanovic, John Wawrzynek, and Ion Stoica. AutoPhase: Juggling HLS phase orderings in random forests with deep reinforcement learning. In MLSys , 2020.
- [36] Moshe Looks, Marcello Herreshoff, DeLesley Hutchins, and Peter Norvig. Deep learning with dynamic computation graphs. arXiv , 2017.
- [37] Zhihao Jia, Oded Padon, James Thomas, Todd Warszawski, Matei Zaharia, and Alex Aiken. TASO: Optimizing deep learning computation with automatic generation of graph substitutions. In SOSP , 2019.
- [38] Yizhi Liu, Yao Wang, Ruofei Yu, Mu Li, Vin Sharma, and Yida Wang. Optimizing CNN model inference on CPUs. In ATC , 2019.
- [39] Yanqi Zhou, Sudip Roy, Amirali Abdolrashidi, Daniel Wong, Peter Ma, Qiumin Xu, Hanxiao Liu, Mangpo Phitchaya Phothilimtha, Shen Wang, Anna Goldie, Azalia Mirhoseini, and James Laudon. Transferable graph optimizers for ML compilers. In NeurIPS , 2020.
- [40] Yaoyao Ding, Ligeng Zhu, Zhihao Jia, Gennady Pekhimenko, and Song Han. IOS: Interoperator scheduler for CNN acceleration. In MLSys , 2021.
- [41] Zhen Zheng, Pengzhan Zhao, Guoping Long, Feiwen Zhu, Kai Zhu, Wenyi Zhao, Lansong Diao, Jun Yang, and Wei Lin. FusionStitching: Boosting memory intensive computations for deep learning workloads. arXiv , 2021.
- [42] Yichen Yang, Phitchaya Phothilimthana, Yisu Wang, Max Willsey, Sudip Roy, and Jacques Pienaar. Equality saturation for tensor graph superoptimization. In MLSys , 2021.
- [43] Jie Zhao, Xiong Gao, Ruijie Xia, Zhaochuang Zhang, Deshi Chen, Lei Chen, Renwei Zhang, Zhen Geng, Bin Cheng, and Xuefeng Jin. Apollo: Automatic partition-based operator fusion through layer by layer optimization. In MLSys , 2022.
- [44] Riyadh Baghdadi, Jessica Ray, Malek Ben Romdhane, Emanuele Del Sozzo, Abdurrahman Akkas, Yunming Zhang, Patricia Suriana, Shoaib Kamil, and Saman Amarasinghe. Tiramisu: A polyhedral compiler for expressing fast and portable code. In CGO , 2019.
- [45] Bastian Hagedorn, Archibald Samuel Elliott, Henrik Barthels, Rastislav Bodik, and Vinod Grover. Fireiron: A data-movement-aware scheduling language for GPUs. In PACT , 2020.

- [46] Jian Weng, Animesh Jain, Jie Wang, Leyuan Wang, Yida Wang, and Tony Nowatzki. UNIT: Unifying tensorized instruction compilation. In CGO , 2021.
- [47] Rui Li, Yufan Xu, Aravind Sukumaran-Rajam, Atanas Rountev, and P. Sadayappan. Analytical characterization and design space exploration for optimization of CNNs. In ASPLOS , 2021.
- [48] Wookeun Jung, Thanh Tuan Dao, and Jaejin Lee. DeepCuts: A deep learning optimization framework for versatile GPU workloads. In PLDI , 2021.
- [49] Yaoyao Ding, Cody Hao Yu, Bojian Zheng, Yizhi Liu, Yida Wang, and Gennady Pekhimenko. Hidet: Task-mapping programming paradigm for deep learning tensor programs. In ASPLOS , 2023.
- [50] Size Zheng, Yun Liang, Shuo Wang, Renze Chen, and Kaiwen Sheng. FlexTensor: An automatic schedule exploration and optimization framework for tensor computation on heterogeneous system. In ASPLOS , 2020.
- [51] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. Code Llama: Open foundation models for code. arXiv , 2023.
- [52] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. StarCoder 2 and The Stack v2: The next generation. arXiv , 2024.
- [53] Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Empowering code generation with OSS-instruct. arXiv , 2023.
- [54] Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Lei Shen, Zihan Wang, Andi Wang, Yang Li, et al. CodeGeeX: A pre-trained model for code generation with multilingual benchmarking on HumanEval-X. In KDD , 2023.
- [55] Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and Steven CH Hoi. CodeT5+: Open code large language models for code understanding and generation. arXiv , 2023.
- [56] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. DeepSeek-Coder: When the large language model meets programming-the rise of code intelligence. arXiv , 2024.
- [57] Chunqiu Steven Xia, Matteo Paltenghi, Jia Le Tian, Michael Pradel, and Lingming Zhang. Universal fuzzing via large language models. arXiv , 2023.
- [58] Chunqiu Steven Xia, Yuxiang Wei, and Lingming Zhang. Automated program repair in the era of large pre-trained language models. In ICSE , 2023.
- [59] Alexander Shypula, Aman Madaan, Yimeng Zeng, Uri Alon, Jacob R. Gardner, Yiming Yang, Milad Hashemi, Graham Neubig, Parthasarathy Ranganathan, Osbert Bastani, and Amir Yazdanbakhsh. Learning performance-improving code edits. In ICLR , 2024.
- [60] Chris Cummins, Volker Seeker, Dejan Grubisic, Mostafa Elhoushi, Youwei Liang, Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Kim Hazelwood, Gabriel Synnaeve, and Hugh Leather. Large language models for compiler optimization. arXiv , 2023.
- [61] Chris Cummins, Volker Seeker, Dejan Grubisic, Baptiste Roziere, Jonas Gehring, Gabriel Synnaeve, and Hugh Leather. LLM Compiler: Foundation language models for compiler optimization. In CC , 2025.

## A LLMPrompt Example

Below we show an example prompt used in our LLM-Guided MCTS framework (refer to Figure 1).

## Example Code to be Optimized:

```
@tvm.script.ir_module class MyModule: @T.prim_func def main( A: T.Buffer((1, 16, 7168), "float32"), B: T.Buffer((7168, 2048), "float32"), C: T.Buffer((1, 16, 2048), "float32"), ): T.func_attr({"global_symbol": "main", "tir.noalias": True}) for b, t, j, k in T.grid(1, 16, 2048, 7168): with T.block("moe"): vb, vt, vj, vk = T.axis.remap("SSSR", [b, t, j, k]) with T.init(): C[vb, vt, vj] = T.float32(0) C[vb, vt, vj] += A[vb, vt, vk] * B[vk, vj]
```

## Prompt

```
You are a code optimization assistant performing Monte Carlo Tree Search (MCTS) on a given code to improve performance. Each code has a corresponding history of transformations and predicted cost. You are given the code of the selected node and its immediate parent. Their main differences include: Loop shapes: Current: for b_1, t_1, j_1 in T.grid(1, 2, 8), and then j_2_init in T.grid(1), and later j_2 in T.grid(1). Index example: vj = ... + j_1 * 64 + j_2 * 64 Parent: for b_1, t_1, j_1 in T.grid(1, 2, 2), then j_2_init in T.grid(4), and later j_2 in T.grid(4). Index example: vj = ... + j_1 * 256 + j_2 * 64 Tile decisions: Current: sample_perfect_tile(..., decision=[4, 8, 1, 64]) Parent: sample_perfect_tile(..., decision=[4, 2, 4, 64]) Performance estimates: Current: 0.773 Parent: 0.313 Available transformations: TileSize, Parallel, ComputeLocation, Unroll Task Analyze the IR, trace, and predicted scores. Then propose a sequence of transformations (you may repeat any) to potentially improve performance. Output your reasoning and your suggested transformations. For example, your answer should be in the following format: Reasoning: This code still has large loop extents, so I'd tile it twice differently, then unroll. Transformations to apply: TileSize, TileSize, Unroll.
```

## LLM Response

Reasoning: The current schedule tiles the j-axis as 2048=4×8×1×64, whereas the parent uses 4×2×4×64. The higher predicted score of the current node (0.773 vs. 0.313) suggests that expanding outer parallel granularity (j\_1=8) helped, but collapsing j\_2 to 1 removed a small register-level tile, limiting unrolling and hurting reuse within each j\_3=64 chunk.

I would (1) reintroduce a non-unit j\_2 while shrinking j\_1 so that j\_1×j\_2 =8; for instance, [4,4,2,64] keeps enough outer tasks but creates a j\_2×j\_3=128-wide microtile that unrolls cleanly. (2) Tile the 7168-step k-reduction to align with j\_3=64, e.g., 7168 = 7×16×64; the 16-step inner slice can be unrolled to cut loop overhead and improve register reuse across the j\_3 tile. (3) Move the compute of block "moe" to the j\_1 level so each j\_1 tile initializes/accumulates C locally, shortening live ranges and reducing write-back pressure. (4) Parallelize the outer j\_1 (or t\_1 if j\_1 becomes too small) to utilize cores without touching the reduction. (5) Unroll the small j\_2 loop and the k-inner microtile. These changes retain the beneficial outer structure of the current node while restoring a compact inner tile for ILP and cache locality, which should further improve throughput.

Transformations to apply: TileSize, TileSize, ComputeLocation, Parallel, Unroll, Unroll.

## B Sample Efficiency and Speedup Results

Table 3 presents the relative speedup of three methodsEvolutionary Search , MCTS , and the REASONING COMPILER -evaluated across the different benchmarks. Speedup is measured as the ratio of execution time for the unoptimized code to that of the optimized code after applying a given number of transformation proposals. The table captures performance as a function of the number of samples explored. Higher values indicate more effective optimization. For instance, the REASONING COMPILER consistently achieves higher speedups with fewer samples, demonstrating superior sample efficiency and faster convergence compared to MCTS and Evolutionary Search . This table corresponds to Figure 3 in the paper.

Table 3: Speedup over unoptimized code across varying numbers of samples for different compiler optimization methods.

| Number of Samples   |    18 |    36 |     72 |    144 |    192 |    600 |     900 |    1632 |    5952 |
|---------------------|-------|-------|--------|--------|--------|--------|---------|---------|---------|
| Evolutionary Search |  4.67 |  5.7  |   7.74 |   7.98 |   9.4  |   9.54 |   11.2  |   12.04 |   13.18 |
| MCTS                |  4.14 |  4.68 |   8.11 |   8.5  |   9.66 |   9.94 |   11.79 |   12.44 |   12.63 |
| REASONING COMPILER  |  4.52 |  7.08 |   8.38 |   8.79 |  10.56 |  11.33 |   12.1  |   12.57 |   12.87 |
| Number of Samples   | 36    | 54    |  72    | 144    | 192    | 600    |  900    | 1632    | 3000    |
| Evolutionary Search |  2.11 |  2.27 |   3.9  |   4.1  |   5.07 |   6.6  |    6.62 |    9.31 |    9.13 |
| MCTS                |  5.93 |  6.33 |   6.79 |   6.87 |   6.93 |   7.24 |    8.18 |    8.84 |    8.9  |
| REASONING COMPILER  |  7.05 |  7.33 |   8.34 |   8.53 |   9.1  |   9.45 |   11.06 |   11.74 |   11.74 |
| Number of Samples   | 36    | 54    |  72    | 150    | 200    | 600    | 1000    | 1500    | 3000    |
| Evolutionary Search |  2.22 |  2.44 |   2.73 |   2.73 |   4.64 |   4.71 |    5.11 |    5.61 |    5.58 |
| MCTS                |  3.62 |  3.79 |   3.85 |   4.04 |   4.37 |   5.09 |    5.56 |    5.4  |    5.64 |
| REASONING COMPILER  |  4.48 |  4.67 |   4.89 |   5.37 |   5.42 |   5.43 |    5.59 |    5.6  |    5.67 |
| Number of Samples   | 36    | 72    | 150    | 200    | 400    | 600    | 1000    | 1600    | 3000    |
| Evolutionary Search |  2.08 |  2.11 |   2.15 |   2.19 |   2.29 |   2.32 |    2.44 |    2.45 |    2.55 |
| MCTS                |  2.11 |  2.13 |   2.18 |   2.37 |   2.38 |   2.38 |    2.44 |    2.44 |    2.45 |
| REASONING COMPILER  |  2.21 |  2.3  |   2.29 |   2.36 |   2.47 |   2.47 |    2.51 |    2.55 |    2.58 |
| Number of Samples   | 20    | 50    | 100    | 250    | 400    | 600    | 1000    | 1500    | 3000    |
| Evolutionary Search |  1.36 |  2.28 |   3.61 |   5.59 |   5.59 |   5.75 |    5.76 |    5.94 |    5.94 |
| MCTS                |  1.76 |  2.51 |   4.05 |   5.41 |   7.83 |   8.13 |    8.58 |    8.9  |    8.9  |
| REASONING COMPILER  | 12.74 | 12.74 |  12.74 |  12.75 |  12.75 |  13.24 |   13.26 |   13.52 |   13.79 |

## C Impact of LLM Choice and Reasoning Strategy

As a continuation of Figure 4(a), Table 4 reports speedup over unoptimized code on three additional benchmarks: DeepSeek-R1 MoE Layer , FLUX Attention Layer , and FLUX Convolution Layer . Each block of the table corresponds to a different benchmark and shows the best speedup achieved by the REASONING COMPILER as a function of the number of schedules sampled using the reasoning model listed in the table. Rows compare different reasoning models used for API call generation, including both proprietary (e.g., GPT-4o mini, OpenAI o1-mini) and open-source models (e.g., Llama3.3-Instruct, DeepSeek-Distill). Across all benchmarks, the results show that more capable models-those that are larger or instruction-tuned-consistently achieve higher speedups with fewer samples. For example, Llama3.3-Instruct (70B) and DeepSeek-Distill (32B) achieve near-maximal speedup within the first 72-150 samples, while smaller models such as DeepSeek-Distill (7B) or Llama3.1-Instruct (8B) reach similar performance more gradually. These results validate the generality of our findings: the use of context-aware LLMs accelerates convergence of the REASONING COMPILER across diverse code domains. Moreover, the performance of open-source models is competitive with proprietary alternatives, further supporting the accessibility and reproducibility of our method.

Table 4: Speedup over unoptimized code across varying numbers of samples for different choices of API call models.

| Number of Samples           |    18 |    36 |    72 |    150 |    200 |    600 |
|-----------------------------|-------|-------|-------|--------|--------|--------|
| GPT-4o mini                 |  4.52 |  7.08 |  8.38 |   8.79 |  10.56 |  11.33 |
| OpenAI o1-mini              |  4.63 |  4.64 |  7.37 |   9.14 |   9.15 |  11.77 |
| Llama3.3-Instruct (70B)     |  5.15 |  9.68 |  9.69 |   9.8  |   9.8  |   9.81 |
| DeepSeek-Distill-Qwen (32B) |  7.07 |  8.14 |  8.23 |   8.77 |   8.78 |   9.98 |
| Llama3.1-Instruct (8B)      |  3.6  |  5.87 |  6.28 |   8.46 |   8.63 |  10.52 |
| DeepSeek-Distill-Qwen (7B)  |  4.06 |  4.86 |  6.68 |   6.82 |   7.94 |  11.58 |
| Number of Samples           | 18    | 36    | 72    | 150    | 200    | 600    |
| GPT-4o mini                 |  6.14 |  7.05 |  8.33 |   8.53 |   9.1  |   9.45 |
| OpenAI o1-mini              |  4.56 |  6.65 |  8.59 |   9.29 |  10.55 |  11.56 |
| Llama3.3-Instruct (70B)     |  7.3  |  7.7  |  7.96 |   8.06 |   8.6  |   9.22 |
| DeepSeek-Distill-Qwen (32B) |  5.56 |  8.11 |  9.49 |  10.17 |  11.02 |  12.02 |
| Llama3.1-Instruct (8B)      |  4.29 |  4.31 |  6.98 |   8.7  |   9.18 |   9.21 |
| DeepSeek-Distill-Qwen (7B)  |  6.89 |  7.35 |  7.35 |  10.22 |  10.34 |  10.44 |
| Number of Samples           | 18    | 36    | 72    | 150    | 200    | 600    |
| GPT-4o mini                 |  4.09 |  4.48 |  4.89 |   5.37 |   5.42 |   5.43 |
| OpenAI o1-mini              |  3.29 |  2.99 |  5.27 |   5.53 |   5.65 |   5.67 |
| Llama3.3-Instruct (70B)     |  2.67 |  3.12 |  4.82 |   4.86 |   5.71 |   5.71 |
| DeepSeek-Distill-Qwen (32B) |  3.56 |  4.29 |  4.29 |   4.54 |   4.99 |   5.21 |
| Llama3.1-Instruct (8B)      |  2.01 |  3.43 |  3.55 |   3.8  |   3.87 |   5.21 |
| DeepSeek-Distill-Qwen (7B)  |  3.02 |  3.76 |  3.83 |   4.54 |   4.94 |   5.17 |
| Number of Samples           | 18    | 36    | 72    | 150    | 200    | 600    |
| GPT-4o mini                 |  1.65 |  2.21 |  2.3  |   2.29 |   2.36 |   2.47 |
| OpenAI o1-mini              |  2.37 |  2.37 |  2.38 |   2.39 |   2.45 |   2.54 |
| Llama3.3-Instruct (70B)     |  2.3  |  2.35 |  2.47 |   2.51 |   2.56 |   2.57 |
| DeepSeek-Distill-Qwen (32B) |  1.41 |  2.26 |  2.32 |   2.35 |   2.4  |   2.45 |
| Llama3.1-Instruct (8B)      |  2.11 |  2.3  |  2.39 |   2.55 |   2.55 |   2.56 |
| DeepSeek-Distill-Qwen (7B)  |  1.56 |  2.18 |  2.42 |   2.44 |   2.46 |   2.45 |

## D Impact of Historical Trace Depth on Optimization Efficiency

As a continuation of Figure 4(b), Table 5 presents the data for the ablation study on the depth of historical trace included in the prompt sent to the LLM. Specifically, we compare two configurations: the 'Parent + Grandparent' setting, where the prompt contains information from the current node and its two immediate ancestors, and the 'Parent + Grandparent + Great-Grandparent' setting, where the prompt additionally includes the great-grandparent node. These variations allow us to assess the impact of deeper context windows on the effectiveness of the REASONING COMPILER .

Results show that increasing the historical context generally improves sample efficiency across all benchmarks. For example, on DeepSeek-R1 MoE Layer , adding one more ancestral node boosts early performance significantly, achieving a 9.39 × speedup at just 18 samples compared to 6.14 × for the shallower context. Similarly, on Llama-3-8B Attention Layer , the extended context leads to a higher final speedup (11.87 × vs. 11.33 × ) and earlier convergence. The performance gains, while smaller, are also consistent on FLUX Attention Layer and FLUX Convolution Layer , with improvements observed across all sample budgets. These findings confirm that providing richer historical context enables the LLM to make more informed decisions at each step of the search, ultimately enhancing the sample efficiency of the REASONING COMPILER

Table 5: Speedup over unoptimized code across varying numbers of samples for different context lengths.

| Number of Samples                        |    18 |    36 |    72 |    150 |    200 |    600 |
|------------------------------------------|-------|-------|-------|--------|--------|--------|
| Parent + Grandparent                     |  4.52 |  7.08 |  8.38 |   8.79 |  10.56 |  11.33 |
| Parent + Grandparent + Great-Grandparent |  3.63 |  7.13 | 11.36 |  11.86 |  11.86 |  11.87 |
| Number of Samples                        | 18    | 36    | 72    | 150    | 200    | 600    |
| Parent + Grandparent                     |  6.14 |  7.05 |  8.33 |   8.53 |   9.1  |   9.45 |
| Parent + Grandparent + Great-Grandparent |  9.39 | 10.31 | 10.31 |  10.49 |  10.59 |  10.65 |
| Number of Samples                        | 18    | 36    | 72    | 150    | 200    | 600    |
| Parent + Grandparent                     |  4.09 |  4.48 |  4.89 |   5.37 |   5.42 |   5.43 |
| Parent + Grandparent + Great-Grandparent |  4.21 |  4.55 |  4.81 |   5.47 |   5.53 |   5.61 |
| Number of Samples                        | 18    | 36    | 72    | 150    | 200    | 600    |
| Parent + Grandparent                     |  1.65 |  2.21 |  2.3  |   2.29 |   2.36 |   2.47 |
| Parent + Grandparent + Great-Grandparent |  1.73 |  2.22 |  2.32 |   2.35 |   2.49 |   2.5  |

## E Ablations of MCTS Branching Factor

To determine the value of MCTS branching factor ( B ), we ablate on B = 2 and B = 4 . In Table 6, results show that when branching factor B = 2 , the REASONING COMPILER is more sample-efficient than when B = 4 . Our choice of B = 2 aligns with prior works [21, 22]. If a higher branching factor is chosen, then there are more possible next steps, which require more sampling effort (i.e., more simulations) to cover these expanded possibilities at the same level of thoroughness.

Table 6: Speedup over unoptimized code across varying numbers of samples for different branching factors.

| Llama-3-8B Attention Layer   | Number of Samples   |    18 |    36 |    72 |    150 |    200 |    600 |
|------------------------------|---------------------|-------|-------|-------|--------|--------|--------|
| Llama-3-8B Attention Layer   | B = 2               |  4.52 |  7.08 |  8.38 |   8.79 |  10.56 |  11.33 |
| Llama-3-8B Attention Layer   | B = 4               |  4.16 |  7.88 |  8.35 |   8.89 |   9.86 |  10.99 |
|                              | Number of Samples   | 18    | 36    | 72    | 150    | 200    | 600    |
|                              | B = 2               |  6.14 |  7.05 |  8.33 |   8.53 |   9.1  |   9.45 |
|                              | B = 4               |  2.98 |  4.29 |  4.29 |   7.28 |   7.29 |   9.1  |
|                              | Number of Samples   | 18    | 36    | 72    | 150    | 200    | 600    |
|                              | B = 2               |  4.09 |  4.48 |  4.89 |   5.37 |   5.42 |   5.43 |
|                              | B = 4               |  2.4  |  3.48 |  3.97 |   4.95 |   4.97 |   5.55 |
|                              | Number of Samples   | 18    | 36    | 72    | 150    | 200    | 600    |
|                              | B = 2               |  1.65 |  2.21 |  2.3  |   2.29 |   2.36 |   2.47 |
|                              | B = 4               |  1.91 |  1.97 |  2.23 |   2.23 |   2.25 |   2.43 |

## F Cost of LLMs Used in Experiments

In Table 7, for each benchmark, we report the API cost of running a full experiment with every LLM used to generate transformation proposals. We run a high number of samples to understand the boundary of performance improvements and allow the algorithm to saturate. For OpenAI, our main results used GPT-4o mini, the lowest-cost model available at submission time. For open-source models, we used Hugging Face APIs through the Nscale hyperscaler provider. Across benchmarks, these open-source models achieved competitive speedups and sample efficiency relative to GPT4o mini, indicating that open-source models are a viable alternative when commercial APIs are impractical. Costs of open-source models could be further reduced by local deployment.

Table 7: Cost of different LLM APIs per entire experiment (USD) across layer-wise and end-toend benchmarks.

|                            | Model       | Model          | Model                    | Model                   | Model                   | Model                  |
|----------------------------|-------------|----------------|--------------------------|-------------------------|-------------------------|------------------------|
| Layer / Task               | GPT-4o mini | OpenAI o1-mini | Llama3.3- Instruct (70B) | DeepSeek- Distill (32B) | Llama3.1- Instruct (8B) | DeepSeek- Distill (7B) |
| Llama-3-8B Attention Layer | $0.89       | $6.56          | $2.07                    | $1.55                   | $0.31                   | $2.07                  |
| DeepSeek-R1 MoE Layer      | $0.90       | $6.63          | $2.09                    | $1.57                   | $0.31                   | $2.09                  |
| FLUX Attention Layer       | $0.88       | $6.47          | $2.03                    | $1.52                   | $0.30                   | $2.03                  |
| FLUX Convolution Layer     | $1.12       | $8.25          | $2.67                    | $2.00                   | $0.40                   | $2.67                  |
| Llama-4-Scout MLP Layer    | $0.90       | -              | -                        | -                       | -                       | -                      |
| End-to-End Llama-3-8B      | $1.59       | -              | -                        | -                       | -                       | -                      |

## G LLMProposal Validity and Fallback Rates

LLM-generated transformations can occasionally be syntactically valid but semantically redundant or performance-regressive. During any single MCTS expansion, proposals that fail basic validity checks (e.g., naming or use-context non-compliance) are simply discarded while the remaining valid proposals proceed, and no fallback is triggered. A fallback occurs only when all LLM-generated proposals in that expansion are invalid, in which case the search reverts to the default, non-LLM expansion policy and continues without interruption. In Table 8, we report the fallback rate as the average fraction of expansions that trigger this non-LLM path (i.e., expansions in which all LLM proposals are invalid). To prevent downstream harm from poor but valid transformations, the cost model evaluates all proposed transformations before they are added to the tree; proposals with low estimated values are naturally pruned. Because the transformation space is a known, finite set of legal rewrites, most correctness issues reduce to naming compliance and use-context, which modern instruction-tuned LLMs typically handle well. Empirically, commercial models (GPT-4o mini and OpenAI o1-mini) show 0% fallback rates, larger open-source models perform similarly (Llama3.3-Instruct 70B at 0.08% and DeepSeek-Distill 32B at 0.17%), whereas smaller models exhibit higher fallback rates (Llama3.1-Instruct 8B at 10.50% and DeepSeek-Distill 7B at 17.20%).

Table 8: Fallback rate by model used as the transformation proposal generator.

| Model                   | Fallback Rate   |
|-------------------------|-----------------|
| GPT-4o mini             | 0%              |
| OpenAI o1-mini          | 0%              |
| Llama3.3-Instruct (70B) | 0.08%           |
| DeepSeek-Distill (32B)  | 0.17%           |
| Llama3.1-Instruct (8B)  | 10.50%          |
| DeepSeek-Distill (7B)   | 17.20%          |

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We demonstrate accurately the paper's contributions and scope in the abstract, §1 (introduction), and §4 (results) to support the claims.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The authors acknowledge that their method currently depends on external APIs for querying large language models, which may pose reproducibility and scalability concerns due to cost and access restrictions. They also recognize that the system's performance can vary across model types and that the evaluation is limited to six representative state-of-the-art benchmarks. Moreover, since the approach relies on prompt formatting and reasoning traces, its effectiveness may degrade in settings where context length or LLM interpretability is constrained.

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

Answer: [NA]

Justification: This paper focuses on practical compiler optimization techniques rather than theoretical developments. As such, it does not present formal theorems or proofs. However, in §2, we provide a formal problem formulation to clearly define the optimization setting and guide our methodology. No theoretical claims are made that would require formal assumptions or correctness proofs.

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

Justification: We have provided a detailed experimental setup and included the link to our GitHub repository. We also described in detail our method in §3.1 and §3.2 to make sure our experiments can be reproduced.

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

Justification: We included the link to our repository in the abstract. The repository contains instructions on how to set up and run the experiments.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/ guides/CodeSubmissionPolicy) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: In §4.1, we specified all the experiment details necessary to understand the results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## Answer: [Yes]

Justification: All experiments are repeated 20 times, and the results are averaged to ensure statistical stability, as described in §4.1.

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

Justification: We mention the machine details in §4.1, and the README in the GitHub repository provides the steps.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: This work presents a compiler optimization framework that leverages LLM reasoning for efficient model serving. The positive societal impacts include reducing the computational cost of deploying large machine learning models, which in turn improves accessibility and scalability, as discussed in abstract, introduction, and conclusion (see §1 and §6).

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

Justification: The paper does not release any models or associated datasets which have high risk of misuse. It rather focuses on compiler-level optimizations for efficient ML model serving, which poses no direct safety or misuse concerns that would warrant safeguards.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Our method is integrated with Apache TVM v0.20.0 [10], an open-source machine learning compiler stack released under the Apache License 2.0. We properly cite the original work [8, 10] and ensure full compliance with its licensing terms. We also use OpenAI or HuggingFace's model serving and utilize their APIs to access the models.

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

Justification: We integrate the proposed approach into open-source TVM scheduling and also make our code open source, as discussed in §4.1.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: Large language models are an integral part of our method. We use them in the program optimization process to guide transformation proposals in compiler optimization search. This use of LLMs is central, and is described in detail in 3.1. All of our usage complies with responsible AI guidelines, and models used (e.g., OpenAI's models, LLaMA-3, DeepSeek) are publicly accessible using APIs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.