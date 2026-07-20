# A Learning Agent over Boolean Maps

## TODO

- [ ] Write an introduction and motivation section explaining the goals of
      this project.
- [ ] Relabel answer $a$ (a generic element of $A$) so it no longer
      collides with input bit $a$ (one of the two Boolean variables in the
      running $n=2$ example) — the same symbol currently means two
      different things.
- [x] Explain why $\sum_k \Delta H(q_k)$, summed over all rounds of a
      realized trace, does not equal the total entropy drop
      $H(p_0)\to 0$ — the gap between an expectation computed fresh each
      round and the actual realized reduction along one particular path.
      *(Done — see the chain-rule discussion closing "General form of the
      entropy reduction": the realized drops do telescope to $H(p)$; it's
      the per-round forecasts that need not match them.)*
- [ ] Formalize the optimal (greedy-max) $\Delta H(q_k)$ as a function of
      round $k$ and the evolving belief $p$ — is there a general
      pattern/bound for how it behaves as rounds proceed?
- [ ] Reframe around predictive power, not internal state: what we
      actually care about is $B$'s predictive quality (the three metrics
      from "Modeling the quality of $B$"), not just how fast its internal
      belief entropy shrinks. We have an example, work out the general case.
- [ ] Map the update rule operators.
- [ ] Frame as: we know the distribution of maps, how to efficiently 
      narrow down.
- [ ] Protocol algorithm for actual implementation: Start with all constant maps, give them a high prior, all others low and uniform. When all constant maps eliminated, re-update prior with all 1-circuit maps (that would survive), etc, constructively.
- [ ] Track how each of the three quality metrics (stochastic $B$,
      best-guess $B$, perplexity) evolves round-by-round as questions are
      asked? Nah...
- [ ] Work a concrete example using the Gibbs prior (circuit-complexity
      based) on the $2\to1$ hypothesis space.
- [ ] Run numerical simulations over the full 65536-row classification
      tables for both flavors ($4\to1$ and $3\to2$).
- [ ] Explore the effect of the Gibbs prior's temperature $\beta$ on the
      learning dynamics.
- [ ] Investigate whether this framework can be turned around to *infer*
      a map's circuit complexity from how learning unfolds, rather than
      assuming complexity is known in advance. I.e. how many questions until
      the bayesian estimator of the map is the true map. 
- [ ] Expanding on that: search for the prior that best allows the box to 
      learn.
- [ ] Can we prove this is optimal? Probably fastest finding vs most
      correct are in tension
- [ ] Think through greedy vs. random question choice — the document
      currently introduces both (Plackett-Luce random order in "The
      question order", greedy max-$\Delta H$ in "Example: splitting the
      open hypotheses") without clearly explaining *why* you'd pick one
      over the other; make that explicit, and survey other greedy
      selection criteria beyond maximizing expected entropy reduction
      (e.g. worst-case reduction, expected rounds-to-completion).
- [ ] Unify writing style throughout the document.
- [ ] Read up and connect to: KL divergence (between $B$'s model and the
      truth, or between two models); the notion of *regret* — the gap
      between the best achievable model and $B$'s actual model; free
      energy; square loss.

## Definitions

- **Questions** $Q = \{0,1\}^n$: binary strings of length $n$.
- **Answers** $A = \{0,1\}^m$: binary strings of length $m$.
- **Truth** $\psi: Q \to A$: the one true function nature uses to answer
  questions. $\psi$ is unknown to the learner.
- **Question prevalence** $P(q)$: the real-world frequency of question $q$
  (or our estimate of how often $q$ actually occurs), a distribution over
  $Q$ with $\sum_{q \in Q} P(q) = 1$.
- **Hypothesis space** $\Phi = \{\phi_1, \dots, \phi_N\}$: the set of *all*
  possible functions $Q \to A$, i.e. every candidate $\psi$ could be. Since
  each of the $2^n$ inputs can map to any of the $2^m$ outputs independently,

$$N = |\Phi| = 2^{m \cdot 2^n}.$$

  This is exactly the row-count of the classification tables, as explained
  in the appendix ($N=65536$ for $(n,m)=(4,1)$ and $(n,m)=(3,2)$).

- **Consistent-hypothesis sets** $K_a(q) = \{\, j : \phi_j(q) = a \,\}$: the
  indices of hypotheses that would answer $a$ to question $q$. For fixed
  $q$, the sets $K_a(q)$ ($a \in A$) partition $\{1,\dots,N\}$, since every
  $\phi_j$ answers exactly one $a$ to a given $q$. In fact, we will show each of the $2^m$ sets is the same size.
- **Per-hypothesis accuracy** $\epsilon_j$: the probability, under $P$, that
  $\phi_j$ would agree with $\psi$ on a random question — i.e. the accuracy
  $\phi_j$ *would have if it were the true map*:

$$\epsilon_j = \sum_{q \in Q} P(q)\, \delta\big(\psi(q), \phi_j(q)\big),$$

  where $\delta(x,y)=1$ if $x=y$ and $0$ otherwise. $\epsilon_j$ is a fixed
  property of $\phi_j$ (relative to $\psi$ and $P$) — it doesn't depend on
  $B$'s belief at all.
- **Learner (box)** $B$: maintains a prior (belief) $p_j = P(\psi = \phi_j)$
  over every hypothesis in $\Phi$, with

$$\sum_{j=1}^{N} p_j = 1.$$

  $B$'s prior encodes what it expects the truth to look like *before* seeing
  any question-answer pairs; later, evidence updates $p_j$ (e.g. via Bayes'
  rule), which is the subject of a later section of this project.

## Example: $n=2$, $m=1$

Here $Q=\{00,01,10,11\}$, $A=\{0,1\}$, and $N = 2^{1 \cdot 4} = 16$: every
2-input, 1-output Boolean function. Each $\phi_j$ is shown as its truth
table, read in input order $(00,01,10,11)$.

The example prior $p_j$ below is conjured for illustration, not fit to data. 

| $j$ | $\phi_j(00,01,10,11)$ | name | $p_j$ |
|---|---|---|---|
| 1  | 0000 | FALSE (constant 0) | 0.150 |
| 2  | 0001 | $a \wedge b$ (AND) | 0.149 |
| 3  | 0010 | $a \wedge \lnot b$ | 0.094 |
| 4  | 0011 | $a$ (projection) | 0.102 |
| 5  | 0100 | $\lnot a \wedge b$ | 0.060 |
| 6  | 0101 | $b$ (projection) | 0.082 |
| 7  | 0110 | $a \oplus b$ (XOR) | 0.012 |
| 8  | 0111 | $a \vee b$ (OR) | 0.014 |
| 9  | 1000 | $\lnot(a \vee b)$ (NOR) | 0.141 |
| 10 | 1001 | $a = b$ (XNOR) | 0.114 |
| 11 | 1010 | $\lnot b$ | 0.015 |
| 12 | 1011 | $b \to a$ (IF THEN) | 0.013 |
| 13 | 1100 | $\lnot a$ | 0.017 |
| 14 | 1101 | $a \to b$ (IF THEN) | 0.016 |
| 15 | 1110 | $\lnot(a \wedge b)$ (NAND) | 0.011 |
| 16 | 1111 | TRUE (constant 1) | 0.010 |

$\sum_j p_j = 1.000$; all 16 values distinct. 

## How $B$ learns

For this experiment, $B$ is fed questions one at a time, without
repeats. Asking a question twice teaches $B$ nothing it doesn't already know, so we only care about the order in which the $2^n$ distinct questions are first asked.

### The question order

We could imagine drawing $q \sim P$ repeatedly and discarding repeats, but it's cleaner to describe the resulting process directly: draw $q_1$ with probability $P(q_1)$; having removed it, draw $q_2$ from what remains with probability $P(q_2)$ renormalized by the remaining mass $1-P(q_1)$; and so on until all $2^n$ questions have been drawn. This gives a full ordering
$(q_1,\ldots,q_{2^n})$ of $Q$ with probability

$$
\Pr(q_1,\ldots,q_{2^n}) = \prod_{k=1}^{2^n}\frac{P(q_k)}{\sum_{i=k}^{2^n}P(q_i)},
$$

the **Plackett-Luce distribution**: each factor is $q_k$'s share of the probability mass still remaining once $q_1,\ldots,q_{k-1}$ have already been removed. (Under uniform $P$, every factor's denominator just counts remaining questions, and this reduces to the uniform distribution over all $2^n!$ orderings, as expected.)

### The update rule

At round $k$, $B$ is asked $q_k$, produces a guess (how it decides is the subject of the models below), and then is told the true answer $a = \psi(q_k)$.

Recall $K_a(q)$ from the Definitions above: the hypotheses consistent with answer $a$ to question $q$. Learning $a = \psi(q_k)$ rules out every hypothesis *not* in $K_a(q_k)$ outright:

$$
p_j \leftarrow 0 \quad \text{for } j \notin K_a(q_k).
$$

What about the survivors? Take two surviving hypotheses
$j,l \in K_a(q_k)$: both $\phi_j$ and $\phi_l$ predicted $a$ on $q_k$, so this
round of evidence doesn't distinguish between them at all. If $\phi_j$ was
$r$ times as likely as $\phi_l$ before this round, it must still be $r$
times as likely afterward, nothing has come in to change that ratio. $p_j = p_l \cdot r$ remains. The
only update consistent with preserving every such ratio among survivors is
renormalizing them to sum to 1:

$$
p_j \leftarrow \frac{p_j}{\sum_{i \in K_a(q_k)} p_i} \quad \text{for } j \in K_a(q_k).
$$

(This is Bayes' rule in disguise, not a separate assumption: the
likelihood of observing $a$ given $\phi_j$ is $1$ if $\phi_j(q_k)=a$ and
$0$ otherwise, so Bayes' rule zeroes out the inconsistent hypotheses and
rescales the rest by the same normalizing constant — exactly the
ratio-preserving renormalization above.)

This elimination has a clean, predictable size, because $\Phi$ is the
*full, unconstrained* space of functions $Q\to A$: we're assuming every
input can independently take any of the $2^m$ answers, with a function's
value at one input placing no restriction whatsoever on its values
elsewhere, so $\Phi$ is exactly the Cartesian product $A^{|Q|}$. Having
survived rounds $1,\ldots,k-1$ means exactly: the values at
$q_1,\ldots,q_{k-1}$ are fixed to what was observed, while every other
input, including the fresh $q_k$, is still completely free and
independent. Fixing that still-free value to each of the $2^m$ possible
answers therefore divides whatever's currently open into $2^m$ 
equal groups, of which the true answer keeps exactly one. Since a fresh
question always removes exactly one input from the "still free" pool, the
support of $B$'s distribution shrinks geometrically and predictably every
round:

$$
(2^m)^{2^n+1-k} \;\longrightarrow\; (2^m)^{2^n-k}
$$

at round $k$, depending only on $k$ (and $n,m$). It does not depend on which particular hypotheses survived or which answers were given.

### Collapse entropy

For a not-yet-asked question $q$ and a hypothetical answer $a\in A$,
define the **collapse entropy**

$$
H(a\mid q) := -\sum_{j\in K_a(q)} \frac{p_j}{P(a\mid q)}\,\log_2\frac{p_j}{P(a\mid q)},
\qquad
P(a\mid q) := \sum_{j\in K_a(q)} p_j,
$$

the entropy of $B$'s belief *after* it would collapse via the update rule
above onto $K_a(q)$ — were $a$ the answer actually received. Unlike the
update rule itself, which only fires once $a$ is known, $H(a\mid q)$ can
be computed for every $(q,a)$ pair in advance, from $p$ alone: it's the
"what if" entropy for a hypothetical round that hasn't happened yet.

Averaging over both possible answers (weighted by how likely $B$ currently
thinks each is) gives the expected entropy *after* asking $q$, and hence
the expected entropy *reduction*:

$$
\mathbb{E}_a[H(a\mid q)] = \sum_{a\in A} P(a\mid q)\, H(a\mid q),
\qquad
\Delta H(q) := H(p) - \mathbb{E}_a[H(a\mid q)],
$$

where $H(p) = -\sum_j p_j \log_2 p_j$ is $B$'s current belief entropy.
$\Delta H(q)$ is exactly the quantity the lookahead matrix computed one
question at a time in the next section; asking the question with the
largest $\Delta H(q)$ is the natural greedy criterion for narrowing things
down as fast as possible in expectation.

### Example: splitting the open hypotheses

Take the running $n=2,m=1$ example (16 hypotheses) with $\psi=$ AND, asking
the four questions $00, 01, 10, 11$. Each fresh question $q_k$
splits whatever's currently open into $2^1$ groups $K_a(q_k)$, one per
possible answer, and the true answer $a=\psi(q_k)$ keeps one.

Which first question is the most powerful? Under the prior from the Example section
(all 16 hypotheses, nothing yet asked), each of the four questions divides
the full set into two size-8 groups $K_0(q), K_1(q)$ (equal count, as
established above); here is every cell's membership (names match the
numbering in the Example table):

| $a$ | $q=00$ | $q=01$ | $q=10$ | $q=11$ |
|---|---|---|---|---|
| $0$ | FALSE, AND, a∧¬b, a(proj), ¬a∧b, b(proj), XOR, OR | FALSE, AND, a∧¬b, a(proj), NOR, XNOR, ¬b, b→a | FALSE, AND, ¬a∧b, b(proj), NOR, XNOR, ¬a, a→b | FALSE, a∧¬b, ¬a∧b, XOR, NOR, ¬b, ¬a, NAND |
| $1$ | NOR, XNOR, ¬b, b→a, ¬a, a→b, NAND, TRUE | ¬a∧b, b(proj), XOR, OR, ¬a, a→b, NAND, TRUE | a∧¬b, a(proj), XOR, OR, ¬b, b→a, NAND, TRUE | AND, a(proj), b(proj), OR, XNOR, b→a, a→b, TRUE |

Equal *count* doesn't mean equal *probability*, since this prior is far
from uniform. Summing each cell's actual weights and computing its
collapse entropy:

| | $q=00$ | $q=01$ | $q=10$ | $q=11$ |
|---|---|---|---|---|
| $P(0\mid q)$ | 0.663 | 0.778 | 0.729 | 0.500 |
| $H(0\mid q)$ | 2.693 | 2.728 | 2.713 | 2.424 |
||||||
| $P(1\mid q)$ | 0.337 | 0.222 | 0.271 | 0.500 |
| $H(1\mid q)$ | 2.174 | 2.493 | 2.285 | 2.456 |
||||||
| $\mathbb{E}_a[H(a\mid q)]$ | 2.518 | 2.676 | 2.597 | 2.440 |
| $\Delta H(q)$ | 0.922 | 0.764 | 0.843 | 1.000 |

($H(p)=3.440$ bits at the very start.) This time the four questions are
clearly separated, not near-tied: $q=11$ splits the hypothesis set exactly
in half by probability ($0.500/0.500$, despite the prior's heavy skew
elsewhere) and is worth the full $1$ bit — strictly more informative than
every other option — while $q=01,10,00$ trail at $0.764, 0.843, 0.922$
bits respectively, each separated from its neighbors by at least $0.08$
bits.

Following the greedy rule from the previous section — always ask the
unasked question with the largest $\Delta H(q)$ — the first question is
$q_1=11$. The true answer is $\psi(11)=\mathrm{AND}(1,1)=1$, so belief
collapses onto $K_1(11)$ (renormalized) and 8 hypotheses survive:
AND, OR, TRUE, XNOR, $a$(proj), $a\to b$, $b$(proj), $b\to a$. Recomputing
$\Delta H(q)$ for the three remaining questions against this new belief,
then repeating the whole process at each subsequent round:

| round $k$ | remaining questions | $\Delta H(q)$ for each | chosen $q_k$ | true $a$ | survivors after |
|---|---|---|---|---|---|
| 1 | 00, 01, 10, 11 | 0.922, 0.764, 0.843, **1.000** | 11 | 1 | 8: AND, OR, TRUE, XNOR, a(proj), a→b, b(proj), b→a |
| 2 | 00, 01, 10 | **0.889**, 0.802, 0.853 | 00 | 0 | 4: AND, OR, a(proj), b(proj) |
| 3 | 01, 10 | 0.851, **0.919** | 10 | 0 | 2: AND, b(proj) |
| 4 | 01 | **0.938** | 01 | 0 | 1: AND |

Every round asks the single best remaining question, and every round's
true answer happens to be $0$ except the first. By round 4 only AND
survives — matching $\psi$, as it must. Interestingly, this greedy order
still takes all four questions to fully pin down the truth (as any order
must here: after round 3, AND and $b$(proj) still agree everywhere except
$q=01$, so nothing shorter than the full $2^n=4$ rounds could have
separated them) — maximizing expected information each round narrows
things down as fast as possible on average, but doesn't shrink the
worst-case number of rounds needed.

### Information in a question

How much does asking $q_k$ actually tell $B$ about the identity of the
truth? It's convenient to make explicit something already implicit in
$p_j = P(\psi=\phi_j)$: from $B$'s own subjective point of view, treat the
index of the true map as a random variable $J$ with $P(J=j) = p_j$, and
let $A_{q_k} = \phi_J(q_k)$ be the (as yet unseen) answer it implies. Two
related quantities measure the information $q_k$ carries: surprisal, and expected entropy reduction.

Let us define the surprisal of the answer actually received. Before hearing $a$, $B$'s
own predictive distribution over what the answer will be is
$P(a\mid q_k) = \sum_{j \in K_a(q_k)} p_j$ (this reappears below as $B$'s
induced answer distribution $P_B(a\mid q)$, in the Perplexity subsection).
The specific answer that comes back carries $-\log_2 P(a\mid q_k)$ bits of
Shannon self-information: if $B$ already expected $a$, little is learned
and few hypotheses are eliminated; if $a$ was one $B$ thought unlikely, a
large chunk of belief mass is wiped out in one round.

## Modeling the quality of $B$

There are several equivalent ways to model how good $B$'s current belief
is, and hence how fast $B$ learns: **stochastic $B$**, **best-guess $B$**,
and **perplexity of $B$**. We start with the first — though answering by
random sampling seems like an odd protocol for an agent, it makes the math
particularly simple.

### Stochastic $B$

On question $q$, $B$ samples a hypothesis $\phi_j \sim p$ (draws index $j$
with probability $p_j$) and answers with $\phi_j(q)$.

The expectation that $B$ answers correctly, over both a random question
(drawn from $P$) and $B$'s own random sampling (drawn from $p$), is

$$
\mathbb{E}[\text{correct}] = \sum_{q \in Q} P(q) \sum_{j=1}^{N} p_j\, \delta\big(\psi(q), \phi_j(q)\big).
$$

Since this is just a finite double sum, we can reorder it:

$$
\mathbb{E}[\text{correct}] = \sum_{j=1}^{N} p_j \underbrace{\sum_{q \in Q} P(q)\, \delta\big(\psi(q), \phi_j(q)\big)}_{\epsilon_j} = \sum_{j=1}^{N} p_j\, \epsilon_j.
$$

So under the stochastic protocol, $B$'s expected correctness is simply the
belief-weighted average of each hypothesis's own accuracy $\epsilon_j$ (as
defined above) — a dot product $p \cdot \epsilon$, linear in $B$'s belief.
This linearity is what makes the stochastic model the simplest of the
three to reason about.

### Best-guess $B$

A more operationally sensible protocol: $B$ always answers with its single
most-believed hypothesis, regardless of $q$. Let $j^* = \arg\max_j p_j$.
Then

$$
\mathbb{E}[\text{correct}] = \sum_{q \in Q} P(q)\, \delta\big(\psi(q), \phi_{j^*}(q)\big) = \epsilon_{j^*}.
$$

A single number, not a weighted average — simpler to *state* than the
stochastic result, but harder to *reason about* as $p$ evolves with
evidence: $\arg\max$ is discontinuous in $p$, so a small update can flip
$j^*$ from one hypothesis to an entirely different one, and $\epsilon_{j^*}$
jumps with it. The stochastic model's $p \cdot \epsilon$ is smooth and
differentiable in $p$; best-guess $B$'s performance is not. Best-guess is,
however, exactly what a decision-theoretic agent minimizing 0-1 loss would
do (MAP estimation) — it's the realistic protocol, even if it's the
harder one to analyze.

### Perplexity of $B$

Both previous models score $B$ against the truth $\psi$ — something $B$
itself never has access to. Perplexity instead measures $B$'s own
*uncertainty* about the answer, computable entirely from $B$'s belief $p$
and the hypothesis space $\Phi$, with no reference to $\psi$ at all.

Recall $K_a(q)$ from the Definitions above. This gives $B$'s induced
**answer distribution** at $q$:

$$
P_B(a \mid q) = \sum_{j \in K_a(q)} p_j,
$$

the probability that stochastic-$B$ would answer $a$ to $q$ — a genuine
distribution over $A$, since $\sum_{a} P_B(a\mid q) = \sum_j p_j = 1$.

$B$'s uncertainty at $q$ is the entropy of that distribution, and its
**perplexity at $q$** the exponentiated entropy:

$$
H_B(q) = -\sum_{a \in A} P_B(a\mid q) \log_2 P_B(a\mid q), \qquad
\mathrm{PPL}_B(q) = 2^{H_B(q)} \in [1, 2^m].
$$

$\mathrm{PPL}_B(q)$ reads as the *effective number of equally-likely
answers* $B$ is torn between at $q$: $1$ when $B$'s belief is fully
concentrated on hypotheses that agree on $q$ (no uncertainty), up to
$2^m = |A|$ when $B$'s answer distribution is uniform over every possible
answer. Averaging over questions by prevalence gives an overall figure:

$$
H_B = \sum_{q \in Q} P(q)\, H_B(q), \qquad \mathrm{PPL}_B = 2^{H_B}.
$$

Because none of $K_a(q)$, $P_B(a\mid q)$, $H_B(q)$, or $\mathrm{PPL}_B$
mention $\psi$, $B$ can compute its own perplexity at any time — unlike
$\epsilon_j$ or $\mathbb{E}[\text{correct}]$, which require an outside
evaluator who knows the truth. This makes perplexity a natural
self-assessed confidence signal (e.g. for deciding when to ask another
question rather than commit to an answer).

### $B$ as a stochastic matrix

A fourth way to picture $B$: as a $2^m\times 2^n$ column-stochastic matrix
$M$, one column per question $q\in Q$, one row per answer $a\in A$, with
$M_{a,q}\geq 0$ and $\sum_a M_{a,q}=1$ for every column. Feed in $q$ as a
one-hot vector $e_q\in\mathbb{R}^{2^n}$; $Me_q$ is just column $q$ of $M$,
 the resulting distribution over answers.

A binary map $\phi_j$ is a special case of exactly this shape: the matrix
with a single $1$ in each column, at row $\phi_j(q)$. A *deterministic*
column-stochastic matrix. As a point in the $2^m\times 2^n$ real matrix
space (dimension $2^{n+m}$), $B$'s general $M$ can be any column-stochastic
matrix, while a binary map is confined to one of the $N=2^{m\cdot 2^n}$
all-or-nothing corners.

$N$ vastly exceeds the matrix's own dimension $2^{n+m}$, so an $M$ could result from many combinations of 
mixtures $p_j=P(\psi=\phi_j)$ of them. It's an overcomplete basis for the convex polytope in which $M$ lives. concretely,

$$
M_{a,q} = \sum_{j\in K_a(q)} p_j = P_B(a\mid q),
$$

reusing the induced answer distribution from above. The entries of
$M$ are literally sums of $p_j$ over whichever binary maps have a $1$ in
that slot.

For the running $n=2,m=1$ example, $M$ is just $2\times 4$: two rows
($a=0,1$), four columns ($q=00,01,10,11$), each entry the sum of the
$p_j$ (indices from the Example table) whose $\phi_j$ agrees with that
row on that column:

$$
M =
\begin{pmatrix}
\sum_{j=1}^ 8 p_j &
p_1{+}\ldots{+}p_4{+}p_9{+}\ldots{+}p_{12} &
p_1{+}p_2{+}p_5{+}p_6{+}p_9{+}p_{10}{+}p_{13}{+}p_{14} &
\sum_{j \text{ ODD}} p_j \\[4pt]
\sum_{j=9}^{16} p_j &
p_5{+}\ldots{+}p_8{+}p_{13}{+}\ldots{+}p_{16} &
p_3{+}p_4{+}p_7{+}p_8{+}p_{11}{+}p_{12}{+}p_{15}{+}p_{16} &
\sum_{j \text{ EVEN}} p_j 
\end{pmatrix}
$$

Due to our naming convention, this happens to follow the rule *column $k$, row $l$, has $j$ such that the binary representation of $j-1$ has bit $k=l$*. This rule extends naturally to larger $m$ and allows for efficient lookup.

which, plugging in $B$'s prior from the Example table, is

$$
M = \begin{pmatrix} 0.663 & 0.778 & 0.729 & 0.500 \\ 0.337 & 0.222 & 0.271 & 0.500 \end{pmatrix},
$$

exactly the $P_B(a\mid q)$ values already used in the table above (each
column sums to $1$, as it must).

This connects to the models above:

- **Stochastic $B$** is recovered by sampling $a$ from column $q$ of
  $M$ — identical to sampling $\phi_j\sim p$ first and reading off
  $\phi_j(q)$, since both give $a$ with probability $M_{a,q}$.
- The stochastic-matrix picture suggests a *different* best-guess rule:
  for each $q$ independently, answer $\arg\max_a M_{a,q}$ — the
  column-wise mode. Call this **matrix best-guess**. It's the genuine
  Bayes-optimal answer to each question in isolation (it directly
  maximizes $P_B(a\mid q)$ per question, so it's at least as accurate,
  question by question, as the original best-guess). As it produces a definite answer for each a, it is also a binary map. It is interesting to 
  investigate when this map is not the same as the $phi_j$ that maximizes $p_j$.

The update rule has a clean picture here too. Learning $a^*=\psi(q_k)$
doesn't just zero out $K_a(q_k)$'s complement in $p$: it collapses column
$q_k$ of $M$ to a one-hot vector (all mass at row $a^*$, since every
surviving $\phi_j$ answers $a^*$ there by construction), *and*
simultaneously renormalizes every *other* column, since the same
zeroed-out hypotheses were contributing to their entries too. We get a condition that all $p_j$ that contributed to the wrong row in that column vanish. The ones in the right row get blown up. As we recall, every binary map had a value in each column, so this rule touches all $p_j$.

How do we update the stochastic matrix? This applied to every entry of $M$ at once:

$$
M_{a,q} \;\longleftarrow\; \frac{\displaystyle\sum_{j\,\in\, K_a(q)\,\cap\,K_{a^*}(q_k)} p_j}{P(a^*\mid q_k)}
\qquad \text{for every } a\in A,\ q\in Q,
$$

keep only the hypotheses consistent with *both* the column being read off
and the new evidence, then rescale by however much probability mass
survived. Column $q_k$ itself collapses to one-hot as a special case of
this same formula, not a separate rule: setting $q=q_k$, the numerator
is $P(a^*\mid q_k)$ when $a=a^*$ (giving $M_{a^*,q_k}=1$) and $0$
otherwise, since $K_a(q_k)$ and $K_{a^*}(q_k)$ are disjoint for
$a\neq a^*$. Exactly "zero in the wrong row, one in the right row."

It is important to choose an intelligent prior. If every $p_j$
starts uniform, symmetry guarantees every not-yet-asked column stays
exactly uniform ($1/2^m$ in every row) no matter what's been learned
elsewhere. A uniform prior over the full hypothesis space of binary maps
space has, by construction, no correlation between any two questions'
answers, so nothing transfers between columns. A smarter prior breaks
that symmetry: if $p_j$ favors hypotheses whose answers
correlate across questions (e.g. low-complexity functions, which tend to
repeat structure), then resolving one column sharpens others *before
they're ever asked*. So a smart prior allows us to learn on new $q$ we haven't seen, from others we have. This may even be a working definition of intelligence: an instinct for patterns that allows us to learn faster.

### How much does $q$ reduce the entropy of $a$?

Column $q$ of $M$ is a distribution over $A$ in its own right, with its
own entropy: a genuinely different quantity from $H(p)$, the entropy of
$B$'s belief over the $N$ possible $\Phi$. But we could argue that the physical uncertainty we care about is not the internal model of the world in the box, but its predictive power. 

With no information at all, $a$
could be any of the $2^m$ possible answers with equal probability, for a
maximum entropy of $\log_2(2^m)=m$ bits: total ignorance about the
answer. $B$'s actual column-$q$ entropy is exactly $H_B(q)$ from the
Perplexity subsection above, so the *reduction* already achieved by $B$'s
current prior — before any round has actually been played — is

$$
m - H_B(q) \quad \text{bits}.
$$

This is a different quantity from $\Delta H(q)$: $\Delta H(q)$ measures
how much *asking and hearing the answer to* $q$ would shrink $B$'s belief
over all $N$ hypotheses (ranging up to $H(p)$ bits); $m-H_B(q)$ measures
how much $B$'s *current* prior already constrains the *answer itself*
(ranging up to $m$ bits), with no question asked at all — pure structure
already baked into $p$.

For the running $n=2,m=1$ example, $m=1$ bit is the maximum possible:
total ignorance about a single bit. Column by column, using $B$'s prior
from the Example table:

| $q$ | $P_B(0\mid q)$ | $P_B(1\mid q)$ | $H_B(q)$ | reduction $1-H_B(q)$ |
|---|---|---|---|---|
| $00$ | $0.663$ | $0.337$ | $0.922$ | $0.078$ |
| $01$ | $0.778$ | $0.222$ | $0.764$ | $0.236$ |
| $10$ | $0.729$ | $0.271$ | $0.843$ | $0.157$ |
| $11$ | $0.500$ | $0.500$ | $1.000$ | $0.000$ |

$q=01$ is the column $B$ is most confident about *before asking anything*
(only $0.764$ bits of uncertainty in the answer, a $0.236$-bit reduction
from total ignorance), precisely because its split is the most lopsided
($0.778/0.222$). $q=11$, by contrast, is a coin flip: $B$'s prior offers
zero reduction there, a full $1$ bit of irreducible uncertainty about the
answer. 

That's the entropy of $a$ *before* anything is asked. More interesting:
after actually learning the true answer to some $q_k$, how does the
entropy of every *other* column change? Trivially $H_B(q_k)$ itself drops
to $0$ — $B$ just observed $\psi(q_k)$ directly. But the update rule
collapses every other column too, and by how much depends entirely on
whether the prior correlates their answers with $q_k$'s. Row $q_k$ = the
question actually asked and answered (true $\psi=$AND); each column is the
resulting $H_B(q')$ immediately afterward, for every $q'$ including $q_k$
itself. We also copy the prior into the first row, for reference:

| learned $q_k$ | $H_B(00)$ | $H_B(01)$ | $H_B(10)$ | $H_B(11)$ | ‖ | $\langle H_B\rangle_Q$ |
|---|---|---|---|---|---|---|
| - | $0.922$ | $0.764$ | $0.843$ | $1.000$ | ‖ | $0.882$ |
||||||||
| $00$ | $0$ | $0.817$ | $0.920$ | $0.998$ | ‖ | $0.684$ |
| $01$ | $0.946$ | $0$ | $0.866$ | $0.999$ | ‖ | $0.703$ |
| $10$ | $0.968$ | $0.795$ | $0$ | $1.000$ | ‖ | $0.691$ |
| $11$ | $0.889$ | $0.802$ | $0.853$ | $0$ | ‖ | $0.636$ |

Zero down the diagonal, as it must be. Off the diagonal, the picture is
mixed, not uniformly downward: compare each entry against that column's
own baseline in the top row. Learning $q_k=11$ genuinely
helps $q'=00$ ($0.922\to0.889$); but learning $q_k=10$ *raises* $H_B(01)$
above its own baseline ($0.764\to0.795$), and $q_k=00,01$ do the same to
several other columns. This isn't a mistake: the guarantee that
conditioning helps *on average* is a statement about the reduction in
$B$'s belief-space entropy $H(p)$, averaged over $q_k$'s own possible
answers — not a promise that any one *realized* answer must lower every
other column's entropy individually. A correlated prior can just as
easily point the wrong way for a particular realization, the same
phenomenon already seen for surprisal and entropy-drop earlier. 

$\langle H_B\rangle_Q$ is the *uniform* average $\frac{1}{2^n}\sum_{q'\in Q} H_B(q')$
The baseline row's $0.882$ bits is $B$'s
average per-question uncertainty before anything is asked; every learned
$q_k$ pulls that average down. $q_k=11$ gives the largest drop on average
(to $0.636$), which is partly because itself had so much uncertainty, but also due to the interaction with other questions.

In fact, whether one $q$ positively affects another is a *property* of a good prior: a process of elimination should move us through a sequence of good guesses (or best contingent guesses). As this prior was chosen randomly, it is not expected to have that property. Unfortunately, what constitutes a 'good guess' is a property of the true distribution of $\Phi$, and so is by definition empirical.

### General form of the entropy reduction

The examples above suggest a general law, and there is one. The key
observation: under $B$'s belief, the answers $A_q = \phi_J(q)$ (one random
variable per column, $J\sim p$) collectively *are* the hypothesis — a
binary map is nothing but its full table of answers, so the tuple
$(A_q)_{q\in Q}$ determines $J$ and vice versa. Their joint entropy is
therefore exactly $B$'s belief entropy:

$$
H\big(A_{q_1},\ldots,A_{q_{2^n}}\big) = H(p).
$$

Meanwhile the uniform column average $\langle H_B\rangle_Q$ sums the
*marginal* entropies $H_B(q) = H(A_q)$. Marginal entropies always sum to
at least the joint entropy (subadditivity), and the gap is the **total
correlation** of the answers under $p$:

$$
C(p) := \sum_{q\in Q} H_B(q) - H(p) \;\geq\; 0,
$$

zero exactly when the prior makes all answers independent of each other.
This gives an exact identity, valid for any $n$, $m$, and prior $p$:

$$
\langle H_B\rangle_Q = \frac{H(p) + C(p)}{2^n},
\qquad
m - \langle H_B\rangle_Q = \frac{\big(m\,2^n - H(p)\big) - C(p)}{2^n}.
$$

Read the second form as a budget: $m\,2^n = \log_2 N$ is the maximum
possible belief entropy, so $m\,2^n - H(p)$ is how much $B$'s prior
already "knows" in belief-space — and the average sharpness of its
*predictions* is that knowledge minus the correlation overhead $C(p)$,
spread over the $2^n$ questions. Correlation is knowledge that doesn't
show up in any single marginal. The two extremes make this vivid:

- A **product prior** (answers independent across questions) has
  $C=0$: every bit of belief-space knowledge appears in the marginals,
  and $\langle H_B\rangle_Q = H(p)/2^n$, the minimum possible for a given
  $H(p)$.
- A **maximally redundant prior**, e.g. $p=\tfrac12$ on FALSE and
  $\tfrac12$ on TRUE ($n=2,m=1$): belief entropy is a tiny $H(p)=1$ bit
  (only two live hypotheses!), yet every column is a 50/50 coin flip.
  Here $C = 4-1 = 3$ bits eats the *entire* deficit:
  $m-\langle H_B\rangle_Q = (4-1-3)/4 = 0$. $B$ is almost certain which
  world it's in, and can predict nothing — the two remaining worlds
  disagree everywhere.

For the running example's prior: $H(p)=3.440$, the column entropies sum
to $3.529$, so $C(p)=0.089$ — a weakly correlated prior — and indeed
$\langle H_B\rangle_Q = (3.440+0.089)/4 = 0.882$, matching the baseline
row of the cross table, with reduction $(4 - 3.440 - 0.089)/4 = 0.118$
bits.

**As a function of the round $k$.** The identity is preserved verbatim by
the update rule: after learning the answers to $S_k=\{q_1,\ldots,q_k\}$,
the asked columns contribute $0$ to both sides, and

$$
\langle H_B\rangle_Q^{(k)} = \frac{H(p^{(k)}) + C_k}{2^n},
\qquad
\frac{H(p^{(k)})}{2^n} \;\leq\; \langle H_B\rangle_Q^{(k)} \;\leq\; m\Big(1-\frac{k}{2^n}\Big),
$$

with $C_k$ the total correlation of the *remaining* answers under the
updated belief $p^{(k)}$. Checking this against every row of the cross
table (true $\psi=$AND):

| learned $q_k$ | $H(p^{(1)})$ | $C_1$ | $(H(p^{(1)})+C_1)/4$ | $\langle H_B\rangle_Q$ from cross table |
|---|---|---|---|---|
| $00$ | $2.693$ | $0.042$ | $0.684$ | $0.684$ |
| $01$ | $2.728$ | $0.083$ | $0.703$ | $0.703$ |
| $10$ | $2.713$ | $0.050$ | $0.691$ | $0.691$ |
| $11$ | $2.456$ | $0.087$ | $0.636$ | $0.636$ |

The upper bound $m(1-k/2^n)$ is the **uniform-prior envelope**: with
$p_j$ uniform, every unasked column stays exactly at $m$ bits forever
(as argued in the stochastic-matrix section), so
$\langle H_B\rangle_Q^{(k)} = m(1-k/2^n)$ — a straight line from $m$ down
to $0$, each question earning exactly its own $m/2^n$ share and nothing
more. Any prior sits on or below this line, and *how far below* is
precisely the generalization: for a product prior the unasked columns are
frozen at their prior marginals ($C_k=0$ always); only a correlated prior
($C>0$) can move unasked columns at all.

How much, exactly? In expectation — averaging over the answer $B$ expects
for the asked block, rather than fixing one realization — the entropy of
an unasked column $q'$ after learning $S_k$ is

$$
\mathbb{E}\big[H_B^{(k)}(q')\big] = H(A_{q'}) - I\big(A_{q'};\, A_{S_k}\big):
$$

**expected generalization is mutual information**, the information the
asked block carries about the unasked column under the prior. Since
$I\geq 0$ always, no column's entropy rises *in expectation* — which
finally resolves the cross-table anomaly cleanly. There, learning
$q_k=10$ (realized answer $0$) raised $H_B(01)$ from $0.764$ to $0.795$;
but the other branch (answer $1$, probability $0.271$) would have dropped
it to $0.666$, and the expectation $0.729(0.795)+0.271(0.666)=0.760$ is
below the baseline $0.764$ by exactly $I(A_{01};A_{10})=0.004$ bits. The
prior's correlations point the right way on average and can still point
the wrong way on a given draw.

Finally, the chain rule of entropy ties the whole run together: for any
*fixed* question order, the expected belief-entropy drops per round are
$\mathbb{E}[\Delta H_k] = H(A_{q_k}\mid A_{S_{k-1}})$, and these telescope
exactly:

$$
\sum_{k=1}^{2^n} H\big(A_{q_k}\mid A_{S_{k-1}}\big) = H\big(A_{q_1},\ldots,A_{q_{2^n}}\big) = H(p).
$$

(Numerically confirmed for the running example: the four conditional
answer entropies along the order $00,01,10,11$ sum to $3.440 = H(p)$.)
Every question order spends the same total budget $H(p)$ *in
expectation* — greedy max-$\Delta H$ selection can only front-load the
spending, not increase it. On a single realized trace the picture
differs in a specific way: the *realized* drops
$H(p^{(k-1)})-H(p^{(k)})$ also telescope perfectly (from $H(p)$ down to
$0$), but each round's *forecast* $\Delta H(q_k)$ — an expectation
computed fresh from the belief of that moment — need not equal the drop
that then actually occurs, which is why the running example's realized
drops kept undershooting their forecasts (AND, a high-prior hypothesis,
kept surviving; surprises never materialized).

### Closed forms from the index representation

Can the calculation be carried all the way through to elementary
functions? For a general prior, no — but the index representation from
the stochastic-matrix section shows exactly *which* priors allow it, and
gives a sharp limit law for generic ones.

Recall the indexing rule ($m=1$): writing $j-1$ in binary, bit $q$ of
$j-1$ is $\phi_j(q)$ — the hypothesis index *is* the truth table. Column
$q$ of $M$ is then a subset sum over half the indices, selected by one
bit:

$$
M_{1,q} = \sum_{j\,:\,\mathrm{bit}_q(j-1)=1} p_j,
\qquad
\langle H_B\rangle_Q = \frac{1}{2^n}\sum_{q} h\big(M_{1,q}\big),
$$

where $h(x)=-x\log_2 x-(1-x)\log_2(1-x)$ is the binary entropy function.
The $2^n$ bit-mask subset sums are the only aggregates of $p$ that
matter. Two cases collapse to elementary functions.

**Solvable case 1: energy additive over answer bits.** Take a Gibbs prior
whose energy is the *popcount* of the truth table,
$E(j)=\mathrm{popcount}(j-1)$ — the exact version of the "$0$-over-$1$
bias" the running example gestures at. Because the energy is a sum over
bits, the partition function factorizes over the $2^n$ bit positions:

$$
Z=\sum_{j} e^{-\beta\,\mathrm{popcount}(j-1)}
 =\prod_{q}\big(1+e^{-\beta}\big) = \big(1+e^{-\beta}\big)^{2^n},
$$

and with it the whole prior: each answer bit is independently $1$ with
probability $\sigma(-\beta)=1/(1+e^{\beta})$, the logistic function.
Everything is then elementary, exactly, for every $n$:

$$
M_{1,q}=\sigma(-\beta)\ \ \forall q,\qquad
\langle H_B\rangle_Q = h\big(\sigma(\beta)\big),\qquad
H(p)=2^n\,h\big(\sigma(\beta)\big),\qquad
C(p)=0,
$$

and as a function of the round, $\langle H_B\rangle_Q^{(k)} =
(1-k/2^n)\,h(\sigma(\beta))$ — asked columns zero out, unasked columns
never move. (Verified to machine precision in `entropy_identities.py`;
the same factorization gives $m\,h(\sigma(\beta))$ per column for $m>1$,
and survives replacing popcount by Hamming distance to any fixed
reference map, since that's just a per-column relabeling $0\leftrightarrow 1$.)
The moral cuts both ways: this whole family of priors is exactly
solvable *because* $C=0$ — and $C=0$ means zero generalization. An
additive-over-bits energy is a "non-interacting" prior. Genuine circuit
complexity is not of this form — an AIG's size couples the answer bits
to each other — and that coupling is precisely what lets a
complexity-based prior transfer information to unasked columns.
(Energies that couple *pairs* of answer bits turn the prior into an
Ising model on the question set, with column marginals as magnetizations
and the mutual-information transfer as spin-spin correlations — beyond
elementary functions, but squarely in statistical-mechanics territory.)

**Solvable case 2: a generic prior, in the limit.** Draw $p$ uniformly at
random from the $N$-simplex (Dirichlet with all parameters $1$). Each
column probability is a sum of exactly $N/2$ coordinates, which is
$\mathrm{Beta}(N/2,\,N/2)$-distributed: sharply concentrated at
$\tfrac12$ with variance $\tfrac{1}{4(N+1)}$. Expanding
$h(\tfrac12+\varepsilon) = 1 - \tfrac{2}{\ln 2}\varepsilon^2 +
O(\varepsilon^4)$ and taking expectations,

$$
\mathbb{E}\big[\,m-\langle H_B\rangle_Q\,\big]
= \frac{1}{2\ln 2\,(N+1)} + O(N^{-2}),
\qquad N = 2^{2^n},
$$

doubly exponentially small in $n$ (numerics at $n=3$: measured
$0.002812$ vs. predicted $0.002807$). And the $k$-dependence rides along
for free: conditioning a flat Dirichlet on the survivors of $k$ answers
leaves a flat Dirichlet on the $N_k = 2^{2^n-k}$ survivors, so the same
formula applies with $N\to N_k$ — the reduction stays negligible until
$2^n - k$ is $O(1)$. A generic prior predicts essentially nothing until
nearly every question has already been asked: the mass of the simplex
sits at "maximally uncertain in every column," and only the collapse of
$N_k$ to a handful of survivors pulls the columns away from the fair
coin. Structure (case 1's bias, or a complexity energy) is not a nicety
— without it, prediction before the final rounds is doubly exponentially
close to worthless.

### General $m$: base-$2^m$ digits

Everything above fixed $m=1$ so the index could be read off in plain
binary. The natural generalization: write $j-1$ in base $2^m$ instead of
base $2$ — one *digit* per question, each digit ranging over all of
$A=\{0,\ldots,2^m-1\}$ rather than just a bit:

$$
j-1 = \sum_{q=0}^{2^n-1} \phi_j(q)\cdot(2^m)^q,
\qquad
\phi_j(q) = \mathrm{digit}_q^{(2^m)}(j-1),
$$

directly generalizing "bit $q$ of $j-1$" to "base-$2^m$ digit $q$ of
$j-1$." Column $q$ of $M$ is the full distribution of that digit's $2^m$
possible values, and $H_B(q)=H(M_{\cdot,q})$ is now bounded by $m$, not
$1$.

Nothing in the derivation of the general identity used $m=1$: the
answers still jointly determine the hypothesis, so
$H(A_{q_1},\ldots)=H(p)$ still holds, subadditivity still gives
$C(p)\geq0$, and

$$
\langle H_B\rangle_Q^{(k)} = \frac{H(p^{(k)})+C_k}{2^n},
\qquad
0\;\leq\;\langle H_B\rangle_Q^{(k)}\;\leq\; m\Big(1-\frac{k}{2^n}\Big),
$$

is exact for every $m$, unchanged. What *does* change with $m$ is the two
solvable cases.

**Case 1, revisited.** An energy additive over positions,
$E(j)=\sum_q c(\phi_j(q))$ for any cost $c:A\to\mathbb R$, still
factorizes the partition function over the $2^n$ questions — the
argument never used $|A|=2$ — giving $2^n$ independent, identically
distributed digits drawn from a single-symbol Boltzmann distribution
$\pi_a\propto e^{-\beta c(a)}$, so $\langle H_B\rangle_Q^{(k)} =
(1-k/2^n)H(\pi_\beta)$ and $C_k=0$ again. Now *which* $c$ you pick starts
to matter, in a way it couldn't when there was only one bit to cost:

- $c(a)=\mathrm{popcount}_2(a)$ — penalize each of the $m$ *bits within
  the answer* independently — makes $\pi_\beta$ a product of $m$
  independent Bernoulli$(\sigma(-\beta))$ bits, so
  $H(\pi_\beta)=m\,h(\sigma(\beta))$: exactly $m$ non-interacting copies
  of the $m=1$ result. The answer's own bits carry no information about
  each other.
- $c(a)=a$ — cost the answer's integer value directly, which *does*
  couple its $m$ bits (symbol $3$ costs more than symbol $2$, even
  though both have different bit-popcounts) — makes $\pi_\beta$ a
  truncated geometric distribution on $\{0,\ldots,2^m-1\}$ with ratio
  $r=e^{-\beta}$:

$$
\pi_a = \frac{r^a}{z(\beta)},\qquad
z(\beta)=\sum_{a=0}^{2^m-1} r^a = \frac{1-r^{2^m}}{1-r},
$$

  still an elementary (finite geometric) closed form, with entropy

$$
H(\pi_\beta) = \log_2 z(\beta) + \frac{\beta}{\ln 2}\left[\frac{r}{1-r} -
\frac{2^m r^{2^m}}{1-r^{2^m}}\right]
$$

  (the bracket is the mean of the truncated geometric — verified to $6$
  decimal places against direct computation of $H(\pi_\beta)$ for
  $n=2,m=2$). The two choices of $c$ agree at $m=1$ (both give
  $h(\sigma(\beta))$) and diverge for $m\geq2$: either way $H_B(q)$ is
  the marginal uncertainty of a whole answer, but only the second energy
  makes the *bits within one answer* informative about each other — a
  distinction that simply doesn't exist when $m=1$.

**Case 2, revisited.** The generic-prior calculation goes through with
one change: each column is now a sum over $2^m$ (not $2$) equal-size
blocks of a flat Dirichlet($1$) prior on $N=(2^m)^{2^n}$ hypotheses, so
by the aggregation property of the Dirichlet distribution, column $q$'s
answer distribution is *exactly* $\mathrm{Dirichlet}(N/2^m,\ldots,N/2^m)$
with $2^m$ parts. The same second-order expansion around the uniform
point — now uniform over $2^m$ symbols instead of $2$ — gives

$$
\mathbb{E}\big[m-\langle H_B\rangle_Q\big] = \frac{2^m-1}{2\ln2\,(N+1)}
+ O(N^{-2}), \qquad N=2^{m\cdot2^n},
$$

the $m=1$ formula times $(2^m-1)$: a generic prior's ignorance about
*which of the $2^m$ answers* is right scales with how many wrong answers
there are to be ignorant about, while the doubly-exponential smallness in
$N$ — and hence in both $n$ *and* $m$ — is unchanged. This survives
conditioning on $k$ answered questions with $N\to N_k=2^{m(2^n-k)}$
exactly as before (verified numerically, including the more demanding
$n=3,m=2,k=3$ case: measured $0.00207$ vs. predicted $0.00211$).

## $\epsilon_j$ under uniform $P(q)$

One immediate simplification we will consider is taking $P(q)$ uniform.

This isn't as strange as it looks: by a Shannon source-coding
argument, any question source can be re-encoded to be arbitrarily close to
white noise, so uniform $P$ is close to the general case after the right
encoding, not a special one. The author is aware that this is a bit at odds with the philosophy of working in a representation that also expects low circuit complexity. Nonetheless, it's worth it for the simplification it affords.

With uniform question frequency, every term in the sum defining $\epsilon_j$ carries
the same weight $P(q) = 2^{-n}$, so it factors out:

$$
\epsilon_j = \sum_{q\in Q} 2^{-n}\,\delta\big(\psi(q),\phi_j(q)\big) = 2^{-n}\big|\{q \in Q : \psi(q) = \phi_j(q)\}\big|.
$$

That is, $\epsilon_j$ is just $2^{-n}$ times a *count*: the number of
questions where $\psi$ and $\phi_j$ agree, out of $2^n$ total. Since that
count is an integer between $0$ and $2^n$, $\epsilon_j$ can only take the
$2^n+1$ values $0, \tfrac{1}{2^n}, \tfrac{2}{2^n}, \dots, 1$ — a discrete
set of fractional options, not a continuum.

### Counting hypotheses by $\epsilon_j$ (a binomial)

How many hypotheses share a given $\epsilon_j$ is a simple counting
argument. Fix $\psi$. Under uniform $P(q)$, $\epsilon_j$ depends only on
$d$, the number of questions (out of $2^n$) where $\phi_j$ disagrees with
$\psi$: $\epsilon_j = (2^n-d)/2^n$. To build such a $\phi_j$, choose which
$d$ of the $2^n$ questions are wrong ($\binom{2^n}{d}$ ways), and for each
of those give any of the $2^m-1$ incorrect answers:

$$
\#\{\phi_j : \epsilon_j = (2^n-d)/2^n\} = \binom{2^n}{d}(2^m-1)^d.
$$

This is independent of which $\psi$ we fixed (relabeling doesn't change
shell sizes), and sums to $N$ by the binomial theorem:
$\sum_{d=0}^{2^n}\binom{2^n}{d}(2^m-1)^d = (2^m)^{2^n} \equiv N$.

## Gibbs prior over circuit complexity

A more principled version of a hand-picked complexity-biased prior: treat
circuit complexity $C_j$ (AIG size, from the classification tables) as an
energy, and set

$$
p_j^{\mathrm{Boltz}}(\beta) = \frac{e^{-\beta C_j}}{\sum_{k=1}^N e^{-\beta C_k}}.
$$

Higher $\beta$ ("colder") concentrates belief ever more sharply on the
lowest-complexity hypotheses (a stronger Occam's-razor bias); $\beta \to 0$
flattens toward the uniform prior. For the running $n=2,m=1$ hypothesis
space, the 16 complexities are:

| $j$ | name | $C_j$ (AIG size) |
|---|---|---|
| 1  | FALSE (constant 0) | 0 |
| 2  | $a \wedge b$ (AND) | 1 |
| 3  | $a \wedge \lnot b$ | 1 |
| 4  | $a$ (projection) | 0 |
| 5  | $\lnot a \wedge b$ | 1 |
| 6  | $b$ (projection) | 0 |
| 7  | $a \oplus b$ (XOR) | 3 |
| 8  | $a \vee b$ (OR) | 1 |
| 9  | $\lnot(a \vee b)$ (NOR) | 1 |
| 10 | $a = b$ (XNOR) | 3 |
| 11 | $\lnot b$ | 0 |
| 12 | $b \to a$ (IF THEN) | 1 |
| 13 | $\lnot a$ | 0 |
| 14 | $a \to b$ (IF THEN) | 1 |
| 15 | $\lnot(a \wedge b)$ (NAND) | 1 |
| 16 | TRUE (constant 1) | 0 |

i.e. 6 functions at complexity 0, 8 at complexity 1, and 2 (XOR/XNOR) at
complexity 3 — three clean tiers, distinct from the $a$-vs-$b$ and
$0$-vs-$1$ biases used for the prior in the Example section above.

## Worked example: $\epsilon_j$ and a complexity-based prior

Take the running $n=2, m=1$ hypothesis space (16 functions) and assume
$P(q)$ uniform ($P(q)=1/4$ for each of the four questions). 

Fix a true map, $\psi = $ AND, and compute $\epsilon_j$ for all 16
hypotheses (fraction of the 4 questions each $\phi_j$ gets right against
$\psi$ — as derived above, just an agreement count over $2^n=4$). Names
and truth tables for each $\phi_j$ are the same as in the Example section
above; $C_j$ is tabulated just above, in the Gibbs-prior section; sorted
ascending by $\epsilon_j$ they give the staircase below:

![epsilon_j sorted ascending against psi=AND](figures/epsilon_sorted.png)

The staircase shape is just an artifact of Hamming distance on 4-bit truth
tables: only steps of $1/4$ are possible, so $\epsilon_j \in \{0, .25, .5,
.75, 1\}$. Applying the counting argument above with $n=2,m=1$
($2^n=4$, $2^m-1=1$), the plateau widths are exactly $\binom{4}{d}$ — a row
of Pascal's triangle: $1,4,6,4,1$.

Now apply the Gibbs prior over complexity to this example:

![Boltzmann prior over complexity, same rank order as epsilon_j](figures/epsilon_boltzmann.png)

Plotted in the *same* rank order as the $\epsilon_j$ chart above: AND — the
true map here — sits at $C_j=1$, one of the eight complexity-1 functions,
so it gets a solidly middling Boltzmann weight at every $\beta$ shown
(tied with NAND, OR, NOR, and the other complexity-1 functions) —
comfortably ahead of the two complexity-3 functions (XOR, XNOR), which get
driven toward zero weight as $\beta$ grows, but behind the six
complexity-0 functions (the constants, projections, and negations), which
the prior favors most of all. Unlike a case where the truth itself happens
to be the rarest, most complex function, here the complexity bias is
roughly neutral toward the truth: it neither actively fights learning AND
nor goes out of its way to help beyond what a flat preference for
"not too complex" already provides.

## Appendix: circuit complexity methods

**Circuit complexity** here means minimum **AIG size**: the fewest 2-input
AND gates needed to realize a function, where any wire (a gate's input, or
the circuit's output) may be freely inverted — inversions are edge
attributes, not gates, and don't add to the count. This is the standard
And-Inverter-Graph metric used in logic synthesis (e.g. ABC, mockturtle).
Reference values: AND = 1, OR = 1, XOR = 3, NAND = 1.

For the **3-bit -> 2-bit** table, complexity is the size of one *joint*
circuit computing both output bits together, allowing gates to be shared
between them (the realistic "minimum hardware" cost), not the sum of two
independent single-output circuits.

### Equivalence classes

Both AIG size and joint AIG size are invariant under relabeling: permuting
inputs, inverting any input, inverting either output, and (for the 2-output
case) swapping which output is which — all free operations. So instead of
solving all 65536 functions per table, each table collapses to a much
smaller set of equivalence classes:

- **4 -> 1** (NPN group: permute + negate inputs, negate output, order
  768): **222 classes**.
- **3 -> 2** (permute + negate inputs, negate/swap outputs, order 384):
  **308 classes**, computed the same way (`groups.py`).

Every one of the 65536 rows in each table inherits its complexity value
from its class's representative.

### Our synthesis method

For a target function (or pair of functions, for joint synthesis), we
search circuit sizes r = 0, 1, 2, ... and encode "does an r-AND-gate AIG
compute this?" as a SAT instance (`aig_synth.py`, via CaDiCaL through
python-sat): each gate selects two earlier signals (inputs, constant 0, or
earlier gates) with independent free-inversion flags, and each output
selects a signal and inversion. The first satisfiable r is the minimum.
Symmetry-breaking clauses (no two gates computing an identical function; no
gate that is neither an output nor feeds a later gate) prune the search
substantially. The first SAT r found is provably minimal, since smaller r
was already proven UNSAT.

We validated this against the krinkin/bounds data on 10 random NPN4 classes
(0 mismatches) before trusting it for the 3->2 table, which has no external
reference.

### What we tried and discarded

- A **"pooled closure" search** (iteratively taking the frontier of all
  reachable functions and combining all pairs) looked like it computed
  minimum size cheaply, but turned out to compute minimum circuit *depth*
  instead — it silently let two derivations that don't actually share any
  gates split their cost for free. Caught by hand-tracing NAND-basis XOR:
  the closure reported 3, matching XOR's true circuit *depth*, but true
  minimum NAND *size* is 4 (confirmed by SAT) — the two intermediate terms
  in the closure's derivation both depended on a common subterm that isn't
  actually free to reuse across them.
- **Random circuit construction** (building genuine random AND-gate chains
  and reading off true minimum sizes for whatever functions appear) is
  sound but has poor coverage: even 3,000,000 random trials found only
  ~35% of the 222 NPN4 classes, because random small circuits are strongly
  biased toward "simple" functions. Useful as a fast partial upper-bound
  pass, not as a complete method.

## Sources

- **[krinkin/bounds](https://github.com/krinkin/bounds)** (reproducibility
  data for *"A Simple Constructive Bound on Circuit Size Change Under Truth
  Table Perturbation"*): published optimal AIG sizes for all 222 NPN classes
  of 4-variable functions (220 exact, 2 upper bounds). No synthesis method
  was documented in that repo — it's a data artifact, not a tool.
- Our own **SAT-based exact synthesis**, used for the 3->2 table, which has
  no equivalent published dataset.
- **Shannon**, *A Mathematical Theory of Communication* (1948): the
  source-coding argument behind treating $P(q)$ as uniform in the worked
  example — a source can always be re-encoded arbitrarily close to its
  entropy bound (i.e. toward white noise) via a suitable code.
