# The thermodynamic limit

The earlier chapters describe learning on a finite question set. This chapter asks what remains when the number of input bits tends to infinity. The question set and its size are

$$
Q:=\{0,1\}^n,
\qquad
|Q|=2^n,
$$

so $n\to\infty$ is an exponentially growing-table limit. We measure time by the fraction of the table already queried,

$$
x=\frac{\ell}{|Q|}\in[0,1].
$$

The limit at fixed $x>0$ is macroscopic: it records learning that occupies a nonzero fraction of the run. Learning completed in $O(1)$, $O(\log |Q|)$, or more generally $o(|Q|)$ questions is compressed into a boundary layer at $x=0$. This distinction will explain both the hyperbolas below and several apparent noncommutations of limits.

Throughout, an answer contains $m$ bits. Write

$$
A:=\{0,1\}^m,
\qquad
|A|=2^m,
\qquad
u:=|A|^{-1}=2^{-m},
\qquad
N:=|A|^{|Q|}=u^{-|Q|}=2^{m2^n}
$$

for the answer-alphabet size and the number of maps $Q\to A$. Enumerate those maps as $\phi_0,\ldots,\phi_{N-1}$, with prior weights $p_0,\ldots,p_{N-1}$. The random true map is $\psi$, with

$$
\Pr(\psi=\phi_j)=p_j,
\qquad
\sum_{j=0}^{N-1}p_j=1.
$$

One value of $\psi$ is drawn at the start of a learning run and supplies every answer in that run. Bayesian posteriors are conditional distributions of this same random map, and expected trajectories average over its prior law.

Fix an ordering $q_1,\ldots,q_{|Q|}$ of the questions. For a fixed set of indices $T\subseteq\{1,\ldots,|Q|\}$, define

$$
\psi(T):=\bigl(\psi(q_k)\bigr)_{k\in T}\in A^{|T|},
$$

where the components are written in increasing order of $k$. Thus $T$ specifies which positions in the fixed question ordering are included. For one question $q_k$, the realized answer is simply

$$
a=\psi(q_k)\in A.
$$

For a fixed candidate map, use the parallel notation $\phi_j(T):=(\phi_j(q_k))_{k\in T}$. Hence, for $y\in A^{|T|}$,

$$
\Pr\bigl(\psi(T)=y\bigr)
=\sum_{j=0}^{N-1}p_j\,
\mathbf 1\!\left[\phi_j(T)=y\right].
$$

The joint Shannon entropy of this random answer vector is

$$
H(\psi(T))
:=-\sum_{y\in A^{|T|}}\Pr(\psi(T)=y)
\log_2\Pr(\psi(T)=y).
$$

Here $T$ is fixed: the entropy measures uncertainty in the answers, not uncertainty about which questions were selected.

All entropies are in bits. As in the chapter on expected trajectories, the truth is drawn from the learner's prior and the questions are asked in uniformly random order. Leverage is always the ratio of expected totals, not the expectation of branchwise ratios.

For later use, the binary entropy function is

$$
h_2(z):=-z\log_2z-(1-z)\log_2(1-z),
$$

with $0\log 0:=0$. The subscript emphasizes that $h_2$ is binary entropy; categorical entropies continue to use $H$.

## 1. A toy example: the spike prior and its renormalization flow

The spike prior places a distinguished weight on one map and spreads the remaining mass uniformly over every other map. We take the special map to be the all-zero map,

$$
\phi_0(q)=0^m\qquad(q\in Q),
$$

so its answer to every question is the $m$-bit zero word and its answer to any $k$ questions is the all-zero block $0^{mk}$. This is without loss of generality: the answer alphabet can be relabelled separately at each question.

$$
p_0=p,
\qquad
p_j=\omega:=\frac{1-p}{N-1}\quad(j\ne0).
\tag{1.1}
$$

The symbols $p_j$ denote the map probabilities; the plain $p=p_0$ is only the mass of the all-zero map.

This parameterization makes $p$ the exact mass of the special map. Its uniform point is $p=1/N$, not $p=0$. We normally consider $p\ge 1/N$; values below that point describe an anti-spike.

The example is useful because it is closed under Bayesian conditioning. A confirming answer produces a sharper spike on a smaller map space; a refuting answer removes the special map and leaves equal weights on all survivors. The posterior therefore has a two-state flow. The uniform family is the stable, absorbing class: once a trajectory enters it, later conditioning remains uniform. An interior spike is a source of one-way probability current into that class. Conditional on avoiding refutation, however, the surviving spike sharpens toward one. The flow and the leverage calculation below are two descriptions of this same branch structure.

### 1.1 The two-state flow and exact leverage

Let $T_k:=\{1,\ldots,k\}$ in the fixed ordering above. A history remains on the spike branch for $k$ steps exactly when $\psi(T_k)=0^{mk}$. Exactly

$$
N_k:=Nu^k
$$

maps produce this confirming history, one of which is $\phi_0$. Its prior probability is therefore

$$
P_k:=\Pr\bigl(\psi(T_k)=0^{mk}\bigr)
=p+(N_k-1)\omega
=p+(Nu^k-1)\omega.
\tag{1.2}
$$

On this event, the posterior is another spike on those $N_k$ maps, with

$$
p_k=\Pr\bigl(\psi=\phi_0\mid\psi(T_k)=0^{mk}\bigr)
=\frac{p}{P_k},
\qquad
\omega_k=\frac{\omega}{P_k}.
\tag{1.3}
$$

Here $p_k$ is the time-indexed posterior height of the spike, evaluated on the realized branch $\psi(T_k)=0^{mk}$; at $k=0$ it is exactly the original map weight $p_0=p$.

For $0\le k<|Q|$, the next zero answer is predicted by the special map and by $uN_k-1$ of the surviving background maps. Its conditional probability is

$$
\Pr\bigl(\psi(q_{k+1})=0^m\mid\psi(T_k)=0^{mk}\bigr)
=p_k+(uN_k-1)\omega_k
=\frac{P_{k+1}}{P_k},
\qquad
p_{k+1}=p_k\frac{P_k}{P_{k+1}},
\qquad
\boxed{P_kp_k=p_0=p}.
\tag{1.4}
$$

The boxed product is the Bayesian consistency condition that organizes the example. The probability $P_k$ of remaining on the spike branch decreases while the conditional spike height $p_k$ increases, but their product stays equal to the original mass $p$. Equivalently, $\Pr(\psi=\phi_0\mid\psi(T_k))$ is a martingale under the Bayes average over histories. It is $p_k$ after $k$ confirmations and zero after refutation, so its expectation is always $P_kp_k=p$. Renormalization therefore sharpens the surviving spike without creating unconditional probability.

If any answer is nonzero, $\phi_0$ is eliminated. Every compatible survivor then had the same mass $\omega$, so the posterior is uniform. Subsequent conditioning preserves uniformity. Thus the two macrostates and their directions are

$$
\text{surviving spike}\longrightarrow
\begin{cases}
\text{surviving, sharper spike},&\text{with probability }P_{k+1}/P_k,\\
\text{uniform posterior},&\text{with probability }1-P_{k+1}/P_k,
\end{cases}
\qquad
\text{uniform}\longrightarrow\text{uniform}.
$$

For $0\le k<|Q|$ and $p<1$, the exact odds on the surviving branch are

$$
\frac{p_k}{1-p_k}
=\frac{p(N-1)}{(1-p)(Nu^k-1)}.
\tag{1.5}
$$

For fixed $p>0$, (1.2)--(1.3) give

$$
P_k=p+(1-p)u^k+O(N^{-1}),
\qquad
p_k=\frac{p}{p+(1-p)u^k}+O(N^{-1}),
\tag{1.6}
$$

If, in addition, $0<p<1$ and $u^{|Q|-k}\to0$, then the finite-population correction disappears and

$$
\frac{p_k}{1-p_k}
\simeq u^{-k}\frac{p}{1-p}.
\tag{1.7}
$$

Each confirmation supplies approximately $-\log_2u=m$ bits of evidence for the spike. Because $N=u^{-|Q|}$, (1.2) gives $P_{|Q|}=p$ and (1.3) then gives $p_{|Q|}=1$: the histories that confirm through the complete table are precisely those with $\psi=\phi_0$. At the uniform point $p=1/N$, both posterior branches are uniform on their surviving map spaces. At the opposite endpoint $p=1$, the delta state is also absorbing. For $1/N<p<1$, false-spike histories flow irreversibly into the uniform class, while the confirming history sharpens toward the delta state.

The same $P_k$ also has a second, equivalent interpretation: it is the probability of the single $k$-answer block $0^{mk}$. There is no new distribution to calculate. For $1\le k\le |Q|$, every other answer block is realized by $N_k$ background maps and therefore has the common probability

$$
U_k
:=\frac{1-P_k}{u^{-k}-1}
=N_k\omega.
\tag{1.8}
$$

Thus, for any index set $T\subseteq\{1,\ldots,|Q|\}$ of size $k$,

$$
\Pr(\psi(T)=y)
=
\begin{cases}
P_k,&y=0^{mk},\\
U_k,&y\ne0^{mk}.
\end{cases}
\tag{1.9}
$$

The entropy of this block distribution is

$$
G_k
=-P_k\log_2P_k-(u^{-k}-1)U_k\log_2U_k.
\tag{1.10}
$$

Every choice of $k$ question indices has this same law, so (1.10) is also the mean $k$-question block entropy defined in the earlier chapter. In particular,

$$
G_0=0,
\qquad
G_{|Q|}=H(\psi),
\qquad
H(M)_0=|Q|G_1.
\tag{1.11}
$$

The exact prior entropy is therefore

$$
G_{|Q|}
=h_2(p)+(1-p)\log_2(N-1),
\tag{1.12}
$$

where $h_2$ is binary entropy. This is extensive in $|Q|$ at fixed $0<p<1$, but vanishes at $p=1$.

The two-state flow now gives the finite learning trajectory without another posterior calculation. The entropy chain rule says that joint uncertainty equals the entropy of what is observed first plus the average uncertainty left in what is observed next. For any $T\subseteq\{1,\ldots,|Q|\}$ and $i\notin T$,

$$
H\bigl(\psi(T\cup\{i\})\bigr)
=H\bigl(\psi(T)\bigr)
+H\bigl(\psi(q_i)\mid\psi(T)\bigr).
$$

Here the conditional entropy averages over the possible values of the random answer history $\psi(T)$. Consequently, $G_{k+1}-G_k$ is the expected conditional entropy of one fresh answer after $k$ answers have been observed. The expected information received is the entropy of the observed block,

$$
\mathbb E[\text{received after }k]=G_k,
\tag{1.13}
$$

while each of the $|Q|-k$ unasked questions has mean conditional entropy $G_{k+1}-G_k$. Hence

$$
\mathbb E[H(M_k)]=(|Q|-k)(G_{k+1}-G_k),
\tag{1.14}
$$

for $0\le k<|Q|$, while the remaining entropy is zero at $k=|Q|$. Therefore, for $1\le k<|Q|$ and $p<1$,

$$
\boxed{
\langle L_k\rangle
=\frac{|Q|G_1-(|Q|-k)(G_{k+1}-G_k)}{G_k}.}
\tag{1.15}
$$

At completion, again for $p<1$,

$$
\langle L_{|Q|}\rangle=\frac{|Q|G_1}{G_{|Q|}}.
\tag{1.16}
$$

Thus one counting formula, (1.2), determines the Bayesian flow, the answer-block entropy, and the exact leverage. No separate finite-trajectory ansatz is required.

### 1.2 One question, branch by branch

The next four graphs plot only the following branch formulas. Here “correct” means that the observed answer agrees with the spike map at this question; it does not assert that the entire spike map is the truth. For one question, abbreviate

$$
P:=P_1
=p+(Nu-1)\omega,
\qquad
U:=U_1=Nu\omega.
\tag{1.17}
$$

Here $P+(|A|-1)U=1$: the zero answer has probability $P$, while each particular nonzero answer has probability $U$.

**Surprisal curves.** The green, red, and gray solid curves are

$$
s_{\rm correct}=-\log_2P,
\qquad
s_{\rm wrong}=-\log_2U,
\qquad
\mathbb E[s]
=P s_{\rm correct}+(|A|-1)U s_{\rm wrong}
=G_1.
\tag{1.18}
$$

Surprisal is minus the logarithm of the probability of the answer actually observed; a particular wrong answer has probability $U$, not $1-P$. Averaging over the correct symbol and the $|A|-1$ wrong symbols gives the one-answer entropy $G_1$.

**Entropy before asking.** The blue dotted curve is

$$
H_{\rm before}:=H(M)_0=|Q|G_1.
\tag{1.19}
$$

Every one of the $|Q|$ columns initially has the same answer distribution $(P,U,\ldots,U)$ and hence entropy $G_1$. Summing those marginal entropies gives $|Q|G_1$.

**Entropy after a correct answer.** Conditional on confirmation, the next unasked answer has probabilities $P_2/P$ for zero and $U_2/P$ for each nonzero symbol. Its entropy is therefore

$$
H_{\rm c}
:=-\frac{P_2}{P}\log_2\frac{P_2}{P}
  -(|A|-1)\frac{U_2}{P}\log_2\frac{U_2}{P}.
\tag{1.20}
$$

The asked column is fixed, while all $|Q|-1$ unasked columns have this conditional marginal. The green dashed curve is consequently

$$
H_{\rm after,correct}=(|Q|-1)H_{\rm c}.
\tag{1.21}
$$

**Entropy after a wrong answer.** Refutation eliminates $\phi_0$ and leaves equal posterior weights on all compatible maps. Every unasked answer is therefore uniform on $|A|=2^m$ symbols, giving the red dashed curve

$$
H_{\rm after,wrong}=(|Q|-1)m.
\tag{1.22}
$$

**Expected entropy after asking.** Weighting the two posterior branches gives the gray dashed curve

$$
\mathbb E[H(M_1)]
=P H_{\rm after,correct}
 +(|A|-1)U H_{\rm after,wrong}
=(|Q|-1)(G_2-G_1).
\tag{1.23}
$$

The first equality is ordinary branch averaging. The second is the $k=1$ case of the chain-rule trajectory (1.14).

![One-question spike anatomy for n=4, m=1: branch surprisals and answer-matrix entropy before and after the update.](figures/ansatz_spike_4to1.png)

![One-question spike anatomy for n=3, m=2. A particular wrong answer has probability U, not 1-P.](figures/ansatz_spike_3to2.png)

For $p<1$, the leverage plots divide the vertical difference between the blue curve and each dashed branch curve by that branch's surprisal. Their green and red curves are therefore

$$
L_{\rm correct}
=\frac{|Q|G_1-(|Q|-1)H_{\rm c}}{-\log_2P},
\qquad
L_{\rm wrong}
=\frac{|Q|G_1-(|Q|-1)m}{-\log_2U}.
\tag{1.24}
$$

The gray curve is the ratio of the two expected totals:

$$
\boxed{
\langle L_1\rangle
=\frac{
P[|Q|G_1-(|Q|-1)H_{\rm c}]
+(|A|-1)U[|Q|G_1-(|Q|-1)m]
}{G_1}
=\frac{|Q|G_1-(|Q|-1)(G_2-G_1)}{G_1}.}
\tag{1.25}
$$

The branch formulas use the realized surprisal, whereas the gray formula divides the expected entropy drop by the expected surprisal $G_1$; it is not the probability-weighted average of the green and red ratios. The dotted guide $L=1$ is the uniform-prior baseline, and the black line is the zero reference.

![Branchwise and expected first-question leverage for n=4, m=1.](figures/ansatz_leverage_4to1.png)

![Branchwise and expected first-question leverage for n=3, m=2.](figures/ansatz_leverage_3to2.png)

### 1.3 The extraction ratio

The total correlation

$$
C:=H(M)_0-G_{|Q|}=|Q|G_1-G_{|Q|}
\tag{1.26}
$$

is the information stored in dependencies among answers rather than visible in their separate marginals. From (1.25), the expected entropy drop after one question is

$$
\mathbb E[\Delta H(M)]
=|Q|G_1-(|Q|-1)(G_2-G_1).
\tag{1.27}
$$

Only the part beyond the received entropy $G_1$ comes from the dependency store. When $C>0$, its extraction ratio is therefore

$$
\boxed{
R_1
:=\frac{\mathbb E[\Delta H(M)]-G_1}{C}
=\frac{(|Q|-1)(2G_1-G_2)}{|Q|G_1-G_{|Q|}}.}
\tag{1.28}
$$

The symbol $R_1$ will mean this extraction ratio throughout the chapter. It uses the entire dependency store $C$ as its denominator. Leverage instead uses the received surprisal $G_1$:

$$
\boxed{
R_1=\frac{G_1}{C}\bigl(\langle L_1\rangle-1\bigr),
\qquad
\langle L_1\rangle=1+\frac{C}{G_1}R_1.}
$$

Thus $R_1$ is a stock-extraction fraction, whereas $\langle L_1\rangle-1$ is released dependency information per received bit.

At $p=1$, both the numerator of $R_1$ and $C$ vanish. Expanding (1.10) with $\varepsilon=1-p\to0^+$ gives the exact finite-$(n,m)$ ratio limit

$$
\boxed{
\lim_{p\to1^-}R_1(p)
=\frac{(|Q|-1)N(1-u)^2}
       {|Q|N(1-u)-(N-1)}.}
\tag{1.29}
$$

The simpler expression

$$
\frac{(|Q|-1)(1-u)^2}{|Q|(1-u)-1}
\tag{1.30}
$$

is the additional large-$N$ approximation. For example, at $(n,m)=(2,1)$, (1.29) is $12/17\simeq0.706$, whereas (1.30) gives $0.75$. Both have the same thermodynamic limit:

$$
\boxed{
\lim_{n\to\infty}\lim_{p\to1^-}R_1(p)=1-u=1-2^{-m}.}
\tag{1.31}
$$

![Exact one-question extraction ratio for the spike prior, with varying n and m.](figures/extraction_ratio.png)

The solid curves are the exact finite-$N$ values of $R_1$ from (1.28), plotted only where the ratio is defined. The dashed lines are the exact finite-$N$ limits (1.29) as $p\to1^-$.

For the spike-plus-uniform family, (1.31) is a theorem. Extending it to a broader class of “uninformative hedges” would require a precise definition and a separate extremal argument; full support alone is not sufficient.

### 1.4 The sharp-prior and thermodynamic limits

The sharp-prior calculation means taking $\varepsilon=1-p\to0^+$ with $n$ and $m$ fixed. It is a limit toward the delta prior, not evaluation at $p=1$, where leverage is $0/0$. Define the entropy coefficient

$$
\iota_k
:=\frac{N}{N-1}(1-u^k).
\tag{1.32}
$$

Then $P_k=1-\varepsilon\iota_k$, the other $u^{-k}-1$ outcomes share mass $\varepsilon\iota_k$, and

$$
G_k
=h_2(\varepsilon\iota_k)
+\varepsilon\iota_k\log_2(u^{-k}-1).
\tag{1.33}
$$

At fixed finite $|Q|$ and $k$,

$$
G_k
\sim\varepsilon\iota_k\log_2\frac1\varepsilon
\qquad(\varepsilon\downarrow0).
\tag{1.34}
$$

Substitution into (1.15), with (1.16) at completion, cancels the common factor and gives, for $1\le k\le |Q|$,

$$
\boxed{
\lim_{p\to1^-}\langle L_k\rangle
=(1-u)\frac{|Q|-(|Q|-k)u^k}{1-u^k},
\qquad u=2^{-m}.}
\tag{1.35}
$$

Equivalently,

$$
\lim_{p\to1^-}\langle L_k\rangle
=|Q|(1-u)+(1-u)\frac{k u^k}{1-u^k}.
\tag{1.36}
$$

The first term is extensive in the number of questions. The second is an $O(1)$ boundary correction that decays geometrically with $k$. In particular,

$$
\lim_{p\to1^-}\langle L_1\rangle
=|Q|-(|Q|-1)2^{-m},
\tag{1.37}
$$

and therefore

$$
\boxed{
\lim_{n\to\infty}\lim_{p\to1^-}
\frac{\langle L_1\rangle}{|Q|}
=1-2^{-m}.}
\tag{1.38}
$$

The coefficient $1-2^{-m}$ is the fraction of the hypothesis count eliminated by one answer. The equality with normalized leverage follows from the entropy asymptotics; it should not be interpreted as literally settling the same fraction of every individual column.

![The sharp-prior first-question leverage, normalized by the number of questions, approaches the culling coefficient.](figures/ansatz_culling_ratio.png)

The limiting operation is necessary. At $p=1$, the learner already assigns probability one to the truth, so both received and destroyed information are zero and leverage is $0/0$. The finite value in (1.37) is the ratio at which these two quantities vanish as $p\to1^-$.

For $k=\lfloor x|Q|\rfloor$, $x>0$, (1.36) also gives

$$
\lim_{n\to\infty}\lim_{p\to1^-}
\frac{\langle L_{\lfloor x|Q|\rfloor}\rangle}{|Q|}
=1-2^{-m}.
\tag{1.39}
$$

Thus the normalized sharp-prior curve approaches a constant plateau. In fact, (1.36) gives the uniform bound

$$
\sup_{1\le k\le |Q|}
\left|
\frac1{|Q|}\lim_{p\to1^-}\langle L_k\rangle-(1-u)
\right|
=O(|Q|^{-1}).
\tag{1.40}
$$

The microscopic boundary term survives only as an $O(1)$ correction to the unnormalized leverage.

![Sharp-prior spike flow as a function of the asked fraction.](figures/spike_flow_leverage.png)

The other order of limits has a different scale. Fix $0<p<1$ and $x>0$, then send $n\to\infty$. The limiting probability of the spike answer in one column is

$$
p+(1-p)u.
\tag{1.41}
$$

and the corresponding one-column entropy is

$$
h_{\rm spike}(p,m)
:=h_2\!\big(p+(1-p)u\big)
 +(1-p)(1-u)\log_2(|A|-1).
\tag{1.42}
$$

At any fixed $x>0$, the conditional spike height $p_k$ on a surviving confirming branch has approached one within a vanishing fraction of the run. The branch itself still occurs with limiting probability $p$; the complementary probability $1-p$ lies on uniform refuted branches. Consequently

$$
\frac{\mathbb E[H(M_{\lfloor x|Q|\rfloor})]}{|Q|}
\longrightarrow(1-x)(1-p)m,
\tag{1.43}
$$

$$
\frac{G_{\lfloor x|Q|\rfloor}}{|Q|}
\longrightarrow x(1-p)m.
\tag{1.44}
$$

To obtain leverage, invoke the exact finite formula (1.15). Since $G_1\to h_{\rm spike}(p,m)$, divide its numerator and denominator by $|Q|$ and apply (1.43)--(1.44):

$$
\begin{aligned}
L_p(x)
&=\frac{h_{\rm spike}(p,m)-(1-x)(1-p)m}
        {x(1-p)m}\\
&=1+
\frac{h_{\rm spike}(p,m)-(1-p)m}
     {x(1-p)m},
\qquad 0<x<1.
\end{aligned}
$$

Therefore

$$
\boxed{
L_p(x)
=1+\frac{h_{\rm spike}(p,m)-(1-p)m}
          {x(1-p)m}.}
\tag{1.45}
$$

At $x=1$, the same value follows from the completion formula (1.16). The $1/x$ term records information released in a vanishing initial fraction of the run, divided by the $O(x|Q|)$ information subsequently received.

![Fixed-p thermodynamic spike leverage for m=1 and m=2.](figures/spike_flow_finite_p.png)

As $p\to1^-$,

$$
\frac{h_{\rm spike}(p,m)-(1-p)m}{(1-p)m}
\sim\frac{1-2^{-m}}m\log_2\frac1{1-p}.
\tag{1.46}
$$

Define

$$
L_{p,n}(x)
:=\left\langle L_{\lfloor x|Q|\rfloor}\right\rangle.
\tag{1.47}
$$

It is clearest to compare the two iterated limits using the same normalization:

$$
\lim_{n\to\infty}\lim_{p\to1^-}\frac{L_{p,n}(x)}{|Q|}
=1-2^{-m},
\tag{1.48}
$$

whereas

$$
\lim_{p\to1^-}\lim_{n\to\infty}\frac{L_{p,n}(x)}{|Q|}
=0.
\tag{1.49}
$$

Before division by $|Q|$, the first ordering is extensive and the second diverges only logarithmically in $1/(1-p)$. The two orders correspond to two extreme rates of approach. To interpolate between them, choose a sequence $p_n\to1$ and prescribe its exponential sharpness rate by

$$
\frac{\log_2(1/(1-p_n))}{|Q|}\longrightarrow\lambda\in(0,\infty).
\tag{1.50}
$$

The endpoint $\lambda=0$ recovers the scaling behind (1.49), where $n\to\infty$ before the prior becomes sharp; the formal endpoint $\lambda=\infty$ recovers (1.48), where the sharp-prior limit is taken first. A finite positive $\lambda$ selects a joint scaling strictly between them. Put $\varepsilon_n=1-p_n$. For $0<x<1$ and $k=\lfloor x|Q|\rfloor$, the leading entropies are

$$
G_1\sim(1-u)\varepsilon_n\lambda |Q|,
\qquad
G_k\sim\varepsilon_n|Q|(\lambda+mx),
\qquad
G_{k+1}-G_k\sim\varepsilon_nm.
\tag{1.51}
$$

Substituting these estimates into (1.15) gives the crossover

$$
\frac{L_{p_n,n}(x)}{|Q|}
\longrightarrow
 (1-u)\frac{\lambda}{\lambda+mx}.
\tag{1.52}
$$

At $x=1$, the increment $G_{k+1}-G_k$ is not defined. The same value in (1.52) follows separately from the completion formula (1.16).

The two familiar orders are therefore endpoints of a larger family of scalings.

## 2. Thermodynamic-limit calculus

The spike is solvable because all subsets of the same size have the same entropy. The general theory replaces that symmetry by an average over subsets.

### 2.1 The finite sequence from which the limit is taken

Use the fixed ordering $q_1,\ldots,q_{|Q|}$ and the index-set notation from the introduction. For each fixed $T\subseteq\{1,\ldots,|Q|\}$, the vector $\psi(T)\in A^{|T|}$ contains the answers at those indices. Its entropy is taken under the prior law of $\psi$; $T$ itself is held fixed. Define

$$
G_{n,k}
:=\binom{|Q|}{k}^{-1}
\sum_{\substack{T\subseteq\{1,\ldots,|Q|\}\\|T|=k}}H\bigl(\psi(T)\bigr),
\qquad 0\le k\le |Q|.
\tag{2.1}
$$

Then

$$
G_{n,0}=0,
\qquad
G_{n,|Q|}=H(\psi)=H\bigl(p^{(n)}\bigr),
\tag{2.2}
$$

where $p^{(n)}$ is the prior law of the random map $\psi$ at size $n$.

Define its increments

$$
\gamma_{n,k}:=G_{n,k+1}-G_{n,k}.
\tag{2.3}
$$

The **entropy chain rule** separates uncertainty already present in one variable from the uncertainty left in another after the first is known. For discrete random variables $X_1$ and $X_2$,

$$
H(X_1,X_2)=H(X_1)+H(X_2\mid X_1),
\qquad
H(X_2\mid X_1)=\sum_x\Pr(X_1=x)H(X_2\mid X_1=x).
$$

Thus joint entropy is “entropy learned from $X_1$” plus the average additional entropy learned from $X_2$ once $X_1$ has been revealed. If $T=\{t_1,\ldots,t_k\}$ with $t_1<\cdots<t_k$, repeating the identity gives

$$
H\bigl(\psi(T)\bigr)
=\sum_{i=1}^k H\bigl(\psi(q_{t_i})\mid
\psi(q_{t_1}),\ldots,\psi(q_{t_{i-1}})\bigr).
$$

Apply the two-variable form to $\psi(T)$ and one fresh answer $\psi(q_i)$, with $i\in\{1,\ldots,|Q|\}\setminus T$. There are two separate averages: $H(\psi(q_i)\mid\psi(T))$ averages over answer histories under the prior on the random map, while $\mathbb E_{T,i}$ below averages that entropy over the externally selected indices $(T,i)$. Thus

$$
\gamma_{n,k}
=\mathbb E_{T,i}\big[H(\psi(q_i)\mid\psi(T))\big].
\tag{2.4}
$$

Conditioning reduces entropy. Coupling a random $k$-set to a random $(k+1)$-set therefore yields

$$
m\ge\gamma_{n,0}\ge\gamma_{n,1}\ge\cdots\ge
\gamma_{n,|Q|-1}\ge0.
\tag{2.5}
$$

This monotonicity is deterministic: the truth and subset have already been averaged. It should not be called self-averaging; concentration of individual learning trajectories would be a separate result.

The exact finite-$n$ identities are

$$
\mathbb E[\text{received through }\ell]=G_{n,\ell},
\tag{2.6}
$$

$$
\mathbb E[\text{remaining table entropy}]
=(|Q|-\ell)\gamma_{n,\ell},
\tag{2.7}
$$

for $0\le\ell<|Q|$, with zero remaining entropy at $\ell=|Q|$, and

$$
\boxed{
\langle L_{n,\ell}\rangle
=\frac{|Q|G_{n,1}-(|Q|-\ell)\gamma_{n,\ell}}
       {G_{n,\ell}}.}
\tag{2.8}
$$

for $1\le\ell<|Q|$. At the completed run,

$$
\langle L_{n,|Q|}\rangle
=\frac{|Q|G_{n,1}}{G_{n,|Q|}}.
\tag{2.8a}
$$

### 2.2 A precise macroscopic-limit statement

It is useful to keep the initial marginal entropy separate from the right-hand macroscopic profile. This allows an $o(|Q|)$-question boundary layer at $x=0$.

Let

$$
h_0:=\lim_{n\to\infty}G_{n,1},
\tag{2.9}
$$

and suppose the staircases have an almost-everywhere limit $\gamma$ on $(0,1)$. At any particular $x\in(0,1)$ where we evaluate the remaining entropy, also suppose

$$
\gamma_{n,\lfloor x|Q|\rfloor}
\longrightarrow\gamma(x).
\tag{2.10}
$$

At such a point, $\gamma(x)$ is the expected entropy of one more answer after a random $x$-fraction has been observed. The following hypotheses are sufficient for the macroscopic limit.

1. **Fixed answer width.** The integer $m$ is fixed. This supplies the uniform bound $0\le\gamma_{n,k}\le m$.
2. **Initial-entropy convergence.** The limit (2.9) exists.
3. **Profile convergence.** The staircase $t\mapsto\gamma_{n,\lfloor t|Q|\rfloor}$ converges almost everywhere on $(0,1)$. At a particular $x$ where remaining entropy is evaluated, direct convergence as in (2.10) is also required. At continuity points of a selected monotone limit this causes no ambiguity.
4. **Positive entropy density.** The limiting area is positive:

   $$
   \int_0^1\gamma(t)\,dt>0.
   $$

The fourth condition is the nondegenerate form of extensivity. It implies

$$
\frac{G_{n,|Q|}}{|Q|}
\longrightarrow\int_0^1\gamma(t)\,dt>0.
\tag{2.11}
$$

Merely writing $G_{n,|Q|}=\Theta(|Q|)$ is not sufficient: the entropy density may oscillate with $n$, and even a convergent scalar entropy density does not determine the profile. Conversely, profile convergence plus positive area implies the required extensivity. No Gibbs representation and no full-support lower bound on individual map probabilities is necessary.

To derive the limit, extend the finite increments to a staircase on $[0,1]$ using the same symbol:

$$
\gamma_n(t):=\gamma_{n,\min(\lfloor t|Q|\rfloor,|Q|-1)}.
$$

The value assigned at the single endpoint $t=1$ is irrelevant to the integral.

Telescoping gives an exact Riemann-sum identity at grid points:

$$
\frac{G_{n,\ell}}{|Q|}
=\frac1{|Q|}\sum_{k=0}^{\ell-1}\gamma_{n,k}
=\int_0^{\ell/|Q|}\gamma_n(t)\,dt.
\tag{2.12}
$$

Because $0\le\gamma_n(t)\le m$, dominated convergence gives, for $\ell_n=\lfloor x|Q|\rfloor$,

$$
\frac{G_{n,\ell_n}}{|Q|}
\longrightarrow
g(x):=\int_0^x\gamma(t)\,dt.
\tag{2.13}
$$

At a point of profile convergence,

$$
\frac{(|Q|-\ell_n)\gamma_{n,\ell_n}}{|Q|}
\longrightarrow(1-x)\gamma(x).
\tag{2.14}
$$

Substitution into (2.8) proves

$$
\boxed{
L(x)
=\frac{h_0-(1-x)\gamma(x)}{g(x)},
\qquad
g(x)=\int_0^x\gamma(t)\,dt,
\qquad 0<x<1.}
\tag{2.15}
$$

Since $\gamma$ is nonnegative and nonincreasing, its positive integral implies $g(x)>0$ for every $x>0$.

Monotonicity and boundedness guarantee convergent subsequences by a Helly selection argument. They do not guarantee that the full sequence of priors selects a unique profile. For example, a family can alternate between two constructions on even and odd $n$. At the prior level, one therefore needs a coherent construction—such as a fixed exchangeable directing measure or fixed local block types with convergent frequencies—or must take profile convergence itself as the hypothesis.

Endpoint expansions, boundary-layer singularities, and discontinuous profiles are collected in Appendix H.

## 3. The coin-mixture prior

The simplest exchangeable scaling family is a mixture of independent-answer models. It gives a complete calculus exercise and a clean example of the boundary layer at $x=0$. Throughout this section, the answer width $m$ and the component family are fixed while $n\to\infty$.

### 3.1 Categorical formulation

Let $Z$ take values in a finite set of components indexed by $r$, with

$$
\Pr(Z=r)=w_r,
\qquad
w_r\ge0,
\qquad
\sum_r w_r=1.
$$

The mixture defines a joint distribution for the component $Z$ and random map $\psi$. The component is sampled once for the entire map, not once per question. Conditional on $Z=r$, the answers $\psi(q_i)$ at distinct question indices are independent and have categorical distribution $\nu_r$ on the $|A|=2^m$ possible answers. Thus, for every $i\in\{1,\ldots,|Q|\}$,

$$
\nu_r:A=\{0,1\}^m\longrightarrow[0,1],
\qquad
\sum_{a\in A}\nu_r(a)=1,
\qquad
\Pr(\psi(q_i)=a\mid Z=r)=\nu_r(a).
$$

Thus $\nu_r$ acts on **one possible answer** $a$, not on a whole map: it returns the probability that component $r$ assigns to that answer at any question. For $m=1$, a Bernoulli component with bias $\theta_r$ is

$$
\nu_r(0)=1-\theta_r,
\qquad
\nu_r(1)=\theta_r.
$$

For $m=2$, a component could instead be

| answer $a$ | $00$ | $01$ | $10$ | $11$ |
|---|---:|---:|---:|---:|
| $\nu_r(a)$ | $1/2$ | $1/4$ | $1/8$ | $1/8$ |

This is a distribution on the four two-bit answers; it need not factor into two independent bits. If a map answers $00$ at one question and $10$ at another, those two questions contribute the factor $\nu_r(00)\nu_r(10)=1/16$ to its likelihood under component $r$. More generally, the joint law is

$$
\Pr(Z=r,\psi=\phi_j)
=w_r\prod_{i=1}^{|Q|}\nu_r\bigl(\phi_j(q_i)\bigr),
\qquad
p_j:=\Pr(\psi=\phi_j)
=\sum_r w_r\prod_{i=1}^{|Q|}\nu_r\bigl(\phi_j(q_i)\bigr).
\tag{3.1}
$$

Here $\phi_j(q_i)\in A$ is the answer block produced by candidate map $\phi_j$ at question $q_i$. The product is its probability conditional on component $r$, and the outer sum averages those component likelihoods with weights $w_r$.

The one-question mixture distribution is

$$
\nu_{\rm mix}:=\sum_r w_r\nu_r,
$$

so

$$
h_0=H(\nu_{\rm mix}).
\tag{3.2}
$$

Conditional on knowing $Z$, the entropy per question is

$$
h_{\rm cond}:=\sum_r w_rH(\nu_r).
\tag{3.3}
$$

Because the construction treats all questions alike, $H(\psi(T))$ has the same value $G_k$ for every $T\subseteq\{1,\ldots,|Q|\}$ with $|T|=k$. Apply the chain rule introduced in Section 2 to the joint distribution of $Z$ and $\psi(T)$:

$$
H\bigl(Z,\psi(T)\bigr)
=H(Z)+H\bigl(\psi(T)\mid Z\bigr)
=H\bigl(\psi(T)\bigr)+H\bigl(Z\mid\psi(T)\bigr).
$$

Conditional independence gives

$$
H\bigl(\psi(T)\mid Z\bigr)
=\sum_{i\in T}H\bigl(\psi(q_i)\mid Z\bigr)
=k\sum_rw_rH(\nu_r)
=k h_{\rm cond}.
$$

Define

$$
I_k:=I\bigl(Z;\psi(T)\bigr)
=H(Z)-H\bigl(Z\mid\psi(T)\bigr),
\qquad |T|=k,
$$

as the mutual information between the selected component and the first $k$ answers. Rearranging the two chain-rule expansions gives the exact decomposition

$$
G_k=k h_{\rm cond}+I_k.
\tag{3.4}
$$

The first immediate consequence is

$$
\gamma_k=h_{\rm cond}+(I_{k+1}-I_k),
\tag{3.4a}
$$

The correction has the conditional-information form

$$
I_{k+1}-I_k
=I\bigl(Z;\psi(q_i)\mid\psi(T)\bigr)\ge0,
\qquad i\in\{1,\ldots,|Q|\}\setminus T,
$$

Here $I_{k+1}$ is evaluated on $T\cup\{i\}$. The difference is exactly the additional information that the next answer reveals about which component was selected.

The second consequence is the exact finite-$n$ leverage formula. For $1\le\ell<|Q|$,

$$
\langle L_{n,\ell}\rangle
=\frac{
|Q|H(\nu_{\rm mix})-(|Q|-\ell)\big[h_{\rm cond}+I_{\ell+1}-I_\ell\big]
}{
\ell h_{\rm cond}+I_\ell
}.
\tag{3.4b}
$$

Provided $h_{\rm cond}>0$, $I_\ell\ge0$ and $I_{\ell+1}-I_\ell\ge0$ show that the finite curve lies below its limiting hyperbola at the same $x=\ell/|Q|$.

For finitely many components,

$$
0\le I_k\le H(Z)<\infty.
\tag{3.5}
$$

Thus the information required to identify the component is subextensive.

Discard any zero-weight components. If the remaining distinct components are identifiable and mutually absolutely continuous, Bayesian posterior odds decay at a rate set by relative entropy. Choose nested index sets $T_k\subseteq\{1,\ldots,|Q|\}$ with $|T_k|=k$. For data generated from a positive-weight component $r$, and any positive-weight competitor $s$, almost surely under $\nu_r^{\otimes\infty}$,

$$
\frac1k\log_2
\frac{\Pr(Z=s\mid\psi(T_k))}
     {\Pr(Z=r\mid\psi(T_k))}
\overset{\rm a.s.}{\longrightarrow}
-D_{\rm KL}(\nu_r\Vert\nu_s)<0.
\tag{3.6}
$$

Here $D_{\rm KL}$ is measured with base-two logarithms, consistently with the entropy convention of this chapter.

With a support mismatch, the divergence may be $+\infty$, and one revealing symbol can instead set the wrong component's posterior probability to zero.

The posterior predictive distribution therefore converges to the true $\nu_r$. Bounded convergence gives

$$
\gamma(x)=h_{\rm cond}
\qquad\text{for every fixed }0<x<1.
\tag{3.7}
$$

Equations (2.13) and (2.15) now give

$$
g(x)=x h_{\rm cond},
$$

and, provided $h_{\rm cond}>0$,

$$
\boxed{
L(x)=1+\frac c x,
\qquad
c=\frac{H(\nu_{\rm mix})-h_{\rm cond}}{h_{\rm cond}}\ge0.}
\tag{3.8}
$$

The inequality is the concavity of entropy:

$$
H\!\left(\sum_rw_r\nu_r\right)
\ge\sum_rw_rH(\nu_r).
$$

If $h_{\rm cond}=0$—for example, a nontrivial mixture of deterministic coins—the map entropy is $O(1)$ and bounded by $H(Z)$ rather than extensive in $|Q|$. The normalization of Section 2 then degenerates, and (3.8) is not a finite thermodynamic curve.

Equality holds exactly when all positive-weight components have the same categorical distribution, after duplicate components are merged. That is the effective one-coin model, for which $L\equiv1$.

The similarity to the fixed-$p$ spike is now explicit. Provided $h_{\rm cond}>0$, in both cases a global latent variable is inferred within $o(|Q|)$ questions, leaving a constant positive entropy density afterward. The macroscopic profile sees only the jump from $h_0$ to $h_{\rm cond}$, and (H.3) turns that jump into $1/x$.

### 3.2 The binary coin and exact finite formulas

For $m=1$, write $\nu_r=\operatorname{Bernoulli}(\theta_r)$ and denote the truth-table Hamming weight by

$$
|\phi_j|:=\sum_{i=1}^{|Q|}\phi_j(q_i).
$$

Then

$$
p_j
=\sum_r w_r\theta_r^{|\phi_j|}(1-\theta_r)^{|Q|-|\phi_j|}.
\tag{3.9}
$$

After $\ell$ observations containing $k$ ones, let

$$
P_{\ell,k}
:=\sum_r w_r\theta_r^k(1-\theta_r)^{\ell-k}.
\tag{3.10}
$$

Every particular binary string of length $\ell$ and weight $k$ has probability $P_{\ell,k}$. Hence

$$
G_\ell
=-\sum_{k=0}^{\ell}\binom{\ell}{k}
P_{\ell,k}\log_2P_{\ell,k}.
\tag{3.11}
$$

The posterior component weights and next-answer probability are

$$
w_r(\ell,k)
=\frac{w_r\theta_r^k(1-\theta_r)^{\ell-k}}
       {P_{\ell,k}},
\tag{3.12}
$$

for count classes with $P_{\ell,k}>0$. Consequently

$$
\Pr\bigl(\psi(q_i)=1\mid\psi(T)=y\bigr)
=\sum_rw_r(\ell,k)\theta_r.
\tag{3.13}
$$

Here $T\subseteq\{1,\ldots,|Q|\}$ has $|T|=\ell$, the fresh index satisfies $i\in\{1,\ldots,|Q|\}\setminus T$, and $y\in\{0,1\}^{\ell}$ is the observed value of $\psi(T)$ with $k$ ones.

Averaging the binary entropy of this prediction over all count classes gives

$$
\gamma_\ell
=\sum_{\substack{0\le k\le\ell\\P_{\ell,k}>0}}
\binom{\ell}{k}P_{\ell,k}
h_2\!\left(\sum_rw_r(\ell,k)\theta_r\right)
=G_{\ell+1}-G_\ell.
\tag{3.14}
$$

In (3.11), zero-probability terms are interpreted using $0\log 0=0$; (3.14) simply omits those classes because their posterior weights are undefined and their contribution is zero. Equation (3.11) is valid through $\ell=|Q|$; the next-answer identity (3.14) is valid through $\ell=|Q|-1$. Apart from that truncation, the formulas do not depend on $n$.

### 3.3 Worked example: $(n,m)=(2,1)$

Take

$$
(\theta_1,\theta_2)=(0.1,0.9),
\qquad
(w_1,w_2)=\left(\frac12,\frac12\right).
\tag{3.15}
$$

Here $|Q|=4$. For a particular length-$\ell$ answer string with $k$ ones, (3.10) becomes

$$
P_{\ell,k}
=\frac12\left(\frac1{10}\right)^k
          \left(\frac9{10}\right)^{\ell-k}
+\frac12\left(\frac9{10}\right)^k
          \left(\frac1{10}\right)^{\ell-k}
=\frac{9^k+9^{\ell-k}}{2\,10^\ell}.
$$

The probability of an individual map therefore depends only on its Hamming weight, with clearly separated masses:

| Hamming weight $|\phi_j|$ | mass of each map | multiplicity |
|---:|---:|---:|
| 0 | $0.3281$ | 1 |
| 1 | $0.0369$ | 4 |
| 2 | $0.0081$ | 6 |
| 3 | $0.0369$ | 4 |
| 4 | $0.3281$ | 1 |

In particular, a weight-one map is more than four times as probable as a weight-two map. The normalization is visible directly:

$$
2(0.3281)+8(0.0369)+6(0.0081)=1.
$$

In the truth-table ordering used in the earlier examples:

| $j$ | $p_j$ | $j$ | $p_j$ | $j$ | $p_j$ | $j$ | $p_j$ |
|---|---:|---|---:|---|---:|---|---:|
| 0000 | 0.3281 | 0001 | 0.0369 | 0010 | 0.0369 | 0011 | 0.0081 |
| 0100 | 0.0369 | 0101 | 0.0081 | 0110 | 0.0081 | 0111 | 0.0369 |
| 1000 | 0.0369 | 1001 | 0.0081 | 1010 | 0.0081 | 1011 | 0.0369 |
| 1100 | 0.0081 | 1101 | 0.0369 | 1110 | 0.0369 | 1111 | 0.3281 |

The mixture mean is $\frac12(0.1)+\frac12(0.9)=1/2$, so the first-answer entropy is

$$
G_1=h_2(1/2)=1.
$$

If the first answer is $1$, Bayes' rule changes the component weights from $(1/2,1/2)$ to $(0.1,0.9)$. The next-one probability is therefore

$$
0.1(0.1)+0.9(0.9)=0.82.
$$

If the first answer is $0$, the posterior weights reverse to $(0.9,0.1)$ and the next-one probability is $0.18$. Both branches consequently have predictive entropy $h_2(0.18)=h_2(0.82)$, so

$$
G_2-G_1=h_2(0.18)=0.680077.
$$

The remaining block entropies follow from the same count-by-weight formula. For two answers, the strings $00,11$ each have probability $0.41$ and $01,10$ each have probability $0.09$, so

$$
G_2=-2(0.41\log_2 0.41)-2(0.09\log_2 0.09)
=1.680077.
$$

For three answers, the two constant strings each have probability $0.365$ and the other six strings each have probability $0.045$. At four answers, use the masses and multiplicities in the table. Thus

$$
G_3=-2(0.365\log_2 0.365)-6(0.045\log_2 0.045)
=2.269405,
$$

$$
G_4=-2(0.3281\log_2 0.3281)
    -8(0.0369\log_2 0.0369)
    -6(0.0081\log_2 0.0081)
=2.797921.
$$

The complete finite sequence is

$$
(G_0,G_1,G_2,G_3,G_4)
=(0,1,1.680077,2.269405,2.797921),
\tag{3.16}
$$

with increments

$$
(1,0.680077,0.589327,0.528517).
\tag{3.17}
$$

At $\ell=1,2,3,4$, the exact cumulative leverages are

$$
\langle L_\ell\rangle
=\frac{4-(4-\ell)(G_{\ell+1}-G_\ell)}{G_\ell}
\quad(\ell<4),
\qquad
\langle L_4\rangle=\frac4{G_4},
$$

namely

$$
1.95977,\quad1.67930,\quad1.52969,\quad1.42963.
$$

The conditional entropy density and hyperbola coefficient are

$$
h_{\rm cond}=\frac12h_2(0.1)+\frac12h_2(0.9)
=h_2(0.1)=0.468996,
$$

$$
c=\frac{1-h_2(0.1)}{h_2(0.1)}=1.132216,
$$

so the limiting curve is

$$
L(x)=1+\frac{1.132216}{x}.
\tag{3.18}
$$

The small-$n$ curve lies well below this limit because four observations are not enough to identify the latent coin sharply. For example, its endpoint is $1.42963$, whereas the limiting endpoint is $1+c=2.13222$.

![Two-component and ten-component binary coin mixtures with the same thermodynamic curve but different finite-size corrections.](figures/coin_mixture_leverage.png)

The panels begin at $x=0.15$ so the common $1/x$ singularity does not dominate the vertical scale. The left panel uses (3.15). The right panel uses ten equally weighted biases

$$
0.005,\ 0.04,\ 0.1,\ 0.2,\ 0.288171,
\quad\text{and their complements}.
$$

The fitted fifth bias is chosen by the condition

$$
h_2(0.005)+h_2(0.04)+h_2(0.1)+h_2(0.2)+h_2(0.288171)
\simeq5h_2(0.1).
$$

Because $h_2(1-\theta)=h_2(\theta)$, this identity covers all ten components, while complement symmetry makes their mean bias $1/2$. Consequently both mixtures have exactly the same two thermodynamic summaries,

$$
h_0=1,
\qquad
h_{\rm cond}=h_2(0.1),
$$

and hence the same $c$ and the same dashed limiting hyperbola. Their finite curves differ because the mutual information $I_k$ in (3.4) depends on the entire collection of component weights and separations, not only on $h_0$ and $h_{\rm cond}$. The number of components alone does not determine the convergence rate; this matched construction is chosen to display one concrete difference.

### 3.4 What changes when $m>1$

For $m>1$, the categorical construction (3.1) is the general model. The shared-scalar coin prior is a structured subfamily of it, not a separate alternative. For an answer $a=(a_1,\ldots,a_m)\in A$ and a map $\phi_j$, use the same Hamming-weight notation at both scales:

$$
|a|:=\sum_{b=1}^m a_b,
\qquad
|\phi_j|:=\sum_{i=1}^{|Q|}|\phi_j(q_i)|.
$$

Thus $|a|$ counts ones in one $m$-bit answer and $|\phi_j|$ counts ones in the complete table. The signed bias $B_j$ used in the source chapter is not another independent statistic: $B_j=2|\phi_j|-m|Q|$. Set

$$
\nu_r(a)
=\theta_r^{|a|}(1-\theta_r)^{m-|a|}.
$$

One component $Z$ is drawn once for the whole map. Conditional on $Z=r$, all $m|Q|$ table bits are independent Bernoulli variables with bias $\theta_r$. Substituting this $\nu_r$ into (3.1) gives

$$
p_j
=\sum_r w_r\theta_r^{|\phi_j|}(1-\theta_r)^{m|Q|-|\phi_j|}.
\tag{3.19}
$$

After the shared $Z$ is averaged out, distinct bits are generally correlated, both within one answer and across different questions. This is why the entropy of one $m$-bit answer is not generally $m h_2(\sum_r w_r\theta_r)$.

Let

$$
F_s
:=-\sum_{k=0}^s\binom{s}{k}P_{s,k}\log_2P_{s,k},
\tag{3.20}
$$

with $P_{s,k}$ as in (3.10). This is the entropy of the complete $s$-bit string, not merely the entropy of its Hamming count. Since $H(\nu_r)=m h_2(\theta_r)$, the categorical formulas specialize exactly to

$$
G_\ell=F_{m\ell},
\qquad
h_0=F_m,
\qquad
h_{\rm cond,m}=m\sum_rw_rh_2(\theta_r).
\tag{3.21}
$$

Therefore the exact finite leverage plotted below is, for $1\le\ell<|Q|$,

$$
\langle L_{n,\ell}\rangle
=\frac{|Q|F_m-(|Q|-\ell)\big(F_{m(\ell+1)}-F_{m\ell}\big)}
       {F_{m\ell}},
\tag{3.21a}
$$

with completion value

$$
\langle L_{n,|Q|}\rangle=\frac{|Q|F_m}{F_{m|Q|}}.
\tag{3.21b}
$$

The latent component contributes only subextensive information, so $F_s/s\to\sum_rw_rh_2(\theta_r)$. Substituting this limit into (3.21a) gives the thermodynamic curve.

Provided $h_{\rm cond,m}=m\sum_rw_rh_2(\theta_r)>0$, the correct thermodynamic curve is therefore

$$
\boxed{
L_m(x)=1+\frac{c_m}{x},
\qquad
c_m=\frac{F_m-h_{\rm cond,m}}{h_{\rm cond,m}}
=\frac{F_m-m\sum_rw_rh_2(\theta_r)}
          {m\sum_rw_rh_2(\theta_r)}.}
\tag{3.22}
$$

For the mixture (3.15),

$$
c_1=1.132216,
\qquad
c_2=0.791144,
$$

with

$$
F_1=1,
\qquad
F_2=1.680077,
\qquad
h_{\rm cond,1}=0.468996,
\qquad
h_{\rm cond,2}=0.937991.
$$

Thus the $m=2$ panel must use $F_2$ for the initial answer entropy, not $2h_2(1/2)=2$. The exact curves approach their respective dashed hyperbolas at every fixed $x>0$; no uniform convergence at the singular origin is claimed. As above, the displayed range starts at $x=0.15$.

![Exact convergence for one-bit and two-bit questions under the same shared scalar latent coin mixture.](figures/coin_mixture_m_comparison.png)

It is useful to distinguish three superficially similar models:

- **One global $Z$, shared by every bit:** the construction used here, with $G_\ell=F_{m\ell}$ and $h_0=F_m$.
- **One independent global component for each output coordinate:** $G_\ell=mF_\ell$, so both initial and residual entropies scale by $m$ and the coefficient remains $c_1$.
- **A fresh component redrawn at every question:** the answers are iid with distribution $\nu_{\rm mix}$, so $G_\ell=\ell H(\nu_{\rm mix})$ and $L\equiv1$; there is no persistent latent variable to learn.


### 3.5 What “two coins are enough” means

For binary answers, a projectively consistent infinitely exchangeable prior is a mixture of iid Bernoulli laws by de Finetti's theorem. Projective consistency matters: an arbitrary exchangeable prior at each finite $n$, or an arbitrary sequence of such priors, need not come from one directing measure and need not converge.

For a fixed directing measure, the macroscopic leverage curve retains only two entropy summaries,

$$
h_0=h_2(\mathbb E\Theta),
\qquad
h_{\rm cond}=\mathbb E[h_2(\Theta)],
$$

and after taking a dimensionless ratio it retains only

$$
c=\frac{h_0-h_{\rm cond}}{h_{\rm cond}}.
$$

For a possibly continuous directing measure, the empirical frequency identifies the directing parameter almost surely; equivalently, the posterior predictive converges to it. Consequently, for $T\subseteq\{1,\ldots,|Q|\}$ with $|T|=k$ and $i\in\{1,\ldots,|Q|\}\setminus T$, $H(\psi(q_i)\mid\psi(T))\downarrow\mathbb E[h_2(\Theta)]=h_{\rm cond}$ and $H(\psi(T))/k\to h_{\rm cond}$. The finite-component bound (3.5) is not needed for this extension.

Two Bernoulli components can realize every finite value of $c\ge0$. For example, at fixed mean $0<\mu<1$, take weights $(1-\mu,\mu)$ and biases

$$
\theta_1=(1-\xi)\mu,
\qquad
\theta_2=(1-\xi)\mu+\xi.
$$

Their mean remains $\mu$. As $\xi$ moves from $0$ toward $1$, their mean conditional entropy moves continuously from $h_2(\mu)$ toward $0$, so every positive intermediate value is attained. The endpoint $\xi=1$ has zero entropy density and is excluded from the finite thermodynamic leverage law.

This does **not** mean that two atoms reproduce an arbitrary exchangeable prior, its finite-$n$ corrections, or its posterior concentration rate. For $|A|>2$, de Finetti gives mixtures of iid categorical distributions, and two atoms do not in general reproduce every marginal-and-entropy pair. The precise conclusion is narrower:

> Every fixed, projectively consistent exchangeable binary prior with positive conditional entropy density $h_{\rm cond}=\mathbb E[h_2(\Theta)]>0$ has a hyperbolic macroscopic leverage curve. That curve contains only two entropy summaries, and a two-coin model can realize every finite value of their scale-free ratio. One effective coin is the trivial case $L\equiv1$.

## 4. Non-hyperbolic solvable priors

A stable, projectively consistent exchangeable prior cannot produce a smooth non-hyperbolic macroscopic curve: de Finetti reduces it to an iid mixture, and the preceding section applies. To obtain a profile that changes throughout $0<x<1$, the prior must retain structure tied to particular groups or addresses of questions. A fixed partition into local blocks is the simplest solvable construction.

This statement requires the consistency qualification. A separate finite exchangeable law can be chosen at every $n$ without forming one infinite exchangeable family; such a sequence need not be an iid mixture and need not possess any thermodynamic limit.

### 4.1 Independent blocks of questions

Partition $Q$ into independent blocks drawn from a fixed finite set of types, and denote the resulting block partition by $\mathcal P_n$. We index block types by $\kappa$. At finite $n$, let $K_{\kappa,n}$ be the number of type-$\kappa$ blocks and set

$$
w_{\kappa,n}:=\frac{r_\kappa K_{\kappa,n}}{|Q|},
$$

the fraction of all questions that belong to them. We assume $w_{\kappa,n}\to w_\kappa$ and $\sum_\kappa w_\kappa=1$. A block type $\kappa$ has:

- size $r_\kappa$;
- a fixed joint distribution on its $r_\kappa$ answers, shared by every block of that type up to relabelling;
- a mean subset-entropy profile $H_\kappa(0),\ldots,H_\kappa(r_\kappa)$, defined below.

Let $V\subseteq Q$ be one particular block of type $\kappa$, so $|V|=r_\kappa$. Fix an ordering

$$
V=(q_1,\ldots,q_{r_\kappa})
$$

once and for all, and give every subset $J\subseteq V$ the inherited order. Thus, if $J=\{q_{i_1},\ldots,q_{i_j}\}$ with $i_1<\cdots<i_j$, its random answer vector is

$$
\psi(J):=\bigl(\psi(q_{i_1}),\ldots,\psi(q_{i_j})\bigr)
\in A^j=A^{|J|}.
$$

Thus $J$ is the set of questions, whereas $\psi(J)$ is a random vector under the prior. A particular realization will be denoted by $y_J$. For an integer $j\in\{0,\ldots,r_\kappa\}$, define

$$
H_\kappa(j):=\binom{r_\kappa}{j}^{-1}
\sum_{\substack{J\subseteq V\\|J|=j}}H\bigl(\psi(J)\bigr).
$$

Here $j$ is the **number of questions selected from the block**, not a question label, and $V$ is the block containing those questions. The entropy $H(\psi(J))$ measures the learner's uncertainty in their **joint answers**. It does not measure uncertainty about which questions were selected: the question set $J$ is treated as known.

Explicitly, if $y_J=(y_{q_{i_1}},\ldots,y_{q_{i_j}})\in A^j=A^{|J|}$ is one possible realized answer tuple on those known questions, then

$$
H\bigl(\psi(J)\bigr)
=-\sum_{y_J\in A^{|J|}}
\Pr\bigl(\psi(J)=y_J\bigr)
\log_2\Pr\bigl(\psi(J)=y_J\bigr).
$$

The uncertainty is therefore over which answer tuple $y_J$ the learner's distribution assigns to those coordinates.

No symmetry is required for this averaged definition. All zoo members below have the stronger property of **entropy homogeneity**, meaning that $H(\psi(J))$ itself depends only on $|J|$. Full permutation invariance of the distribution is sufficient but not necessary. MDS code blocks, for example, are entropy homogeneous even when their coordinate distribution is not invariant under every permutation.

This profile records how much answer uncertainty remains visible after looking at $j$ locations inside a type-$\kappa$ block. Its purpose is that successive differences have an operational meaning. Let $J$ be a uniformly chosen $i$-subset of $V$, and then choose a fresh question $q$ uniformly from $V\setminus J$. Averaging the entropy chain rule over these pairs gives the within-block conditional-entropy increments

$$
\eta_\kappa(i):=H_\kappa(i+1)-H_\kappa(i),
\qquad 0\le i\le r_\kappa-1,
\tag{4.1}
$$

so, under the sampling rule just described,

$$
\eta_\kappa(i)
=\mathbb E_{J,q}\!\left[H\bigl(\psi(q)\mid\psi(J)\bigr)\right]
=\frac{1}{\binom{r_\kappa}{i}(r_\kappa-i)}
\sum_{\substack{J\subseteq V\\|J|=i}}
\sum_{q\in V\setminus J}H\bigl(\psi(q)\mid\psi(J)\bigr).
$$

It is the mean uncertainty in the answer to one fresh question after $i$ of its block partners have already been answered. This is why the block-entropy profile is useful: it converts a joint law on an entire block into exactly the conditional entropies needed by the learning curve.

The endpoints make the meaning concrete: $H_\kappa(0)=0$, while $H_\kappa(r_\kappa)=H(\psi(V))$ is the entropy of every answer in the full block taken jointly. For two independent fair binary answers, the profile is $(0,1,2)$ and the increments are $(1,1)$. If the same fair bit is copied into both questions, the profile is instead $(0,1,1)$ and the increments are $(1,0)$: after either answer is known, the other carries no remaining uncertainty.

For a fresh question in a block of size $r_\kappa$, the number of its $r_\kappa-1$ partners already present in a uniformly random $\ell$-set is hypergeometric. Thus, exactly at finite $|Q|$ for $0\le\ell<|Q|$,

$$
\gamma_{n,\ell}
=\sum_\kappa w_{\kappa,n}\sum_{i=0}^{r_\kappa-1}
\frac{\binom{r_\kappa-1}{i}\binom{|Q|-r_\kappa}{\ell-i}}
     {\binom{|Q|-1}{\ell}}\,
\eta_\kappa(i).
\tag{4.2}
$$

Hold every $r_\kappa$ fixed, take $\ell=\lfloor x|Q|\rfloor$, and assume $w_{\kappa,n}\to w_\kappa$. Sampling without replacement then approaches independent sampling:

$$
\frac{\binom{r-1}{i}\binom{|Q|-r}{\ell-i}}
     {\binom{|Q|-1}{\ell}}
\longrightarrow
\underbrace{\binom{r-1}{i}x^i(1-x)^{r-1-i}}
_{\displaystyle \beta_{r-1,i}(x)}.
\tag{4.3}
$$

The expression on the right simultaneously defines the Bernstein basis polynomial $\beta_{r-1,i}$. Probabilistically, it is the limiting probability that exactly $i$ of the fresh question's $r-1$ block partners have already been asked by macroscopic time $x$.

The limiting profile is

$$
\boxed{
\gamma(x)
=\sum_\kappa w_\kappa\sum_{i=0}^{r_\kappa-1}
\beta_{r_\kappa-1,i}(x)\eta_\kappa(i).}
\tag{4.4}
$$

At the start of the run, no partner of the fresh question has been asked, so only the $i=0$ basis term survives. Therefore

$$
h_0=\gamma(0)
=\sum_\kappa w_\kappa\eta_\kappa(0)
=\sum_\kappa w_\kappa H_\kappa(1).
$$

Thus the same block-entropy profile supplies both the initial marginal entropy and the entire macroscopic conditional-entropy curve.

To construct a sequence with prescribed fractions $w_\kappa$, choose integer block counts $K_{\kappa,n}$ satisfying

$$
\frac{r_\kappa K_{\kappa,n}}{|Q|}\longrightarrow w_\kappa,
\qquad
\sum_\kappa r_\kappa K_{\kappa,n}\le |Q|,
$$

and assign only $o(|Q|)$ leftover sites. Their treatment does not affect the limit. If block sizes grow with $n$, the proof changes; a useful sufficient condition for the hypergeometric-to-binomial approximation is

$$
\frac{\big(\max_{V\in\mathcal P_n}|V|\big)^2}{|Q|}\longrightarrow0.
$$

For countably many types, pointwise frequency convergence must be replaced by an $\ell^1$ or tightness condition.

At finite $n$, any leftovers can be recorded as an additional block type, so (4.2) still includes every question and remains exact. Their weight vanishes in the limit.

### 4.2 Bernstein polynomials as a profile language

For a single block type, suppress the type label. For $r=3$, (4.4) reads

$$
\gamma(x)
=\eta(0)(1-x)^2
+2\eta(1)x(1-x)
+\eta(2)x^2.
\tag{4.5}
$$

For $r=4$,

$$
\gamma(x)
=\eta(0)(1-x)^3
+3\eta(1)x(1-x)^2
+3\eta(2)x^2(1-x)
+\eta(3)x^3.
\tag{4.6}
$$

Some useful coefficient vectors are listed below. Here $h_{\rm fresh}$ is the entropy of one independent fresh answer, and $h_{\rm clique}$ is the entropy of the value copied through a clique.

| block | $(\eta(0),\ldots,\eta(r-1))$ | profile |
|---|---|---|
| independent fresh answers | $(h_{\rm fresh},h_{\rm fresh},\ldots,h_{\rm fresh})$ | $h_{\rm fresh}$ |
| clique | $(h_{\rm clique},0,\ldots,0)$ | $h_{\rm clique}(1-x)^{r-1}$ |
| parity | $(m,\ldots,m,0)$ | $m(1-x^{r-1})$ |
| $(r,k)$ MDS | $k$ entries $m$, then zeros | $m\,\Pr[\operatorname{Bin}(r-1,x)\le k-1]$ |

Entropy submodularity gives

$$
m\ge\eta_\kappa(0)\ge\eta_\kappa(1)\ge\cdots
\ge\eta_\kappa(r_\kappa-1)\ge0.
\tag{4.7}
$$

The derivative identity for Bernstein polynomials makes the macroscopic monotonicity explicit:

$$
\gamma'(x)
=\sum_\kappa w_\kappa(r_\kappa-1)
\sum_{i=0}^{r_\kappa-2}
\big(\eta_\kappa(i+1)-\eta_\kappa(i)\big)
\beta_{r_\kappa-2,i}(x)
\le0.
\tag{4.8}
$$

Thus $\gamma$ is nonincreasing, as required by the general entropy calculus. Leverage itself need not be monotone.

Equation (4.4) describes every prior in the stated class of independent blocks drawn from a fixed finite list of bounded types, and mixing types mixes their profiles linearly. There is also a useful approximation result when the answer alphabet is allowed to grow: mixtures of common-length MDS blocks can realize any nonincreasing Bernstein coefficient sequence, and Bernstein polynomials approximate every continuous nonincreasing profile. A Reed–Solomon block of length $r$ needs $|A|=2^m\ge r$, however. Sending $r\to\infty$ in this approximation therefore also requires $m\to\infty$, or a different family of realizable entropy vectors; this is not a completeness theorem at fixed answer width.

### 4.3 Why leverage can have an interior maximum

The destroyed-entropy density is $h_0-(1-x)\gamma(x)$. After subtracting the received density, the cumulative deduced density is

$$
h_0-(1-x)\gamma(x)-g(x).
$$

If $\gamma$ is differentiable, its derivative is

$$
\frac{d}{dx}\left[h_0-(1-x)\gamma(x)-g(x)\right]
=-(1-x)\gamma'(x)\ge0.
\tag{4.9}
$$

Hence $-\gamma'(x)$, weighted by the unasked fraction, specifies when stored information is released. Since

$$
L(x)=1+\frac{h_0-(1-x)\gamma(x)-g(x)}{g(x)},
\tag{4.10}
$$

we have

$$
L'(x)
=\frac{
-(1-x)\gamma'(x)g(x)
-\left[h_0-(1-x)\gamma(x)-g(x)\right]\gamma(x)
}{g(x)^2}.
\tag{4.11}
$$

Leverage rises when the current deduction-to-reception rate exceeds its accumulated average, and falls when it drops below that average.

For an explicit example, mix a fraction $w$ of $(r,k)=(4,2)$ MDS questions with a fraction $1-w$ of fresh questions. Then

$$
\frac{\gamma(x)}m
=1-3wx^2+2wx^3,
\tag{4.12}
$$

$$
\frac{g(x)}m
=x-wx^3+\frac w2x^4.
\tag{4.13}
$$

The cumulative deduced-information density in this example is

$$
\frac{h_0-(1-x)\gamma(x)-g(x)}m
=3wx^2-4wx^3+\frac{3w}{2}x^4.
\tag{4.14}
$$

Consequently,

$$
L(x)=1+3wx+O(x^2)
\qquad(x\downarrow0),
\tag{4.15}
$$

whereas direct differentiation at the other endpoint gives

$$
L'(1)=-\frac{w}{2}\,
\frac{1-w}{(1-w/2)^2}<0
\qquad(0<w<1).
\tag{4.16}
$$

Thus $L'(0^+)=3w>0$ but $L'(1)<0$, so continuity forces an interior maximum. The peak is produced by a threshold release followed by continued ordinary reception.

## 5. An exotic zoo of solvable thermodynamic priors

The following priors are designed to release stored information at different stages of the run. Their profiles are explicit, so their leverage follows from (2.15). Full derivations appear in the appendix.

### 5.1 Cliques: constant leverage

For a clique block $V\in\mathcal P_n$ of size $r$, draw one shared value $Z_V\sim\nu_V$ and copy it to all questions in $V$. If $h_V:=H(\nu_V)$, then

$$
H_V(0)=0,
\qquad
H_V(j)=h_V\quad(j\ge1),
$$

so

$$
\bigl(\eta_V(0),\ldots,\eta_V(r-1)\bigr)
=(h_V,0,\ldots,0).
$$

For a common block size, let $h_{\rm clique}$ denote the question-weighted mean of the $h_V$. Then

$$
\gamma(x)=h_{\rm clique}(1-x)^{r-1},
\qquad
g(x)=\frac{h_{\rm clique}}{r}\big[1-(1-x)^r\big].
$$

Therefore

$$
\boxed{L(x)\equiv r.}
\tag{5.1}
$$

The result is exact at finite $n$ for an exact clique partition, or with only deterministic zero-entropy leftovers. Mixing clique sizes at a common entropy scale does not interpolate between constants: the initial leverage is the arithmetic mean of the sizes, while the final leverage is their harmonic mean.

### 5.2 Parity: deductions near the end

For a binary block $V$ of size $r$, fix a coset label $\sigma_V\in\{0,1\}$ and choose its answer tuple uniformly from that parity coset,

$$
\Pr\bigl(\psi(V)=y_V\bigr)
=2^{-(r-1)}
\mathbf 1\!\left[\bigoplus_{q\in V}y_q=\sigma_V\right].
\tag{5.2}
$$

Every proper subset is uniform. Thus

$$
\eta(i)=
\begin{cases}
1,&0\le i\le r-2,\\
0,&i=r-1,
\end{cases}
$$

and

$$
\gamma(x)=1-x^{r-1},
\qquad
g(x)=x-\frac{x^r}{r}.
$$

It follows that

$$
\boxed{
L(x)
=\frac{x+(1-x)x^{r-1}}{x-x^r/r}.}
\tag{5.3}
$$

For $r\ge3$, this curve rises from $1$ to $r/(r-1)$. The case $r=2$ is exceptional: it is an equality or inequality clique and has $L\equiv2$. The derivative of the cumulative deduced-information density is

$$
-(1-x)\gamma'(x)=(r-1)x^{r-2}(1-x),
\tag{5.4}
$$

which concentrates late in the run.

### 5.3 Tilted parity

Uniform parity is not the only solvable law on a parity coset. For $0<\theta<1$, draw iid $\operatorname{Bernoulli}(\theta)$ bits and condition their parity to be $\sigma$. With

$$
\varepsilon:=1-2\theta,
\qquad
\mathcal Z_{r,\sigma}
=\frac{1+(-1)^\sigma\varepsilon^r}{2},
\tag{5.5}
$$

the block law is

$$
\Pr_\sigma\bigl(\psi(V)=y_V\bigr)
=\frac{\theta^{|y_V|}(1-\theta)^{r-|y_V|}
\mathbf 1[\oplus_{q\in V}y_q=\sigma]}
{\mathcal Z_{r,\sigma}}.
\tag{5.6}
$$

If $v$ bits remain unseen and their required parity is $\sigma_{\rm rem}$, the next bit is $1$ with probability

$$
P_{\rm next}(v,\sigma_{\rm rem})
=\frac{\theta\big[1-(-1)^{\sigma_{\rm rem}}\varepsilon^{v-1}\big]}
       {1+(-1)^{\sigma_{\rm rem}}\varepsilon^v}.
\tag{5.7}
$$

A finite average of $h_2(P_{\rm next}(v,\sigma_{\rm rem}))$ gives each $\eta(i)$; the exact weights are derived in Appendix C. The final member remains determined, so $\eta(r-1)=0$, while earlier coefficients now decrease gradually rather than staying exactly flat. At $r=4,\theta=0.3,\sigma=1$,

$$
\eta=(0.912441,\ 0.898876,\ 0.811317,\ 0).
\tag{5.8}
$$

This gives $L(0^+)\simeq1.0446$ and $L(1)\simeq1.3915$. The parameter $\varepsilon$ controls how far the profile bends away from uniform parity.

### 5.4 MDS blocks: a movable threshold

Let answers be symbols of $\mathbb F_{2^m}$, and let a block $V$ of size $r$ be uniform on an $[r,k]_{2^m}$ MDS code. A Reed–Solomon realization uses $r$ distinct field points and therefore requires $2^m\ge r$. Any $k$ answers determine the whole block, while any $j\le k$ answers are uniform. Hence

$$
H_{\rm MDS}(j)=m\min(j,k),
$$

and

$$
\eta(i)=
\begin{cases}
m,&i<k,\\
0,&i\ge k.
\end{cases}
$$

Writing

$$
\tau_{r,k}(x)
:=\Pr[\operatorname{Bin}(r-1,x)\le k-1],
\tag{5.9}
$$

we obtain

$$
\gamma(x)=m\tau_{r,k}(x).
\tag{5.10}
$$

The beta integral gives

$$
\rho:=\frac{k}{r},
\qquad
\int_0^1\tau_{r,k}(x)\,dx=\rho.
\tag{5.11}
$$

Thus $L(1)=r/k=1/\rho$. If $n\to\infty$ is taken first and then $r\to\infty$ with $k/r\to\rho$, the profile approaches a step at $x=\rho$. The answer width must grow sufficiently for such Reed–Solomon blocks.

To produce a peak, mix a code fraction $w$ with a fresh fraction $1-w$:

$$
\frac{\gamma(x)}m=w\tau_{r,k}(x)+(1-w),
\tag{5.12}
$$

$$
\boxed{
L(x)
=\frac{
w[1-(1-x)\tau_{r,k}(x)]+(1-w)x
}{
w\int_0^x\tau_{r,k}(t)\,dt+(1-w)x
}.}
\tag{5.13}
$$

The code releases deductions around its threshold; fresh questions continue to add received information afterward, making the cumulative curve decline after its maximum.

### 5.5 Local noise and full support

The hard clique, parity, and code priors assign probability zero to many maps. For $0\le\delta\le1$, a general local lapse replaces the hard law $P_V$ on a block $V\in\mathcal P_n$ by

$$
P_{V,\delta}(y_V)
=(1-\delta)P_V(y_V)+\delta |A|^{-|V|}.
\tag{5.14}
$$

Term by term, the map prior becomes

$$
\boxed{
p_{\delta,j}:=\Pr_\delta(\psi=\phi_j)
=\prod_{V\in\mathcal P_n}\left[
(1-\delta_{V,n})P_V\bigl(\phi_j(V)\bigr)
+\delta_{V,n}|A|^{-|V|}
\right].}
\tag{5.15}
$$

Here every block rate satisfies $0\le\delta_{V,n}\le1$. The map prior has full support whenever every $\delta_{V,n}>0$. The lapse is local: blocks remain independent, so only their entropy coefficients change.

For fixed block sizes, entropy is continuous on the finite probability simplex. Therefore

$$
\eta_{V,\delta}(i)\longrightarrow\eta_{V,0}(i)
\qquad(\delta\to0).
\tag{5.16}
$$

If $\max_{V\in\mathcal P_n}\delta_{V,n}\to0$, the hard and noisy priors have the same thermodynamic profile. More generally, it is enough that the question-weighted coefficient perturbation tends to zero:

$$
\sum_{V\in\mathcal P_n}\frac{|V|}{|Q|}
\max_i|\eta_{V,\delta_{V,n}}(i)-\eta_{V,0}(i)|
\longrightarrow0.
\tag{5.17}
$$

At fixed $\delta>0$, the noise does **not** vanish merely because $n\to\infty$; it produces a solvable but displaced profile. If block sizes also grow, $\delta_n\to0$ alone may be insufficient. For noisy parity blocks of size $r(n)$, for example, one needs roughly $r(n)\delta_n\to0$.

Two explicit coefficient changes are especially simple:

- Uniform parity followed by independent bit flips at rate $\delta$ has

  $$
  \eta(i)=1\quad(i\le r-2),
  \qquad
  \eta(r-1)
  =h_2\!\left(\frac{1-(1-2\delta)^r}{2}\right).
  \tag{5.18}
  $$

- An MDS block lapsed to the uniform block distribution has an exact entropy $H_\delta(j)$ given in Appendix D. Below threshold, $H_\delta(j)=jm$ remains unchanged; above threshold, the zero coefficients become a positive tail.

Merely giving the shared clique value full alphabet support is not enough: the block would still contain only constant strings. Independent output noise or a lapse such as (5.14) is required for full map support.

### 5.6 Arbitrary mixtures and the cocktail

Suppose component type $\kappa$ occupies a fraction $w_\kappa$ of questions and has initial marginal entropy $h_{0,\kappa}$, profile $\gamma_\kappa$, integral $g_\kappa$, and leverage $L_\kappa$. Independence across disjoint question sets gives

$$
h_0=\sum_\kappa w_\kappa h_{0,\kappa},
\qquad
\gamma(x)=\sum_\kappa w_\kappa\gamma_\kappa(x),
\qquad
g(x)=\sum_\kappa w_\kappa g_\kappa(x).
\tag{5.19}
$$

Therefore

$$
\boxed{
L(x)
=\frac{\sum_\kappa w_\kappa g_\kappa(x)L_\kappa(x)}
       {\sum_\kappa w_\kappa g_\kappa(x)}.}
\tag{5.20}
$$

Leverage is a received-entropy-weighted average of component leverages, not a simple question-fraction average. Equation (5.20) is a symbolic solution for any mixture whose component profiles are known.

As a worked cocktail, take $m=4$:

- one quarter of the questions follow the two-coin mixture $\theta\in\{0.2,0.7\}$, with weights $(1/4,3/4)$, and one scalar latent bias shared by all coin-component bits;
- one half belong to lapsed $(r,k)=(16,4)$ Reed–Solomon blocks with lapse $\delta=0.1$;
- one quarter are fresh, with independent bit bias $0.3$.

Normalize entropy per output bit and define

$$
h_{\rm cocktail}:=\frac{h_0}{m},
\qquad
\gamma_{\rm cocktail}(x):=\frac{\gamma(x)}m,
\qquad
g_{\rm cocktail}(x):=\frac{g(x)}m
=\int_0^x\gamma_{\rm cocktail}(t)\,dt.
$$

For the coin component, let $h_{\rm cond,coin}$ denote its constant interior conditional entropy per output bit. Then

$$
\frac{F_4}{4}=0.94828746,
\qquad
h_{\rm cond,coin}=0.84145020.
$$

If $\gamma_{\rm lapse}$ is the lapsed-code profile per bit, then

$$
h_{\rm cocktail}
=\frac14(0.94828746)+\frac12+\frac14h_2(0.3)
=0.95739459,
\tag{5.21}
$$

while, for $x>0$,

$$
\gamma_{\rm cocktail}(x)
=\frac14h_{\rm cond,coin}
+\frac12\gamma_{\rm lapse}(x)
+\frac14h_2(0.3).
\tag{5.22}
$$

Also,

$$
\int_0^1\gamma_{\rm lapse}(x)\,dx=0.33232806.
$$

Substitution into (2.15) yields

$$
L(x)=\frac{h_{\rm cocktail}-(1-x)\gamma_{\rm cocktail}(x)}
           {g_{\rm cocktail}(x)}.
\tag{5.23}
$$

Numerically, its initial divergence has coefficient

$$
\lim_{x\downarrow0}xL(x)
=\frac{h_{\rm cocktail}-\gamma_{\rm cocktail}(0^+)}
       {\gamma_{\rm cocktail}(0^+)}
\simeq0.02870,
$$

a dip of approximately $1.499$ at $x\simeq0.0836$, a peak of approximately $2.114$ at $x\simeq0.350$, and

$$
L(1)\simeq1.60408.
\tag{5.24}
$$

The initial hyperbolic decline comes from the global coin latent; the middle peak comes from the code threshold; the final decline comes from fresh and lapsed entropy received after the code deductions have mostly been released.

![Finite learning curves approaching the thermodynamic profiles for four solvable priors.](figures/leverage_shapes.png)

*Finite learning curves and their dashed thermodynamic limits. In the cocktail panel, the dashed curve uses the shared-latent initial entropy $F_4/4$ and equations (5.21)--(5.23).*

### 5.7 The zoo side by side

| member | microscopic rule | limiting profile | leverage shape |
|---|---|---|---|
| coin mixture | one global latent selects an iid component | constant for $x>0$, with a jump at $0$ | $1+c/x$ |
| clique | one value copied through each block | $h_{\rm clique}(1-x)^{r-1}$ | exactly $r$ |
| parity | one final parity constraint | $m(1-x^{r-1})$ | rising for $r\ge3$ |
| tilted parity | biased product law conditioned on parity | (4.4), with coefficients from (C.17) | tilt-dependent; retains the final parity constraint |
| MDS | any $k$ answers determine an $r$-block | $m\tau_{r,k}(x)$ | threshold rise; pure code tends to a plateau |
| MDS + fresh | threshold block plus ordinary sites | $m[w\tau_{r,k}+(1-w)]$ | interior peak |
| cocktail | coin + lapsed code + fresh | weighted sum | decline, dip, peak, decline |

The shape of a well-measured learning curve can therefore provide evidence about the **timing** of usable correlations: global and front-loaded, first-touch, last-touch, or threshold-like. It does not identify a unique microscopic prior. Different priors can share the same entropy profile, and an actual trained model may be misspecified, non-Bayesian, or imperfectly calibrated.

The central scale distinction is now explicit:

> The $n\to\infty$ curve sees only extensive-time learning. Any structure learned in $O(1)$ or $o(|Q|)$ questions collapses into a singularity or jump at $x=0$.

For Bayesian priors, this means that a finite-dimensional global latent may profoundly affect early learning yet disappear from the interior profile. To obtain nontrivial behavior throughout $0<x<1$, the prior must distribute uncertainty and correlations across $\Theta(|Q|)$ local degrees of freedom. Macroscopic leverage classifies when stored information becomes available, not every detail of how the prior represents it.

## 6. Beyond blocks: shape-based priors (proposal)

The zoo breaks question-permutation symmetry by decree, with a partition into blocks. This section records a construction that breaks it through the geometry of $\mathcal Q$ itself, so that $p_j$ is determined by the *shape* of the map $\phi_j$, and that is still solvable in the limit.

Two constraints drive the design. First, a projectively consistent exchangeable prior has a flat interior profile by the de Finetti argument, hence a hyperbola: a non-hyperbolic prior **must** be non-exchangeable. Second, refusing blocks leaves only one natural structure on $\mathcal Q=\{0,1\}^n$ to break the symmetry with: the hypercube.

### 6.1 The low-degree (Reed--Muller) prior: definition and rate

Every map $\phi:\{0,1\}^n\to\{0,1\}$ is uniquely an $\mathbb F_2$ polynomial in the input bits, its *algebraic normal form*

$$
\phi(x)=\bigoplus_{S\subseteq[n]}a_S\prod_{i\in S}x_i,
\qquad
a_S=\bigoplus_{y\,:\,\mathrm{supp}(y)\subseteq S}\phi(y),
$$

the coefficients recovered by M\"obius inversion: $a_S$ is the XOR of the truth-table entries on the subcube below $S$ (uniqueness because the $2^n$ monomials are triangular in the subset order, hence independent). The full coefficient vector costs $n2^{n-1}$ in-place XORs by the standard butterfly, whose per-bit kernel $Z=\binom{1\ 0}{1\ 1}$ is the triangular half of the Walsh--Hadamard kernel of the correlator ledger:

$$
W=\begin{pmatrix}1&1\\1&-1\end{pmatrix}
=Z\begin{pmatrix}1&0\\0&-2\end{pmatrix}Z^{\mathsf T},
\qquad
W^{\otimes n}=Z^{\otimes n}\,\mathrm{diag}_S\big((-2)^{|S|}\big)\,(Z^{\mathsf T})^{\otimes n}.
$$

The Walsh--Hadamard transform is two M\"obius passes with a $(-2)^{|S|}$ rescaling between them, the matrix form of the spin expansion $\prod_q(1-2y_q)=\sum_T(-2)^{|T|}\prod_{q\in T}y_q$; reducing mod $2$ kills the middle diagonal, makes $Z$ self-inverse, and leaves exactly the XOR butterfly. Same tensor machine, two arithmetics ($W^2=2I$ over $\mathbb R$, $Z^2=I$ over $\mathbb F_2$) and two cubes: the ledger transform acts on the map-space cube $\{0,1\}^Q$, the normal form on the question cube $\{0,1\}^n$. Write $\deg\phi$ for the largest $|S|$ with $a_S=1$. Take

$$
p_j\;\propto\;\mathbf 1[\deg\phi_j\le d(n)],
\tag{6.1}
$$

so the support is the Reed--Muller code $\mathrm{RM}(d,n)$, a linear code of length $Q=2^n$ and dimension

$$
K=\sum_{i=0}^{d}\binom ni
=Q\cdot\Pr\Big[\operatorname{Bin}\big(n,\tfrac12\big)\le d\Big].
\tag{6.2}
$$

The dimension counts monomials, and the binomial form dictates where the interesting degrees live. For fixed $d$ (or $d=\alpha n$ with $\alpha<\tfrac12$) the rate $R_n=K/Q$ tends to $0$; for $\alpha>\tfrac12$ it tends to $1$. Extensive-rate priors sit at the binomial median: by de Moivre--Laplace,

$$
d(n)=\frac n2+\frac c2\sqrt n
\qquad\Longrightarrow\qquad
R_n\to R=\Phi(c)\in(0,1).
\tag{6.3}
$$

The family interpolates between the zoo's two extreme members stretched over the whole table: $\mathrm{RM}(0,n)$ is the repetition code, i.e.\ a single global clique (two constant maps), and $\mathrm{RM}(n-1,n)$ is the even-weight code, i.e.\ a single global parity block. Both are degenerate in the limit ($R\to0$ and $R\to1$); the median-degree regime between them is where a genuine macroscopic profile lives.

**Why a hard cutoff rather than a Gibbs weight on degree.** One can write $p_j\propto e^{-\beta\deg\phi_j}$, but the density of states collapses it back onto (6.1). There are $2^{K_k}$ maps of degree $\le k$, so allowing one more degree adds $\binom n{k+1}\ln2$ nats of entropy against a linear energy cost $\beta$: the balance point pins an effective degree $d^*$ with $\binom n{d^*}\approx\beta/\ln2$, the degree concentrates there, and lower degrees occupy a $2^{-\binom n{d^*}}$ fraction of the support. Canonical equals microcanonical in extreme form: $e^{-\beta\deg}$ is asymptotically uniform on $\mathrm{RM}(d^*,n)$, with $\beta$ forced to scale like $\ln2\binom n{d^*}$ and an equilibrium that exists only on the decreasing flank $d^*\ge n/2$, i.e.\ $R\ge\tfrac12$. The soft variant that stays solvable is the mixture over cutoffs of 6.6; the per-map alternative $p_j\propto e^{-\lambda\,\#\{\text{monomials}\}}$ (iid Bernoulli coefficients in the polynomial basis) is natural but breaks the rank identity (6.4), which requires uniform coefficients.

### 6.2 Exact finite-$n$ structure: entropies are ranks

Write $\psi=u^{\mathsf T}G$ with $G$ a $K\times Q$ generator matrix and $u$ uniform on $\mathbb F_2^K$. For any question set $\mathcal S$, the answer vector $\psi(\mathcal S)$ is uniform on the image of the column submatrix, so

$$
H\big(\psi(\mathcal S)\big)=\operatorname{rank}\big(G_{\cdot,\mathcal S}\big)
\quad\text{bits, exactly at every finite }n.
\tag{6.4}
$$

Three consequences, all before any limit is taken:

- $h_0=G_{n,1}=1$: the all-ones word (degree $0$) is in the code, so every column of $G$ is nonzero and a single answer is a fair bit.
- $\gamma_{n,\ell}=G_{n,\ell+1}-G_{n,\ell}=\Pr[\text{a fresh random column increases the rank}]$, i.e.\ the probability that question $\ell+1$ is *not* determined by $\ell$ random answers. Monotonicity of $\gamma_{n,\ell}$ in $\ell$ is submodularity of matroid rank; the boundedness and monotonicity hypotheses of the macroscopic-limit statement hold with no entropy argument.
- The completion leverage is exact: $G_{n,Q}=H(\psi)=K$, so
$$
\langle L_{Q}\rangle=\frac{Q\,G_{n,1}}{G_{n,Q}}=\frac QK=\frac1{R_n}
\qquad\text{at every finite }n.
\tag{6.5}
$$

For small $n$ every $G_{n,\ell}$ is computable by enumerating subsets and taking ranks, giving exact finite-size curves converging to the limit below.

### 6.3 The EXIT identity and the area theorem

The **average EXIT function** of a code at erasure rate $\varepsilon$ is the mean entropy of one coordinate given all others observed through a $\mathrm{BEC}(\varepsilon)$:

$$
h_n(\varepsilon)=\frac1Q\sum_{q}H\big(\psi(q)\mid \psi(\mathcal S_\varepsilon\setminus q)\big),
\qquad \mathcal S_\varepsilon=\{\text{unerased coordinates}\}.
\tag{6.6}
$$

On the erasure channel there is no soft information: a coordinate is either determined by the revealed set (conditional entropy $0$) or uniform given it (entropy $1$), so $h_n(\varepsilon)$ is again a rank statement, the probability of *not* being determined. The only difference from $\gamma_{n,\ell}$ is Bernoulli sampling of the revealed set instead of a fixed size $\ell$. Since rank changes by at most $1$ per coordinate and the Bernoulli count concentrates, the two agree in the limit:

$$
\gamma_{n,\lfloor tQ\rfloor}=h_n(1-t)+o(1).
\tag{6.7}
$$

The area theorem is the telescoping identity in this notation:

$$
\int_0^1\gamma_n\big(t\big)\,dt=\frac{G_{n,Q}}Q=R_n\;\to\;R.
\tag{6.8}
$$

Received information per question over the whole run is the rate, exactly.

### 6.4 Capacity forces the step

The imported theorem (Kudekar--Kumar--Mondelli--Pfister--\c Sa\c so\u glu--Urbanke, IEEE-IT 2017): Reed--Muller codes of rate $R_n\to R$ achieve capacity on the BEC under bit-MAP decoding. On the erasure channel "bit-MAP error $\to0$" *is* the entropy statement: for every $\varepsilon<1-R$ the probability that a coordinate is undetermined tends to $0$. In profile language, revealing a fraction $t>R$ determines a fresh answer with probability $\to1$:

$$
\gamma(t)=0\qquad\text{for } t>R.
\tag{6.9}
$$

The other half needs no second theorem, only the area. $\gamma$ is nonincreasing and bounded by $h_0=1$, it vanishes on $(R,1)$, and its integral must be $R$ by (6.8); the mass has nowhere to sit except at height $1$ on $(0,R)$:

$$
\int_0^{R}\gamma= R,\quad \gamma\le1
\qquad\Longrightarrow\qquad
\gamma(t)=1\ \text{ for } 0<t<R.
\tag{6.10}
$$

So the profile is the exact step $\gamma(t)=\mathbf 1[t<R]$, obtained from a one-sided decoding theorem plus bookkeeping.

### 6.5 The curve

Integrate and substitute into the master law with $h_0=1$:

$$
g(t)=\int_0^t\gamma=\min(t,R),
\qquad
L(t)=\frac{1-(1-t)\gamma(t)}{g(t)}
=\begin{cases}
\dfrac{1-(1-t)}{t}=1, & t<R,\\[8pt]
\dfrac{1-0}{R}=\dfrac1R, & t>R.
\end{cases}
\tag{6.11}
$$

Before $t=R$ every answer is a fresh fair bit and nothing can be deduced: leverage sits exactly at the uniform baseline. At $t=R$ the $K$ received bits pin the codeword, and every one of the remaining $(1-R)Q$ answers is deduced for free; thereafter the cumulative ratio is locked at $1/R$, in agreement with the exact finite-$n$ completion value (6.5). For the median degree $c=0$: $R=\tfrac12$, and $L$ jumps from $1$ to $2$ at half time. At finite $n$ the jump is an analytic S-curve through $t=R_n$ (computable exactly by (6.4) for small $n$), sharpening as $n$ grows.

The information is held until a tunable macroscopic time $t=R=\Phi(c)$, the curve is a step rather than a hyperbola, no blocks appear anywhere, and $p_j$ reads only the algebraic shape of the map. This also upgrades 5.4: the MDS step needed blocks plus the secondary limit $r\to\infty$; here the same profile is a single thermodynamic limit, the law of large numbers replaced by the capacity theorem.

![Reed--Muller profiles and leverage](figures/rm_step.png)

*Median-degree Reed--Muller priors at $R=\tfrac12$ exactly ($n$ odd). Left: the exact $n=3$ profile (full subset enumeration) and Monte Carlo rank profiles at $n=5,7$ (6000 and 2500 random question orders), sharpening onto the step $\mathbf 1[t<\tfrac12]$. Right: the finite-$n$ leverage from the exact finite law, converging to the step $1\to1/R=2$ at half time.*

### 6.6 Extensions

- **Degree mixtures.** $p_j\propto\sum_d w_d\,\mathbf 1[\deg\phi_j\le d]$ with distinct limiting rates gives a staircase $\gamma$; refining the mixture approximates any nonincreasing profile with purely shape-based ensembles. The latent "which $d$" carries $O(\log n)$ bits and vanishes from the interior.
- **The multicanonical continuum.** No temperature schedule $\beta_n$ on the degree escapes the step: consecutive-degree mass ratios move by factors $e^{\Theta(2^n/n)}$ through the critical window, so any $\beta_n$ concentrates the degree on at most two adjacent values, one limiting rate. What does work is cancelling the density of states. Weight
$$
p_j\;\propto\;2^{-K_{\deg\phi_j}}\;g\!\Big(\frac{2\deg\phi_j-n}{\sqrt n}\Big),
\tag{6.15}
$$
a flat-histogram (Wang--Landau) reweighting with an arbitrary fixed density $g$ on the CLT window. The rate $R=\Phi(c)$ then inherits a continuous law $\mu$, and the profile is the mixture of steps $\gamma(t)=\mu(R>t)$: the continuum staircase. In particular $g=$ standard normal makes $R$ uniform on $(0,1)$, so $\gamma(t)=1-t$, $g(t)=t-t^2/2$, and the master law collapses to
$$
L(t)=\frac{1-(1-t)^2}{t-t^2/2}\;\equiv\;2:
\tag{6.16}
$$
constant leverage, the clique signature, from a blockless mechanism -- a sharp example for Appendix H that the flattest curve does not identify its microscopics.
- **What cancelling the density of states means.** Three readings of (6.15). *Hierarchical Bayes*: the nested classes $\mathrm{RM}(0)\subset\cdots\subset\mathrm{RM}(n)$ are model classes; a temperature prices hypotheses and is always outbid by the doubly exponential class sizes, while the reweighting prices the class itself and is indifferent within it -- Occam on classes, the only level where Occam survives super-exponential multiplicity. *Two-part MDL*: $-\log_2 p_j\approx|\text{class code}|+K_{\deg\phi_j}$, the exact codelength of writing the hypothesis down; the honest energy of a degree step is its $\binom nk$ coefficient bits, and $2^{-K}$ charges precisely that where $e^{-\beta k}$ charged a constant. *Statistical mechanics*: at a strongly first-order transition the canonical ensemble pins the order parameter; the flat-histogram ensemble holds the whole coexistence interval open, and here the order parameter is the rate $R$, whose law $\mu$ becomes a free dial.
- **Profile completeness.** The family realizes *every* admissible profile at $m=1$. A limiting profile must be nonincreasing with $0\le\gamma\le h_0=1$; any such $\gamma$ is its own layer-cake mixture of steps,
$$
\mu=-d\gamma\ \text{on}\ (0,1)\;+\;\big(1-\gamma(0^+)\big)\,\delta_0\;+\;\gamma(1^-)\,\delta_1,
\qquad
\gamma(t)=\mu(R>t),
\tag{6.17}
$$
realized by pulling $\mu$ back through $R=\Phi(c)$ into the window weight $g$. The atoms are meaningful: mass at $R=0$ is boundary-layer components ($\gamma(0^+)<1$), mass at $R=1$ is a uniform component ($\gamma(1^-)>0$), and an atom at interior $R$ is an interior jump of $\gamma$ -- primary here, in one limit, where the block machinery needs the secondary limit $r\to\infty$. Monotone-bounded being the complete constraint on profiles, no prior of any kind can reach outside this family's range.
- **The inverse recipe.** Given a target curve $L(t)$ and $h_0=1$: (i) solve the linear ODE of the inverse problem, $(1-t)g'+L\,g=1$, $g(0)=0$, by integrating factor $I(t)=\exp\int_0^t\frac{L(s)}{1-s}ds$, so $g=I^{-1}\int_0^tI(s)/(1-s)\,ds$; (ii) set $\gamma=g'$; (iii) check $\gamma$ nonincreasing with values in $[0,1]$ -- if it fails, *no* prior produces this $L$; (iv) layer-cake $\gamma$ into $\mu$ by (6.17); (v) at size $n$, give degree class $d$ the weight $w_d=\mu\big((R_{n,d-1},R_{n,d}]\big)$, the slice of $\mu$ between consecutive rates, i.e.
$$
p_j=\sum_d w_d\,2^{-K_d}\,\mathbf 1[\deg\phi_j\le d].
\tag{6.18}
$$
For $L\equiv2$ the recipe returns $g=t-t^2/2$, $\gamma=1-t$, $\mu$ uniform: each class weighted by the rate gap it spans, recovering (6.16).
- **Full support with every class alive.** Demand that $\mu$ have a strictly positive density $f$ on $(0,1)$. The consecutive rates differ by $\Delta R_d=\binom nd/2^n$, so
$$
w_d=\int_{R_{n,d-1}}^{R_{n,d}}f(R)\,dR\;\approx\;f(R_d)\,\frac{\binom nd}{2^n}\;>\;0
\quad\text{for every } d\in\{0,\dots,n\},
\tag{6.19}
$$
and since $\mathrm{RM}(n,n)$ is the whole map space, $p_j\ge w_n2^{-Q}>0$ everywhere: full support at every finite $n$ with no noise channel, from the nested supports alone. Each class contributes visibly: $w_d=\gamma(R_{d-1})-\gamma(R_d)$ is the profile's drop across the class's own rate gap, so class identity survives as the local slope of $\gamma$; the individual weights necessarily vanish (as $f/\sqrt n$ in the window, exponentially on the flanks), but the ratio weight-to-rate-gap is the invariant $f$. The support floor $w_n2^{-Q}$ is doubly exponentially thin; a fat floor requires an atom $\bar w$ at $R=1$ and costs exactly $\gamma(1^-)=\bar w$, a permanent full-support tail. In the inverse recipe, this version is available precisely when $\gamma=g'$ comes out strictly decreasing and continuous.
- **Full support.** Passing answers through the bit-flip channel of Appendix E preserves the limit for vanishing noise and smooths the step at fixed noise, exactly as for the block codes.
- **Other code families.** Any code sequence with a proven EXIT limit yields a solvable shape-based prior; the EXIT literature (polar, spatially coupled, LDPC under density evolution) is a dictionary of achievable profiles.

### 6.7 A worked prescription: a smooth interior maximum

The requirements: a simply constructible microscopic prior, monotone in complexity (every more complex map strictly less likely than every simpler one), full support, all classes contributing, and clean algebra ending in a smooth $L(t)$ with an interior peak. The prescription:

$$
\mu\;=\;w\,\mathrm{Beta}(2,2)\;+\;(1-w)\,\delta_1,
\qquad
f_{\mathrm{Beta}(2,2)}(R)=6R(1-R).
\tag{6.20}
$$

Microscopically: flip a $w$-coin. Heads: draw $R$ as the *median of three uniform numbers* (that is the Beta(2,2) law), find the degree class whose rate gap contains $R$, and draw a uniform polynomial of that class. Tails: draw a fully uniform map. The class weights are the slices $w_d=w\int_{R_{n,d-1}}^{R_{n,d}}6R(1-R)\,dR$ plus the atom $(1-w)$ on $d=n$; all are positive, so every class contributes, $p_j\ge(1-w)2^{-Q}>0$ has a fat full-support floor, and $p_j$ is strictly decreasing in $\deg\phi_j$ because the classes are nested.

The promised cancellation is the beta--binomial duality: the Beta(2,2) survival function with integer parameters is a Bernstein sum, and precisely the smoothstep polynomial already tabulated at (5.9),

$$
\mu\big((t,1)\big)\Big/w\;\Big|_{\rm cont}
=1-I_t(2,2)=\tau_{4,2}(t)=1-3t^2+2t^3,
\qquad
\gamma(t)=1-w\big(3t^2-2t^3\big).
\tag{6.21}
$$

Everything downstream is polynomial:

$$
g(t)=t-w\,t^3\Big(1-\frac t2\Big),
\qquad
1-(1-t)\gamma(t)=t\big[1+w\,t(1-t)(3-2t)\big],
$$

$$
L(t)=\frac{1+w\,t(1-t)(3-2t)}{1-w\,t^2\big(1-\tfrac t2\big)}.
\tag{6.22}
$$

Endpoints: $\gamma'(0)=0$, so $L(0^+)=1$ exactly with initial slope $L'(0)=3w>0$, and $g(1)=\mathbb E_\mu[R]=1-\tfrac w2$ gives $L(1)=1/\mathbb E_\mu[R]$, an instance of the general identity that completion leverage is the reciprocal mean class rate. Since $L'(0)>0$ and $L'(1)<0$ (this $\gamma$ is the interior-maximum example of 4.3 with $m=1$), the peak is interior and smooth: numerically $L^*=1.44$ at $t^*=0.65$ for $w=0.55$, $L^*=1.70$ at $t^*=0.75$ for $w=0.8$, and $L^*=1.91$ at $t^*=0.87$ for $w=0.95$, against the endpoint values $1/(1-w/2)=1.38,\,1.67,\,1.90$.

![Multicanonical beta-bump profiles and leverage](figures/rm_peak.png)

*The prescription (6.20) at $w=0.55,\,0.8,\,0.95$: smoothstep profiles (left) and leverage curves with smooth interior maxima (right, dots), computed from (6.22).*

The timing story: early on most classes have rates above $t$, so answers are mostly fresh and $L$ hugs the baseline; as $t$ sweeps through the beta bump, class after class is exhausted and deductions surge; past the bump only the uniform remnant keeps paying full price with no deductions left, and the cumulative ratio sags toward $1/\mathbb E_\mu[R]$. Sharper beliefs steepen the surge: replacing $\mathrm{Beta}(2,2)$ by $\mathrm{Beta}(k,r-k)$ concentrates the bump, raises the peak toward the step of 6.4--6.5, and moves it to $t\approx k/r$, with all integrals staying polynomial by the same duality.

### 6.8 The smoothness (Ising) prior: model and derivation roadmap

The geometric alternative encodes "similar questions have similar answers." Encode answers as spins $\sigma_q(\phi):=(-1)^{\phi(q)}$ and weight maps by agreement along the $nQ/2$ hypercube edges:

$$
p_j\;\propto\;\exp\Big(\beta\sum_{q\sim q'}\sigma_q(\phi_j)\,\sigma_{q'}(\phi_j)\Big),
\qquad q\sim q'\iff\text{Hamming distance }1.
\tag{6.12}
$$

This is a ferromagnetic Ising model whose lattice is the $n$-cube and whose spins are the answers; agreements and the spin product differ only by constants absorbed into $\beta$. It is non-exchangeable, block-free, full-support, and its $p_j$ depends only on the geometry of the map. The derivation it would take, in order:

1. **Choose the scaling.** Each edge carries mutual information $\Theta(\beta^2)$, and there are $nQ/2$ edges, so extensive stored correlation, $\Theta(Q)$ bits, requires $\beta=b/\sqrt n$. The neighboring scalings both degenerate: at $\beta=\Theta(1)$ the diverging degree freezes the model onto the two constant maps (a global clique), and at $\beta=\Theta(1/n)$ the correlation store is $O(Q/n)$, subextensive, and the profile is flat.
2. **Symmetries first.** At zero external field the $\pm$ gauge gives $h_0=1$ exactly. Full support means $\gamma(1)>0$: unlike the code priors, the profile does not vanish at the end of the run.
3. **Condition on a macroscopic subset.** $\gamma(t)$ is the mean conditional entropy of one unrevealed spin given the answers on a uniform $tQ$-subset. The cavity field on an unrevealed question is $(b/\sqrt n)\big[\sum_{\text{revealed nbrs}}\sigma+\sum_{\text{unrevealed nbrs}}\langle\sigma\rangle\big]$: a sum of $\approx tn$ pinned terms and $(1-t)n$ fluctuating ones, each of size $1/\sqrt n$. By the CLT the field converges to a centered Gaussian of variance $v(t)$.
4. **Close the self-consistency.** The variance obeys a fixed point coupling the revealed density to the conditional magnetizations of the unrevealed spins,
$$
v(t)=b^2\big[t+(1-t)\,q(t)\big],
\qquad
q(t)=\mathbb E_Z\tanh^2\!\big(\sqrt{v(t)}\,Z\big),
\tag{6.13}
$$
the ferromagnetic analogue of a replica-symmetric cavity equation. For $b<1$ the paramagnetic branch is unique; for $b>1$ a one-bit global latent (the pure-state sign) appears and the computation proceeds within a pure state.
5. **Read off the profile.** $\gamma(t)=\mathbb E_Z\,h_2\big(\tfrac12(1+\tanh(\sqrt{v(t)}\,Z))\big)$. Sanity checks: $v(0)=0$ in the paramagnetic phase, so $\gamma(0^+)=1=h_0$, a genuinely smooth curve with **no boundary layer**; $v$ increases with $t$, so $\gamma$ decreases; $v(1)=b^2$ gives $\gamma(1)>0$.
6. **Integrate and substitute.** $g(t)=\int_0^t\gamma$ and the master law give a smooth $L(t)$ with no jumps or singularities. Its starting value is *not* the baseline: expanding the fixed point for small $t$ ($q\approx v$, $v\approx b^2t/(1-b^2)$, $\gamma\approx1-v/(2\ln2)$) gives a finite slope for $\gamma$ at zero and hence
$$
L(0^+)=1+\frac{|\gamma'(0)|}{h_0}
=1+\frac{b^2}{2\ln2\,(1-b^2)},
\tag{6.14}
$$
from which $L$ *declines* smoothly toward $L(1)$. The timing is first-touch, clique-like but saturating: early answers immediately bias all their neighbours, and later answers are increasingly redundant. This is a shape the zoo does not contain in pure form: a smooth declining curve with full support, no jump at $t=0$ and no $1/t$ singularity.

![Ising smoothness-prior profiles and leverage](figures/ising_smooth.png)

*The cavity fixed point (6.13) solved numerically at $b=0.35,\,0.7,\,0.95$. Left: smooth profiles with $\gamma(0^+)=h_0=1$ (no boundary layer) and $\gamma(1)>0$ (full support). Right: the leverage starts at the finite plateau (6.14), equal to $1.10$, $1.69$ and $7.68$ respectively, and declines smoothly with no jumps.*

Two honest caveats. In this scaling the cube's geometry washes out: any $n$-regular expander with $1/\sqrt n$ couplings gives the same limit, so the "smoothness on the cube" story survives as motivation while the answer is mean-field universal. And step 3--4 are cavity-method level: provable by Gaussian interpolation at high temperature ($b$ small), open near criticality. It is the right second target; the code prior is the first.

### 6.9 Instructive failures

- **$k$-junta priors** (map depends on a random set of $k$ relevant inputs): the latent is $2^k m=O(1)$ bits, so everything collapses into the boundary layer and the interior is a hyperbola-with-jump.
- **Bias-Gibbs priors** ($p_j$ a function of the truth-table weight): exchangeable, hence a coin mixture in the limit, hence a hyperbola.
- **Uniform over monotone maps**: entropy $\sim\binom{n}{n/2}=Q/\Theta(\sqrt n)$ is subextensive, so the normalized profile degenerates.

Each fails one constraint; together they show that a non-hyperbolic block-free prior needs $\Theta(Q)$ bits of stored structure organized non-exchangeably, and the low-degree prior is the minimal natural object that does it.

## 7. The finite law as the engine: from a sampling rule to $L(t)$

This chapter is self-contained: every object is introduced where it first
appears. The engine is the exact finite law (2.8),

$$
\langle L_{n,\ell}\rangle
=\frac{Q\,G_{n,1}-(Q-\ell)\,\gamma_{n,\ell}}{G_{n,\ell}},
\qquad \ell=\lfloor tQ\rfloor,
\tag{7.1}
$$

whose three ingredients -- the prior table entropy $Q\,G_{n,1}$, the
received entropy $G_{n,\ell}$, and the per-question increment
$\gamma_{n,\ell}=G_{n,\ell+1}-G_{n,\ell}$ -- will each be computed for a
prior built from a *generic* sampling rule: a single random number $R$
drawn from an arbitrary distribution. The three limits come out as
functionals of that distribution alone, the leverage curve follows for
the whole family at once, and only at the very end is the distribution
chosen, to recover one concrete curve.

First, the constituents of these entropies. Fix a set of questions
$\mathcal S=\{q_1,\dots,q_\ell\}$ and write
$\psi(\mathcal S)=(\psi(q_1),\dots,\psi(q_\ell))$ for the answer
vector, a random element of $\{0,1\}^\ell$ (any fixed ordering of
$\mathcal S$ will do; entropy is blind to relabelings). Its
distribution is induced by the prior: the probability of the answer
pattern $y$ is the total prior weight of the maps that answer $y$ on
$\mathcal S$, and $H(\psi(\mathcal S))$ is the Shannon entropy of
that $2^\ell$-outcome distribution,

$$
\Pr\big(\psi(\mathcal S)=y\big)
=\sum_{j\,:\,\phi_j(\mathcal S)=y}p_j,
\qquad
H\big(\psi(\mathcal S)\big)
=-\sum_{y\in\{0,1\}^\ell}
\Pr\big(\psi(\mathcal S)=y\big)\,
\log_2\Pr\big(\psi(\mathcal S)=y\big):
\tag{7.2}
$$

Werner's uncertainty, in bits, about the joint answers on
$\mathcal S$ before any are asked. The quantities in (7.1) are the
uniform averages of these entropies over question sets and their
increments,

$$
G_{n,\ell}
=\binom Q\ell^{-1}\sum_{\mathcal S\,:\,|\mathcal S|=\ell}
H\big(\psi(\mathcal S)\big),
\qquad
\gamma_{n,\ell}=G_{n,\ell+1}-G_{n,\ell},
\tag{7.3}
$$

with $G_{n,1}$ the $\ell=1$ case: the mean entropy of a single answer.

### 7.1 The hypothesis classes

The truth is a Boolean map $\phi:\{0,1\}^n\to\{0,1\}$, a table of
$Q=2^n$ answer bits. To organize the $2^Q$ candidate maps by
complexity, write them as polynomials. The arithmetic lives in
$\mathbb F_2$, the two-element field $\{0,1\}$ in which addition is XOR
($1\oplus1=0$) and multiplication is AND; all linear algebra in this
chapter -- spans, ranks, matrix products -- is over this field. Since a
bit satisfies $x^2=x$, no variable is ever squared, so a monomial is
labelled by the *set* $S\subseteq\{1,\dots,n\}$ of variables it
contains, and every map is a formal XOR of monomials,

$$
\phi(x)=\bigoplus_{S}a_S\prod_{i\in S}x_i,
\qquad a_S\in\{0,1\}.
$$

The *coefficients* $a_S$ are single bits: monomial $S$ is present or
absent. The *weight* of a monomial is $|S|$, and the *degree* of the
map, $\deg\phi$, is the largest weight of a present monomial. There are
$2^n$ monomials, hence $2^Q$ coefficient vectors -- exactly as many as
maps, and the correspondence is a bijection (the *algebraic normal
form*): the coefficients are recovered from the truth table by M\"obius
inversion, $a_S=\bigoplus_{y\subseteq S}\phi(y)$, an XOR over the
subcube below $S$. Example: OR answers $1$ except at $00$, and
$\mathrm{OR}=b_1b_0\oplus b_1\oplus b_0$ (check all four inputs), so
its degree is $2$.

Mechanically the inversion is the $Z$-butterfly. Per bit it is an
upper-triangular $2\times2$ kernel, and on $n$ bits its tensor power is
the subset indicator; in the document's descending question order
$11,10,01,00$ (per bit: $1,0$),

$$
Z=\begin{pmatrix}1&1\\0&1\end{pmatrix},
\qquad
\big(Z^{\otimes n}\big)_{S,y}=\mathbf 1[\,y\subseteq S\,],
\qquad\text{e.g.}\quad
Z^{\otimes 2}=
\begin{pmatrix}
1&1&1&1\\
0&1&0&1\\
0&0&1&1\\
0&0&0&1
\end{pmatrix}.
$$

(Here $\mathbf 1[\cdot]$, used throughout, is the *indicator*: a
function that evaluates a statement to a bit, $1$ if true and $0$ if
false.) Over $\mathbb F_2$ the kernel is its own inverse, $Z^2=I$. As
an in-place algorithm: for each bit $i$, for every input $x$ with
$x_i=1$, do $T[x]\mathrel{\oplus=}T[x\text{ with bit }i\text{
cleared}]$; $n2^{n-1}$ XORs in total. The name "triangular half of
Walsh--Hadamard" is the factorization (same $1,0$ order)

$$
W=\begin{pmatrix}-1&1\\1&1\end{pmatrix}
=Z\begin{pmatrix}-2&0\\0&1\end{pmatrix}Z^{\mathsf T},
$$

so the correlator ledger's transform is two $Z$-passes with a
$(-2)^{|S|}$ rescaling between them; mod $2$ the rescaling dies and the
XOR butterfly is what remains. Worked at $n=2$, truth tables written in
the numeral convention $T=(\phi(11),\phi(10),\phi(01),\phi(00))$ -- so
that $T$ read as a binary number *is* the map index $j$ -- and
coefficients ordered $(a_{b_1b_0},a_{b_1},a_{b_0},a_\emptyset)$:

| map ($j$) | $T$ | $a$ | polynomial | $\deg$ |
|---|---|---|---|---|
| TRUE (15) | $(1,1,1,1)$ | $(0,0,0,1)$ | $1$ | $0$ |
| $\lnot b_1$ (3) | $(0,0,1,1)$ | $(0,1,0,1)$ | $b_1\oplus 1$ | $1$ |
| XOR (6) | $(0,1,1,0)$ | $(0,1,1,0)$ | $b_1\oplus b_0$ | $1$ |
| AND (8) | $(1,0,0,0)$ | $(1,0,0,0)$ | $b_1b_0$ | $2$ |
| NAND (7) | $(0,1,1,1)$ | $(1,0,0,1)$ | $b_1b_0\oplus 1$ | $2$ |
| $b_1\to b_0$ (11) | $(1,0,1,1)$ | $(1,1,0,1)$ | $b_1b_0\oplus b_1\oplus 1$ | $2$ |

Now grade the hypothesis space by degree. The *class*
$\mathrm{RM}(d,n)$ -- the Reed--Muller code -- is the set of all maps
of degree at most $d$: a linear space of dimension
$K_d=\sum_{i\le d}\binom ni$ (one fair coin per allowed monomial), and
the classes are nested, $\mathrm{RM}(0,n)\subset\cdots\subset
\mathrm{RM}(n,n)$, the last being the entire map space. The *rate* of a
class is its dimension per question,

$$
R_{n,d}=\frac{K_d}Q\in(0,1],
$$

and it will turn out to be the fraction of a full run after which the
class's member is pinned. Consecutive rates slice $(0,1]$ into the
*rate gaps* $(R_{n,d-1},R_{n,d}]$ of lengths
$\Delta R_d=\binom nd/2^n\le\sqrt{2/\pi n}$: a partition of the unit
interval, one cell per class, refining as $n$ grows.

### 7.2 The sampling rule and the prior

The prior is built by a lottery with one free choice. Fix a
distribution $\mu$ on $[0,1]$ -- described equivalently by its CDF
$F(x)=\Pr(R\le x)$ or, when it has one, its density $f=F'$ -- and:

1. draw a single random number $R\sim\mu$ (the *statistic*);
2. let $D$ be the class whose rate gap contains $R$;
3. draw the truth uniformly from class $D$: each of the $K_D$ allowed
   monomial coefficients by a fair coin.

In every expectation below, $\mathbb E[\min(t,R)]$, $\mathbb E_\mu[R]$,
the random object is this one draw $R$; the $R_{n,d}$ are the
deterministic grid it lands among. The class index $D$ is a *latent
label*: latent because Werner never observes it directly, a label
because it is one value among $n+1$, hence worth at most
$\log_2(n+1)$ bits. Integrating the lottery, the class weights are the
$\mu$-masses of the gaps, and the prior weight of a map is the sum over
the classes that contain it -- by nestedness, all classes at or above
its degree:

$$
w_d=F(R_{n,d})-F(R_{n,d-1}),
\qquad
p_j=\sum_{d\ge\deg\phi_j}w_d\,2^{-K_d},
\tag{7.4}
$$

with $\sum_dw_d=F(1)-F(0)=1$ telescoping. Two structural properties,
inherited from nestedness alone. If $F$ increases across every gap
(e.g.\ $\mu$ has a positive density), then every $w_d>0$, so $p_j$ is
*strictly decreasing in the degree*: Occam, with each degree step
costing a factor of roughly $2^{-\binom nd}$. And the $d=n$ term gives
$p_j\ge w_n2^{-Q}>0$ for every map: *full support*, with no noise
channel, from the nested supports alone. Atoms of $\mu$ are allowed --
an atom inside a gap simply loads that one class -- at the price that
classes in any flat stretch of $F$ carry zero weight.

### 7.3 Term one: the prior table entropy

A class is a linear code, and it pays to make its linear structure
explicit once. The *generator matrix* $\mathsf G$ of class $d$
(sans serif, to keep it distinct from the block entropies $G_{n,\ell}$)
is the $K_d\times Q$ matrix over $\mathbb F_2$ whose rows are the
allowed monomials evaluated at every question,
$\mathsf G_{S,q}=\prod_{i\in S}q_i$; a uniform member of the class is
$\psi=u^{\mathsf T}\mathsf G$ with $u$ uniform on $\{0,1\}^{K_d}$.

One column of the answer matrix is the law of a single answer
$\psi(q)$. The $q$-column of $\mathsf G$ is nonzero -- its
constant-monomial entry is $\prod_{i\in\emptyset}q_i=1$ -- so
$\psi(q)$ is the image of a uniform $u$ under a surjective linear map
onto $\mathbb F_2$: a fair bit, exactly, in every class, whatever the
class. A mixture of fair bits is a fair bit. Hence at every finite
$n$, with no limit taken and no reference to $\mu$,

$$
G_{n,1}=1,
\qquad
H\big(M^{(0)}\big)=Q .
\tag{7.5}
$$

The entropy reading: this prior predicts nothing about any single
answer; its entire content is correlational. The identity entropy is
$H(p)\approx\sum_dw_dK_d\to Q\,\mathbb E_\mu[R]$, so the dependency
stock is extensive, $C=H(M^{(0)})-H(p)\approx Q\int_0^1F$: the fraction
of the table stored in correlations is the area *under* the CDF.

### 7.4 Term two: the received entropy

$G_{n,\ell}$ is the mean entropy of the answers to $\ell$ uniformly
random questions. Four elementary steps take it to a functional of
$F$.

**(a) Ungroup the mixture.** Computing the joint entropy
$H(\psi(\mathcal S),D)$ by the chain rule in both orders and
rearranging,

$$
H\big(\psi(\mathcal S)\big)
=\sum_dw_d\,H_d\big(\psi(\mathcal S)\big)
+I\big(D;\psi(\mathcal S)\big),
\qquad
0\le I\le H(w)\le\log_2(n+1):
\tag{7.6}
$$

mixture entropy is mean class entropy plus the information the answers
carry about the latent label. The learner does learn *which class*
along the way, but that is a label among $n+1$ options -- at most
$\log_2(n+1)$ bits across the whole run, $o(1)$ per question.
Identifying the class is cheap; identifying the member is the expensive
part. Averaged over question sets:
$G_{n,\ell}=\sum_dw_dG^{(d)}_{n,\ell}+O(\log n)$.

**(b) Class entropy is rank.** Within class $d$, the answers on
$\mathcal S$ are $\psi(\mathcal S)=u^{\mathsf T}\mathsf
G_{\cdot,\mathcal S}$: a uniform $u$ pushed through the column
submatrix, hence uniform on its image, a subspace whose size is
$2^{\mathrm{rank}}$. So $H_d(\psi(\mathcal S))
=\operatorname{rank}(\mathsf G_{\cdot,\mathcal S})$ exactly, where the
*rank* of a set of columns is the dimension of their span over
$\mathbb F_2$ -- operationally, the number of bits needed to describe
those answers.

**(c) The clipped clock.** Trivially
$\mathrm{rank}\le\min(\ell,K_d)$: a rank is bounded by its column count
and by the ambient dimension. Three facts force the matching limit.
*Monotone increments*: the mean increment
$\gamma^{(d)}_{n,\ell}=\Pr[\text{a fresh column is independent of }\ell
\text{ random ones}]$ is nonincreasing in $\ell$ by rank
*submodularity* -- diminishing returns,
$\mathrm{rank}(\mathcal B\cup q)-\mathrm{rank}(\mathcal B)
\le\mathrm{rank}(\mathcal A\cup q)-\mathrm{rank}(\mathcal A)$ for
$\mathcal A\subseteq\mathcal B$: what is dependent on few columns is
dependent on more. *The finite-$n$ area theorem*: increments telescope
to the dimension, $\sum_{\ell<Q}\gamma^{(d)}_{n,\ell}=K_d$ -- received
information over a complete run equals the size of what there was to
learn (in coding theory, the area under the EXIT curve equals the
rate). *The capacity theorem*, the one imported result: Reed--Muller codes
achieve capacity on the binary erasure channel
(Kudekar--Kumar--Mondelli--Pfister--\c Sa\c so\u glu--Urbanke, 2017).
In the vocabulary here: reveal any fraction of the questions beyond
the class rate, and a fresh answer is determined with probability
tending to one,

$$
R_{n,d_n}\to R\in(0,1)
\quad\Longrightarrow\quad
\lim_{n\to\infty}\gamma^{(d_n)}_{n,\lfloor tQ\rfloor}=0
\quad\text{for every fixed }t>R.
\tag{KKMPSU}
$$

A bounded nonincreasing sequence with area $R_d$ and no mass above
$R_d$ has nowhere to sit but at height $1$ below, so the increments
converge to a pure step, written with the Heaviside function
$\theta(x)=\mathbf 1[x>0]$:

$$
\lim_{n\to\infty}\gamma^{(d)}_{n,\lfloor tQ\rfloor}
=\mathbf 1[t<R_d]
=\theta(R_d-t).
\tag{7.7}
$$

Integrating the step,

$$
\frac{G^{(d)}_{n,\lfloor tQ\rfloor}}Q
=\frac1Q\sum_{\ell'<tQ}\gamma^{(d)}_{n,\ell'}
\;\longrightarrow\;
\int_0^t\theta(R_d-x)\,dx=\min(t,R_d):
\tag{7.8}
$$

each class pays one full bit per question until its clock stops at its
own rate, and nothing after.

**(d) From the class grid to the statistic.** Let $R^{(n)}$ be the
draw $R$ rounded up to the rate at the top of its gap, so that
$\sum_dw_d\min(t,R_{n,d})=\mathbb E[\min(t,R^{(n)})]$. Since
$\min(t,\cdot)$ is $1$-Lipschitz and the rounding moves $R$ by at most
the largest gap, the error is at most
$\max_d\Delta R_d\le\sqrt{2/\pi n}\to0$, uniformly in $t$. Hence

$$
\frac{G_{n,\lfloor tQ\rfloor}}Q
\;\longrightarrow\;
\mathbb E_\mu\big[\min(t,R)\big]
\;=\;\int_0^t\big(1-F(x)\big)\,dx\;=:\;g(t).
\tag{7.9}
$$

The last equality is the *layer-cake* identity, and it deserves one
line: pointwise, $\min(t,R)=\int_0^t\mathbf 1[x<R]\,dx$ (if $R\ge t$
the integral is $t$, otherwise it stops at $R$); take expectations and
swap integral and expectation, and $\mathbb E\,\mathbf 1[x<R]
=\Pr(R>x)=1-F(x)$. Received information per question is the average of
clipped clocks, and equals the area under the survival function up to
time $t$.

### 7.5 Term three: the increment

The sequence $\ell\mapsto G_{n,\ell}$ is concave: its increments
$\gamma_{n,\ell}$ are nonincreasing, by the same conditioning argument
as in the general theory. Slopes of concave functions converge wherever
the limit is differentiable -- the elementary convexity lemma of
calculus -- and by (7.9) the normalized values converge to $g$, which
is differentiable at every continuity point of $F$. Therefore

$$
\gamma_{n,\lfloor tQ\rfloor}
\;\longrightarrow\;
g'(t)=1-F(t)=\Pr(R>t),
\tag{7.10}
$$

at every such point. This is the heart of the construction, and it is
worth saying in words: the mean conditional entropy of a fresh answer
is the probability that the true class is not yet exhausted. The
survival function of the statistic is not a postulate; it *is* the
increment term of the finite law. The remaining-entropy term follows by
multiplying by the unasked fraction:

$$
\frac{(Q-\ell)\,\gamma_{n,\ell}}Q
\;\longrightarrow\;
(1-t)\,\big(1-F(t)\big).
\tag{7.11}
$$

### 7.6 The generic curve

Substitute (7.5), (7.9), (7.11) into (7.1) and cancel the common factor
$Q$. The numerator tidies up: $1-(1-t)(1-F(t))=t+(1-t)F(t)$. So, for
any sampling law with CDF $F$,

$$
L_F(t)
=\frac{t+(1-t)\,F(t)}{\displaystyle\int_0^t\big(1-F(x)\big)\,dx}.
\tag{7.12}
$$

The numerator has its own entropy reading: destroyed table density
equals the asked fraction $t$ (every asked column is settled by its own
answer) plus $F(t)$ per *unasked* column -- an unasked answer is
deducible exactly when the class died before $t$, an event of
probability $F(t)$. Asked plus deduced over received. Two generic
endpoint identities follow by expanding at the ends:

$$
L_F(0^+)=1+\lim_{t\downarrow0}\frac{F(t)}t=1+f(0^+),
\qquad
L_F(1)=\frac1{\int_0^1(1-F)}=\frac1{\mathbb E_\mu[R]}:
$$

the curve starts at the baseline plus the density of the statistic at
zero, and completes at the reciprocal mean rate. Since any nonincreasing
profile is the survival function of exactly one law (the layer-cake
read backwards, as in 6.6), equation (7.12) is not one family among
many: it is the general admissible curve written in microscopic form,
with the statistic as its parameter.

### 7.7 Recovering the curve: $R=\max(U_1,U_2)$

Now make the one free choice. Draw two uniform numbers and keep the
larger. Both draws land below $x$ with probability $x\cdot x$, so

$$
F(x)=x^2,
\qquad
f(x)=2x,
\qquad
\mathbb E[R]=\int_0^1(1-x^2)\,dx=\tfrac23 .
\tag{7.13}
$$

The weights (7.4) become gaps of squared rates,
$w_d=R_{n,d}^2-R_{n,d-1}^2$. Worked at $n=2$: the classes have
$(K_0,K_1,K_2)=(1,3,4)$ and rates $(\tfrac14,\tfrac34,1)$, so
$w=(\tfrac1{16},\tfrac12,\tfrac7{16})$, and (7.4) prices the sixteen
maps of the butterfly table in three tiers: the two constants at
$\tfrac1{16}\cdot\tfrac12+\tfrac12\cdot\tfrac18+\tfrac7{16}\cdot
\tfrac1{16}=\tfrac{31}{256}$, the six affine maps at $\tfrac{23}{256}$,
the eight curved maps at $\tfrac{7}{256}$; the check
$2\cdot31+6\cdot23+8\cdot7=256$ closes.

The three terms specialize in one line each:

$$
g(t)=\int_0^t(1-x^2)\,dx=t-\frac{t^3}3,
\qquad
\gamma(t)=1-t^2,
\qquad
t+(1-t)t^2=t\,(1+t-t^2),
$$

and (7.12) becomes

$$
L(t)=\frac{t\,(1+t-t^2)}{t\,\dfrac{3-t^2}3}
=\frac{3\,(1+t-t^2)}{3-t^2}.
\tag{7.14}
$$

The generic endpoint identities check out: $L(0^+)=1+f(0^+)=1$ (the
density vanishes at zero, so the curve leaves the baseline flat), and
$L(1)=1/\mathbb E[R]=\tfrac32$; in between $L'\propto(1-t)(3-t)>0$, a
smooth strictly rising curve.

![Order-statistic class priors](figures/rm_orderstat.png)

*One uniform draw ($F=t$) gives $L\equiv2$; the smaller of two draws
($1-F=(1-t)^2$) gives the clique-triple profile with $L\equiv3$; the
larger of two draws gives (7.14), rising from $1$ to $\tfrac32$.*

Other choices of the statistic, all through the same (7.12):

| statistic | $F(t)$ | $\gamma(t)=1-F$ | $L_F(t)$ |
|---|---|---|---|
| one draw $U$ | $t$ | $1-t$ | $\equiv2$ |
| $\min(U_1,U_2)$ | $2t-t^2$ | $(1-t)^2$ | $\equiv3$ (clique triple) |
| $\max(U_1,U_2)$ | $t^2$ | $1-t^2$ | rising $1\to\tfrac32$, eq.\ (7.14) |
| $k$-th of $m$ draws | $I_t(k,m{+}1{-}k)$ | $\tau_{m+1,k}(t)$ | the MDS family of 5.4 |
| $k/m\to\rho$, $m\to\infty$ | $\to\mathbf 1[t\ge\rho]$ | $\to\mathbf 1[t<\rho]$ | the capacity step of 6.5 |
| any of the above $+\,(1-w)\delta_1$ | $wF$ | $1-wF$ | tail tilts down; interior peak as in 6.7 |

Choosing the statistic is choosing the timetable on which complexity
classes exhaust: early-loaded statistics ($\min$) front-load the
deductions and flatten the curve to a clique-like constant, late-loaded
ones ($\max$, higher order statistics) delay them and make the curve
rise, concentration sharpens toward the step, and an atom at $R=1$ -- a
share of "anything goes" -- keeps paying fresh bits forever and bends
the tail down into an interior maximum.

## 8. The weight prior: sparsity as complexity

Chapter 7 graded maps by the *degree* of their polynomial, the width of the widest monomial. This chapter asks about the other natural size of a coefficient vector: its Hamming weight $\mathrm{wt}(\rho)$, the *number* of monomials present. Two questions drive it. Is weight a better proxy for circuit complexity than degree? And does the leverage TDL survive the change of measure?

### 8.1 What circuits actually charge for

The Gibbs chapter tabulated the AIG complexity $X$ of all $65{,}536$ maps at $(4,1)$; the butterfly gives each map's degree and weight; correlate. Spearman rank correlations with $X$:

| statistic | tractable classes? | $\rho_{\rm Spearman}$ with $X$ |
|---|---|---|
| ANF degree | yes (RM chain) | $+0.017$ |
| max-index in *any* monomial order | yes (any chain) | $-0.003$ |
| ANF weight | no (Hamming balls) | $+0.286$ |
| support (junta size) | components only | $+0.027$ |
| $|$weight bias$|$ | no | $-0.137$ |
| footprint $nI+S$ | no | $+0.027$ |

Degree is nearly uncorrelated with $X$ in the bulk -- $96\%$ of maps sit at degree $3$--$4$, where the mean $X$ is flat at $7.0$. What degree does control is the *floor*: the cheapest map in each degree shell costs exactly $d-1$ gates (the bare product; the classical multiplicative-complexity bound $X \ge \deg - 1$, tight at every $d$). The named examples say it plainly:

| map | deg | wt | AIG $X$ |
|---|---|---|---|
| $b_3b_2b_1b_0$ | $4$ | $1$ | $3$ |
| $b_2b_1b_0$ | $3$ | $1$ | $2$ |
| $b_3\oplus b_2\oplus b_1\oplus b_0$ | $1$ | $4$ | $9$ |
| all six pair monomials XORed | $2$ | $6$ | $9$ |

Needing a term that multiplies many bits is *complex as a floor, not as a price*: it forces $d-1$ unavoidable nonlinear gates, but the floor is cheap, and almost all AIG cost lives in how many terms are XORed together ($3$ gates per XOR), which is the weight. So weight is the more faithful Occam -- and the table's second row is a no-free-lunch in advance: the *entire* class of statistics whose level sets are nested linear spaces (pick any order on the monomials; complexity $=$ position of the last monomial present) is max-type, and max-type statistics saturate on the populous shells exactly as degree does. Any statistic that sees accumulation cost is not of that class. Tractability and fidelity to $X$ pull in opposite directions.

### 8.2 The prior: one temperature suffices

Where the degree's density of states is doubly exponential and forces the sampling rule, the weight's is binomial, $\binom Qw$ coefficient vectors of weight $w$, and a plain Gibbs weight works:

$$
p_j \;\propto\; e^{-\lambda\,\mathrm{wt}(\rho_j)}
\qquad\Longleftrightarrow\qquad
\rho_S \;\overset{\text{iid}}{\sim}\;\mathrm{Bernoulli}(\theta),
\quad \theta=\frac1{1+e^\lambda},
\tag{8.1}
$$

every coefficient an independent biased coin. The energy-entropy balance $e^{-\lambda} \cdot \frac{Q-w}{w+1} = 1$ has an interior solution $w^* = \theta Q$ with $O(\sqrt Q)$ fluctuations: no shell cliff, no multicanonical rescue. The prior is constructive (butterfly, then a popcount), has full support, and is strictly Occam in weight, $p_j \propto \theta^{\mathrm{wt}}(1-\theta)^{Q-\mathrm{wt}}$ decreasing in $\mathrm{wt}$ for $\theta < \tfrac12$.

### 8.3 What survives exactly

The map table is an invertible linear scramble of iid bits: $\psi = \rho^{\mathsf T} Z^{\otimes n}$, with the butterfly kernel itself as the generator. Exact consequences, no limits taken:

- **Total entropy.** The scramble is a bijection, so $H(\psi) = Q\,h_2(\theta)$: the correlation store is extensive, $C/Q \to 1-h_2(\theta)$.
- **Area theorem.** Telescoping never needed linearity of the classes: $\sum_{\ell<Q}\gamma_{n,\ell} = Q\,h_2(\theta)$ exactly, so $\int_0^1\gamma = h_2(\theta)$ in any limit.
- **Monotonicity.** Entropy submodularity makes $\gamma_{n,\ell}$ nonincreasing, as always.
- **Single-answer bias.** $\psi(q)$ XORs the $2^{|q|}$ coefficients below $q$ ($|q|$ the question's Hamming weight), so with $\varepsilon := 1-2\theta$,
$$
\mathbb E\,(-1)^{\psi(q)} = \varepsilon^{\,2^{|q|}},
\qquad
h_0^{(n)} = 2^{-n}\sum_k \binom nk\, h_2\!\Big(\tfrac{1+\varepsilon^{2^k}}2\Big)
\;\longrightarrow\; 1 .
\tag{8.2}
$$
A typical question has weight $\approx n/2$ and XORs $\sqrt Q$ coins: its answer is a fair bit to doubly-exponential accuracy. Only the $o(Q)$ low-weight questions carry visible bias -- the prior is automatically non-exchangeable, graded by the *questions'* weights where chapter 7 graded the *monomials'*.
- **The block law.** For a question set $\mathcal S$, Fourier inversion gives the exact distribution
$$
\Pr\big(\psi(\mathcal S)=y\big)
=2^{-\ell}\sum_{c\in\{0,1\}^\ell}(-1)^{c\cdot y}\,
\varepsilon^{\,\mathrm{wt}(\mathsf G_{\cdot,\mathcal S}\,c)} .
\tag{8.3}
$$

### 8.4 What breaks

Everything above is exact; what is lost is the *one-line evaluation*. The level sets $\{\mathrm{wt}\le w\}$ are Hamming balls, not subspaces, so $\psi(\mathcal S)$ is not uniform on an image and entropy is no longer a rank. By (8.3) the block entropy is a functional of the **weight enumerator** of the span of the observed columns -- how many of the $2^\ell$ combinations $\mathsf G_{\cdot,\mathcal S}c$ have low weight -- and no clipped-clock argument, no rank submodularity shortcut, and no KKMPSU import apply. Computing $\gamma(t)$ is now a genuine statistical-mechanics problem: infer an iid biased vector from $tQ$ noiseless linear observations with the deterministic subset-indicator matrix.

### 8.5 The conjectured TDL of leverage

The scaffolding that pinned the degree prior's step is still standing: $\gamma_n$ nonincreasing, bounded by $h_0 \to 1$, with exact area $h_2(\theta)$. So by the same no-room argument, the *entire* limit hinges on one missing theorem, the source-coding analogue of KKMPSU:

> revealing any fraction $t > h_2(\theta)$ of the table determines a fresh answer with probability tending to one.

That is the counting bound (you cannot pin $Q\,h_2(\theta)$ bits with fewer than $Q\,h_2(\theta)$ nearly-fair answers; the missing part is that the deterministic matrix achieves it). Three supports: dense random linear observations of a Bernoulli-$\theta$ source do reconstruct at any rate above $h_2(\theta)$, and typical columns here are $\sqrt Q$-dense; the same matrix $Z^{\otimes n}$ is Ar{\i}kan's polar kernel, and polarization theory proves the *sequential-order* version of the statement; and the finite-$n$ curves drift the right way. If the theorem holds,

$$
\gamma(t) \;=\; \theta_{\rm H}\big(h_2(\theta)-t\big),
\qquad
L(t)=
\begin{cases}
1, & t < h_2(\theta),\\[2pt]
1/h_2(\theta), & t > h_2(\theta):
\end{cases}
\tag{8.4}
$$

the capacity step again, with the rate dialed continuously by the temperature -- what took the degree prior a multicanonical mixture, the weight prior would do with a single scalar $\theta$. One more identifiability entry: two entirely different Occams, degree with sampled levels and weight with one temperature, conjecturally share the same macroscopic curve family.

![Weight-prior finite-n curves](figures/weight_step.png)

*Exact-entropy Monte Carlo at $n=4$ ($300$ question sets per size, block entropies exact via (8.3) and a fast WHT). Landmarks confirmed to machine precision: $G_{n,1}$ equals the bias formula (8.2), and $G_{n,Q} = Q h_2(\theta)$ exactly. The $\theta=0.11$ profile crosses half-height almost exactly at its conjectured step $t = h_2(0.11) = 0.500$; the leverage hovers near $1/h_2(\theta)$ over the whole run rather than stepping, because at $n=4$ the low-weight questions leak biased answers from the first draw -- a fat boundary layer, opposite in character to the degree prior's label-blocked start.*

### 8.6 Verdict

Would it work? As a prior: yes, and arguably better than degree -- constructive, full-support, single-temperature, and the more faithful proxy of gate cost. As a solvable thermodynamic limit: not with this document's tools. The rank identity was load-bearing, and it is exactly the price paid for fidelity to $X$: statistics with linear level sets are max-type and blind to accumulation, statistics that see accumulation have Hamming-ball level sets and turn block entropies into weight-enumerator problems. The weight prior sits at the best available compromise: every exact ingredient of the finite law survives (area, monotonicity, bias structure, an explicit block law), the limiting curve is pinned to a single conjectured threshold at $h_2(\theta)$, and the finite-$n$ evidence points at it -- but the step itself awaits a theorem that this document can only name.

### 8.7 The tractability ladder

How far can the complexity statistic be varied before the TDL is
lost? Level sets $\{\kappa\le c\}$ nest for free; what the rank
identity needs is each level set closed under XOR. If every level set
is a subspace, they form a flag $V_0\subset V_1\subset\cdots$ and
$\kappa(\rho)=\min\{c:\rho\in V_c\}$ -- after a change of basis,
the position of the last monomial present in some fixed order. The
fully tractable statistics are therefore *exactly* the max-index
family: not an accident, a characterization. This sorts every
candidate onto one of three rungs.

1. **Flag statistics** (degree, or any monomial order): classes are
   codes, entropy is rank, the finite law evaluates in one line, and
   with the capacity input the step is a theorem. Max-type, hence
   blind to accumulation and nearly uncorrelated with $X$.
2. **Additive statistics** ($\mathrm{wt}=\sum_S\rho_S$; sum of
   degrees $\sum_S|S|\rho_S$; any energy $\sum_S w(S)\rho_S$): the
   Gibbs prior factorizes into independent Bernoulli coefficients
   $\theta_S=1/(1+e^{\lambda w(S)})$, and the whole chapter survives:
   exact total entropy $\sum_S h_2(\theta_S)$, exact area, exact
   Fourier block law with $\prod_S\varepsilon_S$ per span vector,
   monotone increments. For *nondegenerate* biases the no-room argument pins the limit
   to one conjectured step at
   $R=\lim\,2^{-n}\sum_S h_2(\theta_S)$; degenerate patterns
   are richer (8.8). For
   the sum of degrees the temperature must scale, $\lambda_n=2c/n$
   (at fixed $\lambda$ the typical coins freeze and $R\to0$), giving
   $R\to h_2(1/(1+e^c))$: one more single-dial step. Hybrids -- a
   flag class sampled first, an additive tilt within it -- stay on
   this rung, Fourier inside the class plus the $O(\log n)$ label.
3. **Everything else** (ratios such as the average degree of the
   present monomials, maxima of sums, weight-and-degree combinations
   that respect neither structure): even the product form is gone.
   Only the generic theorems remain -- submodularity, boundedness,
   the exact area, Helly subsequences -- which guarantee that a
   limiting profile exists along subsequences and identify nothing
   about it.

Descending the ladder trades provability for fidelity to circuit
cost, and apparently nothing else: every rung's conjectured curves
lie in the same step-and-mixture family, so the phenomenology of
macroscopic leverage looks universal across complexity measures even
where the proofs give out.

### 8.8 Rung two examined: the missing theorem, and the shapes

What exactly is missing, and surprisingly little else. Exact at every
$n$: monotone increments, $\gamma\le h_0^{(n)}\to1$, and the area
$\int\gamma=R$. Helly gives subsequential limits for free. The one
open ingredient is achievability,

> for every fixed $t$ above the pattern's reconstruction threshold,
> $\lfloor tQ\rfloor$ random answers determine a fresh answer with
> probability tending to one, and the threshold equals the counting
> bound $R$;

granted that, the no-room argument forces $\gamma=1$ below $R$, the
limit is unique, and full-sequence convergence follows -- converse,
uniqueness and convergence all come free. There is also a sharp
reason achievability is harder than on rung one: rank arguments
provably give nothing. Any $\ell<Q$ columns of the invertible
$Z^{\otimes n}$ are linearly independent, so no answer is ever
exactly determined at finite $n$: $\gamma_{n,\ell}>0$ strictly, all
deduction is soft Bayesian decoding of the bias, and the step's flat
tail is an asymptotic collapse of soft uncertainty rather than a
literal determination -- which is also why the finite-$n$ leverage in
the figure hovers smoothly instead of stepping.

As for the shapes: the universal constraints are only that $\gamma$
be nonincreasing with $\gamma(0^+)\le1$ and $\int_0^1\gamma=R$,
and the step conjecture is *not* general. Freeze every monomial
containing $b_0$ ($\theta_S=0$) and leave the rest fair
($\theta_S=\tfrac12$): a legitimate product prior, but $\psi(q)$
then ignores the last bit of $q$, questions pair into equality
cliques, and the TDL is $\gamma(t)=1-t$ with $L\equiv2$ -- smooth,
no step. The conjectured dictionary:

- **homogeneous nondegenerate biases** (all $\theta_S$ in a band
  away from $0$): the pure step at $R$ -- the missing theorem's home;
- **frozen patterns** ($\theta_S\in\{0,\tfrac12\}$): the EXIT
  profile of the unfrozen monomial code -- rung two contains rung one
  as its extreme-bias boundary, cliques and parities included;
- **graded biases** (sum of degrees at $\lambda_n\sim1/n$):
  interpolations between the two.

Rung two therefore adds no shapes beyond the universal monotone
family -- nothing can, that being the layer-cake closure -- but it
realizes them by bias patterns instead of class mixtures, at the
price of one missing achievability theorem per pattern.

### 8.9 Punishing degree hard: a provable corner of rung two

The freedom in $w$ rescues provability for a large subfamily, with
no new coding theorem. Take
$$
w(S)=0\ \text{ for } |S|\le d^*,
\qquad
w(S)\ge n\ \text{ for } |S|>d^*,
\qquad d^*=\tfrac n2+\tfrac c2\sqrt n:
\tag{8.5}
$$
coins below the threshold are exactly fair, coins above are nearly
frozen ($\theta_S\le e^{-n}$), the prior has full support at every
$n$ with strict degree-Occam, and the table splits into independent
parts, $\psi=\psi_{\mathrm{RM}}\oplus\psi_{\rm junk}$ with
$H(\rho_{\rm junk})\le Q\,h_2(e^{-n})=o(Q)$. Two elementary
sandwiches then finish it. Subextensive additive junk is invisible:
for independent $A,B$,
$$
H(A)\;\le\;H(A\oplus B)\;\le\;H(A)+H(B),
\qquad\text{so}\qquad
\big|G_{n,\ell}-G^{\mathrm{RM}}_{n,\ell}\big|\le o(Q)
\ \text{ uniformly.}
\tag{8.6}
$$
And near-fair coins are as good as fair: the deficit is the KL
divergence $D(P\Vert U)=\sum_S(1-h_2(\theta_S))$, data processing
pushes it through any projection, and
$H_U(\psi(\mathcal S))-H_P(\psi(\mathcal S))
=D(P_{\mathcal S}\Vert U_{\mathcal S})\le D(P\Vert U)$ exactly.
Both sandwiches give $g(t)$ equal to the Reed--Muller prior's, the
concavity lemma upgrades that to $\gamma(t)$, and the capacity
theorem -- already imported -- gives the step:
$$
\gamma(t)=\theta_{\rm H}\big(\Phi(c)-t\big),
\qquad
L(t)=\begin{cases}1,&t<\Phi(c),\\ 1/\Phi(c),&t>\Phi(c),
\end{cases}
\tag{8.7}
$$
now for a full-support, single-formula additive prior: the one thing
rung one could not offer without a lapse channel.

The boundary of this provable corner is sharp, and it is the
hardness of the punishment. Pure linear punishment $w(S)=\beta|S|$
provably cannot satisfy both sides: fairness of the typical low coins
needs $\beta\,n/2\to0$, freezing of the high coins needs
$\beta\,n\gtrsim n\ln2$ -- a contradiction -- so linear $w$
produces graded biases across the CLT window, which is exactly the
open case of 8.8. Provability requires the fair-to-frozen transition
to cross within $o(\sqrt n)$ degrees, so that the transition coins
number $o(Q)$: punishing high degrees \emph{hard}, harder than
linearly, is precisely what buys the limit. Within the corner, the
fair set need not be a degree prefix: any monomial class whose code
has a known EXIT limit works the same way, and mixtures over $w$
recover the layer-cake family as before.

**Does $w(S)=|S|^2$ qualify?** No, on either reading, and checking it
turns the criterion into a formula. Unscaled, quadratic punishment is
too harsh everywhere: only the $O(n^2)$ coins of degree $\le2$ stay
alive, the total entropy is subextensive, the rate vanishes, and the
TDL degenerates into a pure boundary layer -- the junta failure in
Gibbs clothing. Scaled so the crossover sits in the CLT window
($\lambda_n\approx4/n^2$), the problem inverts to flatness: the
climb from $w\approx1$ to the freezing scale $w\approx n$ spans
$k^*\,(n^{1/p}-1)\gtrsim k^*(\ln n)/p$ degrees, which at $p=2$ is
$\sim n^{3/2}$, wider than the whole degree range -- the high coins
never freeze and one lands in the graded-bias open case of 8.8. No
fixed power works: the width condition demands
$p\gg\sqrt n\,\log n$, at which point the power law is a
threshold in disguise. The provable corner is genuinely
threshold-shaped: $w$ must climb by $\Omega(n)$ within $o(\sqrt n)$
degrees of the crossover.




# Appendices

Appendices A--F derive the block formulas used above in a common order:

1. specify the probability law inside one block;
2. compute its subset entropies $H_\kappa(j)$;
3. take differences $\eta_\kappa(i)=H_\kappa(i+1)-H_\kappa(i)$;
4. insert those differences into the Bernstein profile;
5. integrate the profile and substitute it into the leverage law.

Appendix G discusses the inverse problem, and Appendix H collects endpoint and boundary-layer consequences of the general limit.

## A. The general block calculation

### A.1 Exact mean block entropy at finite $n$

Let $V\in\mathcal P_n$ be a block. Define the entropy averaged over all $j$-subsets of that block:

$$
H_V(j)
:=\binom{|V|}{j}^{-1}
\sum_{\substack{J\subseteq V\\|J|=j}}H\bigl(\psi(J)\bigr).
\tag{A.1}
$$

When the block is entropy homogeneous, every term in this average is equal.

If $T$ is a uniformly random $\ell$-subset of $Q$, then its intersection count with $V$ is hypergeometric:

$$
\Pr(|T\cap V|=j)
=\frac{\binom{|V|}{j}\binom{|Q|-|V|}{\ell-j}}
       {\binom{|Q|}{\ell}}.
\tag{A.2}
$$

Different blocks have dependent intersection counts because their total is $\ell$, but entropy adds across independent blocks and expectation is linear. Therefore

$$
\boxed{
G_{n,\ell}
=\sum_{V\in\mathcal P_n}\sum_{j=0}^{|V|}
\frac{\binom{|V|}{j}\binom{|Q|-|V|}{\ell-j}}
     {\binom{|Q|}{\ell}}\,
H_V(j).}
\tag{A.3}
$$

This formula is exact and contains no large-$n$ approximation.

### A.2 Exact conditional-entropy increments

Choose a fresh question $q$ uniformly, then choose $\ell$ of the remaining $|Q|-1$ questions. Conditional on $q$ belonging to a type-$\kappa$ block of size $r_\kappa$, the probability of seeing exactly $i$ already-asked partners is

$$
\Pr(\text{exactly }i\text{ partners})
=\frac{\binom{r_\kappa-1}{i}\binom{|Q|-r_\kappa}{\ell-i}}
       {\binom{|Q|-1}{\ell}}.
\tag{A.4}
$$

Only those partners affect the fresh answer because blocks are independent. Define

$$
\eta_\kappa(i)=H_\kappa(i+1)-H_\kappa(i).
\tag{A.5}
$$

Choosing an $i$-subset and then a fresh coordinate uniformly shows that $\eta_\kappa(i)$ is the corresponding mean conditional entropy, even without entropy homogeneity. If $w_{\kappa,n}$ is the fraction of questions in type $\kappa$, then, for $0\le \ell<|Q|$,

$$
\boxed{
\gamma_{n,\ell}
=G_{n,\ell+1}-G_{n,\ell}
=\sum_\kappa w_{\kappa,n}\sum_{i=0}^{r_\kappa-1}
\frac{\binom{r_\kappa-1}{i}\binom{|Q|-r_\kappa}{\ell-i}}
     {\binom{|Q|-1}{\ell}}\,
\eta_\kappa(i).}
\tag{A.6}
$$

### A.3 Hypergeometric to binomial

Fix $r$ and $i$, set $\ell=\lfloor x|Q|\rfloor$, and expose a particular set of $i$ partners that are selected and $r-1-i$ that are not. Its probability is a product of $r-1$ factors of the form

$$
\frac{\ell+O(r)}{|Q|+O(r)}
\quad\text{or}\quad
\frac{|Q|-\ell+O(r)}{|Q|+O(r)}.
$$

Each selected factor tends to $x$, and each unselected factor tends to $1-x$. There are $\binom{r-1}{i}$ choices, so

$$
\Pr(\text{exactly }i\text{ partners})\longrightarrow
\binom{r-1}{i}x^i(1-x)^{r-1-i}.
\tag{A.7}
$$

The total-variation error is $O(r^2/|Q|)$ for fixed $r$. Thus (A.6) tends to

$$
\gamma(x)
=\sum_\kappa w_\kappa\sum_{i=0}^{r_\kappa-1}
\beta_{r_\kappa-1,i}(x)\eta_\kappa(i).
\tag{A.8}
$$

### A.4 Integrating a Bernstein profile

Direct integration of one basis function gives

$$
\int_0^x \beta_{r-1,i}(t)\,dt
=\binom{r-1}{i}\int_0^x t^i(1-t)^{r-1-i}\,dt,
\qquad
\int_0^1 \beta_{r-1,i}(t)\,dt=\frac1r.
\tag{A.9}
$$

Therefore

$$
\boxed{
g(x)
=\sum_\kappa w_\kappa
\sum_{i=0}^{r_\kappa-1}\eta_\kappa(i)
\binom{r_\kappa-1}{i}
\int_0^x t^i(1-t)^{r_\kappa-1-i}\,dt.}
\tag{A.10}
$$

At the end of the run,

$$
g(1)
=\sum_\kappa\frac{w_\kappa}{r_\kappa}\sum_i\eta_\kappa(i)
=\sum_\kappa\frac{w_\kappa}{r_\kappa}H_\kappa(r_\kappa).
\tag{A.11}
$$

This is exactly the prior entropy per question: there are $w_\kappa|Q|/r_\kappa$ independent type-$\kappa$ blocks, each carrying $H_\kappa(r_\kappa)$ bits.

### A.5 MDS mixtures as Bernstein coefficient synthesizers

Fix a common block length $r$ over $\mathbb F_{2^m}$, with $2^m\ge r$ so that $[r,k]_{2^m}$ Reed–Solomon codes exist for $k=1,\ldots,r$. Normalize entropies by $m$. A dimension-$k$ block contributes coefficient $1$ at indices $i<k$ and $0$ afterward. If dimension $k$ is used with question fraction $w_k$, the mixed normalized entropy coefficients are

$$
\eta_{\rm target}(i)=\sum_{k=i+1}^{r}w_k,
\qquad 0\le i\le r-1.
\tag{A.12}
$$

Conversely, take any nonincreasing sequence

$$
1\ge \eta_{\rm target}(0)\ge \eta_{\rm target}(1)\ge\cdots
\ge \eta_{\rm target}(r-1)\ge0
$$

and set $\eta_{\rm target}(r)=0$. The choice

$$
w_k=\eta_{\rm target}(k-1)-\eta_{\rm target}(k)
\qquad(1\le k\le r)
\tag{A.13}
$$

is nonnegative and reproduces the sequence through (A.12). Any slack $1-\eta_{\rm target}(0)$ can be assigned to frozen zero-entropy blocks.

Bernstein's approximation theorem then approximates a continuous nonincreasing normalized target profile $\gamma_{\rm target}:[0,1]\to[0,1]$ by taking $\eta_{\rm target}(i)=\gamma_{\rm target}(i/(r-1))$. This establishes a profile-synthesis result only under the accompanying realization conditions: $r^2/|Q|\to0$ along a diagonal limit, $2^m\ge r$, and divisibility leftovers must have vanishing density. Thus it is not a completeness theorem at fixed $m$.

## B. Cliques

### B.1 Block law and entropy vector

Let $V=\{q_1,\ldots,q_r\}\in\mathcal P_n$. Draw an $m$-bit shared value $Z_V\sim\nu_V$, and set

$$
\psi(q)=Z_V\qquad(q\in V).
\tag{B.1}
$$

The block distribution is

$$
\Pr\bigl(\psi(V)=y_V\bigr)
=\nu_V(y_{q_1})\mathbf 1[y_{q_1}=\cdots=y_{q_r}].
\tag{B.2}
$$

Let $h_V:=H(\nu_V)$. Observing no coordinate has zero entropy; observing at least one reveals $Z_V$, after which additional coordinates add nothing:

$$
H_V(0)=0,
\qquad
H_V(j)=h_V\quad(1\le j\le r).
\tag{B.3}
$$

Thus

$$
\eta_V(0)=h_V,
\qquad
\eta_V(i)=0\quad(i\ge1).
\tag{B.4}
$$

### B.2 Profile, integral, and leverage

For a common size $r$, let $h_{\rm clique}$ be the question-weighted mean of $h_V$. Only the $i=0$ Bernstein term survives:

$$
\gamma(x)=h_{\rm clique}(1-x)^{r-1}.
\tag{B.5}
$$

Integrating,

$$
g(x)
=h_{\rm clique}\int_0^x(1-t)^{r-1}\,dt
=\frac{h_{\rm clique}}{r}\big[1-(1-x)^r\big].
\tag{B.6}
$$

The initial marginal entropy is $h_0=h_{\rm clique}$. The destroyed density is

$$
h_0-(1-x)\gamma(x)
=h_{\rm clique}\big[1-(1-x)^r\big]
=r\,g(x).
\tag{B.7}
$$

Therefore

$$
L(x)=r.
\tag{B.8}
$$

For example, one four-question clique with a fair binary value has prior mass $1/2$ on $0000$ and $1/2$ on $1111$. Its profile is $(1-x)^3$, its received density is $[1-(1-x)^4]/4$, and its leverage is $4$ at every stage.

### B.3 Why the result is exact at finite size

A block is learned if and only if it has been touched at least once. Its probability of remaining untouched after a random $\ell$-subset is

$$
\frac{\binom{|Q|-r}{\ell}}{\binom{|Q|}{\ell}}.
\tag{B.9}
$$

For $|Q|/r$ equal-entropy blocks, expected received information is

$$
\frac{|Q|}{r}\,h_{\rm clique}
\left[1-\frac{\binom{|Q|-r}{\ell}}{\binom{|Q|}{\ell}}\right],
$$

whereas expected destroyed table entropy is

$$
|Q|\,h_{\rm clique}
\left[1-\frac{\binom{|Q|-r}{\ell}}{\binom{|Q|}{\ell}}\right].
$$

Their ratio is exactly $r$ for every non-null stage.

### B.4 Mixing clique sizes

Use the block-type notation of Section 4, and suppose a question fraction $w_\kappa$ lies in size-$r_\kappa$ cliques with a common entropy scale. Substituting each type's destroyed factor $1-(1-x)^{r_\kappa}$ and received factor $r_\kappa^{-1}[1-(1-x)^{r_\kappa}]$ gives

$$
L(x)
=\frac{\sum_\kappa w_\kappa[1-(1-x)^{r_\kappa}]}
       {\sum_\kappa(w_\kappa/r_\kappa)[1-(1-x)^{r_\kappa}]}.
\tag{B.10}
$$

At the two endpoints,

$$
L(0^+)=\sum_\kappa w_\kappa r_\kappa,
\qquad
L(1)=\left(\sum_\kappa\frac{w_\kappa}{r_\kappa}\right)^{-1}.
\tag{B.11}
$$

These are the arithmetic and harmonic means. A mixture of flat clique curves is generally a decreasing curve, not another constant.

If type-$\kappa$ cliques instead have mean entropy $h_\kappa$, the entropy-weighted endpoints are

$$
L(0^+)
=\frac{\sum_\kappa w_\kappa h_\kappa r_\kappa}
       {\sum_\kappa w_\kappa h_\kappa},
\qquad
L(1)
=\frac{\sum_\kappa w_\kappa h_\kappa}
       {\sum_\kappa w_\kappa h_\kappa/r_\kappa}.
\tag{B.11a}
$$

### B.5 Noisy cliques

For binary answers, draw the shared bit $Z\sim\nu$ and let

$$
\psi(q_j)=Z\oplus E_j,
\qquad
E_j\stackrel{\rm iid}{\sim}\operatorname{Bernoulli}(\delta).
\tag{B.12}
$$

Let $J_i:=\{q_1,\ldots,q_i\}$. After observing $y_{J_i}=\psi(J_i)$, the posterior on the shared value is

$$
P_i(z\mid y_{J_i})
\propto
\nu(z)\prod_{j=1}^i
\big[(1-\delta)\mathbf1(y_{q_j}=z)+\delta\mathbf1(y_{q_j}\ne z)\big].
\tag{B.13}
$$

The next-one probability is

$$
\Pr\bigl(\psi(q_{i+1})=1\mid\psi(J_i)=y_{J_i}\bigr)
=\delta+(1-2\delta)P_i(1\mid y_{J_i}),
\tag{B.14}
$$

so

$$
\eta_\delta(i)
=\sum_{y_{J_i}\in\{0,1\}^{i}}
\Pr\bigl(\psi(J_i)=y_{J_i}\bigr)
h_2\!\left(
\delta+(1-2\delta)P_i(1\mid y_{J_i})
\right).
\tag{B.15}
$$

For every $0<\delta<1$, all binary strings have positive probability. As $\delta\to0$, (B.15) tends to (B.4) term by term. At fixed nontrivial noise, the exact $L\equiv r>1$ law is lost; at $\delta=1/2$, the output is iid fair and the different flat law $L\equiv1$ results.

## C. Uniform, tilted, and noisy parity

### C.1 Uniform parity

Let $V=\{q_1,\ldots,q_r\}\in\mathcal P_n$. Fix $\sigma\in\{0,1\}$ and choose the binary answers in $V$ uniformly subject to

$$
\bigoplus_{q\in V}\psi(q)=\sigma.
\tag{C.1}
$$

There are $2^{r-1}$ admissible strings. Fix any $j\le r-1$ coordinates and any values for them. The remaining $r-j$ coordinates obey one parity equation and therefore have $2^{r-j-1}$ completions, independent of the fixed values. Every proper subset is consequently uniform:

$$
H_V(j)=j\quad(0\le j\le r-1),
\qquad
H_V(r)=r-1.
\tag{C.2}
$$

Thus

$$
\eta(i)=1\quad(0\le i\le r-2),
\qquad
\eta(r-1)=0.
\tag{C.3}
$$

For $m$-bit answers, the same formulas acquire a factor $m$ if parity is imposed independently on each bit or as one additive equation over $\mathbb F_{2^m}$.

### C.2 Profile and leverage

The Bernstein basis sums to one. Removing only the final term gives

$$
\gamma(x)
=\sum_{i=0}^{r-2}\beta_{r-1,i}(x)
=1-x^{r-1}.
\tag{C.4}
$$

Hence

$$
g(x)=x-\frac{x^r}{r}.
\tag{C.5}
$$

The destroyed density is

$$
1-(1-x)(1-x^{r-1})
=x+(1-x)x^{r-1},
$$

so

$$
L(x)
=\frac{x+(1-x)x^{r-1}}
       {x-x^r/r}.
\tag{C.6}
$$

For $r\ge3$, direct differentiation yields

$$
L'(x)
=\frac{x^{r-1}(1-x)
\left[r(r-2)-\sum_{j=1}^{r-2}x^j\right]}
{r\left(x-x^r/r\right)^2}
\ge0.
\tag{C.7}
$$

Moreover,

$$
L(0^+)=1,
\qquad
L(1)=\frac r{r-1}.
\tag{C.8}
$$

For $r=2$, (C.6) simplifies to $L\equiv2$.

For the smallest genuinely different example, take one odd-parity block with $r=4$. Its prior is uniform on the eight odd-weight strings. Then

$$
\gamma(x)=1-x^3,
\qquad
g(x)=x-\frac{x^4}{4},
\qquad
L(x)=\frac{x+(1-x)x^3}{x-x^4/4}.
$$

The cumulative deduced-information density is

$$
1-(1-x)(1-x^{r-1})
-\left(x-\frac{x^r}{r}\right),
$$

and its derivative is

$$
-(1-x)\gamma'(x)=(r-1)x^{r-2}(1-x).
\tag{C.9}
$$

This is proportional to a $\operatorname{Beta}(r-1,2)$ density. It is also the density of the stage at which a block's penultimate queried member determines its final member, divided by $r$ questions per block.

### C.3 Tilted parity: exact subset probabilities

For $0<\theta<1$, start with iid $\operatorname{Bernoulli}(\theta)$ bits and condition on (C.1). Let

$$
\varepsilon=1-2\theta.
$$

For $v$ iid bits, the probability of parity $\sigma_{\rm rem}$ is

$$
\mathcal Z_{v,\sigma_{\rm rem}}
=\frac{1+(-1)^{\sigma_{\rm rem}}\varepsilon^v}{2}.
\tag{C.10}
$$

The normalized tilted-coset law is

$$
\Pr_\sigma\bigl(\psi(V)=y_V\bigr)
=\frac{\theta^{|y_V|}(1-\theta)^{r-|y_V|}
\mathbf1[\oplus_{q\in V}y_q=\sigma]}
{\mathcal Z_{r,\sigma}}.
\tag{C.11}
$$

Take a particular realized tuple $y_J$ on $J\subset V$, with $|J|=j$ and Hamming weight $z:=|y_J|$. Its unobserved coordinates must have parity $\sigma\oplus(z\bmod2)$. Summing them out gives

$$
\Pr_\sigma\bigl(\psi(J)=y_J\bigr)
=\theta^z(1-\theta)^{j-z}
\frac{1+(-1)^{\sigma\oplus(z\bmod2)}\varepsilon^{r-j}}
     {1+(-1)^\sigma\varepsilon^r}.
\tag{C.12}
$$

Write the right-hand side of (C.12), for any particular weight-$z$ tuple, as $\chi_\sigma(j,z)$. The subset entropy is then the finite sum

$$
H_\sigma(j)
=-\sum_{z=0}^j\binom jz
\chi_\sigma(j,z)\log_2\chi_\sigma(j,z),
\tag{C.13}
$$

The coefficients are then exactly

$$
\eta(i)=H_\sigma(i+1)-H_\sigma(i).
\tag{C.14}
$$

### C.4 Tilted parity: predictive form

The same result can be organized as a conditional prediction. Suppose the questions in $J\subset V$ have been seen, with $|J|=i$; then $v:=|V\setminus J|=r-i$ bits remain. If their required parity is $\sigma_{\rm rem}$, then for any $q\in V\setminus J$,

$$
P_{\rm next}(v,\sigma_{\rm rem})
:=\Pr\!\left(
\psi(q)=1\;\middle|\;
\bigoplus_{q'\in V\setminus J}\psi(q')=\sigma_{\rm rem}
\right)
=\frac{\theta[1-(-1)^{\sigma_{\rm rem}}\varepsilon^{v-1}]}
       {1+(-1)^{\sigma_{\rm rem}}\varepsilon^v}.
\tag{C.15}
$$

Let $\sigma_{\rm seen}$ be the parity of the $i$ seen bits. Conditional on total parity $\sigma$,

$$
\Pr(\sigma_{\rm seen}=z\mid\sigma)
=\frac{
[1+(-1)^z\varepsilon^i]
[1+(-1)^{\sigma\oplus z}\varepsilon^v]
}{
2[1+(-1)^\sigma\varepsilon^r]
}.
\tag{C.16}
$$

Thus

$$
\boxed{
\eta(i)
=\sum_{z=0}^1
\Pr(\sigma_{\rm seen}=z\mid\sigma)\,
h_2\!\left(P_{\rm next}(v,\sigma\oplus z)\right).}
\tag{C.17}
$$

When $\theta=1/2$, $\varepsilon=0$ and (C.17) reduces to (C.3). When $v=1$, the remaining bit is fixed and $\eta(r-1)=0$. For $r=4,\theta=0.3,\sigma=1$, evaluation gives (5.8).

### C.5 Uniform parity followed by bit-flip noise

Now return to the **uniform** parity coset and flip every bit independently with probability $\delta$. The parity changes precisely when an odd number of flips occurs. Let

$$
P_{\rm odd}(r,\delta)
:=\Pr(\text{odd number of flips})
=\frac{1-(1-2\delta)^r}{2}.
\tag{C.18}
$$

Convolution with the flip channel produces a mixture of the two uniform parity cosets, with weights $1-P_{\rm odd}(r,\delta)$ and $P_{\rm odd}(r,\delta)$. Any $r-1$ coordinates remain uniform, so

$$
\eta_\delta(i)=1\quad(i\le r-2).
\tag{C.19}
$$

Given $r-1$ coordinates, uncertainty in the final bit is exactly uncertainty in the resulting parity label:

$$
\eta_\delta(r-1)=h_2(P_{\rm odd}(r,\delta)).
\tag{C.20}
$$

Therefore

$$
\boxed{
\gamma_\delta(x)
=1-\big[1-h_2(P_{\rm odd}(r,\delta))\big]x^{r-1}.}
\tag{C.21}
$$

For fixed $r$, any $\delta_n\to0$ recovers the hard profile. If the block size grows as $r(n)\to\infty$, one instead needs $P_{\rm odd}(r(n),\delta_n)\to0$, which for small noise requires approximately $r(n)\delta_n\to0$. Equations (C.19)--(C.21) rely on a uniform starting coset; tilted parity followed by noise remains solvable by finite sums but is not uniform within each resulting coset.

## D. MDS and Reed–Solomon blocks

### D.1 Block construction

Let $V=\{q_1,\ldots,q_r\}\in\mathcal P_n$ and work over $\mathbb F_{2^m}$. Choose $r$ distinct evaluation points $\alpha_1,\ldots,\alpha_r$, which requires $r\le2^m$. Draw a polynomial

$$
f(z)=d_0+d_1z+\cdots+d_{k-1}z^{k-1}
$$

uniformly from the $2^{mk}$ polynomials of degree below $k$, and set

$$
\psi(q_j)=f(\alpha_j).
\tag{D.1}
$$

This is a uniform Reed–Solomon codeword.

Two elementary interpolation facts determine all entropies.

1. Values at any $k$ distinct points determine $f$, and hence all $r$ answers.
2. Values at any $j\le k$ points are uniform on $\mathbb F_{2^m}^j$. To see this, prescribe those $j$ values, freely choose values at another $k-j$ points, and interpolate. Exactly $2^{m(k-j)}$ polynomials realize every prescription.

Therefore

$$
H_{\rm MDS}(j)=m\min(j,k).
\tag{D.2}
$$

The increment vector is

$$
\eta(i)=
\begin{cases}
m,&0\le i\le k-1,\\
0,&k\le i\le r-1.
\end{cases}
\tag{D.3}
$$

The same entropy law holds for any uniform coset of an $[r,k]_{2^m}$ MDS code. Repetition ($k=1$) and single-parity-check ($k=r-1$) codes give the clique and parity entropy laws over any alphabet, although for small fields and large $r$ they are not obtained from $r$ distinct Reed–Solomon evaluation points.

### D.2 Profile and code rate

Define the binomial lower-tail polynomial

$$
\tau_{r,k}(x)
:=\sum_{i=0}^{k-1}
\binom{r-1}{i}x^i(1-x)^{r-1-i}.
\tag{D.4}
$$

Inserting (D.3) into the Bernstein law gives $\gamma(x)=m\tau_{r,k}(x)$. Its integral is

$$
\begin{aligned}
\int_0^1\tau_{r,k}(x)\,dx
&=\sum_{i=0}^{k-1}\binom{r-1}{i}
\int_0^1x^i(1-x)^{r-1-i}\,dx\\
&=\sum_{i=0}^{k-1}\binom{r-1}{i}
\frac{i!(r-1-i)!}{r!}\\
&=\sum_{i=0}^{k-1}\frac1r
=\rho,
\qquad \rho:=\frac kr.
\end{aligned}
\tag{D.5}
$$

The total entropy per block is $km$, so the same result also follows by dividing $km$ by $r$ questions.

For a pure code prior,

$$
g(x)=m\int_0^x\tau_{r,k}(t)\,dt,
\tag{D.6}
$$

and

$$
\boxed{
L_{\rm code}(x)
=\frac{1-(1-x)\tau_{r,k}(x)}
       {\int_0^x\tau_{r,k}(t)\,dt}.}
\tag{D.7}
$$

In particular,

$$
L_{\rm code}(1)=\frac1\rho=\frac rk.
\tag{D.8}
$$

For $2^m=4$, $r=4$, and $k=2$, a random affine polynomial over $\mathbb F_4$ gives $4^2=16$ codewords. Any two answers determine the other two. In this case

$$
\tau_{4,2}(x)
=(1-x)^3+3x(1-x)^2
=1-3x^2+2x^3,
$$

which is the threshold component used in the explicit non-monotonic example (4.12).

### D.3 The secondary large-block limit

Now take $r\to\infty$ with $k/r\to\rho$, **after** taking $n\to\infty$ at fixed $r$. By the law of large numbers, a $\operatorname{Bin}(r-1,x)$ count divided by $r-1$ converges in probability to $x$.

Away from $x=\rho$,

$$
\tau_{r,k}(x)
\longrightarrow\mathbf1[x<\rho].
\tag{D.9}
$$

Hence

$$
L_{\rm code}(x)\longrightarrow
\begin{cases}
1,&x<\rho,\\[2mm]
1/\rho,&x>\rho.
\end{cases}
\tag{D.10}
$$

At finite $r$, the transition is smooth and $\gamma(x)>0$ for every $x<1$. The jump appears only in the secondary limit. A Reed–Solomon realization of this limit requires $m\ge\log_2r$ to grow.

### D.4 Dilution with fresh questions

Let a fraction $w$ of questions lie in code blocks and a fraction $1-w$ be independent uniform answers. Their normalized profiles are $\tau_{r,k}$ and $1$, so

$$
\frac{\gamma(x)}m=w\tau_{r,k}(x)+(1-w),
\tag{D.11}
$$

$$
\frac{g(x)}m
=w\int_0^x\tau_{r,k}(t)\,dt+(1-w)x.
\tag{D.12}
$$

Since the initial marginal entropy is $m$,

$$
L(x)
=\frac{
w[1-(1-x)\tau_{r,k}(x)]+(1-w)x
}{
w\int_0^x\tau_{r,k}(t)\,dt+(1-w)x
}.
\tag{D.13}
$$

For $1\le k<r$, differentiate the binomial tail:

$$
\tau_{r,k}'(x)
=-(r-1)\binom{r-2}{k-1}
x^{k-1}(1-x)^{r-1-k}.
\tag{D.14}
$$

The derivative of the cumulative deduced-information density is

$$
\boxed{
-(1-x)\gamma'(x)
=wm(r-1)\binom{r-2}{k-1}
x^{k-1}(1-x)^{r-k}.}
\tag{D.15}
$$

This is proportional to a $\operatorname{Beta}(k,r-k+1)$ density. Its mode is $(k-1)/(r-1)$ when $k>1$, and its total mass is

$$
\int_0^1[-(1-x)\gamma'(x)]\,dx
=wm\left(1-\frac kr\right).
\tag{D.16}
$$

The release occurs near the code threshold. Fresh questions continue to add entropy at marginal leverage one afterward, which creates the declining side of the peak.

At $k=r$, the code is the full product space: $\tau_{r,r}\equiv1$ and the deduction density is identically zero.

### D.5 A lapsed MDS block

Let $\mathcal K+y\subset\mathbb F_{2^m}^r$ be an MDS coset, where $y$ is a fixed offset. For $0\le\delta\le1$, draw uniformly from the coset with probability $1-\delta$ and uniformly from all $2^{mr}$ answer strings with probability $\delta$:

$$
P_\delta(y_V)
=(1-\delta)2^{-mk}\mathbf1[y_V\in\mathcal K+y]
+\delta 2^{-mr}.
\tag{D.17}
$$

This has full support for every $0<\delta\le1$.

For $j\le k$, both mixture components project to the uniform distribution on $\mathbb F_{2^m}^j$, so

$$
H_\delta(j)=jm.
\tag{D.18}
$$

For $j\ge k$, the projection of the code has $2^{mk}$ consistent patterns. Each has probability

$$
(1-\delta)2^{-mk}+\delta 2^{-mj}
=2^{-mk}\big[(1-\delta)+\delta 2^{m(k-j)}\big].
\tag{D.19}
$$

The other $2^{mj}-2^{mk}$ patterns each have probability $\delta2^{-mj}$. Summing the two entropy contributions gives

$$
\boxed{
H_\delta(j)
=\big[(1-\delta)+\delta2^{m(k-j)}\big]
\left[km-\log_2\!\big((1-\delta)+\delta2^{m(k-j)}\big)\right]
+\delta(1-2^{m(k-j)})
\big(jm-\log_2\delta\big),
\qquad j\ge k.}
\tag{D.20}
$$

The exact increment vector is

$$
\eta_\delta(i)=H_\delta(i+1)-H_\delta(i).
\tag{D.21}
$$

It equals $m$ below threshold. Above threshold it declines toward a positive lapse tail; the exact finite differences should be used rather than replacing the whole tail immediately by $\delta m$.

## E. Local noise as a term-by-term perturbation

### E.1 A general channel construction

Using the fixed ordering of $V$ from Section 4.1, let $P_V$ be any law on $A^{|V|}$, and, for $0\le\delta\le1$, let $\Pr_\delta(y\mid a)$ be a full-support channel on the answer alphabet that tends to the identity as $\delta\to0$. Apply it independently to each coordinate of $V$:

$$
P_{V,\delta}(y_V)
=\sum_{a_V\in A^{|V|}}P_V(a_V)
\prod_{q\in V}\Pr_\delta(y_q\mid a_q).
\tag{E.1}
$$

Alternatively, use the lapse (5.14). The full map law remains a product over blocks:

$$
p_{\delta,j}:=\Pr_\delta(\psi=\phi_j)
=\prod_{V\in\mathcal P_n}
P_{V,\delta_{V,n}}\bigl(\phi_j(V)\bigr).
\tag{E.2}
$$

For block rates in $0<\delta_{V,n}\le1$, it has full support if each noisy block law does.

### E.2 Why vanishing local noise preserves the limit

For fixed $|V|$, the block probability simplex is finite-dimensional. As $\delta\to0$,

$$
\|P_{V,\delta}-P_V\|_{\rm TV}\longrightarrow0.
$$

Entropy is continuous on a finite alphabet, so every subset entropy and increment converges:

$$
H_{V,\delta}(j)\to H_V(j),
\qquad
\eta_{V,\delta}(i)\to\eta_V(i).
\tag{E.3}
$$

The finite hypergeometric weights—and their limiting Bernstein weights—are nonnegative and sum to one. Let $\gamma_{\rm noisy,n}$ and $\gamma_{\rm hard,n}$ denote the profiles computed from the noisy and hard block coefficients at size $n$. Hence

$$
\sup_{x\in[0,1]}
|\gamma_{\rm noisy,n}(x)-\gamma_{\rm hard,n}(x)|
\le
\sum_{V\in\mathcal P_n}\frac{|V|}{|Q|}
\max_i|\eta_{V,\delta_{V,n}}(i)-\eta_{V,0}(i)|
\longrightarrow0.
\tag{E.4}
$$

The integrals converge uniformly as well. Wherever the limiting received density is bounded away from zero, leverage converges. For fixed block sizes, continuity makes the last limit automatic if $\max_V\delta_{V,n}\to0$; the aggregate condition displayed above is the more general statement.

For varying block sizes, a uniform condition is needed. One sufficient aggregate condition is (5.17); simple model-specific conditions, such as $r(n)\delta_n\to0$ for parity blocks of size $r(n)$, can be sharper.

At fixed $\delta>0$, none of these perturbations disappears with $n$. The exact noisy profile remains the thermodynamic limit.

## F. Arbitrary mixtures and the corrected cocktail

### F.1 Symbolic mixture law

Partition the questions into independent component families. Component type $\kappa$ occupies fraction $w_\kappa$, has initial one-question entropy $h_{0,\kappa}$, macroscopic profile $\gamma_\kappa(x)$, and accumulated received density

$$
g_\kappa(x)=\int_0^x\gamma_\kappa(t)\,dt.
$$

Entropy adds across the independent components, so

$$
h_0=\sum_\kappa w_\kappa h_{0,\kappa},
\qquad
\gamma(x)=\sum_\kappa w_\kappa\gamma_\kappa(x),
\qquad
g(x)=\sum_\kappa w_\kappa g_\kappa(x).
\tag{F.1}
$$

For each component,

$$
h_{0,\kappa}-(1-x)\gamma_\kappa(x)=L_\kappa(x)g_\kappa(x).
$$

Summing gives

$$
\boxed{
L(x)
=\frac{\sum_\kappa w_\kappa g_\kappa(x)L_\kappa(x)}
       {\sum_\kappa w_\kappa g_\kappa(x)}.}
\tag{F.2}
$$

The mixture weights in leverage are therefore dynamic received-entropy shares $w_\kappa g_\kappa(x)$, not the fixed question shares $w_\kappa$.

### F.2 Coin component of the cocktail

For the two-coin component, let

$$
(\theta_1,\theta_2)=(0.2,0.7),
\qquad
(w_1,w_2)=\left(\frac14,\frac34\right),
$$

and let the same scalar latent bias govern all bits in this component. Define

$$
P_{s,z}
=\sum_{r=1}^2w_r\theta_r^z(1-\theta_r)^{s-z},
$$

$$
F_s
=-\sum_{z=0}^s\binom sz
P_{s,z}\log_2P_{s,z}.
\tag{F.3}
$$

For an $m=4$-bit answer, the initial entropy per bit is

$$
\frac{F_4}{4}
=\frac{3.793149840}{4}
=0.948287460.
\tag{F.4}
$$

It is not $h_2(0.575)=0.983708263$, because marginalizing the shared latent makes the four bits dependent. After a vanishing-fraction identification layer, the entropy per bit is

$$
h_{\rm cond,coin}
=\frac14h_2(0.2)+\frac34h_2(0.7)
=0.841450198.
\tag{F.5}
$$

### F.3 Lapsed-code and fresh components

For $2^m=16,r=16,k=4,\delta=0.1$, compute $H_\delta(j)$ from (D.18)--(D.20), define

$$
\eta_{\delta}(i)=H_\delta(i+1)-H_\delta(i),
$$

and normalize the code profile per bit:

$$
\gamma_{\rm lapse}(x)
=\frac14\sum_{i=0}^{15}
\beta_{15,i}(x)\eta_\delta(i).
\tag{F.6}
$$

Its integral is

$$
\int_0^1\gamma_{\rm lapse}(x)\,dx
=\frac{H_\delta(16)}{16\cdot4}
=0.332328056.
\tag{F.7}
$$

The fresh component consists of independent bits with bias $0.3$, so its per-bit profile and initial entropy are both $h_2(0.3)$.

### F.4 Assemble the cocktail

With question fractions $1/4,1/2,1/4$, the initial per-bit entropy is

$$
h_{\rm cocktail}
=\frac14\frac{F_4}{4}
+\frac12
+\frac14h_2(0.3)
=0.957394590.
\tag{F.8}
$$

For every $x>0$,

$$
\gamma_{\rm cocktail}(x)
=\frac14h_{\rm cond,coin}
+\frac12\gamma_{\rm lapse}(x)
+\frac14h_2(0.3),
\tag{F.9}
$$

and

$$
g_{\rm cocktail}(x)
=\frac14h_{\rm cond,coin}x
+\frac12\int_0^x\gamma_{\rm lapse}(t)\,dt
+\frac14h_2(0.3)x.
\tag{F.10}
$$

The scale factor $m$ cancels, leaving

$$
L(x)
=\frac{h_{\rm cocktail}-(1-x)\gamma_{\rm cocktail}(x)}
       {g_{\rm cocktail}(x)}.
\tag{F.11}
$$

At the origin, the profile's right limit is

$$
\gamma_{\rm cocktail}(0^+)
=\frac14(0.841450198)+\frac12+\frac14h_2(0.3)
=0.930685274.
$$

Thus (H.3) gives the exact singular coefficient

$$
\lim_{x\downarrow0}xL(x)
=\frac{h_{\rm cocktail}-\gamma_{\rm cocktail}(0^+)}
       {\gamma_{\rm cocktail}(0^+)}
\simeq0.02870.
\tag{F.12}
$$

At the endpoint,

$$
g_{\rm cocktail}(1)
=\frac14(0.841450198)
+\frac12(0.332328056)
+\frac14h_2(0.3)
=0.596849302,
$$

so

$$
L(1)
=\frac{0.957394590}{0.596849302}
\simeq1.60408.
\tag{F.13}
$$

Numerical evaluation of the same closed form gives the dip and peak quoted in Section 5.

## G. What can be inferred from an observed curve?

If $h_0$ and an ideal macroscopic leverage curve $L(x)$ are known, then $g'(x)=\gamma(x)$ and (2.15) imply the linear differential equation

$$
(1-x)g'(x)+L(x)g(x)=h_0,
\qquad
g(0)=0.
\tag{G.1}
$$

For an admissible $L$, solving (G.1) for the regular, finite, nonnegative solution recovers the entropy profile $\gamma=g'$. This qualification matters when $L$ has a $1/x$ singularity: $g(0)=0$ alone is then a singular initial condition rather than a standard nonsingular initial-value problem.

The inverse map does not recover a unique prior. Many microscopic distributions have the same subset-entropy profile, and finite data introduce further ambiguity. For a trained model, question selection, optimization dynamics, misspecification, and calibration also affect the measured curve. The defensible inference is therefore about broad timing classes:

- a $1/x$ term indicates information learned on a subextensive clock;
- a flat curve with $L>1$ is consistent with first-touch duplication, while $L\equiv1$ is also the independent baseline;
- a rising curve is consistent with late constraints;
- an interior peak is consistent with threshold release followed by continued reception.

This is useful structural evidence, but not an identification theorem for the real-world prior.

## H. Endpoints, jumps, and boundary layers

This appendix records the endpoint and discontinuity consequences of the macroscopic law (2.15).

At the end of the run,

$$
L(1)=\frac{h_0}{g(1)}
=\lim_{n\to\infty}\frac{|Q|G_{n,1}}{G_{n,|Q|}}.
\tag{H.1}
$$

At the origin there are two different cases. If $h_0=\gamma(0^+)>0$ and $\gamma$ is right-differentiable, first-order expansion gives

$$
L(0^+)
=1-\frac{\gamma'(0^+)}{\gamma(0^+)}.
\tag{H.2}
$$

If instead $h_0>\gamma(0^+)>0$, an initial boundary layer has disappeared into $x=0$, and

$$
L(x)
\sim\frac{h_0-\gamma(0^+)}{\gamma(0^+)}\frac1x
\qquad(x\downarrow0).
\tag{H.3}
$$

This is the source of the spike and coin-mixture hyperbolas.

A separate caveat concerns interior thresholds. If $\gamma$ merely reaches zero continuously, no instantaneous avalanche follows. A macroscopic jump occurs when

$$
\gamma(x_*^-)>\gamma(x_*^+),
\tag{H.4}
$$

especially when the profile jumps to zero. Then the remaining-entropy density drops by

$$
(1-x_*)\big[\gamma(x_*^-)-\gamma(x_*^+)\big],
$$

and cumulative leverage can jump as well. The local per-step leverage inside the shrinking transition layer may diverge; the cumulative $L(x)$ need not.

Finally, if $G_{n,|Q|}=o(|Q|)$, the normalization of Section 2 erases the entire received channel. One should then introduce a microscopic or mesoscopic time scale rather than conclude automatically that leverage has a meaningful infinite value.
