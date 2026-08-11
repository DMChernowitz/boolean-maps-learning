"""Exact finite-size figures for the coin-mixture chapter.

The binary latent-bias model has component parameters (theta_r, w_r).
For s observed bits, every particular string of weight j has probability

    a_{s,j} = sum_r w_r theta_r^j (1-theta_r)^(s-j),

so its exact entropy is

    F_s = -sum_j binom(s,j) a_{s,j} log2(a_{s,j}).

For an m-bit answer whose bits share the same scalar latent bias,

    G_l = F_(m l),           G_1 = F_m,
    h_cond,m = m sum_r w_r h(theta_r),
    c_m = (F_m - h_cond,m) / h_cond,m.

Outputs:
  figures/coin_mixture_leverage.png
      Two- and ten-component binary mixtures with the same F_1,
      h_cond,1, and c_1, but visibly different finite-n convergence.
  figures/coin_mixture_m_comparison.png
      The chapter's shared-scalar mixture at m=1 and m=2, using F_m rather
      than the incorrect m h(mean theta) initial entropy.
"""
from functools import lru_cache
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOG2 = math.log(2.0)


def h(theta):
    """Binary entropy in bits."""
    if theta <= 0.0 or theta >= 1.0:
        return 0.0
    return (-theta * math.log2(theta)
            - (1.0 - theta) * math.log2(1.0 - theta))


def normalized_coins(coins):
    """Return a hashable validated tuple of (theta, weight) pairs."""
    out = tuple((float(theta), float(weight)) for theta, weight in coins)
    if not out or any(not 0.0 < theta < 1.0 or weight <= 0.0
                      for theta, weight in out):
        raise ValueError("coin biases must lie in (0,1) and weights be positive")
    if not math.isclose(sum(weight for _, weight in out), 1.0,
                        rel_tol=0.0, abs_tol=2e-14):
        raise ValueError("coin weights must sum to one")
    return out


@lru_cache(maxsize=None)
def F_bits(s, coins):
    """Exact entropy F_s of s exchangeable bits, evaluated in log space."""
    if s < 0:
        raise ValueError("the number of bits must be nonnegative")
    coins = normalized_coins(coins)
    log_parameters = [(math.log(weight), math.log(theta),
                       math.log1p(-theta))
                      for theta, weight in coins]
    entropy = 0.0
    total_mass = 0.0
    for ones in range(s + 1):
        logs = [log_weight + ones * log_theta
                + (s - ones) * log_one_minus_theta
                for log_weight, log_theta, log_one_minus_theta
                in log_parameters]
        peak = max(logs)
        log_string_probability = (
            peak + math.log(sum(math.exp(value - peak) for value in logs))
        )
        log_multiplicity = (math.lgamma(s + 1)
                            - math.lgamma(ones + 1)
                            - math.lgamma(s - ones + 1))
        weight_class_mass = math.exp(log_multiplicity
                                     + log_string_probability)
        total_mass += weight_class_mass
        entropy -= weight_class_mass * log_string_probability / LOG2
    if not math.isclose(total_mass, 1.0, rel_tol=3e-11, abs_tol=3e-11):
        raise ArithmeticError(f"block probabilities sum to {total_mass}, not one")
    return entropy


def tdl_parameters(m, coins):
    """Return (F_m, h_cond,m, c_m) for the shared scalar latent model."""
    coins = normalized_coins(coins)
    initial_entropy = F_bits(m, coins)
    entropy_density = m * sum(weight * h(theta)
                              for theta, weight in coins)
    if entropy_density <= 0.0:
        raise ValueError("the thermodynamic entropy density must be positive")
    c_m = (initial_entropy - entropy_density) / entropy_density
    return initial_entropy, entropy_density, c_m


def L_curve(n, m, coins):
    """Exact finite-n cumulative leverage for |Q|=2^n m-bit questions."""
    coins = normalized_coins(coins)
    num_questions = 2**n
    Gs = [F_bits(m * ell, coins)
          for ell in range(num_questions + 1)]
    initial_matrix_entropy = num_questions * Gs[1]
    xs, leverages = [], []
    for ell in range(1, num_questions + 1):
        remaining = ((num_questions - ell) * (Gs[ell + 1] - Gs[ell])
                     if ell < num_questions else 0.0)
        xs.append(ell / num_questions)
        leverages.append((initial_matrix_entropy - remaining) / Gs[ell])
    return np.asarray(xs), np.asarray(leverages)


def matched_ten_component_mixture(target_c):
    """Equal-weight symmetric ten-coin mixture with prescribed binary c.

    Five distinct low biases and their complements make a deliberately
    broad ten-component directing measure.  Symmetry fixes mean(theta)=1/2
    and hence F_1=1.  Four low biases are fixed; bisection chooses the fifth
    so mean h(theta)=1/(1+target_c).  The broad support makes the finite-size
    approach visibly different from the matched two-component prior.
    """
    fixed_low_biases = (0.005, 0.04, 0.1, 0.2)
    target_entropy_density = 1.0 / (1.0 + target_c)
    required_entropy = (5.0 * target_entropy_density
                        - sum(h(theta) for theta in fixed_low_biases))
    low, high = 0.2, 0.499999
    if not h(low) < required_entropy < h(high):
        raise ValueError("target c is outside the matching family")
    for _ in range(100):
        middle = (low + high) / 2.0
        if h(middle) < required_entropy:
            low = middle
        else:
            high = middle
    fitted_bias = (low + high) / 2.0
    low_biases = fixed_low_biases + (fitted_bias,)
    biases = low_biases + tuple(1.0 - theta
                               for theta in reversed(low_biases))
    coins = normalized_coins(
        [(theta, 0.1) for theta in biases]
    )
    return coins, fitted_bias


def verify_curve(n, m, coins):
    """Check exact finite values against entropy and TDL bounds."""
    coins = normalized_coins(coins)
    initial_entropy, h_cond, c_m = tdl_parameters(m, coins)
    xs, exact = L_curve(n, m, coins)
    limit = 1.0 + c_m / xs
    if not np.all(np.isfinite(exact)):
        raise AssertionError("non-finite leverage in exact curve")
    # G_s = s*h_cond,1 + I(Z;Y_1^s), with 0 <= I <= H(Z).
    latent_entropy = -sum(weight * math.log2(weight) for _, weight in coins)
    bits = m * 2**n
    block_entropy = F_bits(bits, coins)
    bit_density = h_cond / m
    if not (bits * bit_density - 2e-9
            <= block_entropy
            <= bits * bit_density + latent_entropy + 2e-9):
        raise AssertionError("block entropy violates its latent-information bound")
    # I_{ell+1}-I_ell >= 0 makes every finite curve no larger than its TDL.
    if np.any(exact > limit + 2e-8):
        raise AssertionError("finite curve lies above its limiting hyperbola")
    if not initial_entropy > 0.0:
        raise AssertionError("initial answer entropy must be positive")
    return xs, exact


def plot_family_panel(ax, coins, m, title, n_values, colors, x_min=0.15):
    """Draw exact finite curves and their dashed thermodynamic limit."""
    initial_entropy, h_cond, c_m = tdl_parameters(m, coins)
    for n, color in zip(n_values, colors):
        xs, exact = verify_curve(n, m, coins)
        shown = xs >= x_min
        ax.plot(xs[shown], exact[shown], color=color, lw=1.45,
                label=f"exact $n={n}$")
    continuum_x = np.linspace(x_min, 1.0, 700)
    ax.plot(continuum_x, 1.0 + c_m / continuum_x,
            "--", color="#d84a3a", lw=1.8,
            label="TDL $1+c_m/x$")
    ax.set_title(title + "\n"
                 + f"$F_m={initial_entropy:.6f}$, "
                   f"$h_{{\\mathrm{{cond}},m}}={h_cond:.6f}$, "
                   f"$c_m={c_m:.6f}$")
    ax.set_xlabel("asked fraction $x=\\ell/|Q|$")
    ax.set_xlim(x_min, 1.0)
    ax.set_ylim(0.95, 9.0)
    ax.grid(alpha=0.25)
    return c_m


# Chapter mixture (3.15): two equally weighted, complementary biases.
TWO_COMPONENT = normalized_coins(((0.1, 0.5), (0.9, 0.5)))
TWO_F1, TWO_HCOND1, TWO_C1 = tdl_parameters(1, TWO_COMPONENT)
TEN_COMPONENT, TEN_FITTED_BIAS = matched_ten_component_mixture(TWO_C1)
TEN_F1, TEN_HCOND1, TEN_C1 = tdl_parameters(1, TEN_COMPONENT)
assert math.isclose(TWO_C1, TEN_C1, rel_tol=0.0, abs_tol=3e-15)
assert math.isclose(TWO_F1, TEN_F1, rel_tol=0.0, abs_tol=3e-15)
assert math.isclose(TWO_HCOND1, TEN_HCOND1, rel_tol=0.0, abs_tol=3e-15)
assert math.isclose(sum(theta * weight for theta, weight in TEN_COMPONENT),
                    0.5, rel_tol=0.0, abs_tol=2e-15)
assert math.isclose(
    sum(weight * h(theta) for theta, weight in TWO_COMPONENT),
    sum(weight * h(theta) for theta, weight in TEN_COMPONENT),
    rel_tol=0.0, abs_tol=3e-15)

# Regression checks against the chapter's worked binary example and a
# direct enumeration of the four outcomes of one shared-bias m=2 answer.
worked_entropies = tuple(F_bits(ell, TWO_COMPONENT) for ell in range(5))
assert np.allclose(
    worked_entropies,
    (0.0, 1.0, 1.680077045728280, 2.269404511402888,
     2.797921032936441),
    atol=3e-15)
_, worked_leverage = L_curve(2, 1, TWO_COMPONENT)
assert np.allclose(worked_leverage,
                   (1.959768863, 1.679295051,
                    1.529689159, 1.429632914), atol=6e-9)
a20 = sum(weight * (1 - theta)**2 for theta, weight in TWO_COMPONENT)
a21 = sum(weight * theta * (1 - theta) for theta, weight in TWO_COMPONENT)
a22 = sum(weight * theta**2 for theta, weight in TWO_COMPONENT)
direct_F2 = -sum(probability * math.log2(probability)
                 for probability in (a20, a21, a21, a22))
assert math.isclose(F_bits(2, TWO_COMPONENT), direct_F2,
                    rel_tol=0.0, abs_tol=2e-15)

N_VALUES = (4, 6, 8, 10)
N_COLORS = ("#a8ccef", "#72a9df", "#367fc4", "#174f91")


# Figure 1: same thermodynamic coefficient, different finite-size approach.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharex=True, sharey=True)
c_left = plot_family_panel(
    axes[0], TWO_COMPONENT, 1, "2 components: chapter mixture",
    N_VALUES, N_COLORS)
c_right = plot_family_panel(
    axes[1], TEN_COMPONENT, 1, "10 components: equal-weight matched mixture",
    N_VALUES, N_COLORS)
assert math.isclose(c_left, c_right, rel_tol=0.0, abs_tol=3e-15)
axes[0].set_ylabel("expected cumulative leverage $\\langle L\\rangle$")
axes[0].legend(fontsize=8.5, loc="upper right")
fig.suptitle("Binary coin mixtures with the same thermodynamic curve "
             f"($c_1={TWO_C1:.6f}$)\n"
             "Different latent supports leave different finite-$n$ corrections")
fig.text(0.5, 0.012,
         "The 2-component biases are $(0.1,0.9)$ with equal weights.\n"
         "The 10-component mixture matches both $F_1$ and "
         "$h_{\\mathrm{cond},1}$; "
         "its biases and checks are printed by the generating script.",
         ha="center", va="bottom", fontsize=8.5)
fig.tight_layout(rect=(0, 0.11, 1, 0.89))
fig.savefig("figures/coin_mixture_leverage.png", dpi=150)
print("wrote figures/coin_mixture_leverage.png")


# Figure 2: same scalar latent construction, m=1 versus m=2.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), sharex=True, sharey=True)
plot_family_panel(axes[0], TWO_COMPONENT, 1,
                  "$m=1$: one bit per answer", N_VALUES, N_COLORS)
plot_family_panel(axes[1], TWO_COMPONENT, 2,
                  "$m=2$: two bits share the same bias", N_VALUES, N_COLORS)
axes[0].set_ylabel("expected cumulative leverage $\\langle L\\rangle$")
axes[0].legend(fontsize=8.5, loc="upper right")
fig.suptitle("Shared scalar latent bias: exact $m=1$ and $m=2$ convergence\n"
             "Exact blocks use $G_\\ell=F_{m\\ell}$; "
             "dashes use $c_m=(F_m-h_{\\mathrm{cond},m})/"
             "h_{\\mathrm{cond},m}$")
fig.text(0.5, 0.012,
         "Both panels use $\\theta=(0.1,0.9)$ with equal weights.\n"
         "The $m=2$ initial entropy is $F_2$, "
         "not $2h(\\mathbb{E}[\\Theta])$.",
         ha="center", va="bottom", fontsize=8.5)
fig.tight_layout(rect=(0, 0.17, 1, 0.89))
fig.savefig("figures/coin_mixture_m_comparison.png", dpi=150)
print("wrote figures/coin_mixture_m_comparison.png")


# Concise reproducibility and verification report.
two_f1, two_hcond1, two_c1 = tdl_parameters(1, TWO_COMPONENT)
two_f2, two_hcond2, two_c2 = tdl_parameters(2, TWO_COMPONENT)
print("2-component parameters:", TWO_COMPONENT)
print(f"m=1: F_1={two_f1:.9f}, h_cond,1={two_hcond1:.9f}, "
      f"c_1={two_c1:.9f}")
print(f"m=2: F_2={two_f2:.9f}, h_cond,2={two_hcond2:.9f}, "
      f"c_2={two_c2:.9f}")
print(f"10-component fitted low bias: {TEN_FITTED_BIAS:.12f}")
print("10-component biases:",
      tuple(round(theta, 9) for theta, _ in TEN_COMPONENT))
print(f"matched summaries: F_1={TEN_F1:.12f}, "
      f"h_cond,1={TEN_HCOND1:.12f}")
print(f"matched coefficients: c_2coin={TWO_C1:.12f}, "
      f"c_10coin={TEN_C1:.12f}, "
      f"difference={TEN_C1-TWO_C1:.3e}")
print("verified: probability normalization, latent-information entropy "
      "bounds, finite curves below TDL, and equal matched coefficients")
