"""
   Copyright 2020 (https://github.com/IBM/discrete-gaussian-differential-privacy)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Code for computing approximate differential privacy guarantees
# for discrete Gaussian and, more generally, concentrated DP
# See https://arxiv.org/abs/2004.00010
# - Thomas Steinke dgauss@thomas-steinke.net 2020

import math
import matplotlib.pyplot as plt
import numpy as np
import matrix
import argparse
from scipy import sparse, optimize, special
import six


# *********************************************************************
# Now we move on to concentrated DP

# compute delta such that
# rho-CDP implies (eps,delta)-DP
# Note that adding cts or discrete N(0,sigma2) to sens-1 gives rho=1/(2*sigma2)

# start with standard P[privloss>eps] bound via markov
def cdp_delta_standard(rho, eps):
    assert rho >= 0
    assert eps >= 0
    if rho == 0: return 0  # degenerate case
    # https://arxiv.org/pdf/1605.02065.pdf#page=15
    return math.exp(-((eps - rho) ** 2) / (4 * rho))


# Our new bound:
# https://arxiv.org/pdf/2004.00010v3.pdf#page=13
def cdp_delta(rho, eps):
    assert rho >= 0
    assert eps >= 0
    if rho == 0: return 0  # degenerate case

    # search for best alpha
    # Note that any alpha in (1,infty) yields a valid upper bound on delta
    # Thus if this search is slightly "incorrect" it will only result in larger delta (still valid)
    # This code has two "hacks".
    # First the binary search is run for a pre-specificed length.
    # 1000 iterations should be sufficient to converge to a good solution.
    # Second we set a minimum value of alpha to avoid numerical stability issues.
    # Note that the optimal alpha is at least (1+eps/rho)/2. Thus we only hit this constraint
    # when eps<=rho or close to it. This is not an interesting parameter regime, as you will
    # inherently get large delta in this regime.
    amin = 1.01  # don't let alpha be too small, due to numerical stability
    amax = (eps + 1) / (2 * rho) + 2
    for i in range(1000):  # should be enough iterations
        alpha = (amin + amax) / 2
        derivative = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
        if derivative < 0:
            amin = alpha
        else:
            amax = alpha
    # now calculate delta
    delta = math.exp((alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)) / (alpha - 1.0)
    return min(delta, 1.0)  # delta<=1 always


# Above we compute delta given rho and eps, now we compute eps instead
# That is we wish to compute the smallest eps such that rho-CDP implies (eps,delta)-DP
def cdp_eps(rho, delta):
    assert rho >= 0
    assert delta > 0
    if delta >= 1 or rho == 0: return 0.0  # if delta>=1 or rho=0 then anything goes
    epsmin = 0.0  # maintain cdp_delta(rho,eps)>=delta
    epsmax = rho + 2 * math.sqrt(rho * math.log(1 / delta))  # maintain cdp_delta(rho,eps)<=delta
    # to compute epsmax we use the standard bound
    for i in range(1000):
        eps = (epsmin + epsmax) / 2
        if cdp_delta(rho, eps) <= delta:
            epsmax = eps
        else:
            epsmin = eps
    return epsmax


# Now we compute rho
# Given (eps,delta) find the smallest rho such that rho-CDP implies (eps,delta)-DP
def cdp_rho(eps, delta):
    assert eps >= 0
    assert delta > 0
    if delta >= 1: return 0.0  # if delta>=1 anything goes
    rhomin = 0.0  # maintain cdp_delta(rho,eps)<=delta
    rhomax = eps + 1  # maintain cdp_delta(rhomax,eps)>delta
    for i in range(1000):
        rho = (rhomin + rhomax) / 2
        if cdp_delta(rho, eps) <= delta:
            rhomin = rho
        else:
            rhomax = rho
    return rhomin


def _log_add(logx, logy):
    """Add two numbers in the log space."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx, logy):
    """Subtract two numbers in the log space. Answer must be non-negative."""
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _log_sub_sign(logx, logy):
    """Returns log(exp(logx)-exp(logy)) and its sign."""
    if logx > logy:
        s = True
        mag = logx + np.log(1 - np.exp(logy - logx))
    elif logx < logy:
        s = False
        mag = logy + np.log(1 - np.exp(logx - logy))
    else:
        s = True
        mag = -np.inf

    return s, mag


def _log_print(logx):
    """Pretty print."""
    if logx < math.log(sys.float_info.max):
        return "{}".format(math.exp(logx))
    else:
        return "exp({})".format(logx)


def _log_comb(n, k):
    return (special.gammaln(n + 1) - special.gammaln(k + 1) -
            special.gammaln(n - k + 1))


def _compute_log_a_int(q, sigma, alpha):
    """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
    assert isinstance(alpha, six.integer_types)

    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = (
                _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * math.log(1 - q))

        s = log_coef_i + (i * i - i) / (2 * (sigma ** 2))
        log_a = _log_add(log_a, s)

    return float(log_a)


def _compute_log_a_frac(q, sigma, alpha):
    """Compute log(A_alpha) for fractional alpha. 0 < q < 1."""
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + .5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


def _compute_log_a(q, sigma, alpha):
    """Compute log(A_alpha) for any positive finite alpha."""
    if float(alpha).is_integer():
        return _compute_log_a_int(q, sigma, int(alpha))
    else:
        return _compute_log_a_frac(q, sigma, alpha)


def _log_erfc(x):
    """Compute log(erfc(x)) with high accuracy for large x."""
    try:
        return math.log(2) + special.log_ndtr(-x * 2 ** .5)
    except NameError:
        # If log_ndtr is not available, approximate as follows:
        r = special.erfc(x)
        if r == 0.0:
            # Using the Laurent series at infinity for the tail of the erfc function:
            #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
            # To verify in Mathematica:
            #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
            return (-math.log(math.pi) / 2 - math.log(x) - x ** 2 - .5 * x ** -2 +
                    .625 * x ** -4 - 37. / 24. * x ** -6 + 353. / 64. * x ** -8)
        else:
            return math.log(r)


def _compute_delta(orders, rdp, eps):
    """Compute delta given a list of RDP values and target epsilon.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    eps: The target epsilon.
  Returns:
    Pair of (delta, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if eps < 0:
        raise ValueError("Value of privacy loss bound epsilon must be >=0.")
    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")

    # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
    #   delta = min( np.exp((rdp_vec - eps) * (orders_vec - 1)) )

    # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4):
    logdeltas = []  # work in log space to avoid overflows
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1:
            raise ValueError("Renyi divergence order must be >=1.")
        if r < 0:
            raise ValueError("Renyi divergence must be >=0.")
        # For small alpha, we are better of with bound via KL divergence:
        # delta <= sqrt(1-exp(-KL)).
        # Take a min of the two bounds.
        logdelta = 0.5 * math.log1p(-math.exp(-r))
        if a > 1.01:
            # This bound is not numerically stable as alpha->1.
            # Thus we have a min value for alpha.
            # The bound is also not useful for small alpha, so doesn't matter.
            rdp_bound = (a - 1) * (r - eps + math.log1p(-1 / a)) - math.log(a)
            logdelta = min(logdelta, rdp_bound)

        logdeltas.append(logdelta)

    idx_opt = np.argmin(logdeltas)
    return min(math.exp(logdeltas[idx_opt]), 1.), orders_vec[idx_opt]


def _compute_eps(orders, rdp, delta):
    """Compute epsilon given a list of RDP values and target delta.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if delta <= 0:
        raise ValueError("Privacy failure probability bound delta must be >0.")
    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")

    # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
    #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

    # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
    # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
    eps_vec = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1:
            raise ValueError("Renyi divergence order must be >=1.")
        if r < 0:
            raise ValueError("Renyi divergence must be >=0.")

        if delta ** 2 + math.expm1(-r) >= 0:
            # In this case, we can simply bound via KL divergence:
            # delta <= sqrt(1-exp(-KL)).
            eps = 0  # No need to try further computation if we have eps = 0.
        elif a > 1.01:
            # This bound is not numerically stable as alpha->1.
            # Thus we have a min value of alpha.
            # The bound is also not useful for small alpha, so doesn't matter.
            eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
        else:
            # In this case we can't do anything. E.g., asking for delta = 0.
            eps = np.inf
        eps_vec.append(eps)

    idx_opt = np.argmin(eps_vec)
    return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def _compute_rdp(q, sigma, alpha):
    """Compute RDP of the Sampled Gaussian mechanism at order alpha.
  Args:
    q: The sampling rate.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.
  Returns:
    RDP at alpha, can be np.inf.
  """
    if q == 0:
        return 0

    if q == 1.:
        return alpha / (2 * sigma ** 2)

    if np.isinf(alpha):
        return np.inf

    return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(q, noise_multiplier, steps, orders):
    """Computes RDP of the Sampled Gaussian Mechanism.
  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
    if np.isscalar(orders):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array(
            [_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps


def search_sigma(sample_rate, step, epsilon, delta, sigma_low=0, sigma_high=1000):  # 找到最佳匹配的噪音sigma
    sigma = (sigma_low + sigma_high) / 2
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
              list(range(5, 64)) + [128, 256, 512])
    rdp = compute_rdp(sample_rate, sigma, step, orders)
    eps, opt_order = _compute_eps(orders, rdp, delta)
    eps = round(eps, 4)
    if abs(eps - epsilon) <= 0.01:
        return sigma
    else:
        if (eps > epsilon):
            return search_sigma(sample_rate, step, epsilon, delta, sigma, sigma_high)
        else:
            return search_sigma(sample_rate, step, epsilon, delta, sigma_low, sigma)


if __name__ == '__main__':
    eps = 1
    delta = 1e-5
    rho = cdp_rho(eps, delta)
    print("rho", rho)
    sigma = np.sqrt(20 / 2 / rho / 9)
    print("sigma", sigma)
    print("rho", rho*9/20)
    epsilon = cdp_eps(rho * 9 / 20, delta)
    sig = search_sigma(0.1, 20, epsilon, delta, 0, 1000)
    print("sigma", sig)
