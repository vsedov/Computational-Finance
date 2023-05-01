# CS3930/CS3930R
# 3. (a) Consider a European put option on a share currently worth 120p maturing in
# two years with the strike price of 85p. Assume that the volatility of the share
# is 40% pa and the risk-free interest rate (with continuous compounding) is
# 12% pa. Use the three-step binomial tree to calculate the value of the option.
# Do the calculation in the following steps.

# Do binomial trees for both american and European options
from numpy import exp, sqrt, zeros

s0 = 120
k = 85
t = 2
r = 0.12
sigma = 0.4
n = 3


# Generate the tree
def generate_tree(s0, k, t, r, sigma, n):
    delta_t = t / n
    u = exp(sigma * sqrt(delta_t))
    d = 1 / u
    p = (exp(r * delta_t) - d) / (u - d)
    tree = zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            tree[j, i] = s0 * (d**j) * (u ** (i - j))
    return tree


print(generate_tree(s0, k, t, r, sigma, n))


def binomial_tree_put_eu_american(s0, k, t, r, sigma, n, american):
    delta_t = t / n
    u = exp(sigma * sqrt(delta_t))
    d = 1 / u
    p = (exp(r * delta_t) - d) / (u - d)
    tree = generate_tree(s0, k, t, r, sigma, n)
    # Calculate the payoff at maturity
    for i in range(n + 1):
        tree[i, n] = max(k - tree[i, n], 0)
    # Calculate the option price at t=0
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            tree[j, i] = exp(-r * delta_t) * (
                p * tree[j, i + 1] + (1 - p) * tree[j + 1, i + 1]
            )
            if american:
                tree[j, i] = max(tree[j, i], k - tree[j, i])
    return tree[0, 0]


print(binomial_tree_put_eu_american(s0, k, t, r, sigma, n, True))
print(binomial_tree_put_eu_american(s0, k, t, r, sigma, n, False))
