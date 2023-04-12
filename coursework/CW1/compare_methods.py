import math

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st

np.random.seed(0)


def bs(S0, K, T, r, sigma, option_type):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (
        sigma * math.sqrt(T)
    )
    d2 = (math.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (
        sigma * math.sqrt(T)
    )
    if option_type == "call":
        return S0 * st.norm.cdf(d1) - K * math.exp(-r * T) * st.norm.cdf(d2)
    elif option_type == "put":
        return K * math.exp(-r * T) * st.norm.cdf(-d2) - S0 * st.norm.cdf(-d1)


def montecarlo(S0, K, T, r, sigma, option_type, n_simulations, n_steps):
    dt = T / n_steps
    stock_prices = np.zeros((n_simulations, n_steps + 1))
    stock_prices[:, 0] = S0
    for i in range(n_steps):
        eps = np.random.normal(0, 1, n_simulations)
        stock_prices[:, i + 1] = stock_prices[:, i] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps
        )
    if option_type == "call":
        payoff = np.maximum(stock_prices[:, -1] - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - stock_prices[:, -1], 0)
    else:
        raise ValueError("Invalid option type")

    fig, ax = plt.subplots(1)
    for i in range(10):
        ax.plot(np.linspace(0, T, n_steps + 1), stock_prices[i, :])
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock price")
    ax.set_title("Stock price paths")
    fig.show()

    discounted_payoff = np.exp(-r * T) * payoff
    return np.mean(discounted_payoff)


def trees(S0, K, T, r, sigma, option_type, n=1000, american=False):
    delta_t = T / n
    u = np.exp(sigma * np.sqrt(delta_t))
    d = 1 / u
    p = (np.exp(r * delta_t) - d) / (u - d)
    tree = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            tree[j, i] = S0 * (d**j) * (u ** (i - j))
    if option_type == "call":
        option_values = np.maximum(tree[:, -1] - K, 0)
    elif option_type == "put":
        option_values = np.maximum(K - tree[:, -1], 0)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = np.exp(-r * delta_t) * (
                p * option_values[j] + (1 - p) * option_values[j + 1]
            )
            if american:
                if option_type == "call":
                    option_values[j] = np.maximum(
                        option_values[j], tree[j, i] - K
                    )
                elif option_type == "put":
                    option_values[j] = np.maximum(
                        option_values[j], K - tree[j, i]
                    )
    return option_values[0]


S0 = 100
X1 = 120
X2 = 98
T = 0.5
r = 0.05
sigma = 0.2


# European call and put options computed from the Black-Scholes formula
print(
    "European call option price (Black-Scholes):",
    bs(S0, X1, T, r, sigma, "call"),
)
print(
    "European put option price (Black-Scholes):",
    bs(S0, X1, T, r, sigma, "put"),
)

print(
    "European call option price (Black-Scholes):",
    bs(S0, X2, T, r, sigma, "call"),
)
print(
    "European put option price (Black-Scholes):",
    bs(S0, X2, T, r, sigma, "put"),
)
print("\n\n")

print(
    f"European call option price (Binomial Tree) => {X1}:",
    trees(S0, X1, T, r, sigma, "call"),
)
print(
    f"European put option price (Binomial Tree) => {X1}:",
    trees(S0, X1, T, r, sigma, "put"),
)
print(
    f"European call option price (Binomial Tree) => {X2}:",
    trees(S0, X2, T, r, sigma, "call"),
)
print(
    f"European put option price (Binomial Tree) => {X2}:",
    trees(S0, X2, T, r, sigma, "put"),
)
print(
    f"American call option price (Binomial Tree) => {X1}:",
    trees(S0, X1, T, r, sigma, "call", n=100, american=True),
)
print(
    f"American put option price (Binomial Tree) => {X1}:",
    trees(S0, X1, T, r, sigma, "put", n=100, american=True),
)
print(
    f"American call option price (Binomial Tree) => {X2}:",
    trees(S0, X2, T, r, sigma, "call", n=100, american=True),
)
print(
    f"American put option price (Binomial Tree) => {X2}:",
    trees(S0, X2, T, r, sigma, "put", n=100, american=True),
)

print("\n\n")

call1_bs = bs(S0, X1, T, r, sigma, "call")
put1_bs = bs(S0, X1, T, r, sigma, "put")
call2_bs = bs(S0, X2, T, r, sigma, "call")
put2_bs = bs(S0, X2, T, r, sigma, "put")


n_simulations = 100000
n_steps = 180

call1_mc = montecarlo(S0, X1, T, r, sigma, "call", n_simulations, n_steps)
put1_mc = montecarlo(S0, X1, T, r, sigma, "put", n_simulations, n_steps)
call2_mc = montecarlo(S0, X2, T, r, sigma, "call", n_simulations, n_steps)
put2_mc = montecarlo(S0, X2, T, r, sigma, "put", n_simulations, n_steps)

print("Theoretical European call option 1 price: ", call1_bs)
print("Monte Carlo European call option 1 price: ", call1_mc)
print("Theoretical European put option 1 price: ", put1_bs)
print("Monte Carlo European put option 1 price: ", put1_mc)
print("Theoretical European call option 2 price: ", call2_bs)
print("Monte Carlo European call option 2 price: ", call2_mc)
print("Theoretical European put option 2 price: ", put2_bs)
print("Monte Carlo European put option 2 price: ", put2_mc)
fig, ax = plt.subplots(1)
stock_prices = []
dt = T / n_steps
for i in range(10):
    stock_prices = np.zeros((n_simulations, n_steps + 1))
    stock_prices[:, 0] = S0
for j in range(n_steps):
    eps = np.random.normal(0, 1, n_simulations)
    stock_prices[:, j + 1] = stock_prices[:, j] * np.exp(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps
    )
ax.plot(np.linspace(0, T, n_steps + 1), stock_prices[i, :])
ax.set_xlabel("Time")
ax.set_ylabel("Stock price")
ax.set_title("Stock price paths for option 1")
plt.show()
fig, ax = plt.subplots(1)
for i in range(10):
    stock_prices = np.zeros((n_simulations, n_steps + 1))
    stock_prices[:, 0] = S0
for j in range(n_steps):
    eps = np.random.normal(0, 1, n_simulations)
    stock_prices[:, j + 1] = stock_prices[:, j] * np.exp(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps
    )
ax.plot(np.linspace(0, T, n_steps + 1), stock_prices[i, :])
ax.set_xlabel("Time")
ax.set_ylabel("Stock price")
ax.set_title("Stock price paths for option 2")
plt.show()

fig, ax = plt.subplots(1)
ax.bar([0, 1, 2, 3], [call1_bs, call1_mc, put1_bs, put1_mc])
ax.bar([4, 5, 6, 7], [call2_bs, call2_mc, put2_bs, put2_mc])
ax.set_xticks([0.5, 4.5])
ax.set_xticklabels(["Option 1", "Option 2"])
ax.set_ylabel("Option price")
ax.set_title("European option prices")
ax.legend(
    [
        "Theoretical call",
        "Monte Carlo call",
        "Theoretical put",
        "Monte Carlo put",
    ]
)
plt.show()
