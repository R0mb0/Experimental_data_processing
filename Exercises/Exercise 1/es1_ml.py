"""
Esercizio 1 - Stima ML a due parametri (Cowan, par. 6.8)

Modello: f(x; a, b) = (1 + a*x + b*x^2) / N(a,b)
dove N(a,b) e' la costante di normalizzazione sull'intervallo osservato
[xmin, xmax], scelto qui come min/max campionari (eq. 6.27 di Cowan
generalizzata a xmin, xmax non necessariamente simmetrici).
"""

import numpy as np

# --- 1. Dati ---
data = np.loadtxt("data.txt")
xmin, xmax = data.min(), data.max()
n = len(data)
print(f"n = {n}, xmin = {xmin:.4f}, xmax = {xmax:.4f}")


# --- 2. pdf normalizzata sull'intervallo [xmin, xmax] ---
def norm_const(a, b, xmin, xmax):
    # integrale di (1 + a*x + b*x^2) da xmin a xmax, calcolato analiticamente
    return (
        (xmax - xmin)
        + a / 2 * (xmax**2 - xmin**2)
        + b / 3 * (xmax**3 - xmin**3)
    )


def pdf(x, a, b, xmin, xmax):
    return (1 + a * x + b * x**2) / norm_const(a, b, xmin, xmax)


# --- 3. log-verosimiglianza negativa ---
def neg_log_likelihood(params, x, xmin, xmax):
    a, b = params
    f = pdf(x, a, b, xmin, xmax)
    if np.any(f <= 0):
        return np.inf  # fuori dal dominio fisico (pdf negativa)
    return -np.sum(np.log(f))


# --- 4. Massimizzazione numerica: Newton-Raphson scritto a mano ---
# (scipy non e' disponibile in questo ambiente: implementiamo noi stessi
#  l'equivalente di MIGRAD, un metodo quasi-Newton che usa gradiente e
#  Hessiana di -logL per aggiornare i parametri ad ogni iterazione)
def gradient_finite_diff(func, params, x, xmin, xmax, eps=1e-5):
    n_params = len(params)
    grad = np.zeros(n_params)
    for i in range(n_params):
        pp = np.array(params, dtype=float)
        pp[i] += eps
        pm = np.array(params, dtype=float)
        pm[i] -= eps
        grad[i] = (func(pp, x, xmin, xmax) - func(pm, x, xmin, xmax)) / (2 * eps)
    return grad


def hessian_finite_diff(func, params, x, xmin, xmax, eps=1e-4):
    n_params = len(params)
    H = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            pp = np.array(params, dtype=float)
            pp[i] += eps
            pp[j] += eps
            f_pp = func(pp, x, xmin, xmax)

            pm = np.array(params, dtype=float)
            pm[i] += eps
            pm[j] -= eps
            f_pm = func(pm, x, xmin, xmax)

            mp = np.array(params, dtype=float)
            mp[i] -= eps
            mp[j] += eps
            f_mp = func(mp, x, xmin, xmax)

            mm = np.array(params, dtype=float)
            mm[i] -= eps
            mm[j] -= eps
            f_mm = func(mm, x, xmin, xmax)

            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
    return H


# --- Newton-Raphson con backtracking: theta_{k+1} = theta_k - t*H^-1 grad(-logL) ---
# Il passo pieno (t=1) puo' portare (a,b) in una zona dove la pdf diventa
# negativa (-logL = inf). Si dimezza t finche' il nuovo punto non migliora
# la funzione obiettivo e resta nel dominio ammesso.
theta = np.array([0.0, 0.0])  # punto di partenza
f_curr = neg_log_likelihood(theta, data, xmin, xmax)
for it in range(50):
    grad = gradient_finite_diff(neg_log_likelihood, theta, data, xmin, xmax)
    H = hessian_finite_diff(neg_log_likelihood, theta, data, xmin, xmax)
    step = np.linalg.solve(H, grad)

    t = 1.0
    while t > 1e-6:
        theta_new = theta - t * step
        f_new = neg_log_likelihood(theta_new, data, xmin, xmax)
        if np.isfinite(f_new) and f_new < f_curr:
            break
        t /= 2
    theta, f_curr = theta_new, f_new

    if np.linalg.norm(t * step) < 1e-8:
        break

a_hat, b_hat = theta
nll_min = neg_log_likelihood(theta, data, xmin, xmax)
print(f"\nConvergenza in {it+1} iterazioni")
print(f"Stime ML:  alpha_hat = {a_hat:.4f},  beta_hat = {b_hat:.4f}")
print(f"-logL_min = {nll_min:.4f}")


# --- 5. Matrice Hessiana numerica di -logL nel punto di massimo (eq. 6.21/6.22) ---
H = hessian_finite_diff(neg_log_likelihood, theta, data, xmin, xmax)
cov = np.linalg.inv(H)  # H e' l'Hessiana di -logL, quindi cov = H^-1 direttamente

sigma_a = np.sqrt(cov[0, 0])
sigma_b = np.sqrt(cov[1, 1])
rho = cov[0, 1] / (sigma_a * sigma_b)

print(f"\nalpha = {a_hat:.3f} +/- {sigma_a:.3f}")
print(f"beta  = {b_hat:.3f} +/- {sigma_b:.3f}")
print(f"cov[a,b] = {cov[0,1]:.4f},  correlazione r = {rho:.3f}")
