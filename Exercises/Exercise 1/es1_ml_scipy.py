"""
Esercizio 1 - Stima ML a due parametri (Cowan, par. 6.8)
Versione con scipy.optimize (equivalente diretto di MIGRAD/HESSE di MINUIT).

Modello: f(x; a, b) = (1 + a*x + b*x^2) / N(a,b)
dove N(a,b) e' la costante di normalizzazione sull'intervallo osservato
[xmin, xmax] (eq. 6.27 di Cowan, generalizzata a xmin, xmax non
necessariamente simmetrici).
"""

import numpy as np
from scipy.optimize import minimize

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


# --- 4. Massimizzazione numerica (equivalente a MIGRAD) ---
# BFGS e' un metodo quasi-Newton, concettualmente lo stesso algoritmo
# usato da MIGRAD in MINUIT. minimize() gestisce da solo eventuali passi
# che uscirebbero dal dominio ammesso grazie alla propria line search
# interna (motivo per cui qui non serve il backtracking manuale).
result = minimize(
    neg_log_likelihood,
    x0=[0.0, 0.0],
    args=(data, xmin, xmax),
    method="BFGS",
)
a_hat, b_hat = result.x
print(f"\nConvergenza: {result.success}  (metodo BFGS, {result.nit} iterazioni)")
print(f"Stime ML:  alpha_hat = {a_hat:.4f},  beta_hat = {b_hat:.4f}")
print(f"-logL_min = {result.fun:.4f}")


# --- 5. Errori: Hessiana di -logL nel punto di massimo (eq. 6.21/6.22) ---
# Non usiamo l'Hessiana approssimata restituita da BFGS (result.hess_inv,
# costruita iterativamente e solo approssimata): la ricalcoliamo per
# differenze finite esattamente nel punto di massimo, come fa HESSE in
# MINUIT, per avere la stima "vera" della matrice di covarianza.
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


H = hessian_finite_diff(neg_log_likelihood, result.x, data, xmin, xmax)
cov = np.linalg.inv(H)  # H e' l'Hessiana di -logL, quindi cov = H^-1 direttamente

sigma_a = np.sqrt(cov[0, 0])
sigma_b = np.sqrt(cov[1, 1])
rho = cov[0, 1] / (sigma_a * sigma_b)

print(f"\nalpha = {a_hat:.3f} +/- {sigma_a:.3f}")
print(f"beta  = {b_hat:.3f} +/- {sigma_b:.3f}")
print(f"cov[a,b] = {cov[0,1]:.4f},  correlazione r = {rho:.3f}")

# Confronto facoltativo con l'Hessiana approssimata interna a BFGS:
# cov_bfgs = result.hess_inv  (di solito molto simile a "cov" sopra,
# ma leggermente meno accurata perche' costruita per approssimazioni
# successive durante la discesa, non ricalcolata esattamente al minimo)
