"""
Esercizio 2 - Metodo dei minimi quadrati (Cowan, cap. 7)
Versione con scipy: curve_fit per il modello non lineare, chi2.sf per il P-value.

9 misure indipendenti (x_i, y_i, sigma_i), x nota senza errore.
Tre modelli candidati:
  M1: y = t0 * x^t1                     (non lineare in t1)
  M2: y = t0 + t1*x + t2*x^2            (lineare nei parametri)
  M3: y = t0 + t1*x + t2*exp(x)         (lineare nei parametri)

M2, M3 restano risolti con la formula analitica (Cowan eq. 7.10): e' la scelta
corretta indicata dal Readme ("quando possibile"), scipy non aggiunge nulla qui.
Per M1 si usa scipy.optimize.curve_fit al posto del Gauss-Newton/Levenberg-
Marquardt scritto a mano. Il P-value usa scipy.stats.chi2.sf al posto della
gamma incompleta implementata a mano.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

# --- 1. Dati ---
data = np.genfromtxt("dati.txt", names=True)
x = data["x"]
y = data["y_mean"]
sigma = data["std"]
n = len(x)
print(f"n = {n} punti")


# =====================================================================
# 2. Least squares lineare analitico (Cowan eq. 7.10, 7.11/7.12, 7.3)
#    -- invariato: e' gia' il metodo corretto per M2, M3, scipy non serve.
# =====================================================================
def linear_ls_fit(x, y, sigma, basis_funcs):
    n = len(x)
    m = len(basis_funcs)
    A = np.zeros((n, m))
    for j, a_j in enumerate(basis_funcs):
        A[:, j] = a_j(x)

    Vinv = np.diag(1.0 / sigma**2)
    AtVinvA = A.T @ Vinv @ A
    AtVinvy = A.T @ Vinv @ y
    cov = np.linalg.inv(AtVinvA)
    theta_hat = cov @ AtVinvy

    resid = y - A @ theta_hat
    chi2_val = resid @ Vinv @ resid

    return theta_hat, cov, chi2_val


basis_M2 = [lambda x: np.ones_like(x), lambda x: x, lambda x: x**2]
theta_M2, cov_M2, chi2_M2 = linear_ls_fit(x, y, sigma, basis_M2)
m_M2 = 3

print("\n--- Modello 2: y = t0 + t1*x + t2*x^2 ---")
for i, name in enumerate(["t0", "t1", "t2"]):
    print(f"  {name} = {theta_M2[i]:.4f} +/- {np.sqrt(cov_M2[i,i]):.4f}")
print(f"  chi2 = {chi2_M2:.3f}")

basis_M3 = [lambda x: np.ones_like(x), lambda x: x, lambda x: np.exp(x)]
theta_M3, cov_M3, chi2_M3 = linear_ls_fit(x, y, sigma, basis_M3)
m_M3 = 3

print("\n--- Modello 3: y = t0 + t1*x + t2*exp(x) ---")
for i, name in enumerate(["t0", "t1", "t2"]):
    print(f"  {name} = {theta_M3[i]:.4f} +/- {np.sqrt(cov_M3[i,i]):.4f}")
print(f"  chi2 = {chi2_M3:.3f}")


# =====================================================================
# 3. Modello 1 (non lineare): y = t0 * x^t1  -> scipy.optimize.curve_fit
# =====================================================================
def model_M1(x, t0, t1):
    return t0 * x**t1


# stima iniziale (log-log), stessa idea di prima, giusto per aiutare la
# convergenza; curve_fit userebbe comunque un default se la si omette.
logA = np.column_stack([np.ones_like(x), np.log(x)])
coef0, *_ = np.linalg.lstsq(logA, np.log(y), rcond=None)
p0 = [np.exp(coef0[0]), coef0[1]]

# sigma=sigma, absolute_sigma=True: dice a curve_fit di trattare sigma come
# le vere deviazioni standard (necessario per ottenere chi2 ed errori
# assoluti corretti, non solo relativi)
popt, pcov = curve_fit(model_M1, x, y, p0=p0, sigma=sigma, absolute_sigma=True)
theta_M1 = popt
cov_M1 = pcov
m_M1 = 2

resid_M1 = (y - model_M1(x, *theta_M1)) / sigma
chi2_M1 = np.sum(resid_M1**2)

print("\n--- Modello 1: y = t0 * x^t1 (scipy.optimize.curve_fit) ---")
for i, name in enumerate(["t0", "t1"]):
    print(f"  {name} = {theta_M1[i]:.4f} +/- {np.sqrt(cov_M1[i,i]):.4f}")
print(f"  chi2 = {chi2_M1:.3f}")


# =====================================================================
# 4. P-value con scipy.stats.chi2.sf (eq. 7.24)
# =====================================================================
p_check = chi2.sf(3.99, 3)
print(f"\n[verifica] chi2.sf(3.99, 3) = {p_check:.4f}  (il libro riporta 0.263)")


# =====================================================================
# 5. Confronto dei tre modelli
# =====================================================================
results = [
    ("M1: t0*x^t1", chi2_M1, n - m_M1),
    ("M2: t0+t1*x+t2*x^2", chi2_M2, n - m_M2),
    ("M3: t0+t1*x+t2*exp(x)", chi2_M3, n - m_M3),
]

print("\n=== Confronto modelli (test del chi2 / Pearson) ===")
print(f"{'modello':<24}{'chi2':>8}{'nd':>5}{'chi2/nd':>10}{'P-value':>10}")
for name, c2, nd in results:
    p = chi2.sf(c2, nd)
    print(f"{name:<24}{c2:>8.3f}{nd:>5d}{c2/nd:>10.3f}{p:>10.4f}")
