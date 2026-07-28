"""
Esercizio 2 - Metodo dei minimi quadrati (Cowan, cap. 7)

9 misure indipendenti (x_i, y_i, sigma_i), x nota senza errore.
Tre modelli candidati:
  M1: y = t0 * x^t1                     (non lineare in t1)
  M2: y = t0 + t1*x + t2*x^2            (lineare nei parametri)
  M3: y = t0 + t1*x + t2*exp(x)         (lineare nei parametri)

Per M2, M3 si usa la soluzione analitica (Cowan eq. 7.10).
Per M1 si minimizza numericamente il chi2 con Gauss-Newton/Levenberg-Marquardt
(scipy non e' disponibile in questo sandbox, come nell'Esercizio 1).
"""

import math
import numpy as np

# --- 1. Dati ---
data = np.genfromtxt("dati.txt", names=True)
x = data["x"]
y = data["y_mean"]
sigma = data["std"]
n = len(x)
print(f"n = {n} punti")


# =====================================================================
# 2. Least squares lineare generico (Cowan eq. 7.10, 7.11/7.12, 7.3)
# =====================================================================
def linear_ls_fit(x, y, sigma, basis_funcs):
    """
    basis_funcs: lista di funzioni a_j(x), j=0..m-1.
    Modello: lambda(x; theta) = sum_j a_j(x) * theta_j
    V = diag(sigma_i^2)  (misure indipendenti)
    """
    n = len(x)
    m = len(basis_funcs)
    A = np.zeros((n, m))
    for j, a_j in enumerate(basis_funcs):
        A[:, j] = a_j(x)

    Vinv = np.diag(1.0 / sigma**2)

    # theta_hat = (A^T Vinv A)^-1 A^T Vinv y      (eq. 7.10)
    AtVinvA = A.T @ Vinv @ A
    AtVinvy = A.T @ Vinv @ y
    cov = np.linalg.inv(AtVinvA)  # eq. 7.12 (inversa della matrice di informazione)
    theta_hat = cov @ AtVinvy

    resid = y - A @ theta_hat
    chi2 = resid @ Vinv @ resid  # eq. 7.5 / 7.3 (V diagonale)

    return theta_hat, cov, chi2


# --- Modello 2: parabola  y = t0 + t1*x + t2*x^2 ---
basis_M2 = [lambda x: np.ones_like(x), lambda x: x, lambda x: x**2]
theta_M2, cov_M2, chi2_M2 = linear_ls_fit(x, y, sigma, basis_M2)
m_M2 = 3

print("\n--- Modello 2: y = t0 + t1*x + t2*x^2 ---")
for i, name in enumerate(["t0", "t1", "t2"]):
    print(f"  {name} = {theta_M2[i]:.4f} +/- {np.sqrt(cov_M2[i,i]):.4f}")
print(f"  chi2 = {chi2_M2:.3f}")


# --- Modello 3: y = t0 + t1*x + t2*exp(x) ---
basis_M3 = [lambda x: np.ones_like(x), lambda x: x, lambda x: np.exp(x)]
theta_M3, cov_M3, chi2_M3 = linear_ls_fit(x, y, sigma, basis_M3)
m_M3 = 3

print("\n--- Modello 3: y = t0 + t1*x + t2*exp(x) ---")
for i, name in enumerate(["t0", "t1", "t2"]):
    print(f"  {name} = {theta_M3[i]:.4f} +/- {np.sqrt(cov_M3[i,i]):.4f}")
print(f"  chi2 = {chi2_M3:.3f}")


# =====================================================================
# 3. Modello 1 (non lineare): y = t0 * x^t1  -> Gauss-Newton/Levenberg-Marquardt
# =====================================================================
def model_M1(theta, x):
    t0, t1 = theta
    return t0 * x**t1


def residuals_M1(theta, x, y, sigma):
    return (y - model_M1(theta, x)) / sigma


def jacobian_M1(theta, x, sigma):
    # derivate del RESIDUO (non del modello!) rispetto a t0, t1
    t0, t1 = theta
    d_dt0 = -(x**t1) / sigma
    d_dt1 = -(t0 * x**t1 * np.log(x)) / sigma
    return np.column_stack([d_dt0, d_dt1])


# Punto di partenza: log(y) = log(t0) + t1*log(x) e' lineare, usato SOLO
# per ottenere una stima iniziale ragionevole (non e' il fit finale: minimizza
# i residui in scala logaritmica, non il chi2 originale in y).
logA = np.column_stack([np.ones_like(x), np.log(x)])
coef0, *_ = np.linalg.lstsq(logA, np.log(y), rcond=None)
theta = np.array([np.exp(coef0[0]), coef0[1]])
print(f"\nStima iniziale (da log-log): t0={theta[0]:.3f}, t1={theta[1]:.3f}")

# Levenberg-Marquardt: (J^T J + lambda*diag(J^T J)) delta = -J^T r
chi2_curr = np.sum(residuals_M1(theta, x, y, sigma) ** 2)
lam = 1e-3
for it in range(200):
    r = residuals_M1(theta, x, y, sigma)
    J = jacobian_M1(theta, x, sigma)
    JtJ = J.T @ J
    Jtr = J.T @ r

    damped = JtJ + lam * np.diag(np.diag(JtJ))
    delta = np.linalg.solve(damped, -Jtr)
    theta_new = theta + delta
    chi2_new = np.sum(residuals_M1(theta_new, x, y, sigma) ** 2)

    if chi2_new < chi2_curr:
        theta, chi2_curr = theta_new, chi2_new
        lam /= 3
        if np.linalg.norm(delta) < 1e-10:
            break
    else:
        lam *= 3  # passo troppo aggressivo: aumenta lo smorzamento e riprova

theta_M1 = theta
chi2_M1 = chi2_curr
m_M1 = 2

# Covarianza: approssimazione Gauss-Newton dell'informazione osservata,
# cov = (J^T J)^-1 valutata al minimo (J e' gia' pesato da 1/sigma)
J_final = jacobian_M1(theta_M1, x, sigma)
cov_M1 = np.linalg.inv(J_final.T @ J_final)

print(f"\n--- Modello 1: y = t0 * x^t1  (convergenza in {it+1} iterazioni) ---")
for i, name in enumerate(["t0", "t1"]):
    print(f"  {name} = {theta_M1[i]:.4f} +/- {np.sqrt(cov_M1[i,i]):.4f}")
print(f"  chi2 = {chi2_M1:.3f}")


# =====================================================================
# 4. P-value: funzione di sopravvivenza della chi2 (eq. 7.24), a mano
#    Q(a,x) = gamma incompleta superiore regolarizzata; per chi2 con k
#    gradi di liberta': P(chi2 > z) = Q(k/2, z/2).
#    Algoritmo standard (Numerical Recipes): serie per x < a+1,
#    frazione continua per x >= a+1.
# =====================================================================
def _gser(a, x, itmax=200, eps=1e-12):
    """Serie per la gamma incompleta inferiore regolarizzata P(a,x)."""
    gln = math.lgamma(a)
    if x <= 0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(itmax):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gcf(a, x, itmax=200, eps=1e-12, tiny=1e-300):
    """Frazione continua (Lentz) per la gamma incompleta superiore Q(a,x)."""
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def gammaq(a, x):
    """Gamma incompleta superiore regolarizzata Q(a,x) = 1 - P(a,x)."""
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    else:
        return _gcf(a, x)


def chi2_sf(chi2_value, ndof):
    """P(chi2_ndof > chi2_value): eq. (7.24) di Cowan."""
    return gammaq(ndof / 2.0, chi2_value / 2.0)


# Validazione: esempio del libro (Sez. 7.3/7.5), fit lineare a 5 punti,
# chi2 = 3.99, nd = 3 -> il libro riporta P = 0.263
p_check = chi2_sf(3.99, 3)
print(f"\n[verifica] chi2_sf(3.99, 3) = {p_check:.4f}  (il libro riporta 0.263)")


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
    p = chi2_sf(c2, nd)
    print(f"{name:<24}{c2:>8.3f}{nd:>5d}{c2/nd:>10.3f}{p:>10.4f}")
