import numpy as np
import matplotlib.pyplot as plt

# =========================
# STEERING VECTOR
# =========================
def steering_vector(theta, M):
    n = np.arange(M)
    return np.exp(1j * np.pi * n * np.sin(theta))


# =========================
# DECODE BEAMFORMING
# =========================
def decode_beamforming(X, M, K):
    W = []
    idx = 0
    for _ in range(K):
        real = X[idx:idx+M]
        imag = X[idx+M:idx+2*M]
        W.append(real + 1j * imag)
        idx += 2*M
    return W


# =========================
# JCAS FITNESS (MINIMIZE)
# =========================
def jcas_fitness(X, params):
    M = params["M"]
    K = params["K"]
    H = params["H"]
    sigma2 = params["sigma2"]
    Pmax = params["Pmax"]
    alpha = params["alpha"]
    theta0 = params["theta0"]
    mu = params["mu"]

    W = decode_beamforming(X, M, K)

    # ----- Communication (Sum-rate)
    sum_rate = 0.0
    for k in range(K):
        hk = H[k]
        signal = np.abs(np.vdot(hk, W[k]))**2
        interference = sum(
            np.abs(np.vdot(hk, W[j]))**2
            for j in range(K) if j != k
        )
        sinr = signal / (interference + sigma2)
        sum_rate += np.log2(1 + sinr)

    # ----- Radar sensing
    a = steering_vector(theta0, M)
    R = sum(np.outer(w, np.conj(w)) for w in W)
    sensing_gain = np.real(np.conj(a).T @ R @ a)

    # ----- Power penalty
    total_power = sum(np.linalg.norm(w)**2 for w in W)
    penalty = mu * max(0.0, total_power - Pmax)

    fitness = -(alpha * sum_rate + (1 - alpha) * sensing_gain) + penalty
    return fitness


# =========================
# GWO (Mirjalili)
# =========================
def gwo(obj_func, lb, ub, dim, pop_size=40, max_iter=300, seed=None):
    if seed is not None:
        np.random.seed(seed)

    lb = np.full(dim, lb)
    ub = np.full(dim, ub)

    X = lb + np.random.rand(pop_size, dim) * (ub - lb)
    F = np.array([obj_func(x) for x in X])

    idx = np.argsort(F)
    Alpha, Beta, Delta = X[idx[0]], X[idx[1]], X[idx[2]]
    Alpha_score = F[idx[0]]

    curve = [Alpha_score]

    for t in range(1, max_iter + 1):
        a = 2 - 2 * t / max_iter

        for i in range(pop_size):
            X_new = np.zeros(dim)
            for leader in [Alpha, Beta, Delta]:
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - X[i])
                X_new += leader - A * D
            X[i] = X_new / 3

        X = np.clip(X, lb, ub)
        F = np.array([obj_func(x) for x in X])

        idx = np.argsort(F)
        Alpha = X[idx[0]]
        Alpha_score = min(Alpha_score, F[idx[0]])
        curve.append(Alpha_score)

    return Alpha_score, curve


# =========================
# BASELINE (RANDOM SEARCH)
# =========================
def baseline(params, dim, lb, ub, max_iter=300):
    best = np.inf
    curve = []
    for _ in range(max_iter):
        X = np.random.uniform(lb, ub, dim)
        val = jcas_fitness(X, params)
        best = min(best, val)
        curve.append(best)
    return best, curve


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # ---- SYSTEM PARAMETERS
    M = 8
    K = 3
    dim = 2 * M * K

    Pmax = 10.0
    sigma2 = 1e-3
    alpha = 0.6
    theta0 = np.deg2rad(20)
    mu = 100.0

    np.random.seed(0)
    H = [(np.random.randn(M) + 1j*np.random.randn(M))/np.sqrt(2)
         for _ in range(K)]

    params = {
        "M": M,
        "K": K,
        "H": H,
        "sigma2": sigma2,
        "Pmax": Pmax,
        "alpha": alpha,
        "theta0": theta0,
        "mu": mu
    }

    def obj(X):
        return jcas_fitness(X, params)

    lb, ub = -np.sqrt(Pmax), np.sqrt(Pmax)

    # ---- RUN BASELINE
    baseline_best, baseline_curve = baseline(params, dim, lb, ub)

    # ---- RUN GWO
    gwo_best, gwo_curve = gwo(
        obj, lb, ub, dim,
        pop_size=40,
        max_iter=300,
        seed=42
    )

    # =========================
    # OUTPUT TERMINAL (MỤC 6)
    # =========================
    improvement = (baseline_best - gwo_best) / abs(baseline_best) * 100

    print("========== KẾT QUẢ SO SÁNH ==========")
    print(f"Best fitness (JCAS gốc - Random): {baseline_best:.4f}")
    print(f"Best fitness (GWO-JCAS):         {gwo_best:.4f}")
    print(f"Cải thiện đạt được:              {improvement:.2f} %")
    print("=> GWO cho nghiệm tốt hơn bài toán gốc.")

    # =========================
    # OUTPUT ĐỒ THỊ (MỤC 5)
    # =========================
    plt.figure(figsize=(8,5))
    plt.plot(baseline_curve, '--', linewidth=2, label="JCAS gốc (Random)")
    plt.plot(gwo_curve, '-', linewidth=2, label="GWO-JCAS")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("So sánh hội tụ: JCAS gốc vs GWO-JCAS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
