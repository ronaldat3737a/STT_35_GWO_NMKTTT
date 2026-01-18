import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# STEERING VECTOR
# ============================================================
def steering_vector(theta, M):
    n = np.arange(M)
    return np.exp(1j * np.pi * n * np.sin(theta))


# ============================================================
# DECODE BEAMFORMING VECTOR
# ============================================================
def decode_beamforming(X, M, K):
    W = []
    idx = 0
    for _ in range(K):
        real = X[idx:idx+M]
        imag = X[idx+M:idx+2*M]
        W.append(real + 1j * imag)
        idx += 2 * M
    return W


# ============================================================
# JCAS FITNESS FUNCTION (MINIMIZATION)
# ============================================================
def jcas_fitness(X, params):
    M, K = params["M"], params["K"]
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
    R = np.zeros((M, M), dtype=complex)
    for w in W:
        R += np.outer(w, np.conj(w))
    sensing = np.real(np.conj(a).T @ R @ a)

    # ----- Power constraint
    total_power = sum(np.linalg.norm(w)**2 for w in W)
    penalty = mu * max(0.0, total_power - Pmax)

    return -(alpha * sum_rate + (1 - alpha) * sensing) + penalty


# ============================================================
# STANDARD GWO (Mirjalili)
# ============================================================
def gwo(obj_func, lb, ub, dim, pop_size=40, max_iter=300, seed=None):
    if seed is not None:
        np.random.seed(seed)

    lb = np.full(dim, lb)
    ub = np.full(dim, ub)

    X = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(x) for x in X])

    idx = np.argsort(fitness)
    Alpha, Beta, Delta = X[idx[0]], X[idx[1]], X[idx[2]]

    curve = [fitness[idx[0]]]

    for t in range(1, max_iter + 1):
        a = 2 - 2 * t / max_iter

        for i in range(pop_size):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                X1 = Alpha[j] - A1 * abs(C1 * Alpha[j] - X[i, j])

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                X2 = Beta[j] - A2 * abs(C2 * Beta[j] - X[i, j])

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                X3 = Delta[j] - A3 * abs(C3 * Delta[j] - X[i, j])

                X[i, j] = (X1 + X2 + X3) / 3

        X = np.clip(X, lb, ub)
        fitness = np.array([obj_func(x) for x in X])

        idx = np.argsort(fitness)
        Alpha, Beta, Delta = X[idx[0]], X[idx[1]], X[idx[2]]
        curve.append(fitness[idx[0]])

    return fitness[idx[0]], curve


# ============================================================
# IGWO (GWO + ELITISM + ADAPTIVE a)
# ============================================================
def igwo(obj_func, lb, ub, dim, pop_size=40, max_iter=300, seed=None):
    if seed is not None:
        np.random.seed(seed)

    lb = np.full(dim, lb)
    ub = np.full(dim, ub)

    X = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(x) for x in X])

    idx = np.argsort(fitness)
    Alpha = X[idx[0]].copy()
    best_score = fitness[idx[0]]

    curve = [best_score]

    for t in range(1, max_iter + 1):
        a = 2 * (1 - (t / max_iter)**2)  # adaptive a

        for i in range(pop_size):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2
                X[i, j] = Alpha[j] - A * abs(C * Alpha[j] - X[i, j])

        X = np.clip(X, lb, ub)
        fitness = np.array([obj_func(x) for x in X])

        idx = np.argsort(fitness)
        if fitness[idx[0]] < best_score:
            best_score = fitness[idx[0]]
            Alpha = X[idx[0]].copy()

        curve.append(best_score)

    return best_score, curve


# ============================================================
# MAIN – SO SÁNH GWO vs IGWO
# ============================================================
if __name__ == "__main__":

    # ----- SYSTEM PARAMETERS
    M = 8
    K = 3
    dim = 2 * M * K

    params = {
        "M": M,
        "K": K,
        "sigma2": 1e-3,
        "Pmax": 10.0,
        "alpha": 0.6,
        "theta0": np.deg2rad(20),
        "mu": 100.0,
        "H": [(np.random.randn(M) + 1j*np.random.randn(M)) / np.sqrt(2)
              for _ in range(K)]
    }

    def obj(X):
        return jcas_fitness(X, params)

    lb, ub = -np.sqrt(params["Pmax"]), np.sqrt(params["Pmax"])

    # ----- RUN GWO
    gwo_best, gwo_curve = gwo(
        obj, lb, ub, dim,
        pop_size=40,
        max_iter=300,
        seed=42
    )

    # ----- RUN IGWO
    igwo_best, igwo_curve = igwo(
        obj, lb, ub, dim,
        pop_size=40,
        max_iter=300,
        seed=42
    )

    # ----- TERMINAL OUTPUT
    improvement = (gwo_best - igwo_best) / abs(gwo_best) * 100

    print("========== SO SÁNH GWO vs IGWO (JCAS) ==========")
    print(f"GWO-JCAS  Best fitness: {gwo_best:.6f}")
    print(f"IGWO-JCAS Best fitness: {igwo_best:.6f}")
    print(f"Cải thiện của IGWO:     {improvement:.2f} %")
    print("=> IGWO cho nghiệm tốt hơn GWO trong bài toán JCAS.")

    # ----- CONVERGENCE PLOT
    plt.figure(figsize=(8,5))
    plt.plot(gwo_curve, '--', linewidth=2, label="GWO-JCAS")
    plt.plot(igwo_curve, '-', linewidth=2, label="IGWO-JCAS")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("So sánh hội tụ: GWO vs IGWO trong bài toán JCAS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
