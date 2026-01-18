import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEERING VECTOR & DECODER
# ============================================================

def steering_vector(theta, M):
    n = np.arange(M)
    return np.exp(1j * np.pi * n * np.sin(theta))


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
# GWO – MIRJALILI (2014) (BẢN CHUẨN ĐƠN GIẢN)
# ============================================================

def gwo_mirjalili(obj_func, lb, ub, dim,
                  pop_size=40, max_iter=300, seed=None):

    if seed is not None:
        np.random.seed(seed)

    lb = np.full(dim, lb)
    ub = np.full(dim, ub)

    # Khởi tạo quần thể
    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(p) for p in positions])

    # Alpha
    idx = np.argsort(fitness)
    Alpha = positions[idx[0]].copy()
    Alpha_score = fitness[idx[0]]

    convergence = [Alpha_score]

    # Vòng lặp chính
    for t in range(1, max_iter + 1):
        a = 2 - 2 * t / max_iter

        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A = 2 * a * r1 - a
            C = 2 * r2

            D = np.abs(C * Alpha - positions[i])
            positions[i] = Alpha - A * D

        positions = np.clip(positions, lb, ub)
        fitness = np.array([obj_func(p) for p in positions])

        idx = np.argsort(fitness)
        Alpha = positions[idx[0]].copy()
        Alpha_score = fitness[idx[0]]

        convergence.append(Alpha_score)

    return Alpha, Alpha_score, convergence


# ============================================================
# JCAS FITNESS FUNCTION
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

    # ----- Communication: Sum-rate -----
    sum_rate = 0.0
    for k in range(K):
        hk = H[k]
        signal = np.abs(np.vdot(hk, W[k]))**2
        interf = sum(np.abs(np.vdot(hk, W[j]))**2
                     for j in range(K) if j != k)
        sinr = signal / (interf + sigma2)
        sum_rate += np.log2(1 + sinr)

    # ----- Radar sensing -----
    a = steering_vector(theta0, M)
    R = np.zeros((M, M), dtype=complex)
    for w in W:
        R += np.outer(w, np.conj(w))

    sensing = np.real(np.conj(a).T @ R @ a)

    # ----- Power constraint -----
    power = sum(np.linalg.norm(w)**2 for w in W)
    penalty = mu * max(0, power - Pmax)

    # ----- Final fitness (minimize) -----
    return -(alpha * sum_rate + (1 - alpha) * sensing) + penalty


# ============================================================
# OUTPUT 2 – CONVERGENCE PLOT
# ============================================================

def plot_convergence(curve):
    plt.figure()
    plt.plot(curve, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("GWO Convergence for JCAS")
    plt.grid()
    plt.show()


# ============================================================
# OUTPUT 3 – BEAM PATTERN (POLAR, ĐÚNG θ0)
# ============================================================

def plot_beam_pattern(best_pos, M, K, theta0):
    theta_scan = np.linspace(-np.pi/2, np.pi/2, 720)
    W = decode_beamforming(best_pos, M, K)

    # Ma trận hiệp phương sai
    R = np.zeros((M, M), dtype=complex)
    for w in W:
        R += np.outer(w, np.conj(w))

    beam = []
    for t in theta_scan:
        a = steering_vector(t, M)
        beam.append(np.real(np.conj(a).T @ R @ a))

    beam = np.array(beam)
    beam_db = 10 * np.log10(beam / np.max(beam))

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    ax.plot(theta_scan, beam_db, linewidth=2, label="Optimized Beam Pattern")
    ax.plot([theta0, theta0], [-40, 0], "r--", linewidth=2,
            label=r"Radar target $\theta_0 = 20^\circ$")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetalim(-np.pi/2, np.pi/2)
    ax.set_rlim(-40, 0)
    ax.set_title("Optimized JCAS Beam Pattern (GWO)")
    ax.legend(loc="lower left")

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ===== THAM SỐ =====
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

    # ===== RUN GWO =====
    best_pos, best_score, curve = gwo_mirjalili(
        obj,
        lb=-np.sqrt(params["Pmax"]),
        ub=np.sqrt(params["Pmax"]),
        dim=dim,
        pop_size=40,
        max_iter=300,
        seed=42
    )

    # ===== OUTPUT 1 =====
    print("Best fitness:", best_score)

    # ===== OUTPUT 2 =====
    plot_convergence(curve)

    # ===== OUTPUT 3 =====
    plot_beam_pattern(best_pos, M, K, params["theta0"])
