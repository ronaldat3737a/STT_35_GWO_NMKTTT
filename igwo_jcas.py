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
# LEVY FLIGHT (NHẸ – CHỈ ĐẦU KỲ)
# ============================================================

def levy_flight(dim, beta=1.5):
    sigma = (math.gamma(1+beta) * math.sin(math.pi*beta/2) /
            (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / (np.abs(v)**(1/beta))


# ============================================================
# SAFE IGWO FOR JCAS
# ============================================================

def igwo(obj_func, lb, ub, dim,
         pop_size=40, max_iter=300, seed=0):

    np.random.seed(seed)
    lb, ub = np.full(dim, lb), np.full(dim, ub)

    X = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(x) for x in X])

    idx = np.argsort(fitness)
    Alpha, Beta, Delta = X[idx[0]], X[idx[1]], X[idx[2]]
    best = Alpha.copy()
    best_score = fitness[idx[0]]

    curve = [best_score]

    for t in range(1, max_iter+1):

        # Adaptive a (ổn định)
        a = 2 * (1 - t/max_iter)

        for i in range(pop_size):
            for leader in [Alpha, Beta, Delta]:
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2*a*r1 - a
                C = 2*r2
                D = np.abs(C*leader - X[i])
                X[i] += (leader - A*D) / 3

        # Levy chỉ 10% đầu
        if t < 0.1*max_iter:
            step = levy_flight(dim)
            X[0] += 0.01 * step * (ub - lb)

        X = np.clip(X, lb, ub)
        fitness = np.array([obj_func(x) for x in X])

        idx = np.argsort(fitness)
        Alpha, Beta, Delta = X[idx[0]], X[idx[1]], X[idx[2]]

        if fitness[idx[0]] < best_score:
            best_score = fitness[idx[0]]
            best = Alpha.copy()

        curve.append(best_score)

    return best, best_score, curve


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

    # Communication: sum-rate
    sum_rate = 0
    for k in range(K):
        signal = np.abs(np.vdot(H[k], W[k]))**2
        interf = sum(np.abs(np.vdot(H[k], W[j]))**2
                     for j in range(K) if j != k)
        sinr = signal / (interf + sigma2)
        sum_rate += np.log2(1 + sinr)

    # Radar sensing
    a = steering_vector(theta0, M)
    R = sum(np.outer(w, np.conj(w)) for w in W)
    sensing = np.real(a.conj().T @ R @ a)

    # Power penalty
    power = sum(np.linalg.norm(w)**2 for w in W)
    penalty = mu * max(0, power - Pmax)

    return -(alpha*sum_rate + (1-alpha)*sensing) + penalty


# ============================================================
# PLOT CONVERGENCE
# ============================================================

def plot_convergence(curve):
    plt.figure()
    plt.plot(curve, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("IGWO Convergence for JCAS")
    plt.grid()
    plt.show()


# ============================================================
# PLOT BEAM PATTERN (θ0 = 20°)
# ============================================================

def plot_beam(best, M, K, theta0):

    theta = np.linspace(-np.pi/2, np.pi/2, 720)
    W = decode_beamforming(best, M, K)

    R = sum(np.outer(w, np.conj(w)) for w in W)

    beam = []
    for t in theta:
        a = steering_vector(t, M)
        beam.append(np.real(a.conj().T @ R @ a))

    beam = np.array(beam)
    beam_db = 10*np.log10(beam/np.max(beam))

    plt.figure(figsize=(7,7))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, beam_db, linewidth=2)
    ax.plot([theta0, theta0], [-40, 0], 'r--', linewidth=2)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetalim(-np.pi/2, np.pi/2)
    ax.set_rlim(-40, 0)
    ax.set_title("Optimized JCAS Beam Pattern (θ₀ = 20°)")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    M, K = 8, 3
    dim = 2*M*K

    params = {
        "M": M,
        "K": K,
        "sigma2": 1e-3,
        "Pmax": 10,
        "alpha": 0.6,
        "theta0": np.deg2rad(20),
        "mu": 100,
        "H": [(np.random.randn(M)+1j*np.random.randn(M))/np.sqrt(2)
              for _ in range(K)]
    }

    def obj(x): return jcas_fitness(x, params)

    best, score, curve = igwo(
        obj,
        lb=-np.sqrt(params["Pmax"]),
        ub=np.sqrt(params["Pmax"]),
        dim=dim
    )

    print("Best fitness:", score)
    plot_convergence(curve)
    plot_beam(best, M, K, params["theta0"])
