import numpy as np
import math

def _levy_flight(step_dim, beta=1.5):
    # Mantegna's algorithm
    # returns 1D array of size step_dim
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(step_dim) * sigma_u
    v = np.random.randn(step_dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step

def _chaotic_sequence(size, x0=0.7):
    # logistic map chaotic generator in (0,1)
    seq = np.empty(size)
    x = float(x0)
    for i in range(size):
        x = 4.0 * x * (1.0 - x)
        seq[i] = x
        # avoid stuck at 0 or 1
        if x == 0 or x == 1:
            x = 0.3749
    return seq

def gwo_igwo(obj_func, lb, ub, dim, pop_size=30, max_iter=500,
             seed=None, verbose=False, use_levy=True, levy_beta=1.5,
             chaos=False, elitism=True):
    """
    Improved GWO (IGWO) — giữ lõi Mirjalili nhưng thêm:
      - Elitism: lưu best_so_far và chèn lại nếu bị mất
      - Adaptive a: phi tuyến (mềm hơn tuyến tính)
      - Lévy flight perturbation cho Alpha (thỉnh thoảng)
      - Tùy chọn: chaotic RNG cho r1/r2 (logistic map)
    """
    if seed is not None:
        np.random.seed(seed)

    lb = np.asarray(lb); ub = np.asarray(ub)
    if lb.size == 1:
        lb = np.full(dim, float(lb))
    if ub.size == 1:
        ub = np.full(dim, float(ub))
    assert lb.shape == (dim,) and ub.shape == (dim,), "lb/ub must match dim"

    # init positions
    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    positions = np.clip(positions, lb, ub)

    # initial fitness & leaders
    fitness = np.array([obj_func(positions[i]) for i in range(pop_size)])
    idx = np.argsort(fitness)
    Alpha_idx = idx[0]; Beta_idx = idx[1] if pop_size>1 else Alpha_idx; Delta_idx = idx[2] if pop_size>2 else Beta_idx

    Alpha_pos = positions[Alpha_idx].copy(); Alpha_score = fitness[Alpha_idx]
    Beta_pos  = positions[Beta_idx].copy();  Beta_score  = fitness[Beta_idx]
    Delta_pos = positions[Delta_idx].copy(); Delta_score = fitness[Delta_idx]

    # global best (elitism)
    best_so_far_pos = Alpha_pos.copy()
    best_so_far_score = Alpha_score

    convergence = [best_so_far_score]

    # logistic map seed for chaos
    chaotic_x0 = 0.61803398875  # phi-1 seed (just a value in (0,1))
    for t in range(1, max_iter + 1):
        # adaptive 'a' (phi tuyến): nhẹ nhàng hơn tuyến tính
        r_t = t / float(max_iter)
        a = 2.0 * (1.0 - r_t ** 2)  # alternative to linear: 2*(1-(t/max)^2)

        # generate random matrices r1/r2 either via chaos or uniform
        if chaos:
            total = pop_size * dim
            seq1 = _chaotic_sequence(total, x0=chaotic_x0 + 0.001 * t)
            seq2 = _chaotic_sequence(total, x0=chaotic_x0 + 0.003 * t)
            r1_1 = seq1.reshape(pop_size, dim)
            r2_1 = seq2.reshape(pop_size, dim)
            # for A2/C2 and A3/C3 generate more sequences (shift seeds)
            seq3 = _chaotic_sequence(total, x0=chaotic_x0 + 0.007 * t)
            seq4 = _chaotic_sequence(total, x0=chaotic_x0 + 0.011 * t)
            r1_2 = seq3.reshape(pop_size, dim); r2_2 = seq4.reshape(pop_size, dim)
            seq5 = _chaotic_sequence(total, x0=chaotic_x0 + 0.013 * t)
            seq6 = _chaotic_sequence(total, x0=chaotic_x0 + 0.017 * t)
            r1_3 = seq5.reshape(pop_size, dim); r2_3 = seq6.reshape(pop_size, dim)
        else:
            r1_1 = np.random.rand(pop_size, dim); r2_1 = np.random.rand(pop_size, dim)
            r1_2 = np.random.rand(pop_size, dim); r2_2 = np.random.rand(pop_size, dim)
            r1_3 = np.random.rand(pop_size, dim); r2_3 = np.random.rand(pop_size, dim)

        A1 = 2.0 * a * r1_1 - a; C1 = 2.0 * r2_1
        A2 = 2.0 * a * r1_2 - a; C2 = 2.0 * r2_2
        A3 = 2.0 * a * r1_3 - a; C3 = 2.0 * r2_3

        Alpha_mat = np.tile(Alpha_pos, (pop_size, 1))
        Beta_mat  = np.tile(Beta_pos,  (pop_size, 1))
        Delta_mat = np.tile(Delta_pos, (pop_size, 1))

        D_alpha = np.abs(C1 * Alpha_mat - positions)
        X1 = Alpha_mat - A1 * D_alpha

        D_beta = np.abs(C2 * Beta_mat - positions)
        X2 = Beta_mat - A2 * D_beta

        D_delta = np.abs(C3 * Delta_mat - positions)
        X3 = Delta_mat - A3 * D_delta

        new_positions = (X1 + X2 + X3) / 3.0
        new_positions = np.clip(new_positions, lb, ub)
        positions = new_positions

        # evaluate
        fitness = np.array([obj_func(positions[i]) for i in range(pop_size)])

        # update global best (elitism)
        current_min_idx = np.argmin(fitness)
        current_min_score = fitness[current_min_idx]
        if current_min_score < best_so_far_score:
            best_so_far_score = current_min_score
            best_so_far_pos = positions[current_min_idx].copy()

        # if best_so_far disappeared (worse than current worst), re-insert it
        if elitism:
            worst_idx = np.argmax(fitness)
            if best_so_far_score < fitness[worst_idx]:
                # replace worst with best_so_far
                positions[worst_idx] = best_so_far_pos.copy()
                fitness[worst_idx] = best_so_far_score

        # Optional: Lévy perturbation on Alpha to escape local minima occasionally
        if use_levy:
            if np.random.rand() < 0.2:  # 20% chance each iteration
                step = _levy_flight(dim, beta=levy_beta)
                # scale step relative to search range: small factor
                scale = 0.01 * (ub - lb)
                alpha_perturb = best_so_far_pos + step * scale
                alpha_perturb = np.clip(alpha_perturb, lb, ub)
                # insert perturbation into population by replacing worst (or keep separate)
                worst_idx = np.argmax(fitness)
                positions[worst_idx] = alpha_perturb
                fitness[worst_idx] = obj_func(positions[worst_idx])
                # update best_so_far if improved
                if fitness[worst_idx] < best_so_far_score:
                    best_so_far_score = fitness[worst_idx]
                    best_so_far_pos = positions[worst_idx].copy()

        # re-calc leaders based on possibly updated fitness
        idx = np.argsort(fitness)
        Alpha_idx = idx[0]; Beta_idx = idx[1] if pop_size>1 else Alpha_idx; Delta_idx = idx[2] if pop_size>2 else Beta_idx
        Alpha_pos = positions[Alpha_idx].copy(); Alpha_score = fitness[Alpha_idx]
        Beta_pos  = positions[Beta_idx].copy();  Beta_score  = fitness[Beta_idx]
        Delta_pos = positions[Delta_idx].copy(); Delta_score = fitness[Delta_idx]

        # ensure best_so_far is tracked (safeguard)
        if best_so_far_score < Alpha_score:
            # optionally make best_so_far the Alpha for reporting
            Alpha_pos = best_so_far_pos.copy(); Alpha_score = best_so_far_score

        convergence.append(best_so_far_score)

        if verbose and (t % 10 == 0 or t == 1 or t == max_iter):
            print(f"[IGWO] Iter {t}/{max_iter} — best = {best_so_far_score:.6e}")

    return best_so_far_pos, best_so_far_score, convergence

# -----------------------------------------------------------
# BÀI TOÁN THỬ THÁCH: HÀM ACKLEY (KHÓ)
# -----------------------------------------------------------
if __name__ == "__main__":
    
    # --- 1. ĐỊNH NGHĨA HÀM MỤC TIÊU: ACKLEY ---
    # Global Minimum = 0 tại vị trí [0, 0, ..., 0]
    def ackley_function(x):
        dim = len(x)
        # Phần 1: Tính trung bình bình phương (độ dốc tổng thể)
        sum_sq = np.sum(x**2)
        # Phần 2: Tính trung bình cos (độ gồ ghề lồi lõm)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        
        # Công thức Ackley: Kết hợp cả hai
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
        term2 = -np.exp(sum_cos / dim)
        
        return term1 + term2 + 20 + np.e

    # --- 2. CẤU HÌNH THAM SỐ (KHÓ) ---
    dim = 10           # Tăng lên 10 chiều (Càng nhiều chiều càng nhiều bẫy)
    lb = -32.768        # Giới hạn chuẩn của Ackley
    ub = 32.768
    pop_size = 50       # Cần đông quân hơn chút
    max_iter = 1000     # Cần thời gian dài để thoát bẫy
    seed = 42           

    print(f"\n--- BẮT ĐẦU TEST IGWO VỚI HÀM ACKLEY ({dim} CHIỀU) ---")
    print("Đang chạy... (Có thể mất vài giây vì thuật toán phức tạp hơn)")

    # --- 3. CHẠY THUẬT TOÁN IGWO ---
    # Bật hết các tính năng nâng cao: Chaos, Levy, Elitism
    best_pos, best_score, curve = gwo_igwo(
        ackley_function, lb, ub, dim,
        pop_size=pop_size, max_iter=max_iter, seed=seed, 
        verbose=True,       # Hiện thông báo
        use_levy=True,      # Bật nhảy cóc (Quan trọng để thoát hố Ackley)
        chaos=True,         # Bật hỗn loạn
        elitism=True        # Bật bảo toàn tinh hoa
    )

    # --- 4. IN KẾT QUẢ ---
    print("\n" + "="*50)
    print(" KẾT QUẢ CUỐI CÙNG (HÀM ACKLEY)")
    print("="*50)
    
    print(f"Giá trị lỗi thấp nhất (Best Score): {best_score:.10e}") 
    
    # Đánh giá độ xịn
    if best_score < 1e-10:
        print("=> ĐÁNH GIÁ: XUẤT SẮC! Đã tìm được đáy vực thẳm Ackley.")
    elif best_score < 1e-5:
        print("=> ĐÁNH GIÁ: Tốt. Đã thoát được các bẫy xa, nhưng chưa chạm đáy tâm.")
    else:
        print("=> ĐÁNH GIÁ: Khá. Vẫn bị kẹt ở các hố cục bộ gần tâm.")
        
    print("-" * 50)
    # Chỉ in 5 chiều đầu tiên để đỡ rối mắt
    print("Vị trí tối ưu (5 chiều đầu):")
    print(np.round(best_pos[:5], 5))