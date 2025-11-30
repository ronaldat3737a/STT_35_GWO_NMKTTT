import numpy as np

def gwo_mirjalili_strict(obj_func, lb, ub, dim, pop_size=30, max_iter=500, seed=None, verbose=False):
    """
    Hàm thuật toán GWO chuẩn theo Mirjalili (2014) với các đặc điểm:
      - Đánh giá quần thể khởi tạo TRƯỚC khi vào vòng lặp (Pre-evaluation).
      - Cập nhật Alpha/Beta/Delta CHỈ SAU KHI toàn bộ bầy đã di chuyển và được đánh giá lại.
      - Sử dụng Vector hóa (Vectorization) để tính toán ma trận nhanh gấp nhiều lần vòng lặp for.
      - Lịch trình giảm tham số 'a' chuẩn: a = 2 - 2*(t/max_iter).
    """
    
    # 1. THIẾT LẬP CƠ BẢN
    # Thiết lập hạt giống ngẫu nhiên để kết quả có thể tái lập (nếu cần)
    if seed is not None:
        np.random.seed(seed)

    # Chuẩn hóa giới hạn dưới (lb) và trên (ub) thành dạng vector
    # Điều này giúp xử lý linh hoạt dù người dùng nhập số đơn hay mảng
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    if lb.size == 1:
        lb = np.full(dim, float(lb))
    if ub.size == 1:
        ub = np.full(dim, float(ub))
    
    # Kiểm tra kích thước đầu vào để tránh lỗi
    assert lb.shape == (dim,) and ub.shape == (dim,), "Lỗi: lb và ub phải là số đơn hoặc vector cùng kích thước chiều."

    # 2. KHỞI TẠO QUẦN THỂ (INITIALIZATION)
    # Tạo vị trí ngẫu nhiên cho toàn bộ bầy sói trong không gian tìm kiếm
    positions = lb + np.random.rand(pop_size, dim) * (ub - lb)
    positions = np.clip(positions, lb, ub) # Đảm bảo nằm trong biên

    # Đánh giá độ thích nghi (Fitness) ban đầu cho cả bầy
    # Bước này quan trọng để tìm ra Alpha, Beta, Delta ngay từ đầu
    fitness = np.array([obj_func(positions[i]) for i in range(pop_size)])

    # 3. CHỌN RA 3 CON ĐẦU ĐÀN (LEADERS)
    # Sắp xếp bầy sói theo điểm số từ tốt nhất đến tệ nhất (bài toán tối ưu hóa nhỏ nhất)
    idx = np.argsort(fitness)
    Alpha_idx = idx[0]                                   # Con tốt nhất
    Beta_idx  = idx[1] if pop_size > 1 else Alpha_idx    # Con nhì
    Delta_idx = idx[2] if pop_size > 2 else Beta_idx     # Con ba

    # Lưu lại vị trí và điểm số của 3 thủ lĩnh
    Alpha_pos = positions[Alpha_idx].copy(); Alpha_score = fitness[Alpha_idx]
    Beta_pos  = positions[Beta_idx].copy();  Beta_score  = fitness[Beta_idx]
    Delta_pos = positions[Delta_idx].copy(); Delta_score = fitness[Delta_idx]

    # Lưu lịch sử hội tụ để vẽ biểu đồ sau này
    convergence = [Alpha_score]

    # 4. VÒNG LẶP CHÍNH: QUÁ TRÌNH SĂN MỒI
    # t chạy từ 1 đến max_iter
    for t in range(1, max_iter + 1):
        
        # --- CẬP NHẬT THAM SỐ a ---
        # a giảm tuyến tính từ 2 xuống 0.
        # Khi a > 1: Ưu tiên tìm kiếm (Exploration). Khi a < 1: Ưu tiên tấn công (Exploitation).
        a = 2.0 - 2.0 * (t / float(max_iter))

        # --- TÍNH TOÁN VECTOR HÓA (VECTORIZATION) ---
        # Thay vì dùng vòng lặp, ta sinh số ngẫu nhiên cho CẢ BẦY cùng lúc.
        # r1, r2 kích thước (pop_size, dim) -> Mỗi con sói, ở mỗi chiều có r1, r2 riêng.
        
        # > Tính toán hệ số cho hướng Alpha
        r1_1 = np.random.rand(pop_size, dim); r2_1 = np.random.rand(pop_size, dim)
        A1 = 2.0 * a * r1_1 - a   # Vector A1
        C1 = 2.0 * r2_1           # Vector C1
        
        # > Tính toán hệ số cho hướng Beta
        r1_2 = np.random.rand(pop_size, dim); r2_2 = np.random.rand(pop_size, dim)
        A2 = 2.0 * a * r1_2 - a
        C2 = 2.0 * r2_2
        
        # > Tính toán hệ số cho hướng Delta
        r1_3 = np.random.rand(pop_size, dim); r2_3 = np.random.rand(pop_size, dim)
        A3 = 2.0 * a * r1_3 - a
        C3 = 2.0 * r2_3

        # --- CHUẨN BỊ DỮ LIỆU ĐỂ TÍNH TOÁN ---
        # Nhân bản vị trí của Alpha, Beta, Delta ra thành ma trận 
        # để khớp kích thước với ma trận positions của toàn bầy sói
        Alpha_mat = np.tile(Alpha_pos, (pop_size, 1))
        Beta_mat  = np.tile(Beta_pos,  (pop_size, 1))
        Delta_mat = np.tile(Delta_pos, (pop_size, 1))

        # --- CẬP NHẬT VỊ TRÍ THEO CÔNG THỨC GWO ---
        # 1. Tính khoảng cách (D) và hướng đi (X1) dựa trên Alpha
        D_alpha = np.abs(C1 * Alpha_mat - positions)
        X1 = Alpha_mat - A1 * D_alpha

        # 2. Tính khoảng cách (D) và hướng đi (X2) dựa trên Beta
        D_beta = np.abs(C2 * Beta_mat - positions)
        X2 = Beta_mat - A2 * D_beta

        # 3. Tính khoảng cách (D) và hướng đi (X3) dựa trên Delta
        D_delta = np.abs(C3 * Delta_mat - positions)
        X3 = Delta_mat - A3 * D_delta

        # 4. Vị trí mới là trung bình cộng của 3 hướng đi (Sự đồng thuận)
        # Đây là bước cập nhật cho TOÀN BỘ quần thể (bao gồm cả Alpha cũ)
        new_positions = (X1 + X2 + X3) / 3.0

        # --- XỬ LÝ BIÊN & CẬP NHẬT ---
        # Kéo sói quay lại nếu chạy ra ngoài vùng tìm kiếm
        new_positions = np.clip(new_positions, lb, ub)

        # Cập nhật vị trí mới cho cả bầy
        positions = new_positions
        
        # Đánh giá lại điểm số (Fitness) cho vị trí mới
        fitness = np.array([obj_func(positions[i]) for i in range(pop_size)])

        # --- BẦU CHỌN LẠI THỦ LĨNH (UPDATE LEADERS) ---
        # Sắp xếp lại xem ai tốt nhất thì lên làm Alpha, Beta, Delta mới
        idx = np.argsort(fitness)
        Alpha_idx = idx[0]
        Beta_idx  = idx[1] if pop_size > 1 else Alpha_idx
        Delta_idx = idx[2] if pop_size > 2 else Beta_idx

        # Cập nhật thông tin thủ lĩnh
        Alpha_pos = positions[Alpha_idx].copy(); Alpha_score = fitness[Alpha_idx]
        Beta_pos  = positions[Beta_idx].copy();  Beta_score  = fitness[Beta_idx]
        Delta_pos = positions[Delta_idx].copy(); Delta_score = fitness[Delta_idx]

        convergence.append(Alpha_score)

        # In tiến độ chạy (nếu bật verbose)
        if verbose and (t % 10 == 0 or t == 1 or t == max_iter):
            print(f"Vòng lặp {t}/{max_iter} — Tốt nhất = {Alpha_score:.6e}")

    return Alpha_pos, Alpha_score, convergence


# -----------------------------------------------------------
# PHẦN TEST: DÙNG HÀM SPHERE ĐỂ DEMO
# -----------------------------------------------------------
if __name__ == "__main__":
    
    # --- 1. ĐỊNH NGHĨA BÀI TOÁN: HÀM SPHERE ---
    # Hàm Sphere: f(x) = x1^2 + x2^2 + ... + xn^2
    # Hình dáng: Giống cái bát tô tròn.
    # Mục tiêu: Tìm đáy bát (giá trị 0) tại tọa độ [0, 0, ..., 0].

    def sphere_function(x):
        return np.sum(x**2)

    # --- 2. CẤU HÌNH THAM SỐ ---
    dim = 10            # Số chiều (10 biến số cần tìm)
    lb = -100.0         # Giới hạn dưới (-100)
    ub = 100.0          # Giới hạn trên (100)
    pop_size = 30       # Bầy sói 30 con
    max_iter = 500      # Chạy 500 vòng (Đủ để về 0)
    seed = 42           # Khóa kết quả (Để chạy lại vẫn ra y hệt)

    print(f"\n--- BẮT ĐẦU CHẠY DEMO VỚI HÀM SPHERE ({dim} CHIỀU) ---")
    
    # --- 3. GỌI HÀM GWO ---
    best_pos, best_score, curve = gwo_mirjalili_strict(
        sphere_function, lb, ub, dim,
        pop_size=pop_size, max_iter=max_iter, seed=seed, verbose=True
    )

    # --- 4. IN KẾT QUẢ ---
    print("\n" + "="*50)
    print(" KẾT QUẢ CUỐI CÙNG")
    print("="*50)
    
    # In ra dạng số khoa học (e-xx). 
    # Ví dụ: 6.42e-32 nghĩa là 0.000...(31 số 0)...642 -> Gần như bằng 0 tuyệt đối.
    print(f"Giá trị lỗi thấp nhất (Best Score): {best_score:.5e}") 
    
    # Đánh giá xem kết quả có 'Xịn' không
    if best_score < 1e-10:
        print("=> ĐÁNH GIÁ: TUYỆT VỜI! Đã tìm được chính xác đáy bát.")
    else:
        print("=> ĐÁNH GIÁ: Tốt, nhưng chưa hội tụ hoàn toàn.")
        
    print("-" * 50)
    # In tọa độ. Nó phải xấp xỉ [0, 0, 0, ...]
    print("Vị trí tối ưu tìm được (Tọa độ):")
    print(np.round(best_pos, 5)) # Làm tròn 5 số thập phân cho dễ nhìn