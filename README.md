So Sánh Thuật Toán Grey Wolf Optimizer (GWO) & Improved GWO (IGWO)Dự án này triển khai và so sánh hiệu năng giữa thuật toán GWO Cơ bản và phiên bản GWO Cải tiến (IGWO) trên các bài toán tối ưu hóa khác nhau. Đây là mã nguồn phục vụ cho báo cáo môn Nhập môn Kĩ thuật Truyền thông - Nhóm 35.👥 Thành viên thực hiệnNguyễn Công Đạt - MSSV: 20236023Nguyễn Mạnh Hùng - MSSV: 20236033📂 Danh sách Mã nguồnDự án bao gồm 2 file chính:1. gwo_original_sphere.py (GWO Cơ bản)Mô tả: Triển khai thuật toán GWO chuẩn theo bài báo gốc của Mirjalili (2014).Bài toán: Hàm Sphere (Hàm lồi đơn giản, đáy tại 0).Đặc điểm kỹ thuật:Sử dụng tham số $a$ giảm tuyến tính từ 2 xuống 0.Cơ chế tìm kiếm dựa trên trung bình cộng vị trí của Alpha, Beta, Delta.2. gwo_improved_ackley.py (IGWO Cải tiến)Mô tả: Phiên bản nâng cấp tích hợp nhiều kỹ thuật hiện đại để giải quyết các bài toán "hóc búa".Bài toán: Hàm Ackley (Hàm phức tạp với nhiều cực trị địa phương, rất khó hội tụ về tâm).Các kỹ thuật cải tiến (Highlights):✅ Chaotic Maps (Logistic Map): Thay thế số ngẫu nhiên thường bằng dãy số hỗn loạn để tăng tính đa dạng cho quần thể.✅ Lévy Flight: Sử dụng bước nhảy vọt (theo phân phối Lévy) giúp Alpha thoát khỏi các bẫy cục bộ (Local Minima).✅ Elitism (Bảo toàn tinh hoa): Lưu trữ nghiệm tốt nhất lịch sử (best_so_far) và chèn lại vào quần thể nếu nó bị mất đi qua các vòng lặp.✅ Non-linear a: Tham số $a$ giảm theo hàm phi tuyến $2(1 - t^2)$, giúp kéo dài thời gian tìm kiếm (Exploration) ở giai đoạn đầu.⚙️ Cài đặt Môi trườngCode yêu cầu Python 3 và thư viện NumPy.Cài đặt thư viện:pip install numpy

🚀 Hướng dẫn Chạy DemoChạy thuật toán GốcKiểm tra khả năng hội tụ cơ bản:python gwo_original_sphere.py

Chạy thuật toán Cải tiến (IGWO)Kiểm tra khả năng thoát bẫy trên hàm Ackley:python gwo_improved_ackley.py

# Kết quả kỳ vọng: IGWO sẽ tìm được giá trị lỗi cực thấp (gần 0) trên hàm Ackley, chứng minh hiệu quả của các kỹ thuật cải tiến so với thuật toán gốc.📊 Bảng so sánh tóm tắt| Tính năng | GWO Cơ bản | IGWO Cải tiến || Sinh số ngẫu nhiên | random.rand (Đều) | Chaotic Sequence (Hỗn loạn) || Cơ chế thoát bẫy | Không có | Lévy Flight (Nhảy cóc) || Lưu nghiệm tốt nhất | Không (chỉ lưu Alpha hiện tại) | Elitism (Lưu Best-so-far) || Tham số hội tụ $a$ | Tuyến tính | Phi tuyến (Mềm dẻo hơn) || Độ phức tạp | Thấp | Cao |Nhóm 35 - Nhập môn Kĩ thuật Truyền thông

# Thuật Toán Grey Wolf Optimizer (GWO) & Improved GWO (IGWO)

Dự án này triển khai thuật toán **GWO Cơ bản** và **GWO Cải tiến (IGWO)** trên các bài toán tối ưu hóa khác nhau. Đây là mã nguồn phục vụ cho báo cáo môn _Nhập môn Kĩ thuật Truyền thông - Nhóm 35_.

---

## 👥 Thành viên thực hiện

- **Nguyễn Công Đạt** – MSSV: _20236023_
- **Nguyễn Mạnh Hùng** – MSSV: _20236033_

---

## 📂 Danh sách Mã nguồn

Dự án bao gồm 2 file chính:

### 1. `gwo_original_sphere.py` (GWO Cơ bản)

**Mô tả:** Triển khai thuật toán GWO chuẩn theo bài báo gốc của _Mirjalili (2014)_.

**Bài toán:** Hàm _Sphere_ (hàm lồi đơn giản, đáy tại 0).

**Đặc điểm kỹ thuật:**

- Sử dụng tham số $a$ giảm tuyến tính từ 2 xuống 0.
- Cơ chế tìm kiếm dựa trên trung bình cộng vị trí của **Alpha**, **Beta**, **Delta**.

---

### 2. `gwo_improved_ackley.py` (IGWO Cải tiến)

**Mô tả:** Phiên bản nâng cấp tích hợp nhiều kỹ thuật hiện đại để giải quyết các bài toán _hóc búa_.

**Bài toán:** Hàm _Ackley_ (hàm phức tạp nhiều cực trị địa phương, rất khó hội tụ về tâm).

#### 🔥 Các kỹ thuật cải tiến (Highlights):

- ✅ **Chaotic Maps (Logistic Map):** thay thế random bằng dãy hỗn loạn.
- ✅ **Lévy Flight:** bước nhảy lớn giúp thoát local minima.
- ✅ **Elitism:** bảo toàn nghiệm tốt nhất lịch sử.
- ✅ **Non-linear a:** $a = 2(1 - t^2)$ kéo dài thời gian exploration.

---

## ⚙️ Cài đặt Môi trường

Yêu cầu Python 3 + NumPy.

Cài đặt thư viện:

```bash
pip install numpy
```

---

## 🚀 Hướng dẫn Chạy Demo

### Chạy thuật toán Gốc (GWO)

```bash
python gwo_original_sphere.py
```

Kiểm tra khả năng hội tụ cơ bản.

### Chạy thuật toán Cải tiến (IGWO)

```bash
python gwo_improved_ackley.py
```

Kiểm tra khả năng thoát bẫy trên hàm Ackley.

---
