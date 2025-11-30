# Thuật Toán Grey Wolf Optimizer (GWO) & Improved GWO (IGWO)

Dự án này triển khai thuật toán **GWO Cơ bản** và **GWO Cải tiến (IGWO)** trên các bài toán tối ưu hóa khác nhau. Đây là mã nguồn phục vụ cho báo cáo môn *Nhập môn Kĩ thuật Truyền thông - Nhóm 35*.

---

## 👥 Thành viên thực hiện
- **Nguyễn Công Đạt** – MSSV: *20236023*
- **Nguyễn Mạnh Hùng** – MSSV: *20236033*

---

## 📂 Danh sách Mã nguồn
Dự án bao gồm 2 file chính:

### 1. `gwo_original_sphere.py` (GWO Cơ bản)
**Mô tả:** Triển khai thuật toán GWO chuẩn theo bài báo gốc của *Mirjalili (2014)*.

**Bài toán:** Hàm *Sphere* (hàm lồi đơn giản, đáy tại 0).

**Đặc điểm kỹ thuật:**
- Sử dụng tham số $a$ giảm tuyến tính từ 2 xuống 0.
- Cơ chế tìm kiếm dựa trên trung bình cộng vị trí của **Alpha**, **Beta**, **Delta**.

---

### 2. `gwo_improved_ackley.py` (IGWO Cải tiến)
**Mô tả:** Phiên bản nâng cấp tích hợp nhiều kỹ thuật hiện đại để giải quyết các bài toán *hóc búa*.

**Bài toán:** Hàm *Ackley* (hàm phức tạp nhiều cực trị địa phương, rất khó hội tụ về tâm).

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
