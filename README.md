# 🏥 viegrand_HAR
### Fall Detection & Stroke Risk Assessment System using Edge Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Prototype-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> Hệ thống xử lý trung tâm cho thiết bị đeo tay thông minh (Smart Wristband), sử dụng thuật toán Random Forest để phát hiện té ngã với độ chính xác **>99%** và cảnh báo nguy cơ đột quỵ dựa trên phân tích trạng thái bất động sau ngã.

---

## 📖 Mục lục
- [Giới thiệu](#-giới-thiệu)
- [Tính năng nổi bật](#-tính-năng-nổi-bật)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt & Sử dụng](#-cài-đặt--sử-dụng)
- [Hiệu năng Model](#-hiệu-năng-model)
- [Liên hệ](#-liên-hệ)

---

## 🚀 Giới thiệu
Dự án này là phần lõi AI (AI Backend) phục vụ cho hệ sinh thái **VieGrand** - giải pháp chăm sóc người cao tuổi. Hệ thống nhận dữ liệu thô (Raw Data) từ cảm biến **MPU6050** thông qua vi điều khiển **ESP32**, xử lý tín hiệu và đưa ra cảnh báo thời gian thực.

## ✨ Tính năng nổi bật
*   **Real-time Processing:** Xử lý dữ liệu streaming từ cảm biến với độ trễ thấp.
*   **Advanced Feature Extraction:** Trích xuất 14 đặc trưng vật lý (Jerk, SVM, Tilt Angle...) tối ưu cho thiết bị đeo cổ tay.
*   **Two-stage Analysis:**
    1.  **Fall Detection:** Phát hiện cú ngã (Accuracy 99.26%).
    2.  **Stroke Risk:** Đánh giá sự bất động sau ngã (Post-fall immobility).
*   **Robustness:** Loại bỏ báo động giả từ các hành động mạnh (vỗ tay, đập bàn).

---

## 📂 Cấu trúc dự án

```text
VieGrand-HAR/
├── models/                 
│   ├── fall_detection_rf.pkl  
│   ├── scaler.pkl             
│   └── feature_names.pkl      
│
├── notebooks/              
│   ├── 01_Data_Analysis.ipynb 
│   └── 02_Model_Training.ipynb
│
├── server/                 
│   ├── app.py              
│   └── utils.py            
│
├── requirements.txt        
└── README.md               
