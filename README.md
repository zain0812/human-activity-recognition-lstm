# 🧠 Human Activity Recognition using MediaPipe + LSTM

This project is a **Human Activity Recognition (HAR) system** that uses **pose estimation** and **deep learning (LSTM)** to classify human actions from video data.

It extracts body keypoints using MediaPipe, converts them into meaningful motion features, and trains an LSTM model to recognize different activities.

---

# 🚀 Project Overview

The system works in two main stages:

## 1️⃣ Feature Extraction (Computer Vision)
- Video is processed frame by frame
- Human pose landmarks are detected using MediaPipe
- 9 key features are extracted:
  - Joint angles (elbow, knee)
  - Average knee angle
  - Normalized body positions (wrist, shoulder)
- Features are saved as sequences

## 2️⃣ Model Training (Deep Learning)
- Features are grouped into sequences (30 frames each)
- LSTM model learns temporal motion patterns
- Model predicts human actions based on movement

---

# 📂 Project Structure
