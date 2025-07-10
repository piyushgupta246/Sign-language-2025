
# 🤟 Sign Language Detection System 2025

An intelligent AI-powered computer vision system designed to recognize hand gestures and translate them into meaningful text using deep learning techniques. This project aims to assist individuals with hearing or speech impairments by bridging communication gaps through real-time sign language recognition.

---

![Alt text](https://github.com/piyushgupta246/ML/blob/main/1.jpg)

---

## 🔍 About

Sign language is a vital communication method for people with hearing or speech disabilities. However, the lack of widespread understanding and support tools often creates barriers in daily interactions.

This project uses **MediaPipe** and a trained **machine learning classifier** (e.g., Random Forest or CNN) to detect and interpret American Sign Language (ASL) hand gestures in real time via a webcam. It enables users to perform signs, which the system then converts into readable text on screen, promoting inclusive communication.

---

## ⚙️ Features

- 📹 Real-time hand tracking using **MediaPipe**
- 🧠 Gesture recognition with trained **ML classifier**
- 🖥️ Live webcam interface for continuous detection
- 🗂️ Support for custom-trained sign language dataset
- 🧪 Expandable for additional gestures or phrases

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/piyushgupta246/Sign-Language-Detection-2025.git

# Navigate to the project folder
cd Sign-Language-Detection-2025

# (Optional) Set up virtual environment
python -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
