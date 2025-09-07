# 🐱 Face Recognition System
#### A real-time face recognition system built with Python and OpenCV.
#### Register, train, and recognize faces via webcam in a few simple steps! 🧑‍🤝‍🧑
---

## ✨ Features

#### 🆕 Register New Person – Capture faces and store personal details
#### 🤖 Train Model – Train the LBPH face recognizer
#### 👀 Recognize Person – Real-time detection & recognition
#### 🗑 Delete All Data – Clear all faces, CSV, and trained models
#### 💻 Interactive CLI – Easy-to-use menu interface

---

# 📁 Project Structure
```bash
.
├── main.py               # Main menu
├── register.py           # Register new person
├── train.py              # Train face recognition model
├── recognize.py          # Recognize faces in real-time
├── delete_data.py        # Delete all stored data
├── utils.py              # Utility functions & constants
├── faces/                # Folder for captured face images
├── details.csv           # CSV file with person details
└── trainer.yml           # Saved trained model

```
---

# ⚙ Installation

### 1. Clone the repository:
```bash
git clone https://github.com/<your-username>/face-recognition-system.git
cd face-recognition-system
```
### 2. Install dependencies:
```bash
pip install opencv-python opencv-contrib-python pandas numpy
```
---

# 🏃 Usage
Run the main program:
```bash
python main.py
```

# Interactive menu:

1. Register new person
2. Train model
3. Recognize person
4. Delete all stored data
5. Exit

Register: Capture your face & save details.
Train: Train LBPH model on stored faces.
Recognize: Identify faces from webcam feed.
Delete Data: Remove all images, CSV, and model.

> Press Q in OpenCV windows to quit any process.

---

# 🧩 How it Works

1. Face Detection: OpenCV Haar Cascade detects faces.
2. Face Recognition: LBPH algorithm predicts registered faces.
3. Data Storage: Details saved in details.csv and face images in faces/.
4. Lockout Mechanism: Avoid repeated recognition prints within 5 seconds.

# 📝 License
MIT License © 2025
You are free to use, modify, and distribute this project.
