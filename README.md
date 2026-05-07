# Astra Pro AMR Vision Tests

Computer vision and depth sensing experiments using the Orbbec Astra Pro camera for AMR (Autonomous Mobile Robot) development

---

# Features

- Depth streaming using OpenNI2
- RGB camera integration
- RGB + Depth visualization
- QR code detection
- Box detection
- Distance estimation
- Visual servoing
- Basic robotics vision pipeline



# Hardware

- Orbbec Astra Pro
- Windows 10/11
- USB 2.0
- Python 3.10



# Technologies

- Python
- OpenCV
- OpenNI2
- NumPy
- Pyzbar



# Installation

## 1. Clone Repository

```bash
git clone https://github.com/USERNAME/astra-pro-amr-vision.git
cd astra-pro-amr-vision
```



## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

### Windows

```powershell
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install OpenNI Python Wrapper

```bash
pip install git+https://github.com/severin-lemaignan/openni-python.git
```

---

# OpenNI2 Setup

Place OpenNI2 SDK in:

```text
C:\OpenNI2
```

The folder should contain:

```text
C:\OpenNI2\Bin
C:\OpenNI2\Include
C:\OpenNI2\Redist
```

---

# Tests

## Depth Stream

```bash
python Depth-stream.py
```

---

## RGB Camera Test

```bash
python RGBaswebcam.py
```

---

## RGB + Depth Viewer

```bash
python RGB+Depth-viewer.py
```

---

## QR Scanner (Pyzbar)

```bash
python QR-Scanner-pyzbar.py
```

---

## QR Scanner (OpenCV)

```bash
python QR-Scanner-OpenCV.py
```

---

## Box Measurement

```bash
python Box-measurement.py
```

---

## Visual Servoing

```bash
python visualservoing.py
```

---

# Future Improvements

- ROS2 integration
- Jetson Orin Nano support
- YOLO object detection
- SLAM
- Autonomous navigation
- Obstacle avoidance
- Motor control

---

# Author

Abdalla Elradi

Informatics Engineering Student  