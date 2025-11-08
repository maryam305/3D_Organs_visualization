<div align="center">

# 🫀 Human Body Systems Viewer

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![VTK](https://img.shields.io/badge/VTK-9.0+-red.svg)

**🩺 Interactive 3D medical visualization system for exploring human anatomy**

[✨ Features](#-features) • [📥 Installation](#-installation) • [🚀 Quick Start](#-quick-start) • [📖 Usage](#-usage) 

---

</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/46a0a135-abf9-42c7-a209-45aa5f1d6983" alt="Main Launcher" width="800"/>
</p>

---

## 📑 Table of Contents

- [🩺 Overview](#-overview)
- [✨ Features](#-features)
- [🧱 Architecture](#-architecture)
- [📥 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📖 Usage](#-usage)
- [🎮 Interactive Controls](#-interactive-controls)
- [🎨 Color Coding](#-color-coding)
- [📂 Supported Formats](#-supported-formats)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [ Acknowledgments](#-acknowledgments)
- [📧 Contact](#-contact)

---

## 🩺 Overview
This repository provides **interactive human body system viewers** using **Python/Qt** and **VTK**, plus standalone 3D **clipping** and **curved MPR** tools for NIfTI/DICOM volumes and PNG stacks — along with an embeddable ECG widget and realistic signal-driven animations.

It includes dedicated modules for:
🫀 Cardiovascular | 🧠 Nervous | 💪 Musculoskeletal | 🦷 Dental systems  
All systems are launched via a Tkinter-based main interface or standalone entry points.

---

## ✨ Features

### 🎯 Core Systems

#### 🫀 Cardiovascular System
- 💓 Real-time ECG simulation with conduction modeling  
- 🔊 Heart sounds and audio cues  
- 🗣️ Text-to-speech annotations  
- 🩸 Interactive vessel visualization  

https://github.com/user-attachments/assets/cf411a22-6278-40b3-8a52-e793f859b8e8

#### 🧠 Nervous System
- 🧬 Brain and spinal cord visualization  
- 📊 EEG-driven surface coloring  
- ✂️ Multi-plane clipping tools  
- 📐 Curved MPR (Multiplanar Reconstruction)  

https://github.com/user-attachments/assets/dea9e450-0fd2-4720-aac7-ae6f6a9e7322

#### 🦴 Musculoskeletal System
- 💪 Bone and muscle rendering  
- ⚡ Neural signal visualization with glow effects  
- 🎨 Advanced quality controls  
- ✨ Edge enhancement options  

https://github.com/user-attachments/assets/14e4e32c-3b81-4cb0-9cd5-bddac1712f44

#### 🦷 Dental System
- 😁 Teeth and jaw segmentation  
- 🎨 Color preset management  
- 🔧 Procedural tooth generation  
- 💫 Neural signal animation  

https://github.com/user-attachments/assets/349d61ec-7f28-4be6-94aa-31921308e7dc

---

---

## 🛠️ Advanced Tools

- **✂️ Clipping App**: PyVista-based 3D clipping with axial/coronal/sagittal planes  
- **📊 Curved MPR**: Interactive curved multiplanar reconstruction for NIfTI, DICOM, and PNG stacks  
- **📈 ECG Widget**: Embeddable real-time ECG display with dark theme  
- **🧪 Experimental Lab**: Blood-flow particle animation and in-scene curved paths  

---

## 📥 Installation

### ⚙️ Prerequisites

- 🐍 Python 3.10+  
- 🖥️ CPU (optional, for enhanced rendering)

### 📦 Install Dependencies

```bash
pip install PyQt5
pip install vtk
pip install pyvista
pip install pyvistaqt
pip install numpy
pip install scipy
pip install matplotlib
pip install nibabel
pip install pydicom
pip install imageio
pip install pandas
pip install pillow
pip install pyttsx3


```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 🎮 Launch Main Interface

```bash
python main.py
```

### 🏃 Run Individual Systems

```bash
# 🫀 Cardiovascular system
python cardiovascular_system.py

# 🧠 Nervous system
python nervous_system.py

# 🦴 Musculoskeletal system
python musculoskeletal_system.py

# 🦷 Dental system
python dental_system.py
```

### 🔧 Standalone Tools

```bash
# ✂️ 3D clipping tool
python clipping.py

# 📊 Curved MPR utility
python mpr.py
```

---

## 📖 Usage

### 📐 Curved MPR Tool

1. 📂 Load NIfTI/DICOM/PNG volume
2. 🎯 Select plane (axial/coronal/sagittal) and slice
3. 🖱️ Click to draw curve points on the image
4. 🎨 Generate straightened CPR image along the path
<p align="center">
 <img width="1562" height="965" alt="Screenshot 2025-11-07 233922" src="https://github.com/user-attachments/assets/5344f807-92e0-44fe-b980-300ddff4506f" />
</p>


### ✂️ Advanced Clipping

- 🎚️ Position clipping planes interactively
- 👁️ Toggle half-space visibility
- 🧭 Show anatomical plane orientations
- ⌨️ Keyboard shortcuts for precise rotation
<img width="1574" height="924" alt="Screenshot 2025-11-07 234304" src="https://github.com/user-attachments/assets/0c79b197-9012-4337-9059-c664420fa6e5" />


### 💓 ECG Simulation

The cardiovascular module includes:
- 📈 P-QRS-T wave generation
- 💗 Atrial and ventricular contraction mapping
- 🔊 Synchronized heart sounds
- 🗣️ Optional text-to-speech guidance
<p align="center">
  <img width="1563" height="872" alt="Screenshot 2025-11-07 234803" src="https://github.com/user-attachments/assets/d788f3bb-114d-408b-b1f8-5006af5da12a" />
</p>


---

## 📁 Project Structure

```
📦 human-body-systems/
├── 🚀 main.py                      # Tkinter launcher
├── 🫀 cardiovascular_system.py     # Heart & vessels with ECG
├── 🧠 nervous_system.py            # Nervous system viewer
├── 🦴 musculoskeletal_system.py    # Bones & muscles
├── 🦷 dental_system.py             # Dental visualization
├── ✂️ clipping.py                  # Standalone clipping tool
├── 📊 mpr.py                       # Curved MPR utility
├── 📈 ecg_widget.py                # Reusable ECG widget
└── 🧪 Test.py                      # Experimental features
```

---

## 📂 Supported Formats


https://github.com/user-attachments/assets/2d56d4a3-1536-4cd0-8e55-63dae8739217



| Type    | Extensions                           | Description                   |
| :------ | :----------------------------------- | :---------------------------- |
| Meshes  | `.stl`, `.obj`, `.ply`, `.vtk`       | 3D anatomical models          |
| Volumes | `.nii`, `.nii.gz`, DICOM, PNG stacks | Medical imaging volumes       |
| Audio   | `.wav`, `.mp3`                       | Heart sounds / voice guidance |

---
## 🧩Data Sources & Citations

These datasets were partially used in this project for testing, visualization, and evaluation purposes.

3D Multimodal Dental Dataset based on CBCT and Oral Scan — Figshare
A multimodal 3D dataset combining Cone Beam CT (CBCT) and intraoral scans, designed for dental anatomy and visualization research.
Please refer to the original Figshare page for citation details and licensing information.

https://figshare.com/articles/dataset/_b_3D_multimodal_dental_dataset_based_on_CBCT_and_oral_scan_b_/26965903?file=49086406

-Healthy-Total-Body-CTs — The Cancer Imaging Archive (TCIA) 
https://www.cancerimagingarchive.net/collection/healthy-total-body-cts.com

A dataset containing low-dose, whole-body CT scans of 30 healthy subjects with detailed tissue segmentation (organs, fat, muscle, etc.).

Data Citation:
Selfridge, A. R., Spencer, B., Shiyam Sundar, L. K., Abdelhafez, Y., Nardo, L., Cherry, S. R., & Badawi, R. D. (2023).
Low-Dose CT Images of Healthy Cohort (Healthy-Total-Body-CTs) (Version 2) [Dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/NC7Z-4F76


---
## 🎮 Interactive Controls
| Action            | Control          |
| :---------------- | :--------------- |
| Rotate 3D view    | Left mouse drag  |
| Pan camera        | Right mouse drag |
| Zoom              | Mouse scroll     |
| Reset camera      | `R` key          |
| Toggle clipping   | `C` key          |
| Adjust opacity    | `O` slider       |
| Enable curved MPR | `M` key          |
| Focus on region   | Double-click     |
| Toggle wireframe  | `W` key          |


## 📸Muscle

<div align="center">

|Opacity | Clipping |
|:---:|:---:|
| <img width="1562" height="923" alt="Screenshot 2025-11-07 232407" src="https://github.com/user-attachments/assets/0f9d2384-aa16-4c21-9163-bf832f3fd647" />| <img width="1577" height="944" alt="Screenshot 2025-11-07 232618" src="https://github.com/user-attachments/assets/bee77c92-1e0e-4426-9576-1568e975f17f" />|

|Opacity  | Colours |
|:---:|:---:|
|<img width="1569" height="864" alt="Screenshot 2025-11-07 232422" src="https://github.com/user-attachments/assets/fd0eda4d-8fcf-4df4-be35-59ff8efc96f5" />|<img width="1563" height="902" alt="Screenshot 2025-11-07 232522" src="https://github.com/user-attachments/assets/3617750c-25ab-451a-a4de-ca5ff11d2bf2" />|


|  Curved MPR | Navigation |
|:---:|:---:|
| <img width="1547" height="945" alt="Screenshot 2025-11-07 232739" src="https://github.com/user-attachments/assets/d71ba3a6-3578-477d-b7b8-b7b1b8ddcd43" /> | <img width="1561" height="887" alt="Screenshot 2025-11-07 232915" src="https://github.com/user-attachments/assets/b34f49a4-44ca-4807-9f0b-27cb4a2a190f" />|

</div>
</div>
</div>


## 📸Brain

<div align="center">

|Opacity | Clipping |
|:---:|:---:|
| <img width="1780" height="912" alt="Screenshot 2025-11-07 231121" src="https://github.com/user-attachments/assets/f9b78a19-09ef-4e7a-a245-6380fa28ece3" />| <img width="1768" height="876" alt="Screenshot 2025-11-07 231447" src="https://github.com/user-attachments/assets/cd3d81a6-e0ac-4fa8-b51a-b5f4b6e956e8" />|


|  Curved MPR |Focus navigation |
|:---:|:---:|
| <img width="1734" height="917" alt="Screenshot 2025-11-07 231707" src="https://github.com/user-attachments/assets/64a2b14a-0c60-4089-bebb-5a1c6b6ce061" /> | <img width="1769" height="885" alt="Screenshot 2025-11-07 232053" src="https://github.com/user-attachments/assets/2ab3a840-dad6-458f-9ff5-596923e23834" />|

</div>

---


## 📸Teeth

<div align="center">

|Opacity of upper jaw | Opacity of lower jaw  |
|:---:|:---:|
| <img width="1563" height="834" alt="Screenshot 2025-11-07 234200" src="https://github.com/user-attachments/assets/f4388025-7997-4d0d-a2d3-57c020227310" />| <img width="1583" height="791" alt="Screenshot 2025-11-07 234212" src="https://github.com/user-attachments/assets/dbbb7458-bd3f-4b00-9979-5e458ce171ff" />|
</div>

## 📸Heart
<div align="center">
  
|Opacity |Curved MPR |
|:---:|:---:|
|<img width="1580" height="900" alt="Screenshot 2025-11-07 234830" src="https://github.com/user-attachments/assets/6bda63b8-5dbd-417b-ae9e-ec76c9115113" />| <img width="1861" height="951" alt="Screenshot 2025-11-06 194052" src="https://github.com/user-attachments/assets/a58a0f19-2965-4fed-ab45-5b1be02b2cf6" />|

</div>

---

## 🤝 Contributing

Contributions are welcome! 🎉
To contribute:

Fork this repo

Create a feature branch (git checkout -b feature-name)

Commit changes (git commit -m "Add feature")

Push and open a PR

Please ensure code follows PEP8 and includes minimal documentation.

## 📜 License
This project is provided as-is for educational and research purposes. Not intended for clinical diagnostic use

##  Acknowledgments

- 🛠️ Built with VTK, PyVista, and Qt
- 🏥 Medical imaging support via nibabel and pydicom
- 💓 ECG simulation based on physiological models

---
## 📧 Contact
**Project Contributer**: Maryam Moustafa
- Email: maryam23shabaan@gmail.com
- [GitHub](https://github.com/maryam305)
- [LinkedIn](https://www.linkedin.com/in/maryam-moustafa-653257378)

**Project Contributer**: Nour Ahmed
- [GitHub](https://github.com/nourahmedmohamed1)
- [LinkedIn](https://linkedin.com/in/nn-anwar)

**Project Contributer**: Aya Sayed
- Email: aya.sayed14827@gmail.com
- [GitHub](https://github.com/14930)
- [LinkedIn](https://www.linkedin.com/in/aya-sayed-bb6a80397?utm_source=share_via&utm_content=profile&utm_medium=member_android)

**Project Contributer**: Mahmoud Mazen
- [GitHub](https://github.com/MahmoudMazen0)




---

<div align="center">



</div>
