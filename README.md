<div align="center">

# 🫀 Human Body Systems Viewer

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![VTK](https://img.shields.io/badge/VTK-9.0+-red.svg)

**🩺 Interactive 3D medical visualization system for exploring human anatomy**

[✨ Features](#-features) • [📥 Installation](#-installation) • [🚀 Quick Start](#-quick-start) • [📖 Usage](#-usage) 

---

</div>





<img width="1916" height="986" alt="Screenshot 2025-11-10 111920" src="https://github.com/user-attachments/assets/fe017e2a-e897-4def-8325-994f197158ff" />




---

## 📑 Table of Contents

- [🩺 Overview](#-overview)
- [✨ Features](#-features)
- [📥 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📖 Usage](#-usage)
- [🎮 Interactive Controls](#-interactive-controls)
- [📂 Supported Formats](#-supported-formats)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📧 Contact](#-contact)

---

## 🩺 Overview
This repository provides **interactive human body system viewers** using **Python/Qt** and **VTK**, featuring integrated 3D **clipping** and **curved MPR** tools for NIfTI/DICOM volumes — along with realistic signal-driven animations and ECG simulation.

It includes dedicated modules for:
🫀 Cardiovascular | 🧠 Nervous | 💪 Musculoskeletal | 🦷 Dental systems  
All systems are launched via a Tkinter-based main interface or standalone entry points.

---

## ✨ Features

### 🎯 Core Systems

#### 🫀 Cardiovascular System
- 💗 Real-time ECG simulation with conduction modeling  
- 📊 Heart sounds and audio cues  
- 🗣️ Text-to-speech annotations  
- 🩸 Interactive vessel visualization  
- ✂️ **Integrated advanced clipping with anatomical planes**
- 🔍 **Integrated curved MPR for volume exploration**


https://github.com/user-attachments/assets/1ce0cc42-e0b6-4a82-bd3a-5f7adc6cdc45


#### 🧠 Nervous System
- 🧬 Brain and spinal cord visualization  
- 📊 EEG-driven surface coloring  
- ⚡ Neural signal pathway animation with glow effects
- ✂️ **Integrated multi-plane clipping tools**  
- 🔍 **Integrated curved MPR (Multiplanar Reconstruction)**  


https://github.com/user-attachments/assets/43b43fc7-c90c-4b51-b1d6-29a6aec2bc46


#### 🦴 Musculoskeletal System
- 💪 Bone and muscle rendering  
- ⚡ Neural signal visualization with motor pathway simulation
- 🦵 Stair climbing animation sequence (signal → knee flex)
- 🎨 Advanced quality controls  
- ✨ Edge enhancement options  
- ✂️ **Integrated advanced clipping**
- 🔍 **Integrated curved MPR**


https://github.com/user-attachments/assets/dbbb5aca-8ad5-471b-8f91-fb698f0c2fc1


#### 🦷 Dental System
- 😁 Teeth and jaw segmentation  
- 🎨 Color preset management  
- 🔧 Procedural tooth generation  
- 💫 Neural signal animation  
- 🦴 Jaw movement control (open/close)
- ✂️ **Integrated advanced clipping**
- 🔍 **Integrated curved MPR**


https://github.com/user-attachments/assets/abb83cd5-13b1-433c-b023-cc106cb5048c


---

## 🛠️ Integrated Advanced Tools

All systems now include built-in access to:

- **✂️ Advanced Clipping**: Interactive 3D clipping with axial/coronal/sagittal planes
  - Octant clipping (hide specific regions: left/right/front/back/top/bottom)
  - Visual anatomical plane overlays (colored by orientation)
  - Real-time plane position adjustment
  
- **📊 Curved MPR**: Interactive curved multiplanar reconstruction
  - Load NIfTI, DICOM
  - Draw custom curved paths on 2D slices
  - Generate straightened CPR images along the path
  - Adjustable slice range selection

- **📈 ECG Widget**: Embeddable real-time ECG display with dark theme (Cardiovascular)
  
- **🧪 Enhanced Animations**: 
  - Blood-flow particle animation (Cardiovascular)
  - Neural pathway visualization (Nervous & Musculoskeletal)
  - Jaw movement cycles (Dental)

---

## 📥 Installation

### ⚙️ Prerequisites

- 🐍 Python 3.10+  
- 🖥️ GPU recommended (optional, for enhanced rendering)

### 📦 Install Dependencies

```bash
pip install PyQt5
pip install vtk
pip install numpy
pip install scipy
pip install matplotlib
pip install pandas
pip install pillow
pip install pyttsx3
pip install nibabel  # Required for MPR functionality
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

The main launcher provides buttons to start each system module in a separate process.

### 🏃 Run Individual Systems Directly

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

---

## 📖 Usage

### 🔍 Curved MPR Tool (Integrated)

Access from the **"🔍 Curved MPR"** tab in any system:

1. Click **"Open MPR Tool"** button
2. 📂 Load NIfTI/DICOM
3. 🎯 Use **'Display Slice'** slider to browse through the volume
4. 🎨 Set **'CPR Start/End'** slice range for the reconstruction depth
5. 🖱️ Click on the 2D image to draw curve points along your desired path
6. ✨ Click **"Generate CPR"** to create straightened view

### ✂️ Advanced Clipping (Integrated)

Access from the **"✂️ Clipping"** tab in any system:

1. Click **"Open Advanced Clipping"** button
2. 🎚️ Position clipping planes using X/Y/Z sliders (0-100%)
3. 👁️ Toggle anatomical plane overlays (Axial=Blue, Sagittal=Red, Coronal=Green)
4. 🧭 Select octant regions to hide (Left/Right/Front/Back/Top/Bottom)
5. ⌨️ Use **"Reset All"** to restore default view

### 💗 ECG Simulation (Cardiovascular)

The cardiovascular module includes:
- 📈 P-QRS-T wave generation
- 💗 Atrial and ventricular contraction mapping
- 📊 Synchronized heart sounds
- 🗣️ Optional text-to-speech guidance

### ⚡ Neural Signal Animation (Nervous & Musculoskeletal)

**Nervous System:**
- Watch neural pathways light up on the brain surface
- Choose from Pain, Vision, or Thinking pathways
- Adjustable animation speed

**Musculoskeletal System:**
- Stair climbing sequence: Neural signal → Knee flexion → Reset
- Starts with left leg, then right leg
- Realistic motor pathway simulation

### 🦷 Dental Animations

- Neural signal propagation through teeth (root to crown)
- Jaw movement control (open/close with sound effects)
- Combined sequence: Signal → Open → Signal → Close

---

## 📁 Project Structure

```
📦 human-body-systems/
├── 🚀 main.py                      # Tkinter launcher (subprocess-based)
├── 🫀 cardiovascular_system.py     # Heart & vessels with ECG + integrated tools
├── 🧠 nervous_system.py            # Brain viewer + integrated tools
├── 🦴 musculoskeletal_system.py    # Bones & muscles + integrated tools
├── 🦷 dental_system.py             # Dental visualization + integrated tools
├── 📋 requirements.txt             # Python dependencies
└── 📄 README.md                    # This file
```

**Note:** `clipping.py` and `mpr.py` have been removed. Their functionality is now integrated into each system module.

---

## 📂 Supported Formats

| Type    | Extensions                           | Description                   |
| :------ | :----------------------------------- | :---------------------------- |
| Meshes  | `.stl`, `.obj`, `.ply`, `.vtk`       | 3D anatomical models          |
| Volumes | `.nii`, `.nii.gz`, DICOM, PNG stacks | Medical imaging volumes (for MPR)       |
| Audio   | `.wav`, `.mp3`                       | Heart sounds / voice guidance |

---

## 🧩 Data Sources & Citations

These datasets were partially used in this project for testing, visualization, and evaluation purposes.

**3D Multimodal Dental Dataset based on CBCT and Oral Scan** — Figshare  
A multimodal 3D dataset combining Cone Beam CT (CBCT) and intraoral scans, designed for dental anatomy and visualization research.  
Please refer to the original Figshare page for citation details and licensing information.

https://figshare.com/articles/dataset/_b_3D_multimodal_dental_dataset_based_on_CBCT_and_oral_scan_b_/26965903?file=49086406

**Healthy-Total-Body-CTs** — The Cancer Imaging Archive (TCIA)  
https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/

A dataset containing low-dose, whole-body CT scans of 30 healthy subjects with detailed tissue segmentation (organs, fat, muscle, etc.).

**Data Citation:**  
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
| Toggle clipping   | Access via Clipping tab          |
| Adjust opacity    | Use opacity sliders in UI       |
| Enable curved MPR | Access via MPR tab          |
| Focus on region   | Enable Focus Mode, then click     |
| Toggle wireframe  | `W` key          |

---

## 🤝 Contributing

Contributions are welcome! 🎉

To contribute:

1. Fork this repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push and open a PR

Please ensure code follows PEP8 and includes minimal documentation.

---

## 📜 License

This project is provided as-is for educational and research purposes. Not intended for clinical diagnostic use.

---

## 🙏 Acknowledgments

- 🛠️ Built with VTK, PyVista, and Qt
- 🥼 Medical imaging support via nibabel and pydicom
- 💗 ECG simulation based on physiological models

---

## 📧 Contact

**Project Contributor**: Maryam Moustafa
- Email: maryam23shabaan@gmail.com
- [GitHub](https://github.com/maryam305)
- [LinkedIn](https://www.linkedin.com/in/maryam-moustafa-653257378)
- 
**Project Contributor**: Aya Sayed
- Email: aya.sayed14827@gmail.com
- [GitHub](https://github.com/14930)
- [LinkedIn](https://www.linkedin.com/in/aya-sayed-bb6a80397?utm_source=share_via&utm_content=profile&utm_medium=member_android)
  
**Project Contributor**: Nour Ahmed
- [GitHub](https://github.com/nourahmedmohamed1)
- [LinkedIn](https://linkedin.com/in/nn-anwar)

**Project Contributor**: Mahmoud Mazen
- Email: mmmahmoudmazen208@gmail.com
- [GitHub](https://github.com/MahmoudMazen0)

---

<div align="center">

Made with ❤️ for medical visualization and education

</div>
