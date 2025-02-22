Here’s a README for your project:  

---

## 🔥 FireGuard: A CRNN-based Fire Detection and Alert System  

FireGuard is a deep learning-based fire detection system that improves upon the baseline model from the IEEE paper **"EdgeFireSmoke: A Novel Lightweight CNN Model for Real-Time Video Fire/Smoke Detection."** While the original work uses a **CNN-based model (CNN.py)**, we enhance its capabilities by implementing a **CRNN model (CRNN.py)** for better feature extraction and temporal understanding.  

---

### 📌 Features  
✅ Fire and smoke detection using deep learning  
✅ Comparison between CNN and CRNN models  
✅ Trained on a large dataset of **49.5K images**  
✅ Real-time alert system integration (optional)  

---

### 📂 Dataset  
The dataset used for training and evaluation can be accessed from:  
🔗 **[Dataset Link](https://drive.google.com/drive/folders/1k23qNjH_nDxi6auUEgj6BXPiCutcxDYd)**  

⚠ **Note:** The dataset size is **13GB** with **49.5K images**. Due to its large size, we recommend using an IDE like **PyCharm** or **Spyder**. If using **Google Colab**, you must have a **premium version** for smooth execution.  

---

### 🛠 Requirements  
Ensure you have the following packages installed:  

```bash
pip install tensorflow numpy pandas keras pillow matplotlib opencv-python flask
```

---

### 📜 Code Structure  

- **CNN.py** - Implements the baseline **CNN model Based in the IEEE Paper**  
- **CRNN.py** - Implements the **CRNN model** with enhancements  
---

### 🚀 Usage  

#### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/BhanuSaketh/FireGuard-A-CRNN-based-Fire-Detection-and-Alert-System-  
cd FireGuard  
```

#### 2️⃣ Run Training  
To train the CNN model:  
```bash
python CNN.py --model CNN
```
To train the CRNN model:  
```bash
python CRNN.py --model CRNN
```

---

### 📢 Citation  
If you use this work, consider citing the original IEEE paper:  
> *"EdgeFireSmoke: A Novel Lightweight CNN Model for Real-Time Video Fire/Smoke Detection."*  

---
