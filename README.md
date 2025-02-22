## 🔥 FireGuard: A CRNN-based Fire Detection and Alert System  

FireGuard is a deep learning-based fire detection system that improves upon the baseline model from the IEEE paper **"EdgeFireSmoke: A Novel Lightweight CNN Model for Real-Time Video Fire/Smoke Detection."** While the original work uses a **CNN-based model (CNN.py)**, we enhance its capabilities by implementing a **CRNN model (CRNN.py)** for better feature extraction and temporal understanding.  

---

### 📌 Features  
✅ Fire and smoke , Green-Area and Burned-Area detection using deep learning  
✅ Comparison between CNN and CRNN models  
✅ Trained on a large dataset of **49.5K images**  
✅ Real-time alert system integration (optional)  
✅ Web interface using **Flask** for testing and visualization  

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
- **static/** - Contains static files (CSS, JavaScript, Images) for the Flask web application  
- **templates/** - Contains HTML templates for the Flask web application  
- **app.py** - The Flask web application for testing the models  
- **model/** - Directory where trained models (`model.h5` and `model.json`) are saved  

---

### 🚀 Usage  

#### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/BhanuSaketh/FireGuard-A-CRNN-based-Fire-Detection-and-Alert-System-  
cd FireGuard  
```

#### 2️⃣ Train the Models  

To train the CNN model:  
```bash
python CNN.py --model CNN
```
This will generate two files:  
- `model.h5` (Trained CNN model)  
- `model.json` (Model architecture)  
- It will also display the **Accuracy, Precision, Recall, and F1-score** from the base paper.  

To train the CRNN model:  
```bash
python CRNN.py --model CRNN
```
This will generate the same files but for the CRNN model.  

---

### 🌐 Running the Flask Web Application  

Once you have trained models (`model.h5` and `model.json`), you can test them using the web interface.  

#### 1️⃣ Place Model Files  
Move the generated `model.h5` and `model.json` files into the `model/` directory.  

#### 2️⃣ Start the Flask App  
```bash
python app.py
```

#### 3️⃣ Access the Web Interface  
Open your browser and go to:  
```
http://127.0.0.1:5000/
```
Here, you can upload images and test fire detection using the trained models.  

---

### 📢 Citation  
If you use this work, consider citing the original IEEE paper:  
> *"EdgeFireSmoke: A Novel Lightweight CNN Model for Real-Time Video Fire/Smoke Detection."*  

---

🔥 **FireGuard: Detect Fires Before They Spread!**

