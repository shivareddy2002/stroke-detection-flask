# 🧠 Advanced Stroke Prediction Web App (Flask)

An interactive **Machine Learning & Deep Learning web application** built with **Flask** for **stroke risk prediction** from **CT and MRI medical image data**.  
This project integrates **computer vision**, **AI models**, and **data analytics** into a seamless web-based platform for medical insights and stroke type classification.

---

## 🚀 Project Overview  

This Flask-based system leverages **multimodal medical images (CT & MRI scans)** and **clinical features** to **analyze and predict stroke types**.  
It provides **AI-driven inference**, **dynamic visualization**, and an **intuitive web dashboard** for healthcare professionals and researchers.

---

## ✨ Key Highlights  

- 🧩 Integrates **image-based and patient-level data** for multimodal analysis  
- 🤖 **AI-powered prediction engine** using deep learning (CNN + BiGRU / ML hybrid)  
- 📊 **Interactive visualizations** powered by Plotly and Matplotlib  
- 🧠 Combines **deep learning + classical ML** pipelines for enhanced accuracy  
- 🎨 **Modern Flask + HTML/CSS/JS** UI for smooth and responsive interaction  
- ☁️ **Production-ready deployment** via Flask (can be hosted on Render / Heroku / AWS)  

---

## 📊 Dataset Description  

**Dataset Source:** [Multimodal Stroke Image Dataset (Kaggle)](https://www.kaggle.com/datasets/turkertuncer/multimodal-stroke-image-dataset?select=deep)

The dataset contains **CT and MRI images** categorized by **stroke type**, allowing for robust AI training and model validation.  

### 🩻 Data Overview  

| Data Type | Description |
|------------|-------------|
| CT Images | Computed Tomography scans representing various stroke types |
| MRI Images | Magnetic Resonance Imaging scans with stroke localization |
| Labels | Stroke category labels (e.g., ischemic, hemorrhagic, normal) |
| Directory Structure | Organized folders under `/deep` for each stroke type |
| Use Case | Suitable for CNNs, transfer learning, and multimodal fusion models |

---

## 🧠 Features  

- 🩺 **Stroke Type Detection:** Upload a CT or MRI scan and get instant predictions  
- 🧮 **Hybrid AI Model:** CNN feature extraction + ML classifier (SVM / RandomForest / BiGRU)  
- 📈 **Dynamic Data Visualization:** Class distribution, image statistics, and model metrics  
- 🔍 **Image Viewer:** Displays uploaded scans with preprocessing preview  
- ⚙️ **Backend Inference Engine:** Handles preprocessing, prediction, and visualization dynamically  
- 🌈 **Flask-Based UI:** Built using HTML5, Bootstrap, and Jinja2 templating  
- 🔒 **Error Handling:** Robust upload validation and runtime error management  

---

## 🧰 Tech Stack  

| Category          | Tools / Libraries           |
| ----------------- | --------------------------- |
| Web Framework     | Flask, Jinja2, Werkzeug     |
| Image Processing  | OpenCV, Pillow (PIL)        |
| Data Analysis     | Pandas, NumPy               |
| Visualization     | Matplotlib, Plotly, Seaborn |
| Machine Learning  | Scikit-learn                |
| Deep Learning     | TensorFlow / Keras          |
| File Handling     | OS, UUID, Secure Filename   |
| Model Persistence | joblib / pickle             |
| Environment       | Python 3.8+                 |

---
