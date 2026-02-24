<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:6a11cb,100:2575fc&height=180&section=header&text=Advanced%20Stroke%20Prediction%20Web%20App%20(Flask)&fontSize=34&fontColor=ffffff&animation=fadeIn&fontAlignY=35"/>
</p>

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

## ☁️ Vercel Deployment Fix (Lambda Size Limit)

If your Vercel deployment fails with:

`Total dependency size exceeds Lambda ephemeral storage limit (500 MB)`

use this lightweight serverless setup:

- Keep `requirements.txt` minimal (`flask`, `numpy`, `pillow`, `werkzeug`)
- Remove heavyweight training/runtime-only libraries from production builds
- The app automatically uses:
  - **GA-BiGRU model** when TensorFlow + model files are available
  - **Heuristic fallback inference** when they are not available in serverless

This keeps deployments under serverless limits while still returning a prediction.

## 🎯 How to Get Better Predictions

To improve medical prediction quality in practice:

- Use consistently preprocessed grayscale scans (same orientation and resize policy)
- Train/calibrate the GA-BiGRU model on more balanced CT/MRI stroke classes
- Track metrics beyond accuracy (AUC, sensitivity, specificity, F1)
- Tune decision threshold using validation ROC (instead of fixed 0.5)
- Add explainability (Grad-CAM / saliency) and clinician validation workflow
- For production, host full TensorFlow inference on GPU-enabled services (not tiny Lambdas)

---
## 👨‍💻 Author  

**Lomada Siva Gangi Reddy**  
- 🎓 B.Tech CSE (Data Science), RGMCET (2021–2025)  
- 💡 Interests: Python | Machine Learning | Deep Learning | Data Science  
- 📍 Open to **Internships & Job Offers**

 **Contact Me**:  

- 📧 **Email**: lomadasivagangireddy3@gmail.com  
- 📞 **Phone**: 9346493592  
- 💼 [LinkedIn](https://www.linkedin.com/in/lomada-siva-gangi-reddy-a64197280/)  🌐 [GitHub](https://github.com/shivareddy2002)  🚀 [Portfolio](https://lsgr-portfolio-pulse.lovable.app/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:f9c74f,100:ff4b4b&height=120&section=footer"/>
</p>
