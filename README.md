# ✍️ Handwritten Math Solver

A Flask application that uses **OCR + CNN** to recognize handwritten digits and math operators from an image, reconstruct the expression, and **safely evaluate** it (no insecure `eval()`).

---

## 📸 Example
<img width="427" height="180" alt="image-1" src="https://github.com/user-attachments/assets/4635a5f7-f495-4ad3-b34b-bf9640dc4735" />


➡️ The app predicts `2+3*4` and computes the result: `= 14`.

---

## 🔧 Features
- 🖼️ Image preprocessing with OpenCV  
- 🔤 CNN model (Keras/TensorFlow) for digit/operator recognition  
- 🧮 Safe arithmetic evaluator (no arbitrary code execution)  
- 🌐 Flask API endpoint `/predict` for uploads  
- ✅ Unit tests and GitHub Actions CI  

---

## ⚡ Quick Start

### 1. Clone & install
```bash
git clone https://github.com/iot-lab-kiit/handwritten-math-solver.git
cd handwritten-math-solver
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Add model files
Place your trained model files in the `model/` folder:
- `model/model.json`
- `model/model_weights.h5`



### 3. Run the server
```bash
python app.py
```
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000).

---




## 🤝 Contributing
We welcome contributions!  
Check issues labeled **“good first issue”** or **“help wanted”** if you’re new.  

Ways to help:
- Improve preprocessing or recognition
- Add more operators or equation solving
- Write more tests
- Improve docs or CI/CD pipeline

See [contributing.md](https://github.com/iot-lab-kiit/handwritten-math-solver/blob/main/contributing.md) for details.


---

## 📚 Tech Stack
- Python 3.9+  
- Flask  
- TensorFlow / Keras  
- OpenCV  
- NumPy / Pillow  
- Pytest  
- GitHub Actions  

---

## 📄 License
MIT License — see [LICENSE](https://github.com/iot-lab-kiit/handwritten-math-solver/blob/main/LICENSE)
