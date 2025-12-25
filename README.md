# Face Mask Detection


An end-to-end **Face Mask Detection System** built using **YOLOv8**, deployed with **Streamlit**, and integrated with **GitHub Actions CI** for automated validation. This project demonstrates computer vision, deep learning deployment, and CI pipeline practices in a real-world–style setup.

---

## 🚀 Features

* **✅ Precise Classification:** Detects `with_mask`, `without_mask`, and `mask_weared_incorrect`.
* **🖼 Multi-Media Support:** Inference for images, videos, and batch processing.
* **📊 Visual Analytics:** Detection summary with confidence scores and class counters.
* **🧪 DevOps Ready:** Integrated GitHub Actions CI pipeline for automated testing.
* **🧠 Deployment Optimized:** Features ONNX-optimized inference for faster performance.

---

##  Project Structure

```text
face_mask_detection_yolo/
├── .github/workflows/    # CI/CD automation scripts
├── app/                  # Streamlit-based web application
├── inference/            # Prediction & visualization logic
├── model/                # Training, evaluation & export scripts
├── utils/                # Data loading & preprocessing helpers
├── best_face_mask.onnx   # Optimized model for deployment
├── best_face_mask.pt     # Trained YOLOv8 weights
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
⚙️ Installation & Setup
1️⃣ Clone the Repository
Bash

git clone [https://github.com/Nik9296/DL_labAssesment.git](https://github.com/Nik9296/DL_labAssesment.git)
cd DL_labAssesment
2️⃣ Create Virtual Environment
Bash

python -m venv venv
# For Windows:
venv\Scripts\activate
# For Mac/Linux:
source venv/bin/activate
3️⃣ Install Dependencies
Bash

pip install -r requirements.txt
▶️ Running the Application
Launch the Streamlit dashboard locally:

Bash

streamlit run app.py
Then open your browser at: http://localhost:8501

🔄 CI Pipeline (GitHub Actions)
This project uses Continuous Integration to ensure code quality:

Automated Checks: Triggered on every push or pull_request to the main branch.

Validation: Installs dependencies, checks for syntax errors, and verifies file integrity.

🧪 Technologies Used
Core: Python, YOLOv8 (Ultralytics)

UI: Streamlit, Plotly

Inference: OpenCV, ONNX Runtime

DevOps: GitHub Actions

📊 Future Roadmap
[ ] Two-stage pipeline (dedicated face detector + classifier).

[ ] Real-time WebRTC support for browser-based webcam streaming.

[ ] Dockerization for cloud deployment.

Maintained by Nik9296


---

### How to push this to GitHub now:
1. Save the content above into a file named **README.md**.
2. Run these commands in your terminal:
```bash
git add README.md
git commit -m "Update README with full project details"
git push origin main
