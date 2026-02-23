---
title: banking-insurance-ml-platform
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🏦 Banking & Insurance ML Platform</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3b82f6&center=true&vCenter=true&width=700&lines=Credit+Risk+%C2%B7+Churn+%C2%B7+Fraud+%C2%B7+Insurance+Pricing;Decision+Trees+%C2%B7+Random+Forest+%C2%B7+GBM+%C2%B7+AdaBoost+%C2%B7+GLM;Built+for+Data+Scientists+%26+Business+Stakeholders" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🏦 Banking & Insurance ML Platform** — End-to-end ML platform covering credit risk, churn, fraud detection and insurance pricing using Decision Trees, Bagging, Boosting and GLM families — with both stakeholder-friendly dashboards and full technical model details, all trained on synthetic data.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Dashboard Modules](#-dashboard-modules)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>🏦 <b>Banking & Insurance ML</b></td>
    <td>Five production-grade ML use cases on synthetic data covering the full analytics lifecycle</td>
  </tr>
  <tr>
    <td>🌳 <b>Full Model Family</b></td>
    <td>Decision Tree, Bagging, AdaBoost, Gradient Boosting, Logistic GLM, statsmodels Tweedie/Gamma</td>
  </tr>
  <tr>
    <td>📊 <b>Dual-Audience UI</b></td>
    <td>Stakeholder plain-language cards alongside full technical deep-dives, ROC curves, and coefficients</td>
  </tr>
  <tr>
    <td>🔬 <b>Fully Transparent</b></td>
    <td>Printable decision tree rules, GLM coefficients with p-values, and feature importance charts</td>
  </tr>
  <tr>
    <td>🔒 <b>Secure by Design</b></td>
    <td>Role-based access, audit logs, encrypted data pipelines</td>
  </tr>
  <tr>
    <td>🐳 <b>Containerized Deployment</b></td>
    <td>Docker-first architecture, cloud-ready and scalable on HuggingFace Spaces</td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│            banking-insurance-ml-platform                │
│                                                         │
│  ┌───────────┐    ┌───────────┐    ┌───────────────┐  │
│  │ Synthetic │───▶│    ML     │───▶│   Flask API   │  │
│  │   Data    │    │  Engine   │    │   Backend     │  │
│  └───────────┘    └───────────┘    └───────┬───────┘  │
│                                            │           │
│                                   ┌────────▼────────┐  │
│                                   │  Plotly.js      │  │
│                                   │   Dashboard     │  │
│                                   └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/banking-insurance-ml-platform.git
cd banking-insurance-ml-platform

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker
docker build -t banking-insurance-ml-platform .
docker run -p 7860:7860 banking-insurance-ml-platform
```

---

## 📊 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| 🏠 Overview | Executive KPIs, radar chart, and model family explainer for stakeholders | ✅ Live |
| 💳 Credit Risk | Default prediction — all classifiers + statsmodels Binomial GLM | ✅ Live |
| 🛡️ Insurance Pricing | Premium regression — Tweedie/Gamma GLM vs GBM vs Random Forest | ✅ Live |
| 👤 Churn Prediction | Customer retention scoring — AdaBoost, RF, DT, GBM side-by-side | ✅ Live |
| 🚨 Fraud Detection | Anomaly detection — Isolation Forest + supervised DT & GBM | ✅ Live |
| 📊 Model Comparison | AUC heatmap, Accuracy & F1 grouped bars across all tasks | ✅ Live |

---

## 🧠 ML Models

```python
# Core Models Used in banking-insurance-ml-platform
models = {
    "decision_tree":      "sklearn.tree.DecisionTreeClassifier / Regressor",
    "bagging_rf":         "sklearn.ensemble.RandomForestClassifier / Regressor",
    "adaboost":           "sklearn.ensemble.AdaBoostClassifier",
    "gradient_boosting":  "sklearn.ensemble.GradientBoostingClassifier / Regressor",
    "logistic_glm":       "sklearn.linear_model.LogisticRegression",
    "glm_statsmodels":    "statsmodels.api.GLM — Binomial / Gamma / Tweedie",
    "isolation_forest":   "sklearn.ensemble.IsolationForest",
    "tweedie_glm":        "sklearn.linear_model.TweedieRegressor (power=1.5)",
}
```

---

## 📁 Project Structure

```
banking-insurance-ml-platform/
│
├── 📄 app.py                  # Application entry point — data generation, model training, routes
│
├── 📂 templates/
│   └── 📄 index.html          # Full single-page dashboard (7 modules, Bootstrap 5)
│
├── 📂 static/
│   ├── 📄 style.css           # Dark-theme design system
│   └── 📄 charts.js           # All Plotly.js chart rendering logic
│
├── 📄 requirements.txt        # Python dependencies
├── 📄 Dockerfile              # Container definition (HuggingFace Spaces ready)
└── 📄 README.md               # This file
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## Disclaimer

<span style="color:red">This project is developed strictly for educational and research purposes and does not constitute professional advice of any kind. All datasets used are synthetically generated — no real user data is stored or used. This software is provided "as is" without warranty of any kind; use at your own risk.</span>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/banking-insurance-ml-platform?style=social)](https://github.com/mnoorchenar/banking-insurance-ml-platform)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/banking-insurance-ml-platform?style=social)](https://github.com/mnoorchenar/banking-insurance-ml-platform/fork)

<sub>This project is used purely for academic and research purposes. Any similarity to existing company names, products, or trademarks is entirely coincidental and unintentional. This project has no affiliation with any commercial entity.</sub>

</div>