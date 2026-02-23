---
title: banking-insurance-ml-platform
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🏦 Banking & Insurance ML Platform</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=4F46E5&center=true&vCenter=true&width=700&lines=GLM+%7C+Decision+Tree+%7C+Random+Forest+%7C+Gradient+Boosting;Banking+%26+Insurance+Risk+Analytics;Dual+View%3A+Data+Scientist+%2B+Stakeholder;100%25+Synthetic+Data+%E2%80%94+No+Real+PII" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-f97316?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🏦 Banking & Insurance ML Platform** — An interactive, end-to-end portfolio demonstrating Generalized Linear Models, Decision Trees, Random Forest (Bagging), and Gradient Boosting applied to credit-default and insurance high-claim prediction. Features a dual Data Scientist / Stakeholder view — all powered by realistic synthetic data.

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
    <td>🧪 <b>Fully Synthetic Datasets</b></td>
    <td>Realistic banking (credit default) and insurance (high-claim) portfolios generated with statistically grounded log-odds models — zero real PII</td>
  </tr>
  <tr>
    <td>🤖 <b>Four Model Families</b></td>
    <td>GLM (Logistic Regression), Decision Tree, Random Forest (Bagging), and Gradient Boosting — all configurable via interactive controls</td>
  </tr>
  <tr>
    <td>👥 <b>Dual Audience Views</b></td>
    <td>Every model page offers a toggle between a full Data Scientist view (metrics, charts, hyperparameters) and a plain-English Stakeholder view (KPIs, business impact, recommendations)</td>
  </tr>
  <tr>
    <td>📊 <b>Rich Plotly Visualizations</b></td>
    <td>ROC curves, confusion matrices, calibration curves, feature importance, depth-complexity tradeoffs, staged boosting curves, and correlation heatmaps</td>
  </tr>
  <tr>
    <td>🔒 <b>Secure by Design</b></td>
    <td>Non-root Docker execution, no external data dependencies, stateless API backend</td>
  </tr>
  <tr>
    <td>🐳 <b>Containerized Deployment</b></td>
    <td>Docker-first architecture with gunicorn; HuggingFace Spaces ready on port 7860</td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│              Banking & Insurance ML Platform                        │
│                                                                     │
│  ┌──────────────────┐    ┌────────────────┐    ┌────────────────┐  │
│  │  Synthetic Data  │───▶│   ML Engine    │───▶│  Flask API     │  │
│  │  Generator       │    │  (sklearn)     │    │  /api/model/*  │  │
│  │  Banking Dataset │    │  GLM / Tree /  │    │  /api/data/*   │  │
│  │  Insurance Data  │    │  RF / GBM      │    └───────┬────────┘  │
│  └──────────────────┘    └────────────────┘            │           │
│                                                ┌────────▼────────┐  │
│                                                │  Plotly.js      │  │
│                                                │  Flask/Jinja2   │  │
│                                                │  Bootstrap 5    │  │
│                                                │  Dashboard UI   │  │
│                                                └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
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

# 5. Run the application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker
docker build -t banking-insurance-ml-platform .
docker run -p 7860:7860 banking-insurance-ml-platform

# Or with Docker Compose (if you add docker-compose.yml)
docker compose up --build
```

---

## 📊 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| 📊 Data Explorer | Interactive EDA — distributions, correlation heatmap, categorical analysis, sample table | ✅ Live |
| 📈 GLM — Logistic Regression | Coefficients, odds ratios, ROC, calibration curve, confusion matrix | ✅ Live |
| 🌳 Decision Tree | Depth-complexity tradeoff, rule extraction, feature importance, CV score | ✅ Live |
| 🌲 Random Forest | OOB learning curve, bagging benefit analysis, feature importance | ✅ Live |
| ⚡ Gradient Boosting | Staged-score curves, overfitting diagnostics, feature importance | ✅ Live |
| 🏆 Model Comparison | Side-by-side ROC overlay, metric heatmap, radar chart, selection guidance | ✅ Live |
| 👥 Stakeholder Dashboard | Plain-English KPIs, risk segments, business impact, AI recommendations | ✅ Live |

---

## 🧠 ML Models

```python
# Core Models in Banking & Insurance ML Platform
models = {
    "GLM":               "LogisticRegression (sklearn) — L1/L2 regularization, calibration curve",
    "Decision Tree":     "DecisionTreeClassifier — Gini/Entropy, depth-complexity analysis, rule export",
    "Random Forest":     "RandomForestClassifier — Bagging, OOB score, learning curve vs n_estimators",
    "Gradient Boosting": "GradientBoostingClassifier — staged scores, subsample, learning rate tuning",
    "Evaluation":        "AUC-ROC, F1, Precision, Recall, Confusion Matrix, 5-fold CV, Calibration"
}
```

---

## 📁 Project Structure

```
banking-insurance-ml-platform/
│
├── 📂 models/
│   ├── 📄 __init__.py
│   ├── 📄 data_generator.py    # Synthetic banking & insurance data
│   └── 📄 ml_models.py         # GLM, Decision Tree, RF, GBM training & evaluation
│
├── 📂 routes/
│   ├── 📄 __init__.py
│   ├── 📄 main.py              # Page routes (Flask Blueprints)
│   └── 📄 api.py               # JSON API endpoints for model training & data
│
├── 📂 templates/
│   ├── 📄 base.html            # Sidebar layout, topbar, dataset toggle
│   ├── 📄 index.html           # Home / landing page
│   ├── 📄 data_explorer.html   # EDA module
│   ├── 📄 glm.html             # Logistic Regression module
│   ├── 📄 decision_tree.html   # Decision Tree module
│   ├── 📄 random_forest.html   # Random Forest module
│   ├── 📄 gradient_boosting.html # Gradient Boosting module
│   ├── 📄 model_comparison.html  # All-models comparison
│   └── 📄 stakeholder.html     # Executive stakeholder view
│
├── 📂 static/
│   ├── 📂 css/
│   │   └── 📄 style.css        # Custom styling (dark sidebar, cards, KPIs)
│   └── 📂 js/
│       └── 📄 charts.js        # Shared Plotly helpers, API wrappers
│
├── 📄 app.py                   # Application entry point
├── 📄 Dockerfile               # HuggingFace Spaces ready (port 7860)
├── 📄 requirements.txt         # Python dependencies
└── 📄 README.md                # This file
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

<span style="color:red">This project is developed strictly for educational and research purposes and does not constitute professional advice of any kind. All datasets used are synthetically generated — no real user data is stored or processed. This software is provided "as is" without warranty of any kind; use at your own risk.</span>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/banking-insurance-ml-platform?style=social)](https://github.com/mnoorchenar/banking-insurance-ml-platform)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/banking-insurance-ml-platform?style=social)](https://github.com/mnoorchenar/banking-insurance-ml-platform/fork)

<sub>This project is for academic and research purposes only. No affiliation with any financial institution or commercial entity. All data is synthetically generated.</sub>

</div>
