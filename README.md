
# Telco Customer Churn Prediction | End-to-End MLOps

[![CI/CD Pipeline](https://github.com/Sikorski06/ML-Churn-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Sikorski06/ML-Churn-project/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-orange.svg)](https://huggingface.co/spaces/Sikorski06/Telco-Churn-Predictor)

> **Live Demo:** [Try the deployed application here](https://huggingface.co/spaces/Sikorski06/Telco-Churn-Predictor)

## Project Overview
This project goes beyond a standard Jupyter Notebook analysis. It is a **production-ready Machine Learning pipeline** designed to predict telecommunications customer churn. Built with a strong MLOps mindset, the system emphasizes business value, robust software engineering principles, and fully automated deployment.

## Key Differentiators (Why this project stands out)
* **Business-Driven Metric Optimization:** Instead of blindly chasing standard accuracy, the XGBoost model is dynamically threshold-tuned specifically for **Recall**. In the Telco industry, the cost of a false negative (missing a churning customer) is significantly higher than a false positive.
* **Resilient Architecture:** The machine learning inference logic is strictly decoupled from the serving layer (`FastAPI`).
* **Foolproof UI/UX:** The `Gradio` frontend includes strict input validation (e.g., numeric-only inputs for charges, default safe states, and dynamic fallback mapping for categories like "Other" gender) to prevent backend API crashes and ensure inclusivity.
* **Fully Automated CI/CD:** A custom `GitHub Actions` workflow automatically runs smoke tests, builds the Docker image, pushes it to `Docker Hub` as an artifact, and continuously deploys to `Hugging Face Spaces` upon every push to the main branch.

## Technology Stack
* **Modeling & Data:** Python, Pandas, Scikit-learn, XGBoost, Optuna (Hyperparameter Tuning)
* **API & Serving:** FastAPI, Uvicorn, Pydantic (Data Validation)
* **Frontend:** Gradio
* **Infrastructure & MLOps:** Docker, GitHub Actions (CI/CD), Docker Hub, Hugging Face Spaces

## Architecture & Repository Structure
The codebase follows strict structural standards, separating data processing, model training, and serving infrastructure:

```text
ML-Churn-project/
├── .github/workflows/   # CI/CD automation pipelines (YAML)
├── src/                 # Main source code
│   ├── data/            # Data loading and validation scripts
│   ├── features/        # Feature engineering and encoding logic
│   ├── model/           # XGBoost training, evaluation, and thresholding
│   ├── serving/         # Inference logic and model loading classes
│   └── app/             # FastAPI backend and Gradio UI integration
├── Dockerfile           # Container definition
├── requirements.txt     # Project dependencies
└── README.md