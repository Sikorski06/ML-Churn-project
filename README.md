# End-to-End Machine Learning Project: Telco Customer Churn 

[![CI/CD Pipeline](https://github.com/Sikorski06/ML-Churn-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Sikorski06/ML-Churn-project/actions)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face-orange.svg)](https://huggingface.co/spaces/Sikorski06/Telco-Churn_Predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Try the Live Application:** [Hugging Face Spaces Deployment](https://huggingface.co/spaces/Sikorski06/Telco-Churn-Predictor)


## Project Overview & Business Problem
Customer retention is one of the most critical metrics in the telecommunications industry. Acquiring a new customer can cost up to 5 times more than retaining an existing one. 

The goal of this project is to build an **End-to-End Machine Learning pipeline** that predicts whether a customer is likely to churn (leave the company). This allows the business to proactively identify at-risk customers and offer targeted retention campaigns.

**Unlike standard notebook tutorials, this project is fully modularized, containerized, and deployed to the cloud using CI/CD automation.**

## Project Architecture
The system is designed with MLOps best practices, separating the data processing, model training, and the serving API.

1. **Data processing & Modeling:** XGBoost classifier tuned via Optuna.
2. **Serving Layer:** A robust `FastAPI` backend wrapped around the model.
3. **User Interface:** A foolproof `Gradio` frontend designed for end-users (e.g., call center agents).
4. **Containerization:** The entire application is packaged inside a `Docker` container.
5. **CI/CD Pipeline:** `GitHub Actions` automatically builds, tests (Smoke Testing), and pushes the image to `Docker Hub` and `Hugging Face Spaces`.

## Key Engineering Highlights (What makes this unique)
* **Optimized for Business Value (Recall):** Instead of standard accuracy, the threshold of the XGBoost model is dynamically tuned specifically for **Recall**. In churn prediction, False Negatives (missing a churning customer) are the most expensive mistakes.
* **Foolproof UI / Backend Protection:** The frontend handles edge cases gracefully. For example, selecting a non-standard "Other" gender dynamically maps to a safe fallback (`safe_gender`) to prevent internal server errors during inference, ensuring an inclusive yet unbreakable application.
* **Automated Cloud Deployment:** Every push to the `main` branch triggers a workflow that safely deploys the latest version to the public internet.

## Tech Stack
* **Machine Learning:** `Python`, `Scikit-learn`, `XGBoost`, `Optuna`
* **API Development:** `FastAPI`, `Uvicorn`, `Pydantic`
* **Frontend:** `Gradio`
* **DevOps / MLOps:** `Docker`, `GitHub Actions`, `Docker Hub`, `Hugging Face Spaces`

## Getting Started (Local Development)

Because the project is fully Dockerized, running it locally requires zero manual Python environment configuration.

**Step 1: Clone the repository**
```bash
git clone [https://github.com/Sikorski06/ML-Churn-project.git](https://github.com/Sikorski06/ML-Churn-project.git)
cd ML-Churn-project
```
**Step 2: Build the Docker Image**
```bash
docker build -t churn-api .
```
**Step 3: Run the Container**
```bash
docker run -p 7860:7860 churn-api
```
**Step 4: Access the UI**


Navigate to `http://localhost:7860/ui` in your web browser.
