# Wellness Tourism Package Purchase Prediction (MLOps)

This repository contains an end-to-end **Machine Learning and MLOps pipeline** for predicting whether a customer is likely to purchase a **Wellness Tourism Package**.  
The project demonstrates data versioning, model training, deployment, and CI/CD automation using modern MLOps tools.

---

## 🔍 Problem Overview

The goal is to predict customer purchase intent **before** outreach, enabling targeted marketing and improved conversion efficiency.  
This is formulated as a **binary classification problem** using customer demographic and interaction data.

---

## 🏗️ Solution Architecture

The project follows a complete MLOps lifecycle:

- **Data Registration:** Hugging Face Dataset Hub  
- **Model Training & Tuning:** Scikit-learn pipeline with Random Forest  
- **Model Registry:** Hugging Face Model Hub  
- **Deployment:** Streamlit app on Hugging Face Spaces (Docker-based)  
- **CI/CD:** GitHub Actions for automated training and registration  

---

## 🔗 Project Assets

- 📊 **Dataset:**  
  https://huggingface.co/datasets/ashishgupttt/visit-with-us-wellness-dataset

- 🧠 **Model Repository:**  
  https://huggingface.co/ashishgupttt/wellness-purchase-model

- 🚀 **Deployed Application:**  
  https://huggingface.co/spaces/ashishgupttt/wellness-purchase-app

---

## 📁 Repository Structure

├── app/ # Streamlit app, Dockerfile, deployment scripts
├── src/ # Training and CI scripts
├── reports/ # Experiment logs and evaluation metrics
├── .github/workflows/ # GitHub Actions CI/CD pipeline
├── requirements-train.txt # Training dependencies
└── README.md


---

## ⚙️ Automation

The repository includes a **GitHub Actions workflow** that:
- Loads data from the Hugging Face Dataset Hub
- Trains and evaluates the model
- Logs metrics
- Registers the model automatically on the Hugging Face Model Hub

---

## 📌 Notes

This repository is intended for academic and demonstration purposes as part of an **Advanced Machine Learning and MLOps project**.
