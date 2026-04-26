<h1 align="center">💳 UPI Fraud Detection System</h1>

<p align="center">
  <b>Machine Learning Based Real-Time Fraud Detection</b><br>
  <i>Secure • Intelligent • Interactive</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Streamlit-Deployed-ff4b4b?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Machine Learning-Scikit Learn-orange?style=for-the-badge&logo=scikitlearn"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://upifrauddetection-xzmxv9ve9d2thclklmige9.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀 Live Demo-Click Here-0A66C2?style=for-the-badge&logo=streamlit&logoColor=white"/>
  </a>
</p>

---

## ✨ Overview

This project is a **real-time UPI Fraud Detection System** built using **Machine Learning models** and deployed with **Streamlit**.

It analyzes transaction details like:
- Time ⏰  
- Amount 💰  
- Transaction Type 🔁  
- Sender & Receiver 📤📥  

…and predicts whether the transaction is:

✅ **Legitimate**  
🚨 **Fraudulent**

---

## 🎯 Features

✔️ Dual Model Prediction (Logistic Regression + SVM)  
✔️ Real-time Fraud Probability  
✔️ Smart Feature Engineering  
✔️ User Confirmation (Bank-like OTP logic)  
✔️ Interactive Dashboard 📊  
✔️ Visual Analytics (Graphs + Gauge)  

---

## 🧠 Machine Learning Models

| Model | Purpose |
|------|--------|
| Logistic Regression | Probability-based prediction |
| Support Vector Machine (SVM) | Decision boundary classification |

---


## 📊 How It Works

```text
User Input → Feature Engineering → Encoding → Scaling → Model Prediction → Result
```

---


## 🚀 Live Demo

<p align="center">
  <a href="https://upifrauddetection-xzmxv9ve9d2thclklmige9.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀 Live Demo-Click Here-0A66C2?style=for-the-badge&logo=streamlit&logoColor=white"/>
  </a>
</p>



## 📁 Project Structure

```bash
├── app.py
├── lr_model.pkl
├── svm_model.pkl
├── scaler.pkl
├── le_type.pkl
├── le_sender.pkl
├── le_receiver.pkl
├── median.pkl
├── lr_acc.pkl
├── svm_acc.pkl
├── requirements.txt
└── README.md
```

---

## ⚡ Installation (Run Locally)

```bash
git clone https://github.com/Yamuna-97/Upi_Fraud_detection_using_machine_learning.git
cd Upi_Fraud_detection_using_machine_learning

pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Key Highlights

- Detects **midnight transactions 🌙**
- Identifies **high-value anomalies 💰**
- Flags **unknown UPI IDs ⚠️**
- Uses **dual-model verification**
- Simulates **real banking security flow 🏦**

---

## 🛡️ Real Banking Logic

If fraud is detected:
- ⚠️ User gets warning  
- 🔐 User confirmation required  
- ❌ Transaction can be blocked  
- ✅ Genuine transactions allowed  

---


<p align="center">
  💡 <i>"Smart Systems make Secure Transactions"</i>
</p>
