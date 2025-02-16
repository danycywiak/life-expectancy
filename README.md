# 📊 Life Expectancy Prediction with Neural Networks

## 🌟 Overview
This project uses a neural network to predict life expectancy based on socioeconomic and health-related data. The model is implemented using **TensorFlow** and deployed on **Hugging Face Spaces** with **Streamlit**.

## 🔍 Problem Statement
Life expectancy varies across countries due to multiple factors such as **GDP, healthcare access, education, and environmental conditions**. This project aims to analyze these factors and predict life expectancy using a machine learning approach.

## 🛠 Technologies Used
- **Programming:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, TensorFlow, Matplotlib, Seaborn
- **Model Deployment:** Hugging Face, Streamlit
- **Data Visualization:** Matplotlib, Seaborn

## 📂 Dataset
The dataset used for training the model comes from **WHO and the United Nations**. It includes various indicators such as GDP per capita, immunization rates, and mortality rates.
- [Dataset Source (if public)](https://www.kaggle.com/datasets/lashagoch/life-expectancy-who-updated)

## 🚀 Project Demo
Try the deployed model here:  
[![Hugging Face](https://img.shields.io/badge/🤗-Try%20it%20on%20Hugging%20Face-blue)](https://huggingface.co/spaces/danycywiak/life-expectancy)

## 📥 Installation & Usage
To run this project locally, follow these steps:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/life-expectancy-prediction.git
cd life-expectancy-prediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

## 📊 Model Training
The model is a **deep neural network (DNN)** trained with the following approach:
1. **Data Preprocessing:** Handling missing values, feature scaling, and encoding categorical variables.
2. **Model Architecture:** Multiple dense layers with **ReLU activation** and dropout regularization.
3. **Evaluation Metrics:** Mean Absolute Error (MAE) and Mean Squared Error (MSE) to assess performance.
