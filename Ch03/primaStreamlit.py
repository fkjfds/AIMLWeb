import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, precision_recall_curve,
                             roc_curve, auc)
import io

# 애플리케이션 제목
st.title("Pima Indians Diabetes Prediction APP")

# 파일 업로드
uploaded_file = st.file_uploader("Upload your CSV file", type=["CSV"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("File Uploaded Successfully!")

    # 데이터 확인
    if st.checkbox("Show Raw Data"):
        st.write(data)

    # 데이터 전처리 : min 값이 0인 피처 평균값으로 대체
    for col in data.columns[:-1]:
        if data[col].min() == 0 and col != "Pregnancies":
            if data[col].dtype == "int64":
                mean_value = int(data[col][data[col] != 0].mean())
            elif data[col].dtype == "float64":
                mean_value = data[col].replace(0, mean_value)
    #데이터 확인
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_ouput = buffer.getvalue()
    st.text(info_ouput)
    st.write("Processed Data (0 values replaced with mean): ")
    st.write(data.describe())