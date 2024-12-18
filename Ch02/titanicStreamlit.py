import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## Load the dataset
st.sidebar.header("Upload yoir CSV file")
uploadedFile = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploadedFile is not None:
    data = pd.read_csv(uploadedFile)

    st.title("Titanic Survivours Prediction APP")

    ## Show raw dataset
    if "data" in locals() or "data" in globals():
        st.header("Raw dataset")
        isCheck = st.checkbox("Show raw dataset")
        if isCheck:
            st.write(data)

    ## Feature Selection
    # 열의 이름을 제거한 후 리스트로 변환
    importanceFeatures = data.columns.drop(["PassengerID", "Name", "Ticket",
                                            "Cabin", "Survived"]).tolist()
    ## Selecting Features for Prediction
    st.header("Select Features for Prediction")
    # multiselect(설명텍스트, 선택가능한 피처의 전체목록, 기본적으로 선택될 피처)
    selectedFeatures = st.multiselect("Select features to use for predction",
                                      options=importanceFeatures,
                                      default=["Pclass", "Sex", "Age", "Fare", "Embraked"])

    # 현재 선택된 피처를 세션 상태로 저장
    if "selectedFeatures" not in st.session_state:
        st.session_state["selectedFeatures"] = selectedFeatures
    #피처가 바뀌면 모델을 다시 학습하도록 설정
    if selectedFeatures != st.session_state["selectedFeatures"]:
        del st.session_state["gridBestModel"]
        st.session_state["selectedFeatures"] = selectedFeatures

    ## Data Preprocessing

else:
    st.error("No dataset available. Please upload a CSV file.")
