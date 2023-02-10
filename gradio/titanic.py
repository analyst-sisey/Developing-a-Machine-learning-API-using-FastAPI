#Import the required Libraries
import json
import streamlit as st
import pickle, os
import pandas as pd
import base64
import requests


DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH, "assets")
ml_components_pkl = os.path.join(ASSETSDIRPATH, "logistic_reg.pkl")

# load the model
@st.cache(persist=True,allow_output_mutation=True)
def load_model():
   with open('assets/logistic_reg.pkl','rb') as file:
      RF_model = pickle.load(file)
      return RF_model


#function to load scaler
def load_scaler():
   with open("assets/scalerr.pkl", 'rb') as file:
      scaler = pickle.load(file) 
      return scaler

model = load_model()
scaler = load_scaler()
#add background image to th app
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('image1.jpg')    
# Add a title and intro text
st.title('Titanic Disaster Survivors Prediction') 
st.text( 'This is a web app that interracts with the API for the model predicting')
st.text( 'the survivors of the infamous Titanic Disaster')
st.markdown('***Select your features and click on Submit***')

#Form to collect user input
form = st.form(key="information", clear_on_submit=True)

with form:

    cols = st.columns((1, 1))
    Pclass = cols[0].selectbox("Ticket class:", ['1st', '2nd', '3rd'], index=0)

    Age = cols[0].number_input("Enter Passenger's Age")
    SibSp = cols[1].number_input('Enter Number of siblings / spouses aboard')
    Parch = cols[0].number_input('Enter Number of parents / children aboard')
    Fare = cols[1].number_input('Enter Fare paid by Passenger')
    sex = cols[0].selectbox("Gender:", ['Male', 'Female'], index=0)
    Embarked = cols[1].selectbox("Port of Embarkation:", ['Cherbourg', 'Queenstown', 'Southampton'], index=0)
    
    cols = st.columns(2)
    
    submitted = st.form_submit_button(label="Submit")
    Sex_female= 0
    Sex_male=0
    Embarked_C=0 
    Embarked_Q=0
    Embarked_S = 0

    if Pclass == "1st":
        Pclass = 1
    elif Pclass =="2nd":
        Pclass = 2
    else: Pclass = 3

    if sex == "Male":
        Sex_female = 0.0
        Sex_male = 0.1
    elif sex =="Female":
        Sex_female = 0.0
        Sex_male = 0.1
    #else: Pclass = 2
    if Embarked == "Cherbourg":
       Embarked_C = 0.1
       Embarked_Q = 0.0
       Embarked_S = 0.0
    elif Embarked =="Queenstown":
         Embarked_C = 0.0
         Embarked_Q = 0.1
         Embarked_S = 0.0
    elif Embarked == "Southampton":
         Embarked_C = 0.0
         Embarked_Q = 0.0
         Embarked_S = 0.1
    inputs ={
  "Pclass": Pclass,
  "Age": Age,
  "SibSp": SibSp,
  "Parch": Parch,
  "Fare": Fare,
  "Sex_female": Sex_female,
  "Sex_male": Sex_male,
  "Embarked_C": Embarked_C,
  "Embarked_Q": Embarked_Q,
  "Embarked_S": Embarked_S
}

if submitted:
    st.success("Processing!")
    


    #res = requests.post(url= "http://127.0.0.1:8000/predict", data = data)
    res = requests.post(url = "http://127.0.0.1:8000/predict", data = json.dumps(inputs))
    st.subheader(f"Response from API is= {res.text}")
   
    x =res.text
    y = json.loads(x)
    st.subheader(f"Response from API = {y['prediction']}")
    
    st.balloons()
