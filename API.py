# Imports
from fastapi import FastAPI
import pickle, uvicorn, os
from pydantic import BaseModel
import pandas as pd
import numpy as np

####################################################################
# Config & Setup
## Variables of environment
DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH, "assets")
#ml_components_pkl = os.path.join(ASSETSDIRPATH, "logistic_reg.pkl")


# loading the model

def load_model():
   with open('assets/logistic_reg.pkl','rb') as file:
      RF_model = pickle.load(file)
      return RF_model


#loading the  encoder
def load_label_encoder():
   with open("assets/scalerr.pkl", 'rb') as file:
      encoder = pickle.load(file) 
      return encoder

scaler = load_label_encoder()
clasifier_model = load_model()

## API Basic Config
app = FastAPI(
    title="Predicting the Titanic Survivors API",
    version="0.1.0",
    description="Predicting the survivors of the infamous Titanic Disaster",
)


## Loading of assets


####################################################################
# API Core
## BaseModel
class Model_Input(BaseModel):
    Pclass: int
    Age: int
    SibSp: int
    Parch: int
    Fare: int
    Sex_female: int
    Sex_male: int
    Embarked_C: int
    Embarked_Q: int
    Embarked_S: int




## Utils
def feature_engeneering(
    dataset, scaler, 
):  # FE : ColumnTransfromer, Pipeline
    "Cleaning, Processing and Feature Engineering of the input dataset."
    """:dataset pandas.DataFrame"""

    output_dataset = dataset.copy()

    output_dataset = scaler.transform(output_dataset)

    return output_dataset


def make_prediction(
     Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S
):
    """"""
    df = pd.DataFrame(
        [[Pclass,Age,SibSp,Parch,Fare,Sex_female,Sex_male,Embarked_C,Embarked_Q,Embarked_S]],
        columns=[
            'Pclass','Age','SibSp','Parch','Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S'
        ],
    )

    X = feature_engeneering(dataset=df, scaler=scaler,)

    model_output = clasifier_model.predict(X).tolist()
    #Logistic regression is a binary classifier. Hence, predict_proba(X_test) returns 2 columns for the negative and positive class. 
    # We use the loc to return only the positive probability.
    model_prob = clasifier_model.predict_proba(X)[:,1].tolist()
   # return model_output
    return model_output, model_prob

## Endpoints
@app.post("/")
def index():
    return {
        "message": 'welcome to the API for the model predicting the survivors of the infamous Titanic Disaster',
    
    }
@app.post("/predict")
async def predict(input: Model_Input):
    
    output_pred,model_prob = make_prediction(
        Pclass = input.Pclass,
        Age = input.Age,
        SibSp = input.SibSp,
        Parch = input.Parch,
        Fare = input.Fare,
        Sex_female = input.Sex_female,
        Sex_male = input.Sex_male,
        Embarked_C = input.Embarked_C,
        Embarked_Q=input.Embarked_Q,
        Embarked_S=input.Embarked_S,

    )
    if (output_pred[0]>0):
        interprete ="This passenger DID NOT survive the Titanic shipwreck"
    else:
        interprete ="This passenger SURVIVED the Titanic shipwreck" 
    return {
        "prediction": output_pred,
        "Interpretation": interprete,
        "Probability Score": model_prob,
        "input": input,
    }


####################################################################
# Execution

if __name__ == "__main__":
    uvicorn.run(
        "API:app",
        reload=True,
    )