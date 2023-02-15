# From Notebook to Production: A Step-by-Step Guide to Deploying Your ML Model using FastAPI

The primary objective of creating a machine learning model is to solve real-world problems. Consequently, models created in notebooks must be made available for the intended users through deployment. There are several methods for deploying ML models and one of such method is through an API. This article explains how to deploy a model using an API, specifically FastApi. My article Model Deployment -The last stage of the CRISP-DM Method explains the importance of model deployment and the various deployment methods.

An API, or application programming interface, is a set of defined rules that enable different applications to communicate with each other. APIs allow different applications to communicate with each other using a set of defined rules. They provide a set of rules, protocols, and tools for building software and enable the exchange of data and services between various applications and systems. They are used to deploy machine learning models by exposing the model through a RESTful API. This allows other applications and users to communicate with the model and make predictions by sending data to the API endpoint. The API then returns the model’s predictions to the requesting applications and users. Using APIs for ML model deployment provides several benefits, including scalability, reliability, and ease of maintenance. APIs also provide a well-defined interface for communication with the model, making it easier to test, monitor and maintain.

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python. It is fast, robust, and easy to code thereby increasing the speed to develop features by about 200% to 300%. You can check the full FastAPI documentation for more information.

In this article, we will discuss the following steps needed to accomplish our task. Serving a model using FastApi involves 4 steps:

 ## Creating a Model

 ## Saving and loading the Model

 ## Create the FastAPI endpoints

 ## Convert (encode) input data and make a prediction.

**The full code is available in the project folder on Github**

 ## Creating a Model

We will use the model we created earlier using the titanic dataset. For a detailed step-by-step process of creating and saving the model, read my article, [From Data to Prediction: A Comprehensive Guide to Analyzing, Visualizing, and Modeling the Titanic Dataset](https://medium.com/@alihu.alhassan/from-data-to-prediction-a-comprehensive-guide-to-analyzing-visualizing-and-modeling-the-titanic-3ca458d4da83)

After creating the model, we need to create an endpoint that accepts input data, extracts the relevant features, makes a prediction using the loaded model, and returns the predicted class as a JSON response. We will be using FastAPI for this process. To use FastAPI, we need to install FastAPI and a few other dependencies. Follow this step-by-step instruction on how to install it and other dependencies.

Next, we will create a new file api.py to contain our FastAPI application code. This file will contain our API endpoints, routes, and the functions to load and serve our model.

## Loading the saved (Serialized) model

Model serialization is the process of converting a trained machine-learning model into a format that can be stored in a file or database and later loaded back into memory for prediction. The serialization process typically involves converting the model’s parameters, architecture, and/or weights into a binary representation that can be written to disk or transferred over a network. There are several methods for serializing models including pickle, joblib, and ONNX. During the process of creating the model, we explained how to save a model using pickle. You can read it here.

To load the Model, open the pickled model file using the open function in read-binary mode and then load the pickled model from the file using the pickle.load function. Repeat the same process for the encoder and other pickled assets.

import pickle

# Load the saved model from file
with open('LOGISTIC_modeL.pkl', 'rb') as file:
    model = pickle.load(file)
Next, we will create the route(/predict) that receives the POST request. When a request is received, the make_prediction function loads the model and uses it to make a prediction. The result is returned as a JSON response that includes the predicted variable, probability score as well as input variables.
```
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
 ```
    
## Conclusion

In this article, we learned how to serve a machine-learning model through an API using FastAPI. FastAPI makes it easy to build high-performance APIs with Python, and it is a great choice for serving machine learning models in production. With a little bit of Python code, we can create an API that makes predictions based on data, allowing us to build intelligent applications to solve business problems.

## The full code is available in the API.py file in this folder.

