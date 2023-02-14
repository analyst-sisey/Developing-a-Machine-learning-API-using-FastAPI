# Introduction
This repository contains a comprehensive projecct that first analysed and visualised the titanic dataset, 2 predictive machine learning models were then created using the dataset. The model was then deployed using FastAPI. The project was the containerized using docker.

# Data Analysis and Model Creation
The goal of this project is to use the analyse the dataset and create a predictive model that can accurately determine whether a given passenger survived or died in the sinking of the ship. We wexplore the processes of accomplishing this goal using the following CRISP-DM methodology. 

- [x] Perform EDA
- [x] Answer some questions
- [x] Test some hypothesis
- [x] Create 2 different models and compare their results and confusion matrixes

**The project notebook titanic_survival_prediction.ipynb in this repository contains detailed explanations and code for this project**


# Developing-a-Machine-learning-API-using-FastAPI
This project creates an API that is requested to interact with a the model. The model is an ML model that predicts the survivors of the famous titanic disaster.
This is an interesting solution when you want to keep your model architecture secret or to make your model available to users already having an API. By creating an API, and deploying it, your model can easily receive request using the internet protocol.

# Containerizing the Model using Docker
Finally we wrap our project in a container using Docker. Containerizing a model provides a lightweight, portable, and reproducible environment for running the model, regardless of the underlying system. This allows us to easily package the model and its dependencies into a single container that can be deployed and run consistently across different machines, making it easier to share and scale the model. Additionally, containerization can help isolate the model from other processes running on the host system, improving security and reducing potential conflicts with other software installed on the same machine.
