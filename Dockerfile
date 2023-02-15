FROM python:3.9

# 
WORKDIR /Developing-a-Machine-learning-API-using-FastAPI

# 
COPY ./requirements.txt /Developing-a-Machine-learning-API-using-FastAPI/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /Developing-a-Machine-learning-API-using-FastAPI/requirements.txt

# 
COPY ./app /Developing-a-Machine-learning-API-using-FastAPI/app

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]