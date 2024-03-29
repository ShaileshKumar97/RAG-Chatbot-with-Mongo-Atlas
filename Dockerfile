# Use a base image with Python
FROM python:3.9.7

# Define Arguments
ARG OPENAI_API_KEY_ARG
ARG MONGO_URI
ARG MONGODB_NAME
ARG MONGODB_COLLECTION
ARG ATLAS_VECTOR_SEARCH_INDEX_NAME

WORKDIR /app

# Define environmental variables
ENV OPENAI_API_KEY=${OPENAI_API_KEY_ARG}
ENV MONGO_URI=${MONGO_URI}
ENV MONGODB_NAME=${MONGODB_NAME}
ENV MONGODB_COLLECTION=${MONGODB_COLLECTION}
ENV ATLAS_VECTOR_SEARCH_INDEX_NAME=${ATLAS_VECTOR_SEARCH_INDEX_NAME}

COPY app.py /app/
COPY requirements.txt /app/

RUN pip install -r requirements.txt

CMD ["python", "app.py"]