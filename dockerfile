# Use the official Python image as a base image
FROM --platform=linux/amd64 python:3.12-slim as build

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy NLTK data to the container
COPY nltk_data /usr/local/share/nltk_data

# Set NLTK data path in environment variable
ENV NLTK_DATA=/usr/local/share/nltk_data

# Copy the current directory contents into the container at /app
COPY . .

# Command to run the application
ENTRYPOINT ["python", "model.py"]
