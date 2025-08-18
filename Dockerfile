# Use an official Python runtime as a parent image
FROM python:3.10.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container at /app
COPY . .

# Command to run your API when the container launches
# It tells Uvicorn to run on host 0.0.0.0, which is necessary for cloud services
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]