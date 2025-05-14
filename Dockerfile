# Base image: Using Python 3.10.11 (which is the one that I'm currently using)
FROM python:3.10.11

# Set the working directory inside the container
# All subsequent commands will be run from this directory
WORKDIR /app

# Copy only requirements.txt first
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces the image size by not caching pip packages (this is nice if the place where I'm deploying has limited size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container (taking into account the .dockerimage obviously)
COPY . .

# Command to run when the container starts
# Format ["command", "argument"]
CMD ["python", "run.py"]