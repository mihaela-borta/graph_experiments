# Base image with Python
FROM python:3.10-slim AS builder

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the rest of the files
ADD ./data /app/data
ADD ./src /app/src
COPY shared_config/ /app/config

# Declare volumes
VOLUME ["/app/data", "/app/src", "/app/config"]

# Expose any required ports (optional, if needed for other purposes)
EXPOSE 5000

# Command to keep the container running (you can replace this with your app start command)
CMD ["tail", "-f", "/dev/null"]