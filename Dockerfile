# 1. Base Image: Start with a lightweight Linux + Python 3.10
FROM python:3.10-slim

# 2. Setup: Create a folder inside the container called '/app'
WORKDIR /app

# 3. System Tools: Install basic Linux tools needed for Python libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Dependencies: Copy the list and install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. The App: Copy your actual code and data into the container
COPY src/ ./src/
COPY data/ ./data/
COPY checkpoints/ ./checkpoints/

# 6. Network: Open the port so we can talk to the app
EXPOSE 8000

# 7. Start: The command to turn on the server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]