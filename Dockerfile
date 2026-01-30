FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Command to run the app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
