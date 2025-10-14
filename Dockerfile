FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY asthama_app/ /app/

COPY models/model.pkl /app/models/model.pkl


# Install Python dependencies
RUN pip install -r requirements.txt


# Expose the Flask app port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
