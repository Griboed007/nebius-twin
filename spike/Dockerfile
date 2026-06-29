FROM python:3.11-slim
RUN pip install fastapi uvicorn
COPY spike_app.py .
CMD ["uvicorn","spike_app:app","--host","0.0.0.0","--port","8080"]
