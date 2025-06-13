FROM python:3.11-slim

WORKDIR /app

COPY . /app

ENV PYTHONPATH="/app"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
