FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# This module is fully self-contained: build context = this directory, so the whole
# thing (code, docs corpus, and the vendored embedding model under models/) lands at
# /app/support_assistant. No network call is needed at build or request-serving time.
COPY . ./support_assistant

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENV MOCK_LLM=1
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "support_assistant.support_assistant:app", "--host", "0.0.0.0", "--port", "7860"]
