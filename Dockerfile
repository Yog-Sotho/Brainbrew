FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
# FIX: added curl for healthcheck
RUN apt-get update && apt-get install -y python3.12 python3-pip git curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash appuser && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser

WORKDIR /app
COPY --chown=appuser:appuser . .
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER appuser

EXPOSE 8501

# FIX: added healthcheck â€” Streamlit exposes /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
