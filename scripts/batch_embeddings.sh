#!/bin/bash

# Activate virtual environment
source /opt/fastapi-app/venv/bin/activate

# Change to app directory
cd /opt/fastapi-app

# Run batch embedding generation
python -m src.recommendations.batch_embedding_service >>/var/log/batch_embeddings.log 2>&1

# Exit with the command's exit code
exit $?
