#!/usr/bin/env bash
# Launch MLflow UI for browsing experiment runs.
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
