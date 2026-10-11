# ML Serving Platform

An end-to-end MLOps project that trains a machine learning model, tracks every experiment, serves the best model through a REST API, and deploys it automatically with CI/CD.

> **Status:** 🚧 Work in progress. See the [Roadmap](#roadmap) for current progress.

---

## Objective

Most ML projects stop at a Jupyter notebook. This project takes a model all the way to production:

1. Train a text classification model (spam detection).
2. Track every experiment (parameters, metrics, models) with MLflow.
3. Serve the best model through a FastAPI inference service.
4. Package the service with Docker.
5. Test, build, and deploy automatically with GitHub Actions.

## Architecture

```
 Dataset
    |
    v
 train.py  ------>  MLflow (experiment tracking + model    registry)
    |
    v
 Saved model
    |
    v
 FastAPI service  (/predict, /health)
    |
    v
 Docker image  ------>  GitHub Actions (test + build)  ------>  Cloud deployment
```



## Project Structure

```
ml-serving-platform/
├── data/                # Dataset files
├── src/
│   └── train.py         # Model training and MLflow logging
├── api/
│   └── main.py          # FastAPI application
├── tests/               # Unit and API tests
├── requirements.txt     # Python dependencies
└── README.md
```

