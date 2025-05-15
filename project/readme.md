# Project: Automated MLOps Pipeline with Ray

This project aims to design and implement a fully automated MLOps platform, inspired by the feature set of Anyscale. Students will collaboratively construct a modular, distributed machine learning pipeline using Ray, focusing on **automation**, **scalability**, and **reproducibility**. 

The pipeline will cover:

- data ingestion,
- preprocessing,
- model training (with a focus on LLMs),
- deployment,
- monitoring, and
- orchestration.

## Objectives

- Automate the entire ML workflow, from data scraping to model monitoring.
- Enable distributed computing for scalability and efficiency.
- Ensure modularity, allowing each group to focus on a distinct pipeline component.
- Promote reproducibility and best practices in MLOps.

## Project Structure and Group Responsibilities

The project is divided into **five** main phases, each assigned to specific student groups:

### Phase 1: Data Ingestion & Cleaning <mark>Groups 1, 2 & 3</mark>

- Implement automated web scraping using Ray Actors for parallelization.
- Perform data validation and cleaning to ensure high-quality input for model training.
- Store and version datasets using Ray Datasets and cloud storage (e.g., AWS S3 compatible open source storage, MinIO).

### Phase 2: Model Training <mark>Groups 4, 5 & 6</mark>

- Automate hyperparameter tuning with Ray Tune.
- Implement distributed training for LLMs (e.g., **Llama**) using Ray Train.
- Track experiments and results with Ray and **MLflow**.

### Phase 3: Model Serving & Monitoring <mark>Groups 7, 8 & 9</mark>

- Deploy trained models using Ray Serve, enabling scalable API endpoints.
- Integrate monitoring tools (Prometheus, Grafana) for real-time performance tracking.
- Implement data drift detection and automate retraining triggers.

### Phase 4: Pipeline Orchestration <mark>Groups 10 & 11</mark>

- Orchestrate the end-to-end workflow using **Apache Airflow**.
- Automate task scheduling, error handling, and notifications.

### Phase 5: Infrastructure & Security <mark>Groups 12 & 13</mark>

- Configure and manage Ray clusters.
- Implement secure data access and Attribute Base Access Control (ABAC) management in Ray.

## Technical Requirements

1. Programming Language: Python 3.8+
2. Core Framework: Ray
3. Orchestration: Apache Airflow
4. Experiment Tracking: MLflow
5. Monitoring: Prometheus, Grafana
6. Cloud Storage: MinIO, or similar
7. Model: Llama or equivalent LLM

## Automation Focus

<mark>**Automation is a key requirement at every stage**</mark>:

- **Data ingestion must automatically scrape and validate new data**.
- **Model training should trigger on new data or drift detection**.
- **Deployment must support continuous integration and canary releases**.
- **Monitoring should provide alerts and auto-retraining capabilities**.
- **Orchestration must ensure seamless, hands-off pipeline execution**.

## Evaluation Criteria

- Correctness and completeness of each pipeline component
- Degree of automation achieved
- Scalability and efficiency of distributed components
- Quality of documentation and code readability
- Collaboration and integration between groups

## Notes:

- Deployment Environment: Pure code file are not acceptable, the deployment must be dockerize and tested on Kubernetes platform.

- Teams Collaboration: Teams must collaborate with each other, some parts required the other team results and team which are unwilling to collaborate will get negative point as mentioned in `Evaluation Criteria` section.

- Using any other tools for automation such as `n8n` is also acceptable but it must not limit the development/ deployment process, in other words it must be robust.

## References

- [MLOps: Continuous Delivery and Automation Pipelines in Machine Learning](https://ml-ops.org/)
- [Anyscale Documentation](https://docs.anyscale.com/)
- [Ray Documentation](https://docs.ray.io/en/latest/)
- [Ray Serve Deployment Graphs](https://docs.ray.io/en/latest/serve/production-guide.html#deployment-graphs)
- [Ray Tune for Hyperparameter Optimization](https://docs.ray.io/en/latest/tune/index.html)
- [Ray Train for Distributed Training](https://docs.ray.io/en/latest/train/index.html)
- [Ray Datasets](https://docs.ray.io/en/latest/data/dataset.html)
- [Airflow and Ray Integration](https://docs.anyscale.com/reference/integrations/airflow)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)
