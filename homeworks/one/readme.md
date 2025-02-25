# First Homework 
Foundational Cloud Infrastructure and Performance Analysis of a Containerized LLM Inference Service using Docker Ecosystem

## Description
This introductory homework assignment is designed to establish a foundational understanding of core cloud infrastructure technologies, specifically focusing on containerization, multi-container orchestration, and basic service orchestration using the Docker ecosystem.  Within the context of a cloud-native paradigm, this exercise explores the practical application of Docker, Docker Compose, and Docker Swarm for deploying and managing a representative microservice: <mark>an LLM-based sentiment analysis inference service</mark>.

The assignment emphasizes empirical performance analysis as a critical aspect of cloud deployments. Students will systematically progress through containerizing the LLM service with Docker, orchestrating it as a multi-container application using Docker Compose, and subsequently deploying and scaling it using Docker Swarm.  At each stage, students are tasked with rigorously measuring and documenting key performance indicators, namely, **inference latency and resource utilization** (CPU and memory).

This methodical approach enables students to directly observe and quantify the impact of containerization and orchestration on the performance characteristics of the LLM inference service. By comparing performance metrics across different deployment configurations, students will gain practical insights into the benefits and potential trade-offs associated with each Docker tool in terms of latency and resource consumption.

## Objective

*   **Containerize an Application using Docker:**  Successfully package an LLM inference service into a Docker container, demonstrating proficiency in Dockerfile creation and image management.
*   **Orchestrate Multi-Container Applications with Docker Compose:**  Utilize Docker Compose to define and manage a single-service application, understanding the benefits of declarative configuration for containerized workloads.
*   **Deploy and Scale Services with Docker Swarm:**  Employ Docker Swarm to deploy the LLM inference service as a scalable service, exploring basic service orchestration and replica management.
*   **Measure and Analyze Performance Metrics:**  Implement and execute a performance measurement methodology to quantify inference latency and resource utilization (CPU and memory) for the LLM inference service across different Docker deployment configurations.
*   **Compare Performance across Docker Tools:**  Analyze and compare the measured performance data obtained from Docker containers, Docker Compose, and Docker Swarm deployments, drawing informed conclusions about the performance implications of each technology.
*   **Establish a Baseline for Cloud Performance Analysis:**  Develop a foundational understanding of performance metrics and measurement techniques in cloud environments, setting the stage for more sophisticated monitoring and analysis in subsequent assignments and research endeavors.