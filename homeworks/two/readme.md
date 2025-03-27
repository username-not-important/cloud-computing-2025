# Second Homework

**MapReduce, Hadoop, and Spark Technologies for Cloud Computing in the Context of Federated Learning**

Due date until <mark>1404-02-07</mark> at [this](https://forms.gle/XoVpXBufGLWH3WeT6) google form.

## Objective

 To understand the core concepts of MapReduce, Hadoop, and Spark and analyze their potential applications and synergies within the context of Federated Learning.

## Homework Assignment Outline

### Part 1: Foundational Knowledge & Environment Setup(20%)

Define and explain the core concepts of:

- MapReduce, including its Map, Shuffle, and Reduce phases.
- Hadoop, including its core modules (HDFS, YARN, MapReduce).
- Spark, highlighting its key features like RDDs/DataFrames.
- Describe the architectural relationship between Hadoop and MapReduce, detailing how Hadoop provides the necessary infrastructure for MapReduce jobs.

Using docker & docker compose with knowledge coming from previous homework refer to:

- [**docker-hadoop**](https://github.com/bigdatafoundation/docker-hadoop) [_tested_],
- [**docker-spark**](https://github.com/big-data-europe/docker-spark) [_tested_], or
- any available docker deployment on the internet.

for running Hadoop and Spark. You also can locally install them on your system but it is time consuming and might cause lots of issues, do it with your own responsibility.

**Deliverables:**

Written Analysis

### Part 2: Comparative Analysis (10%)

Compare and contrast Spark and Hadoop in terms of:

- Processing speed and the reasons behind the differences.
- How they handle data and their respective data structures (HDFS files vs. RDDs/DataFrames).
- Typical use cases where each technology excels (batch processing vs. versatile workloads).

**Deliverables:**

Written Analysis

### Part 3: Practical Application in Federated Learning (70%)

#### Scenario

Imagine a distributed environment where multiple clients hold local datasets. The goal is to train a simple model (e.g., averaging) across these datasets without sharing the raw data.

#### Tasks

- Explain the principles of Federated Learning and its advantages in privacy-preserving distributed machine learning.
- [Hadoop: Distributed Data Processing for Initial Aggregation](./docs/task_1.md)
- [Spark: Global Model Aggregation and Update](./docs/task_2.md)
- [Analysis](./docs/task_3.md)

## Grading Rubric

 The report will be evaluated based on the accuracy and completeness of the definitions, the depth of the comparative analysis, the understanding of Federated Learning principles, and the thoughtfulness of the discussion on the potential applications and synergies of MapReduce, Hadoop, and Spark within the context of Federated Learning. Specific attention will be paid to the clarity of explanations and the use of supporting details.