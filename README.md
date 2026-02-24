# Monitoring a company's online reputation

This project implements a scalable **MLOps ecosystem** for online reputation management by automating the end-to-end lifecycle of a sentiment analysis system. 

Technically, the solution centers on fine-tuning a **transformer-based architecture**, specifically the `cardiffnlp/twitter-roberta-base-sentiment-latest` model, integrated with **FastText** for efficient text classification. 

The architecture is operationalized through a robust **CI/CD pipeline** that automates model training, integration testing,

The built model was finally released on **Hugging Face** and the project is also available on **Docker**.

## Project structure

<pre>
.
├── .github/
│   └── workflows/
|        ├── CI_CD_config.yml
├── datasets/
│   ├── product_reviews.csv
├── src/
│   ├── main.py
│   ├── test.py
│   └── train.py
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── prometheus.yml
├── README.md
└── requirements.txt
</pre>

## File details

### Model building

* **src/main.py:** the script uses **FastAPI** to create a RESTful web service that serves sentiment predictions. Critically, it incorporates **Prometheus** instrumentation to track live performance metrics—such as request volume and latency—fulfilling the "continuous monitoring" requirement of the MLOps lifecycle;

* **src/train.py:** the script automates the fine-tuning of the **Twitter-RoBERTa** model using a custom dataset. It performs end-to-end data preprocessing, manages the training loop via the Hugging Face `Trainer` API, and exports both the optimized model weights and technical performance metrics (*Accuracy* and *F1-Score*) for version control;

* **src/test.py:** the script implements **Automated Integration Testing** using the `pytest` framework and FastAPI's `TestClient`. It simulates real-user interactions by sending sample social media text to the API and verifying that the response matches the expected technical schema (labels and confidence scores);

* **datasets/product_reviews.csv:** the `product_reviews.csv` file is a structured dataset containing user-generated text paired with sentiment classifications (*Positive*, *Neutral*, *Negative*). In the `train.py` script, this data is loaded via Pandas and converted into a Hugging Face `Dataset` object, forming the foundation for the model's ability to learn and classify online company reputation accurately. The original dataset can be found [here](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset).

### CI/CD pipeline

* **.github/workflows:** the script automates a **Continuous Integration (CI)** pipeline that triggers whenever code is pushed to the `main` branch. It sets up a standardized Linux environment, installs the machine learning dependencies, performs "linting" (code style checks), and executes the automated tests to prevent bugs from reaching production;

### Docker package

* **Docker:** this script defines the **containerization strategy** for the sentiment analysis API. It uses a lightweight Python 3.12 base image, installs the necessary machine learning libraries, and configures the container to launch the FastAPI server (`main.py`) as its primary process upon startup;

* **docker-compose.yml:** the script defines a multi-container environment where the **FastAPI application**, a **Prometheus** time-series database, and a **Grafana** dashboard work in tandem. This setup transitions the project from a simple script into a production-ready "Observability Stack," enabling the "Continuous Monitoring" required by the MLOps methodologies;

* **promethues.yml:** the script defines a **"Scrape Job"** that targets the FastAPI application. It instructs the monitoring engine to ping the service every 15 seconds to pull the current metrics, such as request counts and model latency, which are then stored for visualization in Grafana.

### Utilities

* .gitignore;
* LICENSE;
* README.md;
* requirements.txt.

