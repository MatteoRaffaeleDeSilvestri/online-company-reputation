# Monitoring a company's online reputation

MachineInnovators Inc. is a leader in developing scalable, production-ready machine learning applications. The main focus of the project is to integrate MLOps methodologies to facilitate the development, implementation, continuous monitoring and retraining of sentiment analysis models. The goal is to enable the company to improve and monitor its reputation on social media through automatic sentiment analysis.

Businesses often face the challenge of managing and improving their social media reputation effectively and in a timely manner. Manually monitoring user sentiment can be inefficient and prone to human error, while the need to respond quickly to changes in user sentiment is crucial to maintaining a positive image of the company.

**Benefits of the solution**

1. **Automating sentiment analysis:** by implementing a FastText-based sentiment analysis model, MLOps Innovators Inc. will automate data processing from social media to identify positive, neutral, and negative sentiment. This will allow for a rapid and targeted response to user feedback

2. **Continuous reputation monitoring:** using MLOps methodologies, the company will implement a continuous monitoring system to assess user sentiment trends over time. This will enable changes in the perception of the company to be detected quickly and prompt action to be taken if necessary

3. **Model retraining:** introducing an automatic retraining system for the sentiment analysis model will ensure that the algorithm dynamically adapts to new data and variations in user language and behavior on social media. Maintaining the model's predictive accuracy is essential for proper sentiment assessment

**Project details**

- **Phase 1: implementing the sentiment analysis model with FastText**
    - **Model:** use a pre-trained FastText model for sentiment analysis that can classify texts from social media into positive, neutral, or negative sentiment. Use this model: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    - **Dataset:** use public datasets containing text and their respective sentiment labels

- **Phase 2: creating the CI/CD pipeline**
  - **CI/CD pipeline:** develop an automated pipeline for model training, integration testing, and application deployment to HuggingFace

- **Phase 3: deploy and continuous monitoring**
    - **Deploy on HuggingFace (optional):** implement the sentiment analysis model, including data and application, on HuggingFace to facilitate integration and scalability
    - **Monitoring system:** configure a monitoring system to continuously evaluate model performance and detected sentiment

- **Delivery**
    - **Source code:** public repository on GitHub with well-documented code for the CI/CD pipeline and model implementation. The actual delivery must take place via a Google Colab notebook with a link to the GitHub repository inside
    - **Documentation:** description of the design choices, implementations and results obtained during the project

**Motivation of the project**

Implementing FastText for sentiment analysis enables MLOps Innovators Inc. to significantly improve social media reputation management. By automating sentiment analysis, the company will be able to respond more quickly to user needs, improving satisfaction and strengthening the company's image in the market. With this project, MLOps Innovators Inc. promotes innovation in AI technologies, offering advanced and scalable solutions to modern business reputation management challenges.

## Project structure

<pre>
.
├── .github/
│   └── workflows/
|        ├── CI_CD_config.yml
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

