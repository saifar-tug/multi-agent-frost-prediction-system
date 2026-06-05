# Frost Multi-Agent Prediction System

A modular multi-agent AI system for frost risk prediction and agricultural decision support using real meteorological observations from Austria.

---

## Project Objective

The goal of this project is to predict frost events from historical weather observations and demonstrate how multiple specialized agents can collaborate to generate actionable recommendations for agricultural decision support.

The system combines:

* Machine Learning (Random Forest)
* Rule-Based Agents
* Large Language Models (Llama 3 via Ollama)

to produce both predictions and human-readable explanations.

---

## Dataset

### Data Source

GeoSphere Austria

### Station Information

| Attribute          | Value                             |
| ------------------ | --------------------------------- |
| Station            | Graz Universität / Heinrichstraße |
| Latitude           | 47.08° N                          |
| Longitude          | 15.448056° E                      |
| Elevation          | 366 m                             |
| Period             | 2000–2026                         |
| Sampling Frequency | Daily                             |
| Observation Times  | 07:00, 14:00, 19:00 CET           |

---

## Target Variable

The current system uses the observed frost label:

```text
reif
```

where:

```text
1 = Frost observed
0 = No frost observed
```

This replaces the earlier temperature-threshold approach:

```python
tlmin <= 0
```

because the observed label represents actual recorded frost events.

---

## Features

The model uses meteorological variables including:

* Minimum temperature
* Maximum temperature
* Mean temperature
* Minimum soil temperature
* Relative humidity
* Vapor pressure
* Atmospheric pressure
* Cloud cover
* Visibility
* Precipitation
* Fog
* Dew
* Wind indicators

After preprocessing, the final modeling dataset contains:

* 9,240 observations
* 17 predictor variables
* 1 target variable (`frost`)

---

## Machine Learning Models

Two baseline models were evaluated:

### Logistic Regression

Used as a simple and interpretable baseline model.

### Random Forest

Used as the primary frost prediction model and integrated into the Weather Agent.

---

## Validation Strategy

Chronological time-series cross-validation was used to avoid data leakage.

### Fold 1

* Train: 2000-01-01 → 2004-03-19
* Test: 2004-03-20 → 2008-06-06

### Fold 2

* Train: 2000-01-01 → 2008-06-06
* Test: 2008-06-07 → 2012-08-24

### Fold 3

* Train: 2000-01-01 → 2012-08-24
* Test: 2012-08-25 → 2016-11-11

### Fold 4

* Train: 2000-01-01 → 2016-11-11
* Test: 2016-11-12 → 2021-01-29

### Fold 5

* Train: 2000-01-01 → 2021-01-29
* Test: 2021-01-30 → 2026-04-18

This approach ensures that future observations are never used during model training.

---

## Model Performance

| Model               | AUC    | F1     | Recall |
| ------------------- | ------ | ------ | ------ |
| Logistic Regression | 0.9575 | 0.6920 | 0.9428 |
| Random Forest       | 0.9643 | 0.7226 | 0.7457 |

The Random Forest achieved the strongest overall performance and was selected for deployment within the multi-agent architecture.

---

## Feature Importance

The most influential variables identified by the Random Forest model were:

| Feature                  | Importance |
| ------------------------ | ---------- |
| Soil Temperature Minimum | 0.254      |
| Air Temperature Minimum  | 0.226      |
| Air Temperature Mean     | 0.129      |
| Vapor Pressure Mean      | 0.115      |
| Air Temperature Maximum  | 0.065      |

These results are physically plausible because frost formation is strongly related to air and soil temperature conditions.

---

## Multi-Agent Architecture

### Weather Agent

Loads the trained Random Forest model and predicts:

* Frost probability
* Frost classification

### Soil Agent

Evaluates soil-related frost risk using minimum soil temperature.

Outputs:

* Soil temperature
* Soil risk level
* Explanation

### Planner Agent

Combines weather and soil assessments to determine:

* Priority level
* Recommended action

Example output:

```text
Activate frost protection.
```

### LLM Orchestrator

Uses a local Llama 3 model through Ollama to transform structured agent outputs into a concise human-readable decision support report.

The LLM does not perform prediction. Its role is to explain and summarize the outputs produced by the prediction and planning agents.

---

## Saved Artifacts

The training notebook exports the following artifacts:

| File                          | Purpose                                         |
| ----------------------------- | ----------------------------------------------- |
| random_forest_frost_model.pkl | Trained Random Forest model                     |
| feature_columns.pkl           | Feature ordering used during training           |
| sample_row_recent.pkl         | Latest available observation from the dataset   |
| sample_row_frost.pkl          | Historical frost example used for demonstration |

---

## Project Structure

```text
multi-agent-frost-prediction-system/

├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── random_forest_frost_model.pkl
│   ├── feature_columns.pkl
│   ├── sample_row_recent.pkl
│   └── sample_row_frost.pkl
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── src/
│   ├── agents/
│   │   ├── weather_agent.py
│   │   ├── soil_agent.py
│   │   ├── planner_agent.py
│   │   └── llm_orchestrator.py
│   │
│   └── main.py
│
└── README.md
```

---

## Current Status

* GeoSphere Austria dataset integrated
* Observed frost labels adopted
* Time-series validation implemented
* Logistic Regression baseline completed
* Random Forest baseline completed
* Feature importance analysis completed
* Multi-agent architecture operational
* Llama 3 orchestration operational
* End-to-end demonstration pipeline completed

---

## Example Workflow

```text
Meteorological Observation
            ↓
      Weather Agent
            ↓
        Soil Agent
            ↓
      Planner Agent
            ↓
     LLM Orchestrator
            ↓
 Decision Support Report
```

This project demonstrates how machine learning, agent-based reasoning, and large language models can be combined to support frost-risk assessment and agricultural decision making.

## Next Steps

* Atomation of the Data collection (API or data scraping)
* Automation of the Data Preprocessing
* Refine the Agentic Architecture and Behaviour 