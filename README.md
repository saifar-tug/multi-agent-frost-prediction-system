# Frost Multi-Agent System

A modular multi-agent AI system for frost risk prediction and agricultural decision support.

## Phase 1
- Weather agent (frost prediction)
- Soil agent (crop vulnerability)
- Planner agent (contingency logic)
- LLM coordinator (structured explanation)

## Project Structure
- src/agents: agent implementations
- src/models: ML models
- data/: datasets
- notebooks/: experiments


## Phase 1: Local Multi-Agent Frost Prototype
### Objective

Phase 1 aims to validate the modular multi-agent architecture using real meteorological data and to establish a reproducible baseline frost prediction model.

### Dataset

Daily minimum temperature data were obtained from the European Climate Assessment & Dataset (ECA&D) for the GRAZ-UNIVERSITAET station. The dataset was filtered to the period 1990–2024 and cleaned to remove missing or suspect observations.

Frost labels were defined as days with minimum temperature ≤ 0°C.

t = an index inside our historical dataset.
Tmin(t) → minimum temperature today
Tmin(t−1) → minimum temperature yesterday
Tmin(t−2) → minimum temperature two days ago
Frost(t) → frost today
Frost(t+1) → frost tomorrow

Our dataset ends in:
2023-06-27 (after filtering)

So inside our dataset:
t = 1990-01-01 (first sample)
t = 1990-01-02
...
t = 2023-06-26
t = 2023-06-27 (last usable sample)

When we say:
Frost(t+1)

we mean:
Frost on the next row in the dataset

### System Components

The current implementation consists of:

Weather Agent
Trains a logistic regression model to classify frost events using daily minimum temperature.
Soil Agent
Computes crop vulnerability based on soil temperature and moisture inputs.
Planner Agent
Generates contingency decisions based on frost probability and crop vulnerability.
LLM Coordinator
Produces structured, human-readable explanations of system decisions.

### Current Status
Real Austrian meteorological data integrated
Modular multi-agent architecture operational
Frost classifier trained on 11,988 daily samples
End-to-end system producing coordinated decisions

Phase 1 establishes architectural correctness and data integration. Subsequent phases will extend the system toward forecasting, benchmarking, knowledge grounding (RAG), and integration with large-scale climate data infrastructures.