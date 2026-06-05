# src/utils/data_generator.py
# this was for the initial structure of the project, but we ended up not using it in the final version.

import numpy as np
import pandas as pd


def generate_weather_data(n_samples=500):
    np.random.seed(42)

    temperature = np.random.normal(loc=3, scale=5, size=n_samples)
    humidity = np.random.uniform(0.3, 1.0, n_samples)

    #define frost condition
    frost = (temperature <= 0).astype(int)

    df = pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "frost": frost
    })

    return df