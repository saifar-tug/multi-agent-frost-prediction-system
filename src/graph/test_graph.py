# src/graph/test_graph.py

import joblib

from src.graph.frost_graph import (
    frost_graph
)


sample_data = joblib.load(
    "models/sample_row_frost.pkl"
)

result = frost_graph.invoke(
    {
        "question": (
            "Should I activate frost protection?"
        ),
        "sample_data": sample_data
    }
)

print("\nFINAL RESPONSE\n")

print(
    result["final_response"]
)