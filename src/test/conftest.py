# src/test/conftest.py
"""Shared external-dependency checks used to skip tests when a live service is unreachable."""

import requests


def ollama_available() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=1)
        return True
    except requests.RequestException:
        return False


def live_data_available() -> bool:
    try:
        requests.get("https://api.open-meteo.com/v1/forecast", timeout=3)
        return True
    except requests.RequestException:
        return False
