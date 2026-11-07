#!/usr/bin/env python
"""
Script to generate the dataset and save validation report.
"""
import os
import sys

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ola_support_agent.dataset import main

if __name__ == "__main__":
    main()