#!/usr/bin/env python3
"""
Stand-in for sync_manifest_status.py used by test_refresh_dashboard_sh.py: echoes back
whatever the test wrote into --pr-data's file as the "corrected" list, instead of
reading a real plan.yaml and computing a correction from live pull request state.
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--plan")
parser.add_argument("--pr-data")
arguments = parser.parse_args()

with open(arguments.pr_data) as pull_request_data_file:
    corrected = json.load(pull_request_data_file)

print(json.dumps({"corrected": corrected}))
