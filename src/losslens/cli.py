import argparse
import json

from .pipeline import run


def main():
    p = argparse.ArgumentParser(description="Fit exposure-aware pricing models and create the review pack")
    p.add_argument("command", choices=["run"], nargs="?", default="run")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--max-policies", type=int, help="Optional development sample; omitted for full data")
    a = p.parse_args()
    print(json.dumps(run(a.refresh, a.max_policies), indent=2))
