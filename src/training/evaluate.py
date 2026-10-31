import argparse
import json
import sys

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--gate", type=float, required=True)
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
name = cfg["name"]
metrics = json.load(open(f"metrics/{name}.json"))
passed = metrics["value"] >= args.gate

result = {"metric": metrics["metric"], "value": metrics["value"],
          "gate": args.gate, "passed": passed}
with open(f"metrics/{name}_gate.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"{name}: {metrics['metric']}={metrics['value']} gate={args.gate} passed={passed}")
sys.exit(0 if passed else 1)