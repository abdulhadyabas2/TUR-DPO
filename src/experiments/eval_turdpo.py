from __future__ import annotations
import argparse
import json
from src.utils.logging_utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, required=True)
    ap.add_argument('--out', type=str, default='results/tables/eval_summary.json')
    args = ap.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    # For now, passthrough
    write_json(obj, args.out)
    print(json.dumps(obj, indent=2))


if __name__ == '__main__':
    main()
