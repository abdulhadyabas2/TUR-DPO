from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def exact_match(pred: str, ref: str) -> int:
    return int(pred.strip().lower() == ref.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', type=str, required=True, help='JSONL with fields: prompt, pred, ref')
    ap.add_argument('--out', type=str, default='results/tables/em_results.csv')
    args = ap.parse_args()

    rows = []
    with open(args.preds, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            em = exact_match(obj.get('pred',''), obj.get('ref',''))
            rows.append({'prompt': obj.get('prompt',''), 'pred': obj.get('pred',''), 'ref': obj.get('ref',''), 'em': em})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['prompt','pred','ref','em'])
        w.writeheader()
        w.writerows(rows)
    acc = sum(r['em'] for r in rows) / max(1, len(rows))
    print(json.dumps({'n': len(rows), 'em': acc}, indent=2))


if __name__ == '__main__':
    main()