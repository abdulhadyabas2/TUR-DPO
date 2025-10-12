from __future__ import annotations
import argparse
import csv
from src.utils.plotting import plot_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, required=True, help='CSV with x,y columns')
    ap.add_argument('--out', type=str, default='results/figures/curve.png')
    args = ap.parse_args()

    xs, ys = [], []
    with open(args.input, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row['x']))
            ys.append(float(row['y']))

    plot_curve(xs, ys, xlabel='x', ylabel='y', title='Demo curve', out_path=args.out)
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
