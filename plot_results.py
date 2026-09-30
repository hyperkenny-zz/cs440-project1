"""Optional plotting dependency: python3 -m pip install matplotlib."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with args.summary.open() as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(8,5))
    for bot in range(1,5):
        group = sorted((r for r in rows if int(r['bot']) == bot), key=lambda r: float(r['q']))
        xs = [float(r['q']) for r in group]
        ys = [float(r['success_rate']) for r in group]
        ax.plot(xs, ys, marker='o', label=f'Bot {bot}')
        ax.fill_between(xs, [float(r['ci_low']) for r in group],
                        [float(r['ci_high']) for r in group], alpha=.12)
    ax.set(xlabel='Flammability q', ylabel='Success frequency', ylim=(0,1.02),
           title='Ship fire: success frequency and 95% Wilson intervals')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    output = args.summary.with_name('success_rates.png')
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == '__main__':
    main()
