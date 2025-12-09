# creates: benchmark.csv, benchmark.png, systems.csv, score.png, H2-0.xyz
from gpaw.benchmark.plot import tables, plot_score, NAMES, REFERENCES
from pathlib import Path
import json


def plot() -> None:
    import matplotlib.pyplot as plt

    data = json.loads(Path('benchmarks.json').read_text())

    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    titles = [
        '$t_i^0/t_i$',
        'Second step [%]',
        'max_rss [Gbytes]']

    for n, (title, ax) in enumerate(zip(titles, axs)):
        for tag, (day, score, results) in data.items():
            Y = []
            for name in NAMES:
                if name in results:
                    r = results[name]
                    print(r)
                    print(REFERENCES[name]);asdg
                    if n == 0:
                        y = REFERENCES[name][3] / (r[1] + r[5])
                    elif n == 1:
                        y = 100 * r[5] / (r[1] + r[5])
                    else:
                        y = r[7] * 1e-9
                else:
                    y = None
                Y.append(y)
            ax.plot(Y, 'x-', label=f'{tag} {day} ({score:.1f})')
        ax.set_ylabel(title)
        if n == 2:
            ax.set_xticks(range(len(NAMES)), NAMES, rotation=70, ha='center')
            ax.legend()
        else:
            ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig('benchmark.png')


def main():
    plot()
    # plot_score()
    # tables()


if __name__ == '__main__':
    main()
