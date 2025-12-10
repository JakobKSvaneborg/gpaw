"""This module generates badges such as the coverage badge.

To make a nice custom badge, try for example:

  https://img.shields.io/badge/pw--perf--index-103.26-purple

On gitlab, these can be configured in Settings -> General (under "Badges").
Also maybe add them to README.
"""
import json
import sys
from pathlib import Path


def coverage_badge(percentage: str | int | float) -> str:
    # This creates a fancy coverage bagde for README, gitlab, etc.
    #
    # We cannot use gitlab's own coverage badge because it always reflects
    # the most recent pipeline, and the full coverage is only generated
    # nightly-or-manually.  Therefore this custom module.
    return makebadge('Coverage', f'{percentage}%',
                     color=getcolor(float(percentage)))


def getcolor(x: float) -> str:
    return ('4c1' if x >= 95 else
            'a3c51c' if x >= 90 else
            'dfb317' if x >= 75 else
            'e05d44')


def perf_index(mode: str) -> str:
    import gpaw_web_page_data
    dir = Path(gpaw_web_page_data.__file__).parent
    data = json.loads((dir / 'benchmarks/benchmarks.json').read_text())
    last, latest = (score for day, score in data['scores'][mode][-2:])
    change = (latest - last) / last * 100
    score = f'{last:.1f} ({change:+.1f})'
    return makebadge(
        f'{mode.upper()}-perf-index',
        score,
        getcolor(80 + change * 10))


def makebadge(text1: str, text2: str, color='purple') -> str:
    t1 = len(text1) / 13 * 770
    w1 = t1 / 10 + 10
    t2 = len(text2) / 6 * 390
    w2 = t2 / 10 + 10
    return f"""\
    <svg xmlns="http://www.w3.org/2000/svg"
     width="{w1 + w2}"
     height="20"
     role="img"
     aria-label="{text1} {text2}">
     <title>{text1} {text2}</title>
     <linearGradient id="s" x2="0" y2="100%">
      <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
      <stop offset="1" stop-opacity=".1"/>
     </linearGradient>
     <clipPath id="r">
      <rect width="{w1 + w2}" height="20" rx="3" fill="#fff"/>
     </clipPath>
     <g clip-path="url(#r)">
      <rect width="{w1}" height="20" fill="#555"/>
      <rect x="{w1}" width="{w2}" height="20" fill="{color}"/>
      <rect width="{w1 + w2}" height="20" fill="url(#s)"/>
     </g>
     <g fill="#fff" text-anchor="middle"
      font-family="Verdana,Geneva,DejaVu Sans,sans-serif"
      text-rendering="geometricPrecision"
      font-size="110">
      <text aria-hidden="true"
       x="{50 + t1 / 2}" y="150" fill="#010101" fill-opacity=".3"
       transform="scale(.1)" textLength="{t1}">
       {text1}
      </text>
      <text x="{50 + t1 / 2}" y="140"
       transform="scale(.1)" fill="#fff" textLength="{t1}">
       {text1}
      </text>
      <text aria-hidden="true"
       x="{w1 * 10 + 50 + t2 / 2}" y="150"
       fill="#010101" fill-opacity=".3"
       transform="scale(.1)" textLength="{t2}">
       {text2}
      </text>
      <text x="{w1 * 10 + 50 + t2 / 2}" y="140"
       transform="scale(.1)" fill="#fff" textLength="{t2}">
       {text2}
      </text>
     </g>
    </svg>"""


def main(args: list[str]) -> None:
    if args[1] == 'coverage':
        print(coverage_badge(args[2]))
    else:
        print(perf_index(mode=args[1]))


if __name__ == '__main__':
    main(sys.argv)
