"""This module generates badges such as the coverage badge.

To make a nice custom badge, try for example:

  https://img.shields.io/badge/pw--perf--index-103.26-purple

On gitlab, these can be configured in Settings -> General (under "Badges").
Also maybe add them to README.
"""


import sys


def coverage_badge(percentage: str | int | float) -> str:
    # This creates a fancy coverage bagde for README, gitlab, etc.
    #
    # We cannot use gitlab's own coverage badge because it always reflects
    # the most recent pipeline, and the full coverage is only generated
    # nightly-or-manually.  Therefore this custom module.
    return makebadge(value=f'{percentage}%',
                     color=getcolor(float(percentage)))


def getcolor(x: int | float) -> str:
    return ('4c1' if x >= 95 else
            'a3c51c' if x >= 90 else
            'dfb317' if x >= 75 else
            'e05d44')


def makebadge(value: str, color: str) -> str:
    # Note: If text changes length you need to revisit the widths
    # including textLength, offsets etc., better use shields.io if you
    # need to generate a badge with different text lengths.
    width = 98
    width_small = 35

    return """\
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" role="img" aria-label="Coverage: {value}"><title>Coverage: {value}</title><linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="r"><rect width="{width}" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#r)"><rect width="63" height="20" fill="#555"/><rect x="63" width="{width_small}" height="20" fill="#{color}"/><rect width="{width}" height="20" fill="url(#s)"/></g><g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110"><text aria-hidden="true" x="325" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="530">Coverage</text><text x="325" y="140" transform="scale(.1)" fill="#fff" textLength="530">Coverage</text><text aria-hidden="true" x="795" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="250">{value}</text><text x="795" y="140" transform="scale(.1)" fill="#fff" textLength="250">{value}</text></g></svg>\
""".format(value=value, color=color, width=width, width_small=width_small)  # noqa

"""
<svg xmlns="http://www.w3.org/2000/svg" width="136" height="20" role="img" aria-label="pw-perf-index: 103.26">
<title>pw-perf-index: 103.26</title>
<linearGradient id="s" x2="0" y2="100%">
<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
<stop offset="1" stop-opacity=".1"/>
</linearGradient>
<clipPath id="r">
<rect width="136" height="20" rx="3" fill="#fff"/>
</clipPath><g clip-path="url(#r)">
<rect width="87" height="20" fill="#555"/>
<rect x="87" width="49" height="20" fill="purple"/><rect width="136" height="20" fill="url(#s)"/></g><g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110"><text aria-hidden="true" x="445" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="770">pw-perf-index</text><text x="445" y="140" transform="scale(.1)" fill="#fff" textLength="770">pw-perf-index</text><text aria-hidden="true" x="1105" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="390">103.26</text><text x="1105" y="140" transform="scale(.1)" fill="#fff" textLength="390">103.26</text></g></svg>"""

def main(args: list[str]) -> None:
    print(coverage_badge(args[1]))


if __name__ == '__main__':
    main(sys.argv)
