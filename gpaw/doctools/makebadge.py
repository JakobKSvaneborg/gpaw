def coverage_badge(percentage):
    return makebadge(value=f'{percentage}%',
                     color=getcolor(float(percentage)))

def getcolor(x):
    return ('4c1' if x >= 95 else
            'a3c51c' if x >= 90 else
            'dfb317' if x >= 75 else
            'e05d44')


def makebadge(value, color):
    # Note: If text changes you need to revisit the widths.
    return """\
<svg xmlns="http://www.w3.org/2000/svg" width="106" height="20" role="img" aria-label="Coverage: {value}"><title>Coverage: {value}</title><linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="r"><rect width="106" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#r)"><rect width="63" height="20" fill="#555"/><rect x="63" width="43" height="20" fill="#{color}"/><rect width="106" height="20" fill="url(#s)"/></g><g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110"><text aria-hidden="true" x="325" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="530">Coverage</text><text x="325" y="140" transform="scale(.1)" fill="#fff" textLength="530">Coverage</text><text aria-hidden="true" x="835" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="330">{value}</text><text x="835" y="140" transform="scale(.1)" fill="#fff" textLength="330">{value}</text></g></svg>\
""".format(value=value, color=color)


if __name__ == '__main__':
    import sys
    print(coverage_badge(sys.argv[1]))
