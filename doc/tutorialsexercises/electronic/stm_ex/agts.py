def workflow():
    from myqueue.workflow import run
    with run(script='HAl100.py'):
        run(function=go)


def go():
    import sys
    sys.argv = ['', 'HAl100.gpw']
    exec(open('stm.py').read())
