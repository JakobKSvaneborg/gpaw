from gpaw.benchmark.performance_index import workflow as wf


def workflow():
    wf(skip=['ErGe-2M', 'Fe8O8-3M'])
