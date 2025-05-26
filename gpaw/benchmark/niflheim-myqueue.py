from myqueue.workflow import run

from gpaw.benchmark import benchmark_main, get_benchmarks
from os import makedirs

platforms = [('xeon24el8_test', 24, 24, 0, '10m'),
             ('a100', 8, 1, 1, '10m'),
             ('sm3090el8', 8, 1, 1, '10m'),
             ('epyc96', 96, 96, 0, '30m'),
             ('epyc96', 96 * 2, 96 * 2, 0, '30m')]


def workflow():
    for partition, ncores, nprocs, ngpus, time in platforms:
        for benchmark in get_benchmarks(
                cores=nprocs, memory='10000G', gpus=ngpus):
            makedirs(f'./{partition}-{nprocs}', exist_ok=True)
            run(function=benchmark_main, args=(benchmark,),
                nodename=partition, cores=ncores, processes=nprocs,
                gpus=ngpus, tmax=time,
                name=f'{benchmark}-{partition}-{nprocs}',
                folder=f'{partition}-{nprocs}')
