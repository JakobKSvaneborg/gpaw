from myqueue.workflow import run

from gpaw.benchmark import benchmark_main, get_benchmarks

platforms = [('xeon24el8_test', 24, 24, 0, '10m'),
             ('a100', 8, 1, 1, '10m'),
             ('sm3090_devel', 8, 1, 1, '10m'),
             ('epyc96', 96, 96, 0, '30m'),
             ('epyc96', 96 * 2, 96 * 2, 0, '30m')]

for partition, ncores, nprocs, ngpus, time in platforms:
    for benchmark in get_benchmarks(cores=nprocs, memory='10000G'):
        print(partition, ncores, nprocs, ngpus, time, benchmark)
asd
niflheim_target_nodes = {'C60_pw': ('xeon24el8_test', 24, 24, 0, '10m'),
                         'C60_lcao': ('xeon24el8_test', 24, 24, 0, '10m'),
                         'C60_lowpw_gpu': ('a100', 8, 1, 1, '10m'),
                         'C60_lowpw_float_gpu': ('sm3090_devel', 8, 1, 1, '10m'),
                         'MoS2_tube': ('epyc96', 96, 96, 0, '30m'),
                         'C676_graphene': ('epyc96', 96, 96, 0, '30m'),
                         'diamond_pw': ('xeon24el8_test', 24, 24, 0, '10m')}

def workflow():
    for key in niflheim_target_nodes:
        name, cores, procs, gpus, time = niflheim_target_nodes[key]
        run(function=benchmark_main, args=(key,),
            nodename=name, cores=cores, processes=procs,
            gpus= gpus, tmax=time, name=key)
