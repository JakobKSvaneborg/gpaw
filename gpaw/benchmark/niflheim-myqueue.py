from myqueue.workflow import run

from gpaw.benchmark import benchmark_main 

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
        run(function=benchmark_main, args=key,
            nodename=name, cores=cores, processes=procs,
            gpus= gpus, tmax=time, name=key)
