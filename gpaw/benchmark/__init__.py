import collections.abc
from copy import deepcopy
from pprint import pp
import numpy as np
from gpaw.mpi import world
from time import time
from json import dumps
from pathlib import Path

from gpaw.benchmark.systems import parse_system

# A parameter set is a 2-tuple with dictionary for gpaw-parameters,
# and additional dictionary with named sub parameter sets

pw_default_parameters = {'mode': {'name': 'pw', 'ecut': 400}}

pw_parameter_subsets = {'high': {'mode': {'ecut': 800}},
                        'low': {'mode': {'ecut': 400}},
                        'float32': {'mode': {'dtype': np.float32}}}

lcao_default_parameters = {'mode': {'name': 'lcao'}}

lcao_parameter_subsets = {'sz': {'basis': 'sz(dzp)'},
                          'dzp': {'basis': 'dzp'}}

kpts_parameter_subsets = {'gamma': {'kpts': (1, 1, 1)},
                          'density6': {'kpts': {'density': 6}},
                          '411': ({'kpts': (4, 1, 1)})}

xc_parameter_subsets = {'PBE': {'xc': 'PBE'},
                        'LDA': {'xc': 'LDA'}}

gpu_default_parameters = {'parallel': {'gpu': True}, 'random': True}

parallel_parameter_subsets = {'parallel': {'scalapack':
                              {'parallel': {'sl_auto': True}}}}

gpaw_parameter_sets = {'pw': (pw_default_parameters, pw_parameter_subsets),
                       'lcao': (lcao_default_parameters,
                                lcao_parameter_subsets),
                       'kpts': ({}, kpts_parameter_subsets),
                       'gpu': (gpu_default_parameters, {}),
                       'xc': ({}, xc_parameter_subsets),
                       'parallel': ({}, parallel_parameter_subsets)}


benchmarks_str = """\
C60_pw               C60-pw.high:kpts.gamma                                low_req
C60_lcao             C60-lcao.dzp                                          low_req
C60_lowpw_gpu        C60-pw.low:kpts.gamma:gpu                             low_req_gpu
C60_lowpw_float_gpu  C60-pw.low.float32:kpts.gamma:gpu                     low_req_gpu
MoS2_tube            MoS2_tube-pw.high:kpts.411:xc.PBE:parallel.scalapack  low_req
676_graphene         676_graphene-pw:kpts.gamma:xc.PBE:parallel.scalapack  high_req
diamond_pw           diamond-pw.high:kpts.density6                         low_req"""

low_req = {'mincores': 16, 'maxcores': 256, 'minmem': '4G'}
low_req_gpu = {'mincores': 1, 'maxcores': 4, 'minmem': '4G'}
high_req = {'high_req': 96, 'maxcores': 96, 'minmem': '4G'}

requirements = {'low_req': low_req,
                'low_req_gpu': low_req_gpu,
                'high_req': high_req}

benchmarks = {}
for benchmark_line in benchmarks_str.split('\n'):
    for nickname, definition, req in benchmark_line.split():
        benchmarks[nickname] = definition, requirements[req]


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_parameters(parameter_sets):
    """Parses parameter_sets descriptor into a dictionary

    First, individual parameter sets are separated by :
    And a single parameter set contains first the main paramter descriptor,
    which can be further refined by . for parameters subsets.

    For example valid paramter strings are:
         pw.high:gamma
         pw.high:gamma:parallel.gpu
         lcao.dzp:kpt.density4:noscalapack
    """

    kwargs = {}
    parameter_sets = parameter_sets.split(':')
    for parameter_set in parameter_sets:
        firstsplit = parameter_set.split('.', 1)
        if len(firstsplit) == 1:
            firstsplit.append(None)
        print(firstsplit)
        set_name, parameter_subsets = firstsplit
        default_parameter_set, subsets = gpaw_parameter_sets[set_name]
        recursive_update(kwargs, deepcopy(default_parameter_set))
        if parameter_subsets is None:
            continue
        for subsetname in parameter_subsets.split('.'):
            recursive_update(kwargs, deepcopy(subsets[subsetname]))
    return kwargs


def benchmarks_error(name):
    err = f'Cannot find benckmark with name {name}\n\n'
    err += 'Available benchmarks\n'
    header = '{:20s} | {:35s}\n'.format('name', 'system-parameter sets')
    err += header + '-' * len(header) + '\n'

    for benchmark, system_and_parameter_set in benchmarks.items():
        err += f'{benchmark:20s} | {system_and_parameter_set:35s}\n'
    return err


def shell_command(cmd):
    import subprocess
    try:
        output = subprocess.run(cmd.split(' '),
                                capture_output=True,
                                text=True,
                                check=True,
                                shell=True).stdout
    except subprocess.CalledProcessError as e:
        output = f'{e.output} {e.stderr}'

    return output


def gather_system_information():
    return {'processor': shell_command('lscpu'),
            'memory': shell_command('lsmem'),
            'mpi-ranks': world.size,
            'nvidia-smi': shell_command('nvidia-smi'),
            'rocm-smi': shell_command('rocm-smi'),
            'hostname': shell_command('hostname')}


def benchmark_atoms_and_calc(name):
    from gpaw.new.ase_interface import GPAW

    # Replace nickname with long name
    if '-' not in name:
        if name in benchmarks:
            name = benchmarks[name]
        else:
            raise Exception(benchmarks_error(name))

    system, parameter_sets = name.split('-')
    atoms = parse_system(system)
    parameters = parse_parameters(parameter_sets)
    pp(parameters, indent=4, sort_dicts=True)
    atoms.calc = GPAW(**parameters)
    return atoms, atoms.calc


def gs_and_move_atoms(name):
    atoms, calc = benchmark_atoms_and_calc(name)
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    atoms.positions += 0.1 * F
    atoms.get_potential_energy()
    return {'energy': E,
            'forces': F.tolist()}


class Benchmark:
    def __init__(self, system_info):
        self.error = None
        self.system_info = system_info
        self.results = None

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            self.error = (exc_type, exc_value, exc_traceback)
        self.end = time()

    @property
    def walltime(self):
        return self.end - self.start

    def todict(self):
        return {'system_info': self.system_info,
                'walltime': self.walltime,
                'error': self.error,
                'results': self.results}

    def write_json(self, fname):
        Path(fname).write_text(dumps(self.todict()))


def benchmark_main(name):
    if world.rank == 0:
        system_info = gather_system_information()
        print('Running benchmark', name)
    else:
        system_info = None

    world.barrier()
    with Benchmark(system_info) as results:
        results.results = gs_and_move_atoms(name)
    if world.rank == 0:
        results.write_json(f'{name}-benchmark.json')


def parse_mem(memstr):
    mul = {'G': 1024**3,
           'M': 1024**2,
           'K': 1024**1}[memstr[-1]]
    return float(memstr[:-1]) * mul


def get_benchmarks(memory='8G', cores=16, gpu=False):
    for benchmark, (long_name, requirements) in benchmarks.items():
        if cores < requirements.get('mincores', 1):
            continue
        if cores > requirements.get('maxcores', np.inf):
            continue
        if parse_mem(memory) <= parse_mem(requirements.get('minmem', np.inf)):
            continue
        if ('gpu' in long_name) != gpu:
            continue
        yield benchmark
