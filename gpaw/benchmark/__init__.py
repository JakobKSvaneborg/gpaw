import collections.abc
from copy import deepcopy
from pprint import pp
import numpy as np


# A parameter set is a 2-tuple with dictionary for gpaw-parameters,
# and additional dictionary with named sub parameter sets

pw_default_parameters = {'mode': {'name': 'pw', 'ecut': 400}}

pw_parameter_subsets = {'high': {'mode': {'ecut': 800}},
                        'low': {'mode': {'ecut': 400}},
                        'float32': {'mode': {'dtype': np.float32}}}

lcao_default_parameters = {'mode': {'name': 'lcao'}}

lcao_parameter_subsets = {'sz': {'basis': 'sz(dzp)'},
                          'dzp': {'basis': 'dzp'}}

general_parameter_sets = {'gamma': ({'kpts': (1, 1, 1)}, {}),
                          'kpts': ({'kpts': {'density': 6}}, {}),
                          'parallel': ({},
                                       {'scalapack': {'parallel': {'sl_auto': True}}})}


gpu_default_parameters = {'parallel': {'gpu': True}, 'random': True}

gpaw_parameter_sets = {'pw': (pw_default_parameters, pw_parameter_subsets),
                       'lcao': (lcao_default_parameters, lcao_parameter_subsets),
                       'gpu': (gpu_default_parameters, {}),
                       **general_parameter_sets}


def system_magic_graphene():
    from gpaw.benchmark.generate_twisted import make_heterostructure
    from ase.build import graphene
    atoms = graphene(vacuum=5)
    transa_cc = np.array([[29, -30, 0], [59, 29, 0], [0, 0, 1]])
    transb_cc = np.array([[30, -29, 0], [59, 30, 0], [0, 0, 1]])
    atoms = make_heterostructure(atoms, atoms,
                                 transa_cc=transa_cc,
                                 transb_cc=transb_cc,
                                 straina_vv=np.eye(3),
                                 interlayer_dist=3.35)
    return atoms

def system_6000_bl_graphene():
    from gpaw.benchmark.generate_twisted import make_heterostructure
    from ase.build import graphene
    atoms = graphene(vacuum=5)
    transa_cc = np.array([[23, 45, 0], [-22, 23, 0], [0, 0, 1]])
    transb_cc = np.array([[22, 45, 0], [-23, 22, 0], [0, 0, 1]])
    atoms = make_heterostructure(atoms, atoms,
                                 transa_cc=transa_cc,
                                 transb_cc=transb_cc,
                                 straina_vv=np.eye(3),
                                 interlayer_dist=3.35)
    return atoms

def system_C60():
    from ase.build import molecule
    atoms = molecule('C60')
    atoms.center(vacuum=5)
    return atoms

def system_diamond():
    from ase.build import bulk
    atoms = bulk('C')
    return atoms


systems = {'C60': system_C60,
           'diamond': system_diamond}

benchmarks =  {'C60_pw': 'C60-pw.high:gamma',
               'C60_lcao': 'C60-lcao.dzp',
               'C60_lowpw_gpu': 'C60-pw.low:gamma:gpu',
               'C60_lowpw_float_gpu': 'C60-pw.low.float32:gamma:gpu',
               'diamond_pw': 'diamond-pw.high:kpts'}

        
def parse_system(name):
    return systems[name]()


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
    header = '{:15s} | {:15s} | {:15s}\n'.format('name','system', 'parameter sets')
    err += header + '-' * len(header) + '\n'

    for benchmark, (system, parameter_set) in benchmarks.items():
        err += f'{benchmark:15s} | {system:15s} | {parameter_set:15s}\n'
    return err


def benchmark_atoms_and_calc(name):
    from gpaw.new.ase_interface import GPAW
    if '-' in name:
        system, parameter_sets = name.split('-')
    else:
        try:
            system, parameter_sets = benchmarks[name]
        except KeyError:
            print(benchmarks_error(name))
            raise
    atoms = parse_system(system)
    parameters = parse_parameters(parameter_sets)
    pp(parameters, indent=4, sort_dicts=True)
    atoms.calc = GPAW(**parameters)
    return atoms, atoms.calc 


def benchmark_main(name):
    atoms, calc = benchmark_atoms_and_calc(name)
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    atoms.positions += 0.1 * F
    atoms.get_potential_energy()
