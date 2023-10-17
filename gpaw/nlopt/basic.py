import numpy as np
from pathlib import Path
from gpaw.mpi import world, broadcast


def load_data(mml_name='mml.npz'):
    """
    Load the data and distribute among cores

    Input:
        mml_name        NLO data filename (default mml.npz)
    Output:

        p_kvnn          The mometum matrix elements, dimension (nk,3,nb,nb)
    """

    # Load the data to the master
    if world.rank == 0:
        nlo = np.load(mml_name)
        # print(nlo.files)
        # print(nlo['p_kvnn'])
    else:
        nlo = dict.fromkeys(['w_sk', 'f_skn', 'E_skn', 'p_skvnn'])

    # Distribute the data among cores
    k_info = distribute_data(
        [nlo['w_sk'], nlo['f_skn'], nlo['E_skn'], nlo['p_skvnn']])

    return k_info


def distribute_data(arr_list):
    """
    Distribute the data among the cores

    Input:
        arr_list        A list of numpy array (the first two should be s,k)
    Output:
        k_info          A  dictionary of data with key of s,k index
    """

    # Check the array shape
    size = world.size
    rank = world.rank
    if rank == 0:
        nk = 0
        arr_shape = []
        for ii, arr in enumerate(arr_list):
            ar_shape = arr.shape
            arr_shape.append(ar_shape)
            if nk == 0:
                ns = ar_shape[0]
                nk = ar_shape[1]
            else:
                assert ar_shape[1] == nk, 'Wrong shape for array.'
    else:
        arr_shape = None
        nk = None
        ns = None
    arr_shape = broadcast(arr_shape, root=0)
    nk = broadcast(nk, root=0)
    ns = broadcast(ns, root=0)

    # Distribute the data of k-points between cores
    k_info = {}

    # Loop over k points
    for s1 in range(ns):
        for kk in range(nk):
            if rank == 0:
                if kk % size == rank:
                    k_info[s1 * nk + kk] = [arr[s1, kk] for arr in arr_list]
                else:
                    for ii, arr in enumerate(arr_list):
                        data_k = np.array(arr[s1, kk], dtype=complex)
                        world.send(
                            data_k, kk % size, tag=ii * nk + kk)
            else:
                if kk % size == rank:
                    dataset = []
                    for ii, cshape in enumerate(arr_shape):
                        data_k = np.empty(cshape[2:], dtype=complex)
                        world.receive(data_k, 0, tag=ii * nk + kk)
                        dataset.append(data_k)
                    k_info[s1 * nk + kk] = dataset

    return k_info


def is_file(filename):
    """
    Check if the file exist

    Input:
        filename        Filename to check
    Output:
        file_exist      Flag for file
    """
    if world.rank == 0:
        if (not Path(filename).is_file()):
            file_exist = True
        else:
            file_exist = False
    else:
        file_exist = None

    file_exist = broadcast(file_exist, 0)

    return file_exist
