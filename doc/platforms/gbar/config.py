config = {
    'scheduler': 'lsf',
    'nodes': [
        ('XeonE5_2650v4', {'cores': 24, 'memory': '252GB'}),
        ('XeonE5_2660v3', {'cores': 20, 'memory': '126GB'})],
    'extra_args': ['-q', 'hpc']}
