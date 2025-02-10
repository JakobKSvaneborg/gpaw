config = {
    'scheduler': 'slurm',
    'extra_args': ['--account=project_4xxxxxxxx', '--mem=0'],
    'mpiexec': 'srun',
    'parallel_python': 'gpaw python',
    'nodes': [
        ('standard', {'cores': 128}),
        ('small', {'cores': 128}),
        ('debug', {'cores': 128}),
        ('largemem', {'cores': 128}),
        ('standard-g', {'cores': 56}),
        ('small-g', {'cores': 56}),
        ('dev-g', {'cores': 56})]}
