def assert_legacy_gpaw(calc):
    if not calc.old:
        raise ValueError(
            'New-GPAW not supported.  '
            'Please use legacy GPAW like this: GPAW(..., legacy_gpaw=True)')