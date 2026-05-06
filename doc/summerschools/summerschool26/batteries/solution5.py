for xc in ['LDA', 'PBE', 'DFTD3']:
    calcname = f'Li-C6-{xc}'
    Li_gra = Atoms('CCCCCCLi', positions=[(0, 0, 0), (0, ccdist, 0), (a, 0, 0),
                                          (-a, 0, 0), (-a / 2, -ccdist / 2, 0),
                                          (a / 2, -ccdist / 2, 0), (0, -ccdist, c / 2)],
                   cell=([1.5 * a, -1.5 * ccdist, 0],
                         [1.5 * a, 1.5 * ccdist, 0],
                         [0, 0, c]),
                   pbc=(1, 1, 1))

    if xc == 'DFTD3':
        dft = GPAW(mode=PW(500), kpts=(5, 5, 6), xc='PBE', txt=calcname + '.log')
        calc = DFTD3(dft=dft, xc='PBE')
    else:
        calc = GPAW(mode=PW(500), kpts=(5, 5, 6), xc=xc, txt=calcname + '.log')

    Li_gra.calc = calc  # Connect system and calculator

    sf = StrainFilter(Li_gra, mask=[1, 1, 1, 0, 0, 0])
    opt = BFGS(sf, trajectory=calcname + '.traj')
    opt.run(fmax=0.01)
