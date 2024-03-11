#!/usr/bin/env python
'''
This is an old script but it should work
'''

if __name__ == "__main__":

    from docsic import USIC, dump_scf_summary, docsic_rden_l, lowdin_pop
    import docsic
    from pyscf.gto import Mole 
    from pyscf import dft
    from pyscf import scf
    import numpy as np
    import sys
    from pyscf.tools import cubegen
    from pyscf.gto import mole
    
    molecule = 'bz.xyz'
    charge = 0
    spin= 0
    functional = 'slater,vwn5'

    mol = Mole()
    mol.basis = 'sto-3g' 
    mol.cart=False
    mol.spin=spin
    mol.charge=charge
    mol.fromfile(filename=molecule,format='xyz')
    mol.build()
    S = mol.intor('int1e_ovlp')


    mf = scf.UHF(mol)

    mf.kernel(  )
    dump_scf_summary(mf,14)
    P = mf.make_rdm1()
    lowdin_pop(mol, P, S)

###############################################################

    print('********* START PBE ******************')
    mf = USIC(mol)
    mf.sic='doc-sic'
    mf.verbose = 4
    mf.grids.level = 5
    mf.scale_sic = 1.00
    mf.reuse_piv = False
    mf.xc = 'pbe,pbe'
    mf.max_cycle = 500
    mf.diis_start_cycle = 40
    mf.damp = 0.85
    mf.kernel( dm0=P )

    Pa,Pb = mf.make_rdm1()
    P = (Pa + Pb)
    lowdin_pop(mol, P, S)


    print('********* END PBE ******************')
    print('********* START SCAN ******************')

    mf.xc = 'scan'

    mf.kernel( dm0=(Pa,Pb) )
    Pa,Pb = mf.make_rdm1()
    P = (Pa + Pb)
    lowdin_pop(mol, P, S)



    print('********* END SCAN ******************')
    print('********* START LDA ******************')
    mf.xc = 'slater,vwn5'
    mf.diis_start_cycle = 50
    mf.max_cycle = 500
    mf.damp = 0.75
    mf.kernel( dm0=(Pa,Pb) )

    Pa,Pb = mf.make_rdm1()
    P = (Pa + Pb)
    lowdin_pop(mol, P, S)

    print('********* END LDA ******************')
    print('********* END ALL ******************')

    










