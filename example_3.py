#!/usr/bin/env python
'''
DOC-SIC calculation
use: python3.11 example_1.py methane.xyz
'''

if __name__ == "__main__":

    from docsic import USIC, dump_scf_summary, lowdin_pop
    from pyscf.gto import Mole 
    import sys
    
    spin = 1
    charge = 1
    functional = 'scan'


    mol = Mole()
    mol.atom='''
    C   0.000000   0.000000   0.668163
    C  -0.000000  -0.000000  -0.668163
    H  -0.000000   0.927900   1.243034
    H   0.000000  -0.927900   1.243034
    H  -0.000000   0.927900  -1.243034
    H   0.000000  -0.927900  -1.243034
    '''
    mol.basis = '6-31G**' 
    mol.cart=False
    mol.spin=spin
    mol.charge=charge
    mol.build()


    mf = USIC(mol)
    mf.xc = functional
    mf.sic='doc-sic'
    mf.verbose = 4
    mf.scale_sic = 1.00
    mf.diis_start_cycle = 10
    mf.damp = 0.75
    mf.grids.level = 5
    print('Functional ',functional)
    print('SIC        ',mf.sic)
    print('Basis      ',mol.basis)
    print('charge ',charge)
    print('spin   ',spin)
    mf.kernel( )
    dump_scf_summary(mf)

   
    dm = mf.make_rdm1()
    lowdin_pop(mol,dm, mol.intor('int1e_ovlp') )




    










