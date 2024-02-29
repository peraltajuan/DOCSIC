#!/usr/bin/env python
'''
DOC-SIC calculation
use: python3.11 example_1.py methane.xyz
'''

if __name__ == "__main__":

    from docsic import USIC, dump_scf_summary, lowdin_pop
    from pyscf.gto import Mole 
    import sys
    
    molecule = sys.argv[1]
    spin = 0
    charge = 0
    functional = 'slater,vwn5'


    mol = Mole()
    mol.fromfile(filename=molecule+'.xyz',format='xyz')
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
    print('Functional ',functional)
    print('SIC        ',mf.sic)
    print('Basis      ',mol.basis)
    print('charge ',charge)
    print('spin   ',spin)
    mf.kernel( )
    dump_scf_summary(mf)

   
    dm = mf.make_rdm1()
    lowdin_pop(mol,dm, mol.intor('int1e_ovlp') )




    










