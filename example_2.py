#!/usr/bin/env python
'''
DOC-SIC calculation
use: python3.11 example_1.py methane.xyz
'''

if __name__ == "__main__":

    from docsic import USIC, lowdin_pop
    from pyscf.gto import Mole 
    from pyscf import dft, scf
    
    spin = 0
    charge = 0
    functional = 'pbe,pbe'


    mol = Mole()
    mol.atom = '''
    O
    H  1  1.2
    H  1  1.2  2 105
    '''
    mol.basis = 'aug-cc-pvdz' 
    mol.cart=False
    mol.spin=spin
    mol.charge=charge
    mol.build()
#   DFT calculation
    mf = dft.UKS(mol)
    mf.xc = functional
    mf.verbose = 4
    print('Functional ',functional)
    print('Basis      ',mol.basis)
    print('charge ',charge)
    print('spin   ',spin)
    mf.kernel()
    dm = mf.make_rdm1()
    lowdin_pop(mol,dm, mol.intor('int1e_ovlp') )



#   SIC calculation with MOM
    mo0 = mf.mo_coeff
    occ = mf.mo_occ
    dm_u = mf.make_rdm1(mo0, occ)

    mom = USIC(mol)
    mom.sic='doc-sic'
    mom.verbose = 4
    mom.scale_sic = 1.00
    mom = scf.addons.mom_occ(mom, mo0, occ)
    print('START MOM ',functional)
    print('SIC        ',mom.sic)
    mom.scf(dm_u)
    dm = mom.make_rdm1()
    lowdin_pop(mol,dm, mol.intor('int1e_ovlp') )
#   regular SCF now 
    print('START regular SCF ',functional)
    print('SIC        ',mom.sic)
    mom.kernel( dm0=mom.make_rdm1() )



    dm = mom.make_rdm1()
    lowdin_pop(mol,dm, mol.intor('int1e_ovlp') )







    










