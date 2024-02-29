#!/usr/bin/env python
'''
DOC-SIC calculation
use: python3.11 example_4.py filename.xyz
this example produces cube files with the localized orbitals
'''

if __name__ == "__main__":

    from docsic import USIC, dump_scf_summary, lowdin_pop,docsic_rden_l
    from pyscf.gto import Mole 
    from pyscf.tools import cubegen
    import sys
    
    filename = sys.argv[1]
    spin = 0
    charge = 0
    functional = 'slater,vwn5'

    mol = Mole()
    mol.fromfile(filename=filename+'.xyz',format='xyz')
    mol.basis = 'sto-3g' 
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


    Pa,Pb = mf.make_rdm1()
    P = (Pa+Pb)
    # total density
    cubegen.density(mol, filename+f'_alpha.cube', Pa ,nx=120, ny=120, nz=120)
    cubegen.density(mol, filename+f'_beta.cube', Pb,nx=120, ny=120, nz=120)




    S = mol.intor('int1e_ovlp') 
    nocc =  mol.nelec[0]
    C_loc_a = docsic_rden_l(mf,Pa,nocc,S,spin='a',what='orbital')
    for i,C in enumerate(C_loc_a):
         cubegen.orbital(mol, filename+f'_alpha_docsic_orb_{i+1}.cube', C,nx=120, ny=120, nz=120)

    nocc =  mol.nelec[1]
    C_loc_b = docsic_rden_l(mf,Pb,nocc,S,spin='b',what='orbital')
    for i,C in enumerate(C_loc_b):
         cubegen.orbital(mol, filename+f'_beta_docsic_orb_{i+1}.cube', C,nx=120, ny=120, nz=120)


# uncomment this code for densities
#    S = mol.intor('int1e_ovlp') 
#    nocc =  mol.nelec[0]
#    P_loc_a = docsic_rden_l(mf,Pa,nocc,S,spin='a')
#    for i,P in enumerate(P_loc_a):
#         cubegen.density(mol, filename+f'_alpha_docsic_den_{i+1}.cube', P,nx=120, ny=120, nz=120)
#
#    nocc =  mol.nelec[1]
#    P_loc_b = docsic_rden_l(mf,Pb,nocc,S,spin='b')
#    for i,P in enumerate(P_loc_b):
#         cubegen.density(mol, filename+f'_beta_docsic_den_{i+1}.cube', P,nx=120, ny=120, nz=120)


    










