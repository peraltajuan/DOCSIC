#!/usr/bin/env python
'''
Test the SIC matrix elements using finite differences
'''

if __name__ == "__main__":

    from docsic import USIC, dump_scf_summary, energy_sic, printM
    from docsic import make_linear, make_square, d_energy_sic
    import docsic
    from pyscf.gto import Mole
    from pyscf import dft
    import numpy as np


    def locateLargest(matrix,N):
      list_largest = []
      copy_m =  matrix.copy()
      for l in range(N):
        largest_num = 0.0
        for row_idx, row in enumerate(copy_m):
            for col_idx, num in enumerate(row):
                if np.absolute(num) > largest_num:
                    largest_num = num
                    irow = row_idx
                    icol = col_idx
        copy_m[irow,icol] = 0.0
        copy_m[icol,irow] = 0.0
        list_largest.append((irow,icol))

      return list_largest





    mol = Mole()
    mol.atom = '''O 0 0 0; H  0 1.1 0; H 0. 0. 1'''


    mol.basis = '6-31G*'
    mol.charge=0
    mol.spin = 0
    mol.build()
    S = mol.intor('int1e_ovlp')


    mf = dft.UKS(mol)
    mf.xc = 'lda'
    mf.verbose = 4
    mf.kernel()
    dump_scf_summary(mf,10)
    dm = mf.make_rdm1()


    mf = USIC(mol)
    mf.xc = 'scan'
    mf.sic='doc-sic'
    mf.scale_sic=1.0
    mf.verbose = 4
    mf.max_cycle = 1
    mf.kernel(dm0=dm)

    dump_scf_summary(mf,10)

    SPIN = 1


    dm1 = mf.make_rdm1()[SPIN]
    dm2 = mf.make_rdm1()[SPIN-1]
    Der_an = d_energy_sic(dm1,mf,SPIN,dm2)
    Der_an = make_square(Der_an)
#    Der_s = 0.5*(Der_an + Der_an.T)
    Der_s = Der_an




    dm1 = mf.make_rdm1()[SPIN]
    dm2 = mf.make_rdm1()[SPIN-1]

   
    Nao = len(dm1)
    nelem = 12

    Lmax = locateLargest(np.asarray(dm1),nelem)
    print(Lmax)
    
    Der_sym = np.zeros_like(dm1)

    epsilon = 1.e-6

    Dev = 0.0
    for nn,ij in enumerate(Lmax):
        i = ij[0]
        j = ij[1]
        DM = dm1
        print('***********************************************')
        print(f'Numerical derivatives, symmetric {i:3d} {j:3d}')
        print(f'      Step {nn+1:3d} of {len(Lmax):3d}')
        print('***********************************************')
        DM[i,j] = dm1[i,j] + epsilon
        if i !=j : DM[j,i] = dm1[j,i] + epsilon
        Eplus,pa,pb = energy_sic(DM,mf,SPIN,dm2)
        DM[i,j] = dm1[i,j] - epsilon
        if i!=j : DM[j,i] = dm1[j,i] - epsilon
        Eminus,pa,pb = energy_sic(DM,mf,SPIN,dm2)
        Der_sym[i,j]=(Eplus-Eminus)/2/epsilon
        Der_sym[j,i]=(Eplus-Eminus)/2/epsilon

        print('Numer ',i,j, Der_sym[i,j])
        print('Analy ',i,j, Der_s[i,j])
        Dev += (Der_sym[i,j] - Der_s[i,j])**2
        Dev += (Der_sym[j,i] - Der_s[j,i])**2
    Dev = np.sqrt(Dev)
    print('*'*34)
    print(f'  The deviation is {Dev:12.7e} ')
    print('*'*34)



    










