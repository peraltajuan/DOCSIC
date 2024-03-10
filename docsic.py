from pyscf.lib import logger
from pyscf import lib
import scipy
import torch
from scipy.linalg import qr
from scipy.linalg import inv, pinv
from torch.autograd import Function
import numpy as np
from  pyscf.scf import  hf, uhf, chkfile
from  pyscf.dft import rks, uks
from  pyscf import  dft,  __config__
from scipy.linalg import eigh

#print(np.__version__)
#print(scipy.__version__)
#import pyscf
#print(pyscf.__version__)
#print(torch.__version__)


def printM(a):
    for row in a:
        for col in row:
            print("{:9.6f}".format(col), end=" ")
        print("")

def printL(a):
    for i,r in enumerate(a):
          print(f"{r:9.6f}", end=" ")
          if ((i+1) % 8) == 0: print("")
    print("")

def printLi(s,a):
    print(s)
    for i,r in enumerate(a):
          print(f"{r:3d}", end=" ")
          if ((i+1) % 20) == 0: print("")
    print("")





def lowdin_pop(mol, dm, S ):
    '''Lowdin population analysis

    Prints Lowdin Population for charge and spin    

    '''
    print('')
    print(' Lodwin Population')
    print(' Center         charge       spin ')
    s12 = scipy.linalg.sqrtm(S).real
    SPSa = s12 @ dm[0] @s12
    SPSb = s12 @ dm[1] @s12
    popa= np.diag(SPSa)
    popb= np.diag(SPSb)
    chga = np.zeros(mol.natm)
    chgb = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        chga[s[0]] += popa[i]
        chgb[s[0]] += popb[i]
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        achg = mol.atom_charge(ia)
        CH = -chga[ia] -chgb[ia] + achg 
        SP = chga[ia] - chgb[ia]
        print(f' {ia:3d}{symb:s}       {CH:10.5f} {SP:10.5f} ')



class MatrixSQRT(Function):
    """Square root of a positive definite matrix to use with torch autograd.
    NOTE: matrix square root will break if there are zero eigenvalues
          This version uses the analytical expression instead of the Sylvester solver
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float64)
        sqrtma = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtma)
        return sqrtma

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtma, = ctx.saved_tensors
            sqrtma = sqrtma.data.cpu().numpy().astype(np.float64)
            gm = grad_output.data.cpu().numpy().astype(np.float64)

#           https://doi.org/10.1063/1.3624397 Cheng and Gauss, J. Chem. Phys. 135, 084114 (2011)
            Z, Y = eigh(sqrtma)
            YTAY =  Y.T @ gm @ Y
            Zdd = np.add.outer(Z, Z) 
            YTBY = YTAY/Zdd
            grad_sqrtm = Y @ YTBY @ Y.T
            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSQRT.apply


def dump_scf_summary(mf, verbose=logger.DEBUG):
    '''
    **** This is an in-house modified version to work with doc-sic ****
    '''
    if not mf.scf_summary:
        return

    log = logger.new_logger(mf, verbose)
    summary = mf.scf_summary
    def write(fmt, key):
        if key in summary:
            log.info(fmt, summary[key])
    log.info('**** SCF Summaries ****')
    log.info('Total Energy =                    %24.15f', mf.e_tot)
    write('Nuclear Repulsion Energy =        %24.15f', 'nuc')
    write('One-electron Energy =             %24.15f', 'e1')
    write('Two-electron Energy =             %24.15f', 'e2')
    write('Two-electron Coulomb Energy =     %24.15f', 'coul')
    write('DFT Exchange-Correlation Energy = %24.15f', 'exc')
    write('Total DFT Energy                = %24.15f', 'dft')
    write('SIC  Energy                     = %24.15f', 'esic')
    write('Empirical Dispersion Energy =     %24.15f', 'dispersion')
    write('PCM Polarization Energy =         %24.15f', 'epcm')
    write('EFP Energy =                      %24.15f', 'efp')
    if getattr(mf, 'entropy', None):
        log.info('(Electronic) Entropy              %24.15f', mf.entropy)
        log.info('(Electronic) Zero Point Energy    %24.15f', mf.e_zero)
        log.info('Free Energy =                     %24.15f', mf.e_free)




def docsic_rden_l(ks,P,nmo,S,spin=None,what='density'):
    '''
    Do the localization for one type of spin
    '''
    S12 = scipy.linalg.sqrtm(S).real  # this should be done only once 
    Sm12 = pinv(S12,rtol=1e-8)  # this should be done only once 
    Pb = S12@P@S12
    if 'sic_iter' not in dir(rks): rks.sic_iter = 1
# THIS  needs a revision
    reuse = False


    if  'reuse_piv' in dir(ks): 
         if ks.reuse_piv and rks.sic_iter > 1:
             print('Reusing the pivoting array for doc-sic spin '+ spin)
             reuse = True

    if not reuse:
#       print('Determining the pivoting array for doc-sic spin=',spin)
       Q,  R, piv = scipy.linalg.qr(Pb , pivoting=True);
       if spin =='a' : rks.piv_a =  np.sort(piv[0:nmo]  )  
       if spin =='b' : rks.piv_b =  np.sort(piv[0:nmo]  ) 
       if spin ==None: rks.piv   =  np.sort(piv[0:nmo]  )   
    if spin =='a' : piv = rks.piv_a 
    if spin =='b' : piv = rks.piv_b
    if spin ==None: piv = rks.piv

    Xt = Pb[:,piv]
    St = (Xt.T @  Xt).real
    St12 = scipy.linalg.sqrtm(St).real 
    Stm12 = inv(St12)
    X  = Sm12 @ Xt @ Stm12
    P_loc = []
    for i in range(len(piv)): 
        P_loc.append(np.outer( X.T[i] , X.T[i] ) )
    if what =='density':
        return np.array(P_loc)
    elif what =='orbital':
        return np.array(X.T)
    else:
        return np.array(P_loc)



def get_vsic(rks,mol,vhj,vxc,P,spin=None):
    ov = mol.intor('int1e_ovlp')
    ov12 = scipy.linalg.sqrtm(ov).real  # this should be done only once 
    ovm12 = pinv(ov12,rtol=1e-8)  # this should be done only once 

    ht = torch.tensor( vhj + vxc  , dtype=torch.float64, requires_grad=False)
    P1 = torch.tensor( P  , dtype=torch.float64, requires_grad=True)
    S12 = torch.tensor( ov12  , dtype=torch.float64, requires_grad=False)
    Sm12 = torch.tensor( ovm12  , dtype=torch.float64, requires_grad=False)
    
    P0 = torch.mm(P1 , S12) 
    Pt = torch.mm(S12, P0)

    if spin=='a' : piv = rks.piv_a
    if spin=='b' : piv = rks.piv_b
    if spin==None: piv = rks.piv
    Xt = Pt[:,piv]
    
    Xtt= torch.transpose(Xt,0,1)
    St  = torch.mm( Xtt  , Xt) 
    St12 = sqrtm(St).real
    Stm12 = torch.linalg.inv(St12)
    z0 = torch.mm(Xt ,  Stm12) 
    z = torch.mm(Sm12 ,  z0) 
    XT  = torch.transpose( z , 0,1 )

    P_loc = torch.tensor(() ,dtype=torch.float64, requires_grad=True ) # torch.Tensor()
    
    if spin=='a' : pivs ='alpha'
    if spin=='b' : pivs =' beta'
    print('Pivoting vector '+pivs , piv)
    for i in range( len(XT) ): 
        Pi = torch.outer( XT[i] , XT[i] )
        P_loc = torch.cat((P_loc, Pi.unsqueeze(0))  ,  dim=0 )  

    ESICt =   torch.sum( ht * P_loc ) 
    ESICt.backward()
    dEdP = P1.grad

    dedp = dEdP.detach().numpy()
    return -0.5*(dedp + dedp.T.conj()).real 












def get_veff_usic(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC +SIC functional for UKS.  See pyscf/dft/rks.py
    **** This is an in-house modified version to work with doc-sic ****
    :func:`get_veff` fore more details.
    [adapted from get_veff]
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    if not isinstance(dm, np.ndarray):
        dm = np.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        dm = numpy.asarray((dm*.5,dm*.5))
    ks.initialize_grids(mol, dm)



    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        exc_save = exc
        esic = 0.0 
        xcsic = 0.0 
        gradlam = 0.0 
        n = mol.tot_electrons()
      
        if ks.sic == 'doc-sic':
             rks.sic_iter = 1
             calc_vsic = True
             nocc_a =  mol.nelec[0]  
             nocc_b =  mol.nelec[1]
             S = mol.intor('int1e_ovlp')
             if not('piv_a' in dir(rks)):
               rks.sic_iter = 1
             else: 
               rks.sic_iter += 1
#
             if rks.sic_iter == 1: calc_vsic = True
  
             if 'calc_vsic' in dir(rks): calc_vsic = rks.calc_vsic
             scale_j  = 1.00
             scale_xc = 1.00
             if calc_vsic:

                P_loc_a = docsic_rden_l(ks,dm[0],nocc_a ,S,spin='a')  

                if nocc_b == 1: 
                   P_loc_b = np.asarray([dm[1]])
                   rks.piv_b  = [0]
                else:
                   P_loc_b = docsic_rden_l(ks,dm[1],nocc_b ,S,spin='b')  
 
                nsic, excsic1, v1sic = ni.nr_uks(mol, ks.grids, ks.xc, [P_loc_a,0.0*P_loc_a], max_memory=max_memory)
                nsic_a = nsic[0]
                excsic_a = excsic1*scale_xc
                v1sic_a = v1sic[0]*scale_xc
                print('')
                print('Evaluating self-XC for SIC alpha')
                printL(nsic_a)
                printL(excsic_a)
                excsic_a = sum(excsic_a)
                nsic, excsic1, v1sic = ni.nr_uks(mol, ks.grids, ks.xc, [P_loc_b,0.0*P_loc_b], max_memory=max_memory)
                nsic_b = nsic[0]
                excsic_b = excsic1*scale_xc
                v1sic_b = v1sic[0]*scale_xc
                print('Evaluating self-XC for SIC beta')
                printL(nsic_b)
                printL(excsic_b)
                excsic_b = sum(excsic_b)

    

        if ks.nlc or ni.libxc.is_nlc(ks.xc):
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm[0]+dm[1],
                                          max_memory=max_memory)
            exc += enlc
            vxc += vnlc
            logger.debug(ks, 'nelec with nlc grids = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)


    if not ni.libxc.is_hybrid_xc(ks.xc):
        vk = None
        if False:
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj = ks.get_j(mol, ddm[0]+ddm[1], hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm[0]+dm[1], hermi)
            if (ks.sic == 'doc-sic' and calc_vsic):
               v1j_a = scale_j*ks.get_j(mol, P_loc_a, hermi)
               v1j_b = scale_j*ks.get_j(mol, P_loc_b, hermi)

        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if omega != 0:
                vklr = ks.get_k(mol, ddm, hermi, omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vj = vj[0] + vj[1] + vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vj = vj[0] + vj[1]
            vk *= hyb
            if omega != 0:
                vklr = ks.get_k(mol, dm, hermi, omega)
                vklr *= (alpha - hyb)
                vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -=(np.einsum('ij,ji', dm[0], vk[0]).real +
                   np.einsum('ij,ji', dm[1], vk[1]).real) * .5
    if ground_state:
        ecoul = np.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
    else:
        ecoul = None

    if (ks.sic == 'doc-sic' and calc_vsic):
           e1_a=[]
           e1_b=[]
           for i in range(len(P_loc_a)):
              e1_a= np.append(e1_a, np.einsum('ij,ji', P_loc_a[i], v1j_a[i]).real  * .5  )
           for i in range(len(P_loc_b)):
              e1_b= np.append(e1_b, np.einsum('ij,ji', P_loc_b[i], v1j_b[i]).real  * .5  )
    
           ecoul1_a = np.einsum('kij,kji', P_loc_a, v1j_a).real  * .5
           ecoul1_b = np.einsum('kij,kji', P_loc_b, v1j_b).real  * .5
           print('Evaluating self-Hartree for SIC alpha')
           printL(e1_a) 
           print('Evaluating self-Hartree for SIC beta')
           printL(e1_b) 
           print('')
           scale_sic = ks.scale_sic # 0.2 # /(1.+1./rks.sic_iter**2)
           esic = (-ecoul1_a -ecoul1_b -excsic_a -excsic_b)*scale_sic




    if (ks.sic == 'doc-sic' and calc_vsic):
       vsic_a = get_vsic(rks,mol,v1j_a,v1sic_a,dm[0],spin='a')*scale_sic
       vsic_b = get_vsic(rks,mol,v1j_b,v1sic_b,dm[1],spin='b')*scale_sic
       veff = vxc + np.asarray((vsic_a,vsic_b))  # the plus is correct
       rks.esic = esic
    else:
       veff = vxc


    if not calc_vsic:
       veff = vxc + ks.vsic
       esic = rks.esic
       print('Reusing VSIC')
    else:
       if ks.sic == 'doc-sic':
           ks.vxc  = vxc
           ks.vsic = np.asarray((vsic_a,vsic_b))  # test: hold here vsic 
    ks.exc = exc
    ks.ecoul =ecoul
    vhf = lib.tag_array(veff, ecoul=ecoul, exc=exc, vj=vj, vk=vk,esic=esic)

    return vhf



def make_square(P):
   if P.ndim == 1:  
     N = int(np.sqrt(len(P)))
     R =  np.reshape(P, (N,N) )
   else:
     R = P 
   return R 

def make_linear(P):
   if P.ndim == 2:  
      N = len(P)
      R = np.reshape(P, N*N ) 
   else:
      R = P 
   return R



def energy_sic(dm1, ks,s,dm2):
    '''Electronic SIC part of RKS energy.
    Returns:
         SIC electronic 
    '''
    dm1 = make_square(dm1)
    dm2 = make_square(dm2)
    if s==0: dm =[dm1,dm2]
    if s==1: dm =[dm2,dm1]
    v = get_veff_usic(ks, ks.mol, dm )
    esic = v.esic.real
    return esic


def d_energy_sic(dm1, ks,s,dm2):
    '''
    Wrapper to  get_veff_usic to get only the SIC KS matrix
    '''
    dm1 = make_square(dm1)
    dm2 = make_square(dm2)
    if s==0: dm =[dm1,dm2]
    if s==1: dm =[dm2,dm1]
    get_veff_usic(ks, ks.mol, dm )
    return ks.vsic [s]


def energy_sic_piv(dm, ks,piv):
    '''Electronic SIC part of RKS energy.
    Returns:
         SIC electronic energy for a set of pivoting vectors
    '''
    ks.reuse_piv = True
    rks.piv_a  = piv[0]
    rks.piv_b  = piv[1]


    v = get_veff_usic(ks, ks.mol, dm )
    esic = v.esic.real
    ecoul = ks.ecoul.real
    exc = ks.exc.real
    return   esic  #exc+ecoul+esic














def energy_uelec(ks, dm=None, h1e=None, vhf=None):
    '''
    **** This is an in-house modified version to work with doc-sic ****
    energy_elec adapted for the UKS case with SIC
    '''
    if(ks.sic=='doc-sic'): 
       if dm is None: dm = ks.make_rdm1()
       if h1e is None: h1e = ks.get_hcore()
       if vhf is None or getattr(vhf, 'ecoul', None) is None:
           vhf = get_veff_usic(ks, ks.mol, dm)
       dm = dm[0] + dm[1]
       e1 = np.einsum('ij,ji->', h1e, dm).real
       ecoul = ks.ecoul.real
       exc = ks.exc.real
       esic = vhf.esic.real
       e2 = ecoul + exc
       nuc = ks.energy_nuc()
       ks.scf_summary['e1'] = e1
       ks.scf_summary['e2'] = e2
       ks.scf_summary['coul'] = ecoul
       ks.scf_summary['exc'] = exc
       ks.scf_summary['esic'] = esic
       ks.scf_summary['e_tot'] = e1+e2+esic+nuc
       ks.scf_summary['e_dft'] = e1+e2+nuc
       vhf.e_tot = e1+e2+esic+nuc
       vhf.e_dft = e1+e2
       verbose=4
       log = logger.new_logger(ks, verbose)
       summary = ks.scf_summary
       def write(fmt, key):
           if key in summary: log.info(fmt, summary[key])
       write('Total Energy =                    %24.15f', 'e_tot')
       write('Nuclear Repulsion Energy =        %24.15f', 'nuc')
       write('One-electron Energy =             %24.15f', 'e1')
       write('Two-electron Energy =             %24.15f', 'e2')
       write('Two-electron Coulomb Energy =     %24.15f', 'coul')
       write('DFT Exchange-Correlation Energy = %24.15f', 'exc')
       write('Total DFT Energy (no SIC)       = %24.15f', 'e_dft')
       write('SIC  Energy                     = %24.15f', 'esic')
       return  e1+e2+esic, e2   
    else:  # this is not working....
      if dm is None: dm = ks.make_rdm1()
      if h1e is None: h1e = ks.get_hcore()
      if vhf is None or getattr(vhf, 'ecoul', None) is None:
          vhf = ks.get_veff(ks.mol, dm)
          print(dir(vhf))
      if not (isinstance(dm, np.ndarray) and dm.ndim == 2): 
          dm = dm[0] + dm[1]
      return rks.energy_elec(ks, dm, h1e, vhf)



def kernel_sic(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    ''' 
    **** This is an in-house modified version to work with doc-sic ****
    kernel: the SCF driver.

    Args:
        mf : an instance of SCF class
            mf object holds all parameters to control SCF.  One can modify its
            member functions to change the behavior of SCF.  The member
            functions which are called in kernel are

            | mf.get_init_guess
            | mf.get_hcore
            | mf.get_ovlp
            | mf.get_veff
            | mf.get_fock
            | mf.get_grad
            | mf.eig
            | mf.get_occ
            | mf.make_rdm1
            | mf.energy_tot
            | mf.dump_chk

    Kwargs:
        conv_tol : float
            converge threshold.
        conv_tol_grad : float
            gradients converge threshold.
        dump_chk : bool
            Whether to save SCF intermediate results in the checkpoint file
        dm0 : ndarray
            Initial guess density matrix.  If not given (the default), the kernel
            takes the density matrix generated by ``mf.get_init_guess``.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.

    Returns:
        A list :   scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

        scf_conv : bool
            True means SCF converged
        e_tot : float
            Hartree-Fock energy of last iteration
        mo_energy : 1D float array
            Orbital energies.  Depending the eig function provided by mf
            object, the orbital energies may NOT be sorted.
        mo_coeff : 2D array
            Orbital coefficients.
        mo_occ : 1D array
            Orbital occupancies.  The occupancies may NOT be sorted from large
            to small.

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    >>> conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=np.eye(mol.nao_nr()))
    >>> print('conv = %s, E(HF) = %.12f' % (conv, e))
    conv = True, E(HF) = -1.081170784378
    '''
    if 'init_dm' in kwargs:
        raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.11.
Keyword argument "init_dm" is replaced by "dm0"''')
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %s', cond)
    if np.max(cond)*1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', np.max(cond))

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    TIGHT_GRAD_CONV_TOL = getattr(__config__, 'scf_hf_kernel_tight_grad_conv_tol', True)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot
        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf_last=vhf
        vhf = mf.get_veff(mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf)
        d_fock_0 = np.linalg.norm(vhf[0] - vhf_last[0] ) 
        d_fock_1 = np.linalg.norm(vhf[1] - vhf_last[1] ) 
        e_tot = mf.energy_tot(dm, h1e, vhf)

        # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        diis_e_0 = np.linalg.norm(fock[0]@dm[0]@s1e - s1e@dm[0]@fock[0] ) 
        diis_e_1 = np.linalg.norm(fock[1]@dm[1]@s1e - s1e@dm[1]@fock[1] ) 
        CErr = np.sqrt(diis_e_0**2 + diis_e_1**2)
        print(f'Commutator Error:{CErr:10.4e} ')
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))

        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / np.sqrt(norm_gorb.size)
        norm_ddm = np.linalg.norm(dm-dm_last)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / np.sqrt(norm_gorb.size)
        norm_ddm = np.linalg.norm(dm-dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    logger.timer(mf, 'scf_cycle', *cput0)



# calculate and print lambda matrices (from localization equations) 
 
    nocca =  mol.nelec[0]  
    noccb =  mol.nelec[1]  
    fock_a = h1e  +  mf.vxc[0] #+ mf.vsic[0] 
    fock_b = h1e  +  mf.vxc[1] #+ mf.vsic[1] 
    print('','*'*75) 
    print('','*'*5,' OCCUPIED ORBITALS ','*'*49) 
    print('','*'*75) 
    loc_mat(mf,dm[0],nocca,s1e,fock_a,spin='a',virtual=False) 
    loc_mat(mf,dm[1],noccb,s1e,fock_b,spin='b',virtual=False) 
 
    print('','*'*75) 
    print('','*'*5,' VIRTUAL  ORBITALS ','*'*49) 
    print('','*'*75) 
    loc_mat(mf,dm[0],nocca,s1e,fock_a,spin='a',virtual=True) 
    loc_mat(mf,dm[1],noccb,s1e,fock_b,spin='b',virtual=True) 




    # A post-processing hook before return
    mf.post_kernel(locals())
    # why???
    mf.mo_coeff = mo_coeff
    mf.mo_occ = mo_occ
    mf.scf_conv =scf_conv
    mf.e_tot = e_tot
    mf.mo_energy = mo_energy

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ




def make_loc(Fo,fock,Z):
    Nbas = len(Z[:,0])
    NFO =  len(Z[0,:])
    LocMat_SIC = np.zeros((NFO,NFO) )
    LocMat_DFT = np.zeros((NFO,NFO) )


    for a in range(0,NFO):
      for b in range(0,NFO):
        LocMat_DFT[b,a] =  (Z[:,a].T.conj() @ fock @ Z[:,b]).real 


    for a in range(0,NFO):
      for b in range(0,NFO):
        LocMat_SIC[b,a] =  (Z[:,a].T.conj() @ Fo[b,:,:] @ Z[:,b]).real  
#    print("Lambda SIC matrices (Ha):")
#    printM(LocMat_SIC) 

    ASim_loc_mat = LocMat_SIC - LocMat_SIC.T.conj()
    As = np.linalg.norm(ASim_loc_mat)/np.sqrt(ASim_loc_mat.size)
    print(f" RMS asymm Lambda matrix = {As:g}" )




    print("\n Eigenvalues of the symmetrized Lambda matrix (Ha):")
    OrbEn =  eigh( (LocMat_SIC +LocMat_SIC.T.conj())/2.0E0  + LocMat_DFT    )[0]
    printL(OrbEn)

    return






def loc_mat(ks,P,nmo,S,fock,spin=None,virtual=False):
    '''
    Do the localization for one type of spin
    '''
    S12 = scipy.linalg.sqrtm(S).real  # this should be done only once 
    Sm12 = pinv(S12,rtol=1e-8)  # this should be done only once 
    Pb = S12@P@S12
    print('\n', '*'*75)
    if spin =='a' : print(' ' ,'*** SPIN ALPHA '*5)    
    if spin =='b' : print(' ','*** SPIN BETA  '*5)  
    print('', '*'*75)


    if spin=='a' : pivs ='alpha'
    if spin=='b' : pivs =' beta'

    reuse = False
    if  'reuse_piv' in dir(ks): 
         if ks.reuse_piv and rks.sic_iter > 1:
             print('Reusing the pivoting array for doc-sic spin '+ pivs)
             reuse = True

    if not virtual: 
         piv_array = np.arange(0,nmo)
         W = np.zeros_like(S)
         s = 1.0
    if virtual: 
         piv_array = np.arange(0,len(S)-nmo )
         W = np.eye(len(S))
         s = -1.0
    if not reuse:
       M = W + s*Pb 
       print('\n Determining the pivoting array for doc-sic spin '+pivs)
       Q,  R, piv = scipy.linalg.qr( M , pivoting=True);
       if spin =='a' : rks.piv_a =  np.sort(piv[piv_array]  )  
       if spin =='b' : rks.piv_b =  np.sort(piv[piv_array]  ) 
       if spin ==None: rks.piv   =  np.sort(piv[piv_array]  )   
    if spin =='a' : piv = rks.piv_a 
    if spin =='b' : piv = rks.piv_b
    if spin ==None: piv = rks.piv



    printLi(' Using columns: ', piv)


    RR = np.abs(  np.diagonal(R).round(decimals=3)  )
    Rank = sum(x > 0.0 for x in RR)
    print(f'\n Rank of the Density Matrix = {Rank:g}\n')

    Xt = M[:,piv]
    St = Xt.T @  Xt
    St12 = scipy.linalg.sqrtm(St).real
    Stm12 = inv(St12)
    X  = Sm12 @ Xt @ Stm12

    P_loc = [] 
    for i in range(len(piv)): 
        P_loc.append(np.outer( X.T[i] , X.T[i].conj() ).real )

    P_loc = np.asarray(P_loc)
    ni = ks._numint
    mol=ks.mol
    max_memory = ks.max_memory - lib.current_memory()[0]
    nsic, excsic1, v1sic = ni.nr_uks(mol, ks.grids, ks.xc, [P_loc,0.0*P_loc], max_memory=max_memory)
    v1j = ks.get_j(mol, P_loc, hermi=1)

    
    make_loc(-v1sic[0]-v1j,fock,X)

    return



def get_vsic(rks,mol,vhj,vxc,P,spin=None):
    ov = mol.intor('int1e_ovlp')
    ov12 = scipy.linalg.sqrtm(ov).real  # this should be done only once 
    ovm12 = scipy.linalg.pinv(ov12,rtol=1e-8)  # this should be done only once 

    ht = torch.tensor( vhj + vxc  , dtype=torch.float64, requires_grad=False)
    P1 = torch.tensor( P  , dtype=torch.float64, requires_grad=True)
    S12 = torch.tensor( ov12  , dtype=torch.float64, requires_grad=False)
    Sm12 = torch.tensor( ovm12  , dtype=torch.float64, requires_grad=False)
    
    P0 = torch.mm(P1 , S12) 
    Pt = torch.mm(S12, P0)

    if spin=='a' : piv = rks.piv_a
    if spin=='b' : piv = rks.piv_b
    if spin==None: piv = rks.piv
    Xt = Pt[:,piv]
    
    Xtt= torch.transpose(Xt,0,1)
    St  = torch.mm( Xtt  , Xt) 
    St12 = sqrtm(St).real
    Stm12 = torch.linalg.inv(St12)
    z0 = torch.mm(Xt ,  Stm12) 
    z = torch.mm(Sm12 ,  z0) 
    XT  = torch.transpose( z , 0,1 )

    P_loc = torch.tensor(() ,dtype=torch.float64, requires_grad=True ) # torch.Tensor()

    if spin=='a' : pivs ='alpha'
    if spin=='b' : pivs =' beta'
    printLi('Pivoting vector: '+pivs,piv)
    for i in range( len(XT) ): 
        Pi = torch.outer( XT[i] , XT[i] )
        P_loc = torch.cat((P_loc, Pi.unsqueeze(0))  ,  dim=0 )  

    ESICt =   torch.sum( ht * P_loc ) 
    ESICt.backward()
    dEdP = P1.grad

    dedp = dEdP.detach().numpy()
    return -0.5*(dedp + dedp.T.conj()).real 








class USIC(rks.KohnShamDFT, uhf.UHF):
    '''
    Unrestricted Kohn-Sham with DOC-SIC
    See pyscf/dft/rks.py RKS class for document of the attributes'''
    def __init__(self, mol, xc='LDA,VWN'):
        self.sic=''
        uhf.UHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        uhf.UHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def initialize_grids(self, mol=None, dm=None):
        ground_state = (isinstance(dm, np.ndarray)
                        and dm.ndim == 3 and dm.shape[0] == 2)
        if ground_state:
            super().initialize_grids(mol, dm[0]+dm[1])
        else:
            super().initialize_grids(mol)
        return self
    get_veff = get_veff_usic
    get_vsap = uks.get_vsap
    energy_elec = energy_uelec
    kernel = kernel_sic
    get_vsap = uks.get_vsap
    energy_elec = energy_uelec
       
    init_guess_by_vsap = rks.init_guess_by_vsap

    def nuc_grad_method(self):
        from pyscf.grad import uks
        return uks.Gradients(self)

    def to_hf(self):
        '''Convert to UHF object.'''
        return self._transfer_attrs_(self.mol.UHF())



