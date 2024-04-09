# DOCSIC
## Density as Orbitals by Columns Self-Interaction Correction 

This code implements self-consistently the Perdew-Zunger self-interaction correction to approximate density functional calculations using orbitals taken from columns of the density matrix. Give a set of indices chosen from a pivoted QR decomposition of the density matrix, the energy is minimized using standard self-consistent iteration with an effective self-interaction multiplicative term added to the generalized Kohn-Sham Hamiltonian. 

The resulting localized orbitals are identical to those obtained with the SCDM-L method reported in 

Selected Columns of the Density Matrix in an Atomic Orbital Basis I: An Intrinsic and Non-iterative Orbital Localization
Scheme for the Occupied Space, E. G. Fuemmeler, A. Damle,  R. A. J. DiStasio,  J. Chem. Theory Comput. 2023, 19, 8572–8586.

Details will be published as

A Mean-field, Orbital-by-orbital Method for Self-interaction Correction, 
J. E. Peralta, V. Barone, J. I. Melo, D. R. Alcoba, L.
Lain, A. Torre, G. E. Massaccesi, and O. B. Oña.

##

Dependencies (developed and tested, but other versions might work):

- Python 3.11
- Numpy 1.25.0
- Scipy 1.11.2
- PySCF 2.5.0
- Torch 2.2.1


##

How to run the examples:

Methane:
```
python3.11 example_1.py methane
```
Water (with maximum overlap method):
```
python3.11 example_2.py
```
Ethylene+:
```
python3.11 example_3.py
```
Generate cube files:
```
python3.11 example_3.py isobutene
```


 
  
