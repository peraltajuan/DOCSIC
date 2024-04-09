# DOCSIC
## Density as Orbitals by Columns Self-Interaction Correction 

This code implements self-consistently the Perdew-Zunger self-interaction correction to approximate density functional calculations using orbitals taken from columns of the density matrix. 
The resulting localized orbitals are identical to the SCDM-L method reported in 

Fuemmeler, E. G.; Damle, A.; DiStasio, R. A. J. Selected Columns of the Density
Matrix in an Atomic Orbital Basis I: An Intrinsic and Non-iterative Orbital Localization
Scheme for the Occupied Space. J. Chem. Theory Comput. 2023, 19, 8572–8586.

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


 
  
