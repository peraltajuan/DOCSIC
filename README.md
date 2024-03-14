# DOCSIC
## Density as Orbitals by Columns Self-Interaction Correction 

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
##

**On the HPCC:**

use

```
module load GCCcore/12.3.0
```

```
module load Python/3.11.3
```

```
module load CUDA/12.3.0
```

##
To do:

- Sqrtm + inv for torch can be replaced by one inverse-sqrtm instead. It saves one step.
- Add option to fix the SIC KS matrices in the AO basis (save time re-computing them).    
  
