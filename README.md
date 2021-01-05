# APAC

Automated Parcellation tool for human Auditory Cortex (APAC)

APAC is a lightweight module to parcellate human primary auditory cortex comparable to the core in non-human primates using structural MRI (myelin-sensitive index and curvature).

## Requirements

- Required files
```
*.?.sphere.*.surf.gii
*.?.SmoothedMyelinMap_BC.*.func.gii
*.?.curvature.*.shape.gii
```
The current version was developed using the data obtained from [HCP pipelines](https://github.com/Washington-University/HCPpipelines). User-specific files are approved.

- Required library (python>=3.6)
```
nilearn
sklearn
scipy
os
glob
numpy
nibabel
sklearn.mixture
```

## Implementation
 
 Usage:

```python
import apac
OutDir  = '/directory/that/output/will/be/saved'
DaraDir = '/directory/containing/above/required/files'

# initialize core processor
pcore = apac.core.core(OutDir)

# call the input data
pcore.call(DaraDir)  

# define ppcore
min_out = 1  
# 1: remain the minimum output (pcore & pcore_c)
# 0: remain intermediate files (initial_roi, clustK, curv_border)
pcore.def_pcore(min_out)

'''
Minimum ouputs
[hemi].pcore.func.gii       : Putative core (highly myelinated region) 
[hemi].pcore_c.func.gii     : Putative core adjusted for curvature (ideally located within Heschl's gyrus)

Additiaonl outputs
[hemi].initial_roi.func.gii : Initial ROI for parcellation
[hemi].clustK?.func.gii     : Clusters derived from Gaussian Mixture Model with k=?
[hemi].curv_border.func.gii : Suli border (limit of pCore expansion)
'''
```
 
## License

- APAC is licensed under the terms of the MIT license.

## Related papers

- 'Fully automated parcellation of the primary auditory cortex', Society for Neuroscience 2019 [Link](https://www.abstractsonline.com/pp8/#!/7883/presentation/50268)


## Core developers

- Kyoungseob Byeon: MIP Lab, Sunkyunkwan University
- Bo-yong Park: MICA Lab, Montreal Neurological Institute and Hospital
- Sean H. Lee: Max Planck Institute for Empirical Aesthetics

