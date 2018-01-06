# susceptibility-imaging-toolkit
The Susceptibility Imaging Toolkit provides functions and a UI for Quantitative Susceptibility Mapping (QSM) analysis of brain MRI images.

The toolkit reads in MRI scans from either .mat files or .dcm files and processes the phase using the following pipeline:
phase ---Laplacian unwrapping---> unwrapped phase ---V-SHARP background removal---> tissue phase ---QSM solver---> susceptibility
The functions are integrated into a UI based on PyQt5 for ease of use.

Related papers:
Phase unwrapping: equation 1 in http://onlinelibrary.wiley.com/doi/10.1002/nbm.3056/full
Background phase removal: Schweser F, Deistung A, Berengar WL, Reichenbach JR. Quantitative imaging of intrinsic magnetic tissue properties using MRI signal phase: An approach to in vivo brain iron metabolism? NeuroImage. 2010 doi:10.1016/j.neuroimage.2010.10.070.
QSM inversion: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3249423/, https://www.ncbi.nlm.nih.gov/pubmed/26313885
