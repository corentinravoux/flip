<img src="docs/_static/flip_logo.webp" width=350>

Field Level Inference Package: a python package to infer cosmological parameters from density and velocity field

Flip is a Python package that uses the Maximum likelihood method to fit growth rate based on velocity and density field. The first part of the software is the computation of a covariance matrix from a model power spectrum and the considered coordinates. This part is generalized to work for any linear power spectrum models, both for velocities, densities, and cross-terms, and it is optimized with Hankel transform for any model. In the second part, the covariance is used to create a likelihood by multiplying it by velocities or densities. Finally, this package includes some integrated fitter such as Minuit and MCMC (with emcee) to fit the growth rate of structure.


[![Documentation Status](https://readthedocs.org/projects/flip/badge/?version=latest)](https://flip.readthedocs.io/en/latest/?badge=latest)

## Quick install
```bash
git clone https://github.com/corentinravoux/flip.git
cd flip
pip install .
```

## Need help?
Documentation available on [ReadTheDoc](https://flip.readthedocs.io/) 

## How to cite

This package was started on the previous work of [@bastiencarreres](https://github.com/bastiencarreres), for now, please cite [this article](https://arxiv.org/abs/2303.01198)
A paper describing the concept of the code is in preparation.
