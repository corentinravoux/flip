# Covariance models & emulators

The analytical covariance model registry and the covariance emulators. Each model
module documents its free parameters, variants and required coordinate keys; the
generated `flip_terms.py` / `coefficients.py` / `fisher_terms.py` kernel files are
omitted (see [`symbolic`](#symbolic-code-generation)). See the
[Covariance models](../covariance-models.md) guide for an overview.

## Registry

::: flip.covariance.analytical

## Models

::: flip.covariance.analytical.adamsblake17

::: flip.covariance.analytical.adamsblake17plane

::: flip.covariance.analytical.adamsblake20

::: flip.covariance.analytical.carreres23

::: flip.covariance.analytical.lai22

::: flip.covariance.analytical.rcrk24

::: flip.covariance.analytical.ravouxcarreres

::: flip.covariance.analytical.ravouxnoanchor25

::: flip.covariance.analytical.ravouxqin26

::: flip.covariance.analytical.genericzdep

## Emulators

::: flip.covariance.emulators.generator

::: flip.covariance.emulators.gpmatrix

::: flip.covariance.emulators.skgpmatrix

::: flip.covariance.emulators.nnmatrix

## Symbolic code generation

::: flip.covariance.symbolic
