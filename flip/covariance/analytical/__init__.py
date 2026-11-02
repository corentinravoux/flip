from flip._config import __minimal__

# Minimal mode: no model is imported here. CovMatrix loads the requested
# model by name, so a sparse checkout may hold any subset of them.
if not __minimal__:
    from . import (
        adamsblake17,
        adamsblake17plane,
        adamsblake20,
        carreres23,
        genericzdep,
        lai22,
        ravouxcarreres,
        ravouxnoanchor25,
        ravouxqin26,
        rcrk24,
    )
