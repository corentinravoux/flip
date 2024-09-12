_free_par = {
    'density': {'baseline' : ['sigv', 'bs8']},
    'velocity': {'baseline': ['fs8', 'sigv']}
    }

_free_par['density_velocity'] = {
    'baseline': [
    i for k in _free_par.keys() for i in _free_par[k]['baseline']
    ]
    }

_free_par['full'] = _free_par['density_velocity']