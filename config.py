import itertools

FEATURES = {'xy': {'cols': ['COORDENADA_X', 'COORDENADA_Y']},
            'class': {'cols': ['ID_CLASSE']},
            'voltage': {'cols': ['ID_GRUPO']},
            'no_wires': {'cols': ['ID_FASE'], 'force-one-hot': True},
            'contract_status': {'cols': ['ID_SITUACAO'], 'force-one-hot': True},
            'meter_type': {'cols': ['ID_TIPO_DE_MEDIDOR']},
            }

# All two combinations of features + all
COMPOUND_FEATURES = dict(('_'.join(f), f) for f in itertools.combinations(FEATURES, 2))
# Generalized: ['_'.join(f) for k in xrange(2, 7) for f in itertools.combinations(features, k)]
COMPOUND_FEATURES['all'] = ['xy', 'class', 'voltage', 'no_wires', 'contract_status', 'meter_type']

# TODO Double check
LEVELS = {1: 'ID_REGIONAL',
          2: 'ID_MUNICIPIO',
          3: 'ID_LOCALIDADE',
          4: 'ID_BAIRRO'}
