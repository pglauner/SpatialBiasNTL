from os.path import expanduser
import pandas as pd


DB_ROOT = '{0}/DATA'.format(expanduser('~'))
DB_PATH = '{0}/{1}'.format(DB_ROOT, 'celpe_latest')
CONSUMPTIONS_FILENAMES = ['fato_201201.dsv', 'fato_201301.dsv', 'fato_201401.dsv', 'fato_201501.dsv', 'fato_201601.dsv']
CUSTOMER_FILENAME = 'uc.dsv'
INSPECTIONS_FILENAME = 'ocorrencia.dsv'

CONSUMPTIONS_HEADER_RELEVANT = ['DATA_REFERENCIA', 'CONSUMO_MEDIDO', 'CONSUMO_FATURADO', 'DIAS_FATURADOS']

INSPECTIONS_HEADER_RELEVANT = ['DATA_REFERENCIA', 'ID_RESULTADO']


def get_raw_consumptions(extra_columns = []):
    frames = []
    for file in CONSUMPTIONS_FILENAMES:
        filepath = '{0}/{1}'.format(DB_PATH, file)
        X = pd.read_csv(filepath, sep=';', usecols=['ID_UC'] + extra_columns)
        frames.append(X)
        print 'Read raw consumptions {0}'.format(file)
    return pd.concat(frames)


def get_raw_inspections(extra_columns= []):
    filepath = '{0}/{1}'.format(DB_PATH, INSPECTIONS_FILENAME)
    X = pd.read_csv(filepath, sep=';', usecols=['ID_UC'] + extra_columns)
    print 'Read raw inspections'
    return X


def get_raw_ID_UC(extra_columns = []):
    filepath = '{0}/{1}'.format(DB_PATH, CUSTOMER_FILENAME)
    X = pd.read_csv(filepath, sep=';', usecols=['ID_UC'] + extra_columns, error_bad_lines=False)
    print 'Read raw customers'
    return X
