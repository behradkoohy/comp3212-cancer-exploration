import numpy as np
import pandas as pd

def get_datasets():
    brca = pd.read_csv('data/Gistic2_CopyNumber_BRCA', sep='\t')
    ovca = pd.read_csv('data/Gistic2_CopyNumber_OVCA', sep='\t')
    brca.set_index('Gene Symbol', inplace=True)
    ovca.set_index('Gene Symbol', inplace=True)
    brca = brca.transpose()
    ovca = ovca.transpose()
    brca['BRCA'] = 1
    ovca['BRCA'] = 0
    df = pd.concat([brca, ovca[1:]])
    print(len(df), len(brca.transpose()-2) + len(ovca.transpose()-2))
    return df

