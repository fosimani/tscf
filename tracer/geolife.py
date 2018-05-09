"""Extract information from the GeoLife dataset"""
import numpy as np
import os
import pandas as pd

from dateutil import parser
from time import time

from joblib import Parallel, delayed


def decode_str(s):
    return s.decode('utf-8')


class GeoLifeExtractor(object):

    def __init__(self, base_path='./geolife-gps-trajectory-dataset/'):
        self.base_path = base_path

    def extract_user_information(self, user):
        #
        # Get users dataset files
        #
        files = []
        user_dir = os.path.join(self.base_path, f'{user}/Trajectory/')

        for (_, _, filenames) in os.walk(user_dir):
            files.extend([f for f in filenames if f.endswith('.plt')])

        #
        # Build a pandas DataFrame gathering data for each file of the user
        #
        df = None

        columns = ['latitude', 'longitude', 'Unknown1', 'Unknown2', 'Unknown3', 'date', 'time']

        for file in files:
            data = np.genfromtxt(
                os.path.join(user_dir, file),
                delimiter=',',
                skip_header=6,
                converters={
                    0: float,
                    1: float,
                    2: float,
                    3: float,
                    4: float,
                    5: decode_str,
                    6: decode_str,
                }
            )

            if df is None:
                df = pd.DataFrame([list(l) for l in data], columns=columns)
            else:
                df_aux = pd.DataFrame([list(l) for l in data], columns=columns)
                df = pd.concat([df, df_aux])

        df['user'] = user

        print(f'{user}: Number of locations: {len(df)}')
        print(f'{user}: Trajectories from {df.date.min()} until {df.date.max()}')

        df.to_pickle(f'.tmp_user_tracjectories_{user}.pkl')

        return df

    def extract_all_users_information(self, n_jobs=1):
        self.users = [
            d
            for d in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, d))
        ]

        t = time()
        dfs = Parallel(n_jobs=n_jobs)(
            delayed(self.extract_user_information)(user)
            for user in self.users
        )
        print(f'Took {time() - t} gather information from all users')

        return pd.concat(dfs)
