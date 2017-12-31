import pandas as pd
import glob
import os


class Wikihoops:
    def __init__(self, static_path = '../static'):
        print(os.getcwd())
        list_ = []
        self.frame = pd.DataFrame()
        for game_file in glob.glob("{0}/wikihoops/*.csv".format(static_path)):
            df = pd.read_csv(game_file)
            df['code'] = df['gameId'].apply(lambda x: x[0:8] + x[11:14])
            df['dbGameId'] = df['gameId'].apply(lambda x: x[0:8] + '/' + x[8:14])
            list_.append(df)
        self.frame = pd.concat(list_)
