"""
Opens the objects panda and random forest and predicts the data.
"""

import sys
import os
import django

sys.path.append('/Users/benjaminkuehnis/Documents/hsr/hs17/gameprediction/nsfs/nsfs/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'nsfs.settings'
django.setup()

from schedule.models import Game
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from game_data.gamescore.wikihoops import Wikihoops
from game_data.dataexport.dataframe_helper import remove_temporary_colums, get_results

from game_data.gamescore import basketballgame
import glob


def migrate_game_code_nba():
    for g in Game.objects.all():
        print(g.game_code[0:8] + g.game_code[12:15])
        g.game_code_nba = g.game_code[0:8] + g.game_code[12:15]
        g.save()

def reset():
    for g in Game.objects.exclude(random_forest_rating__exact = 0):
        g.random_forest_rating = 0
        g.save()

def import_random_forest_value():
    with open('../static/dumps/panda_dataset.p', 'rb') as pickle_file:
        df = pickle.load(pickle_file)

    with open('../static/dumps/random_forest.p', 'rb') as pickle_file:
        rf = pickle.load(pickle_file)

    for g in Game.objects.filter(random_forest_rating__exact=0):

        df_index = df.index[df['dbGameId'] == g.game_code_nba].tolist()
        print(df_index)
        if len(df_index) == 1:
            #print(df_index[0])
            #print(df.loc[[df_index[0]]])
            d = df.loc[[df_index[0]]]
            remove_temporary_colums(d)
            random_forest_prediction = rf.predict(d)
            g.random_forest_rating = random_forest_prediction
            g.save()
            print("{} | {}", g.game_code_nba, random_forest_prediction)
            print(d)


#migrate_game_code_nba()


reset()
import_random_forest_value()

game_data = []
w = Wikihoops()
#print(w.frame.head(2))

with open('../static/dumps/panda_dataset.p', 'rb') as pickle_file:
    df = pickle.load(pickle_file)

print(df.head(2))
print(df['dbGameId'].values)


print(df.shape)

print(df.head(2))
y = get_results(df)

remove_temporary_colums(df)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)

print(df.shape, X_train.shape, X_test.shape)

print("open file")
with open('../static/dumps/random_forest.p', 'rb') as pickle_file:
    rf = pickle.load(pickle_file)

print("Start calc")
predictions = rf.predict(X_test.head(2))

print(predictions)

plt.show(block=True)

