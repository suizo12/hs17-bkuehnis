import sys, os, django
sys.path.append('/Users/benjaminkuehnis/Documents/hsr/hs17/gameprediction/nsfs/nsfs/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'nsfs.settings'
django.setup()
from schedule.models import WikiHoops, NbaGame
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data():
    games = list(NbaGame.objects.all().values('game_id', 'date'))
    df_nba_games = pd.DataFrame(games)
    df_nba_games['game_id'] = df_nba_games['game_id'].astype('int')
    df_nba_games['d'] = df_nba_games['date'].str[0:4] + '-' + df_nba_games['date'].str[4:6] + '-' + df_nba_games[
                                                                                                        'date'].str[6:8]
    df_nba_games['d'] = pd.to_datetime(df_nba_games['d'])
    df_nba_games = df_nba_games.set_index('game_id')



    w_df = pd.DataFrame(list(
        WikiHoops.objects.all().values('game_rating', 'game_star', 'nba_game_id', 'rating_up', 'rating_down',
                                       'rating_percentage')))
    w_df = w_df.rename(columns={'nba_game_id': 'game_id'})
    w_df = w_df.set_index('game_id')


    df = pd.concat([df_nba_games, w_df], axis=1)
    df = df[pd.notnull(df['game_rating'])]
    return df


def game_rating():
    df = get_data()
    dates = df['d'].apply(pd.to_datetime).values

    print(df.sample(10))


    fig, ax = plt.subplots()
    ax.set_xlabel('Datum')
    ax.set_ylabel('User-Rating')
    plt.scatter(dates, df['game_rating'], s =1, c = 'red')
    plt.show()

def game_user_percentage():
    df = get_data()

    print(df.sample(10))
    print(df['rating_down'].sum())

    print(df['rating_up'].sum())
    print(df['game_star'].sum())
    print(3482+10214-8191)
    print(len(df[(df['rating_up'] != 0.0) & (df['rating_down'] != 0.0)]))
    print(len(df))

    df = df[(df['rating_up'] != 0.0) & (df['rating_down'] != 0.0)]
    print(len(df))
    df['rating_percentage'] = df['rating_percentage'].apply(int)
    print(len(df[df['rating_percentage'] > 74]))
    dates = df['d'].apply(pd.to_datetime).values
    fig, ax = plt.subplots()
    ax.set_xlabel('Datum')
    ax.set_ylabel('Rating Percentage')
    plt.scatter(dates, df['rating_percentage'], s =1, c = 'red')
    plt.show()

def game_star():
    df = get_data()
    dates = df['d'].apply(pd.to_datetime).values

    print(df.sample(10))


    fig, ax = plt.subplots()
    ax.set_xlabel('Datum')
    ax.set_ylabel('Game-Rating')
    plt.scatter(dates, df['game_star'], s =1, c = 'red')
    plt.show()

def only_season_2017():
    a = list(a.game_star for a in WikiHoops.objects.all())
    print(sorted(a))


    date = (datetime.utcfromtimestamp(a.nba_game.date).strftime('%Y-%m-%d') for a in
            WikiHoops.objects.filter(nba_game__isnull=False).order_by('created'))
    dates = [pd.to_datetime(d) for d in date]

    plt.scatter(dates, a, s=1, c='red')
    plt.show()

#game_star()
#game_rating()
game_user_percentage()