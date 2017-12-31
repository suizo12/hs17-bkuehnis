"""
Predicts the rating for the games
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
from schedule.models import NbaGame

warnings.filterwarnings("ignore")




def rf_modell(dataframe, y, title=''):
    print('-----------------------Start {0}-----------------------'.format(title))
    X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=0.2, random_state=1)

    rf = RandomForestRegressor(n_jobs=-1, random_state=72)
    rf.fit(X_train, y_train)  # rf ist nun unser trainiertes Model (Random Forests)
    return rf


def predict_rating(static_path='../static'):
    df_f_path = '{0}/dumps/{1}'.format(static_path, 'df_nba_boxscores.csv')
    df = pd.DataFrame.from_csv(df_f_path)
    print(21700528 in df.index)
    # Reddit ups
    df_ups = df[pd.notnull(df['ups'])].copy()
    print(21700528 in df_ups.index)
    df_ups.drop(['game_rating'], axis=1, inplace=True)
    df_ups.drop(['game_star'], axis=1, inplace=True)
    df_ups.drop(['rating_up'], axis=1, inplace=True)
    df_ups.drop(['rating_down'], axis=1, inplace=True)
    df_ups.drop(['rating_percentage'], axis=1, inplace=True)
    df_ups = df_ups[~df_ups.isnull().any(1)]
    df_ups['ups'] = df_ups['ups'].where(df_ups['ups'] <= 2500, 2500)
    df_ups.drop(['d'], axis=1, inplace=True)
    normalized_ups = df_ups['ups']
    df_ups.drop(['ups'], axis=1, inplace=True)

    #Wikihoops


    df = df[pd.notnull(df['attendance'])]
    df = df[pd.notnull(df['game_rating'])]
    df['game_rating'] = df['game_rating'].where(df['game_rating'] <= 15, 15)
    df['game_rating'] = df['game_rating'].where(df['game_rating'] >= -5, -5)
    game_rating = df['game_rating']
    df.drop(['game_rating'], axis=1, inplace=True)
    df.drop(['game_star'], axis=1, inplace=True)
    df.drop(['rating_up'], axis=1, inplace=True)
    df.drop(['rating_down'], axis=1, inplace=True)

    df.drop(['rating_percentage'], axis=1, inplace=True)
    df.drop(['d'], axis=1, inplace=True)
    df.drop(['ups'], axis=1, inplace=True)
    #p_game_rating = run_random_forest(df, original_game_rating, 'original wikihoops rating')
    #p_game_rating = run_random_forest(df, game_rating, 'wikihoops rating')
    #p_game_stars = run_random_forest(df, game_stars, 'wikihoops stars')
    #
    rf_reddit = rf_modell(df_ups, normalized_ups, 'reddit normalized ups')

    for nba_games in NbaGame.objects.filter(reddit_rating__isnull=True):
        i_id = int(nba_games.game_id[2:])
        if i_id in df_ups.index:
            nba_games.reddit_rating = int(rf_reddit.predict(df_ups.loc[[i_id]]))
            nba_games.save()


    rf = rf_modell(df, game_rating, 'wh normalized user-rating')

    for nba_games in NbaGame.objects.filter(wh_user_rating__isnull=True):
        i_id = int(nba_games.game_id[2:])
        if i_id == 21700528:
            print(nba_games.wh_user_rating)

        if i_id in df.index:
            nba_games.wh_user_rating = int(rf.predict(df.loc[[i_id]]))
            nba_games.save()



