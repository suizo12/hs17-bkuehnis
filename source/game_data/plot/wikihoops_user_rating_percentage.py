import sys
import os
import django
sys.path.append('/Users/benjaminkuehnis/Documents/hsr/hs17/gameprediction/nsfs/nsfs/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'nsfs.settings'
django.setup()
from schedule.models import NbaGame, WikiHoops, NbaReddit
from game_data.nbadataimport.box_score import BoxScore
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from game_data.plot.utils import get_score, get_percent_score
import seaborn as sns
from game_data.utils.variable_to_file import load_variable_from_file, save_variable_to_file, \
    WIKIHOOPS_RATING_RANDOM_FOREST_PREDICTIONS_FILE, WIKIHOOPS_STAR_RANDOM_FOREST_PREDICTIONS_FILE
import warnings
import datetime
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")

# sns.set()
# uniform_data = np.random.rand(10, 12)
# ax = sns.heatmap(uniform_data)

df_f_path = '../static/dumps/{0}'.format('df_nba_boxscore_v2')

def save_nba_df_to_file():
    game_data = []

    for game_file in glob.glob("../static/nba/boxscore/2015//*.json"):
        game_data.append(BoxScore(game_file).data)

    for game_file in glob.glob("../static/nba/boxscore/2016//*.json"):
        game_data.append(BoxScore(game_file).data)

    for game_file in glob.glob("../static/nba/boxscore/2017//*.json"):
        game_data.append(BoxScore(game_file).data)

    games = list(NbaGame.objects.all().values('game_id', 'date'))
    df_nba_games = pd.DataFrame(games)
    df_nba_games['game_id'] = df_nba_games['game_id'].astype('int')
    df_nba_games['d'] = df_nba_games['date'].str[0:4] + '-' + df_nba_games['date'].str[4:6] + '-' + df_nba_games[
                                                                                                        'date'].str[6:8]
    df_nba_games['d'] = pd.to_datetime(df_nba_games['d'])
    df_nba_games = df_nba_games.set_index('game_id')


    df = pd.DataFrame.from_records(game_data)
    df = df.rename(columns={'gameId': 'game_id'})
    df['game_id'] = df['game_id'].astype('int')
    df = df.set_index('game_id')

    df = pd.concat([df, df_nba_games], axis=1)

    w_df = pd.DataFrame(list(WikiHoops.objects.all().values('game_rating', 'game_star', 'nba_game_id', 'rating_up', 'rating_down', 'rating_percentage')))
    w_df = w_df.rename(columns={'nba_game_id': 'game_id'})
    w_df = w_df.set_index('game_id')


    # print(w_df.loc[[11700007]])

    r_df = pd.DataFrame(list(NbaReddit.objects.filter(nba_game_id__isnull=False).values('nba_game_id', 'ups')))
    r_df = r_df.rename(columns={'nba_game_id': 'game_id'})
    r_df = r_df.set_index('game_id')
    r_df = r_df[~r_df.index.duplicated(keep='first')]

    df = pd.concat([df, r_df], axis=1)

    df = pd.concat([df, w_df], axis=1)
    df = df[pd.notnull(df['game_rating'])]

    df = df[pd.notnull(df['attendance'])]
    # print(len(df))
    # print(df.head(5))
    # print(df.dtypes)

    for header in list(df):
        try:
            df[header] = df[header].astype(str).astype(float)
        except:
            print(header)
    # df = df[keep_columns]

    # df.plot(kind='box', subplots=True, sharex=False, layout=(2, 3), figsize=(18, 8))

    # df.plot(kind='kde', subplots=True, sharex=False, layout=(2, 3), figsize=(18, 8))

    # print(df.head(5))

    #
    # print('Is there any null values:')
    # print(df.isnull().any())
    #
    # print(df.shape)
    # with open('../static/dumps/panda_dataset.p', 'wb') as pickle_file:
    #     pickle.dump(df, pickle_file)
    # .

    df.to_csv(df_f_path)


def feature_importance(df, forest, title):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(df.columns, forest.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance')[:10].plot(title=title, kind='bar', rot=45)


def remove_some_bad_data(x, y):
    i = x.index(max(x))

    y = np.delete(y, i)
    # print(y.delete(i))

    x.remove(max(x))
    print(min(x))
    print(max(x))
    print(x)
    return x, y


def run_random_forest(dataframe, y, title=''):
    print('-----------------------Start {0}-----------------------'.format(title))
    X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=0.2, random_state=1)
    #
    # print(df.shape, X_train.shape, X_test.shape)
    #
    lr = linear_model.Ridge(alpha=0.000001, normalize=True)
    lr.fit(X_train, y_train)  # lr ist nun unser trainiertes Model (Linear Regression)

    lasso = linear_model.Lasso(alpha=0.000001)
    lasso.fit(X_train, y_train)  # lr ist nun unser trainiertes Model (Linear Regression)
    pred_lasso = lasso.predict(
        X_test)  # Linear RegressionX_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    pred_lasso = lasso.predict(X_test)  # Random Forests
    #
    rf = RandomForestRegressor(n_jobs=-1, random_state=72)
    rf.fit(X_train, y_train)  # rf ist nun unser trainiertes Model (Random Forests)
    #
    pred_lr = lr.predict(
        X_test)  # Linear RegressionX_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
    pred_rf = rf.predict(X_test)  # Random Forests
    #
    # # lasso Regression
    rmse_lasso = np.sqrt(mean_squared_error(y_test, pred_lr))
    #print(rmse_lasso)

    # # Linear Regression
    rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
    #print(rmse_lr)
    #
    # # Random Forests
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    #print(rmse_rf)
    #
    scores = cross_val_score(rf, pd.concat([X_train, X_test]),
                             pd.concat([y_train, y_test]),
                             cv=5, scoring='neg_mean_squared_error')
    print(np.sqrt(-1 * scores))
    #
    combined_error = rmse_lr + rmse_rf
    weight_lr = 1 - rmse_lr / combined_error
    weight_rf = 1 - rmse_rf / combined_error
    #
    print("Lineare Regression:\t {}".format(rmse_lr))
    print("Random Forests:\t\t {}".format(rmse_rf))
    print("Random Forests:\t\t {}".format(rmse_lr))
    print("Weighted Avgerage:\t {}".format(
        np.sqrt(mean_squared_error(y_test, weight_lr * pred_lr + weight_rf * pred_rf))))

    #
    print('Ridge Trainnig set score: {0}'.format(lr.score(X_train, y_train)))
    print('Ridge Test set score: {0}'.format(lr.score(X_test, y_test)))

    print('Lasso Trainnig set score: {0}'.format(lasso.score(X_train, y_train)))
    print('Lasso Test set score: {0}'.format(lasso.score(X_test, y_test)))

    print('Random Forest Trainnig set score: {0}'.format(rf.score(X_train, y_train)))
    print('Random Forest Test set score: {0}'.format(rf.score(X_test, y_test)))

    #
    predictions = rf.predict(dataframe)
    feature_importance(dataframe, rf, title)

    y = pd.concat([y_test])
    s_df = pd.DataFrame(y)
    t_df = pd.DataFrame(X_test)
    all_data = pd.concat([t_df, s_df], axis=1)
    print(all_data.sample(10))
    print(len(df))
    print(len(t_df))
    X_test['rating_percentage'] = X_test['rating_percentage'].apply(int)
    all_data['rating_percentage'] = all_data['rating_percentage'].apply(int)
    top_df = all_data[all_data['rating_percentage'] >= 75]
    bad_df = all_data[all_data['rating_percentage'] < 75]


    print(len(all_data))

    print(len(bad_df) + len(top_df))

    predicted_top = cross_val_predict(rf, pd.concat([X_test[X_test['rating_percentage'] >= 75]]), top_df['game_rating'].values, cv=10)
    predicted_bad = cross_val_predict(rf, pd.concat([X_test[X_test['rating_percentage'] < 75]]), bad_df['game_rating'].values, cv=10)
    predicted_bad = list(map(int, predicted_bad))
    predicted_top = list(map(int, predicted_top))

    print(predicted_top)
    print('---')
    print(len(predicted_top))
    print(len(top_df['game_rating'].values))
    print(len(predicted_bad))
    print(len(bad_df['game_rating'].values))
    fig, ax = plt.subplots()

    ax.scatter(top_df['game_rating'].values, predicted_top, edgecolors=(0, 0, 0), alpha=0.3, label='User agreement >= 75%')
    ax.scatter(bad_df['game_rating'].values, predicted_bad, edgecolors=(0, 0, 0), alpha=0.3, label='User agreement < 75%')
    ax.plot([min(y),max(y)], [min(y), max(y)], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.legend(fontsize='small')
    plt.show()
    print('-----------------------End {0}-----------------------'.format(title))
    return predictions


# save_nba_df_to_file()
df = pd.DataFrame.from_csv(df_f_path)
df = df[~(df['rating_percentage'] == '?')]

#Start original wikihoops rating-
#Random Forest Trainnig set score: 0.9380983892546473
#Random Forest Test set score: 0.6391530046154357
#Random Forest Trainnig set score: 0.9482792532694236
#Random Forest Test set score: 0.6653169400651051
#
#df = df[(df['rating_up'] != 0.0) & (df['rating_down'] != 0.0)]
#df['rating_percentage'] = df['rating_percentage'].apply(int)
#print(len(df[df['rating_percentage'] > 74]))
#df = df[df['rating_percentage'] > 74]
#normalize y
original_game_rating = df['game_rating'].copy()

df['game_rating'] = df['game_rating'].where(df['game_rating'] <= 15, 15)
df['game_rating'] = df['game_rating'].where(df['game_rating'] >= -5, -5)


game_rating = df['game_rating']
game_stars = df['game_star']

df.drop(['game_rating'], axis=1, inplace=True)
df.drop(['game_star'], axis=1, inplace=True)
df.drop(['rating_up'], axis=1, inplace=True)
df.drop(['rating_down'], axis=1, inplace=True)
df.drop(['d'], axis=1, inplace=True)
df.drop(['ups'], axis=1, inplace=True)


p_game_rating = run_random_forest(df, original_game_rating, 'original wikihoops rating')
#p_game_rating = run_random_forest(df, game_rating, 'wikihoops rating')

