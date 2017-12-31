import sys
import os
import django
sys.path.append('/Users/benjaminkuehnis/Documents/hsr/hs17/gameprediction/nsfs/nsfs/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'nsfs.settings'
django.setup()
from schedule.models import NbaGame, WikiHoops, NbaReddit
from game_data.nbadataimport.box_score import BoxScore, save_nba_df_to_file
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


def feature_importance(df, forest, title):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(df.columns, forest.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance')[:10].plot(title=title, kind='bar', rot=45)


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

    y = pd.concat([y_train, y_test])
    predicted = cross_val_predict(lr, pd.concat([X_test]), pd.concat([y_test]), cv=10)
    predicted = list(map(int, predicted))
    i = predicted.index(max(predicted))
    y = y_test.values
    y = np.delete(y, i)
    print(y)
    #print(y.delete(i))

    predicted.remove(max(predicted))
    print(min(predicted))
    print(max(predicted))
    print(predicted)
    fig, ax = plt.subplots()
    print(len(predicted))

    print(len(y))
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([min(y),max(y)], [min(y), max(y)], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    print('-----------------------End {0}-----------------------'.format(title))
    return predictions


# save_nba_df_to_file()
#df = df[pd.notnull(df['game_rating'])]
#df = df[pd.notnull(df['attendance'])]

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
print(len(df))
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


# Reddit ups



df_ups = df[pd.notnull(df['ups'])]

base = datetime.date(2017, 6, 15)
date_list = [(base - datetime.timedelta(days=x)).isoformat() for x in range(0, 63)]
#playoff = df_ups[(df_ups['d'].isin(date_list))]
#df_ups = df_ups[~(df_ups['d'].isin(date_list))]
df_ups['ups'] = df_ups['ups'].where(df_ups['ups'] <= 2500, 2500)

print(len(df_ups))
l = list()
normalized_rating = list()


df_date_ups = pd.DataFrame(df_ups[['d', 'ups']])


df_ups['ups'] = df_ups['ups'].astype('int').copy()


df_date_ups['d'] = pd.to_datetime(df_date_ups['d'])
df_date_ups = df_date_ups.set_index('d')

for date, row in df_date_ups.iterrows():

    a = df_date_ups.loc[df_date_ups.index.year == date.year]
    l = list(a[(a.index.month == date.month) & (a.index.day == date.day)]['ups'])

    ns = get_score(row['ups'], l)
    #print('{2} : {0}: {1}'.format(date, sorted(l), ns))
    normalized_rating.append(ns)

df_ups['normalized_ups'] = pd.Series(normalized_rating, index=df_ups.index)
#print(df_ups['normalized_ups'].sort_values())
ups = df_ups['ups']
df_ups.drop(['ups'], axis=1, inplace=True)

normalized_ups = df_ups['normalized_ups']

df_ups.drop(['normalized_ups'], axis=1, inplace=True)

df.drop(['d'], axis=1, inplace=True)
df.drop(['ups'], axis=1, inplace=True)
df_ups.drop(['d'], axis=1, inplace=True)


p_game_rating = run_random_forest(df, original_game_rating, 'original wikihoops rating')
p_game_rating = run_random_forest(df, game_rating, 'wikihoops rating')
#p_game_stars = run_random_forest(df, game_stars, 'wikihoops stars')
#plt.show()
#p_ups_normalized = run_random_forest(df_ups, normalized_ups, 'reddit normalized ups')
#p_ups = run_random_forest(df_ups, ups, 'reddit ups')


#plt.hist(p_ups_normalized, alpha=0.5, label='reddit normalized ups')

#plt.hist(p_game_stars, alpha=0.5, label='wikihoops star')

#plt.legend()
#plt.xlabel("Prediction Values")
#plt.ylabel("Frequency")



