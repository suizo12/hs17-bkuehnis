from sklearn.model_selection import train_test_split
import pandas as pd
from game_data.gamescore import basketballgame
from game_data.dataexport.dataframe_helper import remove_temporary_colums, get_results
import glob
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn.model_selection import cross_val_score
from game_data.gamescore.wikihoops import Wikihoops
import pickle
#from game_data.dataexport.plot_lc import plot_learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample
'''
Get all data from 2014-2018 season
Transform the data to panda
dump the panda file to '../static/dumps/panda_dataset.p'
'''
keep_columns = ['away_+/-', 'away_TS%', 'away_TOV', 'home_+/-', 'home_TS%', 'home_TOV', 'rating', 'dbGameId']

game_data = []
w = Wikihoops()

#
#for game_file in glob.glob("../static/matches/united_states/nba/2003-2004//*.json"):
#    game_data.append(basketballgame.BasketballGame(game_file, w.frame, False).data)


# for game_file in glob.glob("../static/matches/united_states/nba/2014-2015//*.json"):
#     game_data.append(basketballgame.BasketballGame(game_file, w.frame, False).data)
# for game_file in glob.glob("../static/matches/united_states/nba/2015-2016//*.json"):
#     game_data.append(basketballgame.BasketballGame(game_file, w.frame, False).data)
# for game_file in glob.glob("../static/matches/united_states/nba/2016-2017//*.json"):
#     game_data.append(basketballgame.BasketballGame(game_file, w.frame, False).data)
# for game_file in glob.glob("../static/matches/united_states/nba/2017-2018//*.json"):
#     game_data.append(basketballgame.BasketballGame(game_file, w.frame, False).data)

#
#df = pd.DataFrame.from_records(game_data)
#df.to_csv('prototype.csv')
df = pd.DataFrame.from_csv('prototype.csv')
print(len(df))
#df = df[keep_columns]

#df.plot(kind='box', subplots=True, sharex=False, layout=(2, 3), figsize=(18, 8))

#df.plot(kind='kde', subplots=True, sharex=False, layout=(2, 3), figsize=(18, 8))



# print('Is there any null values:')
# print(df.isnull().any())


y = get_results(df)
#y = y.where(y <= 15, 15)
#print(y.min())
#print(y.max())

remove_temporary_colums(df)

n_p = int(len(df) / 10)
for c in df.columns:
    if c != 'dbGameId':
        #print(df.nlargest(n_p, c))
        print('{0}, {1}'.format(c, df.nlargest(n_p, c).iloc[-1][c]))

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)


lr = linear_model.LinearRegression(normalize=True)
#lr = linear_model.Ridge(alpha=0.001, normalize=True)
print(lr.fit(X_train, y_train))  # lr ist nun unser trainiertes Model (Linear Regression)


pred_lr = lr.predict(X_test)  # Linear Regression

# Linear Regression
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
print(rmse_lr)

# Random Forests

scores = cross_val_score(lr, pd.concat([X_train, X_test]),
                         pd.concat([y_train, y_test]),
                         cv=5, scoring='neg_mean_squared_error')
print(np.sqrt(-1*scores))

params = pd.Series(lr.coef_, index=df.columns)

print("Lineare Regression:\t {}".format(rmse_lr))
# The coefficients
print('Coefficients: \n', lr.coef_)
# The mean squared error
predictions = lr.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, predictions))
#err = np.std([lr.fit(*resample(X_test, y_test)).coef_ for i in range(1000)], 0)
#print(pd.DataFrame({'effect': params.round(0), 'error': err.round(0)}))

y = pd.concat([y_train, y_test])
predicted = cross_val_predict(lr, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=10)
print('LR Train set score: {0}'.format(lr.score(X_train, y_train)))
print('LR set score: {0}'.format(lr.score(X_test, y_test)))

fig, ax = plt.subplots()
ax.scatter(y_test, predictions, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
