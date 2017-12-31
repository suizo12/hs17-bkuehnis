from sklearn.model_selection import train_test_split
import pandas as pd
from game_data.gamescore import basketballgame
from game_data.dataexport.dataframe_helper import remove_temporary_colums, get_results
import glob
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score
from game_data.gamescore.wikihoops import Wikihoops
import pickle
from sklearn.model_selection import ShuffleSplit

'''
Get all data from 2014-2018 season
Transform the data to panda
dump the panda file to '../static/dumps/panda_dataset.p'
'''
keep_columns = ['away_+/-', 'away_TS%', 'away_TOV', 'home_+/-', 'home_TS%', 'home_TOV', 'rating', 'game']

game_data = []
w = Wikihoops()

#for game_file in glob.glob("../static/matches/united_states/nba/2003-2004//*.json"):
#    game = basketballgame.BasketballGame(game_file, w.frame, True)
#    game_data.append(basketballgame.BasketballGame(game_file).data)
#for game_file in glob.glob("../static/matches/united_states/nba/2014-2015//*.json"):
#    game_data.append(basketballgame.BasketballGame(game_file, w.frame, True).data)
#for game_file in glob.glob("../static/matches/united_states/nba/2015-2016//*.json"):
#    game_data.append(basketballgame.BasketballGame(game_file, w.frame, True).data)
for game_file in glob.glob("../static/matches/united_states/nba/2016-2017//*.json"):
    #game = basketballgame.BasketballGame(game_file, w.frame, True)
    game_data.append(basketballgame.BasketballGame(game_file, w.frame, False).data)
for game_file in glob.glob("../static/matches/united_states/nba/2017-2018//*.json"):
    # game = basketballgame.BasketballGame(game_file, w.frame, True)
    game_data.append(basketballgame.BasketballGame(game_file, w.frame, False).data)


df = pd.DataFrame.from_records(game_data)
#df = df[keep_columns]

#df.plot(kind='box', subplots=True, sharex=False, layout=(2, 3), figsize=(18, 8))

#df.plot(kind='kde', subplots=True, sharex=False, layout=(2, 3), figsize=(18, 8))



# print('Is there any null values:')
# print(df.isnull().any())

print(df.shape)
with open('../static/dumps/panda_dataset.p', 'wb') as pickle_file:
    pickle.dump(df, pickle_file)

y = get_results(df)

remove_temporary_colums(df)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)

print(df.shape, X_train.shape, X_test.shape)

lr = linear_model.Ridge(alpha=.5, normalize=True)
print(lr.fit(X_train, y_train))  # lr ist nun unser trainiertes Model (Linear Regression)

rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=72)
print(rf.fit(X_train, y_train))  # rf ist nun unser trainiertes Model (Random Forests)

pred_lr = lr.predict(X_test)  # Linear Regression
pred_rf = rf.predict(X_test)  # Random Forests

# Linear Regression
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
print(rmse_lr)

# Random Forests
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
print(rmse_rf)

scores = cross_val_score(rf, pd.concat([X_train, X_test]),
                         pd.concat([y_train, y_test]),
                         cv=5, scoring='neg_mean_squared_error')
print(np.sqrt(-1*scores))

combined_error = rmse_lr + rmse_rf
weight_lr = 1-rmse_lr/combined_error
weight_rf = 1-rmse_rf/combined_error

print("Lineare Regression:\t {}".format(rmse_lr))
print("Random Forests:\t\t {}".format(rmse_rf))
print("Weighted Avgerage:\t {}".format(np.sqrt(mean_squared_error(y_test, weight_lr*pred_lr + weight_rf*pred_rf))))

predictions = rf.predict(X_test)

print(predictions)
with open('../static/dumps/random_forest.p', 'wb') as pickle_file:
    pickle.dump(predictions, pickle_file)
