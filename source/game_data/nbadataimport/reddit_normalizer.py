import pandas as pd
import numpy as np
df_f_path = '../static/dumps/{0}'.format('df_nba_boxscore_v2')

df = pd.DataFrame.from_csv(df_f_path)
print(df.columns)
print(df.sample(10))
df = df[pd.notnull(df['ups'])][['ups', 'game_rating', 'game_star', 'd']]
print(df.sort_values('d'))