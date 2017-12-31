import sys, os, django
sys.path.append('/Users/benjaminkuehnis/Documents/hsr/hs17/gameprediction/nsfs/nsfs/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'nsfs.settings'
django.setup()
from schedule.models import NbaReddit, NbaGame, Team, WikiHoops
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from game_data.plot.utils import get_score

games = list(NbaGame.objects.all().values())
df = pd.DataFrame(games)
df['game_id'] = df['game_id'].astype('int')
df['d'] = df['date'].str[0:4] + '-' + df['date'].str[4:6] + '-' + df['date'].str[6:8]
df['d'] = pd.to_datetime(df['d'])
df = df.set_index('game_id')


w_df = pd.DataFrame(list(WikiHoops.objects.all().values('game_rating', 'game_star', 'nba_game_id')))
w_df = w_df.rename(columns = {'nba_game_id': 'game_id'})
w_df = w_df.set_index('game_id')


r_df = pd.DataFrame(list(NbaReddit.objects.filter(nba_game_id__isnull=False).values('nba_game_id', 'ups')))
r_df = r_df.rename(columns = {'nba_game_id': 'game_id'})
r_df = r_df.set_index('game_id')
r_df = r_df[~r_df.index.duplicated(keep='first')]


df = pd.concat([df, w_df], axis=1)
df = pd.concat([df, r_df], axis=1)

# remove nan values
df = df[pd.notnull(df['ups'])]
df = df[pd.notnull(df['game_star'])]
df = df[pd.notnull(df['game_rating'])]


df['normalized_game_rating'] = df['game_rating']
df['normalized_game_rating'] = df['normalized_game_rating'].where(df['normalized_game_rating'] <= 15, 15)
df['normalized_game_rating'] = df['normalized_game_rating'].where(df['normalized_game_rating'] >= -5, -5)


df['ups'] = df['ups'].where(df['ups'] <= 2500, 2500)

df['ups'] = df['ups'] * 100 / 2500
print(df['ups'])
df['game_star'] = df['game_star'] * 10

df['normalized_game_rating'] = df['normalized_game_rating'] + 5
df['normalized_game_rating'] = df['normalized_game_rating'] * 100 / 20
df['w_diff'] = df['ups'] - df['normalized_game_rating']
df['w_diff'] = df['w_diff'].where(df['w_diff'] >= 0 , df['w_diff'] * -1)

df['normalized_game_rating'] = df['normalized_game_rating'].apply(int)
df['ups'] = df['ups'].apply(int)


print(df['w_diff'].head(10))
print(df['w_diff'].sum() / len(df['w_diff']))
print(len(df[(df['normalized_game_rating'] > 75) & (df['ups'] > 75)]))
print(len(df[(df['normalized_game_rating'] > 66) & (df['ups'] > 66)]))

print(len(df[(df['normalized_game_rating'] < 33) & (df['ups'] < 33)]))
print(len(df[(df['w_diff'] < 15)]))
print('-----')
game_rating = list()
e_game_rating = list()
ups = list()
e_ups = list()

for i in range(0,100,10):
    i2 = (i +10)
    print(i, i2)
    if i2 == 100:
        #np.mean()
        e_ups.append(np.mean(df[(df['ups'] >= i) & (df['ups'] <= i2)]['w_diff'].values))
        e_game_rating.append(np.mean(df[(df['normalized_game_rating'] >= i) & (df['normalized_game_rating'] <= i2)][
                                       'w_diff'].values))

        ups.append(np.mean(df[(df['ups'] >= i) & (df['ups'] <= i2)]['ups'].values))
        game_rating.append(np.mean(df[(df['normalized_game_rating'] >= i) & (df['normalized_game_rating'] <= i2)]['normalized_game_rating'].values))
    else:
        e_ups.append(np.mean(df[(df['ups'] >= i) & (df['ups'] <= i2)]['w_diff'].values))
        e_game_rating.append(np.mean(df[(df['normalized_game_rating'] >= i) & (df['normalized_game_rating'] <= i2)][
                                         'w_diff'].values))
        ups.append(np.mean(df[(df['ups'] >= i) & (df['ups'] < i2)]['ups'].values))
        game_rating.append(np.mean(df[(df['normalized_game_rating'] >= i) & (df['normalized_game_rating'] < i2)]['normalized_game_rating'].values))
print(ups)
print(game_rating)

print('-----')
width = 0.35       # the width of the bars

fig, ax = plt.subplots()

N = 10
men_means = (20, 35, 30, 35, 27,20, 35, 30, 35, 27)
men_std = (2, 3, 4, 1, 2,2, 3, 4, 1, 2)

ind = np.arange(N)

rects1 = ax.bar(ind, game_rating, width, yerr=e_game_rating)

women_means = (25, 32, 34, 20, 25,25, 32, 34, 20, 25)
women_std = (20, 3, 4, 1, 2,2, 3, 4, 1, 2)
rects2 = ax.bar(ind + width, ups, width, yerr=e_ups)

# add some text for labels, title and axes ticks
ax.set_ylabel('Durchschnittlicher Sehenswürdigkeitswert')
ax.set_xlabel('Gruppierte Sehenswürdigkeit in %')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100'))

ax.legend((rects1[0], rects2[0]), ('Wikihoops User-Rating', 'Reddit-Ups'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()

#print(df.head(20))
##print(df['w_diff'].sum())
#print(len(df['w_diff']))
# plot

take = 1500
dates = [pd.to_datetime(d) for d in df['d'].tail(take)]
#plt.scatter(dates, df['game_star'].tail(take), label='star')
plt.scatter(dates, df['normalized_game_rating'].tail(take), label='Wikihoops User-Rating')

plt.scatter(dates, df['w_diff'].tail(take), label='Difference between User-Rating and Reddit-Ups')
plt.scatter(dates, df['ups'].tail(take), label='Reddit-Ups')
plt.xticks(rotation=90)
plt.legend()
# pd.DataFrame(pf[['d', 'game_star']].tail(20)).plot.scatter(x='d', y='game_star')
#plt.show()


