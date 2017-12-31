import sys, os, django
sys.path.append('/Users/benjaminkuehnis/Documents/hsr/hs17/gameprediction/nsfs/nsfs/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'nsfs.settings'
django.setup()
from schedule.models import NbaReddit, NbaGame, Team, WikiHoops
from game_data.plot.utils import get_score


scores = sorted((g.ups for g in NbaReddit.objects.filter(nba_game__isnull=False)))




d1 = list()
d2 = list()
d3 = list()
date = list()
for game in NbaReddit.objects.filter(nba_game__isnull=False).order_by('created'):

    w = WikiHoops.objects.filter(nba_game_id=game.nba_game_id)
    if w.exists():
        d1.append(get_score(game.ups, scores))
        d2.append(w.first().game_star)
        d3.append(w.first().game_rating)
        date.append(game.nba_game.start_time_utc.strftime('%Y-%m-%d'))



import numpy as np
import matplotlib.pyplot as plt

# Create data
N = 60


data = (d1, d2, d3)
colors = ("red", "green", "blue")
groups = ("reddit", "w_star", "w_rating")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
from datetime import datetime
import pandas as pd
dates = [pd.to_datetime(d) for d in date]

for date, data, color, group in zip(dates, data, colors, groups):
    x = date
    y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()