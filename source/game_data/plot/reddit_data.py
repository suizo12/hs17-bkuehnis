import sys, os, django
sys.path.append('/Users/benjaminkuehnis/Documents/hsr/hs17/gameprediction/nsfs/nsfs/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'nsfs.settings'
django.setup()
from schedule.models import NbaReddit
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def all_data():
    a = list(a.ups for a in NbaReddit.objects.filter(nba_game__isnull=False).order_by('created'))
    print(len(a))
    date = (datetime.utcfromtimestamp(a.created).strftime('%Y-%m-%d') for a in NbaReddit.objects.filter(nba_game__isnull=False).order_by('created'))
    dates = [pd.to_datetime(d) for d in date]

    fig, ax = plt.subplots()
    ax.set_xlabel('Datum')
    ax.set_ylabel('Reddit Wertung')
    plt.scatter(dates, a, s =1, c = 'red')
    plt.show()

def only_season_2017():
    a = list(a.ups for a in NbaReddit.objects.filter(nba_game__isnull=False).order_by('created'))
    print(sorted(a))


    date = (datetime.utcfromtimestamp(a.created).strftime('%Y-%m-%d') for a in
            NbaReddit.objects.filter(nba_game__isnull=False).order_by('created'))
    dates = [pd.to_datetime(d) for d in date]

    plt.scatter(dates, a, s=1, c='red')
    plt.show()

all_data()