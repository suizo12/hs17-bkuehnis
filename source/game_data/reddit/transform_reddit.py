import sys, os, django
sys.path.append('/Users/benjaminkuehnis/Documents/hsr/hs17/gameprediction/nsfs/nsfs/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'nsfs.settings'
django.setup()
from schedule.models import NbaReddit, NbaGame, Team
from datetime import datetime, timedelta
from django.db.models import Q
import pytz     ## pip install pytz


def getTeams(teams, title):
    r = list()
    for t in teams:
        if t.name in title:
            r.append(t)
    return r

def getTeamsByCity(teams, title):
    r = list()
    for t in teams:
        if t.city in title:
            r.append(t)
    return r

def update_nba_reddit(date, teams, nba_reddit):
    nbagames = NbaGame.objects.filter(date__exact=date.strftime('%Y%m%d')).filter(
        Q(v_team_id__exact=teams[0].tid) | Q(v_team_id__exact=teams[1].tid)).filter(
        Q(h_team_id__exact=teams[0].tid) | Q(h_team_id__exact=teams[1].tid))
    if nbagames.exists():
        nba_reddit.nba_game_id = nbagames.first().game_id
        nba_reddit.save()
        return True
    return False

def merge_nba_game_with_reddit_thread():
    teams = list(Team.objects.all())
    team_names = (t.name for t in teams)


    c = 0
    b = 0
    for a in NbaReddit.objects.filter(nba_game__isnull=True):
        d = datetime.utcfromtimestamp(a.created)
        selected_teams = getTeams(teams, a.title)
        if len(selected_teams) == 2:
            if not update_nba_reddit(d, selected_teams, a):
                if not update_nba_reddit((d - timedelta(days=1)), selected_teams, a):
                    if not update_nba_reddit((d - timedelta(days=2)), selected_teams, a):
                        if not update_nba_reddit((d + timedelta(days=1)), selected_teams, a):
                            print('No Nba Game Found for day {0}:'.format(d))
                            print(a.title)
                            c = c + 1
        else:
            selected_teams = getTeamsByCity(teams, a.title)
            if len(selected_teams) == 2:
                if not update_nba_reddit(d, selected_teams, a):
                    if not update_nba_reddit((d - timedelta(days=1)), selected_teams, a):
                        if not update_nba_reddit((d - timedelta(days=2)), selected_teams, a):
                            if not update_nba_reddit((d + timedelta(days=1)), selected_teams, a):
                                print('No Nba Game Found for day {0}:'.format(d))
                                print(a.title)
                                c = c + 1
    print(c)
    print(b)
    data = NbaReddit.objects.filter(nba_game__isnull=True)
    data2 = NbaReddit.objects.filter(nba_game__isnull=False)
    print(len(data))
    print(len(data2))
    #print(datetime.utcfromtimestamp(a.created))
    #print(a.title)

