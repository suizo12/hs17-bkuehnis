from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from .models import Schedule, Team, Game, NbaGame
import json
import locale
from datetime import datetime
from django.core.urlresolvers import resolve
from .utils import file_path

from datetime import timedelta

# Create your views here.


def import_team(param):
    t = Team(tid=param['tid'], name=param['tn'], short_name=param['ta'], city=param['tc'])
    t.save()


def nba_today():
    return (datetime.today() - timedelta(days=1)).__format__('%Y-%m-%d')
    #return datetime.today().__format__('%Y-%m-%d')


def import_game(request):
    locale.setlocale(locale.LC_ALL, 'en_US')
    global_file_path = file_path(resolve(request.path).app_name,
                                 '{0}/{1}/{2}'.format('static', 'data', '2015_schedule.json'))
    with open(global_file_path) as data_file:
        data = json.load(data_file)

    for month in data['lscd']:
        for g in month['mscd']['g']:
            if not Team.objects.filter(tid__exact=g['v']['tid']).exists():
                import_team(g['v'])
            if not Team.objects.filter(tid__exact=g['h']['tid']):
                import_team(g['h'])
            if not Game.objects.filter(game_id__exact=g['gid']):
                g = Game(game_id=g['gid'], team_away_id=g['v']['tid'], team_home_id=g['h']['tid'], date=g['gdtutc'],
                         date_uct=datetime.strptime(g['gdtutc'] + g['utctm'], '%Y-%m-%d%H:%M'),
                         game_code=g['gcode'], location=g['ac'], state=g['as'])
                g.save()

        return gameday(request, nba_today())


def gameday(request, param_date):
    db_date = param_date.replace('-', '')
    games = list(NbaGame.objects.filter(date__exact=db_date))
    year = param_date[:4]
    month = param_date[5:7]
    day = param_date[8:10]
    ddate = datetime.strptime(param_date, '%Y-%m-%d')

    return render(request, 'schedule/index.html',
                  dict(games=games, date=ddate.strftime('%A, %d %B %Y'), year=year, month=month, day=day))


def index(request):
    return gameday(request, nba_today())


def date(request):
    return render(request, 'schedule/datetime.html')


def about(request):
    return render(request, 'schedule/about.html')


def schedule(request):
    from schedule.models import Schedule
    schedule = get_object_or_404(Schedule, pk=1)
    template_name = 'schedule.html'
    return render(request, 'schedule/schedule.html', {'schedule': schedule})


## Import Data
def import_schedule(request, year):
    """
    Import the schedule from 'data.nba.com' into the table 'Schedule'
    :param request:
    :param year:
    :return:
    """
    import requests

    import json
    try:
        print(year)
        data = Schedule.objects.get(year__exact=year)
        return HttpResponse(json.dumps(data.schedule_json, sort_keys=True, indent=4), content_type="application/json")
        # j = json.loads(data.schedule_json)
        # return HttpResponse(json.dumps(data.schedule_json, sort_keys = False, indent = 4, separators=(',', ': ')))
    # except ObjectDoesNotExist:
    except Schedule.DoesNotExist:
        url = 'http://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{}/league/00_full_schedule.json'
        r = requests.get(url=url.format(year))
        s = Schedule(year=year, schedule_json=r.json())
        s.save()
        return HttpResponse(r.json())