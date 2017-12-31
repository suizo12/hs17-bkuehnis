from django.test import TestCase
from .models import Team, Game
import json
import locale
from django.contrib.staticfiles import finders
from datetime import datetime


# Create your tests here.


def import_team(param):
    t = Team(tid=param['tid'], name=param['tn'], short_name=param['ta'], city=param['tc'])
    t.save()


class ImportDataTest(TestCase):
    def import_json_season_2015_2016_2017(self):
        """
        Import the data from static/data/*.json into the db.
        :return:
        """

        result = finders.find('2017_schedule.json')
        locale.setlocale(locale.LC_ALL, 'en_US')

        print(result)
        with open(result) as data_file:
            data = json.load(data_file)

        for month in data['lscd']:
            for g in month['mscd']['g']:
                if not Team.objects.exists(tid__exact=g['v']['tid']):
                    import_team(g['v'])
                if not Team.objects.exists(tid__exact=g['h']['tid']):
                    import_team(g['h'])
                if not Game.objects.exists(game_id__exact=g['gid']):
                    g = Game(game_id=g['gid'], team_away_id=g['v']['tid'], team_home_id=g['h']['tid'], date=g['gdtutc'],
                             date_uct=datetime.strptime(g['gdtutc'] + g['utctm'], '%Y-%m-%d%H:%M'),
                             game_code=g['gcode'], location=g['ac'], state=g['as'])
                    g.save()
