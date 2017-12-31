"""
Imports data from data.nba.net
see https://github.com/kshvmdn/nba.js/blob/master/docs/api/DATA.md
"""
import json


def transform_date():
    from schedule.models import NbaGame
    for g in NbaGame.objects.all():
        print(g.start_time_utc)
        print(g.start_time_utc.strftime('%Y%m%d'))
        print(g.game_code_url[:8])
        g.date = g.game_code_url[:8]
        g.save()


def import_nba_schedule(static_path = '../static'):
    from schedule.models import NbaGame
    # with open('{0}/nba/schedule/nba_schedule_2014.json'.format(static_path), 'rb') as nba_schedule_file:
    # with open('{0}/nba/schedule/nba_schedule_2015.json'.format(static_path), 'rb') as nba_schedule_file:
    # with open('{0}/nba/schedule/nba_schedule_2016.json'.format(static_path), 'rb') as nba_schedule_file:
    with open('{0}/nba/schedule/nba_schedule_2017.json'.format(static_path), 'rb') as nba_schedule_file:
        schedule = json.load(nba_schedule_file)

    for game in schedule['league']['standard']:
        if not NbaGame.objects.filter(game_id=game['gameId']).exists():
            # print('----------')
            nba_game = NbaGame(game_id=game['gameId'])
            nba_game.game_code_url = game['gameUrlCode']
            nba_game.h_team_id = game['hTeam']['teamId']
            nba_game.v_team_id = game['vTeam']['teamId']

            # print(game["gameId"])
            # print(game["gameUrlCode"])
            # print(game["hTeam"]['teamId'])

            if 'score' in game:
                nba_game.v_team = game['vTeam']['score']
                nba_game.v_win = game['vTeam']['win']
                nba_game.v_loss = game['vTeam']['loss']
                nba_game.v_team = game['hTeam']['score']
                nba_game.v_win = game['hTeam']['win']
                nba_game.v_loss = game['hTeam']['loss']
                # print(game["vTeam"]['score'])
                # print(game["hTeam"]['score'])
                # print(game["hTeam"]['win'])
                # print(game["hTeam"]['loss'])
                # print(game["vTeam"]['win'])
                # print(game["vTeam"]['loss'])

            if 'startTimeUTC' in game:
                nba_game.start_time_utc = game['startTimeUTC']
                # print(game['startTimeUTC'])
            else:
                d = game['formatted']['startDate']
                t = game['formatted']['startTimeV2']
                nba_game.start_time_utc = d[0:4] + '-' + d[4:6] + '-' + d[6:8] + ' ' + t[0:2] + ':' + t[2:4]
                # 'startDate' (4586548464) '1930' 'startTimeV2' (4586548592)

            nba_game.date = nba_game.game_code_url[:8]

            # print('start: ' + nba_game.start_time_utc)
            nba_game.season_stage_id = game['seasonStageId']

            # print(game['seasonStageId'])
            if 'isBuzzerBeater' in game:
                nba_game.is_buzzer_beater = game['isBuzzerBeater']
                # print(game['isBuzzerBeater'])

            nba_game.save()


def download_boxscore(static_path = '../static'):

    # http://data.nba.net/data/10s/prod/v1/20151005/0011500007_boxscore.json
    from schedule.models import NbaGame
    import datetime
    from datetime import timedelta
    # g = NbaGame.objects.first()
    for game in NbaGame.objects.filter(downloaded_box_score__exact=False):

        if game.start_time_utc.date() <= datetime.date.today():
            try:
                download_json(static_path, game)
            except:
                print(game.date)


def download_json(static_path, game):
    import urllib.request
    import os.path

    file_path = '{0}/nba/boxscore/{1}/{2}.json'.format(static_path, game.start_time_utc.strftime('%Y'), game.game_id)
    if not os.path.isfile(file_path):
        url = 'http://data.nba.net/data/10s/prod/v1/{0}/{1}_boxscore.json'.format(game.date,
                                                                                  game.game_id)
        with urllib.request.urlopen(url) as url:
            data = json.loads(url.read().decode())
            if data['basicGameData']['statusNum'] == 3:
                with open(file_path, "w") as f:
                    json.dump(data, f)

                game.downloaded_box_score = True
                game.save()

    else:
        game.downloaded_box_score = True
        game.save()
