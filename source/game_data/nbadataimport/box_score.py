import pandas as pd
import glob
from schedule.models import NbaGame, WikiHoops, NbaReddit


def save_nba_df_to_file(static_path='../static'):
    """
    Save the boxscores to a file.
    """
    df_f_path = '{0}/dumps/{1}'.format(static_path, 'df_nba_boxscores.csv')
    game_data = []

    for game_file in glob.glob("{0}/nba/boxscore/2015//*.json".format(static_path)):
        game_data.append(BoxScore(game_file).data)

    for game_file in glob.glob("{0}/nba/boxscore/2016//*.json".format(static_path)):
        game_data.append(BoxScore(game_file).data)

    for game_file in glob.glob("{0}/nba/boxscore/2017//*.json".format(static_path)):
        bbb = BoxScore(game_file)
        if bbb.data:
            game_data.append(bbb.data)

    for game_file in glob.glob("{0}/nba/boxscore/2018//*.json".format(static_path)):
        bbb = BoxScore(game_file)
        if bbb.data:
            game_data.append(bbb.data)

    games = list(NbaGame.objects.all().values('game_id', 'date'))
    df_nba_games = pd.DataFrame(games)
    df_nba_games['game_id'] = df_nba_games['game_id'].astype('int')
    df_nba_games['d'] = df_nba_games['date'].str[0:4] + '-' + df_nba_games['date'].str[4:6] + '-' + df_nba_games[
                                                                                                        'date'].str[6:8]
    df_nba_games['d'] = pd.to_datetime(df_nba_games['d'])
    df_nba_games = df_nba_games.set_index('game_id')

    df = pd.DataFrame.from_records(game_data)
    df = df.rename(columns={'gameId': 'game_id'})
    df['game_id'] = df['game_id'].astype('int')
    df = df.set_index('game_id')

    df = pd.concat([df, df_nba_games], axis=1)

    w_df = pd.DataFrame(list(WikiHoops.objects.all().values('game_rating', 'game_star', 'nba_game_id', 'rating_up', 'rating_down', 'rating_percentage')))
    w_df = w_df.rename(columns={'nba_game_id': 'game_id'})
    w_df = w_df.set_index('game_id')

    r_df = pd.DataFrame(list(NbaReddit.objects.filter(nba_game_id__isnull=False).values('nba_game_id', 'ups')))
    r_df = r_df.rename(columns={'nba_game_id': 'game_id'})
    r_df = r_df.set_index('game_id')
    r_df = r_df[~r_df.index.duplicated(keep='first')]

    df = pd.concat([df, r_df], axis=1)
    print(len(df))
    df = pd.concat([df, w_df], axis=1)
    print(len(df))
    for header in list(df):
        try:
            df[header] = df[header].astype(str).astype(float)
        except:
            print(header)
    df.to_csv(df_f_path)


def extract_team_data(dict, team_data, key):
    dict[key + '_fastBreakPoints'] = team_data['fastBreakPoints'] or "0"
    dict[key + '_pointsInPaint'] = team_data['pointsInPaint'] or "0"
    dict[key + '_biggestLead'] = team_data['biggestLead'] or "0"
    dict[key + '_secondChancePoints'] = team_data['secondChancePoints'] or "0"
    dict[key + '_pointsOffTurnovers'] = team_data['pointsOffTurnovers'] or "0"
    dict[key + '_longestRun'] = team_data['longestRun'] or "0"

    totals = team_data['totals']
    dict[key + '_points'] = totals['points']
    dict[key + '_fgm'] = totals['fgm']
    dict[key + '_fga'] = totals['fga']
    dict[key + '_fgp'] = totals['fgp']
    dict[key + '_ftm'] = totals['ftm']
    dict[key + '_fta'] = totals['fta']
    dict[key + '_ftp'] = totals['ftp']
    dict[key + '_tpm'] = totals['tpm']
    dict[key + '_tpa'] = totals['tpa']
    dict[key + '_tpp'] = totals['tpp']
    dict[key + '_offReb'] = totals['offReb']
    dict[key + '_defReb'] = totals['defReb']
    dict[key + '_assists'] = totals['assists']
    dict[key + '_pFouls'] = totals['pFouls']
    dict[key + '_steals'] = totals['steals']
    dict[key + '_turnovers'] = totals['turnovers']
    dict[key + '_blocks'] = totals['blocks']
    dict[key + '_plusMinus'] = totals['plusMinus']
    # dict[key + '_min'] = totals['min']

    leaders = team_data['leaders']
    dict[key + '_max_player_points'] = leaders['points']['value']
    dict[key + '_max_player_rebounds'] = leaders['rebounds']['value']
    dict[key + '_max_player_assists'] = leaders['assists']['value']


def extract_basic_team_data(dict, team_data, key):
    dict[key + '_teamId'] = team_data['teamId']
    dict[key + '_win'] = team_data['win']
    dict[key + '_loss'] = team_data['loss']
    dict[key + '_seriesWin'] = team_data['seriesWin'] or "0"
    dict[key + '_seriesLoss'] = team_data['seriesLoss'] or "0"
    dict[key + '_score'] = team_data['score']
    dict[key + '_linescore_p1'] = team_data['linescore'][0]['score']
    dict[key + '_linescore_p2'] = team_data['linescore'][1]['score']
    dict[key + '_linescore_p3'] = team_data['linescore'][2]['score']
    dict[key + '_linescore_p4'] = team_data['linescore'][3]['score']


class BoxScore:
    def __init__(self, game_path):
        self.data = dict()
        try:
            self.json_data = pd.read_json(game_path)

            basic_game_data = self.json_data['basicGameData']
            self.data['gameId'] = basic_game_data['gameId']

            self.data['isBuzzerBeater'] = int(basic_game_data['isBuzzerBeater'])
            self.data['attendance'] = basic_game_data['attendance']
            #self.data['gameDuration_h'] = basic_game_data['gameDuration']['hours']
            #self.data['gameDuration_min'] = basic_game_data['gameDuration']['minutes']
            period = basic_game_data['period']
            self.data['period_current'] = period['current']
            h_team = basic_game_data['hTeam']
            v_team = basic_game_data['vTeam']
            extract_basic_team_data(self.data, h_team, 'h')
            extract_basic_team_data(self.data, v_team, 'v')

            stats = self.json_data['stats']
            self.data['timesTied'] = stats['timesTied']
            self.data['leadChanges'] = stats['leadChanges']
            v_team_stats = stats['vTeam']
            h_team_stats = stats['hTeam']

            extract_team_data(self.data, v_team_stats, 'v')
            extract_team_data(self.data, h_team_stats, 'h')

        except:
            print(game_path)


