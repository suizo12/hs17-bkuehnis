import pandas as pd


def quarter_diff(home_quarter: int, away_quarter: int) -> int:
    return 1 if abs(home_quarter - away_quarter) < 4 else 0


def quarter_score(home_quarter: int, away_quarter: int) -> int:
    return 1 if home_quarter >= 35 or away_quarter >= 35 else 0


def quarter_sum_score(home_quarter: int, away_quarter: int) -> int:
    return 1 if home_quarter + away_quarter >= 60 else 0


def ts(team_ts: int) -> int:
    return 1 if team_ts >= 50 else 0


def ts(team_ts: int) -> int:
    return 1 if team_ts >= 50 else 0



game_dict = {
"2P": 36.0,
"2P%": 0.5875,
"2PA": 71.0,
"2PAr": 0.8117647058823521,
"3P": 13.0,
"3P%": 0.482758620689655,
"3PA": 33.0,
"3PAr": 0.3918918918918911,
"AST": 29.0,
"AST%": 72.1,
"AST/TOV": 2.8,
"BLK": 9.0,
"BLK%": 14.1,
"DRB": 40.0,
"DRB%": 85.7,
"DRBr": 0.854545454545454,
"DRtg": 121.1,
"FG": 45.0,
"FG%": 0.528571428571428,
"FGA": 93.0,
"FIC": 100.5,
"FT": 26.0,
"FT%": 0.8888888888888881,
"FT/FGA": 0.32051282051282004,
"FTA": 34.0,
"FTAr": 0.4155844155844151,
"FTr": 0.416,
"HOB": 1.7209302325581393,
"ORB": 16.0,
"ORB%": 34.1,
"ORBr": 0.33333333333333304,
"ORtg": 123.3,
"PF": 26.0,
"PTS": 119.0,
"STL": 12.0,
"STL%": 12.1,
"STL/TOV": 1.0,
"TOV": 19.0,
"TOV%": 17.1,
"TRB": 52.0,
"TRB%": 57.3,
"TS%": 0.6262278978388991,
"TSA": 103.8,
"USG%": 100.0,
"eFG%": 0.595238095238095
    # "2P": 40,
    # "2P%": 0.5,
    # "2PA": 40,
    # "2PAr": 1,
    # "3P": 10,
    # "3P%": 0.4,
    # "3PA": 20,
    # "3PAr": 1,
    # "AST": 20,
    # "AST%": 51,
    # "AST/TOV": 0.9,
    # "BLK": 5,
    # "BLK%": 10,
    # "DRB": 40,
    # "DRB%": 85,
    # "DRBr": 0.8,
    # "DRtg": 90,
    # "eFG%": 0.51,
    # "FG": 30,
    # "FG%": 0.51,
    # "FGA": 100,
    # "FIC": 60,
    # "FT": 20,
    # "FT%": 0.9,
    # "FT/FGA": 0.1,
    # "FTA": 25,
    # "FTAr": 0.5,
    # "FTr": 0.5,
    # "HOB": 10,
    # "ORB": 20,
    # "ORB%": 35,
    # "ORBr": 0.4,
    # "ORtg": 120,
    # "PF": 50,
    # "PTS": 120,
    # "STL": 10,
    # "STL%": 8,
    # "STL/TOV": 0.1,
    # "TOV": 5,
    # "TOV%": 5,
    # "TRB": 51,
    # "TRB%": 70,
    # "TS%": 0.51,
    # "TSA": 95,
    # "USG%": 100
}


def zero_is_one(val):
    if val == 0:
        return 1
    return val

class BasketballGame:
    def __init__(self, game_path, wikihoops, use_wikihoops=False):
        self.json_data = pd.read_json(game_path)

        self.away_team = self.json_data['away']['name']
        self.home_team = self.json_data['home']['name']
        self.date = self.json_data['date'][0]
        self.code = self.json_data['code'][0]
        self.code = self.code[0:8] + self.code[9:12]

        self.home_scores = self.json_data['home']['scores']
        self.away_scores = self.json_data['away']['scores']
        self.home_totals = self.json_data['home']['totals']
        self.away_totals = self.json_data['away']['totals']

        self.home_score = int(self.home_scores['T'])
        self.away_score = int(self.away_scores['T'])
        self.total_score = self.home_score + self.away_score
        self.game_type = self.json_data['type']

        self.data = dict()

        for key, value in self.json_data['home']['totals'].items():
            self.data['home_' + key] = value

        for key, value in self.json_data['away']['totals'].items():
            self.data['away_' + key] = value

        self.rating = 0
        if use_wikihoops:
            current_game = wikihoops[wikihoops['code'] == self.code]
            if current_game.size != 0:
                self.rating = zero_is_one(current_game['gameRating'].values[0]) * zero_is_one(current_game['gameStars'].values[0])


            #20170930/DENGSW            DATE/AWAY_HOME
        else:
            # for each overtime add 3
            self.rating += 10 * (self.rating + len(self.home_scores) - 5)
            for key, value in self.home_scores.items():
                if key != 'T':
                    self.rating += quarter_diff(int(value), int(self.away_scores[key]))
                    self.rating += quarter_score(int(value), int(self.away_scores[key]))
                    self.rating += quarter_sum_score(int(value), int(self.away_scores[key]))

            #for key, value in game_dict.items():
            #    self.rating += 1 if value >= self.home_totals[key] else 0
            #    self.rating += 1 if value >= self.away_totals[key] else 0

        self.data['rating'] = self.rating
        self.data['dbGameId'] = self.code

    def __repr__(self):
        return "{} vs {} [{} | {}] game rating: {}".format(self.home_team, self.away_team, self.home_score,
                                                           self.away_score, self.rating)

