from schedule.models import WikiHoops, NbaGame

from bs4 import BeautifulSoup

from urllib.request import Request, urlopen
from urllib.error import URLError
import pandas as pd
from datetime import date
from datetime import timedelta
from time import sleep
import sys

def is_digit(n):
    try:
        int(n)
        return True
    except ValueError:
        return False


def import_wikihoop_data():

    from_date = date.today() - timedelta(weeks=1)
    
    #2012-10-30 - 2013-06-21
    #2013-10-29 - 2014-06-15
    #2014-10-28 - 2015-06-16
    #2015-10-27 - 2016-06-19
    #2016-10-25 - 2017-06-12
    #2017-10-10 - 2018-01-01
    
    data_list = []
    while date.today() >= from_date:
    #while c < 3:

        sleep(0.1)
        url = 'https://wikihoops.com/games/{}/'.format(from_date.isoformat())
        #url = 'https://wikihoops.com/games/2017-01-07'
        from_date = from_date + timedelta(days=1)
    
        req = Request(url)
        try:
            response = urlopen(req)
        except URLError as e:
            if hasattr(e, 'reason'):
                print('We failed to reach a server.')
                print('Reason: ', e.reason)
            elif hasattr(e, 'code'):
                print('The server couldn\'t fulfill the request.')
                print('Error code: ', e.code)
        else:
            # everything is fine
            soup = BeautifulSoup(response.read(), 'html.parser')
            games = soup.find_all(class_='game')

    
            for game in games:
                try:
                    data = {}
                    awayTeam = game.find(itemprop='awayTeam')
                    homeTeam = game.find(itemprop='homeTeam')
                    data['gameId'] = game['data-gameid']
                    data['gameDate'] = '{0}-{1}-{2}'.format(data['gameId'][0:4], data['gameId'][4:6], data['gameId'][6:8])
                    data['awayTeamId'] = awayTeam['data-teamid']
                    data['awayTeamScore'] = awayTeam['data-teamscore']
                    data['homeTeamId'] = homeTeam['data-teamid']
                    data['homeTeamScore'] = homeTeam['data-teamscore']
                    data['gameStars'] = game.find(itemprop='ratingValue').get_text()
                    data['gameRating'] = game.find('span', attrs={'data-count': True})['data-count']
                    data['ratingPercentage'] = game.find('span', attrs={'data-count': True}).find('span', class_='ratingLabel').string
                    data['ratingUp'] = game.find('a', attrs={'data-vote': 'up'})['data-count']
                    data['ratingDown'] = game.find('a', attrs={'data-vote': 'down'})['data-count']                    
                    data_list.append(data)
                except:
                    print('error processing', sys.exc_info()[0])
    if data_list:
        df = pd.DataFrame.from_records(data_list)

        df['code'] = df['gameId'].apply(lambda x: x[0:8] + x[11:14])
        df['dbGameId'] = df['gameId'].apply(lambda x: x[0:8] + '/' + x[8:14])

        db_game_ids = df['dbGameId'].tolist()
        for game_id in db_game_ids:
            #print(id)
            nba_game = NbaGame.objects.filter(game_code_url__exact=game_id)
            if nba_game:

                try:
                    current_game = df[df['dbGameId'] == game_id]
                    percentage = current_game['ratingPercentage'].values[0]
                    percentage = percentage.replace("%", "")
                    if isinstance(int(current_game['gameStars'].values[0]), int):

                        db_wikihoop = WikiHoops.objects.filter(w_game_id__exact=current_game['gameId'].values[0])
                        if db_wikihoop.exists():
                            db_wikihoop = db_wikihoop.first()
                            db_wikihoop.rating_up = current_game['ratingUp'].values[0]
                            db_wikihoop.rating_down = current_game['ratingDown'].values[0]
                            db_wikihoop.rating_percentage = percentage
                            db_wikihoop.save()
                        else:
                            db_wikihoop = WikiHoops(w_game_id=current_game['gameId'].values[0],
                                                    game_rating=current_game['gameRating'].values[0],
                                                    game_star=current_game['gameStars'].values[0],
                                                    rating_up=current_game['ratingUp'].values[0],
                                                    rating_down=current_game['ratingDown'].values[0],
                                                    rating_percentage=percentage,
                                                    )
                            db_wikihoop.nba_game_id = nba_game[0].game_id
                            db_wikihoop.save()
                    else:
                        print('not saved')
                        print(current_game['ratingPercentage'].values[0])
                        print(isinstance(percentage, int))

                except:
                    print("Unexpected error:", sys.exc_info())