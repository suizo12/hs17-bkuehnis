#https://wikihoops.com/games/2012-04-28/
import sys, os, django
sys.path.append('/Users/benjaminkuehnis/Documents/hsr/hs17/gameprediction/nsfs/nsfs/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'nsfs.settings'
django.setup()
from game_data.gamescore.wikihoops import Wikihoops
from schedule.models import NbaGame, WikiHoops as W
from game_data.wikihoops.import_wikihoop import import_wikihoop_data

def add_wikihoops_data(static_path = '../static'):
    """
    import data from csv file
    :param static_path:
    :return:
    """
    wikihoop_panda = Wikihoops(static_path)
    # print(wikihoop_panda.frame)



    #print(wikihoop_panda.frame[wikihoop_panda.frame['gameStars'] == '–'])
    print(len(wikihoop_panda.frame))
    wikihoop_panda.frame.drop(wikihoop_panda.frame[wikihoop_panda.frame['ratingPercentage'] == '–'].index, inplace=True)
    wikihoop_panda.frame.drop(wikihoop_panda.frame[wikihoop_panda.frame['gameStars'] == '–'].index, inplace=True)
    wikihoop_panda.frame.drop(wikihoop_panda.frame[wikihoop_panda.frame['gameStars'] == '–'].index, inplace=True)
    print(len(wikihoop_panda.frame))

    print(wikihoop_panda.frame[wikihoop_panda.frame['gameStars'] == '–'])
    dbGameIds = wikihoop_panda.frame['dbGameId'].tolist()
    for id in dbGameIds:
        #print(id)
        nba_game = NbaGame.objects.filter(game_code_url__exact=id)
        if nba_game:

            try:
                current_game = wikihoop_panda.frame[wikihoop_panda.frame['dbGameId'] == id]
                percentage = current_game['ratingPercentage'].values[0]
                percentage = percentage.replace("%", "")
                if isinstance(int(percentage), int) and isinstance(int(current_game['gameStars'].values[0]), int):

                    db_wikihoop = W.objects.filter(w_game_id__exact=current_game['gameId'].values[0])
                    if db_wikihoop.exists():
                        db_wikihoop = db_wikihoop.first()
                        db_wikihoop.rating_up = current_game['ratingUp'].values[0]
                        db_wikihoop.rating_down = current_game['ratingDown'].values[0]
                        db_wikihoop.rating_percentage = percentage
                        db_wikihoop.save()
                    else:
                        db_wikihoop = W(w_game_id=current_game['gameId'].values[0],
                                        game_rating=current_game['gameRating'].values[0],
                                        game_star=current_game['gameStars'].values[0],
                                        rating_up=current_game['ratingUp'].values[0],
                                        rating_down=current_game['ratingDown'].values[0],
                                        rating_percentage=percentage,
                                        )
                        db_wikihoop.nba_game_id = nba_game[0].game_id
                        print(db_wikihoop)
                        db_wikihoop.save()
                else:
                    print('not saved')
                    print(current_game['ratingPercentage'].values[0])
                    print(isinstance(percentage, int))

            except:
                print("Unexpected error:", sys.exc_info()[0])


import_wikihoop_data()
#add_wikihoops_data()