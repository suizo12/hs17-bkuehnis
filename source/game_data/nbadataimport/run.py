import sys
import os
import django
sys.path.append('/Users/benjaminkuehnis/Documents/hsr/hs17/gameprediction/nsfs/nsfs/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'nsfs.settings'
django.setup()
from game_data.nbadataimport.box_score import save_nba_df_to_file
from game_data.reddit.import_reddit_data import import_reddit_game_thread
from schedule.models import NbaReddit
from game_data.nbadataimport.import_nba_data import download_boxscore, import_nba_schedule
from game_data.nbadataimport.predict_rating import predict_rating
from game_data.reddit.transform_reddit import merge_nba_game_with_reddit_thread


import_reddit_game_thread()
nba_reddit = NbaReddit.objects.filter(name__exact='t3_7mzwdw')
print(nba_reddit)

save_nba_df_to_file()
merge_nba_game_with_reddit_thread()
predict_rating()
#import_nba_schedule()
#download_boxscore()