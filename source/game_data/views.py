from django.http import HttpResponse
from django.core.urlresolvers import resolve
from schedule.utils import file_path
from .nbadataimport.import_nba_data import download_boxscore, import_nba_schedule
from .reddit.import_reddit_data import import_reddit_game_thread
from .wikihoops.import_wikihoop import import_wikihoop_data
from .nbadataimport.predict_rating import predict_rating
from .reddit.transform_reddit import merge_nba_game_with_reddit_thread
from game_data.nbadataimport.box_score import save_nba_df_to_file
# Create your views here.

def create_file(request):

    static_path = file_path(resolve(request.path).app_name, '{0}'.format('static'))
    save_nba_df_to_file(static_path)
    return HttpResponse('file saved!')

def import_data(request):
    """
    Import the data from data.nba.net

    Update schedule and nba_game models with new data.
    :param request:
    :return:
    """
    static_path = file_path(resolve(request.path).app_name, '{0}'.format('static'))

    import_nba_schedule(static_path)
    download_boxscore(static_path)
    import_reddit_game_thread()
    merge_nba_game_with_reddit_thread()
    import_wikihoop_data()
    save_nba_df_to_file(static_path)
    predict_rating(static_path)

    return HttpResponse('imported!')