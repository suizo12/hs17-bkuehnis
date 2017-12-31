import os
import logging.config
import json
from argparse import ArgumentParser

from game_data.dataimport.utils import get_seasons
from game_data.dataimport.nba import NbaBRefSeason
from game_data.dataimport.constants import MATCHES_DIR

with open('logging.json', 'r') as f:
    logging.config.dictConfig(json.load(f))
logger = logging.getLogger('stringer-bell')


def main(league, seasons):
    seasons = get_seasons(seasons)
    for season in seasons:
        path = '{0}/{1}/{2}/{3}'.format(MATCHES_DIR, 'united_states', 'nba', season)
        if not os.path.exists(path):
            os.makedirs(path)
        logger.info('Crawling season {0}'.format(season))
        b_ref = NbaBRefSeason('united_states', league, season)
        b_ref.crawl_season()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--league', default='nba')
    parser.add_argument('--seasons', nargs='+', default=['2017-2018'])
    parser.add_argument('--date', default='10')
    args = parser.parse_args()
    main(args.league, args.seasons)
