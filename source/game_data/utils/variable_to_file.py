import pickle

WIKIHOOPS_STAR_RANDOM_FOREST_PREDICTIONS_FILE = 'wikihoops_star_rf_predictions.p'
WIKIHOOPS_RATING_RANDOM_FOREST_PREDICTIONS_FILE = 'wikihoops_rating_rf_predictions.p'

REDDIT_RATING_RANDOM_FOREST_PREDICTIONS_FILE = 'reddit_rating_rf_predictions.p'


def save_variable_to_file(var, filename):
    with open('../static/dumps/{0}'.format(filename), 'wb') as pickle_file:
        pickle.dump(var, pickle_file)


def load_variable_from_file(filename):
    with open('../static/dumps/{0}'.format(filename), 'rb') as pickle_file:
        return pickle.load(pickle_file)