def remove_temporary_colums(data_frame):
    data_frame.drop(['dbGameId'], axis=1, inplace=True)
    data_frame.drop(['rating'], axis=1, inplace=True)

def get_results(data_frame):
    return data_frame['rating']