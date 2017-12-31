from nsfs.settings import BASE_DIR
import os


def file_path(app_name, static_file_path):
    """
    search for a file in statics
    :param app_name:
    :param static_file_path:
    :return: file path
    """
    local_file_path = '{0}/{1}'.format(app_name, static_file_path)
    return os.path.join(BASE_DIR, local_file_path)