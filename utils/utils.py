import os


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
