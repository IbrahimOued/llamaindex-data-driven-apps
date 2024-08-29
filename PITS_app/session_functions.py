# contains functions that handle the saving, loading, and
# deleting of a user’s session state:

from global_settings import SESSION_FILE
import yaml
import os

def save_session(state):
    sate_to_save = {key: value for key, value in state.items()}
    with open(SESSION_FILE, "w") as file:
        yaml.dump(sate_to_save, file)

def load_session(state):
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as file:
            try:
                loaded_state = yaml.load(file, Loader=yaml.FullLoader)
                for key, value in loaded_state.items():
                    state[key] = value
                return True
            except yaml.YAMLError:
                return False
    return False

def delete_session(state):
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    for key in list(state.keys()):
        del state[key]