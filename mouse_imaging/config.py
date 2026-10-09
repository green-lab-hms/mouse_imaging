"""
System-specific settings, such as where raw and derived data live.

Settings are read from the first of these that exists:
    1. the file named by the MOUSE_IMAGING_CONFIG environment variable
    2. ~/.config/mouse_imaging/config.toml
    3. DEFAULTS below (the green lab's spinoza paths)
A config file only needs the keys it changes. See config.example.toml in the repository.
"""
import os
import tomllib
from pathlib import Path

ENV_VAR = 'MOUSE_IMAGING_CONFIG'
USER_CONFIG = Path.home() / '.config' / 'mouse_imaging' / 'config.toml'

DEFAULTS = {
    'paths': {
        'raw_root': '/data/green_lab/shared/data/raw', # contains twophoton/, virmen/ and sync/
        'derived_root': '/data/green_lab/shared/data/derived/twophoton', # preprocessing output
    },
}

def config_file():
    """
    Path of the config file in use, or None when using the built-in defaults.
    """
    if os.environ.get(ENV_VAR):
        filename = Path(os.environ[ENV_VAR]).expanduser()
        if not filename.is_file():
            raise FileNotFoundError(f'{ENV_VAR} is set to {filename}, which does not exist.')
        return filename
    if USER_CONFIG.is_file():
        return USER_CONFIG
    return None

def load_config():
    """
    Built-in defaults, updated with the keys set in the config file.
    """
    config = {section: dict(values) for section, values in DEFAULTS.items()}
    filename = config_file()
    if filename is not None:
        with open(filename, 'rb') as fh:
            user_config = tomllib.load(fh)
        for section, values in user_config.items():
            config.setdefault(section, {}).update(values)
    return config

def describe_source():
    filename = config_file()
    return f'config file {filename}' if filename else f'built-in defaults (no {USER_CONFIG} and {ENV_VAR} not set)'

def get_path(key):
    """
    One of the [paths] settings as a Path, with ~ expanded.
    """
    return Path(load_config()['paths'][key]).expanduser()

def check_dir(path, key):
    """
    Raise an error naming the config setting to change if a configured directory doesn't exist.
    """
    if not Path(path).is_dir():
        raise FileNotFoundError(
            f'{key} = {path} does not exist. This was read from {describe_source()}. '
            f'Set [paths] {key} in {USER_CONFIG} or in the file named by {ENV_VAR}; '
            f'see config.example.toml in the mouse_imaging repository.')
