"""
This script is used to load app's config contents which is in yaml format.
"""
import yaml
import pathlib


CFGFILE = pathlib.Path("./src/config.yaml")


def load_config():
    try:
        # using 'rb' mode to avoid UnicodeDecodeError
        # such as 'gbk' codec can't decode 中文字符
        with CFGFILE.open("rb") as cfg:
            data = yaml.safe_load(cfg.read())
    except Exception as e:
        raise e

    return data
