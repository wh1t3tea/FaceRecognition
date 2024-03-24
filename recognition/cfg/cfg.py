import json
import os.path as osp
import os
import argparse

def load_cfg(cfg):
    cfg = cfg.config
    with open(cfg) as cfg_json:
        cfg = json.load(cfg_json)
        return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load your config")
    parser.add_argument("config",
                        type=str,
                        default='recognition/cfg/config.json',
                        help='path to your config')
    load_cfg(parser.parse_args())