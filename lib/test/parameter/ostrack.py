from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.ostrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/ostrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # # for got10k
    # params.checkpoint = os.path.join(save_dir, "OSTrack_got10k_100E.pth.tar") 

    # for trackingnet/lasot/tnl2k...
    #params.checkpoint = os.path.join(save_dir, "OSTrack_alldata_300E.pth.tar")
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/ostrack/%s/OSTrack_ep%04d.pth.tar" % (yaml_name, cfg.TEST.EPOCH))
    print(params.checkpoint)

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
