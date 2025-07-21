from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/root/autodl-tmp/got10k_lmdb'
    settings.got10k_path = '/root/autodl-tmp/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/root/autodl-tmp/itb'
    settings.lasot_extension_subset_path_path = '/root/autodl-tmp/lasot_extension_subset'
    settings.lasot_lmdb_path = '/root/autodl-tmp/lasot_lmdb'
    settings.lasot_path = '/root/autodl-tmp/lasot'
    settings.network_path = '/root/autodl-tmp/HGETtrack-main/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/root/autodl-tmp/nfs'
    settings.otb_path = '/root/autodl-tmp/otb'
    settings.prj_dir = '/root/autodl-tmp/HGETtrack-main'
    settings.result_plot_path = '/root/autodl-tmp/HGETtrack-main/output/test/result_plots'
    settings.results_path = '/root/autodl-tmp/HGETtrack-main/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/root/autodl-tmp/HGETtrack-main/output'
    settings.segmentation_path = '/root/autodl-tmp/HGETtrack-main/output/test/segmentation_results'
    settings.tc128_path = '/root/autodl-tmp/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/root/autodl-tmp/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/autodl-tmp/trackingnet'
    settings.uav_path = '/root/autodl-tmp/uav'
    settings.vot18_path = '/root/autodl-tmp/vot2018'
    settings.vot22_path = '/root/autodl-tmp/vot2022'
    settings.vot_path = '/root/autodl-tmp/VOT2019'
    settings.youtubevos_dir = ''
    settings.hot_dir = '/root/autodl-tmp/challenge2023/test'
    return settings

