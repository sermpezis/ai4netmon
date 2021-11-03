import data_analysis_tool as dat

ALL_DATASETS = ['Ripe_Ris_monitors', 'Ripe_Atlas_probes', 'RouteViews_peers', 'Compare_All']

if __name__ == "__main__":

    dat.plot_analysis(ALL_DATASETS)