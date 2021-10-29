import data_analysis_tool as dat


ALL_DATASETS1 = ['Ripe_Atlas_probes']
ALL_DATASETS2 = ['Ripe_Ris_monitors']

if __name__ == "__main__":

    dat.plot_analysis(ALL_DATASETS1)
    dat.plot_analysis(ALL_DATASETS2)