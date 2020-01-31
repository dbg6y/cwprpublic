#!/miniconda3/envs/py3_env/bin/python


### Create json parameter files for simulations


## Import Modules
import numpy as np
import itertools
import json


## Create template parameter dictionary
para_temp = {'CLIM': {'ARMN_CLIM': 0.012, 'ARCV_CLIM': 0.20, 'ADMN_CLIM': 0.012, 
                      'ALDF_CLIM': 0.009, 'LRMN_CLIM': 0.5, 'LRCV_CLIM': 0.20, 
                      'LDMN_CLIM': 0.10, 'ETMX_CLIM': 0.005, 'EWLT_CLIM': 0.0001, 
                      'TSEA_CLIM': 110}, 
             'CATC': {'CAMN_CATC': '1 * 1000**2', 'CACV_CATC': 0.20, 
                      'CTMU_CATC': '24 / 24 / 1000**0.76', 'CTUV_CATC': '0.75 * 60 * 60 * 24', 
                      'CTWC_CATC': '(3 / 2) * CTUV_CATC', 'CTDC_CATC': '1000 * 60 * 60 * 24', 
                      'BFPR_CATC': 0.5, 'APFC_CATC': 100, 'FLMN_CATC': '24 * 60 * 60 * 0.00001'}, 
             'SOIL': {'KSAT_SOIL': 0.8, 'PORO_SOIL': 0.43, 'BETA_SOIL': 13.8, 
                      'SHYG_SOIL': 0.14, 'SWLT_SOIL': 0.18, 'SSTR_SOIL': 0.46 , 
                      'SFLD_SOIL': 0.56}, 
             'CROP': {'ZRCT_CROP': 0.5, 'ZRLC_CROP': 0.5, 'YMAX_CROP': 0.3, 
                      'QPAR_CROP': 2, 'KPMN_CROP': 0.5, 'KPCV_CROP': 0.20, 
                      'RPAR_CROP': 0.5, 'COST_CROP': 0.2, 'TGRW_CROP': 110},  
             'CWPR': {'NMMN_CWPR': 150, 'NMCV_CWPR': 0.2, 'BYVL_CWPR': 'NA', 
                      'CAMO_CWPR': 10, 'PFMX_CWPR': 'NA', 'RDYS_CWPR': 'NA', 
                      'MBLM_CWPR': 3, 'COST_CWPR': 'NA', 'STOR_CWPR': 50, 
                      'FCCP_CWPR': 'NA', 'FLMN_CWPR': '24 * 60 * 60 * 0.0001'},
             'MEMB': {'HAMN_MEMB': 5000, 'HACV_MEMB': 0.20,'HPMN_MEMB': 1, 
                      'HPCV_MEMB': 0.20, 'HSMN_MEMB': 5, 'HSCV_MEMB': 0.20, 
                      'CMIN_MEMB': 200, 'SMNC_MEMB': 'SSTR_SOIL', 'SMND_MEMB': 'SFLD_SOIL', 
                      'SMXC_MEMB': 'SFLD_SOIL', 'SMXD_MEMB': 'SFLD_SOIL'}, 
             'SIMU': {'NMBR_SIMU': 'NA', 'NRUN_SIMU': 10000, 'BUFF_SIMU' : 'int(0.3 * TGRW_CROP)', 
                      'INVL_SIMU': 24},
             'DBSE_FILE': 'results/cwprresult.db'}


## Define simulation-specific parameters
rdys_cwpr = [1, 2]
cost_cwpr = [0.01, 0.03]
pfmx_cwpr = [40]
byvl_cwpr = ['False', 'True']
fccp_cwpr = [0.0, 0.5, 1.0]
para_simu = list(itertools.product(rdys_cwpr, cost_cwpr, pfmx_cwpr, byvl_cwpr, fccp_cwpr))


## Loop through simulations
for indx, para in enumerate(para_simu):
    
    # Define new dictionary
    para_data = para_temp
    
    # Fill in simulation-specific parameters
    para_data['SIMU']['NMBR_SIMU'] = indx
    para_data['CWPR']['RDYS_CWPR'] = para[0]
    para_data['CWPR']['COST_CWPR'] = para[1]
    para_data['CWPR']['PFMX_CWPR'] = para[2]
    para_data['CWPR']['BYVL_CWPR'] = para[3]
    para_data['CWPR']['FCCP_CWPR'] = para[4]
    
    # Save as json file
    with open('cwprparam_' + ('%02d' % indx) + '.json', 'w') as para_file:
        json.dump(para_data, para_file, indent = 2)
