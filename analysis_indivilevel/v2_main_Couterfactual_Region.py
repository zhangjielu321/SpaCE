import pandas as pd
from v2_utils.generate_counterfactual import generate_CFs
from v2_utils.feature_importance import feature_importance_calculation
import geopandas as gpd
import multiprocessing as mp
import pickle
from tqdm import tqdm

# The outcome (0:expired, 2:survived)
data_concat=pd.read_csv('./v2_data/v2_data_concat.csv')
data_concat_outcome_0 = data_concat[data_concat['outcome']==0]
data_concat_outcome_0.reset_index(drop=True, inplace=True)
df_prototypes=pd.read_csv('./v2_data/data_prototypes.csv')


allcols=['lon', 'lat', 'code_cpr_type', 'code_etiology', 'code_gender',
       'code_race', 'code_location_type', 'code_acuity', 'code_witness',
       'code_cpr_used', 'code_who_cpr', 'code_aed_used', 'code_age',
       'code_duration']

features_not_vary=['code_gender','code_race']

continuous_feature_names=['lon','lat']
categorical_feature_names=['code_cpr_type', 'code_etiology', 'code_gender',
       'code_race', 'code_location_type', 'code_acuity', 'code_witness',
       'code_cpr_used', 'code_who_cpr', 'code_aed_used',  'code_duration','code_age']


# Use Arcgispro to link the gdf_concat_outcome_0 and the county_polygon to get the county information
gdf_concat_outcome_0_county=gpd.read_file('./v2_data/v2_data_concat_outcome_0_county.shp')
gdf_concat_outcome_0_county=gdf_concat_outcome_0_county.drop(columns=['TARGET_FID','geometry'])
gdf_concat_outcome_0_county.columns=['lat', 'lon', 'code_cpr_type', 'code_etiology', 'code_gender',
       'code_race', 'code_location_type', 'code_acuity', 'code_witness',
       'code_cpr_used', 'code_who_cpr', 'code_aed_used', 'code_age',
       'code_duration', 'outcome', 'county_name']



def process_group(name_group_tuple):
    name, group = name_group_tuple
    group=group.drop(columns=['county_name'])
    generate_cfs_total=generate_CFs(query_instance=group, num_cfs=5, df_prototypes=df_prototypes,
                                features_not_vary=features_not_vary, 
                                continuous_feature_names=continuous_feature_names,
                                categorical_feature_names=categorical_feature_names)
    lst_object_cfs=generate_cfs_total.generate_all_counterfactuals()
    importance=feature_importance_calculation(data_all=data_concat_outcome_0,global_importance=True, lst_object_cfs=lst_object_cfs,
                                local_importance=False, object_cf=None, 
                                allcols=allcols, continuous_feature_names=continuous_feature_names, 
                                categorical_feature_names=categorical_feature_names, features_not_vary=features_not_vary)
    return (name, importance.cf_importance())

grouped = list(gdf_concat_outcome_0_county.groupby('county_name'))  # Convert to list for tqdm
region_importance= {}
with mp.Pool(mp.cpu_count()) as pool:
    max_ = len(grouped)
    with tqdm(total=max_) as pbar:
        for name, importance in pool.imap_unordered(process_group, grouped):
            pbar.update()
            region_importance[name] = importance

with open('./v2_output/region_global_importance.pickle', 'wb') as handle:
    pickle.dump(region_importance, handle, protocol=pickle.HIGHEST_PROTOCOL)

