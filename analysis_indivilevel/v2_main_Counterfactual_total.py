
import pandas as pd
from v2_utils.generate_counterfactual import generate_CFs
from v2_utils.feature_importance import feature_importance_calculation
# The outcome (0:expired, 2:survived)
data_concat=pd.read_csv('./v2_data/v2_data_concat.csv')
data_concat_outcome_0 = data_concat[data_concat['outcome']==0]
data_concat_outcome_0.reset_index(drop=True, inplace=True)
df_prototypes=pd.read_csv('./v2_data/data_prototypes.csv')

#Generate counterfactuals for the query instance
allcols=['lon', 'lat', 'code_cpr_type', 'code_etiology', 'code_gender',
       'code_race', 'code_location_type', 'code_acuity', 'code_witness',
       'code_cpr_used', 'code_who_cpr', 'code_aed_used', 'code_age',
       'code_duration']
features_not_vary=['code_gender','code_race']
continuous_feature_names=['lon','lat']
categorical_feature_names=['code_cpr_type', 'code_etiology', 'code_gender',
       'code_race', 'code_location_type', 'code_acuity', 'code_witness',
       'code_cpr_used', 'code_who_cpr', 'code_aed_used',  'code_duration','code_age']


# Featurs_not_vary was set.
generate_cfs_total=generate_CFs(query_instance=data_concat_outcome_0, num_cfs=5, df_prototypes=df_prototypes,
                                features_not_vary=features_not_vary, 
                                continuous_feature_names=continuous_feature_names,
                                categorical_feature_names=categorical_feature_names)
lst_object_cfs=generate_cfs_total.generate_all_counterfactuals()

# with open('v2_output/total_counterfactuals.pkl', 'wb') as f:
#     pickle.dump(lst_object_cfs, f)
print("----------------------counterfactual finished----------------------------")


importance=feature_importance_calculation(data_all=data_concat_outcome_0,global_importance=True, lst_object_cfs=lst_object_cfs,
                            local_importance=False, object_cf=None, 
                            allcols=allcols, continuous_feature_names=continuous_feature_names, 
                            categorical_feature_names=categorical_feature_names, features_not_vary=features_not_vary)
global_importance=importance.cf_importance()
# with open('v2_output/total_global_importance.pkl', 'wb') as f:
#     pickle.dump(global_importance, f)
print("----------------------global importance finished--------------------------")
