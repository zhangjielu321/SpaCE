import numpy as np
from visualize_dataframe import Instances
import pandas as pd

class feature_importance_calculation:
    def __init__(self, data_all, global_importance, lst_object_cfs,
                    local_importance, object_cf, 
                    allcols, continuous_feature_names, categorical_feature_names,
                features_not_vary): 
        self.data_all=data_all
        self.global_importance=global_importance
        self.lst_object_cfs=lst_object_cfs
        self.local_importance=local_importance
        self.object_cf=object_cf
        self.allcols=allcols
        self.continuous_feature_names=continuous_feature_names
        self.categorical_feature_names=categorical_feature_names
        self.features_not_vary=features_not_vary
        
    def cf_importance(self):
        if self.local_importance is True:
            local_importances = {}
            local_impor_continus_pos = {}
            local_impor_continus_neg = {}
            local_impor_categorical_class={}
            

            for col in self.allcols:
                    local_importances[col] = 0
                    
            if self.continuous_feature_names is not None:      
                for col in self.continuous_feature_names:
                        local_impor_continus_pos[col] = 0
                        local_impor_continus_neg[col] = 0
                    
            single_calss=[]
            if self.categorical_feature_names is not None:
                for col in self.categorical_feature_names:
                    for i in self.data_all[col].unique():
                        dict_class_name=col+"_class_"+str(int(i))
                        single_calss.append(dict_class_name)  
                for col_class in single_calss:
                    local_impor_categorical_class[col_class]=0


            # calculate local importance
            org_instance = self.object_cf.query_instance
            df = self.object_cf.df_cf
            
            for _, row in df.iterrows():
                if self.continuous_feature_names is not None:
                    for col in self.continuous_feature_names:
                        if self.features_not_vary is not None:
                            if col not in self.features_not_vary:
                                # if quantile change then we add 1 to the local importance
                                instance_q=(self.data_all[col].values<np.array(org_instance[col])).mean()
                                row_q=(self.data_all[col].values<np.array(row[col])).mean()

                                if round(instance_q,1)!=round(row_q,1):
                                    if local_importances is not None:
                                        local_importances[col] += 1

                                if round(instance_q,1)!=round(row_q,1) and org_instance[col].iat[0]-row[col]<0:
                                    if local_impor_continus_pos is not None:
                                        local_impor_continus_pos[col] += 1

                                if round(instance_q,1)!=round(row_q,1) and org_instance[col].iat[0]-row[col]>0:
                                    if local_impor_continus_neg is not None:
                                        local_impor_continus_neg[col] += 1  

                            if col in self.features_not_vary:
                                if local_importances is not None:
                                    local_importances[col]+=0
                                if local_impor_continus_pos is not None:
                                    local_impor_continus_pos[col] +=0
                                if local_impor_continus_neg is not None:
                                    local_impor_continus_neg[col] +=0
                        if self.features_not_vary is None:
                            instance_q=(self.data_all[col].values<np.array(org_instance[col])).mean()
                            row_q=(self.data_all[col].values<np.array(row[col])).mean()

                            if round(instance_q,1)!=round(row_q,1):
                                if local_importances is not None:
                                    local_importances[col] += 1

                            if round(instance_q,1)!=round(row_q,1) and org_instance[col].iat[0]-row[col]<0:
                                if local_impor_continus_pos is not None:
                                    local_impor_continus_pos[col] += 1

                            if round(instance_q,1)!=round(row_q,1) and org_instance[col].iat[0]-row[col]>0:
                                if local_impor_continus_neg is not None:
                                    local_impor_continus_neg[col] += 1
                            
                                

                if self.categorical_feature_names is not None:
                    for col in self.categorical_feature_names:
                        if org_instance[col].iat[0]!= row[col]:
                            if local_importances is not None:
                                local_importances[col] += 1
                                
                    for col in self.categorical_feature_names:
                        if org_instance[col].iat[0]!= row[col]:
                            if local_impor_categorical_class is not None:
                                local_impor_categorical_class[col+"_class_"+str(int(row[col]))] += 1


            if local_importances is not None:
                for col in self.allcols:
                    local_importances[col] /= len(self.object_cf.df_cf)
                    
            if local_impor_categorical_class is not None:
                for col_class in single_calss:
                    local_impor_categorical_class[col_class]/= len(self.object_cf.df_cf)

            if local_impor_continus_pos is not None:
                for col in self.continuous_feature_names:
                    local_impor_continus_pos[col] /= len(self.object_cf.df_cf)

            if local_impor_continus_neg is not None:
                for col in self.continuous_feature_names:
                    local_impor_continus_neg[col] /= len(self.object_cf.df_cf)
                    

            df_local_importance=pd.DataFrame.from_dict(local_importances, orient='index',
                        columns=['local_importance'])
            df_local_impor_continus_pos=pd.DataFrame.from_dict(local_impor_continus_pos, orient='index',
                        columns=['local_impor_continus_pos'])
            df_local_impor_continus_neg=pd.DataFrame.from_dict(local_impor_continus_neg, orient='index',
                        columns=['local_impor_continus_neg'])
            df_local_impor_categorical_class=pd.DataFrame.from_dict(local_impor_categorical_class, orient='index',
                        columns=['local_impor_categorical'])
            

            instances=Instances(query_instance=self.object_cf.query_instance, df_cf=self.object_cf.df_cf,local_importance=df_local_importance,
                            local_impor_continus_pos=df_local_impor_continus_pos,local_impor_continus_neg=df_local_impor_continus_neg, local_impor_categorical_class=df_local_impor_categorical_class)

            dict_local_importance={"summary_importance:": local_importances, 
                    "summary_importance_continuous_positive":local_impor_continus_pos, 
                    "summary_importance_continuous_negative":local_impor_continus_neg, 
                    "summary_impor_categorical_class":local_impor_categorical_class}
            
            # return instances.display()
            return dict_local_importance


        if self.global_importance is True:
            summary_importance = {}
            summary_impor_continus_pos={}
            summary_impor_continus_neg={}
            summary_impor_categorical_class={}
            
            for col in self.allcols:
                summary_importance[col] = 0
            
            if self.continuous_feature_names is not None:
                for col in self.continuous_feature_names:
                    summary_impor_continus_pos[col] = 0
                    summary_impor_continus_neg[col] = 0
                    
            if self.categorical_feature_names is not None:    
                single_class=[]
                for col in self.categorical_feature_names:
                    for i in self.data_all[col].unique():
                        dict_class_name=col+"_class_"+str(int(i))
                        single_class.append(dict_class_name)
                for col_class in single_class:
                    # print(f"col_classs:{col_class}")
                    summary_impor_categorical_class[col_class]=0
            # print(f"summary_impor_categorical_class:{summary_impor_categorical_class}")

            # calculate the global importance

            for i in self.lst_object_cfs:
                # print(f"feature importance {i}th instance/{len(self.lst_object_cfs)}")
                org_instance = i.query_instance
                df = i.df_cf 
                for _, row in df.iterrows():
                    if self.continuous_feature_names is not None:
                        for col in self.continuous_feature_names:
                            if self.features_not_vary is not None:
                                if col not in self.features_not_vary:
                                    instance_q=(self.data_all[col].values<np.array(org_instance.loc[col])).mean()
                                    row_q=(self.data_all[col].values<np.array(row[col])).mean()
                                    if round(instance_q,1)!=round(row_q,1):
                                        if summary_importance is not None:
                                            summary_importance[col] += 1
                                    if round(instance_q,1)!=round(row_q,1) and org_instance.loc[col].iat[0]-row[col]<0:
                                        if summary_impor_continus_pos is not None:
                                            summary_impor_continus_pos[col] += 1
                                    if round(instance_q,1)!=round(row_q,1) and org_instance.loc[col].iat[0]-row[col]>0:
                                        if summary_impor_continus_neg is not None:
                                            summary_impor_continus_neg[col] += 1
                                if col in self.features_not_vary:
                                    if summary_importance is not None:
                                        summary_importance[col]+=0
                                    if summary_impor_continus_pos is not None:
                                        summary_impor_continus_pos[col] +=0
                                    if summary_impor_continus_neg is not None:
                                        summary_impor_continus_neg[col] +=0
                            if self.features_not_vary is None:
                                instance_q=(self.data_all[col].values<np.array(org_instance.loc[col])).mean()
                                row_q=(self.data_all[col].values<np.array(row[col])).mean()
                                if round(instance_q,1)!=round(row_q,1):
                                    if summary_importance is not None:
                                        summary_importance[col] += 1
                                if round(instance_q,1)!=round(row_q,1) and org_instance.loc[col].iat[0]-row[col]<0:
                                    if summary_impor_continus_pos is not None:
                                        summary_impor_continus_pos[col] += 1
                                if round(instance_q,1)!=round(row_q,1) and org_instance.loc[col].iat[0]-row[col]>0:
                                    if summary_impor_continus_neg is not None:
                                        summary_impor_continus_neg[col] += 1
                                

                    if self.categorical_feature_names is not None:        
                        for col in self.categorical_feature_names:
                            if org_instance.loc[col].iat[0]!= row[col]:
                                if summary_importance is not None:
                                    summary_importance[col] += 1
                                    
                        for col in self.categorical_feature_names:
                            if org_instance.loc[col].iat[0]!= row[col]:
                                if summary_impor_categorical_class is not None:
                                    summary_impor_categorical_class[col+"_class_"+str(int(row[col]))] += 1
            
            overall_num_cfs = sum([i.df_cf.shape[0] for i in self.lst_object_cfs])
            
            
            if summary_importance is not None:
                for col in self.allcols:
                    if overall_num_cfs > 0:
                        summary_importance[col] /= overall_num_cfs
                        
            if summary_impor_categorical_class is not None:
                if self.categorical_feature_names is not None:
                    single_class=[]
                    for col in self.categorical_feature_names:
                        for i in self.data_all[col].unique():
                            dict_class_name=col+"_class_"+str(int(i))
                            single_class.append(dict_class_name)
                    for col_class in single_class:
                        if overall_num_cfs > 0:
                            summary_impor_categorical_class[col_class] /= overall_num_cfs
                    
            if summary_impor_continus_pos is not None:
                if self.continuous_feature_names is not None:
                    for col in self.continuous_feature_names:
                        if overall_num_cfs > 0:
                            summary_impor_continus_pos[col] /= overall_num_cfs
            if summary_impor_continus_neg is not None:
                if self.continuous_feature_names is not None:
                    for col in self.continuous_feature_names:
                        if overall_num_cfs > 0:
                            summary_impor_continus_neg[col] /= overall_num_cfs
            
            
            dict_global_importance={"summary_importance:": summary_importance, 
                    "summary_importance_continuous_positive":summary_impor_continus_pos, 
                    "summary_importance_continuous_negative":summary_impor_continus_neg, 
                    "summary_impor_categorical_class":summary_impor_categorical_class}
            

            return dict_global_importance

