import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from visualize_dataframe import Instances
from sklearn.preprocessing import MinMaxScaler

class generate_CFs:
    def __init__(self, query_instance, num_cfs, df_prototypes, features_not_vary, continuous_feature_names, categorical_feature_names):
        self.query_instance=query_instance
        self.num_cfs=num_cfs
        self.df_prototypes=df_prototypes
        self.features_not_vary=features_not_vary
        self.continuous_feature_names=continuous_feature_names
        self.categorical_feature_names=categorical_feature_names

    def build_tree(self, T, n):
        tree = KDTree(T, leaf_size=2)
        dist, ind = tree.query(np.reshape(T[0,:],(1, len(self.query_instance.columns))), k=n)
        return dist, ind
    
    def comparison_vlaue_not_vary(self, df_tree, features_not_vary, index_neighbors):
        #  get the tolerance range of a variable which is considered as not varied features.
        #  when we set a cetain feature not to vary when getting counterfactuals,
        #  the value of query instance on that varaible: 56.5
        #  then the value of cfs on that varaible can be from min(50) to max(60)
        #  Simialry, if the value of query instance is 5.65, then the cfs is from min(5) to max（6）
        list_int=np.round(df_tree.iloc[0][features_not_vary].values.astype(int)).tolist()
        list_string=list(map(str, list_int))
        zip_element1=[int(str(i)[0]) for i in list_string]
        zip_element2=[10**(len(i)-1) for i in list_string]
        value_0_min=[a*b for a,b in zip(zip_element1,zip_element2)]
        value_0_max=[sum(x) for x in zip(value_0_min, zip_element2)]

        value_i=df_tree.iloc[index_neighbors][features_not_vary]
        bool_result=all(va_min<=va_i<=va_max for va_min, va_i,va_max in zip(value_0_min,value_i,value_0_max))
        return bool_result
        # df_tree.iloc[0][features_not_vary], value_i, value_0_min,value_0_max
    
    def generate_counterfactuals(self):
        lst_cf_ind=[]
        num_neighbors=self.num_cfs-1
        for __, row in self.query_instance.iterrows():
            while (len(lst_cf_ind))<self.num_cfs:
                num_neighbors=num_neighbors+1
                lst_cf_ind=[]
                
                # normalize the query instance(row) and prototypes
                T=np.row_stack([row.array,self.df_prototypes.to_numpy()])
                df_T_origin=pd.DataFrame(T, columns=self.query_instance.columns)
                # print(f"original df_T:{df_T_origin.head(5)}")
                outcome_column = df_T_origin['outcome']
                df_T = df_T_origin.drop(columns=['outcome'])
                scaler = MinMaxScaler()
                df_T = pd.DataFrame(scaler.fit_transform(df_T), columns=df_T.columns)
                df_T['outcome'] = outcome_column
                # print(f"normalized df_T:{df_T.head(5)}")
                
                if num_neighbors<round(len(df_T)/2):
                    __, ind=self.build_tree(T, num_neighbors)
                    for i in list(ind.ravel()):
                        if self.features_not_vary is not None and self.continuous_feature_names is not None and all(item in self.continuous_feature_names for item in self.features_not_vary)==True:
                            bool_result=self.comparison_vlaue_not_vary(df_T, self.features_not_vary, i)
                            if df_T.loc[0, 'outcome'] != df_T.loc[i, 'outcome'] and bool_result==True:
                                lst_cf_ind.append(i)
                        if self.features_not_vary is not None and self.categorical_feature_names is not None and all(item in self.categorical_feature_names for item in self.features_not_vary)==True:
                            if df_T.loc[0, 'outcome'] != df_T.loc[i, 'outcome'] and all(df_T.iloc[0][self.features_not_vary]==df_T.iloc[i][self.features_not_vary]) is True:
                                lst_cf_ind.append(i)
                        if self.features_not_vary is None:
                            if df_T.loc[0, 'outcome']!= df_T.loc[i, 'outcome']:
                                lst_cf_ind.append(i)
                        
                        
                elif num_neighbors==round(len(df_T)/2):
                    __, ind=self.build_tree(T, num_neighbors)
                    for i in list(ind.ravel()):
                        if self.features_not_vary is not None and self.continuous_feature_names is not None and all(item in self.continuous_feature_names for item in self.features_not_vary)==True:
                            bool_result=self.comparison_vlaue_not_vary(df_T, self.features_not_vary, i)
                            if df_T.loc[0, 'outcome'] != df_T.loc[i, 'outcome'] and bool_result==True:
                                lst_cf_ind.append(i) 
                        if self.features_not_vary is not None and self.categorical_feature_names is not None and all(item in self.categorical_feature_names for item in self.features_not_vary)==True:
                            if df_T.loc[0, 'outcome'] != df_T.loc[i, 'outcome'] and all(df_T.iloc[0][self.features_not_vary]==df_T.iloc[i][self.features_not_vary]) is True:
                                lst_cf_ind.append(i)
                        if self.features_not_vary is None:
                            if df_T.loc[0, 'outcome'] != df_T.loc[i, 'outcome']:
                                lst_cf_ind.append(i)
                    break
        df_cf=df_T_origin.loc[df_T_origin.index[lst_cf_ind]]  
        object_cf=Instances(self.query_instance, df_cf, local_importance=None, local_impor_continus_pos=None,local_impor_continus_neg=None, local_impor_categorical_class=None)
        return object_cf
    
    def generate_all_counterfactuals(self):
        all_object_cf=[]
        lst_cf_ind=[]
        num_neighbors=self.num_cfs-1
        for index, row in self.query_instance.iterrows(): 
            # print(f"generate counterfactual index:{index}/{len(self.query_instance)}")
            while (len(lst_cf_ind))<self.num_cfs:
                # print("1")
                num_neighbors=num_neighbors+1
                lst_cf_ind=[]
                # normalize the query instance(row) and prototypes
                T=np.row_stack([row.array,self.df_prototypes.to_numpy()])
                df_T_origin=pd.DataFrame(T, columns=self.query_instance.columns)
                # print(f"original df_T:{df_T_origin.head(5)}")
                outcome_column = df_T_origin['outcome']
                df_T = df_T_origin.drop(columns=['outcome'])
                scaler = MinMaxScaler()
                df_T = pd.DataFrame(scaler.fit_transform(df_T), columns=df_T.columns)
                df_T['outcome'] = outcome_column
                # print(f"normalized df_T:{df_T.head(5)}")
                
                if num_neighbors<round(len(df_T)/2):
                    # print("2-1")
                    __, ind=self.build_tree(T, num_neighbors)
                    for i in list(ind.ravel()):
                        if self.features_not_vary is not None and self.continuous_feature_names is not None and all(item in self.continuous_feature_names for item in self.features_not_vary)==True:
                            bool_result=self.comparison_vlaue_not_vary(df_T, self.features_not_vary, i)
                            if df_T.loc[0, 'outcome']!= df_T.loc[i, 'outcome'] and bool_result==True:
                                lst_cf_ind.append(i)
                        if self.features_not_vary is not None and self.categorical_feature_names is not None and all(item in self.categorical_feature_names for item in self.features_not_vary)==True:
                            if df_T.loc[0, 'outcome']!= df_T.loc[i, 'outcome'] and all(df_T.iloc[0][self.features_not_vary]==df_T.iloc[i][self.features_not_vary]) is True:
                                lst_cf_ind.append(i)
                        if self.features_not_vary is None:
                            if df_T.loc[0, 'outcome'] != df_T.loc[i, 'outcome']:
                                lst_cf_ind.append(i)
                        
                elif num_neighbors==round(len(df_T)/2):
                    # print("2-2")
                    __, ind=self.build_tree(T, num_neighbors)
                    for i in list(ind.ravel()):
                        if self.features_not_vary is not None and self.continuous_feature_names is not None and all(item in self.continuous_feature_names for item in self.features_not_vary)==True:
                            bool_result=self.comparison_vlaue_not_vary(df_T, self.features_not_vary, i)
                            if df_T.loc[0, 'outcome'] != df_T.loc[i, 'outcome'] and bool_result==True:
                                lst_cf_ind.append(i)
                        if self.features_not_vary is not None and self.categorical_feature_names is not None and all(item in self.categorical_feature_names for item in self.features_not_vary)==True:
                            if df_T.loc[0, 'outcome'] != df_T.loc[i, 'outcome'] and all(df_T.iloc[0][self.features_not_vary]==df_T.iloc[i][self.features_not_vary]) is True:
                                lst_cf_ind.append(i)
                        if self.features_not_vary is None:
                            if df_T.loc[0, 'outcome'] != df_T.loc[i, 'outcome']:
                                lst_cf_ind.append(i)
                                
                    break 
            # print("3")
            df_cf=df_T_origin.loc[df_T_origin.index[lst_cf_ind]] 
            # print(df_cf)
            object_cf=Instances(row.to_frame(), df_cf, local_importance=None, local_impor_continus_pos=None,local_impor_continus_neg=None, local_impor_categorical_class=None)
            
            #reset
            lst_cf_ind=[]
            num_neighbors=self.num_cfs-1
            all_object_cf.append(object_cf)
        return all_object_cf

