import sys
from v2_MMD_critic.mmd_critic import Dataset, select_prototypes, select_criticisms
from sklearn import metrics
import torch
import numpy as np

class prototype_generation:
    def __init__(self):
        pass
        # default gamma equals to 1/num_features
        # df_all contains generated dataframe and original dataframe

    def load_data(self,df_all):
        self.df_all=df_all
        self.X_arr=self.df_all.iloc[:,:-1].to_numpy()
        self.y_arr=self.df_all.iloc[:,-1].to_numpy()
        
        self.X_tensor = torch.tensor(self.X_arr, dtype=torch.float)
        self.y_tensor = torch.tensor(self.y_arr, dtype=torch.long)
        # sort_indices = y_tensor.argsort() # torch.argsort does not match np.argsort
        # X_tensor = X_tensor[sort_indices, :]
        #y_tensory = y_tensor[sort_indices] 
        
    def generate_prototypes(self,num_prototypes,type_kernel):
        self.gamma=1/self.X_arr.shape[1]
        self.d_train=Dataset(self.X_tensor, self.y_tensor)
        
        if type_kernel=='global':
            self.d_train.compute_rbf_kernel(self.gamma)
        elif type_kernel == 'local':
            self.d_train.compute_local_rbf_kernel(self.gamma)
        else:
            raise KeyError('kernel_type must be either "global" or "local"')
            
        mmd_lst=[]
        for i in range(len(num_prototypes)):
            # print(i, len(num_prototypes),num_prototypes[i])
            if num_prototypes[i] > 0:
                # print('Computing prototypes...', end='', flush=True)
                # s1 is a list in which each element represents a MMD for each prototype       
                prototype_indices= select_prototypes(self.d_train.K, num_prototypes[i])
                prototypes = self.d_train.X[prototype_indices]
                prototype_labels = self.d_train.y[prototype_indices]

                # sorted_by_y_indices = prototype_labels.argsort()
                # prototypes_sorted = prototypes[sorted_by_y_indices]
                # prototype_labels = prototype_labels[sorted_by_y_indices]
                # print('Done.', flush=True)
                # print(prototype_indices.sort()[0].tolist())

                XX = metrics.pairwise.rbf_kernel(self.X_tensor, self.X_tensor, self.gamma)
                ZZ = metrics.pairwise.rbf_kernel(prototypes, prototypes, self.gamma)
                XZ = metrics.pairwise.rbf_kernel(self.X_tensor, prototypes, self.gamma)
                MMD=XX.mean() + ZZ.mean() - 2 * XZ.mean()
                mmd_lst.append(MMD)
        # print(mmd_lst)
        # print(num_prototypes)
        np.set_printoptions(suppress=True)
        
        arr_trend_mmd_lst=np.array([num_prototypes,mmd_lst])
        return arr_trend_mmd_lst

        #         # Criticisms
        #     if num_criticisms > 0:
        #         # print('Computing criticisms...', end='', flush=True)
        #         criticism_indices = select_criticisms(self.d_train.K, prototype_indices, num_criticisms, regularizer)

        #         criticisms = self.d_train.X[criticism_indices]
        #         criticism_labels = self.d_train.y[criticism_indices]

        #         sorted_by_y_indices = criticism_labels.argsort()
        #         criticisms_sorted = criticisms[sorted_by_y_indices]
        #         criticism_labels = criticism_labels[sorted_by_y_indices]
        #         # print('Done.', flush=True)
        #         # print(criticism_indices.sort()[0].tolist())
        
    def generate_prototype_with_optimal_number(self, opt_num_prototype,type_kernel):
        self.gamma=1/self.X_arr.shape[1]
        self.d_train=Dataset(self.X_tensor, self.y_tensor)
        
        if type_kernel=='global':
            self.d_train.compute_rbf_kernel(self.gamma)
        elif type_kernel == 'local':
            self.d_train.compute_local_rbf_kernel(self.gamma)
        else:
            raise KeyError('kernel_type must be either "global" or "local"')
            
        if opt_num_prototype > 0:
            prototype_indices = select_prototypes(self.d_train.K, opt_num_prototype)
            prototypes = self.d_train.X[prototype_indices]
            prototype_labels = self.d_train.y[prototype_indices]

            XX = metrics.pairwise.rbf_kernel(self.X_tensor, self.X_tensor, self.gamma)
            ZZ = metrics.pairwise.rbf_kernel(prototypes, prototypes, self.gamma)
            XZ = metrics.pairwise.rbf_kernel(self.X_tensor, prototypes, self.gamma)
            MMD=XX.mean() + ZZ.mean() - 2 * XZ.mean()
            return prototype_indices.numpy()
