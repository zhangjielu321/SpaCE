# class for data attribute
import pandas as pd

class Instances:
    def __init__(self, query_instance, df_cf, local_importance,local_impor_continus_pos,local_impor_continus_neg, local_impor_categorical_class):
        self.query_instance=query_instance
        self.df_cf=df_cf
        self.original_outcome=self.query_instance.iloc[:,-1].iat[0]
        self.desired_outcome=1.0 - round(self.original_outcome)
        self.column_names=self.query_instance.columns
        self.local_importance=local_importance
        self.local_impor_continus_pos=local_impor_continus_pos
        self.local_impor_continus_neg=local_impor_continus_neg
        self.local_impor_categorical_class=local_impor_categorical_class
        
    def display_df_cf(self, df_cf, query_instance, show_only_changes):
        from IPython.display import display
        if show_only_changes is False:
            display(self.df_cf)  # works only in Jupyter notebook
        else:
            newdf = self.df_cf.values.tolist()
            org = self.query_instance.values.tolist()[0]
            for ix in range(self.df_cf.shape[0]):
                for jx in range(len(org)):
                    if newdf[ix][jx] == org[jx]:
                        newdf[ix][jx] = '-'
                    else:
                        newdf[ix][jx] = str(newdf[ix][jx])
            display(pd.DataFrame(newdf, columns=self.df_cf.columns, index=self.df_cf.index))
            
    # def display(self, query_instance, df_cf, local_importance):
    def display(self):
        from IPython.display import display
        print('Query instance (original outcome : %i)' % round(self.original_outcome))
        display(self.query_instance)

        print('Diverse Counterfactual set (new outcome : %i)' % round(self.desired_outcome))
        self.display_df_cf(self.df_cf, self.query_instance, show_only_changes=True)
        
        if self.local_importance is not None:
            print('Local importance')
            display(self.local_importance)
            
        if self.local_impor_continus_pos is not None:
            print('Local importance continuous positive')
            display(self.local_impor_continus_pos)
            
        if self.local_impor_continus_neg is not None:
            print('Local importance continuous negative')
            display(self.local_impor_continus_neg)
            
        if self.local_impor_categorical_class is not None:
            print('Local importance categorical class')
            display(self.local_impor_categorical_class)