
import dice_ml
import torch
from torch import nn
import pandas as pd
from geoclip import LocationEncoder
from sklearn.preprocessing import MinMaxScaler

data_concat=pd.read_csv('./v2_analysis_indivilevel/v2_data/v2_data_concat.csv')
data_concat_outcome_0 = data_concat[data_concat['outcome']==0]
data_concat_outcome_0.reset_index(drop=True, inplace=True)

X = data_concat_outcome_0.drop('outcome', axis=1)
y = data_concat_outcome_0['outcome']
object = MinMaxScaler()
X_scaled=object.fit_transform(X.drop(['lat', 'lon'], axis=1))
X_scaled=pd.DataFrame(X_scaled, columns=X.drop(columns=['lat','lon']).columns)


X_coord=data_concat_outcome_0[['lat', 'lon']]
X_scaled_coord = pd.concat([X_coord, X_scaled], axis=1)

data_sacled_coord=X_scaled_coord.copy()
data_sacled_coord['outcome']=data_concat_outcome_0['outcome']


# Define the VAE model
class SAVAE(nn.Module):
    def __init__(self):
        super(SAVAE, self).__init__()
        # Define the concatenation part
        self.fc11 = nn.Linear(12, 64)
        self.fc12 = nn.Linear(512, 64)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        # Define the second fully connected layer for mu
        self.fc21 = nn.Linear(128, 16)
        # Define the third fully connected layer for logvar
        self.fc22 = nn.Linear(128, 16)
        # Define the fourth fully connected layer for decoding
        self.fc3 = nn.Linear(16, 12)
        # Define the classifier layer
        self.classifier = nn.Linear(16, 3)
        self.LocEncoder = LocationEncoder()

    # Define the encoder part of VAE
    def encode(self, x_concat):
        h1 = self.LeakyReLU(x_concat)
        return self.fc21(h1), self.fc22(h1)

    # Define the reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # Define the decoder part of VAE
    def decode(self, z):
        h3=self.fc3(z)
        return torch.sigmoid(h3)

    def forward(self,x):
        num_dims = x.ndim
        # If the array is 2D
        if num_dims == 2:
            x1=x[:,2:]
            x2=x[:,:2]
        # If the array is 1D
        elif num_dims == 1:
            x1=x[2:]
            x2=x[:2]
        
        loc_embed=self.LocEncoder(x2)
        x_concat=torch.cat((self.fc12(loc_embed), self.fc11(x1)), dim=1)
        mu, logvar = self.encode(x_concat.view(-1, x_concat.shape[1]))
        z = self.reparameterize(mu, logvar)
        # dimension of z is 16
        # return the reconstructed x, the classifier output, mu and logvar
        return self.decode(z), self.classifier(z), mu, logvar, z

# repackage the model into the modified model, this modified model will only return the predicted class
class ModifiedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        _, output, _, _, _ = self.model(x)
        # get the predicted class (3 classes, 0: expired, 1:still doing resuscitation, 2: survived)
        y_pred = output.argmax(dim=1, keepdim=True)
        # make the num of predicted class as 2 (0: expired, 1: survived)
        y_pred[y_pred == 2] = 1
        return y_pred

new_model=torch.load('./v2_analysis_indivilevel/v2_best_savae_model.pth')
model_repackaged = ModifiedModel(new_model)
model_repackaged=model_repackaged.to('cpu')
torch.save(model_repackaged,'./v2_analysis_indivilevel/v2_repackaged_forDice_model.pth')

continuous_feature_names=['lat','lon']
d = dice_ml.Data(dataframe=data_sacled_coord, continuous_features=continuous_feature_names, outcome_name='outcome')
backend = 'PYT'  # needs pytorch installed
ML_modelpath = './v2_analysis_indivilevel/v2_repackaged_forDice_model.pth'
m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
exp = dice_ml.Dice(d, m, method="gradient")

# generate counterfactuals
X_scaled_coord=X_scaled_coord.reset_index(drop=True)
dice_exp = exp.generate_counterfactuals(X_scaled_coord, total_CFs=1, desired_class="opposite")
# highlight only the changes

# with open("./v2_output_baselines/v2_dice_cfs_new", "wb") as fp:
#     pickle.dump(dice_exp, fp)