
import torch
from torch import nn
import pandas as pd
import numpy as np
from geoclip import LocationEncoder
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point,Polygon
import random 

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
        x1=x[:,2:]
        x2=x[:,:2]
        
        loc_embed=self.LocEncoder(x2)
        x_concat=torch.cat((self.fc12(loc_embed), self.fc11(x1)), dim=1)
        mu, logvar = self.encode(x_concat.view(-1, x_concat.shape[1]))
        z = self.reparameterize(mu, logvar)
        # dimension of z is 16
        # return the reconstructed x, the classifier output, mu and logvar
        return self.decode(z), self.classifier(z), mu, logvar, z

# Define the loss function
def loss_function(recon_x, x1, mu, logvar, y, y_pred, class_weights):

    # Calculate the Mean Squared Error loss
    MSE = F.mse_loss(recon_x, x1, reduction='sum')
    # Calculate the KL Divergence
    KLD = beta_value*(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    # Calculate the Cross Entropy loss
    CE = F.cross_entropy(y_pred, y, weight=class_weights)
    # The total loss is the sum of MSE, KLD, and CE
    return MSE, KLD, CE



def Random_Points_in_Bounds(zip_shapefile):
    gdf_polygon = gpd.read_file(zip_shapefile)
    shapely_polygon=Polygon(gdf_polygon['geometry'].iloc[0])
    minx, miny, maxx, maxy = shapely_polygon.bounds

    while True:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        random_point = Point(x,y)
        if shapely_polygon.contains(random_point)== True:
            return x, y 


def get_random_feature_value(feature_name, zip_shapefile, data_input):
    x,y=Random_Points_in_Bounds(zip_shapefile)
    if feature_name == 'lon':
        return x
    elif feature_name == 'lat':
        return y
    else:
        return random.randint(data_input[feature_name].unique().min(), data_input[feature_name].unique().max())
    
# Generate peusdo data
import warnings
warnings.filterwarnings('ignore')

data_final=pd.read_csv('../v2_data_all/v2_data_coordinate_encoded_final.csv')
GA_boundary_zip = "./v2_data/Georgia_State_Boundary.zip"
# we want to generate the same number of peusdo data as the original data, but considering some are not in the polygon of GA, we need to generate 1000 more, than use Arcgis pro to remove those not in GA
num_of_peusdo=len(data_final)+1000
data_peusdo=pd.DataFrame(columns=data_final.columns, index=range(num_of_peusdo))
for i in range(num_of_peusdo):
    print(f"{i} among {num_of_peusdo} is done")
    for feature in data_final.drop(columns=['outcome']).columns:
        data_peusdo[feature].iloc[i] = get_random_feature_value(feature_name=feature, zip_shapefile=GA_boundary_zip, data_input=data_final)
        
        
# Fit data_peusdo_X into trained Model to calculate the outcome
# prepare the data for the model
data_peusdo_X=data_peusdo.drop(columns=['outcome'])
object = MinMaxScaler()
object.fit(data_final.drop(columns=['lat','lon','outcome']))
arr_scaled=object.transform(data_peusdo_X.drop(columns=['lat','lon']))
df_coord = data_peusdo_X[['lat', 'lon']]
df_scaled=pd.DataFrame(arr_scaled, columns=data_peusdo_X.drop(columns=['lat','lon']).columns)
df_scaled_coord = pd.concat([df_coord, df_scaled], axis=1)

# convert object to float
df_scaled_coord = df_scaled_coord.astype(float)
df_scaled_coord_tensor=torch.tensor(df_scaled_coord.values).float()

# Load the saved state dictionary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=SAVAE().to(device)

model.load_state_dict(torch.load('v2_best_savae_model_dict.pth'))
model.eval()
df_scaled_coord_tensor=df_scaled_coord_tensor.to(device)
__, y_pred, __, __, __ = model(df_scaled_coord_tensor)
y_pred_class = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

# get the predicted outcome
data_peusdo_all=data_peusdo_X.copy()
data_peusdo_all['outcome']=y_pred_class   

# data_peusdo_all.to_csv('v2_data_peusdo.csv', index=False)
