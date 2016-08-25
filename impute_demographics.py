# Load all the packages needed
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import numpy as np

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import Imputer
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

pd.options.display.max_columns = 50
pd.options.display.max_rows = 1000

experian = pd.read_csv("ppl_experian_data.csv")

# filled = pd.read_csv("ppl_experian_filled.csv")
# experian = experian.loc[experian['account.id'].isin(filled['account.id']) == False]

experian = experian.sample(n=5000)

experian['heatIndicator'] =  experian['heatIndicator'].map(lambda x: 2 if x == 'Electric' else 1)
experian['dwellingType'] =  experian['dwellingType'].map(lambda x: 2 if x == 'SingleFamily' else 1)
experian['homeOwnerRenter'] =  experian['homeOwnerRenter'].map(lambda x: 2 if x == 'H' or x == "9" or x == "7" or x == "8" else 1)
experian['sqFootage'] = experian['sqFootage'].replace(0, np.nan)
experian['homeLandSqFootage'] = experian['homeLandSqFootage'].replace(0, np.nan)
experian['estHouseholdIncome'] = experian['estHouseholdIncome'].replace(0, np.nan)
experian['homeTotalValue'] = experian['homeTotalValue'].replace(0, np.nan)
experian['homeIprovementValue'] = experian['homeIprovementValue'].replace(0, np.nan)
experian['estimatedValue'] = experian['estimatedValue'].replace(0, np.nan)
experian['homeTotalTax'] = experian['homeTotalTax'].replace(0, np.nan)

account_id = experian[['account.id']]
experian = experian[['sqFootage', 'yearBuilt','homeLandSqFootage',
                         'estHouseholdIncome','homeTotalValue',
                         'greenAware', 'heatIndicator', 'dwellingType','homeOwnerRenter',
                         "zipCode", "latitude", "longitude",
         'hybridCars',  'totalRooms', 'numRooms', 'homeBath',
        'homeStories',  'homePropertyIndicator','numChildren', 'numAdults', 'lengthOfResidence',
        'homeIprovementValue', 'estimatedValue', 'homeTotalTax']]

#experian = experian.fillna(0)
#experian = experian.dropna()
#experian = experian.replace(0, np.nan)

experian_needfit = experian.drop(["zipCode", "latitude", "longitude",
         'hybridCars',  'totalRooms', 'numRooms', 'homeBath', 'heatIndicator', 'dwellingType','homeOwnerRenter',
        'homeStories',  'homePropertyIndicator','numChildren', 'numAdults', 'lengthOfResidence',
        'homeIprovementValue', 'estimatedValue', 'homeTotalTax'], axis = 1)

experian_rest = experian[["zipCode", "latitude", "longitude",
         'hybridCars',  'totalRooms', 'numRooms', 'homeBath','heatIndicator', 'dwellingType','homeOwnerRenter',
        'homeStories',  'homePropertyIndicator','numChildren', 'numAdults', 'lengthOfResidence',
        'homeIprovementValue', 'estimatedValue', 'homeTotalTax']]

X = experian_needfit.as_matrix()

# Step for validating only-compare real to predicted
# X is a data matrix which we're going to randomly drop entries from
missing_mask = np.random.randn(*X.shape) > 0
print(missing_mask.shape)
missing_mask_rest = np.zeros(experian_rest.shape, dtype=bool)
missing_mask = np.concatenate((missing_mask, missing_mask_rest), axis=1)

X = experian.as_matrix()
X_incomplete = X.copy()
# missing entries indicated with NaN
missing_mask = np.array(experian.isnull())
X_incomplete[missing_mask] = np.nan
print(X.shape, experian.shape, missing_mask.shape)

# # Use 3 nearest rows which have a feature to fill in each row's missing features
# knnImpute = KNN(k=1000)
# X_filled_knn = knnImpute.complete(X_incomplete)
# pd_missing_mask = pd.DataFrame(missing_mask, index = experian.index.tolist(), columns = experian.columns.values.tolist() )
# pd_filled_knn = pd.DataFrame(X_filled_knn, index = experian.index.tolist(), columns = experian.columns.values.tolist())
# pd_filled_knn = pd.concat([account_id, pd_filled_knn], axis = 1)
# pd_filled_knn.to_csv("ppl_experian_filled_k_closestrows.csv")


# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
softImpute = SoftImpute()

# simultaneously normalizes the rows and columns of your observed data,
# sometimes useful for low-rank imputation methods
biscaler = BiScaler()

# rescale both rows and columns to have zero mean and unit variance
X_incomplete_normalized = biscaler.fit_transform(X_incomplete)

X_filled_softimpute_normalized = softImpute.complete(X_incomplete_normalized)
X_filled_softimpute = biscaler.inverse_transform(X_filled_softimpute_normalized)
pd_missing_mask = pd.DataFrame(missing_mask, index = experian.index.tolist(), columns = experian.columns.values.tolist() )
pd_filled_softimpute = pd.DataFrame(X_filled_softimpute, index = experian.index.tolist(), columns = experian.columns.values.tolist())

pd_filled_softimpute = pd.concat([account_id, pd_filled_softimpute], axis = 1)

#pd_filled_softimpute = filled.append(pd_filled_softimpute)
pd_filled_softimpute.to_csv("ppl_experian_filled_softimpute.csv", sep = ",")


#
# # matrix completion using convex optimization to find low-rank solution
# # that still matches observed values.
# # This is very slow!!!!
# X_filled_nnm = NuclearNormMinimization().complete(X_incomplete)
#
#
# # print mean squared error for the three imputation methods above
# nnm_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
# #print("Nuclear norm minimization MSE: %f" % nnm_mse)
#
# softImpute_mse = ((X_filled_softimpute[missing_mask] - X[missing_mask]) ** 2).mean()
# print("SoftImpute MSE: %f" % softImpute_mse)
#
# knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
# print("knnImpute MSE: %f" % knn_mse)
#
# # Bound the fitted values within ranges
#
#
# X_filled_knn[missing_mask]
# X_filled_softimpute[missing_mask]
#
# # Calculate the mean average percentage error for each variables
# # First convert to dataframes
# pd_filled_knn = pd.DataFrame(X_filled_knn, index = experian.index.tolist(), columns = experian.columns.values.tolist())
# pd_filled_knn = pd.concat([account_id, pd_filled_knn], axis = 1)
#
# pd_filled_knn.to_csv("ppl_experian_filled_k_closestrows.csv")
#
#
# pd_X = pd.DataFrame(X, index = experian.index.tolist(), columns = experian.columns.values.tolist())
#
# # bound the yearBuilt
# pd_filled_softimpute['yearBuilt'][pd_filled_softimpute['yearBuilt'] >= 2015] = 2015
# # Calculate percentages of missing values of each variables
# pd_missing_mask.mean()
#
# pd_error_sqfootage = pd.concat([pd.Series(pd_X['sqFootage'], name='sqFootage_real'),
#                       pd.Series(pd_filled_softimpute['sqFootage'], name='sqFootage_fitted'),
#                       pd.Series(pd_missing_mask['sqFootage'], name='missing_mask')], axis=1)
# pd_error_sqfootage = pd_error_sqfootage.loc[pd_error_sqfootage['missing_mask'] == True]
# pd_error_sqfootage.head()
#
# # Percentage Error
# pd_error_sqfootage['MAPE'] = (pd_error_sqfootage['sqFootage_fitted'] - pd_error_sqfootage['sqFootage_real']) / pd_error_sqfootage['sqFootage_real'] * 100
#
# print(max(pd_error_sqfootage['MAPE']), min(pd_error_sqfootage['MAPE']), pd_error_sqfootage['MAPE'].mean())
# print(min(pd_error_sqfootage['sqFootage_real']), max(pd_error_sqfootage['sqFootage_real']))
#
# pd_error_yearbuilt = pd.concat([pd.Series(pd_X['yearBuilt'], name='yearBuilt_real'),
#                       pd.Series(pd_filled_softimpute['yearBuilt'], name='yearBuilt_fitted'),
#                       pd.Series(pd_missing_mask['yearBuilt'], name='missing_mask')], axis=1)
# pd_error_yearbuilt = pd_error_yearbuilt.loc[pd_error_yearbuilt['missing_mask'] == True]
#
# # Percentage Error
# pd_error_yearbuilt['MAPE'] = (pd_error_yearbuilt['yearBuilt_fitted'] - pd_error_yearbuilt['yearBuilt_real']) / pd_error_yearbuilt['yearBuilt_real'] * 100
#
# print(max(pd_error_yearbuilt['MAPE']), min(pd_error_yearbuilt['MAPE']), pd_error_yearbuilt['MAPE'].mean())
# print(pd_error_yearbuilt['yearBuilt_fitted'].max(), pd_error_yearbuilt['yearBuilt_fitted'].min())
#
#
# pd_error_lengthresidence = pd.concat([pd.Series(pd_X['lengthOfResidence'], name='lengthOfResidence_real'),
#                                 pd.Series(pd_filled_softimpute['lengthOfResidence'], name='lengthOfResidence_fitted'),
#                                pd.Series(pd_missing_mask['lengthOfResidence'], name='missing_mask')], axis=1)
# pd_error_lengthresidence = pd_error_lengthresidence.loc[pd_error_lengthresidence['missing_mask'] == True]
#
# # Percentage Error
# pd_error_lengthresidence['MAPE'] = (pd_error_lengthresidence['lengthOfResidence_fitted']- pd_error_lengthresidence['lengthOfResidence_real'])/pd_error_lengthresidence['lengthOfResidence_real']*100
#
# print(max(pd_error_lengthresidence['MAPE']),min(pd_error_lengthresidence['MAPE']), pd_error_lengthresidence['MAPE'].mean())
#
# pd_error_houseincome = pd.concat([pd.Series(pd_X['estHouseholdIncome'], name='estHouseholdIncome_real'),
#                       pd.Series(pd_filled_softimpute['estHouseholdIncome'], name='estHouseholdIncome_fitted'),
#                       pd.Series(pd_missing_mask['estHouseholdIncome'], name='missing_mask')], axis=1)
# pd_error_houseincome = pd_error_houseincome.loc[pd_error_houseincome['missing_mask'] == True]
#
# # Percentage Error
# pd_error_houseincome['MAPE'] = (pd_error_houseincome['estHouseholdIncome_fitted'] - pd_error_houseincome['estHouseholdIncome_real']) / pd_error_houseincome[
#     'estHouseholdIncome_real'] * 100
#
# print(max(pd_error_houseincome['MAPE']), min(pd_error_houseincome['MAPE']), pd_error_houseincome['MAPE'].mean())
#
# pd_error_numadults = pd.concat([pd.Series(pd_X['numAdults'], name='numAdults_real'),
#                       pd.Series(pd_filled_softimpute['numAdults'], name='numAdults_fitted'),
#                       pd.Series(pd_missing_mask['numAdults'], name='missing_mask')], axis=1)
# pd_error_numadults = pd_error_numadults.loc[pd_error_numadults['missing_mask'] == True]
#
# # Percentage Error
# pd_error_numadults['MAPE'] = (pd_error_numadults['numAdults_fitted'] - pd_error_numadults['numAdults_real']) / pd_error_numadults['numAdults_real'] * 100
#
# print(max(pd_error_numadults['MAPE']), min(pd_error_numadults['MAPE']), pd_error_numadults['MAPE'].mean())
#
# pd_error_landsqfootage = pd.concat([pd.Series(pd_X['homeLandSqFootage'], name='homeLandSqFootage_real'),
#                       pd.Series(pd_filled_softimpute['homeLandSqFootage'], name='homeLandSqFootage_fitted'),
#                       pd.Series(pd_missing_mask['homeLandSqFootage'], name='missing_mask')], axis=1)
# pd_error_landsqfootage = pd_error_landsqfootage.loc[pd_error_landsqfootage['missing_mask'] == True]
#
# # Percentage Error
# pd_error_landsqfootage['MAPE'] = (pd_error_landsqfootage['homeLandSqFootage_fitted'] - pd_error_landsqfootage['homeLandSqFootage_real']) / pd_error_landsqfootage[
#     'homeLandSqFootage_real'] * 100
#
# print(max(pd_error_landsqfootage['MAPE']), min(pd_error_landsqfootage['MAPE']), pd_error_landsqfootage['MAPE'].mean())
#
# pd_error_landsqfootage = pd_error_landsqfootage.loc[pd_error_landsqfootage['missing_mask'] == True]
#
#
# # plot the Percentage Error
# import matplotlib.pyplot as plt
#
# bins = np.linspace(-1,20,50 )
# plt.hist(pd_error['MAPE'].values, bins)
# plt.title("Percentage Error")
# plt.xlabel("Value (%)")
# plt.ylabel("Frequency")
#
# fig = plt.gcf()
#
#
# # plot the Percentage Error
# import matplotlib.pyplot as plt
#
# bins = np.linspace(-50,200,50 )
# plt.hist(pd_error['MAPE'].values, bins)
# plt.title("Percentage Error")
# plt.xlabel("Value (%)")
# plt.ylabel("Frequency")
#
# fig = plt.gcf()
