# Import libraries for data manipulation
import pandas as pd

import numpy as np

# Import libraries for data visualization
import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.graphics.gofplots import ProbPlot

# Import libraries for building linear regression model
from statsmodels.formula.api import ols

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

# Import library for preparing data
from sklearn.model_selection import train_test_split

# Import library for data preprocessing
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/Users/waltershields/Downloads/masterclass_dataset.csv")

df.head()

df.info()

df = pd.read_csv("/Users/waltershields/Downloads/masterclass_dataset.csv")
summary_statistics = df.describe()

print(summary_statistics)

# Plotting all the columns to look at their distributions
for i in df.columns:
    
    plt.figure(figsize = (7, 4))
    
    sns.histplot(data = df, x = i, kde = True)
    
    plt.show()

df['Home_Value_log'] = np.log(df['Home_Value'])

sns.histplot(data = df, x = 'Home_Value_log', kde = True)


plt.figure(figsize = (12, 8))

cmap = sns.diverging_palette(230, 20, as_cmap = True)

sns.heatmap(df.corr(), annot = True, fmt = '.2f', cmap = cmap)

plt.show()

# Scatterplot to visualize the relationship between Age of Homes and Distance to Work
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'Age_of_Homes', y = 'Distance_to_Work', data = df)

plt.show()


# Scatterplot to visulaize the relationship between Highway Access and Property Tax
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'Highway_Access', y = 'Property_Tax', data = df)

plt.show()

# Remove the data corresponding to high tax rate
df1 = df[df['Property_Tax'] < 600]

# Import the required function
from scipy.stats import pearsonr

# Calculate the correlation
print('The correlation between Property Tax and Highway Access is', pearsonr(df1['Property_Tax'], df1['Highway_Access'])[0])


# Scatterplot to visualize the relationship between Business Land and Property Tax
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'Business_Land', y = 'Property_Tax', data = df)

plt.show()

# Scatterplot to visulaize the relationship between Number of Rooms and Home Value
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'Number_of_Rooms', y = 'Home_Value', data = df)

plt.show()

# Scatterplot to visulaize the relationship between Population Status and Home Value
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'Population_Status', y = 'Home_Value', data = df)

plt.show()

# Scatterplot to visualize the relationship between Business Land and Air Quality
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'Business_Land', y = 'Air_Quality', data = df)

plt.show()

# Scatterplot to visualize the relationship between Age of Homes and Air Quality
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'Age_of_Homes', y = 'Air_Quality', data = df)

plt.show()

# Scatterplot to visualize the relationship between Distance to Work and Air Quality
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'Distance_to_Work', y = 'Air_Quality', data = df)

plt.show()

# Separate the dependent variable and independent variables
Y = df['Home_Value_log']

X = df.drop(columns = {'Home_Value', 'Home_Value_log'})

# Add the intercept term
X = sm.add_constant(X)

# splitting the data in 70:30 ratio of train to test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 1)


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to check VIF
def checking_vif(train):
    vif = pd.DataFrame()
    vif["feature"] = train.columns

    # Calculating VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(train.values, i) for i in range(len(train.columns))
    ]
    return vif


print(checking_vif(X_train))


# Create the model after dropping Property Tax
X_train = X_train.drop('Property_Tax', axis=1)

# Check for VIF - (see if multicollinearity is removed)
print(checking_vif(X_train))

model1 = sm.OLS(y_train, X_train).fit()
# Get the model summary
model1.summary()

# Create the model after dropping columns 'Home_Value', 'Home_Value_log', 'Property_Tax:', 'Land_Size', 'Age_of_Homes', 'Business_Land' from df DataFrame
Y = df['Home_Value_log']

X = df.drop(['Home_Value', 'Home_Value_log', 'Property_Tax', 'Land_Size', 'Age_of_Homes', 'Business_Land'], axis = 1)
X = sm.add_constant(X)

# Splitting the data in 70:30 ratio of train to test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30 , random_state = 1)

# Create the model
model2 = sm.OLS(y_train, X_train).fit()

# Get the model summary
model2.summary()

residuals = model2.resid
np.mean(residuals)


from statsmodels.stats.diagnostic import het_white

from statsmodels.compat import lzip

import statsmodels.stats.api as sms

name = ["F statistic", "p-value"]

test = sms.het_goldfeldquandt(model2.resid, model2.model.exog)

lzip(name, test)

# Predicted values
fitted = model2.fittedvalues

# sns.set_style("whitegrid")

sns.residplot(x=fitted, y=y_train, color="lightblue", lowess=True)

plt.xlabel("Fitted Values")

plt.ylabel("Residual")

plt.title("Residual PLOT")

plt.show()

# Plot histogram of residuals

residuals = model2.resid

plt.hist(residuals, bins=20)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()


# Plot q-q plot of residuals
import pylab

import scipy.stats as stats

stats.probplot(residuals, dist = "norm", plot = pylab)

plt.show()


# RMSE
def rmse(predictions, targets):
    return np.sqrt(((targets - predictions) ** 2).mean())


# MAPE
def mape(predictions, targets):
    return np.mean(np.abs((targets - predictions)) / targets) * 100


# MAE
def mae(predictions, targets):
    return np.mean(np.abs((targets - predictions)))


# Model Performance on test and train data
def model_pref(olsmodel, x_train, x_test):

    # In-sample Prediction
    y_pred_train = olsmodel.predict(x_train)
    y_observed_train = y_train

    # Prediction on test data
    y_pred_test = olsmodel.predict(x_test)
    y_observed_test = y_test

    print(
        pd.DataFrame(
            {
                "Data": ["Train", "Test"],
                "RMSE": [
                    rmse(y_pred_train, y_observed_train),
                    rmse(y_pred_test, y_observed_test),
                ],
                "MAE": [
                    mae(y_pred_train, y_observed_train),
                    mae(y_pred_test, y_observed_test),
                ],
                "MAPE": [
                    mape(y_pred_train, y_observed_train),
                    mape(y_pred_test, y_observed_test),
                ],
            }
        )
    )


# Checking model performance
model_pref(model2, X_train, X_test)  


# Import the required function

from sklearn.model_selection import cross_val_score

# Build the regression model and cross-validate
linearregression = LinearRegression()                                    

cv_Score11 = cross_val_score(linearregression, X_train, y_train, cv = 10)
cv_Score12 = cross_val_score(linearregression, X_train, y_train, cv = 10, 
                             scoring = 'neg_mean_squared_error')                                  


print("RSquared: %0.3f (+/- %0.3f)" % (cv_Score11.mean(), cv_Score11.std() * 2))
print("Mean Squared Error: %0.3f (+/- %0.3f)" % (-1*cv_Score12.mean(), cv_Score12.std() * 2))

coef = model2.params

pd.DataFrame({'Feature' : coef.index, 'Coefs' : coef.values})


# Write the equation of the fit

Equation = "log (House Price) = "

print(Equation, end = '\t')

for i in range(len(coef)):
    print('(', coef[i], ') * ', coef.index[i], '+', end = ' ')