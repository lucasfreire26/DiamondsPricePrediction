from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
from sklearn import metrics

# Importing dataset and examining it
dataset = pd.read_csv("../../Downloads/diamonds.csv", sep = ';')
pd.set_option('display.max_columns', None) # to make sure you can see all the columns in output window
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# #creating dummy columns DATA PREPARATION
categorical_features = ['cut', 'clarity', 'color']
final_data = pd.get_dummies(dataset, columns=categorical_features)
print(final_data.info())
print(final_data.head(2))
print(final_data.describe())

# Dividing final_data into label and feature sets
X = final_data.drop('price', axis=1)  # Features
Y = final_data['price']  # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)
print(X_scaled)

# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
figure.show()

# Dividing final_data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=100)

print(X_train.shape)
print(X_test.shape)

oversample = RandomOverSampler(random_state=101)
X_train, Y_train = oversample.fit_resample(X_train, Y_train)

####################### Random Forest Regressor ###############
# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
rfr = RandomForestRegressor(criterion='mse', max_features='sqrt', random_state=1)
grid_param = {'n_estimators': [50,100,150,200,250, 300, 350, 400]}

gd_sr = GridSearchCV(estimator=rfr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_train, Y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_

# Mean cross-validated score of the best_estimator
print(best_result)

# # Building random forest using the tuned parameter
rfr = RandomForestRegressor(n_estimators=300, criterion='mse', max_features='sqrt', random_state=1)
rfr.fit(X_train,Y_train)
featimp = pd.Series(rfr.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)

Y_pred = rfr.predict(X_test)
print('MSE: ', metrics.mean_squared_error(Y_test, Y_pred))
print('R2 score: ', metrics.r2_score(Y_test, Y_pred))

# Selecting features with higher sifnificance and redefining feature set
X_ = dataset[['y', 'carat', 'x']]

feature_scaler = StandardScaler()
X_scaled_ = feature_scaler.fit_transform(X_)

# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
rfr = RandomForestRegressor(criterion='mse', max_features='sqrt', random_state=1)
grid_param = {'n_estimators': [50,100,150,200,250]}

gd_sr = GridSearchCV(estimator=rfr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_test, Y_test)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)


################################################################################
# # # Implementing Gradient Boost

gbr = GradientBoostingRegressor()
grid_param = {'learning_rate': [0.01, 0.02, 0.03, 0.04],
              'subsample': [0.9, 0.5, 0.2, 0.1],
              'n_estimators': [100, 500, 1000, 1500],
              'max_depth': [4, 6, 8, 10]
              }

gd_sr = GridSearchCV(estimator=gbr, param_grid=grid_param, cv=2, n_jobs=-1)

gd_sr.fit(X_scaled, Y)

print(" Results from Grid Search ")
print("\n The best estimator across ALL searched params:\n", gd_sr.best_estimator_)
print("\n The best score across ALL searched params:\n", gd_sr.best_score_)
print("\n The best parameters across ALL searched params:\n", gd_sr.best_params_)

################################################################################

# Implementing Linear Regression
# Tuning the SGDRegressor parameters 'eta0' (learning rate) and 'max_iter' using Grid Search
sgdr = SGDRegressor(penalty=None, random_state = 1)
grid_param = {'eta0': [.001, .01, .1, 1], 'max_iter':[10000, 20000, 40000, 50000, 60000, 70000]}

gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

##############################################################################
# Implementing Elastic Net Regularization (Elastic Net Regression)
# Tuning Regularization parameter alpha and l1_ratio
sgdr = SGDRegressor(eta0=.001, max_iter=10000, penalty='elasticnet', random_state=1)
grid_param = {'alpha': [.0001, .001, .01, .1, 1,10,50,80,100],'l1_ratio':[0, 0.1, 0.3,0.5,0.7,0.9,1]}

gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

# Building SGDRegressor using the tuned parameters
sgdr = SGDRegressor(eta0=.001, max_iter=10000, penalty='elasticnet', alpha=0.1, l1_ratio=0, random_state=1)
sgdr.fit(X_scaled,Y)
print('Intercept', sgdr.intercept_)
print(pd.DataFrame(zip(X.columns, sgdr.coef_), columns=['Features','Coefficients']).sort_values(by=['Coefficients'],ascending=False))
################################################################
