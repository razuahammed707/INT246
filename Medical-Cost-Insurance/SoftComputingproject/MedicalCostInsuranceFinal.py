
# coding: utf-8

# In[5]:


## Convention:
## Sex for female = 0 and male = 1

## Smoker = 1 and non-smoker = 0

## Region
## Northeast=0
## Northwest=1
## Southeast=2
## Southwest=3

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('insurance.csv')
prices = data['charges']
data['region'] = data['region'].astype('category')
data['sex'] = data['sex'].astype('category')
data['smoker'] = data['smoker'].astype('category')
data['region']=data['region'].cat.codes
data['smoker']=data['smoker'].cat.codes
data['sex']=data['sex'].cat.codes
print(data)

features = data.drop(['charges'], axis = 1)
    

print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))


# In[13]:



from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    score = r2_score(y_true,y_predict)
    return score


# In[8]:


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,prices,test_size = 0.20 )

print("Training and testing split was successful.")


# In[9]:


# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
data['region'].replace(data['region'])
def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':range(1,11)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    
    grid = GridSearchCV(regressor,params,scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_
print(data)


# In[10]:


# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
#print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))


# In[11]:


# Produce a matrix for client data
client_data = [[19, 0,27.9, 0,1,3],[18,1,33.77,1,0,2],[61,0,29.07,0,1,1],[21,0,25.8,0,0,3]]
# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

