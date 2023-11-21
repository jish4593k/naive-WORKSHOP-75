import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Importing the dataset
datasets = pd.read_csv('Social_Network_Ads.csv')
X = datasets.iloc[:, [2, 3]].values
Y = datasets.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Define a Bayesian Naive Bayes model using PyMC3
with pm.Model() as naive_bayes_model:
    alpha = pm.Normal('alpha', mu=0, sigma=10, shape=2)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    mu = pm.Deterministic('mu', alpha + pm.math.dot(X_Train, beta))
    theta = pm.math.sigmoid(mu)
    
    # Likelihood
    y_obs = pm.Bernoulli('y_obs', p=theta, observed=Y_Train)

# Perform Bayesian inference using Markov Chain Monte Carlo (MCMC)
with naive_bayes_model:
    trace = pm.sample(2000, tune=1000, target_accept=0.9)

# Plot the trace of parameters
pm.plot_posterior(trace, var_names=['alpha', 'beta'], text_size=12, figsize=(12, 6))
plt.suptitle('Posterior Distributions of Parameters', y=1.02)
plt.show()

# Extracting posterior samples
posterior_samples = pm.sample_posterior_predictive(trace, samples=500, model=naive_bayes_model)['y_obs']

# Predictions
y_pred_prob = np.mean(posterior_samples, axis=0)
y_pred = np.round(y_pred_prob)

# Model Evaluation
accuracy = accuracy_score(Y_Train, y_pred)
cm = confusion_matrix(Y_Train, y_pred)
classification_report_str = classification_report(Y_Train, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{cm}")
print(f"Classification Report:\n{classification_report_str}")
