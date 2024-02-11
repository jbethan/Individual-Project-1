# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# %%
# Train
url1 = 'https://raw.githubusercontent.com/jbethan/Individual-Project-1/main/train.csv'
df_train = pd.read_csv(url1, index_col=None)
df_train = df_train.drop(columns=['name', 'neighborhood_overview', 'host_id', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_listings_count', 'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type', 'room_type', 'bathrooms_text', 'bedrooms', 'amenities',	'has_availability',	'first_review', 'last_review', 'instant_bookable'])
df_train = df_train.dropna()

Xtrain = df_train.drop(columns=['price', 'Id'])
Ytrain = df_train['price'] 

# %%
# Test
url2 = 'https://raw.githubusercontent.com/jbethan/Individual-Project-1/main/test.csv'
df_test = pd.read_csv(url2, index_col=None)
df_test = df_test.drop(columns=['name', 'neighborhood_overview', 'host_id', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_listings_count', 'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type', 'room_type', 'bathrooms_text', 'bedrooms', 'amenities',	'has_availability',	'first_review', 'last_review', 'instant_bookable'])
df_test = df_test.dropna()

Xtest = df_test.drop(columns=['Id'])

# %%
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xtrain, Ytrain)

# %%
pred = knn.predict(Xtest)

# %%
df = pd.DataFrame({'Id': df_test['Id'], 'price': pred})

# Write the DataFrame to a CSV file
df.to_csv('submission.csv', index=False)

# %%