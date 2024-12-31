import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Step 1: Load the CSV file into a DataFrame
data = pd.read_csv('lifestyle_data.csv')

# Step 2: Prepare the data
X = data[['age', 'calories', 'sleep', 'income', 'working_hours']]
y = data['healthy_life_percent']

# Step 3: Train the model
model = RandomForestRegressor()
model.fit(X, y)

# Step 4: Save the trained model as a pickle file
joblib.dump(model, 'lifestyle_model.pkl')
