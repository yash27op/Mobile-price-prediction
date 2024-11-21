import pandas as pd
data = pd.read_csv('dataset.csv')  # Replace with your dataset path
print(data.isnull().sum())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_features = ['battery_power', 'clock_speed', 'fc', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'mobile_wt']

data[numeric_features] = scaler.fit_transform(data[numeric_features])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])
from sklearn.model_selection import train_test_split
X = data.drop(columns=['price_range'])
y = data['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
import pandas as pd

# Example input data (ensure all features are included)
new_product = pd.DataFrame({
    'battery_power': [1500],
    'blue': [1],
    'clock_speed': [2.5],
    'dual_sim': [1],
    'fc': [5],
    'four_g': [1],
    'int_memory': [32],
    'mobile_wt': [140],
    'n_cores': [8],
    'pc': [13],
    'px_height': [720],
    'px_width': [1280],
    'ram': [4096],
    'sc_h': [14],
    'sc_w': [7],
    'talk_time': [20],
    'three_g': [1],
    'touch_screen': [1],
    'wifi': [1]
})
from sklearn.preprocessing import StandardScaler

# Ensure you use the same scaler instance fitted on training data
numeric_features = ['battery_power', 'clock_speed', 'fc', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'mobile_wt']
new_product[numeric_features] = scaler.transform(new_product[numeric_features])
# Predict the price range
print("Training features:", model.feature_names_in_)
print("Prediction features:", new_product.columns)
new_product.rename(columns={'m_dep': 'm_deep'}, inplace=True)  # Example

# predicted_price_range = model.predict(new_product)

# # Map the numeric prediction to the actual price range category
# price_map = {0: 'Low Cost', 1: 'Medium Cost', 2: 'High Cost', 3: 'Very High Cost'}
# predicted_category = price_map[predicted_price_range[0]]

# print(f"The predicted price range is: {predicted_category}")
new_products = pd.DataFrame({
    'battery_power': [1500, 2000],
    'blue': [1, 0],
    'clock_speed': [2.5, 1.8],
    'dual_sim': [1, 1],
    'fc': [5, 8],
    'four_g': [1, 0],
    'int_memory': [32, 16],
    'mobile_wt': [140, 170],
    'n_cores': [8, 4],
    'pc': [13, 5],
    'px_height': [720, 480],
    'px_width': [1280, 640],
    'ram': [4096, 2048],
    'sc_h': [14, 12],
    'sc_w': [7, 6],
    'talk_time': [20, 15],
    'three_g': [1, 1],
    'touch_screen': [1, 1],
    'wifi': [1, 0]
})

# Add missing features if necessary
if 'm_dep' not in new_products.columns:
    new_products['m_dep'] = 0.5  # Replace with appropriate value

# Ensure columns match training features
new_products = new_products[model.feature_names_in_]

# Apply preprocessing (e.g., scaling)
new_products[numeric_features] = scaler.transform(new_products[numeric_features])

# Predict the price range
predictions = model.predict(new_products)

# Map predictions to categories
price_map = {0: 'Low Cost', 1: 'Medium Cost', 2: 'High Cost', 3: 'Very High Cost'}
predicted_categories = [price_map[p] for p in predictions]

print(predicted_categories)
