# Install required packages
!pip install streamlit pyngrok folium geopy lightgbm scikit-learn
from google.colab import files
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.distance import great_circle
from pyngrok import ngrok
import numpy as np
from geopy.geocoders import Nominatim
import subprocess
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# Initialize geolocator
geolocator = Nominatim(user_agent="hospital_locator")

# Function to reverse geocode coordinates
def reverse_geocode(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        if location:
            return location.raw['address']
    except Exception as e:
        print(f"Error in reverse geocoding: {e}")
    return None

# Function to check if the location is uninhabitable based on heuristic rules
def is_uninhabitable(address):
    if not address:
        return True
    if 'water' in address.get('natural', '') or 'lake' in address.get('natural', '') or 'forest' in address.get('natural', '') or 'ocean' in address.get('natural', ''):
        return True
    if 'park' in address.get('leisure', ''):
        return True
    return False

# Load and preprocess data
uploaded = files.upload()
data = pd.read_csv("Hospital General Information final.csv")

mapping = {
    'Above the national average': 3,
    'Not Available': 0,
    'Same as the national average': 2,
    'Below the national average': 1
}
mapping_rev = {
    'Above the national average': 1,
    'Not Available': 0,
    'Same as the national average': 2,
    'Below the national average': 3
}
columns_to_map = [
    'Safety of care national comparison', 'Readmission national comparison',
    'Patient experience national comparison', 'Effectiveness of care national comparison',
    'Timeliness of care national comparison', 'Efficient use of medical imaging national comparison'
]

data[columns_to_map] = data[columns_to_map].applymap(lambda x: mapping.get(x))
data['Mortality national comparison'] = data['Mortality national comparison'].map(mapping_rev)

mapping = {"Yes": 1, "No": 0}
mapping_rev = {"Y": 1, "N": 0}
data['Emergency Services'] = data['Emergency Services'].map(mapping)
data['Meets criteria for meaningful use of EHRs'] = data['Meets criteria for meaningful use of EHRs'].map(mapping_rev)

label_encoder = LabelEncoder()
data['new_color_1'] = label_encoder.fit_transform(data['Hospital Ownership'])
data = data.drop(columns=['Hospital Ownership'])
data['new_color_2'] = label_encoder.fit_transform(data['Hospital Type'])
data = data.drop(columns=['Hospital Type'])

data.dropna(how='any', inplace=True)

mapping = {
    "Not Available": 0,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
    "1": 1,
    "0": 0
}
data['Hospital overall rating'] = data['Hospital overall rating'].map(mapping)
data = data.drop(columns=['Emergency Services'])
data = data[(data['latitude'] > 7) & (data['latitude'] < 83) & (data['longitude'] > -169) & (data['longitude'] < -52)]

# Define the prediction model
def Dragon(data, lat, long):
    X = data.drop(columns=['latitude', 'longitude'])
    scaler = QuantileTransformer(output_distribution='uniform')
    X = scaler.fit_transform(X)
    y1 = data.latitude
    y2 = data.longitude
    y = np.column_stack((y1, y2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.1
    }

    lgb_regressor = lgb.LGBMRegressor(**params)
    multi_output_regressor = MultiOutputRegressor(lgb_regressor)
    multi_output_regressor.fit(X_train, y_train)

    multi_output_regressor.estimators_[0].booster_.save_model('model_latitude.txt')
    multi_output_regressor.estimators_[1].booster_.save_model('model_longitude.txt')

    params['learning_rate'] = 0.01
    y_new = []
    X_new = []

    for i in range(len(y1)):
        k = np.sqrt((y1[i] - long) ** 2 + (y2[i] - lat) ** 2)
        if 2 < k <= 5:
            y_new.append(y[i])
            X_new.append(X[i])

    y_new = np.array(y_new)
    X_new = np.array(X_new)

    Xtestnew = []
    for i in range(len(y1)):
        k = np.sqrt((y1[i] - long) ** 2 + (y2[i] - lat) ** 2)
        if k < 2:
            Xtestnew.append(X[i])
    Xtestnew = np.array(Xtestnew)

    lgb_regressor_latitude = lgb.Booster(model_file='model_latitude.txt')
    lgb_regressor_longitude = lgb.Booster(model_file='model_longitude.txt')

    lgb_regressor_latitude = lgb.train(params, lgb.Dataset(X_new, label=y_new[:, 0]), num_boost_round=50, init_model=lgb_regressor_latitude)
    lgb_regressor_longitude = lgb.train(params, lgb.Dataset(X_new, label=y_new[:, 1]), num_boost_round=50, init_model=lgb_regressor_longitude)

    class CustomMultiOutputRegressor:
        def _init_(self, regressor_latitude, regressor_longitude):
            self.regressor_latitude = regressor_latitude
            self.regressor_longitude = regressor_longitude

        def predict(self, X):
            pred_latitude = self.regressor_latitude.predict(X, num_iteration=self.regressor_latitude.best_iteration)
            pred_longitude = self.regressor_longitude.predict(X, num_iteration=self.regressor_longitude.best_iteration)
            return np.column_stack((pred_longitude, pred_latitude))

    custom_multi_output_regressor = CustomMultiOutputRegressor(lgb_regressor_latitude, lgb_regressor_longitude)
    y_pred = custom_multi_output_regressor.predict(Xtestnew)

    y1_nw = []
    for i in range(len(y_pred)):
        k = np.sqrt((y_pred[i, 1] - long) ** 2 + (y_pred[i, 0] - lat) ** 2)
        if k <= 1.5:
            y1_nw.append(y_pred[i])
    y1_nw = np.array(y1_nw)
    return y1_nw

# Streamlit application
def find_nearby_locations(clicked_location, locations, distance_range):
    nearby_locations = []
    for loc in locations:
        if great_circle(clicked_location, loc).km <= distance_range:
            nearby_locations.append(loc)
    return nearby_locations

def find_locations_in_ranges(clicked_location, locations):
    within_1km = find_nearby_locations(clicked_location, locations, 1)
    between_1_and_2km = find_nearby_locations(clicked_location, locations, 2)
    between_1_and_2km = [loc for loc in between_1_and_2km if loc not in within_1km]
    return within_1km, between_1_and_2km

ngrok.set_auth_token('2k0RxjYoA0xdMKlJDbu70MK1vIk_3dq3qhYvWC351dA92C5x1')  # Replace with your ngrok auth token

# Define the Streamlit code as a string
streamlit_code = """
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from geopy.distance import great_circle
import numpy as np
from geopy.geocoders import Nominatim

# Initialize geolocator
geolocator = Nominatim(user_agent="hospital_locator")

def reverse_geocode(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        if location:
            return location.raw['address']
    except Exception as e:
        print(f"Error in reverse geocoding: {e}")
    return None

def is_uninhabitable(address):
    if not address:
        return True
    if 'water' in address.get('natural', '') or 'lake' in address.get('natural', '') or 'forest' in address.get('natural', '') or 'ocean' in address.get('natural', ''):
        return True
    if 'park' in address.get('leisure', ''):
        return True
    return False

def find_nearby_locations(clicked_location, locations, distance_range):
    nearby_locations = []
    for loc in locations:
        if great_circle(clicked_location, loc).km <= distance_range:
            nearby_locations.append(loc)
    return nearby_locations

def find_locations_in_ranges(clicked_location, locations):
    within_1km = find_nearby_locations(clicked_location, locations, 1)
    between_1_and_2km = find_nearby_locations(clicked_location, locations, 2)
    between_1_and_2km = [loc for loc in between_1_and_2km if loc not in within_1km]
    return within_1km, between_1_and_2km

st.title("Hospital Location Predictor")
st.write("Select a location on the map to predict hospital locations.")

# Initialize map
m = folium.Map(location=[20, 0], zoom_start=2)

# Initialize data
data = pd.read_csv("Hospital General Information final.csv")

# Display map
map_data = st_folium(m, width=700, height=500)

if map_data["last_clicked"] is not None:
    lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    st.write(f"Selected coordinates: Latitude={lat}, Longitude={lon}")

    # Show marker
    folium.Marker([lat, lon], popup=f"Latitude={lat}, Longitude={lon}").add_to(m)
    st_folium(m, width=700, height=500)

    # Process the selected location
    address = reverse_geocode(lat, lon)
    if is_uninhabitable(address):
        st.warning("The selected location is uninhabitable.")
    else:
        locations = data[["latitude", "longitude"]].values.tolist()
        within_1km, between_1_and_2km = find_locations_in_ranges((lat, lon), locations)

        st.write(f"Number of hospitals within 1 km: {len(within_1km)}")
        st.write(f"Number of hospitals between 1 km and 2 km: {len(between_1_and_2km)}")

        for loc in within_1km:
            folium.Marker([loc[0], loc[1]], popup="Existing Hospital", icon=folium.Icon(color="blue")).add_to(m)

        for loc in between_1_and_2km:
            folium.Marker([loc[0], loc[1]], popup="Existing Hospital", icon=folium.Icon(color="green")).add_to(m)

        st_folium(m, width=700, height=500)

        # Predict new locations
        predicted_locations = Dragon(data, lat, lon)
        if len(predicted_locations) > 0:
            st.write("Suggested new hospital locations:")
            for pred_loc in predicted_locations:
                folium.Marker([pred_loc[1], pred_loc[0]], popup="Suggested Location", icon=folium.Icon(color="red")).add_to(m)
            st_folium(m, width=700, height=500)
        else:
            st.write("No suitable location for a new hospital within 1.5 km.")
"""

# Save the Streamlit code to a .py file
with open("hospital_location_app.py", "w") as f:
    f.write(streamlit_code)

# Start the Streamlit app using the ngrok tunnel
process = subprocess.Popen(["streamlit", "run", "hospital_location_app.py"])

# Open the Streamlit app using the ngrok tunnel
url = ngrok.connect(8501)
print(f"Public URL: {url}")
