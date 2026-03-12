import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np

model = joblib.load("csat_model.pkl")

X_new = np.array([[4,3,5,2,1,4,3,2,5,1,3,4,2]])

prediction = model.predict(X_new)

print(f"Predicted CSAT Score: {prediction[0]}")