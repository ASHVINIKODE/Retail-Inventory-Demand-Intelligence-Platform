# demand forcasting model 
#Predict future monthly demand using Machine Learning.We'll use Random Forest Regressor
#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np

#load dataset
monthly_demand=pd.read_csv("data/featured/monthly_demand.csv")
#prepare features-convert the year month column into a number
monthly_demand["year_month"]=pd.to_datetime(monthly_demand["year_month"])
monthly_demand["year"] = (
    monthly_demand["year_month"]
    .dt.year
)

monthly_demand["month"] = (
    monthly_demand["year_month"]
    .dt.month
)

#encode product category
monthly_demand["product_category"] = (
    monthly_demand["product_category_name"]
    .astype("category")
    .cat.codes
)
#Features & Target
X = monthly_demand[
    [
        "year",
        "month",
        "product_category",
        "average_price"
    ]
]

y = monthly_demand["total_quantity"]

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#Train Model
model=RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(
    X_train,
    X_test
)

#predictions
predictions=model.predict(
    X_test
)
#Model Evaluation
mae=mean_squared_error(
    y_test,
    predictions
)
rmse=np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)
r2=r2_score(
    y_test,
    predictions
)
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE : {mae:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R² Score : {r2:.3f}")

#feature importance
importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print()

print(importance)

#Save Predictions
results = X_test.copy()

results["Actual"] = y_test.values

results["Predicted"] = predictions

results.to_csv(
    "data/featured/demand_predictions.csv",
    index=False
)

# Why Random Forest?
# Handles non-linear relationships
# Robust to noise
# Little preprocessing required
# Performs well on tabular retail data
# What are the features?
# Year
# Month
# Product Category
# Average Price
# What is the target?

# Monthly quantity sold.

