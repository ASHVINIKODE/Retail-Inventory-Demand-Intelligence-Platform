# Exploratory data analysis eda
# The purpose of EDA is to understand the data, discover patterns, identify trends, and generate business insights before building dashboards or ML models.
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Load Feature Engineered Datasets
# =====================================================

orders = pd.read_csv(
    "data/featured/orders_featured.csv",
    parse_dates=[
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
)

order_items = pd.read_csv(
    "data/featured/order_items_featured.csv",
    parse_dates=["shipping_limit_date"]
)

customer_features = pd.read_csv(
    "data/featured/customer_features.csv",
    parse_dates=["last_purchase_date"]
)

product_features = pd.read_csv(
    "data/featured/product_features.csv"
)

seller_features = pd.read_csv(
    "data/featured/seller_features.csv"
)

abc_analysis = pd.read_csv(
    "data/featured/abc_analysis.csv"
)

monthly_demand = pd.read_csv(
    "data/featured/monthly_demand.csv"
)

payments = pd.read_csv(
    "data/processed/payments_clean.csv"
)

reviews = pd.read_csv(
    "data/processed/reviews_clean.csv",
    parse_dates=[
        "review_creation_date",
        "review_answer_timestamp"
    ]
)

print("=" * 60)
print("EDA DATASETS LOADED SUCCESSFULLY")
print("=" * 60)

datasets = {
    "Orders": orders,
    "Order Items": order_items,
    "Customer Features": customer_features,
    "Product Features": product_features,
    "Seller Features": seller_features,
    "ABC Analysis": abc_analysis,
    "Monthly Demand": monthly_demand,
    "Payments": payments,
    "Reviews": reviews
}

for name, df in datasets.items():
    print(f"{name:<20} : {df.shape}")