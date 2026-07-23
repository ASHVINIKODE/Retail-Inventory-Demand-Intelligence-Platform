import pandas as pd
import os

# =====================================================
# Load Datasets
# =====================================================

customers = pd.read_csv("data/raw/olist_customers_dataset.csv")
orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
order_items = pd.read_csv("data/raw/olist_order_items_dataset.csv")
payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")
products = pd.read_csv("data/raw/olist_products_dataset.csv")
reviews = pd.read_csv("data/raw/olist_order_reviews_dataset.csv")
sellers = pd.read_csv("data/raw/olist_sellers_dataset.csv")
geolocation = pd.read_csv("data/raw/olist_geolocation_dataset.csv")
category_translation = pd.read_csv("data/raw/product_category_name_translation.csv")

# =====================================================
# Convert Date Columns
# =====================================================

order_date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in order_date_columns:
    orders[col] = pd.to_datetime(orders[col])

reviews["review_creation_date"] = pd.to_datetime(
    reviews["review_creation_date"]
)

reviews["review_answer_timestamp"] = pd.to_datetime(
    reviews["review_answer_timestamp"]
)

order_items["shipping_limit_date"] = pd.to_datetime(
    order_items["shipping_limit_date"]
)

# =====================================================
# Geolocation Cleaning
# =====================================================
# Multiple latitude/longitude values exist for the same ZIP code.
# Aggregate them into one representative record per ZIP code.

geolocation = (
    geolocation
    .groupby("geolocation_zip_code_prefix", as_index=False)
    .agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean",
        "geolocation_city": "first",
        "geolocation_state": "first"
    })
)

# =====================================================
# Reviews Cleaning
# =====================================================

reviews["review_comment_title"] = reviews[
    "review_comment_title"
].fillna("No Title")

reviews["review_comment_message"] = reviews[
    "review_comment_message"
].fillna("No Comment")

# =====================================================
# Products Cleaning
# =====================================================

# Replace missing category with "Unknown"
products["product_category_name"] = products[
    "product_category_name"
].fillna("Unknown")

# NOTE:
# We intentionally DO NOT fill missing numeric values
# (weight, dimensions, name length, description length,
# photo quantity).
#
# Reason:
# - They represent genuine missing information.
# - Imputing values like median would create artificial data.
# - These columns are not required for the current analytics.
# - They can be handled later if a specific ML model requires it.

# =====================================================
# Standardize Column Names
# =====================================================

datasets = [
    customers,
    orders,
    order_items,
    payments,
    products,
    reviews,
    sellers,
    geolocation,
    category_translation
]

for df in datasets:
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

# =====================================================
# Create Processed Folder
# =====================================================

os.makedirs("data/processed", exist_ok=True)

# =====================================================
# Save Cleaned Data
# =====================================================

customers.to_csv(
    "data/processed/customers_clean.csv",
    index=False
)

orders.to_csv(
    "data/processed/orders_clean.csv",
    index=False
)

order_items.to_csv(
    "data/processed/order_items_clean.csv",
    index=False
)

payments.to_csv(
    "data/processed/payments_clean.csv",
    index=False
)

products.to_csv(
    "data/processed/products_clean.csv",
    index=False
)

reviews.to_csv(
    "data/processed/reviews_clean.csv",
    index=False
)

sellers.to_csv(
    "data/processed/sellers_clean.csv",
    index=False
)

geolocation.to_csv(
    "data/processed/geolocation_clean.csv",
    index=False
)

category_translation.to_csv(
    "data/processed/category_translation_clean.csv",
    index=False
)

# =====================================================
# Cleaning Summary
# =====================================================

print("=" * 60)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nCleaned datasets saved to:")
print("data/processed/")

print("\nFiles Created:")

files = [
    "customers_clean.csv",
    "orders_clean.csv",
    "order_items_clean.csv",
    "payments_clean.csv",
    "products_clean.csv",
    "reviews_clean.csv",
    "sellers_clean.csv",
    "geolocation_clean.csv",
    "category_translation_clean.csv"
]

for file in files:
    print(f"✔ {file}")

print("\nGeolocation aggregated by ZIP code.")
print("Review comments filled with default values.")
print("Missing product categories replaced with 'Unknown'.")
print("Numeric missing values intentionally preserved.")
print("\nETL Transform Phase Completed!")