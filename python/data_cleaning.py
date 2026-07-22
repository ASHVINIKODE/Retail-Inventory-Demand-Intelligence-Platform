import pandas as pd
import os
#load dataset
customers = pd.read_csv("data/raw/olist_customers_dataset.csv")
orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
order_items = pd.read_csv("data/raw/olist_order_items_dataset.csv")
payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")
products = pd.read_csv("data/raw/olist_products_dataset.csv")
reviews = pd.read_csv("data/raw/olist_order_reviews_dataset.csv")
sellers = pd.read_csv("data/raw/olist_sellers_dataset.csv")
geolocation = pd.read_csv("data/raw/olist_geolocation_dataset.csv")
category_translation = pd.read_csv("data/raw/product_category_name_translation.csv")
#convert date column
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

#GEOLOCATION conatins 261831 duplicate values but it contains different latitude and longitude so the best approach is to average the latitude and longitude for each ZIP code
geolocation=(geolocation
             .groupby("geolocation_zip_code_prefix",as_index=False)
             .agg({
                "geolocation_lat": "mean",
                "geolocation_lng": "mean",
                "geolocation_city": "first",
                "geolocation_state": "first"
             })
             )

reviews["review_comment_title"] = reviews[
    "review_comment_title"
].fillna("No Title")

reviews["review_comment_message"] = reviews[
    "review_comment_message"
].fillna("No Comment")

# Products

products["product_category_name"] = products[
    "product_category_name"
].fillna("Unknown")
# Keep numeric missing values for now.
# We'll handle them during Feature Engineering.

# Standardize Column Names
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
    df.columns=(
        df.columns.str.lower().str.strip().str.replace(" ","_")
    )
#creating processed folder
os.makedirs("data/processed",exist_ok=True)
#save cleaned datasets
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
#cleaning summary
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

print("\nDuplicate rows removed from Geolocation dataset.")

print("Review comments filled with default values.")

print("Product category missing values replaced with 'Unknown'.")

print("\nETL Transform Phase Completed!")