import pandas as pd
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
#dict
datasets = {
    "Customers": customers,
    "Orders": orders,
    "Order Items": order_items,
    "Payments": payments,
    "Products": products,
    "Reviews": reviews,
    "Sellers": sellers,
    "Geolocation": geolocation,
    "Category Translation": category_translation
}

# Data Quality Report
for name, df in datasets.items():

    print("=" * 70)
    print(name.upper())
    print("=" * 70)

    print(f"Rows, Columns : {df.shape}")

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nMemory Usage")
    print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

    print("\n")