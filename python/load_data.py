import pandas as pd
#load all datasets
customers=pd.read_csv("data/raw/olist_customers_dataset.csv")
orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
order_items = pd.read_csv("data/raw/olist_order_items_dataset.csv")
payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")
products = pd.read_csv("data/raw/olist_products_dataset.csv")
reviews = pd.read_csv("data/raw/olist_order_reviews_dataset.csv")
sellers = pd.read_csv("data/raw/olist_sellers_dataset.csv")
geolocation = pd.read_csv("data/raw/olist_geolocation_dataset.csv")
category_translation = pd.read_csv("data/raw/product_category_name_translation.csv")
#now store in dictionary
datasets={
    "Customers":customers,
    "Orders":orders,
    "Order Items":order_items,
    "Payments":payments,
    "Reviews": reviews,
    "Sellers": sellers,
    "Geolocation": geolocation,
    "Category Translation": category_translation
}
#now show the basic information
for name,df in datasets.items():
    print("="*60)
    print(name)
    print("="*60)
    print("Shape:",df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\n")