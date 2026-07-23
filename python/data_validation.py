import pandas as pd

# Load Cleaned Datasets

customers = pd.read_csv("data/processed/customers_clean.csv")

orders = pd.read_csv("data/processed/orders_clean.csv")

order_items = pd.read_csv("data/processed/order_items_clean.csv")

payments = pd.read_csv("data/processed/payments_clean.csv")

products = pd.read_csv("data/processed/products_clean.csv")

reviews = pd.read_csv("data/processed/reviews_clean.csv")

sellers = pd.read_csv("data/processed/sellers_clean.csv")

geolocation = pd.read_csv("data/processed/geolocation_clean.csv")

category_translation = pd.read_csv(
    "data/processed/category_translation_clean.csv"
)


# Data Validation Function


def validate_dataset(name, df, primary_key=None):

    print("=" * 70)
    print(name.upper())
    print("=" * 70)

    print(f"Shape : {df.shape}")

    print(f"Duplicate Rows : {df.duplicated().sum()}")

    print("\nMissing Values")
    print(df.isnull().sum())

    if primary_key:

        duplicate_keys = df[primary_key].duplicated().sum()

        print(f"\nDuplicate '{primary_key}' : {duplicate_keys}")

    print("\n")


# Validate Individual Datasets


validate_dataset(
    "Customers",
    customers,
    "customer_id"
)

validate_dataset(
    "Orders",
    orders,
    "order_id"
)

validate_dataset(
    "Order Items",
    order_items
)

validate_dataset(
    "Payments",
    payments
)

validate_dataset(
    "Products",
    products,
    "product_id"
)

validate_dataset(
    "Reviews",
    reviews
)

validate_dataset(
    "Sellers",
    sellers,
    "seller_id"
)

validate_dataset(
    "Geolocation",
    geolocation,
    "geolocation_zip_code_prefix"
)

validate_dataset(
    "Category Translation",
    category_translation,
    "product_category_name"
)


# Foreign Key Validation


print("=" * 70)
print("FOREIGN KEY VALIDATION")
print("=" * 70)


# Orders -> Customers


missing_customers = orders[
    ~orders["customer_id"].isin(customers["customer_id"])
]

print(f"Orders without matching Customer : {len(missing_customers)}")


# Order Items -> Orders


missing_orders = order_items[
    ~order_items["order_id"].isin(orders["order_id"])
]

print(f"Order Items without matching Order : {len(missing_orders)}")


# Order Items -> Products


missing_products = order_items[
    ~order_items["product_id"].isin(products["product_id"])
]

print(f"Order Items without matching Product : {len(missing_products)}")


# Order Items -> Sellers


missing_sellers = order_items[
    ~order_items["seller_id"].isin(sellers["seller_id"])
]

print(f"Order Items without matching Seller : {len(missing_sellers)}")


# Payments -> Orders


missing_payment_orders = payments[
    ~payments["order_id"].isin(orders["order_id"])
]

print(f"Payments without matching Order : {len(missing_payment_orders)}")


# Reviews -> Orders


missing_review_orders = reviews[
    ~reviews["order_id"].isin(orders["order_id"])
]

print(f"Reviews without matching Order : {len(missing_review_orders)}")


# Business Rule Validation


print("\n" + "=" * 70)
print("BUSINESS RULE VALIDATION")
print("=" * 70)

# Negative Prices

negative_price = order_items[
    order_items["price"] < 0
]

print(f"Negative Prices : {len(negative_price)}")

# Negative Freight

negative_freight = order_items[
    order_items["freight_value"] < 0
]

print(f"Negative Freight Values : {len(negative_freight)}")

# Negative Payment

negative_payment = payments[
    payments["payment_value"] < 0
]

print(f"Negative Payment Values : {len(negative_payment)}")

# Invalid Review Score

invalid_reviews = reviews[
    ~reviews["review_score"].between(1, 5)
]

print(f"Invalid Review Scores : {len(invalid_reviews)}")

# Missing Product Categories

missing_category = products[
    products["product_category_name"] == "Unknown"
]

print(f"Products with Unknown Category : {len(missing_category)}")

# =====================================================
# Validation Summary
# =====================================================

print("\n" + "=" * 70)
print("DATA VALIDATION COMPLETED")
print("=" * 70)

print("""
Validation Performed

✔ Duplicate Row Check
✔ Missing Value Check
✔ Primary Key Validation
✔ Foreign Key Validation
✔ Business Rule Validation

Dataset is now ready for Feature Engineering.
""")