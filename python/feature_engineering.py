import pandas as pd
import numpy as np
import os
# for orders dataset
#load cleaned  dataset
customers = pd.read_csv(
    "data/processed/customers_clean.csv"
)

orders = pd.read_csv(
    "data/processed/orders_clean.csv",
    parse_dates=[
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
)

order_items = pd.read_csv(
    "data/processed/order_items_clean.csv",
    parse_dates=["shipping_limit_date"]
)

payments = pd.read_csv(
    "data/processed/payments_clean.csv"
)

products = pd.read_csv(
    "data/processed/products_clean.csv"
)

reviews = pd.read_csv(
    "data/processed/reviews_clean.csv",
    parse_dates=[
        "review_creation_date",
        "review_answer_timestamp"
    ]
)

sellers = pd.read_csv(
    "data/processed/sellers_clean.csv"
)

geolocation = pd.read_csv(
    "data/processed/geolocation_clean.csv"
)

category_translation = pd.read_csv(
    "data/processed/category_translation_clean.csv"
)
#delivery days
#delivered date-purchase date
orders["delivery_days"]=(
    orders["order_delivered_customer_date"]- orders["order_purchase_timestamp"]
).dt.days

#approval days
orders["approval_days"] = (
    orders["order_approved_at"]
    - orders["order_purchase_timestamp"]
).dt.days

#shipping days
orders["approval_days"] = (
    orders["order_approved_at"]
    - orders["order_purchase_timestamp"]
).dt.days

#estimated delivery date
orders["estimated_delivery_gap"] = (
    orders["order_delivered_customer_date"]
    - orders["order_estimated_delivery_date"]
).dt.days

#order year
orders["order_year"] = (
    orders["order_purchase_timestamp"]
    .dt.year
)
#order month
orders["order_month"] = (
    orders["order_purchase_timestamp"]
    .dt.month
)
#month name
orders["order_month_name"] = (
    orders["order_purchase_timestamp"]
    .dt.month_name()
)
#day name
orders["order_day_name"] = (
    orders["order_purchase_timestamp"]
    .dt.day_name()
)
#quarter
orders["order_quarter"] = (
    orders["order_purchase_timestamp"]
    .dt.quarter
)
#weekend order
orders["is_weekend"] = (
    orders["order_purchase_timestamp"]
    .dt.dayofweek >= 5
)
#delivered flag
orders["is_delivered"] = (
    orders["order_status"] == "delivered"
)
#checking new features
print("=" * 60)
print("NEW FEATURES CREATED")
print("=" * 60)

print(orders.head())

print("\n")

print(orders.info())

#saving the dataset

os.makedirs("data/featured",
            exist_ok=True
)
orders.to_csv(
    "data/featured/orders_featured.csv",
    index=False
)

#for order items dataset

#Total Item Value -The actual cost associated with an item.->Price + Freight
order_items["total_item_value"] = (
    order_items["price"] +
    order_items["freight_value"]
)
#Freight Percentage-How much of the total price is spent on shipping?
#(Freight / Price) × 100
order_items["freight_percentage"] = (
    order_items["freight_value"]
    / order_items["price"]
) * 100

#Sometimes price can be zero, so avoid division by zero
order_items["freight_percentage"] = np.where(
    order_items["price"] > 0,
    (order_items["freight_value"] / order_items["price"]) * 100,
    0
)
#simulated profit...Cost = 70% of Selling Price....Profit = Selling Price − Cost
order_items["estimated_profit"] = (
    order_items["price"] * 0.30
)
#Profit Margin %
order_items["profit_margin_percent"] = (
    order_items["estimated_profit"]
    / order_items["price"]
) * 100

#Shipping Month
order_items["shipping_month"] = (
    order_items["shipping_limit_date"]
    .dt.month
)
#Shipping Year
order_items["shipping_year"] = (
    order_items["shipping_limit_date"]
    .dt.year
)
#Shipping Quarter
order_items["shipping_quarter"] = (
    order_items["shipping_limit_date"]
    .dt.quarter
)
#Shipping Month Name
order_items["shipping_month_name"] = (
    order_items["shipping_limit_date"]
    .dt.month_name()
)
print("=" * 60)
print("ORDER ITEMS FEATURES")
print("=" * 60)

print(order_items.head())

print()

print(order_items.info())
order_items.to_csv(
    "data/featured/order_items_featured.csv",
    index=False
)

# Customer dataset Feature Engineering
customer_orders = (
    customers
    .merge(
        orders,
        on="customer_id",
        how="inner"
    )
    .merge(
        order_items,
        on="order_id",
        how="inner"
    )
)
# Total Revenue per Customer
customer_revenue = (
    customer_orders
    .groupby("customer_unique_id")["price"]
    .sum()
    .reset_index()
)

customer_revenue.rename(
    columns={
        "price": "total_revenue"
    },
    inplace=True
)
# Total Orders per Customer
customer_total_orders = (
    customer_orders
    .groupby("customer_unique_id")["order_id"]
    .nunique()
    .reset_index()
)

customer_total_orders.rename(
    columns={
        "order_id": "total_orders"
    },
    inplace=True
)

# Last Purchase Date
customer_last_purchase = (
    customer_orders
    .groupby("customer_unique_id")[
        "order_purchase_timestamp"
    ]
    .max()
    .reset_index()
)

customer_last_purchase.rename(
    columns={
        "order_purchase_timestamp":
        "last_purchase_date"
    },
    inplace=True
)

# Merge Customer Features
customer_features = (
    customer_revenue
    .merge(
        customer_total_orders,
        on="customer_unique_id"
    )
    .merge(
        customer_last_purchase,
        on="customer_unique_id"
    )
)
# Average Order Value (AOV)

customer_features["average_order_value"] = (
    customer_features["total_revenue"]
    / customer_features["total_orders"]
)

# RFM Analysis
# Reference Date (latest purchase in dataset)
reference_date = (
    customer_orders[
        "order_purchase_timestamp"
    ].max()
)

# Recency

customer_features["recency_days"] = (
    reference_date
    - customer_features["last_purchase_date"]
).dt.days

# Frequency

customer_features["frequency"] = (
    customer_features["total_orders"]
)

# Monetary

customer_features["monetary"] = (
    customer_features["total_revenue"]
)
print("=" * 60)
print("CUSTOMER FEATURES")
print("=" * 60)

print(customer_features.head())

print()

print(customer_features.info())
customer_features.to_csv(
    "data/featured/customer_features.csv",
    index=False
)

#Product Feature Engineering

product_data = (
    order_items
    .merge(
        products,
        on="product_id",
        how="left"
    )
)
#Total Quantity Sold - How many units of each product were sold?
product_quantity = (
    product_data
    .groupby("product_id")["order_item_id"]
    .count()
    .reset_index()
)

product_quantity.rename(
    columns={
        "order_item_id": "quantity_sold"
    },
    inplace=True
)
#Total Revenue
product_revenue = (
    product_data
    .groupby("product_id")["price"]
    .sum()
    .reset_index()
)

product_revenue.rename(
    columns={
        "price": "total_revenue"
    },
    inplace=True
)

#Average Selling Price
product_avg_price = (
    product_data
    .groupby("product_id")["price"]
    .mean()
    .reset_index()
)

product_avg_price.rename(
    columns={
        "price": "average_price"
    },
    inplace=True
)

#Average Freight Cost
product_avg_freight = (
    product_data
    .groupby("product_id")["freight_value"]
    .mean()
    .reset_index()
)

product_avg_freight.rename(
    columns={
        "freight_value": "average_freight"
    },
    inplace=True
)

#Number of Sellers
product_sellers = (
    product_data
    .groupby("product_id")["seller_id"]
    .nunique()
    .reset_index()
)

product_sellers.rename(
    columns={
        "seller_id": "seller_count"
    },
    inplace=True
)

#Merge Everything
product_features = (
    products
    .merge(
        product_quantity,
        on="product_id",
        how="left"
    )
    .merge(
        product_revenue,
        on="product_id",
        how="left"
    )
    .merge(
        product_avg_price,
        on="product_id",
        how="left"
    )
    .merge(
        product_avg_freight,
        on="product_id",
        how="left"
    )
    .merge(
        product_sellers,
        on="product_id",
        how="left"
    )
)

#Fill Missing Values
columns = [
    "quantity_sold",
    "total_revenue",
    "average_price",
    "average_freight",
    "seller_count"
]

product_features[columns] = (
    product_features[columns]
    .fillna(0)
)


print("=" * 60)
print("PRODUCT FEATURES")
print("=" * 60)

print(product_features.head())

print()

print(product_features.info())

product_features.to_csv(
    "data/featured/product_features.csv",
    index=False
)

#Seller Feature Engineering

#Merge Required Tables
seller_data = (
    order_items
    .merge(
        orders,
        on="order_id",
        how="left"
    )
)

# Total Orders Handled
seller_orders = (
    seller_data
    .groupby("seller_id")["order_id"]
    .nunique()
    .reset_index()
)

seller_orders.rename(
    columns={
        "order_id": "total_orders"
    },
    inplace=True
)

#Total Products Sold
seller_products = (
    seller_data
    .groupby("seller_id")["order_item_id"]
    .count()
    .reset_index()
)

seller_products.rename(
    columns={
        "order_item_id": "products_sold"
    },
    inplace=True
)

# Total Revenue
seller_revenue = (
    seller_data
    .groupby("seller_id")["price"]
    .sum()
    .reset_index()
)

seller_revenue.rename(
    columns={
        "price": "total_revenue"
    },
    inplace=True
)

#Average Selling Price
seller_avg_price = (
    seller_data
    .groupby("seller_id")["price"]
    .mean()
    .reset_index()
)

seller_avg_price.rename(
    columns={
        "price": "average_price"
    },
    inplace=True
)

#Average Freight Cost
seller_avg_freight = (
    seller_data
    .groupby("seller_id")["freight_value"]
    .mean()
    .reset_index()
)

seller_avg_freight.rename(
    columns={
        "freight_value": "average_freight"
    },
    inplace=True
)

#Average Delivery Time
seller_data["delivery_days"] = (
    seller_data["order_delivered_customer_date"]
    - seller_data["order_purchase_timestamp"]
).dt.days

seller_delivery = (
    seller_data
    .groupby("seller_id")["delivery_days"]
    .mean()
    .reset_index()
)

seller_delivery.rename(
    columns={
        "delivery_days": "average_delivery_days"
    },
    inplace=True
)

#Merge Everything
seller_features = (
    sellers
    .merge(
        seller_orders,
        on="seller_id",
        how="left"
    )
    .merge(
        seller_products,
        on="seller_id",
        how="left"
    )
    .merge(
        seller_revenue,
        on="seller_id",
        how="left"
    )
    .merge(
        seller_avg_price,
        on="seller_id",
        how="left"
    )
    .merge(
        seller_avg_freight,
        on="seller_id",
        how="left"
    )
    .merge(
        seller_delivery,
        on="seller_id",
        how="left"
    )
)

#Fill Missing Values
seller_features = seller_features.fillna(0)

print("=" * 60)
print("SELLER FEATURES")
print("=" * 60)

print(seller_features.head())

print()

print(seller_features.info())

seller_features.to_csv(
    "data/featured/seller_features.csv",
    index=False
)



#ABC Inventory Analysis

#Sort Products by Revenue
# =====================================================
# ABC Analysis
# =====================================================

# Why ABC Analysis?

# ABC Analysis is one of the most common inventory optimization techniques used by companies like:

# Lowe's
# Amazon
# Walmart
# Blue Yonder
# Target

# It classifies products based on their revenue contribution.

# A Products → Top ~80% of revenue (highest priority)
# B Products → Next ~15% of revenue
# C Products → Remaining ~5% of revenue

# This helps businesses decide:

# Which products need the most attention
# Which products should always stay in stock
# Where to invest inventory budget

abc_analysis = (
    product_features
    .sort_values(
        by="total_revenue",
        ascending=False
    )
    .reset_index(drop=True)
)

#Total Revenue
total_revenue = abc_analysis["total_revenue"].sum()

#Revenue Percentage
abc_analysis["revenue_percentage"] = (
    abc_analysis["total_revenue"]
    / total_revenue
) * 100
#Cumulative Revenue
abc_analysis["cumulative_percentage"] = (
    abc_analysis["revenue_percentage"]
    .cumsum()
)

#Create ABC Categories
def classify_abc(value):

    if value <= 80:
        return "A"

    elif value <= 95:
        return "B"

    else:
        return "C"

abc_analysis["abc_category"] = (
    abc_analysis["cumulative_percentage"]
    .apply(classify_abc)
)  

#Check Distribution
print("=" * 60)
print("ABC ANALYSIS")
print("=" * 60)

print()

print(
    abc_analysis["abc_category"]
    .value_counts()
)

print()

print(
    abc_analysis[
        [
            "product_id",
            "total_revenue",
            "cumulative_percentage",
            "abc_category"
        ]
    ].head(20)
)

#Save Dataset
abc_analysis.to_csv(
    "data/featured/abc_analysis.csv",
    index=False
)

#=====================================
#Demand Forecasting Dataset Creation
#======================================
# Objective

# Before building a forecasting model, we need to prepare a time-series dataset.

# We'll aggregate order data at the monthly × product category level.

# The output will later be used for:

# ARIMA
# Prophet
# XGBoost
# Random Forest
# Power BI Forecasting

#Merge Required Tables
# =====================================================
# Demand Forecasting Dataset
# =====================================================

forecast_data = (
    orders
    .merge(
        order_items,
        on="order_id",
        how="inner"
    )
    .merge(
        products,
        on="product_id",
        how="left"
    )
)

#Add Year
forecast_data["year"] = (
    forecast_data["order_purchase_timestamp"]
    .dt.year
)
#Add Month
forecast_data["month"] = (
    forecast_data["order_purchase_timestamp"]
    .dt.month
)
#add month name
forecast_data["month_name"] = (
    forecast_data["order_purchase_timestamp"]
    .dt.month
)
#create year month column
forecast_data["year_month"] = (
    forecast_data["order_purchase_timestamp"]
    .dt.to_period("M")
    .astype(str)
)

#Aggregate Monthly Demand
monthly_demand = (
    forecast_data
    .groupby(
        [
            "year_month",
            "product_category_name"
        ]
    )
    .agg(
        total_quantity=(
            "order_item_id",
            "count"
        ),
        total_sales=(
            "price",
            "sum"
        ),
        average_price=(
            "price",
            "mean"
        )
    )
    .reset_index()
)

print("=" * 60)
print("MONTHLY DEMAND DATASET")
print("=" * 60)

print(monthly_demand.head())

print()

print(monthly_demand.info())

monthly_demand = monthly_demand.sort_values(
    by=[
        "year_month",
        "product_category_name"
    ]
).reset_index(drop=True)
monthly_demand.to_csv(
    "data/featured/monthly_demand.csv",
    index=False
)

# Why are we creating this dataset?

# This is the training dataset for demand forecasting.

# Instead of predicting demand from raw transactional data, we'll forecast using monthly aggregated demand, which is how retail forecasting is commonly performed.



# =====================================================
# Feature Engineering Summary
# =====================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFeature Engineered Datasets Created:")

feature_files = [
    "orders_featured.csv",
    "order_items_featured.csv",
    "customer_features.csv",
    "product_features.csv",
    "seller_features.csv",
    "abc_analysis.csv",
    "monthly_demand.csv"
]

for file in feature_files:
    print(f"✔ {file}")

print("\nLocation:")
print("data/featured/")

print("\nFeatures Created:")
print("✔ Order Features")
print("✔ Order Item Features")
print("✔ Customer Features (RFM)")
print("✔ Product Features")
print("✔ Seller Features")
print("✔ ABC Inventory Classification")
print("✔ Monthly Demand Forecasting Dataset")

print("\nProject is now ready for:")
print("✔ Exploratory Data Analysis (EDA)")
print("✔ Power BI Dashboard")
print("✔ Machine Learning")
print("✔ Demand Forecasting")
print("✔ Customer Segmentation")

print("\nFeature Engineering Phase Completed!")