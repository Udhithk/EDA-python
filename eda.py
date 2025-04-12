
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("python project.csv", encoding='latin1')

# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# Drop rows where 'Order Date' or 'Sales' is missing
df = df.dropna(subset=['Order Date', 'Sales'])

# Set seaborn theme
sns.set(style='whitegrid', palette='Set2')


# 1. EDA - Basic Overview
print("Dataset shape:", df.shape)
print("\nColumn info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Value counts for important categorical columns
print("\nSegment distribution:\n", df['Segment'].value_counts())
print("\nRegion distribution:\n", df['Region'].value_counts())
print("\nCategory distribution:\n", df['Category'].value_counts())

# Distribution of Sub-Categories
print("\nSub-Category distribution:\n", df['Sub-Category'].value_counts())

# Top 10 Most Frequently Ordered Products
print("\nTop 10 Most Frequently Ordered Products:\n", df['Product Name'].value_counts().head(10))

# Top 10 Most Active Customers (by number of orders)
print("\nTop 10 Customers by Order Frequency:\n", df['Customer ID'].value_counts().head(10))

# Most common Ship Modes (if present)
if 'Ship Mode' in df.columns:
    print("\nShipping Mode distribution:\n", df['Ship Mode'].value_counts())

# Sales summary by Discount level (grouped)
if 'Discount' in df.columns:
    print("\nUnique Discount Levels:\n", df['Discount'].value_counts().sort_index())



# 2. TIME SERIES: Monthly Sales Trend
df['Month'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].astype(str)

plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o')
plt.xticks(rotation=45)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()


# 3. TOP PRODUCTS (Sales and Quantity)
# Top 10 Products by Sales
top_sales = df.groupby('Product Name')['Sales'].sum().nlargest(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_sales.values, y=top_sales.index, palette='viridis')
plt.title("Top 10 Products by Total Sales")
plt.xlabel("Sales")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

# Top 10 Products by Quantity
top_quantity = df.groupby('Product Name')['Quantity'].sum().nlargest(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_quantity.values, y=top_quantity.index, palette='magma')
plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Quantity Sold")
plt.ylabel("Product")
plt.tight_layout()
plt.show()


# 4. CUSTOMER SEGMENTATION
segment_sales = df.groupby('Segment')['Sales'].sum().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(data=segment_sales, x='Segment', y='Sales', palette='pastel')
plt.title("Sales by Customer Segment")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Pie chart for proportion of total sales
plt.figure(figsize=(6, 6))
plt.pie(segment_sales['Sales'], labels=segment_sales['Segment'], autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title("Sales Share by Segment")
plt.tight_layout()
plt.show()


# 5. REGION AND CATEGORY ANALYSIS
# Sales by Region
region_sales = df.groupby('Region')['Sales'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=region_sales, x='Region', y='Sales', palette='coolwarm')
plt.title("Sales by Region")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Sales by Category
category_sales = df.groupby('Category')['Sales'].sum().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(data=category_sales, x='Category', y='Sales', palette='Set3')
plt.title("Sales by Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Heatmap: Region vs Category
heatmap_data = df.pivot_table(index='Region', columns='Category', values='Sales', aggfunc='sum')

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Sales Heatmap: Region vs Category")
plt.tight_layout()
plt.show()





# 6. BOXPLOT ANALYSIS (Distribution of Sales and Profit)
# Boxplot: Sales distribution by Category
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Category', y='Sales', palette='Set2')
plt.title("Sales Distribution by Category")
plt.ylabel("Sales")
plt.xlabel("Category")
plt.tight_layout()
plt.show()

# Boxplot: Profit distribution by Region
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Region', y='Profit', palette='Set3')
plt.title("Profit Distribution by Region")
plt.ylabel("Profit")
plt.xlabel("Region")
plt.tight_layout()
plt.show()


# 8. CORRELATION ANALYSIS (Numerical Features)
# Selecting numerical columns for correlation
num_cols = ['Sales', 'Profit', 'Quantity', 'Discount']
corr_matrix = df[num_cols].corr()

# Plotting correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Sales, Profit, Quantity, Discount")
plt.tight_layout()
plt.show()
