"""
generate_sample_data.py
Generate airline customer sample data for RFM clustering analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate customer data
n_customers = 5000

# Generate customer IDs
customer_ids = [f'CUST_{str(i).zfill(6)}' for i in range(1, n_customers + 1)]


# Simulate different customer behavior patterns
def generate_customer_data(n):
    data = []
    current_date = datetime(2024, 12, 31)

    for i in range(n):
        customer_id = customer_ids[i]

        # Randomly assign customer type
        customer_type = np.random.choice(
            ['HighValue', 'Potential', 'Regular', 'AtRisk'],
            p=[0.15, 0.25, 0.35, 0.25]
        )

        # Generate RFM features based on customer type
        if customer_type == 'HighValue':
            recency_days = np.random.randint(1, 15)
            frequency = np.random.randint(15, 30)
            monetary = np.random.uniform(5000, 20000)
            membership_years = np.random.uniform(2, 8)
            discount_usage = np.random.uniform(0.1, 0.3)

        elif customer_type == 'Potential':
            recency_days = np.random.randint(5, 30)
            frequency = np.random.randint(8, 15)
            monetary = np.random.uniform(3000, 8000)
            membership_years = np.random.uniform(1, 4)
            discount_usage = np.random.uniform(0.2, 0.4)

        elif customer_type == 'Regular':
            recency_days = np.random.randint(15, 60)
            frequency = np.random.randint(3, 10)
            monetary = np.random.uniform(1000, 4000)
            membership_years = np.random.uniform(0.5, 3)
            discount_usage = np.random.uniform(0.3, 0.6)

        else:  # AtRisk
            recency_days = np.random.randint(45, 180)
            frequency = np.random.randint(1, 5)
            monetary = np.random.uniform(500, 2000)
            membership_years = np.random.uniform(1, 5)
            discount_usage = np.random.uniform(0.4, 0.8)

        # Calculate last purchase date
        last_purchase_date = current_date - timedelta(days=int(recency_days))

        # Generate other features
        age = np.random.randint(22, 65)
        gender = np.random.choice(['Male', 'Female'], p=[0.55, 0.45])
        city_tier = np.random.choice(['Tier1', 'Tier2', 'Tier3', 'Other'], p=[0.25, 0.35, 0.25, 0.15])

        # Average amount per trip
        avg_amount_per_trip = monetary / frequency if frequency > 0 else monetary

        # Satisfaction score (1-5)
        satisfaction_score = np.random.choice(
            [3, 4, 5],
            p=[0.2, 0.4, 0.4] if customer_type == 'HighValue' else [0.4, 0.35, 0.25]
        )

        data.append({
            'CustomerID': customer_id,
            'MembershipYears': round(membership_years, 1),
            'Age': age,
            'Gender': gender,
            'CityTier': city_tier,
            'LastPurchaseDate': last_purchase_date.strftime('%Y-%m-%d'),
            'Frequency': frequency,
            'TotalAmount': round(monetary, 2),
            'AvgAmountPerTrip': round(avg_amount_per_trip, 2),
            'DiscountUsage': round(discount_usage, 2),
            'SatisfactionScore': satisfaction_score,
            'CustomerType': customer_type
        })

    return pd.DataFrame(data)


# Generate data
df = generate_customer_data(n_customers)

# Add some missing values (simulate real data)
for col in ['Age', 'SatisfactionScore']:
    mask = np.random.random(len(df)) < 0.02
    df.loc[mask, col] = np.nan

# Save as CSV file
df.to_csv('airline_customers.csv', index=False, encoding='utf-8')
print(f"Generated {len(df)} customer records, saved as airline_customers.csv")

# Save as Excel file
df.to_excel('airline_customers.xlsx', index=False)
print("Also generated airline_customers.xlsx")

# Display data preview
print("\nData preview:")
print(df.head(10))
print(f"\nBasic information:")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nData types:")
print(df.dtypes)
