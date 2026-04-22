"""
generate_sample_data.py
生成航空客户示例数据 - 适用于RFM聚类分析
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置随机种子，确保数据可复现
np.random.seed(42)

# 生成客户数据
n_customers = 5000

# 生成客户ID
customer_ids = [f'CUST_{str(i).zfill(6)}' for i in range(1, n_customers + 1)]


# 模拟不同客户群体的行为模式
def generate_customer_data(n):
    data = []
    current_date = datetime(2024, 12, 31)

    for i in range(n):
        customer_id = customer_ids[i]

        # 随机分配客户类型（用于模拟不同的消费模式）
        customer_type = np.random.choice(
            ['高价值', '潜力客户', '一般客户', '流失风险'],
            p=[0.15, 0.25, 0.35, 0.25]
        )

        # 根据客户类型生成不同的RFM特征
        if customer_type == '高价值':
            recency_days = np.random.randint(1, 15)  # 最近消费时间（天）
            frequency = np.random.randint(15, 30)  # 消费频率
            monetary = np.random.uniform(5000, 20000)  # 消费金额
            membership_years = np.random.uniform(2, 8)
            discount_usage = np.random.uniform(0.1, 0.3)

        elif customer_type == '潜力客户':
            recency_days = np.random.randint(5, 30)
            frequency = np.random.randint(8, 15)
            monetary = np.random.uniform(3000, 8000)
            membership_years = np.random.uniform(1, 4)
            discount_usage = np.random.uniform(0.2, 0.4)

        elif customer_type == '一般客户':
            recency_days = np.random.randint(15, 60)
            frequency = np.random.randint(3, 10)
            monetary = np.random.uniform(1000, 4000)
            membership_years = np.random.uniform(0.5, 3)
            discount_usage = np.random.uniform(0.3, 0.6)

        else:  # 流失风险
            recency_days = np.random.randint(45, 180)
            frequency = np.random.randint(1, 5)
            monetary = np.random.uniform(500, 2000)
            membership_years = np.random.uniform(1, 5)
            discount_usage = np.random.uniform(0.4, 0.8)

        # 计算最近消费日期
        last_purchase_date = current_date - timedelta(days=int(recency_days))

        # 生成其他特征
        age = np.random.randint(22, 65)
        gender = np.random.choice(['男', '女'], p=[0.55, 0.45])
        city_tier = np.random.choice(['一线', '二线', '三线', '其他'], p=[0.25, 0.35, 0.25, 0.15])

        # 平均每次消费金额
        avg_amount_per_trip = monetary / frequency if frequency > 0 else monetary

        # 满意度评分（1-5分）
        satisfaction_score = np.random.choice(
            [3, 4, 5],
            p=[0.2, 0.4, 0.4] if customer_type == '高价值' else [0.4, 0.35, 0.25]
        )

        data.append({
            '客户ID': customer_id,
            '会员年限': round(membership_years, 1),
            '年龄': age,
            '性别': gender,
            '城市等级': city_tier,
            '最近消费日期': last_purchase_date.strftime('%Y-%m-%d'),
            '消费频率': frequency,
            '总消费金额': round(monetary, 2),
            '平均单次消费': round(avg_amount_per_trip, 2),
            '折扣使用率': round(discount_usage, 2),
            '满意度评分': satisfaction_score,
            '客户类型标签': customer_type
        })

    return pd.DataFrame(data)


# 生成数据
df = generate_customer_data(n_customers)

# 添加一些缺失值（模拟真实数据）
for col in ['年龄', '满意度评分']:
    mask = np.random.random(len(df)) < 0.02  # 2%的缺失率
    df.loc[mask, col] = np.nan

# 保存为CSV文件
df.to_csv('airline_customers.csv', index=False, encoding='utf-8-sig')
print(f"已生成 {len(df)} 条客户数据，保存为 airline_customers.csv")

# 同时保存一份Excel格式
df.to_excel('airline_customers.xlsx', index=False)
print("已同时生成 airline_customers.xlsx")

# 显示数据预览
print("\n数据预览：")
print(df.head(10))
print(f"\n数据基本信息：")
print(f"总行数: {len(df)}")
print(f"总列数: {len(df.columns)}")
print(f"\n各列数据类型：")
print(df.dtypes)