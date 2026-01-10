import pandas as pd
import numpy as np

# 產生 200 筆模擬數據
np.random.seed(42)
n = 200
data = {
    'Weight': np.random.normal(70, 15, n),      # 體重
    'Duration': np.random.normal(30, 10, n),    # 運動時間 (分鐘)
    'Heart_Rate': np.random.normal(130, 20, n), # 平均心率
    'Body_Fat': np.random.uniform(10, 35, n),   # 體脂率
    'Calories': 0 # 待會計算
}
df = pd.DataFrame(data)

# 模擬一個簡單的公式來產生標籤 (Calories)
df['Calories'] = df['Duration'] * 0.1 * df['Heart_Rate'] + df['Weight'] * 0.5 + np.random.normal(0, 5, n)
df.to_csv('fitness_data.csv', index=False)
print("資料集 fitness_data.csv 已產生！")