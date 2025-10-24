import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Приклад даних (кількість логінів, час читання, кількість завантажень)
df = pd.DataFrame({
    'logins_per_week':[15,2,8,20,1,12,6,18,0,5],
    'avg_read_time_mins':[120,10,45,150,5,90,30,140,0,20],
    'downloads':[5,0,1,6,0,3,1,4,0,1]
})

# Нормалізація даних
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Кластеризація (наприклад, на 3 групи)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
df['cluster'] = kmeans.labels_

# Оцінка якості кластеризації
silhouette = silhouette_score(X, kmeans.labels_)
dbi = davies_bouldin_score(X, kmeans.labels_)

print("Silhouette Score:", silhouette)
print("Davies-Bouldin Index:", dbi)
print(df)
