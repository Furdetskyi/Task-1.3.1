import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Дані: активність користувачів бібліотеки
data = {
    'logins_per_week': [15, 2, 8, 20, 1, 12, 6, 18, 0, 5],
    'avg_read_time': [120, 10, 45, 150, 5, 90, 30, 140, 0, 20],
    'downloads': [5, 0, 1, 6, 0, 3, 1, 4, 0, 1],
    'active': [1,0,1,1,0,1,0,1,0,0]  # ціль
}
df = pd.DataFrame(data)

X = df[['logins_per_week','avg_read_time','downloads']]
y = df['active']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Логістична регресія
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Логістична регресія точність:", accuracy_score(y_test, log_pred))
print("Random Forest точність:", accuracy_score(y_test, rf_pred))
