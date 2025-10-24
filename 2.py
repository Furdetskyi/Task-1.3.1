import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Дані прикладні
data = {
    'logins_per_week': [1,2,0,15,18,20,3,0,2,25,30,1,0,22,5,6,7,0,19,21],
    'avg_read_time': [5,10,0,120,140,150,20,0,15,200,210,3,0,190,50,60,70,0,160,170],
    'downloads': [0,0,0,5,4,6,1,0,0,7,8,0,0,5,1,2,2,0,6,7],
    'active': [0,0,0,1,1,1,0,0,0,1,1,0,0,1,0,0,0,0,1,1]
}
df = pd.DataFrame(data)

X = df[['logins_per_week','avg_read_time','downloads']]
y = df['active']

# Балансування класів
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Навчання моделі
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
