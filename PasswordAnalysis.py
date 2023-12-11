import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import files
import argparse
#data = files.upload()

#sample_size = 10000
parser = argparse.ArgumentParser(description='Process the path to the password file.')
parser.add_argument('file_path', type=str, help='The path to the password file')
args = parser.parse_args()

password_data = pd.read_csv(args.file_path, header=None, names=['password'], encoding='latin-1', error_bad_lines=False)

password_data['password'] = password_data['password'].astype(str)

password_data['length'] = password_data['password'].apply(len)
password_data['has_number'] = password_data['password'].apply(lambda x: any(char.isdigit() for char in x))
password_data['has_special'] = password_data['password'].apply(lambda x: bool(re.search('[^A-Za-z0-9]', x)))
password_data['has_uppercase'] = password_data['password'].apply(lambda x: any(char.isupper() for char in x))

sequential_patterns = re.compile(r'(012|123|234|345|456|567|678|789|890|abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mn|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)')
password_data['has_sequential'] = password_data['password'].apply(lambda x: bool(sequential_patterns.search(x.lower())))

plt.figure(figsize=(10, 6))
sns.histplot(password_data['length'], bins=30, kde=False)
plt.title('Password Length Distribution')
plt.xlabel('Length of Password')
plt.ylabel('Frequency')
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

sns.countplot(x='has_number', data=password_data, ax=axs[0, 0])
axs[0, 0].set_title('Presence of Numbers')

sns.countplot(x='has_special', data=password_data, ax=axs[0, 1])
axs[0, 1].set_title('Presence of Special Characters')

sns.countplot(x='has_uppercase', data=password_data, ax=axs[1, 0])
axs[1, 0].set_title('Presence of Uppercase Letters')

sns.countplot(x='has_sequential', data=password_data, ax=axs[1, 1])
axs[1, 1].set_title('Presence of Sequential Characters')

plt.tight_layout()
plt.show()

X = password_data[['length', 'has_number', 'has_special', 'has_uppercase', 'has_sequential']]
y = (password_data['length'] >= 8) & password_data['has_number'] & password_data['has_special'] & password_data['has_uppercase'] & ~password_data['has_sequential']
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

importances = model.feature_importances_
print("Feature Importances:", importances)

feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
sorted_feature_importances = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=sorted_feature_importances, y=sorted_feature_importances.index)
plt.title('Feature Importances in the Password Strength Prediction Model')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.show()