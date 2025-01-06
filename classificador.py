#load useful libaries
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


#  separando caracteristicas e valores alvos
diabetes_data = pd.read_csv('database/diabetes.csv')

x = diabetes_data.drop(columns=['Outcome'])
y = diabetes_data['Outcome']

#criando dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=13)

#criar o modelo
rf_clf = RandomForestClassifier(n_estimators=100, max_features=2, bootstrap=True)
rf_clf.fit(X_train, y_train)

#criando predições para o dados de teste
y_pred = rf_clf.predict(X_test)

print(classification_report(y_test, y_pred))