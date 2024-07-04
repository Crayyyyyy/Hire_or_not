import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = 'recruitment_data.csv'
data = pd.read_csv(csv_path)

INPUT = data.drop('HiringDecision', axis=1)
OUT = data['HiringDecision']

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ['RecruitmentStrategy']
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough'
)

INPUT_train, INPUT_test, OUT_train, OUT_test  = train_test_split(INPUT, OUT, train_size=0.85, random_state=13)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1200, random_state=13))])
model.fit(INPUT_train, OUT_train)

OUT_predict = model.predict(INPUT_test)
acc = accuracy_score(OUT_test, OUT_predict)

print(acc)