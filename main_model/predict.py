
import pandas as pd

from preprocess import TitanicPreprocessor
from random_forest import RandomForest


train_df = pd.read_csv("dev_df.csv")
test_df = pd.read_csv("test_df.csv")

preprocessor = TitanicPreprocessor()

train_df = preprocessor.fit_transform(train_df)
test_df = preprocessor.transform(test_df)

X_train = train_df.drop(columns=["survived"]).values
y_train = train_df["survived"].values

X_test = test_df.drop(columns=["survived"]).values

model = RandomForest(
    n_estimators=20,
    max_depth=8,
    min_samples_split=4
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

output = pd.DataFrame({
    "Prediction": predictions
})

output.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")
