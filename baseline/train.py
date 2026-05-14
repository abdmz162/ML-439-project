import pandas as pd
from preprocessor import Preprocessor
from logistic_regression import LogisticRegression

def main():
    # Load training data
    train_df = pd.read_csv('dev_df.csv')

    # Preprocess data
    preprocessor = Preprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)

    # Train model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)

    # Report training status
    print("Training completed.")
    y_train_pred = model.predict(X_train)
    train_accuracy = (y_train_pred == y_train).mean()
    print(f"Train accuracy: {train_accuracy:.4f}")

if __name__ == "__main__":
    main()