import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    try:
        df = pd.read_csv("Student Mental health.csv")
    except Exception as e:
        print(f"Could not load data: {e}")
        return

    # Assuming 'Depression' or last col is target
    target_cols = [col for col in df.columns if 'depression' in col.lower() or 'anxiety' in col.lower() or 'panic' in col.lower()]
    target_col = target_cols[0] if target_cols else df.columns[-1]
    print(f"Using {target_col} as target.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
