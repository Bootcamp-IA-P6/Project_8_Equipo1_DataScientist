from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(X, model_ty="linear"):

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    num_transformer = StandardScaler() if model_ty == "linear" else "passthrough"

    return ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])