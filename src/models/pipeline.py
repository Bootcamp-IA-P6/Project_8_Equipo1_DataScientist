from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def build_pipeline(model, preprocessor, use_smote=False):

    steps = [('prep', preprocessor)]

    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))

    steps.append(('model', model))

    return Pipeline(steps)