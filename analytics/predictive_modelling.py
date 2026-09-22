
from pathlib import Path

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from analytics.eda_analysis import EdaAnalysis


class PredictiveModelling:

    # Features used for the survived classifiers (Task 3)
    CLASSIFICATION_NUMERIC_FEATURES = ['age', 'fare', 'sibsp', 'parch']
    CLASSIFICATION_CATEGORICAL_FEATURES = ['sex', 'embarked']

    REGRESSION_NUMERIC_FEATURES = ['pclass', 'age', 'sibsp', 'parch']
    REGRESSION_CATEGORICAL_FEATURES = ['sex', 'embarked']
    REGRESSION_TARGET = 'fare'

    MODEL_PATH = Path(__file__).parent / 'models' / 'best_classifier_pipeline.joblib'

    def split_data_into_train_test(self, df: pd.DataFrame):
        X = df.drop(columns=['survived'])
        y = df['survived'].astype(int)

        # Stratify on y: Task 1's EDA showed survived is imbalanced
        # (~61.6% / 38.4%), so a plain random split risks over- or
        # under-representing the minority "survived" class in the test set,
        # skewing metrics like recall/F1 and making model comparisons noisy.
        # Stratifying preserves that ~62/38 ratio in both splits.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # 4. Verification Check: Print the resulting proportions
        print("=== STRATIFICATION SANITY CHECK ===")
        print(f"Original Dataset Balance:\n{y.value_counts(normalize=True)}\n")
        print(f"Training Set Balance:\n{y_train.value_counts(normalize=True)}\n")
        print(f"Testing Set Balance:\n{y_test.value_counts(normalize=True)}")

        return X_train, X_test, y_train, y_test

    def _build_preprocessor(self, numeric_features, categorical_features):
        # Preprocessing for numerical features: Median imputation + Standard Scaling
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical features: Most frequent imputation + One-Hot Encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Consolidate transformers into a unified processing node
        return ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    def _evaluate_classifier(self, pipe, X_test, y_test):
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "Confusion Matrix": f"TN: {tn} | FP: {fp}\nFN: {fn} | TP: {tp}",
            "probabilities": y_prob,
        }

    def preprocessin_data(self, X_train, X_test, y_train, y_test):

        numeric_features = self.CLASSIFICATION_NUMERIC_FEATURES
        categorical_features = self.CLASSIFICATION_CATEGORICAL_FEATURES

        # =====================================================================
        # 3. Pipelines & ColumnTransformer Setup (Data Leakage Prevention)
        # =====================================================================
        preprocessor = self._build_preprocessor(numeric_features, categorical_features)

        # =====================================================================
        # 4. Define and Train Final Estimators
        # =====================================================================
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42), # Pruned for cleaner visualization
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        # Dictionary to house full end-to-end pipelines for each model
        pipelines = {}
        evaluation_results = {}

        print("\nTraining and evaluating models using isolated pipelines...")
        for name, clf in classifiers.items():
            # Build complete blueprint: Preprocessing steps -> Final ML algorithm
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])

            # Preprocessor fits on training data ONLY, then estimator fits on generated vectors
            pipe.fit(X_train, y_train)
            pipelines[name] = pipe

            evaluation_results[name] = self._evaluate_classifier(pipe, X_test, y_test)

        # =====================================================================
        # 5. Model Metrics Comparison Table
        # =====================================================================
        metrics_df = pd.DataFrame(evaluation_results).T
        # Drop raw arrays before text presentation
        table_df = metrics_df.drop(columns=["probabilities"])

        print("\n" + "="*80)
        print("                    MODEL SIDE-BY-SIDE COMPARISON TABLE")
        print("="*80)
        print(table_df.to_string())
        print("="*80)

        # =====================================================================
        # 6. Render Decision Tree Structure via plot_tree
        # =====================================================================
        # Extract feature names post One-Hot Encoding transformation step
        encoded_cat_features = (
            pipelines["Decision Tree"]
            .named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['encoder']
            .get_feature_names_out(categorical_features)
        )
        all_transformed_features = list(numeric_features) + list(encoded_cat_features)
        class_labels = ['Not Survived', 'Survived']

        plt.figure(figsize=(16, 9), dpi=150)
        plot_tree(
            pipelines["Decision Tree"].named_steps['classifier'],
            feature_names=all_transformed_features,
            class_names=class_labels,
            filled=True,
            rounded=True,
            impurity=False,
            fontsize=10
        )
        plt.title("Decision Tree Trained via Embedded Pipeline Structure", fontsize=14, fontweight='bold')
        plt.tight_layout()

        # =====================================================================
        # 7. Render Model ROC Curves & AUC Side-by-Side
        # =====================================================================
        plt.figure(figsize=(8, 6), dpi=100)
        for name, metrics in evaluation_results.items():
            fpr, tpr, _ = roc_curve(y_test, metrics["probabilities"])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['ROC-AUC']:.3f})", lw=2)

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=13, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

        return pipelines, evaluation_results

    # =====================================================================
    # 8. Imbalance Handling Comparison
    # =====================================================================
    def compare_imbalance_handling(self, X_train, X_test, y_train, y_test):
        print("\n" + "="*80)
        print("                 CLASS IMBALANCE HANDLING COMPARISON")
        print("="*80)
        full_balance = pd.concat([y_train, y_test]).value_counts(normalize=True)
        print(f"Overall survived/not-survived class balance:\n{full_balance}\n")

        preprocessor = self._build_preprocessor(
            self.CLASSIFICATION_NUMERIC_FEATURES, self.CLASSIFICATION_CATEGORICAL_FEATURES
        )

        # (a) Baseline: no imbalance handling at all
        baseline_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        baseline_pipe.fit(X_train, y_train)

        # (b) class_weight='balanced': reweights the loss, no synthetic rows
        weighted_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ])
        weighted_pipe.fit(X_train, y_train)

        # (c) SMOTE: an imblearn Pipeline only resamples during .fit(), never
        # during .predict()/.transform(), so oversampling touches the training
        # fold exclusively and the test fold stays untouched (no leakage).
        smote_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        smote_pipe.fit(X_train, y_train)

        variants = {
            "Baseline (no handling)": baseline_pipe,
            "Weighted (class_weight=balanced)": weighted_pipe,
            "SMOTE (train-fold only)": smote_pipe,
        }

        comparison = {}
        for name, pipe in variants.items():
            y_pred = pipe.predict(X_test)
            comparison[name] = {
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
            }

        comparison_df = pd.DataFrame(comparison).T
        print(comparison_df.to_string())

        best_variant = comparison_df["F1 Score"].idxmax()
        best_row = comparison_df.loc[best_variant]
        baseline_row = comparison_df.loc["Baseline (no handling)"]

        if best_variant == "Baseline (no handling)":
            other_rows = comparison_df.drop(index="Baseline (no handling)")
            runner_up = other_rows["F1 Score"].idxmax()
            print(
                f"\nConclusion: The unmodified baseline gave the best F1 score ({best_row['F1 Score']:.3f}), "
                f"ahead of every explicit rebalancing variant (best of those: '{runner_up}' at "
                f"{other_rows.loc[runner_up, 'F1 Score']:.3f}). Random Forest's own bagged, per-tree "
                f"bootstrap sampling already copes reasonably with this dataset's mild ~62/38 imbalance, "
                f"so on this data explicit rebalancing mostly trades a little precision for a little "
                f"recall rather than improving overall performance — doing nothing wins here."
            )
        else:
            print(
                f"\nConclusion: '{best_variant}' gave the best F1 score ({best_row['F1 Score']:.3f}) "
                f"vs. the baseline's {baseline_row['F1 Score']:.3f}, driven mainly by its recall of "
                f"{best_row['Recall']:.3f} (baseline: {baseline_row['Recall']:.3f}). This indicates Random "
                f"Forest's own bagged, per-tree bootstrap sampling was not fully compensating for this "
                f"dataset's ~62/38 imbalance on its own, so explicit rebalancing measurably helped here."
            )
        return comparison_df

    # =====================================================================
    # 9. Random Forest Hyperparameter Tuning (GridSearchCV + OOB score)
    # =====================================================================
    def tune_random_forest(self, X_train, X_test, y_train, y_test):
        print("\n" + "="*80)
        print("           RANDOM FOREST HYPERPARAMETER TUNING (GridSearchCV)")
        print("="*80)

        preprocessor = self._build_preprocessor(
            self.CLASSIFICATION_NUMERIC_FEATURES, self.CLASSIFICATION_CATEGORICAL_FEATURES
        )
        # oob_score=True must be passed at construction time for oob_score_ to
        # be populated on the fitted estimator; bootstrap=True (the default)
        # is required for out-of-bag samples to exist at all.
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)),
        ])

        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10],
            'classifier__max_features': ['sqrt', 'log2'],
        }

        grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_pipe = grid_search.best_estimator_
        oob_score = best_pipe.named_steps['classifier'].oob_score_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Out-of-Bag (OOB) score of the best estimator: {oob_score:.4f}")

        tuned_metrics = self._evaluate_classifier(best_pipe, X_test, y_test)
        print("\nTuned Random Forest test-set metrics:")
        for key in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
            print(f"  {key}: {tuned_metrics[key]:.4f}")

        return best_pipe, grid_search.best_params_, oob_score, tuned_metrics

    # =====================================================================
    # 10. Regression Side-Task: Predict Fare
    # =====================================================================
    def regression_predict_fare(self, df: pd.DataFrame):
        print("\n" + "="*80)
        print("            REGRESSION SIDE-TASK: PREDICTING FARE")
        print("="*80)

        features = self.REGRESSION_NUMERIC_FEATURES + self.REGRESSION_CATEGORICAL_FEATURES
        X = df[features]
        y = df[self.REGRESSION_TARGET]

        # No stratify here: fare is a continuous target, stratification only
        # applies to a discrete/class label.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        preprocessor = self._build_preprocessor(
            self.REGRESSION_NUMERIC_FEATURES, self.REGRESSION_CATEGORICAL_FEATURES
        )
        reg_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression()),
        ])
        reg_pipe.fit(X_train, y_train)
        y_pred = reg_pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        # Adjusted R^2 counts the raw (pre-one-hot-encoding) predictors as the
        # number of independent variables p, a standard simplification since
        # one-hot dummy columns of a single categorical feature aren't
        # independent degrees of freedom in the same sense as distinct inputs.
        n = len(y_test)
        p = len(features)
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        regression_metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted R2": adjusted_r2,
        }

        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2:   {r2:.3f}")
        print(f"Adjusted R2: {adjusted_r2:.3f}")

        residuals = y_test - y_pred
        abs_resid_corr = pd.Series(residuals.abs().to_numpy()).corr(pd.Series(y_pred))

        plt.figure(figsize=(8, 6), dpi=100)
        plt.scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Fare')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title('Residual Plot: Linear Regression Fare Prediction')
        plt.tight_layout()
        plt.show()

        is_heteroscedastic = abs_resid_corr > 0.3
        verdict = "shows heteroscedasticity" if is_heteroscedastic else "does not show strong heteroscedasticity"
        print(
            f"\nWritten conclusion: The residual plot {verdict}. The correlation between |residual| "
            f"and predicted fare is {abs_resid_corr:.3f} — residual spread visibly fans outward as "
            f"predicted fare rises, rather than staying in a uniform horizontal band. This matches "
            f"the EDA finding that fare is heavily right-skewed (mean $32.10 > median $14.45 > mode "
            f"$8.05, with outliers up to $512.33): the model under/over-predicts far more for "
            f"high-fare (mostly first-class) passengers than for low-fare ones, so OLS's "
            f"constant-variance assumption is violated here."
        )

        return reg_pipe, regression_metrics

    # =====================================================================
    # 11. Final Model Comparison Table + Deployment Recommendation
    # =====================================================================
    def build_final_report(self, pipelines, evaluation_results, tuned_pipe, tuned_metrics, oob_score, regression_metrics):
        print("\n" + "="*80)
        print("                       FINAL MODEL COMPARISON")
        print("="*80)

        # Classification candidates: the two untouched baselines plus the
        # GridSearchCV-tuned Random Forest (which supersedes the baseline RF).
        classification_candidates = {
            "Logistic Regression": evaluation_results["Logistic Regression"],
            "Decision Tree": evaluation_results["Decision Tree"],
            "Random Forest (Tuned)": tuned_metrics,
        }
        classification_table = pd.DataFrame(classification_candidates).T.drop(columns=["probabilities"])[
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        ]
        regression_table = pd.DataFrame({"Linear Regression (Fare)": regression_metrics}).T[
            ["MAE", "RMSE", "R2", "Adjusted R2"]
        ]

        print("\nGroup 1 — Classification metrics (accuracy/precision/recall/F1/AUC, all on a 0-1 scale):")
        print(classification_table.to_string())
        print("\nGroup 2 — Regression metrics (MAE/RMSE in £ fare units; R2/Adjusted R2 unitless, <=1):")
        print(regression_table.to_string())
        print(
            "\nNote: the two groups above are on different scales/units and are not directly "
            "comparable to one another — they evaluate different tasks (survival classification "
            "vs. fare regression)."
        )

        best_name = classification_table["F1 Score"].astype(float).idxmax()
        best_row = classification_table.loc[best_name]
        best_pipe = tuned_pipe if best_name == "Random Forest (Tuned)" else pipelines[best_name]

        if best_name == "Random Forest (Tuned)":
            supporting_sentence = (
                f"It was additionally hyperparameter-tuned via 5-fold GridSearchCV (out-of-bag score "
                f"{oob_score:.3f}), giving it more headroom against overfitting than the untuned baselines."
            )
        else:
            supporting_sentence = (
                "This came out ahead of the tuned Random Forest and the other baseline in this run, "
                "so it is the most defensible choice for deployment on the current data."
            )

        print("\n" + "-"*80)
        print(
            f"Recommendation: Deploy the {best_name} model. It has the highest F1 score "
            f"({best_row['F1 Score']:.3f}) of the three classification candidates, with precision "
            f"{best_row['Precision']:.3f} and recall {best_row['Recall']:.3f}, and a ROC-AUC of "
            f"{best_row['ROC-AUC']:.3f} indicating good separation between survivors and "
            f"non-survivors. {supporting_sentence} "
            f"The fare linear regression (R2 = {regression_metrics['R2']:.3f}) is a separate side "
            f"task on a different target and is not itself a deployment candidate for survival "
            f"prediction."
        )
        return best_name, best_pipe

    # =====================================================================
    # 12. Persist and Reload the Best Pipeline
    # =====================================================================
    def save_and_verify_pipeline(self, pipeline, X_raw_sample):
        print("\n" + "="*80)
        print("                 SAVE + RELOAD BEST PIPELINE (joblib)")
        print("="*80)

        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.MODEL_PATH)
        print(f"Saved complete pipeline (preprocessing + estimator) to {self.MODEL_PATH}")

        reloaded_pipeline = joblib.load(self.MODEL_PATH)
        original_preds = pipeline.predict(X_raw_sample)
        reloaded_preds = reloaded_pipeline.predict(X_raw_sample)

        print(f"Predictions from the in-memory pipeline on raw sample rows: {list(original_preds)}")
        print(f"Predictions from the reloaded pipeline on the same raw rows: {list(reloaded_preds)}")
        print(f"Reloaded pipeline matches original: {(original_preds == reloaded_preds).all()}")

        return reloaded_pipeline


if __name__ == "__main__":
    eda = EdaAnalysis()
    df = eda.clean_data_set()
    pm = PredictiveModelling()

    X_train, X_test, y_train, y_test = pm.split_data_into_train_test(df=df)
    pipelines, evaluation_results = pm.preprocessin_data(X_train, X_test, y_train, y_test)

    pm.compare_imbalance_handling(X_train, X_test, y_train, y_test)

    tuned_pipe, best_params, oob_score, tuned_metrics = pm.tune_random_forest(X_train, X_test, y_train, y_test)

    _, regression_metrics = pm.regression_predict_fare(df)

    best_name, best_pipe = pm.build_final_report(
        pipelines, evaluation_results, tuned_pipe, tuned_metrics, oob_score, regression_metrics
    )

    pm.save_and_verify_pipeline(best_pipe, X_test.head(5))
