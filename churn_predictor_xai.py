"""
ChurnPredictorXAI — End-to-end churn prediction with SHAP & pyGAM explainability.

This module builds a production-style sklearn Pipeline around a
RandomForestClassifier, then layers two complementary explanation
strategies on top:

    1. **SHAP (TreeExplainer)** — global feature importance & local
       waterfall explanations for individual customers.
    2. **pyGAM (LogisticGAM)** — intrinsically interpretable partial-
       dependence term plots for every continuous feature, with 95 %
       confidence intervals.

Usage:
    python churn_predictor_xai.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pygam import LogisticGAM, s
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ChurnPredictorXAI:
    """Predict telecom customer churn and explain every prediction.

    Attributes:
        pipeline: Fitted sklearn ``Pipeline`` (preprocessor + Random Forest).
        preprocessor: Fitted ``ColumnTransformer`` extracted from the pipeline.
        gam: Fitted ``LogisticGAM`` trained on the continuous features.
        continuous_features: Names of the continuous columns.
        categorical_features: Names of the categorical columns.
        output_dir: Directory where generated plots are saved.
    """

    # ── column groups ──────────────────────────────────────────────
    CONTINUOUS_FEATURES: list[str] = [
        "tenure",
        "monthly_charges",
        "total_charges",
    ]
    CONTINUOUS_MODEL: list[str] = [
        "tenure",
        "monthly_charges",
    ]
    CATEGORICAL_FEATURES: list[str] = [
        "contract_type",
        "internet_service",
        "tech_support",
    ]

    def __init__(self, output_dir: str | Path = "xai_outputs") -> None:
        """Initialise the predictor.

        Args:
            output_dir: Folder for saved plots and artefacts.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.gam: Optional[LogisticGAM] = None

        self.continuous_features = self.CONTINUOUS_MODEL
        self.categorical_features = self.CATEGORICAL_FEATURES

    # ── 1. mock data ───────────────────────────────────────────────
    @staticmethod
    def generate_mock_data(n_samples: int = 8000) -> pd.DataFrame:
        """Create a synthetic telecom-churn DataFrame.

        Churn probability is engineered so that **lower tenure**,
        **month-to-month contracts**, and **higher monthly charges**
        increase the likelihood of churn.

        Args:
            n_samples: Number of rows to generate.

        Returns:
            A ``pd.DataFrame`` with six feature columns and one
            binary ``churn`` target.
        """
        rng = np.random.default_rng(seed=42)

        tenure = rng.integers(0, 73, size=n_samples).astype(float)
        monthly_charges = rng.uniform(20.0, 120.0, size=n_samples)
        # total_charges: weakly correlated so RF can't use as a
        # proxy for tenure
        total_charges = np.clip(
            rng.uniform(100, 5000, n_samples)
            + 0.3 * tenure * monthly_charges
            + rng.normal(0, 500, n_samples),
            0, None,
        )

        contract_type = rng.choice(
            ["Month-to-month", "One year", "Two year"],
            size=n_samples,
            p=[0.50, 0.25, 0.25],
        )
        internet_service = rng.choice(
            ["DSL", "Fiber optic", "No"],
            size=n_samples,
            p=[0.40, 0.45, 0.15],
        )
        tech_support = rng.choice(
            ["Yes", "No", "No internet"],
            size=n_samples,
            p=[0.30, 0.55, 0.15],
        )

        # --- churn probability logic ---
        logit = (
            -0.5
            - 0.10 * tenure
            + 0.04 * monthly_charges
            + 2.5 * (contract_type == "Month-to-month").astype(float)
            - 2.0 * (contract_type == "Two year").astype(float)
            + 1.0 * (internet_service == "Fiber optic").astype(float)
            - 0.8 * (tech_support == "Yes").astype(float)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        churn = rng.binomial(1, prob).astype(int)

        return pd.DataFrame(
            {
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "contract_type": contract_type,
                "internet_service": internet_service,
                "tech_support": tech_support,
                "churn": churn,
            }
        )

    # ── 2. build preprocessing + pipeline ─────────────────────────
    def _build_pipeline(self) -> Pipeline:
        """Construct the full sklearn Pipeline.

        Returns:
            An **unfitted** ``Pipeline`` (``ColumnTransformer`` ➜
            ``RandomForestClassifier``).
        """
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.continuous_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        max_depth=15,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        return pipeline

    # ── 3. train & evaluate ────────────────────────────────────────
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Train the pipeline and print evaluation metrics.

        Args:
            df: Full dataset including the ``churn`` target.
            test_size: Fraction held out for testing.
            random_state: Seed for reproducibility.

        Returns:
            A 4-tuple ``(X_train, y_train, X_test, y_test)`` for
            downstream use.
        """
        feature_cols = self.continuous_features + self.categorical_features
        X = df[feature_cols]
        y = df["churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]

        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        print("=" * 60)
        print("  Random Forest — Hold-out Evaluation")
        print("=" * 60)
        print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
        print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
        print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
        print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
        print("-" * 60)
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

        return X_train, y_train, X_test, y_test

    # ── helpers ────────────────────────────────────────────────────
    def _get_feature_names(self) -> list[str]:
        """Retrieve human-readable feature names from the fitted preprocessor.

        Returns:
            Ordered list of feature names after transformation.
        """
        if self.preprocessor is None:
            raise RuntimeError("Pipeline has not been trained yet.")

        num_names: list[str] = list(self.continuous_features)

        ohe: OneHotEncoder = (
            self.preprocessor.named_transformers_["cat"]
            .named_steps["encoder"]
        )
        cat_names: list[str] = list(ohe.get_feature_names_out(self.categorical_features))

        return num_names + cat_names

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted preprocessor to raw feature data.

        Args:
            X: Raw feature DataFrame (same schema as training data).

        Returns:
            2-D numpy array of transformed features.
        """
        if self.preprocessor is None:
            raise RuntimeError("Pipeline has not been trained yet.")
        return self.preprocessor.transform(X)

    # ── 4. pyGAM term plots ───────────────────────────────────────
    def train_gam(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit a LogisticGAM on the (scaled) continuous features only.

        The GAM is trained on the **transformed** continuous columns
        (after imputation + scaling) so that its spline terms are on
        the same scale the Random Forest sees.

        Args:
            X_train: Raw training features (pre-transformation).
            y_train: Binary churn labels.
        """
        X_transformed = self._transform(X_train)
        n_cont = len(self.continuous_features)
        X_cont = X_transformed[:, :n_cont]

        # Build a sum-of-splines formula: s(0) + s(1) + ... + s(n-1)
        terms = s(0)
        for i in range(1, n_cont):
            terms += s(i)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gam = LogisticGAM(terms).gridsearch(
                X_cont, y_train.values, progress=False
            )

        print("\n✅ LogisticGAM fitted successfully.")
        print(self.gam.summary())

    def plot_gam_terms(self) -> None:
        """Generate partial-dependence term plots for every continuous feature.

        Each subplot shows the smoothed log-odds contribution of a
        single feature with a 95 % confidence band.  Plots are saved
        to ``<output_dir>/gam_term_plots.png``.
        """
        if self.gam is None:
            raise RuntimeError("GAM has not been trained yet. Call train_gam() first.")

        n_terms = len(self.continuous_features)
        fig, axes = plt.subplots(1, n_terms, figsize=(6 * n_terms, 5))
        if n_terms == 1:
            axes = [axes]

        for idx, (ax, feat_name) in enumerate(zip(axes, self.continuous_features)):
            XX = self.gam.generate_X_grid(term=idx)
            pdep, confi = self.gam.partial_dependence(term=idx, X=XX, width=0.95)

            ax.plot(XX[:, idx], pdep, color="#2563eb", lw=2.5)
            ax.fill_between(
                XX[:, idx],
                confi[:, 0],
                confi[:, 1],
                alpha=0.20,
                color="#60a5fa",
            )
            ax.set_title(f"GAM Term: {feat_name}", fontsize=14, fontweight="bold")
            ax.set_xlabel(f"{feat_name} (scaled)", fontsize=12)
            ax.set_ylabel("Partial Dependence (log-odds)", fontsize=12)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            "pyGAM — Non-linear Partial Dependence (95% CI)",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        save_path = self.output_dir / "gam_term_plots.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n📊 GAM term plots saved → {save_path}")

    # ── 5. SHAP explanations ──────────────────────────────────────
    def explain_with_shap(self, X_test: pd.DataFrame) -> None:
        """Generate SHAP global summary and local waterfall plots.

        Args:
            X_test: Raw test-set features (before transformation).
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been trained yet.")

        rf_model: RandomForestClassifier = self.pipeline.named_steps["classifier"]
        feature_names = self._get_feature_names()
        X_test_transformed = self._transform(X_test)

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer(X_test_transformed)

        # For binary classification TreeExplainer may return shape
        # (n_samples, n_features, 2). We want the "churn = 1" slice.
        if shap_values.values.ndim == 3:
            shap_values_class1 = shap.Explanation(
                values=shap_values.values[:, :, 1],
                base_values=shap_values.base_values[:, 1]
                if shap_values.base_values.ndim == 2
                else shap_values.base_values,
                data=shap_values.data,
                feature_names=feature_names,
            )
        else:
            shap_values_class1 = shap_values
            shap_values_class1.feature_names = feature_names

        # ── Global summary plot ──
        fig_summary, ax_summary = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values_class1.values,
            X_test_transformed,
            feature_names=feature_names,
            show=False,
        )
        summary_path = self.output_dir / "shap_global_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"\n🌍 SHAP global summary saved → {summary_path}")

        # ── Local waterfall plot for a high-risk customer ──
        proba = rf_model.predict_proba(X_test_transformed)[:, 1]
        high_risk_idx = int(np.argmax(proba))

        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(
            shap_values_class1[high_risk_idx],
            show=False,
        )
        waterfall_path = self.output_dir / "shap_local_waterfall.png"
        plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"🔍 SHAP waterfall (customer #{high_risk_idx}) saved → {waterfall_path}")


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("━" * 60)
    print("  ChurnPredictorXAI — end-to-end demo")
    print("━" * 60)

    xai = ChurnPredictorXAI(output_dir="xai_outputs")

    # 1. generate data
    print("\n[1/5] Generating mock telecom churn data …")
    df = ChurnPredictorXAI.generate_mock_data(n_samples=8000)
    print(f"      Shape: {df.shape}  |  Churn rate: {df['churn'].mean():.2%}")

    # 2. train random-forest pipeline
    print("\n[2/5] Training Random Forest pipeline …")
    X_train, y_train, X_test, y_test = xai.train(df)

    # 3. fit GAM
    print("\n[3/5] Training LogisticGAM on continuous features …")
    xai.train_gam(X_train, y_train)

    # 4. plot GAM terms
    print("\n[4/5] Generating GAM term plots …")
    xai.plot_gam_terms()

    # 5. SHAP explanations
    print("\n[5/5] Computing SHAP explanations …")
    xai.explain_with_shap(X_test)

    print("\n" + "━" * 60)
    print("  ✅ All outputs saved to ./xai_outputs/")
    print("━" * 60)
