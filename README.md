# ChurnPredictorXAI

An end-to-end, locally deployed Machine Learning system for predicting telecom customer churn, built with a strong focus on **Explainable AI (XAI)**. The project trains a robust Random Forest classifier and demystifies its predictions using **SHAP** and **pyGAM**.

It features a premium, interactive dark-mode web dashboard built with Flask to serve real-time predictions and visualizations.

## Features

- **End-to-end ML Pipeline**: scikit-learn pipeline with median/mode imputation, scaling, one-hot encoding, and a tuned Random Forest.
- **Deep Explainability**:
  - **SHAP (TreeExplainer)**: Provides global feature importance (beeswarm plots) and local waterfall plots explaining the exact log-odds contribution for individual high-risk customers.
  - **pyGAM (LogisticGAM)**: Models the continuous features independently to display precise, non-linear partial-dependence term plots with 95% confidence intervals.
- **Web Dashboard**: An interactive, responsive UI to submit custom customer profiles and get instant probability scores alongside the model's performance metrics, confusion matrix, and ROC curve.
- **Robustness**: Handles data collinearity gracefully (e.g., decoupling `total_charges` from the tree splits to enforce strict logical monotonicity for `tenure`).

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hritikkumarpradhan/ChurnPredictorXAI.git
   cd ChurnPredictorXAI
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ installed. 
   ```bash
   pip install scikit-learn pandas numpy matplotlib shap pygam flask
   ```

3. **Run the Dashboard:**
   ```bash
   python app.py
   ```
   *The script will generate 8000 mock records, train the Pipeline and GAM, compute SHAP values, and start a local web server.*

4. **View the App:**
   Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Repository Structure

- `app.py`: The Flask web application containing the ML engine (`ChurnEngine`), data generation, model training, and all HTML/CSS/JS for the frontend.
- `churn_predictor_xai.py`: A standalone CLI version of the system that trains the model and saves all XAI plots directly to disk (in `xai_outputs/`).
- `test_predictor.py` & `read_results.py`: Programmatic testing suite to verify the logical monotonicity (e.g., ensuring higher tenure reduces churn risk) and the impact of categorical features.

## Technologies Used

- **Python**
- **scikit-learn** (RandomForestClassifier, ColumnTransformer, Pipelines)
- **SHAP** (TreeExplainer)
- **pyGAM** (LogisticGAM)
- **Flask**
- **Matplotlib / Pandas / NumPy**

## Cloud Deployment (Render)

This project includes a `render.yaml` configuration for seamless Infrastructure-as-Code deployments. To publish your app live:

1. **Push your code to GitHub** (make sure `app.py`, `requirements.txt`, and `render.yaml` are pushed).
2. Go to **[Render.com](https://render.com/)**, sign up/log in, and click **New → Blueprint**.
3. Connect your GitHub account and select your `ChurnPredictorXAI` repository.
4. Render will automatically detect the configuration and deploy your web app for free.
*Note: Because the model generates synthetic data and trains an ensemble on boot, the `gunicorn` start command uses an extended `--timeout 200` to prevent timeouts on the free tier.*

## License

This project is open-source and available under the MIT License.
