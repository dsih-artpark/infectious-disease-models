import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from config import CONFIG, setup_logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

app = typer.Typer()


def generate_synthetic_data(n_samples: int, n_features: int, random_seed: int):
    """Generate reproducible synthetic features and binary labels."""
    np.random.seed(random_seed)
    X = np.random.randn(n_samples, n_features)
    # simple linear combination + sigmoid for probabilities
    coefs = np.arange(1, n_features + 1)
    logits = X @ coefs
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
    df["target"] = y
    return df


def train_model(df: pd.DataFrame, test_size: float, random_seed: int, epochs: int):
    """Train logistic regression, simulate epochs, and return accuracy."""
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    model = LogisticRegression()
    for _ in tqdm(range(epochs), desc="Training epochs"):
        model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, accuracy_score(y_test, preds), X_test, y_test, preds


def save_outputs(output_dir: str, X_test: pd.DataFrame, y_test: pd.Series, preds: np.ndarray, metadata: dict):
    """Save predictions and run metadata."""
    os.makedirs(output_dir, exist_ok=True)
    # Predictions CSV
    results = X_test.copy()
    results["true"] = y_test.values
    results["predicted"] = preds
    results.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    # Metadata JSON
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def get_model_params(model_id: str, config: dict) -> dict:
    for model in config.get("models", []):
        if model.get("id") == model_id:
            return model.get("parameters", {})
    return {}


MODEL_ID = "logistic-regression-example-v0.1.0"
MODEL_PARAMS = get_model_params(MODEL_ID, CONFIG)


@app.command()
def main(
    n_samples: int = typer.Option(MODEL_PARAMS.get("n_samples", 1000), help="Number of synthetic samples"),
    n_features: int = typer.Option(MODEL_PARAMS.get("n_features", 5), help="Number of features"),
    test_size: float = typer.Option(MODEL_PARAMS.get("test_size", 0.2), help="Test set proportion"),
    epochs: int = typer.Option(MODEL_PARAMS.get("epochs", 10), help="Number of training epochs"),
    random_seed: int = typer.Option(MODEL_PARAMS.get("random_seed", 42), help="Random seed for reproducibility"),
):
    """
    Reproducible logistic regression on synthetic data.
    """
    # Setup logging with timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / MODEL_ID
    log_file = log_dir / f"run_{timestamp}.log"
    logger = setup_logging(str(log_file))

    # Log experiment start and parameters
    logger.info(f"Starting experiment with model {MODEL_ID}")
    logger.info(f"Parameters: n_samples={n_samples}, n_features={n_features}, test_size={test_size}, epochs={epochs}, random_seed={random_seed}")

    try:
        df = generate_synthetic_data(n_samples, n_features, random_seed)
        logger.info(f"Generated synthetic data with shape {df.shape}")

        model, accuracy, X_test, y_test, preds = train_model(df, test_size, random_seed, epochs)
        logger.info(f"Model training completed with accuracy: {accuracy:.4f}")

        run_metadata = {
            "model_id": MODEL_ID,
            "run_id": uuid.uuid4().hex,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "random_seed": random_seed,
            "n_samples": n_samples,
            "n_features": n_features,
            "test_size": test_size,
            "epochs": epochs,
            "accuracy": accuracy
        }

        output_dir = os.path.join("outputs", MODEL_ID, f"run_{run_metadata['run_id']}")
        save_outputs(output_dir, X_test, y_test, preds, run_metadata)
        logger.info(f"Outputs and metadata saved to: {output_dir}")

        # Use Typer for user-facing output
        typer.echo(f"✅ Run complete — accuracy: {accuracy:.4f}")
        typer.echo(f"📁 Outputs saved to: {output_dir}")
        typer.echo(f"📝 Log file: {log_file}")

    except Exception as e:
        logger.error(f"Experiment failed: {e!s}", exc_info=True)
        typer.echo(f"❌ Experiment failed: {e!s}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
