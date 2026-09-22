import os
import sys

import mlflow
from mlflow import MlflowClient



TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
)

MODEL_NAME = "pothole-segmentation"


METRIC_NAME = "mask_map50_95"


MAX_ALLOWED_DROP = 0.05


def get_metric(client, model_version):
    run = client.get_run(model_version.run_id)

    if METRIC_NAME not in run.data.metrics:
        available = ", ".join(sorted(run.data.metrics)) or "none"
        raise RuntimeError(
            f"Metric '{METRIC_NAME}' was not found in run "
            f"{model_version.run_id}. Available metrics: {available}"
        )

    return float(run.data.metrics[METRIC_NAME])


def main():
    
    if not 0 <= MAX_ALLOWED_DROP <= 1:
        raise ValueError(
            "MLFLOW_MAX_ALLOWED_DROP must be between 0 and 1."
        )
        
    
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    latest = client.get_model_version_by_alias(MODEL_NAME, "latest")
    production = client.get_model_version_by_alias(MODEL_NAME, "Production")

    latest_metric = get_metric(client, latest)
    production_metric = get_metric(client, production)

    print(
        f"latest: version={latest.version}, "
        f"{METRIC_NAME}={latest_metric:.6f}"
    )
    print(
        f"Production: version={production.version}, "
        f"{METRIC_NAME}={production_metric:.6f}"
    )
    
    if production_metric == 0:
        raise RuntimeError(
            "Production metric is zero, cannot compute relative drop."
        )
    

    relative_drop = (production_metric - latest_metric ) / production_metric
    reject = relative_drop > MAX_ALLOWED_DROP

    if reject:
        print(
            f"REJECTED: latest is {relative_drop * 100:.2f}% lower than "
            "Production. Aliases were not changed."
        )
        return 1


    return 0


    # # Reassigning Production automatically removes it from the old version.
    # # The promoted version keeps its existing `latest` alias.
    # client.set_registered_model_alias(
    #     name=MODEL_NAME,
    #     alias="Production",
    #     version=latest.version,
    # )

    # updated = client.get_model_version_by_alias(MODEL_NAME, "Production")
    # if str(updated.version) != str(latest.version):
    #     raise RuntimeError("Production alias verification failed.")

    # print(
    #     f"PROMOTED: Production moved from version {production.version} "
    #     f"to version {latest.version}."
    # )
    # return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
