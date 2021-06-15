#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save to output files

    trainval.to_csv("trainval_data.csv", index=False)
    test.to_csv("test_data.csv", index=False)

    artifact = wandb.Artifact(
     "trainval_data.csv",
     type="trainval_data",
     description="trainval_data_split",
    )
    artifact.add_file("trainval_data.csv")
    logger.info(f"Uploading trainval_data.csv dataset")
    run.log_artifact(artifact)

    artifact2 = wandb.Artifact(
     "test_data.csv",
     type="test_data",
     description="test_data_split",
    )
    artifact2.add_file("test_data.csv")
    logger.info(f"Uploading test_data.csv dataset")

    run.log_artifact(artifact2)
    run.finish()
    logger.info(f"Done")

    """ for df, k in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {k}_data.csv dataset")
        with tempfile.NamedTemporaryFile("w") as fp:

            df.to_csv(fp.name, index=False)

            log_artifact(
                f"{k}_data.csv",
                f"{k}_data",
                f"{k} split of dataset",
                fp.name,
                run,
            ) """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )

    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )

    args = parser.parse_args()

    go(args)
