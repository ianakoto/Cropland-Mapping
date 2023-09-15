from __future__ import annotations

from typing import List, Optional

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions


from croplandclassification.config import *


def run(
    bucket: str,
    train_input_path: str,
    output_path: str,
    max_requests: int = MAX_REQUESTS,
    beam_args: Optional[List[str]] = None,
) -> None:
    """Runs an Apache Beam pipeline to create a dataset.
    This fetches data from Earth Engine and writes compressed NumPy files.
    We use `max_requests` to limit the number of concurrent requests to Earth Engine
    to avoid quota issues. You can request for an increas of quota if you need it.
    Args:
        output_data_path: Directory path to save the data files.
        max_requests: Limit the number of concurrent requests to Earth Engine.
        min_batch_size: Minimum number of examples to write per data file.
        beam_args: Apache Beam command line arguments to parse as pipeline options.
    """

    from croplandclassification.beam_func import (
        ExtractFieldsFn,
        split_dataset,
        serialize,
        TryGetExample,
    )

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        direct_running_mode="multi_processing",
        num_workers=max_requests,
        max_num_workers=max_requests,  # distributed runners
    )

    data_input_path = f"gs://{bucket}/{train_input_path}"

    with beam.Pipeline(options=beam_options) as pipeline:
        # Read input data from Google Cloud Storage
        input_data = (
            pipeline
            | "📖 Read CropLand CSV File"
            >> beam.io.ReadFromText(data_input_path, skip_header_lines=1)
            | "Flatten the CSV" >> beam.Map(lambda row: row.split(","))
            | "🔍 Extract Fields" >> beam.ParDo(ExtractFieldsFn())
        )

        # Split dataset into training and validation
        training_data, test_data = input_data | "📝 Split dataset" >> beam.Partition(
            split_dataset, 2
        )
        # Process and serialize sampled points
        processed_training_features = (
            training_data
            | "📑 Get Training patch" >> beam.ParDo(TryGetExample())
            | "🗂️ Serialize Training" >> beam.Map(serialize)
        )

        processed_test_features = (
            test_data
            | "📑 Get Test patch" >> beam.ParDo(TryGetExample())
            | "🗂️ Serialize Test" >> beam.Map(serialize)
        )

        # Write features to TFRecord files
        processed_training_features | "Write Train TFRecords" >> beam.io.WriteToTFRecord(
            file_path_prefix=f"gs://{bucket}/{output_path}/train",
            file_name_suffix=".tfrecord.gz",
        )

        processed_test_features | "Write Test TFRecords" >> beam.io.WriteToTFRecord(
            file_path_prefix=f"gs://{bucket}/{output_path}/test",
            file_name_suffix=".tfrecord.gz",
        )


def main() -> None:
    import argparse
    import logging
    from croplandclassification.config import MAX_REQUESTS

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        required=True,
        help="Cloud storage Bucket Name",
    )
    parser.add_argument(
        "--train-input-path",
        required=True,
        help="Input path for training data",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path to save the data files",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=MAX_REQUESTS,
        help="Limit the number of concurrent requests to Earth Engine",
    )
    args, beam_args = parser.parse_known_args()

    run(
        bucket=args.bucket,
        train_input_path=args.train_input_path,
        output_path=args.output_path,
        max_requests=args.max_requests,
        beam_args=beam_args,
    )


if __name__ == "__main__":
    main()
