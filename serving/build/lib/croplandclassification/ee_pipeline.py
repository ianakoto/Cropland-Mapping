from google.api_core import exceptions, retry
import google.auth

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
import numpy as np
import pandas as pd
import requests

import sys
import os

# Import the Image function from the IPython.display module.
from IPython.display import Image

import tensorflow as tf
from tensorflow import keras

from .config import *


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

    from google.api_core import exceptions, retry
    import google.auth
    import logging
    import requests
    from .data import *
    import numpy as np
    from numpy.lib.recfunctions import structured_to_unstructured
    from typing import List

    import tensorflow as tf

    # Authenticate and initialize Earth Engine with the default credentials.
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine",
        ]
    )

    # Use the Earth Engine High Volume endpoint.
    #   https://developers.google.com/earth-engine/cloud/highvolume
    ee.Initialize(
        credentials.with_quota_project(None),
        project=project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )

    TEST_RATIO = 0.1

    def create_features_dict() -> dict:
        """Creates dict of features."""
        from .config import *

        train_features = FEATURES + BANDS
        features_dict = {
            name: tf.io.FixedLenFeature(shape=[33, 33], dtype=tf.float32)
            for name in train_features
        }
        features_dict[LABEL] = tf.io.FixedLenFeature(shape=[1, 1], dtype=tf.float32)
        return features_dict

    def serialize(patch: List[np.ndarray]):
        from .config import *

        train_features = FEATURES + BANDS
        feature = create_features_dict()
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

        # Set features for each band
        for i, name in enumerate(train_features):
            example_proto.features.feature[name].float_list.value.extend(
                patch[i].flatten()
            )

        # Set the label feature
        example_proto.features.feature[LABEL].float_list.value.extend(
            patch[-1].flatten()
        )

        return example_proto.SerializeToString()

    def split_dataset(
        element,
        num_partitions: int,
    ) -> int:
        import random

        weights = [1 - TEST_RATIO, TEST_RATIO]
        return random.choices([0, 1], weights)[0]

    class ExtractFieldsFn(beam.DoFn):
        """
        Define a DoFn to extract theid, lat, lon, target columns from the CSV file
        """

        def process(self, element):
            id, lat, lon, target = element
            yield {
                "ID": id,
                "Lat": float(lat),
                "Lon": float(lon),
                "Target": float(target),
            }

    @retry.Retry()
    def get_patch(
        image: ee.Image,
        geometry: ee.Geometry,
    ) -> np.ndarray:
        """Fetches a patch of pixels from Earth Engine.
        It retries if we get error "429: Too Many Requests".
        Args:
            image: Image to get the patch from.
            point: A (longitude, latitude) pair for the point of interest.
        Raises:
            requests.exceptions.RequestException
        Returns:
            The requested patch of pixels as a structured
            NumPy array with shape (width, height).
        """
        from .config import *

        bands = BANDS + FEATURES + LABEL

        url = image.getDownloadURL(
            {
                "region": geometry,
                "dimensions": [POINT_PATCH_SIZE, POINT_PATCH_SIZE],
                "format": "NPY",
                "bands": bands,
            }
        )

        # If we get "429: Too Many Requests" errors, it's safe to retry the request.
        # The Retry library only works with `google.api_core` exceptions.
        response = requests.get(url)
        if response.status_code == 429:
            raise exceptions.TooManyRequests(response.text)

        # Still raise any other exceptions to make sure we got valid data.
        response.raise_for_status()
        return np.load(io.BytesIO(response.content), allow_pickle=True)

    def get_training_example(long, lat, target):
        from .data import labeled_feature
        from .config import *
        import ee

        select_point = ee.Geometry.Point([long, lat]).buffer(PATCH_SIZE)

        image = labeled_feature(long, lat, target)
        patch = get_patch(image, select_point)

        return structured_to_unstructured(patch)

    def try_get_example(long, lat, target) -> Iterator[tuple]:
        """Wrapper over `get_training_examples` that allows it to simply log errors instead of crashing."""
        try:
            yield get_training_example(long, lat, target)
        except (requests.exceptions.HTTPError, ee.ee_exception.EEException) as e:
            logging.error(f"🛑 failed to get example at: {long} {lat}")
            logging.exception(e)

    class TryGetExample(beam.DoFn):
        def process(self, element):
            lat = element["Lat"]
            long = element["Lon"]
            target = element["Target"]
            return try_get_example(long, lat, target)

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
    from config import *

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
