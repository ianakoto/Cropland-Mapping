import logging
import requests
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import io
import ee
import tensorflow as tf

from typing import List

from collections.abc import Iterator

from .data import *
from .config import *

import numpy as np
import pandas as pd
import requests

import sys
import os
import io


import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions


from google.api_core import exceptions, retry
import google.auth


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
    train_features = FEATURES + BANDS
    features_dict = {
        name: tf.io.FixedLenFeature(shape=[33, 33], dtype=tf.float32)
        for name in train_features
    }
    features_dict[LABEL] = tf.io.FixedLenFeature(shape=[1, 1], dtype=tf.float32)
    return features_dict


def serialize(patch: List[np.ndarray]):
    train_features = FEATURES + BANDS
    feature = create_features_dict()
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    # Set features for each band
    for i, name in enumerate(train_features):
        example_proto.features.feature[name].float_list.value.extend(patch[i].flatten())

    # Set the label feature
    example_proto.features.feature[LABEL].float_list.value.extend(patch[-1].flatten())

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
