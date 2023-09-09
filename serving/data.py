from datetime import datetime, timedelta
import io

import ee
from google.api_core import exceptions, retry
import google.auth
import requests

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured



import sys
import os


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
    opt_url="https://earthengine-highvolume.googleapis.com",)






LABEL = "is_crop_or_land"
IMAGE_COLLECTION = "COPERNICUS/S2"
BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
FEATURES = ["NDVI", "EVI"]
SCALE = 10
PATCH_SIZE = 16


# For this Project we focus on 3 areas. 
# Change this part if you want to to focus on a different area.


countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')

# Filter countries by name to obtain the ROIs for Iran, Sudan, and Afghanistan
iran_roi = countries.filter(ee.Filter.eq('country_na', 'Iran'))
sudan_roi = countries.filter(ee.Filter.eq('country_na', 'Sudan'))
afghanistan_roi = countries.filter(ee.Filter.eq('country_na', 'Afghanistan'))

# Get the geometries of the ROIs
iran_geometry = iran_roi.geometry()
sudan_geometry = sudan_roi.geometry()
afghanistan_geometry = afghanistan_roi.geometry()



def get_prediction_data(lon, lat, start, end):
    """Extracts Sentinel image as json at specific lat/lon and timestamp."""

    location = ee.Feature(ee.Geometry.Point([lon, lat]))
    image = (
        ee.ImageCollection(IMAGE_COLLECTION)
        .filterDate(start, end)
        .select(BANDS)
        .mosaic()
    )

    feature = image.neighborhoodToArray(ee.Kernel.square(PATCH_SIZE)).sampleRegions(
        collection=ee.FeatureCollection([location]), scale=SCALE
    )

    return feature.getInfo()["features"][0]["properties"]


def labeled_feature(row):

    select_point = ee.Geometry.Point([row.Lon, row.Lat])

    selected_collection = select_collection_by_point(
        select_point,
        iran_geometry,
        sudan_geometry,
        afghanistan_geometry,
        iran_collection,
        sudan_collection,
        afghanistan_collection
    )

    image = (
        selected_collection
        .mosaic()
    )
    point = ee.Feature(
        select_point,
        {LABEL: row.Target},
    )
    return (
        image.neighborhoodToArray(ee.Kernel.square(PATCH_SIZE))
        .sampleRegions(ee.FeatureCollection([point]), scale=SCALE)
        .first()
    )



def select_collection_by_point(point, 
                               iran_geometry, 
                               sudan_geometry, 
                               afghanistan_geometry, 
                               iran_collection, 
                               sudan_collection,
                               afghanistan_collection):
    """
    Selects an image collection based on whether a given point is within the bounds of a geometry.

    Args:
        point (ee.Geometry.Point): The point to check.
        iran_geometry (ee.Geometry): The geometry representing the bounds of Iran.
        sudan_geometry (ee.Geometry): The geometry representing the bounds of Sudan.
        afghanistan_geometry (ee.Geometry): The geometry representing the bounds of Afghanistan.
        iran_collection (ee.ImageCollection): The Sentinel-2 image collection for Iran.
        sudan_collection (ee.ImageCollection): The Sentinel-2 image collection for Sudan.
        afghanistan_collection (ee.ImageCollection): The Sentinel-2 image collection for Afghanistan.

    Returns:
        ee.ImageCollection or None: The selected image collection or None if the point is not within any geometry.
    """
    # Check if the point is within the bounds of the geometries
    is_in_iran = iran_geometry.contains(point)
    is_in_sudan = sudan_geometry.contains(point)
    is_in_afghanistan = afghanistan_geometry.contains(point)

    # Depending on the results, choose which collection to return
    if is_in_iran.getInfo():
        selected_collection = iran_collection
    elif is_in_sudan.getInfo():
        selected_collection = sudan_collection
    elif is_in_afghanistan.getInfo():
        selected_collection = afghanistan_collection
    else:
        selected_collection = None  # Point is not within any of the geometries

    return selected_collection


def create_composited_sentinel2_collection(roi,
                                           start_date_str, 
                                           end_date_str, 
                                           interval=15, 
                                           limit=10, 
                                           include_ndvi=True, 
                                           include_evi=True):
    """
    Creates a 15-day composited Sentinel-2 image collection within the specified ROI and time range.

    Args:
        roi (ee.Geometry): The region of interest as an Earth Engine geometry.
        start_date_str (str): The start date in 'yyyy-mm-dd' format.
        end_date_str (str): The end date in 'yyyy-mm-dd' format.
        interval (int, optional): The number of days for each composite interval. Default is 15.
        limit (int, optional): The maximum number of images to include in each composite. Default is 10.
        include_ndvi (bool, optional): Whether to calculate and include NDVI bands. Default is True.
        include_evi (bool, optional): Whether to calculate and include EVI bands. Default is True.

    Returns:
        ee.ImageCollection: The composited Sentinel-2 image collection.
    """
    # Convert start_date_str and end_date_str to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Create an empty ImageCollection to store the composites
    composited_collection = ee.ImageCollection([])

    # Loop through the dates and create composites
    while start_date < end_date:
        # Calculate the end date for the current interval
        interval_end_date = start_date + timedelta(days=interval)
        if interval_end_date > end_date:
            interval_end_date = end_date

        # Format dates as strings
        interval_start_date_str = start_date.strftime('%Y-%m-%d')
        interval_end_date_str = interval_end_date.strftime('%Y-%m-%d')

        # Filter Sentinel-2 data for the current date range
        collection = ee.ImageCollection('COPERNICUS/S2') \
            .filterBounds(roi) \
            .filterDate(interval_start_date_str, interval_end_date_str) \
            .select(BANDS) \
            .sort('CLOUDY_PIXEL_PERCENTAGE') \
            #.limit(limit)

        # Calculate NDVI and EVI if requested
        if include_ndvi:
            collection = collection.map(calculate_ndvi)
        if include_evi:
            collection = collection.map(calculate_evi)

        # Create a median composite of the image collection
        median_image = collection.median()

        # Add the composite to the ImageCollection
        composited_collection = composited_collection.merge(ee.ImageCollection([median_image]))

        # Move to the next interval
        start_date = interval_end_date + timedelta(days=1)

    return composited_collection
   

def calculate_ndvi(image):
    """Calculate NDVI for an image."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def calculate_evi(image):
    """Calculate EVI for an image."""
    evi = image.expression(
        '2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))', {
            'B8': image.select('B8'),
            'B4': image.select('B4'),
            'B2': image.select('B2')
        }
    ).rename('EVI')
    return image.addBands(evi)