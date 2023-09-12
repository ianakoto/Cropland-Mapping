from datetime import datetime, timedelta
import ee
import numpy as np
from config import *


countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')

# Filter countries by name to obtain the ROIs for Iran, Sudan, and Afghanistan
iran_roi = countries.filter(ee.Filter.eq('country_na', 'Iran'))
sudan_roi = countries.filter(ee.Filter.eq('country_na', 'Sudan'))
afghanistan_roi = countries.filter(ee.Filter.eq('country_na', 'Afghanistan'))

# Get the geometries of the ROIs
iran_geometry = iran_roi.geometry()
sudan_geometry = sudan_roi.geometry()
afghanistan_geometry = afghanistan_roi.geometry()



def get_prediction_data(lon, 
                        lat, 
                        iran_collection, 
                        sudan_collection, 
                        afghanistan_collection):
    """Extracts Sentinel image as json at specific lat/lon and timestamp."""

    location = ee.Feature(ee.Geometry.Point([lon, lat]))

    selected_collection = select_collection_by_point(
        location,
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

    feature = image.neighborhoodToArray(ee.Kernel.square(PATCH_SIZE)).sampleRegions(
        collection=ee.FeatureCollection([location]), scale=SCALE
    )

    return feature.getInfo()["features"][0]["properties"]


def labeled_feature(row, iran_collection, sudan_collection, afghanistan_collection):
    """
    Extract labeled features from satellite imagery at a specific point.

    This function extracts labeled features from satellite imagery at a specific point
    defined by latitude (Lat) and longitude (Lon) provided in the 'row' parameter. The
    function selects the appropriate image collection (Iran, Sudan, or Afghanistan) based
    on the point's location and then extracts features from the mosaic of that collection.

    Parameters:
    - row (pd.Series or dict): A row or dictionary containing 'Lat', 'Lon', and 'Target' fields,
      where 'Lat' and 'Lon' are the latitude and longitude coordinates of the point, and 'Target'
      is the label associated with the point.
    - iran_collection (Sentinel2ImageCollection): A composited Sentinel-2 image collection for Iran.
    - sudan_collection (Sentinel2ImageCollection): A composited Sentinel-2 image collection for Sudan.
    - afghanistan_collection (Sentinel2ImageCollection): A composited Sentinel-2 image collection for Afghanistan.

    Returns:
    - labeled_feature (ee.Feature): A feature representing the labeled feature extracted from the satellite imagery.

    Note:
    - To use this function, you must have access to the 'select_collection_by_point' function,
      'iran_geometry', 'sudan_geometry', 'afghanistan_geometry', 'PATCH_SIZE', and 'SCALE'
      defined elsewhere in your code.

    Example:
    row = {'Lat': 35.1234, 'Lon': 51.5678, 'Target': 'urban'}
    labeled_feature = labeled_feature(row, iran_collection, sudan_collection, afghanistan_collection)
    # Now you can work with the labeled feature for further analysis or modeling.
    """
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


def get_collections():
    """
    Retrieve composited Sentinel-2 image collections for specific regions and time frame.

    This function retrieves composited Sentinel-2 image collections for three different regions:
    Iran, Sudan, and Afghanistan, for a specified time period between 'start_date' and 'end_date'.

    Returns:
    - iran_collection (Sentinel2ImageCollection): A composited Sentinel-2 image collection for Iran.
    - sudan_collection (Sentinel2ImageCollection): A composited Sentinel-2 image collection for Sudan.
    - afghanistan_collection (Sentinel2ImageCollection): A composited Sentinel-2 image collection for Afghanistan.
    
    Note:
    - To use this function, you must have access to the 'create_composited_sentinel2_collection'
      function and the geometries (iran_geometry, sudan_geometry, and afghanistan_geometry)
      defined elsewhere in your code.

    Example:
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    iran_collection, sudan_collection, afghanistan_collection = get_collections(start_date, end_date)
    # Now you can work with these collections for further analysis or visualization.
    """
    iran_collection = create_composited_sentinel2_collection(iran_geometry,
                                                             IRAN_START_DATE,
                                                             IRAN_END_DATE)
    sudan_collection = create_composited_sentinel2_collection(sudan_geometry,
                                                              SUDAN_START_DATE,
                                                             SUDAN_END_DATE)
    afghanistan_collection = create_composited_sentinel2_collection(afghanistan_geometry,
                                                                   AFGHANISTAN_START_DATE,
                                                                   AFGHANISTAN_END_DATE)
    return iran_collection, sudan_collection, afghanistan_collection


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
        collection = ee.ImageCollection(IMAGE_COLLECTION) \
            .filterBounds(roi) \
            .filterDate(interval_start_date_str, interval_end_date_str)

        # Apply the cloud mask function to the image collection
        collection = collection.map(mask_clouds)

        collection = collection.select(BANDS)

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


def mask_clouds(image):
    """
          Create a function to mask clouds using the Sentinel-2 QA60 band
    """

    QA60 = image.select(['QA60'])
    cloud_mask = QA60.bitwiseAnd(1 << 10).eq(0).And(QA60.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

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