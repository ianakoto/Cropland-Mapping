from datetime import datetime, timedelta
import ee
import numpy as np
from config import *


countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

# Filter countries by name to obtain the ROIs for Iran, Sudan, and Afghanistan
iran_roi = countries.filter(ee.Filter.eq("country_na", "Iran"))
sudan_roi = countries.filter(ee.Filter.eq("country_na", "Sudan"))
afghanistan_roi = countries.filter(ee.Filter.eq("country_na", "Afghanistan"))

# Get the geometries of the ROIs
iran_geometry = iran_roi.geometry()
sudan_geometry = sudan_roi.geometry()
afghanistan_geometry = afghanistan_roi.geometry()


def get_prediction_data(
    lon, lat, iran_collection, sudan_collection, afghanistan_collection
):
    """Extracts Sentinel image as json at specific lat/lon and timestamp."""

    location = ee.Feature(ee.Geometry.Point([lon, lat]))

    selected_collection = select_collection_by_point(
        location,
        iran_geometry,
        sudan_geometry,
        afghanistan_geometry,
        iran_collection,
        sudan_collection,
        afghanistan_collection,
    )

    image = selected_collection.mosaic()

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
        afghanistan_collection,
    )

    image = selected_collection.mosaic()
    point = ee.Feature(
        select_point,
        {LABEL: row.Target},
    )
    return (
        image.neighborhoodToArray(ee.Kernel.square(PATCH_SIZE))
        .sampleRegions(ee.FeatureCollection([point]), scale=SCALE)
        .first()
    )


def sample_cropland_points(scale=500, sample_size=1000):
    """
    Sample cropland points from a specified Earth Engine dataset for Iran, Sudan, and Afghanistan.

    Args:
        scale (int, optional): The scale at which to sample points. Defaults to 500.
        sample_size (int, optional): The number of points to sample per region. Defaults to 1000.

    Returns:
        ee.FeatureCollection, ee.FeatureCollection: Sampled cropland points for Iran and Sudan, Sampled cropland points for Afghanistan.

    """

    # Load the specified dataset
    iran_sudan_dataset = get_training_dataset(
        MODIS_LANDCOVER_ID, IRAN_START_DATE, IRAN_END_DATE, "LC_Type1"
    )

    iran_sampled = export_cropland_samples(
        iran_sudan_dataset, iran_geometry, 12, 1000, 500
    )

    sudan_sampled = export_cropland_samples(
        iran_sudan_dataset, sudan_geometry, 12, 1000, 500
    )

    afghan_dataset = get_training_dataset(
        DYNAMIC_WORLD_ID, AFGHANISTAN_START_DATE, AFGHANISTAN_END_DATE, "crop"
    )

    afghan_sampled = export_cropland_samples(
        afghan_dataset, afghanistan_geometry, 4, 500, 500
    )

    return iran_sampled, sudan_sampled, afghan_sampled


def export_cropland_samples(image_collection, class_label, region, sample_size, scale):
    """
    Export cropland samples from an image collection within a specified region.

    Parameters:
        image_collection (ee.ImageCollection): The Earth Engine ImageCollection to analyze.
        region (ee.Geometry): The region of interest (ROI) defined as an Earth Engine Geometry.
        class_label (int):  The class label for the cropland
        sample_size (int): The number of random sample points to generate.
        scale (int): The scale (pixel size) for stratified sampling.

    Returns:
        ee.FeatureCollection: A FeatureCollection containing cropland samples with added ID, Latitude, and Longitude properties.
    """

    # Create a binary mask where cropland pixels are assigned 1 and others as 0
    cropland_mask = image_collection.eq(
        class_label
    )  # 12 represents cropland in the MODIS dataset, 4 for dynamic world

    # Convert the binary mask to an image with 1s and 0s
    cropland_image = cropland_mask.rename("Cropland")

    # Clip the image to the specified ROI
    cropland_image_clipped = cropland_image.clip(region)

    # Create random sample points for cropland (class 1)
    cropland_samples = (
        cropland_image_clipped.select("Cropland")
        .toInt()
        .stratifiedSample(
            classBand="Cropland",  # Specify the band used for stratification
            scale=scale,
            numPoints=sample_size,
            region=region,
            geometries=True,
        )
    )

    # Create an ID property for each feature
    def add_id(feature):
        return feature.set(
            "ID", ee.String("ID_").cat(ee.String(ee.Number(feature.id())))
        )

    samples_with_id = cropland_samples.map(add_id)

    # Split the geometry into latitude and longitude
    def split_geometry(feature):
        lat_lon = ee.Geometry(feature.geometry()).coordinates()
        return feature.setMulti(
            {"Latitude": lat_lon.get(1), "Longitude": lat_lon.get(0)}
        )

    samples_with_lat_lon = samples_with_id.map(split_geometry)

    return samples_with_lat_lon


def get_training_dataset(id, start_date, end_date, band_name):
    """
    Retrieve and process an Earth Engine image collection to create a training dataset.

    Parameters:
        id (str): The Earth Engine image collection ID to retrieve.
        start_date (str): The start date for filtering the image collection (e.g., "YYYY-MM-DD").
        end_date (str): The end date for filtering the image collection (e.g., "YYYY-MM-DD").
        band_name (str): The name of the band to select and process.

    Returns:
        ee.Image: A processed image representing the maximum value of the specified band
                   within the specified date range.
    """
    # Load the specified image collection, filter by date, and select the specified band
    dataset = (
        ee.ImageCollection(id).filterDate(start_date, end_date).select(band_name).max()
    )

    return dataset


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
    iran_collection = create_composited_sentinel2_collection(
        iran_geometry, IRAN_START_DATE, IRAN_END_DATE
    )
    sudan_collection = create_composited_sentinel2_collection(
        sudan_geometry, SUDAN_START_DATE, SUDAN_END_DATE
    )
    afghanistan_collection = create_composited_sentinel2_collection(
        afghanistan_geometry, AFGHANISTAN_START_DATE, AFGHANISTAN_END_DATE
    )
    return iran_collection, sudan_collection, afghanistan_collection


def select_collection_by_point(
    point,
    iran_geometry,
    sudan_geometry,
    afghanistan_geometry,
    iran_collection,
    sudan_collection,
    afghanistan_collection,
):
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


def create_composited_sentinel2_collection(
    roi,
    start_date_str,
    end_date_str,
    interval=15,
    limit=10,
    include_ndvi=True,
    include_ndwi=True,
    include_evi=True,
):
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
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # Create an empty ImageCollection to store the composites
    composited_collection = ee.ImageCollection([])

    # Loop through the dates and create composites
    while start_date < end_date:
        # Calculate the end date for the current interval
        interval_end_date = start_date + timedelta(days=interval)
        if interval_end_date > end_date:
            interval_end_date = end_date

        # Format dates as strings
        interval_start_date_str = start_date.strftime("%Y-%m-%d")
        interval_end_date_str = interval_end_date.strftime("%Y-%m-%d")

        # Filter Sentinel-2 data for the current date range
        collection = (
            ee.ImageCollection(IMAGE_COLLECTION)
            .filterBounds(roi)
            .filterDate(interval_start_date_str, interval_end_date_str)
        )

        # Apply the cloud & shadow mask function to the image collection
        collection = collection.map(mask_clouds_and_shadows)

        # Calculate NDVI and EVI if requested
        if include_ndvi:
            collection = collection.map(calculate_ndvi)
        if include_ndwi:
            collection = collection.map(calculate_ndwi)
        if include_evi:
            collection = collection.map(calculate_evi)

        collection = collection.select([BANDS] + [FEATURES])
        # Create a median composite of the image collection
        collection = collection.map(normalize_sentinel2)
        median_image = collection.median()

        # Add the composite to the ImageCollection
        composited_collection = composited_collection.merge(
            ee.ImageCollection([median_image])
        )

        # Move to the next interval
        start_date = interval_end_date + timedelta(days=1)

    return composited_collection


def mask_clouds_and_shadows(image):
    """
    Define a function to mask clouds and their shadows.
    Load the Sentinel-2 Cloud Probability (S2C) product.
    """
    s2c = (
        ee.ImageCollection(SENTINEL_CLOUD_PROB_ID)
        .filterBounds(image.geometry())
        .filterDate(image.date())
    )

    # Select the cloud probability band.
    cloud_probability = s2c.first().select("probability")

    # Create a cloud mask by thresholding the cloud probability.
    cloud_mask = cloud_probability.lt(CLOUD_THRESHOLD)

    # Create a shadow mask using the cloud probability and Sentinel-2 image's cirrus band (B10).
    cloud_displacement = image.select("B10").subtract(cloud_probability).abs()
    shadow_mask = cloud_displacement.lt(
        CLOUD_DISPLACEMENT_THRESHOLD
    )  # Adjust the threshold as needed.

    # Combine the cloud mask and shadow mask.
    final_mask = cloud_mask.And(shadow_mask)

    # Apply the mask to the image.
    masked_image = image.updateMask(final_mask)

    return masked_image


def normalize_sentinel2(image):
    """
    Define a function to normalize Sentinel-2 data.
    """
    # Normalize the reflectance values to the range [0, 1].
    normalized_image = image.divide(10000)

    return normalized_image


def mask_clouds(image):
    """
    Create a function to mask clouds using the Sentinel-2 QA60 band
    """

    QA60 = image.select(["QA60"])
    cloud_mask = QA60.bitwiseAnd(1 << 10).eq(0).And(QA60.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)


def calculate_ndvi(image):
    """Calculate NDVI for an image."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def calculate_ndwi(image):
    """Calculate NDWI for an image"""
    nir_band = "B8"  # Near-Infrared (NIR) band
    swir_band = "B11"  # Shortwave Infrared (SWIR) band
    ndwi = image.normalizedDifference([nir_band, swir_band]).rename("NDWI")
    return image.addBands(ndwi)


def calculate_evi(image):
    """Calculate EVI for an image."""
    evi = image.expression(
        "2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))",
        {"B8": image.select("B8"), "B4": image.select("B4"), "B2": image.select("B2")},
    ).rename("EVI")
    return image.addBands(evi)
