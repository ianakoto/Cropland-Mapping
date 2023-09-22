IMAGE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
SENTINEL_CLOUD_PROB_ID = "COPERNICUS/S2_CLOUD_PROBABILITY"
DYNAMIC_WORLD_ID = "GOOGLE/DYNAMICWORLD/V1"
MODIS_LANDCOVER_ID = "MODIS/061/MCD12Q1"
CLOUD_PIXEL_PERCENTAGE = 20  # Define a cloud pixel threshold (e.g., 20%).
CLOUD_DISPLACEMENT_THRESHOLD = 0.2
BANDS = ["B2", "B3", "B4", "B8"]
FEATURES = ["NDVI", "NDWI", "EVI"]
LABEL = "is_cropland"

SCALE = 10
# For this Project we focus on 3 areas.
# Change this part if you want to to focus on a different area.
# For Iran and Sudan, data can span from  july 2019 to 2022
# for Afghanistan, only data for the whole year 2022 should be used

IRAN_START_DATE = "2019-07-01"
IRAN_END_DATE = "2020-06-30"
SUDAN_START_DATE = "2019-07-01"
SUDAN_END_DATE = "2020-06-30"
AFGHANISTAN_START_DATE = "2022-05-01"
AFGHANISTAN_END_DATE = "2022-05-31"
