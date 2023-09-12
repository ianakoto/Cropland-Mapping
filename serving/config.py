LABEL = "is_cropland"
IMAGE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
BANDS = [
    "B2",
    "B3",
    "B4",
    "B8"
]
FEATURES = ["NDVI", "EVI"]
SCALE = 10
PATCH_SIZE =16
BATCH_SIZE = 64


# For this Project we focus on 3 areas. 
# Change this part if you want to to focus on a different area.
# For Iran and Sudan, data can span from  july 2019 to 2022
# for Afghanistan, only data for the whole year 2022 should be used

IRAN_START_DATE = "2019-07-01"
IRAN_END_DATE = "2022-12-31"
SUDAN_START_DATE = "2019-07-01"
SUDAN_END_DATE = "2022-12-31"
AFGHANISTAN_START_DATE = "2022-01-01"
AFGHANISTAN_END_DATE = "2022-12-31"

