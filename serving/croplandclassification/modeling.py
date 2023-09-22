
from __future__ import annotations


from datetime import datetime, timedelta
import ee
from .config import *
import time

from google.api_core import exceptions, retry
import google.auth


import os
import joblib


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


def classify_dataframe_with_regions(df, region_geometries, image_collections, classifiers, bands):
    """
    Classify the DataFrame using the appropriate classifier based on the region and update the DataFrame with predictions.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns: 'id', 'lat', 'long'.
        region_geometries (dict): Dictionary with region names as keys and their respective geometry as values.
        image_collections (dict): Dictionary with region names as keys and their respective ImageCollection as values.
        classifiers (dict): Dictionary with region names as keys and their respective classifier as values.

    Returns:
        pd.DataFrame: DataFrame with predictions added as a new column 'prediction'.
    """
    # Function to extract band values for a given lat, lon and make predictions
    def classify_point(lat, lon):
        point = ee.Geometry.Point(lon, lat)
        region = None

        # Determine the region for the point
        for reg, geom in region_geometries.items():
            if geom.contains(point):
                region = reg
                break

        if region:
            # Extract band values for the appropriate region
            image = image_collections[region].filterBounds(point).first()

            values = image.select(bands).reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=10
            )
            features = ee.Feature(point, values)

            # Apply the appropriate classifier to make predictions
            classified_features = features.classify(classifiers[region])

            # Retrieve the prediction
            prediction = classified_features.get('classification')

            return prediction
        else:
            print(f"No region found for lat: {lat}, lon: {lon}")
            return None

    # Make predictions for each point in the DataFrame
    predictions = []
    for _, row in df.iterrows():
        lat = row['lat']
        lon = row['long']
        prediction = classify_point(lat, lon)
        if prediction is not None:
            predictions.append(prediction.getInfo())
        else:
            predictions.append(None)

    # Update the DataFrame with the predictions
    df['prediction'] = predictions

    return df


def save_best_models(best_models, base_directory):
    """
    Save the best models with a professional directory structure.

    Parameters:
        best_models (dict): A dictionary containing the best models information.
        Structure: {region: {"best_model_name": str, "best_accuracy": float, "best_model": model}}
        base_directory (str): Base directory to save the models'.
    """
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # Iterate through the regions and save the models
    for region, model_info in best_models.items():
        model_name = model_info["best_model_name"]
        model = model_info["best_model"]

        # Create a subdirectory for each region
        region_directory = os.path.join(base_directory, region)
        if not os.path.exists(region_directory):
            os.makedirs(region_directory)

        # Save the model using joblib
        model_filename = os.path.join(
            region_directory, f'{region}_{model_name}.joblib')
        joblib.dump(model, model_filename)
        print("Models Saved")


def sample_features_by_class(table, sample_per_class_type):
    """
    Randomly samples points from a table based on the 'Target' property.

    Parameters:
    table (ee.FeatureCollection): Earth Engine FeatureCollection to sample from.
    sample_per_class_type (int): Number of samples per class (cropland and non-cropland).

    Returns:
    ee.FeatureCollection: A FeatureCollection containing the final sampled points.
    """

    # Filter based on the "target" property (cropland or not cropland)
    cropland_features = table.filter(ee.Filter.eq('Target', 1))
    non_cropland_features = table.filter(ee.Filter.eq('Target', 0))

    # Randomly sample points from each category
    sampled_cropland = cropland_features.randomColumn().sort(
        'random').limit(sample_per_class_type)
    sampled_non_cropland = non_cropland_features.randomColumn().sort(
        'random').limit(sample_per_class_type)

    # Combine the sampled points into a single FeatureCollection
    final_sampled_features = sampled_cropland.merge(sampled_non_cropland)

    # Print the size of the final sampled FeatureCollection
    print('Final sampled FeatureCollection size:',
          final_sampled_features.size().getInfo())

    return final_sampled_features


def train_and_evaluate_models(region, region_collection, final_sampled_features):
    """Train and evaluate models for a specific region.

    Args:
        region (str): The name of the region.
        region_collection (ee.ImageCollection): The collection for the specified region.
        final_sampled_features (ee.FeatureCollection): Sampled features for training.

    Returns:
        dict: Dictionary containing the best model and its accuracy for the region.
    """
    # Define training features
    TRAIN_FEATURES = ["NDWI", "NDVI", "B2", "EVI"]
    y = 'Target'  # Target property indicating cropland (1) or not (0)

    # Models to train
    models = [
        {"name": "Gradient Tree Boosting",
            "classifier": ee.Classifier.smileGradientTreeBoost(numberOfTrees=350)},
        {"name": "CART", "classifier": ee.Classifier.smileCart()},
        {"name": "Random Forest",
            "classifier": ee.Classifier.smileRandomForest(numberOfTrees=350)},
        {"name": "Minimum Distance",
            "classifier": ee.Classifier.minimumDistance('mahalanobis')},
        {"name": "Naive Bayes", "classifier": ee.Classifier.smileNaiveBayes()},
        {"name": "SVM", "classifier": ee.Classifier.libsvm(
            kernelType='RBF', gamma=0.95, cost=10)}
    ]

    # Dictionary to store the best model, its accuracy, and all trained models
    best_model_info = {"best_model_name": None,
                       "best_accuracy": 0, "best_model": None}

    print(f"\nTraining and evaluating models for {region}...")

    # Merge the region-specific collection
    merged_collection = region_collection

    # Select and mosaic the training images
    # you can also try mean, mode, median
    train_images = merged_collection.select(TRAIN_FEATURES).max()

    # Prepare the training data
    training = train_images.sampleRegions(
        collection=final_sampled_features,
        properties=[y],
        scale=10
    )

    print("\nTraining data prepared.")

    # Adds a column of deterministic pseudorandom numbers.
    sample = training.randomColumn()
    split = 0.8

    # Split the data into train and test sets
    X_train = sample.filter(ee.Filter.lt('random', split))
    X_test = sample.filter(ee.Filter.gte('random', split))

    print("\nData split into training and testing sets.")

    # Train and evaluate different models
    for model_info in models:
        model_name = model_info["name"]
        print(f"\nTraining {model_name} model...")

        classifier = model_info["classifier"]
        # Train the model
        classifier = classifier.train(
            features=X_train,
            classProperty=y,
            inputProperties=TRAIN_FEATURES
        )

        validated = X_test.classify(classifier)

        # Calculate test accuracy
        test_accuracy = validated.errorMatrix(
            'Target', 'classification').accuracy().getInfo()

        print(f"\nTest accuracy for {model_name}: {test_accuracy}")

        print(f"\n{model_name} model trained.")

        # Update the best model for the region based on test accuracy
        if test_accuracy > best_model_info["best_accuracy"]:
            best_model_info["best_model_name"] = model_name
            best_model_info["best_accuracy"] = test_accuracy
            best_model_info["best_model"] = classifier

    print(
        f"\nBest model for {region}: {best_model_info['best_model_name']}, Test Accuracy: {best_model_info['best_accuracy']}")

    return {region: best_model_info}
