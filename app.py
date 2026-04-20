import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
import joblib

from ultralytics import YOLO
from skimage.measure import label, regionprops

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Food Detection and Calorie Estimation",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Food Detection & Calorie Estimation (AI + ML)")

# -----------------------------
# LOAD FILES
# -----------------------------
@st.cache_resource
def load_all():
    model = YOLO("best_new.pt")

    calib_df = pd.read_csv("calibration.csv")

    nutrition_df = pd.read_csv("nutrition.csv")
    nutrition_df["food"] = nutrition_df["food"].str.lower().str.strip()
    calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))

    count_df = pd.read_csv("count_based_config.csv")
    count_df["food"] = count_df["food"].str.lower().str.strip()
    count_weight_dict = dict(zip(count_df["food"], count_df["weight_per_item"]))

    return model, calib_df, calorie_dict, count_weight_dict

model, calib_df, calorie_dict, count_weight_dict = load_all()

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Food Image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # SAVE TEMP FILE
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
    results = model(temp_path, conf=0.25)

    annotated = results[0].plot()

    with col2:
        st.image(annotated, channels="BGR", use_container_width=True)

    # -----------------------------
    # MAIN LOGIC (FROM test_model.py)
    # -----------------------------
    total_calories = 0
    rows = []
    count_items = {}
    count = 1

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i in range(len(masks)):

            mask = (masks[i] > 0.5).astype(np.uint8)

            y_idx, x_idx = np.where(mask)
            if len(x_idx) == 0:
                continue

            x_min, x_max = x_idx.min(), x_idx.max()
            y_min, y_max = y_idx.min(), y_idx.max()
            mask = mask[y_min:y_max+1, x_min:x_max+1]

            mask_area = np.sum(mask)
            height, width = mask.shape
            bbox_area = width * height

            labeled = label(mask)
            regions = regionprops(labeled)
            if len(regions) == 0:
                continue

            region = max(regions, key=lambda r: r.area)

            perimeter = region.perimeter
            convex_area = region.area_convex
            major_axis = region.axis_major_length
            minor_axis = region.axis_minor_length

            food = model.names[int(classes[i])].lower().strip()

            # -----------------------------
            # COUNT-BASED ITEMS
            # -----------------------------
            if food in count_weight_dict:
                count_items[food] = count_items.get(food, 0) + 1
                continue

            # -----------------------------
            # FEATURE EXTRACTION
            # -----------------------------
            area_ratio = mask_area / (bbox_area + 1e-6)
            aspect_ratio = width / (height + 1e-6)
            solidity = mask_area / (convex_area + 1e-6)
            eccentricity = region.eccentricity

            equiv_diameter = np.sqrt(4 * mask_area / np.pi)
            thickness = mask_area / (bbox_area + 1e-6)
            volume_proxy = (equiv_diameter ** 2) * thickness

            roundness = (4 * np.pi * mask_area) / (perimeter**2 + 1e-6)
            compactness = (perimeter**2) / (mask_area + 1e-6)

            elongation = major_axis / (minor_axis + 1e-6)
            fill_ratio = mask_area / (convex_area + 1e-6)

            features = pd.DataFrame([{
                "area_ratio": area_ratio,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "eccentricity": eccentricity,
                "equiv_diameter": equiv_diameter,
                "thickness": thickness,
                "volume_proxy": volume_proxy,
                "roundness": roundness,
                "compactness": compactness,
                "elongation": elongation,
                "fill_ratio": fill_ratio
            }])

            # -----------------------------
            # LOAD MODELS
            # -----------------------------
            try:
                xgb = joblib.load(f"models/xgb_{food}.pkl")
                rf = joblib.load(f"models/rf_{food}.pkl")
                cols = joblib.load(f"models/cols_{food}.pkl")
            except:
                continue

            features = features[cols]

            pred_xgb = np.exp(xgb.predict(features)[0]) - 1
            pred_rf = np.exp(rf.predict(features)[0]) - 1

            pred = 0.5 * pred_xgb + 0.5 * pred_rf

            # -----------------------------
            # CALIBRATION
            # -----------------------------
            row = calib_df[calib_df["food"] == food]
            if len(row) > 0:
                pred = row["a"].values[0] * pred + row["b"].values[0]

            kcal = (pred / 100) * calorie_dict.get(food, 0)

            total_calories += kcal

            rows.append((food, pred, kcal))

    # -----------------------------
    # COUNT ITEMS PROCESSING
    # -----------------------------
    for food, cnt in count_items.items():
        weight_per_item = count_weight_dict[food]
        total_weight = cnt * weight_per_item

        kcal = (total_weight / 100) * calorie_dict.get(food, 0)
        total_calories += kcal

        rows.append((f"{food} x {cnt}", total_weight, kcal))

    # -----------------------------
    # DISPLAY TABLE
    # -----------------------------
    if rows:
        df = pd.DataFrame(rows, columns=["Food", "Weight (g)", "Calories"])
        st.dataframe(df, use_container_width=True)

        st.success(f"🔥 Total Calories: {total_calories:.2f} kcal")

    else:
        st.warning("No food detected.")

    os.remove(temp_path)
