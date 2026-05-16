# Egyptian Archaeological Site Looting Dataset

![Project Banner](data/NotLooted/NotLooted105.jpg) 

[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/abdelazizamr837/egyptian-archaeological-site-looting)

## ?? Project Overview
This project focuses on identifying archaeological looting pits in Egypt using satellite imagery. Following the security vacuum of 2011, many historical sites were subjected to illegal excavations. These "pits" are clearly visible from space, and this dataset was created to provide a foundation for automated detection.

## ?? Dataset Details
The dataset consists of **230 image patches** categorized into two classes:
*   **Looted:** Image patches showing clear evidence of illegal excavation pits.
*   **Not Looted:** Image patches showing pristine terrain or natural desert features.

**Dataset Link:** [Access on Kaggle](https://www.kaggle.com/datasets/abdelazizamr837/egyptian-archaeological-site-looting)

## ?? Data Sourcing & Methodology
The images in this dataset were curated using the following process:

*   **Source:** Captured from **Google Earth Pro’s** historical imagery archive.
*   **Timeframe:** Primarily focused on imagery from **2010 to 2016**, capturing the documented surge in looting activity.
*   **Locations:** Key archaeological sites including **Dashur, Lisht, and Saqqara** were targeted based on academic research regarding looting hotspots.
*   **Labeling:** Image patches were manually inspected and tightly cropped to ensure the visual features of the looting pits are the primary focus.

## ?? Repository Structure
\\\	ext
+-- data/
¦   +-- Looted/        # Image patches with looting pits
¦   +-- NotLooted/     # Image patches of pristine ground
+-- README.md
\\\

## ?? Future Work
The next phase of this project involves training a **Convolutional Neural Network (CNN)** to automatically classify these patches, with the goal of creating a monitoring tool for cultural heritage protection.

---
**Dataset Created by:** [Abdelaziz amr]
