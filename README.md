# CoastSeg-Planet
An extension for CoastSeg that uses planet imagery

The goal of this CoastSeg extension is to extract shorelines from Planet imagery. CoastSeg-Planet aims to utilize rio-xarray and Dask to extract shorelines from imagery faster than CoastSeg and to allow for post processing workflows that can compute average, seasonal and other shorelines using rio-xarray's functionality.

Currently the team behind CoastSeg Planet is researching how to co-register Planet imagery to LandSat imagery as well as exploring tools that could be used.

# Installation

## 1. Install the Source Code

```
git clone https://github.com/2320sharon/CoastSeg-Planet.git
cd CoastSeg-Planet
```

## 2. Install the Dependencies
CoastSeg Planet is under active development so the dependencies are subjuct to change.

Run the code below to create a conda environment that will allow you to download planet imagery and extract shorelines.

Make sure to run all the code below within the `CoastSeg-Planet` directory where the `pyproject.toml` file is located, otherwise the `pip install -e .` command will not work.

```
conda create --name coastseg_planet python=3.10 -y
conda activate coastseg_planet
conda install -c conda-forge coastseg -y
cd <location you installed coastseg-planet>
pip install -e .
pip install tensorflow==2.15
pip install transformers
```

# Step 1: Download Imagery
## 1. Get your planet API key
Copy the planet API key.
![image](https://github.com/user-attachments/assets/efa063b3-2f14-4406-936f-0129cf01e0b7)
## 2. Add it to the config.ini file
Paste it into the config.ini file
![image](https://github.com/user-attachments/assets/b505e4d4-6e6e-45b4-90fe-166fcc9ea807)
## 3. Activate the environment
```
conda activate coastseg_planet
cd <location you installed coastseg-planet>
```
## 4. Run the Script `download_script.py`
Before you run this script make sure to have geojson file containing an ROI.
Then edit this file to select:
- date range
- order name
- max cloud cover
Save the changes, then run the script when you are ready.

The default settings in the download script will create a brand new order and it can take anywhere from 10 minutes to a few hours for planet to finish creating your order during which the status will show "Running". Due to how long it takes you might need to set the  `continue_existing` to be `continue_existing=True` if the script errors out because Planet took too long to create your order.

 Large orders will be split into several small orders automatically by the download script, which is why when you place 1 large order you may see several smaller orders appear on your dashboard. This allows us to place orders larger than the limit set by Planet.



### To Finish Downloading an Existing Order
To finish downloading an existing order set `continue_existing` in `download.download_order_by_name`  to be `continue_existing=True`. This tells the Planet API you want to get your existing order, NOT start a new order with the same name.

### To Overwrite Existing Order
To finish downloading an existing order set `overwrite` in `download.download_order_by_name`  to be `overwrite=True`. This tells the Planet API you want to create an order with the same name that already exists. I don't recommended doing this as it wastes resources, but it may be necessary if you accidently entered the wrong details on your original order

### To Coregister Using the Planet API
To coregister all the imagery in your order `coregister` in `download.download_order_by_name`  to be `coregister=True`. This tells the Planet API you want coregister all the imagery in each order to a single scene. WARNING for large orders that have to be split into several suborders this will result in each suborder being registered to a scene in its order. For example 1 large order split into 4 suborders will result in 4 scenes that are used as the reference scene for coregistration.

Also this cannot be combined with `continue_existing` or `overwrite` as this setting can only be set for a new order NOT an existing order

### Run the Script
```
conda activate coastseg_planet
cd <location you installed coastseg-planet>
python download_script.py
```
### 5. Wait for your order to finish
The planet API prepares your order after you request it. If you realize that you made the wrong order and it hasn't finished on Planet yet, you can cancel the order with the `cancel_order.py` script. However if your order was submitted and is being prepared this script won't work. 

#### Important - What to do for large orders
It can take anywhere from 10 minutes to a few hours for planet to finish your order. During this time the script will show a status of "Running". If it takes too long for Planet to prepare the order the script will eventually quit because the Planet API took too long to finish your order. If that happens change `continue_existing` in `download.download_order_by_name`  to be `continue_existing=True`. This tells the Planet API you want to get your existing order, NOT start a new order with the same name

```
asyncio.run(
    download.download_order_by_name(
        order_name,
        output_path,
        roi,
        start_date,
        end_date,
        overwrite=False,            # if True will overwrite an existing order with the same name and download the new one
        continue_existing=True,    # CHANGE THIS LINE
        cloud_cover=CLOUD_COVER,
        product_bundle="analytic_udm2",
        coregister=False,           # if True will coregister the images to the image with the lowest cloud cover using Planet's coregistration service
    )
)
```

In the screenshot below you can see the orders created on the Planet dashboard. You can also download orders from here and check their status.

![image](https://github.com/user-attachments/assets/3bb06930-9c9c-4d18-9096-02b6b1cdc637)

# Extract Shorelines From a Planet Order 
#### Script : `extract_shorelines_for_order.py`
## Phase 1 : Prepare the Data
### 1. Move all the suborder to a single folder (only for large orders)
- If you had a large order `CoastSeg Planet` automatically will split your order into sub orders and place them each in their own folder under your order.
- Move all the files from the subfolders into one directory

| Small Order | Large Order |
|-------------|-------------|
| ![Small Order](https://github.com/user-attachments/assets/81e10727-4637-465a-b54e-42ccb92d9af0) | ![Large Order](https://github.com/user-attachments/assets/51a70d8c-dadd-42e5-a271-e302b41753fb) |
| No need to move files | Move all the tif, json, xml files to a single directory |

![image](https://github.com/user-attachments/assets/875c83f0-072e-4ee4-abb9-27eb9fcc1882)

### 2. Get a Shoreline GeoJSON File

Use coastseg to get a geojson containing only the reference shoreline. Use the Save Feature to File button in CoastSeg pictured below.

![image](https://github.com/user-attachments/assets/9bd2e252-fa9b-40de-95c2-c8b91abfe7fb)

### 3. Get a Transects GeoJSON File

Use coastseg to get a geojson containing only the transects. Use the Save Feature to File button in CoastSeg pictured below.

![image](https://github.com/user-attachments/assets/9bd2e252-fa9b-40de-95c2-c8b91abfe7fb)

## Phase 2 : Extract Shorelines

![CoastSeg Planet-Extract Shorelines Ref SL](https://github.com/user-attachments/assets/c27386dd-5132-4fc0-8561-ea72421c7171)


### 1. Edit the Extract Shoreline Settings
- `max_dist_ref` Enter the size in meters of the reference shoreline buffer. This is the region in which shorelines can be extracted shown in greenish yellow
![2023-06-16-15-30-32_planet](https://github.com/user-attachments/assets/adf39ffc-0a74-4e5f-838b-a5f9bce65286)

- `dist_clouds`: Enter the distance from the clouds that the shorelines must be.

- `min_beach_area`: Enter the minimum area the beach must be in order to be identified as land. This filters the land mask to remove any segments of land that are less than this area. Usually these are misclassifications

- `min_length_sl`: Enter the minimum length of the shoreline in order for a segment to be considered a valid shoreline.

### 2. Enter the Locations of the ROI, Transects, and Shoreline Files
```
# Enter the locations of the feature inputs
# ----------------
# 1. Enter the path ROI = Region of Interest used to download your order
roi_path = os.path.join(os.getcwd(),"sample_data", "rois.geojson")
# 2. Enter the path to the transects file
# transects_path = r"C:\development\coastseg-planet\downloads\Alaska_TOAR_enabled\42444c87-d914-4330-ba4b-3c948829db3f\PSScene\transects.geojson"
transects_path =os.path.join(os.getcwd(),"sample_data", "transects.geojson")
# 3. Enter the path to the reference shoreline file
# shoreline_path = r"C:\development\coastseg-planet\downloads\Alaska_TOAR_enabled\42444c87-d914-4330-ba4b-3c948829db3f\PSScene\good\shoreline.geojson"
shoreline_path =os.path.join(os.getcwd(),"sample_data", "shoreline.geojson")

```


### 3. Enter the Location where all the Imagery was Downloaded
- This is the directory that contains all the .tif, .xml and .json files
- These should all be in the SAME directory NOT SUBDIRECTORIES

```
# 4. Enter the location of directory containing the downloaded imagery from Planet
#   - Make sure this directory contains the tif,json,and xml files for the entire order
planet_dir = ''
planet_dir = r"C:\development\coastseg-planet\CoastSeg-Planet\downloads\DUCK_pier_cloud_0.7_TOAR_enabled_2023-06-01_to_2023-08-01\31eea859-d1dd-46c2-b6a4-7c15871be926\PSScene"

```

### 4. Run the Script   
`python extract_shorelines_for_order.py`

# Outputs of Extracting Shorelines

All of these outputs will be saved in the good directory of directory where all the planet imagery used to extract shorelines was saved.

| Filename | Data | Screenshot |
|----------|----------|----------|
| raw_extracted_shorelines.geojson   | Raw extracted shorelines as multilinestring vectors   | ![image](https://github.com/user-attachments/assets/9dd35a17-3a90-4810-81e4-be4c342c95ea)|
| raw_transect_time_series.csv   | A csv file containing the shoreline position along the transect for each date and transect id   | ![image](https://github.com/user-attachments/assets/fbe927a4-8ef7-40b5-a492-8a4ebc819ad3)|
| raw_transect_time_series_merged.csv    | A csv file containing the shoreline position along the transect, the XY point of the shoreline, and the XY point of the seaward point of the transect,  for each date and transect id   | ![image](https://github.com/user-attachments/assets/24a36540-714a-4b62-9b57-50be8ab136ff)|
| raw_transect_time_series_points.geojson    | X and Y point of where each median shoreline intersected the transect for each date   | ![image](https://github.com/user-attachments/assets/e994ae7a-4e72-4bca-96dd-8dbf875c3b06)|
| raw_transect_time_series_vectors.geojson    |  a shoreline vector created from raw_transect_time_series_points.geojson this is the shoreline generated by intersecting the shoreline captured on each date with the transects    | ![image](https://github.com/user-attachments/assets/7ac12bfa-91d0-49fe-b946-39368a3005f1)|

# Diagrams

## Generic Workflow for Extracting Shorelines
- Users will need to provide their own ROI and transects as geojson files in order to use CoastSeg Planet
![CoastSeg Planet-Planet Extract Shorelines](https://github.com/2320sharon/CoastSeg-Planet/assets/61564689/166d06d1-d976-4343-83fd-18c7e9fa327f)


# Research

## Data Requirements
- `4-band multispectral Analytic Ortho Scene` from Planet
   - According to [Planet's asset types](https://developers.planet.com/docs/data/psscene/#available-asset-types) this asset is "Radiometrically-calibrated analytic image stored as 16-bit scaled radiance."
   - These radiance values need to be converted to TOA using a TOA conversion function


## Co-Registration
- CoastSeg downloads LandSat as TOA imagery from the tier 1 TOA collection which saves all the landsat values as 32 bit floats instead of unsigned 16 bit ints
- CoastSeg-planet includes a script to convert 32bit float TOA imagery into unsigned 16 bit imagery that can be co-registered with the [arosics](https://git.gfz-potsdam.de/danschef/arosics) `COREG`` function


## Planet Downloads VS CoastSeg Downloads
### LandSat 8
- Both Planet & CoastSeg can download L8 but download it from different collections
- CoastSeg downloads imagery from the  tier 1 TOA collection
   - All the landsat values are saved as 32 bit floats 
   - These images CANNOT be co-registered using the [arosics](https://git.gfz-potsdam.de/danschef/arosics) `COREG`` function until they are converted to `unit16`
- Planet downloads imagery from the tier 1  collection
   - All the landsat values are saved as unsigned 16 bit ints
   - Planet by default downloads each band separately so users will need to combine these bands together
   - 



