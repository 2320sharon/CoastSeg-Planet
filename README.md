# CoastSeg-Planet
An extension for CoastSeg that uses planet imagery

The goal of this CoastSeg extension is to extract shorelines from Planet imagery. CoastSeg-Planet aims to utilize rio-xarray and Dask to extract shorelines from imagery faster than CoastSeg and to allow for post processing workflows that can compute average, seasonal and other shorelines using rio-xarray's functionality.

Currently the team behind CoastSeg Planet is researching how to co-register Planet imagery to LandSat imagery as well as exploring tools that could be used.

## Generic Workflow for Extracting Shorelines
- Users will need to provide their own ROI and transects as geojson files in order to use CoastSeg Planet
![CoastSeg Planet-Planet Extract Shorelines](https://github.com/2320sharon/CoastSeg-Planet/assets/61564689/166d06d1-d976-4343-83fd-18c7e9fa327f)


### Prototype Version 1 Diagram
![CoastSeg Planet-Current Prototyp drawio](https://github.com/2320sharon/CoastSeg-Planet/assets/61564689/cf6a4937-cd1c-49c9-ae37-269867aee030)


# Research

## Data Requirements
- `4-band multispectral Analytic Ortho Scene` from Planet
   - According to [Planet's asset types](https://developers.planet.com/docs/data/psscene/#available-asset-types) this asset is "Radiometrically-calibrated analytic image stored as 16-bit scaled radiance."
   - These radiance values need to be converted to TOA using a TOA conversion function


## Co-Registeration
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


# Installation

CoastSeg Planet is under active development so the dependencies are subjuct to change

```
conda create --name coastseg_planet python=3.10 -y
conda activate coastseg_planet
conda install -c conda-forge coastseg arosics -y
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

```
conda activate coastseg_planet
cd <location you installed coastseg-planet>
```
