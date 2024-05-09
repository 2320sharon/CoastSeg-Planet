# CoastSeg-Planet
An extension for CoastSeg that uses planet imagery

The goal of this CoastSeg extension is to extract shorelines from Planet imagery. CoastSeg-Planet aims to utilize rio-xarray and Dask to extract shorelines from imagery faster than CoastSeg and to allow for post processing workflows that can compute average, seasonal and other shorelines using rio-xarray's functionality.

Currently the team behind CoastSeg Planet is researching how to co-register Planet imagery to LandSat imagery as well as exploring tools that could be used.

### Prototype Version 1 Diagram
![CoastSeg Planet-Current Prototyp drawio](https://github.com/2320sharon/CoastSeg-Planet/assets/61564689/cf6a4937-cd1c-49c9-ae37-269867aee030)

# Research

## Data Requirements
- `4-band multispectral Analytic Ortho Scene` from Planet
- Recommend users to clip to AOI to limit file size

## Co-Registeration
- CoastSeg downloads LandSat as TOA imagery from the tier 1 TOA collection which saves all the landsat values as 32 bit floats instead of unsigned 16 bit ints
- CoastSeg-planet includes a script to convert 32bit float TOA imagery into unsigned 16 bit imagery that can be co-registered with the [arosics](https://git.gfz-potsdam.de/danschef/arosics) `COREG`` function
  

# Installation

