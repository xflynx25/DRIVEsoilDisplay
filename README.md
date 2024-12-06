# DRIVEsoilDisplay

## Purpose
Interpretable visualiation of soil sensor measurements, to aid in scientific decision making, and the furtherment of the IOT-soil project. Smart Agriculture. 


## Workflow 
1. Receive data from trusted source (RSSC), record results in the **input_data_source.csv** file
    
    - If only values are given, leave the rating field blank, and run **add_ratings.py** to automatically fill them in

2. Copy files from some experiment in the **FullDataArchive folder**, into the **ActiveExperiment** folder
    
    - to acquire new data, use the method from the acquisition repo to create an experiment folder 

3. Fill out the **input_measured_data.csv** if you used additional methods for measurement

    - ideally, you will be able to find this from some past experiments in the FullDataArchive

4. Run **processRawData.py** to generate the sample statistics (mean) from the timeseries data. 

    - in this file, you can control if you also want plot generation, just alter the  **plots_to_generate** variable. 

5. Now run **combine_csvs.py** or **combine_csvs_multiple.py** depending on the use case. If you used multiple measurement styles for each sample, utilizing a suffix to signify this (see **FullDataArchive/Nov2024_20samples** for an example), then use the multiple variant to produce multiple files. Otherwise, you can simply use combine_csvs.py. 

6. Finally, run **app.py** and navigate to your browser to see the visualization. 

    - NOTE: you can immediately jump to this step if wanting to visualize a previous experiment which has the combined files in the **CompressedData** folder. Just copy the combined*.csv files into the root directory. 



## Environment

### Setting Up the Environment

To set up the Conda environment for this project, follow these steps:

1. **Install Conda**:
   - If you don’t have Conda installed, download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

2. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

3. **Create the Conda Environment**:
   Run the following command to create the environment from the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

4. **Activate the Environment**:
   ```bash
   conda activate my_environment
   ```

5. **Verify Installation**:
   Ensure all required packages are installed:
   ```bash
   conda list
   ```

6. **Deactivating the Environment** (Optional):
   When done working in the environment, deactivate it using:
   ```bash
   conda deactivate
   ```

---

### Optional Step: Updating the Environment

If you make changes to the environment and want to update the `environment.yml` file, re-run the export command:
   ```bash
   conda env export --no-builds > environment.yml
   ```

To update an existing environment from the modified `environment.yml`, use:
   ```bash
   conda env update -f environment.yml
   ```





## Timeline 

Oct.24 = Built breadboard prototype to measure multiple sensors simultaneously. Synthesis plots allow comparison between different trials. Simple web-app allows timeseries visualization. 

Nov.24 = Significant data collection occurs. A set of 5 samples are provided (+2 of our own), then 20 more. What humidity to measure at is a key question. The present repository is developed. A custom 3d printed measurement device is made to allow for minimal soil usage. 

Dec.24 = Sensors seem to have corroded. Uncorroded ones and pH probe showing moderate results on pH. No data for EC provided by RSSC. NPK is just estimated and we need alternatives. In talks with Punakha Biofablab for ideas/support for NPK methods. 

Next steps should be 
* to get new resistive sensors and treat them properly (so don't corrode)
* rigorously look for the best repeatable measurement procedure before mass data collection
* ask for EC in the next batch from RSSC, and determine if resistive sensors are sufficient for PH and EC 
* look into method for verification of Temperature and Humidity 
* order simple capacitive sensors (or make our own). Should be able to do humidity and often do temp. can cross check against the resistive. 
* Investigate NPK, likely ion-selective electrodes or spectroscopy needed. With spectroscopy, may be able to get additional nutrient detection, at the cost of less accurate P,K and possibly a more difficult user interface.
