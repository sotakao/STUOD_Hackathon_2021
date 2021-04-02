# ERC Stochastic Transport in Upper Ocean Dynamics Hackathon March 29-31 2021
![stuod logo](https://www.imperial.ac.uk/ImageCropToolT4/imageTool/uploaded-images/erc-stuod-logos--tojpeg_1572609986634_x2.jpg)

[Link to the event website](https://www.imperial.ac.uk/events/129398/stuods-hackathon/)

## Challenges 1+2+3: Machine Learning with Weather and Climate Data

__Project leads:__ So Takao (UCL), Bertrand Chapron (IFREMER)

Some weather/climate events such as local precipitation events, tropical cyclone trajectoires, size and intensities, and/or El Nino occurrence and strength characteristics, still present major difficulties to forecast with conventional methods using numerical models. This proposed challenge is to explore the possibility of using recent machine learning techniques to deliver data-driven solutions to such complex problems.

The participants can choose to work on one of the following areas:

### Challenge 1: Predicting the rainfall in Basel, Switzerland during the period 2020-2021
Rainfall intensity can be categorized as follows:

- No rain (0 mm/hr)
- Drizzle (0.01 - 2.5mm/hr)
- Light rain (2.5 - 7.5mm/hr)
- Moderate rain (7.5 - 35.5mm/hr)
- Rather heavy rain (35.5 - 64.5mm/hr)
- Heavy rain (>64.5mm/hr)

Your task is to predict the hourly rainfall in Basel, Switzerland during the period 03/2020-03/2021 based on the 6 categories above from various weather attributes such as temperature, wind velocity and pressure.

__Data:__ The dataset for the period 03/2010-03/2020 (downloaded from [meteoblue.com](https://www.meteoblue.com/en/weather/archive/export/basel_switzerland_2661604?daterange=2019-02-01%20to%202021-03-01&domain=NEMSAUTO&params%5B%5D=temp2m&params%5B%5D=precip&params%5B%5D=relhum2m&min=2020-02-01&max=2021-03-01&utc_offset=1&timeResolution=hourly&temperatureunit=CELSIUS&velocityunit=KILOMETER_PER_HOUR&energyunit=watts&lengthunit=metric&degree_day_type=10%3B30&gddBase=10&gddLimit=30)), is already available and can be found in the `datasets` folder. The dataset for the period 03/2020-03/2021, which is to be used for testing will be uploaded in the `datasets` folder on __30/03/2021 at 2pm__ (second day of hackathon) so keep an eye out!

__Additional comments:__ This project is suitable for beginners, as the dataset is relatively small and easy to handle (a short guide is available in the `tutorials` folder). However, that is not to say that it is not appropriate for more experienced participants as there are many interesting things one can do with this, such as figuring out how to deal with imbalanced classes or predicting the timeseries of all variables at once using more advanced tools such as seq2seq and LSTMs!

### Challenge 2: Predicting the intensity and trajectories of tropical cyclones (TCs)

In this challenge, you are provided with datasets about cyclone tracks (times and positions every 6 hours, with some properties on the storm development and extent) in various regions of origin. From this, we can ask the following questions:

- Can we predict the highest intensity a TC will reach, and how many days in advance can we forecast the max winds?
- Can we predict where the cyclone will be when it reaches highest intensity?
- Can we predict the mean number of TCs in different basins for next year, and how many will reach Categories 4 (113-136 knots) and 5 (>136 knots)?
- Rapid intensification (RI) of TCs is defined as an increase in intensity of at least 30 knots in a 24-h period. Can we find the best statistical RI prediction schemes as function of along-track pre-storm SST, SSS, SLA, oceanic regions?
- With warmer/colder SST, fresher/salitier SSS and subsurface temperature/salinity in El-Nino/La Nina conditions, can we find its impact on TC intensity, location and size?

Your task is to tackle problems such as these (or any problems you can come up with!) using machine learning/data-driven methods.

__Data:__ Two sources of data are available for this challenge: One of them, from the US Naval Research Laboratory (NRL) is available on this github repository in the `datasets` folder and the other, from NOAA is available [here](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/). See more details below about these datasets.

__Additional comments:__ This challenge is relatively open-ended and has room for you to get adventurous.

### Challenge 3: Predicting the occurence of El Nino events

An El Nino event is characterised by a warmer-than-average sea surface temperature in the equatorial pacific that persists for an extended period of time. This has a significant impact on Earth's ecosystem and it is crucial that we are able to predict this. While there is no agreed definition of an El Nino event, a popular one that we will use for this challenge is:

__El Nino ([Definition](https://www.cgd.ucar.edu/staff/trenbert/trenberth.papers/defnBAMS.pdf)):__ An event where the 5 months running average of the sea surface temperature anomaly in the Nino 3.4 region 5°S ~ 5°N, 120° ~ 170°W (equivalently, -5° ~ 5°N, 190° ~ 240°E) exceeds 0.4°C for periods of 6 months or longer.

![el nino](images/nino.png)

Your task is to predict the occurence of an El Nino event 1 ~ 6 months ahead of time using various attributes such as sea surface salinity, precipitation and wind speed.

__Data:__ Dataset for this challenge is available in netCDF format on [ftp://ftp.ifremer.fr/ifremer/cersat/projects/stuod/hackathon/elnino/](ftp://ftp.ifremer.fr/ifremer/cersat/projects/stuod/hackathon/elnino/). More details below on how to access this.

__Additional comments:__ This challenge is appropriate for more experienced participants as it requires knowledge of dealing with spatio-temporal data and large datasets.

### Datasets

Some datasets are already available on this github repository, which includes: 
- Synthetic data from the Lorenz '63 and '96 models (used in `tutorials`)
- Toy version of the meteoblue dataset (2.9MB, used in `tutorials`)
- Full version of the meteoblue dataset for Challenge 1 (14.5MB)
- Tropical cyclone dataset for Challenge 2 (9.1 MB)

You can also download the meteoblue dataset from [google drive](https://drive.google.com/drive/folders/1qFDy1qPg63MNmrFjiBHMlS4Mz14yzJ-C) or from terminal with the command:

```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QtXg1q7xfA1Tn_hBpAkI6nVGCO5_9Gfv' -O filename.csv
```

We have two sources of data available for the tropical cyclones challenge. The first, smaller data obtained from the US NRL can be found in the `datasets` folder, which you can also get on terminal using the command:

```
wget -r ftp://ftp.ifremer.fr/ifremer/cersat/projects/stuod/hackathon/storm/tracks/
```

These data are stored in `.dat` format and can be read in `python` using modules such as `pandas`:

```python
import pandas as pd
import os
directory = "../datasets/tropical_cyclones/data/"
names = ['BASIN', 'CY', 'YYYYMMDDHH', 'TECHNUM/MIN', 'TECH', 'TAU', 'LatN/S', 'LonE/W', 'VMAX', 'MSLP', 'TY', 'RAD', 'WINDCODE', 'RAD1', 'RAD2', 'RAD3', 'RAD4', 'POUTER', 'ROUTER', 'RMW', 'GUSTS', 'EYE', 'SUBREGION', 'MAXSEAS', 'INITIALS', 'DIR', 'SPEED', 'STORMNAME', 'DEPTH', 'SEAS', 'SEASCODE', 'SEAS1', 'SEAS2', 'SEAS3', 'SEAS4', 'USERDEFINED1', 'userdata1', 'USERDEFINED2', 'userdata2', 'USERDEFINED3', 'userdata3', 'USERDEFINED4', 'userdata4', 'USERDEFINED5', 'userdata5']
stormdata = {}
for filename in os.listdir(directory):
    stormid = filename[:9]
    stormdata[stormid] = pd.read_table(directory+filename, sep = ',', names = names)
```

Further information about this dataset can be found in the accompanying README file.

The second, larger version of the cyclone data from the The National Oceanic and Atmospheric Administration can be found [here](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/), which are stored in netCDF format. See the [accompanying documentation](https://www.ncdc.noaa.gov/ibtracs/index.php?name=ib-v4-access) for more details on this extended dataset.

The El Nino dataset is available on [ftp://ftp.ifremer.fr/ifremer/cersat/projects/stuod/hackathon/elnino/](ftp://ftp.ifremer.fr/ifremer/cersat/projects/stuod/hackathon/elnino/) in netCDF format, which you can download in full using the command:

```
wget -r ftp://ftp.ifremer.fr/ifremer/cersat/projects/stuod/hackathon/elnino/
```
or if you prefer to pick and download the files individually, use:

```
wget ftp://ftp.ifremer.fr/ifremer/cersat/projects/stuod/hackathon/elnino//<variable>.nc.gz -O <filename>.nc.gz
```

where `<variable>` is to be replaced by one of the following geophysical fields:
- `cci_sst_1981_2018` or `cci_sst_anomalies_1981_2018`: sea surface temperature/anomalies from ESA CCI SST project (satellite)
- `cci_sss_2010_2019` or `cci_sss_anomalies_2010_2019`:  sea surface salinity/anomalies from ESA CCI SSS project (satellite)
- `isas_temperature_2002_2019` or `isas_temperature_anomalies_2002_2019`: water temperature in the first 300 m from Ifremer ISAS15 dataset (in situ measurements)
- `isas_salinity_2002_2019` or `isas_salinity_anomalies_2002_2019`: water salinity in the first 300 m from Ifremer ISAS15 dataset (in situ measurements)
- `ifremer_wind_1992_2019` or `ifremer_wind_anomalies_1992_2019`: wind_speed, eastward and northward wind components from CMEMS/Ifremer Multi-sensor blended dataset (satellite)
- `gpcp_precipitation_2010_2019` or `gpcp_precipitation_anomalies_2010_2019`: precipitation, from NASA GPCP dataset (satellite)
- `aviso_sla_2002_2019` or `aviso_sla_anomalies_2002_2019`: sea level anomaly from AVISO multi-altimeter dataset (satellite) 

After downloading, you can unzip the file with the command

```
gunzip -d <filename>.nc.gz
```

### Tutorials

We have also provided an interactive three-part machine learning tutorial in the `tutorials` folder that will get all the particpants up to speed with the hackathon.
This includes a detailed guide on:

- The basics of statistical classification using the Lorenz '63 model.
- Introduction to statistical regression using the two-level Lorenz '96 model.
- Handling real data with `pandas` using a simplified version of the meteoblue data.

You can access this tutorial by forking this repo and heading to the `tutorials` folder, or by clicking the icon below:

[![Binder](https://binder.pangeo.io/badge_logo.svg)](https://mybinder.org/v2/gh/sotakao/STUOD_Hackathon_2021/HEAD)

We have also included a short guide on how to use `python`'s `xarray` module for reading and handling netCDF files in `tutorials/getting-started-with-xarray`, which may be useful for Challenges 2 and 3.


### Update

Congratulations to all the teams who participated in the hackathon and completing the challenges! We were very impressed with all of your results.

The winner for this event was Team 6 (Lois Baker, Stuart Patching, Tom Gregory), who worked on the El Nino challenge. The team used linear and logisitic regression with empirical orthogonal functions (EOFs) of SST anomaly as features to predict the occurence of an El Nino event 6 months in advance. We were sold by their creativity to use EOFs as features in order to avoid overfitting and the result they achieved were astonishing, given the limited amount of data they were provided. Congratulations again to Team 6!

The runner-ups were Team 8 (Danila Kurganov, Oliver Phillips), who achieved incredible R2 scores on various aspects of the tropical cyclones challenge, and Team 9 (Phillip Breul, Benjamin Aslan, Jakob Wessel, Fiona Spuler) who used XGBoost to achieve astonishing accuracy for predicting the rainfall in Basel. Congratualations again to Teams 8 and 9!

Some other highlights of the day were:
- Team 1 used Resevoir Computing to predict the Nino 3.4 index and found that the choice of hyperparameters were instrumental in achieving good performance.
- Team 2 experimented with Bayesian techniques to predict the rainfall in Basel, in addition to coming up with an excellent motivation for working on this challenge (involving tourism and Roger Federer).
- Team 3 found that linear regression already works extremely well to predict the highest intensity of TCs using the other variables.
- Despite their first time working on Machine Learning, Team 4 achieved high accuracy to predict the rainfall in Basel, in addition to finding a discrepancy in the performance on the validation and test datasets
- Teams 5 and 7 used SMOTE to deal with the class inbalance in the rainfall dataset and also considered different metrics for model comparison since the persistence baseline already achieved high accuracy due to the fact that there was no rain in most days.
- Team 10 attempted to use Graph Neural Networks to automatially learn which geographical locations affect the occurence of El Nino without relying on expert knowledge (based on [this paper](https://arxiv.org/pdf/2012.01598.pdf)), however ran out of time.

### Acknowledgements
We would like to thank Ronan Fablet (IMT Atlantique) and Jean-Francois Piolle (IFRMER) for agreeing to help us out with the event. We would also like to thank Jean-Francois and Nicolas Reul (IFREMER) for providing us with the Tropical Cyclone and El Nino datasets.
