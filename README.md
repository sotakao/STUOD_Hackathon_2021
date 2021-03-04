# ERC Stochastic Transport in Upper Ocean Dynamics Hackathon March 29-31 2021
![stuod logo](https://www.imperial.ac.uk/ImageCropToolT4/imageTool/uploaded-images/erc-stuod-logos--tojpeg_1572609986634_x2.jpg)
## Challenge 3: Machine Learning with Weather and Climate Data

[Link to the event website](https://www.imperial.ac.uk/events/129398/stuods-hackathon/)

### Task A: Predicting the rainfall in Basel, Switzerland during the period 2019-2021
Rainfall intensity can be categorized as follows:

- No rain (0 mm/hr)
- Drizzle (0.01 - 2.5mm/hr)
- Light rain (2.5 - 7.5mm/hr)
- Moderate rain (7.5 - 35.5mm/hr)
- Rather heavy rain (35.5 - 64.5mm/hr)
- Heavy rain (>64.5mm/hr)

Your task is to predict the hourly rainfall (based on the 6 categories) in Basel, Switzerland during the period 03/2019-03/2021 from various weather attributes such as temperature, wind velocity and pressure (pre-downloaded dataset from [meteoblue](https://www.meteoblue.com/en/weather/archive/export/basel_switzerland_2661604?daterange=2019-02-01%20to%202021-03-01&domain=NEMSAUTO&params%5B%5D=temp2m&params%5B%5D=precip&params%5B%5D=relhum2m&min=2020-02-01&max=2021-03-01&utc_offset=1&timeResolution=hourly&temperatureunit=CELSIUS&velocityunit=KILOMETER_PER_HOUR&energyunit=watts&lengthunit=metric&degree_day_type=10%3B30&gddBase=10&gddLimit=30) can be found in the `datasets` folder. More details below on how to use this data).

### Task B (open-ended): Using data from the Met office to tackle climate change

Some ideas include predicting renewable energy sources such as solar irradiance and wind power.

### Datasets

- Lorenz models (used in `tutorials`)
- Meteoblue dataset
- Met office data

### Tutorials

In the `tutorials` folder, you will find a three-part climate-themed tutorial on machine learning that will get you up to speed with the tools required to start hacking.
This includes discussions on:

- Using the Lorenz 63 model to learn about statistical classification
- Using the Lorenz 96 model to learn about regression
- Handling real data with `pandas` using a simplified version of the meteoblue data

The tutorial is aimed at students and researchers who are new to machine learning or are just getting started.

Todo: Convert it to binder or colab format to make it interactive

### Submissions

### Acknowledgements
