This dataset contains the tracks of tropical cyclones over the last few years. It was obtained from the US Naval Research Laboratory (NRL). The tropical cyclone data comes from a variety of sources. Best track data for the Eastern Pacific, Central Pacific and the Atlantic are from the National Hurricane Center. The remaining data was extracted from the Joint Typhoon Warning Center (JTWC) Tropical Cyclone Data.

These data are the 6-hourly storm positions based on a post-storm, subjectively smoothed path. In some cases, the data obtained was in 12 hour increments. For these storms, 6-hourly positions were interpolated using the method of Akima (1970). The best tracks files are named b<stormid>.dat. For example, bwp132000.dat contains best track positions for the 13th storm of the 2000 western north pacific season.
The tropical cyclone description data is stored in flat ascii files. Each storm is given a unique eight character code called the storm ID to identify it in the database. The storm ID is of the format RECYYYYY, where

RE = Region (basin) of Origin

AL - North Atlantic

WP - Western North Pacific

CP - Central North Pacific

EP - Eastern North Pacific

IO - North Indian Ocean

SH - Southern Hemisphere

LS - Southern Atlantic

CY = Annual Cyclone Number(01-99)

YYYY = Year

For example, WP011992 is the first (01) western North Pacific (WP) storm in 1992 (1992).

The track files contain for each track position and time additional properties of the storm, such as their direction and speed, the maximum wind, the radius for various wind speed levels, etc... The description of each column in the file is available here : https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt

More information on this dataset at the US Naval Research Laboratory: https://www.nrlmry.navy.mil/atcf_web/docs/database/new/database.html

