# IOHandler
> This repo use the LX database to create the numpy array indoor coverage of the map.
> We adopt the EPSG:5179 system for location. (x,y)

1. [navermap](https://wooiljeong.github.io/python/static-map/)
2. [openstreemap](https://pygis.io/docs/d_access_osm.html)
3. LX

-----
### File Usage

> 1. IOHandler.py --> the main library
> 2. analysis.ipynb --> demo of the usage of IOH
> 3. LXDownloader.ipynb --> download the map info from the LX library and convert numpy array
> 4. mapDownloader.ipynb --> download the map from openstreet map library and convert to image
> 5. naverGetMap.ipynb --> use naver API to get image of map

### Note:
1. building height
> The building is blindingly assume the ground floor is the entrance floor. The height is the building. Please "Tune" it according to your need.

-----

## ToDo
1. ~~Elevation acquiring~~
> It is impossible to achieve the goal with the database avaiable.
> Google map resolution is too low, in some cases is only 4 meter, which is taller than a floor.
> SRTM library set the whole KAIST campus as the same height.

2. ~~3D map construction~~
> The result is imcomplete with the database avaiable.

