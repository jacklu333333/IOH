import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


class IOHandler:
    '''
    IOHandler is a class to handle the input and output of the map
    ----------------------------------------------------
    input:
        map: the map in numpy array
        pointStart: starting store in (x,y), bottom left corner
        pointEnd: ending store in (x,y), top right corner
    ----------------------------------------------------
    output:
        the object of the IOHandler
    '''

    def __init__(
        self,
            map: np.array,  # the map in numpy array
            # starting store in (x,y), bottom left corner
            pointStart: np.array,
            pointEnd: np.array  # ending store in (x,y), top right corner
    ):
        self.map = map
        self.width, self.height = map.shape
        # check the ending is larger than the starting
        assert pointStart[0] < pointEnd[0]
        assert pointStart[1] < pointEnd[1]

        # the latitude and longitude the top left corner of the map fill out the matching map
        self.xS = pointStart[0]
        self.yS = pointStart[1]

        self.xE = pointEnd[0]
        self.yE = pointEnd[1]

        self.x_resolution = (
            pointEnd[0] - pointStart[0]) / self.width
        self.y_resolution = (
            pointEnd[1] - pointStart[1]) / self.height

        self.x_coordinates = []
        for i in range(self.width):
            self.x_coordinates.append(
                self.xS + self.x_resolution * i)
        self.x_coordinates = np.array(self.x_coordinates)

        self.y_coordinates = []
        for i in range(self.height):
            self.y_coordinates.append(
                self.yS + self.y_resolution * i)
        self.y_coordinates = np.array(self.y_coordinates)

    def map2coordinate(self, x, y):
        '''
        convert the map index to the coordinate
        ----------------------------------------------------
        input:
            x: the x index of the map
            y: the y index of the map
        ----------------------------------------------------
        output:
            the coordinate of the map
        '''
        return self.y_coordinates[y], self.x_coordinates[x]

    # def latlon2map(self, lat, lon):
    #     # find the closest latitude and longitude
    #     y = np.argmin(np.abs(self.y_coordinates - lat))
    #     x = np.argmin(np.abs(self.x_coordinates - lon))
    #     return x, y

    def coordinate2map(self, x, y):
        '''
        convert the coordinate to the map index
        ----------------------------------------------------
        input:
            x: the x coordinate of the map
            y: the y coordinate of the map
        ----------------------------------------------------
        output:
            the index of the map
        '''
        # find the closest grid
        x = np.argmin(np.abs(self.x_coordinates - x))
        y = np.argmin(np.abs(self.y_coordinates - y))
        return x, y

    def searchValid(self, x: float, y: float):
        '''
        check if the coordinate is valid
        ----------------------------------------------------
        input:
            x: the x coordinate of the map
            y: the y coordinate of the map
        ----------------------------------------------------
        output:
            0-1 if the coordinate is valid
            False if the coordinate is not valid
        '''
        # check if the coordinate is in the map
        if y < self.yS or y > self.yE or x < self.xS or x > self.xE:
            print("The coordinate is not in the map")
            return False

        x, y = self.coordinate2map(x, y)

        return self.map[x, y]

    def searchValidNumpy(self, xs: np.array, ys: np.array):
        '''
        check if the coordinate is valid with numpy array
        ----------------------------------------------------
        input:
            xs: the x coordinate of the map, in numpy array
            ys: the y coordinate of the map, in numpy array
        ----------------------------------------------------
        output:
            0-1 if the coordinate is valid
            False if the coordinate is not valid
        '''
        assert ys.shape == xs.shape

        # the input is a numpy array with a series of points's latitude and longitude to check if it is valid

        result_in = np.ones(ys.shape)
        # check if the coordinate is in the map if not set the value to False
        result_in[(ys < self.yS) | (ys > self.yE) | (
            xs < self.xS) | (xs > self.xE)] = False

        result_v = np.zeros(ys.shape)
        x = np.argmin(
            np.abs(self.x_coordinates - xs.reshape(-1, 1)), axis=1)
        y = np.argmin(np.abs(self.y_coordinates -
                      ys.reshape(-1, 1)), axis=1)
        # using numpy index to find the index of the coordinate
        result_v = self.map[x, y]
        result_v[result_in == False] = False

        return result_v


def recursivePadding(img: np.array, center: np.array, nextpoint: np.array, preDistance: float):
    '''
    recursive padding the img with the diffusion effect
    ----------------------------------------------------
    input:
        img: the numpy array of the image
        center: the center of the padding
        nextpoint: the next point to padding
        preDistance: the distance between the center and the nextpoint
    ----------------------------------------------------
    output:
        None
    '''
    assert center.shape == nextpoint.shape == (2,)
    x, y = nextpoint
    distance = np.linalg.norm(nextpoint-center)
    padding_value = 1 - distance/100

    # print(f'x: {x}, y: {y}, padding_value: {padding_value}')
    # print(
    #     f'center{center} current" {nextpoint}preDistance: {preDistance}, distance: {distance}')

    # check if the padding value is 0
    if padding_value <= 0 or distance < preDistance:
        return

    # check if the coordinate is in img array
    if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1]:
        return

    # check if the coordinate is already padded
    if img[x, y] != 0 and padding_value <= img[x, y]:
        return

    # padding
    img[x, y] = padding_value

    # recursive padding
    recursivePadding(img, center, nextpoint + np.array([1, 0]), distance)
    recursivePadding(img, center, nextpoint + np.array([0, 1]), distance)
    recursivePadding(img, center, nextpoint + np.array([0, -1]), distance)
    recursivePadding(img, center, nextpoint + np.array([-1, 0]), distance)


def image2numpy(dir2img: str, entrance: list = []):
    '''
    convert the image to numpy array
    and padding the entrance with the diffusion effect
    ----------------------------------------------------
    input:
        dir2img: the directory to the image
        entrance: a list of entrance coordinate in the of numpy grid index (x,y)
    ----------------------------------------------------            
    output:
        numpy_data: the numpy array of the image
    '''
    img = Image.open(dir2img)

    # convert gray scale
    img = img.convert('L')
    # invert the image
    img = Image.eval(img, lambda x: 255-x)

    # corp the image to where the numpy array is not 0
    img = img.crop(img.getbbox())

    # invert the image back
    img = Image.eval(img, lambda x: 255-x)

    # convert to numpy array
    numpy_data = np.array(img)
    numpy_data = np.invert(numpy_data.astype(np.bool_)).astype(np.float32)

    # for each entrace contain a pair of x and y
    for x, y in entrance:
        recursivePadding(
            numpy_data,
            np.array([x, y]),
            np.array([x, y]),
            -1
        )

    return numpy_data


def poly2numpy(polygon: list,heights:list=[]) -> np.array:
    '''
    convert the polygon to numpy array map
    ----------------------------------------------------
    input:
        polygon: a list of polygon coordinate in the of numpy grid index (y,x)
    ----------------------------------------------------
    output:
        map_array: the numpy array map
    '''
    if len(heights)==0:
        heights = [1 for _ in range(len(polygon))]
        
    assert len(polygon)==len(heights)
    # top =
    # bottom =
    # left =
    # right =

    # Unpack the polygon points
    points = np.array([p for poly in polygon for p in poly]).reshape(-1, 2)

    # Find the top, bottom, left, right
    top = max(points[:, 0])
    bottom = min(points[:, 0])
    right = max(points[:, 1])
    left = min(points[:, 1])

    print(f'{"bottom:":<8s} {bottom}')
    print(f'{"left:":<8s} {left}')
    print(f'{"top:":<8s} {top}')
    print(f'{"right:":<8s} {right}')

    width = int(round(right - left, 0))
    height = int(round(top-bottom, 0))

    # Your numpy array map
    map_array = np.zeros((width, height))

    for poly,height in zip(polygon,heights):
        poly = np.array(poly).reshape(-1, 2)[:, ::-1]
        # print(poly)
        # c = input("continue?")
        poly[:, 0] -= left
        poly[:, 1] -= bottom

        # Create a Path object from the polygon vertices
        poly_path = mplPath.Path(poly)

        # Iterate over each point in the map array
        for i in range(map_array.shape[0]):
            for j in range(map_array.shape[1]):
                # If the point is inside the polygon, set its value to 1.0
                if poly_path.contains_point((i, j)):
                    map_array[i, j] = height

    return map_array


def latlonTransformer(positions: np.array, source: str = 'epsg:4326', target: str = 'epsg:5179') -> np.array:
    '''
    convert the latitude and longitude to the coordinate
    ----------------------------------------------------
    input:
        positions: a numpy array of latitude and longitude
        source: the epsg of the input
        target: the epsg of the output
    ----------------------------------------------------
    output:
        the coordinate of the map in the target epsg
    '''
    transformer = pyproj.Transformer.from_crs(
        source, target, always_xy=True)
    result = transformer.transform(positions[:, 0], positions[:, 1])
    result = np.array(result).swapaxes(0, 1).reshape(-1, 2)

    return result


def plot3dMap(mapInHeight):
    '''
    plot the 3d map
    ----------------------------------------------------
    input:
        mapInHeight: the map in numpy array
    ----------------------------------------------------
    output:
        fig: the figure of the plot
        ax: the axis of the plot
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(mapInHeight.shape[1])
    y = np.arange(mapInHeight.shape[0])
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, y, mapInHeight, cmap='viridis')
    ax.set_aspect('equal')

    fig.show()
    return fig, ax


def mapHeightTo3D(mapHeight:np.array):
    '''
    convert the map in height to 3d map
    ----------------------------------------------------
    input:
        mapHeight: the map in height
    ----------------------------------------------------
    output:
        spatial: the spatial of the map
    '''
    # round the height
    mapHeight = np.round(mapHeight+0.5, 0).astype(np.int32)
    # find the max height
    maxHeight, minHeight = np.max(mapHeight), np.min(mapHeight)
    # create the spatial
    spatial = np.zeros((mapHeight.shape[0], mapHeight.shape[1], maxHeight-minHeight+1))
    # fill the spatial
    for i in range(mapHeight.shape[0]):
        for j in range(mapHeight.shape[1]):
            spatial[i, j, :mapHeight[i, j]-minHeight] = 1

    return spatial