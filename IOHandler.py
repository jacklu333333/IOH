from PIL import Image
import numpy as np


class IOHandler:
    def __init__(self, map: np.array, latitudeStart: float, longitudeStart: float, latitudeEnd: float, longitudeEnd: float):
        self.map = map
        self.width, self.height = map.shape
        # check the ending is larger than the starting
        assert latitudeStart < latitudeEnd
        assert longitudeStart < longitudeEnd

        # the latitude and longitude the top left corner of the map fill out the matching map
        self.longitudeS = longitudeStart
        self.latitudeS = latitudeStart

        self.longitudeE = longitudeEnd
        self.latitudeE = latitudeEnd

        self.longitudinal_resolution = (
            longitudeEnd - longitudeStart) / self.width
        self.latitudinal_resolution = (
            latitudeEnd - latitudeStart) / self.height

        self.longitudes = []
        for i in range(self.width):
            self.longitudes.append(
                self.longitudeS + self.longitudinal_resolution * i)
        self.longitudes = np.array(self.longitudes)
        self.latitudes = []

        for i in range(self.height):
            self.latitudes.append(
                self.latitudeS + self.latitudinal_resolution * i)
        self.latitudes = np.array(self.latitudes)

    def map2latlon(self, x, y):
        return self.latitudes[y], self.longitudes[x]

    def latlon2map(self, lat, lon):
        # find the closest latitude and longitude
        y = np.argmin(np.abs(self.latitudes - lat))
        x = np.argmin(np.abs(self.longitudes - lon))
        return x, y

    def searchValid(self, latitude: float, lontitude: float):
        # check if the coordinate is in the map
        if latitude < self.latitudeS or latitude > self.latitudeE or lontitude < self.longitudeS or lontitude > self.longitudeE:
            print("The coordinate is not in the map")
            return False

        x, y = self.latlon2map(latitude, lontitude)

        return self.map[x, y]


def recursivePadding(img: np.array, padding_value: int, x: int, y: int):
    # print(x, y, padding_value)
    # check if the padding value is 0
    if padding_value <= 0:
        return

    # check if the coordinate is in img array
    if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1]:
        return

    # check if the coordinate is already padded
    if img[x, y] != 0 and img[x, y] > padding_value:
        return

    img[x, y] = padding_value

    recursivePadding(img, padding_value-0.01, x - 1, y)
    recursivePadding(img, padding_value-0.01, x + 1, y)
    recursivePadding(img, padding_value-0.01, x, y - 1)
    recursivePadding(img, padding_value-0.01, x, y + 1)


def image2numpy(dir2img: str, entrance: list = []):
    img = Image.open(dir2img)

    # convert gray scale
    img = img.convert('L')

    # corp the image to where the numpy array is not null
    img = img.crop(img.getbbox())

    # convert to numpy array
    numpy_data = np.array(img)
    numpy_data = np.invert(numpy_data.astype(np.bool_)).astype(np.float32)

    # for each entrace contain a pair of x and y
    for x, y in entrance:
        recursivePadding(numpy_data, 0.9, x, y)

    return numpy_data
