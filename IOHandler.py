from PIL import Image
import numpy as np


class IOHandler:
    def __init__(self, map: np.array, latitudeStart: float, longitudeStart: float, latitudinalEnd: float, longitudinalEnd: float):
        self.map = map
        self.width, self.height = map.shape

        # the latitude and longitude the top left corner of the map fill out the matching map
        self.latitude = latitudeStart
        self.longitude = longitudeStart

        self.latitudinal_resolution = (
            latitudinalEnd - latitudeStart) / self.width
        self.longitudinal_resolution = (
            longitudinalEnd - longitudeStart) / self.height

        self.latitudes = []
        for i in range(self.width):
            self.latitudes.append(
                self.latitude + self.latitudinal_resolution * i)

        self.longitudes = []
        for i in range(self.height):
            self.longitudes.append(
                self.longitude + self.longitudinal_resolution * i)

    def map2latlon(self, x, y):
        return self.latitudes[x], self.longitudes[y]

    def latlon2map(self, lat, lon):
        # find the closest latitude and longitude
        x = np.argmin(np.abs(self.latitudes - lat))
        y = np.argmin(np.abs(self.longitudes - lon))
        return x, y

    def searchValid(self, lontitude, latitude):
        x, y = self.latlon2map(latitude, lontitude)

        return self.map[x, y]


def recursivePadding(img: np.array, padding_value: int, x: int, y: int):
    # check if the padding value is 0
    if padding_value == 0:
        return

    # check if the coordinate is in img array
    if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1]:
        return

    # check if the coordinate is already padded
    if img[x, y] != 0 and img[x, y] >= padding_value:
        return

    img[x, y] = padding_value

    recursivePadding(img, padding_value-0.1, x - 1, y)
    recursivePadding(img, padding_value-0.1, x + 1, y)
    recursivePadding(img, padding_value-0.1, x, y - 1)
    recursivePadding(img, padding_value-0.1, x, y + 1)


def image2numpy(dir2img: str, entrance: list=[]):
    img = Image.open(dir2img)

    # convert gray scale
    img = img.convert('L')

    # convert to numpy array
    numpy_data = np.array(img)

    # for each entrace contain a pair of x and y
    for x, y in entrance:
        recursivePadding(numpy_data, 255, x, y)

    return numpy_data
