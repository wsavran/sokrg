import numpy as np
import matplotlib.pyplot as plt

class FlatEarthProjection:
    """ Computes flat-earth approximation on an ellipsoid """
    def __init__(self, lat0=None, lon0=None, radius=6371000.0):
        self.lat0 = lat0
        self.lon0 = lon0
        self.radius = radius
        self._d1_lon = None
        self._d1_lat = None
        self.a = 6378137.0
        self.b = 6356752.3142
        self.e = np.sqrt((self.a**2 - self.b**2)/self.a**2)

        # computes meridian distances and central latitude
        if self.lat0 is not None and self.lon0 is not None:
            self._d1_lat, self._d1_lon = self._compute_d1()

    @property
    def d1_lat(self):
        return self._d1_lat

    @property
    def d1_lon(self):
        return self._d1_lon

    @property
    def get_origin(self):
        return (self.lon0, self.lat0)

    def set_origin(self, lat, lon):
        self.lat0 = lat
        self.lon0 = lon
        self._d1_lat, self._d1_lon = self._compute_d1()

    def convert_local_to_geo(self, x, y):
        lon = (y / self._d1_lon) + self.lon0
        lat = x / self._d1_lat + self.lat0
        return lat, lon

    def convert_geo_to_local(self, lat, lon):
        pass

    def _compute_d1(self):
        """ approximates the arc-length of 1 degree on the earth using a sphere """
        if self.lat0 is None or self.lon0 is None:
            raise ValueError('lat0 and lon0 must be defined to compute d1')
        lat0 = np.deg2rad(self.lat0)
        # Approximation to the meridian distances using WGS84
        d1_lat = (np.pi * self.radius * (1 - self.e**2)) / (180 * (1 - self.e**2*np.sin(lat0)**2)**(3/2))
        d1_lon = (np.pi * self.radius * np.cos(lat0)) / (180 * np.sqrt(1 - self.e**2*np.sin(lat0)**2))
        return d1_lat, d1_lon

def geospace(lat0, lon0, length, dx, strike):
    """ returns a series of points in geographic coordinates"""
    pts_a = []
    npts = length // dx + 1
    for idx in range(npts):
        # convert to lat, lon
        new = convert_local_idx_to_geo(idx, lat0, lon0, length, dx, strike)
        pts_a.append(new)
    return np.array(pts_a)


def convert_local_idx_to_geo(idx, lat0, lon0, length, dx, strike):
    """ takes an index along strike and computes the lat, lon of that point. origin at center."""
    a = FlatEarthProjection()
    stk = np.deg2rad(strike)
    a.set_origin(lat0, lon0)
    npts = length // dx + 1
    hlf_pts = npts // 2
    # compute signed distance in grid space
    sd_grid = idx - hlf_pts
    # covert to cartesian 
    sd_cart = sd_grid * dx
    # new cartesian coords in flat-earth plane
    x1 = sd_cart*np.cos(stk)
    x2 = sd_cart*np.sin(stk)
    new = a.convert_local_to_geo(x1,x2)
    return new
    

if __name__ == "__main__":
    # in meters
    dx = 100
    stk = np.deg2rad(151)
    lat0 = 35.269
    lon0 = 133.357
    length = 27000

    pts = geospace(lat0, lon0, length, dx, stk)

    # plot points in space
    fig, ax = plt.subplots()
    lon = pts[:,1]
    lat = pts[:,0]
    ax.scatter(lon, lat)
    plt.show()
