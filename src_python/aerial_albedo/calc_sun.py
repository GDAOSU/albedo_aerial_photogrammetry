import math
from math import degrees, radians

############################################################################
#
# SunClass is used for storing intermediate sun calculations.
#
############################################################################


class SunClass:
    latitude = 0.0
    longitude = 0.0
    elevation = 0.0
    azimuth = 0.0

    month = 0
    day = 0
    year = 0
    day_of_year = 0
    time = 0.0

    UTC_zone = 0


############################################################################
#
# Calculate the actual position of the sun based on input parameters.
#
# The sun positioning algorithms below are based on the National Oceanic
# and Atmospheric Administration's (NOAA) Solar Position Calculator
# which rely on calculations of Jean Meeus' book "Astronomical Algorithms."
# Use of NOAA data and products are in the public domain and may be used
# freely by the public as outlined in their policies at
#               www.nws.noaa.gov/disclaimer.php
#
# The calculations of this script can be verified with those of NOAA's
# using the Azimuth and Solar Elevation displayed in the SunPos_Panel.
# NOAA's web site is:
#               http://www.esrl.noaa.gov/gmd/grad/solcalc
############################################################################


def get_sun_position(local_time, latitude, longitude, north_offset, utc_zone, month, day, year):
    use_refraction = True
    longitude *= -1  # for internal calculations
    utc_time = local_time - utc_zone  # Set Greenwich Meridian Time

    if latitude > 89.93:  # Latitude 90 and -90 gives
        latitude = radians(89.93)  # erroneous results so nudge it
    elif latitude < -89.93:
        latitude = radians(-89.93)
    else:
        latitude = radians(latitude)

    t = julian_time_from_y2k(utc_time, year, month, day)

    e = radians(obliquity_correction(t))
    L = apparent_longitude_of_sun(t)
    solar_dec = sun_declination(e, L)
    eqtime = calc_equation_of_time(t)

    time_correction = (eqtime - 4 * longitude) + 60 * utc_zone
    true_solar_time = ((utc_time - utc_zone) * 60.0 + time_correction) % 1440

    hour_angle = true_solar_time / 4.0 - 180.0
    if hour_angle < -180.0:
        hour_angle += 360.0

    csz = math.sin(latitude) * math.sin(solar_dec) + math.cos(latitude) * math.cos(solar_dec) * math.cos(
        radians(hour_angle)
    )
    if csz > 1.0:
        csz = 1.0
    elif csz < -1.0:
        csz = -1.0

    zenith = math.acos(csz)

    az_denom = math.cos(latitude) * math.sin(zenith)

    if abs(az_denom) > 0.001:
        az_rad = ((math.sin(latitude) * math.cos(zenith)) - math.sin(solar_dec)) / az_denom
        if abs(az_rad) > 1.0:
            az_rad = -1.0 if (az_rad < 0.0) else 1.0
        azimuth = 180.0 - degrees(math.acos(az_rad))
        if hour_angle > 0.0:
            azimuth = -azimuth
    else:
        azimuth = 180.0 if (latitude > 0.0) else 0.0

    if azimuth < 0.0:
        azimuth = azimuth + 360.0

    exoatm_elevation = 90.0 - degrees(zenith)

    if use_refraction:
        if exoatm_elevation > 85.0:
            refraction_correction = 0.0
        else:
            te = math.tan(radians(exoatm_elevation))
            if exoatm_elevation > 5.0:
                refraction_correction = 58.1 / te - 0.07 / (te**3) + 0.000086 / (te**5)
            elif exoatm_elevation > -0.575:
                s1 = -12.79 + exoatm_elevation * 0.711
                s2 = 103.4 + exoatm_elevation * (s1)
                s3 = -518.2 + exoatm_elevation * (s2)
                refraction_correction = 1735.0 + exoatm_elevation * (s3)
            else:
                refraction_correction = -20.774 / te

        refraction_correction = refraction_correction / 3600
        solar_elevation = 90.0 - (degrees(zenith) - refraction_correction)

    else:
        solar_elevation = 90.0 - degrees(zenith)

    solar_azimuth = azimuth
    solar_azimuth += north_offset

    sun = SunClass()
    sun.az_north = solar_azimuth

    sun.theta = math.pi / 2 - radians(solar_elevation)
    sun.phi = radians(solar_azimuth) * -1
    sun.azimuth = azimuth
    sun.elevation = solar_elevation

    return sun


##########################################################################
## Get the elapsed julian time since 1/1/2000 12:00 gmt
## Y2k epoch (1/1/2000 12:00 gmt) is Julian day 2451545.0
##########################################################################


def julian_time_from_y2k(utc_time, year, month, day):
    century = 36525.0  # Days in Julian Century
    epoch = 2451545.0  # Julian Day for 1/1/2000 12:00 gmt
    jd = get_julian_day(year, month, day)
    return ((jd + (utc_time / 24)) - epoch) / century


def get_julian_day(year, month, day):
    if month <= 2:
        year -= 1
        month += 12
    A = math.floor(year / 100)
    B = 2 - A + math.floor(A / 4.0)
    jd = math.floor((365.25 * (year + 4716.0))) + math.floor(30.6001 * (month + 1)) + day + B - 1524.5
    return jd


def calc_time_julian_cent(jd):
    t = (jd - 2451545.0) / 36525.0
    return t


def sun_declination(e, L):
    return math.asin(math.sin(e) * math.sin(L))


def calc_equation_of_time(t):
    epsilon = obliquity_correction(t)
    ml = radians(mean_longitude_sun(t))
    e = eccentricity_earth_orbit(t)
    m = radians(mean_anomaly_sun(t))
    y = math.tan(radians(epsilon) / 2.0)
    y = y * y
    sin2ml = math.sin(2.0 * ml)
    cos2ml = math.cos(2.0 * ml)
    sin4ml = math.sin(4.0 * ml)
    sinm = math.sin(m)
    sin2m = math.sin(2.0 * m)
    etime = y * sin2ml - 2.0 * e * sinm + 4.0 * e * y * sinm * cos2ml - 0.5 * y**2 * sin4ml - 1.25 * e**2 * sin2m
    return degrees(etime) * 4


def obliquity_correction(t):
    ec = obliquity_of_ecliptic(t)
    omega = 125.04 - 1934.136 * t
    return ec + 0.00256 * math.cos(radians(omega))


def obliquity_of_ecliptic(t):
    return 23.0 + 26.0 / 60 + (21.4480 - 46.8150) / 3600 * t - (0.00059 / 3600) * t**2 + (0.001813 / 3600) * t**3


def true_longitude_of_sun(t):
    return mean_longitude_sun(t) + equation_of_sun_center(t)


def calc_sun_apparent_long(t):
    o = true_longitude_of_sun(t)
    omega = 125.04 - 1934.136 * t
    lamb = o - 0.00569 - 0.00478 * math.sin(radians(omega))
    return lamb


def apparent_longitude_of_sun(t):
    return radians(true_longitude_of_sun(t) - 0.00569 - 0.00478 * math.sin(radians(125.04 - 1934.136 * t)))


def mean_longitude_sun(t):
    return (280.46646 + 36000.76983 * t + 0.0003032 * t**2) % 360


def equation_of_sun_center(t):
    m = radians(mean_anomaly_sun(t))
    c = (
        (1.914602 - 0.004817 * t - 0.000014 * t**2) * math.sin(m)
        + (0.019993 - 0.000101 * t) * math.sin(m * 2)
        + 0.000289 * math.sin(m * 3)
    )
    return c


def mean_anomaly_sun(t):
    return 357.52911 + t * (35999.05029 - 0.0001537 * t)


def eccentricity_earth_orbit(t):
    return 0.016708634 - 0.000042037 * t - 0.0000001267 * t**2
