from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1) LOAD RED DATA
# -------------------------
file_path_red = "ADP.2020-06-26T11:14:16.300.fits"

with fits.open(file_path_red) as hdul:
    wave_red = None
    flux_red = None
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and {'WAVE', 'FLUX'} <= set(hdu.columns.names):
            wave_red = np.array(hdu.data['WAVE'][0], dtype=float)
            flux_red = np.array(hdu.data['FLUX'][0], dtype=float)
            print(f"Found RED spectrum in HDU {hdu.name} (ext {hdul.index_of(hdu)})")
            break
    if wave_red is None:
        raise RuntimeError("Could not find WAVE/FLUX columns in red file")

# -------------------------
# 2) LOAD BLUE DATA
# -------------------------
file_path_blue = "ADP.2020-06-26T11:14:16.096.fits"

with fits.open(file_path_blue) as hdul:
    wave_blue = None
    flux_blue = None
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and {'WAVE', 'FLUX'} <= set(hdu.columns.names):
            wave_blue = np.array(hdu.data['WAVE'][0], dtype=float)
            flux_blue = np.array(hdu.data['FLUX'][0], dtype=float)
            print(f"Found BLUE spectrum in HDU {hdu.name} (ext {hdul.index_of(hdu)})")
            break
    if wave_blue is None:
        raise RuntimeError("Could not find WAVE/FLUX columns in blue file")

    mask_b = (wave_blue >= 3300) & (wave_blue <= 4500)
    wave_blue = wave_blue[mask_b]
    flux_blue = flux_blue[mask_b]

# -------------------------
# 3) HELIOCENTRIC CORRECTION – SEPARATE FOR RED AND BLUE
# -------------------------
def get_obs_params(filepath):
    with fits.open(filepath) as hdul:
        lat = lon = elev = None
        date_obs = ra_deg = dec_deg = cor_frame = None
        for hdu in hdul:
            hdr = hdu.header
            if 'ESO TEL GEOLAT' in hdr and lat is None:
                lat = hdr['ESO TEL GEOLAT']
            if 'ESO TEL GEOLON' in hdr and lon is None:
                lon = hdr['ESO TEL GEOLON']
            if 'ESO TEL GEOELEV' in hdr and elev is None:
                elev = hdr['ESO TEL GEOELEV']
            if 'DATE-OBS' in hdr and date_obs is None:
                date_obs = hdr['DATE-OBS']
            if 'RA' in hdr and ra_deg is None:
                ra_deg = hdr['RA']
            if 'DEC' in hdr and dec_deg is None:
                dec_deg = hdr['DEC']
            if 'RADECSYS' in hdr and cor_frame is None:
                cor_frame = hdr['RADECSYS']
    return lat, lon, elev, date_obs, ra_deg, dec_deg, cor_frame

# --- Red ---
lat_r, lon_r, elev_r, date_r, ra_r, dec_r, frame_r = get_obs_params(file_path_red)
target_r   = SkyCoord(ra=ra_r*u.deg, dec=dec_r*u.deg, frame=frame_r.lower())
time_r     = Time(date_r, scale='utc')
location_r = EarthLocation(lat=lat_r*u.deg, lon=lon_r*u.deg, height=elev_r*u.m)
v_heli_red = target_r.radial_velocity_correction(
    'heliocentric', obstime=time_r, location=location_r
).to(u.km/u.s).value

# --- Blue ---
lat_b, lon_b, elev_b, date_b, ra_b, dec_b, frame_b = get_obs_params(file_path_blue)
target_b   = SkyCoord(ra=ra_b*u.deg, dec=dec_b*u.deg, frame=frame_b.lower())
time_b     = Time(date_b, scale='utc')
location_b = EarthLocation(lat=lat_b*u.deg, lon=lon_b*u.deg, height=elev_b*u.m)
v_heli_blue = target_b.radial_velocity_correction(
    'heliocentric', obstime=time_b, location=location_b
).to(u.km/u.s).value

c = 299792.458  # km/s
wave_red_helio  = wave_red  * (1.0 + v_heli_red  / c)
wave_blue_helio = wave_blue * (1.0 + v_heli_blue / c)

print(f"\nHeliocentric correction:")
print(f"  RED  : Δv_heli = {v_heli_red :+8.3f} km/s")
print(f"  BLUE: Δv_heli = {v_heli_blue:+8.3f} km/s")
print(f"  Difference   : {v_heli_red - v_heli_blue:+.3f} km/s")

# =====================================================
# 5) CONVERT WAVELENGTH → VELOCITY (MULTIPLE l0)
# =====================================================
l0_red = {
    "NaI 5889": 5889.951,
    "NaI 5896": 5895.924,
}

l0_blue = {
    "NaI 3303" : 3302.369,
    "NaI 3303.9" : 3302.978,
    "CaII 3934": 3933.663,
    "CN 3874": 3873.994,
    "CN 3875.6": 3874.602,
    "CN 3876": 3875.759,
    "CaII 3969": 3968.469,
    "CaI 4227": 4226.728,
    "CH 4301": 4300.313,
    "CH+ 4233": 4232.548,
    "FeI 3720": 3719.935,
}

def wavelength_to_velocity(wave_helio, l0, c=299792.458):
    return c * (wave_helio - l0) / l0

# =====================================================
# PLOT (ALL l0 IN ONE GRAPH)
# =====================================================
plt.figure(figsize=(13,5))

for label, l0 in l0_red.items():
    vel = wavelength_to_velocity(wave_red_helio, l0)
    plt.plot(vel, flux_red, lw=1.2, label=f"RED: {label}")

for label, l0 in l0_blue.items():
    vel = wavelength_to_velocity(wave_blue_helio, l0)
    plt.plot(vel, flux_blue, lw=1.0, alpha=0.7, label=f"BLUE: {label}")

plt.xlabel("Velocity (km/s)")
plt.ylabel("Flux")
plt.legend(ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.xlim(-120, 120)
plt.show()
