
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================
# 1. LOAD SPECTRUM
# =====================================================
file_path = "ADP.2020-06-26T11:14:16.096.fits"

with fits.open(file_path) as hdul:
    wave_red = None
    flux_red = None

    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU):
            if {'WAVE', 'FLUX'} <= set(hdu.columns.names):
                wave_red = np.array(hdu.data['WAVE'][0], dtype=float)
                flux_red = np.array(hdu.data['FLUX'][0], dtype=float)
                print(f"Found spectrum in HDU {hdul.index_of(hdu)}")
                break

if wave_red is None or flux_red is None:
    raise RuntimeError("WAVE / FLUX columns not found")

# =====================================================
# 2. CONTINUUM NORMALIZATION
# =====================================================
z = np.polyfit(wave_red, flux_red, 1)
p = np.poly1d(z)
continuum = p(wave_red)

flux_norm = flux_red / continuum
flux_norm -= 0.12  # your offset

# =====================================================
# 3. OBSERVATION PARAMETERS
# =====================================================
def extract_observation_params(filepath):
    with fits.open(filepath) as hdul:
        lat = lon = elev = None
        date_obs = ra = dec = frame = None

        for hdu in hdul:
            hdr = hdu.header
            if lat is None and 'ESO TEL GEOLAT' in hdr: lat = hdr['ESO TEL GEOLAT']
            if lon is None and 'ESO TEL GEOLON' in hdr: lon = hdr['ESO TEL GEOLON']
            if elev is None and 'ESO TEL GEOELEV' in hdr: elev = hdr['ESO TEL GEOELEV']
            if date_obs is None and 'DATE-OBS' in hdr: date_obs = hdr['DATE-OBS']
            if ra is None and 'RA' in hdr: ra = hdr['RA']
            if dec is None and 'DEC' in hdr: dec = hdr['DEC']
            if frame is None and 'RADECSYS' in hdr: frame = hdr['RADECSYS']

        return lat, lon, elev, date_obs, ra, dec, frame

lat, lon, elev, date_obs, ra, dec, frame = extract_observation_params(file_path)

target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame=frame.lower())
obs_time = Time(date_obs, scale='utc')
obs_loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)

# =====================================================
# 4. HELIOCENTRIC + ISM CORRECTION
# =====================================================
v_helio = target.radial_velocity_correction(
    kind='heliocentric',
    obstime=obs_time,
    location=obs_loc
).to(u.km/u.s).value

c = 299792.458  # km/s

wave_helio = wave_red * (1.0 + v_helio / c)

# ISM correction (BD+33 2642)
v_ism = -20.37  # km/s
wave_ism = wave_helio * (1.0 - v_ism / c)

# =====================================================
# 5. READ LINE LIST (CSV)
# =====================================================
# CSV columns must be: lambda0, FWHM, EW_base
line_table = pd.read_csv("line_list.csv")

# =====================================================
# 6. GAUSSIAN ABSORPTION FUNCTION
# =====================================================
def gaussian_absorption(wave, lambda0, FWHM, EW):
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    A = EW / (np.sqrt(2 * np.pi) * sigma)
    return 1.0 - A * np.exp(-(wave - lambda0)**2 / (2 * sigma**2))

# =====================================================
# 7. SLIDING 30 Å WINDOWS
# =====================================================
window_width = 30.0
half_window = window_width / 2

wave_min = wave_ism.min()
wave_max = wave_ism.max()

window_centers = np.arange(
    wave_min + half_window,
    wave_max - half_window,
    window_width
)

# =====================================================
# 8. LOOP OVER WINDOWS AND LINES
# =====================================================
for center in window_centers:

    mask = (wave_ism >= center - half_window) & (wave_ism <= center + half_window)
    wave_zoom = wave_ism[mask]
    flux_zoom = flux_norm[mask]

    if len(wave_zoom) < 20:
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(wave_zoom, flux_zoom, color='black', lw=1.2, label='Observed')

    # ---------------------------------
    # loop over all lines in CSV
    # ---------------------------------
    for _, row in line_table.iterrows():

        lambda0 = row['lambda0']
        FWHM = row['FWHM']
        EW_base = row['EW_base']

        if not (center - half_window <= lambda0 <= center + half_window):
            continue

        # your scaling
        EW_2008 = EW_base * 0.06 / 1.11
        EW_2009 = EW_base * 0.06 / 1.27

        wave_model = np.linspace(center - half_window,
                                 center + half_window,
                                 1000)

        model_2008 = gaussian_absorption(wave_model, lambda0, FWHM, EW_2008)
        model_2009 = gaussian_absorption(wave_model, lambda0, FWHM, EW_2009)

        label_2008 = f'2008 λ={lambda0:.2f}' if EW_base != 0 else '_nolegend_'
        label_2009 = f'2009 λ={lambda0:.2f}' if EW_base != 0 else '_nolegend_'

        plt.plot(wave_model, model_2008, '--', lw=2, label=label_2008)
        plt.plot(wave_model, model_2009, ':', lw=2, label=label_2009)

        plt.axvline(lambda0, color='blue', ls='--', alpha=0.6)

    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.title(f"{center-half_window:.1f} – {center+half_window:.1f} Å")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
