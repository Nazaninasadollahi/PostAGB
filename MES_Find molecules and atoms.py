from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. LOAD AND CORRECT SPECTRUM
# =========================
file_path = "ADP.2020-06-26T11:14:16.096.fits"

wave = None
flux = None

with fits.open(file_path) as hdul:
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and {'WAVE', 'FLUX'} <= set(hdu.columns.names):
            wave_red = np.array(hdu.data['WAVE'][0], dtype=float)
            flux_red = np.array(hdu.data['FLUX'][0], dtype=float)
            print(f"Found RED spectrum in HDU '{hdu.name}' (extension {hdul.index_of(hdu)})")
            break

if wave_red is None or flux_red is None:
    raise RuntimeError("Could not find WAVE and FLUX columns in the red FITS file.")

#normalization
z = np.polyfit(wave_red, flux_red, 1)  # 3rd-degree is much more stable
p = np.poly1d(z)
cnt = p(wave_red)
flux_normalized = flux_red / cnt
flux_red = flux_normalized - 0.12

# Extract observation parameters
def extract_observation_params(filepath):
    with fits.open(filepath) as hdul:
        lat = lon = elev = None
        date_obs = ra_deg = dec_deg = frame = None
        for hdu in hdul:
            hdr = hdu.header
            if lat is None and 'ESO TEL GEOLAT' in hdr: lat = hdr['ESO TEL GEOLAT']
            if lon is None and 'ESO TEL GEOLON' in hdr: lon = hdr['ESO TEL GEOLON']
            if elev is None and 'ESO TEL GEOELEV' in hdr: elev = hdr['ESO TEL GEOELEV']
            if date_obs is None and 'DATE-OBS' in hdr: date_obs = hdr['DATE-OBS']
            if ra_deg is None and 'RA' in hdr: ra_deg = hdr['RA']
            if dec_deg is None and 'DEC' in hdr: dec_deg = hdr['DEC']
            if frame is None and 'RADECSYS' in hdr: frame = hdr['RADECSYS']
        return lat, lon, elev, date_obs, ra_deg, dec_deg, frame

lat_r, lon_r, elev_r, date_obs_r, ra_r, dec_r, frame_r = extract_observation_params(file_path)

target_coord = SkyCoord(ra=ra_r * u.deg, dec=dec_r * u.deg, frame=frame_r.lower())
obs_time = Time(date_obs_r, scale='utc')
obs_location = EarthLocation(lat=lat_r * u.deg, lon=lon_r * u.deg, height=elev_r * u.m)

v_heliocentric = target_coord.radial_velocity_correction(
    kind='heliocentric', obstime=obs_time, location=obs_location
).to(u.km / u.s).value

c = 299792.458  # km/s
wave_heliocentric = wave_red * (1.0 + v_heliocentric / c)

# ISM correction for BD+33 2642
v_ism = -20.37  # km/s
wave_ism = wave_heliocentric * (1.0 - v_ism / c)

# =========================
# 2. ZOOM REGION: ±15 Å around 4259.01 Å
# =========================
lambda_0 = 4259.01
zoom_width = 5.0  # Å

mask = (wave_ism >= lambda_0 - zoom_width) & (wave_ism <= lambda_0 + zoom_width)
wave_zoom = wave_ism[mask]
flux_zoom = flux_red[mask]

# =========================
# 3. GAUSSIAN ABSORPTION PROFILES (2008 and 2009 scaled)
# =========================
FWHM = 1.05    # Å (instrumental + intrinsic broadening)
EW_base = 0.0215  # Å (your reference EW)

# Scaled EWs according to your factors
EW_2008 = EW_base * 0.06 / 1.11
EW_2009 = EW_base * 0.06 / 1.27

sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
A_2008 = EW_2008 / (np.sqrt(2 * np.pi) * sigma)
A_2009 = EW_2009 / (np.sqrt(2 * np.pi) * sigma)

# Wavelength grid for smooth Gaussian
wave_gauss = np.linspace(lambda_0 - zoom_width, lambda_0 + zoom_width, 1000)
gauss_2008 = 1.0 - A_2008 * np.exp(-(wave_gauss - lambda_0)**2 / (2 * sigma**2))
gauss_2009 = 1.0 - A_2009 * np.exp(-(wave_gauss - lambda_0)**2 / (2 * sigma**2))

# =========================
# 4. PLOT: DATA + MODELS
# =========================
plt.figure(figsize=(10, 6))

# Plot observed spectrum (normalized)
plt.plot(wave_zoom, flux_zoom, color='black', linewidth=1.2, label='Observed (ISM frame)')

# Plot Gaussian models
plt.plot(wave_gauss, gauss_2008, color='blue', linewidth=2, label='Model 2008 (scaled EW)')
plt.plot(wave_gauss, gauss_2009, color='red', linewidth=2, label='Model 2009 (scaled EW)')

# Aesthetics
plt.axvline(lambda_0, color='gray', linestyle='--', alpha=0.7, label=f'λ₀ = {lambda_0} Å')
plt.xlabel('Wavelength (Å)', fontsize=12)
plt.ylabel('Flux', fontsize=12)
plt.title('Zoom around 4259.01 Å (±15 Å) with Gaussian Absorption Models', fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()