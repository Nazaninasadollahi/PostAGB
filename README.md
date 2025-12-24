Step 1: Read the FITS files and check the reference frame → If it is GEOCENT/TOPOCENT, it is ok.

Step 2: Use SKYCALC for simulation → Fill the first section on the SkyCalc website, including name, RA, and Dec (you can use the SIMBAD website to find other names). Click on “Transfer information to SkyCalc model.” Leave the other sections unchanged. In the Wavelength Grid section, set your wavelength range. Choose air for the simulation. Select Linear binning with a value of 0.002. You can see the simulation resolution here (λ/Δλ). Click Submit to get the output. Download the FITS file and use the Plotting SKYCALC results code. The "TRANS" column shows the telluric lines.

Step 4: Compare the telescope data with the telluric lines from SKYCALC to perform telluric calibration in the red part of the spectrum. (Calibration with telluric lines)

Step 5: Shift the telescope data for calibration and save the new FITS files.

Step 6: Check telluric lines for the blue part of the spectrum too. (If it was possible)

Step 7: Move to the heliocentric reference frame. Blue part and red part should have the same ISM velocity (it needs calibration (read the shift in wavelength and perform it in the data in the GEOCENT frame) otherwise). Then read ISM velocity.

Step 8: Move to the ISM frame and find molecules and atoms.
