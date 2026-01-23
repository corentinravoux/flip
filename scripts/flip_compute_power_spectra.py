from flip.power_spectra import generator

power_spectrum_engine = "class_engine"

power_spectrum_settings = {
    "h": 0.6766,
    "omega_b": 0.02242,
    "omega_cdm": 0.11933,
    "sigma8": 0.8102,
    "n_s": 0.9665,
}


minimal_wavenumber = 0.0005
maximal_wavenumber = 1.000
number_points = 1000
logspace = True
redshift = 0.0

# Can be changed by "growth_rate" (fsigma_8^fid normalization) or "no_normalization"
# Here, it normalize by sigma_8^fid
normalization_power_spectrum = "growth_amplitude"

# If you want to add non linearity, what model to use
power_spectrum_non_linear_model = "halofit"

# Model used to compute matter and velocity divergence power spectra
power_spectrum_model = "linearbel"

save_path = "./"  # If not None, will save all calculated power spectra in this folder

(
    wavenumber,
    power_spectrum_mm,
    power_spectrum_mt,
    power_spectrum_tt,
    fiducial,
) = generator.compute_power_spectra(
    power_spectrum_engine,
    power_spectrum_settings,
    redshift,
    minimal_wavenumber,
    maximal_wavenumber,
    number_points,
    logspace=logspace,
    normalization_power_spectrum=normalization_power_spectrum,
    power_spectrum_non_linear_model=power_spectrum_non_linear_model,
    power_spectrum_model=power_spectrum_model,
    save_path=save_path,
)
