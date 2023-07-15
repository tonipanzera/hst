import numpy as np

def linear_func(x, a, b):
    return a*x + b

def quad_func(x, a, b, c):
    return a*x**2 + b*x + c

def closest_value(input_list, input_value):
 
    arr = np.asarray(input_list)
 
    idx = (np.abs(arr - input_value)).argmin()
 
    return idx

def fit_continuum(file, func_type, wave_range_blue, wave_range_red):
    
    '''
    Creates a fit to the continuum based on a user-defined range for both the red and blue sides.
    file = .txt file containing 3 columns (wavelength, flux, error)
    func_type = type of function to fit, either linear or quadratic
    wave_range_blue = wavelength range of continuum on blue side, in an array like [start_wav, end_wav]
    wave_range_red = same thing for the red side
    '''
    data = ascii.read(file)
    wavelength = data['Wavelength']
    flux = data['Flux']
    error = data['Error']
    
    #Grab the index starts and ends of red and blue regions
    start_idx_blue = closest_value(wavelength, wave_range_blue[0])
    end_idx_blue = closest_value(wavelength ,wave_range_blue[1])
    start_idx_red = closest_value(wavelength, wave_range_red[0])
    end_idx_red = closest_value(wavelength ,wave_range_red[1])
    
    #Isolate the wavelengths in question
    b_wave = wavelength[start_idx_blue:end_idx_blue]
    r_wave = wavelength[start_idx_red:end_idx_red]
    
    #Concatenate these to have an array of all the red + blue wavelengths
    wave = np.concatenate((b_wave, r_wave))
    
    #Isolate the flux values in question
    b_flux = flux[start_idx_blue:end_idx_blue]
    r_flux = flux[start_idx_red:end_idx_red]
    
    #Concatenate these to have an array of all the red + blue continuum fluxes
    flux = np.concatenate((b_flux, r_flux))
    
    #Isolate the errors in question
    b_err = error[start_idx_blue:end_idx_blue]
    r_err = error[start_idx_red:end_idx_red]
    
    #Concatenate these to have an array of all the red + blue flux errors
    error = np.concatenate((b_err, r_err))
    
    #Calculate uncertainty in red + blue regions: sigma**2 = sum_i sigma_i**2 / sqrt(N)
    sigma_c = np.sqrt(np.sum(error**2)/np.sqrt(len(error)))
    
    #This is kind of redundant now but it works so why change it
    if func_type =='linear':
        popt, pcov = curve_fit(linear_func, wave, flux)
        a = popt[0]
        b = popt[1]
    
    else:
        popt, pcov = curve_fit(quad_func, wave, flux)
        cont_flux = popt[2]

    return a, b, sigma_c

def integrate_flux(data, left, right, a, b, sigma_c, newflux):
    '''
    Function to integrate the flux values over a user-defined range.
    The continuum will be subtracted first.
    '''
    wavelength = data['Wavelength']
    flux = newflux
    error = data['Error']
    
    #Isolate indices of start and end of line region
    start = closest_value(wavelength, left)
    end = closest_value(wavelength, right)
    
    #Cut each array according to these indices
    cut = flux[start:end]
    cut_waves = wavelength[start-1:end+1]
    cut_error = error[start:end]
    
    #Create array for continuum fit
    xdata = np.linspace(left, right, len(cut))
    
    #Create continuum fit
    ydata = a*xdata + b
    
    cut = cut - ydata
    
    flux_array = []
    sigma_array = []
    
    for i in range(0, len(cut)):
        
        lambdaLeft = cut_waves[i]-cut_waves[i-1]
        lambdaRight = cut_waves[i+1]-cut_waves[i]
        deltaLambda = (lambdaLeft + lambdaRight)/2
        
        flux_i = cut[i]
        sigma_i = cut_error[i]
        
        flux = flux_i*deltaLambda
        sigma = sigma_i*deltaLambda
        
        flux_array.append(flux)
        sigma_array.append(sigma)
    
    #Calculate uncertainty in flux continuum: sigma Fc = sqrt(sigma**2)*deltaLambda
    #sigma_Fc = np.sqrt(sigma_c**2)*(start-end)
    sigma_Fc = np.sqrt(sigma_c**2)
    
    #Calculate uncertainty in flux
    sigma_m = np.sqrt(np.sum(error[start:end]**2))
    
    #Calculate uncertainty in flux line: sigma Fm = sqrt(sigma**2)*deltaLambda
    #sigma_Fm = np.sqrt(sigma_m**2)*(start-end)
    sigma_Fm = np.sqrt(sigma_m**2)*0.1
    
    #Add up continuum-subtracted flux values
    integrated = np.sum(flux_array)
    
    return integrated, sigma_Fc, sigma_Fm

def remove_H2(file, start, end):
    data = ascii.read(file)
    wavelength = data['Wavelength']
    flux = data['Flux']
    error = data['Error']
    
    start = closest_value(wavelength, start)
    end = closest_value(wavelength, end)
    cut = flux[start:end]
    cut_err = error[start:end]
    
    x = np.array([flux[start], flux[end]])
    
    y = ((flux[end]-flux[start])/(wavelength[end]-wavelength[start]))*x
    
    x_new = np.linspace(flux[start], flux[end], len(cut))
    
    waves = np.linspace(wavelength[start], wavelength[end], len(x_new))
    
    new_flux = np.concatenate((flux[:start], x_new, flux[end:]))
    
    return x_new, waves, new_flux