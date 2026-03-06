from scipy.signal import savgol_filter



def smooth(x, window_length=15, polyorder=3):
    return savgol_filter(x, window_length=window_length, polyorder=polyorder)

def deriv(i, window_length=15, polyorder=3, deriv=1):
    return savgol_filter(i, window_length=window_length, polyorder=polyorder, deriv=deriv)