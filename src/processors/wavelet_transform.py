import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import signal, fft
from scipy.signal import savgol_filter
from pathlib import Path

class processor(object):

    def __init__(self):
        # Basic configuration
        self.dt = 0.04
        self.rate = 1/self.dt  # Sampling rate in Hz
    
    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def final_process(self, df, i, results):
        data = df.iloc[:, i]
        column_name = df.columns[i]
        wavelet = 'sym20'
        #print(pywt.wavelist(kind='discrete'))
        level = 3
        coeffs = pywt.wavedec(data, wavelet, level=level)
        spectrum=np.fft.fft(data)
        freqs = fft.fftfreq(len(data), d=self.dt)
        Fpositive=np.where(freqs>=0)
        # Apply soft thresholding to the detail coefficients
        threshold = 0.5 # Adjust this threshold value
        sigma = (1/0.6745) * self.madev(coeffs[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(data)))
        #coeffs[1:] = [pywt.threshold(detail, uthresh, mode='soft') for detail in coeffs[1:]]
        coeffs[1:] = [pywt.threshold(detail, threshold, mode='soft') for detail in coeffs[1:]]
        # Reconstruct the denoised signal
        denoised_signal = pywt.waverec(coeffs, wavelet)
        results[column_name] = denoised_signal[:-1]

        '''
        window_size = 25
        poly_order = 1
        smoothed_y = savgol_filter(denoised_signal, window_size, poly_order)
        
        # Define filter specifications
        cutoff_freq = 0.9 # Cutoff frequency in Hz
        filter_order = 10 # Filter order (number of taps)
        # Design the FIR filter coefficients using the Hamming window
        #fir_coeffs = signal.firwin(filter_order, cutoff_freq, fs=25, window='hamming', pass_zero=False)
        #denoised_signal = signal.convolve(data, fir_coeffs, mode='same')

        lowcut = 0.3
        highcut = 1
        nyquist = 0.5 * 25
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = signal.butter(1, low, btype='low', output = 'sos', analog=False)
        #denoised_signal = signal.lfilter(sos, denoised_signal)
        #zi = signal.sosfilt_zi(sos) * data[0]
        #denoised_signal, _ = signal.sosfilt(sos,data, zi=zi)
        '''
        spectrum_output=np.fft.fft(denoised_signal)

        fig = plt.figure(figsize=(10, 6))
        gs=fig.add_gridspec(2,2)

        ax1=fig.add_subplot(gs[:,0])
        ax2=fig.add_subplot(gs[0,1])
        ax3=fig.add_subplot(gs[1,1])

        ax1.plot(df['time'].values,data.values,label='Input')
        ax1.plot(df['time'].values,denoised_signal,color='r',label='output')
        ax1.legend(fontsize=14)
        ax1.set_xlabel('Time(sec)',fontsize=14)
        ax1.set_ylabel(column_name,fontsize=14)
        ax1.grid(True)

        ax2.plot(freqs[Fpositive],np.abs(spectrum[Fpositive])/len(denoised_signal),label='Input')
        ax2.grid()
        ax2.legend(fontsize=14)
        ax2.set_xlabel('Hz',fontsize=14)
        ax2.set_ylabel('Spectrum',fontsize=14)
        ax2.grid(True)

        ax3.plot(freqs[Fpositive],np.abs(spectrum_output[Fpositive])/len(denoised_signal),color='r',label='Output')
        ax3.grid()
        ax1.legend(fontsize=14)
        ax3.set_xlabel('Hz',fontsize=14)
        ax3.set_ylabel('Spectrum',fontsize=14)
        ax2.grid(True)

        plt.show()


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Filtering data using wavelet transform.")
    parser.add_argument("--csv", help="Input CSV file path")
    parser.add_argument("--output", help="Output CSV file name")
    parser.add_argument("-cn", "--column_num", type=int, help="Output CSV file name")
    parser.add_argument("--save", action='store_true', help="Save final info in yaml file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.drop(columns=df.columns[0])
    proc = processor()
    results = pd.DataFrame()
    proc.final_process(df, args.column_num, results)

    if args.save:
        file_path = args.output
        if Path(file_path).is_file():
            existing_data = pd.read_csv(args.output)
            updated_data = pd.concat([existing_data, results], axis=1)
            updated_data.to_csv(args.output, index=False)
        else:
            updated_data = pd.concat([df['time'], results], axis=1)
            updated_data.to_csv(args.output, index=False)
    print("Data filtered successfully.")
