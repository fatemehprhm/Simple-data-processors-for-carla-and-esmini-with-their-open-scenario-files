import argparse
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import pywt
import  numpy as np
from scipy import signal, fft


class epsilon_finder(object):
    def range_finder(self, data, threshold):
        ranges = []
        means = []
        max_epsilons = []
        max_epsilons_index = []
        current_range_start = 0
        prev_point = data[0]
        current_mean = data[0]
        max_epsilon = 0
        epsilon_index = 1
        for i in range(1, len(data)):
            epsilon = abs(data[i] - current_mean)
            if epsilon > threshold:
                # If data point deviates from the mean by more than the threshold consider it as a new range
                current_range_end = i - 1
                ranges.append((current_range_start, current_range_end))
                means.append(current_mean)
                max_epsilons.append(max_epsilon)
                max_epsilons_index.append(epsilon_index)

                # Start a new range
                current_range_start = i
                current_mean = data[i]
                max_epsilon = 0  # Reset max_epsilon
            
            # Update the running mean
            current_mean += (data[i] - current_mean) / (i - current_range_start + 1)
            
            # Update max_epsilon for the current range
            max_epsilon = max(max_epsilon, epsilon)
            if max_epsilon == epsilon:
                epsilon_index = i
            prev_point = data[i]
        
        # Add the last range
        ranges.append((current_range_start, len(data) - 1))
        means.append(current_mean)
        max_epsilons.append(max_epsilon)
        max_epsilons_index.append(epsilon_index)
        return ranges, means, max_epsilons, max_epsilons_index
    
    def results(self, df, threshold, column_num, ignorance, save):
        time = df['time']
        data_orig = df.iloc[:, column_num]
        column_name = df.columns[column_num]
        start_point = ignorance
        end_point = len(data_orig) - ignorance
        data_changed = pd.Series(data_orig[start_point:end_point]).reset_index(drop=True)
        ranges, means, max_epsilons, max_epsilons_index = self.range_finder(data_changed, threshold)
        final_epsilons = []
        info_list = []

        plt.figure(figsize=(10, 6))
        plt.plot(time.values, data_orig.values, label='Original Data', color='b')
        plt.xlabel('Time(sec)',fontsize=14)
        plt.ylabel(column_name,fontsize=14)  

        for i, (start, end) in enumerate(ranges):
            range_data = data_changed[start:end + 1]
            if len(range_data) < 25:
                continue
            startp = start + start_point
            endp = end + start_point
            range_time = time[startp:endp + 1]
            range_mean = float(means[i])
            range_max_epsilon = float(max_epsilons[i])
            final_epsilons.append(range_max_epsilon)
            range_max_epsilon_index = max_epsilons_index[i]
            mean_values = []
            mean_values.extend([range_mean] * len(range_data))
            plt.plot(range_time.values, mean_values, color='r')
            
            print(f"Range {i + 1}:")
            print(f"Start Index: {start}, End Index: {end}")
            print(f"Mean Value: {range_mean}")
            print(f"Max Epsilon: {range_max_epsilon}")
            print(f"Max Epsilon index: {range_max_epsilon_index}")
            if save:
                info = {} 
                info['Range'] = i + 1
                info['Start Index'] = start
                info['End Index'] = end
                info['Mean Value'] = range_mean
                info['Max Epsilon'] = range_max_epsilon
                info['Max Epsilon index'] = range_max_epsilon_index
                info_list.append(info)

        plt.legend()
        plt.grid(True)
        plt.show()

        mean_epsilon = float(sum(final_epsilons) / len(final_epsilons))
        print('Mean epsilon is: ', mean_epsilon)
        variable_name_info = {"Variable name": column_name,
                'Mean epsilon': mean_epsilon
            }
        info_list.insert(0, variable_name_info)
        return info_list
    
    def calculate_accel(self, df):
        input_y = df.iloc[:, 1]
        column_name = df.columns[1]
        t = 0.04
        data = (input_y - 2 * input_y.shift(1) + input_y.shift(2)) / (t**2)
        column_name = 'ego_accel'
        wavelet = 'sym20'
        level = 3
        coeffs = pywt.wavedec(data, wavelet, level=level)
        spectrum=np.fft.fft(data)
        freqs = fft.fftfreq(len(data), d=t)
        Fpositive=np.where(freqs>=0)
        threshold = 0.5
        coeffs[1:] = [pywt.threshold(detail, threshold, mode='soft') for detail in coeffs[1:]]
        denoised_signal = pywt.waverec(coeffs, wavelet)

        plt.figure(figsize=(10, 6))
        plt.plot(df['time'].values,data.values,label='Input')
        plt.plot(df['time'].values,denoised_signal[:-1],color='r',label='output')
        plt.legend(fontsize=14)
        plt.xlabel('Time(sec)',fontsize=14)
        plt.ylabel(column_name,fontsize=14)
        plt.grid(True)

        plt.show()

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Epsilon Filter")
    parser.add_argument("--csv", help="Input CSV file path")
    parser.add_argument("--threshold", type=float, help="Threshold of the data")
    parser.add_argument("--ignorance", type=int, help="points to ignore")
    parser.add_argument("--save", type=int, help="Save final info in yaml file")
    parser.add_argument("-cn", "--column_num", type=int, help="Column number")
    parser.add_argument("-fa", "--file_address", help="Output file address in yaml format")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    info_list = epsilon_finder().results(df, args.threshold, args.column_num, args.ignorance, args.save)
    epsilon_finder().calculate_accel(df)


    if args.save:
        with open(args.file_address, 'a') as yaml_file: # append data to the previous file
            yaml.dump_all(info_list, yaml_file, default_flow_style=False)
            yaml_file.write('\n')
            yaml_file.write(50*'-')
            yaml_file.write(2*'\n')