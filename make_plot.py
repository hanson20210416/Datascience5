import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    times = []
    workers = []
    
    # Regular expression to match elapsed time line format
    pattern = re.compile(r'(\d+),(\d+\.\d+),(\d+\.\d+),([\d:]+\.\d+)')
    
    for line in lines:
        match = pattern.search(line)
        if match:
            worker_count = int(match.group(1))
            elapsed_time_str = match.group(4)
            
            # Check if the elapsed time has a colon
            if ":" in elapsed_time_str:
                # Convert elapsed time (MM:SS.SS format) to seconds
                try:
                    minutes, seconds = map(float, elapsed_time_str.split(":"))
                    total_seconds = minutes * 60 + seconds
                except ValueError:
                    # Handle unexpected format (e.g., if the split fails)
                    continue
            else:
                # If there is no colon, treat it as seconds only (SS.SS format)
                try:
                    total_seconds = float(elapsed_time_str)
                except ValueError:
                    # Skip if conversion fails
                    continue
            
            workers.append(worker_count)
            times.append(total_seconds)
    
    # Return data as a DataFrame
    df = pd.DataFrame({'Workers': workers, 'Time': times})
    print(df)
    return df

def plot_data1(df):
    # Plot 1: Workers vs Time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Workers'], df['Time'], marker='o', label='Time vs Workers')
    plt.title('Performance: Workers vs Time at week2')
    plt.xlabel('Number of Workers')
    plt.ylabel('Time (seconds)')
    plt.grid()
    plt.legend()
    plt.savefig('workers_vs_time at week2.png')
    plt.show()
    
def plot_data2(df):
    # Plot 1: Workers vs Time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Workers'], df['Time'], marker='o', label='Time vs Workers')
    plt.title('Performance: Workers vs Time at exam')
    plt.xlabel('Number of Workers')
    plt.ylabel('Time (seconds)')
    plt.grid()
    plt.legend()
    plt.savefig('workers_vs_time at exam.png')
    plt.show()

if __name__ == '__main__':
    # Extract data from the results.csv file
    filepath1 = '/homes/zhe/Desktop/programming/p5/week2/results.csv'
    data_frame1 = extract_data(filepath1)
    
    filepath2 = '/homes/zhe/Desktop/programming/p5/exam/results.csv'
    data_frame2 = extract_data(filepath2)
    
    # Plot the extracted data
    plot_data1(data_frame1)
    plot_data2(data_frame2)
