import subprocess
import time
import re

def get_gpu_power_usage():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], encoding='utf-8')
        power_usage = re.findall(r'\d+.\d+', output)
        return power_usage[0] if power_usage else "No data"
    except subprocess.CalledProcessError as e:
        print("not able to fetch GPU power usage:", e)
        return "Failed"

def log_gpu_power_usage(interval_seconds=10, log_file_path='gpu_power_log.txt'):
    with open(log_file_path, 'a') as file:
        while True:
            power_usage = get_gpu_power_usage()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_entry = f"{timestamp}: {power_usage} W\n"
            print(log_entry, end='')  
            file.write(log_entry)
            file.flush()
            time.sleep(interval_seconds)

if __name__ == "__main__":
    log_gpu_power_usage()
