import csv
import os

missing = []
with open('data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:        
        for view in range(3):
            source_path = os.path.basename(line[view].strip())
            file_path = os.path.join('data', 'IMG', source_path)
            if not os.path.exists(file_path):
                missing.append(file_path)
                
print(len(missing))