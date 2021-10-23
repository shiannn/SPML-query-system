import os
import csv
#from django.conf import settings

def handle_uploaded_images(user_dir, file_name):
    print('handle_uploaded_images')
    print(user_dir, file_name)
    ### unzip
    ### predict
    ### output csv and save
    header = ['name', 'area', 'country_code2', 'country_code3']
    data = ['Afghanistan', 652090, 'AF', 'AFG']
    with open(os.path.join(user_dir, 'result.csv'), 'w') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        writer.writerow(data)