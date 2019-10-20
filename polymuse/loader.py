from polymuse import constant

import pandas

import os, requests, zipfile
from google_drive_downloader import GoogleDriveDownloader


def load():
    load_csv(True)
    csv = pandas.read_csv('models.csv')
    id_ = csv[csv['model'] == 'default'].to_numpy()[0][1]
    print(type(id_))

    GoogleDriveDownloader.download_file_from_google_drive(id_, dest_path= './h5_models.zip')

    with zipfile.ZipFile('h5_models.zip', 'r') as zip_ref:
        zip_ref.extractall('h5_models')


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def load_csv(force = False):
    r = requests.get(constant.models_csv)
    if force or not os.path.isfile('models.csv'):
        with open('models.csv', 'wb') as f:
            f.write(r.content)