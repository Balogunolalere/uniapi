import json
import gzip
import pandas as pd
from docarray import DocumentArray, Document
from clip_client import Client
from annlite import AnnLite
from time import sleep


# Load data from a gzipped JSON file
with gzip.open("/home/doombuggy/Downloads/combined.json.gz", "rb") as f:
    print("*** Loading data ***")
    data = json.loads(f.read().decode())

# Convert the data to a pandas DataFrame
data_ = pd.DataFrame(data)

# Drop and fill NA values
columns_to_drop = ['MARITAL STATUS', 'WHAT WAS UPDATED', 'DATE ENTERED', 'QUALIFICATION']
data_ = data_.drop(columns=columns_to_drop).fillna('')

# Process the data
data_['Full Name'] = data_['SURNAME'] + ' ' + data_['OTHERNAMES']
data_['DATE OF BIRTH'] = pd.to_datetime(data_['DATE OF BIRTH'] / 1000, unit='s').astype(str).apply(lambda x: x.split(' ')[0])
data_['MATRICULATION NUMBER'] = data_['MATRICULATION NUMBER'].astype(str)
data_['CorpsId'] = data_['CorpsId'].astype(str)
print("*** Data processed ***")

# Convert the processed data to a list of Document objects
documents = [Document(text=data_['Full Name'][i], tags={'department': data_['DEPARTMENT'][i], 'sex': data_['SEX'][i], 'surname':data_['SURNAME'][i], 'othernames':data_['OTHERNAMES'][i], 'matric':data_['MATRICULATION NUMBER'][i],'dob':data_['DATE OF BIRTH'][i],'jamb':data_['JAMB_NUMBER'][i], 'institution': data_['INSTITUTION CODE'][i], 'corps': data_['CorpsId'][i], 'state':data_['STATE OF ORIGIN'][i]}) for i in range(len(data_))]
print("*** Documents created ***")

# Create an AnnLite instance
ann_da = AnnLite(n_dim=768,metric='cosine', data_path='./data', columns={'othernames': 'str', 'surname': 'str', 'institution': 'str', 'department': 'str', 'matric': 'str', 'dob': 'str', 'corps': 'str', 'jamb': 'str','state': 'str', 'sex' : 'str'})
print("*** AnnLite instance created ***")

# Connect to the CLIP API
c = Client(
    'grpcs://api.clip.jina.ai:2096', credential={'Authorization': '991e3e1eb7c84a1242644521a948e6be'}
)


# Split the list of documents into smaller chunks
chunk_size = 50000
num_chunks = len(documents) // chunk_size + 1

for i in range(num_chunks):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    da = DocumentArray(documents[start:end])
    # Process the current chunk of DocumentArray here
    encoded_da = c.encode(da, show_progress=True)
    ann_da.index(encoded_da)
    print(f'Processed chunk {i + 1} of {num_chunks}')
    
