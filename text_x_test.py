import json
import gzip
import pandas as pd
from docarray import DocumentArray, Document
from clip_client import Client
from annlite import AnnLite
from time import sleep
import asyncio


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
ann_da = AnnLite(n_dim=768,metric='cosine', data_path='./data_', columns={'othernames': 'str', 'surname': 'str', 'institution': 'str', 'department': 'str', 'matric': 'str', 'dob': 'str', 'corps': 'str', 'jamb': 'str','state': 'str', 'sex' : 'str'})
print("*** AnnLite instance created ***")


# Connect to the CLIP API
c = Client(
    'grpcs://api.clip.jina.ai:2096', credential={'Authorization': '991e3e1eb7c84a1242644521a948e6be'}
)

# didvide documents into 8 batches and asynchronously encode each batch and index into ann
da1 = documents[0:352686]
da2 = documents[352686:705372]
da3 = documents[705372:1058058]
da4 = documents[1058058:1410744]
da5 = documents[1410744:1763430]
da6 = documents[1763430:2116116]
da7 = documents[2116116:2468802]
da8 = documents[2468802:2821489]


# for each batch, break into chunks of 1000 and encode each chunk and index into ann at the same time

def encode(da):
    for i in range(0, len(da), 1000):
        print(f"Encoding batch {i}")
        encoded = c.encode(da[i:i+1000])
        print("Encoding done")
        return encoded


async def index(da):
    print("Indexing")
    ann_da.index(encode(da))
    print("Indexing done")

async def main():
    await asyncio.gather(index(da1), index(da2), index(da3), index(da4), index(da5), index(da6), index(da7), index(da8))

asyncio.run(main())

