from docarray import DocumentArray, Document
from clip_client import Client
from annlite import AnnLite
import json
import pandas as pd
import gzip
from time import sleep


from clip_client import Client

c = Client(
    'grpcs://api.clip.jina.ai:2096', credential={'Authorization': '991e3e1eb7c84a1242644521a948e6be'}
)

with gzip.open("/home/doombuggy/Downloads/combined.json.gz", "rb") as f:
    data = json.loads(f.read().decode())

data_ = pd.DataFrame(data)

del [data_['MARITAL STATUS'], data_['WHAT WAS UPDATED'], data_['DATE ENTERED'], data_['QUALIFICATION']]
data_ = data_.fillna('')
data_['Full Name'] = data_['SURNAME'] + ' ' + data_['OTHERNAMES']
data_['DATE OF BIRTH'] = pd.to_datetime(data_['DATE OF BIRTH'] / 1000, unit='s')
data_['DATE OF BIRTH'] = data_['DATE OF BIRTH'].astype(str)
data_['MATRICULATION NUMBER'] = data_['MATRICULATION NUMBER'].astype(str)
data_['CorpsId'] = data_['CorpsId'].astype(str)
data_['DATE OF BIRTH'] = data_['DATE OF BIRTH'].apply(lambda x: x.split(' ')[0])

da = DocumentArray([Document(text=data_['Full Name'][i],tags={'department': data_['DEPARTMENT'][i], 'sex': data_['SEX'][i], 'surname':data_['SURNAME'][i], 'othernames':data_['OTHERNAMES'][i], 'matric':data_['MATRICULATION NUMBER'][i],'dob':data_['DATE OF BIRTH'][i],'jamb':data_['JAMB_NUMBER'][i], 'institution': data_['INSTITUTION CODE'][i], 'corps': data_['CorpsId'][i], 'state':data_['STATE OF ORIGIN'][i]}) for i in range(len(data_))])

# the total number of documents is 2821489
da1 = da[0:50000]
da2 = da[50000:100000]
da3 = da[100000:150000]
da4 = da[150000:200000]
da5 = da[200000:250000]
da6 = da[250000:300000]
da7 = da[300000:350000]
da8 = da[350000:400000]
da9 = da[400000:450000]
da10 = da[450000:500000]
da11 = da[500000:550000]
da12 = da[550000:600000]
da13 = da[600000:650000]
da14 = da[650000:700000]
da15 = da[700000:750000]
da16 = da[750000:800000]
da17 = da[800000:850000]
da18 = da[850000:900000]
da19 = da[900000:950000]
da20 = da[950000:1000000]
da21 = da[1000000:1050000]
da22 = da[1050000:1100000]
da23 = da[1100000:1150000]
da24 = da[1150000:1200000]
da25 = da[1200000:1250000]
da26 = da[1250000:1300000]
da27 = da[1300000:1350000]
da28 = da[1350000:1400000]
da29 = da[1400000:1450000]
da30 = da[1450000:1500000]
da31 = da[1500000:1550000]
da32 = da[1550000:1600000]
da33 = da[1600000:1650000]
da34 = da[1650000:1700000]
da35 = da[1700000:1750000]
da36 = da[1750000:1800000]
da37 = da[1800000:1850000]
da38 = da[1850000:1900000]
da39 = da[1900000:1950000]
da40 = da[1950000:2000000]
da41 = da[2000000:2050000]
da42 = da[2050000:2100000]
da43 = da[2100000:2150000]
da44 = da[2150000:2200000]
da45 = da[2200000:2250000]
da46 = da[2250000:2300000]
da47 = da[2300000:2350000]
da48 = da[2350000:2400000]
da49 = da[2400000:2450000]
da50 = da[2450000:2500000]
da51 = da[2500000:2550000]
da52 = da[2550000:2600000]
da53 = da[2600000:2650000]
da54 = da[2650000:2700000]
da55 = da[2700000:2750000]
da56 = da[2750000:2800000]
da57 = da[2800000:2821489]





ann_da = AnnLite(n_dim=768,metric='cosine', data_path='./data', columns={'othernames': 'str', 'surname': 'str', 'institution': 'str', 'department': 'str', 'matric': 'str', 'dob': 'str', 'corps': 'str', 'jamb': 'str','state': 'str', 'sex' : 'str'})


def encode_and_index(da):
    global ann_da
    encoded_da = c.encode(da, show_progress=True)
    ann_da.index(encoded_da)

def main():
    encode_and_index(da1)
    encode_and_index(da2)
    encode_and_index(da3)
    encode_and_index(da4)
    encode_and_index(da5)
    encode_and_index(da6)
    encode_and_index(da7)
    encode_and_index(da8)
    encode_and_index(da9)
    encode_and_index(da10)
    encode_and_index(da11)
    encode_and_index(da12)
    encode_and_index(da13)
    encode_and_index(da14)
    encode_and_index(da15)
    encode_and_index(da16)
    encode_and_index(da17)
    encode_and_index(da18)
    encode_and_index(da19)
    encode_and_index(da20)
    encode_and_index(da21)
    encode_and_index(da22)
    encode_and_index(da23)
    encode_and_index(da24)
    encode_and_index(da25)
    encode_and_index(da26)
    encode_and_index(da27)
    encode_and_index(da28)
    encode_and_index(da29)
    encode_and_index(da30)
    encode_and_index(da31)
    encode_and_index(da32)
    encode_and_index(da33)
    encode_and_index(da34)
    encode_and_index(da35)
    encode_and_index(da36)
    encode_and_index(da37)
    encode_and_index(da38)
    encode_and_index(da39)
    encode_and_index(da40)
    encode_and_index(da41)
    encode_and_index(da42)
    encode_and_index(da43)
    encode_and_index(da44)
    encode_and_index(da45)
    encode_and_index(da46)
    encode_and_index(da47)
    encode_and_index(da48)
    encode_and_index(da49)
    encode_and_index(da50)
    encode_and_index(da51)
    encode_and_index(da52)
    encode_and_index(da53)
    encode_and_index(da54)
    encode_and_index(da55)
    encode_and_index(da56)
    encode_and_index(da57)
    print('finished')

main()


# query = c.encode([Document(text='Balogun Olalere emmanuel')])
# ann_da.search(query, limit=5)

# for q in query:
#     for k, m in enumerate(q.matches):
#         print(json.dumps({'text': m.text, 'tags': m.tags}))