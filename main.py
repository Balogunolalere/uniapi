from docarray import DocumentArray, Document
from clip_client import Client
from annlite import AnnLite
import json
import asyncio


from clip_client import Client

c = Client(
    'grpcs://api.clip.jina.ai:2096', credential={'Authorization': '991e3e1eb7c84a1242644521a948e6be'}
)


da = DocumentArray().pull('university_2021_dataset', show_progress=True)

# the total number of documents is 2821489. divide into 8 batches and asynchrounously encode each batch and index into ann
da1 = da[0:352686]
da2 = da[352686:705372]
da3 = da[705372:1058058]
da4 = da[1058058:1410744]
da5 = da[1410744:1763430]
da6 = da[1763430:2116116]
da7 = da[2116116:2468802]
da8 = da[2468802:2821489]

ann_da = AnnLite(n_dim=768,metric='cosine', data_path='./data', columns={'othernames': 'str', 'surname': 'str', 'institution_code': 'str', 'department': 'str', 'matric': 'str', 'date_entered': 'str', 'date_of_birth': 'str', 'corps_id': 'str', 'jamb_number': 'string', 'qualification': 'str', 'state_of_origin': 'str', 'sex' : 'str', 'marital_status': 'str'})


async def encode_and_index(da):
    global ann_da
    encoded_da = await c.encode(da, show_progress=True)
    ann_da.index(encoded_da)

async def main():
    await asyncio.gather(encode_and_index(da1), encode_and_index(da2), encode_and_index(da3), encode_and_index(da4), encode_and_index(da5), encode_and_index(da6), encode_and_index(da7), encode_and_index(da8))

asyncio.run(main())



query = c.encode([Document(text='Balogun Olalere emmanuel')])
ann_da.search(query, limit=5)

for q in query:
    for k, m in enumerate(q.matches):
        print(json.dumps({'text': m.text, 'price': m.tags}))