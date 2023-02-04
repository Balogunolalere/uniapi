from docarray import dataclass, DocumentArray, Document
from docarray.typing import Text
import gzip
import json
from time import sleep
from datetime import datetime


# Load data from a gzipped JSON file
with gzip.open("/home/doombuggy/Downloads/combined.json.gz", "rb") as f:
    print("*** Loading data ***")
    data = json.loads(f.read().decode())



@dataclass
class Page:
    name: Text
    #surname : Text
    #othernames : Text
    department : Text
    #institution : Text
    matric : Text
    dob : Text
    #corps : Text
    #jamb : Text
    state : Text
    sex : Text

da = DocumentArray()

# split data(2821489) into 5 parts 
data_1 = data[:564297]
data_2 = data[564297:1128594]
data_3 = data[1128594:1692891]
data_4 = data[1692891:2257188]
data_5 = data[2257188:2821489]

# replace empty strings with None
for i in data_5:
    for key, value in i.items():
        if value == '':
            i[key] = None

for i in data_5:
    name = i['SURNAME'] + ' ' + i['OTHERNAMES']
    sex = i['SEX']
    matric = str(i['MATRICULATION NUMBER'])
    state = i['STATE OF ORIGIN']
    department = i['DEPARTMENT']
    dob = i['DATE OF BIRTH'] // 1000
    dob = str(datetime.fromtimestamp(dob).strftime('%Y-%m-%d'))
    doc = Page(
        name=name,
        department=department,
        matric=matric,
        dob=dob,
        state=state,
        sex=sex
    )
    print('*** Document created ***')
    da.append(Document(doc))


print("*** Data 5 appended to DocumentArray ***")
sleep(20)
try:
    da.push('university_data5', show_progress=True, public=False)
    print('*** Data pushed ***')
except Exception:
    print('*** Error pushing data ***')
    pass


