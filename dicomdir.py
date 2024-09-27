import sys
import pathlib
import pydicom
import pandas as pd

folder = sys.argv[1]
csv_file = sys.argv[2]
file_list =  [str(x) for x in pathlib.Path(folder).rglob("*") if not x.is_dir() and pydicom.misc.is_dicom(str(x))]

mylist = []
for x in file_list:
    ds = pydicom.dcmread(x,stop_before_pixels=True)
    mylist.append(dict(
        FilePath=x,
        StudyInstanceUID=ds.StudyInstanceUID,
        SeriesInstanceUID=ds.SeriesInstanceUID,
        StudyDescription=ds.StudyDescription,
        SeriesDescription=ds.SeriesDescription,
    ))

df = pd.DataFrame(mylist)
df.to_csv(csv_file,index=False)

for SeriesInstanceUID in df.SeriesInstanceUID.unique():
    tmp = df[df.SeriesInstanceUID==SeriesInstanceUID]
    SeriesDescription = tmp.iloc[-1]['SeriesDescription']
    print(f"{SeriesDescription} count:{len(tmp)} SeriesInstanceUID: {SeriesInstanceUID}")

"""
docker run -it -v $PWD:/workdir pangyuteng/dcm:latest bash
python dicomdir.py tmp/dcm_folder ok.csv > ok.md

"""