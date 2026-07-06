import os
import pandas as pd
from datetime import datetime

filename="plates.xlsx"

def log_plate(plate_text,img_name,conf):
    if os.path.exists(filename):
        df=pd.read_excel(filename)
    else:
        df=pd.DataFrame(columns=["plate_text","date_time","img_name","confidence"])
    new_row= {"plate_text":plate_text,
             "date_time":datetime.now(),
             "img_name":img_name,
             "confidence":conf
    }   
    df=pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)
    df.to_excel(filename,index=False)