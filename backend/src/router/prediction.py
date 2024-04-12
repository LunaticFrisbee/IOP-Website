from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
from MLmodels import holtsWinterModel

router = APIRouter(
    prefix="/prediction",
    tags=["prediction"]
)

@router.post('/holtwinter')
def holtWinters(npreds: int, dataSet: UploadFile = File(...)):
    dataFrame = pd.read_csv(dataSet.file)
    print(dataFrame)
    forecast = holtsWinterModel.holtWinterModel(dataFrame, npreds)
    return forecast