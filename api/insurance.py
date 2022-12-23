from fastapi import FastAPI
import pickle
import pandas as pd
from typing import Union
import numpy as np

app = FastAPI()

@app.get("/")
async def root(age: Union[int, None] = 30,
               bmi: Union[float, None] = 25,
               children : Union[int, None] = 0,
               smoker: Union[str, None] = "no",
               sex: Union[str, None] = "male"):

    x_user = get_user_df(age, bmi, children, smoker, sex)
    pickle_model = pickle.load(open("pipeline_deploy_insurance.pkl", "rb"))
    pred = pickle_model.predict(x_user)
    return {"prédiction": f"{round(pred[0])}"}


def get_user_df(age, bmi, children, smoker, sex):
    x_user = pd.DataFrame({
        "age" : [int(age)],
        "bmi" : [float(bmi)],
        "children" : [int(children)],
        "smoker" : [str(smoker).lower()],
        "sex" : [str(sex).lower()]
    })
    return x_user
