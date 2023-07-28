#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union
from queue import Queue

from fastapi import FastAPI
from pydantic import BaseModel

from chunkflow.lib.cartesian_coordinate import BoundingBox

app = FastAPI()

maxid = 85409058

@app.get("/objids/{id_num}")
def get_base_id(id_num: int):
    global maxid
    base_id = maxid

    maxid += id_num
    print(f'updated maxid to {maxid}')
    return base_id

