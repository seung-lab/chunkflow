#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union
from queue import Queue

from fastapi import FastAPI
from pydantic import BaseModel

from chunkflow.lib.cartesian_coordinate import BoundingBox

app = FastAPI()

class Task(BaseModel):
    bbox: 

class Tasks(BaseModel):
    task_queue: queue

@app.get("/task")
def get_task():

