#!/bin/bash
chunkflow \
    fetch-task-kombu -r 1 --queue-name=amqp://172.31.31.249:5672 \
    delete-task-in-queue-kombu

