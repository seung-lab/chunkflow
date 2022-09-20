#!/bin/bash
printf "\n number of generators: "
grep -o '@generator' chunkflow/flow/flow.py| wc -l

printf "\nnumber of operators:"
grep -o '@operator' chunkflow/flow/flow.py| wc -l

printf "\nnumber of plugins in chunkflow: "
\ls chunkflow/plugins/*.py | grep -e __init__.py --invert-match | wc -w

printf "\nnumber of external plugins: "
find chunkflow/plugins/chunkflow-plugins/chunkflowplugins/ -type f | wc -l
