#!/bin/bash

LANGUAGE=$1
GPUIDX=1
BEAMSIZE=10

# Run Training
th main.lua -dev_ref_file <ref_file> -gpuidx 1 -language python

# Run prediction
th nl2code_org.lua -language python -encoder python.encoder -decoder python.decoder 
