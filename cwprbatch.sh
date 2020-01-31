#!/bin/bash

for f in simulations/*.json; do python cwprmodel.py "$f"; done
