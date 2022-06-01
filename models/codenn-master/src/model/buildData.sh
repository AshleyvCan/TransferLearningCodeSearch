#!/bin/bash

MAX_CODE_LENGTH=80
MAX_NL_LENGTH=80
BATCH_SIZE=20

# Create working directory
if [ ! -d "$CODENN_WORK" ]; then
	mkdir $CODENN_WORK
fi


# Prepare C# and SQL data
SQL_UNK_THRESHOLD=3
CSHARP_UNK_THRESHOLD=2
NL_UNK_THRESHOLD=2


python buildData.py python $MAX_CODE_LENGTH $MAX_NL_LENGTH $CSHARP_UNK_THRESHOLD $NL_UNK_THRESHOLD


th buildData.lua -language python
