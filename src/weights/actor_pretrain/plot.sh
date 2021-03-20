#!/bin/sh

EXPERIMENT=2
MODEL=""
EPOCHS=""
ITEM=""

OUT="exp$EXPERIMENT/"
FILE="exp$EXPERIMENT/"

if [ -z "$MODEL" ]
then
    FILE+="loss_"
    OUT+="loss"
else
    FILE+="pretrain_$MODEL/loss_"
    OUT+="pretrain_$MODEL/loss"
fi

if [ -z "$ITEM" ]
then
    FILE+="$EXPERIMENT"
    OUT+=".png"
else
    FILE+="$ITEM"
    FILE+="_$EXPERIMENT"
    OUT+="_$ITEM.png"
fi

if [ -z "$MODEL" ]
then
    true
else
    FILE+="_pretrain_$MODEL"
fi

if [ -z "$EPOCHS" ]
then
    FILE+=".pickle"
else
    FILE+="_$EPOCHS.pickle"
fi


python3 plot.py \
    --file $FILE \
    --out  $OUT
