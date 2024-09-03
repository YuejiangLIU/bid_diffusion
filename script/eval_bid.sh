#!/bin/bash

TH=16
OH=2
OUTDIR=outputs
METHOD=bid
DECAY=0.5
NSAMPLE=15
NMODE=3

# ===================================================================================================

TASK=pusht
DIR=ckpt/${TASK}
CKPT='epoch=0150-test_mean_score=0.909.ckpt'
REF='epoch=0050-test_mean_score=0.250.ckpt'

# ===================================================================================================

NOISE=0.0

# open-loop
AH=8
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}_${NSAMPLE}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --nsample ${NSAMPLE} --nmode ${NMODE} --decay ${DECAY} --reference ${DIR}/${REF} --noise ${NOISE} --ntest 100

# ===================================================================================================

NOISE=1.0

# closed-loop
AH=1
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}_${NSAMPLE}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --nsample ${NSAMPLE} --nmode ${NMODE} --decay ${DECAY} --reference ${DIR}/${REF} --noise ${NOISE} --ntest 100
