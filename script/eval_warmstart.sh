#!/bin/bash

TH=16
OH=2
OUTDIR=outputs
METHOD=warmstart

# ===================================================================================================

TASK=pusht
DIR=ckpt/${TASK}
CKPT='epoch=0150-test_mean_score=0.909.ckpt'

# ===================================================================================================

NOISE=0.0

# open-loop
AH=8
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --noise ${NOISE} --ntest 100

# ===================================================================================================

NOISE=1.0

# closed-loop
AH=1
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --noise ${NOISE} --ntest 100
