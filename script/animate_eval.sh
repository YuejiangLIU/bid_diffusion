#!/bin/bash

TH=16
OH=2
NOISE=0.0
PERTURB=1.0
OUTDIR=animation

# ===================================================================================================

TASK=pusht
DIR=ckpt/${TASK}
CKPT='epoch=0150-test_mean_score=0.909.ckpt'

# ===================================================================================================

# open-loop
METHOD=random
AH=8
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --noise ${NOISE} --perturb ${PERTURB} --ntest 2

# closed-loop
METHOD=random
AH=1
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --noise ${NOISE} --perturb ${PERTURB} --ntest 2

# ===================================================================================================

# warmstart
METHOD=warmstart
AH=1
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --noise ${NOISE} --perturb ${PERTURB} --ntest 2

# ===================================================================================================

# ema
METHOD=ema
DECAY=0.5
AH=1
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}_${DECAY}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --decay ${DECAY} --noise ${NOISE} --perturb ${PERTURB} --ntest 2

# ===================================================================================================

# bid
METHOD=bid
REF='epoch=0050-test_mean_score=0.250.ckpt'
DECAY=0.5
AH=1
NSAMPLE=15
NMODE=3
python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}_${NSAMPLE}_${DECAY}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --nsample ${NSAMPLE} --nmode ${NMODE} --decay ${DECAY} --perturb ${PERTURB} --reference ${DIR}/${REF} --noise ${NOISE} --ntest 2
