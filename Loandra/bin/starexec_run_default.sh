#!/bin/bash
delay=3
wl=$(($STAREXEC_WALLCLOCK_LIMIT-$delay))
timeout -s 15 $wl ./loandra_static -algorithm=1 -pmreslin=1 -pmreslin-varres -pmreslin-cglim=30 -no-pmreslin-incvarres -print-model $1
