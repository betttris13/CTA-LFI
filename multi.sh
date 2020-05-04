#!/bin/bash
read sval fval threads
for (( i=0; i<$threads; i++ ))
do
   echo "Spawning $i"
   python LFIrunmulti.py $sval $fval $threads $i&
done
