#!/bin/bash

command="tar czvf sim.tar.gz ../fourier_prop patch_load.txt profil.txt scalars.txt "

for i in {0..2}
do
	command="${command} Probes${i}.h5 "
done

command="${command}smilei.py"

eval "$command"