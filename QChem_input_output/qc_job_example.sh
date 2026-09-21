#!/bin/bash
#SBATCH --job-name=Cu_H_Bridge
#SBATCH -p joonholee
#SBATCH -t 5:00:00
#SBATCH -n 64
#SBATCH -N 1
#SBATCH --mem=300G
#SBATCH -o qcpbc.out
#SBATCH -e qcpbc.err

tag="Cu_H_bridge"

module load intel/25.0.1-fasrc01 openmpi/5.0.5-fasrc01 intel-mkl/25.0.1-fasrc01

mv INPUT.inp INPUT_${tag}.inp
mv INPUT.scratch INPUT_${tag}.scratch
# Run with tscf and converge to 1e-5
qcpbc -nt 64 -i INPUT_${tag}.inp -o OUTPUT.out -c INPUT_${tag}.scratch > stdout
# Run with DIIS and converge to 1e-8
sed -i 's/^scf_convergence 5$/scf_convergence 8/' INPUT_${tag}.inp
sed -i 's/^tscf true$/!tscf true/' INPUT_${tag}.inp
sed -i 's/^theta 0.001$/!theta 0.001/' INPUT_${tag}.inp
qcpbc -nt 64 -i INPUT_${tag}.inp -o OUTPUT.out -c INPUT_${tag}.scratch > stdout
mv INPUT_${tag}.inp INPUT.inp
mv INPUT_${tag}.scratch INPUT.scratch
