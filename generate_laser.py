from fourier_prop.laser_input import input_laser_field
from fourier_prop.read_laser import read_laser
from fourier_prop.read_laser import read_laser_2d
from fourier_prop.propagator import propagator

from mpi4py import *

def rank0_print(rank, text):
    if rank == 0:
        print(text)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_processes = comm.Get_size()

if rank == 0:
    input_field = input_laser_field.get_input_field(verbose=True)
    for i in range(num_processes - 1):
        comm.send(input_field, dest=i+1)
else:
    input_field = comm.recv(source=0)

input_field.generate_input_Ew_field(req_low_mem=True, comm=comm, rank=rank, num_processes=num_processes)

rank0_print(rank, "Generating output Ew field")
propagator.generate_output_Ew_field(input_field, comm=comm, rank=rank, num_processes=num_processes)


rank0_print(rank, "Generating output Et field")
propagator.generate_output_Et_field_from_Ew(input_field, comm=comm, rank=rank, num_processes=num_processes)


rank0_print(rank, "Computing field at sim grid")
if input_field.prop.spatial_dimensions == 2:
    read_laser.compute_field_at_sim_grid(input_field, comm, rank, num_processes, verbose=True)
else:
    read_laser_2d.compute_field_at_sim_grid(input_field, comm, rank, num_processes, verbose=True)
rank0_print(rank, "Done computing!")
