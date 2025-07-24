import happi
import os

print("Getting gifs...")

DIRECTORY = "./gifs"
folderExists = os.path.exists(DIRECTORY)
if not folderExists:
    os.makedirs(DIRECTORY)
    print("The new gif directory is created!")
else:
    print("Gif folder already exists.")

print("Opening sim")
S = happi.Open("./", verbose=True)

print("Getting Ey XY Plane")
S.Probe(0, "Ey", units=["fs", "um"], vmin=-20, vmax=20, aspect="equal", xlabel="Propagation Axis [um]",
        ylabel="Transverse Axis [um]").animate(movie=DIRECTORY + '/probe_Ey_xy.gif', fps=7)

print("Getting Ey XZ Plane")
S.Probe(1, "Ey", units=["fs", "um"], vmin=-20, vmax=20, aspect="equal", xlabel="Propagation Axis [um]",
        ylabel="Transverse Axis [um]").animate(movie=DIRECTORY + '/probe_Ey_xz.gif', fps=7)

print("Getting Ey YZ Plane")
S.Probe(2, "Ey", units=["fs", "um"], vmin=-20, vmax=20, aspect="equal", xlabel="Propagation Axis [um]",
        ylabel="Transverse Axis [um]").animate(movie=DIRECTORY + '/probe_Ey_yz.gif', fps=7)

print("Getting Ez XY Plane")
S.Probe(0, "Ez", units=["fs", "um"], vmin=-20, vmax=20, aspect="equal", xlabel="Propagation Axis [um]",
        ylabel="Transverse Axis [um]").animate(movie=DIRECTORY + '/probe_Ez_xy.gif', fps=7)



