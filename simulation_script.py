import os, sys
import math
import numpy as np
from pathlib import Path

#x = np.arange(179,201,1)
#x = np.append(x,np.arange(240,250,1))

x = np.arange(186,200,1)
x = np.append(x,340)
for rotation_angle in x:
    file = Path("/home/jacob/xlc-bkg-sim/macfile/Script.mac")
    with file.open('r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if lines[i].startswith("/xlcalibur/det/rotnangle"):
            lines[i] = "/xlcalibur/det/rotnangle " + str(round(rotation_angle)) + " " + "\n"
    with file.open("w") as f:
        f.writelines(lines)


    x = 1
    for theta in np.arange(0, 180):
        x_ = np.cos(theta * math.pi / 180)
        y_ = np.sin(theta * math.pi / 180)
        x_, y_ = np.round(x_, 3), np.round(y_, 3)

    
        with file.open('r') as f:
            lines = f.readlines()
    
        for i in range(len(lines)):
            if lines[i].startswith("/xlcalibur/gun/poldirn"):
                lines[i] = "/xlcalibur/gun/poldirn "+str(x_)+" "+str(y_)+ " "+"0\n"
        with file.open("w") as f:
            f.writelines(lines)
    
        os.system('/home/jacob/xlc-bkg-sim/build/BGOShield /home/jacob/xlc-bkg-sim/macfile/Script.mac 42 /home/jacob/xlc_sim_outputs/level0sim/Script_Test 20')
    
        os.system(f'hadd -f /home/jacob/xlc_sim_outputs/level1sim/Script_PA_{theta}_RA{rotation_angle}.root /home/jacob/xlc_sim_outputs/level0sim/Script_Test_42_t*.root')
        os.system('rm  /home/jacob/xlc_sim_outputs/level0sim/Script_Test*.root')
        os.system(f'/home/jacob/xlc-sim-analysis/build/XCAnalysis -f /home/jacob/xlc_sim_outputs/level1sim/Script_PA_{theta}_RA{rotation_angle}.root -o /home/jacob/xlc_sim_outputs/flight_like/ScriptGenerated/Script_PA_{theta}_RA{rotation_angle}')
    
        os.system(f'python3 /home/jacob/RootFileAnalysis.py -f /home/jacob/xlc_sim_outputs/flight_like/ScriptGenerated/Script_PA_{theta}_RA{rotation_angle}.root -r {rotation_angle}')
    
print("Successfully Completed")
