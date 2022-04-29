# Minimize Your Quartus Installation with ModelSim Support

Installing the full features of Quartus Lite requires over 25G on your computer. The following is an installation guide to help you get the essentials for **Lab 5**. 

### ***PLEASE FOLLOW THE STEPS IN THE GIVEN ORDER.***

1. Download [Quartus Lite 21.1 Setup](https://download.altera.com/akdlm/software/acdsinst/21.1std/842/ib_installers/QuartusLiteSetup-21.1.0.842-windows.exe) and [Intel Cyclone V Device Support 21.1](https://download.altera.com/akdlm/software/acdsinst/21.1std/842/ib_installers/cyclonev-21.1.0.842.qdz). Make sure the two files appear in the **same folder**. (For Windows 11 users, Version 21.1 is known to have some bugs interacting with the resource manager, which causes the program to crash. Download [Quartus Lite 20.1.1 Setup](https://download.altera.com/akdlm/software/acdsinst/20.1std.1/720/ib_installers/QuartusLiteSetup-20.1.1.720-windows.exe) and [Intel Cyclone V Device Support 20.1.1](https://download.altera.com/akdlm/software/acdsinst/20.1std.1/720/ib_installers/cyclonev-20.1.1.720.qdz) instead.)
3. Run the setup. In the "**Select components**" step, make sure to select `Devices - Cyclone V (1430.9MB)`. **Record** the installation path -- it could be something like `D:\intelFPGA_lite\21.1`. 
3. Download [ModelSim 20.1.1 Setup](https://download.altera.com/akdlm/software/acdsinst/20.1std.1/720/ib_installers/ModelSimSetup-20.1.1.720-windows.exe) and run it. Choose the Starter version. You probably need to **manually alter the installation path** to match the path previously recorded. 
4. Run the Quartus software located in `[INSTALLATION_PATH]\quartus\bin64\quartus.exe`. In **Tools -> Options** menu, go to **General / EDA Tool Options** section and specify the location of ModelSim as `[INSTALLATION_PATH]\modelsim_ase\win32aloem`. There does *not* exist a field called ModelSim-Altera -- don't worry. 
6. Create (1) the project [.qpf], (2) the circuit [.bdf], (3) the waveform [.vwf], and compile the files as specified in the [Lab 5 Guide](https://wiki.illinois.edu/wiki/display/zjuiece120/Lab+5). 
7. A little tweaking is required when you are about to run the simulation. In the waveform file, (1) go to **Simulation -> Simulation Settings**, (2) switch **HDL Language** from Verilog to VHDL, and (3) delete the ``-novopt`` flag in the 5th line of **ModelSim Script**. 

Everything should work fine now. Seek the TAs' help in case new problems arise! 

## Useful Links

[Intel Official Support](https://www.intel.com/content/www/us/en/support.html) (Your first go-to site in case of bugs)
[Intel Product Support Forum - Quartus](https://community.intel.com/t5/Intel-Quartus-Prime-Software/bd-p/quartus-prime-software)
[How to Begin a Simple FPGA Design](https://learning.intel.com/developer/learn/course/external/view/elearning/192/how-to-begin-a-simple-fpga-design)
[Become an FPGA Designer in 4 Hours](https://learning.intel.com/developer/learn/course/external/view/elearning/210/university-self-guided-lab-become-an-fpga-designer-in-4-hours)