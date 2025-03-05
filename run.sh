#!/bin/bash

# `sudo visudo`` to let sudo without password
# <username> ALL=(ALL) NOPASSWD: ALL

# Change fan speed to prvent from huge noise during the Demo
sudo chown $(whoami) /sys/class/hwmon/hwmon1/pwm1_enable
sudo chown $(whoami) /sys/class/hwmon/hwmon1/pwm1
echo "1" >  /sys/class/hwmon/hwmon1/pwm1_enable

# Manually change the fan speed, 0 to 255 (max)
echo "70" > /sys/class/hwmon/hwmon1/pwm1

cd /home/dvt/Desktop/NPL54_ES2/app/NPL54Capture/
gnome-terminal -- bash -c "cd /home/dvt/Desktop/NPL54_ES2/app/NPL54Capture/; sudo ./NPL54Capture -b 1" 

sleep 5

#gnome-terminal -- ffplay /dev/video102
#gnome-terminal -- ffplay /dev/video103

sudo pkill python3

gnome-terminal -- bash -c "cd /home/dvt/Desktop/Demo_Nuvo-9166GC_PCIe-NPL54; python3 demo_main_102.py" 

sleep 0.1

gnome-terminal -- bash -c "cd /home/dvt/Desktop/Demo_Nuvo-9166GC_PCIe-NPL54; python3 demo_main_103.py" 







