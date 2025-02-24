#!/bin/bash

# `sudo visudo`` to let sudo without password
# <username> ALL=(ALL) NOPASSWD: ALL

# Change fan speed to prvent from huge noise during the Demo
sudo chown $(whoami) /sys/class/hwmon/hwmon1/pwm1_enable
sudo chown $(whoami) /sys/class/hwmon/hwmon1/pwm1
echo "30" > /sys/class/hwmon/hwmon1/pwm1

cd /home/dvt/Desktop/NPL54_ES2/app/NPL54Capture/
gnome-terminal -- bash -c "cd /home/dvt/Desktop/NPL54_ES2/app/NPL54Capture/; sudo ./NPL54Capture -b 1" 

sleep 5

gnome-terminal -- ffplay /dev/video102
gnome-terminal -- ffplay /dev/video103



