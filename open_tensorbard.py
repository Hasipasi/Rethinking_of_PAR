#This script is used to open tensorboard in the browser
# Usage: python open_tensorboard.py

import os
import sys
import subprocess
import time
import tensorboard

logdir = os.path.join("/net/cremi/bbodis/Bureau/espaces/travail/Rethinking_of_PAR/exp_result")

print("Log directory: ", logdir)

# Check if the log directory exists

if not os.path.exists(logdir):

    print("Log directory does not exist")
    sys.exit()

# open tensorboard in the browser

subprocess.Popen(['tensorboard', '--logdir', logdir])
