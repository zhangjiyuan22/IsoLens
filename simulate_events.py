import sys
# sys.path.insert(1,'pycode/')
# from functions_cpu import *
from functions_cpu_parallel import *
from getconfig import *
import time


if __name__ == "__main__":
    time_begin = time.time()
    ###############################

    # eventname = 'ob110950'
    # eventname = 'ob110950_uniform_mass_prior'
    # eventname = 'ob110950_test'
    # eventname = 'ob110950_5e10'
    eventname = 'ob110950_isolens'
    # eventname = 'ob110950_5e10based_rerun'

    config = EventInfo()
    config = getEventInfo(eventname)

    SimulateEvents(config)

    time_end = time.time()
    print('\nTotal cost: %.2f min'%((time_end-time_begin)/60))
