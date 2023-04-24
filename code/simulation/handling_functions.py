import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def check_script_call(arguments: list[str]):
    accepted_script_arguments: list[str] = [
        'same_transport_network_new_agents',
        'same_agents_new_sc_network',
        'same_sc_network_new_logistic_routes',
        'same_logistic_routes',
    ]
    if len(arguments) > 2:
        raise ValueError('The script does not take more than 1 arguments')
    if len(arguments) > 1:
        if sys.argv[1] not in accepted_script_arguments:
            raise ValueError("First argument " + sys.argv[1] + " is not valid.\
                Possible values are: " + ','.join(accepted_script_arguments))

