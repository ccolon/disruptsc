import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def check_script_call(arguments: list[str]):
    """
    Usage should be python main.py region <one optional argument>

    Parameters
    ----------
    arguments

    Returns
    -------

    """
    if len(arguments) < 2:
        raise ValueError('The script takes at least one argument: the region to be studied. '
                         'Ex. python main.py region <one optional argument>')
    if len(arguments) > 3:
        raise ValueError('The script does not take more than one optional argument '
                         'Ex. python main.py region <one optional argument>')
    accepted_optional_arguments: list[str] = [
        'same_transport_network_new_agents',
        'same_agents_new_sc_network',
        'same_sc_network_new_logistic_routes',
        'same_logistic_routes',
        'same_agents_new_transport_network'
    ]
    if len(arguments) > 2:
        if arguments[2] not in accepted_optional_arguments:
            raise ValueError("Argument " + arguments[2] + " is not valid.\
                Possible values are: " + ','.join(accepted_optional_arguments))

