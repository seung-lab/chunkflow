import sys
from typing import Union

from functools import update_wrapper, wraps

import click
from .cartesian_coordinate import Cartesian


class CartesianParamType(click.ParamType):
    name = 'Cartesian'
    
    def convert(self, value: Union[list, tuple], param, ctx):
        assert len(value) == 3
        return Cartesian.from_collection(value)        

CartesianParam = CartesianParamType()

# global dict to hold the operators and parameters
state = {'operators': {}}
DEFAULT_CHUNK_NAME = 'chunk'
DEFAULT_SYNAPSES_NAME = 'syns'
DEFAULT_SKELETON_NAME = 'skels'


def get_initial_task():
    return {'log': {'timer': {}}}


def default_none(ctx, _, value):
    """
    click currently can not use None with tuple type
    it will return an empty tuple if the default=None details:
    https://github.com/pallets/click/issues/789
    """
    if not value:
        return None
    else:
        return value


# the code design is based on:
# https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py
@click.group(chain=True)
@click.option('--mip', '-m',
              type=click.INT, default=0,
              help='default mip level of chunks.')
@click.option('--dry-run/--real-run', default=False,
              help='dry run or real run. default is real run.')
@click.option('--verbose/--quiet', default=False, 
    help='show more information or not. default is False.')
def main(mip, dry_run, verbose):
    """Compose operators and create your own pipeline."""
    
    state['mip'] = mip
    state['dry_run'] = dry_run
    state['verbose'] = verbose
    if dry_run:
        print('\nYou are using dry-run mode, will not do the work!')


@main.result_callback()
def process_commands(operators, mip, dry_run, verbose):
    """This result callback is invoked with an iterable of all 
    the chained subcommands. As in this example each subcommand 
    returns a function we can chain them together to feed one 
    into the other, similar to how a pipe on unix works.
    """
    # It turns out that a tuple will not work correctly!
    stream = [get_initial_task(), ]

    # Pipe it through all stream operators.
    for operator in operators:
        stream = operator(stream)
        # task = next(stream)

    # Evaluate the stream and throw away the items.
    for _ in stream:
        pass


def operator(func):
    """
    Help decorator to rewrite a function so that
    it returns another function from it.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        def operator(stream):
            return func(stream, *args, **kwargs)
        return operator

    return wrapper


def generator(func):
    """Similar to the :func:`operator` but passes through old values unchanged 
    and does not pass through the values as parameter.
    """
    @operator
    def new_func(stream, *args, **kwargs):
        for item in func(*args, **kwargs):
            yield item

    return update_wrapper(new_func, func)