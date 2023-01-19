import sys
from typing import Union, Callable, Iterable
import logging
logging.getLogger().setLevel(logging.INFO)

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


# https://stackoverflow.com/questions/58745652/how-can-command-list-display-be-categorised-within-a-click-chained-group
class GroupedGroup(click.Group):
    def command(self, *args, **kwargs):
        """Gather the command help groups"""
        help_group = kwargs.pop('group', None)
        decorator = super(GroupedGroup, self).command(*args, **kwargs)

        def wrapper(f):
            cmd = decorator(f)
            cmd.help_group = help_group
            return cmd

        return wrapper

    def format_commands(self, ctx, formatter):
        # Modified fom the base class method

        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if not (cmd is None or cmd.hidden):
                commands.append((subcommand, cmd))

        if commands:
            longest = max(len(cmd[0]) for cmd in commands)
            # allow for 3 times the default spacing
            limit = formatter.width - 6 - longest

            groups = {}
            for subcommand, cmd in commands:
                help_str = cmd.get_short_help_str(limit)
                subcommand += ' ' * (longest - len(subcommand))
                groups.setdefault(
                    cmd.help_group, []).append((subcommand, help_str))


# the code design is based on:
# https://github.com/pallets/click/blob/main/examples/imagepipe/imagepipe.py
@click.group(name='chunkflow', chain=True, cls=GroupedGroup)
@click.option('--log-level', '-l',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical']), 
              default='debug',
              help='print informations level. default is level INFO.')
@click.option('--log-file', '-f',
    type=click.Path(exists=False), default=None,
    help='log file path.')
@click.option('--mip', '-m',
              type=click.INT, default=0,
              help='default mip level of chunks.')
@click.option('--dry-run/--real-run', default=False,
              help='dry run or real run. default is real run.')
@click.option('--verbose/--quiet', default=False, 
    help='show more information or not. default is False.')
def main(log_level, log_file, mip, dry_run, verbose):
    """Compose operators and create your own pipeline."""
    str2level = {
        'debug'     : logging.DEBUG,
        'info'      : logging.INFO,
        'warning'   : logging.WARNING,
        'error'     : logging.ERROR,
        'critical'  : logging.CRITICAL
    }
    log_level = str2level[log_level]
    logging.getLogger().setLevel(log_level)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    formater = logging.Formatter('%(name)-13s: %(levelname)-8s %(message)s')
    console.setFormatter(formater)
    logging.getLogger().addHandler(console)

    if log_file is not None and len(log_file)>0:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    state['mip'] = mip
    state['dry_run'] = dry_run
    state['verbose'] = verbose
    if dry_run:
        logging.warning('\nYou are using dry-run mode, will not do the work!')


@main.result_callback()
def process_commands(processors: Iterable, log_level, log_file, mip, dry_run, verbose):
    """This result callback is invoked with an iterable of all 
    the chained subcommands. As in this example each subcommand 
    returns a function we can chain them together to feed one 
    into the other, similar to how a pipe on unix works.
    """
    # It turns out that a tuple will not work correctly!
    stream = [get_initial_task(), ]

    # Pipe it through all stream operators.
    for processor in processors:
        breakpoint()
        stream = processor(stream)
        # task = next(stream)

    # Evaluate the stream and throw away the items.
    for item in stream:
        print(f'item in stream: {item}')
        if item is None:
            breakpoint()
            continue
        pass


def operator(func: Callable):
    """
    Help decorator to rewrite a function so that
    it returns another function from it.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        def processor(stream: Iterable):
            return func(stream, *args, **kwargs)
        return processor

    return wrapper


def generator(func: Callable):
    """Similar to the :func:`operator` but passes through old values unchanged 
    and does not pass through the values as parameter.
    """
    @operator
    def new_func(stream: Iterable, *args, **kwargs):
        for task in func(*args, **kwargs):
            yield task

    return update_wrapper(new_func, func)

def initiator(func: Callable):
    """Setup some basic parameters for the task.
    Note that a pipeline should be composed by initiator-->generator-->operator in order.
    """
    
    # @wraps(func)
    # def wrapper(*args, **kwargs):
    #     def initiator(stream):
    #         return func(stream, *args, **kwargs)
    #     return initiator
    # return wrapper

    @operator
    def new_func(stream: Iterable, *args, **kwargs):
        ret = func(*args, **kwargs)
        # while True:
        yield ret
        
    return update_wrapper(new_func, func)