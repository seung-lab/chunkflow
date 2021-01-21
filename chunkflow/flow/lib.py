
import logging
from functools import update_wrapper, wraps
import click
from cloudvolume.lib import yellow
import numpy as np

# global dict to hold the operators and parameters
state = {'operators': {}}
DEFAULT_CHUNK_NAME = 'chunk'


def get_initial_task():
    return {'skip': False, 'log': {'timer': {}}}


def handle_task_skip(task, name):
    if task['skip'] and task['skip_to'] == name:
        # have already skipped to target operator
        task['skip'] = False


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
@click.option('--log-level', '-l', 
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical']), 
              default='info',
              help='print informations level. default is level 1.')
@click.option('--mip', '-m',
              type=int, default=0,
              help='default mip level of chunks.')
@click.option('--dry-run/--real-run', default=False,
              help='dry run or real run. default is real run.')
def main(log_level, mip, dry_run):
    """Compose operators and create your own pipeline."""
    str2level = {
        'debug'     : logging.DEBUG,
        'info'      : logging.INFO,
        'warning'   : logging.WARNING,
        'error'     : logging.ERROR,
        'critical'  : logging.CRITICAL
    }
    logging.basicConfig(filename='chunkflow.log', 
                        level=str2level[log_level])
    state['mip'] = mip
    state['dry_run'] = dry_run
    if dry_run:
        logging.warning('\nYou are using dry-run mode, will not do the work!')
    pass


@main.resultcallback()
def process_commands(operators, log_level, mip, dry_run):
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

    # Evaluate the stream and throw away the items.
    if stream:
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