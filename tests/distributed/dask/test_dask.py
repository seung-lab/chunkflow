#%%

import dask



@dask.delayed
def add(x,y):
    return x + y

@dask.delayed
def inc(x):
    return x + 1


if __name__ == '__main__':
    from dask.distributed import Client

    client = Client()
    print(f'client: {client}')
    a = inc(1)
    b = inc(2)
    c = add(a, b)

    c = c.compute()

    print(f'c: {c}')

    print(client.dashboard_link)


# %%


