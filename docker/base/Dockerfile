FROM ubuntu:18.04

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
        apt-utils \
        python3 \
        python3-pip \
    && pip3 install -U pip \
    # test whether pip is working 
    # there is an issue of pip:
    # https://github.com/laradock/laradock/issues/1496
	# we need this hash to solve this issue
    && hash -r pip3 \ 
    && pip3

