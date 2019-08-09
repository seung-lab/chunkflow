.. _development:

Development
############

Install
**********
Install with development mode (preferably in a python virtual environment)::
   
   git clone https://github.com/seung-lab/chunkflow.git
   cd chunkflow
   pip3 install -r requirements.txt
   python3 setup.py develop

.. note::

    we support python version >=3.5

Release
***********
Create a new release in PyPi::

   python setup.py sdist
   twine upload dist/chunkflow-version.tar.gz

Documentation
***************
We use `Sphinx`_ with `reStructuredText`_ to make documentation. You can make it locally for tests::

   cd docs
   pip3 install -r requirements.txt
   make html

.. _Sphinx: https://www.sphinx-doc.org
.. _reStructuredText: http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html

You can also make other formats, such as pdf and epub. Checkout it out with `make help`.

Add a New Operator
*******************

