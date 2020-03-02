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

#. Update version number. The version number is defined in `chunkflow/__version__.py`, increase the version number before releasing. 

#. Create a new release in PyPi.

Our travis build/test system can automatically release/update a version if you tag the commit. It is recommended to use travis system for new version release.

.. code-block:: console

   git tag v1.0.0
   git push origin v1.0.0

Of course, you can also do it manually.

.. code-block:: console

   python setup.py sdist
   twine upload dist/chunkflow-version.tar.gz

.. note::

    If you would like to include/exclude some files/folders, please edit the MANIFEST.in file.

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
#. Take a look of the existing simple operators, such as :file:`chunkflow/flow/create_chunk.py`.

#. Create your own operator file in the `flow` folder.

#. Create your own operator class such as `MyOperator`, and this class should inherit from the `.base.OperatorBase`.

#. Define the job of you operator in `__call__(self, chunk, *other_args, **kwargs)`.

#. add the import operator code in :file:`chunkflow/flow/operators/__init__.py`.

#. Add unit test in :file:`tests/flow/`. 

#. Run your unit test by :code:`pytest tests/flow/my_operator.py`
   
#. Add documentation in :ref:`api`.
