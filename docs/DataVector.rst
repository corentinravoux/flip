DataVector class
================

FLIP includes a :py:class:`~flip.data_vector.basic.DataVector` abstract class that is used to build different classes 
to easily handle the data you want to use and to pass them to one of the different likelihood 
implemented in the :py:mod:`flip.likelihood` module.


Using the :py:class:`~flip.data_vector.basic.DataVector` class you can obtain data and variance / covariance:

.. code-block:: python 

    data, var_data = DataVector(parameter_dic)


You also can compute covariance:

.. code-block:: python 

    Cov = DataVector.compute_cov(
        # model and pw used
        model_name, 
        power_spectrum_dict, 
        # parallelization
        size_batch=size_batch, 
        number_worker=number_worker, 
        # additional parameters
        additional_parameters_values=()
        )



Density
-------


Direct Density
~~~~~~~~~~~~~~

The :py:class:`~flip.data_vector.basic.Dens` class is used on example data as:

.. code-block:: python 

    import pandas as pd
    from flip import data_vector

    grid = pd.read_parquet("flip/flip/data/density_data.parquet")
    grid.rename(columns={'density_err': 'density_error', 
                        'rcom': 'rcom_zobs'}, inplace=True)
    DataDensity = data_vector.Dens(grid.to_dict(orient='list'))


Velocity
--------


Direct Velocity
~~~~~~~~~~~~~~~

The :py:class:`~flip.data_vector.basic.DirectVel` class is used on example data as:

.. code-block:: python 

    import pandas as pd
    import numpy as np
    from flip import data_vector

    data_velocity = pd.read_parquet("flip/flip/data/velocity_data.parquet"))
    data_velocity.rename(columns={'vpec': 'velocity'}, inplace=True)
    data_velocity["velocity_error"] = np.zeros(len(data_velocity["vpec"])

    DataTrueVel = data_vector.DirectVel(data_velocity_true)


Velocity from Hubble diagram residuals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the :py:class:`~flip.data_vector.basic.VelFromHDres` class different estimator of velocities can be used from HD residuals. 
They are described in `Velocity estimators <vel_estimators.html>`_.


The DataVector is initialised as:

.. code-block:: python 

    from flip import data_vector

    DataVel = data_vector.VelFromHDres(data, vel_estimator=estimator_name, **kwargs)


Velocity from SNe Ia SALT2 parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:class:`~flip.data_vector.snia_vectors.VelFromSALTfit` class use the same estimators as :py:class:`~flip.data_vector.basic.VelFromHDres`,
but the input data are the SALT2 fit parameters :code:`mb`, :code:`x1` and :code:`c` alongs with their errors and covariance. 
Also this class always require the :code:`rcom_zobs` field in the data.

The HD residuals are computed using the Tripp relation:

.. math::

    \Delta\mu = m_b  + \alpha x_1 - \beta c - M_0 - 5\log_{10}\left[(1+z)r(z)\right] - 25

The DataVector is initialised as:

.. code-block:: python 

    import pandas as pd
    from flip import data_vector

    data_velocity = pd.read_parquet("flip/flip/data/velocity_data.parquet"))
    DataVel = data_vector.snia_vectors.VelFromSALTfit(
        data_velocity.to_dict(orient='list'), 
        vel_estimator='full'
        )

    mu = DataVel.compute_observed_distance_modulus(test_parameters)
    variance_mu = DataVel.compute_observed_distance_modulus_error(test_parameters)

When using the :code:`__call__` method the SNe Ia HD parameters need to be passed:

.. code-block:: python 

    test_parameters = {
    "alpha":0.14,
    "beta": 3.1,
    "M_0": -19.133,
    "sigma_M": 0.12
    }

    velocity, velocity_error = DataVel(test_parameters)


Density X Velocity
------------------

The :py:class:`~flip.data_vector.basic.DensVel` class allows to init a DataVector with density and velocity. 
It is initialised as:

.. code-block:: python 

    import pandas as pd
    from flip import data_vector
    
    grid = pd.read_parquet("flip/flip/data/density_data.parquet")
    grid.rename(columns={'density_err': 'density_error', 
                        'rcom': 'rcom_zobs'}, inplace=True)

    DataDensity = data_vector.Dens(grid.to_dict(orient='list'))

    data_velocity = pd.read_parquet("flip/flip/data/velocity_data.parquet"))
    DataVel = data_vector.snia_vectors.VelFromSALTfit(
        data_velocity.to_dict(orient='list'), 
        vel_estimator='full'
        )

    DensCrossVel = data_vector.DensVel(DataDensity, DataTrueVel)






