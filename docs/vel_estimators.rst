Velocity estimators
===================

The velocities estimators from the Hubble diagram residulas :math:`\Delta\mu` are implemeted in the 
:py:meth:`~flip.data_vector.basic.redshift_dependence_velocity`. 
This function return the coefficient :math:`J(z)` such that :math:`\hat{v} = J(z)\Delta\mu`.

Watkins estimator
-----------------

Watkins estimator from `Watkins and Feldman 2015 <http://academic.oup.com/mnras/article/450/2/1868/980317/An-unbiased-estimator-of-peculiar-velocity-with>`_ is such that

.. math::

    J(z) = \frac{c\ln10}{5} \frac{z}{1+z}
    
Low z estimator
----------------

The low-z estimator is such as:

.. math::

    J(z) = \frac{c\ln10}{5} z


Hubble highorder estimator
--------------------------

The Hubble highorder estimator use an order 3 expansion with respect to :math:`z` of the Hubble law:

.. math::

    J(z) = \frac{\ln10}{5}\frac{z}{1 + z}\left[ 1 + \frac{1}{2} (1 - q_0)z - \frac{1}{6}(1 - q_0 - 3 q_0^2 + j_0) z^2\right]

When using this estimator you need to pass the deceleration :math:`q_0` and jerk :math:`j_0` parameters. 


Example with the :py:class:`~flip.data_vector.basic.VelFromHDres` class:

.. code-block:: python 

    from flip import data_vector

    DataVel = data_vector.VelFromHDres(data, velocity_estimator="hubble highorder", q0=-0.55,j0=-1)


Full estimator
--------------

The Full estimator need to assume a cosmology it is such as:

.. math::

    J(z) = \frac{c\ln10}{5}\left(c\frac{1 + z}{r(z)H(z)} -1\right)^{-1}

where :math:`r(z)` is the comoving distance and :math:`H(z)` the hubble function.

When using this estimator your data need to contain 
the :code:`hubble_norm` and :code:`rcom_zobs` fields such that :code:`hubble_norm` is :math:`h(z) = H(z) / 100` and :code:`rcom_zobs` is the comoving distance in Mpc :math:`h^{-1}`.


