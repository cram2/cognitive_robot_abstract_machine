import numpy as np
import pytest
import scipy.sparse as sp
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from sqlalchemy import select
from sqlalchemy.orm import Session

from giskardpy.orm.ormatic_interface import Base, QPDataExplicitDAO
from giskardpy.qp.qp_data import QPDataExplicit

# %% fixtures


@pytest.fixture
def session():
    session = Session(create_engine("sqlite:///:memory:"))
    Base.metadata.create_all(bind=session.bind)
    yield session
    Base.metadata.drop_all(session.bind)
    session.close()


@pytest.fixture
def qp_data():
    return QPDataExplicit(
        num_equality_slack_variables=1,
        num_inequality_slack_variables=2,
        quadratic_weights=np.array([1.0, 2.0, 3.0]),
        linear_weights=np.array([0.1, 0.2, 0.3]),
        box_lower_constraints=np.array([-1.0, -2.0, -3.0]),
        box_upper_constraints=np.array([1.0, 2.0, 3.0]),
        equality_matrix=sp.eye(3, format="csc"),
        equality_bounds=np.array([0.5, 0.6, 0.7]),
        inequality_matrix=sp.eye(3, format="csc"),
        inequality_lower_bounds=np.array([-4.0, -5.0, -6.0]),
        inequality_upper_bounds=np.array([4.0, 5.0, 6.0]),
    )


# %% numpy column persistence


class TestNumpyColumnsSurviveRoundTrip:
    """
    The array fields of a QP problem are mapped through NumpyType, which giskardpy's
    generator registers for :class:`numpy.ndarray`.

    Without that registration the columns are dropped and the table persists nothing but
    its identifier.
    """

    def test_arrays_are_read_back_unchanged(self, session, qp_data):
        session.add(to_dao(qp_data))
        session.commit()
        session.expunge_all()

        loaded = session.scalars(select(QPDataExplicitDAO)).one()

        np.testing.assert_array_equal(
            loaded.quadratic_weights, qp_data.quadratic_weights
        )
        np.testing.assert_array_equal(loaded.linear_weights, qp_data.linear_weights)
        np.testing.assert_array_equal(
            loaded.box_lower_constraints, qp_data.box_lower_constraints
        )
        np.testing.assert_array_equal(
            loaded.box_upper_constraints, qp_data.box_upper_constraints
        )
        np.testing.assert_array_equal(loaded.equality_bounds, qp_data.equality_bounds)
        np.testing.assert_array_equal(
            loaded.inequality_lower_bounds, qp_data.inequality_lower_bounds
        )
        np.testing.assert_array_equal(
            loaded.inequality_upper_bounds, qp_data.inequality_upper_bounds
        )

    def test_scalar_fields_are_read_back_unchanged(self, session, qp_data):
        session.add(to_dao(qp_data))
        session.commit()
        session.expunge_all()

        loaded = session.scalars(select(QPDataExplicitDAO)).one()

        assert loaded.num_equality_slack_variables == 1
        assert loaded.num_inequality_slack_variables == 2
