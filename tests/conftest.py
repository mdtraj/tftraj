import pytest
import tensorflow as tf

@pytest.fixture(scope='session')
def sess():
    sess = tf.Session()
    yield sess
    sess.close()
