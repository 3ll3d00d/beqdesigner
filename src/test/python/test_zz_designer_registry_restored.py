'''
conftest.py's `_designer_registry_restored`: a designer one test registers and never removes -- as a service job or CLI
run does with its profile's designers -- is gone before the next test starts.
'''
from pipeline.designer.registry import register_designer, registered_designers, takes_sources


def test_a_test_that_registers_a_designer_and_does_not_remove_it():
    register_designer('test.left_behind', lambda request, sources=None: None, takes_sources=True)
    assert 'test.left_behind' in registered_designers()


def test_the_next_test_does_not_see_it():
    assert 'test.left_behind' not in registered_designers()
    assert not takes_sources('test.left_behind')
