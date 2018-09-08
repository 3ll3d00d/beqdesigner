from model.link import LinkedSignalsModel


class DummySignal:
    def __init__(self, name):
        self.name = name
        self.slaves = []
        self.master = None


def test_no_masters():
    size = 10
    expected_names = [f"Mock {x}" for x in range(0, size)]
    signal_model = [DummySignal(expected_names[x]) for x in range(0, size)]
    lsm = LinkedSignalsModel(signal_model)
    assert len(lsm.rows) == 0
    assert len(lsm.columns) == size
    assert lsm.columns == expected_names


def test_one_master():
    size = 10
    expected_names = [f"Mock {x}" for x in range(0, size)]
    signal_model = [DummySignal(expected_names[x]) for x in range(0, size)]
    signal_model[0].slaves = signal_model[1:]
    for x in range(1, size):
        signal_model[x].master = signal_model[0]
    lsm = LinkedSignalsModel(signal_model)
    assert len(lsm.rows) == 1
    assert len(lsm.columns) == size - 1
    assert lsm.columns == [s.name for s in signal_model[0].slaves]
    for y in range(0, size - 1):
        assert lsm.is_slave(0, y) is True


def test_two_master():
    size = 10
    expected_names = [f"Mock {x}" for x in range(0, size)]
    signal_model = [DummySignal(expected_names[x]) for x in range(0, size)]
    signal_model[0].slaves = signal_model[1:5]
    signal_model[5].slaves = signal_model[6:]
    for x in signal_model[0].slaves:
        x.master = signal_model[0]
    for x in signal_model[5].slaves:
        x.master = signal_model[5]
    lsm = LinkedSignalsModel(signal_model)
    assert len(lsm.rows) == 2
    assert len(lsm.columns) == size - 2
    expected_columns = [s.name for s in signal_model[0].slaves] + [s.name for s in signal_model[5].slaves]
    assert lsm.columns == expected_columns
    for y in range(0, size - 2):
        if y < 4:
            assert lsm.is_slave(0, y) is True
            assert lsm.is_slave(1, y) is False
        else:
            assert lsm.is_slave(0, y) is False
            assert lsm.is_slave(1, y) is True


def test_make_master():
    size = 3
    expected_names = [f"Mock {x}" for x in range(0, size)]
    signal_model = [DummySignal(expected_names[x]) for x in range(0, size)]
    # 0 -> 1, 2
    signal_model[0].slaves = [signal_model[1]]
    signal_model[1].master = signal_model[0]
    lsm = LinkedSignalsModel(signal_model)
    assert lsm.is_slave(0, 0) is True
    assert lsm.is_slave(0, 1) is False
    assert len(lsm.rows) == 1
    assert len(lsm.columns) == 2
    lsm.make_master('Mock 2')
    assert len(lsm.rows) == 2
    assert len(lsm.columns) == 1
    assert list(lsm.row_keys) == ['Mock 0', 'Mock 2']
    assert list(lsm.columns) == ['Mock 1']
    assert lsm.is_slave(0, 0) is True
    assert lsm.is_slave(1, 0) is False
    # now toggle to prove we can only have one master for slave
    lsm.toggle(1, 0)
    assert lsm.is_slave(0, 0) is False
    assert lsm.is_slave(1, 0) is True
    # now remove the master
    lsm.remove_master('Mock 0')
    assert len(lsm.rows) == 1
    assert len(lsm.columns) == 2
    assert list(lsm.row_keys) == ['Mock 2']
    assert list(lsm.columns) == ['Mock 0', 'Mock 1']


def test_slave_to_master_to_slave():
    size = 3
    expected_names = [f"Mock {x}" for x in range(0, size)]
    signal_model = [DummySignal(expected_names[x]) for x in range(0, size)]
    # 0 -> 1, 2
    signal_model[0].slaves = [signal_model[1]]
    signal_model[1].master = signal_model[0]
    lsm = LinkedSignalsModel(signal_model)
    assert lsm.is_slave(0, 0) is True
    assert lsm.is_slave(0, 1) is False
    lsm.toggle(0, 1)
    assert lsm.is_slave(0, 0) is True
    assert lsm.is_slave(0, 1) is True
    lsm.toggle(0, 0)
    assert lsm.is_slave(0, 0) is False
    assert lsm.is_slave(0, 1) is True
