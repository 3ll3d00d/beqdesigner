USER_CHANNELS = ['User 1', 'User 2']
SHORT_USER_CHANNELS = ['U1', 'U2']
JRIVER_NAMED_CHANNELS = [None, None, 'Left', 'Right', 'Centre', 'Subwoofer', 'Surround Left', 'Surround Right',
                         'Rear Left', 'Rear Right', None] + USER_CHANNELS
JRIVER_SHORT_NAMED_CHANNELS = [None, None, 'L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR', None] + SHORT_USER_CHANNELS
JRIVER_CHANNELS = JRIVER_NAMED_CHANNELS + [f"Channel {i + 9}" for i in range(24)]
JRIVER_SHORT_CHANNELS = JRIVER_SHORT_NAMED_CHANNELS + [f"C{i + 9}" for i in range(24)]
