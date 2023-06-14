from enum import Enum


class VideoMode(Enum):
    ONLINE = 1
    OFFLINE = 2


class OfflineMode(Enum):
    BATCH = 1
    CONCURRENT = 2
