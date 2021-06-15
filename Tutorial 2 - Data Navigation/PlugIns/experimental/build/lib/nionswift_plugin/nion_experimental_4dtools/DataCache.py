import threading
import time
import numpy as np

class DataCache:
    def __init__(self, lifetime=30, modify_data_fn=None):
        self.__cached_data = None
        self.__cached_uuid = None
        self.__last_requested = time.time()

        self.lifetime = lifetime
        self.modify_data_fn = modify_data_fn if callable(modify_data_fn) else np.array

        self.__thread = threading.Thread(target=self.__cache_loop, daemon=True)
        self.__lock = threading.Lock()
        self.__thread.start()

    def __cache_loop(self):
        while True:
            with self.__lock:
                now = time.time()
                if now - self.__last_requested > self.lifetime:
                    self.__cached_data = None
                    self.__cached_uuid = None
            time.sleep(0.5)

    def get_cached_data(self, data_source):
        uuid = str(data_source.uuid)
        with self.__lock:
            if self.__cached_uuid != uuid:
                xdata = data_source.xdata
                self.__cached_uuid = uuid
                self.__cached_data = self.modify_data_fn(xdata.data)
                #self.__cached_data = np.reshape(xdata.data, xdata.data.shape[:2] + (-1,))
            self.__last_requested = time.time()
            return self.__cached_data
