import time

class ModelTimer():
    def __init__(self, elapsed=None, verbose=False):
        self.verbose = verbose
        self.total_time = 0
        self.start_time = time.time()
        if elapsed:
            self.total_time = elapsed
            
    def start_epoch_timer(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"Epoch start time: {self.__format(self.start_time)}")
        
    def stop_epoch_timer(self, update=True):
        epoch_time = time.time() - self.start_time
        if update:
            self.total_time += epoch_time
        if self.verbose:
            print(f"Epoch time taken: {self.__format(epoch_time)}, Total time taken: {str(self)}")
    
    def __format(self, ftime):
        return time.strftime("%H:%M:%S", time.gmtime(ftime))
    
    def __str__(self):
        return self.__format(self.total_time)