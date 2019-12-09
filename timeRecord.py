import time

class timeRecord:
    def __init__(self):
        self.timelists = []
        self.taglists = []
    
    def start(self):
        self.timelists.append(time.clock())

    def record(self, tag):
        self.timelists.append(time.clock())
        self.taglists.append(tag)
    
    def print(self):
        l = len(self.taglists)
        for i in range(l):
            print(self.taglists[i] + ': ' + str(round(self.timelists[i + 1] - self.timelists[i], 4)))
