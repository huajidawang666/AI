class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n 
    
    def add(self, *args):
        self.data = [a + b for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        res = self.data[idx]
        return res.item() if hasattr(res, 'item') else res