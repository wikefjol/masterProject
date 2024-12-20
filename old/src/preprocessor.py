
import src

class Preprocessor():

    def __init__(self, config):
        self.config = config
        self.augmentor = None
        self.padder = None # Padder(config)
        self.tokenizor = None
        self.lengthAdjuster = None
        self.encoder = None

    def process(self):
        ...
    
        











