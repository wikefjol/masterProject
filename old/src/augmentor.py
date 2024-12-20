import random
from itertools import zip_longest

class Augmentor():

    def __init__(self, config):
        self.config: dict = config
        self.strategy: str = self.config['preprocessing']['strategy']['augmenting']
        self.nucleotide_bases: list[str] = ['A','C','T','G']
        self.strategies: dict = {'base': self._baseStrategy, 'nothing' : self._noStrategy}

    def augmentSequence(self, seq: str, strategy: str) -> str:
        '''
        Wrapper method. Runs chosen strategy if valid.
        '''
        if strategy not in self.strategies:
            raise ValueError(f"Augmentation strategy {strategy} is not valid. Select from {list( self.strategies.keys() )}.")
    
        # match 
        # case
        return self.strategies[strategy](seq)

    
    def _baseStrategy(self, seq: str) -> str:
        '''
        This strategy is almost identical to base. But when doing forward swapping it skips over the next 
        character so that we don't propagate a character forward by repeated swapping. Could possibly 
        replace base if marcus agrees. 

        For every character in seq, with probability 5% augmentation happens
        If augmentation happens then 4 different actions can happen with 25% probability each
        1) 25%: delete: char is deleted and sequence gets shorter by 1 char
        2) 25%: insert: char is inserted before or after this character and sequence gets longer by 1 char #TODO: Left/right insert?
        3) 25%: replace: char is replaced with random character from ['A', 'C', 'T', 'G']
        4) 12.5%: swap forward: character is swaped with neighbour to the right - skips next char so we don't get double modulation
        5) 12.5%: swap backward: character is swaped with neighbour to the left
        '''

        augmentation_prob = 0.05  # Probability of altering each character
        methods = ['delete', 'insert', 'replace', 'swap forward', 'swap backward']
        methods_prob = [0.25, 0.25, 0.25, 0.125, 0.125]
        
        seq = list(seq)  # Convert sequence to list for mutation
        seq_length = len(seq)

        if not seq: # Handle empty sequence case
            return random.choice(self.nucleotide_bases)
        
        # --- loop over chars in sequence ---
        idx = 0
        while idx < seq_length: 
            
            r = random.random()
            if r < augmentation_prob:
                
                # Select which method to use at this index. Adjust sequence length and idx to compensate for method
                method = random.choices(methods, weights=methods_prob, k=1)[0] 

                if method == 'delete':
                    seq = self._delete(seq, idx)
                    seq_length -= 1

                elif method == 'insert':
                    seq = self._insert(seq, idx)
                    idx += 1 
                    seq_length += 1

                elif method == 'replace':
                    seq = self._replace(seq, idx)
                    idx += 1  
                 
                elif method == 'swap forward':
                    if idx < (seq_length-1):
                        seq = self._swapForward(seq, idx)
                        idx += 2  
                    else:
                        seq = self._swapBackward(seq,idx)
                        idx +=1 

                elif method == 'swap backward':
                    if idx > 0:
                        seq = self._swapBackward(seq, idx)
                        idx += 1  
                    else:
                        seq = self._swapForward(seq,idx)
                        idx +=2 

            else:
                idx += 1 # Normal increment
    
        if len(seq)<1:
            seq = [random.choice(self.nucleotide_bases)]
        return ''.join(seq)
    
    def _noStrategy(self, seq: str) -> str:
        '''
        The trivial do-nothing-strategy. Returns input unmodified.
        '''
        return seq


    def _delete(self, seq: list[str], idx: int) -> list[str]:

        self._validate_index(seq, idx)
        return seq[:idx] + seq[idx + 1:]  # Return the sequence without the character at idx


    def _insert(self, seq: list[str], idx: int) -> list[str]:

        self._validate_index(seq, idx)
        
        insert_idx = random.choice([idx, idx + 1])  # Randomize if insert is before or after input-idx
        nucleotide = random.choice(self.nucleotide_bases)  # Insert a random nucleotide
        return seq[:insert_idx] + [nucleotide] + seq[insert_idx:]  # Insert the new nucleotide


    def _replace(self, seq: list[str], idx: int) -> list[str]:

        self._validate_index(seq, idx)
        
        nucleotide = random.choice(self.nucleotide_bases)  # Replace with a random nucleotide
        seq[idx] = nucleotide
        return seq


    def _swap(self, seq: list[str], idx: int) -> list[str]:
        '''
        Swap character at idx either with character at idx+1 or at idx-1
        '''
        self._validate_index(seq, idx)
        choices = [i for i in [idx - 1, idx + 1] if 0 <= i < len(seq)]  # Swap with valid neighbors

        if choices:
            swap_idx = random.choice(choices)
            seq[idx], seq[swap_idx] = seq[swap_idx], seq[idx]  # Swap the characters
        return seq
    
    def _swapForward(self, seq: list[str], idx: int) -> list[str]:
        '''
        Swap character att idx with character at idx+ 1
        '''
        swap_idx = idx+1

        self._validate_index(seq, idx)

        seq[idx], seq[swap_idx] = seq[swap_idx], seq[idx]  # Swap the characters
        return seq
    
    def _swapBackward(self, seq: list[str], idx: int) -> list[str]:
        '''
        Swap character att idx with character at idx-1
        '''
        swap_idx = idx-1

        self._validate_index(seq, idx)
        
        seq[idx], seq[swap_idx] = seq[swap_idx], seq[idx]  # Swap the characters
        return seq

    def _validate_index(self, seq: list, idx: int) -> None:
        if idx < 0 or idx >= len(seq):

            raise ValueError(f"Index {idx} is out of range for sequence of length {len(seq)}")
        


#%% Graveyard

    def _oldbaseStrategy(self, seq: str) -> str:
        '''
        For every character in seq, with probability 5% augmentation happens
        If augmentation happens then 4 different actions can happen with 25% probability each
        1) delete: char is deleted and sequence gets shorter by 1 char
        2) insert: char is inserted before or after this character and sequence gets longer by 1 char
        3) replace: char is replaced with random character from ['A', 'C', 'T', 'G']
        4) swap: character is swaped with neighbour, either forwards or backwards
        '''
        
        augmentation_prob = 0.05  # Probability of altering each character
        
        # methods and their probabilities
        methods = ['delete', 'insert', 'replace', 'swap']
        methods_prob = [0.25, 0.25, 0.25, 0.25]
        
        seq = list(seq)
        seq_length = len(seq)

        # Handle empty sequence case
        if not seq: 
            return random.choice(self.nucleotide_bases)
        

        # --- loop over chars in sequence ---
        idx = 0
        while idx < seq_length: 
            
            r = random.random()
            if r < augmentation_prob:
                
                # Select which method to use at this index. Adjust sequence length and idx to compensate for method
                method = random.choices(methods, weights=methods_prob, k=1)[0] 

                if method == 'delete':
                    seq = self._delete(seq, idx)
                    seq_length -= 1

                elif method == 'insert':
                    seq = self._insert(seq, idx)
                    idx += 1 
                    seq_length += 1

                elif method == 'replace':
                    seq = self._replace(seq, idx)
                    idx += 1  

                elif method == 'swap':
                    seq = self._swap(seq, idx)
                    idx += 1 
            else:
                # If no method implemented - move on
                idx += 1
        
        # If we by chance removed chars all the way down to length 0, randomize nucleotide
        if len(seq)<1:
            seq = [random.choice(self.nucleotide_bases)]

        return ''.join(seq)
