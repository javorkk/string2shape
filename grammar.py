import h5py
import numpy as np



class TilingGrammar():

    DIGITS =  "0123456789"
    NUM_DELIMITER = "%"
    BRANCH_START = "("
    BRANCH_END = ")"

    def __init__(self, word_set):
        self.charset = []
        self.neighbor_counts = []
        self.neighbor_types = []
        
        if(len(word_set) <= 0):
            return

        for w in word_set:
            self.parse_valid_word(w, 0)
        print("node types: ")
        for c in self.charset:
            print(c)
        print("neighbor types: ")
        for pair in self.neighbor_types:
            print("(" + pair[0] + ", " + pair[1] + ") ")
        print("neighbor counts: ")
        for pair in self.neighbor_counts:
            print("(" + self.charset[pair[0]] + ", " + str(pair[1]) + ") ")

    def _parse_number(self, word, last_non_number, start_char_id = 0, cycle_ids = [], cycle_vals = []):
        next_char_id = start_char_id 
        #skip until end of number
        while(next_char_id < len(word) and self.DIGITS.find(word[next_char_id]) != -1):
            next_char_id += 1
        cycle_edge_id = int(word[start_char_id: next_char_id])
        if(cycle_edge_id in cycle_ids):
            neighbor = cycle_vals[cycle_ids.index(cycle_edge_id)]
            if([last_non_number, neighbor] not in self.neighbor_types ):
                self.neighbor_types.append([last_non_number, neighbor])
        else:
            cycle_ids.append(cycle_edge_id)
            cycle_vals.append(last_non_number)
        return next_char_id

    def parse_valid_word(self, word, char_id = 0, cycle_ids = [], cycle_vals = []):
        if char_id >= len(word):
            return char_id
        char = word[char_id]

        charset_id = len(self.charset)
        if(char in self.charset ):
            charset_id = self.charset.index(char)
        else:
            self.charset.append(char)
            

        num_neighbors = 0
        if(char_id > 0):
            num_neighbors += 1
        next_char_id = char_id + 1

        while(next_char_id < len(word) and self.DIGITS.find(word[next_char_id]) != -1):
            num_neighbors += 1
            next_char_id = self._parse_number(word, word[char_id], next_char_id, cycle_ids, cycle_vals)           
            if(next_char_id < len(word) and word[next_char_id] == self.NUM_DELIMITER):
                next_char_id += 1
        
        while(next_char_id < len(word) and word[next_char_id] == self.BRANCH_START):
            num_neighbors += 1
            next_char_id += 1
            if(next_char_id >= len(word)):
                raise ValueError("Unexpected end of word")
            if(next_char_id == self.BRANCH_END):
                raise ValueError("Unexpected () at " + str(next_char_id) + " in " + word)
            if(next_char_id == self.BRANCH_START):
                raise ValueError("Unexpected (( at " + str(next_char_id) + " in " + word)
            if(self.DIGITS.find(word[next_char_id]) != -1 or word[next_char_id] == self.NUM_DELIMITER):
                raise ValueError("Unexpected number character succeding ( at " + str(next_char_id) + " in " + word) 
          
            if([word[char_id], word[next_char_id]] not in self.neighbor_types ):
                self.neighbor_types.append([word[char_id], word[next_char_id]])

            while(next_char_id < len(word) and word[next_char_id] != self.BRANCH_END):
                next_char_id = self.parse_valid_word(word, next_char_id, cycle_ids, cycle_vals)
            if(next_char_id >= len(word)):
                raise ValueError("Missing ) at the end of a subtree at " + str(next_char_id) + " in " + word)
            else:
                next_char_id += 1
        
        if(next_char_id < len(word) and word[next_char_id] != self.BRANCH_END):
            num_neighbors += 1
            if([word[char_id], word[next_char_id]] not in self.neighbor_types ):
                self.neighbor_types.append([word[char_id], word[next_char_id]])
            next_char_id = self.parse_valid_word(word, next_char_id, cycle_ids, cycle_vals)

        if([charset_id, num_neighbors] not in self.neighbor_counts ):               
            self.neighbor_counts.append([charset_id, num_neighbors])
        
        return next_char_id

    def _check_number(self, word, last_non_number, start_char_id = 0, cycle_ids = [], cycle_vals = []):
            next_char_id = start_char_id 
            #skip until end of number
            while(next_char_id < len(word) and self.DIGITS.find(word[next_char_id]) != -1):
                next_char_id += 1
            cycle_edge_id = int(word[start_char_id: next_char_id])
            if(cycle_edge_id in cycle_ids):
                neighbor = cycle_vals[cycle_ids.index(cycle_edge_id)]
                if([last_non_number, neighbor] not in self.neighbor_types ):
                    return False, next_char_id
            else:
                cycle_ids.append(cycle_edge_id)
                cycle_vals.append(last_non_number)
            return True, next_char_id
    
    def _check_word(self, word, char_id = 0, branch_start_count = 0, cycle_ids = [], cycle_vals = []):
        if char_id >= len(word):
            return True, char_id
        char = word[char_id]

        charset_id = len(self.charset)
        if(char in self.charset ):
            charset_id = self.charset.index(char)
        else:
            return False, char_id
            

        num_neighbors = 0
        if(char_id > 0):
            num_neighbors += 1
        next_char_id = char_id + 1

        while(next_char_id < len(word) and self.DIGITS.find(word[next_char_id]) != -1):
            num_neighbors += 1
            valid, next_char_id = self._check_number(word, word[char_id], next_char_id, cycle_ids, cycle_vals)
            if(valid == False):
                return False, next_char_id           
            if(next_char_id < len(word) and word[next_char_id] == self.NUM_DELIMITER):
                next_char_id += 1

        while(next_char_id < len(word) and word[next_char_id] == self.BRANCH_START):
            num_neighbors += 1
            next_char_id += 1
            if(next_char_id >= len(word)):
                return False, next_char_id
            if(next_char_id == self.BRANCH_END):
                return False, next_char_id
            if(next_char_id == self.BRANCH_START):
                return False, next_char_id
            if(self.DIGITS.find(word[next_char_id]) != -1 or word[next_char_id] == self.NUM_DELIMITER):
                return False, next_char_id 
          
            if([word[char_id], word[next_char_id]] not in self.neighbor_types ):
                return False, next_char_id

            while(next_char_id < len(word) and word[next_char_id] != self.BRANCH_END):
                valid, next_char_id = self._check_word(word, next_char_id, branch_start_count + 1, cycle_ids, cycle_vals)
                if(valid == False):
                    return False, next_char_id
            if(next_char_id >= len(word)):
                return False, next_char_id
            else:
                next_char_id += 1
        
        if(next_char_id < len(word) and word[next_char_id] != self.BRANCH_END):
            num_neighbors += 1
            if([word[char_id], word[next_char_id]] not in self.neighbor_types ):
                return False, next_char_id
            valid, next_char_id = self._check_word(word, next_char_id, branch_start_count, cycle_ids, cycle_vals)
            if(valid == False):
                return False, next_char_id

        if(next_char_id < len(word) and branch_start_count == 0 and word[next_char_id] == self.BRANCH_END):
            return False, next_char_id

        if([charset_id, num_neighbors] not in self.neighbor_counts ):               
            return False, next_char_id
        
        return True, next_char_id

    def check_word(self, word):
        if(word.count(self.BRANCH_START) != word.count(self.BRANCH_END)):
            return False;
        for i in range(len(self.DIGITS)):
            if(word.count(self.DIGITS[i]) % 2 != 0):
                return False
        result, dymmy = self._check_word(word)
        return result

    def load(self, filename):
        h5f = h5py.File(filename, "r")
        charset_array = h5f["charset"][:]
        self.charset = charset_array.tolist()
        nbr_counts_array = h5f["neighbor_counts"][:]
        self.neighbor_counts = nbr_counts_array.tolist()
        nbr_type_array = h5f["neighbor_types"][:]
        self.neighbor_types = nbr_type_array.tolist()
        h5f.close()

    def store(self, filename):
        h5f = h5py.File(filename, "w")
        h5f.create_dataset("charset", data = self.charset)
        h5f.create_dataset("neighbor_counts", data = self.neighbor_counts)
        h5f.create_dataset("neighbor_types", data = self.neighbor_types)
        h5f.close()


