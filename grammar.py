import h5py
import numpy as np



class TilingGrammar():

    #NUM_CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '%']
    #SPECIAL_CHARS = ['(', ')']
    DIGITS =  "0123456789"
    NUM_DELIMITER = "%"
    BRANCH_START = "("
    BRANCH_END = ")"

    def __init__(self, word_set):
        self.charset = []
        self.neighbor_counts = []
        self.neighbor_types = []
        for w in word_set:
            self.parse_valid_word(w, 0)
        print('neighbor types: ')
        for pair in self.neighbor_types:
            print('(' + pair[0] + ', ' + pair[1] + ') ')
        print('neighbor counts: ')
        for pair in self.neighbor_counts:
            print('(' + self.charset[pair[0]] + ', ' + str(pair[1]) + ') ')


    def _parse_number(self, word, last_non_number, start_char_id = 0):
        next_char_id = start_char_id 
        #skip until end of number
        while(NUM_CHARS.find(word[next_char_id]) != -1):
            next_char_id += 1
        #TODO: record a pair of char and number
        return next_char_id

    def _add_neighbor_pair(self, word, next_char_id, char_id):
        charset_id = self.charset.index(word[char_id])
        nbr_charset_id = len(self.charset)
        if(word[next_char_id] in self.charset ):
            nbr_charset_id = self.charset.index(word[next_char_id])
        self.neighbor_types.append([charset_id, nbr_charset_id])

    def parse_valid_word(self, word, char_id = 0):
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
            next_char_id = self._parse_number(word, word[char_id], next_char_id)           
            if(word[next_char_id] == self.NUM_DELIMITER):
                next_char_id += 1
        
        while(next_char_id < len(word) and word[next_char_id] == self.BRANCH_START):
            num_neighbors += 1
            next_char_id += 1
            if(next_char_id >= len(word)):
                raise ValueError('Unexpected end of word')
            if(next_char_id == self.BRANCH_END):
                raise ValueError('Unexpected () at ' + str(next_char_id) + ' in ' + word)
            if(next_char_id == self.BRANCH_START):
                raise ValueError('Unexpected (( at ' + str(next_char_id) + ' in ' + word)
            if(self.DIGITS.find(word[next_char_id]) != -1 or word[next_char_id] == self.NUM_DELIMITER):
                raise ValueError('Unexpected number character succeding ( at ' + str(next_char_id) + ' in ' + word) 
          
            if([word[char_id], word[next_char_id]] not in self.neighbor_types ):
                self.neighbor_types.append([word[char_id], word[next_char_id]])

            while(word[next_char_id] != self.BRANCH_END):
                next_char_id = self.parse_valid_word(word, next_char_id)
            if(next_char_id >= len(word)):
                raise ValueError('Missing ) at the end of a subtree at ' + str(next_char_id) + ' in ' + word)
            else:
                next_char_id += 1
        
        if(next_char_id < len(word) and word[next_char_id] != self.BRANCH_END):
            num_neighbors += 1
            if([word[char_id], word[next_char_id]] not in self.neighbor_types ):
                self.neighbor_types.append([word[char_id], word[next_char_id]])
            self.parse_valid_word(word, next_char_id)

        if([charset_id, num_neighbors] not in self.neighbor_counts ):               
            self.neighbor_counts.append([charset_id, num_neighbors])
        
        return next_char_id

            
    def load(self, filename):
        h5f = h5py.File(filename, 'r')
        self.charset = h5f['charset'][:]
        self.neighbor_counts = h5f['neighbor_counts'][:]
        self.neighbor_types = h5f['neighbor_types'][:]
        h5f.close()

    def store(self, filename):
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset('charset', data = self.charset)
        h5f.create_dataset('neighbor_counts', data = self.neighbor_counts)
        h5f.create_dataset('neighbor_types', data = self.neighbor_types)
        h5f.close()


