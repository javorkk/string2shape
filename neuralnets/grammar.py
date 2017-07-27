import h5py
import numpy as np

EPS = 0.0001

class TilingGrammar():

    DIGITS =  "0123456789"
    NUM_DELIMITER = "%"
    BRANCH_START = "("
    BRANCH_END = ")"
    
    def __init__(self, word_set):
        self.charset = [" "]
        self.neighbor_counts = []
        self.neighbor_types = []
        
        if(len(word_set) <= 0):
            return

        for w in word_set:
            self.parse_valid_word(w, 0)

        self.charset.sort()
        self.neighbor_types.sort()
        self.neighbor_counts.sort()

        print("node types: ")
        print(self.charset)
        print("neighbor types: ")
        for c in self.charset:
            print([pair for pair in self.neighbor_types if pair[0] == c])
        print("neighbor counts: ")
        for c in self.charset:
            print([pair for pair in self.neighbor_counts if pair[0] == c])


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
                if neighbor != last_non_number:
                    self.neighbor_types.append([neighbor, last_non_number])
        else:
            cycle_ids.append(cycle_edge_id)
            cycle_vals.append(last_non_number)
        return next_char_id

    def parse_valid_word(self, word, char_id = 0, cycle_ids = [], cycle_vals = []):
        #TODO: this should not be necessary!
        if char_id == 0 and len(cycle_ids) + len(cycle_vals) > 0:
            while len(cycle_ids) > 0:
                cycle_ids.pop()
            while len(cycle_vals) > 0:
                cycle_vals.pop()

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
                if word[char_id] != word[next_char_id]:
                    self.neighbor_types.append([word[next_char_id], word[char_id]])

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
                if word[char_id] != word[next_char_id]:
                    self.neighbor_types.append([word[next_char_id], word[char_id]])
            next_char_id = self.parse_valid_word(word, next_char_id, cycle_ids, cycle_vals)

        if([char, num_neighbors] not in self.neighbor_counts ):               
            self.neighbor_counts.append([char, num_neighbors])
        
        return next_char_id

    def _check_number(self, word, last_non_number_id, start_char_id = 0, cycle_ids = [], cycle_vals = [], edges = []):
            next_char_id = start_char_id 
            #skip until end of number
            while(next_char_id < len(word) and self.DIGITS.find(word[next_char_id]) != -1):
                next_char_id += 1
            cycle_edge_id = int(word[start_char_id: next_char_id])
            if(cycle_edge_id in cycle_ids):
                neighbor_id = cycle_vals[cycle_ids.index(cycle_edge_id)]
                if neighbor_id == last_non_number_id:
                    return True, next_char_id, edges #second sighting of the same character
                if([word[last_non_number_id], word[neighbor_id]] not in self.neighbor_types ):
                    return False, next_char_id, []
                else:
                    edges.append([last_non_number_id, neighbor_id])
                    edges.append([neighbor_id, last_non_number_id])
            else:
                cycle_ids.append(cycle_edge_id)
                cycle_vals.append(last_non_number_id)
            return True, next_char_id, edges
    
    def _check_word(self, word, char_id = 0, branch_start_count = 0, cycle_ids = [], cycle_vals = [], edges = []):
        #TODO: this should not be necessary!
        if char_id == 0 and len(cycle_ids) + len(cycle_vals) > 0:
            while len(cycle_ids) > 0:
                cycle_ids.pop()
            while len(cycle_vals) > 0:
                cycle_vals.pop()

        if char_id >= len(word):
            return True, char_id, edges
        char = word[char_id]

        charset_id = len(self.charset)
        if(char in self.charset ):
            charset_id = self.charset.index(char)
        else:
            return False, char_id, []
            

        num_neighbors = 0
        if(char_id > 0):
            num_neighbors += 1
        next_char_id = char_id + 1

        while(next_char_id < len(word) and self.DIGITS.find(word[next_char_id]) != -1):
            num_neighbors += 1
            valid, next_char_id, edges = self._check_number(word, char_id, next_char_id, cycle_ids, cycle_vals, edges)
            if not valid:
                return False, next_char_id, []           
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
                return False, next_char_id, []
            if next_char_id < len(word):
                edges.append([char_id, next_char_id])
                edges.append([next_char_id, char_id])

            while(next_char_id < len(word) and word[next_char_id] != self.BRANCH_END):
                valid, next_char_id, edges = self._check_word(word, next_char_id, branch_start_count + 1, cycle_ids, cycle_vals, edges)
                if not valid:
                    return False, next_char_id, []
            if(next_char_id >= len(word)):
                return False, next_char_id, []
            else:
                next_char_id += 1
        
        if(next_char_id < len(word) and word[next_char_id] != self.BRANCH_END):
            num_neighbors += 1
            if([word[char_id], word[next_char_id]] not in self.neighbor_types ):
                return False, next_char_id, []
            if next_char_id < len(word):
                edges.append([char_id, next_char_id])
                edges.append([next_char_id, char_id])
            valid, next_char_id, edges = self._check_word(word, next_char_id, branch_start_count, cycle_ids, cycle_vals, edges)
            if next_char_id < len(word) and not valid:
                return False, next_char_id, []            

        if(next_char_id < len(word) and branch_start_count == 0 and word[next_char_id] == self.BRANCH_END):
            return False, next_char_id, []

        if([char, num_neighbors] not in self.neighbor_counts ):
            return False, next_char_id, []
        
        return True, next_char_id, edges

    def check_word(self, word):
        if(word.count(self.BRANCH_START) != word.count(self.BRANCH_END)):
            return False;
        for i in range(len(self.DIGITS)):
            if(word.count(self.DIGITS[i]) % 2 != 0):
                return False
        result, dummy_id, dummy_edge_list = self._check_word(word)
        while len(dummy_edge_list) > 0:
            dummy_edge_list.pop()
        #if not result:
        #    print("Error at char id: " + str(dummy_id) + " " + word)
        return result

    def max_degree(self):
        #return max([pair[1] for pair in self.neighbor_counts])
        return reduce(lambda x, y: max(x, y[1]), self.neighbor_counts, 0)

    #def encode_to_one_hot(self, word, max_length):
        #result = np.zeros((max_length, len(self.charset) + self.max_degree()))
    def encode_to_one_hot(self, word, max_length = 0):
        node_count = len([char for char in word if char in self.charset])
        result = np.zeros((node_count, len(self.charset) + self.max_degree()), dtype=np.float)
        if(len(word) == 0):
            return result
        if(word.count(self.BRANCH_START) != word.count(self.BRANCH_END)):
            return result
        for i in range(len(self.DIGITS)):
            if(word.count(self.DIGITS[i]) % 2 != 0):
                return result
        valid, dymmy_id, edge_list = self._check_word(word)
        if not valid:
            while len(edge_list) > 0:
                edge_list.pop()
            return result
        edge_list.sort()
        edge_set = [edge_list[0]] + [edge_list[pair_id] for pair_id in range(1, len(edge_list)) \
                                              if edge_list[pair_id][0] != edge_list[pair_id - 1][0] or \
                                                 edge_list[pair_id][1] != edge_list[pair_id - 1][1] ]
        while len(edge_list) > 0:
            edge_list.pop()
        
        #print("word: " + word)
        #print("edge set: ")
        #print(edge_set)
        
        #node types
        node_ids = np.zeros(len(word), dtype=np.int)
        last_node_id = 0
        for char_id in range(len(word)):
            if word[char_id] in self.charset:
                node_ids[char_id] = last_node_id
                result[last_node_id][self.charset.index(word[char_id])] = 1
                last_node_id += 1
            else:
                node_ids[char_id] = -1
        
        #connectivity
        for node_id in range(last_node_id):
            my_edges = [[node_ids[pair[0]], node_ids[pair[1]]] for pair in edge_set if node_ids[pair[0]] == node_id and node_ids[pair[1]] != -1]
            if len(my_edges) > self.max_degree():
                raise ValueError("More neighbors that allowed by the grammar! Got " + str(len(my_edges)) + " expected up to " + str(self.max_degree()) + "\n word is " + word + "\n node id " + str(node_id))                
            for edge_id in range(len(my_edges)):
                #difference + direction between the node index and its neigbor's
                delta = my_edges[edge_id][1] - my_edges[edge_id][0]
                if max_length > 0:
                    #delta = delta / (2.0 * max_length) + 0.5 #values in [0,1]
                    delta = delta / (1.0 * max_length) #values in [-1,1]
                result[node_id][len(self.charset) + edge_id] = delta
            for edge_id in range(len(my_edges), self.max_degree()):
                #result[node_id][len(self.charset) + edge_id] = 0.5 #values in [0,1]
                result[node_id][len(self.charset) + edge_id] = 0.0 #values in [-1,1]


        #print("one_hot_encoding :")    
        #print(result)
        if max_length > node_count:
            #result = np.lib.pad(result, ((0,max_length - node_count), (0,0)), mode='constant', constant_values=0.5) #values in [0,1]
            result = np.lib.pad(result, ((0,max_length - node_count), (0,0)), mode='constant', constant_values=0.0) #values in [-1,1]
        for node_id in range(node_count, max_length):
            result[node_id][0] = 1.0

        while len(edge_set) > 0:
            edge_set.pop()

        return result       

    def check_one_hot(self, vec):
        num_dims = len(self.charset) + self.max_degree()
        non_epty_node_ids = [node_id for node_id in range(vec.shape[0]) if np.amax(vec[node_id]) > EPS or np.amin(vec[node_id]) < -EPS ]
        num_vecs = len(non_epty_node_ids)
        if vec.shape[1] != num_dims:
            return False
        
        #node types
        node_types = []
        for node_id in range(vec.shape[0]):
            if node_id not in non_epty_node_ids:
                node_types.append(" ")
                continue
            #node_type_id = len(self.charset);
            node_type_id = vec[node_id][0 : len(self.charset)].argmax()
            if abs(vec[node_id][node_type_id]) < EPS:
                return False
            else:
                node_types.append(self.charset[node_type_id])

        #connectivity
        for node_id in non_epty_node_ids:
            if node_types[node_id] == " ":
                continue

            num_neighbors = 0
            for neighbor_id_d in vec[node_id][len(self.charset): num_dims]:
                #if abs(neighbor_id_d - 0.5) < EPS: #values in [0,1]
                if abs(neighbor_id_d) < EPS: #values in [-1,1]
                    continue
                num_neighbors += 1
                #nbr_delta = int(round(neighbor_id_d * 2 * vec.shape[0] - vec.shape[0])) #values in [0,1]
                nbr_delta = int(round(neighbor_id_d * vec.shape[0])) #values in [-1,1]
                neighbor_id = (node_id + nbr_delta) % num_vecs
                valid_neighbor = False
                for nbr_nbr_d in vec[neighbor_id][len(self.charset): num_dims]:
                    #if abs(nbr_nbr_d - 0.5) < EPS: #values in [0,1]
                    if abs(nbr_nbr_d) < EPS: #values in [-1,1]
                        continue
                    #nbr_nbr_delta = int(round(nbr_nbr_d * 2 * vec.shape[0] - vec.shape[0])) #values in [0,1]
                    nbr_nbr_delta = int(round(nbr_nbr_d * vec.shape[0])) #values in [-1,1]
                    nbr_nbr = (neighbor_id + nbr_nbr_delta) % num_vecs
                    if nbr_nbr == node_id:
                        for allowed_type in [pair[1] for pair in self.neighbor_types if pair[0] == node_types[node_id] ]:
                            if node_types[neighbor_id] == allowed_type:
                                valid_neighbor = True
                                break
                        if valid_neighbor:
                            break

                if not  valid_neighbor:
                    return False

            if num_neighbors not in [pair[1] for pair in self.neighbor_counts if pair[0] == node_types[node_id] ]:
                return False
        
        return True
    
    def one_hot_to_graph(self, vec):
        non_epty_node_ids = [node_id for node_id in range(vec.shape[0]) if np.amax(vec[node_id]) > EPS or np.amin(vec[node_id]) < -EPS ]
        num_dims = len(self.charset) + self.max_degree()
        if vec.shape[1] != num_dims:
            raise ValueError("Wrong one_hot vector shape! Got " + str(vec.shape) + " expected (, " + str(num_dims) +")")

        #node types
        node_types = []
        for node_id in range(vec.shape[0]):
            if node_id not in non_epty_node_ids:
                node_types.append(" ")
                continue
            node_type_id = vec[node_id][0 : len(self.charset)].argmax()
            if abs(vec[node_id][node_type_id]) < EPS:
                return False
            else:
                node_types.append(self.charset[node_type_id])
        
        #neighbors
        neighbors = []
        for node_id in non_epty_node_ids:
            #neighbors.append([ int(round(x * 2 * vec.shape[0] - vec.shape[0])) for x in vec[node_id][len(self.charset): num_dims] ]) #values in [0,1]
            neighbors.append([ int(round(x * vec.shape[0])) for x in vec[node_id][len(self.charset): num_dims] ]) #values in [-1,1]
            #neighbors.append(vec[node_id][len(self.charset): num_dims])#values in [-word_len, word_len]
        return node_types, neighbors

    def print_one_hot(self, vec):
        non_epty_node_ids = [node_id for node_id in range(vec.shape[0]) if np.amax(vec[node_id]) > EPS or np.amin(vec[node_id]) < -EPS ]
        num_vecs = len(non_epty_node_ids)

        node_types, neighbors = self.one_hot_to_graph(vec)

        result_str = ""
        for node_id in non_epty_node_ids:
            result_str += (str(node_id) + ": " + node_types[node_id] + "  " + str(neighbors[node_id]) + "\n")
        return result_str

    def load(self, filename):
        h5f = h5py.File(filename, "r")
        charset_array = h5f["charset"][:]
        self.charset = charset_array.tolist()
        nbr_counts_array = h5f["neighbor_counts"][:]
        self.neighbor_counts = [[pair[0], int(pair[1])] for pair in nbr_counts_array]
        nbr_type_array = h5f["neighbor_types"][:]
        self.neighbor_types = nbr_type_array.tolist()
        h5f.close()

    def store(self, filename):
        h5f = h5py.File(filename, "w")
        h5f.create_dataset("charset", data = self.charset)
        h5f.create_dataset("neighbor_counts", data = self.neighbor_counts)
        h5f.create_dataset("neighbor_types", data = self.neighbor_types)
        h5f.close()


