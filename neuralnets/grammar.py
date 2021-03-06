from __future__ import print_function #pylint bug workaround
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
        self.categories_prefix = []
        
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

    def convert_to_tree_grammar(self):
        for [char, num_neighbors] in self.neighbor_counts:
            for i in range(1,num_neighbors):
                if([char, i] not in self.neighbor_counts ):
                    self.neighbor_counts.append([char, i])
        self.neighbor_counts.sort()

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
    
    def _check_word(self, word, char_id=0, branch_start_count=0, cycle_ids=[], cycle_vals=[], edges=[]):
        #TODO: this should not be necessary!
        if char_id == 0 and len(cycle_ids) + len(cycle_vals) > 0:
            while len(cycle_ids) > 0:
                cycle_ids.pop()
            while len(cycle_vals) > 0:
                cycle_vals.pop()

        if char_id >= len(word):
            return True, char_id, edges
        char = word[char_id]

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
                return False, next_char_id, []
            if(next_char_id == self.BRANCH_END):
                return False, next_char_id, []
            if(next_char_id == self.BRANCH_START):
                return False, next_char_id, []
            if(self.DIGITS.find(word[next_char_id]) != -1 or word[next_char_id] == self.NUM_DELIMITER):
                return False, next_char_id, [] 
          
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

    def check_word(self, word, verbose=False):
        if(word.count(self.BRANCH_START) != word.count(self.BRANCH_END)):
            if verbose:
                print("Number of ( does not match number of ):")
                print(word)
            return False
        for i in range(len(self.DIGITS)):
            if(word.count(self.DIGITS[i]) % 2 != 0):
                if verbose:
                    print("Uneven number of ",self.DIGITS[i],"s in the word:")
                    print(word)
                return False
        result, dummy_id, dummy_edge_list = self._check_word(word)
        if result == False and verbose:
            print("Error at char id ", dummy_id, " in: ")
            if dummy_id < len(word):
                print(word[0:dummy_id], " >",word[dummy_id],"< ", word[dummy_id + 1:])
            else:
                print(word[:-1], " >",word[-1],"< ")
        while len(dummy_edge_list) > 0:
            dummy_edge_list.pop()
        #if not result:
        #    print("Error at char id: " + str(dummy_id) + " " + word)
        return result

    def similar_words(self, str1, str2):
        for i in range(len(self.DIGITS)):
            if(str1.count(self.DIGITS[i]) != str2.count(self.DIGITS[i])):
                return False #different number of cycles
        for i in range(1, len(self.charset)):
            if(str1.count(self.charset[i]) != str2.count(self.charset[i])):
                return False #different node type histogram
        #equal number of nodes of each type
        #equal number of graph edges

        dummy_type = str(unichr(127))
        padded_node_types_1 = []
        for i, char in enumerate(str1):
            if char in self.charset:
                padded_node_types_1.append(char)
            else:
                padded_node_types_1.append(dummy_type) #dummy node need max id
        padded_node_types_1.append(dummy_type)#ensure at least one occurrence

        node_types_list_1 = self.smiles_to_edges(str1, padded_node_types_1)

        #patch edge type list by removing multiple cycle edge entries
        for i in range(1, len(str1)):
            prev = str1[i - 1]
            char = str1[i]
            if char in self.DIGITS and prev in self.DIGITS:
                node_types_list_1[i] = [dummy_type, dummy_type]

        padded_node_types_2 = []
        for i, char in enumerate(str2):
            if char in self.charset:
                padded_node_types_2.append(char)
            else:
                padded_node_types_2.append(dummy_type) #dummy node need max id
        padded_node_types_2.append(dummy_type)#ensure at least one occurrence

        node_types_list_2 = self.smiles_to_edges(str2, padded_node_types_2)

        #patch edge type list by removing multiple cycle edge entries
        for i in range(1, len(str2)):
            prev = str2[i - 1]
            char = str2[i]
            if char in self.DIGITS and prev in self.DIGITS:
                node_types_list_2[i] = [dummy_type, dummy_type]

        for edge in self.neighbor_types:
            flipped = [edge[1], edge[0]]
            count_1 = node_types_list_1.count(edge) + node_types_list_1.count(flipped)
            count_2 = node_types_list_2.count(edge) + node_types_list_2.count(flipped)
            if count_1 != count_2:
                return False #different number of edges of a certain type
        return True

    def word_similarity(self, str1, str2):
        retval = 0.0
        num_nodes = 0.0
        for i in range(1, len(self.charset)):
            num_nodes += str1.count(self.charset[i])
            num_nodes += str2.count(self.charset[i])

        for i in range(1, len(self.charset)):
            retval += abs(str1.count(self.charset[i]) - str2.count(self.charset[i])) / num_nodes

        return retval

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

        node_types, neighbors = self.one_hot_to_graph(vec)

        result_str = ""
        for node_id in non_epty_node_ids:
            result_str += (str(node_id) + ": " + node_types[node_id] + "  " + str(neighbors[node_id]) + "\n")
        return result_str

    def set_categories_prefix(self, categories_prefix):
        if len(categories_prefix) != len(self.neighbor_types) + 1:
            print("Number of edge categories does not match number of edge types")
        else:
            for item in categories_prefix:
                self.categories_prefix.append(item)
    
    def smiles_to_edges(self, word, padded_node_ids):
        dummy_node_id = max(padded_node_ids)

        edge_list = [[dummy_node_id, dummy_node_id]]
        node_id_stack = []
        cycle_vals = []
        cycle_ids = []
        last_char = word[0]
        last_node_id = padded_node_ids[0]
        digit_skip_counter = 0
        for char_id in range(1,len(word)):
            if digit_skip_counter > 0:
                digit_skip_counter -= 1
                continue
            if word[char_id] in self.charset:
                if last_char in self.charset:
                    edge_list.append([padded_node_ids[char_id], last_node_id])
                elif self.DIGITS.find(last_char) != -1:
                    edge_list.append([padded_node_ids[char_id], last_node_id])
                elif last_char != self.BRANCH_END:
                    edge_list.append([padded_node_ids[char_id], node_id_stack[len(node_id_stack) - 1]])
                elif last_char == self.BRANCH_END:
                    #last subtree of parent node, pop the stack top
                    edge_list.append([padded_node_ids[char_id], node_id_stack[len(node_id_stack) - 1]])
                    node_id_stack.pop()

                last_node_id = padded_node_ids[char_id]
                last_char = word[char_id]
            elif word[char_id] == self.BRANCH_START and last_char != self.BRANCH_END:
                edge_list.append([dummy_node_id, dummy_node_id])
                node_id_stack.append(last_node_id)
                last_char = word[char_id]
            elif word[char_id] == self.BRANCH_START and last_char == self.BRANCH_END: #do nothing
                edge_list.append([dummy_node_id, dummy_node_id])
                last_char = word[char_id]
            elif word[char_id] == self.BRANCH_END:
                edge_list.append([dummy_node_id, dummy_node_id])
                last_char = word[char_id]
            elif self.DIGITS.find(word[char_id]) != -1:
                last_digit_id = char_id
                while last_digit_id < len(word) and self.DIGITS.find(word[last_digit_id]) != -1:
                    last_digit_id += 1
                    digit_skip_counter += 1
                digit_skip_counter -= 1
                cycle_edge_id = int(word[char_id: last_digit_id])
                if cycle_edge_id in cycle_ids:
                    neighbor_node_id = cycle_vals[cycle_ids.index(cycle_edge_id)]
                    last_digit_id = char_id
                    while last_digit_id < len(word) and self.DIGITS.find(word[last_digit_id]) != -1:
                        last_digit_id += 1
                        edge_list.append([neighbor_node_id, last_node_id])
                else:
                    cycle_ids.append(cycle_edge_id)
                    cycle_vals.append(last_node_id)
                    last_digit_id = char_id
                    while last_digit_id < len(word) and self.DIGITS.find(word[last_digit_id]) != -1:
                        last_digit_id += 1
                        edge_list.append([dummy_node_id, dummy_node_id])
                if last_digit_id == len(word):
                    break
                last_char = word[last_digit_id - 1]
            elif word[char_id] == self.NUM_DELIMITER:
                edge_list.append([dummy_node_id, dummy_node_id])
                last_char = word[char_id]

        return edge_list

    def smiles_to_categories_bounds(self, word, invert_edge_direction=False):
        dummy_type = str(unichr(127))
        padded_node_types = []
        for i, char in enumerate(word):
            if char in self.charset:
                padded_node_types.append(char)
            else:
                padded_node_types.append(dummy_type) #dummy node need max id
        padded_node_types.append(dummy_type)#ensure at least one occurrence

        node_types_list = self.smiles_to_edges(word, padded_node_types)
        bounds_list = []

        if  not self.categories_prefix:
            return bounds_list

        num_categories = self.categories_prefix[-1]
        for type_pair in node_types_list:
            if type_pair == [dummy_type, dummy_type]:
                bounds_list.append([num_categories, num_categories + 1])
            elif invert_edge_direction:
                inverted_pair = [type_pair[1], type_pair[0]]
                i_type_id = self.neighbor_types.index(inverted_pair)
                bounds_list.append([self.categories_prefix[i_type_id], self.categories_prefix[i_type_id + 1]])
            else:
                type_id = self.neighbor_types.index(type_pair)
                bounds_list.append([self.categories_prefix[type_id], self.categories_prefix[type_id + 1]])
        return bounds_list

    def smiles_to_one_hot(self, word, charset):
        num_chars = len(charset)
        one_hot = np.zeros(dtype='float32', shape=(len(word), num_chars))
        for j in range(len(word)):
            one_hot[j][np.where(charset == word[j])] = 1
        return one_hot

    def smiles_to_mask(self, word, max_length=120):
        bounds_list = self.smiles_to_categories_bounds(word)
        num_categories = self.categories_prefix[-1] + 1
        masks = np.zeros(dtype='float32', shape=(max_length, num_categories))
        for j, bounds in enumerate(bounds_list):
            for k in range(bounds[0], bounds[1]):
                masks[j][k] = 1
        for j in range(len(bounds_list), max_length):
            masks[j][num_categories - 1] = 1
        return masks

    def load(self, filename):
        h5f = h5py.File(filename, "r")
        charset_array = h5f["charset"][:]
        self.charset = charset_array.tolist()
        nbr_counts_array = h5f["neighbor_counts"][:]
        self.neighbor_counts = [[pair[0], int(pair[1])] for pair in nbr_counts_array]
        nbr_type_array = h5f["neighbor_types"][:]
        self.neighbor_types = nbr_type_array.tolist()
        categories_prefix_array = h5f["categories_prefix"][:]
        self.categories_prefix = categories_prefix_array.tolist()

        h5f.close()

    def store(self, filename):
        h5f = h5py.File(filename, "w")
        h5f.create_dataset("charset", data=self.charset)
        h5f.create_dataset("neighbor_counts", data=self.neighbor_counts)
        h5f.create_dataset("neighbor_types", data=self.neighbor_types)
        h5f.create_dataset("categories_prefix", data=self.categories_prefix)
        h5f.close()


