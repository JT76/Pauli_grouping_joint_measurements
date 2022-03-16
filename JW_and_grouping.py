import numpy as np
import scipy.optimize
from itertools import combinations, product, permutations
from functools import partial
from itertools import repeat
import multiprocessing
import itertools

def jordan_wigner(fermion_operator, n_orbs, dagger=True):
    sig_x = ['X']
    sig_y = ['Y']
    sig_z = ['Z']
    identity = ['I']
    sig_plus = ['X', 'Y']
    weights_plus = [1.0, 1.0j] 
    sig_minus = ['X', 'Y'] 
    weights_minus = [1.0, -1.0j]
    if fermion_operator==0:
        if dagger:
            mapped_operator = sig_minus
            weights = weights_minus
        else:
            mapped_operator = sig_plus
            weights = weights_plus
    else: 
        mapped_operator = sig_z 
    for i in range(1, n_orbs):
        if i < fermion_operator:
            mapped_operator = [i + sig_z[0] for i in mapped_operator]

        elif i == fermion_operator:
            if dagger:
                mapped_operator = [mapped_operator[0] + j for j in sig_minus]
                weights = weights_minus
            else:
                mapped_operator = [mapped_operator[0] + j for j in sig_plus]
                weights = weights_plus
        else:
            mapped_operator = [i + identity[0] for i in mapped_operator]
    return mapped_operator, weights      

def jordan_wigner_map(index_list, n_orbs):
    identity = 'I'
    mid_range = int(len(index_list)/2)
    paulis_res = identity
    num_qubits = n_orbs
    dagger = True
    ops_list = []
    weights = []
    for i in range(len(index_list)):
        if i >= mid_range:
            dagger = False
        new_ops_list, new_weights = jordan_wigner(index_list[i], n_orbs, dagger)
        if i == 0:
            ops_list = new_ops_list
            weights = new_weights
        else:
            ops_list, weights = multiply_letter_ops(ops_list, weights, new_ops_list, new_weights)
    operator = {'paulis': ops_list, 'weights': weights}
    operator = simplify_operator(operator)
    operator['fermion_idx'] = index_list
    return operator

def simplify_operator(operator):
    new_paulis = []
    new_weights = []
    shift = 0.0
    for i in range(len(operator['weights'])):
        if abs(operator['weights'][i]) >= 1e-10:
            keep_it = False
            for letter in operator['paulis'][i]:
                if letter != 'I':
                    keep_it = True
            if keep_it:
                if operator['paulis'][i] in new_paulis:
                    new_weights[new_paulis.index(operator['paulis'][i])] += operator['weights'][i]
                else:
                    new_paulis.append(operator['paulis'][i])
                    new_weights.append(operator['weights'][i])
            else:
                shift += operator['weights'][i]
    return {'paulis': new_paulis, 'weights': new_weights, 'shift': shift}


def multiply_letter_ops(ops_1, w_1, ops_2, w_2):
    new_ops_list = []
    new_weights_list = []
    for op_id_1 in range(len(ops_1)):
        for op_id_2 in range(len(ops_2)):
            new_ops = ''
            new_weight = w_1[op_id_1]*w_2[op_id_2]
            operator_1 = ops_1[op_id_1]
            operator_2 = ops_2[op_id_2]
            for index in range(len(operator_1)):
                intra_weight = 1.0
                if operator_1[index] == 'I':
                    new_letter = operator_2[index]
                elif operator_2[index] == 'I':
                    new_letter = operator_1[index]
                elif operator_1[index] == operator_2[index]:
                    new_letter = 'I'
                elif operator_1[index] == 'X':
                    if operator_2[index] == 'Y':
                       new_letter = 'Z'
                       intra_weight = 1.0j
                    else:
                       new_letter = 'Y'
                       intra_weight = -1.0j
                elif operator_1[index] == 'Y':
                    if operator_2[index] == 'X':
                       new_letter = 'Z'
                       intra_weight = -1.0j
                    else:
                       new_letter = 'X'
                       intra_weight = 1.0j
                else:
                    if operator_2[index] == 'X':
                       new_letter = 'Y'
                       intra_weight = 1.0j
                    else:
                       new_letter = 'X'
                       intra_weight = -1.0j
                new_ops += new_letter
                new_weight *= intra_weight
            new_ops_list.append(new_ops)
            new_weights_list.append(new_weight)
    return new_ops_list, new_weights_list

#######################JW example######################

print('- - - - Example for Jordan-Wigner - - - -')
print('Fermionic operators: a_dagger_1, a_dagger_3, a_1, a_3, on 4 spin orbitals')
print(jordan_wigner_map([1, 3, 1, 3], 4))


################## Grouping code ##########################

def index_ordering(edge_count):
    index_list_descending = []
    new_list = edge_count.copy()
    dummy_edge_count = edge_count.copy() 
    for i in range(len(edge_count)):
        max_new_list = max(new_list)
        index_list_descending.append(dummy_edge_count.index(max_new_list))
        dummy_edge_count[index_list_descending[-1]] = -1        
        new_list.pop(new_list.index(max_new_list))
    assert len(index_list_descending)==len(edge_count)
    return index_list_descending

def ops_commute(p_string_1, p_string_2):
    count = 0
    for index in range(len(p_string_1)):
        count += p_matrix_commute(p_string_1[index], p_string_2[index])
    return count % 2


def p_matrix_commute(p_mat_1, p_mat_2):
    if p_mat_1 == 'I':
        return 0
    elif p_mat_2 == 'I':
        return 0
    elif p_mat_1 == p_mat_2:
        return 0
    else: 
        return 1

def commute_matrix_mem_eff(ops_list):
    edge_count = []
    with multiprocessing.Pool() as p:
        for i in range(len(ops_list)):
            res = p.starmap(ops_commute, zip(repeat(ops_list[i]), ops_list))
            op_edge_count = np.sum(res)
            edge_count.append(op_edge_count)
    return edge_count

def color_set_mem_eff(ops_list):
    edge_count = commute_matrix_mem_eff(ops_list)
    list_ops_descending = index_ordering(edge_count)
    color_list = np.zeros([len(ops_list)])
    for i in range(len(list_ops_descending)):
        ops_index = list_ops_descending[i]
        neighbours_colors = []
        for j in range(len(edge_count)):
            if ops_commute(ops_list[ops_index], ops_list[j]) == 1:
                neighbours_colors.append(int(color_list[j]))
        found_color = True
        color_index = 1
        
        while found_color:
            if color_index in neighbours_colors:
                color_index += 1
            else:
                found_color = False
        color_list[ops_index] = color_index
    num_cliques = int(max(color_list))
    return color_list, num_cliques

def map_colors(ops_list, color_list):
    group_map = {}
    for i in range(len(color_list)):
        group_map[str(color_list[i])] = []
    for i in range(len(color_list)):
        group_map[str(color_list[i])].append(ops_list[i])
    return group_map


################## Grouping example ##########################
print('- - - - Example for grouping - - - -')
print('From operator list in attachement')
print('Grouping ongoing, can take minutes...')
ops_list = np.load('operator_list.npy')
color_list, num_cliques = color_set_mem_eff(ops_list)
group_map = map_colors(ops_list, color_list)
print('Operators mapped.....')
print('Number of groups found through LDCF: ', num_cliques)

##############################################################


################### Joint Measurement code ####################################



def create_stabiliser_matrix(pauli_set):
    string_length = len(pauli_set[0])
    stab_matrix = np.zeros((string_length*2, len(pauli_set)))
    for index_string, string in zip(range(len(pauli_set)), pauli_set):
        for index_letter in range(len(string)):
            index_letter_shift = index_letter + string_length
            if string[index_letter] == 'X':
                stab_matrix[index_letter_shift][index_string] = 1
            if string[index_letter] == 'Y':
                stab_matrix[index_letter][index_string] = 1
                stab_matrix[index_letter_shift][index_string] = 1
            if string[index_letter] == 'Z':
                stab_matrix[index_letter][index_string] = 1
    return stab_matrix

def fully_rankify(stab_matrix, n):
    gate_set = []
    m = int(stab_matrix.shape[0]/2)
    X_matrix = stab_matrix[m:, :n]
    X_matrix_rank = np.linalg.matrix_rank(X_matrix)
    num_changes = n - X_matrix_rank
    no_redundancy_X = scipy.optimize._remove_redundancy._remove_redundancy(X_matrix, np.zeros_like(X_matrix[:, 0]))[0]
    redun = []
    for row in X_matrix:
        add_row = True
        for row_no_redun in no_redundancy_X:
            if np.array_equal(row, row_no_redun): 
                row_no_redun *= row_no_redun*2
                add_row = False
        if add_row:
           redun.append(row)
    for row_index in range(len(X_matrix)):
        for row_redun_index in range(len(redun)):
            if np.array_equal(X_matrix[row_index], redun[row_redun_index]):
                stab_matrix[[row_index, row_index + m]] = stab_matrix[[row_index + m, row_index]]
                gate_set.append({'H': row_index})  
                redun[row_redun_index] = 2*redun[row_redun_index] 
    X_matrix = stab_matrix[m:, :n]
    X_matrix_rank = np.linalg.matrix_rank(X_matrix)              
    assert X_matrix_rank == n
    return stab_matrix, gate_set


def apply_CNOT(stab_matrix, control_id, target_id, n):
    m = int(stab_matrix.shape[0]/2)
    stab_matrix[control_id] = np.mod(stab_matrix[control_id] + stab_matrix[target_id], 2)
    stab_matrix[target_id + m] = np.mod(stab_matrix[control_id + m] + stab_matrix[target_id + m], 2)
    return stab_matrix

def row_order(array):
    row_sums = array.sum(axis = 1)
    unique_rows = np.unique(row_sums)
    row_ordering = []
    for value in unique_rows: 
        for i in range(n):
            if row_sums[i] == value:
                row_ordering.append(i)
    return(row_ordering)

def find_CNOTs(stab_matrix, n):
    m = int(stab_matrix.shape[0]/2)
    X_matrix = stab_matrix[m:, :n]
    marker_list = []
    gate_set = []
    for column_index in range(n):
        for row_index in range(m):
            if X_matrix[row_index][column_index] == 1:
                if row_index not in marker_list:
                    marker_list.append(row_index)
                    for new_row_index in range(m):
                        if X_matrix[new_row_index][column_index] == 1:
                            if new_row_index != marker_list[-1]:
                                gate_set.append({'CNOT': [marker_list[-1], new_row_index]})
                                stab_matrix = apply_CNOT(stab_matrix, marker_list[-1], new_row_index, n)
                                
                                X_matrix = stab_matrix[m:, :n]
                    break
    return stab_matrix, gate_set


                            
def apply_SWAP(stab_matrix, id_1, id_2, n):
    m = int(stab_matrix.shape[0]/2)
    stab_matrix[[id_1, id_2]] = stab_matrix[[id_2, id_1]]
    stab_matrix[[id_1 + m, id_2 + m]] = stab_matrix[[id_2 + m, id_1 + m]]
    return stab_matrix

def find_SWAPs(stab_matrix, n):
    m = int(stab_matrix.shape[0]/2)
    X_matrix = stab_matrix[m:, :n]
    gate_set = []
    for column_index in range(n):
        for row_index in range(m):
            if column_index == row_index:
                if X_matrix[row_index][column_index] == 0: 
                    for new_row_index in range(m):
                        if X_matrix[new_row_index][column_index] ==1: 
                            gate_set.append({'SWAP': [row_index, new_row_index]})
                            stab_matrix = apply_SWAP(stab_matrix, row_index, new_row_index, n) 
                break
    return stab_matrix, gate_set

def clean_Z_matrix(stab_matrix, n):
    m = int(stab_matrix.shape[0]/2)
    Z_matrix = stab_matrix[:m, :n]
    gate_set = []
    for column_index in range(n):
        for row_index in range(m):
            if column_index == row_index: 
                if Z_matrix[row_index][column_index] == 1: 
                    gate_set.append({'S': row_index})
                    stab_matrix[row_index][column_index] = 0
            else:
                if Z_matrix[row_index][column_index] == 1:
                    if row_index < n:
                        if Z_matrix[column_index][row_index] == 1:
                            gate_set.append({'CZ': [row_index, column_index]})
                            stab_matrix[row_index][column_index] = 0
                            stab_matrix[column_index][row_index] = 0
    for index in range(n):
        stab_matrix[[index, index + m]] = stab_matrix[[index + m, index]]
        gate_set.append({'H': index}) 
    return stab_matrix, gate_set                     
 
def multiply_operators(operator_1, operator_2):
    new_weight = 1.0
    new_ops =''
    for index in range(len(operator_1)):
        intra_weight = 1.0
        if operator_1[index] == 'I':
            new_letter = operator_2[index]
        elif operator_2[index] == 'I':
            new_letter = operator_1[index]
        elif operator_1[index] == operator_2[index]:
            new_letter = 'I'
        elif operator_1[index] == 'X':
            if operator_2[index] == 'Y':
                new_letter = 'Z'
                intra_weight = 1.0j
            else:
                new_letter = 'Y'
                intra_weight = -1.0j
        elif operator_1[index] == 'Y':
            if operator_2[index] == 'X':
                new_letter = 'Z'
                intra_weight = -1.0j
            else:
                new_letter = 'X'
                intra_weight = 1.0j
        else:
            if operator_2[index] == 'X':
                new_letter = 'Y'
                intra_weight = 1.0j
            else:
                new_letter = 'X'
                intra_weight = -1.0j
        new_ops += new_letter
        new_weight *= intra_weight
    return new_ops, new_weight


def find_basis(pauli_string_set):
    basis_length = len(pauli_string_set[0])
    basis = [pauli_string_set[0], pauli_string_set[1]]
    constructs = {}
    for index in range(2, len(pauli_string_set)):
        construct_map, phase_weight = construct_pauli(pauli_string_set[index], basis)
        if construct_map[-1] == len(basis):
            basis.append(pauli_string_set[index])
        else:
            constructs[pauli_string_set[index]] = {'Map' : construct_map, 'Phase' : phase_weight}
    return basis, constructs

def update_index_list(index_list, basis, update_index):
    more_index = True
    max_range = len(basis) - update_index
    if update_index > len(index_list):
        more_index = False
        return index_list, more_index
    index_list[-update_index] += 1
    if index_list[-update_index] > max_range:
        index_list, more_index = update_index_list(index_list, basis, update_index + 1)
        if update_index + 1 > len(index_list):
            more_index = False
            return index_list, more_index
        index_list[-update_index] = index_list[-(update_index + 1)] + 1 
        return index_list, more_index   
    else: 
        return index_list, more_index

def construct_pauli(pauli_string, basis):
    search_flag = True
    rank = 2
    while rank <= len(basis) and search_flag:
        index_list = list(range(rank))
        more_index = True
        while more_index:
            dummy = pauli_string
            weight = 1.
            for index in index_list:
                dummy, new_weight = multiply_operators(dummy, basis[index])
                weight = weight*new_weight
            if dummy == 'I'*len(basis[0]): 
                search_flag = False
                dummy_save = index_list
                break
            else:
                index_list, more_index = update_index_list(index_list, basis, update_index = 1)
        rank += 1
    if not search_flag:
        return dummy_save, weight    
    else:
        return [len(basis)], 1.0
    
def convert_to_matrix(pauli_string):
    Id = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    if pauli_string[0]=='I':
        matrix = Id
    if pauli_string[0]=='X':
        matrix = X
    if pauli_string[0]=='Y':
        matrix = Y
    if pauli_string[0]=='Z':
        matrix = Z
    for i in range(1, len(pauli_string)):
        if pauli_string[i]=='I':
            matrix = np.kron(matrix, Id)
        if pauli_string[i]=='X':
            matrix = np.kron(matrix, X)    
        if pauli_string[i]=='Y':
            matrix = np.kron(matrix, Y)
        if pauli_string[i]=='Z':
            matrix = np.kron(matrix, Z)
    return matrix

def produce_circuit(pauli_string_set):
    pauli_set_restricted, construct = find_basis(pauli_string_set)
    stab_matrix = create_stabiliser_matrix(pauli_set_restricted)
    n = len(pauli_set_restricted)
    m = int(stab_matrix.shape[0]/2)
    gate_set = []
    stab_matrix, new_gates = fully_rankify(stab_matrix, n)
    gate_set.append(new_gates)
    stab_matrix, new_gates = find_CNOTs(stab_matrix, n)
    X_matrix = stab_matrix[m:, :n]
    X_rank = np.linalg.matrix_rank(X_matrix)
    if X_rank != n: 
        stab_matrix, new_gates = fully_rankify(stab_matrix, n)
        gate_set.append(new_gates)
        stab_matrix, new_gates = find_CNOTs(stab_matrix, n)
        gate_set.append(new_gates)

    stab_matrix, new_gates = find_SWAPs(stab_matrix, n)
    gate_set.append(new_gates)
    stab_matrix, new_gates = clean_Z_matrix(stab_matrix, n)
    gate_set.append(new_gates)
    return pauli_set_restricted, gate_set


####################### Joint Measurement code #####################################
print('- - - - Joint Measurement example - - - -')
print('Start from a commutative group and produce a circuit for joint measurement')
commute_group = group_map['1.0']
print('Example commutative group: ', commute_group)
basis, gate_set = produce_circuit(commute_group)
print('Basis for measurements: ', basis)
print('Circuit for joint measurement: ', gate_set)


