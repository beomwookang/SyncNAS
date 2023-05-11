import numpy as np
import random
import os, json, pickle
import copy
from itertools import combinations, product
from local_worker import LocalWorker
import threading
import time


class MCTSNode():
    def __init__(self, parent=None, action=None, action_category=None, model='mobilenet_v2', scale=1.0, target='firefly', cpu_cores='b2', gpu_freq='800M', resume=None):
        self.parent = parent
        self.action_category = action_category
        self.action = action

        if self.parent != None:
            self.is_root = False
            self.state = copy.deepcopy(parent.state)
            self.state[self.action_category] = action
            self.cpu_cores = self.parent.cpu_cores
            self.gpu_freq = self.parent.gpu_freq
        else:
            self.is_root = True
            self.state = self.init_state()       #for root node
            self.cpu_cores = cpu_cores
            self.gpu_freq = gpu_freq


        self.results = []
        self.reward = 0.0
        self.visit = 0

        self.next_action_category = self.get_next_action_category()
        if self.next_action_category == 'sync_determined':  #model sampled
            self.is_terminal = True
            self.get_sync_info()
        else:
            self.is_terminal = False
            self.unexpanded_actions = self.get_next_actions_list()
            self.children = []

        self.model = model
        self.scale = scale
        self.target = target

        self.name = self.get_sample_name()
        self.searched = False

        #load previous result if resume
        if resume != None:
            assert os.path.exists(resume)
            print("previoius search loaded!")
            with open(resume, 'r') as f:
                self.searched_models = json.loads(f.read())
        else:
            self.searched_models = dict()

        #True if no mbconv budget is available
        self.invalid = False


    ### initialization-related methods 
    def init_state(self):
        a_category=['sc',
                    'sp_pick1', 'sp_pick2', 'sp_pick3',
                    'sp_pick4', 'sp_pick5', 'sp_pick6',
                    'st_pick1', 'st_pick2', 'st_pick3', 'st_pick4',
                    'st_pick5', 'st_pick6', 'st_pick7']

        state = dict()
        state['sync'] = ['n','n','n','n','n','n','n']
        for category in a_category:
            state[category] = None

        return state


    def print_state(self):
        for k in self.state:
            print(k, ": ", self.state[k])


    def print_model(self):
        assert hasattr(self, "sample_model")
        for k in self.sample_model:
            print(k, ": ", self.sample_model[k])


    def save_result(self, path='searched_samples'):
        with open(path, 'w') as f:
            f.write(json.dumps(self.searched_models))


    def search(self, max_trial):
        with open('Servers.config') as f:
            server_pool = json.loads(f.read())

        available_servers = []

        #check available servers from server pool
        for server in server_pool:
            f_path = "available_" + server
            if os.path.exists(f_path):
                available_servers.append(server_pool[server])
                os.remove(f_path)

        #add available servers to workers list
        workers=[]
        threads=dict()
        samples=dict()
        for server_config in available_servers:
            worker = LocalWorker(server_config)
            threads[worker.server_name] = None
            samples[worker.server_name] = None           
            workers.append(worker)

        #resume previous record if exists
        if len(self.searched_models) > 0:
            print("search resume, re-tracking saved result...")
            for sample_name in self.searched_models:
                sample = self.get_leaf_by_name(sample_name)
                result = self.searched_models[sample_name]
                sample.backpropagate(result)
            print("search resumed!")
        
        trial=0
        while trial < max_trial:
            #check if servers to exclude exist, remove from server_pool, available_servers
            if os.path.exists('delete_servers'):
                with open('delete_servers', 'r') as f:
                    del_servs = json.loads(f.read())
                for d_serv in del_servs:
                    for worker in workers:
                        if worker.server_name == d_serv:
                            workers.remove(worker) 
                    del server_pool[d_serv]
                os.remove('delete_servers')

            #check if new server is ready, check 'Servers.config' and update server_pool
            if os.path.exists('new_server'):
                with open('Servers.config') as f:
                    server_pool = json.loads(f.read())
                os.remove('new_server')

            #check new workers(servers) -> once every round for current workers
            for server in server_pool:
                f_path = "available_" + server
                if os.path.exists(f_path):
                    #add new server to workers list 
                    worker = LocalWorker(server_pool[server])
                    threads[worker.server_name] = None
                    samples[worker.server_name] = None           
                    workers.append(worker)
                    os.remove(f_path)

            #check worker state
            for worker in workers:
                if worker.state == 'IDLE':
                    print("                                        ", end="\r")
                    print("(",worker.server_name,"): ", worker.state, end="\r")
                    if trial < max_trial:
                        sample = self.tree_policy()
                        print("\n[trial ", trial,"]\tSync Case Sampled!")
                        if sample.is_root:
                            print("Search Done!")
                            return
                        sample.determine_channel()
                        if sample.invalid:
                             print("[trial ", trial,"]\tNo MBConv Budget is Available, Skipping...")
                             sample.backpropagate(0)
                             self.searched_models[sample.name] = 0
                             self.save_result()
                             if self.is_descendants_searched():
                                 print("entire space has been searched")
                                 break
                        else:
                            print("[trial ", trial,"]\tChannel Sizes Determined!")
                            threads[worker.server_name] = threading.Thread(target=worker.send_and_wait, 
                                                                           args=(sample,))
                            print("[trial ", trial,"]\tSample [",sample.name,"] Sent!  -------------->  Server ["
                                    +worker.server_name+"]\n")
                            threads[worker.server_name].start()
                            samples[worker.server_name] = sample
                        trial += 1

                elif worker.state == 'WORKING':
                    print("                                        ", end="\r")
                    print("(",worker.server_name,"): ", worker.state, end="\r")
                    pass

                elif worker.state == 'WORKING_PREV':
                    print("                                        ", end="\r")
                    print("(",worker.server_name,"): ", worker.state, end="\r")
                    threads[worker.server_name] = threading.Thread(target=worker.wait_previous)
                    threads[worker.server_name].start()

                elif worker.state == 'PENDING':
                    print("                                        ", end="\r")
                    print("(",worker.server_name,"): ", worker.state, end="\r")
                    result = worker.result
                    threads[worker.server_name] = threading.Thread(target=worker.wrap_up_and_wait)
                    threads[worker.server_name].start()
                    sample_name = list(result)[0]
                    sample_acc = result[sample_name]

                    if samples[worker.server_name] != None:
                        sample = samples[worker.server_name]
                    else:
                        sample = self.get_leaf_by_name(sample_name)

                    sample.backpropagate(sample_acc)
                    self.searched_models.update(result)
                    self.save_result()
                    
                    #break if search done
                    if self.is_descendants_searched():
                        print("entire space has been searched")
                        break

                time.sleep(1)   


    ### get leaf node by sample name (for resume)
    def get_leaf_by_name(self, sample_name):
        current_node = self
        action_sequence = self.name_to_action_sequence(sample_name)
        action_index = 0
        while not current_node.is_terminal:
            if current_node.is_descendants_searched():
                current_node = current_node.parent
                action_index -= 1
            else:
                action = action_sequence[action_index]
                current_node = current_node.pick_with_action(action)
                action_index += 1
        return current_node


    def name_to_action_sequence(self, sample_name):
        configs = sample_name.split('_')
        sync_count = [int(configs[0][-1])]
        sync_position = [int(i) for i in configs[1][:-1]]
        sync_type = list(configs[2])
        return sync_count + sync_position + sync_type

        
    def pick_with_action(self, action):
        if action in self.unexpanded_actions:
            self.unexpanded_actions.pop(self.unexpanded_actions.index(action))
            child_node = MCTSNode(parent=self, action=action, action_category=self.next_action_category, 
                                  model=self.model, scale=self.scale)
            self.children.append(child_node)
            return child_node
        else:
            for child in self.children:
                if child.action == action:
                    return child
                    

    def tree_policy(self):
        current_node = self
        while not current_node.is_terminal:
            if not current_node.is_fully_expanded():
                current_node = current_node.expand()
            else:
                if current_node.is_descendants_searched():
                    current_node = current_node.parent
                    if current_node == None:
                        return self
                else:
                    current_node = current_node.best_child()    #fully expanded; exploitation
        current_node.searched = True
        return current_node


    def expand(self):
        action = self.unexpanded_actions.pop(random.randint(0,len(self.unexpanded_actions)-1))
        child_node = MCTSNode(parent=self, action=action, action_category=self.next_action_category, 
                              model=self.model, scale=self.scale)
        self.children.append(child_node)
        return child_node
        

    def best_child(self):
        f_exps = []
        unsearched_children = []
        for child in self.children:
            if not child.is_descendants_searched():
                unsearched_children.append(child)

        for _c in unsearched_children:
            if _c.visit == 0:
                #for parallel search, pick random
                f_exp = random.randint(1,len(unsearched_children))
            else:
                f_exploit = _c.reward / _c.visit
                f_explore = 2*1.0*np.sqrt((2*np.log(self.visit) / _c.visit))
                f_exp = f_exploit + f_explore
            f_exps.append(f_exp)
        return unsearched_children[np.argmax(f_exps)]


    def is_fully_expanded(self):
        return len(self.unexpanded_actions) == 0


    def is_children_searched(self):
        if not self.is_fully_expanded():
            return False
        else:
            return all(child.searched for child in self.children)


    def is_descendants_searched(self):
        if self.is_terminal:
            return self.searched
        else:
            if not self.is_fully_expanded():
                return False
            else:
                return all(child.is_descendants_searched() for child in self.children)

    
    def backpropagate(self, result):
        if self.is_terminal:
            self.searched = True
        else:
            self.searched = self.is_descendants_searched()
        self.visit += 1
        self.results.append(result)
        self.reward = np.sum(self.results)
        if not self.is_root:
            self.parent.backpropagate(result)


    ### search space-related methods
    def get_next_action_category(self):
        ac = self.action_category
        if ac == None:
            nac = 'sc'
        elif ac == 'sc':
            if self.state[ac] != 1:      #more than 1 sync
                nac = 'sp_pick1'
            else:
                nac = 'st_pick1'
        elif ac.startswith('sp_pick'):
            pick_count = int(ac[-1])
            if pick_count < self.state['sc'] - 1:
                pick_count += 1
                nac = 'sp_pick' + str(pick_count)
            else:
                nac = 'st_pick1'
        elif ac.startswith('st_pick'):
            pick_count = int(ac[-1])
            if pick_count < self.state['sc']:
                pick_count += 1
                nac = 'st_pick' + str(pick_count)
            else:
                nac = 'sync_determined'
        
        return nac


    def get_next_actions_list(self):
        nac = self.next_action_category
        if nac == 'sc':
            actions_list = [1,2,3,4,5,6,7]
        elif nac.startswith('sp_pick'):
            pick_count = int(nac[-1])
            max_sp = 6 - self.state['sc'] + pick_count
            if pick_count == 1:                         #first sp pick
                min_sp = 0
            else:
                min_sp = self.state[self.action_category] + 1
            actions_list = [i for i in range(min_sp, max_sp+1)]
        elif nac.startswith('st_pick'):
            actions_list = ['a', 'c']
        
        return actions_list


    ### Model Configuration & Train Methods
    def get_sync_info(self):
        sync_positions = []
        sync_types = []
        for key in self.state:
            if key.startswith('sp_pick'):
                if self.state[key] != None:
                    sync_positions.append(self.state[key])
        sync_positions.append(6)    #last stage

        stage_count = 1
        for position in sync_positions:
            sync_type = self.state['st_pick'+str(stage_count)]
            self.state['sync'][position] = sync_type
            sync_types.append(sync_type)
            stage_count += 1


    def get_sample_name(self):
        name = 'sync'
        if self.state['sc'] != None:
            name += str(self.state['sc'])
            if self.state['sc'] == 1:
                if self.state['st_pick1'] != None:
                    name += '_6_' 
                    name += self.state['st_pick1']
                else:
                    return name
        else:
            name = 'root'
            return name

        if self.state['sp_pick1'] != None:
            name += '_'
            for i in range(1,7):
                if self.state['sp_pick'+str(i)] != None:
                    name += str(self.state['sp_pick'+str(i)])
                else:
                    break
        else:
            return name

        if self.state['st_pick1'] != None:
            name += '6_'
            for i in range(1,8):
                if self.state['st_pick'+str(i)] != None:
                    name += self.state['st_pick'+str(i)]
                else:
                    return name
            return name
        else:
            return name

    def export_to_config(self, filename):
        assert hasattr(self, 'sample_model')
        with open(filename, 'w') as f:
            f.write(json.dumps(self.sample_model))
       

    def overlap_model(self, overlapped, new_sample):
        for stage in range(7):
            for branch in ['0', '1']:
                _overlapped = overlapped[str(stage)][branch]
                _new_sample = new_sample[str(stage)][branch]
                for block in range(len(_overlapped)):
                    if _overlapped[block][4] < _new_sample[block][4]:
                        _overlapped[block][4] = _new_sample[block][4]
                    if _overlapped[block][5] < _new_sample[block][5]:
                        _overlapped[block][5] = _new_sample[block][5]
            overlapped['sync'][stage] = 'a'
        return overlapped

        
