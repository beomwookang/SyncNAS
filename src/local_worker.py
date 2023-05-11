import os, json

class LocalWorker:
    def __init__(self, server_config:dict):
        self.state = 'IDLE'
        self.result = None

        self.host = server_config['host']
        self.port = server_config['port']
        self.server_name = server_config['name']

        self.workload_path = 'sample_model_'+self.server_name+'.json'
        self.result_path = 'result_'+self.server_name

        #local <-> remote_host communication
        self.receive_path = 'rhost_to_local_'+self.server_name
        self.send_path = 'local_to_rhost_'+self.server_name

        #go to pending if result exists
        if os.path.exists(self.receive_path):
            with open(self.result_path, 'r') as f:
                self.result = json.loads(f.read())
            self.state = 'PENDING'

        #go to working if server working
        self.working_path = 'working_'+self.server_name
        if os.path.exists(self.working_path):
            os.remove(self.working_path)
            self.state = 'WORKING_PREV'


    def wait_previous(self):
        while not os.path.exists(self.receive_path):
            pass
        with open(self.result_path, 'r') as f:
            self.result = json.loads(f.read())
        self.state = 'PENDING'


    def send_and_wait(self, sample):
        self.state = 'WORKING'

        #save sample to uniform format
        self.sample = sample
        sample.export_to_config(self.workload_path)

        #send uniform format to server
        os.system("scp -P "+self.port+" "+self.workload_path+" user@"+self.host+":/home/user/")
        os.remove(self.workload_path)

        #send signal
        os.system("touch "+self.send_path)
        os.system("scp -P "+self.port+" "+self.send_path+" user@"+self.host+":/home/user/")
        os.remove(self.send_path)

        #wait until result received
        while not os.path.exists(self.receive_path):
            pass
        
        #save result and wait until new workload
        with open(self.result_path, 'r') as f:
            self.result = json.loads(f.read())
            #self.result = float(f.read())

        #set state as pending
        self.state = 'PENDING'


    def wrap_up_and_wait(self):
        #gather and remove record
        self.result = None

        #remove files
        os.remove(self.result_path)
        os.remove(self.receive_path)

        #set state to IDLE
        self.state = 'IDLE'
