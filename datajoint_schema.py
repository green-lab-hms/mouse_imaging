import datajoint as dj
import os
import anndata
import numpy as np
from session import as sess
import tmaze

local_root = '/n/data2/hms/neurobio/harvey/jonathan/data/imaging'

def config_dj():	
	dj.config['database.host'] = 'mysql.orchestra'
	dj.config['database.user'] = 'jg319'
	dj.config['database.password'] = 'TitMifsAv9'
	dj.config['stores'] = {
	    'external': dict(protocol='file',
	                location=os.path.join(local_root, 'djblobs'))
	}
	dj.config['cache'] = os.path.join(local_root, 'djcache')
	dj.config['enable_python_native_blobs'] = True

dj.conn()
schema = dj.schema('harvey_datajoint_jg319') # Schema is created by research computing
config_dj()

@schema
class Allele(dj.Lookup):
    definition = """
    # Allele
    allele_id : varchar(40)
    ---
    description = null : varchar(50)
    cell_type : varchar(20)
    color : varchar(20)
    function : varchar(20)
    """
Allele.insert([
    {'allele_id': 'Sst-Cre;Ai14', 'cell_type': 'Sst', 'color': 'red', 'function': 'label'}], skip_duplicates=True)

@schema
class Virus(dj.Lookup):
    definition = """
    # Virus
    virus_id : varchar(20)
    ---
    description : varchar(50)
    cell_type : varchar(20)
    color : varchar(20)
    function : varchar(20)
    """
Virus.insert([
    {'virus_id': 'JG43', 'description': 'syn-jGCaMP7f', 'cell_type': 'neuron', 'color': 'green', 'function': 'activity'},
    {'virus_id': 'JG119', 'description': 'Sst44-NLS.mTagBFP2', 'cell_type': 'Sst44', 'color': 'blue', 'function': 'label'}], skip_duplicates=True)

@schema
class Mouse(dj.Manual):
    definition = """
    # Experimental animals
    mouse_id : varchar(10)
    ---
    dob = null : date
    sex = 'unknown' : enum('M', 'F', 'unknown')
    """
    class Genotype(dj.Part):
        definition = """
        # Genotype of animal
        -> master
        -> Allele
        """
    
    class Injection(dj.Part):
        definition = """
        # Virus that was injected
        -> master
        -> Virus
        """
Mouse.insert([
    {'mouse_id': 'JG166', 'dob': '2019-12-28', 'sex': 'M'},
    {'mouse_id': 'JG167', 'dob': '2019-12-28', 'sex': 'M'},
    {'mouse_id': 'JG177', 'dob': '2020-05-07', 'sex': 'M'},
    {'mouse_id': 'JG178', 'dob': '2020-05-07', 'sex': 'M'},
    {'mouse_id': 'JG179', 'dob': '2020-05-07', 'sex': 'M'},
    {'mouse_id': 'JG180', 'dob': '2020-05-07', 'sex': 'M'},
    {'mouse_id': 'JG181', 'dob': '2020-05-07', 'sex': 'M'},], skip_duplicates=True)

Mouse.Genotype.insert([
	{'mouse_id': 'JG166', 'allele_id': 'Sst-Cre;Ai14'},
    {'mouse_id': 'JG167', 'allele_id': 'Sst-Cre;Ai14'}], skip_duplicates=True)

Mouse.Injection.insert([
    {'mouse_id': 'JG166', 'virus_id': 'JG43'},
    {'mouse_id': 'JG166', 'virus_id': 'JG119'},    
    {'mouse_id': 'JG167', 'virus_id': 'JG43'},
    {'mouse_id': 'JG167', 'virus_id': 'JG119'},
    {'mouse_id': 'JG177', 'virus_id': 'JG43'},
    {'mouse_id': 'JG177', 'virus_id': 'JG119'},
    {'mouse_id': 'JG178', 'virus_id': 'JG43'},
    {'mouse_id': 'JG178', 'virus_id': 'JG119'},
    {'mouse_id': 'JG179', 'virus_id': 'JG43'},
    {'mouse_id': 'JG179', 'virus_id': 'JG119'},
    {'mouse_id': 'JG180', 'virus_id': 'JG43'},
    {'mouse_id': 'JG180', 'virus_id': 'JG119'},
    {'mouse_id': 'JG181', 'virus_id': 'JG43'},
    {'mouse_id': 'JG181', 'virus_id': 'JG119'},
    ], skip_duplicates=True)

@schema
class Session(dj.Manual):
    definition = """
    # Experiment session
    -> Mouse
    session_date : date
    ---
    vr_id : varchar(50)
    adata_h5ad : varchar(255)
    region='' : varchar(20)
    """

def create_session(mouse, date, session='session_1'):
    path = sess.define_path(mouse=mouse, date=date, session=session)
    metadata = sess.get_metadata(path['raw_image1.tif'])
    key = {
    'mouse_id' : mouse,
    'session_date': '20%s-%s-%s' %(date[:2], date[2:4], date[4:6]),
    'vr_env': sess.maze_id(path['virmen.mat']),
    'adata_h5ad': path['adata.h5ad'],
    'region': metadata['region'],
    }
    return key
    
Session.insert([
    create_session('JG166', '200909',),
    create_session('JG167', '200909',)], skip_duplicates=True)

@schema
class Activity(dj.Imported):
    definition = """
    -> Session
    ---
    time : blob@external
    """
    class Neuron(dj.Part):
        definition = """
        -> master
        neuron_id : int
        ---
        activity : blob@external
        """
        
    def make(self, key):
        adata_file = (Session & key).fetch1('adata_h5ad')
        adata = anndata.read_h5ad(adata_file)
        
        time = np.array(adata.obs['t'])
        self.insert1(dict(**key, time=time), skip_duplicates=True)
        for icell, cell_activity in enumerate(adata.X.T):
            key['neuron_id'] = icell
            key['activity'] = cell_activity
            self.Activity.insert1(key, skip_duplicates=True)

Activity.populate()

@schema
class VR(dj.Imported):
    definition = """
    -> Session
    ---
    time : blob@external
    dx : blob@external
    dy : blob@external
    dh : blob@external
    x : blob@external
    y : blob@external
    h : blob@external
    inITI : blob@external
    reward : blob@external
    lick : blob@external
    world : blob@external
    trial : blob@external

    """
    
    def make(self, key):
        adata_file = (Session & key).fetch1('adata_h5ad')
        adata = anndata.read_h5ad(adata_file)
        time = np.array(adata.obs['t'])
        obs = adata.obs.to_records()
        self.insert1(dict(**key, time=time, obs=obs), skip_duplicates=True)

VR.populate()

class DynamicNeuronComputation(dj.Computed):
    definition = """
    -> Neuron.Activity

    """

