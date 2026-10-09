import suite2p
from suite2p.run_s2p import logger_setup
from mouse_imaging import *

def main(mouse, date, session='session_1', ops_name='default_ops'):
    path = sess.define_path(mouse, date, session)
    md = sess.get_metadata(path)
    ops = getattr(options, ops_name)()
    tif_files = list(path['twophoton_dir'].glob('*.tif'))
    assert len(tif_files) > 0

    db = {
        'data_path': [str(path['twophoton_dir'])], # Directory where your input files are located
        'save_path0': str(path['suite2p_dir'].parent), # suite2p writes to save_path0/save_folder, so outputs land directly in suite2p_dir
        'save_folder': path['suite2p_dir'].name,
        # 'file_list': tif_files, # Specify files you'd like to specifically use in the data_path
        'input_format': 'tif',
        'nplanes': md['nslices'] + md['nflyback'], # each tiff has these many planes in sequence, including flyback frames
        'nchannels': md['nchannels'], # each tiff has these many channels per plane
        'keep_movie_raw': False,
        'batch_size': 200, # we will decrease the batch_size in case low RAM on computer
        'functional_chan': 1,
        'ignore_flyback': list(range(md['nslices'], md['nslices'] + md['nflyback'])), # 0-based plane indices of flyback frames
    }

    settings = ops['suite2p_settings']
    settings['fs'] = md['volume_rate'] # sampling rate of recording, determines binning for cell detection

    logger_setup(path['suite2p_dir'])
    suite2p.run_s2p(settings=settings, db=db)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess session.')
    parser.add_argument('--mouse', required=True, type=str)
    parser.add_argument('--date', required=True, type=str)
    parser.add_argument('--session', required=False, type=str, default='session_1')
    parser.add_argument('--ops', required=False, type=str, default='default_ops')
    args = parser.parse_args()
    main(args.mouse, args.date, args.session, ops_name=args.ops)
