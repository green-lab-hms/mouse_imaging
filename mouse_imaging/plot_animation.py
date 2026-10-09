import mouse_imaging.session as sess
import mouse_imaging.plot_jg as pl

import scipy.io
import scipy.ndimage
import scipy.interpolate
import os
import h5py
import numpy as np
import subprocess

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.font_manager
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['font.size'] = 12
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

def sbatch_plot_animation(**kwargs):
    shell_script = """
    sbatch ~/code/mouse_imaging/mouse_imaging/plot_animation.slurm {mouse} {date} {session} {trial} {path_out} {fps}
    """.format(**kwargs)
    print(shell_script)
    out = subprocess.call(shell_script, shell=True)

def plot_title_movie_segment(title, path_out, length_s=4, title_start_s=1, title_on_s=2, fps=30, test=False):
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    for spine in ['left', 'right', 'bottom', 'top']:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    if test:
        ax.text(10, 10, title, ha='center', fontsize=30)

    iterations = length_s * fps
    def update_img(i):
        t_mov = i / fps
        if (i)%10==0:
            print(f'Plotting frame {i+1} out of {iterations}.')
        elif i+1==iterations:
            print(f'Plotting frame {i+1} out of {iterations}.')
            print('Done plotting frames.')
        
        ax.cla()
        if (t_mov >= title_start_s) & (t_mov < title_start_s+title_on_s):
            for spine in ['left', 'right', 'bottom', 'top']:
                ax.spines[spine].set_visible(False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            ax.text(10, 10, title, ha='center', fontsize=30)

    #legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, iterations, interval=1/fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    
    if not test:
        name = '_'.join(title.split('\n'))
        filepath = os.path.join(path_out, f'Title_movie_{name}_{fps}fps.mp4')
        ani.save(filepath, writer=writer, dpi=300)
    return ani

def plot_vr_movie(adata, trial, path_out, fps=30, test=False, save_suffix=None):
    is_heading_pulse = 'h_pulse' in adata.obs.columns

    # Select vr data
    vr = adata.uns['vr'].copy()
    vr = vr[(vr['trial']==trial)]
    t0 = vr['t'].iloc[0]
    t1 = vr['t'].iloc[-1]
    vr['t'] -= t0
    dt_vr = vr['dt'].mean()
    
    # Select obs data
    adatai = adata[(adata.obs['t']>=t0) & (adata.obs['t']<t1)]
    world = adatai.obs['world'].iloc[0]
    obs = adatai.obs.copy()
    obs['ddh_0.25sigma'] = scipy.ndimage.gaussian_filter1d(obs['ddh'], 0.25/obs['dt'].mean())
    obs['t'] -= t0
    
    # Import frame grabs
    key = sess.adata_to_key(adata)
    path = sess.define_path(**key)
    trial_str = '%03d' %trial
    frame_file = path['frameGrabs_mat'].format(trial=trial_str)
    try:
        data = scipy.io.loadmat(frame_file)
        frames = data['frameData']
    except NotImplementedError:
        with h5py.File(frame_file, 'r') as f:
            frames = np.array(f['frameData']).T

    # print(frames.shape[2])
    # Remove previous ITI, adata trial starts after ITI
    trial_ind0 = np.where(frames.max(axis=(0, 1))>0)[0][0]
    frames = frames[:, :, trial_ind0:]
    # print(frames.shape[2])
    
    iterations = int(frames.shape[2] + fps*1.5)
    if test:
        iterations = 10
    fig = plt.figure(figsize=(10, 10))
    
    nrows = 2
    height_ratios = [1, 1]
    gs = mpl.gridspec.GridSpec(nrows=nrows, ncols=1, height_ratios=height_ratios)
    gs_top = gs[0, 0].subgridspec(nrows=1, ncols=3, width_ratios=[2, 1, 0.2])
    gs_bot = gs[1, 0].subgridspec(nrows=4, ncols=3, width_ratios=[0.5, 2, 0.5], hspace=0.3)
    bot_axs = []

    # VR view
    ax0 = plt.subplot(gs_top[0, 0])
    ax0.set_title('Mouse view of\nvirtual environment')
    for spine in ['top', 'bottom', 'left', 'right']:
        ax0.spines[spine].set_visible(False)
    # ax0.xaxis.set_visible(False)
    ax0.set_xticks([])
    ax0.yaxis.set_visible(False)
    im = ax0.imshow(frames[:, :, 0], cmap='Greys_r')
    ax0.invert_yaxis()
    ax0.invert_xaxis()
    ax0.set_xlabel('1/2X playback speed')
    
    # Trajectory
    ax1 = plt.subplot(gs_top[0, 1])
    ax1.set_title('Mouse trajectory')
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax1.spines[spine].set_visible(False)
    if world =='black_left':
        x_reward = -12
    else:
        x_reward = 12
    pl.draw_maze(ax1)
    ax1.text(x_reward-5.5, 340, 'Reward')
    ax1.arrow(x_reward, 335, 0, -15, width=0.25, head_width=1, head_length=6, color='black')
    xlim = (-14, 14)
    ylim = (-50, 400)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    # axcbar = plt.subplot(gs_top[0, 2])
    vmax = 1
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, aspect=5, shrink=0.2)
    cbar.ax.tick_params(width=1, length=8)
    cbar.ax.set_title('Sst44 cell\nactivity', size=14, pad=15)
    
    # Lineplot 0
    i = 0
    ax2 = plt.subplot(gs_bot[i, 1])
    bot_axs.append(ax2)
    ax2.axhline(0, ls='--', lw=1, dashes=[3, 3], color='black')
    ax2.xaxis.set_visible(False)
    spines_remove = ['top', 'right', 'bottom']
    for spine in spines_remove:
        ax2.spines[spine].set_visible(False)
    signal = 'Sst44_dcnv_norm'
    line = ax2.plot(obs['t'], obs[signal], color='tab:blue')
    ymax = obs[signal].max()
    alim = [0, 1.5]
    if ymax > alim[1]:
        alim[1] = np.ceil(ymax)
    ax2.set_ylim(alim)
    ax2.tick_params(width=1, length=8)
    ax2.set_ylabel('Sst44 cell\nactivity', fontsize=12, rotation=0, ha='left', labelpad=80, va='center')

    # Lineplot 1
    i += 1
    ax3 = plt.subplot(gs_bot[i, 1], sharex=ax2)
    bot_axs.append(ax3)
    ax3.axhline(0, ls='--', lw=0.5, color='black', dashes=[6, 6])
    ax3.xaxis.set_visible(False)
    spines_remove = ['top', 'right', 'bottom']
    for spine in spines_remove:
        ax3.spines[spine].set_visible(False)
    ax3.plot(obs['t'], obs['h_mt_error'], color='black')
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_yticks([-np.pi, 0, np.pi])
    ax3.set_yticklabels(['-π', '0', '+π'])
    ax3.tick_params(width=1, length=8)
    ax3.set_ylabel('Heading\ndeviation\n(rad)', fontsize=12, rotation=0, ma='left', ha='left', labelpad=80, va='center')

    # ax3b = plt.subplot(gs_bot[2, 1], sharex=ax2)
    # ax3b.axhline(0, ls='--', lw=0.5, color='black', dashes=[6, 6])
    # spines_remove = ['top', 'right',]
    # for spine in spines_remove:
    #     ax3b.spines[spine].set_visible(False)
    # if 'h_pulse' in obs.keys():
    #     col = 'yaw_corr'
    # else:
    #     col = 'dh'
    # ax3b.plot(obs['t'], obs[col], color='black')
    # ax3b.set_ylim(-2, 2)
    # ax3b.set_yticks([-2, 0, 2])
    # ax3b.tick_params(width=1, length=8)
    # ax3b.set_ylabel('Turning\nvelocity\n(rad/s)', fontsize=12, rotation=0, ma='left', ha='left', labelpad=80, va='center')
    # spines_remove = ['top', 'right', 'bottom']
    # for spine in spines_remove:
    #     ax3b.spines[spine].set_visible(False)
    # ax3b.xaxis.set_visible(False)

    # Lineplot 2
    i += 1
    ax4 = plt.subplot(gs_bot[i, 1], sharex=ax2)
    bot_axs.append(ax4)
    ax4.axhline(0, ls='--', lw=0.5, color='black', dashes=[6, 6])
    ax4.plot(obs['t'], obs['ddh_0.25sigma'], color='black')
    ax4.set_ylim(-5, 5)
    ax4.set_yticks([-5, 0, 5])
    ax4.tick_params(width=1, length=8)
    ax4.set_ylabel('Turning\nacceleration\n(rad/s\u00b2)', fontsize=12, rotation=0, ma='left', ha='left', labelpad=80, va='center')
    if is_heading_pulse:
        spines_remove = ['top', 'right', 'bottom']
        ax4.xaxis.set_visible(False)
    else:
        spines_remove = ['top', 'right']
        ax4.set_xlabel('Time from trial start (s)')
    for spine in spines_remove:
        ax4.spines[spine].set_visible(False)

    if is_heading_pulse:
        i += 1
        ax5 = plt.subplot(gs_bot[i, 1], sharex=ax2)
        bot_axs.append(ax5)
        ax5.axhline(0, ls='--', lw=0.5, color='black', dashes=[6, 6])
        for spine in ['top', 'right']:
            ax5.spines[spine].set_visible(False)

        ax5.plot(obs['t'], obs['h_bias_radpersec'], color='grey')
        ax5.set_ylim(-2, 2)
        ax5.tick_params(width=1, length=8)
        ax5.set_ylabel('Heading\nvelocity\nbias (rad/s)', fontsize=12, rotation=0, ma='left', ha='left', labelpad=80, va='center')
        ax5.set_xlabel('Time from trial start (s)')
    
    fig.tight_layout()
    fig.align_ylabels(bot_axs)

    def update_img(i):
        if i > frames.shape[2]:
            return im
        t_mov = i / 60
        if (i)%10==0:
            print(f'Plotting frame {i+1} out of {iterations}.')
        elif i+1==iterations:
            print(f'Plotting frame {i+1} out of {iterations}.')
            print('Done plotting frames.')
        
        frame_inds = np.where(vr['t']>=t_mov)[0]
        if len(frame_inds) and (frame_inds[0] < (frames.shape[2]-1)):
            im.set_array(frames[:, :, frame_inds[0]])
        else:
            im.set_array(np.zeros_like(frames[:, :, 0]))
        
        ax1.cla()
        ax1.set_title('Mouse trajectory')
        pl.draw_maze(ax1)
        obs_ind = np.where(obs['t']>=t_mov)[0][0]
        q = pl.quiver_activity(adatai[obs_ind], obs_col='Sst44_dcnv_norm', ax=ax1, scale=12, show_ITI=True, width=0.05)
        q.set_clim(0, vmax)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        
        ax1.text(x_reward-5.5, 340, 'Reward')
        ax1.arrow(x_reward, 335, 0, -15, width=0.25, head_width=1, head_length=6, color='black')
        
        ax2.set_xlim(t_mov-10, t_mov)

        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, iterations, interval=1/fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    
    if not test:
        filepath = os.path.join(path_out, '{mouse}_{date}_trial{trial_str}_{fps}fps_behavior_traces.mp4'.format(**key, **dict(fps=fps, trial_str=trial_str)))
        ani.save(filepath, writer=writer, dpi=300)
    return ani

def plot_vr_movie2(adata, trial, path_out, fps=30, iterations=None, save=True, save_suffix=''):
    is_heading_pulse = 'h_pulse' in adata.obs.columns

    # Select vr data
    vr = adata.uns['vr'].copy()
    vr = vr[(vr['trial']==trial)]
    t0 = vr['t'].iloc[0]
    t1 = vr['t'].iloc[-1]
    vr['t'] -= t0
    dt_vr = vr['dt'].mean()
    
    # Select obs data
    adatai = adata[(adata.obs['t']>=t0) & (adata.obs['t']<t1)]
    world = adatai.obs['world'].iloc[0]
    obs = adatai.obs.copy()
    obs['ddh_0.25sigma'] = scipy.ndimage.gaussian_filter1d(obs['ddh'], 0.25/obs['dt'].mean())
    obs['t'] -= t0
    
    # Import frame grabs
    key = sess.adata_to_key(adata)
    path = sess.define_path(**key)
    trial_str = '%03d' %trial
    frame_file = path['frameGrabs_mat'].format(trial=trial_str)
    try:
        data = scipy.io.loadmat(frame_file)
        frames = data['frameData']
    except NotImplementedError:
        with h5py.File(frame_file, 'r') as f:
            frames = np.array(f['frameData']).T

    # print(frames.shape[2])
    # Remove previous ITI, adata trial starts after ITI
    trial_ind0 = np.where(frames.max(axis=(0, 1))>0)[0][0]
    frames = frames[:, :, trial_ind0:]
    # print(frames.shape[2])
    
    if iterations is None:
        iterations = int(frames.shape[2] + fps*1.5)

    fig = plt.figure(figsize=(12, 8))
    
    nrows = 2
    height_ratios = [1, 1]
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=2)
    gs_right = gs[0, 1].subgridspec(nrows=3, ncols=1)
    line_axs = []

    # VR view
    ax0 = plt.subplot(gs[0, 0])
    ax0.set_title('Mouse view of\nvirtual environment')
    for spine in ['top', 'bottom', 'left', 'right']:
        ax0.spines[spine].set_visible(False)
    # ax0.xaxis.set_visible(False)
    ax0.set_xticks([])
    ax0.yaxis.set_visible(False)
    im = ax0.imshow(frames[:, :, 0], cmap='Greys_r')
    ax0.invert_yaxis()
    ax0.invert_xaxis()
    ax0.set_xlabel('1/2X playback speed')
   
    # Lineplot 0
    i = 1
    ax2 = plt.subplot(gs_right[i, 0])
    line_axs.append(ax2)
    ax2.axhline(0, ls='--', lw=1, dashes=[3, 3], color='black')
    ax2.xaxis.set_visible(False)
    spines_remove = ['top', 'right', 'bottom']
    for spine in spines_remove:
        ax2.spines[spine].set_visible(False)
    signal = 'Sst44_dcnv_norm'
    line = ax2.plot(obs['t'], obs[signal], color='tab:blue')
    ymax = obs[signal].max()
    alim = [0, 1.5]
    if ymax > alim[1]:
        alim[1] = np.ceil(ymax)
    ax2.set_ylim(alim)
    ax2.tick_params(width=1, length=8)
    ax2.set_ylabel('Sst44 cell\nactivity', fontsize=12, rotation=0, ha='left', labelpad=80, va='center')

    if is_heading_pulse:
        i += 1
        ax5 = plt.subplot(gs_right[i, 0], sharex=ax2)
        line_axs.append(ax5)
        ax5.axhline(0, ls='--', lw=0.5, color='black', dashes=[6, 6])
        for spine in ['top', 'right']:
            ax5.spines[spine].set_visible(False)

        ax5.plot(obs['t'], obs['h_bias_radpersec'], color='grey')
        ax5.set_ylim(-2, 2)
        ax5.tick_params(width=1, length=8)
        ax5.set_ylabel('Heading\nvelocity\nbias (rad/s)', fontsize=12, rotation=0, ma='left', ha='left', labelpad=80, va='center')
        ax5.set_xlabel('Time from trial start (s)')
    
    fig.tight_layout()
    fig.align_ylabels(line_axs)

    def update_img(i):
        if i > frames.shape[2]:
            return im
        t_mov = i / 60
        if (i)%10==0:
            print(f'Plotting frame {i+1} out of {iterations}.')
        elif i+1==iterations:
            print(f'Plotting frame {i+1} out of {iterations}.')
            print('Done plotting frames.')
        
        frame_inds = np.where(vr['t']>=t_mov)[0]
        if len(frame_inds) and (frame_inds[0] < (frames.shape[2]-1)):
            im.set_array(frames[:, :, frame_inds[0]])
        else:
            im.set_array(np.zeros_like(frames[:, :, 0]))
        
        ax2.set_xlim(t_mov-10, t_mov)

        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, iterations, interval=1/fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    
    if save:
        filepath = os.path.join(path_out, '{mouse}_{date}_trial{trial_str}_{fps}fps_{save_suffix}.mp4'.format(**key, **dict(fps=fps, trial_str=trial_str, save_suffix=save_suffix)))
        print(filepath)
        ani.save(filepath, writer=writer, dpi=300)
    return ani

def plot_vr_movie_simple(adata, trial, path_out=None, fps=30, save=True, fontsize=18, iterations=None):
    is_heading_pulse = 'h_pulse' in adata.obs.columns

    # Select vr data
    vr = adata.uns['vr'].copy()
    vr = vr[(vr['trial']==trial)]
    t0 = vr['t'].iloc[0]
    t1 = vr['t'].iloc[-1]
    vr['t'] -= t0
    dt_vr = vr['dt'].mean()
    
    # Select obs data
    adatai = adata[(adata.obs['t']>=t0) & (adata.obs['t']<t1)]
    world = adatai.obs['world'].iloc[0]
    obs = adatai.obs.copy()
    obs['ddh_0.25sigma'] = scipy.ndimage.gaussian_filter1d(obs['ddh'], 0.25/obs['dt'].mean())
    obs['t'] -= t0
    
    # Import frame grabs
    key = sess.adata_to_key(adata)
    path = sess.define_path(**key)
    trial_str = '%03d' %trial
    frame_file = path['frameGrabs_mat'].format(trial=trial_str)
    try:
        data = scipy.io.loadmat(frame_file)
        frames = data['frameData']
    except NotImplementedError:
        with h5py.File(frame_file, 'r') as f:
            frames = np.array(f['frameData']).T

    # print(frames.shape[2])
    # Remove previous ITI, adata trial starts after ITI
    trial_ind0 = np.where(frames.max(axis=(0, 1))>0)[0][0]
    frames = frames[:, :, trial_ind0:]
    # print(frames.shape[2])
    
    if iterations is None:
        iterations = int(frames.shape[2] + fps*1.5)
    fig = plt.figure(figsize=(25, 10))
    
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.3)
    if is_heading_pulse:
        gs_right = gs[0, 1].subgridspec(nrows=4, ncols=1, height_ratios=[0.05, 0.5, 0.5, 0.05], hspace=0.6)
    else:
        gs_right = gs[0, 1].subgridspec(nrows=3, ncols=1, height_ratios=[0.2, 0.5, 0.2])

    # VR view
    ax0 = plt.subplot(gs[0, 0])
    ax0.set_title('Mouse view of virtual environment (1/2X playback speed)', fontsize=fontsize)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax0.spines[spine].set_visible(False)
    ax0.set_xticks([])
    ax0.yaxis.set_visible(False)
    im = ax0.imshow(frames[:, :, 0], cmap='Greys_r')
    ax0.invert_yaxis()
    ax0.invert_xaxis()
    
    lineplot_axs = []
    # Lineplots
    i = 1
    ax2 = plt.subplot(gs_right[i, 0])
    lineplot_axs.append(ax2)
    ax2.axhline(0, ls='--', lw=1, dashes=[3, 3], color='black')
    
    spines_remove = ['top', 'right']
    if is_heading_pulse:
        spines_remove.append('bottom')
        ax2.xaxis.set_visible(False)
    for spine in spines_remove:
        ax2.spines[spine].set_visible(False)
    signal = 'Sst44_dcnv_norm'
    line = ax2.plot(obs['t'], obs[signal], color='tab:blue')
    ymax = obs[signal].max()
    alim = [0, 1.5]
    if ymax > alim[1]:
        alim[1] = np.ceil(ymax)
    ax2.set_ylim(alim)
    ax2.tick_params(width=1, length=8, labelsize=fontsize)
    ax2.set_ylabel('Sst44 cell\nactivity', fontsize=fontsize, rotation=0, ha='left', labelpad=120, va='center')

    if is_heading_pulse:
        i += 1
        ax5 = plt.subplot(gs_right[i, 0], sharex=ax2)
        lineplot_axs.append(ax5)
        ax5.axhline(0, ls='--', lw=0.5, color='black', dashes=[6, 6])
        for spine in ['top', 'right']:
            ax5.spines[spine].set_visible(False)

        ax5.plot(obs['t'], obs['h_bias_radpersec'], color='grey')
        ax5.set_ylim(-2, 2)
        ax5.tick_params(width=1, length=8, labelsize=fontsize)
        ax5.set_ylabel('Heading\nvelocity\nbias (rad/s)', fontsize=fontsize, rotation=0, ma='left', ha='left', labelpad=120, va='center')
        ax5.set_xlabel('Time (s)', fontsize=fontsize, labelpad=20)
    else:
        ax2.set_xlabel('Time (s)', fontsize=fontsize, labelpad=20)
    
    # fig.tight_layout()
    fig.align_ylabels(lineplot_axs)

    def update_img(i):
        if i > frames.shape[2]:
            return im
        t_mov = i / 60
        if (i)%10==0:
            print(f'Plotting frame {i+1} out of {iterations}.')
        elif i+1==iterations:
            print(f'Plotting frame {i+1} out of {iterations}.')
            print('Done plotting frames.')
        
        frame_inds = np.where(vr['t']>=t_mov)[0]
        if len(frame_inds) and (frame_inds[0] < (frames.shape[2]-1)):
            im.set_array(frames[:, :, frame_inds[0]])
        else:
            im.set_array(np.zeros_like(frames[:, :, 0]))
        
        ax2.set_xlim(t_mov-10, t_mov)

        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, iterations, interval=1/fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    
    if save:
        filepath = os.path.join(path_out, '{mouse}_{date}_trial{trial_str}_{fps}fps.mp4'.format(**key, **dict(fps=fps, trial_str=trial_str)))
        ani.save(filepath, writer=writer, dpi=300)
    return ani

def plot_vr_movie_simple_forward_trace(adata, trial, path_out=None, save_suffix='', fps=30, save=True, fontsize=12, iterations=None):
    is_heading_pulse = 'h_pulse' in adata.obs.columns

    # Select vr data
    vr = adata.uns['vr'].copy()
    vr = vr[(vr['trial']==trial)]
    t0 = vr['t'].iloc[0]
    t1 = vr['t'].iloc[-1]
    vr['t'] -= t0
    dt_vr = vr['dt'].mean()
    tlim = (0, vr['t'].max())
    
    # Select obs data
    adatai = adata[(adata.obs['t']>=t0) & (adata.obs['t']<t1)]
    world = adatai.obs['world'].iloc[0]
    obs = adatai.obs.copy()
    obs['ddh_0.25sigma'] = scipy.ndimage.gaussian_filter1d(obs['ddh'], 0.25/obs['dt'].mean())
    obs['t'] -= t0
    
    # Import frame grabs
    key = sess.adata_to_key(adata)
    path = sess.define_path(**key)
    trial_str = '%03d' %trial
    frame_file = path['frameGrabs_mat'].format(trial=trial_str)
    try:
        data = scipy.io.loadmat(frame_file)
        frames = data['frameData']
    except NotImplementedError:
        with h5py.File(frame_file, 'r') as f:
            frames = np.array(f['frameData']).T

    # print(frames.shape[2])
    # Remove previous ITI, adata trial starts after ITI
    trial_ind0 = np.where(frames.max(axis=(0, 1))>0)[0][0]
    frames = frames[:, :, trial_ind0:]
    # print(frames.shape[2])
    
    if iterations is None:
        iterations = int(frames.shape[2] + fps*1.5)
    fig = plt.figure(figsize=(12, 8))
    
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.6)
    if is_heading_pulse:
        gs_right = gs[0, 1].subgridspec(nrows=4, ncols=1, height_ratios=[0.05, 0.5, 0.5, 0.05], hspace=0.6)
    else:
        gs_right = gs[0, 1].subgridspec(nrows=3, ncols=1, height_ratios=[0.2, 0.5, 0.2])

    # VR view
    ax0 = plt.subplot(gs[0, 0])
    ax0.set_title('Mouse view of virtual environment\n(1/2X playback speed)', fontsize=fontsize, va='bottom')
    for spine in ['top', 'bottom', 'left', 'right']:
        ax0.spines[spine].set_visible(False)
    ax0.set_xticks([])
    ax0.yaxis.set_visible(False)
    im = ax0.imshow(frames[:, :, 0], cmap='Greys_r')
    ax0.invert_yaxis()
    ax0.invert_xaxis()
    
    lineplot_axs = []
    # Lineplots
    i = 1
    ax2 = plt.subplot(gs_right[i, 0])
    lineplot_axs.append(ax2)
    ax2.axhline(0, ls='--', lw=1, dashes=[3, 3], color='black')
    
    spines_remove = ['top', 'right']
    if is_heading_pulse:
        spines_remove.append('bottom')
        ax2.xaxis.set_visible(False)
    for spine in spines_remove:
        ax2.spines[spine].set_visible(False)
    signal = 'Sst44_dcnv_norm'
    l_activity, = plt.plot([], [], color='tab:blue', lw=2)
    ax2.set_xlim(tlim)
    ymax = obs[signal].max()
    alim = [0, 1.5]
    if ymax > alim[1]:
        alim[1] = np.ceil(ymax)
    ax2.set_ylim(alim)
    ax2.tick_params(width=1, length=8, labelsize=fontsize)
    ax2.set_ylabel('Sst44 cell\nactivity', fontsize=fontsize, rotation=0, ha='left', labelpad=70, va='center')

    if is_heading_pulse:
        i += 1
        ax5 = plt.subplot(gs_right[i, 0], sharex=ax2)
        lineplot_axs.append(ax5)
        ax5.axhline(0, ls='--', lw=0.5, color='black', dashes=[6, 6])
        for spine in ['top', 'right']:
            ax5.spines[spine].set_visible(False)

        l_hpulse, = plt.plot([], [], color='grey', lw=2)
        ax5.set_xlim(tlim)
        ax5.set_ylim(-2, 2)
        ax5.tick_params(width=1, length=8, labelsize=fontsize)
        ax5.set_ylabel('Heading\nvelocity\nbias (rad/s)', fontsize=fontsize, rotation=0, ma='left', ha='left', labelpad=70, va='center',)
        ax5.set_xlabel('Time (s)', fontsize=fontsize, labelpad=10)
    else:
        ax2.set_xlabel('Time (s)', fontsize=fontsize, labelpad=10)
    
    # fig.tight_layout()
    fig.align_ylabels(lineplot_axs)
    obs_ind0 = np.where(obs['t']>=0)[0][0]
    t_interp = np.arange(obs['t'][0], obs['t'][-1], 0.0167)
    f = scipy.interpolate.interp1d(obs['t'], obs[signal], kind='linear')
    activity_interp = f(t_interp)
    if is_heading_pulse:
        f = scipy.interpolate.interp1d(obs['t'], obs['h_bias_radpersec'], kind='linear')
        hpulse_interp = f(t_interp)
    interp_ind0 = np.where(t_interp>=0)[0][0]
    def update(i):
        if i > frames.shape[2]:
            return im
        t_mov = i / 60
        if (i)%10==0:
            print(f'Plotting frame {i+1} out of {iterations}.')
        elif i+1==iterations:
            print(f'Plotting frame {i+1} out of {iterations}.')
            print('Done plotting frames.')
        
        frame_inds = np.where(vr['t']>=t_mov)[0]
        if len(frame_inds) and (frame_inds[0] < (frames.shape[2]-1)):
            im.set_array(frames[:, :, frame_inds[0]])
        else:
            im.set_array(np.zeros_like(frames[:, :, 0]))
        
        obs_ind1 = np.where(obs['t']>=t_mov)[0][0]
        interp_ind1 = np.where(t_interp>=t_mov)[0][0]
        l_activity.set_data(t_interp[interp_ind0:interp_ind1], activity_interp[interp_ind0:interp_ind1])
        if is_heading_pulse:
            l_hpulse.set_data(t_interp[interp_ind0:interp_ind1], hpulse_interp[interp_ind0:interp_ind1])

        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig, update, iterations, interval=1/fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    
    if save:
        filepath = os.path.join(path_out, '{mouse}_{date}_trial{trial_str}_{fps}fps_{save_suffix}.mp4'.format(**key, **dict(fps=fps, trial_str=trial_str, save_suffix=save_suffix)))
        print(filepath)
        ani.save(filepath, writer=writer, dpi=300)
    return ani

def plot_vr_movie_forward_trace_cell_types(adata, trial, path_out=None, save_suffix='', fps=30, save=True, fontsize=12, iterations=None, obs_cols=None, iti_s=0, remove_prev_iti=True):
    
    # Select vr data
    vr = adata.uns['vr'].copy()
    if type(trial) is int:
        vr = vr[vr['trial']==trial]
    else:
        vr = vr[(vr['trial']>=trial[0]) & (vr['trial']<=trial[1])]
    t0 = vr['t'].iloc[0]
    t1 = vr['t'].iloc[-1]
    vr['t'] -= t0
    dt_vr = vr['dt'].mean()
    tlim = (0, vr['t'].max())
    
    # Select obs data
    adatai = adata[(adata.obs['t']>=t0) & (adata.obs['t']<t1)]
    # world = adatai.obs['world'].iloc[0]
    obs = adatai.obs.copy()
    # obs['ddh_0.25sigma'] = scipy.ndimage.gaussian_filter1d(obs['ddh'], 0.25/obs['dt'].mean())
    obs['t'] -= t0
    
    # Import frame grabs
    key = sess.adata_to_key(adata)
    path = sess.define_path(**key)
    if type(trial) is int:
        trial_str = '%03d' %trial
        frame_file = path['frameGrabs_mat'].format(trial=trial_str)
    else:
        trial_str = 'trial_%i-%i' %trial
        frame_file = os.path.join(path['preprocessed_dir'], 'virmen', 'frameGrabs', trial_str + '.mat')
    print(frame_file)
    
    try:
        data = scipy.io.loadmat(frame_file)
        frames = data['frameData']
    except NotImplementedError:
        with h5py.File(frame_file, 'r') as f:
            frames = np.array(f['frameData']).T

    if remove_prev_iti:
        # Remove previous ITI, adata trial starts after ITI
        trial_ind0 = np.where(frames.max(axis=(0, 1))>0)[0][0]
        frames = frames[:, :, trial_ind0:]
    
    if iterations is None:
        iterations = int(frames.shape[2] + 60*iti_s)

    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.6)
    nlineplots = len(obs_cols)
    nrows = nlineplots + 2
    gs_right = gs[0, 1].subgridspec(nrows=nrows, ncols=1, height_ratios=[1] * nrows, hspace=0.4)

    # Setup VR plot
    ax0 = plt.subplot(gs[0, 0])
    ax0.set_title('Mouse view of virtual environment\n(1/2X playback speed)', fontsize=fontsize, va='bottom',)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax0.spines[spine].set_visible(False)
    ax0.set_xticks([])
    ax0.yaxis.set_visible(False)
    im = ax0.imshow(frames[:, :, 0], cmap='Greys_r')
    ax0.invert_yaxis()
    ax0.invert_xaxis()
    
    # Setup lineplots
    lineplot_axs = []
    lineplots = []
    irow = 0
    for obs_col in obs_cols:
        obs_col = obs_col.copy()
        irow += 1
        signal = obs_col.pop('signal')
        ylabel = obs_col.pop('ylabel')
        ylim = obs_col.pop('ylim')

        # Define axis
        ax2 = plt.subplot(gs_right[irow, 0])
        lineplot_axs.append(ax2)
        ax2.axhline(0, ls='--', lw=1, dashes=[3, 3], color='black')
        spines_remove = ['top', 'right']
        if irow < nrows - 2:
            spines_remove.append('bottom')
            ax2.xaxis.set_visible(False)
        for spine in spines_remove:
            ax2.spines[spine].set_visible(False)
            ax2.set_xlabel('Time (s)', fontsize=fontsize, labelpad=10,)
        ax2.set_xlim(tlim)
        ax2.set_ylim(ylim)
        ax2.tick_params(width=1, length=8, labelsize=fontsize)
        ax2.set_ylabel(ylabel, fontsize=fontsize, rotation=0, ha='left', labelpad=70, va='center')

        # Set plot
        lineplot, = plt.plot([], [], **obs_col)
        lineplots.append(lineplot)
    fig.align_ylabels(lineplot_axs)

    # Interpolate imaging timeframe data to vr 60Hz data
    obs_ind0 = np.where(obs['t']>=0)[0][0]
    t_interp = np.arange(obs['t'].iloc[0], obs['t'].iloc[-1], 0.0166667)
    interpolations = []
    for obs_col in obs_cols:
        signal = obs_col['signal']
        f = scipy.interpolate.interp1d(obs['t'], obs[signal], kind='linear')
        interpolation = f(t_interp)
        interpolations.append(interpolation)
    interp_ind0 = np.where(t_interp>=0)[0][0]

    # Populate plots for each frame
    def update(i):
        # if i > frames.shape[2]:
        #     return im
        t_mov = i / 60
        if (i)%10==0:
            print(f'Plotting frame {i+1} out of {iterations}.')
        elif i+1==iterations:
            print(f'Plotting frame {i+1} out of {iterations}.')
            print('Done plotting frames.')
        
        frame_inds = np.where(vr['t']>=t_mov)[0]
        if len(frame_inds) and (frame_inds[0] < (frames.shape[2]-1)):
            im.set_array(frames[:, :, frame_inds[0]])
        else:
            im.set_array(np.zeros_like(frames[:, :, 0]))
        
        obs_ind1 = np.where(obs['t']>=t_mov)[0][0]
        interp_ind1 = np.where(t_interp>=t_mov)[0][0]
        for lineplot, interpolation in zip(lineplots, interpolations):
            lineplot.set_data(t_interp[interp_ind0:interp_ind1], interpolation[interp_ind0:interp_ind1])

        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig, update, iterations, interval=1/fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    
    if save:
        filepath = os.path.join(path_out, '{mouse}_{date}_trial{trial_str}_{fps}fps_{save_suffix}.mp4'.format(**key, **dict(fps=fps, trial_str=trial_str, save_suffix=save_suffix)))
        ani.save(filepath, writer=writer, dpi=300)
    return ani

def main(mouse=None, date=None, session=None, plot='plot_vr_movie', path_out=None, **kwargs):
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    adata = sess.load_as_anndata(mouse=mouse, date=date, session=session, adata_filekey='adata_h5ad', recompute=False, save=False, skip_if_not_saved=False)
    plot_function = globals()[plot]
    plot_function(adata, path_out=path_out, **kwargs)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load session as anndata and save pickle.')
    parser.add_argument('--mouse', required=True, type=str)
    parser.add_argument('--date', required=True, type=str)
    parser.add_argument('--session', required=True, type=str)
    parser.add_argument('--trial', required=True, type=int)
    parser.add_argument('--path_out', required=True, type=str)
    parser.add_argument('--fps', required=False, type=int, default=30)
    parser.add_argument('--plot', required=False, type=str, default='plot_vr_movie')
    parser.add_argument('--save_suffix', required=False, type=str, default='')
    
    args = vars(parser.parse_args())
    main(**args)