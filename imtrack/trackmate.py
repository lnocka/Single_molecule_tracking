"""
Important features of TrackMate tracking:
    * Singleton spots do not show up in any output track.csv
    * Filtered out tracks and spots do not show up in output csv (a good thing)

Gotchas:
If you select XML, and have SpotFilters present, and custom spots added, you may get double
coincidence detection.

See 'trackmateXML' module for more information

Created by Darren McAffee
"""

import numpy as np
import pandas as pd
import os

from .trackmateXML import tmxml


def trackmate(spot_source, folder, prefix):
    track, spot = tmcsv(folder,prefix)
    if spot_source == 'csv':
        return track, spot
    if spot_source == 'xml':
        _, spot = tmxml(folder, prefix)
        # I don't want to have to compute the statistics for each track. Let TrackMate do that.
        # I'll only add the singletons
        track, spot = add_spot_to_track(spot, track)
        return track, spot
    else:
        raise ValueError(f"Unrecognized spotfile type {spot_source}")


def read_csv(path, id=None):
    df = pd.read_csv(path, index_col=id)
    if ' ' in df.columns:
        df = df.drop(' ', axis=1)
    df.columns = [s.lower() for s in df.columns]
    df.index.name = df.index.name.lower()
    return df


def tmcsv(folder, prefix):
    dfs = []
    for s, i in zip([' tracks.csv', ' spots.csv'], ['TRACK_ID', 'ID', None]):
        df = read_csv(os.path.join(folder, prefix + s), i)
        df.columns = df.columns.map(lambda x: x.lower())
        dfs.append(df)
    return dfs


def add_spot_to_track(spot, track):
    # 5 names per row for easier counting
    trackcols = ['label', 'number_spots', 'number_gaps', 'longest_gap', 'number_splits',
                 'number_merges', 'number_complex', 'track_duration', 'track_start', 'track_stop',
                 'track_displacement', 'track_index', 'track_x_location', 'track_y_location', 'track_z_location',
                 'track_mean_speed', 'track_max_speed', 'track_min_speed', 'track_median_speed', 'track_std_speed',
                 'track_mean_quality', 'track_max_quality', 'track_min_quality', 'track_median_quality', 'track_std_quality']
    tdict = {k: [] for k in trackcols}
    lasttrack = track.index.max() + 1
    singletons = spot[spot.track_id == -1].reset_index().groupby('id')
    for sid, sdf in singletons:
        tdict['label'].append('Track_'+str(lasttrack))
        tdict['track_index'].append(lasttrack)
        spot.loc[sid, 'track_id'] = lasttrack
        lasttrack += 1
        tdict['number_spots'].append(1)
        for k in trackcols[2:8]:
            tdict[k].append(0)
        tdict['track_start'].append(sdf['frame'].iloc[0])
        tdict['track_stop'].append(sdf['frame'].iloc[0])
        tdict['track_displacement'].append(0.)
        tdict['track_x_location'].append(sdf['position_x'].iloc[0])
        tdict['track_y_location'].append(sdf['position_y'].iloc[0])
        tdict['track_z_location'].append(0.)
        for k in trackcols[15:20]:
            tdict[k].append(0.)
        for k in trackcols[20:24]:
            tdict[k].append(sdf['quality'].iloc[0])
        tdict['track_std_quality'].append(0.)
    tsingles = pd.DataFrame(tdict)
    track = pd.concat((track, tsingles))
    return track, spot
