"""
Important features of TrackMate tracking:
    * Singleton spots do not show up in any output track.csv
    * Filtered out tracks and spots do not show up in output csv (a good thing)

So when one desires only filtered tracks / spots, use the csv.

HOWEVER, if one wants to include singleton spots in a spot dataframe, and to have them included in a track dataframe.
Then some manual work must be done.

The AIM of this module is to take XML -> SPOT dataframe, and TRACK dataframe that include singleton spots.

Note: there is no 'track_id' in the 'spot' features, we must add these from the 'Edge' features.

The XML output of TrackMate has the following structure:
<TrackMate version="3.8.0">
    <Log>...</Log>
    <Model spatialunits="pixel" timeunits="frame">
      <FeatureDeclarations>...</FeatureDeclarations>
      <AllSpots nspots="1196">
        <SpotsInFrame frame="0">
          ...
          <Spot ID="250595" name="ID250595" QUALITY="14682.9521484375" POSITION_T="0.0" MAX_INTENSITY="50913.0" FRAME="0" MEDIAN_INTENSITY="21743.0" VISIBILITY="1"
           MEAN_INTENSITY="25272.47619047619" TOTAL_INTENSITY="530722.0" ESTIMATED_DIAMETER="2.2500000052938334" RADIUS="2.25" SNR="0.8873450151521538"
           POSITION_X="247.42050471050712" POSITION_Y="293.7728514745919" STANDARD_DEVIATION="15241.86393004165" CONTRAST="0.3653358557458769" MANUAL_COLOR="-10921639"
           MIN_INTENSITY="7031.0" POSITION_Z="0.0" />
           ...
           </SpotsInFrame frame="0">
        ...
        </AllSpots>
      <AllTracks>
        ...
        <Track name="Track_0" TRACK_ID="0" NUMBER_SPOTS="51" NUMBER_GAPS="0" LONGEST_GAP="0" NUMBER_SPLITS="0" NUMBER_MERGES="0" NUMBER_COMPLEX="0"
        TRACK_DURATION="50.0" TRACK_START="0.0" TRACK_STOP="50.0" TRACK_DISPLACEMENT="5.194169383548677" TRACK_INDEX="0" TRACK_X_LOCATION="249.578116610045"
        TRACK_Y_LOCATION="291.97859442797187" TRACK_Z_LOCATION="0.0" TRACK_MEAN_SPEED="0.9000246620355484" TRACK_MAX_SPEED="2.392235370142079"
        TRACK_MIN_SPEED="0.11579882188397175" TRACK_MEDIAN_SPEED="0.7715707228478237" TRACK_STD_SPEED="0.6012492060775483" TRACK_MEAN_QUALITY="14296.215973498774"
        TRACK_MAX_QUALITY="21959.5078125" TRACK_MIN_QUALITY="9743.107421875" TRACK_MEDIAN_QUALITY="14295.44140625" TRACK_STD_QUALITY="2358.8248069352476">
          ..
          <Edge SPOT_SOURCE_ID="250872" SPOT_TARGET_ID="250859" LINK_COST="0.8739312117421943" EDGE_TIME="20.5" EDGE_X_LOCATION="247.65753934890876"
          EDGE_Y_LOCATION="292.25872187999516" EDGE_Z_LOCATION="0.0" VELOCITY="0.934842880778473" DISPLACEMENT="0.934842880778473"/>
          ...
        ...
        </AllTracks>
      <FilteredTracks>...</FilteredTracks>
      </Model>
    <Settings>...</Settings>
    <GUIState state="ConfigureViews">...</GUIState>
</TrackMate>

Created by Darren McAffee

"""
import os
import pandas as pd
import xml.etree.ElementTree as ET

# Future: use same_structure and group_children with ObjFromXML
# to determine similarly structured xml elements
# and then convert into a multilevel dataframe (could try with nested dicts)

# def same_structure(e1, e2):
#     if type(e1) != type(e2):
#         return False
#     if e1.tag != e1.tag: return False
#     if e1.attrib.keys() != e2.attrib.keys(): return False
#     if len(e1) != len(e2): return False
#     return all([same_structure(c1, c2) for c1, c2 in zip(e1, e2)])
#
# def group_children(element):
#     tags = set([c.tag for c in element])
#     for group in tags:
#         pairs = combinations(element.findall(group))
#         if all(same_structure(a, b) for a, b in pairs):
#             outer = []
#             i
#             for num, child in enumerate(element.findall(group)):

# A recursive constructor that will make each attribute of the object
# another object with appropriately named attributes until you get to the
# leaves of the XML. Duplicate elements, e.g. [spot, spot, spot, ...] will
# be overwritten. Use special functions (e.g. spots2df) for certain tags.

class ObjFromXML:
    def __init__(self, element):
        for child in element:
            siblings = vars(self).keys()
            if child.tag == 'AllSpots':
                self.AllSpots = spots2df(child)
            elif child.tag == 'AllTracks':
                self.AllTracks = tracks2df(child, self.AllSpots)
            # An okay way to deal with many siblings
            elif child.tag in siblings:
                tag = child.tag
                num = len(child.tag)
                idx = [int(s[num:] or 0) for s in siblings if s.startswith(tag)]
                mx = max(idx)
                ix = mx + 1
                new_tag = tag + str(ix)
                vars(self).update({new_tag: ObjFromXML(child)})
                vars(getattr(self, new_tag)).update(child.attrib)
            else:
                vars(self).update({child.tag: ObjFromXML(child)})
                vars(getattr(self, child.tag)).update(child.attrib)


def spots2df(AllSpots):
    # Collect spot information as a dict, and collect said dicts.
    spot_list = []
    for spot in AllSpots.iter('Spot'):
        s = dict(spot.items())
        spot_list.append(s)

    # Convert to DF and clean up spot data
    spotdf = pd.DataFrame(spot_list)
    spotdf.index = spotdf.pop('ID')
    spotdf = spotdf.drop('name', 1)

    # Spot ID column list
    # Spot ID name QUALITY POSITION_T MAX_INTENSITY FRAME MEDIAN_INTENSITY VISIBILITY
    # MEAN_INTENSITY TOTAL_INTENSITY ESTIMATED_DIAMETER RADIUS SNR
    # POSITION_X POSITION_Y STANDARD_DEVIATION CONTRAST MANUAL_COLOR
    # MIN_INTENSITY POSITION_Z
    float_columns = ['QUALITY', 'MAX_INTENSITY', 'MEDIAN_INTENSITY', 'VISIBILITY', 'MEAN_INTENSITY', \
                     'TOTAL_INTENSITY', 'ESTIMATED_DIAMETER', 'RADIUS', 'SNR', 'POSITION_X', 'POSITION_Y',
                     'STANDARD_DEVIATION', 'CONTRAST', 'MIN_INTENSITY', 'POSITION_Z']
    spotdf[float_columns] = spotdf[float_columns].astype(float)
    float_columns.append('FRAME')
    spotdf = spotdf[float_columns]
    spotdf.columns = spotdf.columns.map(lambda x: x.lower())
    spotdf['frame'] = spotdf['frame'].astype(int)
    return spotdf


def tracks2df(AllTracks, spotdf):
    # Similarly collect track data.
    track_list = []
    for track in AllTracks.iter('Track'):
        # Sometimes in the CSV track output of TrackMate Label (Track_1) does not match Track_ID = 0
        # Especially if there has been filtering. Use Track_ID, which is used in CSV spot output.
        track_id = dict(track.items())['TRACK_ID']
        for ind, edge in enumerate(track.iter('Edge')):
            edge_dict = dict(edge.items())
            edge_dict.update({'TRACK_ID':track_id})
            track_list.append(edge_dict)

    # Clean up track data
    trackdf = pd.DataFrame(track_list)
    trackdf['SOURCE_FRAME'] = spotdf.loc[trackdf['SPOT_SOURCE_ID']].frame.values
    trackdf = trackdf.sort_values(by=['TRACK_ID','SOURCE_FRAME']).reset_index(drop=True)
    desired_columns = ['TRACK_ID', 'SPOT_SOURCE_ID', 'SPOT_TARGET_ID','SOURCE_FRAME']
    trackdf = trackdf[desired_columns]
    trackdf = trackdf.sort_values(by=["TRACK_ID", "SOURCE_FRAME"])
    trackdf.columns = trackdf.columns.map(lambda x: x.lower())
    return trackdf


def parsexml(filepath):
    '''filpath must be an accesssible .xml file from TrackMate output'''
    tree = ET.parse(filepath)
    root = tree.getroot()
    obj = ObjFromXML(root)
    return obj


def add_track_to_spot(spot, track):
    spot['track_id'] = -1  # sentinel value
    # Assign source spots to their tracks
    spot.loc[track['spot_source_id'], 'track_id'] = track['track_id'].values
    # Assign target spots to their tracks
    spot.loc[track['spot_target_id'], 'track_id'] = track['track_id'].values
    spot.track_id = spot.track_id.astype(int)


def tmxml(folder, prefix):
    path = os.path.join(folder, prefix + '.xml')
    obj = parsexml(path)
    track = obj.Model.AllTracks
    spot = obj.Model.AllSpots
    spot.index.name = spot.index.name.lower()
    add_track_to_spot(spot, track)
    return track, spot
