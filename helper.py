import pandas as pd
import geopandas as gpd
from multiprocessing import Pool, cpu_count
from shapely.ops import unary_union
from shapely.geometry import Point, LineString, shape
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestCentroid



def getGeoDataFrame_v2(df, coordinate_columns=['lon', 'lat'], crs={'init':'epsg:4326'}):
	'''
		Create a GeoDataFrame from a DataFrame in a much more generalized form.
	'''
	
	df.loc[:, 'geom'] = np.nan
	df.geom = df[coordinate_columns].apply(lambda x: Point(*x), axis=1)
	
	return gpd.GeoDataFrame(df, geometry='geom', crs=crs)


def haversine(p_1, p_2):
	'''
		Return the Haversine Distance of two points in KM
	'''
	lon1, lat1, lon2, lat2 = map(np.deg2rad, [p_1[0], p_1[1], p_2[0], p_2[1]])   
	
	dlon = lon2 - lon1
	dlat = lat2 - lat1    
	a = np.power(np.sin(dlat * 0.5), 2) + np.cos(lat1) * np.cos(lat2) * np.power(np.sin(dlon * 0.5), 2)    
	
	return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def calculate_acceleration(gdf, spd_column='velocity', ts_column='ts'):
	'''
	Return given dataframe with an extra acceleration column that
	is calculated using the rate of change of velocity over time.
	'''
	# if there is only one point in the trajectory its acceleration will be zero (i.e. constant speed)
	if len(gdf) == 1:
		gdf.loc[:, 'acceleration'] = 0
		return gdf
	
	gdf.loc[:, 'acceleration'] = gdf[spd_column].diff(-1).divide(gdf[ts_column].diff(-1).abs())
	gdf.dropna(subset=['geom'], inplace=True)
	
	return gdf


def calculate_velocity(gdf, spd_column='velocity', ts_column='ts'):
	'''
	Return given dataframe with an extra velocity column that 
	is calculated using the distance covered in a given amount of time.
	TODO - use the get distance method to save some space
	'''
	# if there is only one point in the trajectory its velocity will be the one measured from the speedometer
	if len(gdf) == 1:
		if spd_column is not None:
			gdf.loc[:, 'velocity'] = gdf[spd_column]
		else:
			gdf.loc[:, 'velocity'] = np.nan
		return gdf

	# create columns for current and next location. Drop the last columns that contains the nan value
	gdf.loc[:, 'current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
	gdf.loc[:, 'next_loc'] = gdf.geom.shift(-1)
	gdf.loc[:, 'dt'] = gdf[ts_column].diff(-1).abs()
	
	gdf = gdf.iloc[:-1]
	gdf.next_loc = gdf.next_loc.apply(lambda x : (x.x,x.y)) 
		
	# get the distance traveled in n-miles and multiply by the rate given (3600/secs for knots)
	gdf.loc[:,'velocity'] = gdf[['current_loc', 'next_loc']].apply(lambda x : haversine(x[0], x[1])*0.539956803 , axis=1).multiply(3600/gdf.dt)

	gdf.drop(['current_loc', 'next_loc', 'dt'], axis=1, inplace=True)
	gdf.dropna(subset=['geom'], inplace=True)
	
	return gdf


def calculate_angle(point1, point2):
	'''
		Calculating initial bearing between two points
	'''
	lon1, lat1 = point1[0], point1[1]
	lon2, lat2 = point2[0], point2[1]

	dlat = (lat2 - lat1)
	dlon = (lon2 - lon1)
	numerator = np.sin(dlon) * np.cos(lat2)
	denominator = (
		np.cos(lat1) * np.sin(lat2) -
		(np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
	)

	theta = np.arctan2(numerator, denominator)
	theta_deg = (np.degrees(theta) + 360) % 360
	return theta_deg


def calculate_bearing(gdf, course_column='course'):
	'''
	Return given dataframe with an extra bearing column that
	is calculated using the course over ground (in degrees in range [0, 360))
	'''
	# if there is only one point in the trajectory its bearing will be the one measured from the accelerometer
	if len(gdf) == 1:
		if course_column is not None:
			gdf.loc[:, 'bearing'] = gdf[course_column]
		else:
			gdf.loc[:, 'bearing'] = np.nan
		return gdf

	# create columns for current and next location. Drop the last columns that contains the nan value
	gdf.loc[:, 'current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
	gdf.loc[:, 'next_loc'] = gdf.geom.shift(-1)
	gdf = gdf.iloc[:-1]
	
	gdf.next_loc = gdf.next_loc.apply(lambda x : (x.x,x.y))
	
	gdf.loc[:,'bearing'] = gdf[['current_loc', 'next_loc']].apply(lambda x: calculate_angle(x[0], x[1]), axis=1)

	gdf.drop(['current_loc', 'next_loc'], axis=1, inplace=True)
	gdf.dropna(subset=['geom'], inplace=True)
	
	return gdf


def create_radial_chart(ax, df, bins_number, degree_intervals):
	ticks = [r'${0}^o$'.format(degree_intervals*i) for i in range (bins_number)]
	bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
	angles = np.radians(df.bearing.loc[~df.bearing.isin([0.0, 180.0])].dropna().values)

	n, bins, _ = ax.hist(angles, bins)


	fig = plt.figure()
	ax = plt.subplot(1, 1, 1, projection='polar')

	width = 2 * np.pi / bins_number - 0.02
	bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0, align='edge')

	for bar in bars:
		bar.set_facecolor(plt.cm.tab20(0))
		bar.set_alpha(1.0)
		
	plt.suptitle('Vessel course polar chart', fontsize=8, y=1.02)

	lines, labels = plt.thetagrids(range(0, 360, degree_intervals), ticks, fontsize=8)
	
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax.yaxis.offsetText.set_visible(False)

	ax.tick_params(pad=0)
	ax.set_rlabel_position(140)

	ax.set_theta_zero_location("N")
	ax.set_theta_direction(-1)

	return ax, lines, labels


def highlight_outliers(s, indices):
	if s.name in indices:
		return ['background-color: lightblue']*len(s.index)
	else:
		return ['background-color: white']*len(s.index)
    
    
def create_trajectories(traj_gdf):
	traj_gdf.loc[:, 'label'] = traj_gdf['traj_id']
	traj_gdf_trimmed = traj_gdf.groupby('mmsi', group_keys = False).apply(lambda gdf: gdf.loc[gdf.traj_id[gdf.traj_id.replace(-1, np.nan).ffill(limit = 1).bfill(limit = 1).notnull()].index])    
	return traj_gdf_trimmed


def fix_trajectories(traj_ves):
	traj_segments = np.split(traj_ves, traj_ves.loc[traj_ves.traj_id == -1].index)
	traj_segments = [df for df in traj_segments if len(df) != 0]    # remove the fragments that have 0 points
	traj_segments[0].loc[:,'traj_id'] = 0
    
	for i in range(1,len(traj_segments)):        
		if (len(traj_segments[i]) == 1):
			traj_segments[i].loc[:,'traj_id'] = traj_segments[i-1].traj_id.max()
		else:
			traj_segments[i].loc[:,'traj_id'] = traj_segments[i-1].traj_id.max()+1
            
	traj_segments_id_fix = pd.concat(traj_segments)
	traj_segments_id_fix.sort_values('timestamp', inplace=True)
	traj_segments_id_fix.reset_index(inplace=True, drop=True)

	print (f'(Initial) Number of segments: {len(traj_segments)}')
	print (f'(Final-Useful) Number of port-based segments produced: {len(traj_segments_id_fix["traj_id"].unique())}')

	return traj_segments_id_fix


def temporal_segmentation(traj_segments_id_fix, TEMPORAL_THRESHOLD = 60*60*12, CARDINALITY_THRESHOLD = 3):

	traj_ves_trips = []
	print (f'(Initial) Number of port-based segments: {len(traj_segments_id_fix["traj_id"].unique())}')

	for traj_id, sdf in traj_segments_id_fix.groupby('traj_id'):
		df = sdf.reset_index()
		break_points = df.loc[df['timestamp'].diff() > TEMPORAL_THRESHOLD].index

		sdfs = np.split(df, break_points)
		traj_ves_trips.extend(sdfs)

	print (f'(Intermediate) Number of temporal-gap-based segments: {len(traj_ves_trips)}')

	traj_ves_trips = [tmp_df for tmp_df in traj_ves_trips if len(tmp_df) >= CARDINALITY_THRESHOLD]
	print (f'(Final-Useful) Number of trips produced: {len(traj_ves_trips)}')

	traj_ves_trips[0].loc[:,'trip_id'] = 0
	for idx in range(1, len(traj_ves_trips)):
		traj_ves_trips[idx].loc[:,'trip_id'] = traj_ves_trips[idx-1].trip_id.max()+1
    
	traj_ves_trips_gdf = pd.concat(traj_ves_trips)
	traj_ves_trips_gdf.sort_values('timestamp', inplace=True)
	traj_ves_trips_gdf.reset_index(inplace=True, drop=True)
    
	return traj_ves_trips_gdf


def temporal_alignment_v2(df, features=['lat', 'lon'], temporal_axis_name='datetime', temporal_name='ts', temporal_unit='s', rate=1, method='linear', crs={'init': 'epsg:4326'}, asGeoDataFrame=False):
	df.loc[:, temporal_axis_name] = pd.to_datetime(df[temporal_name], unit=temporal_unit)
	x = df[temporal_axis_name].values.astype(np.int64)
	y = df[features].values
	
	# scipy interpolate needs at least 2 records 
	if (len(df) <= 1):
		if asGeoDataFrame: df.insert(len(df.columns), 'geom', '')
		return df.iloc[0:0]
	
	dt_start = df[temporal_axis_name].min().replace(second=0)
	dt_end = df[temporal_axis_name].max().replace(second=0)
	
	f = interp1d(x, y, kind=method, axis=0)
	xnew_V3 = pd.date_range(start=dt_start.replace(minute=rate*(dt_start.minute//rate)), end=dt_end, freq=f'{rate*60}S', closed='right') 
   
	df_RESAMPLED = pd.DataFrame(f(xnew_V3), columns=features)      
	df_RESAMPLED.loc[:, temporal_axis_name] = xnew_V3
	
	if asGeoDataFrame:		
		if (len(df_RESAMPLED) == 0):
			df_RESAMPLED.insert(len(df_RESAMPLED.columns), 'geom', '')
		else:
			df_RESAMPLED.loc[:, 'geom'] = df_RESAMPLED[['lon', 'lat']].apply(lambda x: Point(x[0], x[1]), axis=1)
			return gpd.GeoDataFrame(df_RESAMPLED, crs=crs, geometry='geom')
	else:
		return df_RESAMPLED
    
    
def get_clusters_centers(X, labels, ignore=-1):
	clf = NearestCentroid()
	clf.fit(X[labels!=ignore].values, labels[labels!=ignore])
	return np.degrees(clf.centroids_)