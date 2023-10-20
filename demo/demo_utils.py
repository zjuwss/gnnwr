import folium
import json
from folium.plugins import HeatMap, MarkerCluster
def marker_map(markers:list,center:list,zoom=4,border=None):
    tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=en&size=1&scl=1&style=7'
    map = folium.Map(location=center,zoom_start=zoom,tiles = tiles,attr="高德")
    if border != None:
        with open(border,encoding='utf-8') as f:
            t = f.readline()
            geojson = json.loads(t)
            folium.GeoJson(geojson).add_to(map)
    for item in markers:
        if not 'location' in item: raise ValueError('location of markers is neccessary')
        if not 'color' in item: item['color'] = 'blue'
        if not 'desc' in item: item['desc'] = [str(item['location'][i][1])+'  '+str(item['location'][i][0]) for i in range(len(item['location']))]
        else : item['desc'] = [item['desc'] for i in range(len(item['location']))]
        mc = MarkerCluster(locations=item['location'],icons=[folium.Icon(color=item['color'],icon='location') for i in range(len(item['location']))],popups=item['desc'])
        mc.add_to(map)
    return map

def Heatmap(data,center,zoom=4):
    tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=en&size=1&scl=1&style=7'
    map = folium.Map(location=center,zoom_start=zoom,tiles = tiles,attr="高德")
    map.add_child(HeatMap(data))
    return map
