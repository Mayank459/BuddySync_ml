"""City centre coordinates for origin cities (travel distance for train fares). Approximate city centres.

ponytail: a hand list of ~45 cities; swap for a geocoder (or the datameet stations list) when users come
from smaller towns. Unknown city → None → the leg keeps its search links without a fare.
"""
CITY_COORDS = {
    "mumbai": (19.0760, 72.8777), "delhi": (28.6139, 77.2090), "new delhi": (28.6139, 77.2090),
    "bengaluru": (12.9716, 77.5946), "bangalore": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867),
    "chennai": (13.0827, 80.2707), "kolkata": (22.5726, 88.3639), "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714), "goa": (15.2832, 73.9862), "jaipur": (26.9124, 75.7873),
    "kochi": (9.9312, 76.2673), "cochin": (9.9312, 76.2673), "lucknow": (26.8467, 80.9462),
    "chandigarh": (30.7333, 76.7794), "indore": (22.7196, 75.8577), "nagpur": (21.1458, 79.0882),
    "surat": (21.1702, 72.8311), "vadodara": (22.3072, 73.1812), "bhopal": (23.2599, 77.4126),
    "patna": (25.5941, 85.1376), "bhubaneswar": (20.2961, 85.8245), "guwahati": (26.1445, 91.7362),
    "thiruvananthapuram": (8.5241, 76.9366), "trivandrum": (8.5241, 76.9366), "coimbatore": (11.0168, 76.9558),
    "visakhapatnam": (17.6868, 83.2185), "vizag": (17.6868, 83.2185), "varanasi": (25.3176, 82.9739),
    "amritsar": (31.6340, 74.8723), "udaipur": (24.5854, 73.7125), "mangaluru": (12.9141, 74.8560),
    "mangalore": (12.9141, 74.8560), "raipur": (21.2514, 81.6296), "ranchi": (23.3441, 85.3096),
    "dehradun": (30.3165, 78.0322), "madurai": (9.9252, 78.1198), "nashik": (19.9975, 73.7898),
    "kanpur": (26.4499, 80.3319), "agra": (27.1767, 78.0081), "mysuru": (12.2958, 76.6394),
    "mysore": (12.2958, 76.6394), "vijayawada": (16.5062, 80.6480), "jodhpur": (26.2389, 73.0243),
    "gwalior": (26.2183, 78.1828), "aurangabad": (19.8762, 75.3433), "kolhapur": (16.7050, 74.2433),
    "thane": (19.2183, 72.9781), "navi mumbai": (19.0330, 73.0297), "noida": (28.5355, 77.3910),
    "gurugram": (28.4595, 77.0266), "gurgaon": (28.4595, 77.0266), "ghaziabad": (28.6692, 77.4538),
}


def coords(city: str):
    return CITY_COORDS.get((city or "").strip().lower())
