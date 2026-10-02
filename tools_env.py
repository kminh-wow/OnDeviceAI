"""Deterministic mock tool environment.

All tool outputs are a pure function of their (normalized) arguments, so every
precision (FP16 / Q8 / Q4) sees exactly the same world.
"""
import re
import zlib


def _h(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))


# ---------------------------------------------------------------- static data
CITIES = ["Seoul", "Busan", "Daejeon", "Gwangju", "Jeju"]
CITY_KO = {"Seoul": "서울", "Busan": "부산", "Daejeon": "대전", "Gwangju": "광주", "Jeju": "제주"}
CITY_CODE = {"Seoul": "SEL", "Busan": "PUS", "Daejeon": "DJN", "Gwangju": "GWJ", "Jeju": "JEJ"}

CATEGORIES = ["cafe", "restaurant", "museum", "park"]
CATEGORY_KO = {"cafe": "카페", "restaurant": "식당", "museum": "박물관", "park": "공원"}
CAT_CODE = {"cafe": "CAF", "restaurant": "RES", "museum": "MUS", "park": "PRK"}

ORIGINS = ["home", "office"]
ORIGIN_KO = {"home": "집", "office": "회사"}

DATES = ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"]
DAY_START, DAY_END = "09:00", "21:00"
CALENDAR = {
    "2026-10-05": [("Team meeting", "09:00", "11:30"), ("Lunch with Jisu", "12:00", "13:00"),
                   ("Project review", "15:00", "16:00")],
    "2026-10-06": [("Dentist", "10:00", "11:00"), ("Study group", "11:00", "14:00")],
    "2026-10-07": [("Class", "09:00", "12:00"), ("Lab meeting", "13:00", "17:00")],
    "2026-10-08": [("Gym", "09:00", "10:00"), ("Team project", "10:30", "13:30"),
                   ("Part-time job", "14:00", "18:30")],
    "2026-10-09": [("Morning lecture", "09:00", "10:30")],
}

CONDITIONS = ["sunny", "cloudy", "rain"]

ITEMS = ["umbrella", "keyboard", "mouse", "headphones", "backpack", "tumbler"]
ITEM_KO = {"umbrella": "우산", "keyboard": "키보드", "mouse": "마우스",
           "headphones": "헤드폰", "backpack": "백팩", "tumbler": "텀블러"}
ITEM_CODE = {"umbrella": "UMB", "keyboard": "KEY", "mouse": "MOU",
             "headphones": "HPH", "backpack": "BAG", "tumbler": "TUM"}
_PRICE_LEVELS = {"umbrella": [9000, 15000, 24000], "keyboard": [35000, 59000, 89000],
                 "mouse": [18000, 32000, 49000], "headphones": [45000, 79000, 129000],
                 "backpack": [32000, 48000, 69000], "tumbler": [12000, 19000, 27000]}


def _to_min(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _to_hm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def free_slots(events, min_len=60):
    """Gaps of at least `min_len` minutes between DAY_START and DAY_END."""
    busy = sorted((_to_min(s), _to_min(e)) for s, e in events)
    cur, end, out = _to_min(DAY_START), _to_min(DAY_END), []
    for s, e in busy:
        if s - cur >= min_len:
            out.append((cur, s))
        cur = max(cur, e)
    if end - cur >= min_len:
        out.append((cur, end))
    return [{"start": _to_hm(a), "end": _to_hm(b)} for a, b in out]


def weather_of(city, date):
    cond = CONDITIONS[_h("weather", city, date) % 3]
    return {"city": city, "date": date, "condition": cond,
            "temperature_c": 11 + _h("temp", city, date) % 12,
            "precipitation_chance": {"rain": 80, "cloudy": 30, "sunny": 10}[cond]}


def _build_places():
    suffix, ratings, places = ["Lumen", "Maru", "Haneul"], [4.2, 4.5, 4.8], {}
    for city in CITIES:
        for cat in CATEGORIES:
            rot = _h("rot", city, cat) % 3
            outdoor_top = _h("outdoor", city, cat) % 2 == 0
            for i in range(3):
                pid = f"{CITY_CODE[city]}-{CAT_CODE[cat]}-{i + 1}"
                rating = ratings[(i + rot) % 3]
                if cat == "park":
                    indoor = False
                elif cat == "museum":
                    indoor = True
                else:  # the top-rated cafe/restaurant is sometimes an outdoor terrace
                    indoor = not (outdoor_top and rating == 4.8)
                price = {"cafe": 5000 + 1000 * (_h("p", pid) % 8),
                         "restaurant": 15000 + 5000 * (_h("p", pid) % 7),
                         "museum": 3000 * (_h("p", pid) % 5),
                         "park": 0}[cat]
                places[pid] = {"place_id": pid, "name": f"{suffix[i]} {cat.title()} {city}",
                               "city": city, "category": cat, "indoor": indoor,
                               "rating": rating, "price_per_person": price}
    return places


def _build_products():
    brands, ratings, products = ["Nova", "Orbit", "Pine"], [4.1, 4.4, 4.7], {}
    for item in ITEMS:
        rr, rp = _h("pr", item) % 3, _h("pp", item) % 3
        for i in range(3):
            pid = f"P-{ITEM_CODE[item]}-{i + 1}"
            products[pid] = {"product_id": pid, "name": f"{brands[i]} {item.title()}",
                             "item": item, "price": _PRICE_LEVELS[item][(i + rp) % 3],
                             "rating": ratings[(i + rr) % 3]}
    return products


PLACES = _build_places()
PRODUCTS = _build_products()
_PLACE_BY_NAME = {p["name"].lower(): pid for pid, p in PLACES.items()}
_PRODUCT_BY_NAME = {p["name"].lower(): pid for pid, p in PRODUCTS.items()}


def route_of(origin, pid, mode="transit"):
    base = 12 + _h("route", origin, pid) % 38
    if mode == "taxi":
        dur = max(5, round(base * 0.6))
        cost = int(round((4800 + dur * 300) / 100) * 100)
    else:
        dur, cost = base, 1500
    return {"origin": origin, "destination": pid, "destination_name": PLACES[pid]["name"],
            "mode": mode, "duration_min": dur, "cost_krw": cost}


# ---------------------------------------------------------------- normalizers
def _clean(x):
    return re.sub(r"\s+", " ", str(x)).strip().lower()


_CITY_ALIASES = {c.lower(): c for c in CITIES}
_CITY_ALIASES.update({ko: en for en, ko in CITY_KO.items()})
_CITY_ALIASES.update({"서울시": "Seoul", "부산시": "Busan", "대전시": "Daejeon", "광주시": "Gwangju",
                      "제주도": "Jeju", "제주시": "Jeju", "jeju island": "Jeju", "seoul city": "Seoul"})


def norm_city(x):
    s = _clean(x).split(",")[0].strip()
    for suf in ["특별시", "광역시", "특별자치도", "특별자치시"]:
        s = s.replace(suf, "")
    return _CITY_ALIASES.get(s.strip())


_CAT_ALIASES = {}
for _k, _v in {"cafe": ["cafe", "café", "coffee", "coffee shop", "카페", "커피숍"],
               "restaurant": ["restaurant", "식당", "레스토랑", "음식점", "맛집"],
               "museum": ["museum", "박물관", "미술관", "gallery"],
               "park": ["park", "공원"]}.items():
    for _a in _v:
        _CAT_ALIASES[_a] = _k


def norm_category(x):
    s = _clean(x)
    if s in _CAT_ALIASES:
        return _CAT_ALIASES[s]
    if s.endswith("s") and s[:-1] in _CAT_ALIASES:
        return _CAT_ALIASES[s[:-1]]
    for a, k in _CAT_ALIASES.items():
        if a in s:
            return k
    return None


_ORIGIN_ALIASES = {"home": "home", "my home": "home", "house": "home", "집": "home", "우리집": "home",
                   "우리 집": "home", "내 집": "home", "office": "office", "my office": "office",
                   "work": "office", "company": "office", "회사": "office", "사무실": "office", "직장": "office"}


def norm_origin(x):
    return _ORIGIN_ALIASES.get(_clean(x))


_MODE_ALIASES = {"transit": "transit", "public transit": "transit", "public transportation": "transit",
                 "public transport": "transit", "bus": "transit", "subway": "transit", "metro": "transit",
                 "대중교통": "transit", "지하철": "transit", "버스": "transit",
                 "taxi": "taxi", "cab": "taxi", "car": "taxi", "택시": "taxi"}


def norm_mode(x):
    return _MODE_ALIASES.get(_clean(x))


def norm_date(x):
    s = str(x).strip()
    m = re.search(r"(\d{4})\s*[-/.년]\s*(\d{1,2})\s*[-/.월]\s*(\d{1,2})", s)
    if m:
        return f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"
    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", s)
    if m:
        return f"2026-{int(m[1]):02d}-{int(m[2]):02d}"
    return None


def norm_time(x):
    s = _clean(x)
    m = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)?", s)
    if m:
        h, mi, ap = int(m[1]), int(m[2]), m[3]
    else:
        m = re.search(r"(오전|오후)?\s*(\d{1,2})\s*시\s*(?:(\d{1,2})\s*분)?", s)
        if m:
            h = int(m[2])
            mi = int(m[3]) if m[3] else (30 if "반" in s else 0)
            ap = {"오전": "am", "오후": "pm"}.get(m[1])
        else:
            m = re.fullmatch(r"(\d{1,2})\s*(am|pm)?", s)
            if not m:
                return None
            h, mi, ap = int(m[1]), 0, m[2]
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    if not (0 <= h < 24 and 0 <= mi < 60):
        return None
    return f"{h:02d}:{mi:02d}"


def norm_int(x):
    if isinstance(x, bool):
        return None
    try:
        return int(round(float(str(x).replace(",", "").strip())))
    except ValueError:
        m = re.search(r"\d+", str(x))
        return int(m[0]) if m else None


def norm_bool(x):
    if isinstance(x, bool):
        return x
    s = _clean(x)
    if s in ("true", "yes", "1", "y"):
        return True
    if s in ("false", "no", "0", "n"):
        return False
    return None


def norm_place(x):
    s = str(x).strip()
    if s.upper() in PLACES:
        return s.upper()
    return _PLACE_BY_NAME.get(s.lower())


def norm_pid(x):
    s = str(x).strip()
    if s.upper() in PRODUCTS:
        return s.upper()
    return _PRODUCT_BY_NAME.get(s.lower())


_ITEM_ALIASES = {}
for _it in ITEMS:
    _ITEM_ALIASES[_it] = _it
    _ITEM_ALIASES[ITEM_KO[_it]] = _it
_ITEM_ALIASES.update({"headphone": "headphones", "헤드셋": "headphones", "가방": "backpack",
                      "배낭": "backpack", "bag": "backpack", "보온병": "tumbler", "컵": "tumbler"})


def norm_item(x):
    s = _clean(x)
    if s in _ITEM_ALIASES:
        return _ITEM_ALIASES[s]
    if s.endswith("s") and s[:-1] in _ITEM_ALIASES:
        return _ITEM_ALIASES[s[:-1]]
    for a, k in _ITEM_ALIASES.items():
        if a in s:
            return k
    return None


def norm_text(x):
    s = str(x).strip()
    return s or None


# required / optional arguments and their normalizers
TOOL_SPECS = {
    "get_calendar": {"required": {"date": norm_date}, "optional": {}},
    "get_weather": {"required": {"city": norm_city, "date": norm_date}, "optional": {}},
    "search_place": {"required": {"city": norm_city, "category": norm_category},
                     "optional": {"indoor": norm_bool}},
    "get_route": {"required": {"origin": norm_origin, "destination": norm_place},
                  "optional": {"mode": norm_mode}},
    "get_budget": {"required": {}, "optional": {}},
    "book_place": {"required": {"place_id": norm_place, "date": norm_date, "time": norm_time,
                                "people": norm_int}, "optional": {}},
    "create_event": {"required": {"date": norm_date, "start_time": norm_time, "duration_min": norm_int,
                                  "title": norm_text},
                     "optional": {"place_id": norm_place, "transport": norm_mode}},
    "search_product": {"required": {"query": norm_item}, "optional": {"max_price": norm_int}},
    "add_to_cart": {"required": {"product_id": norm_pid}, "optional": {"quantity": norm_int}},
    "get_cart": {"required": {}, "optional": {}},
}
DEFAULTS = {"get_route": {"mode": "transit"}, "add_to_cart": {"quantity": 1}}
TOOL_NAMES = list(TOOL_SPECS)


def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


_S = lambda d: {"type": "string", "description": d}
_I = lambda d: {"type": "integer", "description": d}

TOOL_SCHEMAS = [
    _fn("get_calendar", "Get the user's calendar for a date. Returns events and free_slots "
        "(gaps of at least 60 minutes between 09:00 and 21:00, in chronological order).",
        {"date": _S("Date in YYYY-MM-DD")}, ["date"]),
    _fn("get_weather", "Get the weather forecast for a city on a date. condition is one of sunny, cloudy, rain.",
        {"city": _S("City name, e.g. Seoul"), "date": _S("Date in YYYY-MM-DD")}, ["city", "date"]),
    _fn("search_place", "Search places in a city by category. Returns place_id, name, indoor, rating, "
        "price_per_person (KRW).",
        {"city": _S("City name"), "category": {"type": "string", "enum": CATEGORIES},
         "indoor": {"type": "boolean", "description": "If true, return only indoor places"}},
        ["city", "category"]),
    _fn("get_route", "Get travel time and cost from the user's origin to a place.",
        {"origin": {"type": "string", "enum": ORIGINS},
         "destination": _S("place_id of the destination"),
         "mode": {"type": "string", "enum": ["transit", "taxi"], "description": "Default transit"}},
        ["origin", "destination"]),
    _fn("get_budget", "Get the user's remaining budget in KRW.", {}, []),
    _fn("book_place", "Make a reservation at a place.",
        {"place_id": _S("place_id"), "date": _S("YYYY-MM-DD"), "time": _S("HH:MM"),
         "people": _I("Number of people")}, ["place_id", "date", "time", "people"]),
    _fn("create_event", "Add an event to the user's calendar.",
        {"date": _S("YYYY-MM-DD"), "start_time": _S("HH:MM"), "duration_min": _I("Duration in minutes"),
         "title": _S("Event title"), "place_id": _S("Optional place_id"),
         "transport": {"type": "string", "enum": ["transit", "taxi"], "description": "Optional transport mode"}},
        ["date", "start_time", "duration_min", "title"]),
    _fn("search_product", "Search products. Returns product_id, name, price (KRW), rating.",
        {"query": _S("Product type, e.g. umbrella"), "max_price": _I("Optional maximum price")}, ["query"]),
    _fn("add_to_cart", "Add a product to the shopping cart.",
        {"product_id": _S("product_id"), "quantity": _I("Default 1")}, ["product_id"]),
    _fn("get_cart", "Show the shopping cart and total price.", {}, []),
]


# ---------------------------------------------------------------- world
class World:
    def __init__(self, budget=100000):
        self.budget = budget
        self.reservations, self.events, self.cart = [], [], []

    def call(self, name, args):
        """Returns (result, status). status: ok | tool_error (valid format, bad value) | invalid."""
        if name not in TOOL_SPECS:
            return {"error": f"Unknown tool '{name}'. Available: {', '.join(TOOL_NAMES)}"}, "invalid"
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}, "invalid"
        spec = TOOL_SPECS[name]
        missing = [k for k in spec["required"] if args.get(k) in (None, "")]
        if missing:
            return {"error": f"missing required argument(s): {', '.join(missing)}"}, "invalid"
        norm = {}
        for k, f in {**spec["required"], **spec["optional"]}.items():
            if args.get(k) not in (None, ""):
                v = f(args[k])
                if v is None:
                    return {"error": f"invalid value for '{k}': {args[k]!r}"}, "tool_error"
                norm[k] = v
        return getattr(self, "_" + name)(**norm)

    # --- tools
    def _get_calendar(self, date):
        evs = list(CALENDAR.get(date, [])) + [(e["title"], e["start_time"], _to_hm(
            _to_min(e["start_time"]) + e["duration_min"])) for e in self.events if e["date"] == date]
        evs.sort(key=lambda e: e[1])
        return {"date": date, "events": [{"title": t, "start": s, "end": e} for t, s, e in evs],
                "free_slots": free_slots([(s, e) for _, s, e in evs])}, "ok"

    def _get_weather(self, city, date):
        return weather_of(city, date), "ok"

    def _search_place(self, city, category, indoor=None):
        res = [dict(p) for p in PLACES.values() if p["city"] == city and p["category"] == category
               and (indoor is None or p["indoor"] == indoor)]
        return {"results": res}, "ok"

    def _get_route(self, origin, destination, mode="transit"):
        return route_of(origin, destination, mode), "ok"

    def _get_budget(self):
        return {"remaining_budget_krw": self.budget}, "ok"

    def _book_place(self, place_id, date, time, people):
        if people < 1:
            return {"error": "people must be >= 1"}, "tool_error"
        self.reservations.append({"place_id": place_id, "date": date, "time": time, "people": people})
        return {"status": "confirmed", "reservation_id": f"R-{len(self.reservations)}",
                "place_name": PLACES[place_id]["name"], "date": date, "time": time, "people": people}, "ok"

    def _create_event(self, date, start_time, duration_min, title, place_id=None, transport=None):
        if duration_min < 1:
            return {"error": "duration_min must be >= 1"}, "tool_error"
        self.events.append({"date": date, "start_time": start_time, "duration_min": duration_min,
                            "title": title, "place_id": place_id, "transport": transport})
        return {"status": "created", "event_id": f"E-{len(self.events)}", "date": date,
                "start_time": start_time, "duration_min": duration_min}, "ok"

    def _search_product(self, query, max_price=None):
        res = [dict(p) for p in PRODUCTS.values() if p["item"] == query
               and (max_price is None or p["price"] <= max_price)]
        return {"results": res}, "ok"

    def _add_to_cart(self, product_id, quantity=1):
        if quantity < 1:
            return {"error": "quantity must be >= 1"}, "tool_error"
        self.cart.append({"product_id": product_id, "quantity": quantity})
        return {"status": "added", "product_id": product_id, "quantity": quantity,
                "cart_total_krw": self._total()}, "ok"

    def _get_cart(self):
        items = [{"product_id": c["product_id"], "name": PRODUCTS[c["product_id"]]["name"],
                  "price": PRODUCTS[c["product_id"]]["price"], "quantity": c["quantity"]} for c in self.cart]
        return {"items": items, "total_krw": self._total()}, "ok"

    def _total(self):
        return sum(PRODUCTS[c["product_id"]]["price"] * c["quantity"] for c in self.cart)

    def state(self):
        return {"reservations": list(self.reservations), "events": list(self.events), "cart": list(self.cart)}
