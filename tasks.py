"""60 tasks = 4 domains x 5 instances x 3 horizons (1 / 3 / 5+ steps).

Gold trajectories and expected final states are computed by an oracle from the
same mock world, so they never disagree with the tool outputs.
Usage: python tasks.py --lang ko --out tasks_ko.json
"""
import argparse
import json

from tools_env import (CALENDAR, CITIES, CITY_KO, DATES, ITEMS, ITEM_KO, ORIGINS, ORIGIN_KO, PLACES,
                       PRODUCTS, CATEGORY_KO, free_slots, route_of, weather_of)

COND_RANK = {"sunny": 0, "cloudy": 1, "rain": 2}


def gc(name, derived=(), extra=None, **args):
    """Gold call. args = key arguments that are checked; derived = args that depend on earlier observations."""
    return {"name": name, "args": args, "derived": list(derived), "extra": extra or {}}


def search(city, cat, indoor=None):
    return [p for p in PLACES.values() if p["city"] == city and p["category"] == cat
            and (indoor is None or p["indoor"] == indoor)]


def top(items):
    return max(items, key=lambda p: p["rating"])


def first_slot(date):
    return free_slots([(s, e) for _, s, e in CALENDAR[date]])[0]["start"]


def tv(t):
    """Acceptable spellings of a time in the final answer."""
    h, m = map(int, t.split(":"))
    v = [t, f"{h}:{m:02d}"]
    v += [f"{h}시"] if m == 0 else [f"{h}시 {m}분", f"{h}시{m}분"]
    if h > 12:
        v += [f"{h - 12}:{m:02d}", f"오후 {h - 12}시", f"{h - 12} pm", f"{h - 12}pm"]
    return v


def nv(n):
    v = [str(n)]
    if n >= 10000 and n % 1000 == 0:
        v.append(f"{n / 10000:g}만")
    return v


def task(tid, domain, h, q, gold, budget=100000, state=None, answer=None):
    st = {"reservations": [], "events": [], "cart": []}
    st.update(state or {})
    return {"id": tid, "domain": domain, "horizon": h, "query": q, "world_params": {"budget": budget},
            "gold": gold, "expected_state": st, "answer": answer or []}


def build_tasks(lang="ko"):
    ko = lang == "ko"
    T = []
    for i in range(5):
        # ------------------------------------------------ A. schedule / reservation
        date, city, people = DATES[i], CITIES[i], [2, 3, 4, 2, 3][i]
        ck = CITY_KO[city] if ko else city
        first = CALENDAR[date][0][1]
        T.append(task(f"schedule-h1-{i+1}", "schedule", 1,
                      f"{date} 내 일정을 알려줘. 첫 일정은 몇 시에 시작해?" if ko else
                      f"What is on my calendar for {date}? What time does my first event start?",
                      [gc("get_calendar", date=date)], answer=[tv(first)]))

        slot = first_slot(date)
        best = top(search(city, "restaurant"))["place_id"]
        T.append(task(f"schedule-h3-{i+1}", "schedule", 3,
                      f"{date} 내 일정을 확인하고, {ck}에서 평점이 가장 높은 식당을 찾아서 그날 첫 번째 빈 시간대의 "
                      f"시작 시각에 {people}명으로 예약해줘." if ko else
                      f"Check my calendar for {date}, find the highest-rated restaurant in {city}, and book it for "
                      f"{people} people at the start time of my first free slot that day.",
                      [gc("get_calendar", date=date),
                       gc("search_place", city=city, category="restaurant"),
                       gc("book_place", ["place_id", "time"], place_id=best, date=date, time=slot, people=people)],
                      state={"reservations": [{"place_id": best, "date": date, "time": slot, "people": people}]}))

        rain = weather_of(city, date)["condition"] == "rain"
        cands = search(city, "restaurant", True if rain else None)
        budget = people * 30000
        while not [p for p in cands if p["price_per_person"] * people <= budget]:
            budget += people * 5000
        best = top([p for p in cands if p["price_per_person"] * people <= budget])["place_id"]
        sp = gc("search_place", ["indoor"], city=city, category="restaurant", indoor=True) if rain else \
            gc("search_place", city=city, category="restaurant")
        title = "모임 식사" if ko else "Group dinner"
        T.append(task(f"schedule-h5-{i+1}", "schedule", 5,
                      f"{date} 내 일정과 {ck} 날씨를 확인해줘. 비가 오면 실내 식당만 고려해. 남은 예산 안에서 "
                      f"{people}명이 먹을 수 있는(1인 가격 x {people}명) 식당 중 평점이 가장 높은 곳을 골라서, 그날 첫 번째 "
                      f"빈 시간대의 시작 시각에 {people}명으로 예약해줘. 그리고 같은 시각에 '{title}' 일정을 90분으로 "
                      f"캘린더에 추가해줘(장소 포함)." if ko else
                      f"Check my calendar for {date} and the weather in {city}. If it rains, consider only indoor "
                      f"restaurants. Among restaurants that fit my remaining budget for {people} people (price per "
                      f"person x {people}), pick the highest-rated one and book it for {people} people at the start "
                      f"time of my first free slot that day. Then add a 90-minute event '{title}' at that time to my "
                      f"calendar, including the place.",
                      [gc("get_calendar", date=date), gc("get_weather", city=city, date=date), sp, gc("get_budget"),
                       gc("book_place", ["place_id", "time"], place_id=best, date=date, time=slot, people=people),
                       gc("create_event", ["start_time", "place_id"], {"title": title},
                          date=date, start_time=slot, duration_min=90, place_id=best)],
                      budget=budget,
                      state={"reservations": [{"place_id": best, "date": date, "time": slot, "people": people}],
                             "events": [{"date": date, "start_time": slot, "duration_min": 90, "place_id": best}]}))

        # ------------------------------------------------ B. travel / route
        date, city, origin = DATES[(i + 1) % 5], CITIES[(i + 1) % 5], ORIGINS[i % 2]
        ck = CITY_KO[city] if ko else city
        ok = ORIGIN_KO[origin] if ko else f"my {origin}"
        dest = top(search(city, "museum"))
        dur = route_of(origin, dest["place_id"], "transit")["duration_min"]
        T.append(task(f"travel-h1-{i+1}", "travel", 1,
                      f"{ok}에서 {dest['name']}({dest['place_id']})까지 대중교통으로 몇 분 걸려?" if ko else
                      f"How many minutes does it take from {ok} to {dest['name']} ({dest['place_id']}) by public transit?",
                      [gc("get_route", origin=origin, destination=dest["place_id"], mode="transit")],
                      answer=[[str(dur)]]))

        t0 = ["10:00", "13:00", "15:00", "11:00", "14:00"][i]
        best = dest["place_id"]
        title = "박물관 방문" if ko else "Museum visit"
        T.append(task(f"travel-h3-{i+1}", "travel", 3,
                      f"{ck}에서 평점이 가장 높은 박물관을 찾고, {ok}에서 거기까지 대중교통 경로를 확인한 다음, "
                      f"{date} {t0}에 그 박물관 방문 일정을 120분으로 추가해줘(장소 포함)." if ko else
                      f"Find the highest-rated museum in {city}, check the public transit route from {ok} to it, then "
                      f"add a 120-minute visit to that museum on {date} at {t0} to my calendar, including the place.",
                      [gc("search_place", city=city, category="museum"),
                       gc("get_route", ["destination"], origin=origin, destination=best, mode="transit"),
                       gc("create_event", ["place_id"], {"title": title},
                          date=date, start_time=t0, duration_min=120, place_id=best)],
                      state={"events": [{"date": date, "start_time": t0, "duration_min": 120, "place_id": best}]}))

        cat = "museum" if weather_of(city, date)["condition"] == "rain" else "park"
        best = top(search(city, cat))["place_id"]
        budget = [50000, 3000, 50000, 3000, 50000][i]
        mode = "taxi" if route_of(origin, best, "taxi")["cost_krw"] <= budget else "transit"
        title = "나들이" if ko else "Outing"
        T.append(task(f"travel-h5-{i+1}", "travel", 5,
                      f"{date} {ck} 날씨를 확인해서 비가 오면 박물관, 아니면 공원 중 평점이 가장 높은 곳을 골라줘. "
                      f"{ok}에서 거기까지 대중교통 경로와 택시 경로를 모두 확인하고, 남은 예산으로 택시비를 낼 수 있으면 "
                      f"택시, 아니면 대중교통을 선택해. 그리고 {date} {t0}에 그곳 방문 일정을 120분으로 추가해줘. 장소와 "
                      f"선택한 이동수단(transport)도 같이 기록해줘." if ko else
                      f"Check the weather in {city} on {date}. If it rains pick the highest-rated museum, otherwise the "
                      f"highest-rated park. Check both the transit route and the taxi route from {ok} to it. If my "
                      f"remaining budget covers the taxi cost choose taxi, otherwise transit. Then add a 120-minute "
                      f"visit on {date} at {t0} to my calendar, recording the place and the chosen transport.",
                      [gc("get_weather", city=city, date=date),
                       gc("search_place", ["category"], city=city, category=cat),
                       gc("get_route", ["destination"], origin=origin, destination=best, mode="transit"),
                       gc("get_route", ["destination"], origin=origin, destination=best, mode="taxi"),
                       gc("get_budget"),
                       gc("create_event", ["place_id", "transport"], {"title": title}, date=date, start_time=t0,
                          duration_min=120, place_id=best, transport=mode)],
                      budget=budget,
                      state={"events": [{"date": date, "start_time": t0, "duration_min": 120, "place_id": best,
                                         "transport": mode}]}))

        # ------------------------------------------------ C. place / weather
        date, city, origin = DATES[(i + 2) % 5], CITIES[(i + 2) % 5], ORIGINS[(i + 1) % 2]
        ck = CITY_KO[city] if ko else city
        ok = ORIGIN_KO[origin] if ko else f"my {origin}"
        w = weather_of(city, date)
        T.append(task(f"place-h1-{i+1}", "place", 1,
                      f"{date} {ck} 날씨 어때? 기온도 알려줘." if ko else
                      f"What's the weather in {city} on {date}? Tell me the temperature too.",
                      [gc("get_weather", city=city, date=date)], answer=[[str(w["temperature_c"])]]))

        rain = w["condition"] == "rain"
        best = top(search(city, "cafe", True if rain else None))
        sp = gc("search_place", ["indoor"], city=city, category="cafe", indoor=True) if rain else \
            gc("search_place", city=city, category="cafe")
        T.append(task(f"place-h3-{i+1}", "place", 3,
                      f"{date} 내 일정과 {ck} 날씨를 확인하고, 비가 오면 실내 카페 중에서, 아니면 전체 카페 중에서 "
                      f"평점이 가장 높은 곳을 추천해줘." if ko else
                      f"Check my calendar for {date} and the weather in {city}. If it rains, recommend the "
                      f"highest-rated indoor cafe; otherwise the highest-rated cafe overall.",
                      [gc("get_calendar", date=date), gc("get_weather", city=city, date=date), sp],
                      answer=[[best["name"], best["place_id"]]]))

        d1 = date
        d2 = next((d for d in DATES[DATES.index(d1) + 1:] + DATES[:DATES.index(d1)]
                   if weather_of(city, d)["condition"] != w["condition"]), DATES[(DATES.index(d1) + 1) % 5])
        c1, c2 = w["condition"], weather_of(city, d2)["condition"]
        better = d1 if COND_RANK[c1] <= COND_RANK[c2] else d2
        cat = "park" if weather_of(city, better)["condition"] == "sunny" else "museum"
        best = top(search(city, cat))["place_id"]
        slot = first_slot(better)
        title = "외출" if ko else "Day out"
        T.append(task(f"place-h5-{i+1}", "place", 5,
                      f"{ck}의 {d1}과 {d2} 날씨를 비교해서 더 좋은 날(맑음 > 흐림 > 비, 같으면 앞 날짜)을 골라줘. 그날 내 "
                      f"일정을 확인하고, 그날이 맑으면 공원, 아니면 박물관 중 평점이 가장 높은 곳을 찾아. {ok}에서 거기까지 "
                      f"대중교통 경로를 확인한 뒤, 그날 첫 번째 빈 시간대의 시작 시각에 120분짜리 '{title}' 일정을 추가해줘"
                      f"(장소 포함)." if ko else
                      f"Compare the weather in {city} on {d1} and {d2} and pick the better day (sunny > cloudy > rain; "
                      f"if equal, the earlier date). Check my calendar for that day. If that day is sunny find the "
                      f"highest-rated park, otherwise the highest-rated museum. Check the transit route from {ok} to "
                      f"it, then add a 120-minute event '{title}' at the start time of my first free slot that day, "
                      f"including the place.",
                      [gc("get_weather", city=city, date=d1), gc("get_weather", city=city, date=d2),
                       gc("get_calendar", ["date"], date=better),
                       gc("search_place", ["category"], city=city, category=cat),
                       gc("get_route", ["destination"], origin=origin, destination=best, mode="transit"),
                       gc("create_event", ["date", "start_time", "place_id"], {"title": title},
                          date=better, start_time=slot, duration_min=120, place_id=best)],
                      state={"events": [{"date": better, "start_time": slot, "duration_min": 120,
                                         "place_id": best}]}))

        # ------------------------------------------------ D. shopping / budget
        item = ITEMS[i]
        ik = ITEM_KO[item] if ko else item
        cheapest = min(p["price"] for p in PRODUCTS.values() if p["item"] == item)
        T.append(task(f"shopping-h1-{i+1}", "shopping", 1,
                      f"가장 저렴한 {ik}는 얼마야?" if ko else f"How much is the cheapest {item}?",
                      [gc("search_product", query=item)], answer=[nv(cheapest)]))

        item = ITEMS[(i + 1) % 6]
        ik = ITEM_KO[item] if ko else item
        qty = [1, 2, 1, 2, 1][i]
        prods = [p for p in PRODUCTS.values() if p["item"] == item]
        budget = qty * sorted(p["price"] for p in prods)[1]
        best = top([p for p in prods if p["price"] * qty <= budget])["product_id"]
        T.append(task(f"shopping-h3-{i+1}", "shopping", 3,
                      f"남은 예산을 확인하고, 예산 안에서 {qty}개를 살 수 있는 {ik} 중 평점이 가장 높은 것을 {qty}개 "
                      f"장바구니에 담아줘." if ko else
                      f"Check my remaining budget, and add {qty} of the highest-rated {item} that I can afford "
                      f"({qty} units within budget) to my cart.",
                      [gc("get_budget"), gc("search_product", query=item),
                       gc("add_to_cart", ["product_id"], product_id=best, quantity=qty)],
                      budget=budget, state={"cart": [{"product_id": best, "quantity": qty}]}))

        i1, i2 = [("umbrella", "tumbler"), ("keyboard", "mouse"), ("headphones", "backpack"),
                  ("mouse", "tumbler"), ("backpack", "umbrella")][i]
        p1 = [p for p in PRODUCTS.values() if p["item"] == i1]
        p2 = [p for p in PRODUCTS.values() if p["item"] == i2]
        budget = sorted(p["price"] for p in p1)[1] + sorted(p["price"] for p in p2)[1]
        while True:
            b1 = top([p for p in p1 if p["price"] <= budget])
            rest = [p for p in p2 if p["price"] <= budget - b1["price"]]
            if rest:
                b2 = top(rest)
                break
            budget += 5000
        k1, k2 = (ITEM_KO[i1], ITEM_KO[i2]) if ko else (i1, i2)
        T.append(task(f"shopping-h5-{i+1}", "shopping", 5,
                      f"남은 예산을 확인하고 {k1}와 {k2}를 1개씩 사려고 해. 먼저 예산 안에서 평점이 가장 높은 {k1}를 "
                      f"담고, 남은 금액 안에서 평점이 가장 높은 {k2}를 담은 뒤, 장바구니를 확인해서 총액을 알려줘." if ko else
                      f"Check my remaining budget. I want to buy one {i1} and one {i2}. First add the highest-rated "
                      f"{i1} within budget, then the highest-rated {i2} within what is left, then check the cart and "
                      f"tell me the total.",
                      [gc("get_budget"), gc("search_product", query=i1), gc("search_product", query=i2),
                       gc("add_to_cart", ["product_id"], product_id=b1["product_id"], quantity=1),
                       gc("add_to_cart", ["product_id"], product_id=b2["product_id"], quantity=1),
                       gc("get_cart")],
                      budget=budget,
                      state={"cart": [{"product_id": b1["product_id"], "quantity": 1},
                                      {"product_id": b2["product_id"], "quantity": 1}]},
                      answer=[nv(b1["price"] + b2["price"])]))
    T.sort(key=lambda t: (t["horizon"], t["domain"], t["id"]))
    return T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ko", choices=["ko", "en"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tasks = build_tasks(a.lang)
    out = a.out or f"tasks_{a.lang}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"{len(tasks)} tasks -> {out}")
