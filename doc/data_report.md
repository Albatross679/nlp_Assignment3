# Data Exploration Report

## Dataset Overview

This is the **ATIS (Airline Travel Information System)** benchmark dataset — a classic NL-to-SQL dataset focused on air travel queries against a relational flight database.

| Split | NL Lines | SQL Lines | Purpose |
|-------|----------|-----------|---------|
| **train** | 4,225 | 4,225 | Training pairs |
| **dev** | 466 | 466 | Development/validation pairs |
| **test** | 432 | — | Held-out evaluation (no ground-truth SQL) |

---

## 1. Natural Language Inputs (`.nl` files)

### Sample Queries (train.nl)

```
list all the flights that arrive at general mitchell international from various cities
give me the flights leaving denver august ninth coming back to boston
what flights from tacoma to orlando on saturday
what is the most expensive one way fare from boston to atlanta on american airlines
what flights return from denver to philadelphia on a saturday
can you list all round trip flights from orlando to kansas city and then to minneapolis
```

### Token Length Statistics (whitespace-split)

| File | Lines | Avg Tokens | Min | Max |
|------|-------|------------|-----|-----|
| train.nl | 4,225 | 10.96 | 1 | 42 |
| dev.nl | 466 | 10.91 | 3 | 30 |
| test.nl | 432 | 9.41 | 2 | 22 |

- NL questions are short and conversational (~10-11 tokens on average)
- Test set is slightly shorter on average (9.41) and lower max (22 vs 42)
- Some single-word inputs exist in training (min=1)

### NL Question Style

**Top first words:** "what" (1,000), "show" (812), "i" (460), "list" (229), "please" (189)

**Top 3-word starters:**
| Starter | Count |
|---------|-------|
| "show me the" | 326 |
| "what is the" | 252 |
| "i would like" | 191 |
| "show me all" | 190 |
| "what are the" | 135 |

The language is conversational, informal, and often imperative.

### NL Intent Categories (heuristic)

| Intent | Count | % |
|--------|-------|---|
| Flight queries | 2,607 | 61.7% |
| Fare/cost queries | 597 | 14.1% |
| Airline queries | 377 | 8.9% |
| Ground transport | 222 | 5.3% |
| Other | 267 | 6.3% |
| Aircraft | 73 | 1.7% |
| Airport | 64 | 1.5% |
| Meal | 11 | 0.3% |
| City | 7 | 0.2% |

### Most Referenced Cities in NL

Boston (895), Denver (826), San Francisco (716), Atlanta (615), Pittsburgh (549), Dallas (508), Baltimore (497), Philadelphia (473), Washington (286), Oakland (206).

---

## 2. SQL Queries (`.sql` files)

### Sample Query

```sql
SELECT DISTINCT flight_1.flight_id
FROM flight flight_1 , airport_service airport_service_1 , city city_1 ,
     airport_service airport_service_2 , city city_2
WHERE flight_1.from_airport = airport_service_1.airport_code
  AND airport_service_1.city_code = city_1.city_code
  AND city_1.city_name = 'DENVER'
  AND flight_1.to_airport = airport_service_2.airport_code
  AND airport_service_2.city_code = city_2.city_code
  AND city_2.city_name = 'PHILADELPHIA'
```

### Token Length Statistics

| File | Lines | Avg Tokens | Min | Max |
|------|-------|------------|-----|-----|
| train.sql | 4,225 | 60.90 | 10 | 158 |
| dev.sql | 466 | 58.90 | 9 | 146 |

SQL queries are roughly **6x longer** than their NL counterparts.

### Unique vs Duplicate Queries

- **Total SQL queries:** 4,225
- **Unique SQL queries:** 2,826
- **Duplicate queries:** 1,399 (33.1% are duplicates)
- **SQL queries with multiple NL paraphrases:** 595
- **Max paraphrases for a single SQL query:** 29

The same SQL is expressed using many different NL phrasings — this is the core paraphrase supervision in the dataset.

### SQL Keyword Distribution

| Keyword | Count | Notes |
|---------|-------|-------|
| AND | 34,357 | Very heavy use of conjunctive conditions |
| SELECT | 4,645 | Includes subqueries |
| FROM | 4,645 | |
| WHERE | 4,645 | |
| DISTINCT | 4,229 | Nearly universal |
| BETWEEN | 443 | Time ranges |
| MIN | 334 | Cheapest/earliest queries |
| OR | 264 | |
| NOT | 183 | |
| IS NULL | 108 | |
| MAX | 85 | Most expensive/latest queries |
| COUNT | 46 | |
| LIKE | 12 | |
| GROUP BY | 9 | |
| **JOIN** | **0** | Never used — all implicit joins |
| **ORDER BY** | **0** | Never used |

### SELECT Targets (Top 10)

| Count | Target |
|-------|--------|
| 3,094 | `flight_1.flight_id` |
| 311 | `fare_1.fare_id` |
| 227 | `ground_service_1.transport_type` |
| 154 | `airline_1.airline_code` |
| 75 | `aircraft_1.aircraft_code` |
| 42 | `fare_basis_1.fare_basis_code` |
| 37 | `airport_1.airport_code` |
| 30 | `count(DISTINCT flight_1.flight_id)` |
| 30 | `flight_1.departure_time` |
| 29 | `state_1.state_code` |

**73.2% of queries select `flight_id`** — the dominant task is finding flights matching criteria.

### Query Complexity

| Feature | Count | % |
|---------|-------|---|
| Subqueries (nested SELECT) | 418 | 9.9% |
| OR clauses | 196 | 4.6% |
| BETWEEN | 410 | 9.7% |
| NOT | 181 | 4.3% |
| GROUP BY | 9 | 0.2% |
| ORDER BY | 0 | 0.0% |

### "ASD" Dummy Queries

- **train.sql:** 29 instances, **dev.sql:** 5 instances
- These are placeholder queries: `SELECT DISTINCT state_1.state_code FROM state state_1 WHERE state_1.state_code = 'ASD'`
- Represent "unanswerable" or out-of-scope NL questions

---

## 3. Key Structural Patterns in SQL

1. **All SQL uses implicit joins** — comma-separated tables in FROM + join conditions in WHERE. Never explicit `JOIN...ON` syntax.
2. **Table aliasing is systematic** — every table gets a `_1` suffix (e.g., `flight flight_1`, `city city_1`).
3. **City lookup always routes through airport_service** — the universal pattern:
   ```sql
   flight_1.from_airport = airport_service_1.airport_code
   AND airport_service_1.city_code = city_1.city_code
   AND city_1.city_name = 'CITYNAME'
   ```
4. **Time stored as integers** — e.g., 1800 = 6:00 PM. Time constraints use direct comparisons or BETWEEN.
5. **Day/date filtering** routes through the `days` and `date_day` tables via `flight_days = days_code` joins.

---

## 4. Alignment File (`alignment.txt`)

Maps natural language phrases to canonical forms (13 mappings):

| NL Phrase | Canonical Form |
|-----------|---------------|
| general mitchell international airport | mke |
| general mitchell international | mke |
| baltimore washington airport | bwi |
| rental cars | rental car |
| car rentals | rental car |
| orlando international airport | mco |
| orlando international | mco |
| love field | dal |
| dallas forth worth | dfw |
| la guardia airport | lga |
| la guardia | lga |
| la | los angeles |
| supper | dinner |

Used for normalizing airport names to codes and resolving NL synonyms.

---

## 5. Database Schema (`flight_database.db`)

**25 tables, ~161,441 total rows.**

### All Tables with Row Counts

| Table | Rows | Description |
|-------|------|-------------|
| **flight** | 23,457 | Core flight records |
| **flight_fare** | 67,230 | Flight-to-fare mapping |
| **flight_leg** | 37,021 | Multi-leg flight segments |
| **flight_stop** | 13,564 | Intermediate stops |
| **fare** | 16,252 | Fare pricing |
| **date_day** | 2,557 | Calendar lookup |
| **equipment_sequence** | 952 | Aircraft sequence mapping |
| **food_service** | 374 | Meal information |
| **days** | 233 | Day-of-week codes |
| **ground_service** | 168 | Ground transportation |
| **airport_service** | 64 | Airport-city mapping |
| **fare_basis** | 60 | Fare class details |
| **airport** | 52 | Airport information |
| **airline** | 46 | Airlines |
| **city** | 46 | Cities |
| **aircraft** | 43 | Aircraft types |
| **state** | 26 | US states |
| **dual_carrier** | 23 | Codeshare flights |
| **class_of_service** | 19 | Booking classes |
| **restriction** | 14 | Fare restrictions |
| **time_interval** | 13 | Named time periods (morning, afternoon, etc.) |
| **month** | 12 | Month lookup |
| **code_description** | 7 | Miscellaneous codes |
| **compartment_class** | 6 | Cabin classes |
| **time_zone** | 4 | Time zones |

### Key Table Schemas

**flight** (23,457 rows) — the central table:
```
flight_id (INT, PK), flight_days (TEXT), from_airport (VARCHAR3),
to_airport (VARCHAR3), departure_time (INT), arrival_time (INT),
airline_flight (TEXT), airline_code (VARCHAR3), flight_number (INT),
aircraft_code_sequence (TEXT), meal_code (TEXT), stops (INT),
connections (INT), dual_carrier (TEXT), time_elapsed (INT)
```

**fare** (16,252 rows):
```
fare_id (INT, PK), from_airport (VARCHAR3), to_airport (VARCHAR3),
fare_basis_code (TEXT), fare_airline (TEXT), restriction_code (TEXT),
one_direction_cost (INT), round_trip_cost (INT), round_trip_required (VARCHAR3)
```

**city** (46 rows):
```
city_code (VARCHAR4), city_name (VARCHAR18), state_code (VARCHAR2),
country_name (VARCHAR6), time_zone_code (VARCHAR3)
```

**airport_service** (64 rows) — maps cities to airports:
```
city_code (VARCHAR4), airport_code (VARCHAR3), miles_distant (INT),
direction (VARCHAR2), minutes_distant (INT)
```

### Foreign Key Relationships (from schema `links`)

- `flight` links to: flight_stop, food_service, flight_fare, airport, airline, flight_leg
- `airport` links to: flight, flight_stop, time_zone, airport_service, state, ground_service
- `city` links to: airport_service, state
- `fare` links to: flight_fare, fare_basis, restriction

### Schema File Structure (`flight_database.schema`)

JSON format with four sections:
- **`types`** — 64 named column type categories (e.g., AIRCRAFTCODE=0, CITYNAME=12, DEPARTURETIME=19)
- **`ents`** — Per-table column definitions with `index` (boolean), `type` (category), `utt` (NL utterance)
- **`defaults`** — Default column and NL name per table (e.g., flight -> flight_id / "flight")
- **`links`** — Foreign key relationships between tables

---

## 6. Summary of Key Takeaways for Modeling

1. **Input is short (~11 tokens), output is long (~60 tokens)** — the model must generate significantly more tokens than it receives.
2. **73% of queries are flight lookups** — the model should excel at the `SELECT flight_id FROM flight...` pattern.
3. **33% of training SQL is duplicated** — multiple NL phrasings map to the same SQL, providing paraphrase robustness.
4. **No explicit JOINs, no ORDER BY** — the SQL dialect is restricted, which simplifies the generation target.
5. **Systematic aliasing (`_1` suffix)** and the **city-through-airport_service pattern** are highly regular and learnable.
6. **29 "ASD" dummy queries** in training represent unanswerable questions — the model may need to learn to output these.
7. **The alignment file** provides 13 critical NL-to-canonical mappings that preprocessing should handle.
8. **Time is integer-encoded** (e.g., 1800 = 6pm) — the model must learn this conversion from NL ("before noon" -> `< 1200`).
