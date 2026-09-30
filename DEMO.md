# BuddySync trip planner: demo script (3 minutes)

**The pitch in one line:** travel apps plan trips; none plans around *the show*. BuddySync knows the gate time and when the show ends, gets you back safely late at night, and plans for your whole buddy group.

---

## Before you present (30 min ahead)

1. **Fresh Gemini key.** The free tier allows 20 requests per model per day, and rehearsals use it up. Create a new key in a **new** Google Cloud project at aistudio.google.com.
2. In PowerShell, from `ml/`:
   ```powershell
   . .\demo_env.ps1                          # your keys (copy demo_env.example.ps1 and fill it in)
   python -m trip_planner.demo --days 7      # demo event 7 days out; every line must say READY
   python -m uvicorn app:app --port 8765     # leave running
   ```
   In a second window: `. .\demo_env.ps1; python -m streamlit run playground.py`
3. Open **two browser tabs** on http://127.0.0.1:8501, both on the **Trip planner** tab.
   - **Tab A (solo):** origin `Pune`, leave the rest.
   - **Tab B (clan):** origin `Delhi`, clan members `8:Bengaluru, 9:Mumbai`.
   - The event dropdown starts on **Midnight Echoes Live at NSCI Dome** (show ends 22:45).
4. Don't run `evals` or rehearse chat after this point: it spends the free quota.

## On stage

| Time | Do | Say |
|---|---|---|
| 0:00 | (nothing) | The pitch line above |
| 0:15 | **Tab A → Start new trip.** It appears instantly. | "One tap: the whole trip, built around the show." |
| 0:20 | Scroll the cards | **Train Pune → Mumbai ₹510–560**, estimated from the Railway Board's official fare chart. **Hotel at a live rate** from LiteAPI, with Booking.com and Agoda links. **Dinner after the show**, skipping places whose listed hours say they're closed by 22:45. The next day's places are on the map. |
| 0:35 | Point at the warning | "The show ends at 22:45, but Mumbai's last metro is around 22:30. So it plans a **cab back** and gives you an Uber link. Nobody else does that." |
| 0:45 | **Tab A: type in chat** `Ek din aur rukna hai, aur sirf veg khana` and press Enter | "It speaks Hinglish. It edits the plan with tools, and it can only pick **real places** from our catalog. Code checks every change. It takes about 5–10 seconds." (While it works, move to Tab B.) |
| 1:00 | **Tab B → Start new trip.** Instant. | "A clan of three. Each person gets their own journey: **Delhi flies** (a live fare, with our affiliate link), **Bengaluru** travels too, and **Sana lives in Mumbai**, so she has no leg. Costs are split per person." |
| 1:20 | Point at the 🗳️ vote card | "The group votes on budget, balanced or premium. Everyone's equal: a majority decides straight away, otherwise the most votes win after 24 hours." |
| 1:30 | Vote **budget** as 7; change *As user_id* to **8**; vote **budget** | "Two of three, so it passes, and the hotel switches to the budget option." (Wait about 2 s.) |
| 1:50 | 👍 on the hotel; point at 🔒 if you locked a slot | "Thumbs up per place, and anything the group locks can't be changed, not even by the AI, without another vote." |
| 2:10 | **Back to Tab A:** the Hinglish reply is there | Read one line of it. "It stays one more day and it's veg only; every edit is a new version, so undo is one tap." |
| 2:30 | (close) | The numbers below |

**Numbers worth quoting:**
- 5 cities, with **1,100–3,900 real hotels each** (LiteAPI) plus OpenStreetMap places.
- Live **flight fares** (Travelpayouts), **train fares** from the official 2026 fare chart, weather, and the last metro per city.
- **1,990 plans tested automatically**: 0 invented places, and 0 broken rules.
- Group planning: votes, locks and a per-person cost split. Hindi and Hinglish chat, and voice input.
- We **never book or take payment**. We hand off to booking sites and earn affiliate commission.

## If something goes wrong

| Problem | Say or do |
|---|---|
| The chat reply is still "Thinking…" after about 30 s | All three free Groq models are rate-limited, so it has fallen back to Gemini. Say "it's on a free AI tier today", and move on. |
| The chat shows an error | Same line, and move on. Don't retry on stage. |
| A plan says `auto` instead of `cache:…` | Still fine: that's the code-only planner, the no-AI fallback, and it's complete. |
| No prices on the cards | The demo event date has passed. Re-run `python -m trip_planner.demo --days 7` and click Start new trip again. |
| The playground says "API is down" | Check that the uvicorn window is still running (port 8765). |

Keep judges' questions off the keyboard if you can: a long prompt costs a lot of free-tier tokens and can take minutes.
