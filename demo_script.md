# Live demo script: "Same strategy, same data, two very different answers"

**Runtime:** about 8–10 minutes. **Audience:** anyone, technical or not.
**The one idea to land:** a backtest can look great purely because it cheated
by peeking at the future. This project makes that cheating structurally
impossible, and proves it.

The demo runs in the app. Each step has a **Do** (what you click or type), a
**Show** (what's on screen) and a **Say** (talking points, in your own words).
A command-line version of the same demo is in the appendix.

---

## Step 0: Prep (before the audience arrives)

**Do:**

```bash
cd backtesting
source .venv/bin/activate
pytest -q -m "not network"     # everything green before you start
streamlit run app.py           # http://localhost:8501
```

No data download is needed. The app runs on the Bitcoin (BTCUSDT) 2020–2024
daily sample committed in `data/sample/` (from Binance Vision, CC BY-NC-SA 4.0,
see `data/sample/LICENSE.md`), so it works offline and always produces exactly
the numbers and fingerprint (`a58795bacb31c078`) shown below.

**Optional, for a clean Runs page:** move old runs aside, don't delete them:
`mkdir -p ~/mlflow-backup && mv mlflow.db mlruns ~/mlflow-backup/`.

Have `runner.py` and `broken_runner.py` open in your editor for Step 4.

> **If something goes wrong on stage**
> - *No internet?* Not a problem: the app reads the committed sample.
> - *"No stored data" error?* Open **Data & costs** and choose "Committed sample", or check you started the app from the project root.
> - *Port 8501 busy?* Add `--server.port 8502` and open that port instead.

---

## Step 1: The hook (no clicks, ~1 min)

**Show:** the app's Backtest page, or just your face.

**Say:**
- "Almost every impressive trading backtest you'll see online has a bug
  called **look-ahead bias**: somewhere, the simulation lets the strategy see
  information from the future. It's like grading a trader who's allowed to
  read tomorrow's newspaper."
- "It's rarely deliberate. It's usually one line: an off-by-one slice, or
  trading at the same closing price the decision was calculated from.
  It's invisible in the results. You just get a nicer-looking number."
- "This project is a backtesting engine built around one rule: **the strategy
  physically cannot see the future.** Not 'please don't', but *can't*. And I
  kept a deliberately broken copy of the engine so I can show you what the
  difference looks like."

---

## Step 2: An honest backtest (~2 min)

**Do:** open the **Backtest** page. It opens on a finished run of the default
strategy: a moving-average crossover with 10- and 30-day windows, correct engine.

**Show:** the headline strip:

| Total return | CAGR | Sharpe | Max drawdown | Fills |
|---|---|---|---|---|
| +598.1% ($10,000 became $69,810) | +47.5% | 1.09 | −64.3% | 68 |

and under the total return, **−599.5 pts vs buy & hold**. Point at the equity
chart (the dashed line is buy-and-hold) and the drawdown curve below it.

**Say:**
- "The strategy is a textbook moving-average crossover on Bitcoin over five
  years: buy when the 10-day average crosses above the 30-day average, sell
  when it crosses back below."
- "The engine walks forward **one day at a time**. On each day, the strategy
  is handed a table of prices that *ends at today*. Tomorrow's row simply isn't
  in the object it receives."
- "When it decides to buy, the trade happens at **the next day's opening
  price**, because that's the first price a real trader could actually get.
  Every trade pays a fee, the bid-ask spread and market impact. There's no
  'free trading' switch; the engine refuses a zero fee."
- "Result: **+598%**, but with a **64% drawdown** along the way, and about
  **600 points worse** than simply buying and holding Bitcoin. Honest, and not
  flattering."
- If asked about the settings: everything (capital, fee, spread, impact, volume
  cap, date range) is in the **Data & costs** popover.

---

## Step 3: The leak test (~3 min)

**Do:** open the **Leak test** page. It runs the same strategy through both
engines on first visit.

**Show:** the face-off:

| | Total return | Sharpe | Max drawdown |
|---|---|---|---|
| Correct engine | +598.1% | 1.09 | −64.3% |
| **Broken engine** | **+1,285.7%** | **1.40** | −38.8% |

Then the line underneath: *"Same data (`a58795bacb31c078`), same costs. Only
the engine differs."* Then the overlaid equity chart, and **The leak** box on
the right, which shows the one line that differs.

**Say:**
- "Same strategy, same parameters, same data (look, the **same fingerprint**),
  same fees. The only difference is the engine."
- "The broken engine has two tiny bugs, the kind that show up in real
  code: it hands the strategy **one extra day** of prices, and it lets the
  trade happen at **today's close**, the very price the decision was based on."
- "Return **more than doubles**, from +598% to +1,286%. Sharpe goes from 1.09 to
  1.40, and the worst drawdown shrinks from 64% to 39%, so it even looks *safer*.
  And it now **beats buy-and-hold**. If I'd built the broken one first, I'd
  believe I had an edge."
- On the chart, the 2022 crypto bear market: "From the same November 2021 peak,
  the honest run falls **56%** by November 2022, the broken one only **39%**.
  It kept getting out a day early, because it could already see the next down day."

**Do:** switch the strategy to **Breakout** and click **Run both**.

**Show:** +726.1% correct vs **+9,730.1%** broken.

**Say:**
- "Same trick, different strategy: the leak turns +726% into +9,730%, **13 times**
  the honest number. And +9,730% on Bitcoin doesn't *look* impossible, which is
  exactly why you can't catch a leak by eyeballing the results."

---

## Step 4: Where the guarantee lives in the code (~2 min)

**Show:** `runner.py`, the function `history_up_to`. Then `broken_runner.py`,
the two blocks marked `LEAK 1` and `LEAK 2`.

**Say:**
- "This is the only function in the project that builds what a strategy
  sees. It slices rows zero through today, then *checks* that the last row
  really is today, on every single day. If that check ever fails, the run
  stops with a `LookAheadError`."
- "Here's the broken version: `slice(0, t + 2)` instead of `t + 1`. One
  character. That's all it takes."

**Do:**

```bash
pytest tests/test_no_lookahead.py -v
```

**Say:**
- "I don't just claim the engine is safe, I test it. This test runs every
  strategy, then **poisons the future**: every price after some day gets
  replaced with nonsense. Everything the strategy did *before* that day must
  come out identical. The correct engine passes at every cut-off. The broken
  engine gets caught, and there's a test asserting exactly that."

---

## Step 5: The other kind of cheating (~1 min, optional)

**Do:** open the **Sweep** page and click **Run sweep** with the defaults
(short windows 5,10,20; long windows 30,50,100; split date 2023-01-01).

**Show:** the two heatmaps and the line *"Picked by in-sample Sharpe: 5/100.
Out of sample it ranked 3 of 9 (Sharpe 1.59)."*

**Say:**
- "Trying lots of settings and keeping the best one is peeking too: you chose
  it by looking at the results. So the sweep picks the winner on 2020–2022 only,
  then judges it on 2023–2024, which it never saw. The in-sample winner came 3rd
  of 9. The out-of-sample column is the honest one."

---

## Step 6: Every run is recorded (~1 min)

**Do:** open the **Runs** page. Select the correct and the broken run from
Step 3 to overlay their equity curves. The **Engine** filter shows one kind or both.

**Say:**
- "Every run from every page is logged automatically to MLflow: every parameter,
  every metric, the chart, the full trade log, and the fingerprint of the exact
  data. Nothing lives only on a screen."
- "So I can prove, a year from now, exactly what produced a number."

Optional, for a technical audience: open the full MLflow UI
(`mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns`,
then <http://127.0.0.1:5000>), tick the correct and broken runs and click
**Compare**: identical params and `dataset_snapshot_id`, different metrics.

---

## Step 7: Close (~30 sec)

**Say:**
1. "**Structural, not conventional.** The strategy never receives future data,
   so it can't misuse it. A reviewer checks one function, not the whole codebase."
2. "**Honest costs.** Fees, spread and market impact on every trade, trades at
   the next open, and no frictionless mode."
3. "**Reproducible.** Every run is an MLflow experiment tied to an exact data
   fingerprint."

---

## Appendix A: the numbers

BTCUSDT (Binance Vision, CC BY-NC-SA 4.0), 2020-01-01 → 2024-12-31, daily bars, $10,000 start,
fee 0.1%, slippage 5 bps, square-root market impact (coefficient 1.0), 10% volume cap
(the defaults), `dataset_snapshot_id a58795bacb31c078`.

| Run | Total return | CAGR | Sharpe | Max drawdown | Trades |
|---|---|---|---|---|---|
| MA 10/30, correct runner | +598.1% | +47.5% | 1.09 | −64.3% | 68 |
| MA 10/30, **broken runner** | **+1,285.7%** | **+69.2%** | **1.40** | −38.8% | 68 |
| MA 20/50, correct runner | +578.5% | +46.7% | 1.07 | −59.3% | 37 |
| MA 20/50, broken runner | +803.9% | +55.3% | 1.19 | −54.5% | 37 |
| Buy and hold (baseline) | +1,197.6% | +67.0% | 1.12 | −76.6% | 1 |

| Strategy (defaults) | Correct | Broken |
|---|---|---|
| Momentum 90 | +836.3% (Sharpe 1.20) | +3,866.4% (1.84) |
| Breakout 20/10 | +726.1% (1.18) | +9,730.1% (2.34) |
| RSI 14/30/70 | −13.4% (0.18) | −77.7% (−0.40) |

These are exact for the committed sample (`data/sample/`). A fresh
`fetch_data` download of the same range comes from the same Vision archives
and gives the same fingerprint.

## Appendix B: the same demo from the command line

```bash
# Step 2: correct engine
python -m scripts.run_backtest --source binance --symbol BTCUSDT \
    --start 2020-01-01 --end 2024-12-31 \
    --strategy moving_average_crossover --param short_window=10 --param long_window=30 \
    --data-root data/sample

# Step 3: the identical command through the broken engine
python -m scripts.run_backtest --source binance --symbol BTCUSDT \
    --start 2020-01-01 --end 2024-12-31 \
    --strategy moving_average_crossover --param short_window=10 --param long_window=30 \
    --data-root data/sample --runner broken

# Step 5: the honest sweep
python -m scripts.sweep --source binance --symbol BTCUSDT \
    --start 2020-01-01 --end 2024-12-31 --data-root data/sample \
    --split-date 2023-01-01 --short 5,10,20 --long 30,50,100
```

Expected output of the first command:

```
[correct runner] moving_average_crossover on BTCUSDT 2020-01-01 → 2024-12-31
  total return +598.1% | CAGR +47.5% | Sharpe 1.09 | max drawdown -64.3% | 68 trades
  vs buy-and-hold: return +1197.6%, Sharpe 1.12 | excess return -599.5%
  dataset_snapshot_id a58795bacb31c078 | MLflow run <id>
```

The broken one prints `[BROKEN RUNNER (look-ahead leak)]`, +1285.7%, Sharpe 1.40,
and an excess return of +88.2% over buy-and-hold.
