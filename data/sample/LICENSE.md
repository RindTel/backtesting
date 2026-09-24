# Licence for the data in `data/sample/`

**Data:** BTCUSDT daily klines (spot), 2020-01-01 to 2024-12-31, from
**Binance Vision** (<https://data.binance.vision>), provided under the
[Binance Vision Dataset Terms v1.0](https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf)
(last updated 26 August 2026).

**Licence:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/),
as required by §3.1 and §4.5 of those terms.

**Attribution:** Data © Binance, via Binance Vision (data.binance.vision).

**Changes made:** the original monthly CSV archives (checksum-verified) were
reduced to the open time and OHLCV columns, converted to UTC timestamps and
saved as Parquet in this project's canonical schema, partitioned by year
(`scripts/fetch_data.py --source binance --symbol BTCUSDT --start 2020-01-01
--end 2024-12-31 --data-root data/sample`).

**Non-commercial only.** Commercial use of this data needs a separate licence
from Binance (Dataset Terms §3.4). This project is not sponsored, endorsed or
approved by Binance (§8.2).

This licence covers **only the data files in this directory**. The project's
source code is licensed separately.
