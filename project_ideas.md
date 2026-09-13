- What is the project's primary goal? (Learning GNN+RL? Demonstrating sector diversification benefits?)
Answer: Learning GNN+RL

- What market conditions should the portfolio be resilient to? (Bull markets? Bear markets? Mixed regimes?)
Answer: Mixed Regimes

- What level of complexity can you handle with 15-30 stocks? (More stocks = more complex graph but better diversification)
Answer: 20 stocks across 5 different sectors

# TECHNOLOGY
1. AAPL - Apple Inc. (Consumer Tech)
2. MSFT - Microsoft Corporation (Software/Cloud)
3. NVDA - NVIDIA Corporation (Semiconductors)
4. ADBE - Adobe Inc. (Digital Media)

# FINANCE
1. JPM - JPMorgan Chase & Co. (Banking/Financial Services)
2. BAC - Bank of America Corporation (Banking/Financial Services)
3. GS  - The Goldman Sachs Group, Inc. (Investment Banking/Financial Services)
4. V   - Visa Inc. (Payments/Financial Services)

# ENERGY

1. XOM - Exxon Mobil Corporation (Oil & Gas)
2. CVX - Chevron Corporation (Oil & Gas)
3. COP - ConocoPhillips (Oil & Gas Exploration/Production)
4. SLB - SLB (Oilfield Services)

# UTILITIES

1. NEE - NextEra Energy, Inc. (Electric Utilities/Renewable Energy)
2. DUK - Duke Energy Corporation (Electric Utilities)
3. SO  - The Southern Company (Electric Utilities)
4. AEP - American Electric Power Company, Inc. (Electric Utilities)

# REAL ESTATE

1. PLD - Prologis, Inc. (Industrial REIT)
2. AMT - American Tower Corporation (Telecommunications REIT)
3. O   - Realty Income Corporation (Retail REIT)
4. SPG - Simon Property Group, Inc. (Retail REIT)


# get_AALP.py 
- this file contains the sample dataset extraction test results.

# Split ratio
Train: 2015-01-01 to 2022-12-31 (8 years)
Test:  2022-01-01 to 2024-12-31 (3 years)

# Rolling Window and Correlation Method
```
Rolling window: 60 trading days (~3 months)
Why: Balances responsiveness to regime changes with enough samples for stable correlation estimates.
Method: Pearson correlation on simple returns.
```