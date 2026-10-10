# Phase 3: Person C - Baseline Evaluation & Graph Visualization Guide
## For Solo Implementation (No Vibe Coding)

This guide provides a step-by-step approach to completing Person C's Phase 3 responsibilities when working alone. The focus is on **running the mean-variance baseline on the full train/test split to obtain benchmark performance**, **creating the animated/evolving graph visual** (correlation graph changing over time) using the historical windows already available from Person A's work, and **building the stacked-area allocation-over-time chart** that shows how the RL agent's money shifts between stocks week by week (using early/rough agent outputs from Person B's environment).

---
## 🎯 Phase 3 Objectives (Person C)

1. **Run the mean-variance baseline on the full train/test split** to obtain benchmark performance metrics for comparison with the GNN+RL agent in later phases.
2. **Create the animated/evolving graph visualization** showing how the correlation graph changes over time (week to week), with nodes colored by sector and edges representing significant correlations.
3. **Build the stacked-area allocation-over-time chart** that visualizes how portfolio weights evolve over time for the RL agent (using early outputs from Person B's PortfolioEnv).
4. **Document assumptions, parameters, and limitations** for the baseline performance and visualizations.
5. **Prepare a handoff package** (updated scripts, sample outputs, and documentation) so that Person B can use the baseline for evaluation and Person A can see the expected format of the graph visuals.

> **Key Principle**: The baseline performance metrics should be robust and reproducible, and the visualizations should be clear, labeled, and suitable for inclusion in the final dashboard for comparing the GNN+RL agent against traditional benchmarks.

---
## ---
## 📋 Step‑by‑Step Workflow

### Step 1: Prepare Data Splits for Baseline Evaluation
**Before running baselines, ensure you have proper train/test splits consistent with Person B's environment.**

- Locate the data splits used by Person B in `src/environment/portfolioenv.py` or related scripts.
- Identify the train/test split dates or indices used for Phase 4 RL training.
- Ensure your baseline evaluation uses the same splits to enable fair comparison.

**Action**: 
- Check if train/test splits are already defined (look for files like `data/clean/train.parquet` and `data/clean/test.parquet`).
- If not, create consistent splits based on date ranges (e.g., train: 2015-03-31 to 2020-12-31, test: 2021-01-01 to 2022-12-28).
- Document the split ratios and date ranges used.

### Step 2: Run Mean-Variance Baseline on Full Dataset
**Execute your existing mean-variance baseline on the complete dataset to obtain benchmark performance.**

- Use the `mean_variance_baseline.py` script you created in Phase 2.
- Ensure it runs on both train and test splits separately (or as specified by the project).
- Apply the same transaction cost model as Person B's environment (linear bps model).
- Record key performance metrics: Sharpe ratio, max drawdown, cumulative return, volatility, turnover.

**Action**:
1. Modify `mean_variance_baseline.py` to accept train/test split parameters if needed.
2. Run the baseline on the full dataset (or specified splits).
3. Save outputs to `results/baselines/mean_variance_full/` or similar.
4. Generate performance metrics report.
5. Create equity curve plot for the full period.

### Step 3: Create Animated/Evolving Graph Visualization
**Using the correlation matrices and graph objects from Person A's Phase 2 work, create an animation showing how the graph evolves over time.**

**Action**: Create a script `animate_evolving_graph.py` that:
1. Loads the sequence of graph objects from `graphs/saved_graphs/` (graph_0000.pkl, graph_0001.pkl, etc.).
2. For each graph snapshot:
   - Extracts the correlation matrix or edge list.
   - Applies the same thresholding logic used by Person A (check `graph_construction_config.json`).
   - Builds a NetworkX graph with nodes representing stocks and edges representing significant correlations.
   - Colors nodes by sector (use sector mapping from Person A's work or ticker list).
   - Uses a consistent layout algorithm (e.g., spring layout with fixed seed) for smooth animation.
   - Saves each frame as a PNG image.
3. Combines the frames into an animated GIF or MP4 video.
4. Adds a timestamp/text overlay showing the date each frame represents.
5. Saves the animation to `results/plots/evolving_graph_animation.gif` (or .mp4).

**Technical Details**:
- Use matplotlib's animation functionality or imageio to create the GIF.
- Ensure consistent node ordering and positioning across frames for smooth animation.
- Use a discrete colormap for sector coloring (e.g., tab10).
- Edge width can represent correlation strength (absolute value).
- Title should show the date/time window each frame represents.

### Step 4: Build Stacked-Area Allocation-Over-Time Chart
**Create a visualization showing how the RL agent's portfolio weights change over time using early outputs from Person B's environment.**

**Action**: Create a script `plot_allocation_over_time.py` that:
1. Loads portfolio weight history from Person B's early training runs (check `results/` directory for PPO training outputs).
2. If no RL outputs are available yet, use placeholder/random weights that sum to 1.0 for demonstration.
3. Creates a stacked area chart where:
   - X-axis: time steps/dates
   - Y-axis: portfolio weight allocation (0 to 1.0)
   - Each layer: weight allocated to a specific stock or cash
   - Colors: consistent with sector coloring used in graph visualization
4. Includes a legend showing which color corresponds to which stock/sector.
5. Adds title and axis labels.
6. Saves the plot to `results/plots/allocation_over_time.png`.

**Note**: Since this is Phase 3 and RL training may not be complete, you can:
- Use weights from Person B's testing with placeholder embeddings
- Generate synthetic weights that demonstrate the concept
- Leave clear instructions for updating with real RL outputs in Phase 4

### Step 5: Document Assumptions & Limitations
**For each artifact (baseline performance, evolving graph, allocation chart), document key assumptions.**

**Baseline Performance Assumptions**:
- Same transaction cost model as Person B's environment (linear bps).
- Lookback window for mean-variance estimation (document your choice).
- Risk-aversion parameter λ (document your choice).
- Rebalancing frequency (if applicable).
- No look-ahead bias in estimations.
- Train/test split methodology.

**Evolving Graph Assumptions**:
- Correlation threshold used for edge creation (same as Person A).
- Sector mapping is static and complete.
- Layout algorithm parameters (seed, iterations).
- Temporal frequency of graph snapshots (daily, weekly, etc.).
- Node ordering consistency across frames.

**Allocation Chart Assumptions**:
- Source of weight data (early RL runs, placeholder, etc.).
- Time alignment with graph visualization.
- Normalization of weights (sum to 1.0).
- Sector-color mapping consistency with other visualizations.

### Step 6: Prepare Handoff Package
**Organize your work for easy use by Persons A and B.**

**Action**: Create/update the following structure:
```
results/
├── baselines/
│   ├── mean_variance/
│   │   ├── mean_variance_baseline.py
│   │   ├── params.json
│   │   └── output/
│   │       ├── weights.csv
│   │       ├── portfolio_value.csv
│   │       └── metrics.json
│   └── mean_variance_full/  # New for Phase 3 full dataset run
│       ├── params.json
│       └── output/
│           ├── weights.csv
│           ├── portfolio_value.csv
│           └── metrics.json
├── plots/
│   ├── correlation_heatmap_*.png
│   ├── static_graph_*.png
│   ├── evolving_graph_animation.gif  # New for Phase 3
│   ├── allocation_over_time.png      # New for Phase 3
│   ├── mean_variance_baseline.png
│   └── ...
└── ...
```

Create or update `results/README.md` explaining:
- How to run the mean-variance baseline on full dataset.
- How to generate the evolving graph animation.
- How to create the allocation-over-time chart.
- Where to find performance metrics and visualizations.
- Expected formats for handoff to later phases.

### Step 7: Validate Outputs
**Verify that your work produces sensible, error-free results.**

**Action**:
1. Run the mean-variance baseline on full dataset and verify:
   - No NaN or infinite portfolio values.
   - Reasonable Sharpe ratios (positive for sensible strategies).
   - Transaction costs properly accounted for.
   - Weights sum to 1.0 at each time step.
2. Verify the evolving graph animation:
   - All frames generated without errors.
   - Consistent node positioning across frames.
   - Sector coloring consistent with ticker list.
   - Timestamp overlay accurate.
3. Verify the allocation-over-time chart:
   - Properly formatted stacked area chart.
   - Weights sum to approximately 1.0 at each point.
   - Clear legend and axis labels.
   - Sensible weight evolution patterns.

---
## ---
## 📝 Deliverables for Phase 3 Checkpoint
By the end of Phase 3 (alone), you should have:

1. ✅ **Mean-variance baseline run on full dataset** with performance metrics (Sharpe, max-dd, return, volatility) saved in `results/baselines/mean_variance_full/`.
2. ✅ **Animated/evolving graph visualization** saved as `results/plots/evolving_graph_animation.gif` (or .mp4) showing correlation graph changes over time.
3. ✅ **Stacked-area allocation-over-time chart** saved as `results/plots/allocation_over_time.png` showing RL agent weight evolution.
4. ✅ **Documented assumptions and limitations** for baseline performance and visualizations (in script headers or separate `ASSUMPTIONS.md`).
5. ✅ **Updated handoff package** in `results/` folder with clear README explaining how to reproduce and use your work.

### Checkpoint Verification:
- Can you explain **why** you chose specific parameters for the mean-variance baseline (lookback window, λ, rebalancing freq)?
- Does the baseline use the **same transaction‑cost model** as Person B's environment?
- Is the **evolving graph animation** smooth and visually informative (showing clear temporal patterns)?
- Does the **allocation-over-time chart** properly show weight changes summing to 1.0?
- Have you **prevented look‑ahead bias** in all estimations?
- Are your **visualizations clear, labeled, and suitable** for the final dashboard?
- Can you **reproduce all outputs** from scratch using your scripts and documentation?

---
## ---
## 💡 Tips for Solo Work (Avoiding Vibe Coding)
1. **Leverage Phase 2 work** – reuse your `mean_variance_baseline.py` and visualization scripts with minimal modifications.
2. **Start with the evolving graph** – it depends only on Person A's completed work and gives immediate feedback.
3. **Use consistent styling** – apply the same color schemes, fonts, and formatting across all visualizations for dashboard cohesion.
4. **Validate data pipelines first** – ensure you can load Person A's graph files and Person B's environment outputs before creating visualizations.
5. **Keep animations simple** – focus on clear temporal changes rather than complex visual effects.
6. **Document parameter choices** – explain why you selected specific thresholds, layout parameters, or visualization settings.
7. **Test with small subsets** – verify your scripts work on a few data points before processing the full dataset.
8. **Use fixed seeds** – for reproducible layouts and animations, use fixed random seeds where randomness is involved.
9. **Check file sizes** – ensure animations aren't excessively large for sharing; optimize frame rate and dimensions as needed.
10. **Share early** – send drafts of your evolving graph and allocation chart to Persons A and B for feedback before finalizing.

---
## ---
## 🚀 Next Steps After Phase 3
Once your baseline evaluation and visualizations are complete:

1. **Person A** will have completed GNN embeddings and handed them off to Person B (including trained model, embedding files, and specs).
2. **Person B** will integrate the real GNN embeddings into `PortfolioEnv` and begin PPO training in Phase 4.
3. **In Phase 4 (Weeks 6‑7)**, you (Person C) will:
   - Continue refining the mean-variance baseline if needed based on initial RL results.
   - Update the evolving graph animation with any new graph snapshots from Person A.
   - Update the allocation-over-time chart with real RL agent outputs from Person B's training.
   - Prepare preliminary comparison visuals for Phase 5.
4. **In Phase 5 (Evaluation & Comparison)**, you will:
   - Run a full comparison — GNN+RL vs equal‑weight vs buy‑and‑hold vs Markowitz — on Sharpe ratio, max drawdown, cumulative return, volatility.
   - Build the **portfolio‑value‑over‑time line chart** that overlays all four strategies.
   - Create additional comparison visuals (e.g., bar charts of metrics, scatter plots of risk vs return).
5. **In Phase 6 (Demo Polish & Report)**, you will assemble everything into the final Streamlit dashboard, incorporating:
   - All baseline performance metrics
   - Evolving graph visualization
   - Allocation-over-time charts for all strategies
   - Final comparison dashboard

**You’ve got this!** Take time with each design decision. The quality of your baseline performance metrics and visualizations will heavily influence how meaningful the GNN+RL experiments are — if benchmarks are flawed or visualizations unclear, you cannot trust whether the RL agent is truly adding value. Your work provides the essential context for evaluating whether the sophisticated GNN+RL approach delivers genuine improvement over traditional methods.