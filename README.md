# 2025-MLB-Swing-Decision-Analysis

This project utilizes advanced MLB Statcast data to analyze and cluster Major League hitters based on their physical traits, plate discipline, and batted-ball quality. By integrating "Bat Tracking" metrics with "Swing Decisions," this research identifies distinct hitter archetypes and explores the strategic trade-offs between aggression and efficiency.

## 📊 Project Overview

Traditional baseball analytics often separate physical tools (Power) from mental approach (Discipline). This project bridges that gap by using Unsupervised Learning (K-Means) to profile hitters based on how their swing mechanics interact with their plate discipline.

## 🛠️ Data & Methodology

### Data Sources
The analysis integrates data from **Baseball Savant (Statcast)** and **Baseball Reference** for the 2025 season:
- **Bat Tracking**: Average Bat Speed, Swing Length, Squared-up %, and Blast %.
- **Exit Velocity & Barrels**: Hard Hit Rate (95+ mph), Barrel %, and Average EV.
- **Plate Discipline**: Zone Swing %, Out-of-Zone Swing % (Chase Rate), and Whiff %.

### Workflow
1. **ETL & Data Merging**: Aligning multi-source datasets via `player_id`.
2. **Feature Engineering**:
   - `Discipline Score`: Calculated as `Zone Swing % - Chase %` to measure strike zone judgment.
   - `Power Efficiency`: Ratio of `Average Exit Velocity` to `Bat Speed`.
   - `Chase Efficiency`: `Hard Hit Rate` relative to `Chase Rate` to measure the cost of aggression.
3. **Clustering**: Applied **K-Means Clustering** with the Elbow Method to identify 5 distinct hitter archetypes.
4. **Strategic Mapping**: Developed a 4-Quadrant Strategic Map to visualize the relationship between decision-making and physical output.

## 📈 Key Insights & Results

### Hitter Archetypes (Clusters)
Our model identified 5 meaningful groups, including:
- **The Elites**: High discipline paired with elite hard-hit ability.
- **The Aggressors**: High chase rates but compensated by elite "Bad Ball" hitting skills.
- **The Contact Machines**: Short swing lengths and high squared-up efficiency.
- **High Risk / Low Reward**: Hitters with high whiff rates and poor decision-making quality.

### The Strategic Map (Cost of Chase)
By plotting **Chase Rate** against **Hard Hit Rate**, we categorized hitters into four quadrants:
1. **Elite: Patient & Powerful**: Top-left quadrant. High discipline and high impact.
2. **Aggressive: Bad Ball Hitters**: Top-right quadrant. High chase but high damage output.
3. **Passive / Low Impact**: Bottom-left quadrant. Disciplined but lacking raw power.
4. **High Risk / Low Reward**: Bottom-right quadrant. Excessive chasing with minimal hard contact.

## 🚀 Future Work
- [ ] **2-Strike Approach Analysis**: Comparing swing length and bat speed adjustments in 2-strike counts versus regular counts.
- [ ] **Temporal Analysis**: Tracking how hitter clusters shift throughout the season.
- [ ] **Predictive Modeling**: Using these clusters to predict future SLG or wOBA stability.
