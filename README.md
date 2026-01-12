# 2025-MLB-Swing-Decision-Analysis

This project utilizes advanced MLB Statcast data to analyze and cluster Major League hitters based on their physical traits, plate discipline, and batted-ball quality. By integrating "Bat Tracking" metrics with "Swing Decisions," this research identifies distinct hitter archetypes and explores the strategic trade-offs between aggression and efficiency.

## 📊 Project Overview

Traditional baseball analytics often separate physical tools (Power) from mental approach (Discipline). This project bridges that gap by using Unsupervised Learning (K-Means) to profile hitters based on how their swing mechanics interact with their plate discipline.

## 🛠️ Data & Methodology

### Data Sources
The analysis integrates data from [**Baseball Savant (Statcast)**](https://baseballsavant.mlb.com/) and [**Baseball Reference**](https://www.baseball-reference.com/) for the 2025 season:
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

## 📈 Key Insights & Visualizations

### 1. Batter Archetype Analysis (Clustering)
![Metric Chart](images/cluster-metric.png)
![Radar Chart](images/cluster-radar.png)

Based on our K-Means clustering ($K=5$), the 144 qualified hitters were segmented into the following strategic groups:

| Cluster | Group Name | Characteristics | Representative Stars | Sample Size |
|:---:|:---|:---|:---|:---:|
| **0** | **The All-Around Elites** | Elite bat speed + High discipline + High impact; High Whiff% as a trade-off. | Shohei Ohtani, Aaron Judge, Kyle Schwarber | 45 |
| **1** | **Disciplined Power** | Strong power + Elite zone selection; significantly lower Whiff% than Cluster 0. | Juan Soto, Vladimir Guerrero Jr. | 43 |
| **2** | **Aggressive Swingers** | High bat speed; lower discipline scores; heavy reliance on raw physical tools. | Mookie Betts, Jake Cronenworth | 28 |
| **3** | **The Contact Machines** | Extreme bat control + Short swing path; prioritize "putting the ball in play" over power. | Luis Arraez, Steven Kwan | 4 |
| **4** | **Dynamic Multi-Tool** | Balanced output; value through situational hitting, versatility, and consistent contact. | Trea Turner, Bo Bichette, Salvador Perez | 24 |

---

![Cluster Chart](images/cluster-PCA.png)

While the X and Y axes of the PCA plot do not represent specific physical metrics, they serve as a 2D projection of the 13-dimensional hitter profiles. The clear separation of colors (clusters) validates that our K-Means model has successfully identified distinct, non-overlapping archetypes based on swing physics and plate discipline.

### 2. Cost of Chase: Chase Rate vs. Hard Hit Rate
By plotting decision quality against physical output, we identified four distinct quadrants. This visualization highlights how different hitter profiles navigate the trade-off between aggression and quality of contact.

![Strategic Map](images/cost-of-chase.png)

- **ELITE (Patient & Powerful)**: Hitters like **Aaron Judge**, **Kyle Schwarber** and **Shohei Ohtani** exhibit elite plate discipline while maintaining the league's highest hard-hit rates.
- **AGGRESSIVE (Bad Ball Hitters)**: This group, including stars like **Manny Machado** and **Oniel Cruz**, possesses the rare physical ability to turn "bad balls" into high-velocity contact despite high chase rates.


## 🚀 Future Work
- [ ] **2-Strike Approach Analysis**: Comparing swing length and bat speed adjustments in 2-strike counts versus regular counts.
- [ ] **Predictive Modeling**: Using these clusters to predict future SLG or wOBA stability.
