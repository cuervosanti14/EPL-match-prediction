# EPL Match Outcome Prediction
DS2500 Group Project — Spring 2026

## Team Members
- Santiago Cuervo — SPI Gap Analysis
- Brian Skiles — Home Advantage Analysis
- Maxim Kurdimov — Possession Analysis
- Sifat Anan — Draw Predictability Analysis

## Project Overview
This project investigates how well publicly available statistics
predict EPL match outcomes (Win, Draw, Loss) across four
distinct analytical angles.

## Setup
Clone the repo and install libraries:
pip install -r requirements.txt

## Data Sources
- Football Data UK: https://football-data.co.uk/
- FBref: https://fbref.com/en/
- Kaggle SPI: https://www.kaggle.com/datasets/thedevastator/club-soccer-predictions-spi-ratings-and-forecast?select=spi_global_rankings.csv 

## How to Run (Setup)

### Prerequisites
- Python 3.x
- Required libraries listed in `requirements.txt`

### Steps

1. **Clone the repository:**
```bash
   git clone https://github.com/cuervosanti14/EPL-match-prediction.git
   cd EPL-match-prediction
```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

3. **Run the analysis:**
   - Navigate to `src/` folder
   - Run individual analysis scripts:
     - `python spi_analysis.py` (Santiago's SPI Gap Analysis)
     - `python home_advantage_analysis.py` (Brian's analysis)
     - `python possession_analysis.py` (Maxim's analysis)
     - `python drawprob_analysis.py` (Sifat's analysis)

4. **View results:**
   - Visualizations saved in `visuals/` folder
   - Organized by analysis type (SPI, draw_prob, home_advantage, possession)

## Project Structure
`
EPL-match-prediction/
├── README.md
├── data/ (datasets - CSV files)
│   ├── SPI/
│   ├── draw_prob/
│   ├── home_advantage/
│   └── possession/
├── src/ (analyses)
│   ├── spi_analysis.py
│   ├── drawprob_analysis.py
│   ├── home_advantage_analysis.py
│   └── possession_analysis.py
├── visuals/
│   ├── SPI/
│   ├── draw_prob/
│   ├── home_advantage/
│   └── possession/
├── requirements.txt
└── .gitignore
`
