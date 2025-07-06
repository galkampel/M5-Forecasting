# M5 Forecasting Dataset Features

## Overview
The M5 Forecasting dataset is a hierarchical sales data from Walmart, featuring daily sales data for various products across different stores in multiple states. This dataset was part of the M5 Forecasting competition, focusing on accurate sales prediction.

## Data Files Overview
The M5 forecasting dataset consists of the following main files:
- `calendar.csv`: Contains date-related features and events
- `sales_train_validation.csv`: Main sales data with hierarchical structure
- `sales_train_evaluation.csv`: Extended sales data for evaluation
- `sell_prices.csv`: Price information for items
- `sample_submission.csv`: Template for competition submissions

## Hierarchical Structure
The dataset follows a hierarchical structure with the following levels:

1. **State Level**
   - 3 states: CA (California), TX (Texas), and WI (Wisconsin)
   - Represents the highest geographical aggregation level

2. **Store Level**
   - 10 stores
   - Multiple stores within each state
   - Each store has a unique identifier
   - Allows for store-specific analysis and forecasting

3. **Category Level**
   - 3 categories: FOODS, HOBBIES, HOUSEHOLD
   - Product categories representing broad product groups
   - Helps in grouping similar products together
   - Enables category-level demand analysis

4. **Department Level**
   - 7 departments: FOODS_1, FOODS_2, FOODS_3, HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2
     - `FOODS_3` has 8230 items (1913 days)
   - Departments within categories
   - More granular product grouping than categories
   - Useful for department-specific trends and patterns

5. **Item Level**
   - 3,049 unique items
   - Individual products
   - Most granular level of the hierarchy
   - Contains specific product identifiers and characteristics

Total number of time series = 3 states × 10 stores × 3,049 items = 91,470 series

## Special Events Features
The calendar data includes various types of special events that can impact sales:

### Event Types
1. **Cultural Events**
   - Valentine's Day
   - Mother's Day
   - Father's Day
   - Cinco De Mayo
   - St. Patrick's Day

2. **Religious Events**
   - Easter
   - Orthodox Easter
   - Lent
   - Ramadan
   - Purim
   - Pesach

3. **National Events**
   - Independence Day
   - Memorial Day
   - Presidents Day

4. **Sporting Events**
   - Super Bowl
   - NBA Finals (Start and End)

### SNAP Program Features
- __SNAP__ (Supplemental Nutrition Assistance Program)- a U.S. government program that
helps low-income individuals and families buy food. Walmart accepts SNAP benefits,
allowing customers to use their Electronic Benefit Transfer (EBT) cards for purchases both in-store and online
- SNAP for each state
- Binary indicators (0/1) for:
  - `snap_CA`: SNAP days in California
  - `snap_TX`: SNAP days in Texas
  - `snap_WI`: SNAP days in Wisconsin
  - Each state has 650 days of SNAP (~$\frac{1}{3}$ of the days)

## Calendar Features
- `date`: Calendar date
- `wm_yr_wk`: Week number
- `weekday`: Day of the week (text)
- `wday`: Day of week (number)
- `month`: Month number
- `year`: Year
- `d`: Day identifier (d_1, d_2, etc.)
- `event_name_1`, `event_type_1`: Primary event information
- `event_name_2`, `event_type_2`: Secondary event information

## Price Features
The sell_prices.csv contains:
- Item-level price information
- Weekly price variations
- Temporal price changes and patterns

## Sales Features
The sales training data includes:
- Daily unit sales
- Multiple time series at different hierarchical levels
- 1941 days of historical data
- Zero and non-zero sales patterns
- Seasonal patterns and trends

## Key Statistics
- Time span: 5+ years of daily data (2011-01-29 to 2016-06-19)
- Geographical coverage: 3 states
- Multiple hierarchical levels for aggregation
- Comprehensive event calendar
- Daily sales and price variations

This dataset structure allows for:
- Multi-level forecasting
- Event impact analysis
- Price elasticity studies
- Seasonal pattern detection
- Hierarchical demand planning 