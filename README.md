# Personal Expense Tracker with ML Prediction

A Python-based personal expense tracking application that uses machine learning to categorize expenses automatically and predict future spending patterns.

## 🌟 Features

- **Expense Tracking & Management**
  - Log expenses with date, amount, description, and category
  - View and delete past expenses
  - Organize spending into 7 main categories

- **Intelligent Categorization**
  - Uses machine learning (RandomForest classifier) to automatically suggest categories
  - Learns from your spending patterns over time
  - Becomes smarter as you add more data

- **Predictive Analytics**
  - Forecasts next month's spending by category
  - Uses linear regression models trained on your historical data
  - Shows trends in your spending habits

- **Budget Management**
  - Set monthly budgets for each expense category
  - Visualize budget vs. actual spending
  - Get visual alerts when predicted spending exceeds budget

- **Data Visualization**
  - Interactive pie charts showing expense distribution
  - Bar charts comparing expenses across categories 
  - Trend analysis showing spending patterns over time

## 📋 Requirements

- Python 3.7+
- PyQt5
- pandas
- numpy
- matplotlib
- scikit-learn

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/personal-expense-tracker.git
   cd personal-expense-tracker
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python expense_tracker.py
   ```

## 📘 How to Use

### Adding Expenses
1. Enter the date, amount, and description of your expense
2. Select a category (or let the app suggest one for you)
3. Click "Add Expense"

### Deleting Expenses
1. Select an expense from the table
2. Click "Delete Selected Expense"
3. Confirm the deletion

### Setting a Budget
1. Go to the "Budget" tab
2. Enter budget amounts for each category
3. Click "Save Budget"

### Viewing Predictions
1. Navigate to the "Analysis" tab
2. View predicted spending for the next month
3. Compare predictions with your budget

## 💾 Data Storage

All data is stored locally in CSV files:
- `expenses_data.csv` - Your expense records
- `budget_data.csv` - Your budget settings
- `expense_classifier.pkl` - Trained ML model for categorization
- `expense_predictor.pkl` - Trained ML model for predictions

## 🧠 Machine Learning Components

### Expense Categorization
- Uses RandomForest algorithm
- Features include expense amount and text analysis of descriptions
- Requires at least 10 expenses to begin making predictions

### Spending Prediction
- Uses Linear Regression models
- One model per expense category
- Takes into account seasonal patterns and trends
- Requires at least 3 months of data per category

## 📊 Visualizations

- **Pie Chart**: Shows the distribution of your expenses across categories
- **Bar Chart**: Compares current spending with predictions
- **Trend Line**: Displays your spending patterns over time
- **Budget Progress**: Visualizes how close you are to your budget limits

## 🔜 Future Improvements

- [ ] Export data to CSV/Excel
- [ ] Import transactions from bank statements
- [ ] Dark mode support
- [ ] Mobile companion app
- [ ] Cloud synchronization

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


Created with ❤️ by HAKAN AYDIN
