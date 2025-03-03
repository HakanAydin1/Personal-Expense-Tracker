import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QLineEdit, QComboBox, QTableWidget,
                             QTableWidgetItem, QTabWidget, QFileDialog, QMessageBox, QDateEdit,
                             QTableView, QFormLayout, QGroupBox, QGridLayout, QSlider)
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QColor
import pickle
import datetime
import calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class ExpenseClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.categories = ['Food', 'Transportation', 'Housing', 'Entertainment', 'Utilities', 'Shopping', 'Other']
        self.trained = False
        self.encoder = None

    def prepare_data(self, expenses_df):
        """Prepare data for training the classifier."""
        # Extract features from the expense description and amount
        X = expenses_df[['Description', 'Amount']]
        y = expenses_df['Category']

        # Create word features from description
        X['Description'] = X['Description'].str.lower()

        return X, y

    def train(self, expenses_df):
        """Train the model on the expenses data."""
        if len(expenses_df) < 10:
            return False

        X, y = self.prepare_data(expenses_df)

        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('desc', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='')),
                ]), ['Description']),
                ('amount', SimpleImputer(strategy='median'), ['Amount'])
            ])

        # Train on text and amount
        try:
            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Store description words for feature extraction
            self.common_words = {}
            for cat in self.categories:
                cat_desc = X_train[y_train == cat]['Description'].str.split()
                words = [word for sublist in cat_desc if isinstance(sublist, list) for word in sublist]
                self.common_words[cat] = set(words)

            # Add features based on common words
            X_train = self._add_word_features(X_train)
            X_test = self._add_word_features(X_test)

            # Train the model
            self.model.fit(X_train, y_train)
            self.trained = True

            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def _add_word_features(self, X):
        """Add features based on common words in descriptions."""
        X_with_features = X.copy()

        for category, words in self.common_words.items():
            X_with_features[f'has_{category}_words'] = X['Description'].apply(
                lambda desc: sum(1 for word in str(desc).split() if word in words)
            )

        return X_with_features

    def predict_category(self, description, amount):
        """Predict the category for a new expense."""
        if not self.trained:
            # Default to 'Other' if model is not trained
            return 'Other'

        # Create a dataframe with the input
        input_df = pd.DataFrame({
            'Description': [description.lower()],
            'Amount': [amount]
        })

        # Add word features
        input_df = self._add_word_features(input_df)

        # Make prediction
        prediction = self.model.predict(input_df)
        return prediction[0]

    def save_model(self, filepath):
        """Save the trained model to a file."""
        if self.trained:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'common_words': self.common_words,
                    'trained': self.trained
                }, f)
            return True
        return False

    def load_model(self, filepath):
        """Load a trained model from a file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.common_words = data['common_words']
                self.trained = data['trained']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class ExpensePredictor:
    def __init__(self):
        self.models = {}  # One model for each category
        self.categories = ['Food', 'Transportation', 'Housing', 'Entertainment', 'Utilities', 'Shopping', 'Other']
        self.trained = False

    def train(self, expenses_df):
        """Train the model on the expenses data."""
        if len(expenses_df) < 30:  # Need enough data for reasonable prediction
            return False

        # Group by month and category, and sum the amounts
        expenses_df['Month'] = pd.to_datetime(expenses_df['Date']).dt.month
        expenses_df['Year'] = pd.to_datetime(expenses_df['Date']).dt.year

        monthly_expenses = expenses_df.groupby(['Year', 'Month', 'Category'])['Amount'].sum().reset_index()

        # Create separate models for each category
        for category in self.categories:
            category_data = monthly_expenses[monthly_expenses['Category'] == category]

            if len(category_data) < 3:  # Need at least 3 months of data
                continue

            # Create features: previous month's expense, 2 months ago expense, month of year
            X = []
            y = []

            # Sort by date
            category_data = category_data.sort_values(by=['Year', 'Month'])

            # Create lag features
            for i in range(2, len(category_data)):
                features = [
                    category_data.iloc[i - 1]['Amount'],  # Previous month
                    category_data.iloc[i - 2]['Amount'],  # Two months ago
                    category_data.iloc[i]['Month']  # Month of year (seasonality)
                ]
                X.append(features)
                y.append(category_data.iloc[i]['Amount'])

            if len(X) > 0:
                # Train a model for this category
                model = LinearRegression()
                model.fit(X, y)
                self.models[category] = model

        self.trained = len(self.models) > 0
        return self.trained

    def predict_next_month(self, expenses_df):
        """Predict expenses for the next month."""
        if not self.trained:
            return None

        # Group by month and category, and sum the amounts
        expenses_df['Month'] = pd.to_datetime(expenses_df['Date']).dt.month
        expenses_df['Year'] = pd.to_datetime(expenses_df['Date']).dt.year

        monthly_expenses = expenses_df.groupby(['Year', 'Month', 'Category'])['Amount'].sum().reset_index()

        # Get the current month
        now = datetime.datetime.now()
        current_month = now.month
        current_year = now.year

        # Calculate the next month
        if current_month == 12:
            next_month = 1
            next_year = current_year + 1
        else:
            next_month = current_month + 1
            next_year = current_year

        predictions = {}

        for category in self.categories:
            if category not in self.models:
                predictions[category] = 0
                continue

            category_data = monthly_expenses[monthly_expenses['Category'] == category].sort_values(by=['Year', 'Month'])

            if len(category_data) < 2:
                predictions[category] = 0
                continue

            # Get the last two months of data
            try:
                last_month = category_data.iloc[-1]['Amount']
                two_months_ago = category_data.iloc[-2]['Amount'] if len(category_data) > 1 else last_month

                # Make prediction
                prediction = self.models[category].predict([[last_month, two_months_ago, next_month]])
                predictions[category] = max(0, prediction[0])  # Ensure no negative predictions
            except Exception as e:
                print(f"Error predicting for {category}: {e}")
                predictions[category] = 0

        return predictions

    def save_model(self, filepath):
        """Save the trained model to a file."""
        if self.trained:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'models': self.models,
                    'trained': self.trained
                }, f)
            return True
        return False

    def load_model(self, filepath):
        """Load a trained model from a file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.trained = data['trained']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class PieChartCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

    def update_chart(self, data, title="Expense Distribution"):
        """Update the pie chart with new data."""
        self.ax.clear()

        # Filter out zero values
        labels = [k for k, v in data.items() if v > 0]
        values = [v for v in data.values() if v > 0]

        if sum(values) > 0:
            self.ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            self.ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        else:
            self.ax.text(0.5, 0.5, "No data to display",
                         horizontalalignment='center', verticalalignment='center')

        self.ax.set_title(title)
        self.fig.tight_layout()
        self.draw()


class BarChartCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

    def update_chart(self, data, predicted_data=None, title="Monthly Expenses"):
        """Update the bar chart with new data and predicted data."""
        self.ax.clear()

        categories = list(data.keys())
        values = list(data.values())

        x = range(len(categories))
        width = 0.35

        # Plot actual expenses
        self.ax.bar(x, values, width, label='Actual')

        # If we have predictions, plot them
        if predicted_data:
            predicted_values = [predicted_data.get(cat, 0) for cat in categories]
            self.ax.bar([i + width for i in x], predicted_values, width, label='Predicted')

        self.ax.set_xlabel('Categories')
        self.ax.set_ylabel('Amount')
        self.ax.set_title(title)
        self.ax.set_xticks([i + width / 2 for i in x] if predicted_data else x)
        self.ax.set_xticklabels(categories)

        if predicted_data:
            self.ax.legend()

        self.fig.tight_layout()
        self.draw()


class TrendChartCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

    def update_chart(self, monthly_data, title="Monthly Spending Trend"):
        """Update the trend chart with monthly data."""
        self.ax.clear()

        months = list(monthly_data.keys())
        values = list(monthly_data.values())

        if len(months) > 0:
            self.ax.plot(months, values, 'o-')

            # Add trend line
            if len(months) > 1:
                z = np.polyfit(range(len(months)), values, 1)
                p = np.poly1d(z)
                self.ax.plot(months, p(range(len(months))), "r--", alpha=0.8)

            self.ax.set_xlabel('Month')
            self.ax.set_ylabel('Total Expenses')
            self.ax.set_title(title)

            # Rotate x-axis labels if there are many months
            if len(months) > 6:
                plt.xticks(rotation=45)
        else:
            self.ax.text(0.5, 0.5, "Not enough data for trend analysis",
                         horizontalalignment='center', verticalalignment='center')

        self.fig.tight_layout()
        self.draw()


class ExpenseTracker(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Personal Expense Tracker")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize data storage
        self.expenses_df = pd.DataFrame(columns=['Date', 'Amount', 'Description', 'Category'])
        self.budget = {cat: 0 for cat in
                       ['Food', 'Transportation', 'Housing', 'Entertainment', 'Utilities', 'Shopping', 'Other']}

        # Initialize ML models
        self.classifier = ExpenseClassifier()
        self.predictor = ExpensePredictor()

        # Initialize UI
        self.init_ui()

        # Try to load saved data and models
        self.data_file = "expenses_data.csv"
        self.classifier_file = "expense_classifier.pkl"
        self.predictor_file = "expense_predictor.pkl"
        self.budget_file = "budget_data.csv"

        self.load_data()
        self.load_models()
        self.load_budget()

        # Update all visualizations
        self.update_expense_table()
        self.update_visualizations()

    def init_ui(self):
        """Initialize the user interface."""
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Create tab widget
        tab_widget = QTabWidget()

        # Dashboard tab
        dashboard_tab = QWidget()
        dashboard_layout = QVBoxLayout()

        # Add expense form
        expense_form = QGroupBox("Add New Expense")
        form_layout = QFormLayout()

        # Date input
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        form_layout.addRow("Date:", self.date_edit)

        # Amount input
        self.amount_edit = QLineEdit()
        self.amount_edit.setPlaceholderText("Enter amount")
        form_layout.addRow("Amount:", self.amount_edit)

        # Description input
        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("Enter description")
        form_layout.addRow("Description:", self.desc_edit)

        # Category dropdown
        self.category_combo = QComboBox()
        self.category_combo.addItems(
            ['Food', 'Transportation', 'Housing', 'Entertainment', 'Utilities', 'Shopping', 'Other'])
        form_layout.addRow("Category:", self.category_combo)

        # Add expense button
        self.add_btn = QPushButton("Add Expense")
        self.add_btn.clicked.connect(self.add_expense)
        form_layout.addRow("", self.add_btn)

        expense_form.setLayout(form_layout)
        dashboard_layout.addWidget(expense_form)

        # Charts section
        charts_layout = QHBoxLayout()

        # Pie chart
        pie_box = QGroupBox("Expense Distribution")
        pie_layout = QVBoxLayout()
        self.pie_chart = PieChartCanvas(self, width=5, height=4)
        pie_layout.addWidget(self.pie_chart)
        pie_box.setLayout(pie_layout)
        charts_layout.addWidget(pie_box)

        # Bar chart
        bar_box = QGroupBox("Category Comparison")
        bar_layout = QVBoxLayout()
        self.bar_chart = BarChartCanvas(self, width=5, height=4)
        bar_layout.addWidget(self.bar_chart)
        bar_box.setLayout(bar_layout)
        charts_layout.addWidget(bar_box)

        dashboard_layout.addLayout(charts_layout)

        # Expense table
        table_box = QGroupBox("Recent Expenses")
        table_layout = QVBoxLayout()
        self.expense_table = QTableWidget()
        self.expense_table.setColumnCount(4)
        self.expense_table.setHorizontalHeaderLabels(["Date", "Amount", "Description", "Category"])
        self.expense_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.expense_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.expense_table.setSelectionMode(QTableWidget.SingleSelection)
        table_layout.addWidget(self.expense_table)

        # Add delete button
        delete_btn = QPushButton("Delete Selected Expense")
        delete_btn.clicked.connect(self.delete_expense)
        table_layout.addWidget(delete_btn)

        table_box.setLayout(table_layout)
        dashboard_layout.addWidget(table_box)

        dashboard_tab.setLayout(dashboard_layout)
        tab_widget.addTab(dashboard_tab, "Dashboard")

        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout()

        # Trend chart
        trend_box = QGroupBox("Monthly Spending Trend")
        trend_layout = QVBoxLayout()
        self.trend_chart = TrendChartCanvas(self, width=8, height=4)
        trend_layout.addWidget(self.trend_chart)
        trend_box.setLayout(trend_layout)
        analysis_layout.addWidget(trend_box)

        # Prediction section
        prediction_box = QGroupBox("Spending Predictions")
        prediction_layout = QVBoxLayout()

        prediction_label = QLabel("Predicted spending for next month:")
        prediction_layout.addWidget(prediction_label)

        self.prediction_table = QTableWidget()
        self.prediction_table.setColumnCount(3)
        self.prediction_table.setHorizontalHeaderLabels(["Category", "Predicted Amount", "Current Budget"])
        prediction_layout.addWidget(self.prediction_table)

        prediction_box.setLayout(prediction_layout)
        analysis_layout.addWidget(prediction_box)

        analysis_tab.setLayout(analysis_layout)
        tab_widget.addTab(analysis_tab, "Analysis")

        # Budget tab
        budget_tab = QWidget()
        budget_layout = QVBoxLayout()

        budget_form = QGroupBox("Set Monthly Budget")
        budget_form_layout = QGridLayout()

        self.budget_edits = {}
        row = 0
        for i, category in enumerate(
                ['Food', 'Transportation', 'Housing', 'Entertainment', 'Utilities', 'Shopping', 'Other']):
            label = QLabel(f"{category}:")
            self.budget_edits[category] = QLineEdit()
            self.budget_edits[category].setPlaceholderText(f"Budget for {category}")

            budget_form_layout.addWidget(label, row, 0)
            budget_form_layout.addWidget(self.budget_edits[category], row, 1)
            row += 1

        save_budget_btn = QPushButton("Save Budget")
        save_budget_btn.clicked.connect(self.save_budget)
        budget_form_layout.addWidget(save_budget_btn, row, 0, 1, 2)

        budget_form.setLayout(budget_form_layout)
        budget_layout.addWidget(budget_form)

        # Budget progress visualization
        budget_progress_box = QGroupBox("Budget Progress")
        budget_progress_layout = QVBoxLayout()
        self.budget_bar_chart = BarChartCanvas(self, width=8, height=4)
        budget_progress_layout.addWidget(self.budget_bar_chart)
        budget_progress_box.setLayout(budget_progress_layout)
        budget_layout.addWidget(budget_progress_box)

        budget_tab.setLayout(budget_layout)
        tab_widget.addTab(budget_tab, "Budget")

        main_layout.addWidget(tab_widget)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def add_expense(self):
        """Add a new expense to the tracker."""
        try:
            # Get input values
            date_str = self.date_edit.date().toString("yyyy-MM-dd")
            amount_str = self.amount_edit.text()
            description = self.desc_edit.text()

            # Validate input
            if not amount_str or not description:
                QMessageBox.warning(self, "Input Error", "Please enter both amount and description.")
                return

            try:
                amount = float(amount_str)
                if amount <= 0:
                    raise ValueError("Amount must be positive")
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Please enter a valid positive number for amount.")
                return

            # Get category - use ML prediction if not specified
            category = self.category_combo.currentText()

            # If classifier is trained, use it to suggest a category
            if self.classifier.trained and category == "Other":
                suggested_category = self.classifier.predict_category(description, amount)
                reply = QMessageBox.question(self, "Category Suggestion",
                                             f"Based on your expense pattern, we suggest '{suggested_category}' category. Accept?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    category = suggested_category

            # Add to dataframe
            new_expense = pd.DataFrame({
                'Date': [date_str],
                'Amount': [amount],
                'Description': [description],
                'Category': [category]
            })

            self.expenses_df = pd.concat([self.expenses_df, new_expense], ignore_index=True)

            # Save data
            self.save_data()

            # Update UI
            self.update_expense_table()
            self.update_visualizations()

            # Train models with new data
            self.train_models()

            # Clear input fields
            self.amount_edit.clear()
            self.desc_edit.clear()

            # Show confirmation
            self.statusBar().showMessage(f"Expense added: ${amount:.2f} for {description}", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def update_expense_table(self):
        """Update the expense table with current data."""
        self.expense_table.setRowCount(0)

        # Sort by date, most recent first
        if not self.expenses_df.empty:
            sorted_df = self.expenses_df.sort_values(by='Date', ascending=False)

            # Add rows to table
            for i, (index, row) in enumerate(sorted_df.iterrows()):
                self.expense_table.insertRow(i)

                # Format date
                date_item = QTableWidgetItem(row['Date'])

                # Format amount
                amount_item = QTableWidgetItem(f"${row['Amount']:.2f}")
                amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                # Add description and category
                desc_item = QTableWidgetItem(row['Description'])
                category_item = QTableWidgetItem(row['Category'])

                # Store the original DataFrame index as item data for later retrieval
                date_item.setData(Qt.UserRole, index)

                self.expense_table.setItem(i, 0, date_item)
                self.expense_table.setItem(i, 1, amount_item)
                self.expense_table.setItem(i, 2, desc_item)
                self.expense_table.setItem(i, 3, category_item)

            # Resize columns to contents
            self.expense_table.resizeColumnsToContents()

    def delete_expense(self):
        """Delete the selected expense from the tracker."""
        selected_rows = self.expense_table.selectedItems()

        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select an expense to delete.")
            return

        # Get the row index and original DataFrame index
        table_row = selected_rows[0].row()
        df_index = self.expense_table.item(table_row, 0).data(Qt.UserRole)

        # Confirm deletion
        expense_desc = self.expense_table.item(table_row, 2).text()
        expense_amount = self.expense_table.item(table_row, 1).text()

        reply = QMessageBox.question(self, "Confirm Deletion",
                                     f"Are you sure you want to delete this expense?\n{expense_amount} for {expense_desc}",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Remove from DataFrame
            self.expenses_df = self.expenses_df.drop(df_index).reset_index(drop=True)

            # Save data
            self.save_data()

            # Update UI
            self.update_expense_table()
            self.update_visualizations()

            # Re-train models with updated data
            self.train_models()

            # Show confirmation
            self.statusBar().showMessage(f"Expense deleted: {expense_amount} for {expense_desc}", 3000)

    def update_visualizations(self):
        """Update all visualizations with current data."""
        if self.expenses_df.empty:
            return

        # Get current month's expenses
        current_month = datetime.datetime.now().month
        current_year = datetime.datetime.now().year

        # Convert date to datetime
        self.expenses_df['Date'] = pd.to_datetime(self.expenses_df['Date'])

        current_month_df = self.expenses_df[
            (self.expenses_df['Date'].dt.month == current_month) &
            (self.expenses_df['Date'].dt.year == current_year)
            ]

        # Calculate total by category for pie chart
        category_totals = current_month_df.groupby('Category')['Amount'].sum().to_dict()

        # Ensure all categories exist in the dict
        for cat in ['Food', 'Transportation', 'Housing', 'Entertainment', 'Utilities', 'Shopping', 'Other']:
            if cat not in category_totals:
                category_totals[cat] = 0

        # Update pie chart
        self.pie_chart.update_chart(category_totals, "Current Month Expenses")

        # Get predictions for next month
        predictions = None
        if self.predictor.trained:
            predictions = self.predictor.predict_next_month(self.expenses_df)

        # Update bar chart
        self.bar_chart.update_chart(category_totals, predictions, "Current vs. Predicted Expenses")

        # Update trend chart - monthly totals over time
        monthly_data = self.expenses_df.groupby([
            self.expenses_df['Date'].dt.year,
            self.expenses_df['Date'].dt.month
        ])['Amount'].sum()

        # Format month labels
        month_labels = [f"{year}-{month}" for (year, month) in monthly_data.index]
        monthly_values = monthly_data.values

        # Create a dictionary for the trend chart
        trend_data = dict(zip(month_labels, monthly_values))

        # Update trend chart
        self.trend_chart.update_chart(trend_data)

        # Update prediction table
        self.update_prediction_table(predictions if predictions else {})

        # Update budget progress chart
        self.update_budget_progress()

    def update_prediction_table(self, predictions):
        """Update the prediction table with the latest predictions."""
        self.prediction_table.setRowCount(0)

        for i, category in enumerate(
                ['Food', 'Transportation', 'Housing', 'Entertainment', 'Utilities', 'Shopping', 'Other']):
            self.prediction_table.insertRow(i)

            # Add category name
            self.prediction_table.setItem(i, 0, QTableWidgetItem(category))

            # Add predicted amount
            predicted = predictions.get(category, 0)
            amount_item = QTableWidgetItem(f"${predicted:.2f}")
            amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.prediction_table.setItem(i, 1, amount_item)

            # Add budget
            budget = self.budget.get(category, 0)
            budget_item = QTableWidgetItem(f"${budget:.2f}")
            budget_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            # Highlight if prediction exceeds budget
            if budget > 0 and predicted > budget:
                budget_item.setBackground(QColor(255, 200, 200))  # Light red

            self.prediction_table.setItem(i, 2, budget_item)

        # Resize columns to contents
        self.prediction_table.resizeColumnsToContents()

    def update_budget_progress(self):
        """Update the budget progress visualization."""
        if self.expenses_df.empty:
            return

        # Get current month's expenses
        current_month = datetime.datetime.now().month
        current_year = datetime.datetime.now().year

        # Convert date to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.expenses_df['Date']):
            self.expenses_df['Date'] = pd.to_datetime(self.expenses_df['Date'])

        current_month_df = self.expenses_df[
            (self.expenses_df['Date'].dt.month == current_month) &
            (self.expenses_df['Date'].dt.year == current_year)
            ]

        # Calculate total by category
        category_totals = current_month_df.groupby('Category')['Amount'].sum().to_dict()

        # Ensure all categories exist in the dict
        for cat in ['Food', 'Transportation', 'Housing', 'Entertainment', 'Utilities', 'Shopping', 'Other']:
            if cat not in category_totals:
                category_totals[cat] = 0

        # Update budget progress chart - actual vs. budget
        self.budget_bar_chart.update_chart(category_totals, self.budget, "Budget vs. Actual Spending")

    def save_budget(self):
        """Save the budget values from the form."""
        try:
            for category in self.budget_edits:
                budget_text = self.budget_edits[category].text()
                if budget_text:
                    try:
                        self.budget[category] = float(budget_text)
                    except ValueError:
                        QMessageBox.warning(self, "Input Error", f"Invalid value for {category} budget.")

            # Save to file
            budget_df = pd.DataFrame({
                'Category': list(self.budget.keys()),
                'Amount': list(self.budget.values())
            })
            budget_df.to_csv(self.budget_file, index=False)

            # Update visualizations
            self.update_visualizations()

            QMessageBox.information(self, "Budget Saved", "Your budget has been updated successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def train_models(self):
        """Train the ML models with the current data."""
        if len(self.expenses_df) < 10:
            return

        # Train classifier
        classifier_trained = self.classifier.train(self.expenses_df)

        # Train predictor if we have enough data
        predictor_trained = self.predictor.train(self.expenses_df)

        # Save models
        if classifier_trained:
            self.classifier.save_model(self.classifier_file)

        if predictor_trained:
            self.predictor.save_model(self.predictor_file)

        # Update visualizations with new predictions
        self.update_visualizations()

    def save_data(self):
        """Save the expense data to a file."""
        try:
            self.expenses_df.to_csv(self.data_file, index=False)
        except Exception as e:
            print(f"Error saving data: {e}")

    def load_data(self):
        """Load expense data from file."""
        try:
            if os.path.exists(self.data_file):
                self.expenses_df = pd.read_csv(self.data_file)
                print(f"Loaded {len(self.expenses_df)} expenses")
        except Exception as e:
            print(f"Error loading data: {e}")

    def load_models(self):
        """Load trained models from files."""
        if os.path.exists(self.classifier_file):
            self.classifier.load_model(self.classifier_file)

        if os.path.exists(self.predictor_file):
            self.predictor.load_model(self.predictor_file)

    def load_budget(self):
        """Load budget data from file."""
        try:
            if os.path.exists(self.budget_file):
                budget_df = pd.read_csv(self.budget_file)
                self.budget = dict(zip(budget_df['Category'], budget_df['Amount']))

                # Update budget form
                for category, amount in self.budget.items():
                    if category in self.budget_edits:
                        self.budget_edits[category].setText(str(amount))
        except Exception as e:
            print(f"Error loading budget: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExpenseTracker()
    window.show()
    sys.exit(app.exec_())