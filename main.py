import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tkinter import ttk
import joblib


class ClassifierApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Classifier")
        self.window.geometry("400x400")

        self.model = None
        self.X_train = None
        self.y_train = None

        self.load_data_button = tk.Button(self.window, text="Load Data", command=self.load_data)
        self.load_data_button.pack(pady=10)

        self.train_model_button = tk.Button(self.window, text="Train Model", command=self.train_model)
        self.train_model_button.pack(pady=10)

        self.predict_data_button = tk.Button(self.window, text="Predict Data", command=self.predict_data)
        self.predict_data_button.pack(pady=10)

        self.rebuild_model_button = tk.Button(self.window, text="Rebuild Model", command=self.rebuild_model)
        self.rebuild_model_button.pack(pady=10)

        self.view_data_button = tk.Button(self.window, text="View Data", command=self.view_data)
        self.view_data_button.pack(pady=10)

        self.plot_data_button = tk.Button(self.window, text="Plot Data", command=self.plot_data)
        self.plot_data_button.pack(pady=10)

        self.save_model_button = tk.Button(self.window, text="Save Model", command=self.save_model)
        self.save_model_button.pack(pady=10)

        self.load_model_button = tk.Button(self.window, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

        self.window.mainloop()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if file_path:
            try:
                data = pd.read_csv(file_path, header=None)
                X = data.iloc[:, 1:14]  # Wybierz 13 cech
                y = data.iloc[:, 0]
                self.X_train, _, self.y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
                messagebox.showinfo("Data Loaded", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Warning", "No training data available!")
            return

        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        messagebox.showinfo("Training", "Model trained successfully!")

    def predict_data(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Model not trained yet!")
            return

        file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if file_path:
            try:
                new_data = pd.read_csv(file_path, header=None)
                new_data = new_data.iloc[:, 1:14]
                predictions = self.model.predict(new_data)
                accuracy = accuracy_score(self.y_train, self.model.predict(self.X_train))
                messagebox.showinfo("Predictions", f"Predictions: {predictions}\nAccuracy: {accuracy}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def rebuild_model(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Warning", "No training data available!")
            return

        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        messagebox.showinfo("Model Rebuilt", "Model rebuilt successfully!")

    #tabelka
    def view_data(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Warning", "No training data available!")
            return

        data_window = tk.Toplevel(self.window)
        data_window.title("Data View")
        data_window.geometry("600x400")

        data_table = ttk.Treeview(data_window)

        #kolumny tabeli
        data_table["columns"] = list(range(14))

        #nagłówki kolumn
        for i in range(14):
            data_table.heading(i, text=f"Column {i + 1}")

        #dodawnie danych do tabeli
        for row_index, (x_row, y_value) in enumerate(zip(self.X_train.values, self.y_train.values)):
            data_table.insert("", "end", text=f"Row {row_index + 1}", values=list(x_row) + [y_value])

        #wyświetlanie tabeli
        data_table.pack()

        data_window.mainloop()

    #wykres
    def plot_data(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Warning", "No training data available!")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.X_train, x=self.X_train.iloc[:, 0], y=self.X_train.iloc[:, 1], hue=self.y_train)
        plt.title("Data Plot")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    def save_model(self):
        if self.model is None:
            messagebox.showwarning("Warning", "No model available!")
            return

        file_path = filedialog.asksaveasfilename(filetypes=[('Model Files', '*.joblib')])
        if file_path:
            try:
                joblib.dump(self.model, file_path)
                messagebox.showinfo("Model Saved", "Model saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[('Model Files', '*.joblib')])
        if file_path:
            try:
                self.model = joblib.load(file_path)
                messagebox.showinfo("Model Loaded", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))


app = ClassifierApp()
