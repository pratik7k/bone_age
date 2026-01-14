import tkinter as tk
from tkinter import filedialog, messagebox
import requests


API_URL = "http://localhost:8000/predict"


def select_and_send():
    # Open file browser
    file_path = filedialog.askopenfilename(
        title="Select X-ray Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if not file_path:
        return

    gender = gender_var.get()

    try:
        with open(file_path, "rb") as f:
            files = {
                "image": f
            }
            data = {
                "gender": gender
            }

            response = requests.post(API_URL, files=files, data=data)

        if response.status_code == 200:
            result = response.json()

            messagebox.showinfo(
                "Prediction Result",
                f"""
Age Group: {result['data']['age_group_label']}
Age Range: {result['data']['age_range_months'][0]} - {result['data']['age_range_months'][1]} months
Bone Age: {result['data']['bone_age_months']} months
({result['data']['bone_age_years']} years)
"""
            )
        else:
            messagebox.showerror(
                "Server Error",
                f"Status Code: {response.status_code}\n{response.text}"
            )

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ---------------- UI ----------------
root = tk.Tk()
root.title("Bone Age Predictor - Test Client")
root.geometry("420x200")

gender_var = tk.StringVar(value="male")

tk.Label(root, text="Select Gender:", font=("Arial", 11)).pack(pady=5)

tk.Radiobutton(root, text="Male", variable=gender_var, value="male").pack()
tk.Radiobutton(root, text="Female", variable=gender_var, value="female").pack()

tk.Button(
    root,
    text="Select X-ray Image & Predict",
    command=select_and_send,
    font=("Arial", 12),
    width=30,
    bg="#2E86C1",
    fg="white"
).pack(pady=20)

root.mainloop()
