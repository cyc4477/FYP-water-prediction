from tkinter import *
from tkinter.ttk import *
import pandas as pd
from models import startPrediction

window = Tk()
window.geometry('950x500')
window.title("Welcome to predicting peanut moisture app")

lb1 = Label(window, text="Input the information of \nthe dehydrating process", font=("Arial Bold", 18))
lb1.grid(column=1, row=0)


lb2 = Label(window, text="Choose the time(h):", font=("Arial Bold", 14))
lb2.grid(column=0, row=2)

combo1 = Combobox(window)
combo1['values'] = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)
combo1.current(0)
combo1.grid(column=1, row=2)

lb3 = Label(window, text="Choose the temperature:", font=("Arial Bold", 14))
lb3.grid(column=0, row=3)

combo2 = Combobox(window)
combo2['values'] = (30, 35, 40, 45, 50)
combo2.current(0)
combo2.grid(column=1, row=3)

lb4 = Label(window, text="Input the initial water(%):", font=("Arial Bold", 14))
lb4.grid(column=0, row=4)

txt1 = Entry(window, width=10)
txt1.grid(column=1, row=4)
txt1.focus()

lb5 = Label(window, text="Input the velocity(m^3/(kg,h)):", font=("Arial Bold", 14))
lb5.grid(column=0, row=5)

txt2 = Entry(window, width=10)
txt2.grid(column=1, row=5)
txt2.focus()

lb6 = Label(window, text="Choose the height(mm):", font=("Arial Bold", 14))
lb6.grid(column=0, row=6)

combo3 = Combobox(window)
combo3['values'] = (100, 400)
combo3.current(0)
combo3.grid(column=1, row=6)



def clicked():
    lb7 = Label(window, text="Thanks for trying on my prediction app", font=("Arial Bold", 14))
    lb7.grid(column=1, row=8)
    time = combo1.get()
    print(time)
    temperature = combo2.get()
    print(temperature)
    initial_water = txt1.get()
    print(initial_water)
    velocity = txt2.get()
    print(velocity)
    height = combo3.get()
    print(height)

    if len(time) == 0 or len(velocity) == 0:
        lb8 = Label(window, text="There are some empty information, please re-enter again", font=("Arial Bold", 14))
        lb8.grid(column=1, row=9)

    else:
        lb9 = Label(window, text="Prediction is calculating", font=("Arial Bold", 14))
        lb9.grid(column=1, row=10)
        data = {'时间（h）': [time], 'temperature(℃)': [temperature], 'initial water(%)': [initial_water],
                'velocity(m^3/(kg,h))': [velocity], 'height(mm)': [height]}
        xtest = pd.DataFrame(data)

        prediction = startPrediction(xtest)
        lb10 = Label(window, text="The prediction of water contain after dehydrating process is:", font=("Arial Bold", 14))
        lb10.grid(column=1, row=11)

        lb11.config(text="{0:.2f}%".format(prediction * 100))


btn = Button(window, text="Click Me", command=clicked)
btn.grid(column=1, row=7)

lb11 = Label(window, text="", font=("Arial Bold", 14))
lb11.grid(column=1, row=13)

def quit():
    window.destroy()


quit_btn = Button(window, text="quit", command=quit)
quit_btn.grid(column=1, row=15)


window.mainloop()
print()

# data = {'时间（h）': [df['时间（h）'].min()], 'temperature(℃)': [df['temperature(℃)'].min()], 'initial water(%)': [df['initial water(%)'].min()], 'velocity(m^3/(kg,h))': [df['velocity(m^3/(kg,h))'].min()], 'height(mm)': [df['height(mm)'].min()]}


