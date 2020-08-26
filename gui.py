import numpy as np 
import csv
import matplotlib.pyplot as plt
from first import *
import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

class Application(tk.Frame):
    def __init__(self,motor,master=None):
        super().__init__(master)
        self.master = master
        self.motor = motor
        self.arma=tk.StringVar() 
        self.field=tk.StringVar()
        self.pack()
        self.create_widgets()
        self.master.geometry("1600x1000")
        self.motor.plot_motor()
        self.motor.plot_show(self.master)

    def create_widgets(self):
        vcmd = (self.register(self.callback))
        self.name1 = tk.Label(self)
        self.name1["text"] = "Armature Amp-turns"
        self.name1.pack(side="top")
        self.entry1 = tk.Entry(self, validate='all', validatecommand=(vcmd, '%P'))
        self.entry1["textvariable"] = self.arma
        self.entry1.pack(side="top")
        self.name2 = tk.Label(self)
        self.name2["text"] = "Field Amp-turns"
        self.name2.pack(side="top")
        self.entry2 = tk.Entry(self, validate='all', validatecommand=(vcmd, '%P'))
        self.entry2["textvariable"] = self.field
        self.entry2.pack(side="top")
        self.plot = tk.Button(self)
        self.plot["text"] = "Plot"
        self.plot["command"] = self.ploting
        self.plot.pack(side="top")
        self.canvas = None

    def callback(self, P):
        if str.isdigit(P) or P == "":
            return True
        else:
            return False

    def ploting(self):
        Xi,Yi,Vi = self.motor.Find_potentials(float(self.arma.get()),float(self.field.get()))
        self.motor.plot_motor()
        self.motor.plot_field(Xi,Yi,Vi)
        self.motor.plot_show(self.master)


class plot():
    def __init__(self,X,Y,E,T,C,Ti,n):
        self.n =n
        self.X =X
        self.Y =Y 
        self.E =E
        self.T =T
        self.Ti =Ti
        self.C =C
        self.Source = np.zeros(8)
        self.fig = Figure(figsize=(4, 4), dpi=100)

    def Find_potentials(self,Arma = 0,Field = 0):
          #make in versitile
        self.Source[2]=Field
        self.Source[3]=-Field
        self.Source[4]=Arma
        self.Source[5]=-Arma
        B =calculate_B(self.X,self.Y,self.E,self.T,8,self.Source)
        V = np.linalg.solve(self.C,B)
        Xi,Yi,Vi = calculate_potential(V,self.T,self.Ti,self.X,self.Y,self.n)
        return Xi,Yi,Vi

    def plot_motor(self):
        motor=plt.imread('motor.png')
        self.fig = Figure(figsize=(6,6), dpi=100)
        self.fig.add_subplot().imshow(motor)
        # plt.imshow(motor)


    def plot_field(self,Xi,Yi,Vi):
        Vi[5 > abs(Vi)] = np.NaN
        self.fig.add_subplot().contour(Xi*239+505,Xi*239+334,Vi,25,colors ='green',linewidths = 0.7,linestyles = 'solid')
 

    def plot_show(self, master):
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()
        ax = self.fig.gca()
        ax.set_axis_off()
        ax.autoscale(False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        #plt.show()


def read_data():
    Ti = np.load("Ti.npy",allow_pickle=True)
    C = np.load("C.npy",allow_pickle=True)
    return C,Ti

def calculate_potential(V,T,Ti,X,Y,n):

    potential = []
    elemen_no =0
    for i in Ti.T:
            
        node = T[elemen_no,[1,2,3]]
        X_element = X[node-1]
        Y_element = Y[node-1]
        V_element = V[node-1] 
        mat = np.array([
            [1, 1, 1],X_element,Y_element
                ])
        mat = mat.T
        delta = np.dot(np.linalg.inv(mat),V_element)
        point = np.ones((i[0].shape[0],1))
        point = np.append(point , i[0].reshape(i[0].shape[0],1),axis=1)
        point = np.append(point , i[1].reshape(i[0].shape[0],1),axis=1)
        potential.append([np.dot(point,delta.reshape(delta.shape[0])),i[0],i[1]])
        elemen_no += 1     
    VV = np.zeros((n*2,n*2))
    # VV[:] = np.NaN
    for i in potential:
        i[1] = i[1]*n + n-1
        i[2] = i[2]*n + n-1
        for j in range(len(i[0])):
            if j is not 0:
                VV[int(i[1][j]),int(i[2][j])] = i[0][j]

    X =np.linspace(-1,1,n*2)
    
    return X ,X ,VV.T

def main():
    file = "motor_v2_5254.msh"
    data = load_data(file)
    X,Y = load_nodes(data)
    E,T = load_edges_and_triangles(data)
    C,Ti = read_data()
    n = 500
    motor1 = plot(X,Y,E,T,C,Ti,n)
    root = tk.Tk()
    root.title("Field distribution in a DC motor")
    app = Application(motor1,master=root)
    app.mainloop()

 

if __name__ == "__main__":
    main()


