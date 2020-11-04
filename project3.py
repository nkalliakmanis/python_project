# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:40:22 2020

@author: user
"""

import math
import tkinter as tk 
from tkinter import *
from tkinter import scrolledtext
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas
from pandas import DataFrame

class CreateData:
    def __init__(self):
        self.w=Tk()
        self.w.title("Create data")
        self.w.geometry("600x600")
        
        
        l=Label(self.w,text="Create y=A1*x + B1 + R1*Rt, where Rt=noise")
        l.grid(column=0,row=0)
        
        
        l=Label(self.w,text="Give A1:")
        l.grid(column=0,row=1)
        self.e1=Entry(self.w)
        self.e1.grid(column=1,row=1)
        
        
        self.e2=Entry(self.w)
        self.e2.grid(column=1,row=2)
        l=Label(self.w,text="Give B1:")
        l.grid(column=0,row=2)
        
        self.e3=Entry(self.w)
        self.e3.grid(column=1,row=3)
        l=Label(self.w,text="Give R1:")
        l.grid(column=0,row=3)
        
        l=Label(self.w,text="Create z=A2*y + B2*cos(x/T) + R2*Rt, where Rt=noise")
        l.grid(column=0,row=4)
        
        self.e4=Entry(self.w) 
        self.e4.grid(column=1,row=5)
        l=Label(self.w,text="Give A2:")
        l.grid(column=0,row=5)
        
        self.e5=Entry(self.w)
        self.e5.grid(column=1,row=6) 
        l=Label(self.w,text="Give B1:")
        l.grid(column=0,row=6)
        
        self.e6=Entry(self.w)
        self.e6.grid(column=1,row=7)
        l=Label(self.w,text="Give T:")
        l.grid(column=0,row=7)
        
        self.e7=Entry(self.w)
        self.e7.grid(column=1,row=8)
        l=Label(self.w,text="Give R2")
        l.grid(column=0,row=8)
        
        l=Label(self.w,text="Filename:")
        l.grid(column=0,row=9)
        self.e8=Entry(self.w)
        self.e8.grid(column=1,row=9)
        
        b4=Button(self.w,text="Create",command=self.create2)
        b4.grid(column=0,row=10)
        
        
        self.w.mainloop()
          
    def create2(self):
        error=0
        try:
            A1=float(self.e1.get())
        except:
            self.e1.delete(0,tk.END)
            self.e1.insert(0, "error")
            error=1
            
        try:
            B1=float(self.e2.get())
        except:
            self.e2.delete(0,tk.END)
            self.e2.insert(0, "error")
            error=1
        
        try:
            R1=float(self.e3.get())
        except:
            self.e3.delete(0,tk.END)
            self.e3.insert(0, "error")
            error=1
        
        try:
            A2=float(self.e4.get())
        except:
            self.e4.delete(0,tk.END)
            self.e4.insert(0, "error")
            error=1
            
        try:
            B2=float(self.e5.get())
        except:
            self.e5.delete(0,tk.END)
            self.e5.insert(0, "error")
            error=1
        
        try:
            T=float(self.e6.get())
        except:
            self.e6.delete(0,tk.END)
            self.e6.insert(0, "error")
            error=1
        
        try: 
            R2=float(self.e7.get())
        except:
            self.e7.delete(0,tk.END)
            self.e7.insert(0, "error")
            error=1
        
        
        if(error==0):
            X = np.random.normal(size=100,loc=10,scale=2)
            Y = A1*X + B1 + R1*np.random.normal(size=100,loc=0,scale=1)
            cs=np.array([math.cos(c/T) for c in X])
            Z = A2*Y + B2*cs + R2*np.random.normal(size=100,loc=0,scale=1)
            filename=self.e8.get()
            f=open(filename,"w")
            f.write("X,Y,Z\n")
            for x,y,z in zip(X,Y,Z):
                f.write("%f,%f,%f\n" %(x,y,z))
            f.close()
            
        else:
            print("Give the parameters correctly!") 
    

    
class Analyze: 
    def __init__(self):
        self.w=Tk()
        self.w.title("Analyze data")
        self.w.geometry("600x600")
        self.df=pandas.DataFrame()
        self.e1=Entry(self.w)
        self.e1.grid(column=0,row=1)
        b6=Button(self.w,text="Load file",command=self.load)
        b6.grid(column=1,row=1)
        
        b7=Button(self.w,text="Plot time series",command=self.plot)
        b7.grid(column=0,row=2)
        
        b8=Button(self.w,text="Histograms",command=self.hist)
        b8.grid(column=0,row=3)
        
        b9=Button(self.w,text="Statistics",command=self.statistics)
        b9.grid(column=0,row=4)
        
        b10=Button(self.w,text="Correlations",command=self.correlations)
        b10.grid(column=0,row=5)
        
        b11=Button(self.w,text="Linear Regression",command=self.linear)
        b11.grid(column=0,row=6)
        
        b12=Button(self.w,text="Neural",command=self.neural)
        b12.grid(column=0,row=7)
        
        
    def load(self):
         
        filename=self.e1.get()
        self.df=pandas.read_csv(filename)
        print (self.df)
        
        
    def plot(self):
        t=[i for i in range(len(self.df["X"]))]
        
        plt.plot(t,self.df["X"])
        plt.title("X(t)")
        plt.xlabel("t")
        plt.ylabel("X")
        plt.show()
        
        plt.plot(t,self.df["Y"])
        plt.title("Y(t)")
        plt.xlabel("t")
        plt.ylabel("Y")
        plt.show()
        
        plt.plot(t,self.df["Z"])
        plt.title("Z(t)")
        plt.xlabel("t")
        plt.ylabel("Z")
        plt.show()
        
        plt.scatter(self.df["X"],self.df["Y"])
        plt.title("Y(X)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        
        
        
        plt.scatter(self.df["X"],self.df["Z"])
        plt.title("Z(X)")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.show()
        
        
        
        plt.scatter(self.df["Y"],self.df["Z"])
        plt.title("Z(Y)")
        plt.xlabel("Y")
        plt.ylabel("Z")
        plt.show()
        
        
    def hist(self): 
        
        self.df["X"].hist(bins=5) 
        plt.title("X")
        plt.show() 
        plt.Figure()
        
        self.df["Y"].hist(bins=5)
        plt.title("Y")
        plt.show()
        plt.Figure() 
        
        self.df["Z"].hist(bins=5) 
        plt.title("Z")
        plt.show() 
        plt.Figure()

        
    def statistics(self): 
        x=self.df["X"]
        avg=np.mean(x)
        std=np.std(x)
        mx=np.max(x)
        mn=np.min(x)
        
        s= "Mesos oros=%.2f std=%.2f  Max=%.2f Min=%.2f" %(avg,std,mx,mn)
        print(s)
        
    def correlations(self): 
        print(self.df.corr())
        print(self.df.describe())
        
    def linear(self): 
        xy=self.df[["X","Y"]]
        z=self.df["Z"]
        
        regr = linear_model.LinearRegression() 
        X_train, X_test, z_train, z_test = train_test_split(xy, z, test_size=0.3, random_state=1)
        regr.fit(X_train,z_train)
        P=regr.predict(X_test)
        print(regr.score(X_test,z_test)) 
        
        
        
    def neural(self):   
        
        xy=self.df[["X","Y"]]
        z=self.df["Z"]
        
        X_train, X_test, z_train, z_test = train_test_split(xy, z, test_size=0.3, random_state=1)
        print(X_train)
        mlp = MLPRegressor(hidden_layer_sizes=(10,10,10)) 
        mlp.fit(X_train, z_train)
        P=mlp.predict(X_test)
        print(mlp.score(X_test,z_test)) 
        
        
        
class MyProject: 
    def __init__(self):
        self.w=Tk()
        self.w.title("My project")
        self.w.geometry("600x600")
        

        b1=Button(self.w,text="Create data",command=self.create)
        b2=Button(self.w,text="Analyze date",command=self.analyze)
        b3=Button(self.w,text="Exit",command=self.exit1)
        
        b1.config(width=25)
        b2.config(width=25)
        b3.config(width=25)
        
        
        b1.grid(column=0,row=1)
        b2.grid(column=0,row=2)
        b3.grid(column=0,row=3)
        self.w.mainloop()
        
    def create(self):
        w2=CreateData()

    def analyze(self):
        w2=Analyze()
        
    def exit1(self):
        self.w.destroy()
       
myframe=MyProject()