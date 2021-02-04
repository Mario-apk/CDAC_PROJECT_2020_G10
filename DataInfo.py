# -*- coding: utf-8 -*-
""" Data Information """


def dataInfo(a):
    print(a.head())
    
    print("the shape of dataset is ",a.shape)
    
    print("Info.  regarding dataset",a.info())
    
    #to find the range of tuples in  attributes and other info.
    print("the range of tuples in  attributes ",a.describe())