
import scipy.special
import scipy.integrate 
from scipy.integrate import quad
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import sys    
import os
import xlwt
import xlrd
import pandas as pd
from xlrd.sheet import ctype_text 
import pickle
import sklearn.linear_model
regr = sklearn.linear_model.LinearRegression()
from scipy.optimize import curve_fit, fsolve
import operator
import scipy.stats
#from scipy.stats import norm
from numpy import linspace
from pylab import plot,show,hist,figure,title
import matplotlib
import matplotlib.cm as cm
import math
from scipy.signal import argrelextrema, savgol_filter

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

#unzipping dictionary contents 
def unzip_dict(dictionary):
    keys = []
    values = []
    for k, v in dictionary[soil_type].items():
        keys.append(k)
        values.append(v)
    for var in zip(keys, values):
        exec( "%s=%s" % (var[0], var[1]))

#extract lines of a tex file as an array of numbers and strings
def extract_lines(tex_file_name):
    lines = [] #Declare an empty list named "lines"
    with open (tex_file_name, 'rt') as in_file: 
        for line in in_file: 
            try:
                el = float(line.strip())
            except ValueError:
                el = line.strip()
            lines.append(el)
    return lines

import sys    
import os
import xlwt
import xlrd
import pandas as pd

def xl_output(array_names, array_values, special='', overwrite=0, sheet_name='output_sheet'): #values will typically be arrays themselves
    #create the tex file and place it in a logical area relative to the main script:
    cwd = os.getcwd()
    file_n =  os.path.basename(sys.argv[0])
    if special == '':
        xl_file = cwd + '/xl/' + file_n[:-3] +'_output.xlsx'
    else:
        xl_file = cwd + '/xl/' + file_n[:-3] +'_output_' + special + '.xlsx'
    #doesn't create if file already exists 
    if os.path.isfile(xl_file):
        None
    else:
        os.mknod(xl_file) 
    #does file already have data?
    if overwrite == 0: #do NOT overwite 
        df = pd.read_excel(xl_file) # or
        if df.empty:
            #write data to file
            cols = len(array_names)
            book = xlwt.Workbook()
            #sh = book.add_sheet(sheet)
            sh = book.add_sheet(sheet_name)
            for i in range(0, cols):
                sh.write(0, i, array_names[i])
                data_col = array_values[i]
                for j in range(0, len(data_col)):
                    sh.write(j+1, i, data_col[j])
            book.save(xl_file)
        else:
            print('File nonempty. To write over the current file, specify overwrite=1 while calling the function.')
    else: #directly rewite, no checking to see if empty
        #write data to file
        cols = len(array_names)
        book = xlwt.Workbook()
        #sh = book.add_sheet(sheet)
        sh = book.add_sheet(sheet_name)
        for i in range(0, cols):
            sh.write(0, i, array_names[i])
            data_col = array_values[i]
            for j in range(0, len(data_col)):
                sh.write(j+1, i, data_col[j])
        book.save(xl_file)

from xlrd.sheet import ctype_text 
#extract excel information and convert to python object
#UNPACKING 
def extract_xl(special='', sheet_index=0, name_file='std (format: ~/xl/current_pyfile_name_output.xlsx)'):
    if name_file == 'std (format: ~/xl/current_pyfile_name_output.xlsx)':
        cwd = os.getcwd()
        file_n =  os.path.basename(sys.argv[0])
        if (special == '') or (special == None):
            xl_file = cwd + '/xl/' + file_n[:-3] +'_output.xlsx'
        else:
            xl_file = cwd + '/xl/' + file_n[:-3] +'_output_' + special + '.xlsx'
    else:
        xl_file = name_file
    if os.path.isfile(xl_file): #file exists so we can extract its contents 
        workbook1 = xlrd.open_workbook(xl_file)
        xl_sheet = workbook1.sheet_by_index(sheet_index)
        # Iterate through rows, and print out the column values
        row_vals = []
        for row_idx in range(0, xl_sheet.nrows):
            for col_idx in range(0, xl_sheet.ncols):
                cell_obj = xl_sheet.cell(row_idx, col_idx)
                cell_type_str = ctype_text.get(cell_obj.ctype, 'unknown type')
                print ('(row %s) %s (type:%s)' % (row_idx, cell_obj.value, cell_type_str))
                row_vals.append(cell_obj.value)
        # Retrieve non-empty rows
        nonempty_row_vals = [x for x in row_vals if x]    
        num_rows_missing_vals = xl_sheet.nrows - len(nonempty_row_vals)
        print ('Vals: %d; Rows Missing Vals: %d' % (len(nonempty_row_vals), num_rows_missing_vals))
    else:
        print('This file does not exist---no content to extract.')
        
import re
def urlify(s, replace_with='-'):
     # Remove all non-word characters (everything except numbers and letters)
     s = re.sub(r"[^\w\s]", '', s)
     # Replace all runs of whitespace with a single dash
     s = re.sub(r"\s+", replace_with, s)
     return s

#THIS directories location 
cwd = os.getcwd()
main_dirc = cwd.split('code', 1)[0]
sys.path.insert(0, main_dirc+'code')

#easy picture namer
def gn(pic_name='temporary',sub_dirc=None,redirect=None,f='.pdf'):
    cwd = os.getcwd()
    #mainD = cwd.split('peat_project', 1)[0] + 'peat_project'
    mainD = cwd.split('snow_main', 1)[0] + 'snow_main'
    if sub_dirc!=None:# and redirect!=None:
        main_folder = mainD+'/g/'+sub_dirc+'/'
    else:
        main_folder = mainD+'/g/'
        #if redirect!=None:
            #mainP = redirect
        #else:
            #mainP = mainD
        #mainP = mainD if type(redirect)==type(None) else redirect
        #fol = '' if type(sub_dirc)==type(None) else sub_dirc
        #main_folder = mainP+'/g/'+fol
    return main_folder+urlify(pic_name)+f

import pickle
def save_obj(obj, name,parent_folder=None):
    cwd = os.getcwd()
    #mainD = cwd.split('code', 1)[0]
    #mainD = cwd.split('peat_project', 1)[0] + 'peat_project/'
    mainD = cwd.split('snow_main', 1)[0] + 'snow_main/'
    if parent_folder==None:
        with open(mainD+'obj/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(parent_folder+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, parent_folder=None):
    cwd = os.getcwd()
    #mainD = cwd.split('peat_project', 1)[0] + 'peat_project/'
    mainD = cwd.split('snow_main', 1)[0] + 'snow_main/'
    if parent_folder==None:
        #cwd = os.getcwd()
        #main_folder = cwd[:-5]
        with open(mainD+'obj/'+ name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        with open(parent_folder+ name + '.pkl', 'rb') as f:
            return pickle.load(f)



import xlwt
import xlrd
import pandas as pd

def xl_push(*array_values,names,names2=None,names3=None, file_name='dummy',sheet_name='def',start=1,end=1): #values will typically be arrays themselves
    cwd = os.getcwd()
    mainD = cwd.split('code', 1)[0]
    xl_file = mainD+'xl_outputs/'+ file_name + '.xlsx'
    if start==1:
        #doesn't create if file already exists 
        if os.path.isfile(xl_file):
            None
        else:
            os.mknod(xl_file)
        book = xlwt.Workbook()
    else:
        book = start
    
    #convert dictionary
    if type(array_values[0])==dict:
        dic = array_values[0]
        #print(dic[2009])
        array_values = []
        for nid in range(0,len(names)):
            n = names[nid]
            cold = dic[n]
            array_values.append(cold)

    array_names = names
    cols = len(array_names)
    sh = book.add_sheet(sheet_name)
    for i in range(0, cols):
        print(names[i])
        sh.write(0, i, array_names[i])
        if type(names2)!=type(None):
            sh.write(1, i, names2[i])
        if type(names3)!=type(None):
            sh.write(2, i, names3[i])
        data_col = array_values[i]
        for j in range(0, len(data_col)):
            try:
                sh.write(j+3, i, data_col[j])
            except:
                sh.write(j+3, i, float(data_col[j]))
    if end==1:
        book.save(xl_file)
    return book

from draft_figs import *

#array_names = ['cheese','michael','coffee']
#a1 = [1,2,3]
#a2 = [0,0,9]
#a3 = [12,4,5]
#array_values = [a1,a2,a3]

#bk = xl_push(a1,a2,a2,names=array_names, sheet_name='toads',end=0)
#xl_push(a1,a3,a1,names=array_names, sheet_name='toads2',start=bk)



