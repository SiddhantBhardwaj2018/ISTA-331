"""
Some code that you can mess with as you implement your functions to see
what is going on.  Once your code works, you can switch to main1 and
get a better view of your work in action.
"""

"""
Name - Siddhant Bhardwaj
ISTA 331 HW1
Section Leader - Aleah M Crawford
Collaborators - Shivansh Singh Chauhan,Abhishek Agarwal,Vibhor Mehta, Sriharsha Madhira
"""

import pandas as pd, numpy as np, random, sqlite3
from itertools import product

def isbn_to_title(conn):
    c = conn.cursor()
    query = 'SELECT isbn, book_title FROM Books;'
    return {row['isbn']: row['book_title'] for row in c.execute(query).fetchall()}

def select_book(itt):
    isbns = sorted(itt)
    print('All books:')
    print('----------')
    for i, isbn in enumerate(isbns):
        print(' ', i, '-->', isbn, itt[isbn][:60])
    print('-' * 40)
    selection = input('Enter book number or return to quit: ')
    return isbns[int(selection)] if selection else None
    
def similar_books(key, cm, pm, itt, spm): # an isbn, count_matrix, p_matrix, isbn_to_title
    bk_lst = []
    for isbn in cm.columns:
        if key != isbn:
            bk_lst.append((cm.loc[key, isbn], isbn))
    bk_lst.sort(reverse=True)
    print('Books similar to', itt[key] + ':')
    print('-----------------' + '-' * (len(itt[key]) + 1))
    for i in range(5):
        print(str(i) + ':')
        print(' ', bk_lst[i][0], '--', itt[bk_lst[i][1]][:80])
        print('  spm:', itt[spm[key][i]][:80])
        print('  p_matrix:', pm.loc[key, bk_lst[i][1]])
        
        
        
def get_purchase_matrix(conn):
    '''
    '''
    c = conn.cursor()
    q = "SELECT Orders.cust_id,OrderItems.isbn FROM Orders INNER JOIN OrderItems ON Orders.order_num = OrderItems.order_num"
    lst_cust_id = []
    lst_isbn = []
    for row in c.execute(q).fetchall():
        lst_cust_id.append(row[0])
        lst_isbn.append(row[1])
    d = {"Cust_id":lst_cust_id,"ISBN":lst_isbn}
    df = pd.DataFrame(index = [i for i in range(len(lst_cust_id))],data = d)
    stack1 = df.set_index('Cust_id').stack()
    v = stack1.groupby(level = 0).agg(list)
    d1 = dict(v)
    for i in d1:
        d1[i].sort()
    return(d1)
    
    
    
def get_empty_count_matrix(conn):
    '''
    '''
    c = conn.cursor()
    q = "SELECT isbn FROM Books"
    lst = []
    for row in c.execute(q).fetchall():
        lst.append(row[0])
    df = pd.DataFrame(0,index = lst,columns = lst)
    return(df)
        
        
def fill_count_matrix(count_matrix,purchase_matrix):
    '''
    '''
    for i in purchase_matrix:
        for subset in product(purchase_matrix[i],repeat = 2):
            count_matrix.loc[subset[0],subset[1]] += 1
    
    
def make_probability_matrix(count):
    '''
    '''
    lst = [i for i in count.columns]
    new = pd.DataFrame(0,index = lst,columns = lst)
    for row in count:
        for column in count.columns:
            if row != column:
                new.loc[row,column] = count.loc[row,column] / count.loc[row,row]
        for column in count.columns:
            if row == column:
                new.loc[row,column] = -1.0
    return(new)
    
    
def sparse_p_matrix(p_matrix):
    dict = {}
    for i in p_matrix.index:
        if i not in dict:
            dict[i] = []
        for j in  p_matrix.columns:
            dict[i].append(j)
        l = insertionSort(dict[i],i,p_matrix)
        dict[i] = l
    return dict

def insertionSort(arr,ind,matrix): 
    
    for i in range(1, len(arr)): 
        key = matrix.loc[ind , arr[i]]
        ch = arr[i]
        # Move elements of arr[0..i-1], that are 
        # greater than key, to one position ahead 
        # of their current position 
        j = i-1
        while j >=0 and key >= matrix.loc[ind ,arr[j]]:
                arr[j+1] = arr[j]
                j -= 1
        arr[j+1] = ch 
        
    if(len(arr) > 15):
        return arr[:15]
    return arr
    
    
    
def get_cust_id(conn):
    '''
    '''
    print('CID       Name\n-----     -----\n'+
          '    1     Thompson, Rich\n' + 
          '    2     Marzanna, Alfie\n' +
          '    3     Knut, Dan\n---------------')
    response = input("Enter customer number or enter to quit: ")
    if type(response) == int:
       val1 = response
    else:
        val1 = None
    return(val1)
    
def purchase_history(cust_id,lst_isbn,conn):
    '''
    '''
    c = conn.cursor()
    q =  "SELECT Customers.first || ' ' || Customers.last, Orders.cust_id,OrderItems.isbn,Books.book_title FROM Customers INNER JOIN Orders ON Customers.cust_id = Orders.cust_id INNER JOIN OrderItems ON Orders.order_num = OrderItems.order_num INNER JOIN Books ON OrderItems.isbn = Books.isbn"
    lst_1 = []
    lst_2 = []
    lst_3 = []
    lst_4 = []
    lst_books = []
    for row in c.execute(q).fetchall():
        lst_1.append(row[0])
        lst_2.append(row[1])
        lst_3.append(row[2])
        lst_4.append(row[3])
    d = {'Cust_id':lst_2,'Name':lst_1,'ISBN':lst_3,'Book_Title':lst_4}
    df = pd.DataFrame(index = [i for i in range(len(lst_1))],data = d)
    k = df.iloc[df.loc[df['Cust_id'] == cust_id].index[0], 1]
    for i in lst_isbn:
        for j in df.loc[df['ISBN'] == i].index:
            lst_books.append(df.iloc[j,3])
    lst_books = sorted(list(set(lst_books)),reverse = True)
    string1 = lst_books[0] + '\n'
    string3 = lst_books[1] + '\n'
    string2 = '-'* (21 + len(k)) + '\n'
    string = "Purchase history for " + k +'\n' + string2 + string1 + string3 + "----------------------------------------\n"
    return(string)
    
def get_recent(cust_id,conn):
    '''
    '''
    c = conn.cursor()
    q = "SELECT Orders.cust_id,Orders.order_date,OrderItems.isbn FROM Orders INNER JOIN OrderItems ON Orders.order_num = OrderItems.order_num"
    lst_1 = []
    lst_2 = []
    lst_3 = []
    lst_isbn = []
    for row in c.execute(q).fetchall():
        lst_1.append(row[0])
        lst_2.append(row[1])
        lst_3.append(row[2])
    d = {'Cust_id':lst_1,'ISBN':lst_3,'Order_Date':lst_2}
    df = pd.DataFrame(index = [i for i in range(len(lst_1))],data = d)
    pd.to_datetime(df['Order_Date'])
    req_df = df[df['Cust_id'] == cust_id]
    recent_req_df = req_df[req_df['Order_Date'] == req_df['Order_Date'].max()]
    if len(recent_req_df) == 1:
        return(recent_req_df['ISBN'])
    else:
        for i in recent_req_df['ISBN']:
            lst_isbn.append(i)
    return(lst_isbn[random.randrange(len(lst_isbn))])

def get_recommendation(id,mat,recent_lst,conn):
    '''
    '''
    c = conn.cursor()
    book_isbn = get_recent(id,conn)
    query = ('SELECT cust_id,first ||" "||last FROM Customers')
    name = ""
    for row in c.execute(query).fetchall():
        if(list(row)[0] == id):
            name = (list(row)[1])
    list_isbn = mat[book_isbn]
    new_lst= [i for i in list_isbn if i not in recent_lst]
    strn = "Recommendations for "+name+'\n'
    dict_title = isbn_to_title(conn)
    s = ""
    if (len(new_lst) == 0):
        s = "Out of ideas, go to Amazon\n"
    elif (len(new_lst) == 1):
        s = dict_title[new_lst[0]]  + '\n'
    else:
        s = dict_title[new_lst[0]] +'\n' +dict_title[new_lst[1]] +'\n'
        
    return (strn + ( '-'* ( len(strn) -1 ) + '\n' ) + s)
    
    
    
def main1():
    conn = sqlite3.connect('bookstore.db')
    conn.row_factory = sqlite3.Row
    purchase_matrix = get_purchase_matrix(conn)
    count_matrix = get_empty_count_matrix(conn)
    fill_count_matrix(count_matrix, purchase_matrix)
    p_matrix = make_probability_matrix(count_matrix)
    spm = sparse_p_matrix(p_matrix)
    ######
    itt = isbn_to_title(conn)
    selection = select_book(itt)
    while selection:
        similar_books(selection, count_matrix, p_matrix, itt, spm)
        input('Enter to continue:')
        selection = select_book(itt)
    ######
    cid = get_cust_id(conn)
    while cid:
        print()
        titles = purchase_history(cid, purchase_matrix[cid], conn)
        print(titles)
        print(get_recommendation(cid, spm, purchase_matrix[cid], conn))
        input('Enter to continue:')
        cid = get_cust_id(conn)
    
def main2():
    conn = sqlite3.connect('bookstore.db')
    conn.row_factory = sqlite3.Row
    
    purchase_matrix = get_purchase_matrix(conn)
    print('*' * 20, 'Purchase Matrix', '*' * 20)
    print(purchase_matrix)
    print()
    
    count_matrix = get_empty_count_matrix(conn)
    print('*' * 20, 'Empty Count Matrix', '*' * 20)
    print(count_matrix)
    print()
    
    fill_count_matrix(count_matrix, purchase_matrix)
    print('*' * 20, 'Full Count Matrix', '*' * 20)
    print(count_matrix)
    print()
    
    p_matrix = make_probability_matrix(count_matrix)
    print('*' * 20, 'Probability Matrix', '*' * 20)
    print(p_matrix)
    print()
    
    spm = sparse_p_matrix(p_matrix)
    print('*' * 20, 'Sparse Probability Matrix', '*' * 20)
    print(spm)
    print()
    
    ######
    itt = isbn_to_title(conn)
    print('*' * 20, 'itt dict', '*' * 20)
    print(itt)
    print()
    
    """
    selection = select_book(itt)
    while selection:
        similar_books(selection, count_matrix, p_matrix, itt, spm)
        input('Enter to continue:')
        selection = select_book(itt)
    ######
    cid = get_cust_id(conn)
    while cid:
        print()
        titles = purchase_history(cid, purchase_matrix[cid], conn)
        print(titles)
        print(get_recommendation(cid, spm, purchase_matrix[cid], conn))
        input('Enter to continue:')
        cid = get_cust_id(conn)
    """
if __name__ == "__main__":
    main1()
    
    
    
    
    
    
    
    
