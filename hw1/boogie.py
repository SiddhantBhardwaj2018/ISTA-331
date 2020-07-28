import sqlite3,pandas as pd
import random
'''
def count():
    d = {'971':[-1.0,0.33333,0.33333,0.66667],'972':[1.0,-1.0,0.0,1.0],'973':[1.0,0.0,-1.0,0.0],'974':[1.0,0.5,0.0,-1.0]}
    lst = ['971','972','973','974']
    df = pd.DataFrame(index = lst,data = d)
    return(df)
'''

def main(cust_id):
    '''
    '''
    conn = sqlite3.connect('small.db')
    conn.row_factory = sqlite3.Row
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
    recent_req_df = req_df[req_df['Order_Date'] == req_df['Order_Date'].min()]
    if len(recent_req_df) == 1:
        return(recent_req_df['ISBN'])
    else:
        for i in recent_req_df['ISBN']:
             lst_isbn.append(i)
    print(lst_isbn[random.randrange(len(lst_isbn))])
   
if __name__ == "__main__": 
      
    main(1)