import psycopg2
con = psycopg2.connect(host='localhost', database='teste',user='alo', password='esiAdmins2005')
cur = con.cursor()

sql = 'create table agegender5 (id serial primary key, data DATE , age integer,sex boolean,etiny varchar(30),disp integer)'
#cur.execute(sql)


sql = "insert into agegender5 values (default,'12/12/1999','30','1','white','2')"
cur.execute(sql)

#cur.execute('delete from agegender5')
con.commit()

cur.execute('select * from agegender5')
recset = cur.fetchall()
for rec in recset:
    print (rec)

con.close()


