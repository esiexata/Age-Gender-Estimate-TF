import psycopg2


from datetime import datetime,tzinfo,timedelta

class Zone(tzinfo):
    def __init__(self,offset,isdst,name):
        self.offset = offset
        self.isdst = isdst
        self.name = name
    def utcoffset(self, dt):
        return timedelta(hours=self.offset) + self.dst(dt)
    def dst(self, dt):
            return timedelta(hours=1) if self.isdst else timedelta(0)
    def tzname(self,dt):
         return self.name

GMT = Zone(0,False,'GMT')
EST = Zone(-3,False,'EST')

print (datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S %Z'))
print (datetime.now(GMT).strftime('%m/%d/%Y %H:%M:%S %Z'))
print (datetime.now(EST).strftime('%m/%d/%Y %H:%M:%S %Z'))

t = datetime.strptime('2011-01-21 02:37:21','%Y-%m-%d %H:%M:%S')
t = t.replace(tzinfo=GMT)
print (t)
print (t.astimezone(EST))


def teste (date, age, gender):
    print("campos a serem inseridos", date, age, gender)


def insert_age_gender(date, age, gender):

    if gender == 1:
        gender = True
    else:
        gender = False

    EST = Zone(-3, False, 'EST')
    print(datetime.now(EST).strftime('%m/%d/%Y %H:%M:%S %Z'))
    data = (datetime.now(EST).strftime('%m/%d/%Y %H:%M:%S %Z'))

    try:
        connection = psycopg2.connect(user="alo",
                                      password="esiAdmins2005",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="teste")
        cursor = connection.cursor()
        postgres_insert_query = """ INSERT INTO agegender5 ( data, age, sex, etiny, disp) VALUES (%s,%s,%s,%s,%s)"""
        #cursor.execute('delete from agegender5')
        print("campos a serem inseridos", date, age, gender)

        record_to_insert = (str(date), int(age), gender, 'none', '10')
        cursor.execute(postgres_insert_query, record_to_insert)
        connection.commit()
        count = cursor.rowcount
        print(count, "insrerido com sucesso")
    except (Exception, psycopg2.Error) as error:
        if (connection):
            print("Failed to insert record into database", error)
            return 1
    finally:
        #cursor.execute('select * from agegender5')
        #recset = cursor.fetchall()
        #for rec in recset:
        #    print(rec)
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

    return 0


insert_age_gender('10/01/2015', 10, True)


