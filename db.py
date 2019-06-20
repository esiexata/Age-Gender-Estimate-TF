import psycopg2

def insert_age_gender(date, age, gender,img):

    if gender == 1:
        gender = True
    else:
        gender = False


    try:
        connection = psycopg2.connect(user="alo",
                                      password="esiAdmins2005",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="teste")
        cursor = connection.cursor()
        postgres_insert_query = """ INSERT INTO agegender7 ( data, age, sex, etiny, img, disp) VALUES (%s,%s,%s,%s,%s,%s)"""
        #cursor.execute('delete from agegender5')
        print("campos a serem inseridos", date, age, gender)

        record_to_insert = (str(date), int(age), gender, 'none', img, 99)
        cursor.execute(postgres_insert_query, record_to_insert)
        connection.commit()
        count = cursor.rowcount
        print(count, "insrerido com sucesso")
    except (Exception, psycopg2.Error) as error:
        if (connection):
            print("Failed to insert record into database", error)
            return 1
    finally:

        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

    return 0


def list_rows():
    connection = psycopg2.connect(user="alo",
                                  password="esiAdmins2005",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="teste")
    cursor = connection.cursor()

    cursor.execute('select * from agegender7')
    recset = cursor.fetchall()
    for rec in recset:
        print(rec)
    #closing database connection.
    cursor.close()
    connection.close()



def createDB():

    try:
        connection = psycopg2.connect(user="alo",
                                      password="esiAdmins2005",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="teste")
        cursor = connection.cursor()
        query = 'create table agegender7 (id serial primary key, data timestamp, age integer, sex varchar(10), etiny varchar(10), img text, disp integer)'
        cursor.execute(query)
        connection.commit()
        count = cursor.rowcount
        print(count, "insrerido com sucesso")
    except (Exception, psycopg2.Error) as error:
        if (connection):
            print("Failed to insert record into database", error)
            return 1
    finally:

        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

    return 0

#createDB()

list_rows()











