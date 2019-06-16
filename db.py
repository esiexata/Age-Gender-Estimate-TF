import psycopg2

def teste (date, age, gender):
    print("campos a serem inseridos", date, age, gender)



def insert_age_gender(date, age, gender):

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
        postgres_insert_query = """ INSERT INTO agegender5 ( data, age, sex, etiny, disp) VALUES (%s,%s,%s,%s,%s)"""
        # cursor.execute('delete from agegender5')
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
        cursor.execute('select * from agegender5')
        recset = cursor.fetchall()
        for rec in recset:
            print(rec)
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

    return 0


#insert_age_gender('10/01/2015', 10, True)
